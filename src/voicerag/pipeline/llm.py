"""Groq client with a rotating key pool and per-key circuit breakers.

Requirement 5 asks for structured orchestration rather than a raw prompt-in
text-out call. The constraint that shapes this module is that we hold several
Groq FREE-tier accounts rather than one paid account, so rate limits are hit
routinely and are a normal operating condition, not an exception.

The design that follows:

  * Keys form a pool. A 429 on one key rotates to the next instead of failing
    the request, which turns N free accounts into roughly N times the throughput.
  * Each key has its own circuit breaker. A key that is exhausted or revoked is
    taken out of rotation for a cooldown rather than retried on every request -
    otherwise one dead key would slow every call down by its full timeout.
  * Retries are bounded, jittered, and honour Retry-After. Unjittered backoff
    across a key pool synchronizes retries into thundering herds.
  * When every key is unavailable the failure is explicit (AllKeysUnavailable),
    so the caller can fall back to the extractive path instead of hanging.

The pipeline treats this whole module as optional: if generation fails, the
router answers extractively and the trace records the degradation.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field

import httpx

from voicerag.config import Settings, get_settings

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


class LlmError(RuntimeError):
    """Generation failed."""


class AllKeysUnavailable(LlmError):
    """Every key in the pool is cooling down or exhausted."""


@dataclass
class KeyState:
    key: str
    index: int
    consecutive_failures: int = 0
    cooldown_until: float = 0.0
    calls: int = 0
    rate_limits: int = 0

    @property
    def masked(self) -> str:
        """Never log a raw key."""
        return f"key{self.index}:...{self.key[-4:]}" if len(self.key) >= 4 else f"key{self.index}"

    def available(self, now: float) -> bool:
        return now >= self.cooldown_until


@dataclass
class LlmResult:
    text: str
    key_used: str
    attempts: int
    latency_ms: float
    finish_reason: str | None = None


@dataclass
class GroqPool:
    """Round-robin pool with per-key circuit breaking."""

    keys: list[KeyState] = field(default_factory=list)
    breaker_threshold: int = 3
    breaker_cooldown_s: float = 60.0
    _cursor: int = 0

    @classmethod
    def from_settings(cls, settings: Settings) -> GroqPool:
        return cls(keys=[KeyState(key=k, index=i) for i, k in enumerate(settings.groq_key_pool)])

    def __len__(self) -> int:
        return len(self.keys)

    def acquire(self, now: float | None = None) -> KeyState:
        """Next available key, round-robin. Raises if all are cooling down."""
        if not self.keys:
            raise AllKeysUnavailable("no Groq API keys configured")

        now = time.monotonic() if now is None else now
        for offset in range(len(self.keys)):
            candidate = self.keys[(self._cursor + offset) % len(self.keys)]
            if candidate.available(now):
                self._cursor = (self._cursor + offset + 1) % len(self.keys)
                return candidate

        soonest = min(k.cooldown_until for k in self.keys) - now
        raise AllKeysUnavailable(f"all {len(self.keys)} keys cooling down for {soonest:.1f}s")

    def report_success(self, state: KeyState) -> None:
        state.consecutive_failures = 0
        state.calls += 1

    def report_rate_limit(self, state: KeyState, retry_after: float | None, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        state.rate_limits += 1
        # A rate limit is not a fault - the key is healthy, just busy - so it
        # rests for the advertised window without counting toward the breaker.
        state.cooldown_until = now + (retry_after if retry_after is not None else 20.0)

    def report_failure(self, state: KeyState, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        state.consecutive_failures += 1
        if state.consecutive_failures >= self.breaker_threshold:
            state.cooldown_until = now + self.breaker_cooldown_s

    def snapshot(self) -> list[dict]:
        """Observability without leaking secrets."""
        now = time.monotonic()
        return [
            {
                "key": k.masked,
                "calls": k.calls,
                "rate_limits": k.rate_limits,
                "failures": k.consecutive_failures,
                "available": k.available(now),
            }
            for k in self.keys
        ]


def _parse_retry_after(response: httpx.Response) -> float | None:
    raw = response.headers.get("retry-after")
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    # Groq also reports remaining time on its own headers.
    for header in ("x-ratelimit-reset-requests", "x-ratelimit-reset-tokens"):
        raw = response.headers.get(header)
        if raw:
            try:
                return float(raw.rstrip("s"))
            except ValueError:
                continue
    return None


class GroqClient:
    def __init__(
        self,
        settings: Settings | None = None,
        pool: GroqPool | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.pool = pool or GroqPool.from_settings(self.settings)
        self._client = client

    @property
    def enabled(self) -> bool:
        """False when no keys exist, so the router can skip escalation entirely."""
        return len(self.pool) > 0

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self.settings.groq_timeout_s)
        return self._client

    def complete(
        self,
        system: str,
        user: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        json_mode: bool = False,
    ) -> LlmResult:
        payload: dict = {
            "model": self.settings.groq_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        started = time.perf_counter()
        attempts = 0
        last_error: Exception | None = None
        # One extra pass so a fresh key gets a turn after rotation.
        budget = self.settings.groq_max_retries + 1 + len(self.pool)

        while attempts < budget:
            attempts += 1
            try:
                state = self.pool.acquire()
            except AllKeysUnavailable as exc:
                raise AllKeysUnavailable(f"{exc} (after {attempts - 1} attempts)") from last_error

            try:
                response = self.client.post(
                    GROQ_URL,
                    headers={"Authorization": f"Bearer {state.key}"},
                    json=payload,
                )
            except httpx.HTTPError as exc:
                last_error = exc
                self.pool.report_failure(state)
                self._backoff(attempts)
                continue

            if response.status_code == 200:
                self.pool.report_success(state)
                body = response.json()
                choice = (body.get("choices") or [{}])[0]
                text = (choice.get("message") or {}).get("content") or ""
                return LlmResult(
                    text=text.strip(),
                    key_used=state.masked,
                    attempts=attempts,
                    latency_ms=(time.perf_counter() - started) * 1000,
                    finish_reason=choice.get("finish_reason"),
                )

            if response.status_code == 429:
                self.pool.report_rate_limit(state, _parse_retry_after(response))
                last_error = LlmError(f"{state.masked} rate limited")
                # Rotate immediately: another key is probably free right now.
                continue

            if response.status_code in (401, 403):
                # Bad credentials never recover on retry; park the key for good.
                self.pool.report_failure(state)
                state.cooldown_until = time.monotonic() + 3600
                last_error = LlmError(f"{state.masked} unauthorized ({response.status_code})")
                continue

            if response.status_code >= 500:
                self.pool.report_failure(state)
                last_error = LlmError(f"groq {response.status_code}")
                self._backoff(attempts)
                continue

            # 4xx that is our fault (malformed request): retrying cannot help.
            raise LlmError(f"groq rejected the request ({response.status_code}): {response.text[:300]}")

        raise LlmError(f"groq failed after {attempts} attempts: {last_error}")

    def complete_json(self, system: str, user: str, max_tokens: int = 256) -> dict:
        """Generation constrained to JSON, with defensive parsing.

        Even in JSON mode a model can emit fenced or prefixed output, so parsing
        falls back to extracting the outermost braces before giving up. The
        caller gets a dict or an exception, never a half-parsed string.
        """
        result = self.complete(system, user, max_tokens=max_tokens, json_mode=True)
        text = result.text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError as exc:
                raise LlmError(f"unparseable JSON from model: {text[:200]}") from exc

        raise LlmError(f"model returned no JSON object: {text[:200]}")

    def _backoff(self, attempt: int) -> None:
        # Full jitter. Without it, a pool of keys retries in lockstep and
        # recreates the burst that caused the failure.
        delay = min(2.0, 0.15 * (2 ** (attempt - 1)))
        time.sleep(random.uniform(0, delay))

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
