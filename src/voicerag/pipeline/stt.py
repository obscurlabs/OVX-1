"""Speech-to-text via Sarvam, with a hard credit guard.

The account has **100 free credits total**. A single careless benchmark loop
would consume all of them and leave the demo dead on submission day, so this
module is built defensively rather than conveniently:

  1. Every transcription is cached by a hash of the raw audio bytes. Replaying a
     recording is free and byte-identical to the original call.
  2. Live calls are refused unless SARVAM_ALLOW_LIVE is explicitly enabled. On a
     cache miss with live disabled, it raises SttCreditGuard rather than
     transcribing. Failing loudly is the point: a silent fallback is exactly how
     a credit budget disappears.
  3. Retries are bounded and never retry a 4xx, because a rejected request that
     is retried five times can still be billed five times.

Sarvam's Saarika model is used over ElevenLabs because the corpus is Indic and
Saarika handles Hindi and code-mixed Hindi-English speech, returning a detected
language code we route on downstream.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import time
from pathlib import Path

import httpx

from voicerag.config import Settings, get_settings
from voicerag.contracts import Transcript


class SttError(RuntimeError):
    """Transcription failed."""


class SttCreditGuard(SttError):
    """A live Sarvam call was required but not authorized.

    Raised on a cache miss while SARVAM_ALLOW_LIVE is off. Deliberately not a
    soft failure: the caller must decide to spend a credit, never this module.
    """


class AudioConversionError(SttError):
    """ffmpeg could not decode the supplied audio."""


def audio_hash(data: bytes) -> str:
    """Cache key over the RAW upload bytes.

    Hashing before conversion means the key is stable regardless of ffmpeg
    version or encoder settings, so an upgrade cannot silently invalidate the
    cache and trigger a wave of paid re-transcriptions.
    """
    return hashlib.blake2b(data, digest_size=16).hexdigest()


class TranscriptCache:
    """Persistent audio-hash -> transcript map."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._entries: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            self._entries = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            # A corrupt cache must not block the pipeline; worst case is a
            # re-transcription, which the caller still has to authorize.
            self._entries = {}

    def get(self, key: str) -> dict | None:
        return self._entries.get(key)

    def put(self, key: str, text: str, language: str, model: str) -> None:
        self._entries[key] = {
            "text": text,
            "language": language,
            "model": model,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self._save()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Write-then-rename so an interrupted save cannot corrupt a cache that
        # represents real money already spent.
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._entries, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def __len__(self) -> int:
        return len(self._entries)


def convert_to_wav(data: bytes, ffmpeg_path: str, timeout_s: float = 30.0) -> bytes:
    """Transcode arbitrary browser audio to 16kHz mono PCM WAV.

    Browsers record WebM/Opus (Chrome) or MP4/AAC (Safari); Sarvam expects WAV.
    16kHz mono is both what ASR models want and the smallest payload that keeps
    full accuracy, which also trims upload time from the latency budget.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        src = Path(tmpdir) / "in.bin"
        dst = Path(tmpdir) / "out.wav"
        src.write_bytes(data)

        result = subprocess.run(
            [
                ffmpeg_path,
                "-hide_banner", "-loglevel", "error",
                "-i", str(src),
                "-ar", "16000",
                "-ac", "1",
                "-c:a", "pcm_s16le",
                "-y", str(dst),
            ],
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
        if result.returncode != 0 or not dst.exists():
            detail = result.stderr.decode("utf-8", errors="replace").strip()[:400]
            raise AudioConversionError(f"ffmpeg failed (exit {result.returncode}): {detail}")
        return dst.read_bytes()


class SarvamSTT:
    def __init__(
        self,
        settings: Settings | None = None,
        cache_path: Path | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        from voicerag.config import Paths

        self.settings = settings or get_settings()
        self.cache = TranscriptCache(cache_path or Paths.stt_cache)
        self._client = client
        self.live_calls = 0  # observability: how many credits this process spent

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=30.0)
        return self._client

    def transcribe(
        self,
        audio: bytes,
        lang_hint: str | None = None,
        allow_live: bool | None = None,
    ) -> Transcript:
        """Transcribe audio, preferring the cache.

        `allow_live` overrides the setting for a single call, which is how the
        demo endpoint opts in explicitly while every other caller stays safe.
        """
        if not audio:
            raise SttError("empty audio payload")

        key = audio_hash(audio)

        cached = self.cache.get(key)
        if cached is not None:
            return Transcript(
                text=cached["text"],
                language=cached.get("language", "unknown"),
                cached=True,
                audio_hash=key,
            )

        permitted = self.settings.sarvam_allow_live if allow_live is None else allow_live
        if not permitted:
            raise SttCreditGuard(
                f"cache miss for audio {key[:12]} and live Sarvam calls are disabled. "
                "The account has only 100 credits. Set SARVAM_ALLOW_LIVE=1, or pass "
                "allow_live=True, to authorize spending one."
            )

        if not self.settings.sarvam_api_key:
            raise SttError("SARVAM_API_KEY is not set")

        wav = convert_to_wav(audio, self.settings.ffmpeg_path)
        text, language = self._call_sarvam(wav, lang_hint)

        self.cache.put(key, text, language, self.settings.sarvam_model)
        self.live_calls += 1

        return Transcript(text=text, language=language, cached=False, audio_hash=key)

    def _call_sarvam(self, wav: bytes, lang_hint: str | None) -> tuple[str, str]:
        """POST to Sarvam with bounded retries."""
        # "unknown" asks Saarika to auto-detect, which is what we want for a
        # bilingual demo where the speaker may switch languages mid-session.
        language_code = lang_hint or "unknown"

        last_error: Exception | None = None
        for attempt in range(self.settings.groq_max_retries + 1):
            try:
                response = self.client.post(
                    self.settings.sarvam_stt_url,
                    headers={"api-subscription-key": self.settings.sarvam_api_key},
                    files={"file": ("audio.wav", wav, "audio/wav")},
                    data={"model": self.settings.sarvam_model, "language_code": language_code},
                )
            except httpx.HTTPError as exc:
                last_error = exc
                # Network-level failure: nothing was billed, so retrying is safe.
                if attempt < self.settings.groq_max_retries:
                    time.sleep(0.4 * (2**attempt))
                    continue
                raise SttError(f"Sarvam request failed: {exc}") from exc

            if response.status_code == 200:
                payload = response.json()
                text = (payload.get("transcript") or "").strip()
                if not text:
                    raise SttError(f"Sarvam returned an empty transcript: {payload}")
                return text, payload.get("language_code") or "unknown"

            # 4xx is a client error - bad key, bad audio, quota exhausted. Retrying
            # cannot fix it and may be billed again, so fail immediately.
            if response.status_code < 500 and response.status_code != 429:
                raise SttError(
                    f"Sarvam rejected the request ({response.status_code}): {response.text[:300]}"
                )

            last_error = SttError(f"Sarvam {response.status_code}: {response.text[:200]}")
            if attempt < self.settings.groq_max_retries:
                time.sleep(0.4 * (2**attempt))

        raise SttError(f"Sarvam failed after retries: {last_error}")

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
