"""Tests for the Groq key pool and client.

The behaviours worth pinning down are the ones that only appear under failure:
rotation on rate limit, breakers isolating a dead key, bounded retries, and an
explicit error when the pool is exhausted so the router can fall back rather
than hang. All requests go through a mock transport.
"""

from __future__ import annotations

import httpx
import pytest

from voicerag.config import Settings
from voicerag.pipeline.llm import (
    AllKeysUnavailable,
    GroqClient,
    GroqPool,
    KeyState,
    LlmError,
)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr("voicerag.pipeline.llm.time.sleep", lambda *_: None)


def make_settings(keys: str = "key-aaaa,key-bbbb,key-cccc", **overrides) -> Settings:
    base = {"groq_api_keys": keys, "groq_max_retries": 2, "groq_model": "test-model"}
    base.update(overrides)
    return Settings(**base)


class ScriptedTransport(httpx.MockTransport):
    """Replays a scripted sequence and records which key each call used."""

    def __init__(self, responses: list[httpx.Response]):
        self.keys_used: list[str] = []
        self._responses = responses

        def handler(request: httpx.Request) -> httpx.Response:
            self.keys_used.append(request.headers["Authorization"].removeprefix("Bearer "))
            i = min(len(self.keys_used) - 1, len(self._responses) - 1)
            return self._responses[i]

        super().__init__(handler)

    @property
    def calls(self) -> int:
        return len(self.keys_used)


def ok(text: str = "an answer") -> httpx.Response:
    return httpx.Response(
        200, json={"choices": [{"message": {"content": text}, "finish_reason": "stop"}]}
    )


def build(responses, settings=None) -> tuple[GroqClient, ScriptedTransport]:
    settings = settings or make_settings()
    transport = ScriptedTransport(responses)
    client = GroqClient(settings=settings, client=httpx.Client(transport=transport))
    return client, transport


class TestPool:
    def test_round_robins_across_keys(self):
        pool = GroqPool.from_settings(make_settings())
        used = [pool.acquire().index for _ in range(6)]
        assert used == [0, 1, 2, 0, 1, 2]

    def test_skips_a_cooling_key(self):
        pool = GroqPool.from_settings(make_settings())
        first = pool.acquire()
        pool.report_rate_limit(first, retry_after=30, now=100.0)

        following = [pool.acquire(now=100.0).index for _ in range(4)]
        assert first.index not in following

    def test_breaker_opens_after_repeated_failures(self):
        pool = GroqPool.from_settings(make_settings(), )
        pool.breaker_threshold = 3
        state = pool.keys[0]

        for _ in range(3):
            pool.report_failure(state, now=100.0)

        assert not state.available(100.0)
        assert state.available(100.0 + pool.breaker_cooldown_s + 1)

    def test_success_resets_the_breaker(self):
        pool = GroqPool.from_settings(make_settings())
        state = pool.keys[0]
        pool.report_failure(state, now=100.0)
        pool.report_failure(state, now=100.0)
        pool.report_success(state)
        assert state.consecutive_failures == 0

    def test_raises_when_every_key_is_down(self):
        pool = GroqPool.from_settings(make_settings())
        for state in pool.keys:
            pool.report_rate_limit(state, retry_after=30, now=100.0)

        with pytest.raises(AllKeysUnavailable, match="cooling down"):
            pool.acquire(now=100.0)

    def test_empty_pool_raises(self):
        pool = GroqPool.from_settings(make_settings(keys=""))
        with pytest.raises(AllKeysUnavailable, match="no Groq API keys"):
            pool.acquire()

    def test_snapshot_masks_keys(self):
        pool = GroqPool.from_settings(make_settings())
        for row in pool.snapshot():
            assert "key-aaaa" not in row["key"]
            assert row["key"].startswith("key")

    def test_key_state_masking(self):
        assert "secret" not in KeyState(key="sk-supersecret", index=0).masked


class TestCompletion:
    def test_happy_path(self):
        client, transport = build([ok("hello")])
        result = client.complete("sys", "user")

        assert result.text == "hello"
        assert result.attempts == 1
        assert transport.calls == 1
        assert result.finish_reason == "stop"

    def test_rate_limit_rotates_to_another_key(self):
        """The core value of pooling free accounts."""
        client, transport = build([httpx.Response(429), ok("recovered")])
        result = client.complete("sys", "user")

        assert result.text == "recovered"
        assert transport.calls == 2
        assert transport.keys_used[0] != transport.keys_used[1], "must switch keys, not retry the same one"

    def test_unauthorized_key_is_parked_and_another_used(self):
        client, transport = build([httpx.Response(401), ok("via second key")])
        result = client.complete("sys", "user")

        assert result.text == "via second key"
        assert client.pool.keys[0].cooldown_until > 0

    def test_server_error_retries_then_succeeds(self):
        client, transport = build([httpx.Response(500), httpx.Response(502), ok("third time")])
        assert client.complete("sys", "user").text == "third time"
        assert transport.calls == 3

    def test_gives_up_when_all_keys_rate_limited(self):
        client, transport = build([httpx.Response(429)])
        with pytest.raises(AllKeysUnavailable):
            client.complete("sys", "user")
        # One attempt per key, then stop - no unbounded hammering.
        assert transport.calls == 3

    def test_malformed_request_is_not_retried(self):
        client, transport = build([httpx.Response(400, text="bad payload")])
        with pytest.raises(LlmError, match="rejected"):
            client.complete("sys", "user")
        assert transport.calls == 1

    def test_network_error_is_retried(self):
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] < 3:
                raise httpx.ConnectError("boom")
            return ok("after network trouble")

        client = GroqClient(
            settings=make_settings(),
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        assert client.complete("sys", "user").text == "after network trouble"

    def test_enabled_reflects_key_availability(self):
        assert GroqClient(settings=make_settings()).enabled is True
        assert GroqClient(settings=make_settings(keys="")).enabled is False

    def test_json_mode_sets_response_format(self):
        seen: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            import json as _json

            seen.update(_json.loads(request.content))
            return ok('{"ok": true}')

        client = GroqClient(
            settings=make_settings(),
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        client.complete_json("sys", "user")
        assert seen["response_format"] == {"type": "json_object"}


class TestJsonParsing:
    def test_parses_clean_json(self):
        client, _ = build([ok('{"grounded": true, "score": 0.9}')])
        assert client.complete_json("sys", "user") == {"grounded": True, "score": 0.9}

    def test_recovers_json_from_surrounding_prose(self):
        """Models wrap JSON in fences or preamble even in JSON mode."""
        client, _ = build([ok('Here you go:\n```json\n{"grounded": false}\n```')])
        assert client.complete_json("sys", "user") == {"grounded": False}

    def test_raises_on_unparseable_output(self):
        client, _ = build([ok("no json at all here")])
        with pytest.raises(LlmError, match="no JSON object"):
            client.complete_json("sys", "user")

    def test_raises_on_truncated_json(self):
        """Hitting max_tokens mid-object leaves no closing brace."""
        client, _ = build([ok('{"grounded": tru')])
        with pytest.raises(LlmError, match="no JSON object"):
            client.complete_json("sys", "user")

    def test_raises_when_braces_enclose_invalid_json(self):
        client, _ = build([ok('prefix {"grounded": tru, oops} suffix')])
        with pytest.raises(LlmError, match="unparseable JSON"):
            client.complete_json("sys", "user")
