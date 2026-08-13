"""Tests for the Sarvam STT layer.

These are budget tests as much as correctness tests. The account holds 100
credits total, so the behaviours worth pinning down are: never call out when the
cache can answer, never call out without authorization, and never retry a
request that may already have been billed.

Every test uses a mock transport. Nothing here can reach Sarvam.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import httpx
import pytest

from voicerag.config import Settings
from voicerag.contracts import Transcript
from voicerag.pipeline import stt as stt_module
from voicerag.pipeline.stt import (
    SarvamSTT,
    SttCreditGuard,
    SttError,
    TranscriptCache,
    audio_hash,
    convert_to_wav,
)

AUDIO = b"fake-webm-audio-payload"
OTHER_AUDIO = b"a-different-recording"


@pytest.fixture(autouse=True)
def no_real_ffmpeg(monkeypatch):
    """Conversion is exercised separately; unit tests must not shell out."""
    monkeypatch.setattr(stt_module, "convert_to_wav", lambda data, path, **kw: b"RIFFfake")


def make_settings(**overrides) -> Settings:
    base = {
        "sarvam_api_key": "test-key",
        "sarvam_allow_live": False,
        "groq_max_retries": 2,
        "ffmpeg_path": "ffmpeg",
    }
    base.update(overrides)
    return Settings(**base)


class RecordingTransport(httpx.MockTransport):
    """Mock transport that counts requests, so 'was a credit spent?' is testable."""

    def __init__(self, responses):
        self.calls = 0
        self._responses = list(responses)

        def handler(request: httpx.Request) -> httpx.Response:
            self.calls += 1
            item = self._responses[min(self.calls - 1, len(self._responses) - 1)]
            return item

        super().__init__(handler)


def ok_response(text="नमस्ते दुनिया", language="hi-IN") -> httpx.Response:
    return httpx.Response(200, json={"transcript": text, "language_code": language})


def build(settings, cache_path, responses) -> tuple[SarvamSTT, RecordingTransport]:
    transport = RecordingTransport(responses)
    client = httpx.Client(transport=transport)
    return SarvamSTT(settings=settings, cache_path=cache_path, client=client), transport


class TestCreditGuard:
    def test_refuses_live_call_when_disabled(self, tmp_path):
        engine, transport = build(make_settings(), tmp_path / "c.json", [ok_response()])

        with pytest.raises(SttCreditGuard, match="only 100 credits"):
            engine.transcribe(AUDIO)

        assert transport.calls == 0, "guard must block BEFORE any billable request"
        assert engine.live_calls == 0

    def test_per_call_override_authorizes_one_call(self, tmp_path):
        engine, transport = build(make_settings(), tmp_path / "c.json", [ok_response()])

        result = engine.transcribe(AUDIO, allow_live=True)

        assert transport.calls == 1
        assert engine.live_calls == 1
        assert result.cached is False
        assert result.text == "नमस्ते दुनिया"

    def test_settings_flag_authorizes(self, tmp_path):
        settings = make_settings(sarvam_allow_live=True)
        engine, transport = build(settings, tmp_path / "c.json", [ok_response()])

        engine.transcribe(AUDIO)
        assert transport.calls == 1

    def test_missing_api_key_fails_before_calling(self, tmp_path):
        settings = make_settings(sarvam_api_key="", sarvam_allow_live=True)
        engine, transport = build(settings, tmp_path / "c.json", [ok_response()])

        with pytest.raises(SttError, match="SARVAM_API_KEY"):
            engine.transcribe(AUDIO)
        assert transport.calls == 0


class TestCaching:
    def test_repeat_audio_never_spends_a_second_credit(self, tmp_path):
        """The single most important behaviour in this module."""
        engine, transport = build(make_settings(), tmp_path / "c.json", [ok_response()])

        first = engine.transcribe(AUDIO, allow_live=True)
        for _ in range(20):
            again = engine.transcribe(AUDIO)  # live NOT authorized
            assert again.text == first.text
            assert again.cached is True

        assert transport.calls == 1, "replay must be free"
        assert engine.live_calls == 1

    def test_cache_survives_process_restart(self, tmp_path):
        cache_path = tmp_path / "c.json"
        engine, transport = build(make_settings(), cache_path, [ok_response()])
        engine.transcribe(AUDIO, allow_live=True)

        # A fresh instance, as a redeployed server would be.
        reborn, transport2 = build(make_settings(), cache_path, [ok_response()])
        result = reborn.transcribe(AUDIO)

        assert result.cached is True
        assert transport2.calls == 0

    def test_different_audio_is_a_separate_entry(self, tmp_path):
        engine, transport = build(
            make_settings(), tmp_path / "c.json", [ok_response("first"), ok_response("second")]
        )
        engine.transcribe(AUDIO, allow_live=True)

        with pytest.raises(SttCreditGuard):
            engine.transcribe(OTHER_AUDIO)
        assert transport.calls == 1

    def test_corrupt_cache_does_not_crash(self, tmp_path):
        cache_path = tmp_path / "c.json"
        cache_path.write_text("{ this is not json", encoding="utf-8")

        cache = TranscriptCache(cache_path)
        assert len(cache) == 0

    def test_cache_writes_valid_json(self, tmp_path):
        cache_path = tmp_path / "c.json"
        engine, _ = build(make_settings(), cache_path, [ok_response()])
        engine.transcribe(AUDIO, allow_live=True)

        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        entry = payload[audio_hash(AUDIO)]
        assert entry["text"] == "नमस्ते दुनिया"
        assert entry["language"] == "hi-IN"


class TestRetries:
    def test_client_error_is_not_retried(self, tmp_path):
        """A 4xx may already have been billed; retrying risks paying again."""
        engine, transport = build(
            make_settings(), tmp_path / "c.json", [httpx.Response(400, text="bad audio")]
        )

        with pytest.raises(SttError, match="rejected"):
            engine.transcribe(AUDIO, allow_live=True)

        assert transport.calls == 1, "4xx must not be retried"

    def test_server_error_is_retried_then_fails(self, tmp_path, monkeypatch):
        monkeypatch.setattr(stt_module.time, "sleep", lambda *_: None)
        engine, transport = build(
            make_settings(groq_max_retries=2), tmp_path / "c.json", [httpx.Response(503)]
        )

        with pytest.raises(SttError):
            engine.transcribe(AUDIO, allow_live=True)

        assert transport.calls == 3  # initial + 2 retries

    def test_rate_limit_is_retried(self, tmp_path, monkeypatch):
        monkeypatch.setattr(stt_module.time, "sleep", lambda *_: None)
        engine, transport = build(
            make_settings(groq_max_retries=1),
            tmp_path / "c.json",
            [httpx.Response(429), ok_response("recovered")],
        )

        result = engine.transcribe(AUDIO, allow_live=True)
        assert result.text == "recovered"
        assert transport.calls == 2

    def test_failed_call_is_not_cached(self, tmp_path):
        cache_path = tmp_path / "c.json"
        engine, _ = build(make_settings(), cache_path, [httpx.Response(400)])

        with pytest.raises(SttError):
            engine.transcribe(AUDIO, allow_live=True)

        assert TranscriptCache(cache_path).get(audio_hash(AUDIO)) is None

    def test_empty_transcript_is_an_error(self, tmp_path):
        engine, _ = build(
            make_settings(), tmp_path / "c.json", [httpx.Response(200, json={"transcript": "  "})]
        )
        with pytest.raises(SttError, match="empty transcript"):
            engine.transcribe(AUDIO, allow_live=True)


class TestInput:
    def test_empty_audio_rejected(self, tmp_path):
        engine, transport = build(make_settings(), tmp_path / "c.json", [ok_response()])
        with pytest.raises(SttError, match="empty audio"):
            engine.transcribe(b"", allow_live=True)
        assert transport.calls == 0

    def test_hash_is_stable_and_distinct(self):
        assert audio_hash(AUDIO) == audio_hash(AUDIO)
        assert audio_hash(AUDIO) != audio_hash(OTHER_AUDIO)

    def test_returns_contract_type(self, tmp_path):
        engine, _ = build(make_settings(), tmp_path / "c.json", [ok_response()])
        result = engine.transcribe(AUDIO, allow_live=True)
        assert isinstance(result, Transcript)
        assert result.audio_hash == audio_hash(AUDIO)


def _ffmpeg() -> str | None:
    from voicerag.config import get_settings

    configured = get_settings().ffmpeg_path
    if configured and Path(configured).exists():
        return configured
    return shutil.which("ffmpeg")


@pytest.mark.skipif(_ffmpeg() is None, reason="ffmpeg not available")
class TestConversion:
    def test_converts_real_audio_to_16k_mono_wav(self, tmp_path):
        """Round-trip through real ffmpeg, since the browser path depends on it."""
        ffmpeg = _ffmpeg()
        source = tmp_path / "tone.webm"
        subprocess.run(
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi",
             "-i", "sine=frequency=440:duration=1", "-c:a", "libopus", "-y", str(source)],
            check=True, capture_output=True, timeout=60,
        )

        wav = convert_to_wav(source.read_bytes(), ffmpeg)

        assert wav[:4] == b"RIFF" and wav[8:12] == b"WAVE"
        # Byte 22 of a PCM WAV header is the channel count; 24-27 the sample rate.
        assert int.from_bytes(wav[22:24], "little") == 1, "must be mono"
        assert int.from_bytes(wav[24:28], "little") == 16000, "must be 16kHz"

    def test_garbage_input_raises(self, tmp_path):
        from voicerag.pipeline.stt import AudioConversionError

        with pytest.raises(AudioConversionError):
            convert_to_wav(b"this is definitely not audio", _ffmpeg())
