"""Tests for answer routing.

The routing decision is what keeps the pipeline inside its latency budget, so
the tests assert not just the answer but whether the LLM was called at all.
"""

from __future__ import annotations

import pytest

from voicerag.contracts import Decision, RetrievedChunk, Route
from voicerag.pipeline.router import AnswerRouter, ExtractiveAnswerer, content_terms


def chunk(text: str, cid: str = "c1", score: float = 0.03) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid,
        text=text,
        passage_id=f"p-{cid}",
        lang="en",
        strategy="passage",
        granularity="passage",
        fused_score=score,
    )


class FakeLlm:
    """Records whether escalation actually happened."""

    def __init__(self, reply: str = "A synthesized answer.", fail: bool = False, enabled: bool = True):
        self.reply = reply
        self.fail = fail
        self.enabled = enabled
        self.calls = 0

    def complete(self, system: str, user: str, **kwargs):
        self.calls += 1
        if self.fail:
            raise RuntimeError("groq unavailable")

        class Result:
            text = self.reply

        return Result()


class TestContentTerms:
    def test_strips_question_words(self):
        assert "what" not in content_terms("what is a corporation")
        assert "corporation" in content_terms("what is a corporation")

    def test_handles_hindi_question_words(self):
        terms = content_terms("निगम क्या है")
        assert "निगम" in terms
        assert "क्या" not in terms


class TestExtractiveAnswerer:
    @pytest.fixture
    def answerer(self):
        return ExtractiveAnswerer()

    def test_finds_the_answer_bearing_sentence(self, answerer):
        chunks = [
            chunk(
                "Many things exist in nature. Photosynthesis requires sunlight, water "
                "and carbon dioxide. Other processes are different."
            )
        ]
        candidate = answerer.answer("what does photosynthesis require", chunks)

        assert candidate is not None
        assert "sunlight" in candidate.text
        assert candidate.confidence > 0.4

    def test_prefers_sentences_with_numbers_for_numeric_questions(self, answerer):
        chunks = [
            chunk(
                "Boiling eggs is a common task. Boil the egg for 6 minutes. "
                "Boiling eggs can be done in many ways."
            )
        ]
        candidate = answerer.answer("how long should i boil an egg", chunks)
        assert candidate is not None
        assert "6 minutes" in candidate.text

    def test_returns_none_without_chunks(self, answerer):
        assert answerer.answer("anything", []) is None

    def test_returns_none_for_contentless_query(self, answerer):
        assert answerer.answer("what is the", [chunk("some text here")]) is None

    def test_confidence_is_low_for_unrelated_context(self, answerer):
        chunks = [chunk("The Pacific Ocean is the largest ocean on Earth.")]
        candidate = answerer.answer("how does photosynthesis work in plants", chunks)
        assert candidate is None or candidate.confidence < 0.45


class TestRouting:
    def test_high_confidence_stays_extractive_and_never_calls_the_llm(self):
        """The latency-critical path: no network call at all."""
        llm = FakeLlm()
        router = AnswerRouter(llm=llm)
        chunks = [chunk("Photosynthesis requires sunlight, water and carbon dioxide.")]

        answer, route = router.route("what does photosynthesis require", chunks)

        assert route is Route.EXTRACTIVE
        assert answer.decision is Decision.ANSWER
        assert llm.calls == 0, "extractive path must not touch the LLM"
        assert answer.citations == ["c1"]

    def test_low_confidence_escalates(self):
        llm = FakeLlm(reply="Synthesized from several passages.")
        router = AnswerRouter(llm=llm)
        # Deliberately weak overlap so extraction cannot clear the bar.
        chunks = [chunk("Assorted unrelated commentary about various topics.")]

        answer, route = router.route("explain the causes of the french revolution", chunks)

        assert route is Route.LLM
        assert llm.calls == 1
        assert answer.text == "Synthesized from several passages."

    def test_llm_failure_falls_back_to_extractive(self):
        """Degradation, not a 500."""
        llm = FakeLlm(fail=True)
        router = AnswerRouter(llm=llm)
        chunks = [chunk("Assorted unrelated commentary about various topics.")]

        answer, route = router.route("explain the causes of the french revolution", chunks)

        assert llm.calls == 1
        assert route is Route.EXTRACTIVE
        assert answer.text  # something was still returned

    def test_disabled_llm_is_not_called(self):
        llm = FakeLlm(enabled=False)
        router = AnswerRouter(llm=llm)
        chunks = [chunk("Assorted unrelated commentary about topics.")]

        router.route("explain something obscure entirely", chunks)
        assert llm.calls == 0

    def test_no_chunks_abstains_without_calling_the_llm(self):
        """Escalating with an empty context invites answering from memory."""
        llm = FakeLlm()
        router = AnswerRouter(llm=llm)

        answer, _ = router.route("anything at all", [])

        assert answer.decision is Decision.ABSTAIN
        assert llm.calls == 0, "must never generate without retrieved context"

    def test_insufficient_context_sentinel_abstains(self):
        """The LLM participates in abstention rather than being forced to answer."""
        llm = FakeLlm(reply="INSUFFICIENT_CONTEXT")
        router = AnswerRouter(llm=llm)
        chunks = [chunk("Assorted unrelated commentary about various topics.")]

        answer, route = router.route("what is the population of mars colony", chunks)

        assert answer.decision is Decision.ABSTAIN
        assert route is Route.LLM

    def test_empty_llm_reply_abstains(self):
        llm = FakeLlm(reply="")
        router = AnswerRouter(llm=FakeLlm(reply=""))
        router.llm = llm
        chunks = [chunk("Assorted unrelated commentary about various topics.")]

        answer, _ = router.route("something quite unrelated to this", chunks)
        assert answer.decision is Decision.ABSTAIN


class TestContextBuilding:
    def test_context_is_capped_and_cites_what_it_used(self):
        llm = FakeLlm()
        router = AnswerRouter(llm=llm, max_context_chunks=2, max_context_chars=200)
        chunks = [chunk("x" * 500, cid=f"c{i}") for i in range(5)]

        context, cited = router._build_context(chunks)

        assert len(context) <= 260  # payload plus the "[n] " markers
        assert len(cited) <= 2
        assert cited == ["c0"] or cited == ["c0", "c1"]
