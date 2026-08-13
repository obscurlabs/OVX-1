"""End-to-end orchestration: audio in, grounded answer or refusal out.

This is the harness proper (requirement 5). Every stage is timed, every decision
is recorded, and every failure has a defined next move rather than propagating
as a 500. The trace it emits is what the latency analytics aggregate, so
requirement 4 is satisfied by instrumentation the pipeline needs anyway.

Order matters and is chosen so refusals are cheap:

    STT           only when audio is supplied, cache-first
    GUARD_IN      unsafe / injection / junk rejected before any index is touched
    RETRIEVE      dense + lexical + fusion
    GUARD_IN'     relevance: does the corpus actually cover this?
    ROUTE         extractive span, or escalate to the LLM
    GUARD_OUT     grounding: is the answer supported by the passages?

A refusal at any gate short-circuits the rest, so an off-topic question costs a
few milliseconds rather than a full generation.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from voicerag.config import Paths, Settings, get_settings
from voicerag.contracts import (
    Answer,
    Decision,
    PipelineTrace,
    QueryRequest,
    QueryResponse,
    Stage,
    Timer,
    Transcript,
)
from voicerag.pipeline.guardrails import GroundingGuard, InputGuard, RelevanceGuard
from voicerag.pipeline.retrieval import HybridRetriever
from voicerag.pipeline.router import AnswerRouter


class VoiceRagPipeline:
    def __init__(
        self,
        retriever: HybridRetriever,
        router: AnswerRouter,
        input_guard: InputGuard | None = None,
        relevance_guard: RelevanceGuard | None = None,
        grounding_guard: GroundingGuard | None = None,
        stt=None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.retriever = retriever
        self.router = router
        self.input_guard = input_guard or InputGuard()
        self.relevance_guard = relevance_guard or RelevanceGuard()
        self.grounding_guard = grounding_guard or GroundingGuard(
            threshold=self.settings.grounding_threshold
        )
        self.stt = stt

    @classmethod
    def load(
        cls,
        index_dir: Path | None = None,
        encoder_dir: Path | None = None,
        settings: Settings | None = None,
        with_stt: bool = True,
    ) -> VoiceRagPipeline:
        from voicerag.pipeline.llm import GroqClient
        from voicerag.pipeline.router import ExtractiveAnswerer

        settings = settings or get_settings()
        retriever = HybridRetriever.load(
            index_dir or Paths.indexes,
            encoder_dir or Paths.onnx_encoder,
            dim=settings.embed_dim,
        )
        router = AnswerRouter(
            extractive=ExtractiveAnswerer(),
            llm=GroqClient(settings=settings),
        )

        stt = None
        if with_stt:
            from voicerag.pipeline.stt import SarvamSTT

            stt = SarvamSTT(settings=settings)

        return cls(retriever, router, stt=stt, settings=settings)

    # -- request path -------------------------------------------------------

    def answer(self, request: QueryRequest, allow_live_stt: bool | None = None) -> QueryResponse:
        trace = PipelineTrace(request_id=uuid.uuid4().hex[:12])
        transcript: Transcript | None = None

        # --- speech to text ------------------------------------------------
        if request.audio_b64:
            import base64

            if self.stt is None:
                return self._refuse(
                    trace, "Voice input isn't available right now.", "stt not configured", Stage.STT
                )

            try:
                audio = base64.b64decode(request.audio_b64)
            except Exception as exc:  # noqa: BLE001
                return self._refuse(
                    trace, "I couldn't read that audio.", f"audio decode: {exc}", Stage.STT
                )

            with Timer() as timer:
                try:
                    transcript = self.stt.transcribe(audio, request.lang_hint, allow_live_stt)
                except Exception as exc:  # noqa: BLE001
                    trace.add(Stage.STT, timer_ms(timer), ok=False, note=type(exc).__name__)
                    trace.errors.append(str(exc))
                    return self._refuse(
                        trace, "I couldn't transcribe that audio.", str(exc), stage=None
                    )
            trace.add(
                Stage.STT,
                timer.ms,
                note="cache" if transcript.cached else "live",
            )
            query = transcript.text
        else:
            query = (request.text or "").strip()

        # --- input guard ---------------------------------------------------
        with Timer() as timer:
            verdict = self.input_guard.check(query)
        trace.add(Stage.GUARD_IN, timer.ms, ok=verdict.allowed, note=verdict.category)

        if not verdict.allowed:
            trace.decision = Decision.REFUSE
            return QueryResponse(
                answer=Answer(text=verdict.reason or "I can't help with that.", decision=Decision.REFUSE),
                transcript=transcript,
                chunks=[],
                trace=trace,
            )

        # --- retrieval -----------------------------------------------------
        with Timer() as timer:
            retrieval = self.retriever.retrieve(
                query,
                top_k=self.settings.rerank_top_n,
                candidate_k=self.settings.retrieve_top_k * 2,
                lang=None,
            )
        trace.add(Stage.ENCODE, retrieval.encode_ms)
        trace.add(
            Stage.RETRIEVE,
            timer.ms - retrieval.encode_ms,
            note=f"dense={retrieval.dense_ms:.1f}ms lexical={retrieval.lexical_ms:.1f}ms "
            f"fuse={retrieval.fuse_ms:.1f}ms",
        )
        trace.degradation = retrieval.degradation

        # --- relevance guard -----------------------------------------------
        with Timer() as timer:
            relevance = self.relevance_guard.check(query, retrieval.chunks)
        trace.add(Stage.GUARD_IN, timer.ms, ok=relevance.allowed, note="relevance")

        if not relevance.allowed:
            trace.decision = Decision.ABSTAIN
            return QueryResponse(
                answer=Answer(
                    text=relevance.reason or "I couldn't find anything about that.",
                    decision=Decision.ABSTAIN,
                    grounding_score=relevance.score,
                ),
                transcript=transcript,
                chunks=retrieval.chunks,
                trace=trace,
            )

        # --- answer --------------------------------------------------------
        with Timer() as timer:
            answer, route = self.router.route(query, retrieval.chunks)
        trace.add(
            Stage.GENERATE if route.value == "llm" else Stage.ROUTE,
            timer.ms,
            note=route.value,
        )
        trace.route = route

        # --- grounding guard -------------------------------------------------
        # Only answers need checking. An abstention is already the safe outcome,
        # and scoring it would waste time to reach the same conclusion.
        if answer.decision is Decision.ANSWER:
            with Timer() as timer:
                grounding = self.grounding_guard.check(answer.text, retrieval.chunks)
            trace.add(Stage.GUARD_OUT, timer.ms, ok=grounding.allowed, note=grounding.category)

            if not grounding.allowed:
                trace.decision = Decision.ABSTAIN
                return QueryResponse(
                    answer=Answer(
                        text="I found related passages, but not enough to answer that reliably.",
                        decision=Decision.ABSTAIN,
                        route=route,
                        grounding_score=grounding.score,
                    ),
                    transcript=transcript,
                    chunks=retrieval.chunks,
                    trace=trace,
                )
            answer.grounding_score = grounding.score

        trace.decision = answer.decision
        return QueryResponse(
            answer=answer, transcript=transcript, chunks=retrieval.chunks, trace=trace
        )

    def _refuse(
        self, trace: PipelineTrace, message: str, error: str, stage: Stage | None
    ) -> QueryResponse:
        if stage is not None:
            trace.add(stage, 0.0, ok=False, note=error[:80])
        trace.errors.append(error)
        trace.decision = Decision.REFUSE
        return QueryResponse(
            answer=Answer(text=message, decision=Decision.REFUSE), chunks=[], trace=trace
        )


def timer_ms(timer: Timer) -> float:
    """Timer.ms is only set on clean exit; report 0 when the block raised."""
    return getattr(timer, "ms", 0.0)
