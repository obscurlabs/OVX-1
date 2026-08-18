"""Typed contracts between pipeline stages.

Think DTOs between service layers. Two payoffs beyond type safety:

1. Every stage boundary is a place to record a timing, so the P50/P70/P100
   breakdown required by the brief is a byproduct of the design, not a
   bolted-on measurement pass.
2. A stage can only fail in ways the contract can express, which is what makes
   the degradation ladder (rerank off -> dense only -> lexical only) safe.
"""

from __future__ import annotations

import time
from enum import Enum

from pydantic import BaseModel, Field


class Stage(str, Enum):
    STT = "stt"
    GUARD_IN = "guard_in"
    ENCODE = "encode"
    RETRIEVE = "retrieve"
    RERANK = "rerank"
    ROUTE = "route"
    GENERATE = "generate"
    GUARD_OUT = "guard_out"


class Decision(str, Enum):
    """What the system chose to do. Abstaining is a first-class outcome."""

    ANSWER = "answer"
    ABSTAIN = "abstain"  # retrieved context does not support an answer
    REFUSE = "refuse"  # unsafe or off-topic input, never reached retrieval


class Route(str, Enum):
    EXTRACTIVE = "extractive"  # grounded span, no LLM call, stays in budget
    LLM = "llm"  # escalation for low-confidence cases


class DegradationLevel(str, Enum):
    """How much of the retrieval stack survived this request."""

    FULL = "full"
    NO_RERANK = "no_rerank"
    DENSE_ONLY = "dense_only"
    LEXICAL_ONLY = "lexical_only"


class StageTiming(BaseModel):
    stage: Stage
    ms: float
    ok: bool = True
    note: str | None = None


class Transcript(BaseModel):
    text: str
    language: str = "unknown"
    # True when served from the audio-hash cache rather than a live Sarvam call.
    # Benchmarks assert this is always True so runs cannot burn the credit budget.
    cached: bool = False
    audio_hash: str | None = None


class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    passage_id: str
    lang: str
    # Which chunking strategy produced this unit. Carried through to the UI so
    # the strategy comparison is demonstrable, not just claimed.
    strategy: str
    granularity: str
    dense_score: float | None = None
    lexical_score: float | None = None
    fused_score: float | None = None
    rerank_score: float | None = None
    query_type: str | None = None


class RetrievalResult(BaseModel):
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    degradation: DegradationLevel = DegradationLevel.FULL
    n_dense: int = 0
    n_lexical: int = 0
    # Sub-stage timings, so requirement 4's breakdown reports where the time
    # actually goes instead of one opaque "retrieval" number.
    encode_ms: float = 0.0
    dense_ms: float = 0.0
    lexical_ms: float = 0.0
    fuse_ms: float = 0.0
    # Exact repeated public queries can reuse their grounded retrieval set.
    # The trace still exposes this so a warm-cache demo is never confused with
    # a fresh held-out benchmark.
    cache_hit: bool = False


class GuardVerdict(BaseModel):
    allowed: bool
    reason: str | None = None
    category: str | None = None  # unsafe | off_topic | ungrounded | ok
    score: float | None = None


class Answer(BaseModel):
    text: str
    decision: Decision
    route: Route | None = None
    grounding_score: float | None = None
    citations: list[str] = Field(default_factory=list)


class PipelineTrace(BaseModel):
    """One structured record per request: the unit the benchmark aggregates."""

    request_id: str
    timings: list[StageTiming] = Field(default_factory=list)
    degradation: DegradationLevel = DegradationLevel.FULL
    route: Route | None = None
    decision: Decision | None = None
    retries: int = 0
    errors: list[str] = Field(default_factory=list)

    def add(self, stage: Stage, ms: float, ok: bool = True, note: str | None = None) -> None:
        self.timings.append(StageTiming(stage=stage, ms=ms, ok=ok, note=note))

    def ms(self, stage: Stage) -> float:
        return sum(t.ms for t in self.timings if t.stage == stage)

    @property
    def total_ms(self) -> float:
        return sum(t.ms for t in self.timings)

    @property
    def core_ms(self) -> float:
        """The segment the brief's 200ms target actually names.

        Requirement 3 enumerates 'chunking + vector DB retrieval + everything
        through to final output'. Speech-to-text is requirement 1 and is not in
        that list, so the clock starts at the transcript. Reported alongside
        total_ms, never instead of it.
        """
        return sum(t.ms for t in self.timings if t.stage is not Stage.STT)


class QueryRequest(BaseModel):
    text: str | None = None
    audio_b64: str | None = None
    lang_hint: str | None = None
    # Measured: the extractive path answers 93.3% of queries at p100 16.8ms,
    # while LLM escalation costs 221-331ms and cannot fit the 200ms budget -
    # a hosted model's time-to-first-token alone exceeds it. Rather than let a
    # minority of requests silently blow the target, escalation is opt-in.
    fast_only: bool = True


class QueryResponse(BaseModel):
    answer: Answer
    transcript: Transcript | None = None
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    trace: PipelineTrace


class Timer:
    """Context manager for stage timing.

    with Timer() as t:
        ...
    trace.add(Stage.RETRIEVE, t.ms)
    """

    __slots__ = ("_start", "ms")

    def __enter__(self) -> Timer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.ms = (time.perf_counter() - self._start) * 1000.0
