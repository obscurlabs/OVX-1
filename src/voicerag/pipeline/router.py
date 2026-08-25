"""Answer routing: extractive fast path, LLM escalation only when needed.

This is the decision that makes the latency target reachable. A hosted LLM costs
200-600ms before it emits a first token, so routing every query through one puts
the 200ms budget out of reach regardless of how fast retrieval is.

MS MARCO questions are mostly answerable by a span already present in a
retrieved passage - that is how the dataset was built, with human annotators
marking the passage containing the answer. So the pipeline first tries to find
that span directly:

    high confidence -> return the grounded span   (~2ms, no LLM, no cost)
    low confidence  -> escalate to Groq           (slower, reported separately)

The extractive path is not a degraded mode. It is grounded by construction: the
answer is literally text from a retrieved passage, so it cannot hallucinate.
Escalation exists for questions needing synthesis across passages or rephrasing,
where an extracted sentence would read badly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from voicerag.contracts import Answer, Decision, Route, RetrievedChunk
from voicerag.index.chunking import split_sentences

_WORD = re.compile(r"[a-z0-9]+|[ऀ-ॿ]+")
_HAS_DIGIT = re.compile(r"\d")

# Question words carry intent but no content, so they must not count toward
# term overlap - otherwise "what is X" scores highly against any sentence
# containing "is".
_QUESTION_WORDS = frozenset(
    """what which who whom whose how why when where is are was were do does did
    can could would should the a an of in on at to for and or
    क्या कौन कब कहाँ कैसे क्यों किस है हैं था थे की का के को में से और या
    """.split()
)

# Expected answer shape by question form. Used to prefer sentences that could
# plausibly contain the answer, e.g. a "how many" answer should contain a number.
_NUMERIC_CUES = re.compile(
    r"\b(how many|how much|how long|how old|how far|what year|what time|"
    r"कितन|कब)\b", re.IGNORECASE
)
_PERSON_CUES = re.compile(r"\b(who|whose|whom|कौन)\b", re.IGNORECASE)
_LOCATION_CUES = re.compile(r"\b(where|which country|which city|कहाँ|कहां)\b", re.IGNORECASE)


def content_terms(text: str) -> list[str]:
    return [t for t in _WORD.findall(text.lower()) if t not in _QUESTION_WORDS]


@dataclass
class ExtractiveCandidate:
    text: str
    confidence: float
    chunk: RetrievedChunk


class ExtractiveAnswerer:
    """Select the best answer-bearing sentence from the retrieved chunks."""

    def __init__(self, max_sentences: int = 2, min_confidence: float = 0.45) -> None:
        self.max_sentences = max_sentences
        self.min_confidence = min_confidence

    def answer(self, query: str, chunks: list[RetrievedChunk]) -> ExtractiveCandidate | None:
        query_terms = set(content_terms(query))
        if not query_terms or not chunks:
            return None

        wants_number = bool(_NUMERIC_CUES.search(query))
        wants_person = bool(_PERSON_CUES.search(query))
        wants_place = bool(_LOCATION_CUES.search(query))

        best: ExtractiveCandidate | None = None

        # Only the strongest few chunks are worth scanning: beyond that,
        # retrieval confidence is low enough that extraction is guesswork.
        for rank, chunk in enumerate(chunks[:5]):
            sentences = split_sentences(chunk.text)
            if not sentences:
                continue

            for i, sentence in enumerate(sentences):
                sentence_terms = set(content_terms(sentence))
                if not sentence_terms:
                    continue

                covered = len(query_terms & sentence_terms) / len(query_terms)

                # Answer-shape bonus: a "how many" question is far better served
                # by a sentence that actually contains a figure.
                shape = 0.0
                if wants_number and _HAS_DIGIT.search(sentence):
                    shape = 0.15
                elif wants_person and re.search(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+", sentence):
                    shape = 0.10
                elif wants_place and re.search(r"\b(?:in|at|near|of)\s+[A-Z][a-z]+", sentence):
                    shape = 0.10

                # Retrieval already ranked these; trust it as a mild prior so a
                # weak sentence in the top chunk does not beat a strong one lower
                # down purely by position.
                rank_prior = 1.0 - (rank * 0.04)
                score = (covered + shape) * rank_prior

                if best is None or score > best.confidence:
                    window = " ".join(sentences[i : i + self.max_sentences]).strip()
                    best = ExtractiveCandidate(text=window, confidence=round(score, 4), chunk=chunk)

        return best


SYSTEM_PROMPT = (
    "You answer strictly from the provided context passages. "
    "Rules: use ONLY facts present in the context; never add outside knowledge; "
    "never invent numbers, names or dates. If the context does not contain the "
    'answer, reply with exactly: INSUFFICIENT_CONTEXT. '
    "Answer in the same language as the question. Be concise - two sentences at most."
)

INSUFFICIENT = "INSUFFICIENT_CONTEXT"


class AnswerRouter:
    """Chooses between the extractive path and LLM escalation."""

    def __init__(
        self,
        extractive: ExtractiveAnswerer | None = None,
        llm=None,
        max_context_chunks: int = 4,
        max_context_chars: int = 2400,
    ) -> None:
        self.extractive = extractive or ExtractiveAnswerer()
        self.llm = llm
        self.max_context_chunks = max_context_chunks
        self.max_context_chars = max_context_chars

    def route(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        allow_escalation: bool = True,
    ) -> tuple[Answer, Route]:
        # With no context there is nothing to be grounded in, and escalating
        # would hand the model an empty context - an open invitation to answer
        # from parametric memory, which is the exact failure the guardrails
        # exist to prevent. Abstain before any generation can happen.
        if not chunks:
            return (
                Answer(
                    text="I couldn't find an answer to that in my documents.",
                    decision=Decision.ABSTAIN,
                    route=None,
                    grounding_score=0.0,
                ),
                Route.EXTRACTIVE,
            )

        candidate = self.extractive.answer(query, chunks)

        if candidate is not None and candidate.confidence >= self.extractive.min_confidence:
            return (
                Answer(
                    text=candidate.text,
                    decision=Decision.ANSWER,
                    route=Route.EXTRACTIVE,
                    grounding_score=candidate.confidence,
                    citations=[candidate.chunk.chunk_id],
                ),
                Route.EXTRACTIVE,
            )

        # Escalate. If the LLM is unavailable - no keys, all keys cooling down,
        # upstream failure - fall back to the best extractive candidate rather
        # than failing the request, and let the grounding guard judge it.
        escalation_attempted = False
        if allow_escalation and self.llm is not None and getattr(self.llm, "enabled", False):
            escalation_attempted = True
            try:
                return self._escalate(query, chunks)
            except Exception:  # noqa: BLE001 - degradation is the intended behaviour
                pass

        # Degrading from a failed LLM call is not the same as never having
        # qualified, and the difference decides whether this is a fallback or a
        # fabrication.
        #
        # After a failed escalation the candidate is the best available answer
        # for a request that would otherwise return nothing, so returning it is
        # the intended degradation.
        #
        # Without escalation it has already scored below min_confidence, and
        # returning it anyway asserts something retrieval never supported. The
        # grounding guard cannot catch that: an extractive answer is a verbatim
        # span copied out of a retrieved chunk, so a traceability check passes
        # it by construction. fast_only=True is the default request shape, so
        # this branch is the common path, not an edge case - measured at a 0.920
        # false-confidence rate against MSMARCO-XI's unanswerable queries before
        # this gate existed.
        if candidate is not None and escalation_attempted:
            return (
                Answer(
                    text=candidate.text,
                    decision=Decision.ANSWER,
                    route=Route.EXTRACTIVE,
                    grounding_score=candidate.confidence,
                    citations=[candidate.chunk.chunk_id],
                ),
                Route.EXTRACTIVE,
            )

        return (
            Answer(
                text="I couldn't find an answer to that in my documents.",
                decision=Decision.ABSTAIN,
                route=Route.EXTRACTIVE,
                grounding_score=candidate.confidence if candidate is not None else 0.0,
            ),
            Route.EXTRACTIVE,
        )

    def _escalate(self, query: str, chunks: list[RetrievedChunk]) -> tuple[Answer, Route]:
        context, cited = self._build_context(chunks)
        result = self.llm.complete(
            system=SYSTEM_PROMPT,
            user=f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:",
            max_tokens=160,
        )

        text = result.text.strip()

        # The model was told to emit a sentinel when the context is inadequate.
        # Honouring it is how the LLM path participates in abstention rather
        # than being forced to produce something.
        if INSUFFICIENT in text.upper() or not text:
            return (
                Answer(
                    text="I couldn't find an answer to that in my documents.",
                    decision=Decision.ABSTAIN,
                    route=Route.LLM,
                    grounding_score=0.0,
                ),
                Route.LLM,
            )

        return (
            Answer(text=text, decision=Decision.ANSWER, route=Route.LLM, citations=cited),
            Route.LLM,
        )

    def _build_context(self, chunks: list[RetrievedChunk]) -> tuple[str, list[str]]:
        parts: list[str] = []
        cited: list[str] = []
        budget = self.max_context_chars

        for i, chunk in enumerate(chunks[: self.max_context_chunks], 1):
            text = chunk.text[:budget]
            if not text:
                break
            parts.append(f"[{i}] {text}")
            cited.append(chunk.chunk_id)
            budget -= len(text)
            if budget <= 0:
                break

        return "\n\n".join(parts), cited
