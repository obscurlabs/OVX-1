"""Guardrails: deciding when NOT to answer.

Requirement 6 asks for handling of off-topic queries, unsafe input, and answers
not grounded in retrieved context. Three gates, cheapest first, so a request
that should be refused never pays for retrieval or generation:

    InputGuard      before retrieval  - unsafe content, injection, junk input
    RelevanceGuard  after retrieval   - is anything actually relevant here?
    GroundingGuard  after generation  - is the answer supported by the context?

A note on why RelevanceGuard does not simply threshold cosine similarity.
E5 embeddings are strongly anisotropic - measured mean cosine to the corpus
centroid is 0.867 - so every vector is similar to every other vector and an
irrelevant chunk still scores ~0.85. An absolute threshold would therefore
either accept everything or reject everything. The usable signals are relative:

  * lexical evidence, where BM25's zero is meaningful (no shared terms at all)
  * the margin between the best hit and the rest of the candidate set, which
    collapses when nothing in the corpus is genuinely about the query

GroundingGuard is deliberately lexical rather than model-based. A second LLM
call to check the first would blow the latency budget and could hallucinate its
own verdict. Checking that the answer's content actually appears in the
retrieved text is cheap, deterministic, and catches the failure that matters
most: invented specifics, especially numbers.
"""

from __future__ import annotations

import re
import unicodedata

from voicerag.contracts import GuardVerdict, RetrievedChunk

# --------------------------------------------------------------------------
# Input guard
# --------------------------------------------------------------------------

# A deliberately narrow list covering unambiguous requests for harm. This is a
# first line, not a complete safety classifier: it targets clear cases and
# accepts that a determined adversary can phrase around it. The stronger
# protection is structural - the system only ever answers from a retrieved
# MS MARCO passage, so it has no capability to offer beyond that corpus.
_UNSAFE_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bhow (?:to|do i) (?:make|build|construct)\s+(?:a\s+)?(?:bomb|explosive|nerve agent)",
        r"\b(?:synthesi[sz]e|manufacture)\s+(?:meth|methamphetamine|fentanyl|sarin|ricin)",
        r"\bhow (?:to|do i) (?:kill|murder|poison)\s+(?:someone|a person|my)",
        r"\bchild\s+(?:porn|sexual abuse)",
        r"\bhow (?:to|do i) (?:hack|break into)\s+(?:someone|somebody|my (?:neighbou?r|ex))",
    )
]

# Attempts to override the system prompt. Relevant because the transcript is
# untrusted text that we place into a prompt.
_INJECTION_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"ignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions",
        r"disregard\s+(?:the\s+)?(?:system|previous)\s+(?:prompt|instructions)",
        r"you\s+are\s+now\s+(?:a|an|in)\b",
        r"\breveal\s+(?:your\s+)?(?:system\s+prompt|instructions)",
        r"pretend\s+(?:to\s+be|you\s+are)\b",
    )
]

_WORD = re.compile(r"[a-z0-9]+|[ऀ-ॿ]+")
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")

MIN_QUERY_CHARS = 3
MIN_QUERY_WORDS = 1


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


class InputGuard:
    """Pre-retrieval checks on the transcript."""

    def check(self, text: str) -> GuardVerdict:
        stripped = (text or "").strip()

        if len(stripped) < MIN_QUERY_CHARS or len(_tokens(stripped)) < MIN_QUERY_WORDS:
            return GuardVerdict(
                allowed=False,
                category="empty",
                reason="I didn't catch that. Could you say it again?",
            )

        # Transcription of silence or noise often yields punctuation or repeated
        # single characters; answering those wastes a full pipeline run.
        letters = [c for c in stripped if unicodedata.category(c).startswith("L")]
        if not letters:
            return GuardVerdict(
                allowed=False,
                category="empty",
                reason="I couldn't make out any speech in that recording.",
            )

        for pattern in _UNSAFE_PATTERNS:
            if pattern.search(stripped):
                return GuardVerdict(
                    allowed=False,
                    category="unsafe",
                    reason="I can't help with that request.",
                )

        for pattern in _INJECTION_PATTERNS:
            if pattern.search(stripped):
                return GuardVerdict(
                    allowed=False,
                    category="injection",
                    reason="I can only answer questions about the documents I have access to.",
                )

        return GuardVerdict(allowed=True, category="ok")


# --------------------------------------------------------------------------
# Relevance guard
# --------------------------------------------------------------------------


class RelevanceGuard:
    """Post-retrieval check: does the corpus actually cover this question?

    Thresholds here are calibrated empirically (scripts/calibrate_guards.py)
    against in-corpus gold queries versus deliberately out-of-corpus ones,
    rather than guessed.
    """

    def __init__(
        self,
        min_lexical_overlap: float = 0.15,
        min_margin: float = 0.06,
        min_chunks: int = 1,
    ) -> None:
        self.min_lexical_overlap = min_lexical_overlap
        self.min_margin = min_margin
        self.min_chunks = min_chunks

    def check(self, query: str, chunks: list[RetrievedChunk]) -> GuardVerdict:
        if len(chunks) < self.min_chunks:
            return GuardVerdict(
                allowed=False,
                category="off_topic",
                reason="I couldn't find anything about that in my documents.",
                score=0.0,
            )

        overlap = self._term_overlap(query, chunks)
        margin = self._score_margin(chunks)

        # Either signal alone is weak, so require both to be poor before
        # refusing. Cross-lingual queries legitimately have zero term overlap
        # (Hindi query, English passage), which is why overlap cannot veto alone.
        if overlap < self.min_lexical_overlap and margin < self.min_margin:
            return GuardVerdict(
                allowed=False,
                category="off_topic",
                reason="I couldn't find anything about that in my documents.",
                score=round(max(overlap, margin), 4),
            )

        return GuardVerdict(allowed=True, category="ok", score=round(max(overlap, margin), 4))

    @staticmethod
    def _term_overlap(query: str, chunks: list[RetrievedChunk]) -> float:
        """Fraction of query terms appearing anywhere in the top chunks."""
        query_terms = set(_tokens(query))
        if not query_terms:
            return 0.0
        context = set()
        for chunk in chunks[:5]:
            context.update(_tokens(chunk.text))
        return len(query_terms & context) / len(query_terms)

    @staticmethod
    def _score_margin(chunks: list[RetrievedChunk]) -> float:
        """How far the best hit stands above the rest of the candidates.

        When the corpus genuinely covers a query, a few chunks score clearly
        above the pack. When it does not, retrieval returns an undifferentiated
        smear of weak matches and this collapses toward zero.
        """
        scores = [c.fused_score or 0.0 for c in chunks]
        if len(scores) < 2:
            return 0.0
        best = scores[0]
        rest = sum(scores[1:]) / len(scores[1:])
        return (best - rest) / best if best > 0 else 0.0


# --------------------------------------------------------------------------
# Grounding guard
# --------------------------------------------------------------------------


class GroundingGuard:
    """Post-generation check: is every claim traceable to retrieved text?"""

    # Function words carry no factual content, so including them would inflate
    # the score of an answer whose actual claims were invented.
    STOPWORDS = frozenset(
        """a an the of in on at to for and or is are was were be been being it its this that
        these those with as by from what which who how why when where there here can could
        would should may might will shall do does did has have had not no yes if then than
        है हैं था थे थी का की के को में से पर और या एक यह वह जो कि तो ही भी हो होता होती
        """.split()
    )

    def __init__(self, threshold: float = 0.35, require_number_support: bool = True) -> None:
        self.threshold = threshold
        self.require_number_support = require_number_support

    def check(self, answer: str, chunks: list[RetrievedChunk]) -> GuardVerdict:
        if not answer.strip():
            return GuardVerdict(allowed=False, category="ungrounded", reason="empty answer", score=0.0)
        if not chunks:
            return GuardVerdict(
                allowed=False, category="ungrounded", reason="no supporting context", score=0.0
            )

        context = " ".join(c.text for c in chunks).lower()
        context_terms = set(_tokens(context))

        content_terms = [t for t in _tokens(answer) if t not in self.STOPWORDS]
        if not content_terms:
            return GuardVerdict(allowed=False, category="ungrounded", reason="no content", score=0.0)

        supported = sum(1 for t in content_terms if t in context_terms)
        score = supported / len(content_terms)

        # Numbers get a hard check rather than a proportional one. A fabricated
        # statistic is the most damaging and most common RAG failure, and it
        # barely moves a token-overlap score because it is one token among many.
        if self.require_number_support:
            answer_numbers = set(_NUMBER.findall(answer))
            context_numbers = set(_NUMBER.findall(context))
            invented = answer_numbers - context_numbers
            if invented:
                return GuardVerdict(
                    allowed=False,
                    category="ungrounded",
                    reason=f"answer contains figures absent from the context: {sorted(invented)}",
                    score=round(score, 4),
                )

        if score < self.threshold:
            return GuardVerdict(
                allowed=False,
                category="ungrounded",
                reason="the answer is not sufficiently supported by the retrieved passages",
                score=round(score, 4),
            )

        return GuardVerdict(allowed=True, category="ok", score=round(score, 4))
