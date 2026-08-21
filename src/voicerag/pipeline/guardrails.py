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
        # The gap between the verb and the object has to tolerate modifiers.
        # "how do i build a pipe bomb at home" previously slipped through: the
        # old pattern allowed only an article between "build" and "bomb", so
        # any adjective at all - pipe, nail, pressure cooker - defeated it. The
        # gap is bounded and excludes sentence punctuation so the match cannot
        # run across clauses into an unrelated noun.
        r"\bhow (?:to|do i) (?:make|build|construct|assemble)\b[\w\s'-]{0,30}?"
        r"\b(?:bombs?|explosives?|grenades?|molotov|napalm|nerve agents?|chemical weapons?)\b",
        r"\b(?:synthesi[sz]e|manufacture)\s+(?:meth|methamphetamine|fentanyl|sarin|ricin)",
        r"\bhow (?:to|do i) (?:kill|murder|poison)\s+(?:someone|a person|my)",
        r"\bchild\s+(?:porn|sexual abuse)",
        # "hack into my neighbour's wifi" previously slipped through: the verb
        # and preposition are separated ("hack ... into"), so the alternation
        # must not swallow the preposition.
        r"\bhow (?:to|do i) (?:hack|break)\s+into\s+(?:someone|somebody|my|his|her|their|another)",
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

        # A held key, a stuck mic or a hum transcribes to a run of one repeated
        # character. "aaaaaaaaaa" clears every check above - it is long enough,
        # it is one word, it is letters - and then retrieves real chunks and
        # gets answered from whichever passage happens to share the run. Every
        # token being a single repeated character is the signal; a query with
        # any genuine word in it keeps its distinct characters and passes.
        tokens = _tokens(stripped)
        if tokens and all(len(set(token)) == 1 for token in tokens):
            if max(len(token) for token in tokens) >= 3:
                return GuardVerdict(
                    allowed=False,
                    category="degenerate",
                    reason="I couldn't make out a question there. Could you try again?",
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
        lexical_index=None,
        min_idf_coverage: float = 0.75,
        distinctive_top_n: int = 0,
    ) -> None:
        self.min_lexical_overlap = min_lexical_overlap
        self.min_margin = min_margin
        self.min_chunks = min_chunks
        # Optional BM25 index, used for the vocabulary-coverage check below.
        self.lexical_index = lexical_index
        self.min_idf_coverage = min_idf_coverage
        # Disabled by calibration (scripts/calibrate_guards.py). The
        # distinctive-term check cost 13.7% false refusals at EVERY coverage
        # setting - overwhelmingly on Hindi queries, where inflection and
        # transliteration variance ("लिबेरिया" vs "लाइबेरिया") break exact token
        # matching - while IDF coverage alone already reached the same 3.8%
        # false-answer rate. Kept as a parameter so the sweep is reproducible.
        self.distinctive_top_n = distinctive_top_n

    def check(self, query: str, chunks: list[RetrievedChunk]) -> GuardVerdict:
        if len(chunks) < self.min_chunks:
            return GuardVerdict(
                allowed=False,
                category="off_topic",
                reason="I couldn't find anything about that in my documents.",
                score=0.0,
            )

        # Vocabulary coverage runs first and can refuse on its own, because it
        # catches a failure the other signals structurally cannot.
        #
        # Measured: "what is the population of the zorbian empire" scored 0.67
        # plain term overlap - "population" and "empire" match many passages -
        # and was answered from an unrelated census passage. Counting terms
        # equally lets common words carry a query whose *defining* term does not
        # exist in the corpus at all. Weighting by IDF fixes this: an
        # out-of-vocabulary term has maximal IDF, so a query hinging on one has
        # low coverage no matter how many stopwords-adjacent terms it matches.
        coverage = self._idf_coverage(query)
        if coverage is not None and coverage < self.min_idf_coverage:
            return GuardVerdict(
                allowed=False,
                category="off_topic",
                reason="I couldn't find anything about that in my documents.",
                score=round(coverage, 4),
            )

        # Second check: is the query's MOST DISTINCTIVE term actually present in
        # what we retrieved?
        #
        # Vocabulary coverage cannot catch "who won the interplanetary chess
        # championship" - every term exists in the corpus, it is the combination
        # that does not. But the retrieved passages are about junior chess
        # championships and contain no "interplanetary" anywhere, which is the
        # tell. A query whose defining term is missing from its own best matches
        # was answered by generic neighbours, not by relevant ones.
        if self.distinctive_top_n > 0 and self._distinctive_terms_missing(
            query, chunks, self.distinctive_top_n
        ):
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

    def _idf_coverage(self, query: str) -> float | None:
        """Share of the query's information content that the corpus can cover.

        Each content term is weighted by its IDF, with unknown terms taking the
        maximum. The result is the fraction of total query information that maps
        onto terms the corpus actually contains.

        Returns None when no lexical index is wired in, so the guard degrades to
        its other signals rather than failing.
        """
        if self.lexical_index is None:
            return None

        terms = [t for t in _tokens(query) if t not in GroundingGuard.STOPWORDS]
        if not terms:
            return None

        known_weight = 0.0
        total_weight = 0.0
        for term in terms:
            weight = self.lexical_index.idf(term)
            total_weight += weight
            if self.lexical_index.knows(term):
                known_weight += weight

        return known_weight / total_weight if total_weight > 0 else None

    def _distinctive_terms_missing(
        self, query: str, chunks: list[RetrievedChunk], top_n: int = 2
    ) -> bool:
        """True when none of the query's rarest terms appear in the retrieved text.

        Restricted to chunks in the query's own script. A Hindi query matched to
        an English passage has zero lexical overlap by construction, so applying
        this check across scripts would refuse every cross-lingual answer - the
        exact capability the system is meant to demonstrate.
        """
        if self.lexical_index is None:
            return False

        terms = [t for t in _tokens(query) if t not in GroundingGuard.STOPWORDS]
        if not terms:
            return False

        ranked = sorted(set(terms), key=self.lexical_index.idf, reverse=True)
        distinctive = ranked[:top_n]
        if not distinctive:
            return False

        query_is_devanagari = any("ऀ" <= c <= "ॿ" for c in query)
        same_script = [c for c in chunks[:5] if (c.lang == "hi") == query_is_devanagari]
        if not same_script:
            # Only cross-lingual matches: this check cannot judge them, so defer
            # to the other signals rather than refusing.
            return False

        context = set()
        for chunk in same_script:
            context.update(_tokens(chunk.text))

        return not any(term in context for term in distinctive)

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
