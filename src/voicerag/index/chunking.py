"""Multi-strategy chunking.

Requirement 2 asks for a *vast* chunking approach and explicitly rejects naive
fixed-size splitting. The honest difficulty with MSMARCO-XI is that its passages
are already short (~50-80 words), so plain fixed-size splitting is close to a
no-op: it would emit one chunk per passage and change nothing. Depth therefore
has to come from indexing the same corpus at several granularities and letting
retrieval pick the right one, rather than from cutting text more cleverly.

Five strategies, each earning its place:

  PASSAGE          Atomic passage. The baseline every other strategy is scored
                   against, and the natural unit for short factual answers.
  SENTENCE_WINDOW  Index single sentences (precise matching) but return the
                   sentence plus its neighbours (enough context to answer).
                   Decouples match granularity from answer granularity.
  FIXED_OVERLAP    Token window with stride, applied ONLY to the long tail of
                   passages above a threshold. Overlap stops a fact that
                   straddles a boundary from being lost by both chunks.
  SEMANTIC         Splits where meaning shifts, by measuring cosine distance
                   between consecutive sentence embeddings and cutting at
                   distance percentile breakpoints. Topic-aware, not length-aware.
  PARENT_GROUPED   Dataset-aware. MS MARCO ships ~10 passages per query_id, so
                   those form a natural document. Retrieval can match a precise
                   small chunk and expand to this parent for context.

Script handling matters here: Hindi terminates sentences with the danda (U+0964),
not the full stop, so a naive `.split(".")` silently fails to split Hindi at all
and produces one giant chunk per passage. That bug is why `split_sentences`
handles both scripts explicitly.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class ChunkStrategy(str, Enum):
    PASSAGE = "passage"
    SENTENCE_WINDOW = "sentence_window"
    FIXED_OVERLAP = "fixed_overlap"
    SEMANTIC = "semantic"
    PARENT_GROUPED = "parent_grouped"


class Granularity(str, Enum):
    SENTENCE = "sentence"
    PASSAGE = "passage"
    DOCUMENT = "document"


# Devanagari danda / double danda, plus Latin terminators. Keeping the terminator
# with the sentence (lookbehind) preserves text on rejoin.
_SENT_SPLIT = re.compile(r"(?<=[।॥.!?])\s+")
_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    return _WS.sub(" ", text).strip()


def _cap_length(sentence: str, max_words: int) -> list[str]:
    """Hard-split a runaway 'sentence' into word windows.

    Real corpus text sometimes contains no terminator at all (scraped tables,
    concatenated lists), which yields single 'sentences' of 1500+ words. Left
    alone these silently lose most of their content: the encoder truncates at
    512 tokens, so everything past the cutoff is indexed as if it did not exist.
    Splitting keeps the tail retrievable.
    """
    words = sentence.split()
    if len(words) <= max_words:
        return [sentence]
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]


def split_sentences(text: str, min_chars: int = 15, max_words: int = 120) -> list[str]:
    """Split on Devanagari and Latin terminators.

    Fragments shorter than `min_chars` are merged into the previous sentence:
    abbreviations and decimals ("Dr.", "3.5") otherwise produce useless slivers
    that pollute the sentence index. Units longer than `max_words` are hard-split
    so untermination cannot hide text from the index.
    """
    text = normalize(text)
    if not text:
        return []

    parts = [p.strip() for p in _SENT_SPLIT.split(text) if p.strip()]
    if not parts:
        return [text]

    merged: list[str] = []
    pending = ""  # a short fragment with nothing before it, waiting to attach forward
    for part in parts:
        if pending:
            part = f"{pending} {part}"
            pending = ""
        if len(part) < min_chars:
            if merged:
                merged[-1] = f"{merged[-1]} {part}"
            else:
                pending = part  # leading fragment, e.g. "Dr." -> join to what follows
            continue
        merged.append(part)

    if pending:
        if merged:
            merged[-1] = f"{merged[-1]} {pending}"
        else:
            merged.append(pending)

    capped: list[str] = []
    for sentence in merged:
        capped.extend(_cap_length(sentence, max_words))
    return capped


def word_count(text: str) -> int:
    """Whitespace word count.

    Adequate for both scripts here: Devanagari is space-separated between words,
    so this tracks true token count closely enough for chunk sizing decisions.
    """
    return len(text.split())


def _chunk_id(passage_id: str, strategy: ChunkStrategy, ordinal: int, text: str) -> str:
    """Deterministic id, so re-running indexing is idempotent and diffable."""
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=6).hexdigest()
    return f"{passage_id}:{strategy.value}:{ordinal}:{digest}"


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    text: str
    passage_id: str
    lang: str
    strategy: ChunkStrategy
    granularity: Granularity
    ordinal: int = 0
    # Text actually handed to the answerer. Differs from `text` for
    # SENTENCE_WINDOW, where we match narrow but answer wide.
    context_text: str | None = None
    parent_id: str | None = None
    query_id: str | None = None
    query_type: str | None = None
    is_selected: bool = False
    meta: dict[str, str] = field(default_factory=dict)

    @property
    def answer_text(self) -> str:
        return self.context_text or self.text

    @property
    def n_words(self) -> int:
        return word_count(self.text)


@dataclass(slots=True)
class ChunkConfig:
    # Only passages longer than this get windowed; below it, splitting destroys
    # more context than it buys.
    long_passage_words: int = 120
    fixed_window_words: int = 80
    fixed_stride_words: int = 56  # 30% overlap
    sentence_window_radius: int = 1  # neighbours joined into context_text
    semantic_percentile: float = 75.0  # cut at distances above this percentile
    semantic_min_sentences: int = 4  # below this, semantic splitting is noise
    min_chunk_words: int = 4


@dataclass(slots=True)
class SourcePassage:
    passage_id: str
    text: str
    lang: str
    query_id: str | None = None
    query_type: str | None = None
    is_selected: bool = False


# --------------------------------------------------------------------------
# Individual strategies
# --------------------------------------------------------------------------


def chunk_passage_atomic(p: SourcePassage, cfg: ChunkConfig) -> list[Chunk]:
    text = normalize(p.text)
    if word_count(text) < cfg.min_chunk_words:
        return []
    return [
        Chunk(
            chunk_id=_chunk_id(p.passage_id, ChunkStrategy.PASSAGE, 0, text),
            text=text,
            passage_id=p.passage_id,
            lang=p.lang,
            strategy=ChunkStrategy.PASSAGE,
            granularity=Granularity.PASSAGE,
            query_id=p.query_id,
            query_type=p.query_type,
            is_selected=p.is_selected,
        )
    ]


def chunk_sentence_window(p: SourcePassage, cfg: ChunkConfig) -> list[Chunk]:
    """Match on one sentence, answer from that sentence plus its neighbours."""
    sentences = split_sentences(p.text)
    if len(sentences) < 2:
        return []

    r = cfg.sentence_window_radius
    chunks: list[Chunk] = []
    for i, sent in enumerate(sentences):
        if word_count(sent) < cfg.min_chunk_words:
            continue
        window = " ".join(sentences[max(0, i - r) : min(len(sentences), i + r + 1)])
        chunks.append(
            Chunk(
                chunk_id=_chunk_id(p.passage_id, ChunkStrategy.SENTENCE_WINDOW, i, sent),
                text=sent,
                context_text=window,
                passage_id=p.passage_id,
                lang=p.lang,
                strategy=ChunkStrategy.SENTENCE_WINDOW,
                granularity=Granularity.SENTENCE,
                ordinal=i,
                query_id=p.query_id,
                query_type=p.query_type,
                is_selected=p.is_selected,
            )
        )
    return chunks


def chunk_fixed_overlap(p: SourcePassage, cfg: ChunkConfig) -> list[Chunk]:
    """Sliding token window, long passages only.

    Gated on length so we do not emit a duplicate of every short passage: for a
    60-word passage an 80-word window is just the passage again, which would
    inflate the index with exact duplicates and skew retrieval scores.
    """
    words = normalize(p.text).split()
    if len(words) <= cfg.long_passage_words:
        return []

    chunks: list[Chunk] = []
    step = max(1, cfg.fixed_stride_words)
    for ordinal, start in enumerate(range(0, len(words), step)):
        piece = words[start : start + cfg.fixed_window_words]
        if len(piece) < cfg.min_chunk_words:
            break
        text = " ".join(piece)
        chunks.append(
            Chunk(
                chunk_id=_chunk_id(p.passage_id, ChunkStrategy.FIXED_OVERLAP, ordinal, text),
                text=text,
                passage_id=p.passage_id,
                lang=p.lang,
                strategy=ChunkStrategy.FIXED_OVERLAP,
                granularity=Granularity.PASSAGE,
                ordinal=ordinal,
                query_id=p.query_id,
                query_type=p.query_type,
                is_selected=p.is_selected,
            )
        )
        if start + cfg.fixed_window_words >= len(words):
            break
    return chunks


def chunk_semantic(
    p: SourcePassage,
    cfg: ChunkConfig,
    embed_fn: Callable[[Sequence[str]], np.ndarray],
) -> list[Chunk]:
    """Cut where meaning shifts.

    Embed each sentence, measure cosine distance between consecutive sentences,
    and break at distances above a percentile threshold. A passage that stays on
    one topic yields one chunk; a passage that wanders yields several.

    `embed_fn` is injected rather than imported so this module stays free of
    torch and remains unit-testable with a stub.
    """
    sentences = split_sentences(p.text)
    if len(sentences) < cfg.semantic_min_sentences:
        return []

    vectors = embed_fn(sentences)
    vectors = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9)
    distances = 1.0 - np.sum(vectors[:-1] * vectors[1:], axis=1)
    if distances.size == 0:
        return []

    threshold = float(np.percentile(distances, cfg.semantic_percentile))
    breakpoints = [i + 1 for i, d in enumerate(distances) if d > threshold]

    bounds = [0, *breakpoints, len(sentences)]
    chunks: list[Chunk] = []
    for ordinal, (start, end) in enumerate(zip(bounds, bounds[1:], strict=False)):
        text = " ".join(sentences[start:end]).strip()
        if word_count(text) < cfg.min_chunk_words:
            continue
        chunks.append(
            Chunk(
                chunk_id=_chunk_id(p.passage_id, ChunkStrategy.SEMANTIC, ordinal, text),
                text=text,
                passage_id=p.passage_id,
                lang=p.lang,
                strategy=ChunkStrategy.SEMANTIC,
                granularity=Granularity.PASSAGE,
                ordinal=ordinal,
                query_id=p.query_id,
                query_type=p.query_type,
                is_selected=p.is_selected,
            )
        )
    return chunks


def chunk_parent_grouped(
    passages: Sequence[SourcePassage],
    cfg: ChunkConfig,
    max_words: int = 400,
) -> list[Chunk]:
    """Build one document-level chunk per query_id.

    MS MARCO's ~10 passages per query are retrieved-together candidates on a
    single topic, so concatenating them reconstructs a document the original
    corpus never stored explicitly. Gives the index a coarse tier for broad
    questions that no single passage answers, and a parent to expand into.
    """
    if not passages:
        return []

    query_id = passages[0].query_id
    if query_id is None:
        return []

    parts: list[str] = []
    total = 0
    for p in passages:
        text = normalize(p.text)
        n = word_count(text)
        if total + n > max_words:
            break
        parts.append(text)
        total += n

    if not parts:
        return []

    text = " ".join(parts)
    parent_id = f"doc:{query_id}"
    return [
        Chunk(
            chunk_id=_chunk_id(parent_id, ChunkStrategy.PARENT_GROUPED, 0, text),
            text=text,
            passage_id=parent_id,
            lang=passages[0].lang,
            strategy=ChunkStrategy.PARENT_GROUPED,
            granularity=Granularity.DOCUMENT,
            parent_id=parent_id,
            query_id=query_id,
            query_type=passages[0].query_type,
            is_selected=any(p.is_selected for p in passages),
            meta={"n_passages": str(len(parts))},
        )
    ]


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

DEFAULT_STRATEGIES = (
    ChunkStrategy.PASSAGE,
    ChunkStrategy.SENTENCE_WINDOW,
    ChunkStrategy.FIXED_OVERLAP,
)


def chunk_one(
    passage: SourcePassage,
    cfg: ChunkConfig | None = None,
    strategies: Iterable[ChunkStrategy] = DEFAULT_STRATEGIES,
    embed_fn: Callable[[Sequence[str]], np.ndarray] | None = None,
) -> list[Chunk]:
    """Apply the requested strategies to a single passage.

    Strategies self-skip when inapplicable (a 2-sentence passage produces no
    semantic chunks, a short passage produces no fixed-overlap chunks), so the
    index carries multiple granularities without duplicating short text.
    """
    cfg = cfg or ChunkConfig()
    out: list[Chunk] = []

    for strategy in strategies:
        if strategy is ChunkStrategy.PASSAGE:
            out.extend(chunk_passage_atomic(passage, cfg))
        elif strategy is ChunkStrategy.SENTENCE_WINDOW:
            out.extend(chunk_sentence_window(passage, cfg))
        elif strategy is ChunkStrategy.FIXED_OVERLAP:
            out.extend(chunk_fixed_overlap(passage, cfg))
        elif strategy is ChunkStrategy.SEMANTIC:
            if embed_fn is None:
                raise ValueError("SEMANTIC chunking requires embed_fn")
            out.extend(chunk_semantic(passage, cfg, embed_fn))
        elif strategy is ChunkStrategy.PARENT_GROUPED:
            # Needs sibling passages; handled by chunk_corpus, not per-passage.
            continue

    return out


def chunk_corpus(
    passages: Sequence[SourcePassage],
    cfg: ChunkConfig | None = None,
    strategies: Iterable[ChunkStrategy] = DEFAULT_STRATEGIES,
    embed_fn: Callable[[Sequence[str]], np.ndarray] | None = None,
) -> list[Chunk]:
    """Chunk a whole corpus, including the cross-passage PARENT_GROUPED tier."""
    cfg = cfg or ChunkConfig()
    strategies = list(strategies)

    chunks: list[Chunk] = []
    for p in passages:
        chunks.extend(chunk_one(p, cfg, strategies, embed_fn))

    if ChunkStrategy.PARENT_GROUPED in strategies:
        by_query: dict[str, list[SourcePassage]] = {}
        for p in passages:
            if p.query_id:
                by_query.setdefault(p.query_id, []).append(p)
        for group in by_query.values():
            chunks.extend(chunk_parent_grouped(group, cfg))

    return chunks


def strategy_stats(chunks: Sequence[Chunk]) -> dict[str, dict[str, float]]:
    """Per-strategy counts and size distribution.

    Feeds the comparison table in the README: the brief wants evidence that the
    strategies were evaluated, not merely implemented.
    """
    stats: dict[str, dict[str, float]] = {}
    for strategy in ChunkStrategy:
        subset = [c for c in chunks if c.strategy is strategy]
        if not subset:
            continue
        widths = [c.n_words for c in subset]
        stats[strategy.value] = {
            "count": float(len(subset)),
            "mean_words": float(np.mean(widths)),
            "p50_words": float(np.percentile(widths, 50)),
            "p95_words": float(np.percentile(widths, 95)),
            "min_words": float(np.min(widths)),
            "max_words": float(np.max(widths)),
        }
    return stats
