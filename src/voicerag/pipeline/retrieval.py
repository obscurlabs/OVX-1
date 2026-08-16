"""Hybrid retrieval: dense + lexical, fused with Reciprocal Rank Fusion.

Why fuse at all. The two indexes fail in different, complementary ways:

  dense   understands meaning and crosses languages (a Hindi query can match an
          English passage) but is vague about exact tokens - product codes,
          numbers, rare proper nouns.
  lexical nails exact tokens and is effectively free, but cannot bridge scripts
          at all (see test_cross_script_query_finds_nothing).

RRF combines ranked lists without needing their scores to be comparable:

    score(d) = sum over lists of  1 / (k + rank(d))

That property matters here because cosine similarity and BM25 live on entirely
different scales, so any weighted sum of raw scores would need per-corpus
calibration that would silently rot as the corpus changes.

Degradation is deliberate: if either index fails, retrieval continues on the
other and records what happened in the trace, because a slightly worse answer
beats a 500.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from pyarrow import ipc

from voicerag.contracts import DegradationLevel, RetrievalResult, RetrievedChunk

# What _materialize actually reads. Anything else is build-time metadata and
# would only cost RAM at serve time; query_id in particular is used by the
# chunker to group passages and never again.
SERVING_COLUMNS = ("chunk_id", "text", "passage_id", "lang", "strategy", "query_type")


def merge_display_text(table: pa.Table) -> pa.Table:
    """Collapse text/context_text into the one column serving actually uses.

    Retrieval has always answered with `context_text or text` - the wider window
    when the chunker produced one, the chunk itself otherwise. Storing both
    means carrying 377MB + 340MB to ever read one of them, so the choice is made
    once at build time and only the winner is shipped.
    """
    if "context_text" not in table.schema.names:
        return table.select([c for c in SERVING_COLUMNS if c in table.schema.names])

    merged = pc.coalesce(table.column("context_text"), table.column("text"))
    table = table.set_column(table.schema.get_field_index("text"), "text", merged)
    return table.select(SERVING_COLUMNS)


def load_chunk_meta(index_dir: Path) -> pa.Table:
    """Chunk metadata, memory-mapped when the serving artifact is present.

    Arrow IPC is read through pa.memory_map, so the table's buffers point
    straight at the file and the resident cost is only the pages retrieval
    actually touches. It must be a SINGLE record batch: read_all() on a
    multi-batch file yields a chunked table, and .take() would then need
    combine_chunks(), which concatenates every buffer into fresh heap memory and
    discards the entire benefit of mapping it.

    The parquet branch is the local-development and test fallback. It is the
    2425MB path; scripts/build_serving_meta.py produces the mapped artifact.
    """
    arrow_path = index_dir / "chunks_meta.arrow"
    if arrow_path.exists():
        table = ipc.open_file(pa.memory_map(str(arrow_path), "r")).read_all()
        if table.num_rows and table.column(0).num_chunks != 1:
            # Fail loudly: silently combining here would restore the 2.4GB cost
            # in a place nobody would think to look.
            raise ValueError(
                f"{arrow_path.name} has {table.column(0).num_chunks} batches, expected 1 - "
                "rebuild it with scripts/build_serving_meta.py"
            )
        return table

    return merge_display_text(pq.read_table(index_dir / "chunks_meta.parquet")).combine_chunks()

# Standard RRF constant. Large enough that the top few ranks are not
# overwhelmingly dominant, small enough that deep ranks stop mattering.
RRF_K = 60

# Metadata-aware priors. Different granularities suit different questions, and
# these nudge fusion without overriding it.
STRATEGY_PRIOR = {
    "passage": 1.00,
    "sentence_window": 1.05,  # precise matches answer factoid questions best
    "fixed_overlap": 0.95,
    "parent_grouped": 0.90,  # broad context, useful but rarely the direct answer
    "semantic": 1.00,
}


class HybridRetriever:
    def __init__(
        self,
        dense_index,
        bm25,
        lexical_rows: np.ndarray,
        chunk_meta,
        query_encoder,
        dense_weight: float = 1.0,
        lexical_weight: float = 1.0,
    ) -> None:
        self.dense_index = dense_index
        self.bm25 = bm25
        # bm25 sees a subset of chunks, so its doc ids are not chunk row ids.
        self.lexical_rows = lexical_rows
        self.chunk_meta = chunk_meta
        self.query_encoder = query_encoder
        self.dense_weight = dense_weight
        self.lexical_weight = lexical_weight

    # -- loading ------------------------------------------------------------

    @classmethod
    def load(cls, index_dir: Path, encoder_dir: Path, quantization: str = "i8", dim: int = 384):
        from voicerag.index.dense import DenseIndex
        from voicerag.index.lexical import BM25Index
        from voicerag.pipeline.query_encoder import OnnxQueryEncoder

        index_dir = Path(index_dir)

        dense = DenseIndex.load(
            index_dir / f"dense_{quantization}.usearch",
            dim=dim,
            quantization=quantization,
            view=True,
        )
        bm25 = BM25Index.load(index_dir / "bm25.pkl")
        lexical_rows = np.load(index_dir / "lexical_rows.npy")
        chunk_meta = load_chunk_meta(index_dir)
        encoder = OnnxQueryEncoder(encoder_dir)

        return cls(dense, bm25, lexical_rows, chunk_meta, encoder)

    # -- retrieval ----------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 50,
        candidate_k: int | None = None,
        lang: str | None = None,
    ) -> RetrievalResult:
        # Over-retrieve before fusion: a document ranked 40th by one index may be
        # 2nd by the other, and truncating early throws that signal away.
        candidate_k = candidate_k or top_k * 2

        dense_ranked: list[int] = []
        dense_scores: dict[int, float] = {}
        lexical_ranked: list[int] = []
        lexical_scores: dict[int, float] = {}
        errors: list[str] = []

        encode_ms = dense_ms = lexical_ms = fuse_ms = 0.0

        try:
            started = time.perf_counter()
            vector = self.query_encoder.encode(query)
            encode_ms = (time.perf_counter() - started) * 1000

            started = time.perf_counter()
            rows, sims = self.dense_index.search(vector, top_k=candidate_k)
            dense_ms = (time.perf_counter() - started) * 1000

            dense_ranked = [int(r) for r in rows]
            dense_scores = {int(r): float(s) for r, s in zip(rows, sims, strict=False)}
        except Exception as exc:  # noqa: BLE001
            errors.append(f"dense: {type(exc).__name__}: {exc}")

        try:
            started = time.perf_counter()
            doc_ids, scores = self.bm25.search(query, top_k=candidate_k)
            lexical_ms = (time.perf_counter() - started) * 1000

            lexical_ranked = [int(self.lexical_rows[d]) for d in doc_ids]
            lexical_scores = {
                int(self.lexical_rows[d]): float(s)
                for d, s in zip(doc_ids, scores, strict=False)
            }
        except Exception as exc:  # noqa: BLE001
            errors.append(f"lexical: {type(exc).__name__}: {exc}")

        degradation = DegradationLevel.FULL
        if not dense_ranked and not lexical_ranked:
            return RetrievalResult(chunks=[], degradation=DegradationLevel.LEXICAL_ONLY)
        if not dense_ranked:
            degradation = DegradationLevel.LEXICAL_ONLY
        elif not lexical_ranked:
            degradation = DegradationLevel.DENSE_ONLY

        started = time.perf_counter()
        fused = self._fuse(dense_ranked, lexical_ranked)
        rows = self._materialize(fused, dense_scores, lexical_scores, top_k, lang)
        fuse_ms = (time.perf_counter() - started) * 1000

        return RetrievalResult(
            chunks=rows,
            degradation=degradation,
            n_dense=len(dense_ranked),
            n_lexical=len(lexical_ranked),
            encode_ms=encode_ms,
            dense_ms=dense_ms,
            lexical_ms=lexical_ms,
            fuse_ms=fuse_ms,
        )

    def _fuse(self, dense_ranked: list[int], lexical_ranked: list[int]) -> list[tuple[int, float]]:
        scores: dict[int, float] = {}

        for rank, row in enumerate(dense_ranked):
            scores[row] = scores.get(row, 0.0) + self.dense_weight / (RRF_K + rank + 1)
        for rank, row in enumerate(lexical_ranked):
            scores[row] = scores.get(row, 0.0) + self.lexical_weight / (RRF_K + rank + 1)

        return sorted(scores.items(), key=lambda kv: -kv[1])

    def _materialize(
        self,
        fused: list[tuple[int, float]],
        dense_scores: dict[int, float],
        lexical_scores: dict[int, float],
        top_k: int,
        lang: str | None,
    ) -> list[RetrievedChunk]:
        if not fused:
            return []

        # Pull a surplus so language filtering and priors have room to reorder
        # without leaving us short of top_k.
        head = fused[: top_k * 3]
        rows = [row for row, _ in head]
        table = self.chunk_meta.take(rows).to_pylist()

        out: list[RetrievedChunk] = []
        for (row, fused_score), meta in zip(head, table, strict=False):
            if lang and meta["lang"] != lang:
                continue

            prior = STRATEGY_PRIOR.get(meta["strategy"], 1.0)
            out.append(
                RetrievedChunk(
                    chunk_id=meta["chunk_id"],
                    # Already resolved to context_text-or-text at build time.
                    text=meta["text"],
                    passage_id=meta["passage_id"],
                    lang=meta["lang"],
                    strategy=meta["strategy"],
                    granularity=meta["strategy"],
                    dense_score=dense_scores.get(row),
                    lexical_score=lexical_scores.get(row),
                    fused_score=fused_score * prior,
                    query_type=meta["query_type"],
                )
            )

        out.sort(key=lambda c: -(c.fused_score or 0.0))
        return self._dedupe(out)[:top_k]

    @staticmethod
    def _dedupe(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Collapse chunks that resolve to the same passage.

        Multi-granularity indexing means one passage can surface as an atomic
        chunk, two sentence windows and an overlap window simultaneously. Left
        alone, a single passage would crowd out the rest of the context window
        and give the answerer four copies of one fact instead of four facts.
        """
        seen: set[str] = set()
        unique: list[RetrievedChunk] = []
        for chunk in chunks:
            if chunk.passage_id in seen:
                continue
            seen.add(chunk.passage_id)
            unique.append(chunk)
        return unique
