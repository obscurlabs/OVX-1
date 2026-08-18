"""Exact-query retrieval cache behaviour."""

from __future__ import annotations

import numpy as np
import pyarrow as pa

from voicerag.pipeline.retrieval import HybridRetriever


class FakeEncoder:
    def __init__(self) -> None:
        self.calls = 0

    def encode(self, query: str) -> np.ndarray:
        self.calls += 1
        return np.asarray([1.0, 0.0], dtype=np.float32)


class FakeDense:
    def __init__(self) -> None:
        self.calls = 0

    def search(self, vector: np.ndarray, top_k: int):
        self.calls += 1
        return np.asarray([0]), np.asarray([0.9])


class FakeBm25:
    def __init__(self) -> None:
        self.calls = 0

    def search(self, query: str, top_k: int):
        self.calls += 1
        return np.asarray([0]), np.asarray([1.0])


def make_retriever() -> tuple[HybridRetriever, FakeEncoder, FakeDense, FakeBm25]:
    encoder, dense, bm25 = FakeEncoder(), FakeDense(), FakeBm25()
    metadata = pa.table(
        {
            "chunk_id": ["c1"],
            "text": ["The answer is grounded in this passage."],
            "passage_id": ["p1"],
            "lang": ["en"],
            "strategy": ["passage"],
            "query_type": ["DESCRIPTION"],
        }
    )
    return HybridRetriever(dense, bm25, np.asarray([0]), metadata, encoder), encoder, dense, bm25


def test_exact_repeated_query_reuses_full_retrieval_result():
    retriever, encoder, dense, bm25 = make_retriever()

    first = retriever.retrieve("What is grounding?", top_k=1, candidate_k=2)
    second = retriever.retrieve("  what is grounding?  ", top_k=1, candidate_k=2)

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert (encoder.calls, dense.calls, bm25.calls) == (1, 1, 1)
    assert second.chunks[0] is not first.chunks[0]
    assert second.encode_ms == second.dense_ms == second.lexical_ms == second.fuse_ms == 0.0


def test_degraded_retrieval_is_not_cached():
    retriever, encoder, dense, bm25 = make_retriever()

    def broken_search(vector: np.ndarray, top_k: int):
        dense.calls += 1
        raise RuntimeError("dense offline")

    dense.search = broken_search
    retriever.retrieve("only lexical", top_k=1, candidate_k=2)
    retriever.retrieve("only lexical", top_k=1, candidate_k=2)

    assert encoder.calls == 2
    assert bm25.calls == 2
