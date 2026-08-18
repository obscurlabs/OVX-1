"""Serving encoder cache behaviour without loading a real ONNX graph."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

from voicerag.pipeline.query_encoder import OnnxQueryEncoder


class StubEncoder(OnnxQueryEncoder):
    def __init__(self, cache_size: int = 2) -> None:
        self._cache = OrderedDict()
        self._cache_size = cache_size
        self.calls: list[str] = []

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        self.calls.extend(texts)
        return np.asarray([[float(len(texts[0]))]], dtype=np.float32)


def test_cache_normalizes_equivalent_queries_and_returns_a_copy():
    encoder = StubEncoder()

    first = encoder.encode("  caf\u00e9  ")
    first[0] = -1
    second = encoder.encode("cafe\u0301")

    assert encoder.calls == ["  caf\u00e9  "]
    assert second.tolist() == [8.0]


def test_cache_evicts_the_least_recently_used_vector():
    encoder = StubEncoder(cache_size=2)

    encoder.encode("first")
    encoder.encode("second")
    encoder.encode("first")  # refresh first, so second becomes the LRU
    encoder.encode("third")
    encoder.encode("second")

    assert encoder.calls == ["first", "second", "third", "second"]
