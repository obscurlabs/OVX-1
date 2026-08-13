"""Dense vector index (usearch HNSW) with selectable quantization.

Quantization is the lever that decides where this can be hosted. For 1.17M
chunks at 384 dimensions:

    f32   1790 MB   exact, unhostable on a small instance
    i8     448 MB   near-exact, still too large beside Python + model + BM25
    b1      56 MB   1 bit per dimension, fully resident with room to spare

Binary quantization keeps only the sign of each dimension and compares with
Hamming distance. It loses accuracy, so the choice is made by measuring recall
against the gold set (scripts/eval_retrieval.py) rather than by assertion. The
loss is also partly recoverable: binary retrieves a wider candidate set cheaply,
and fusion with BM25 plus reranking reorders what actually matters.

Normalized inputs are assumed (E5Encoder normalizes), which is what makes the
sign-only representation and inner-product-as-cosine both valid.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

Quantization = Literal["f32", "i8", "b1"]

_METRIC = {"f32": "cos", "i8": "cos", "b1": "hamming"}
_DTYPE = {"f32": "f32", "i8": "i8", "b1": "b1"}


def binarize(vectors: np.ndarray) -> np.ndarray:
    """Pack sign bits: (n, dim) float -> (n, dim/8) uint8."""
    return np.packbits(vectors > 0, axis=1)


def to_int8(vectors: np.ndarray) -> np.ndarray:
    """Scale normalized floats to the full int8 range.

    usearch's i8 storage expects genuine int8 values. Passing normalized floats
    in [-1, 1] truncates them to -1/0/1 - a 3-level quantization that destroys
    retrieval while raising no error. Scaling by 127 first uses all 256 levels.
    """
    return np.clip(np.rint(vectors * 127.0), -127, 127).astype(np.int8)


class DenseIndex:
    def __init__(self, index, dim: int, quantization: Quantization) -> None:
        self.index = index
        self.dim = dim
        self.quantization = quantization

    @classmethod
    def build(
        cls,
        vectors: np.ndarray,
        quantization: Quantization = "i8",
        connectivity: int = 16,
        expansion_add: int = 128,
        batch_size: int = 100_000,
        progress: bool = True,
    ) -> DenseIndex:
        from usearch.index import Index

        n, dim = vectors.shape
        index = Index(
            ndim=dim,
            metric=_METRIC[quantization],
            dtype=_DTYPE[quantization],
            connectivity=connectivity,
            expansion_add=expansion_add,
        )

        # Batched so a 1.8GB memmap is never materialized in RAM at once.
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            payload = np.asarray(vectors[start:end], dtype=np.float32)
            if quantization == "b1":
                payload = binarize(payload)
            elif quantization == "i8":
                payload = to_int8(payload)
            index.add(np.arange(start, end, dtype=np.int64), payload, log=False)
            if progress:
                print(f"    added {end:,}/{n:,}", flush=True)

        return cls(index, dim, quantization)

    def search(self, query: np.ndarray, top_k: int = 50) -> tuple[np.ndarray, np.ndarray]:
        """Return (row_ids, similarities), best first.

        Distances are converted to similarities so callers never have to know
        whether the backing index used cosine or Hamming.
        """
        vector = np.asarray(query, dtype=np.float32).reshape(1, -1)
        if self.quantization == "b1":
            payload = binarize(vector)
        elif self.quantization == "i8":
            payload = to_int8(vector)
        else:
            payload = vector

        matches = self.index.search(payload, top_k, log=False)
        keys = np.asarray(matches.keys, dtype=np.int64).ravel()
        distances = np.asarray(matches.distances, dtype=np.float32).ravel()

        if self.quantization == "b1":
            # Hamming distance counts differing bits; normalize to [0, 1].
            similarity = 1.0 - (distances / float(self.dim))
        else:
            similarity = 1.0 - distances

        return keys, similarity

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.index.save(str(path))

    @classmethod
    def load(
        cls,
        path: Path,
        dim: int,
        quantization: Quantization = "i8",
        view: bool = True,
    ) -> DenseIndex:
        """Load an index.

        `view=True` memory-maps instead of reading into RAM: startup is near
        instant and resident memory tracks the working set rather than the file.
        That is what keeps a cold container from stalling on first request.
        """
        from usearch.index import Index

        index = Index(ndim=dim, metric=_METRIC[quantization], dtype=_DTYPE[quantization])
        if view:
            index.view(str(path))
        else:
            index.load(str(path))
        return cls(index, dim, quantization)

    def __len__(self) -> int:
        return len(self.index)

    @property
    def size_mb(self) -> float:
        return self.index.memory_usage / 1024 / 1024
