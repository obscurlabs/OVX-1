"""Diagnose dense-index recall: is the loss quantization, HNSW, or a bug?

Binary scored 15.6% recall@10 on clustered data, which is too low to be simple
quantization loss. Three hypotheses:

  A. usearch interprets `ndim` for b1 as BYTES, not bits, so declaring 384 makes
     it read past each 48-byte vector into adjacent memory.
  B. HNSW approximation, not quantization, is the real cost (f32 control tells us).
  C. Hamming distance needs different search-time expansion to behave.

Runs each configuration against exact brute-force ground truth.
"""

from __future__ import annotations

import numpy as np
from usearch.index import Index

from voicerag.index.dense import binarize, to_int8


def make_clustered(n: int = 2000, dim: int = 384, n_clusters: int = 40) -> np.ndarray:
    rng = np.random.default_rng(1)
    centers = rng.normal(size=(n_clusters, dim)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    members = np.repeat(centers, n // n_clusters, axis=0)
    noisy = members + rng.normal(scale=0.35, size=members.shape).astype(np.float32)
    return noisy / np.linalg.norm(noisy, axis=1, keepdims=True)


def exact_truth(vectors: np.ndarray, k: int) -> np.ndarray:
    sims = vectors @ vectors.T
    np.fill_diagonal(sims, -np.inf)
    return np.argsort(-sims, axis=1)[:, :k]


def measure(index, queries, truth, k: int, step: int, transform) -> float:
    hits = 0
    sampled = list(range(0, len(queries), step))
    for i in sampled:
        payload = transform(queries[i].reshape(1, -1))
        matches = index.search(payload, k, log=False)
        keys = np.asarray(matches.keys).ravel().tolist()
        hits += len(set(keys) & set(truth[i].tolist()))
    return hits / (len(sampled) * k)


def run(label, ndim, dtype, metric, vectors, truth, transform, expansion_search=64):
    try:
        index = Index(
            ndim=ndim,
            metric=metric,
            dtype=dtype,
            connectivity=16,
            expansion_add=128,
            expansion_search=expansion_search,
        )
        index.add(np.arange(len(vectors), dtype=np.int64), transform(vectors), log=False)
        recall = measure(index, vectors, truth, k=10, step=20, transform=transform)
        print(f"  {label:<46} recall@10 = {recall:6.2%}   mem = {index.memory_usage / 1e6:6.1f} MB")
    except Exception as exc:  # noqa: BLE001
        print(f"  {label:<46} FAILED: {type(exc).__name__}: {exc}")


def load_real(n: int) -> np.ndarray:
    """Sample real chunk embeddings.

    Synthetic clusters turned out to be a poor proxy: the f32 control only
    reached 89% recall, meaning the generated neighbourhoods were nearly
    degenerate. Real sentence embeddings are the only trustworthy input for this
    decision.
    """
    from voicerag.config import Paths

    path = Paths.indexes / "vectors.f32.npy"
    if not path.exists():
        raise SystemExit(f"missing {path}; run scripts/embed_chunks.py first")

    memmap = np.load(path, mmap_mode="r")
    # Strided sample spans the corpus rather than one contiguous region, which
    # would over-represent a single language and chunking strategy.
    stride = max(1, len(memmap) // n)
    return np.ascontiguousarray(memmap[::stride][:n], dtype=np.float32)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--real", type=int, default=0, help="use N real embeddings")
    args = parser.parse_args()

    if args.real:
        vectors = load_real(args.real)
        print(f"{len(vectors):,} REAL chunk embeddings, dim {vectors.shape[1]}\n")
    else:
        vectors = make_clustered()
        print(f"{len(vectors)} synthetic clustered vectors, dim {vectors.shape[1]}\n")

    truth = exact_truth(vectors, 10)

    print("=== control: is HNSW itself lossy here? ===")
    run("f32 / cos / ndim=384", 384, "f32", "cos", vectors, truth, lambda v: v)

    print("\n=== int8 ===")
    run("i8 / cos / ndim=384 (scaled x127)", 384, "i8", "cos", vectors, truth, to_int8)

    print("\n=== binary: hypothesis A (ndim in bits vs bytes) ===")
    run("b1 / hamming / ndim=384 (bits)", 384, "b1", "hamming", vectors, truth, binarize)
    run("b1 / hamming / ndim=48 (bytes)", 48, "b1", "hamming", vectors, truth, binarize)

    print("\n=== binary: hypothesis C (wider search expansion) ===")
    run("b1 / hamming / ndim=384 / exp=256", 384, "b1", "hamming", vectors, truth, binarize,
        expansion_search=256)

    print("\n=== binary: hypothesis D (anisotropy - vectors share a mean direction) ===")
    # Sentence embeddings are notoriously anisotropic: nearly all vectors point
    # into one narrow cone, so their SIGN patterns are almost identical and carry
    # little neighbourhood information. Subtracting the corpus mean before taking
    # signs is the standard remedy and costs nothing at query time.
    mean = vectors.mean(axis=0, keepdims=True)
    centered = vectors - mean
    centered /= np.linalg.norm(centered, axis=1, keepdims=True) + 1e-9

    cos_to_mean = float(np.mean(vectors @ (mean / np.linalg.norm(mean)).T))
    print(f"  mean cosine of corpus to its own centroid: {cos_to_mean:.3f}"
          f"   ({'ANISOTROPIC' if cos_to_mean > 0.5 else 'well spread'})")

    run("b1 / hamming / centered", 384, "b1", "hamming", centered, truth, binarize)

    packed_c = binarize(centered)
    bits_c = np.unpackbits(packed_c, axis=1).astype(np.int8)
    agree = bits_c @ bits_c.T + (1 - bits_c) @ (1 - bits_c).T
    np.fill_diagonal(agree, -1)
    approx_c = np.argsort(-agree, axis=1)[:, :10]
    sampled_c = range(0, len(vectors), 20)
    hits_c = sum(len(set(approx_c[i].tolist()) & set(truth[i].tolist())) for i in sampled_c)
    print(f"  brute-force binary, centered                   recall@10 = "
          f"{hits_c / (len(list(sampled_c)) * 10):6.2%}")

    print("\n=== sanity: does sign-only preserve neighbours at all? ===")
    # Brute-force Hamming, no HNSW involved. Isolates quantization from indexing.
    packed = binarize(vectors)
    bits = np.unpackbits(packed, axis=1).astype(np.int8)
    hamming = bits @ bits.T + (1 - bits) @ (1 - bits).T  # agreements
    np.fill_diagonal(hamming, -1)
    approx = np.argsort(-hamming, axis=1)[:, :10]

    sampled = range(0, len(vectors), 20)
    hits = sum(len(set(approx[i].tolist()) & set(truth[i].tolist())) for i in sampled)
    print(f"  brute-force binary (no HNSW)                   recall@10 = "
          f"{hits / (len(list(sampled)) * 10):6.2%}")


if __name__ == "__main__":
    main()
