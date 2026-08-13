"""Validate the built indexes with real queries, in both languages.

Two concurrent build runs touched these files, so file existence is not proof of
integrity. This exercises the full path - encode, dense search, lexical search,
RRF fusion, metadata lookup - and prints what comes back so the results can be
judged by eye, not just by exit code.

Uses the torch encoder because the ONNX export does not exist yet; the vectors
are identical, only the runtime differs.
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
import pyarrow.parquet as pq

from voicerag.config import Paths, get_settings
from voicerag.index.dense import DenseIndex
from voicerag.index.lexical import BM25Index
from voicerag.pipeline.retrieval import HybridRetriever

QUERIES = [
    ("en", "what is a corporation"),
    ("en", "how long does it take to boil an egg"),
    ("en", "what causes photosynthesis in plants"),
    ("hi", "निगम क्या है"),
    ("hi", "प्रकाश संश्लेषण कैसे होता है"),
    ("hi", "भारत की राजधानी कौन सी है"),
]


class TorchEncoderAdapter:
    """Presents the E5 torch encoder through the serving encoder's interface."""

    def __init__(self, model_name: str) -> None:
        from voicerag.index.encoder import E5Encoder

        self.encoder = E5Encoder(model_name, batch_size=8)

    def encode(self, text: str) -> np.ndarray:
        return self.encoder.encode_queries([text])[0]


def main() -> int:
    settings = get_settings()

    print("=== loading indexes ===")
    t0 = time.perf_counter()

    dense_path = Paths.indexes / "dense_i8.usearch"
    dense = DenseIndex.load(dense_path, dim=settings.embed_dim, quantization="i8", view=True)
    print(f"  dense    : {len(dense):,} vectors  ({dense_path.stat().st_size / 1e6:.0f} MB)")

    bm25 = BM25Index.load(Paths.bm25_index)
    print(f"  lexical  : {bm25.n_docs:,} docs, {len(bm25.vocab):,} terms")

    lexical_rows = np.load(Paths.indexes / "lexical_rows.npy")
    chunk_meta = pq.read_table(Paths.chunk_meta)
    print(f"  metadata : {chunk_meta.num_rows:,} rows")
    print(f"  loaded in {time.perf_counter() - t0:.1f}s\n")

    # Integrity: the three structures must agree on how many chunks exist.
    problems = []
    if len(dense) != chunk_meta.num_rows:
        problems.append(f"dense {len(dense):,} != meta {chunk_meta.num_rows:,}")
    if lexical_rows.max() >= chunk_meta.num_rows:
        problems.append(f"lexical row {lexical_rows.max()} out of range")
    if bm25.n_docs != len(lexical_rows):
        problems.append(f"bm25 {bm25.n_docs:,} != lexical_rows {len(lexical_rows):,}")
    if problems:
        print("INTEGRITY PROBLEMS:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("integrity: dense, lexical and metadata all agree\n")

    print("=== loading encoder ===")
    encoder = TorchEncoderAdapter(settings.embed_model)
    retriever = HybridRetriever(dense, bm25, lexical_rows, chunk_meta, encoder)
    print(f"  {settings.embed_model} on {encoder.encoder.device}\n")

    # Warm up: first call pays CUDA kernel compilation and page-in costs that
    # would otherwise be misread as query latency.
    retriever.retrieve("warmup query", top_k=5, candidate_k=100)

    print("=== component latency (100 candidates) ===")
    for lang, query in QUERIES[:4]:
        t0 = time.perf_counter()
        bm25.search(query, top_k=100)
        lex_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        vector = encoder.encode(query)
        enc_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        dense.search(vector, top_k=100)
        dense_ms = (time.perf_counter() - t0) * 1000

        print(f"  [{lang}] encode {enc_ms:6.1f}ms | dense {dense_ms:6.1f}ms | lexical {lex_ms:6.1f}ms")
    print()

    latencies = []
    for lang, query in QUERIES:
        t0 = time.perf_counter()
        # Production over-retrieves: 100 candidates per index before fusion.
        # Retrieving only 10 from 1.17M vectors misses badly at int8 recall.
        result = retriever.retrieve(query, top_k=5, candidate_k=100)
        ms = (time.perf_counter() - t0) * 1000
        latencies.append(ms)

        print(f"--- [{lang}] {query!r}   {ms:.0f}ms   "
              f"({result.degradation.value}, dense={result.n_dense} lex={result.n_lexical}) ---")
        if not result.chunks:
            print("    NO RESULTS\n")
            continue
        for i, chunk in enumerate(result.chunks[:3], 1):
            snippet = chunk.text[:150].replace("\n", " ")
            print(f"  {i}. [{chunk.lang}/{chunk.strategy}] rrf={chunk.fused_score:.4f} "
                  f"dense={chunk.dense_score if chunk.dense_score is None else round(chunk.dense_score, 3)} "
                  f"lex={chunk.lexical_score if chunk.lexical_score is None else round(chunk.lexical_score, 1)}")
            print(f"     {snippet}")
        print()

    print("=== retrieval latency (torch encoder, includes GPU query encode) ===")
    print(f"  mean {np.mean(latencies):.0f}ms   min {np.min(latencies):.0f}ms   "
          f"max {np.max(latencies):.0f}ms")
    print("  (the served path uses ONNX-int8 on CPU; benchmarked separately)")

    # Repair the manifest entry the concurrent build dropped.
    manifest_path = Paths.indexes / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["indexes"]["bm25"] = {
        "path": Paths.bm25_index.name,
        "n_docs": bm25.n_docs,
        "vocab": len(bm25.vocab),
        "postings": int(bm25.doc_ids.size),
        "size_mb": round(Paths.bm25_index.stat().st_size / 1024 / 1024, 1),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nmanifest repaired: {sorted(manifest['indexes'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
