"""Build the serving indexes from cached vectors.

    python scripts/build_index.py --quantization b1 i8

Dense index: every chunk, at every granularity.

Lexical index: passage- and document-level chunks ONLY. Two reasons, one
principled and one practical. Sentence-level BM25 is noisy - a 15-word chunk has
too few terms for length normalization to behave, and near-duplicate sentences
from overlapping windows distort document frequency. And indexing all 1.17M
chunks lexically would cost ~240MB of postings, which the hosting budget cannot
absorb. Restricting to ~310k passage-level documents costs little recall because
the dense index already covers fine granularity.

Because the lexical index sees a subset, its internal doc ids are not chunk row
ids. The mapping is saved alongside it; fusion depends on getting this right.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from voicerag.config import Paths, get_settings
from voicerag.index.dense import DenseIndex
from voicerag.index.lexical import BM25Index

LEXICAL_STRATEGIES = {"passage", "parent_grouped"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--quantization",
        nargs="+",
        default=["b1", "i8"],
        choices=["f32", "i8", "b1"],
        help="build one index per listed quantization for comparison",
    )
    p.add_argument("--skip-lexical", action="store_true")
    p.add_argument("--skip-dense", action="store_true")
    p.add_argument("--connectivity", type=int, default=16)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    settings = get_settings()
    Paths.ensure()

    vectors_path = Paths.indexes / "vectors.f32.npy"
    if not vectors_path.exists():
        print(f"missing {vectors_path}; run scripts/embed_chunks.py first")
        return 1

    vectors = np.load(vectors_path, mmap_mode="r")
    n, dim = vectors.shape
    print(f"vectors: {n:,} x {dim}  ({vectors_path.stat().st_size / 1e6:.0f} MB on disk)\n")

    table = pq.read_table(Paths.chunks)
    if table.num_rows != n:
        print(f"MISMATCH: {table.num_rows:,} chunks vs {n:,} vectors -- re-run embed_chunks.py")
        return 1

    manifest: dict[str, object] = {
        "n_chunks": n,
        "dim": dim,
        "embed_model": settings.embed_model,
        "indexes": {},
    }

    # Merge with any previous run so dense and lexical can be built separately
    # (each stays inside a sane wall-clock budget).
    manifest_path = Paths.indexes / "manifest.json"
    if manifest_path.exists():
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["indexes"].update(previous.get("indexes", {}))
        except (OSError, ValueError):
            pass

    # --- dense ------------------------------------------------------------
    for quant in [] if args.skip_dense else args.quantization:
        print(f"=== dense index [{quant}] ===")
        t0 = time.perf_counter()
        index = DenseIndex.build(vectors, quantization=quant, connectivity=args.connectivity)
        build_s = time.perf_counter() - t0

        path = Paths.indexes / f"dense_{quant}.usearch"
        index.save(path)
        size_mb = path.stat().st_size / 1024 / 1024

        print(f"  built in {build_s / 60:.1f}m -> {path.name} ({size_mb:.0f} MB)\n")
        manifest["indexes"][f"dense_{quant}"] = {
            "path": path.name,
            "quantization": quant,
            "size_mb": round(size_mb, 1),
            "build_seconds": round(build_s, 1),
        }

    # --- lexical ----------------------------------------------------------
    if not args.skip_lexical:
        print("=== lexical index (BM25) ===")
        strategies = table.column("strategy").to_pylist()
        texts_all = table.column("text").to_pylist()

        rows = [i for i, s in enumerate(strategies) if s in LEXICAL_STRATEGIES]
        docs = [texts_all[i] for i in rows]
        print(f"  {len(docs):,} documents ({', '.join(sorted(LEXICAL_STRATEGIES))})")

        t0 = time.perf_counter()
        bm25 = BM25Index.build(docs)
        build_s = time.perf_counter() - t0

        bm25.save(Paths.bm25_index)
        np.save(Paths.indexes / "lexical_rows.npy", np.asarray(rows, dtype=np.int32))
        size_mb = Paths.bm25_index.stat().st_size / 1024 / 1024

        print(f"  built in {build_s / 60:.1f}m")
        print(f"  vocabulary : {len(bm25.vocab):,} terms")
        print(f"  postings   : {bm25.doc_ids.size:,}")
        print(f"  on disk    : {size_mb:.0f} MB   in memory ~{bm25.memory_mb:.0f} MB\n")

        manifest["indexes"]["bm25"] = {
            "path": Paths.bm25_index.name,
            "n_docs": len(docs),
            "vocab": len(bm25.vocab),
            "postings": int(bm25.doc_ids.size),
            "size_mb": round(size_mb, 1),
        }

    # --- serving metadata --------------------------------------------------
    # Only the columns the request path actually reads; text columns dominate
    # size, and the server never needs the rest.
    meta = table.select(
        ["chunk_id", "text", "context_text", "passage_id", "lang", "strategy", "query_id", "query_type"]
    )
    pq.write_table(meta, Paths.chunk_meta, compression="zstd")
    manifest["chunk_meta_mb"] = round(Paths.chunk_meta.stat().st_size / 1024 / 1024, 1)

    (Paths.indexes / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print("=== manifest ===")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
