"""Rebuild the serving indexes over a subset of the corpus.

Render Free gives 512MB. The fixed cost - interpreter, FastAPI, onnxruntime and
the pruned encoder - is 187MB, so everything that scales with the corpus has to
fit in roughly 240MB with headroom for the eval loop. At full size those parts
come to 1419MB (dense 592, chunk metadata 698, BM25 129), which sets the target
at about 17% of 1,165,508 chunks.

WHAT IS CUT, AND WHY IT IS CUT THIS WAY

The corpus is MS MARCO shaped: each query carries ~10 passages, and the chunker
expands each passage into several granularities. That gives two ways to shrink
it, and they are not equivalent.

  Drop granularities, keep every query.  Keeps corpus breadth, but collapses
    toward one chunk per passage - which is precisely the naive fixed-size
    indexing the brief asks us not to do. It would trade away the requirement we
    are strongest on.

  Drop queries, keep every granularity.  Costs breadth: a question about a
    dropped query's passages can no longer be answered. But every retained
    query keeps its full passage set and all four granularities, so retrieval
    behaves exactly as evaluated - just over a smaller world.

The second is chosen. Sampling is by query_id, never by chunk, because a
partially-retained query is the bad case: the passage holding the answer
disappears while the question still looks answerable. Whole queries in or out
means the system is either right or honestly out-of-corpus, and the guardrails
already handle out-of-corpus by refusing.

The full-size index remains reproducible from this repo; this produces the
deployable artifact.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from voicerag.config import Paths, get_settings
from voicerag.index.dense import DenseIndex
from voicerag.index.lexical import BM25Index
from voicerag.pipeline.retrieval import merge_display_text

LEXICAL_STRATEGIES = {"passage", "parent_grouped"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target-chunks", type=int, default=220_000)
    p.add_argument("--seed", type=int, default=20260822)
    p.add_argument("--connectivity", type=int, default=16)
    p.add_argument("--out", default=None, help="output dir (default: data/indexes/serving)")
    return p.parse_args()


def choose_queries(query_ids: list[str], target: int, seed: int) -> set[str]:
    """Pick whole queries at random until the chunk budget is spent."""
    counts: dict[str, int] = {}
    for qid in query_ids:
        counts[qid] = counts.get(qid, 0) + 1

    order = sorted(counts)  # deterministic before shuffling
    rng = np.random.default_rng(seed)
    rng.shuffle(order)

    keep: set[str] = set()
    total = 0
    for qid in order:
        if total + counts[qid] > target:
            continue
        keep.add(qid)
        total += counts[qid]
    return keep


def main() -> int:
    args = parse_args()
    settings = get_settings()
    out_dir = Path(args.out) if args.out else Paths.indexes / "serving"
    out_dir.mkdir(parents=True, exist_ok=True)

    vectors_path = Paths.indexes / "vectors.f32.npy"
    if not vectors_path.exists():
        print(f"missing {vectors_path}; nothing to subset")
        return 1

    vectors = np.load(vectors_path, mmap_mode="r")
    table = pq.read_table(Paths.chunks)
    if table.num_rows != vectors.shape[0]:
        print(f"MISMATCH: {table.num_rows:,} chunks vs {vectors.shape[0]:,} vectors")
        return 1
    print(f"source: {table.num_rows:,} chunks x {vectors.shape[1]} dims\n")

    # --- choose rows -------------------------------------------------------
    query_ids = table.column("query_id").to_pylist()
    keep_queries = choose_queries(query_ids, args.target_chunks, args.seed)
    mask = np.fromiter((q in keep_queries for q in query_ids), dtype=bool, count=len(query_ids))
    rows = np.flatnonzero(mask)

    print(f"queries : {len(keep_queries):,} of {len(set(query_ids)):,} kept")
    print(f"chunks  : {len(rows):,} of {table.num_rows:,} kept "
          f"({100 * len(rows) / table.num_rows:.1f}%)")

    subset = table.take(pa.array(rows))
    for name, col in (("strategy", "strategy"), ("lang", "lang")):
        grouped = subset.group_by(col).aggregate([(col, "count")])
        pairs = sorted(
            zip(grouped.column(col).to_pylist(), grouped.column(f"{col}_count").to_pylist()),
            key=lambda kv: -kv[1],
        )
        print(f"  by {name:9s}: " + ", ".join(f"{k}={v:,}" for k, v in pairs))
    print()

    manifest: dict[str, object] = {
        "n_chunks": len(rows),
        "n_queries": len(keep_queries),
        "source_chunks": table.num_rows,
        "source_queries": len(set(query_ids)),
        "dim": int(vectors.shape[1]),
        "embed_model": settings.embed_model,
        "trim_seed": args.seed,
        "indexes": {},
    }

    # --- dense -------------------------------------------------------------
    print("=== dense index [i8] ===")
    t0 = time.perf_counter()
    subset_vectors = np.ascontiguousarray(vectors[rows])
    index = DenseIndex.build(subset_vectors, quantization="i8", connectivity=args.connectivity)
    dense_path = out_dir / "dense_i8.usearch"
    index.save(dense_path)
    del subset_vectors
    size_mb = dense_path.stat().st_size / 1e6
    print(f"  built in {(time.perf_counter() - t0) / 60:.1f}m -> {size_mb:.0f} MB\n")
    manifest["indexes"]["dense_i8"] = {"path": dense_path.name, "size_mb": round(size_mb, 1)}

    # --- lexical -----------------------------------------------------------
    print("=== lexical index (BM25) ===")
    strategies = subset.column("strategy").to_pylist()
    texts_all = subset.column("text").to_pylist()
    lex_rows = [i for i, s in enumerate(strategies) if s in LEXICAL_STRATEGIES]
    docs = [texts_all[i] for i in lex_rows]
    print(f"  {len(docs):,} documents ({', '.join(sorted(LEXICAL_STRATEGIES))})")

    t0 = time.perf_counter()
    bm25 = BM25Index.build(docs)
    bm25.save(out_dir / "bm25.pkl")
    np.save(out_dir / "lexical_rows.npy", np.asarray(lex_rows, dtype=np.int32))
    size_mb = (out_dir / "bm25.pkl").stat().st_size / 1e6
    print(f"  built in {(time.perf_counter() - t0) / 60:.1f}m -> {size_mb:.0f} MB\n")
    manifest["indexes"]["bm25"] = {
        "path": "bm25.pkl",
        "n_docs": len(docs),
        "vocab": len(bm25.vocab),
        "size_mb": round(size_mb, 1),
    }
    del texts_all, docs

    # --- serving metadata --------------------------------------------------
    print("=== serving metadata ===")
    meta = merge_display_text(subset).combine_chunks()
    arrow_path = out_dir / "chunks_meta.arrow"
    with pa.ipc.new_file(pa.OSFile(str(arrow_path), "wb"), meta.schema) as writer:
        writer.write_table(meta, max_chunksize=meta.num_rows)
    size_mb = arrow_path.stat().st_size / 1e6
    print(f"  {arrow_path.name}: {size_mb:.0f} MB, 1 batch, {meta.num_rows:,} rows\n")
    manifest["chunk_meta_mb"] = round(size_mb, 1)

    # The pruned encoder travels with the index it was pruned against.
    enc_src = Paths.serving_encoder()
    enc_dst = out_dir / "encoder_onnx"
    if enc_src.exists():
        shutil.copytree(enc_src, enc_dst, dirs_exist_ok=True)
        enc_mb = sum(f.stat().st_size for f in enc_dst.rglob("*")) / 1e6
        print(f"encoder: {enc_src.name} -> {enc_dst.name} ({enc_mb:.0f} MB)")
        manifest["encoder"] = {"source": enc_src.name, "size_mb": round(enc_mb, 1)}

    # Which queries survived. Evaluation must run against these and not the full
    # 15k set: a dropped query is correctly refused as out-of-corpus, and scoring
    # that as a retrieval failure would measure the trim rather than the system.
    (out_dir / "kept_query_ids.json").write_text(
        json.dumps(sorted(keep_queries)), encoding="utf-8"
    )

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    total = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file()) / 1e6
    print(f"\ntotal deployable artifacts: {total:.0f} MB  ->  {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
