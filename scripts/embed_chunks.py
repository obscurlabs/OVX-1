"""Embed every chunk once, on the GPU, and persist raw vectors to disk.

Deliberately separated from index construction. Embedding 1.17M chunks costs
several minutes of GPU time; index *variants* (int8 vs binary quantization,
different connectivity) cost seconds to rebuild from cached vectors. Splitting
the two means the quantization comparison is cheap enough to actually run
instead of being asserted.

Output is a float32 memmap so downstream steps never load 1.8GB into RAM.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import pyarrow.parquet as pq

from voicerag.config import Paths, get_settings


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--limit", type=int, default=0, help="embed only N chunks (smoke test)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    settings = get_settings()
    Paths.ensure()

    if not Paths.chunks.exists():
        print(f"missing {Paths.chunks}; run scripts/build_chunks.py first")
        return 1

    table = pq.read_table(Paths.chunks, columns=["chunk_id", "text"])
    texts = table.column("text").to_pylist()
    chunk_ids = table.column("chunk_id").to_pylist()

    if args.limit:
        texts, chunk_ids = texts[: args.limit], chunk_ids[: args.limit]

    n = len(texts)
    print(f"embedding {n:,} chunks")

    from voicerag.index.encoder import E5Encoder

    encoder = E5Encoder(settings.embed_model, batch_size=args.batch_size)
    print(f"  model  : {settings.embed_model}")
    print(f"  device : {encoder.device}")
    print(f"  dim    : {encoder.dim}\n")

    vectors_path = Paths.indexes / "vectors.f32.npy"
    out = np.lib.format.open_memmap(
        vectors_path, mode="w+", dtype=np.float32, shape=(n, encoder.dim)
    )

    t0 = time.perf_counter()
    step = max(args.batch_size * 16, 4096)
    for start in range(0, n, step):
        end = min(start + step, n)
        out[start:end] = encoder.encode_passages(texts[start:end])

        done = end
        elapsed = time.perf_counter() - t0
        rate = done / max(elapsed, 1e-6)
        eta = (n - done) / max(rate, 1e-6)
        print(f"  {done:>9,}/{n:,}  {rate:>7,.0f}/s  eta {eta / 60:>5.1f}m", flush=True)

    out.flush()
    elapsed = time.perf_counter() - t0

    # chunk_id order is the index's row order; persisted so lookups cannot drift.
    (Paths.indexes / "chunk_ids.json").write_text(json.dumps(chunk_ids), encoding="utf-8")

    size_mb = vectors_path.stat().st_size / 1024 / 1024
    print(f"\nembedded {n:,} chunks in {elapsed / 60:.1f}m ({n / elapsed:,.0f}/s)")
    print(f"wrote {vectors_path} ({size_mb:.0f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
