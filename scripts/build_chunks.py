"""Apply the chunking strategies to the ingested corpus.

Run with --limit first: chunk count drives index size, embedding time and the
Render memory budget, so it is worth measuring the multiplication factor on a
sample before committing to a full pass.

    python scripts/build_chunks.py --limit 5000
    python scripts/build_chunks.py --semantic
"""

from __future__ import annotations

import argparse
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq

from voicerag.config import Paths, get_settings
from voicerag.index.chunking import (
    ChunkConfig,
    ChunkStrategy,
    SourcePassage,
    chunk_corpus,
    strategy_stats,
    word_count,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--limit", type=int, default=0, help="only chunk N passages (0 = all)")
    p.add_argument(
        "--semantic",
        action="store_true",
        help="enable semantic chunking (loads the GPU encoder; long passages only)",
    )
    p.add_argument(
        "--semantic-min-words",
        type=int,
        default=80,
        help="semantic splitting is only worth its cost above this length",
    )
    p.add_argument("--dry-run", action="store_true", help="report stats without writing")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    settings = get_settings()
    Paths.ensure()

    if not Paths.passages.exists():
        print(f"missing {Paths.passages}; run scripts/ingest.py first")
        return 1

    table = pq.read_table(Paths.passages)
    rows = table.to_pylist()
    if args.limit:
        rows = rows[: args.limit]
    print(f"loaded {len(rows):,} passages\n")

    sources = [
        SourcePassage(
            passage_id=r["passage_id"],
            text=r["text"],
            lang=r["lang"],
            query_id=r["query_id"],
            query_type=r["query_type"],
            is_selected=r["is_selected"],
        )
        for r in rows
    ]

    cfg = ChunkConfig()
    strategies = [
        ChunkStrategy.PASSAGE,
        ChunkStrategy.SENTENCE_WINDOW,
        ChunkStrategy.FIXED_OVERLAP,
        ChunkStrategy.PARENT_GROUPED,
    ]

    t0 = time.perf_counter()
    chunks = chunk_corpus(sources, cfg, strategies)
    print(f"base strategies: {len(chunks):,} chunks in {time.perf_counter() - t0:.1f}s")

    # Semantic runs separately and only on long passages: it needs one embedding
    # per sentence just to decide where to cut, so applying it to 60-word
    # passages costs GPU time to reproduce what PASSAGE already emitted.
    if args.semantic:
        from voicerag.index.chunking import chunk_semantic
        from voicerag.index.encoder import E5Encoder

        long_sources = [s for s in sources if word_count(s.text) >= args.semantic_min_words]
        print(f"\nsemantic: {len(long_sources):,} passages over {args.semantic_min_words} words")

        if long_sources:
            t0 = time.perf_counter()
            encoder = E5Encoder(settings.embed_model)
            print(f"  encoder on {encoder.device}")
            embed_fn = encoder.embed_fn_for_chunking()

            added = 0
            for i, src in enumerate(long_sources):
                chunks.extend(chunk_semantic(src, cfg, embed_fn))
                added += 1
                if (i + 1) % 2000 == 0:
                    print(f"  {i + 1:,}/{len(long_sources):,}")
            print(f"  done in {time.perf_counter() - t0:.1f}s")

    print(f"\n=== {len(chunks):,} chunks from {len(sources):,} passages "
          f"(x{len(chunks) / max(1, len(sources)):.2f}) ===\n")

    stats = strategy_stats(chunks)
    header = f"{'strategy':<18}{'count':>10}{'mean_w':>9}{'p50_w':>8}{'p95_w':>8}{'max_w':>8}"
    print(header)
    print("-" * len(header))
    for name, s in sorted(stats.items(), key=lambda kv: -kv[1]["count"]):
        print(
            f"{name:<18}{int(s['count']):>10,}{s['mean_words']:>9.1f}"
            f"{s['p50_words']:>8.0f}{s['p95_words']:>8.0f}{s['max_words']:>8.0f}"
        )

    # Index footprint drives the hosting decision, so state it explicitly.
    n = len(chunks)
    print(
        f"\nprojected index: {n:,} vectors x {settings.embed_dim} dims"
        f"  -> fp32 {n * settings.embed_dim * 4 / 1e6:.0f} MB"
        f" | int8 {n * settings.embed_dim / 1e6:.0f} MB"
    )

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0

    t0 = time.perf_counter()
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "chunk_id": c.chunk_id,
                    "text": c.text,
                    "context_text": c.context_text,
                    "passage_id": c.passage_id,
                    "lang": c.lang,
                    "strategy": c.strategy.value,
                    "granularity": c.granularity.value,
                    "ordinal": c.ordinal,
                    "parent_id": c.parent_id,
                    "query_id": c.query_id,
                    "query_type": c.query_type,
                    "is_selected": c.is_selected,
                }
                for c in chunks
            ]
        ),
        Paths.chunks,
        compression="zstd",
    )
    size_mb = Paths.chunks.stat().st_size / 1024 / 1024
    print(f"\nwrote {Paths.chunks} ({size_mb:.1f} MB) in {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
