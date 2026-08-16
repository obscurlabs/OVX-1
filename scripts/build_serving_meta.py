"""Build the memory-mappable chunk metadata the server reads.

chunks_meta.parquet is the build-time artifact: eight columns, ZSTD, two row
groups. Reading it costs 2425MB of RSS for 138MB of disk, because a read has to
decompress ~680MB of text and hand back Python-owned Arrow buffers. That single
allocation is larger than Render Free's entire 512MB.

This writes the serving form instead:

  - only the six columns _materialize reads (query_id is build-time only)
  - text and context_text collapsed into the one column retrieval answers with
  - uncompressed Arrow IPC, so pa.memory_map gives buffers backed by the file
  - exactly ONE record batch, so .take() never needs combine_chunks()

The file is larger on disk than the parquet and that is the point: uncompressed
is what makes it mappable, and pages arrive only when a row is retrieved.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from pyarrow import ipc

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from voicerag.config import Paths
from voicerag.pipeline.retrieval import SERVING_COLUMNS, merge_display_text


def build(src: Path, dst: Path) -> None:
    print(f"reading {src.name} ({src.stat().st_size / 1e6:.1f} MB)")
    table = pq.read_table(src)
    print(f"  rows    : {table.num_rows:,}")
    print(f"  columns : {table.schema.names}")

    table = merge_display_text(table).combine_chunks()
    print(f"  serving : {list(SERVING_COLUMNS)}")
    print(f"  in-memory: {table.nbytes / 1e6:.1f} MB")

    with ipc.new_file(pa.OSFile(str(dst), "wb"), table.schema) as writer:
        writer.write_table(table, max_chunksize=table.num_rows)

    print(f"wrote {dst.name} ({dst.stat().st_size / 1e6:.1f} MB)")

    check = ipc.open_file(pa.memory_map(str(dst), "r")).read_all()
    batches = check.column(0).num_chunks
    if batches != 1:
        raise SystemExit(f"FAIL: {batches} record batches, must be 1")
    if check.num_rows != table.num_rows:
        raise SystemExit(f"FAIL: {check.num_rows:,} rows, expected {table.num_rows:,}")
    print(f"verified: 1 record batch, {check.num_rows:,} rows, mappable")


if __name__ == "__main__":
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Paths.chunk_meta
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else Paths.chunk_meta_arrow
    build(src, dst)
