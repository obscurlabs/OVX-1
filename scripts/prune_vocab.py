"""Shrink the query encoder by dropping vocabulary the corpus never uses.

multilingual-e5-small ships a 250k-token vocabulary covering 100+ languages.
This corpus is Hindi and English, and measurement says it touches 18.1% of it.

That matters far more than the file size suggests. The table is stored as
uint8 (96MB on disk), but it is per-tensor quantized and ONNX Runtime
materializes it as fp32 at inference:

    250,037 x 384 x 4 bytes = 384MB

which is the single largest allocation in the server - larger than the dense
index, the BM25 index and the interpreter combined. On a 512MB host it is the
difference between fitting and not fitting.

Why this is lossless rather than a quality trade. The scale and zero_point are
SCALARS shared by the whole table (verified: shape ()), so a retained row
dequantizes to bit-identical fp32 values before and after. And the Unigram
tokenizer picks a maximum-likelihood segmentation: deleting pieces that never
appear in any optimal segmentation cannot change the segmentation chosen for
text that did not use them. So embeddings do not move, and the existing index
stays valid - no re-embedding, no rebuild.

Kept, beyond the tokens actually observed:
  - the four specials at ids 0-3, so <s>/<pad>/</s>/<unk> keep their ids and
    the TemplateProcessing post-processor stays correct
  - <mask>, remapped to its new position
  - every single-character piece. This tokenizer has byte_fallback disabled, so
    a character with no piece becomes <unk> and its information is simply lost.
    Single-character pieces are the floor that keeps unseen input degrading
    into smaller pieces rather than into nothing.

Usage:
    python scripts/prune_vocab.py                 # build + verify
    python scripts/prune_vocab.py --rescan        # ignore the cached id scan
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import onnx
import pyarrow.parquet as pq
from onnx import numpy_helper as nh
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from voicerag.config import Paths

EMBED_INITIALIZER = "embeddings.word_embeddings.weight_quantized"
SPECIAL_IDS = (0, 1, 2, 3)  # <s> <pad> </s> <unk>
SCAN_CACHE = Paths.cache / "used_token_ids.npy"


def scan_used_ids(tokenizer: Tokenizer, rescan: bool) -> np.ndarray:
    """Every vocabulary id the corpus and the query set actually produce."""
    if SCAN_CACHE.exists() and not rescan:
        ids = np.load(SCAN_CACHE)
        print(f"reusing cached scan: {len(ids):,} ids ({SCAN_CACHE})")
        return ids

    used = np.zeros(260_000, dtype=bool)

    def scan(texts: list[str], label: str) -> None:
        texts = [t for t in texts if isinstance(t, str) and t]
        for i in range(0, len(texts), 20_000):
            for enc in tokenizer.encode_batch(texts[i : i + 20_000]):
                used[enc.ids] = True
        print(f"  {label:22s} -> {used.sum():,} distinct ids")

    table = pq.read_table(Paths.chunk_meta, columns=["text", "context_text"])
    scan(table.column("text").to_pylist(), "chunk text")
    scan(table.column("context_text").to_pylist(), "chunk context_text")
    del table

    queries = pq.read_table(Paths.queries)
    for col in ("query_hi", "query_en", "answer_hi", "answer_en"):
        if col in queries.schema.names:
            scan(queries.column(col).to_pylist(), f"queries.{col}")

    ids = np.flatnonzero(used)
    SCAN_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.save(SCAN_CACHE, ids)
    return ids


def build_keep_set(vocab: list, used_ids: np.ndarray) -> list[int]:
    keep = set(int(i) for i in used_ids)
    keep.update(SPECIAL_IDS)

    # Robustness floor - see module docstring.
    singles = 0
    for idx, (piece, _score) in enumerate(vocab):
        # Metaspace marks word starts with U+2581; the piece is one real char.
        bare = piece.removeprefix("▁")
        if len(bare) == 1:
            keep.add(idx)
            singles += 1

    keep.add(len(vocab) - 1)  # <mask>, always last
    ordered = sorted(keep)
    print(f"  observed          : {len(used_ids):,}")
    print(f"  + single-char     : {singles:,} pieces considered")
    print(f"  = keeping         : {len(ordered):,} / {len(vocab):,} "
          f"({100 * len(ordered) / len(vocab):.1f}%)")
    return ordered


def prune_tokenizer(src: Path, dst: Path, keep: list[int]) -> None:
    spec = json.loads(src.read_text(encoding="utf-8"))
    vocab = spec["model"]["vocab"]

    old_to_new = {old: new for new, old in enumerate(keep)}
    spec["model"]["vocab"] = [vocab[i] for i in keep]

    assert spec["model"]["unk_id"] == 3, "unk_id moved; specials must stay at 0-3"
    for i in SPECIAL_IDS:
        assert old_to_new[i] == i, f"special {i} was renumbered to {old_to_new[i]}"

    for tok in spec.get("added_tokens", []):
        if tok["id"] not in old_to_new:
            raise ValueError(f"added token {tok['content']!r} was pruned")
        tok["id"] = old_to_new[tok["id"]]

    dst.write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")


def prune_onnx(src: Path, dst: Path, keep: list[int]) -> tuple[int, int]:
    model = onnx.load(str(src))
    target = next(
        (i for i in model.graph.initializer if i.name == EMBED_INITIALIZER), None
    )
    if target is None:
        raise ValueError(f"{EMBED_INITIALIZER} not in graph")

    table = nh.to_array(target)
    before = table.nbytes
    pruned = np.ascontiguousarray(table[keep])
    after = pruned.nbytes

    target.CopyFrom(nh.from_array(pruned, name=EMBED_INITIALIZER))
    onnx.save(model, str(dst))
    return before, after


def verify(old_dir: Path, new_dir: Path, n: int = 400) -> float:
    """Embeddings must not move. Anything above ~1e-6 invalidates the index."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from voicerag.pipeline.query_encoder import OnnxQueryEncoder

    table = pq.read_table(Paths.chunk_meta, columns=["text"])
    texts = table.column("text").to_pylist()[:: max(1, table.num_rows // n)][:n]
    del table

    queries = pq.read_table(Paths.queries)
    for col in ("query_hi", "query_en"):
        if col in queries.schema.names:
            texts += [t for t in queries.column(col).to_pylist()[:100] if t]

    old = OnnxQueryEncoder(old_dir)
    new = OnnxQueryEncoder(new_dir)

    worst = 0.0
    for i in range(0, len(texts), 32):
        batch = texts[i : i + 32]
        a = old.encode_batch(batch)
        b = new.encode_batch(batch)
        worst = max(worst, float(np.abs(a - b).max()))

    print(f"  compared {len(texts):,} texts (corpus + both query languages)")
    return worst


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rescan", action="store_true", help="ignore the cached id scan")
    ap.add_argument("--out", default=None, help="output encoder dir")
    args = ap.parse_args()

    src_dir = Paths.onnx_encoder
    dst_dir = Path(args.out) if args.out else src_dir.parent / "encoder_onnx_pruned"
    dst_dir.mkdir(parents=True, exist_ok=True)

    src_onnx = next(p for p in sorted(src_dir.glob("*.onnx")) if "quantize" in p.name)

    print("scanning corpus for used vocabulary")
    tokenizer = Tokenizer.from_file(str(src_dir / "tokenizer.json"))
    used_ids = scan_used_ids(tokenizer, args.rescan)

    print("\nbuilding keep set")
    spec = json.loads((src_dir / "tokenizer.json").read_text(encoding="utf-8"))
    keep = build_keep_set(spec["model"]["vocab"], used_ids)

    print("\nwriting pruned encoder")
    prune_tokenizer(src_dir / "tokenizer.json", dst_dir / "tokenizer.json", keep)
    before, after = prune_onnx(src_onnx, dst_dir / src_onnx.name, keep)
    print(f"  embedding table   : {before / 1e6:.1f}MB -> {after / 1e6:.1f}MB uint8")
    print(f"  fp32 at runtime   : {before * 4 / 1e6:.1f}MB -> {after * 4 / 1e6:.1f}MB")

    for extra in ("config.json", "tokenizer_config.json", "special_tokens_map.json"):
        if (src_dir / extra).exists():
            shutil.copy2(src_dir / extra, dst_dir / extra)

    print("\nverifying embeddings are unchanged")
    worst = verify(src_dir, dst_dir)
    print(f"  max abs difference: {worst:.3e}")

    if worst > 1e-6:
        print("\nFAIL: embeddings moved. The existing index would be invalid.")
        return 1
    print("\nOK: embeddings identical. Existing index remains valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
