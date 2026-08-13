"""Publish index artifacts to a Hugging Face dataset repo.

GitHub rejects files over 100MB, and our serving artifacts total ~830MB. HF Hub
is built for exactly this, so the split is:

    GitHub   source, tests, benchmarks  (~120KB)
    HF Hub   indexes and ONNX encoder   (~830MB)

The Space downloads the artifacts on boot (see api.fetch_artifacts), which keeps
the container image small and lets the index be rebuilt and republished without
touching the application code.

vectors.f32.npy (1.7GB) is deliberately NOT uploaded: it exists only to rebuild
indexes without re-embedding, and the server never reads it.

    python scripts/upload_index.py --repo obscurlabs/ovx-1-index
"""

from __future__ import annotations

import argparse
import sys

from voicerag.config import Paths, get_settings

ARTIFACTS = (
    "dense_i8.usearch",
    "bm25.pkl",
    "lexical_rows.npy",
    "chunks_meta.parquet",
    "manifest.json",
)

ENCODER_FILES = (
    "model_quantized.onnx",
    "tokenizer.json",
    "config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", required=True, help="e.g. obscurlabs/ovx-1-index")
    p.add_argument("--private", action="store_true", help="create as a private dataset")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    settings = get_settings()

    if not settings.hf_token:
        print("HF_TOKEN is not set in .env")
        print("NOTE: uploading needs a WRITE token; the read-only one used for")
        print("downloading the dataset will be rejected.")
        return 1

    missing = [n for n in ARTIFACTS if not (Paths.indexes / n).exists()]
    if missing:
        print(f"missing artifacts: {missing}")
        print("run scripts/build_index.py first")
        return 1

    total = 0
    print("=== to upload ===")
    for name in ARTIFACTS:
        size = (Paths.indexes / name).stat().st_size
        total += size
        print(f"  {size / 1024 / 1024:8.1f} MB  {name}")
    for name in ENCODER_FILES:
        path = Paths.onnx_encoder / name
        if path.exists():
            size = path.stat().st_size
            total += size
            print(f"  {size / 1024 / 1024:8.1f} MB  encoder_onnx/{name}")
    print(f"  {'-' * 8}")
    print(f"  {total / 1024 / 1024:8.1f} MB  total\n")

    if args.dry_run:
        print("--dry-run: nothing uploaded")
        return 0

    from huggingface_hub import HfApi

    api = HfApi(token=settings.hf_token)

    print(f"ensuring dataset repo {args.repo} exists...")
    api.create_repo(
        repo_id=args.repo, repo_type="dataset", private=args.private, exist_ok=True
    )

    for name in ARTIFACTS:
        print(f"uploading {name} ...")
        api.upload_file(
            path_or_fileobj=str(Paths.indexes / name),
            path_in_repo=name,
            repo_id=args.repo,
            repo_type="dataset",
        )

    for name in ENCODER_FILES:
        path = Paths.onnx_encoder / name
        if not path.exists():
            continue
        print(f"uploading encoder_onnx/{name} ...")
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=f"encoder_onnx/{name}",
            repo_id=args.repo,
            repo_type="dataset",
        )

    print(f"\ndone: https://huggingface.co/datasets/{args.repo}")
    print(f"set VOICERAG_INDEX_REPO={args.repo} on the Space")
    return 0


if __name__ == "__main__":
    sys.exit(main())
