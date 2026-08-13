"""Probe MSMARCO-XI's real structure before committing to a download.

The dataset is 55.6GB across 14 languages. Streaming a couple of rows first tells
us the actual config names and field layout, so ingestion is written against what
is really there rather than against the dataset card's prose.
"""

from __future__ import annotations

import json
import sys

from datasets import get_dataset_config_names, load_dataset

from voicerag.config import get_settings


def preview(value: object, width: int = 220) -> object:
    """Shorten long values so nested passage lists stay readable."""
    if isinstance(value, str):
        return value[:width] + ("..." if len(value) > width else "")
    if isinstance(value, list):
        return [preview(v, 90) for v in value[:3]] + ([f"...+{len(value) - 3} more"] if len(value) > 3 else [])
    if isinstance(value, dict):
        return {k: preview(v, 90) for k, v in value.items()}
    return value


def main() -> int:
    settings = get_settings()
    token = settings.hf_token or None
    if not token:
        print("WARNING: no HF_TOKEN in .env; proceeding anonymously (may be rate limited)\n")

    print(f"=== configs for {settings.hf_dataset} ===")
    try:
        configs = get_dataset_config_names(settings.hf_dataset, token=token)
    except Exception as exc:  # noqa: BLE001 - surfacing the real failure is the point
        print(f"FAILED to list configs: {type(exc).__name__}: {exc}")
        return 1

    print(f"{len(configs)} configs: {configs}\n")

    config = settings.hf_lang_config if settings.hf_lang_config in configs else configs[0]
    if config != settings.hf_lang_config:
        print(f"NOTE: '{settings.hf_lang_config}' not found; probing '{config}' instead\n")

    print(f"=== streaming {settings.hf_dataset} / {config} ===")
    for split in (settings.hf_split, "train"):
        try:
            ds = load_dataset(
                settings.hf_dataset, config, split=split, streaming=True, token=token
            )
            row = next(iter(ds))
        except Exception as exc:  # noqa: BLE001
            print(f"  split '{split}': unavailable ({type(exc).__name__}: {exc})")
            continue

        print(f"\n--- split '{split}' available. Field layout: ---")
        for key, value in row.items():
            print(f"\n  [{key}]  type={type(value).__name__}")
            print(f"    {json.dumps(preview(value), ensure_ascii=False, default=str)}")
        break

    return 0


if __name__ == "__main__":
    sys.exit(main())
