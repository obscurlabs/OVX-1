"""List the repo's parquet layout.

The dataset exposes a single 'default' config with language as a row field, so
`load_dataset(..., 'hi')` is not available. If the parquet shards are named by
language we can point data_files at the Hindi shards only and skip ~50GB of
irrelevant download; if not, we must stream-and-filter.
"""

from __future__ import annotations

import re
from collections import Counter

from huggingface_hub import HfApi

from voicerag.config import get_settings


def main() -> None:
    settings = get_settings()
    api = HfApi(token=settings.hf_token or None)

    files = api.list_repo_files(settings.hf_dataset, repo_type="dataset")
    parquet = [f for f in files if f.endswith(".parquet")]

    print(f"total files: {len(files)}   parquet: {len(parquet)}\n")

    print("=== first 15 parquet paths ===")
    for f in parquet[:15]:
        print(" ", f)

    print("\n=== top-level directories ===")
    tops = Counter(f.split("/")[0] for f in parquet)
    for name, count in sorted(tops.items()):
        print(f"  {name:<30} {count} files")

    # Look for language codes anywhere in the paths.
    print("\n=== paths containing a Hindi marker ===")
    hindi = [f for f in parquet if re.search(r"hin|_hi[_/.]|/hi/", f, re.I)]
    for f in hindi[:20]:
        print(" ", f)
    if not hindi:
        print("  none -- language is not encoded in the file path")

    print(f"\n=== non-parquet files (first 20) ===")
    for f in [x for x in files if not x.endswith(".parquet")][:20]:
        print(" ", f)


if __name__ == "__main__":
    main()
