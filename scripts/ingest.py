"""Ingest MSMARCO-XI into a deduplicated passage pool plus a gold query set.

Observed schema (probe_dataset.py), which differs from the dataset card:

    source_lang   "eng_Latn"
    target_lang   "hin_Deva"
    query_id      int
    query_type    "DESCRIPTION" | "NUMERIC" | "ENTITY" | "PERSON" | "LOCATION"
    query         translated query          Eng_Query  original English query
    Answer        translated answer         Eng_Answer original English answer
    passages      { English_passages: [~10], Translated_passages: [~10],
                    is_selected: [~10] }

Three things this layout gives us that shape the whole build:

1. English and Hindi passages are PARALLEL - same content, same index position.
   Linking them by a shared pair_key makes cross-lingual retrieval measurable:
   ask in Hindi, retrieve the English passage, and we can prove it is the right one.
2. `is_selected` marks the gold passage per query, so retrieval recall is
   measurable rather than asserted.
3. `Answer` / `Eng_Answer` are gold answers, which is what makes the extractive
   fast path viable and gives the grounding check something to be scored against.

Two outputs:
    data/processed/passages.parquet  deduplicated corpus to index
    data/processed/queries.parquet   gold Q/A + gold passage ids (eval + benchmark)
"""

from __future__ import annotations

import hashlib
import re
import sys
import time
from collections import Counter

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from voicerag.config import Paths, get_settings
from voicerag.index.chunking import normalize

# Some queries arrive with leading punctuation artifacts: ". what is a corporation?"
_LEADING_JUNK = re.compile(r"^[^\wऀ-ॿ]+")


def clean_query(text: str) -> str:
    return _LEADING_JUNK.sub("", normalize(text or "")).strip()


def content_id(lang: str, text: str) -> str:
    """Stable id from content, so dedup and re-runs are deterministic."""
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest()
    return f"{lang}:{digest}"


def main() -> int:
    settings = get_settings()
    Paths.ensure()

    print(f"Downloading {settings.hf_data_file} from {settings.hf_dataset} ...")
    t0 = time.perf_counter()
    local_path = hf_hub_download(
        repo_id=settings.hf_dataset,
        filename=settings.hf_data_file,
        repo_type="dataset",
        token=settings.hf_token or None,
    )
    print(f"  -> {local_path}  ({time.perf_counter() - t0:.1f}s)")

    pf = pq.ParquetFile(local_path)
    total_rows = pf.metadata.num_rows
    print(f"  rows in shard: {total_rows:,}")
    print(f"  reading up to {settings.max_queries:,} queries\n")

    # passage_id -> record. Dedup is real here: MS MARCO reuses passages across
    # queries, and every duplicate we keep is wasted index space and a distorted
    # BM25 score (repeated text inflates document frequency).
    passages: dict[str, dict] = {}
    query_rows: list[dict] = []

    seen_queries = 0
    dup_hits = 0
    lang_counter: Counter[str] = Counter()
    type_counter: Counter[str] = Counter()
    target_lang = "unknown"

    t0 = time.perf_counter()
    for batch in pf.iter_batches(batch_size=512):
        if seen_queries >= settings.max_queries:
            break

        for row in batch.to_pylist():
            if seen_queries >= settings.max_queries:
                break

            payload = row.get("passages") or {}
            eng_list = payload.get("English_passages") or []
            tr_list = payload.get("Translated_passages") or []
            selected = payload.get("is_selected") or []
            if not eng_list:
                continue

            target_lang = row.get("target_lang") or target_lang
            query_id = str(row.get("query_id"))
            query_type = (row.get("query_type") or "UNKNOWN").upper()
            type_counter[query_type] += 1

            gold_en: list[str] = []
            gold_hi: list[str] = []

            for position, eng_text in enumerate(eng_list):
                eng_norm = normalize(eng_text or "")
                if not eng_norm:
                    continue

                is_selected = bool(selected[position]) if position < len(selected) else False
                # The English hash is the cross-lingual join key: the Hindi
                # passage at the same position is a translation of this text.
                pair_key = hashlib.blake2b(eng_norm.encode("utf-8"), digest_size=8).hexdigest()

                for lang, text in (
                    ("en", eng_norm),
                    ("hi", normalize(tr_list[position]) if position < len(tr_list) else ""),
                ):
                    if not text:
                        continue
                    pid = content_id(lang, text)
                    if is_selected:
                        (gold_en if lang == "en" else gold_hi).append(pid)

                    existing = passages.get(pid)
                    if existing is not None:
                        dup_hits += 1
                        # A passage can be gold for one query and filler for another.
                        # Keep the positive signal.
                        existing["is_selected"] = existing["is_selected"] or is_selected
                        continue

                    passages[pid] = {
                        "passage_id": pid,
                        "text": text,
                        "lang": lang,
                        "query_id": query_id,
                        "position": position,
                        "is_selected": is_selected,
                        "pair_key": pair_key,
                        "query_type": query_type,
                    }
                    lang_counter[lang] += 1

            query_rows.append(
                {
                    "query_id": query_id,
                    "query_hi": clean_query(row.get("query")),
                    "query_en": clean_query(row.get("Eng_Query")),
                    "answer_hi": normalize(row.get("Answer") or ""),
                    "answer_en": normalize(row.get("Eng_Answer") or ""),
                    "query_type": query_type,
                    "gold_en_ids": gold_en,
                    "gold_hi_ids": gold_hi,
                }
            )
            seen_queries += 1

            if seen_queries % 2000 == 0:
                print(f"  {seen_queries:,} queries -> {len(passages):,} unique passages")

            if len(passages) >= settings.max_passages:
                print(f"  hit max_passages cap ({settings.max_passages:,}), stopping")
                break

    elapsed = time.perf_counter() - t0
    print(f"\nParsed {seen_queries:,} queries in {elapsed:.1f}s")

    pq.write_table(
        pa.Table.from_pylist(list(passages.values())),
        Paths.passages,
        compression="zstd",
    )
    pq.write_table(
        pa.Table.from_pylist(query_rows),
        Paths.queries,
        compression="zstd",
    )

    n_gold = sum(1 for q in query_rows if q["gold_en_ids"] or q["gold_hi_ids"])
    size_mb = Paths.passages.stat().st_size / 1024 / 1024

    print("\n=== corpus ===")
    print(f"  target language     : {target_lang}")
    print(f"  unique passages     : {len(passages):,}")
    print(f"    english           : {lang_counter['en']:,}")
    print(f"    hindi             : {lang_counter['hi']:,}")
    print(f"  duplicates dropped  : {dup_hits:,}")
    print(f"  gold-marked passages: {sum(1 for p in passages.values() if p['is_selected']):,}")
    print(f"\n=== queries ===")
    print(f"  queries             : {len(query_rows):,}")
    print(f"  with a gold passage : {n_gold:,}  ({n_gold / max(1, len(query_rows)):.1%})")
    print(f"  by type             : {dict(type_counter.most_common())}")
    print(f"\n=== written ===")
    print(f"  {Paths.passages}  ({size_mb:.1f} MB)")
    print(f"  {Paths.queries}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
