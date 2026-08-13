"""Calibrate the relevance guard instead of hand-tuning it.

Hand-picked thresholds produced a bad trade: the distinctive-term check cut
false answers from 58% to 33% but pushed false refusals from 0.8% to 4%, and
every new false refusal was a Hindi query. Devanagari is morphologically rich
and the translations vary in transliteration ("लिबेरिया" vs "लाइबेरिया"), so
exact token matching of rare terms is far less reliable in Hindi than in Latin
script - which means one global setting cannot serve both.

This sweeps the parameters over both suites and prints the trade-off curve, so
the operating point is chosen from evidence and the cost of the choice is
stated. Retrieval runs ONCE per query and the guard is then re-evaluated against
the cached chunks, so a full sweep costs one pass over the index rather than one
per configuration.

Two error rates, in tension:
    false refusal  legitimate question declined   (destroys usefulness)
    false answer   out-of-corpus question answered (the hallucination risk)
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pyarrow.parquet as pq

from voicerag.config import Paths, get_settings

# A wider out-of-corpus set than the smoke test: 12 queries is too few to
# calibrate against without overfitting to them.
OUT_OF_CORPUS = [
    "what is the population of the zorbian empire",
    "who won the interplanetary chess championship in 2043",
    "what is quantum flibbertigibbet theory",
    "how do you calibrate a nebulon flux capacitor",
    "what did the treaty of vondelmarch establish",
    "who is the current president of wakanda",
    "what is the melting point of unobtainium",
    "how many moons does the planet zephyria have",
    "what language do the thraxians speak",
    "when was the grand library of xanthos built",
    "how tall is the mount blimborn range",
    "what currency is used in the republic of narnovia",
    "who composed the symphony of the drelthi",
    "what is the average rainfall in glimmerhold",
    "how do quixnar batteries store energy",
    "what year did the vespertine dynasty fall",
    "who discovered the element frobnicium",
    "what is the capital city of eldoria",
    "how fast can a zargon lizard run",
    "what causes the blorxian tides",
    "ज़ोर्बियन साम्राज्य की जनसंख्या कितनी है",
    "थ्रैक्सियन लोग कौन सी भाषा बोलते हैं",
    "एल्डोरिया की राजधानी क्या है",
    "फ्रोबनिशियम तत्व की खोज किसने की",
    "ज़ार्गन छिपकली कितनी तेज़ दौड़ सकती है",
    "वेस्पर्टाइन राजवंश का पतन कब हुआ",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-n", "--in-corpus", type=int, default=300)
    p.add_argument("--seed", type=int, default=11)
    return p.parse_args()


def load_in_corpus(n: int, seed: int) -> list[str]:
    rows = pq.read_table(Paths.queries).to_pylist()
    labelled = [r for r in rows if r["gold_en_ids"] or r["gold_hi_ids"]]
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(labelled), size=min(n, len(labelled)), replace=False)

    out = []
    for i in picks:
        row = labelled[int(i)]
        text = row["query_hi"] if int(i) % 2 and row["query_hi"] else row["query_en"]
        if text.strip():
            out.append(text)
    return out


def main() -> int:
    args = parse_args()
    settings = get_settings()

    from voicerag.pipeline.guardrails import RelevanceGuard
    from voicerag.pipeline.retrieval import HybridRetriever

    print("=== loading ===")
    retriever = HybridRetriever.load(Paths.indexes, Paths.onnx_encoder, dim=settings.embed_dim)

    in_corpus = load_in_corpus(args.in_corpus, args.seed)
    print(f"  in_corpus={len(in_corpus)}  out_of_corpus={len(OUT_OF_CORPUS)}")

    # Retrieve once; every configuration is then evaluated on cached chunks.
    print("\n=== retrieving (once) ===")
    cached: dict[str, list] = {}
    for label, queries in (("in", in_corpus), ("out", OUT_OF_CORPUS)):
        for q in queries:
            cached[f"{label}:{q}"] = retriever.retrieve(
                q, top_k=settings.rerank_top_n, candidate_k=settings.retrieve_top_k * 2
            ).chunks
    print(f"  cached {len(cached)} retrievals")

    print("\n=== sweep ===")
    header = f"{'idf_cov':>8}{'distinct':>10}{'false_refuse':>14}{'false_answer':>14}{'sum':>8}"
    print(header)
    print("-" * len(header))

    results = []
    for coverage in (0.0, 0.45, 0.55, 0.65, 0.75, 0.85):
        for distinctive in (0, 1, 2):
            guard = RelevanceGuard(
                lexical_index=retriever.bm25,
                min_idf_coverage=coverage,
            )
            # top_n = 0 disables the distinctive-term check entirely.
            guard._distinct_top_n = distinctive

            refused_legit = 0
            for q in in_corpus:
                if not _allowed(guard, q, cached[f"in:{q}"], distinctive):
                    refused_legit += 1

            answered_ooc = 0
            for q in OUT_OF_CORPUS:
                if _allowed(guard, q, cached[f"out:{q}"], distinctive):
                    answered_ooc += 1

            fr = refused_legit / len(in_corpus)
            fa = answered_ooc / len(OUT_OF_CORPUS)
            results.append(
                {"idf_coverage": coverage, "distinctive_top_n": distinctive,
                 "false_refusal": round(fr, 4), "false_answer": round(fa, 4)}
            )
            print(f"{coverage:>8.2f}{distinctive:>10}{fr:>13.1%}{fa:>14.1%}{fr + fa:>8.3f}")

    # A false refusal is more damaging than a false answer here: refusing a real
    # question makes the demo look broken, while an occasional over-answer on a
    # fictional entity is a softer failure. Weight refusals accordingly.
    best = min(results, key=lambda r: r["false_refusal"] * 2 + r["false_answer"])
    print("\n=== recommended operating point ===")
    print(f"  min_idf_coverage  = {best['idf_coverage']}")
    print(f"  distinctive_top_n = {best['distinctive_top_n']}")
    print(f"  false refusal     = {best['false_refusal']:.1%}")
    print(f"  false answer      = {best['false_answer']:.1%}")
    print("  (objective weights a false refusal twice a false answer)")

    out_dir = Paths.root / "benchmarks" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "guard_calibration.json"
    path.write_text(
        json.dumps({"sweep": results, "recommended": best}, indent=2), encoding="utf-8"
    )
    print(f"\nwrote {path}")
    return 0


def _allowed(guard, query: str, chunks, distinctive_top_n: int) -> bool:
    """Evaluate the guard with the distinctive-term check set to top_n (0 = off)."""
    original = guard._distinctive_terms_missing

    if distinctive_top_n == 0:
        guard._distinctive_terms_missing = lambda q, c, top_n=2: False
    else:
        guard._distinctive_terms_missing = (
            lambda q, c, top_n=distinctive_top_n: original(q, c, top_n)
        )

    try:
        return guard.check(query, chunks).allowed
    finally:
        guard._distinctive_terms_missing = original


if __name__ == "__main__":
    sys.exit(main())
