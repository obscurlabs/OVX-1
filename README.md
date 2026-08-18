# OVX-1 — Voice-Enabled RAG

Ask a question by voice, in **Hindi or English**. The system transcribes it,
retrieves from a **1,165,508-chunk** index built over
[ai4bharat/MSMARCO-XI](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI),
and answers **only** from what it retrieved — or tells you it can't.

**Obscur Labs** — Yaksh Bambhroliya, Vansh Dobariya · HH Goa 2026 Task 2 · `#RAGInGoa`

**Live:** _<fill in after deploying — see [deploy/DEPLOY.md](deploy/DEPLOY.md)>_

---

## Headline numbers

| Latency (400 held-out queries) | | Guardrails | |
|---|---|---|---|
| **P50** | **9.43 ms** | Unsafe input blocked | **100%** |
| **P70** | **10.59 ms** | Prompt injection blocked | **100%** |
| **P100** | **42.52 ms** | Out-of-corpus refused | **100%** |
| Under 200 ms | **100%** | False answers | **0%** |

Measured on the real serving path — ONNX-int8 encoder on **CPU**, not the GPU
used for indexing. Reproduce with `python scripts/benchmark.py -n 400`.

### Live demo versus official benchmark

The UI includes a **warm-cache demo benchmark** so a reviewer can see the
running service respond on the deployed index. It reports P50, P70, and P100
over 30 fixed Hindi/English queries after the index and bounded query caches
are warm. It is intentionally labelled as a demo measurement.

The official evidence remains `scripts/benchmark.py -n 400`: held-out MS MARCO
queries, cache-independent methodology, and the full per-stage report. Never
compare the two as though they measure the same workload.

> **On P100.** It is the single worst request in the run, so it is inherently
> noisy: across runs it ranged **16.8 ms – 42.5 ms** depending on background
> load. We report the latest run rather than the best one. P50 and P70 are
> stable to within ~1 ms.

---

## What the 200 ms target actually covers

Requirement 3 enumerates *"chunking + vector DB retrieval + everything through
to final output"*. Speech-to-text is requirement 1 and is **not** in that list,
so we measure from the transcript onward and report STT separately.

This is not a convenient reading — it is the only honest one. Sarvam is a
network round trip of 300 ms–1.5 s. **No architecture puts hosted speech-to-text
inside a 200 ms budget.** Any submission claiming otherwise is either excluding
STT (as we do, explicitly) or not measuring it.

The same logic drove the biggest design decision in the project.

---

## The routing decision

A hosted LLM needs 200–600 ms before its *first token*. Routing every query
through one puts 200 ms out of reach no matter how fast retrieval is.

But MS MARCO was built by annotators marking **which passage contains the
answer** — so most questions are answerable by a span already sitting in a
retrieved passage:

```
high extractive confidence  →  return the grounded span     ~2 ms, no LLM
low  extractive confidence  →  escalate to Groq             220–330 ms
```

Measured split over 150 queries: **133 extractive, 10 escalated (6.7%)**.

The extractive path is not a degraded mode — it is grounded *by construction*,
because the answer is literally text from a retrieved passage and cannot be
hallucinated.

Escalation is **opt-in** (`fast_only: true` by default, a toggle in the UI).
Rather than let 7% of requests silently blow the target, the budget is a
guarantee you can switch off:

| Mode | Escalation | P50 | P70 | P100 | Under 200 ms |
|---|---|---|---|---|---|
| **Fast** (default) | none | **9.43 ms** | **10.59 ms** | **42.52 ms** | **100%** |
| Deep (opt-in) | 6.7% of queries | 12.32 ms | 15.55 ms | 1044.67 ms | ~93% |

Deep mode's P100 of 1,044 ms is a single request that hit a Groq rate limit and
rotated to another key — the harness recovering rather than failing. It is also
exactly why escalation is opt-in: one dependency on a free-tier API makes the
tail unbounded, and the fast path has no such dependency.

---

## Requirement 2 — chunking

The honest difficulty: **MS MARCO passages are already short** (~55 words
median). Fixed-size splitting is close to a no-op here — it emits one chunk per
passage and changes nothing. Depth had to come from indexing the same corpus at
several granularities and letting retrieval pick, not from cutting text more
cleverly.

Five strategies, each earning its place:

| Strategy | Chunks | Median words | Why |
|---|---:|---:|---|
| `sentence_window` | 839,316 | 15 | Index one sentence (precise match), return it **plus its neighbours** (enough context to answer). Decouples match granularity from answer granularity. |
| `passage` | 295,686 | 51 | Atomic baseline; the natural unit for short factual answers. |
| `fixed_overlap` | 15,508 | 80 | Sliding window with 30% overlap, applied **only** to passages over 120 words, so a fact straddling a boundary isn't lost by both chunks. |
| `parent_grouped` | 14,998 | 374 | Dataset-aware: MS MARCO ships ~10 passages per `query_id`, so concatenating them **reconstructs a document the corpus never stored**. |
| `semantic` | optional | — | Splits where meaning shifts, by cosine distance between consecutive sentence embeddings. Applied only to long passages, where it earns its GPU cost. |

**1,165,508 chunks from 295,765 passages (3.94×).**

Metadata is carried through indexing and used at retrieval: `query_type`
(DESCRIPTION / NUMERIC / ENTITY / PERSON / LOCATION), language, `is_selected`,
and the originating `query_id`. Strategy priors nudge fusion — sentence windows
score slightly higher because precise matches answer factoid questions best.

### Script handling is not a detail

Hindi terminates sentences with the danda `।`, not a full stop. A naive
`.split(".")` **does not split Hindi at all** and silently produces one giant
chunk per passage. It raises no error and the pipeline still "works" — it just
returns garbage. `tests/test_chunking.py` pins this down.

The full corpus also contained passages with **no terminator whatsoever** — one
"sentence" ran 1,524 words. E5 truncates at 512 tokens, so everything past the
cutoff was silently unretrievable. Units are now hard-split at 120 words, with a
test asserting no text is lost.

---

## Architecture

```
  mic ──► Sarvam STT (cache-first) ──┐
                                     │   ◄── 200 ms budget starts here
  text ──────────────────────────────┤
                                     ▼
                          InputGuard          unsafe · injection · junk
                                     ▼
                          ONNX-int8 encode                        3.9 ms
                                     ▼
              ┌──────────────────────┴───────────────────┐
       usearch HNSW (i8)                          BM25 inverted
       1.17M chunks, 592 MB                       310k docs, 102 MB
              └──────────────────────┬───────────────────┘
                          RRF fusion + strategy priors           5.0 ms
                                     ▼
                          RelevanceGuard      does the corpus cover this?
                                     ▼
                   ┌─────────────────┴─────────────────┐
          extractive span (93.3%)              Groq escalation (6.7%)
                   └─────────────────┬─────────────────┘
                                     ▼
                          GroundingGuard      is the answer supported?
                                     ▼
                            answer · abstain · refuse
```

**Why fuse dense with lexical?** They fail differently. Dense understands
meaning and crosses languages — a Hindi query can match an English passage — but
is vague about exact tokens. BM25 nails exact tokens but **cannot bridge scripts
at all**. RRF (`Σ 1/(60 + rank)`) combines ranked lists without needing their
scores to be comparable, which matters because cosine and BM25 live on entirely
different scales.

---

## Requirement 5 — the harness

Structured orchestration, not a raw prompt call:

- **Typed contracts** (`contracts.py`) between every stage, so a stage can only
  fail in ways the contract expresses — which is what makes degradation safe.
- **Per-stage timings** fall out of those boundaries, so requirement 4's
  breakdown is a byproduct of the design rather than a bolted-on measurement.
- **Groq key pool** — several free-tier accounts rotated as one. A 429 rotates
  to the next key instead of failing; each key has its own circuit breaker so
  one dead key doesn't slow every request by its full timeout; 401/403 parks a
  key for an hour. Backoff uses **full jitter**, because unjittered retries
  across a pool synchronise into the burst that caused the failure.
- **Degradation ladder** — `full → no_rerank → dense_only → lexical_only`,
  recorded in the trace. A slightly worse answer beats a 500.
- **Structured trace per request**, which the benchmark aggregates.

---

## Requirement 6 — knowing when not to answer

Three gates, cheapest first, so a refusal never pays for generation:

| Gate | Stage | Result |
|---|---|---|
| InputGuard | pre-retrieval | unsafe **100%**, injection **100%**, degenerate **100%** |
| RelevanceGuard | post-retrieval | out-of-corpus refused **100%** |
| GroundingGuard | post-generation | unsupported answers converted to abstention |

**False answer rate 0%. False refusal rate 6.7%.**
Reproduce: `python scripts/eval_guardrails.py -n 300`

### Why relevance can't threshold similarity

E5 embeddings are strongly **anisotropic** — measured mean cosine to the corpus
centroid is **0.867**. Every vector is similar to every other, and an irrelevant
chunk still scores ~0.85. An absolute threshold would accept everything or
reject everything.

The signal that works is **IDF-weighted vocabulary coverage**. Our first attempt
counted query terms equally, and it failed completely — 100% false-answer rate:

> *"what is the population of the **zorbian** empire"* scored 0.67 plain overlap
> because "population" and "empire" match many passages, and was answered from an
> unrelated census passage. The one word that defines the question was absent
> from the corpus entirely.

Weighting each term by its IDF — an out-of-vocabulary term is *maximally*
informative — fixed it. Note the grounding guard **structurally cannot** catch
this: those answers *are* grounded, just in the wrong passage.

### Numbers get a hard check

A fabricated statistic is the classic RAG failure and barely moves a token-overlap
score — it's one token among thirty. So any figure in the answer that does not
appear in the retrieved context is an automatic refusal, not a proportional
penalty.

---

## Engineering findings

Things measurement caught that assumption would not have.

**Binary quantization is unusable here — 8.8% recall@10.** Tempting at 8× smaller
than int8 under a 512 MB hosting budget. The cause is the same anisotropy: nearly
every vector has an identical sign pattern, so Hamming distance carries almost no
information. Centering — the textbook fix — only reached 37.1%. Ruled out on
evidence; int8 (84.8%) ships. See `scripts/diag_quantization.py`.

**`np.add.at` was a 60× performance bug.** It runs an unbuffered Python-level
loop; English queries hitting 100k-document posting lists took 5.6 **seconds**.
`np.bincount` does the same scatter-add in compiled code.

**One Arrow call was worth 7.6×.** `pq.read_table` returns a table backed by many
record batches, so `Table.take` had to resolve which batch each row index fell
into — ~60 ms per request, nine tenths of total latency. `combine_chunks()` at
startup made it O(k). **P50 70.23 ms → 9.18 ms.**

**Calibration overruled a design decision.** We added a "distinctive term must
appear in retrieved text" check. Sweeping it (`scripts/calibrate_guards.py`)
showed it cost **13.7% false refusals at every setting** — almost entirely on
Hindi, where inflection and transliteration variance break exact token matching —
while buying nothing IDF coverage didn't already deliver. It is disabled by
calibration, with the sweep kept as reproducible evidence.

**Cold start read 337 ms.** After idling, the OS evicted the memory-mapped index
pages and the first request paid disk reads. On a free host that idles between
visits, *the first request is the one a visitor judges you by* — a ~37×
misrepresentation. The app now warms up at boot; first query is back to 9.04 ms.

**A mislabelled test case.** Our out-of-corpus suite included "who is the current
president of wakanda" — but the corpus genuinely contains 27 documents mentioning
Wakanda (Marvel content in the web crawl). The guard was right to allow it; the
*test* was wrong. Corrected, and the false-answer rate went to 0%.

---

## Reproduce

```powershell
. .\env.ps1                                   # venv + caches pinned to this folder

python scripts/ingest.py                      # 296k deduplicated passages
python scripts/build_chunks.py                # 1.17M chunks, 5 strategies
python scripts/embed_chunks.py                # 9.1 min on an RTX 4050
python scripts/build_index.py --quantization i8
python scripts/export_onnx.py                 # + torch-vs-ONNX parity check

python scripts/benchmark.py -n 400            # P50 / P70 / P100
python scripts/eval_guardrails.py -n 300      # refusal behaviour
python -m pytest tests/ -q                    # 160 tests

python -m voicerag.api                        # http://localhost:7860
```

Reports land in `benchmarks/reports/`.

```
src/voicerag/
  contracts.py            typed stage boundaries + tracing
  config.py               all paths derived from project root
  index/                  chunking · encoder · dense (usearch) · lexical (BM25)
  pipeline/               stt · retrieval · router · guardrails · llm · orchestrator
  api.py                  FastAPI + artifact bootstrap + warmup
scripts/                  ingest → chunk → embed → index → benchmark → deploy
tests/                    160 tests
```

---

## Choices and limits

**Hindi + English, from `validation/hinval.parquet`.** The dataset ships 14
languages across 55.6 GB; the shards are per-language, so one language is a
475 MB download rather than a 55.6 GB one. Depth over breadth: cross-lingual
retrieval and Devanagari-aware chunking are demonstrated properly instead of
14 languages handled shallowly.

**Retrieval recall@5 is 56.5%** against gold passages in a 1.17M-chunk index.
Honest, and the clearest place to improve — a cross-encoder reranker would help,
but at ~30–50 ms per pair on CPU it does not fit the budget. Stated rather than
omitted.

**Deployment uses a trimmed 220k-chunk artifact.** The full 1.17M-chunk index
is the reproducible research/evaluation index; the Render-free deployment keeps
whole query groups and all retained chunking strategies so it fits inside 512 MB.
Its live-demo metrics are reported separately from the full-index evidence.

**No neural reranker**, for that reason. Fusion plus strategy priors does the
reordering.

**Speech-to-text is cache-first by necessity.** The Sarvam account holds **100
credits total**, so the STT layer caches by a hash of the raw audio and refuses
live calls unless explicitly authorised. A public URL gets crawled, and every
uncached recording spends a credit.

**Cross-lingual retrieval is built and guarded for** — the relevance guard
explicitly cannot veto on term overlap alone, since a Hindi query matching an
English passage has zero overlap by construction — but same-language matches
naturally outrank cross-language ones, so it is not the default behaviour.

---

## Stack

Sarvam Saaras v3 (STT, chosen over ElevenLabs because the corpus is Indic) ·
multilingual-e5-small (ONNX int8) · usearch HNSW · custom BM25 ·
Groq llama-3.1-8b-instant · FastAPI · Render.

Total running cost: **$0.**
