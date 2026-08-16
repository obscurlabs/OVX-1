# Deploying OVX-1

Target: **Render free tier** — 512MB RAM, 0.1 CPU, no credit card, HTTPS included.

HTTPS matters more than it looks. `getUserMedia` only works in a secure context,
so the microphone is dead over plain `http://`. Render issues a certificate for
`*.onrender.com` automatically, so voice input works with no extra setup.

## What the free tier forced

Free Hugging Face Spaces were the original target. They no longer exist for
anything with a backend — verified directly against the API:

| SDK | Result |
|---|---|
| `docker` | 402 — PRO subscription required |
| `gradio` | 402 — PRO subscription required |
| `static` | 200 — created |

Static Spaces serve browser assets and run no Python at all, so a lighter model
would not have unlocked them. Render is the remaining host with a real Python
runtime, no card requirement, and free TLS.

Fitting 512MB took two changes, both measured, neither optional:

| | Before | After |
|---|---|---|
| Query encoder (resident) | 422.7 MB | **123.9 MB** |
| Chunk metadata (at load) | 2425 MB | **1.1 MB** |
| Whole service, steady state | — | **457.9 MB** |

The encoder shrank because `multilingual-e5-small` carries a 250k-token
vocabulary for 100+ languages and this corpus touches 18.1% of it. The embedding
table is per-tensor quantized, so dropping unused rows is exact — verified
bit-identical, which is why the index did not need rebuilding for it. Metadata
moved from ZSTD parquet to memory-mapped Arrow, so rows page in as retrieved
instead of decompressing 680MB of text at startup.

The deployed index is trimmed to 219,999 chunks (18.9%) by sampling whole
queries, so every retained query keeps its full passage set and all four
chunking granularities. See `scripts/trim_index.py` for why it is cut that way.

## 1. Publish the index artifacts

Already done — [`obscurlabs/ovx-1-index`](https://huggingface.co/datasets/obscurlabs/ovx-1-index),
public, 303MB. Only needed again after rebuilding the index:

```bash
python scripts/trim_index.py --target-chunks 220000
HF_TOKEN=<write-token> python scripts/upload_index.py --repo obscurlabs/ovx-1-index
```

A **write** token is required; the read-only one in `.env` is rejected. Revoke it
afterwards — the dataset is public, so the server needs no token to read it.

## 2. Create the service

1. Sign in at [render.com](https://render.com) with GitHub — no card.
2. **New → Blueprint**, pick this repo. `render.yaml` supplies everything:
   free plan, Singapore region, build and start commands, health check.
3. Set the three secrets when prompted (they are `sync: false`, so Render asks
   rather than reading them from the file):

| Variable | Value | Notes |
|---|---|---|
| `SARVAM_API_KEY` | your key | speech-to-text |
| `SARVAM_ALLOW_LIVE` | `1` for the demo | `0` restricts voice to cached transcripts |
| `GROQ_API_KEYS` | comma-separated | enables LLM escalation; optional |

First boot takes ~3 minutes: install, then a 303MB artifact download, then
warmup. Later restarts skip the download only if the disk survived — on free
instances it usually does not, so assume the full 3 minutes after a cold start.

## 3. Verify

```bash
curl https://<service>.onrender.com/api/health
```

```json
{ "ready": true, "chunks": 219999, "lexical_docs": 58685, "voice_enabled": true }
```

`ready: false` with an `error` field means the pipeline failed; the message says
why. `ready: false` with `error: null` means it is still loading — wait.

```bash
curl -X POST https://<service>.onrender.com/api/query \
  -H 'Content-Type: application/json' \
  -d '{"text":"what is a corporation"}'
```

Then open the root URL and confirm the microphone prompt appears — that is the
secure-context check passing.

## 4. Keep it warm — do not skip this

**Render free spins down after 15 minutes of inactivity, and a cold hit costs
~60 seconds.** The organisers run an eval loop of ~10 queries and score the
aggregate. One cold request inside that loop is arithmetically fatal:

| Scenario | Mean over 10 |
|---|---|
| Warm throughout | ~10 ms |
| One 1700 ms cold start | 179 ms |
| One 60 s Render cold start | **6,009 ms** |

Point a free uptime monitor at the health endpoint before judging opens:

- [cron-job.org](https://cron-job.org) or [UptimeRobot](https://uptimerobot.com) — no card
- URL `https://<service>.onrender.com/api/health`, every **10 minutes**

Free workspaces get 750 instance-hours per month and a month of continuous
running needs 744, so this fits — but only while this is the **only** free
service in the workspace. A second one will exhaust the quota and suspend both.

## 5. Cleaning up after submission

Everything heavy is regenerable and confined:

```
data/           index artifacts, ~2.5GB   (delete freely)
.cache/         HF + model caches         (delete freely)
```

Deleting the Render service and the HF dataset repo removes the hosted copies.

## Troubleshooting

**`ready: false`, `missing index artifacts ... VOICERAG_INDEX_REPO is not set`**
The env var did not reach the service. Confirm it in Render → Environment.

**`required encoder file model_quantized.onnx unavailable`**
The dataset repo is private, or the upload did not include `encoder_onnx/`.
Re-run `upload_index.py` and check the repo's file list.

**Out of memory / instance restarts under load**
Something reverted to a pre-trim artifact. Check `/api/health`: `chunks` must be
219999. If it reports 1165508 the full index is deployed and will not fit.

**Microphone button does nothing**
The page is being served over `http://`. Use the `https://` URL.

**Build fails on `pip install .`**
Confirm `PYTHON_VERSION` is `3.12.7`; `pyproject.toml` requires `>=3.12,<3.13`.
