# Deploying OVX-1 to Hugging Face Spaces

The build splits across two hosts because of a hard constraint: GitHub rejects
files over 100MB, and the serving artifacts total ~830MB.

```
GitHub          source, tests, benchmarks          ~120 KB
HF dataset repo dense index, BM25, ONNX encoder    ~830 MB
HF Space        Docker container, pulls the above on boot
```

The Space downloads artifacts at startup instead of baking them into the image.
That keeps the image ~200MB and means the index can be rebuilt and republished
without rebuilding the container.

---

## Step 0 — you need a WRITE token

The token currently in `.env` is **read-only**, which is correct for downloading
the dataset but will be rejected when uploading.

1. https://huggingface.co/settings/tokens → **New token**
2. Type: **Write**
3. Keep it separate from the read token; you only need it for step 1.

```powershell
$env:HF_TOKEN_WRITE = "hf_xxxxxxxxxxxx"
```

---

## Step 1 — publish the index artifacts

```powershell
. .\env.ps1
python scripts/upload_index.py --repo obscurlabs/ovx-1-index --dry-run   # inspect
python scripts/upload_index.py --repo obscurlabs/ovx-1-index             # upload
```

Uploads ~830MB: `dense_i8.usearch`, `bm25.pkl`, `lexical_rows.npy`,
`chunks_meta.parquet`, `manifest.json`, and `encoder_onnx/*`.

`vectors.f32.npy` (1.7GB) is intentionally excluded — it only exists to rebuild
indexes without re-embedding, and the server never reads it.

---

## Step 2 — create the Space

1. https://huggingface.co/new-space
2. Owner **obscurlabs**, name **ovx-1**
3. SDK: **Docker** → *Blank*
4. Hardware: **CPU basic (free, 16GB RAM)**
5. Visibility: **Public** (judges need to reach it)

---

## Step 3 — push the code

```powershell
cd D:\Workspace\HHGOA26\townhall-2

git remote add space https://huggingface.co/spaces/obscurlabs/ovx-1
git push space main
```

If prompted for a password, use the **write token**, not your account password.

Then replace the Space's `README.md` with `deploy/SPACE_README.md`. Its YAML
frontmatter is what tells Spaces to use the Dockerfile and expose port 7860 —
without it the Space will not start correctly. Easiest via the web UI:
**Files → README.md → edit → paste → commit**.

---

## Step 4 — set the secrets

Space → **Settings → Variables and secrets**:

| Name | Value | Notes |
|---|---|---|
| `VOICERAG_INDEX_REPO` | `obscurlabs/ovx-1-index` | required |
| `HF_TOKEN` | read token | only if the dataset repo is private |
| `SARVAM_API_KEY` | your key | speech-to-text |
| `SARVAM_ALLOW_LIVE` | `0` | see the warning below |
| `GROQ_API_KEYS` | `key1,key2,key3,key4` | enables deep mode |

### About `SARVAM_ALLOW_LIVE`

Leave it at **`0`** except while recording the demo.

The account holds **100 credits total**. A public URL can be visited by anyone —
crawlers included — and every uncached recording spends one. At `0`, voice input
replays cached transcripts and refuses anything unrecognised, so a shared link
cannot drain the budget. Flip it to `1` immediately before filming, and back to
`0` afterwards.

---

## Step 5 — verify

The build takes 5–10 minutes. Watch **Logs**; a healthy boot ends with:

```
downloading dense_i8.usearch from obscurlabs/ovx-1-index
pipeline ready in NNs
warmed up 6 queries in N.Ns
Uvicorn running on http://0.0.0.0:7860
```

Then:

```powershell
curl https://obscurlabs-ovx-1.hf.space/api/health
```

Expect `ready: true` and `chunks: 1165508`.

---

## Troubleshooting

**Build succeeds, container dies immediately.** Check `VOICERAG_INDEX_REPO` is
set. Without it the app raises at startup — though `/api/health` deliberately
still responds and reports the reason rather than the container exiting silently.

**`401` while downloading artifacts.** The dataset repo is private and `HF_TOKEN`
is missing or read-scoped incorrectly. Making the dataset public is simpler.

**First request is slow (~300ms), then fast.** Should not happen — the app warms
up at boot. If it does, the Space was restarted mid-request.

**Space sleeps.** Free Spaces idle out after ~48h of inactivity. Visit the URL
once before judging; the warmup runs automatically on wake.

**Voice returns "couldn't transcribe".** Expected with `SARVAM_ALLOW_LIVE=0` for
any recording not already in the cache. That is the credit guard doing its job.
