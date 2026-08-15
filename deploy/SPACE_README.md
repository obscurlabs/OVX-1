---
title: OVX-1 Voice RAG
emoji: 🎙️
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: Voice-enabled RAG over MSMARCO-XI, Hindi and English, sub-20ms retrieval
---

# OVX-1 — Voice-Enabled RAG

Speak or type a question in **Hindi or English**; the system transcribes it,
retrieves from a 1.17M-chunk index built over
[ai4bharat/MSMARCO-XI](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI),
and answers **only** from retrieved passages — or refuses.

Built by **Obscur Labs** for HH Goa 2026 · `#RAGInGoa`

## Measured latency (400 held-out queries, ONNX-int8 on CPU)

| Percentile | Core latency |
|---|---|
| P50 | 9.43 ms |
| P70 | 10.59 ms |
| P100 | 42.52 ms |

100% of queries land under the 200ms budget. Core = transcript → final answer,
the segment the brief enumerates. Speech-to-text is a network round trip and is
reported separately.

## Knowing when not to answer

| Suite | Result |
|---|---|
| Unsafe input blocked | 100% |
| Prompt injection blocked | 100% |
| Out-of-corpus refused | 100% |
| Legitimate questions answered | 93.3% |

False answer rate **0%**; false refusal rate **6.7%**.

Try `what is the population of the zorbian empire` — no such thing exists, and
saying so is the correct answer.

## Configuration

Set these as Space **secrets**:

| Variable | Purpose |
|---|---|
| `VOICERAG_INDEX_REPO` | dataset repo holding the index artifacts (required) |
| `HF_TOKEN` | read token, if that dataset repo is private |
| `SARVAM_API_KEY` | speech-to-text |
| `SARVAM_ALLOW_LIVE` | `0` keeps voice on cached transcripts only |
| `GROQ_API_KEYS` | comma-separated; enables the deep-mode path |
