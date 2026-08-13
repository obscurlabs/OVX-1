# HH Goa 2026 — Shortlisting Task 2: Voice-Enabled RAG Model

> Brief as issued. Launched **August 13, 2026** · Deadline **August 22, 2026, 11:59 PM**.
> Team: Obscur Labs — Yaksh Bambhroliya, Vansh Dobariya. Hashtag: `#RAGInGoa`.

## What to build

A voice-enabled Retrieval-Augmented Generation (RAG) system — a user speaks a question, the
pipeline transcribes it, retrieves relevant context from a provided dataset, and returns an
answer, end to end.

**Pipeline shape:** Voice input → Speech-to-text → Chunking/Retrieval (vector DB) → Answer generation

## Dataset

Provided: https://huggingface.co/datasets/ai4bharat/MSMARCO-XI

## Technical requirements

1. **Speech-to-text** — Use either **Sarvam** or **ElevenLabs** for voice-to-text. Pick one.
2. **Chunking** — Strategy must be *vast*; a single naive fixed-size approach is not acceptable.
   They want real thought in how the dataset is split, indexed, and retrieved — e.g. multiple
   chunking strategies, overlap handling, semantic vs. fixed-size splitting, metadata-aware
   chunking.
3. **Latency target** — Full process (chunking + vector DB retrieval + everything through to
   final output) should complete in **under 200 ms**.
4. **Latency analytics** — Submit **P50 / P70 / P100** latency numbers measured across a
   reasonable number of test queries — not a single best-case run.
5. **Harness your model** — Run the pipeline inside a proper harness: structured orchestration
   around the model (tool calls, retries, structured input/output handling, error recovery)
   rather than a single raw prompt-in, text-out call.
6. **Guardrail your model** — Guardrails for off-topic queries, unsafe/inappropriate inputs,
   hallucination checks, and answers not grounded in retrieved context. Show the system knows
   **when not to answer**, not just how to answer.

## Submission requirements

Form: https://forms.gle/MNvCjcv23Hn2Eeu58

- GitHub repo link
- Live working link
- 2 videos (below)

**No resubmissions allowed — submit only when the build is final.**

### Video 1 — Team/process video
90 seconds. Shows how the team is working on this — *process, not the product*.

### Video 2 — Demo video
Demo of the actual project working end to end.

## Promotion requirement (mandatory)

Both videos uploaded to **Instagram, X, and LinkedIn** — by **every individual team member**,
not just one shared team post. At least **1 Instagram account must be public**.

Every post, on every platform, by every member, must include `#RAGInGoa`.

## Timeline

| | |
| --- | --- |
| Task launch | August 13, 2026 |
| Deadline | August 22, 2026, 11:59 PM |
