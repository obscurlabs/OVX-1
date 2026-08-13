"""Export the query encoder to ONNX (int8) and verify parity against torch.

Query encoding is the only neural step on the per-request hot path, and the
deployed server runs it on a shared CPU rather than an RTX 4050. Exporting to
ONNX with dynamic int8 quantization does two things: it is several times faster
than torch-on-CPU for a single short sequence, and it removes the 2.8GB torch
dependency from the deployment image entirely.

The parity check is the important half of this script. A pooling or prefix
mismatch between export and index-build produces vectors that look perfectly
valid but occupy a different space from the index - retrieval quality collapses
and nothing raises an error. Cosine similarity against the torch encoder is the
only way to catch that before it reaches production.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time

import numpy as np

from voicerag.config import Paths, get_settings

PARITY_QUERIES = [
    "what is a corporation",
    "how long does it take to boil an egg",
    "what causes photosynthesis in plants",
    "निगम क्या है",
    "प्रकाश संश्लेषण कैसे होता है",
    "भारत की राजधानी कौन सी है",
    "who invented the telephone",
    "कंप्यूटर का आविष्कार किसने किया",
]

# Below this, the ONNX vectors are not interchangeable with the indexed ones.
PARITY_THRESHOLD = 0.99


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--no-quantize", action="store_true", help="export fp32 only")
    p.add_argument("--force", action="store_true", help="re-export over an existing directory")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    settings = get_settings()
    out_dir = Paths.onnx_encoder

    if out_dir.exists() and args.force:
        shutil.rmtree(out_dir)

    if not (out_dir / "tokenizer.json").exists():
        print(f"exporting {settings.embed_model} -> {out_dir}")
        t0 = time.perf_counter()
        try:
            from voicerag.index.encoder import export_onnx

            export_onnx(settings.embed_model, out_dir, quantize=not args.no_quantize)
        except Exception as exc:  # noqa: BLE001
            print(f"quantized export failed ({type(exc).__name__}: {exc})")
            print("retrying without quantization")
            from voicerag.index.encoder import export_onnx

            export_onnx(settings.embed_model, out_dir, quantize=False)
        print(f"  exported in {time.perf_counter() - t0:.1f}s")
    else:
        print(f"reusing existing export at {out_dir}")

    print("\n=== artifacts ===")
    for path in sorted(out_dir.glob("*")):
        if path.is_file():
            print(f"  {path.stat().st_size / 1024 / 1024:8.1f} MB  {path.name}")

    # --- parity -----------------------------------------------------------
    print("\n=== parity: ONNX vs torch ===")
    from voicerag.index.encoder import E5Encoder
    from voicerag.pipeline.query_encoder import OnnxQueryEncoder

    torch_encoder = E5Encoder(settings.embed_model, device="cpu", batch_size=8)
    onnx_encoder = OnnxQueryEncoder(out_dir)
    print(f"  onnx graph: {onnx_encoder.model_path.name}\n")

    reference = torch_encoder.encode_queries(PARITY_QUERIES)
    produced = onnx_encoder.encode_batch(PARITY_QUERIES)

    worst = 1.0
    for query, a, b in zip(PARITY_QUERIES, reference, produced, strict=False):
        cosine = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        worst = min(worst, cosine)
        flag = "ok " if cosine >= PARITY_THRESHOLD else "BAD"
        print(f"  [{flag}] cos={cosine:.5f}  {query[:44]}")

    print(f"\n  worst cosine: {worst:.5f} (threshold {PARITY_THRESHOLD})")
    if worst < PARITY_THRESHOLD:
        print("\nFAILED: ONNX vectors diverge from the indexed space.")
        print("Retrieval would silently degrade. Check pooling and the 'query: ' prefix.")
        return 1

    # --- speed ------------------------------------------------------------
    print("\n=== single-query encode latency (CPU) ===")
    for label, encode in (
        ("torch cpu", lambda q: torch_encoder.encode_queries([q])),
        ("onnx  cpu", lambda q: onnx_encoder.encode(q)),
    ):
        for _ in range(3):
            encode(PARITY_QUERIES[0])
        samples = []
        for query in PARITY_QUERIES * 5:
            t0 = time.perf_counter()
            encode(query)
            samples.append((time.perf_counter() - t0) * 1000)
        samples.sort()
        print(
            f"  {label}: p50 {samples[len(samples) // 2]:5.1f}ms   "
            f"p100 {samples[-1]:5.1f}ms   mean {sum(samples) / len(samples):5.1f}ms"
        )

    print("\nparity OK - ONNX encoder is safe to serve")
    return 0


if __name__ == "__main__":
    sys.exit(main())
