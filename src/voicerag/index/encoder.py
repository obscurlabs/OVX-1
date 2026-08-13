"""Embedding model wrapper (indexing side, GPU).

multilingual-e5-small is asymmetric: it was trained with "query: " on questions
and "passage: " on documents. Omitting the prefixes, or swapping them, costs a
large chunk of retrieval quality and raises no error at all - the vectors are
still valid, just worse. Encoding both sides through one class is the only
reliable way to keep them consistent between index build and serving.

This module imports torch and is LOCAL ONLY. The deployed server uses the ONNX
export instead (see export_onnx), keeping the Render image free of a 2.8GB dep.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np


class E5Encoder:
    """Batched encoder with the correct asymmetric prefixes."""

    QUERY_PREFIX = "query: "
    PASSAGE_PREFIX = "passage: "

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-small",
        device: str | None = None,
        batch_size: int = 256,
    ) -> None:
        from sentence_transformers import SentenceTransformer  # local import: torch is heavy
        import torch

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=device)
        self.dim = self.model.get_sentence_embedding_dimension()

        # fp16 on GPU roughly doubles indexing throughput at negligible recall cost.
        if device == "cuda":
            self.model = self.model.half()

    def _encode(self, texts: Sequence[str], prefix: str, show_progress: bool) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        prefixed = [prefix + t for t in texts]
        vectors = self.model.encode(
            prefixed,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            # Normalized once here so retrieval can use plain inner product as
            # cosine similarity, and so int8 quantization stays well-conditioned.
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        )
        return vectors.astype(np.float32)

    def encode_passages(self, texts: Sequence[str], show_progress: bool = False) -> np.ndarray:
        return self._encode(texts, self.PASSAGE_PREFIX, show_progress)

    def encode_queries(self, texts: Sequence[str], show_progress: bool = False) -> np.ndarray:
        return self._encode(texts, self.QUERY_PREFIX, show_progress)

    def embed_fn_for_chunking(self):
        """Adapter for semantic chunking.

        Sentences being tested for topical breaks are document text, so they take
        the passage prefix. Small batches: this runs per passage, not per corpus.
        """

        def _fn(sentences: Sequence[str]) -> np.ndarray:
            return self.encode_passages(list(sentences))

        return _fn


def export_onnx(
    model_name: str,
    out_dir: Path,
    quantize: bool = True,
) -> Path:
    """Export the encoder to ONNX (optionally int8) for CPU serving.

    Query encoding is on the hot path of the 200ms budget, and it is the only
    neural step that must run per-request. ONNX int8 on CPU is several times
    faster than torch-on-CPU for a single short query, and it removes torch from
    the deployment entirely.
    """
    from optimum.onnxruntime import ORTModelForFeatureExtraction
    from transformers import AutoTokenizer

    out_dir.mkdir(parents=True, exist_ok=True)

    model = ORTModelForFeatureExtraction.from_pretrained(model_name, export=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    if quantize:
        from optimum.onnxruntime import ORTQuantizer
        from optimum.onnxruntime.configuration import AutoQuantizationConfig

        quantizer = ORTQuantizer.from_pretrained(out_dir)
        qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=True)
        quantizer.quantize(save_dir=out_dir, quantization_config=qconfig)

    return out_dir
