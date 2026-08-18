"""Query encoder for serving (ONNX, CPU).

The only neural step on the per-request hot path. torch is deliberately absent
here: onnxruntime with an int8 graph encodes a short query in single-digit
milliseconds on CPU and keeps the deployed image ~2.8GB smaller.

Pooling must match how the index was built. sentence-transformers applies
attention-masked mean pooling then L2 normalization for E5; reproducing that
exactly is not optional, because a mismatch produces vectors that are still
plausible-looking but land in a different space from the index - retrieval
quality collapses with no error raised anywhere.
"""

from __future__ import annotations

import unicodedata
from collections import OrderedDict
from pathlib import Path

import numpy as np


class OnnxQueryEncoder:
    """E5 query encoder over onnxruntime."""

    QUERY_PREFIX = "query: "

    def __init__(
        self,
        model_dir: Path,
        max_length: int = 128,
        threads: int = 2,
        cache_size: int = 512,
    ) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.model_dir = Path(model_dir)
        self.max_length = max_length
        # A true LRU keeps recurring demo and common queries fast without
        # allowing cache growth to threaten the 512 MB deployment tier.
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cache_size = cache_size

        tokenizer_path = self.model_dir / "tokenizer.json"
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"missing tokenizer at {tokenizer_path}")
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.tokenizer.enable_truncation(max_length=max_length)

        onnx_files = sorted(self.model_dir.glob("*.onnx"))
        if not onnx_files:
            raise FileNotFoundError(f"no .onnx file in {self.model_dir}")
        # Prefer the quantized graph when both are present.
        model_path = next((p for p in onnx_files if "quantize" in p.name), onnx_files[0])

        options = ort.SessionOptions()
        # Queries are one short sequence; extra threads cost more in coordination
        # than they save, and oversubscribing hurts p99 under concurrency.
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.input_names = {i.name for i in self.session.get_inputs()}
        self.model_path = model_path

    def encode(self, text: str) -> np.ndarray:
        # NFC avoids duplicate entries for canonically equivalent Hindi text.
        key = unicodedata.normalize("NFC", text.strip()).casefold()
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached.copy()
        vec = self.encode_batch([text])[0]
        if self._cache_size > 0:
            # Keep an owned copy: downstream consumers must not be able to
            # mutate the cache by modifying the vector returned on a cache miss.
            self._cache[key] = vec.copy()
            self._cache.move_to_end(key)
            if len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return vec

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        encodings = [self.tokenizer.encode(self.QUERY_PREFIX + t) for t in texts]
        max_len = max(len(e.ids) for e in encodings)

        input_ids = np.zeros((len(encodings), max_len), dtype=np.int64)
        attention_mask = np.zeros((len(encodings), max_len), dtype=np.int64)
        for i, enc in enumerate(encodings):
            n = len(enc.ids)
            input_ids[i, :n] = enc.ids
            attention_mask[i, :n] = enc.attention_mask

        feeds = {"input_ids": input_ids, "attention_mask": attention_mask}
        if "token_type_ids" in self.input_names:
            feeds["token_type_ids"] = np.zeros_like(input_ids)

        hidden = self.session.run(None, feeds)[0]  # (batch, seq, dim)

        # Attention-masked mean pooling, matching sentence-transformers.
        mask = attention_mask[..., None].astype(np.float32)
        pooled = (hidden * mask).sum(axis=1) / np.clip(mask.sum(axis=1), 1e-9, None)

        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        return (pooled / np.clip(norms, 1e-9, None)).astype(np.float32)
