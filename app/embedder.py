"""Embedder interface for rag-local-eval-loop using ONNX int8 encoder."""
from __future__ import annotations
import numpy as np
from voicerag.config import Paths
from voicerag.pipeline.query_encoder import OnnxQueryEncoder

_encoder = None

def get_model():
    global _encoder
    if _encoder is None:
        _encoder = OnnxQueryEncoder(Paths.serving_encoder())
    return _encoder

def embed_one(text: str) -> np.ndarray:
    encoder = get_model()
    return encoder.encode(text)

def embed(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    encoder = get_model()
    return encoder.encode_batch(texts)
