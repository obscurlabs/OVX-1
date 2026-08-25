"""Embedder adapter for the Task #2 evaluation suite.

The suite calls `embed()` to build its own throwaway index over MSMARCO-XI
passages, and `embed_one()` to embed each query against it. That asymmetry
matters here: this project uses multilingual-e5-small, which is trained with
*different prefixes* for the two roles ("passage: " vs "query: "). Feeding
passages through the query prefix -- or vice versa -- silently costs recall,
so the two entry points are mapped deliberately rather than aliased:

    embed(texts)    -> passage prefix   (corpus side, eval/index_build.py)
    embed_one(text) -> query prefix     (query side, eval/pipeline.py)

OnnxQueryEncoder hardcodes the query prefix, so the passage side gets its own
instance with the prefix overridden once at construction -- never mutated per
call, so this stays correct under --workers > 1.

Encoder choice: the *unpruned* ONNX graph, not the pruned one the server
loads. scripts/prune_vocab.py drops ~76% of the vocabulary to fit the 512MB
deployment tier, keeping only pieces the production corpus actually uses; it
is verified lossless on that corpus (max abs difference 0.000e+00). The
evaluation suite samples its own MSMARCO-XI rows, which may contain pieces
outside that kept set, so the full-vocabulary graph is the honest choice for
a measurement that is not memory-constrained. Same model, same weights.
"""

from __future__ import annotations

import numpy as np

from voicerag.config import Paths
from voicerag.pipeline.query_encoder import OnnxQueryEncoder

_ENCODERS: dict[str, OnnxQueryEncoder] = {}

# Long enough for MSMARCO-XI passages; the serving default (128) is tuned for
# short spoken queries and would truncate corpus text on the passage side.
_PASSAGE_MAX_LEN = 512


def _encoder_dir():
    full = Paths.indexes / "encoder_onnx"
    if (full / "tokenizer.json").exists():
        return full
    # Fall back to whatever the server uses, so this still works on a checkout
    # that only shipped the pruned artifact.
    return Paths.serving_encoder()


def get_model():
    """Load both prefix variants once. Only the side effect matters to the suite."""
    if not _ENCODERS:
        model_dir = _encoder_dir()
        query = OnnxQueryEncoder(model_dir)
        passage = OnnxQueryEncoder(model_dir, max_length=_PASSAGE_MAX_LEN)
        passage.QUERY_PREFIX = "passage: "
        _ENCODERS["query"] = query
        _ENCODERS["passage"] = passage
    return _ENCODERS


def embed(texts: list[str]) -> np.ndarray:
    """Corpus side: (len(texts), dim), L2-normalised float32."""
    texts = list(texts)
    if not texts:
        return np.zeros((0, embed_one("dimension probe").shape[-1]), dtype=np.float32)
    return get_model()["passage"].encode_batch(texts)


def embed_one(text: str) -> np.ndarray:
    """Query side: (dim,), L2-normalised float32."""
    return get_model()["query"].encode(text)
