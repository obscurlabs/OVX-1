"""Central configuration.

Every path is derived from the project root so the whole build stays self-contained:
model caches, datasets and indexes all live under this folder and vanish with it.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Locally the package is imported from the source tree, so the root is three
# levels up: src/voicerag/config.py -> src/voicerag -> src -> project root.
#
# In the container that derivation is WRONG. `pip install .` puts the package in
# site-packages, so __file__ becomes
#   /usr/local/lib/python3.12/site-packages/voicerag/config.py
# and parents[2] resolves to /usr/local/lib/python3.12 - a root-owned directory
# that is nowhere near web/ and that uid 1000 cannot write to. The failure is
# silent at build time and fatal at boot: mkdir on data/indexes raises
# PermissionError, and WEB_DIR.exists() is False so the UI never mounts.
#
# VOICERAG_ROOT lets the deployment state the root explicitly instead of
# inferring it from an install layout that differs between dev and prod.
_ENV_ROOT = os.environ.get("VOICERAG_ROOT")
PROJECT_ROOT = Path(_ENV_ROOT).resolve() if _ENV_ROOT else Path(__file__).resolve().parents[2]


class Paths:
    """Filesystem layout. Everything here is regenerable and gitignored."""

    root = PROJECT_ROOT
    cache = PROJECT_ROOT / ".cache"
    data = PROJECT_ROOT / "data"

    raw = data / "raw"
    processed = data / "processed"
    indexes = data / "indexes"
    audio = data / "audio"

    # Concrete artifacts, named once here so no module invents its own filename.
    passages = processed / "passages.parquet"
    queries = processed / "queries.parquet"  # gold Q/A pairs: eval + benchmark set
    chunks = processed / "chunks.parquet"
    vector_index = indexes / "dense.usearch"
    bm25_index = indexes / "bm25.pkl"
    chunk_meta = indexes / "chunks_meta.parquet"
    # What the SERVER reads. Same rows as the parquet, but Arrow IPC and
    # uncompressed, so it can be memory-mapped rather than decompressed into the
    # heap. Reading the parquet costs 2425MB of RSS for 138MB of disk (two ZSTD
    # row groups decompress ~680MB of text in one shot); the mapped file costs
    # 1.1MB at load and pages in only the rows actually retrieved.
    chunk_meta_arrow = indexes / "chunks_meta.arrow"
    onnx_encoder = indexes / "encoder_onnx"
    # Same graph with the unused 82% of the vocabulary removed. The word
    # embedding table is per-tensor quantized, so dropping rows is exact -
    # verified bit-identical over corpus and query text - but it takes the
    # encoder's resident cost from 422.7MB to 123.9MB, because ONNX Runtime
    # materializes that uint8 table as fp32 (250,037 x 384 x 4 = 384MB).
    onnx_encoder_pruned = indexes / "encoder_onnx_pruned"
    stt_cache = audio / "transcripts.json"

    @classmethod
    def serving_index(cls) -> Path:
        """The index directory the server loads.

        scripts/trim_index.py writes the deployable subset to indexes/serving,
        so a build machine holds both the full index and the trimmed one. On the
        host only the trimmed artifacts are ever downloaded, and they land
        directly in indexes/ - so this resolves to the right place in both
        without the deployment needing to know which it is.
        """
        candidate = cls.indexes / "serving"
        if (candidate / "manifest.json").exists():
            return candidate
        return cls.indexes

    @classmethod
    def serving_encoder(cls) -> Path:
        """The encoder the server actually loads.

        Benchmarks and guardrail evals resolve through here too, so local
        numbers describe the deployed artifact rather than the fuller model
        that only ever exists on a build machine.
        """
        # The encoder that travelled with the deployable index wins: it was
        # pruned against that corpus, and pairing an index with a differently
        # pruned encoder would shift every vector silently.
        bundled = cls.serving_index() / "encoder_onnx"
        if (bundled / "tokenizer.json").exists():
            return bundled
        if (cls.onnx_encoder_pruned / "tokenizer.json").exists():
            return cls.onnx_encoder_pruned
        return cls.onnx_encoder

    @classmethod
    def ensure(cls) -> None:
        for p in (cls.cache, cls.raw, cls.processed, cls.indexes, cls.audio):
            p.mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Speech-to-text (Sarvam) -------------------------------------------
    sarvam_api_key: str = ""
    # Hard guard on a 100-credit budget. Nothing calls Sarvam for real unless
    # this is explicitly flipped to True; otherwise the cache must satisfy the
    # request or the pipeline fails loudly rather than silently spending credits.
    sarvam_allow_live: bool = False
    sarvam_stt_url: str = "https://api.sarvam.ai/speech-to-text"
    # saarika:v2 was the original choice but that family is being deprecated
    # (v2.5 is already flagged), and a model retirement on demo day with no
    # resubmission allowed is an unacceptable risk. saaras:v3 is Sarvam's
    # current recommended model for transcription.
    sarvam_model: str = "saaras:v3"
    # CRITICAL: saaras defaults toward translation. mode="transcribe" keeps the
    # output in the SPOKEN language. Translating a Hindi question to English
    # would silently break monolingual Hindi retrieval, since the query would
    # then be matched against the corpus in the wrong language.
    sarvam_mode: str = "transcribe"

    # --- Answer generation (Groq) ------------------------------------------
    # Multiple free-tier keys pooled into one effective rate limit; the harness
    # rotates on 429 and trips a breaker per key.
    groq_api_keys: str = ""
    groq_model: str = "llama-3.1-8b-instant"
    groq_timeout_s: float = 8.0
    groq_max_retries: int = 2

    # --- Dataset ------------------------------------------------------------
    hf_token: str = ""
    hf_dataset: str = "ai4bharat/MSMARCO-XI"
    # NOTE: the dataset card implies per-language configs, but the repo actually
    # exposes a single 'default' config with language as a row field. The parquet
    # shards ARE split by language, so we fetch one file directly (~475MB) rather
    # than streaming and filtering the full 55.6GB.
    hf_data_file: str = "validation/hinval.parquet"
    # Cap on source queries read. Each carries ~10 passages in English AND Hindi,
    # so this multiplies by ~20 into passages and ~3x again into chunks. Sized so
    # the final int8 index stays small enough to memory-map on a cheap Render tier.
    max_queries: int = 15_000
    max_passages: int = 400_000

    # --- Models -------------------------------------------------------------
    embed_model: str = "intfloat/multilingual-e5-small"
    embed_dim: int = 384
    rerank_model: str = "BAAI/bge-reranker-base"

    # --- Retrieval tuning ---------------------------------------------------
    retrieve_top_k: int = 50
    rerank_top_n: int = 5
    # Below this grounding score the system abstains instead of answering.
    grounding_threshold: float = 0.35

    # --- Local tooling ------------------------------------------------------
    ffmpeg_path: str = "ffmpeg"

    @field_validator("groq_api_keys")
    @classmethod
    def _strip(cls, v: str) -> str:
        return v.strip()

    @property
    def groq_key_pool(self) -> list[str]:
        """Groq keys as a list. Empty means the LLM escalation path is disabled."""
        return [k.strip() for k in self.groq_api_keys.split(",") if k.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
