"""Central configuration.

Every path is derived from the project root so the whole build stays self-contained:
model caches, datasets and indexes all live under this folder and vanish with it.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# src/voicerag/config.py -> src/voicerag -> src -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]


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
    onnx_encoder = indexes / "encoder_onnx"
    stt_cache = audio / "transcripts.json"

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
    sarvam_model: str = "saarika:v2"

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
