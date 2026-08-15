# Hugging Face Spaces (Docker SDK).
#
# The image stays small on purpose: torch is NOT installed. Indexing ran locally
# on the RTX 4050; serving only needs onnxruntime, which is ~50MB against
# torch's 2.8GB. Index artifacts are fetched from HF Hub at boot rather than
# baked in, so the index can be rebuilt and republished without a rebuild here.

FROM python:3.12-slim

# ffmpeg: browsers record WebM/Opus, Sarvam wants WAV.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Spaces runs containers as uid 1000; writing anywhere else fails at runtime.
RUN useradd -m -u 1000 user
WORKDIR /app

COPY --chown=user:user pyproject.toml ./
COPY --chown=user:user src ./src
COPY --chown=user:user web ./web

# Only the serving dependencies. The [index] extra (torch, sentence-transformers,
# datasets) is local-only tooling and would bloat the image ~30x.
RUN pip install --no-cache-dir . \
    && chown -R user:user /app

USER user

# Keep every cache inside the writable app dir; the default HOME-based paths are
# not reliably writable on Spaces.
# VOICERAG_ROOT is not optional here. The package is installed into
# site-packages, so config.py cannot infer the project root from __file__ - it
# would resolve to /usr/local/lib/python3.12 and the app would look for its
# index and web assets there instead of in /app.
ENV VOICERAG_ROOT=/app \
    HF_HOME=/app/.cache/hf \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860

RUN mkdir -p /app/.cache/hf /app/data/indexes /app/data/audio

EXPOSE 7860

# Boot sequence: download artifacts -> load index -> warm up -> serve.
# Warmup matters on a free host that idles: the first request would otherwise
# pay page-in cost and report ~337ms instead of ~9ms.
CMD ["python", "-m", "voicerag.api"]
