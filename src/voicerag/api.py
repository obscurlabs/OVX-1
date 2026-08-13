"""FastAPI service.

Deployment shape matters here. The index artifacts are ~830MB and cannot live in
git, so on startup the service pulls them from a Hugging Face dataset repo -
the same mechanism that fetched the corpus. Everything is loaded once into
module state; a per-request load would dominate the latency budget many times
over.

Voice input is gated. `SARVAM_ALLOW_LIVE` must be enabled for a cache miss to
reach Sarvam, so a deployed demo cannot quietly drain a 100-credit budget if it
is crawled or shared more widely than intended.
"""

from __future__ import annotations

import base64
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, ORJSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from voicerag.config import Paths, get_settings
from voicerag.contracts import QueryRequest, QueryResponse

log = logging.getLogger("voicerag")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

WEB_DIR = Paths.root / "web"

# Files the server needs. vectors.f32.npy is deliberately absent: it exists only
# to rebuild indexes without re-embedding, and at 1.7GB it would triple startup
# download time for no serving benefit.
REQUIRED_ARTIFACTS = (
    "dense_i8.usearch",
    "bm25.pkl",
    "lexical_rows.npy",
    "chunks_meta.parquet",
    "manifest.json",
)

state: dict = {"pipeline": None, "ready": False, "error": None, "loaded_at": None}


def fetch_artifacts() -> None:
    """Pull index artifacts from the HF dataset repo when absent locally.

    Set VOICERAG_INDEX_REPO to the dataset repo id. Skipped entirely when the
    files already exist, so local development never re-downloads.
    """
    repo = os.environ.get("VOICERAG_INDEX_REPO")
    missing = [name for name in REQUIRED_ARTIFACTS if not (Paths.indexes / name).exists()]
    if not missing:
        return
    if not repo:
        raise RuntimeError(
            f"missing index artifacts {missing} and VOICERAG_INDEX_REPO is not set"
        )

    from huggingface_hub import hf_hub_download

    Paths.indexes.mkdir(parents=True, exist_ok=True)
    for name in missing:
        log.info("downloading %s from %s", name, repo)
        path = hf_hub_download(
            repo_id=repo,
            filename=name,
            repo_type="dataset",
            token=os.environ.get("HF_TOKEN") or None,
            local_dir=Paths.indexes,
        )
        log.info("  -> %s", path)

    # The ONNX encoder ships in the same repo, under a subdirectory.
    if not (Paths.onnx_encoder / "tokenizer.json").exists():
        for name in ("model_quantized.onnx", "tokenizer.json", "config.json",
                     "tokenizer_config.json", "special_tokens_map.json"):
            try:
                hf_hub_download(
                    repo_id=repo,
                    filename=f"encoder_onnx/{name}",
                    repo_type="dataset",
                    token=os.environ.get("HF_TOKEN") or None,
                    local_dir=Paths.indexes.parent,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("optional encoder file %s: %s", name, exc)


WARMUP_QUERIES = (
    "what is a corporation",
    "how long does it take to boil an egg",
    "प्रकाश संश्लेषण कैसे होता है",
    "निगम क्या है",
    "who invented the telephone",
    "भारत की राजधानी कौन सी है",
)


def warm_up(pipeline) -> None:
    """Page the index in before serving real traffic.

    The dense index is memory-mapped, so a cold process (or one whose pages the
    OS evicted after an idle period) pays disk reads on its first queries.
    Measured on a freshly-idle server: 337ms for the first request versus ~9ms
    warm - encode 124ms and retrieve 205ms, entirely page-in cost.

    That matters more than it sounds. On a free host that idles between visits,
    the FIRST request a visitor makes is the one they judge the system by, and
    it would misrepresent the latency by nearly 40x. Warming both scripts also
    touches the ONNX graph and both language regions of the index.
    """
    from voicerag.contracts import QueryRequest

    started = time.perf_counter()
    for text in WARMUP_QUERIES:
        try:
            pipeline.answer(QueryRequest(text=text, fast_only=True))
        except Exception as exc:  # noqa: BLE001 - warmup must never block boot
            log.warning("warmup query failed (%s): %s", text[:30], exc)
    log.info("warmed up %d queries in %.1fs", len(WARMUP_QUERIES), time.perf_counter() - started)


@asynccontextmanager
async def lifespan(app: FastAPI):
    started = time.perf_counter()
    try:
        fetch_artifacts()
        from voicerag.pipeline.orchestrator import VoiceRagPipeline

        state["pipeline"] = VoiceRagPipeline.load()
        warm_up(state["pipeline"])
        state["ready"] = True
        state["loaded_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        log.info("pipeline ready in %.1fs", time.perf_counter() - started)
    except Exception as exc:  # noqa: BLE001
        # Start anyway so /api/health can report WHY it is broken. A container
        # that exits on boot gives the operator nothing to debug from.
        state["error"] = f"{type(exc).__name__}: {exc}"
        log.exception("pipeline failed to load")
    yield


app = FastAPI(
    title="OVX-1 Voice RAG",
    description="Voice-enabled RAG over MSMARCO-XI (Hindi/English)",
    default_response_class=ORJSONResponse,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextQuery(BaseModel):
    text: str
    lang_hint: str | None = None
    # Default keeps every request inside the 200ms budget; opting out allows
    # LLM escalation for the ~7% of queries extraction handles poorly.
    fast_only: bool = True


def require_pipeline():
    if not state["ready"]:
        raise HTTPException(503, detail=state["error"] or "pipeline still loading")
    return state["pipeline"]


@app.get("/api/health")
def health() -> dict:
    settings = get_settings()
    payload = {
        "ready": state["ready"],
        "error": state["error"],
        "loaded_at": state["loaded_at"],
        "voice_enabled": settings.sarvam_allow_live,
    }
    if state["ready"]:
        pipeline = state["pipeline"]
        payload["chunks"] = len(pipeline.retriever.dense_index)
        payload["lexical_docs"] = pipeline.retriever.bm25.n_docs
        payload["llm_keys"] = len(getattr(pipeline.router.llm, "pool", []) or [])
    return payload


@app.post("/api/query", response_model=QueryResponse)
def query(body: TextQuery) -> QueryResponse:
    pipeline = require_pipeline()
    return pipeline.answer(
        QueryRequest(text=body.text, lang_hint=body.lang_hint, fast_only=body.fast_only)
    )


@app.post("/api/voice", response_model=QueryResponse)
async def voice(
    audio: UploadFile = File(...),
    lang_hint: str | None = Form(None),
) -> QueryResponse:
    pipeline = require_pipeline()
    raw = await audio.read()
    if not raw:
        raise HTTPException(400, detail="empty audio upload")

    settings = get_settings()
    return pipeline.answer(
        QueryRequest(audio_b64=base64.b64encode(raw).decode(), lang_hint=lang_hint),
        allow_live_stt=settings.sarvam_allow_live,
    )


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


def main() -> None:
    import uvicorn

    # HF Spaces routes to 7860 by default.
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    main()
