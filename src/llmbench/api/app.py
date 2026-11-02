"""OpenAI-compatible API gateway in front of llama-server, plus a RAG endpoint.

    client --HTTP--> FastAPI (this) --HTTP/SSE--> llama-server (GPU)
                          |--> TEI (embeddings) + Qdrant (top-k chunks)   for /rag/query

Endpoints
* ``POST /v1/chat/completions`` - validated, then proxied to llama-server; ``stream=true``
  relays the Server-Sent Events chunk by chunk (no buffering: the client sees the first
  token as soon as the GPU produces it). Any OpenAI SDK works with ``base_url=.../v1``.
* ``GET  /v1/models``           - the served model
* ``POST /rag/query``           - question -> retrieval -> grounded answer with its sources
* ``GET  /health``              - 200 only if llm, embedder and qdrant all answer (503 otherwise)

Why a gateway instead of exposing llama-server directly: one place for input validation,
request logs with latency, the RAG logic, and later auth / rate limiting / metrics, while
the inference server stays replaceable (vLLM, TGI speak the same API).

Run::  uvicorn --factory llmbench.api.app:create_app --port 8000
"""

from __future__ import annotations

import asyncio
import time
import urllib.error
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from llmbench.api.schemas import (
    ChatCompletionRequest,
    HealthResponse,
    RagQuery,
    RagResponse,
    Source,
)
from llmbench.api.settings import ApiSettings
from llmbench.config import RagConfig, load_rag_config
from llmbench.http import HttpError
from llmbench.log import get_logger
from llmbench.rag.embed import TEIEmbedder
from llmbench.rag.pipeline import RagPipeline, Retriever
from llmbench.rag.store import QdrantStore
from llmbench.serving.client import StreamError

log = get_logger("llmbench.api")


def _rag_config(settings: ApiSettings) -> RagConfig:
    rag = load_rag_config(settings.rag_config)
    updates: dict[str, object] = {}
    if settings.embedder_url:
        updates["embedder"] = rag.embedder.model_copy(update={"url": settings.embedder_url})
    if settings.qdrant_url:
        updates["qdrant"] = rag.qdrant.model_copy(update={"url": settings.qdrant_url})
    return rag.model_copy(update=updates)


def build_pipeline(rag: RagConfig) -> RagPipeline:
    retriever = Retriever(TEIEmbedder(rag.embedder), QdrantStore(rag.qdrant), rag.top_k)
    return RagPipeline(retriever, rag.prompt_template, rag.max_new_tokens)


async def _relay(upstream: httpx.Response) -> AsyncIterator[bytes]:
    """Forward the SSE bytes as they arrive. If the client disconnects, the generator is
    cancelled and closing the upstream response makes llama-server stop generating."""
    try:
        async for chunk in upstream.aiter_raw():
            yield chunk
    finally:
        await upstream.aclose()


def create_app(settings: ApiSettings | None = None, pipeline: RagPipeline | None = None) -> FastAPI:
    settings = settings or ApiSettings.from_env()
    rag = _rag_config(settings)
    rag_pipeline = pipeline or build_pipeline(rag)
    llm = settings.llm_url.rstrip("/")
    health_urls = {
        "llm": f"{llm}/health",
        "embedder": f"{rag.embedder.url}/health",
        "qdrant": f"{rag.qdrant.url}/healthz",
    }

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # One pooled client for the whole process: reuses TCP connections to llama-server.
        timeout = httpx.Timeout(settings.request_timeout_s, connect=5.0)
        limits = httpx.Limits(max_connections=100, max_keepalive_connections=20)
        async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
            app.state.http = client
            log.info("api ready", extra={"llm_url": llm, "model": settings.model_name})
            yield

    app = FastAPI(
        title="llmbench API",
        version="0.1.0",
        description="OpenAI-compatible gateway to a quantized open-weight LLM, with RAG.",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def access_log(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        # For streamed answers this is the time to response headers (the stream continues).
        log.info(
            "request",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "ms": round((time.perf_counter() - start) * 1000, 1),
            },
        )
        return response

    def _client(request: Request) -> httpx.AsyncClient:
        client: httpx.AsyncClient = request.app.state.http
        return client

    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(body: ChatCompletionRequest, request: Request) -> Response:
        client = _client(request)
        payload = body.model_dump(exclude_none=True)
        upstream_request = client.build_request("POST", f"{llm}/v1/chat/completions", json=payload)
        try:
            upstream = await client.send(upstream_request, stream=True)
        except httpx.HTTPError as exc:
            raise HTTPException(502, f"LLM server unreachable: {type(exc).__name__}") from exc
        if body.stream and upstream.status_code == 200:
            return StreamingResponse(
                _relay(upstream),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        content = await upstream.aread()
        await upstream.aclose()
        return Response(
            content,
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type", "application/json"),
        )

    @app.get("/v1/models", response_model=None)
    async def models(request: Request) -> Response:
        try:
            upstream = await _client(request).get(f"{llm}/v1/models")
        except httpx.HTTPError as exc:
            raise HTTPException(502, f"LLM server unreachable: {type(exc).__name__}") from exc
        return Response(upstream.content, status_code=upstream.status_code,
                        media_type="application/json")  # fmt: skip

    # Plain `def`: the RAG pipeline uses blocking I/O, FastAPI runs it in a thread pool.
    @app.post("/rag/query")
    def rag_query(query: RagQuery) -> RagResponse:
        try:
            start = time.perf_counter()
            hits = rag_pipeline.retriever.retrieve(query.question, query.top_k)
            retrieval_ms = (time.perf_counter() - start) * 1000
        except (HttpError, urllib.error.URLError, OSError) as exc:
            raise HTTPException(503, f"retrieval failed (embedder/qdrant): {exc}") from exc
        try:
            ans = rag_pipeline.answer_from_hits(query.question, hits, llm, retrieval_ms)
        except (StreamError, urllib.error.URLError, OSError) as exc:
            raise HTTPException(502, f"generation failed (llm): {exc}") from exc
        return RagResponse(
            answer=ans.answer,
            model=settings.model_name,
            sources=[Source(doc_id=h.doc_id, page=h.page, score=h.score) for h in hits],
            retrieval_ms=round(ans.retrieval_ms, 1),
            ttft_ms=round(ans.ttft_ms, 1),
            total_ms=round(ans.retrieval_ms + ans.total_ms, 1),
            prompt_tokens=ans.prompt_tokens,
            completion_tokens=ans.gen_tokens,
        )

    @app.get("/health", response_model=HealthResponse)
    async def health(request: Request) -> JSONResponse:
        client = _client(request)

        async def check(url: str) -> str:
            try:
                resp = await client.get(url, timeout=3.0)
            except httpx.HTTPError as exc:
                return f"unreachable ({type(exc).__name__})"
            # llama-server answers 503 while the model is loading
            return "ok" if resp.status_code == 200 else f"http {resp.status_code}"

        states = await asyncio.gather(*(check(u) for u in health_urls.values()))
        services = dict(zip(health_urls, states, strict=True))
        ok = all(s == "ok" for s in services.values())
        body = HealthResponse(status="ok" if ok else "degraded", services=services)
        return JSONResponse(body.model_dump(), status_code=200 if ok else 503)

    return app
