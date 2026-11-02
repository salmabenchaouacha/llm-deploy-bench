"""Request / response bodies. FastAPI validates them and publishes them in /docs (OpenAPI)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    """Subset of the OpenAI schema we validate; other fields are forwarded untouched.

    ``extra="allow"`` lets llama-server extensions (``ignore_eos``, ``cache_prompt``,
    ``stream_options``...) through, like a real OpenAI-compatible gateway.
    """

    model_config = ConfigDict(extra="allow")

    model: str | None = None
    messages: list[ChatMessage] = Field(..., min_length=1)
    stream: bool = False
    max_tokens: int | None = Field(None, gt=0, le=4096)
    temperature: float | None = Field(None, ge=0.0, le=2.0)


class RagQuery(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    top_k: int | None = Field(None, ge=1, le=20)


class Source(BaseModel):
    doc_id: str
    page: int
    score: float


class RagResponse(BaseModel):
    answer: str
    model: str
    sources: list[Source]
    retrieval_ms: float
    ttft_ms: float
    total_ms: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    services: dict[str, str]
