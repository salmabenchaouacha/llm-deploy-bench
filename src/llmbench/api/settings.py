"""API settings, read from environment variables (12-factor: same image, any deployment)."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from pydantic import BaseModel, Field

from llmbench.config import DEFAULT_RAG_CONFIG

ENV_VARS = {
    "llm_url": "LLM_URL",
    "model_name": "LLM_MODEL",
    "rag_config": "RAG_CONFIG",
    "embedder_url": "EMBEDDER_URL",
    "qdrant_url": "QDRANT_URL",
    "request_timeout_s": "REQUEST_TIMEOUT_S",
}


class ApiSettings(BaseModel):
    llm_url: str = "http://127.0.0.1:8092"  # llama-server of `docker compose --profile serving`
    model_name: str = "llama3.2-3b-q4_k_m"  # best quality/speed trade-off (results/eval)
    rag_config: Path = DEFAULT_RAG_CONFIG
    embedder_url: str | None = None  # overrides configs/rag.yaml (service names in Docker)
    qdrant_url: str | None = None
    request_timeout_s: float = Field(600.0, gt=0)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> ApiSettings:
        env = os.environ if env is None else env
        values = {field: env[var] for field, var in ENV_VARS.items() if env.get(var)}
        return cls.model_validate(values)
