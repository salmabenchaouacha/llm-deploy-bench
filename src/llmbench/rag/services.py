"""Health checks for the RAG services (Qdrant, TEI) started by docker compose."""

from __future__ import annotations

from llmbench.config import RagConfig
from llmbench.log import get_logger
from llmbench.serving.llama_server import wait_for_health

log = get_logger(__name__)


def wait_for_rag_services(cfg: RagConfig, timeout_s: float = 900.0) -> None:
    """Block until Qdrant and TEI answer. TEI's first start downloads the model (~2.3 GB)."""
    checks = {
        "qdrant": f"{cfg.qdrant.url}/healthz",
        "embedder": f"{cfg.embedder.url}/health",
    }
    for name, url in checks.items():
        log.info("waiting for service", extra={"service": name, "url": url})
        try:
            waited = wait_for_health(url, timeout_s, poll_s=2.0)
        except TimeoutError as exc:
            raise SystemExit(
                f"{name} not reachable at {url}. Is it running? `make rag-up`, then "
                f"`docker compose logs {name}`"
            ) from exc
        log.info("service ready", extra={"service": name, "waited_s": round(waited, 1)})
