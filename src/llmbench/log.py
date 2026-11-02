"""Logging setup shared by every entry point.

- Level from ``LLMBENCH_LOG_LEVEL`` (default ``INFO``).
- ``LLMBENCH_LOG_JSON=1`` switches to one JSON object per line, which log
  collectors (Loki, ELK, Docker logging drivers) can parse without regexes.

Usage::

    from llmbench.log import get_logger
    log = get_logger(__name__)
    log.info("model loaded", extra={"model": "qwen2.5-1.5b", "quant": "gguf_q4_k_m"})
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from typing import Any

_STD_ATTRS = frozenset(vars(logging.makeLogRecord({})).keys()) | {"message", "asctime"}
_configured = False


class JsonFormatter(logging.Formatter):
    """Format records as single-line JSON, keeping fields passed through ``extra=``."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        payload.update({k: v for k, v in vars(record).items() if k not in _STD_ATTRS})
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


class KeyValueFormatter(logging.Formatter):
    """Human-readable format that still shows ``extra=`` fields as key=value."""

    def __init__(self) -> None:
        super().__init__("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S")

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        extras = {k: v for k, v in vars(record).items() if k not in _STD_ATTRS}
        return base + "".join(f" {k}={v}" for k, v in extras.items())


def configure_logging(level: str | None = None, *, json_format: bool | None = None) -> None:
    """Configure the ``llmbench`` logger once; later calls only update the level."""
    global _configured
    resolved_level = (level or os.getenv("LLMBENCH_LOG_LEVEL") or "INFO").upper()
    if json_format is None:
        json_format = os.getenv("LLMBENCH_LOG_JSON", "0").lower() in {"1", "true", "yes"}

    logger = logging.getLogger("llmbench")
    logger.setLevel(resolved_level)
    if not _configured:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter() if json_format else KeyValueFormatter())
        logger.addHandler(handler)
        logger.propagate = False
        _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the ``llmbench`` namespace, configuring logging on first use."""
    configure_logging()
    if not name.startswith("llmbench"):
        name = f"llmbench.{name}"
    return logging.getLogger(name)
