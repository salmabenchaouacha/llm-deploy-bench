"""Tests for the logging helpers."""

from __future__ import annotations

import json
import logging

from llmbench.log import JsonFormatter, KeyValueFormatter, get_logger


def _record(**extra: object) -> logging.LogRecord:
    record = logging.makeLogRecord({"name": "llmbench.test", "levelname": "INFO", "msg": "hi"})
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def test_json_formatter_keeps_extra_fields() -> None:
    line = JsonFormatter().format(_record(model="qwen", tokens_per_s=42.5))
    payload = json.loads(line)
    assert payload["msg"] == "hi"
    assert payload["model"] == "qwen"
    assert payload["tokens_per_s"] == 42.5


def test_key_value_formatter_appends_extras() -> None:
    line = KeyValueFormatter().format(_record(quant="gguf_q4_k_m"))
    assert line.endswith("quant=gguf_q4_k_m")


def test_get_logger_is_namespaced() -> None:
    assert get_logger("rag.retriever").name == "llmbench.rag.retriever"
    assert get_logger("llmbench.eval").name == "llmbench.eval"
