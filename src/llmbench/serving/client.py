"""Minimal streaming client for OpenAI-compatible chat endpoints (llama-server, vLLM...).

Standard library only (``urllib``): the benchmark has no extra dependency and the
measurement path stays simple to audit.

Time-to-first-token (TTFT) is measured client-side, from sending the request to the
first non-empty content delta: it is what a user actually waits for.
"""

from __future__ import annotations

import json
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StreamResult:
    text: str
    ttft_s: float
    total_s: float
    n_chunks: int
    server_timings: dict[str, Any] | None = None
    usage: dict[str, Any] | None = None
    finish_reason: str | None = None
    raw_events: int = field(default=0, repr=False)


class StreamError(RuntimeError):
    """The server answered something we cannot interpret as a completion stream."""


def _post_json(url: str, payload: dict[str, Any], timeout: float) -> Any:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return urllib.request.urlopen(req, timeout=timeout)


def stream_chat(
    base_url: str,
    messages: list[dict[str, str]],
    *,
    max_tokens: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
    seed: int | None = 42,
    ignore_eos: bool = False,
    allow_empty: bool = False,
    cache_prompt: bool | None = None,
    timeout: float = 600.0,
    clock: Callable[[], float] = time.perf_counter,
) -> StreamResult:
    """Send a streaming ``/v1/chat/completions`` request and time it.

    ``ignore_eos`` and ``cache_prompt`` are llama-server extensions (ignored by other
    servers). For benchmarks use ``cache_prompt=False``: otherwise a repeated prompt
    reuses the KV cache and its time-to-first-token is artificially near zero.
    """
    payload: dict[str, Any] = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if seed is not None:
        payload["seed"] = seed
    if ignore_eos:
        payload["ignore_eos"] = True
    if cache_prompt is not None:
        payload["cache_prompt"] = cache_prompt

    parts: list[str] = []
    ttft: float | None = None
    n_chunks = raw_events = 0
    timings: dict[str, Any] | None = None
    usage: dict[str, Any] | None = None
    finish: str | None = None

    start = clock()
    with _post_json(f"{base_url}/v1/chat/completions", payload, timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                break
            raw_events += 1
            try:
                event = json.loads(data)
            except json.JSONDecodeError as exc:
                raise StreamError(f"invalid SSE payload: {data[:200]}") from exc
            if "error" in event:
                raise StreamError(f"server error: {event['error']}")
            timings = event.get("timings") or timings
            usage = event.get("usage") or usage
            for choice in event.get("choices") or []:
                finish = choice.get("finish_reason") or finish
                content = (choice.get("delta") or {}).get("content")
                if content:
                    if ttft is None:
                        ttft = clock() - start
                    parts.append(content)
                    n_chunks += 1
    total = clock() - start

    if ttft is None:
        if not allow_empty:
            raise StreamError("stream ended without any generated content")
        ttft = total  # the model stopped immediately (EOS as first token): an empty answer
    return StreamResult(
        text="".join(parts),
        ttft_s=ttft,
        total_s=total,
        n_chunks=n_chunks,
        server_timings=timings,
        usage=usage,
        finish_reason=finish,
        raw_events=raw_events,
    )


def count_tokens(base_url: str, text: str, timeout: float = 30.0) -> int:
    """Exact token count with the served model's tokenizer (llama-server ``/tokenize``)."""
    with _post_json(f"{base_url}/tokenize", {"content": text}, timeout) as resp:
        body = json.loads(resp.read())
    return len(body["tokens"])
