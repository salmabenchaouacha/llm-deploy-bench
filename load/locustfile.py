"""Locust load test of the llmbench API. Run through `make load` (llmbench.load.run).

Two user types, selected on the command line:
* ``ChatUser`` - POST /v1/chat/completions, streaming, fixed output length (ignore_eos):
  measures the serving capacity of the LLM behind the gateway.
* ``RagUser``  - POST /rag/query with questions of the evaluation set: the full pipeline
  (embedding on CPU + Qdrant + generation).

Closed loop (``wait_time = constant(0)``): N users = N requests in flight at all times.
Each request is also written as one JSON line to ``$LOAD_RECORDS`` with exact client-side
timings (Locust's own stats stop the clock at the response headers of a stream).
"""

from __future__ import annotations

import json
import os
import random
import time
from itertools import count
from pathlib import Path
from typing import Any

from locust import HttpUser, constant, task

ROOT = Path(__file__).resolve().parents[1]
MAX_TOKENS = int(os.environ.get("LOAD_MAX_TOKENS", "128"))
_RECORDS = os.environ.get("LOAD_RECORDS")
_records_fh = open(_RECORDS, "a", encoding="utf-8") if _RECORDS else None  # noqa: SIM115
_user_ids = count()

PROMPTS = [
    "Explain in a few sentences what quantization does to a large language model.",
    "Write a short paragraph describing how a retrieval-augmented generation system works.",
    "List the main steps to deploy a web API in a Docker container.",
]


def _questions() -> list[str]:
    path = ROOT / "data" / "eval" / "questions.jsonl"
    items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return [i["question"] for i in items if i.get("answerable") and i.get("status") != "rejected"]


def record(**fields: Any) -> None:
    if _records_fh is not None:
        _records_fh.write(json.dumps(fields) + "\n")
        _records_fh.flush()


class ChatUser(HttpUser):
    wait_time = constant(0)

    def on_start(self) -> None:
        self.rng = random.Random(42 + next(_user_ids))

    @task
    def chat(self) -> None:
        payload = {
            "messages": [{"role": "user", "content": self.rng.choice(PROMPTS)}],
            "max_tokens": MAX_TOKENS,
            "temperature": 0.0,
            "stream": True,
            "stream_options": {"include_usage": True},
            "ignore_eos": True,  # every request generates exactly MAX_TOKENS tokens
            "cache_prompt": False,
        }
        start = time.time()
        t0 = time.perf_counter()
        ttft: float | None = None
        tokens: int | None = None
        chunks = 0
        error: str | None = None
        with self.client.post("/v1/chat/completions", json=payload, stream=True,
                              catch_response=True, name="chat (stream)") as resp:  # fmt: skip
            if resp.status_code != 200:
                error = f"http {resp.status_code}"
                resp.failure(error)
            else:
                try:
                    for line in resp.iter_lines():
                        if not line.startswith(b"data:"):
                            continue
                        data = line[5:].strip()
                        if data == b"[DONE]":
                            break
                        event = json.loads(data)
                        if event.get("usage"):
                            tokens = event["usage"].get("completion_tokens")
                        for choice in event.get("choices") or []:
                            if (choice.get("delta") or {}).get("content"):
                                chunks += 1
                                if ttft is None:
                                    ttft = time.perf_counter() - t0
                    resp.success()
                except Exception as exc:  # broken stream: count it, keep the test running
                    error = f"{type(exc).__name__}: {exc}"
                    resp.failure(error)
        latency = time.perf_counter() - t0
        record(scenario="chat", start=start, ok=error is None, latency_ms=latency * 1000,
               ttft_ms=None if ttft is None else ttft * 1000,
               tokens=tokens if tokens is not None else chunks, error=error)  # fmt: skip


class RagUser(HttpUser):
    wait_time = constant(0)

    def on_start(self) -> None:
        self.rng = random.Random(42 + next(_user_ids))
        self.questions = _questions()

    @task
    def rag(self) -> None:
        start = time.time()
        t0 = time.perf_counter()
        body: dict[str, Any] = {}
        error: str | None = None
        question = self.rng.choice(self.questions)
        with self.client.post("/rag/query", json={"question": question},
                              catch_response=True, name="rag query") as resp:  # fmt: skip
            if resp.status_code == 200:
                body = resp.json()
                resp.success()
            else:
                error = f"http {resp.status_code}"
                resp.failure(error)
        record(scenario="rag", start=start, ok=error is None,
               latency_ms=(time.perf_counter() - t0) * 1000,
               ttft_ms=None if error else body["retrieval_ms"] + body["ttft_ms"],
               tokens=body.get("completion_tokens"), error=error)  # fmt: skip
