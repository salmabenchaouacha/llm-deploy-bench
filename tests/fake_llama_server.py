"""A tiny HTTP server that mimics llama-server's API, for tests without a GPU.

Implements ``GET /health``, ``POST /tokenize`` (1 token per word) and a streaming
``POST /v1/chat/completions`` that emits one SSE event per word, then a final event
with ``timings`` like llama-server does.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


class FakeState:
    def __init__(self, words: list[str], delay_s: float = 0.0, with_timings: bool = True):
        self.words = words
        self.delay_s = delay_s
        self.with_timings = with_timings
        self.requests: list[dict[str, Any]] = []
        self.healthy = True


def _handler(state: FakeState) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:  # keep pytest output clean
            pass

        def _json(self, code: int, body: dict[str, Any]) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._json(200 if state.healthy else 503, {"status": "ok"})
            else:
                self._json(404, {})

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/tokenize":
                self._json(200, {"tokens": list(range(len(body["content"].split())))})
                return
            state.requests.append(body)
            n = min(body["max_tokens"], len(state.words))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()

            def send(event: dict[str, Any] | str) -> None:
                payload = event if isinstance(event, str) else json.dumps(event)
                self.wfile.write(f"data: {payload}\n\n".encode())
                self.wfile.flush()

            for i, word in enumerate(state.words[:n]):
                time.sleep(state.delay_s)
                send({"choices": [{"index": 0, "delta": {"content": word + " "},
                                   "finish_reason": None}]})  # fmt: skip
                if i == 0:  # SSE comment line: clients must ignore it
                    self.wfile.write(b": keep-alive\n\n")
            final: dict[str, Any] = {
                "choices": [{"index": 0, "delta": {}, "finish_reason": "length"}]
            }
            if state.with_timings:
                final["timings"] = {"prompt_n": 7, "prompt_per_second": 500.0,
                                    "predicted_n": n, "predicted_per_second": 42.0}  # fmt: skip
            send(final)
            send("[DONE]")

    return Handler


@contextmanager
def fake_llama_server(state: FakeState) -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler(state))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
