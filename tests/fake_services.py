"""In-memory fakes of the TEI and Qdrant HTTP APIs, for tests without Docker."""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

DIM = 16


def fake_embedding(text: str, dim: int = DIM) -> list[float]:
    """Bag-of-words hashing: texts sharing words get similar vectors (cosine)."""
    vec = [0.0] * dim
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % dim] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class FakeQdrant:
    def __init__(self) -> None:
        self.collections: dict[str, dict[str, Any]] = {}
        self.requests: list[tuple[str, str]] = []


def _handler(qdrant: FakeQdrant, tei_batches: list[int]) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:
            pass

        def _send(self, code: int, body: Any) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _body(self) -> Any:
            length = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(length)) if length else None

        def _route(self, method: str) -> None:
            path = self.path.split("?")[0]
            body = self._body()
            qdrant.requests.append((method, path))
            if path in ("/health", "/healthz"):
                return self._send(200, {})
            if path == "/embed":
                inputs = body["inputs"]
                tei_batches.append(len(inputs))
                return self._send(200, [fake_embedding(t) for t in inputs])
            m = re.fullmatch(r"/collections/([^/]+)(/.*)?", path)
            if not m:
                return self._send(404, {"status": "not found"})
            name, sub = m.group(1), m.group(2) or ""
            col = qdrant.collections.get(name)
            if sub == "" and method == "GET":
                return self._send(200, {"result": {}}) if col else self._send(404, {})
            if sub == "" and method == "DELETE":
                qdrant.collections.pop(name, None)
                return self._send(200, {"result": True})
            if sub == "" and method == "PUT":
                qdrant.collections[name] = {"size": body["vectors"]["size"], "points": {}}
                return self._send(200, {"result": True})
            if col is None:
                return self._send(404, {"status": f"collection {name} not found"})
            if sub == "/points" and method == "PUT":
                for p in body["points"]:
                    if len(p["vector"]) != col["size"]:
                        return self._send(400, {"status": "wrong vector size"})
                    col["points"][p["id"]] = p
                return self._send(200, {"result": {"status": "completed"}})
            if sub == "/points/count":
                return self._send(200, {"result": {"count": len(col["points"])}})
            if sub == "/points/query":
                q = body["query"]
                scored = sorted(
                    ({"id": p["id"], "payload": p["payload"],
                      "score": sum(a * b for a, b in zip(q, p["vector"], strict=True))}
                     for p in col["points"].values()),
                    key=lambda p: -p["score"],
                )  # fmt: skip
                return self._send(200, {"result": {"points": scored[: body["limit"]]}})
            return self._send(404, {})

        def do_GET(self) -> None:
            self._route("GET")

        def do_PUT(self) -> None:
            self._route("PUT")

        def do_POST(self) -> None:
            self._route("POST")

        def do_DELETE(self) -> None:
            self._route("DELETE")

    return Handler


@contextmanager
def fake_rag_services() -> Iterator[tuple[str, FakeQdrant, list[int]]]:
    """One HTTP server answering both the TEI and the Qdrant routes."""
    qdrant = FakeQdrant()
    batches: list[int] = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler(qdrant, batches))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", qdrant, batches
    finally:
        server.shutdown()
        server.server_close()
