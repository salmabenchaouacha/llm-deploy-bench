"""Tiny JSON-over-HTTP helpers (standard library only).

Every service of the platform (llama-server, TEI, Qdrant) speaks JSON over HTTP, so a
heavyweight SDK per service is not needed. Keeping the transport in one place makes
errors uniform (``HttpError`` carries the status and the server's message) and lets tests
run against small fake servers.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


class HttpError(RuntimeError):
    def __init__(self, method: str, url: str, status: int | None, body: str) -> None:
        self.status = status
        self.body = body
        super().__init__(f"{method} {url} -> {status or 'no response'}: {body[:500]}")


def request_json(
    method: str, url: str, payload: Any | None = None, *, timeout: float = 60.0
) -> Any:
    """Send ``payload`` as JSON and return the decoded JSON answer (None if empty)."""
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, method=method, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        raise HttpError(method, url, exc.code, exc.read().decode(errors="replace")) from exc
    except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
        raise HttpError(method, url, None, str(exc)) from exc
    return json.loads(raw) if raw else None
