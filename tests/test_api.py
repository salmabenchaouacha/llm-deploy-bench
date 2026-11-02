"""API gateway tests against fake llama-server / TEI / Qdrant (no GPU, no Docker)."""

from __future__ import annotations

import json
from collections.abc import Iterator

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from llmbench.api.app import create_app
from llmbench.api.settings import ApiSettings
from llmbench.rag.pipeline import RagPipeline
from llmbench.rag.store import Hit
from tests.fake_llama_server import FakeState, fake_llama_server
from tests.fake_services import fake_rag_services

DEAD_URL = "http://127.0.0.1:9"  # discard port: nothing listens


class StubRetriever:
    k = 5

    def __init__(self) -> None:
        self.calls: list[tuple[str, int | None]] = []

    def retrieve(self, question: str, k: int | None = None) -> list[Hit]:
        self.calls.append((question, k))
        return [Hit("c1", 0.81, "airbus_fy2025", 3, "Airbus delivered 793 aircraft.")]


@pytest.fixture
def llm() -> Iterator[tuple[str, FakeState]]:
    state = FakeState(words=["Airbus", "delivered", "793", "aircraft", "[airbus_fy2025", "p.3]"])
    with fake_llama_server(state) as url:
        yield url, state


def _client(llm_url: str, services_url: str = DEAD_URL) -> tuple[TestClient, StubRetriever]:
    retriever = StubRetriever()
    pipeline = RagPipeline(retriever, "Context:\n{context}\nQuestion: {question}", 64)  # type: ignore[arg-type]
    settings = ApiSettings(llm_url=llm_url, embedder_url=services_url, qdrant_url=services_url)
    return TestClient(create_app(settings, pipeline)), retriever


def test_settings_from_env() -> None:
    s = ApiSettings.from_env({"LLM_URL": "http://llm:8080", "REQUEST_TIMEOUT_S": "30"})
    assert s.llm_url == "http://llm:8080" and s.request_timeout_s == 30.0
    assert s.embedder_url is None  # unset variables keep the defaults


def test_chat_completion_is_proxied(llm: tuple[str, FakeState]) -> None:
    url, state = llm
    client, _ = _client(url)
    with client:
        body = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 3}
        resp = client.post("/v1/chat/completions", json=body)
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == "Airbus delivered 793"
    assert state.requests[-1]["max_tokens"] == 3


def test_streaming_relays_sse_and_forwards_extensions(llm: tuple[str, FakeState]) -> None:
    url, state = llm
    client, _ = _client(url)
    body = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 4,
            "stream": True, "ignore_eos": True}  # fmt: skip
    with client, client.stream("POST", "/v1/chat/completions", json=body) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        lines = [ln for ln in resp.iter_lines() if ln.startswith("data:")]
    assert lines[-1] == "data: [DONE]"
    words = [
        json.loads(ln[5:])["choices"][0]["delta"].get("content", "")
        for ln in lines[:-1]
        if json.loads(ln[5:]).get("choices")
    ]
    assert "".join(words).split() == ["Airbus", "delivered", "793", "aircraft"]
    assert state.requests[-1]["ignore_eos"] is True  # llama-server extension passed through


@pytest.mark.parametrize(
    "body",
    [
        {"messages": []},
        {"messages": [{"role": "robot", "content": "hi"}]},
        {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 0},
    ],
)
def test_invalid_requests_are_rejected_before_the_llm(
    llm: tuple[str, FakeState], body: dict[str, object]
) -> None:
    url, state = llm
    client, _ = _client(url)
    with client:
        assert client.post("/v1/chat/completions", json=body).status_code == 422
    assert state.requests == []


def test_llm_down_gives_502() -> None:
    client, _ = _client(DEAD_URL)
    with client:
        resp = client.post("/v1/chat/completions",
                           json={"messages": [{"role": "user", "content": "hi"}]})  # fmt: skip
    assert resp.status_code == 502


def test_models_lists_served_model(llm: tuple[str, FakeState]) -> None:
    client, _ = _client(llm[0])
    with client:
        assert client.get("/v1/models").json()["data"][0]["id"] == "fake-model"


def test_rag_query_returns_answer_and_sources(llm: tuple[str, FakeState]) -> None:
    url, state = llm
    client, retriever = _client(url)
    with client:
        resp = client.post("/rag/query", json={"question": "How many aircraft?", "top_k": 3})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"].startswith("Airbus delivered 793")
    assert body["sources"] == [{"doc_id": "airbus_fy2025", "page": 3, "score": 0.81}]
    assert body["completion_tokens"] == 6 and body["total_ms"] >= body["retrieval_ms"]
    assert retriever.calls == [("How many aircraft?", 3)]
    prompt = state.requests[-1]["messages"][0]["content"]
    assert "[airbus_fy2025 p.3]" in prompt and "How many aircraft?" in prompt


def test_health_ok_when_all_services_answer(llm: tuple[str, FakeState]) -> None:
    with fake_rag_services() as (services_url, _, _):
        client, _ = _client(llm[0], services_url)
        with client:
            resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok",
                           "services": {"llm": "ok", "embedder": "ok", "qdrant": "ok"}}  # fmt: skip


def test_health_degraded_when_a_service_is_down(llm: tuple[str, FakeState]) -> None:
    url, state = llm
    state.healthy = False  # llama-server answers 503 while loading the model
    client, _ = _client(url)
    with client:
        resp = client.get("/health")
    assert resp.status_code == 503
    services = resp.json()["services"]
    assert services["llm"] == "http 503" and services["qdrant"].startswith("unreachable")
