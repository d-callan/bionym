"""FastAPI smoke tests via TestClient — endpoints, validation, and
error mapping. No network: main._resolver is monkeypatched."""
import os
import sys
from pathlib import Path

os.environ["BIONYM_ALLOW_MOCK"] = "1"  # read at backend import time
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import main  # noqa: E402  (backend/main.py)
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from bionym.graph import KnowledgeGraph  # noqa: E402
from bionym.jev import JevClient, JevError  # noqa: E402
from bionym.llm import LlmClient  # noqa: E402

from fakes import FakeJev, FakeLlm, mini_graph  # noqa: E402

client = TestClient(main.app)


class _StubResolver:
    """Minimal stand-in for resolver-dependent endpoints."""

    def __init__(self):
        self.jev = JevClient(mock=True)
        self.llm = LlmClient(mock=True)
        self.graph = mini_graph()
        self.fail = None

    def resolve(self, identifier, depth=6, summarize=False, propose=False):
        if self.fail:
            raise self.fail
        return self.graph


@pytest.fixture
def resolver(monkeypatch):
    r = _StubResolver()
    monkeypatch.setattr(main, "_resolver", lambda mock_jev=False: r)
    return r


def test_resolve_returns_graph(resolver):
    resp = client.get("/api/resolve", params={
        "identifier": "PF3D7_0710100", "mock_jev": "true"})
    assert resp.status_code == 200
    body = resp.json()
    assert "veupathdb:PF3D7_0710100" in body["nodes"]
    assert "edges" in body and "metadata" in body


def test_ask_returns_answer(resolver):
    resp = client.post("/api/ask", params={"mock_jev": "true"}, json={
        "question": "what datasets exist?",
        "graph": resolver.graph.model_dump(mode="json"),
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "mock answer"
    assert body["scored"] is True  # mock jev nouls (0.5) >= CAT_THRESHOLD


def test_ask_requires_question(resolver):
    resp = client.post("/api/ask", params={"mock_jev": "true"}, json={
        "question": "", "graph": {}})
    assert resp.status_code == 422


def test_ask_rejects_invalid_graph(resolver):
    resp = client.post("/api/ask", params={"mock_jev": "true"}, json={
        "question": "q", "graph": {"nodes": "not-a-dict"}})
    assert resp.status_code == 422


def test_jev_error_maps_to_502(resolver, monkeypatch):
    resolver.fail = JevError("upstream boom")
    resp = client.get("/api/resolve", params={
        "identifier": "x", "mock_jev": "true"})
    assert resp.status_code == 502
    assert "JEV" in resp.json()["detail"]


def test_mock_gate_blocks_without_env():
    """mock_jev=true against the real _resolver must stay gated."""
    monkey_val = main._MOCK_ALLOWED
    try:
        main._MOCK_ALLOWED = False
        resp = client.get("/api/resolve", params={
            "identifier": "x", "mock_jev": "true"})
        assert resp.status_code == 403
    finally:
        main._MOCK_ALLOWED = monkey_val
