import asyncio
from types import SimpleNamespace

from fastapi.testclient import TestClient

import gateway


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


class FakeHTTP:
    def __init__(self, response=None):
        self.response = response or FakeResponse(
            200,
            {
                "text": "Example generated answer.",
                "meta_info": {
                    "prompt_tokens": 20,
                    "completion_tokens": 5,
                    "cached_tokens": 10,
                    "finish_reason": "stop",
                },
            },
        )
        self.post_calls = []

    async def post(self, url, json=None):
        self.post_calls.append({"url": url, "json": json})
        return self.response

    async def get(self, url, timeout=None):
        return FakeResponse(200, payload={}) if url.endswith("/health") else FakeResponse(
            200, payload={}, text="sglang:token_usage 0.1\n"
        )

    async def aclose(self):
        return None


def make_client(monkeypatch, occupancy, fake_http=None):
    async def fake_occupancy():
        return occupancy

    # Admission is based on sglang:token_usage (active tokens), see gateway.py.
    monkeypatch.setattr(gateway, "get_sglang_token_usage_percent", fake_occupancy)
    client = TestClient(gateway.app)
    client.__enter__()
    monkeypatch.setattr(gateway, "HTTP", fake_http or FakeHTTP())
    return client


def close_client(client):
    client.__exit__(None, None, None)


def test_health_endpoint():
    with TestClient(gateway.app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


def test_standard_request_shed_above_threshold(monkeypatch):
    client = make_client(monkeypatch, 98.31)
    try:
        response = client.post("/generate", json={"prompt": "hello", "priority": "standard"})
        assert response.status_code == 503
        assert "KV-cache usage is 98.3%" in response.json()["detail"]["reason"]
    finally:
        close_client(client)


def test_high_priority_allowed_above_threshold(monkeypatch):
    client = make_client(monkeypatch, 98.31)
    try:
        response = client.post("/generate", json={"prompt": "hello", "priority": "high"})
        assert response.status_code == 200
        assert response.json()["gateway"]["kv_occupancy_percent_at_admission"] == 98.31
    finally:
        close_client(client)


def test_standard_request_allowed_below_threshold(monkeypatch):
    client = make_client(monkeypatch, 89.9)
    try:
        response = client.post("/generate", json={"prompt": "hello", "priority": "standard"})
        assert response.status_code == 200
    finally:
        close_client(client)


def test_standard_request_allowed_exactly_at_threshold(monkeypatch):
    # The assignment sheds when usage *exceeds* 90%, so exactly 90.0 is admitted.
    client = make_client(monkeypatch, 90.0)
    try:
        response = client.post("/generate", json={"prompt": "hello", "priority": "standard"})
        assert response.status_code == 200
    finally:
        close_client(client)


def test_standard_request_rejected_just_above_threshold(monkeypatch):
    client = make_client(monkeypatch, 90.1)
    try:
        response = client.post("/generate", json={"prompt": "hello", "priority": "standard"})
        assert response.status_code == 503
        assert response.headers["retry-after"] == "1"
    finally:
        close_client(client)


def test_engine_upstream_error_becomes_502(monkeypatch):
    fake = FakeHTTP(FakeResponse(500, payload={}, text="upstream failure"))
    client = make_client(monkeypatch, 10.0, fake)
    try:
        response = client.post("/generate", json={"prompt": "hello"})
        assert response.status_code == 502
        assert response.json()["detail"]["error"] == "sglang_upstream_error"
    finally:
        close_client(client)

def test_standard_request_fails_closed_when_metrics_unavailable(monkeypatch):
    client = make_client(monkeypatch, None)
    try:
        response = client.post(
            "/generate",
            json={"prompt": "hello", "priority": "standard"},
        )
        assert response.status_code == 503
        assert response.json()["detail"]["error"] == "kv_metrics_unavailable"
        assert response.headers["retry-after"] == "1"
    finally:
        close_client(client)
def test_high_priority_allowed_when_metrics_unavailable(monkeypatch):
    client = make_client(monkeypatch, None)
    try:
        response = client.post(
            "/generate",
            json={"prompt": "hello", "priority": "high"},
        )
        assert response.status_code == 200
        assert response.json()["gateway"]["kv_metrics_available"] is False
    finally:
        close_client(client)
def test_engine_timeout_becomes_504(monkeypatch):
    import httpx
    class TimeoutHTTP(FakeHTTP):
        async def post(self, url, json=None):
            raise httpx.ReadTimeout("simulated engine timeout")
    client = make_client(monkeypatch, 10.0, TimeoutHTTP())
    try:
        response = client.post(
            "/generate",
            json={"prompt": "hello", "priority": "standard"},
        )
        assert response.status_code == 504
        assert response.json()["detail"] == "SGLang request timed out"
    finally:
        close_client(client)