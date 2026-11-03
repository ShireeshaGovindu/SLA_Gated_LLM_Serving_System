import json
from fastapi.testclient import TestClient

import gateway
from test_gateway import FakeHTTP, FakeResponse, make_client, close_client


SCHEMA = {
    "type": "object",
    "properties": {
        "topic": {"type": "string"},
        "summary": {"type": "string"},
        "score": {"type": "integer"},
    },
    "required": ["topic", "summary", "score"],
    "additionalProperties": False,
}


def test_minimal_schema_validator_accepts_valid_object():
    value = {"topic": "GPU", "summary": "Fast inference", "score": 3}
    result = gateway.validate_schema_minimal(value, SCHEMA)
    assert result["valid"] is True
    assert result["errors"] == []


def test_minimal_schema_validator_rejects_missing_property():
    value = {"topic": "GPU", "summary": "Fast inference"}
    result = gateway.validate_schema_minimal(value, SCHEMA)
    assert result["valid"] is False
    assert any("score" in error for error in result["errors"])


def test_minimal_schema_validator_rejects_wrong_type_and_extra_property():
    value = {"topic": "GPU", "summary": "Fast inference", "score": "three", "extra": True}
    result = gateway.validate_schema_minimal(value, SCHEMA)
    assert result["valid"] is False
    assert any("score" in error for error in result["errors"])
    assert any("Unexpected properties" in error for error in result["errors"])


def test_generate_json_returns_parsed_and_validated_object(monkeypatch):
    fake = FakeHTTP(FakeResponse(200, payload={
        "text": json.dumps({"topic": "GPU", "summary": "Fast inference", "score": 3}),
        "meta_info": {"prompt_tokens": 10, "completion_tokens": 8, "cached_tokens": 2},
    }))
    client = make_client(monkeypatch, 10.0, fake)
    try:
        response = client.post("/generate-json", json={"prompt": "Return JSON", "schema": SCHEMA})
        assert response.status_code == 200
        body = response.json()
        assert body["schema_validation"]["valid"] is True
        assert body["parsed_json"]["score"] == 3
        assert fake.post_calls[0]["json"]["sampling_params"]["json_schema"] == json.dumps(SCHEMA, separators=(",", ":"))
    finally:
        close_client(client)


def test_generate_json_invalid_output_returns_502(monkeypatch):
    fake = FakeHTTP(FakeResponse(200, payload={
        "text": '{"topic":"GPU","summary":',
        "meta_info": {},
    }))
    client = make_client(monkeypatch, 10.0, fake)
    try:
        response = client.post("/generate-json", json={"prompt": "Return JSON", "schema": SCHEMA})
        assert response.status_code == 502
        assert response.json()["detail"]["error"] == "invalid_json_output"
    finally:
        close_client(client)
