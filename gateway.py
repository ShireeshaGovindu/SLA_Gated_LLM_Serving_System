#!/usr/bin/env python3
"""SLA-gated FastAPI gateway for an SGLang /generate endpoint.

Environment variables:
  SGLANG_URL=http://127.0.0.1:30000
  KV_LIMIT=90
  GATEWAY_MAX_INFLIGHT=128
  ENGINE_TIMEOUT_SECONDS=300
  METRICS_CACHE_TTL_SECONDS=0.25
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

SGLANG_URL = os.getenv("SGLANG_URL", "http://127.0.0.1:30000").rstrip("/")
KV_LIMIT = float(os.getenv("KV_LIMIT", "90"))
MAX_INFLIGHT = int(os.getenv("GATEWAY_MAX_INFLIGHT", "128"))
ENGINE_TIMEOUT = float(os.getenv("ENGINE_TIMEOUT_SECONDS", "300"))
METRICS_CACHE_TTL = float(os.getenv("METRICS_CACHE_TTL_SECONDS", "0.25"))

HTTP: httpx.AsyncClient | None = None
ENGINE_SEMAPHORE = asyncio.Semaphore(MAX_INFLIGHT)

ADMISSION_LOCK = asyncio.Lock()

_metrics_cache: dict[str, Any] = {"time": 0.0, "text": ""}
_counters = {
    "requests_received": 0,
    "requests_accepted": 0,
    "requests_shed": 0,
    "engine_errors": 0,
    "json_validation_errors": 0,
    "high_priority_allowed": 0,
    "kv_metrics_errors": 0,
}
_counter_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global HTTP
    HTTP = httpx.AsyncClient(timeout=httpx.Timeout(ENGINE_TIMEOUT, connect=15.0))
    try:
        yield
    finally:
        await HTTP.aclose()
        HTTP = None


app = FastAPI(title="SLA-Gated LLM Inference Gateway", version="1.0.0", lifespan=lifespan)


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1)
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    priority: str = Field(default="standard", pattern="^(standard|high)$")
    schema: dict[str, Any] | None = None
    request_id: str | None = None


def _metric_value(metrics_text: str, metric_name: str) -> float | None:
    """Read a Prometheus sample, allowing optional labels after the metric name."""
    pattern = re.compile(r"^" + re.escape(metric_name) + r"(?:\{[^}]*\})?\s+([-+0-9.eE]+)\s*$", re.M)
    match = pattern.search(metrics_text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


async def get_engine_metrics_text() -> str:
    now = time.monotonic()
    if now - _metrics_cache["time"] < METRICS_CACHE_TTL and _metrics_cache["text"]:
        return str(_metrics_cache["text"])
    if HTTP is None:
        return ""
    try:
        response = await HTTP.get(f"{SGLANG_URL}/metrics", timeout=5.0)
        response.raise_for_status()
        _metrics_cache.update({"time": now, "text": response.text})
        return response.text
    except Exception:
        return str(_metrics_cache["text"] or "")


async def get_token_occupancy_percent() -> float | None:
    text = await get_engine_metrics_text()

    capacity = _metric_value(
        text, "sglang:max_total_num_tokens"
    )
    available = _metric_value(
        text, "sglang:kv_available_tokens"
    )

    if (
        capacity is None
        or available is None
        or capacity <= 0
        or available < 0
        or available > capacity
    ):
        return None

    return (capacity - available) / capacity * 100.0


async def _increment(key: str) -> None:
    async with _counter_lock:
        _counters[key] = _counters.get(key, 0) + 1


async def _generate(
    payload: GenerateRequest,
    request: Request,
    json_mode: bool,
) -> dict[str, Any]:

    await _increment("requests_received")

    priority = payload.priority.lower()

    # 1. SLA-based admission and KV-cache pressure check
    async with ADMISSION_LOCK:
        occupancy = await get_token_occupancy_percent()

        if occupancy is None:
            await _increment("kv_metrics_errors")

            if priority != "high":
                await _increment("requests_shed")
                raise HTTPException(
                    status_code=503,
                    detail={
                        "error": "kv_metrics_unavailable",
                        "reason": "Cannot verify KV-cache capacity.",
                        "threshold_percent": KV_LIMIT,
                    },
                    headers={"Retry-After": "1"},
                )

        elif occupancy >= KV_LIMIT and priority != "high":
            await _increment("requests_shed")
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "kv_cache_capacity_exceeded",
                    "reason": (
                        f"Standard request rejected: "
                        f"token usage {occupancy:.1f}%"
                    ),
                    "occupancy_percent": round(occupancy, 3),
                    "threshold_percent": KV_LIMIT,
                },
                headers={"Retry-After": "1"},
            )

        if priority == "high" and (
            occupancy is None or occupancy >= KV_LIMIT
        ):
            await _increment("high_priority_allowed")

        await _increment("requests_accepted")

    # 2. Check engine HTTP client
    if HTTP is None:
        raise HTTPException(
            status_code=503,
            detail="Gateway engine client is not ready",
        )

    # 3. Construct SGLang sampling parameters
    sampling_params: dict[str, Any] = {
        "temperature": payload.temperature,
        "max_new_tokens": payload.max_tokens,
    }

    if json_mode:
        if not payload.schema:
            raise HTTPException(
                status_code=400,
                detail="schema is required for /generate-json",
            )

        sampling_params["json_schema"] = json.dumps(
            payload.schema,
            separators=(",", ":"),
        )

    engine_payload = {
        "text": payload.prompt,
        "sampling_params": sampling_params,
    }

    # 4. Forward admitted request to SGLang
    started = time.perf_counter()

    try:
        async with ENGINE_SEMAPHORE:
            response = await HTTP.post(
                f"{SGLANG_URL}/generate",
                json=engine_payload,
            )

        elapsed = time.perf_counter() - started

        if response.status_code >= 400:
            await _increment("engine_errors")
            raise HTTPException(
                status_code=502,
                detail={
                    "error": "sglang_upstream_error",
                    "upstream_status": response.status_code,
                    "upstream_body": response.text[:4000],
                },
            )

        result = response.json()

    except HTTPException:
        raise

    except httpx.TimeoutException:
        await _increment("engine_errors")
        raise HTTPException(
            status_code=504,
            detail="SGLang request timed out",
        )

    except Exception as exc:
        await _increment("engine_errors")
        raise HTTPException(
            status_code=502,
            detail=(
                f"SGLang request failed: "
                f"{type(exc).__name__}: {exc}"
            ),
        )

    # 5. Construct gateway response
    output_text = result.get("text", "")
    meta = result.get("meta_info") or {}

    answer: dict[str, Any] = {
        "request_id": payload.request_id,
        "text": output_text,
        "meta_info": meta,
        "gateway": {
            "elapsed_seconds": round(elapsed, 6),
            "kv_occupancy_percent_at_admission": (
                round(occupancy, 3)
                if occupancy is not None
                else None
            ),
            "kv_threshold_percent": KV_LIMIT,
            "kv_metrics_available": occupancy is not None,
            "priority": priority,
        },
    }

    # 6. Validate constrained JSON response
    if json_mode:
        try:
            parsed = json.loads(output_text)

            validation = validate_schema_minimal(
                parsed,
                payload.schema or {},
            )

            if not validation["valid"]:
                await _increment("json_validation_errors")
                raise HTTPException(
                    status_code=502,
                    detail={
                        "error": "schema_validation_failed",
                        "validation": validation,
                        "raw_output": output_text[:4000],
                        "meta_info": meta,
                    },
                )

            answer["parsed_json"] = parsed
            answer["schema_validation"] = validation

        except HTTPException:
            raise

        except Exception as exc:
            await _increment("json_validation_errors")
            raise HTTPException(
                status_code=502,
                detail={
                    "error": "invalid_json_output",
                    "reason": str(exc),
                    "raw_output": output_text[:4000],
                    "meta_info": meta,
                },
            )

    return answer

def validate_schema_minimal(value: Any, schema: dict[str, Any]) -> dict[str, Any]:
    """Small dependency-free validator for common JSON Schema object constraints.

    Supports type, required, properties, additionalProperties=false, and primitive
    string/integer/number/boolean types. It is intentionally not a full JSON Schema
    implementation.
    """
    errors: list[str] = []
    expected_type = schema.get("type")
    type_ok = {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
        "null": value is None,
    }
    if expected_type in type_ok and not type_ok[expected_type]:
        errors.append(f"Expected top-level type {expected_type}")

    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"Missing required property: {key}")
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            extras = sorted(set(value) - set(properties))
            if extras:
                errors.append(f"Unexpected properties: {extras}")
        for key, spec in properties.items():
            if key not in value or not isinstance(spec, dict):
                continue
            typ = spec.get("type")
            valid = {
                "object": isinstance(value[key], dict),
                "array": isinstance(value[key], list),
                "string": isinstance(value[key], str),
                "integer": isinstance(value[key], int) and not isinstance(value[key], bool),
                "number": isinstance(value[key], (int, float)) and not isinstance(value[key], bool),
                "boolean": isinstance(value[key], bool),
                "null": value[key] is None,
            }
            if typ in valid and not valid[typ]:
                errors.append(f"Property {key!r} should be {typ}")
    return {"valid": not errors, "errors": errors}


@app.get("/health")
async def health():
    return {"status": "ok", "engine_url": SGLANG_URL, "kv_threshold_percent": KV_LIMIT}


@app.get("/metrics")
async def gateway_metrics():
    occupancy = await get_token_occupancy_percent()
    async with _counter_lock:
        counters = dict(_counters)
    return {
        **counters,
        "kv_occupancy_percent": occupancy,
        "kv_metrics_available": occupancy is not None,
        "kv_threshold_percent": KV_LIMIT,
        "gateway_max_inflight": MAX_INFLIGHT,
        "engine_url": SGLANG_URL,
    }


@app.get("/engine-metrics")
async def engine_metrics():
    return {"prometheus_text": await get_engine_metrics_text()}


@app.post("/generate")
async def generate(payload: GenerateRequest, request: Request):
    return await _generate(payload, request, json_mode=False)


@app.post("/generate-json")
async def generate_json(payload: GenerateRequest, request: Request):
    return await _generate(payload, request, json_mode=True)
