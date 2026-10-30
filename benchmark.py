#!/usr/bin/env python3
"""Assignment benchmark runner for the SLA-Gated LLM Serving Engine.

Scenarios:
 A: 50 requests with distinct long system instructions / leading prefixes.
 B: 100 requests sharing a long (~2k-token target) reference context across 5 domains.
 C: 50 requests using a structured JSON schema.

The runner records per-request latency/status/token metadata, SGLang Prometheus
histogram deltas (TTFT and ITL), cache-hit-rate snapshots, GPU VRAM snapshots when
nvidia-smi is available, and JSON schema validation outcomes.

Run after the gateway and SGLang are healthy:
  python benchmark.py --gateway http://127.0.0.1:8081 --output-dir results

You can run one scenario at a time with --scenario A/B/C. The default runs all.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from transformers import AutoTokenizer

from config.benchmark_config import (SCENARIO_A_DOMAINS, 
                                     SCENARIO_B_QUESTIONS,
                                     SHARED_SYSTEM_PROMPT_BASE,
                                     DEFAULT_SCHEMA,
                                     DOMAINS)

import logging
LOGGER = logging.getLogger("benchmark")


def setup_logging() -> Path:
    log_dir = Path("Logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    log_path = log_dir / f"benchmark_{timestamp}.log"

    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()

    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s"
    ))
    LOGGER.addHandler(handler)
    LOGGER.propagate = False

    return log_path

SHARED_PROMPT_MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"
SHARED_PROMPT_TOKEN_COUNT = 1500

_tokenizer = AutoTokenizer.from_pretrained(
    SHARED_PROMPT_MODEL,
    trust_remote_code=True,
)

def _make_exact_shared_prefix(token_count: int = SHARED_PROMPT_TOKEN_COUNT) -> str:
    text = SHARED_SYSTEM_PROMPT_BASE.strip()
    token_ids = _tokenizer.encode(text, add_special_tokens=False)

    # Extend with relevant operational guidance, not arbitrary filler.
    extensions = [
        " For every recommendation, identify its operational consequence.",
        " Track queue depth, active requests, and rejected requests separately.",
        " Preserve identical shared-prefix text across all benchmark requests.",
        " Report the measurement method and limitations alongside each result.",
        " Treat missing telemetry as unknown rather than interpreting it as zero.",
        " Compare workloads only when their relevant input conditions are equivalent.",
        " Keep request-specific identifiers and questions after the shared prefix.",
    ]

    index = 0
    while len(token_ids) < token_count:
        text += extensions[index % len(extensions)]
        index += 1
        token_ids = _tokenizer.encode(text, add_special_tokens=False)

    # Decode exactly 1,500 tokenizer IDs and verify round-trip tokenization.
    token_ids = token_ids[:token_count]
    while True:
        candidate = _tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        verified_ids = _tokenizer.encode(candidate, add_special_tokens=False)

        if len(verified_ids) == token_count:
            return candidate

        # Adjust the token boundary until the decoded text round-trips exactly.
        if len(verified_ids) > token_count:
            token_ids = token_ids[:token_count - 1]
        else:
            token_ids = verified_ids + _tokenizer.encode(
                " operational guidance",
                add_special_tokens=False,
            )
            token_ids = token_ids[:token_count]

LONG_SHARED_CONTEXT = _make_exact_shared_prefix()

assert len(_tokenizer.encode(
    LONG_SHARED_CONTEXT, add_special_tokens=False
)) == SHARED_PROMPT_TOKEN_COUNT

SCENARIO_B_TOKEN_COUNT = 2000
SCENARIO_B_SHARED_CONTEXT = _make_exact_shared_prefix(SCENARIO_B_TOKEN_COUNT)

assert len(_tokenizer.encode(
    SCENARIO_B_SHARED_CONTEXT, add_special_tokens=False
)) == SCENARIO_B_TOKEN_COUNT

SCENARIO_A_INSTRUCTIONS = [
    "You are a GPU kernel performance specialist. Focus on memory coalescing, occupancy, and register pressure.",
    "You are an inference reliability engineer. Focus on overload behavior, retries, deadlines, and graceful degradation.",
    "You are a transformer serving researcher. Focus on prefill/decode separation, KV state, and batching.",
    "You are a production SRE. Focus on p50/p95/p99 latency, error budgets, and observability.",
    "You are a model compression engineer. Focus on 4-bit quantization, quality trade-offs, and memory footprint.",
    "You are a GPU architecture engineer. Focus on HBM bandwidth, cache locality, and arithmetic intensity.",
    "You are a distributed systems engineer. Focus on queueing, concurrency limits, and backpressure.",
    "You are an inference benchmarking engineer. Focus on repeatability, warmup, baselines, and statistical validity.",
    "You are an API platform engineer. Focus on schema validation, request contracts, and error semantics.",
    "You are an ML systems engineer. Focus on cache reuse, scheduler behavior, and throughput under load.",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    xs = sorted(float(x) for x in values)
    # Linear interpolation, equivalent to common inclusive percentile calculation.
    pos = (len(xs) - 1) * p
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return round(xs[lo], 6)
    return round(xs[lo] + (xs[hi] - xs[lo]) * (pos - lo), 6)


def summarize(values: list[float]) -> dict[str, float | None]:
    return {
        "count": len(values),
        "min": round(min(values), 6) if values else None,
        "mean": round(statistics.mean(values), 6) if values else None,
        "p50": percentile(values, .50),
        "p95": percentile(values, .95),
        "p99": percentile(values, .99),
        "max": round(max(values), 6) if values else None,
    }


def parse_prometheus(text: str) -> dict[str, list[tuple[dict[str, str], float]]]:
    """Parse Prometheus samples into name -> [(labels, value)]."""
    parsed: dict[str, list[tuple[dict[str, str], float]]] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            left, raw_value = line.rsplit(None, 1)
            value = float(raw_value)
        except (ValueError, TypeError):
            continue
        if "{" in left and left.endswith("}"):
            name, label_blob = left.split("{", 1)
            label_blob = label_blob[:-1]
            labels: dict[str, str] = {}
            # SGLang label values are quoted; handle basic escaped quotes.
            for part in label_blob.split(","):
                if "=" in part:
                    key, val = part.split("=", 1)
                    labels[key.strip()] = val.strip().strip('"')
        else:
            name, labels = left, {}
        parsed.setdefault(name, []).append((labels, value))
    return parsed


def metric_samples(parsed: dict[str, list[tuple[dict[str, str], float]]], suffix: str) -> list[tuple[dict[str, str], float]]:
    found = []
    for name, samples in parsed.items():
        if name.endswith(suffix):
            found.extend(samples)
    return found


def gauge_value(parsed: dict[str, list[tuple[dict[str, str], float]]], suffixes: tuple[str, ...]) -> float | None:
    for suffix in suffixes:
        samples = metric_samples(parsed, suffix)
        if samples:
            # If the metric has labels, use the sum as a practical global gauge approximation.
            return sum(v for _, v in samples)
    return None


def histogram_snapshot(
    parsed: dict[str, list[tuple[dict[str, str], float]]],
    metric_fragment: str,
) -> dict[str, Any] | None:
    """Aggregate Prometheus histogram series matching a metric fragment."""
    candidates = [
        name[:-7]
        for name in parsed
        if metric_fragment in name and name.endswith("_bucket")
    ]
    if not candidates:
        return None

    buckets: dict[str, float] = {}
    total_count = 0.0
    total_sum = 0.0
    has_count = False
    has_sum = False

    for base_name in candidates:
        for labels, value in parsed.get(base_name + "_bucket", []):
            le = labels.get("le")
            if le is not None:
                buckets[le] = buckets.get(le, 0.0) + value

        count_samples = parsed.get(base_name + "_count", [])
        sum_samples = parsed.get(base_name + "_sum", [])

        if count_samples:
            total_count += sum(value for _, value in count_samples)
            has_count = True

        if sum_samples:
            total_sum += sum(value for _, value in sum_samples)
            has_sum = True

    return {
        "name": ", ".join(candidates),
        "buckets": buckets,
        "count": total_count if has_count else None,
        "sum": total_sum if has_sum else None,
    }


def histogram_delta(before: dict[str, Any] | None, after: dict[str, Any] | None) -> dict[str, Any] | None:
    if not before or not after:
        return None
    before_b, after_b = before.get("buckets", {}), after.get("buckets", {})
    delta_b = {key: max(0.0, after_b.get(key, 0.0) - before_b.get(key, 0.0)) for key in after_b}
    count = None
    if before.get("count") is not None and after.get("count") is not None:
        count = max(0.0, after["count"] - before["count"])
    total = None
    if before.get("sum") is not None and after.get("sum") is not None:
        total = max(0.0, after["sum"] - before["sum"])
    quantiles: dict[str, float | None] = {}
    # Prometheus buckets are cumulative. Interpolate linearly inside the first bucket reaching target.
    finite = []
    for key, value in delta_b.items():
        try:
            bound = float(key)
            if math.isfinite(bound):
                finite.append((bound, value))
        except ValueError:
            pass
    finite.sort(key=lambda x: x[0])
    for label, q in (("p50", .50), ("p95", .95), ("p99", .99)):
        if count is None or count <= 0:
            quantiles[label] = None
            continue
        target = count * q
        prev_bound, prev_count = 0.0, 0.0
        answer = None
        for bound, cumulative in finite:
            if cumulative >= target:
                in_bucket = max(0.0, cumulative - prev_count)
                if in_bucket <= 0:
                    answer = bound
                else:
                    fraction = min(1.0, max(0.0, (target - prev_count) / in_bucket))
                    answer = prev_bound + (bound - prev_bound) * fraction
                break
            prev_bound, prev_count = bound, cumulative
        quantiles[label] = round(answer, 6) if answer is not None else None
    return {
        "name": after.get("name"),
        "count_delta": count,
        "sum_delta_seconds": round(total, 6) if total is not None else None,
        "mean_seconds": round(total / count, 6) if total is not None and count else None,
        "bucket_count_deltas": delta_b,
        "approx_quantiles_seconds": quantiles,
        "note": "Quantiles are estimates from Prometheus histogram buckets; accuracy is limited by bucket boundaries.",
    }


def get_gpu_snapshot() -> dict[str, Any]:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True,
        )
        rows = []
        for line in proc.stdout.strip().splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) >= 5:
                rows.append({
                    "name": parts[0],
                    "memory_total_mib": float(parts[1]),
                    "memory_used_mib": float(parts[2]),
                    "memory_free_mib": float(parts[3]),
                    "gpu_utilization_percent": float(parts[4]),
                })
        return {"available": True, "gpus": rows}
    except Exception as exc:
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}


async def get_gateway_metrics(client: httpx.AsyncClient, gateway: str) -> dict[str, Any]:
    try:
        response = await client.get(f"{gateway}/metrics", timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


async def get_engine_metrics(client: httpx.AsyncClient, gateway: str, engine_url: str | None) -> dict[str, Any]:
    # Prefer the engine's own Prometheus endpoint; fallback to gateway /engine-metrics.
    urls = [f"{engine_url.rstrip('/')}/metrics"] if engine_url else []
    urls.append(f"{gateway}/engine-metrics")
    for url in urls:
        try:
            response = await client.get(url, timeout=10)
            response.raise_for_status()
            body = response.text
            if url.endswith("/engine-metrics"):
                try:
                    body = response.json().get("prometheus_text", "")
                except Exception:
                    pass
            parsed = parse_prometheus(body)
            return {
                "available": True, "url": url, "raw_text": body,
                "parsed": parsed,
                "cache_hit_rate": gauge_value(parsed, ("sglang:cache_hit_rate", "sglang:prefix_cache_hit_rate")),
                "token_usage": gauge_value(parsed, ("sglang:token_usage",)),
                "ttft_histogram": histogram_snapshot(parsed, "time_to_first_token_seconds"),
                "itl_histogram": histogram_snapshot(parsed, "inter_token_latency_seconds"),
            }
        except Exception:
            continue
    return {"available": False, "error": "Could not retrieve SGLang metrics"}


def metric_public_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    if not snapshot.get("available"):
        return snapshot
    return {k: v for k, v in snapshot.items() if k not in ("raw_text", "parsed")}


def build_prompt_a(i: int, target_tokens: int = 300) -> str:
    """Create deterministic, domain-diverse prompts with exactly target_tokens."""
    domain, question = SCENARIO_A_DOMAINS[i % len(SCENARIO_A_DOMAINS)]
    case_id = f"CASE-{10000 + i * 137}"
    seed = 7919 + i * 104729

    prefix = (
        f"Experiment A request {i:03d}. "
        f"Domain: {domain}. "
        f"Case identifier: {case_id}. "
        f"Scenario seed: {seed}. "
        "Analyze this case independently. "
        "Do not assume information from another request. "
        f"Task: {question} "
        f"Explain the reasoning, practical trade-offs, limitations, "
        f"and two actionable recommendations for {case_id}."
    )

    token_ids = _tokenizer.encode(prefix, add_special_tokens=False)

    if len(token_ids) > target_tokens:
        raise ValueError(
            f"Scenario A prompt {i} exceeds {target_tokens} tokens "
            f"before padding: {len(token_ids)}. Shorten the prompt; do not truncate it."
        )

    filler = (
        f" Additional analysis for {domain}, experiment {i:03d}, "
        f"case {case_id}: discuss assumptions, evidence, alternatives, "
        "practical constraints, and measurable outcomes."
    )

    while len(token_ids) < target_tokens:
        remaining = target_tokens - len(token_ids)
        filler_ids = _tokenizer.encode(filler, add_special_tokens=False)
        token_ids.extend(filler_ids[:remaining])

    prompt = _tokenizer.decode(
        token_ids[:target_tokens],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    verified_count = len(_tokenizer.encode(prompt, add_special_tokens=False))

    if verified_count != target_tokens:
        raise ValueError(
            f"Scenario A prompt {i} round-trip produced {verified_count} tokens; "
            f"expected {target_tokens}."
        )

    if question not in prompt:
        raise ValueError(f"Scenario A prompt {i} lost its complete domain question.")

    return prompt

def build_prompt_b(i: int) -> str:
    """Build a request with the shared 2,000-token prefix and a domain-specific question."""
    domain = DOMAINS[i % len(DOMAINS)]
    question_index = i // len(DOMAINS)
    question = SCENARIO_B_QUESTIONS[domain][question_index]

    suffix = (
        f"Domain: {domain}. "
        f"Request number: {i:03d}. "
        f"Question: {question} "
        "Give a concise, technically grounded response. "
        "Explain relevant trade-offs, measurable evidence, failure modes, "
        "and practical operational recommendations. "
        "Do not return JSON; use plain text."
    )

    return f"{SCENARIO_B_SHARED_CONTEXT}\n\n{suffix}"


def build_prompt_c(i: int) -> str:
    domain = DOMAINS[i % len(DOMAINS)]
    return (
        "Act as a production AI inference engineer. "
        f"Analyze the operational risks and optimization opportunities for {domain}. "
        "Provide a concise, evidence-oriented assessment and identify one practical mitigation. "
        f"Return JSON with request_id exactly 'req-{i:03d}', "
        f"domain exactly '{domain}', a summary under 35 words, "
        "an integer score from 1 to 5, and exactly two short risk strings. "
        "Return no Markdown, extra keys, or surrounding text."
    )


async def run_one(
    client: httpx.AsyncClient, gateway: str, scenario: str, index: int,
    prompt: str, max_tokens: int, timeout: float
) -> dict[str, Any]:
    endpoint = "/generate-json" if scenario == "C" else "/generate"
    payload: dict[str, Any] = {
        "request_id": f"{scenario}-{index:03d}",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "priority": "high" if index % 10 == 0 else "standard",
    }
    if scenario == "C":
        payload["schema"] = DEFAULT_SCHEMA
    start = time.perf_counter()
    try:
        response = await client.post(f"{gateway}{endpoint}", json=payload, timeout=timeout)
        elapsed = time.perf_counter() - start
        body: Any
        try:
            body = response.json()
        except Exception:
            body = {"raw_body": response.text[:3000]}
        result: dict[str, Any] = {
            "scenario": scenario, "index": index, "request_id": payload["request_id"],
            "http_status": response.status_code, "elapsed_seconds": round(elapsed, 6),
            "success": response.status_code == 200, "error": None,
        }
        if response.status_code == 200 and isinstance(body, dict):
            meta = body.get("meta_info") or {}
            gwmeta = body.get("gateway") or {}
            result.update({
                "prompt_tokens": meta.get("prompt_tokens"),
                "completion_tokens": meta.get("completion_tokens"),
                "cached_tokens": meta.get("cached_tokens"),
                "finish_reason": meta.get("finish_reason"),
                "gateway_elapsed_seconds": gwmeta.get("elapsed_seconds"),
                "kv_occupancy_percent_at_admission": gwmeta.get("kv_occupancy_percent_at_admission"),
            })
            if scenario == "C":
                parsed = body.get("parsed_json")
                validation = body.get("schema_validation") or {}
                result["json_valid"] = bool(validation.get("valid", False))
                result["json_validation_errors"] = validation.get("errors", [])
                result["parsed_json"] = parsed
            text = body.get("text", "")
            result["output_chars"] = len(text) if isinstance(text, str) else None
        else:
            result["error"] = body.get("detail", body) if isinstance(body, dict) else body
            if scenario == "C":
                result["json_valid"] = False
        LOGGER.info(
            "REQUEST scenario=%s id=%s status=%s success=%s latency=%.3fs input_tokens=%s output_tokens=%s cached_tokens=%s",
            scenario, payload["request_id"], result["http_status"],
            result["success"], elapsed,
            result.get("prompt_tokens"), result.get("completion_tokens"),
            result.get("cached_tokens"),
        )
        if not result["success"]:
            LOGGER.warning(
                "REQUEST ERROR scenario=%s id=%s status=%s error=%s",
                scenario, payload["request_id"], result["http_status"],
                str(result.get("error"))[:500],
            )
        return result
    except Exception as exc:
        LOGGER.exception("REQUEST FAILED scenario=%s id=%s", scenario, payload["request_id"])
        return {
            "scenario": scenario, "index": index, "request_id": payload["request_id"],
            "http_status": 0, "elapsed_seconds": round(time.perf_counter() - start, 6),
            "success": False, "error": f"{type(exc).__name__}: {exc}",
            "json_valid": False if scenario == "C" else None,
        }


def scenario_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [r["elapsed_seconds"] for r in rows if isinstance(r.get("elapsed_seconds"), (int, float))]
    ttft_values = [r["ttft_seconds"] for r in rows if isinstance(r.get("ttft_seconds"), (int, float))]
    itl_values = [r["itl_seconds"] for r in rows if isinstance(r.get("itl_seconds"), (int, float))]
    prompt_tokens = [r["prompt_tokens"] for r in rows if isinstance(r.get("prompt_tokens"), (int, float))]
    cached_tokens = [r["cached_tokens"] for r in rows if isinstance(r.get("cached_tokens"), (int, float))]
    completion_tokens = [r["completion_tokens"] for r in rows if isinstance(r.get("completion_tokens"), (int, float))]
    statuses: dict[str, int] = {}
    for row in rows:
        key = str(row.get("http_status", 0))
        statuses[key] = statuses.get(key, 0) + 1
    return {
        "request_count": len(rows),
        "status_counts": statuses,
        "success_count": sum(bool(r.get("success")) for r in rows),
        "error_count": sum(not bool(r.get("success")) for r in rows),
        "json_valid_count": sum(r.get("json_valid") is True for r in rows) if rows and rows[0].get("scenario") == "C" else None,
        "json_invalid_count": sum(r.get("json_valid") is False for r in rows) if rows and rows[0].get("scenario") == "C" else None,
        "request_latency_seconds": summarize(latencies),
        "ttft_seconds": summarize(ttft_values) if ttft_values else None,
        "itl_seconds": summarize(itl_values) if itl_values else None,
        "mean_prompt_tokens": round(statistics.mean(prompt_tokens), 3) if prompt_tokens else None,
        "mean_cached_tokens": round(statistics.mean(cached_tokens), 3) if cached_tokens else None,
        "mean_completion_tokens": round(statistics.mean(completion_tokens), 3) if completion_tokens else None,
    }


async def run_scenario(
    client: httpx.AsyncClient,
    args: argparse.Namespace,
    scenario: str,
    count: int,
) -> dict[str, Any]:
    LOGGER.info("SCENARIO START scenario=%s requests=%s max_inflight=%s",
                scenario, count, args.max_inflight)
    print(f"\n=== Scenario {scenario}: {count} requests, full fan-out ===", flush=True)
    before_gpu = get_gpu_snapshot()
    before_gateway = await get_gateway_metrics(client, args.gateway)
    before_engine = await get_engine_metrics(client, args.gateway, args.engine_url)
    start = time.perf_counter()

    if scenario == "A":
        prompts = [build_prompt_a(i) for i in range(count)]
    elif scenario == "B":
        prompts = [build_prompt_b(i) for i in range(count)]
    else:
        prompts = [build_prompt_c(i) for i in range(count)]

    # Full fan-out matches the assignment's requested concurrent request count.
    # --max-inflight can be used to reduce load if the GPU/runtime cannot sustain it.
    semaphore = asyncio.Semaphore(args.max_inflight)
    async def guarded(i: int, prompt: str):
        async with semaphore:
            return await run_one(client, args.gateway, scenario, i, prompt, args.max_tokens, args.request_timeout)
    rows = await asyncio.gather(*(guarded(i, p) for i, p in enumerate(prompts)))
    wall_seconds = time.perf_counter() - start

    after_engine = await get_engine_metrics(client, args.gateway, args.engine_url)
    after_gateway = await get_gateway_metrics(client, args.gateway)
    after_gpu = get_gpu_snapshot()

    # Engine histogram deltas provide measured TTFT/ITL distributions for the whole scenario.
    ttft_delta = histogram_delta(before_engine.get("ttft_histogram"), after_engine.get("ttft_histogram"))
    itl_delta = histogram_delta(before_engine.get("itl_histogram"), after_engine.get("itl_histogram"))
    # Per-request TTFT/ITL are not guessed from end-to-end latency. They are reported from engine histogram deltas.
    summary = scenario_summary(rows)
    successful_rows = [r for r in rows if r.get("success")]

    total_input_tokens = sum(
        r["prompt_tokens"] for r in successful_rows
        if isinstance(r.get("prompt_tokens"), (int, float))
    )
    total_output_tokens = sum(
        r["completion_tokens"] for r in successful_rows
        if isinstance(r.get("completion_tokens"), (int, float))
    )

    summary["total_input_tokens"] = total_input_tokens
    summary["total_output_tokens"] = total_output_tokens
    summary["input_tokens_per_second"] = (
        round(total_input_tokens / wall_seconds, 3) if wall_seconds else None
    )
    summary["output_tokens_per_second"] = (
        round(total_output_tokens / wall_seconds, 3) if wall_seconds else None
    )
    summary["throughput_note"] = (
        "Successful request token counts divided by scenario wall-clock duration; "
        "workload-level throughput, not instantaneous GPU throughput."
    )
    summary["wall_seconds"] = round(wall_seconds, 6)
    summary["request_rate_per_second"] = round(len(rows) / wall_seconds, 6) if wall_seconds else None
    summary["successful_throughput_per_second"] = (
        round(summary["success_count"] / wall_seconds, 6)
        if wall_seconds
        else None
    )
    summary["ttft_histogram_delta"] = ttft_delta
    summary["itl_histogram_delta"] = itl_delta
    summary["cache_metrics"] = {
        "cache_hit_rate_before": before_engine.get("cache_hit_rate"),
        "cache_hit_rate_after": after_engine.get("cache_hit_rate"),
        "token_usage_before": before_engine.get("token_usage"),
        "token_usage_after": after_engine.get("token_usage"),
        "mean_cached_tokens_from_request_metadata": summary.get("mean_cached_tokens"),
        "note": "Cache hit rate is a gauge snapshot, not a hit/miss counter. Use the cached-token metadata and before/after gauge as complementary evidence.",
    }
    summary["gpu_snapshot_before"] = before_gpu
    summary["gpu_snapshot_after"] = after_gpu
    summary["gateway_metrics_before"] = before_gateway
    summary["gateway_metrics_after"] = after_gateway
    summary["engine_metrics_before"] = metric_public_snapshot(before_engine)
    summary["engine_metrics_after"] = metric_public_snapshot(after_engine)
    summary["methodology"] = {
        "requested_concurrency": count,
        "max_inflight_limit": args.max_inflight,
        "full_fan_out": True,
        "target_prompt_tokens": 300 if scenario == "A" else None,
        "shared_context_token_count": 2000 if scenario == "B" else None,
        "prompt_characters_min": min(map(len, prompts)) if prompts else 0,
        "prompt_characters_max": max(map(len, prompts)) if prompts else 0,
        "prompt_token_count_note": "Actual token count is captured from SGLang metadata when provided; prompt character length is not a token-count substitute.",
    }
    result = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "scenario": scenario,
        "summary": summary,
        "requests": rows,
    }
    output_path = Path(args.output_dir) / f"scenario_{scenario.lower()}_results.json"
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Saved {output_path}", flush=True)
    print(json.dumps({
        "scenario": scenario, "success": summary["success_count"],
        "errors": summary["error_count"], "statuses": summary["status_counts"],
        "latency": summary["request_latency_seconds"],
        "ttft_histogram_delta": summary["ttft_histogram_delta"],
        "itl_histogram_delta": summary["itl_histogram_delta"],
        "json_valid": summary["json_valid_count"],
        "cache": summary["cache_metrics"],
    }, indent=2), flush=True)
    LOGGER.info(
        "SCENARIO COMPLETE scenario=%s success=%s errors=%s wall_seconds=%.3f input_tokens_per_sec=%s output_tokens_per_sec=%s",
        scenario, summary["success_count"], summary["error_count"],
        wall_seconds, summary["input_tokens_per_second"],
        summary["output_tokens_per_second"],
    )
    return result


async def run_prefix_warm_cold(client: httpx.AsyncClient, args: argparse.Namespace) -> dict[str, Any]:
    """Small paired probe: first request is cold-ish, repeated shared-prefix request is warm-ish."""
    print("\n=== Prefix warm/cold probe (10 paired requests) ===", flush=True)
    base = LONG_SHARED_CONTEXT + "\nDomain: KV-cache optimization. Explain prefix reuse and its limits."
    rows = []
    for i in range(10):
        prompts = [
            base + f"\nCold-probe unique suffix {i}. Explain the trade-offs.",
            base + f"\nWarm-probe unique suffix {i}. Explain the trade-offs.",
        ]
        pair = []
        for label, prompt in zip(("cold_order", "warm_order"), prompts):
            row = await run_one(client, args.gateway, "B", i, prompt, min(args.max_tokens, 128), args.request_timeout)
            row["probe_label"] = label
            pair.append(row)
        rows.extend(pair)
    cold = [r["elapsed_seconds"] for r in rows if r["probe_label"] == "cold_order" and r["success"]]
    warm = [r["elapsed_seconds"] for r in rows if r["probe_label"] == "warm_order" and r["success"]]
    cold_mean = statistics.mean(cold) if cold else None
    warm_mean = statistics.mean(warm) if warm else None
    reduction = ((cold_mean - warm_mean) / cold_mean * 100.0) if cold_mean and warm_mean is not None else None
    result = {
        "created_at_utc": utc_now(),
        "probe": "paired shared-prefix request latency comparison",
        "request_count": len(rows),
        "cold_order_latency_seconds": summarize(cold),
        "warm_order_latency_seconds": summarize(warm),
        "mean_latency_reduction_percent": round(reduction, 3) if reduction is not None else None,
        "requests": rows,
        "caveat": "This is a practical paired probe, not a perfectly isolated cache experiment: cache state, scheduling, and GPU load can affect latency. Interpret alongside SGLang cache metrics and cached-token metadata.",
    }
    path = Path(args.output_dir) / "prefix_warm_cold_probe.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Saved {path}")
    print(json.dumps({k: v for k, v in result.items() if k != "requests"}, indent=2))
    return result


async def main_async(args: argparse.Namespace) -> None:
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    timeout = httpx.Timeout(args.request_timeout, connect=15.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        # Health checks first; never start a costly benchmark against an unhealthy service.
        for url in (f"{args.gateway}/health", f"{args.engine_url.rstrip('/')}/health" if args.engine_url else None):
            if not url:
                continue
            try:
                response = await client.get(url, timeout=10)
                print(f"Health {url}: HTTP {response.status_code}")
                LOGGER.info("HEALTH CHECK url=%s status=%s", url, response.status_code)
                if response.status_code >= 400:
                    raise SystemExit(f"Health check failed: {url}")
            except Exception as exc:
                raise SystemExit(f"Health check failed for {url}: {exc}")
        selected = [args.scenario] if args.scenario in ("A", "B", "C") else ["A", "B", "C"]
        counts = {"A": args.a_requests, "B": args.b_requests, "C": args.c_requests}
        results = []
        for scenario in selected:
            results.append(await run_scenario(client, args, scenario, counts[scenario]))
        if args.run_warm_cold and (args.scenario == "all"):
            results.append(await run_prefix_warm_cold(client, args))
        index = {
            "created_at_utc": utc_now(),
            "gateway": args.gateway,
            "engine_url": args.engine_url,
            "scenario_files": [
                f"scenario_{s.lower()}_results.json" for s in selected
            ],
            "warm_cold_probe_file": "prefix_warm_cold_probe.json" if args.run_warm_cold and args.scenario == "all" else None,
            "settings": vars(args),
            "notes": [
                "Request latency is measured per request by the client.",
                "TTFT and ITL percentiles are estimated from SGLang Prometheus histogram deltas for each scenario.",
                "If the installed SGLang version lacks the relevant histograms, those fields are null rather than fabricated.",
                "The gateway validates basic JSON Schema constraints for scenario C and records validation per request.",
            ],
        }
        index_path = Path(args.output_dir) / "benchmark_index.json"
        index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False))
        print(f"\nSaved index: {index_path}")
        print("\nAll selected scenarios finished.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gateway", default=os.getenv("GATEWAY_URL", "http://127.0.0.1:8081"))
    parser.add_argument("--engine-url", default=os.getenv("SGLANG_URL", "http://127.0.0.1:30000"))
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--scenario", choices=["all", "A", "B", "C"], default="all")
    parser.add_argument("--a-requests", type=int, default=50)
    parser.add_argument("--b-requests", type=int, default=100)
    parser.add_argument("--c-requests", type=int, default=50)
    parser.add_argument("--max-inflight", type=int, default=128,
                        help="Client-side in-flight cap; default allows full assignment concurrency.")
    parser.add_argument("--request-timeout", type=float, default=360.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--run-warm-cold", action="store_true",
                        help="Run an additional paired prefix-cache latency probe after A/B/C.")
    return parser.parse_args()


if __name__ == "__main__":
    log_path = setup_logging()
    LOGGER.info("BENCHMARK RUN STARTED")
    LOGGER.info("Log file: %s", log_path.resolve())

    try:
        args = parse_args()
        LOGGER.info("Configuration: %s", vars(args))
        asyncio.run(main_async(args))
        LOGGER.info("BENCHMARK RUN COMPLETED SUCCESSFULLY")
    except SystemExit as exc:
        if exc.code == 0:
            LOGGER.info("CLI EXITED SUCCESSFULLY")
        else:
            LOGGER.error("CLI EXITED WITH CODE %s", exc.code)
        raise
    except Exception:
        LOGGER.exception("BENCHMARK RUN FAILED")
        raise
    finally:
        for handler in LOGGER.handlers:
            handler.flush()
            handler.close()
