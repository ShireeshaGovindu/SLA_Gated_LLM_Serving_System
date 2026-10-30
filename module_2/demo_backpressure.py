"""
Module 2: Live Backpressure Demonstration

Runs concurrent inference requests through the existing gateway.
Continuously records KV-cache, GPU, gateway and request metrics.

The gateway handles admission and HTTP 503 shedding.
The demo only stops requests for genuine engine/GPU safety conditions.

Outputs:
  results/module_2/backpressure_live_report.json
  results/module_2/backpressure_metrics.csv
  results/module_2/backpressure.log
"""

import argparse
import asyncio
import csv
import json
import logging
import subprocess
import sys
import time

from datetime import datetime, timezone
from pathlib import Path

import httpx

from module_2.config.backpressure_prompts import (
    OUTPUT_TOKENS,
    INCIDENT_SCENARIOS,
    build_prompt,
    load_tokenizer,
    verify_prompts,
)


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_GATEWAY = "http://127.0.0.1:8081"
DEFAULT_ENGINE = "http://127.0.0.1:30000"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def configure_logging(output_dir):
    log_path = output_dir / "backpressure.log"

    logger = logging.getLogger("backpressure")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    file_handler = logging.FileHandler(
        log_path,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger, log_path


def read_engine_metrics(engine_url):
    """Read SGLang KV-cache and scheduler metrics."""

    try:
        response = httpx.get(
            f"{engine_url}/metrics",
            timeout=5,
        )
        response.raise_for_status()

        lines = response.text.splitlines()

        def metric(name):
            for line in lines:
                if (
                    line.startswith(name + "{")
                    or line.startswith(name + " ")
                ):
                    try:
                        return float(line.split()[-1])
                    except (ValueError, IndexError):
                        return None

            return None

        capacity = metric("sglang:max_total_num_tokens")
        available = metric("sglang:kv_available_tokens")

        occupancy = None

        if capacity and available is not None:
            occupancy = round(
                (capacity - available) / capacity * 100,
                3,
            )

        return {
            "timestamp": utc_now(),
            "max_total_tokens": capacity,
            "kv_available_tokens": available,
            "kv_evictable_tokens": metric(
                "sglang:kv_evictable_tokens"
            ),
            "num_used_tokens": metric(
                "sglang:num_used_tokens"
            ),
            "num_running_reqs": metric(
                "sglang:num_running_reqs"
            ),
            "kv_occupancy_percent": occupancy,
            "metrics_available": (
                capacity is not None
                and available is not None
            ),
        }

    except Exception as exc:
        return {
            "timestamp": utc_now(),
            "metrics_available": False,
            "error": str(exc),
        }


def read_gpu():
    """Read GPU memory and utilization."""

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )

        used, free, utilization = map(
            int,
            result.stdout.strip().splitlines()[0].split(","),
        )

        return {
            "timestamp": utc_now(),
            "memory_used_mib": used,
            "memory_free_mib": free,
            "gpu_utilization_percent": utilization,
        }

    except Exception as exc:
        return {
            "timestamp": utc_now(),
            "error": str(exc),
        }


def read_gateway_metrics(gateway):
    """Read gateway admission counters."""

    try:
        response = httpx.get(
            f"{gateway}/metrics",
            timeout=5,
        )
        response.raise_for_status()

        data = response.json()

        keys = [
            "kv_occupancy_percent",
            "kv_threshold_percent",
            "kv_metrics_available",
            "requests_received",
            "requests_accepted",
            "requests_shed",
            "engine_errors",
            "kv_metrics_errors",
            "high_priority_allowed",
        ]

        return {
            key: data.get(key)
            for key in keys
        }

    except Exception as exc:
        return {"error": str(exc)}


async def send_request(
    client,
    gateway,
    request_id,
    tokenizer,
    priority,
    logger,
):
    prompt = build_prompt(request_id, tokenizer)
    started = time.perf_counter()

    prompt_topic = INCIDENT_SCENARIOS[
        request_id % len(INCIDENT_SCENARIOS)
    ][0]

    try:
        response = await client.post(
            f"{gateway}/generate",
            json={
                "prompt": prompt,
                "max_tokens": OUTPUT_TOKENS,
                "temperature": 0,
                "priority": priority,
            },
        )

        result = {
            "request_id": request_id,
            "prompt_topic": prompt_topic,
            "priority": priority,
            "status_code": response.status_code,
            "latency_seconds": round(
                time.perf_counter() - started,
                3,
            ),
            "response_excerpt": response.text[:250],
        }

        logger.info(
            "request=%d topic=%s priority=%s status=%s latency=%.3fs",
            request_id,
            prompt_topic,
            priority,
            response.status_code,
            result["latency_seconds"],
        )

        return result

    except asyncio.CancelledError:
        logger.warning(
            "request=%d cancelled by safety monitor",
            request_id,
        )
        raise

    except Exception as exc:
        logger.error(
            "request=%d topic=%s error=%s",
            request_id,
            prompt_topic,
            exc,
        )

        return {
            "request_id": request_id,
            "prompt_topic": prompt_topic,
            "priority": priority,
            "error": str(exc),
            "latency_seconds": round(
                time.perf_counter() - started,
                3,
            ),
        }


def metric_csv_row(sample):
    engine = sample.get("engine", {})
    gpu = sample.get("gpu", {})
    gateway = sample.get("gateway", {})

    return {
        "timestamp": sample.get("timestamp"),
        "kv_occupancy_percent": engine.get(
            "kv_occupancy_percent"
        ),
        "max_total_tokens": engine.get("max_total_tokens"),
        "kv_available_tokens": engine.get(
            "kv_available_tokens"
        ),
        "kv_evictable_tokens": engine.get(
            "kv_evictable_tokens"
        ),
        "num_used_tokens": engine.get("num_used_tokens"),
        "num_running_reqs": engine.get("num_running_reqs"),
        "gpu_memory_used_mib": gpu.get("memory_used_mib"),
        "gpu_memory_free_mib": gpu.get("memory_free_mib"),
        "gpu_utilization_percent": gpu.get(
            "gpu_utilization_percent"
        ),
        "requests_received": gateway.get("requests_received"),
        "requests_accepted": gateway.get("requests_accepted"),
        "requests_shed": gateway.get("requests_shed"),
        "engine_health": sample.get("engine_health"),
    }


async def run_burst(
    client,
    gateway,
    engine,
    request_count,
    max_tokens,
    threshold,
    min_free_mib,
    priority,
    tokenizer,
    sample_interval,
    logger,
):
    logger.info(
        "Starting simultaneous burst: requests=%d priority=%s",
        request_count,
        priority,
    )

    stage = {
        "started_at": utc_now(),
        "concurrency": request_count,
        "request_count": request_count,
        "priority": priority,
        "threshold_percent": threshold,
        "samples": [],
        "responses": [],
        "safety_stop": None,
        "threshold_observed": False,
    }

    # Initial measurements
    stage["before"] = {
        "engine": await asyncio.to_thread(
            read_engine_metrics,
            engine,
        ),
        "gpu": await asyncio.to_thread(read_gpu),
        "gateway": await asyncio.to_thread(
            read_gateway_metrics,
            gateway,
        ),
    }

    initial_gpu = stage["before"]["gpu"]
    initial_free = initial_gpu.get("memory_free_mib")

    if initial_free is None or initial_free < min_free_mib:
        stage["safety_stop"] = (
            f"Insufficient GPU free memory: {initial_free} MiB"
        )
        logger.error("SAFETY STOP: %s", stage["safety_stop"])
        return stage

    if not stage["before"]["engine"].get("metrics_available"):
        stage["safety_stop"] = (
            "SGLang KV metrics unavailable before burst"
        )
        logger.error("SAFETY STOP: %s", stage["safety_stop"])
        return stage

    # Launch all requests concurrently.
    request_tasks = [
        asyncio.create_task(
            send_request(
                client,
                gateway,
                i,
                tokenizer,
                priority,
                logger,
            )
        )
        for i in range(request_count)
    ]

    async def monitor():
        consecutive_health_failures = 0
        consecutive_gpu_errors = 0

        while True:
            engine_metrics, gpu_metrics, gateway_metrics = (
                await asyncio.gather(
                    asyncio.to_thread(
                        read_engine_metrics,
                        engine,
                    ),
                    asyncio.to_thread(read_gpu),
                    asyncio.to_thread(
                        read_gateway_metrics,
                        gateway,
                    ),
                )
            )

            # Health check
            health_started = time.perf_counter()
            health_error = None

            try:
                health = await client.get(
                    f"{engine}/health",
                    timeout=15.0,
                )

                health_ok = health.status_code == 200

                if not health_ok:
                    health_error = f"HTTP {health.status_code}"

            except Exception as exc:
                health_ok = False
                health_error = f"{type(exc).__name__}: {exc}"

            health_latency_ms = round(
                (time.perf_counter() - health_started) * 1000,
                2,
            )

            if health_ok:
                consecutive_health_failures = 0
            else:
                consecutive_health_failures += 1

            if gpu_metrics.get("memory_free_mib") is None:
                consecutive_gpu_errors += 1
            else:
                consecutive_gpu_errors = 0

            sample = {
                "timestamp": utc_now(),
                "engine": engine_metrics,
                "gpu": gpu_metrics,
                "gateway": gateway_metrics,
                "engine_health": health_ok,
                "engine_health_latency_ms": health_latency_ms,
                "engine_health_error": health_error,
            }

            stage["samples"].append(sample)

            occupancy = engine_metrics.get(
                "kv_occupancy_percent"
            )

            # IMPORTANT:
            # Do not cancel requests when the KV threshold is reached.
            # The gateway must perform HTTP 503 admission shedding.
            if (
                occupancy is not None
                and occupancy >= threshold
                and not stage["threshold_observed"]
            ):
                stage["threshold_observed"] = True

                logger.warning(
                    "KV threshold observed: %.2f%% (threshold %.2f%%). "
                    "Gateway admission shedding is now being observed.",
                    occupancy,
                    threshold,
                )

            # Actual GPU safety stop
            free = gpu_metrics.get("memory_free_mib")

            if free is not None and free < min_free_mib:
                stage["safety_stop"] = (
                    f"GPU free memory below safety limit: {free} MiB"
                )

            # Require three consecutive failed health checks.
            elif consecutive_health_failures >= 3:
                stage["safety_stop"] = (
                    "SGLang health check failed 3 consecutive times"
                )

            elif consecutive_gpu_errors >= 3:
                stage["safety_stop"] = (
                    "GPU monitoring failed 3 consecutive times"
                )

            if stage["safety_stop"]:
                logger.error(
                    "SAFETY STOP: %s",
                    stage["safety_stop"],
                )

                for task in request_tasks:
                    if not task.done():
                        task.cancel()

                return

            if all(task.done() for task in request_tasks):
                return

            await asyncio.sleep(sample_interval)

    monitor_task = asyncio.create_task(monitor())

    await asyncio.gather(
        monitor_task,
        *request_tasks,
        return_exceptions=True,
    )

    # Collect every request result.
    for request_id, task in enumerate(request_tasks):
        if task.cancelled():
            stage["responses"].append({
                "request_id": request_id,
                "error": "Cancelled by safety monitor",
            })
        else:
            try:
                result = task.result()

                if isinstance(result, dict):
                    stage["responses"].append(result)

            except Exception as exc:
                stage["responses"].append({
                    "request_id": request_id,
                    "error": str(exc),
                })

    # Final measurements
    stage["after"] = {
        "engine": await asyncio.to_thread(
            read_engine_metrics,
            engine,
        ),
        "gpu": await asyncio.to_thread(read_gpu),
        "gateway": await asyncio.to_thread(
            read_gateway_metrics,
            gateway,
        ),
    }

    stage["finished_at"] = utc_now()

    # HTTP outcome counts
    status_counts = {}

    for result in stage["responses"]:
        status = str(
            result.get(
                "status_code",
                "cancelled_or_error",
            )
        )

        status_counts[status] = (
            status_counts.get(status, 0) + 1
        )

    stage["status_counts"] = status_counts

    # Peak resource measurements
    occupancies = [
        sample["engine"].get("kv_occupancy_percent")
        for sample in stage["samples"]
        if sample["engine"].get("kv_occupancy_percent") is not None
    ]

    gpu_used = [
        sample["gpu"].get("memory_used_mib")
        for sample in stage["samples"]
        if sample["gpu"].get("memory_used_mib") is not None
    ]

    gpu_free = [
        sample["gpu"].get("memory_free_mib")
        for sample in stage["samples"]
        if sample["gpu"].get("memory_free_mib") is not None
    ]

    stage["peak_kv_occupancy_percent"] = (
        max(occupancies) if occupancies else None
    )

    stage["peak_gpu_memory_used_mib"] = (
        max(gpu_used) if gpu_used else None
    )

    stage["minimum_gpu_free_mib"] = (
        min(gpu_free) if gpu_free else None
    )

    stage["threshold_reached"] = (
        stage["peak_kv_occupancy_percent"] is not None
        and stage["peak_kv_occupancy_percent"] >= threshold
    )

    logger.info("HTTP outcome counts: %s", status_counts)

    logger.info(
        "Peak KV occupancy: %s%%",
        stage["peak_kv_occupancy_percent"],
    )

    logger.info(
        "Peak GPU memory used: %s MiB",
        stage["peak_gpu_memory_used_mib"],
    )

    logger.info(
        "Minimum GPU free memory: %s MiB",
        stage["minimum_gpu_free_mib"],
    )

    logger.info(
        "Gateway final counters: %s",
        stage["after"]["gateway"],
    )

    return stage


async def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger, log_path = configure_logging(output_dir)

    report = {
        "experiment": "Module 2 live backpressure demonstration",
        "started_at": utc_now(),
        "gateway": args.gateway,
        "engine": args.engine,
        "threshold_percent": args.threshold,
        "minimum_free_vram_mib": args.min_free_mib,
        "request_count": args.requests,
        "max_tokens": args.max_tokens,
        "priority": args.priority,
        "sample_interval_seconds": args.sample_interval,
        "stages": [],
    }

    timeout = httpx.Timeout(
        args.timeout,
        connect=10.0,
    )

    limits = httpx.Limits(
        max_connections=args.requests + 10,
        max_keepalive_connections=args.requests + 10,
    )

    tokenizer = load_tokenizer()
    verify_prompts(tokenizer, args.requests)

    async with httpx.AsyncClient(
        timeout=timeout,
        limits=limits,
    ) as client:

        # Verify both services
        for name, url in [
            ("gateway", f"{args.gateway}/health"),
            ("engine", f"{args.engine}/health"),
        ]:
            response = await client.get(url)
            response.raise_for_status()

            logger.info(
                "%s reachable: HTTP %s",
                name,
                response.status_code,
            )

        logger.info(
            "Initial GPU: %s",
            await asyncio.to_thread(read_gpu),
        )

        logger.info(
            "Initial engine metrics: %s",
            await asyncio.to_thread(
                read_engine_metrics,
                args.engine,
            ),
        )

        logger.info(
            "Initial gateway metrics: %s",
            await asyncio.to_thread(
                read_gateway_metrics,
                args.gateway,
            ),
        )

        stage = await run_burst(
            client=client,
            gateway=args.gateway,
            engine=args.engine,
            request_count=args.requests,
            max_tokens=args.max_tokens,
            threshold=args.threshold,
            min_free_mib=args.min_free_mib,
            priority=args.priority,
            tokenizer=tokenizer,
            sample_interval=args.sample_interval,
            logger=logger,
        )

        report["stages"].append(stage)

    report["finished_at"] = utc_now()

    report["final_gateway_metrics"] = await asyncio.to_thread(
        read_gateway_metrics,
        args.gateway,
    )

    report["final_engine_metrics"] = await asyncio.to_thread(
        read_engine_metrics,
        args.engine,
    )

    report["final_gpu"] = await asyncio.to_thread(read_gpu)

    json_path = output_dir / "backpressure_live_report.json"
    csv_path = output_dir / "backpressure_metrics.csv"

    fieldnames = [
        "timestamp",
        "kv_occupancy_percent",
        "max_total_tokens",
        "kv_available_tokens",
        "kv_evictable_tokens",
        "num_used_tokens",
        "num_running_reqs",
        "gpu_memory_used_mib",
        "gpu_memory_free_mib",
        "gpu_utilization_percent",
        "requests_received",
        "requests_accepted",
        "requests_shed",
        "engine_health",
    ]

    samples = [
        metric_csv_row(sample)
        for stage in report["stages"]
        for sample in stage.get("samples", [])
    ]

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(samples)

    report["artifacts"] = {
        "json": str(json_path),
        "csv": str(csv_path),
        "log": str(log_path),
    }

    json_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    logger.info("=== EXPERIMENT COMPLETE ===")
    logger.info("JSON report: %s", json_path)
    logger.info("Metrics CSV: %s", csv_path)
    logger.info("Execution log: %s", log_path)

    logger.info(
        "Final gateway metrics: %s",
        report["final_gateway_metrics"],
    )

    logger.info(
        "Final engine metrics: %s",
        report["final_engine_metrics"],
    )

    logger.info("Final GPU: %s", report["final_gpu"])


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--gateway",
        default=DEFAULT_GATEWAY,
    )

    parser.add_argument(
        "--engine",
        default=DEFAULT_ENGINE,
    )

    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "results" / "module_2"),
    )

    parser.add_argument(
        "--requests",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=OUTPUT_TOKENS,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=90.0,
    )

    parser.add_argument(
        "--min-free-mib",
        type=int,
        default=1200,
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
    )

    parser.add_argument(
        "--priority",
        choices=["standard", "high"],
        default="standard",
    )

    parser.add_argument(
        "--sample-interval",
        type=float,
        default=0.5,
    )

    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))