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
import itertools
import json
import logging
import math
import subprocess
import sys
import time

from datetime import datetime, timezone
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from module_2.config.backpressure_prompts import (
    OUTPUT_TOKENS,
    INCIDENT_SCENARIOS,
    build_prompt,
    load_tokenizer,
    verify_prompts,
)

DEFAULT_GATEWAY = "http://127.0.0.1:8081"
DEFAULT_ENGINE = "http://127.0.0.1:30000"

# ---------------------------------------------------------
# 1. INITIALIZATION AND LOGGING
# ---------------------------------------------------------

def utc_now():
    return datetime.now(timezone.utc).isoformat()

class EventFormatter(logging.Formatter):
    """Format: [INFO]  message  (WARNING is shown as WARN)."""

    def format(self, record):
        level = "WARN" if record.levelname == "WARNING" else record.levelname
        return f"[{level}]  {record.getMessage()}"


class EventLog:
    """
    One line per request outcome, in the form

      [INFO]  t=473.4s  kv_usage=0.940  tier2 -> admitted
      [WARN]  t=480.4s  kv_usage=0.940  tier2 -> HTTP 503

    t is the time the request was sent (seconds since the run started) and
    kv_usage is the KV token-usage fraction the gateway saw when it admitted
    or rejected the request. The first line of a run of identical outcomes is
    printed at once; the rest are collapsed into one range line:

      [WARN]  t=486.4s .. t=506.4s  kv_usage=0.940  tier2 -> HTTP 503   (11 more)
    """

    def __init__(self, logger, threshold):
        self.logger = logger
        self.threshold = threshold
        self.run = None

    def _line(self, level, t_text, kv, tier, outcome, suffix=""):
        kv_text = f"{kv:.3f}" if kv is not None else "n/a"
        self.logger.log(
            level,
            "%s  kv_usage=%s  %s -> %s%s",
            t_text,
            kv_text,
            tier,
            outcome,
            suffix,
        )

    def emit(self, t, kv, tier, outcome, level):
        key = (tier, outcome, kv)

        if self.run and self.run["key"] == key:
            run = self.run
            run["count"] += 1
            run["lo_t"] = min(run.get("lo_t", t), t)
            run["hi_t"] = max(run.get("hi_t", t), t)
            return

        self.flush()

        self.run = {
            "key": key,
            "level": level,
            "count": 1,
        }

        suffix = ""
        limit = f"{self.threshold:.0f}%"

        if kv is not None:
            above = kv * 100 >= self.threshold

            if outcome == "admitted":
                if tier == "tier1":
                    suffix = (
                        f" (KV >= {limit}: never shed)"
                        if above
                        else f" (KV < {limit})"
                    )
                else:
                    suffix = f" (KV < {limit})"
            elif outcome == "HTTP 503":
                suffix = f" (KV >= {limit}: shed)"

        self._line(level, f"t={t:.1f}s", kv, tier, outcome, suffix)

    def flush(self):
        run, self.run = self.run, None

        if not run or run["count"] == 1:
            return

        tier, outcome, kv = run["key"]

        if run["count"] == 2:
            self._line(run["level"], f"t={run['lo_t']:.1f}s", kv, tier, outcome)
            return

        # Requests finish out of order, so show the earliest..latest send time.
        self._line(
            run["level"],
            f"t={run['lo_t']:.1f}s .. t={run['hi_t']:.1f}s",
            kv,
            tier,
            outcome,
            f"   ({run['count'] - 1} more)",
        )


RUN_T0 = time.monotonic()
EVENTS = None


def record_event(t_sent, priority, status_code, kv_percent, error=None):
    if EVENTS is None:
        return

    tier = "tier1" if priority == "high" else "tier2"
    kv = kv_percent / 100.0 if kv_percent is not None else None

    if error is not None:
        EVENTS.emit(t_sent, kv, tier, f"error ({error})", logging.ERROR)
    elif status_code == 200:
        EVENTS.emit(t_sent, kv, tier, "admitted", logging.INFO)
    elif status_code == 503:
        EVENTS.emit(t_sent, kv, tier, "HTTP 503", logging.WARNING)
    else:
        EVENTS.emit(t_sent, kv, tier, f"HTTP {status_code}", logging.ERROR)


def configure_logging(output_dir):
    log_path = output_dir / "backpressure.log"

    logger = logging.getLogger("backpressure")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

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

    # Per-request outcome lines: same console and the same log file.
    events = logging.getLogger("backpressure.events")
    events.setLevel(logging.INFO)
    events.handlers.clear()
    events.propagate = False

    event_file = logging.FileHandler(log_path, encoding="utf-8")
    event_file.setFormatter(EventFormatter())
    event_stream = logging.StreamHandler()
    event_stream.setFormatter(EventFormatter())

    events.addHandler(event_file)
    events.addHandler(event_stream)

    return logger, log_path

# ---------------------------------------------------------
# 2. ENGINE METRICS
# ---------------------------------------------------------

def read_engine_metrics(engine_url):
    """Read SGLang KV-cache and scheduler metrics."""

    try:
        response = httpx.get(
            f"{engine_url}/metrics",
            timeout=5.0,
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
                        value = float(line.split()[-1])

                        if not math.isfinite(value):
                            return None

                        return value

                    except (ValueError, IndexError):
                        return None

            return None

        capacity = metric("sglang:max_total_num_tokens")
        available = metric("sglang:kv_available_tokens")
        token_usage = metric("sglang:token_usage")

        pool_occupancy = None

        if (
            capacity is not None
            and capacity > 0
            and available is not None
            and 0 <= available <= capacity
        ):
            pool_occupancy = round(
                (capacity - available) / capacity * 100,
                3,
            )

        token_usage_percent = None

        if (
            token_usage is not None
            and 0.0 <= token_usage <= 1.0
        ):
            token_usage_percent = round(
                token_usage * 100,
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
            "token_usage_percent": token_usage_percent,
            "kv_pool_occupancy_percent": pool_occupancy,
            "metrics_available": (
                capacity is not None
                and capacity > 0
                and token_usage_percent is not None
            ),
        }

    except Exception as exc:
        return {
            "timestamp": utc_now(),
            "metrics_available": False,
            "error": repr(exc),
        }

# ---------------------------------------------------------
# 3. GPU METRICS
# ---------------------------------------------------------

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

        lines = result.stdout.strip().splitlines()

        if not lines:
            raise RuntimeError("nvidia-smi returned empty output")

        # This experiment assumes a single-GPU instance.
        values = [
            value.strip()
            for value in lines[0].split(",")
        ]

        if len(values) != 3:
            raise RuntimeError(
                f"Unexpected nvidia-smi output: {result.stdout}"
            )

        used, free, utilization = map(int, values)

        if min(used, free, utilization) < 0:
            raise ValueError("Invalid negative GPU metric")

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

# ---------------------------------------------------------
# 4. GATEWAY METRICS
# ---------------------------------------------------------

def read_gateway_metrics(gateway):
    """Read gateway admission metrics and shedding counters."""

    try:
        response = httpx.get(
            f"{gateway}/metrics",
            timeout=5.0,
        )
        response.raise_for_status()

        data = response.json()

        keys = [
            "kv_occupancy_percent",       # token_usage: admission signal
            "kv_token_usage_percent",
            "kv_pool_occupancy_percent",  # Total KV-pool occupancy
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
        return {
            "error": repr(exc),
        }

# ---------------------------------------------------------
# 5. COMBINED MONITORING SNAPSHOT
# ---------------------------------------------------------

async def collect_snapshot(
    client,
    gateway,
    engine,
):
    """Collect engine, GPU and gateway metrics concurrently."""

    engine_metrics, gpu_metrics, gateway_metrics = (
        await asyncio.gather(
            asyncio.to_thread(
                read_engine_metrics,
                engine,
            ),
            asyncio.to_thread(
                read_gpu,
            ),
            asyncio.to_thread(
                read_gateway_metrics,
                gateway,
            ),
        )
    )

    health_started = time.perf_counter()

    health_ok = False
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
        health_error = (
            f"{type(exc).__name__}: {exc}"
        )

    health_latency_ms = round(
        (time.perf_counter() - health_started) * 1000,
        2,
    )

    return {
        "timestamp": utc_now(),
        "engine": engine_metrics,
        "gpu": gpu_metrics,
        "gateway": gateway_metrics,
        "engine_health": health_ok,
        "engine_health_latency_ms": health_latency_ms,
        "engine_health_error": health_error,
    }

# ---------------------------------------------------------
# 6. REQUEST GENERATION
# ---------------------------------------------------------

async def send_request(
    client,
    gateway,
    request_id,
    tokenizer,
    priority,
    max_tokens,
    logger,
    log_requests=True,
):
    """Generate and send one inference request."""

    prompt = build_prompt(
        request_id,
        tokenizer,
    )

    prompt_topic = INCIDENT_SCENARIOS[
        request_id % len(INCIDENT_SCENARIOS)
    ][0]

    started = time.perf_counter()
    t_sent = time.monotonic() - RUN_T0

    try:
        response = await client.post(
            f"{gateway}/generate",
            json={
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0,
                "priority": priority,
            },
        )

        latency = round(
            time.perf_counter() - started,
            3,
        )

        admission_kv = None

        try:
            body = response.json()

            if response.status_code == 200:
                admission_kv = body["gateway"][
                    "kv_occupancy_percent_at_admission"
                ]
            else:
                admission_kv = body["detail"].get(
                    "occupancy_percent"
                )
        except Exception:
            pass

        result = {
            "request_id": request_id,
            "prompt_topic": prompt_topic,
            "priority": priority,
            "status_code": response.status_code,
            "kv_usage_at_admission_percent": admission_kv,
            "latency_seconds": latency,
            "response_excerpt": response.text[:250],
        }

        record_event(
            t_sent,
            priority,
            response.status_code,
            admission_kv,
        )

        return result

    except asyncio.CancelledError:
        logger.warning(
            "request=%d cancelled by safety monitor",
            request_id,
        )

        raise

    except Exception as exc:
        latency = round(
            time.perf_counter() - started,
            3,
        )

        record_event(
            t_sent,
            priority,
            None,
            None,
            error=type(exc).__name__,
        )

        return {
            "request_id": request_id,
            "prompt_topic": prompt_topic,
            "priority": priority,
            "error": str(exc),
            "latency_seconds": latency,
        }

# ---------------------------------------------------------
# 7. CSV ROW CONVERSION
# ---------------------------------------------------------

def metric_csv_row(sample):
    engine = sample.get("engine", {})
    gpu = sample.get("gpu", {})
    gateway = sample.get("gateway", {})

    return {
        "timestamp": sample.get("timestamp"),

        # Admission metric (active tokens / pool): what the gateway sheds on.
        "kv_token_usage_percent": engine.get(
            "token_usage_percent"
        ),

        # Informational: includes idle, evictable prefix-cache tokens.
        "kv_pool_occupancy_percent": engine.get(
            "kv_pool_occupancy_percent"
        ),

        "max_total_tokens": engine.get(
            "max_total_tokens"
        ),

        "kv_available_tokens": engine.get(
            "kv_available_tokens"
        ),

        "kv_evictable_tokens": engine.get(
            "kv_evictable_tokens"
        ),

        "num_used_tokens": engine.get(
            "num_used_tokens"
        ),

        "num_running_reqs": engine.get(
            "num_running_reqs"
        ),

        "engine_metrics_available": engine.get(
            "metrics_available"
        ),

        "gpu_memory_used_mib": gpu.get(
            "memory_used_mib"
        ),

        "gpu_memory_free_mib": gpu.get(
            "memory_free_mib"
        ),

        "gpu_utilization_percent": gpu.get(
            "gpu_utilization_percent"
        ),

        "requests_received": gateway.get(
            "requests_received"
        ),

        "requests_accepted": gateway.get(
            "requests_accepted"
        ),

        "requests_shed": gateway.get(
            "requests_shed"
        ),

        "engine_errors": gateway.get(
            "engine_errors"
        ),

        "kv_metrics_errors": gateway.get(
            "kv_metrics_errors"
        ),

        "kv_threshold_percent": gateway.get(
            "kv_threshold_percent"
        ),

        "engine_health": sample.get(
            "engine_health"
        ),

        "engine_health_latency_ms": sample.get(
            "engine_health_latency_ms"
        ),
    }

# ---------------------------------------------------------
# 8. BURST EXPERIMENT
# ---------------------------------------------------------

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
        "max_tokens": max_tokens,
        "priority": priority,
        "threshold_percent": threshold,
        "samples": [],
        "responses": [],
        "safety_stop": None,
        "threshold_observed": False,
        "status_counts": {},
    }

    request_tasks = []
    monitor_task = None

    # -----------------------------------------------------
    # Capture initial measurements
    # -----------------------------------------------------

    try:
        stage["before"] = {
            "engine": await asyncio.to_thread(
                read_engine_metrics,
                engine,
            ),

            "gpu": await asyncio.to_thread(
                read_gpu,
            ),

            "gateway": await asyncio.to_thread(
                read_gateway_metrics,
                gateway,
            ),
        }

        initial_gpu = stage["before"]["gpu"]

        initial_free = initial_gpu.get(
            "memory_free_mib"
        )

        initial_engine = stage["before"]["engine"]

        # Initial GPU safety check
        if (
            initial_free is None
            or initial_free < min_free_mib
        ):
            stage["safety_stop"] = (
                "Insufficient GPU free memory before burst: "
                f"{initial_free} MiB"
            )

        # Initial engine metrics check
        elif not initial_engine.get("metrics_available"):
            stage["safety_stop"] = (
                "SGLang KV metrics unavailable before burst"
            )

        if stage["safety_stop"]:
            logger.error(
                "SAFETY STOP: %s",
                stage["safety_stop"],
            )

        else:

            # -------------------------------------------------
            # Launch concurrent requests
            # -------------------------------------------------

            request_tasks = [
                asyncio.create_task(
                    send_request(
                        client=client,
                        gateway=gateway,
                        request_id=i,
                        tokenizer=tokenizer,
                        priority=priority,
                        max_tokens=max_tokens,
                        logger=logger,
                    ),
                    name=f"inference-request-{i}",
                )

                for i in range(request_count)
            ]

            # -------------------------------------------------
            # Monitoring coroutine
            # -------------------------------------------------

            async def monitor():

                consecutive_health_failures = 0
                consecutive_gpu_errors = 0
                consecutive_engine_metric_errors = 0

                next_sample_time = time.perf_counter()

                while True:

                    try:
                        # Collect all monitoring metrics.
                        sample = await collect_snapshot(
                            client,
                            gateway,
                            engine,
                        )

                        stage["samples"].append(sample)

                        engine_metrics = sample["engine"]
                        gpu_metrics = sample["gpu"]

                        # -------------------------------------
                        # Health monitoring
                        # -------------------------------------

                        if sample["engine_health"]:
                            consecutive_health_failures = 0

                        else:
                            consecutive_health_failures += 1

                        # -------------------------------------
                        # GPU monitoring
                        # -------------------------------------

                        free = gpu_metrics.get(
                            "memory_free_mib"
                        )

                        if free is None:
                            consecutive_gpu_errors += 1

                        else:
                            consecutive_gpu_errors = 0

                        # -------------------------------------
                        # Engine metrics monitoring
                        # -------------------------------------

                        if engine_metrics.get(
                            "metrics_available"
                        ):
                            consecutive_engine_metric_errors = 0

                        else:
                            consecutive_engine_metric_errors += 1

                        # -------------------------------------
                        # KV threshold observation
                        # -------------------------------------

                        gateway_metrics = sample.get("gateway", {})

                        occupancy = gateway_metrics.get(
                            "kv_occupancy_percent"
                        )

                        # Threshold crossing is observed,
                        # NOT used to cancel requests.

                        if (
                            occupancy is not None
                            and occupancy >= threshold
                            and not stage["threshold_observed"]
                        ):

                            stage["threshold_observed"] = True

                            logger.warning(
                                "Gateway admission threshold observed: %.2f%% "
                                "(threshold %.2f%%). "
                                "Admission shedding is being observed.",
                                occupancy,
                                threshold,
                            )

                        # -------------------------------------
                        # Actual GPU safety stop
                        # -------------------------------------

                        if (
                            free is not None
                            and free < min_free_mib
                        ):

                            stage["safety_stop"] = (
                                "GPU free memory below safety "
                                f"limit: {free} MiB"
                            )

                        # -------------------------------------
                        # Engine health safety stop
                        # -------------------------------------

                        elif consecutive_health_failures >= 3:

                            stage["safety_stop"] = (
                                "SGLang health check failed "
                                "3 consecutive times"
                            )

                        # -------------------------------------
                        # GPU monitoring safety stop
                        # -------------------------------------

                        elif consecutive_gpu_errors >= 3:

                            stage["safety_stop"] = (
                                "GPU monitoring failed "
                                "3 consecutive times"
                            )

                        # -------------------------------------
                        # Engine metrics monitoring safety
                        # -------------------------------------

                        elif (
                            consecutive_engine_metric_errors >= 3
                        ):

                            stage["safety_stop"] = (
                                "SGLang KV metrics unavailable "
                                "for 3 consecutive samples"
                            )

                        # -------------------------------------
                        # Safety intervention
                        # -------------------------------------

                        if stage["safety_stop"]:

                            logger.error(
                                "SAFETY STOP: %s",
                                stage["safety_stop"],
                            )

                            for task in request_tasks:

                                if not task.done():
                                    task.cancel()

                            # Wait for every client task to finish
                            # its cancellation cleanup.

                            await asyncio.gather(
                                *request_tasks,
                                return_exceptions=True,
                            )

                            return

                        # -------------------------------------
                        # Normal completion
                        # -------------------------------------

                        if all(
                            task.done()
                            for task in request_tasks
                        ):
                            return

                        # -------------------------------------
                        # Accurate scheduled sampling
                        # -------------------------------------

                        next_sample_time += sample_interval

                        delay = max(
                            0.0,
                            next_sample_time - time.perf_counter(),
                        )

                        await asyncio.sleep(delay)

                    except asyncio.CancelledError:
                        raise

                    except Exception as exc:

                        stage["safety_stop"] = (
                            "Monitoring task failed: "
                            f"{type(exc).__name__}: {exc}"
                        )

                        logger.exception(
                            "Monitoring failed unexpectedly"
                        )

                        # Do not leave inference requests running
                        # without a functioning safety monitor.

                        for task in request_tasks:

                            if not task.done():
                                task.cancel()

                        await asyncio.gather(
                            *request_tasks,
                            return_exceptions=True,
                        )

                        return

            # -------------------------------------------------
            # Start monitor and coordinate execution
            # -------------------------------------------------

            monitor_task = asyncio.create_task(
                monitor(),
                name="backpressure-monitor",
            )

            await monitor_task

    except asyncio.CancelledError:

        stage["safety_stop"] = (
            stage["safety_stop"]
            or "Experiment coroutine cancelled externally"
        )

        logger.warning(
            "Experiment cancellation received"
        )

        raise

    except Exception as exc:

        stage["safety_stop"] = (
            stage["safety_stop"]
            or f"Experiment execution failed: {exc}"
        )

        logger.exception(
            "Experiment execution failed"
        )

    finally:

        # -----------------------------------------------------
        # Ensure monitor is not left running
        # -----------------------------------------------------

        if (
            monitor_task is not None
            and not monitor_task.done()
        ):

            monitor_task.cancel()

            await asyncio.gather(
                monitor_task,
                return_exceptions=True,
            )

        # -----------------------------------------------------
        # Ensure all request tasks are accounted for
        # -----------------------------------------------------

        if stage["safety_stop"]:

            for task in request_tasks:

                if not task.done():
                    task.cancel()

        if request_tasks:

            await asyncio.gather(
                *request_tasks,
                return_exceptions=True,
            )

        # -----------------------------------------------------
        # Collect every request result
        # -----------------------------------------------------

        for request_id, task in enumerate(request_tasks):

            if task.cancelled():

                stage["responses"].append({
                    "request_id": request_id,
                    "priority": priority,
                    "error": "Cancelled by safety monitor",
                    "outcome": "cancelled",
                })

            else:

                try:
                    result = task.result()

                    if isinstance(result, dict):

                        if "status_code" in result:
                            result["outcome"] = (
                                "http_response"
                            )

                        else:
                            result["outcome"] = "error"

                        stage["responses"].append(result)

                except Exception as exc:

                    stage["responses"].append({
                        "request_id": request_id,
                        "priority": priority,
                        "error": str(exc),
                        "outcome": "error",
                    })

        # -----------------------------------------------------
        # Final resource measurements
        # -----------------------------------------------------

        stage["after"] = {
            "engine": await asyncio.to_thread(
                read_engine_metrics,
                engine,
            ),

            "gpu": await asyncio.to_thread(
                read_gpu,
            ),

            "gateway": await asyncio.to_thread(
                read_gateway_metrics,
                gateway,
            ),
        }

        stage["finished_at"] = utc_now()

        # -----------------------------------------------------
        # HTTP outcome counts
        # -----------------------------------------------------

        status_counts = {}

        for result in stage["responses"]:

            status = str(
                result.get(
                    "status_code",
                    result.get("outcome", "error"),
                )
            )

            status_counts[status] = (
                status_counts.get(status, 0) + 1
            )

        stage["status_counts"] = status_counts

        # -----------------------------------------------------
        # Calculate peak resource measurements
        # -----------------------------------------------------

        occupancies = [
            sample["gateway"].get("kv_occupancy_percent")

            for sample in stage["samples"]

            if sample.get("gateway", {}).get(
                "kv_occupancy_percent"
            ) is not None
        ]

        pool_occupancies = [
            sample["engine"].get("kv_pool_occupancy_percent")

            for sample in stage["samples"]

            if sample["engine"].get(
                "kv_pool_occupancy_percent"
            ) is not None
        ]

        gpu_used = [
            sample["gpu"].get(
                "memory_used_mib"
            )

            for sample in stage["samples"]

            if sample["gpu"].get(
                "memory_used_mib"
            ) is not None
        ]

        gpu_free = [
            sample["gpu"].get(
                "memory_free_mib"
            )

            for sample in stage["samples"]

            if sample["gpu"].get(
                "memory_free_mib"
            ) is not None
        ]

        stage["peak_kv_occupancy_percent"] = (
            max(occupancies) if occupancies else None
        )

        stage["peak_kv_pool_occupancy_percent"] = (
            max(pool_occupancies) if pool_occupancies else None
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

        # -----------------------------------------------------
        # Final logging
        # -----------------------------------------------------

        logger.info(
            "HTTP outcome counts: %s",
            status_counts,
        )

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
            "Safety stop: %s",
            stage["safety_stop"],
        )

        logger.info(
            "Gateway final counters: %s",
            stage["after"]["gateway"],
        )

    return stage


async def run_sustained_load(
    client,
    gateway,
    engine,
    request_count,
    max_tokens,
    threshold,
    min_free_mib,
    tokenizer,
    sample_interval,
    logger,
    priority="high",
    duration=None,
    total_requests=None,
    max_tokens_min=None,
):
    """
    Keep `request_count` requests in flight at all times (a finished
    request is immediately replaced) so KV-cache occupancy stays up.
    Prints live KV occupancy until Ctrl-C (or --duration seconds).

    The load defaults to Tier 1 ("high") so it is never shed and can
    hold occupancy above the threshold while Tier 2 probes are tested.
    """
    logger.info(
        "Starting sustained load: %d requests continuously in flight, "
        "priority=%s, threshold=%.1f%%",
        request_count,
        priority,
        threshold,
    )
    logger.info(
        "Probe from another terminal with --mode priority-batch "
        "--priority-batch tier1|tier2"
    )

    stage = {
        "name": "sustained_load",
        "priority": priority,
        "request_count": request_count,
        "samples": [],
        "safety_stop": None,
    }

    stop = asyncio.Event()
    counters = {"ok": 0, "shed_503": 0, "other": 0}
    ids = itertools.count()

    async def worker():
        while not stop.is_set():
            rid = next(ids)

            if total_requests and rid >= total_requests:
                return

            # Different generation lengths desynchronize the requests so KV
            # usage stays high instead of rising and collapsing in waves.
            request_tokens = (
                max_tokens_min + (rid * 97) % (max_tokens - max_tokens_min + 1)
                if max_tokens_min
                else max_tokens
            )

            result = await send_request(
                client=client,
                gateway=gateway,
                request_id=rid,
                tokenizer=tokenizer,
                priority=priority,
                max_tokens=request_tokens,
                logger=logger,
                log_requests=False,
            )

            code = result.get("status_code")

            if code == 200:
                counters["ok"] += 1
            elif code == 503:
                counters["shed_503"] += 1
                await asyncio.sleep(0.5)
            else:
                counters["other"] += 1
                await asyncio.sleep(0.5)

    workers = [
        asyncio.create_task(worker())
        for _ in range(request_count)
    ]

    started = time.monotonic()
    health_failures = gpu_errors = engine_errors = 0

    try:
        while not stop.is_set():
            sample = await collect_snapshot(client, gateway, engine)
            stage["samples"].append(sample)

            occupancy = sample["gateway"].get("kv_token_usage_percent")
            free = sample["gpu"].get("memory_free_mib")

            pool = sample["engine"].get("kv_pool_occupancy_percent")

            if EVENTS:
                EVENTS.flush()

            logger.info(
                "LIVE: KV token usage (admission metric)=%s | %s threshold "
                "%.0f%% | pool incl. idle cache=%s%% | running=%s "
                "| load ok=%d shed=%d other=%d | GPU free=%s MiB",
                f"{occupancy:.1f}%" if occupancy is not None else "n/a",
                "ABOVE" if occupancy is not None
                and occupancy >= threshold else "below",
                threshold,
                pool,
                sample["engine"].get("num_running_reqs"),
                counters["ok"],
                counters["shed_503"],
                counters["other"],
                free,
            )

            health_failures = (
                0 if sample["engine_health"] else health_failures + 1
            )
            gpu_errors = gpu_errors + 1 if free is None else 0
            engine_errors = (
                0 if sample["engine"].get("metrics_available")
                else engine_errors + 1
            )

            reason = None

            if free is not None and free < min_free_mib:
                reason = f"GPU free memory below limit: {free} MiB"
            elif health_failures >= 3:
                reason = "Engine health failed 3 times"
            elif gpu_errors >= 3:
                reason = "GPU metrics unavailable 3 times"
            elif engine_errors >= 3:
                reason = "Engine metrics unavailable 3 times"

            if reason:
                logger.error("SAFETY STOP: %s", reason)
                stage["safety_stop"] = reason
                break

            if all(task.done() for task in workers):
                logger.info(
                    "All %s requests finished", total_requests
                )
                break

            if duration and time.monotonic() - started >= duration:
                logger.info("Load duration reached (%ss)", duration)
                break

            await asyncio.sleep(sample_interval)

    except asyncio.CancelledError:
        logger.warning("Sustained load stopped by user")

    finally:
        stop.set()

        for task in workers:
            task.cancel()

        await asyncio.gather(*workers, return_exceptions=True)

        stage["load_counters"] = counters

    return stage

async def run_priority_batch(
    client,
    gateway,
    engine,
    request_count,
    max_tokens,
    threshold,
    priority_batch,
    tokenizer,
    logger,
    wait_above=None,
):
    priority_map = {
        "tier1": "high",
        "tier2": "standard",
    }

    priority = priority_map[priority_batch]

    logger.info(
        "Starting %s batch: priority=%s requests=%d",
        priority_batch.upper(),
        priority,
        request_count,
    )

    if wait_above:
        logger.info(
            "Waiting up to %.0fs for KV token usage >= %.0f%% before sending",
            wait_above,
            threshold,
        )

        deadline = time.monotonic() + wait_above

        while time.monotonic() < deadline:
            probe = await asyncio.to_thread(
                read_gateway_metrics,
                gateway,
            )

            usage = probe.get("kv_token_usage_percent")

            if usage is not None and usage >= threshold:
                logger.info("KV token usage %.1f%% >= threshold: sending", usage)
                break

            await asyncio.sleep(0.2)
        else:
            logger.warning("KV usage never exceeded threshold; sending anyway")

    # Check actual gateway admission metric before submitting.
    gateway_metrics = await asyncio.to_thread(
        read_gateway_metrics,
        gateway,
    )

    occupancy = gateway_metrics.get(
        "kv_token_usage_percent"
    )

    if occupancy is None:
        raise RuntimeError(
            "Gateway KV token usage metric unavailable"
        )

    logger.info(
        "Current KV token usage: %.2f%%",
        occupancy,
    )


    if occupancy >= threshold:
        expected = (
            "all admitted (Tier 1 bypasses shedding)"
            if priority == "high"
            else "HTTP 503 for Tier 2 (KV above threshold)"
        )
    else:
        expected = (
            "all admitted (KV below threshold)"
        )

    logger.info(
        "KV usage %.2f%% is %s threshold %.2f%% -> expected: %s "
        "(usage is re-checked per request by the gateway)",
        occupancy,
        "AT/ABOVE" if occupancy >= threshold else "below",
        threshold,
        expected,
    )

    request_id_base = (
        10000 if priority_batch == "tier1" else 20000
    )

    tasks = [
        asyncio.create_task(
            send_request(
                client=client,
                gateway=gateway,
                request_id=request_id_base + i,
                tokenizer=tokenizer,
                priority=priority,
                max_tokens=max_tokens,
                logger=logger,
            )
        )
        for i in range(request_count)
    ]

    # A 503 returns at once, but an admitted request queues in the engine
    # (FCFS) behind the running load, so print a heartbeat while waiting.
    pending = set(tasks)
    batch_start = time.monotonic()

    while pending:
        _, pending = await asyncio.wait(pending, timeout=10)

        if pending:
            probe = await asyncio.to_thread(read_gateway_metrics, gateway)
            logger.info(
                "waiting %.0fs: %d/%d responses back, %d admitted and "
                "queued in the engine behind the running load "
                "(KV token usage %s%%)",
                time.monotonic() - batch_start,
                request_count - len(pending),
                request_count,
                len(pending),
                probe.get("kv_token_usage_percent"),
            )

    results = [
        t.exception() or t.result() if not t.cancelled() else None
        for t in tasks
    ]
    results = [
        r if r is not None else RuntimeError("cancelled") for r in results
    ]

    if EVENTS:
        EVENTS.flush()

    shed = next(
        (
            r for r in results
            if not isinstance(r, Exception)
            and r.get("status_code") == 503
        ),
        None,
    )

    if shed:
        logger.info(
            "Sample 503 response body: %s",
            shed.get("response_excerpt", "")[:200],
        )

    status_counts = {}

    for result in results:
        if isinstance(result, Exception):
            status = "exception"

        else:
            status = str(
                result.get(
                    "status_code",
                    "error",
                )
            )

        status_counts[status] = (
            status_counts.get(status, 0) + 1
        )

    final_metrics = await asyncio.to_thread(
        read_gateway_metrics,
        gateway,
    )

    observed = [
        r for r in results if not isinstance(r, Exception)
    ]

    above = [
        r for r in observed
        if (r.get("kv_usage_at_admission_percent") or 0) >= threshold
    ]

    below = [
        r for r in observed
        if r.get("kv_usage_at_admission_percent") is not None
        and r["kv_usage_at_admission_percent"] < threshold
    ]

    logger.info(
        "EVIDENCE %s: %d/%d requests saw KV >= %.0f%% at admission "
        "(HTTP %s); %d/%d saw KV < %.0f%% (HTTP %s)",
        priority_batch.upper(),
        len(above),
        len(observed),
        threshold,
        sorted({r.get("status_code") for r in above}),
        len(below),
        len(observed),
        threshold,
        sorted({r.get("status_code") for r in below}),
    )

    logger.info(
        "%s RESULTS: %s",
        priority_batch.upper(),
        status_counts,
    )

    logger.info(
        "Gateway counters after batch: %s",
        final_metrics,
    )

    return {
        "priority_batch": priority_batch,
        "gateway_priority": priority,
        "initial_kv_usage_percent": occupancy,
        "status_counts": status_counts,
        "per_request": [
            r for r in results if not isinstance(r, Exception)
        ],
        "gateway_metrics_after": final_metrics,
    }

# ---------------------------------------------------------
# 9. MAIN EXPERIMENT
# ---------------------------------------------------------

async def main(args):

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger, log_path = configure_logging(output_dir)

    global RUN_T0, EVENTS
    RUN_T0 = time.monotonic()
    EVENTS = EventLog(
        logging.getLogger("backpressure.events"),
        args.threshold,
    )

    report = {
        "experiment": (
            "Module 2 live backpressure demonstration"
        ),

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

    verify_prompts(
        tokenizer,
        args.requests,
    )

    try:

        async with httpx.AsyncClient(
            timeout=timeout,
            limits=limits,
        ) as client:

            # -------------------------------------------------
            # Verify both services
            # -------------------------------------------------

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

            # -------------------------------------------------
            # Initial system measurements
            # -------------------------------------------------

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

            # -------------------------------------------------
            # Run burst experiment
            # -------------------------------------------------

            if args.mode == "burst":

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

            elif args.mode == "load":

                stage = await run_sustained_load(
                    priority=args.load_priority,
                    duration=args.duration,
                    total_requests=args.total_requests,
                    max_tokens_min=args.max_tokens_min,
                    client=client,
                    gateway=args.gateway,
                    engine=args.engine,
                    request_count=args.requests,
                    max_tokens=args.max_tokens,
                    threshold=args.threshold,
                    min_free_mib=args.min_free_mib,
                    tokenizer=tokenizer,
                    sample_interval=args.sample_interval,
                    logger=logger,
                )

                report["stages"].append(stage)

            elif args.mode == "priority-batch":

                result = await run_priority_batch(
                    client=client,
                    gateway=args.gateway,
                    engine=args.engine,
                    request_count=args.requests,
                    max_tokens=args.max_tokens,
                    threshold=args.threshold,
                    priority_batch=args.priority_batch,
                    tokenizer=tokenizer,
                    logger=logger,
                    wait_above=args.wait_above,
                )

                report["priority_batch_result"] = result
    finally:

        if EVENTS:
            EVENTS.flush()

        report["finished_at"] = utc_now()

        # -----------------------------------------------------
        # Final measurements
        # -----------------------------------------------------

        report["final_gateway_metrics"] = await asyncio.to_thread(
            read_gateway_metrics,
            args.gateway,
        )

        report["final_engine_metrics"] = await asyncio.to_thread(
            read_engine_metrics,
            args.engine,
        )

        report["final_gpu"] = await asyncio.to_thread(
            read_gpu
        )

        # -----------------------------------------------------
        # Generate CSV
        # -----------------------------------------------------

        suffix = (
            ""
            if args.mode == "burst"
            else f"_{args.mode}"
            + (f"_{args.priority_batch}" if args.priority_batch else "")
        )

        csv_path = (
            output_dir / f"backpressure_metrics{suffix}.csv"
        )

        fieldnames = [
            "timestamp",

            "kv_token_usage_percent",
            "kv_pool_occupancy_percent",
            "max_total_tokens",
            "kv_available_tokens",
            "kv_evictable_tokens",
            "num_used_tokens",
            "num_running_reqs",
            "engine_metrics_available",

            "gpu_memory_used_mib",
            "gpu_memory_free_mib",
            "gpu_utilization_percent",

            "requests_received",
            "requests_accepted",
            "requests_shed",

            "engine_errors",
            "kv_metrics_errors",
            "kv_threshold_percent",

            "engine_health",
            "engine_health_latency_ms",
        ]

        samples = [
            metric_csv_row(sample)

            for stage in report["stages"]

            for sample in stage.get(
                "samples",
                [],
            )
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

        # -----------------------------------------------------
        # Generate JSON
        # -----------------------------------------------------

        json_path = (
            output_dir / f"backpressure_live_report{suffix}.json"
        )

        report["artifacts"] = {
            "json": str(json_path),
            "csv": str(csv_path),
            "log": str(log_path),
        }

        json_path.write_text(
            json.dumps(
                report,
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )

        # -----------------------------------------------------
        # Final logging
        # -----------------------------------------------------

        logger.info("=== EXPERIMENT COMPLETE ===")

        logger.info(
            "JSON report: %s",
            json_path,
        )

        logger.info(
            "Metrics CSV: %s",
            csv_path,
        )

        logger.info(
            "Execution log: %s",
            log_path,
        )

        logger.info(
            "Final gateway metrics: %s",
            report["final_gateway_metrics"],
        )

        logger.info(
            "Final engine metrics: %s",
            report["final_engine_metrics"],
        )

        logger.info(
            "Final GPU: %s",
            report["final_gpu"],
        )

# ---------------------------------------------------------
# 10. COMMAND-LINE ARGUMENTS
# ---------------------------------------------------------

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
        default=str(
            ROOT / "results" / "module_2"
        ),
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
        default=900.0,
    )

    parser.add_argument(
        "--priority",
        choices=[
            "standard",
            "high",
        ],
        default="standard",
    )

    parser.add_argument(
        "--sample-interval",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--mode",
        choices=[
            "burst",
            "load",
            "priority-batch",
        ],
        default="burst",
    )

    parser.add_argument(
        "--load-priority",
        choices=["high", "standard"],
        default="high",
        help="Priority of the continuous load (--mode load)",
    )

    parser.add_argument(
        "--max-tokens-min",
        type=int,
        default=None,
        help="--mode load: vary max_tokens per request between this and --max-tokens",
    )

    parser.add_argument(
        "--total-requests",
        type=int,
        default=None,
        help="Stop --mode load after this many requests (default: unlimited)",
    )

    parser.add_argument(
        "--wait-above",
        type=float,
        default=None,
        metavar="SECONDS",
        help="priority-batch: wait up to SECONDS for KV usage > threshold before sending",
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Stop --mode load after this many seconds (default: Ctrl-C)",
    )

    parser.add_argument(
        "--priority-batch",
        choices=[
            "tier1",
            "tier2",
        ],
        help="Priority batch to execute",
    )

    args = parser.parse_args()

    # Validate experiment configuration.

    if args.requests < 1:
        parser.error("--requests must be at least 1")

    if args.max_tokens < 1:
        parser.error("--max-tokens must be at least 1")

    if not 0 < args.threshold <= 100:
        parser.error("--threshold must be between 0 and 100")

    if args.min_free_mib < 0:
        parser.error("--min-free-mib cannot be negative")

    if args.timeout <= 0:
        parser.error("--timeout must be positive")

    if args.sample_interval <= 0:
        parser.error("--sample-interval must be positive")

    if args.mode == "priority-batch" and not args.priority_batch:
        parser.error(
            "--priority-batch is required when mode is priority-batch"
        )

    return args

# ---------------------------------------------------------
# 11. ENTRY POINT
# ---------------------------------------------------------

if __name__ == "__main__":

    try:
        asyncio.run(
            main(parse_args())
        )

    except KeyboardInterrupt:
        print("\nExperiment interrupted by user.")

    except Exception:
        logging.exception(
            "Module 2 experiment terminated unexpectedly"
        )
        sys.exit(1)