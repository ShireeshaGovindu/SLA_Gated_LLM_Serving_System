# SLA Gated LLM Serving System

An SLA-aware inference gateway in front of an [SGLang](https://github.com/sgl-project/sglang) engine serving a 4-bit AWQ model (`Qwen/Qwen2.5-7B-Instruct-AWQ`). It combines prefix caching (RadixAttention), schema-constrained JSON output, KV-cache backpressure with priority tiers, and a reproducible benchmark suite.

Developed and measured on a single NVIDIA Tesla T4 (Google Colab, ~15 GB VRAM).

## Contents

1. [System architecture](#system-architecture)
2. [Technology stack](#technology-stack)
3. [Repository layout](#repository-layout)
4. [Environment setup](#environment-setup)
5. [Execution instructions](#execution-instructions)
6. [Benchmark reproduction methodology](#benchmark-reproduction-methodology)
7. [Results summary](#results-summary)
8. [Known limitations](#known-limitations)

---

## System architecture

```
                 ┌──────────────┐   POST /generate, /generate-json
   clients ────▶ │   Gateway    │   {prompt, max_tokens, temperature,
 (benchmark.py,  │  FastAPI     │    priority, schema?, request_id?}
  demo scripts)  │  :8081       │
                 └──────┬───────┘
                        │ 1. read sglang:token_usage (cached 0.25 s)
                        │ 2. admission decision (Tiered)
                        │ 3. bounded concurrency (semaphore)
                        |
                        ▼
                 ┌──────────────┐   POST /generate   ┌─────────────────────┐
                 │ SGLang engine│ ◀───────────────── │ Prometheus metrics  │
                 │ :30000       │ ──── /metrics ───▶ │ (token usage, cache,│
                 │ AWQ 4-bit    │                    │  TTFT/ITL histograms)│
                 │ RadixAttention                    └─────────────────────┘
                 └──────────────┘
                        │
                   NVIDIA T4 GPU
```

## Technology stack

| Layer | Technology |
|---|---|
| Inference engine | SGLang (`sglang[all]`), RadixAttention, Prometheus metrics |
| Model | `Qwen/Qwen2.5-7B-Instruct-AWQ` (4-bit AWQ, fp16 activations) |
| Gateway | Python 3.11, FastAPI, Uvicorn, httpx (async), Pydantic v2 |
| Hardware | NVIDIA Tesla T4 (15,360 MiB), CUDA |
| Packaging | Docker / Docker Compose (gateway only) |
| Observability | Prometheus text metrics, `nvidia-smi`, per-run JSON/CSV/log artifacts |

Python dependencies for the gateway and benchmarks are pinned by range in [requirements.txt](requirements.txt): `fastapi`, `uvicorn[standard]`, `httpx`, `pydantic`, `pytest`. SGLang is installed separately.

---

## Repository layout

| Path | Purpose |
|---|---|
| [gateway.py](gateway.py) | FastAPI gateway: tiers, 503 load shedding, JSON-schema route |
| [benchmark.py](benchmark.py) | Scenarios A/B/C, warm/cold probe, latency/cache/TTFT/ITL analysis |
| [config/benchmark_config.py](config/benchmark_config.py) | Shared ~2k-token system prompt, Scenario A domains, Scenario B questions, JSON schema |
| [config/gateway.env](config/gateway.env) | Gateway environment variables |
| [module_2/demo_backpressure.py](module_2/demo_backpressure.py) | Live backpressure demonstration |
| [module_2/config/backpressure_prompts.py](module_2/config/backpressure_prompts.py) | Prompts for the demo |
| [scripts/start_engine.sh](scripts/start_engine.sh) | Launch SGLang |
| [scripts/start_gateway.sh](scripts/start_gateway.sh) | Launch the gateway |
| [scripts/run_benchmarks.sh](scripts/run_benchmarks.sh) | Run the Module 3 suite |
| [scripts/gateway_healthcheck.sh](scripts/gateway_healthcheck.sh), [scripts/sglang_healthcheck.sh](scripts/sglang_healthcheck.sh) | Engine and Gateway Health checks |
| [Dockerfile](Dockerfile), [docker-compose.yml](docker-compose.yml) | Containerised gateway |
| [results/module_2/](results/module_2/) | Backpressure report, CSV, log |
| [results/module_3/](results/module_3/) | Benchmark results (batch 32 run) |
| [RESULTS.md](RESULTS.md) | Performance report (VRAM, latency, cache, load shedding) |

---


### Gateway (`gateway.py`)

An asyncio FastAPI service that protects the engine from overload.

| Concern | Behaviour |
|---|---|
| **Priority tiers** | `priority: "high"` (Tier 1) is never shed on KV occupancy. `priority: "standard"` (Tier 2, default) is subject to load shedding. |
| **Backpressure** | Reads `sglang:token_usage` from the engine's Prometheus endpoint. If occupancy ≥ `KV_LIMIT` (default 90%), standard requests receive **HTTP 503** with `Retry-After: 1`. |
| **Fail closed** | If occupancy cannot be read, standard requests are also rejected (503, `kv_metrics_unavailable`). Missing metrics are never treated as 0%. |
| **Concurrency cap** | An `asyncio.Semaphore` (`GATEWAY_MAX_INFLIGHT`) bounds in-flight engine calls, including high-priority ones. |
| **Structured output** | `/generate-json` requires a `schema`, passes it to SGLang's constrained decoding (`json_schema` sampling param), then independently validates the returned JSON (required fields, types, unexpected fields). |
| **Error mapping** | Engine errors → 502, timeouts → gateway timeout error.|

#### Endpoints:

| Method | Path | Purpose |
|---|---|---|
| POST | `/generate` | Free-form generation |
| POST | `/generate-json` | Schema-constrained generation (requires `schema`) |
| GET | `/health` | Liveness, engine URL, KV threshold |
| GET | `/metrics` | Gateway counters (received / accepted / shed / high-priority allowed / engine errors / validation errors) and current KV occupancy |
| GET | `/engine-metrics` | Raw SGLang Prometheus text, proxied |

### Engine (`scripts/start_engine.sh`)

SGLang with AWQ 4-bit weights and RadixAttention prefix caching, with metrics enabled so the gateway and benchmarks can observe KV usage, cache hit rate, and TTFT/ITL histograms. T4-specific choices: `--dtype half` (no bf16), `--attention-backend triton`, `--sampling-backend pytorch`.

### Benchmarks (`benchmark.py`, `module_2/`)

- **Module 2** (`module_2/demo_backpressure.py`): a live burst of concurrent, distinct-prefix requests that drives KV occupancy toward the threshold and records 503s, KV metrics, GPU memory, and gateway counters.
- **Module 3** (`benchmark.py`): three scenarios (A/B/C) plus an optional warm/cold prefix-cache probe.

---



## Environment setup

### Prerequisites

- Linux host (or Colab) with an NVIDIA GPU. The reported numbers use a 15 GB T4.
- NVIDIA driver + CUDA compatible with SGLang/PyTorch build.
- Python 3.11 recommended (matches the Dockerfile).
- Network access to download the model from Hugging Face on first launch.

### Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install "sglang[all]"
```

### Configuration

Gateway settings live in [config/gateway.env](config/gateway.env):

| Variable | Default | Meaning |
|---|---|---|
| `SGLANG_URL` | `http://127.0.0.1:30000` | Engine base URL |
| `KV_LIMIT` | `90` | KV occupancy (%) at which standard requests are shed |
| `GATEWAY_MAX_INFLIGHT` | `128` | Max concurrent engine calls from the gateway |
| `ENGINE_TIMEOUT_SECONDS` | `300` | Per-request engine timeout |
| `METRICS_CACHE_TTL_SECONDS` | `0.25` | How long engine metrics are cached |

Other variables: `MODEL` (engine script, default `Qwen/Qwen2.5-7B-Instruct-AWQ`), `GATEWAY_URL`, `OUTPUT_DIR` (benchmark script).

### Docker (gateway only)

The engine needs the GPU and runs on the host; the gateway can be containerised.

```bash
docker compose up --build
```

`SGLANG_URL` defaults to `http://host.docker.internal:30000` in the compose file; override it in the shell if the engine is elsewhere.

---

## Execution instructions

Run each step in its own terminal (or background it). Start the engine first and wait until it says ready.

### 1. Start the engine

```bash
scripts/start_engine.sh
```

In the first launch, it downloads the model. Confirm with:

```bash
scripts/sglang_healthcheck.sh
```

### 2. Start the gateway

```bash
set -a; source config/gateway.env; set +a
python -m uvicorn gateway:app --host 127.0.0.1 --port 8081
```

`scripts/start_gateway.sh` does the same but hardcodes the Colab path `/content/LLM_Serving_System`; edit the two paths in it if you use it elsewhere.

#### Do the health check

```bash
scripts/gateway_healthcheck.sh
```

### 3. Smoke Test by sending a request

```bash
curl -s localhost:8081/generate -H 'content-type: application/json' -d '{
  "prompt": "Explain prefix caching in two sentences.",
  "max_tokens": 128,
  "priority": "standard"
}'

curl -s localhost:8081/generate-json -H 'content-type: application/json' -d '{
  "prompt": "Summarise KV-cache backpressure.",
  "schema": {"type":"object",
             "properties":{"summary":{"type":"string"},"score":{"type":"integer"}},
             "required":["summary","score"],"additionalProperties":false}
}'

curl -s localhost:8081/metrics
```

When KV occupancy is at or above `KV_LIMIT`, `standard` requests return `503` with a `Retry-After` header; `high` requests are still admitted.

### 4. Run the backpressure demo (Module 2)

```bash
python module_2/demo_backpressure.py \
    --gateway http://127.0.0.1:8081 --engine http://127.0.0.1:30000 \
    --requests 64 --max-tokens 128 --threshold 90
```

Writes `results/module_2/backpressure_live_report.json`, `backpressure_metrics.csv`, and `backpressure.log`. 

### 5. Run the benchmark suite (Module 3)

```bash
scripts/run_benchmarks.sh
```

Equivalent to `python benchmark.py --run-warm-cold --output-dir results`. Common flags:

| Flag | Default | Meaning |
|---|---|---|
| `--scenario` | `all` | `A`, `B`, `C`, or `all` |
| `--a-requests` / `--b-requests` / `--c-requests` | 50 / 100 / 50 | Requests per scenario |
| `--max-tokens` | 256 | Output cap per request |
| `--max-inflight` | 128 | Client-side concurrency cap |
| `--request-timeout` | 360 | Seconds per request |
| `--run-warm-cold` | off | Adds a paired warm/cold prefix-cache probe |
| `--output-dir` | `results` | Where result JSON is written |

Per-run logs go to `Logs/benchmark_<timestamp>.log`.

---

## Benchmark reproduction methodology



| Factor | Value used for the reported results |
|---|---|
| GPU | 1× Tesla T4, 15,360 MiB |
| Model | `Qwen/Qwen2.5-7B-Instruct-AWQ` |
| Engine flags | `--quantization awq --dtype half --attention-backend triton --sampling-backend pytorch --mem-fraction-static 0.85 --context-length 4096 --max-running-requests 32 --enable-metrics` |
| Gateway | `KV_LIMIT=90`, `GATEWAY_MAX_INFLIGHT=128` |
| Decoding | `temperature=0.0`, `max_tokens=256` |
| Client concurrency |  concurrency (`--max-inflight 128`) |

### Procedure

1. **Fresh engine.** Restart SGLang before the run so the RadixCache starts empty; otherwise Scenario A is not truly cold.
2. **Verify health.** Both healthcheck scripts pass; `GET /metrics` shows `kv_metrics_available: true`.
3. **Record GPU state.** Capture `nvidia-smi` before, during, and after.
4. **Run.** `scripts/run_benchmarks.sh` (or `--scenario X` for one scenario). Scenarios run sequentially against the same engine in order A → B → C → warm/cold probe.
5. **Collect artifacts.** `results/<dir>/scenario_{a,b,c}_results.json`, `benchmark_index.json` (settings and notes), and `Logs/benchmark_*.log`.
6. **Repeat** at least three times with a fresh engine each time and compare distributions, not single runs.

### Scenarios

| Scenario | Workload | What it isolates |
|---|---|---|
| **A: cold, unique prefixes** | 50 requests, each with a distinct system prompt / domain question | Baseline latency with no cross-request prefix reuse |
| **B: warm, shared prefix** | 100 requests sharing a ~2k-token system prompt, each with a distinct question | RadixAttention reuse of a long common prefix |
| **C: structured output** | 50 requests to `/generate-json` with a fixed schema | Constrained-decoding validity and cost |
| **Warm/cold probe** (`--run-warm-cold`) | Paired cold and warm requests on one prefix | Direct TTFT effect of a cached prefix |

Workload content is defined in [config/benchmark_config.py](config/benchmark_config.py) and is deterministic, so reruns issue the same prompts.

### Metrics and how they are measured

| Metric | Source |
|---|---|
| Request latency p50/p95/p99, throughput, success rate | Client-side timing per request |
| Cached / prompt tokens (cache hit ratio) | Per-request `cached_tokens` metadata returned by the engine |
| TTFT and ITL p50/p95/p99 | **Estimated** from SGLang Prometheus histogram deltas taken before and after each scenario |
| RadixCache hit-rate | Prometheus gauge snapshot after the scenario (a gauge, not a counter) |
| JSON validity | Gateway-side validation recorded per request (Scenario C) |
| KV occupancy, 503 counts | Gateway `/metrics` and engine `sglang:token_usage` |
| VRAM | `nvidia-smi` |

If the installed SGLang version lacks a histogram, the corresponding fields are `null` in the output rather than estimated.

### Interpreting results

- TTFT/ITL percentiles are limited by Prometheus bucket resolution; treat them as coarse.
- Scenario B's TTFT is dominated by **queueing** (100 concurrent requests against 16 running slots), not by prefill. Compare cached-token ratios, not B's TTFT versus A's, to judge caching. Use the warm/cold probe for a like-for-like TTFT comparison.
- Compare equivalent workloads: same prompt lengths, output limits, concurrency, and hardware.
- Raw per-request measurements are preserved in the scenario JSON files; recompute any statistic from them.

---

## Results summary

Batch 32 run on a T4, stored in [results/module_3/](results/module_3/). Full detail in [RESULTS.md](RESULTS.md).

| Scenario | Success | Request latency p50 / p95 / p99 (s) | TTFT p50 / p95 / p99 (s)* | ITL p50 / p95 / p99 (s)* |
|---|---|---|---|---|
| A: 50 unique system prompts | 50/50 | 47.40 / 88.65 / 88.65 | 3.55 / 57.22 / 59.44 | 0.15092 / 0.19675 / 0.28925 |
| B: 100 shared ~2k-token prefix | 100/100 | 172.08 / 250.38 / 291.68 | 128.13 / 198.44 / 350.00 | 0.29583 / 0.38958 / 0.39792 |
| C: 50 JSON-schema requests | 50/50 (all valid JSON) | 16.73 / 25.15 / 25.88 | 3.57 / 18.68 / 19.74 | 0.19004 / 0.37751 / 0.39550 |

\*Estimated from Prometheus histogram deltas.

---

## Known Limitations

- Backpressure uses `sglang:token_usage`, a point-in-time gauge cached for 0.25 s, so admission can lag a rapid burst.
- High-priority requests bypass occupancy shedding but still obey the in-flight semaphore, so they are not guaranteed lower latency.


