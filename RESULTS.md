# SLA-gated LLM serving on a Colab T4

Qwen2.5-7B-Instruct-AWQ (4-bit) · SGLang (Triton attention, FP16) · FastAPI gateway · Tesla T4 15 GB · Python 3.13

Engine flags:
- `--quantization awq --dtype float16 --attention-backend triton --sampling-backend pytorch --mem-fraction-static 0.80 --context-length 4096 --max-running-requests 32 --enable-metrics --enable-cache-report`

## VRAM allocation

| Item | Size |
|---|---|
| GPU total memory | 15,360 MiB |
| GPU memory used | 13,871 MiB |
| GPU memory free | 1,042 MiB |
| Static weights (AWQ 4-bit) | 5.3633 GB |
| KV-cache pool | 6.1717 GB |
| Total weights + KV-cache pool | 11.5350 GB |
| KV-cache capacity | 115,562 tokens |
| Available KV-cache tokens | 115,548 |
| KV-cache memory per token (BF16) | 56 KiB |
| Available GPU memory reported by SGLang at startup | 1.8787 GB |

## Latency and throughput

| Scenario | Requests | Success | TTFT p50 | TTFT p95 | TTFT p99 | ITL p50 | ITL p95 | ITL p99 | E2E p50 | E2E p95 | E2E p99 | Input tok/s | Output tok/s | Request/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A: Distinct prompts | 50 | 50/50 | 3.55s | 57.22s | 59.44s | 150.92ms | 196.75ms | 289.25ms | 47.40s | 88.65s | 88.65s | 168.79 | 144.04 | 0.56 |
| B: Shared ~2k-token prefix | 100 | 100/100 | 128.13s | 198.44s | 350.00s | 295.83ms | 389.58ms | 397.92ms | 172.08s | 250.38s | 291.68s | 706.64 | 87.75 | 0.34 |
| C: Constrained JSON schema | 50 | 50/50 | 3.57s | 18.68s | 19.74s | 190.04ms | 377.51ms | 395.50ms | 16.73s | 25.15s | 25.88s | 174.77 | 127.29 | 1.92 |

## Mean latency and throughput

| Scenario | Mean TTFT | Mean ITL | Mean E2E | Wall time | Input tokens | Output tokens |
|---|---:|---:|---:|---:|---:|---:|
| A: 50 requests | 18.98s | 168.22ms | 62.20s | 88.87s | 15,000 | 12,800 |
| B: 100 requests | 107.52s | 250.83ms | 174.62s | 291.75s | 206,162 | 25,600 |
| C: 50 requests | 6.59s | 181.34ms | 18.04s | 26.03s | 4,550 | 3,314 |

## Cache efficiency

| Scenario | Cache-hit-rate gauge | Mean cached tokens/request |
|---|---:|---:|
| A: Distinct prompts | 8.67% | 299 |
| B: Shared ~2k-token prefix | 97.73% | 1,990.03 |
| C: Constrained JSON | 49.82% | 20.36 |

### Cold vs. warm prefix experiment

| Metric | Cold request | Warm request |
|---|---:|---:|
| Cached tokens | 0 | 1,504 |
| Request latency | 13.353s | 9.362s |
| Cache-hit-rate gauge | 0% | 99.34% |

Observed request-latency reduction: **29.9%**.

## Structured JSON generation

| Metric | Result |
|---|---:|
| Total requests | 50 |
| Successful requests | 50/50 |
| Valid JSON responses | 50/50 |
| Schema validation success rate | 100% |
| HTTP errors | 0 |

- Native SGLang JSON Schema constraint enabled.
- Gateway-side JSON parsing and schema validation enabled.
- All 50 responses passed validation.

*TTFT and ITL percentiles are estimated from SGLang Prometheus histogram buckets. E2E latency is measured by the benchmark client. Cache-hit-rate values are gauge snapshots; cached-token metadata indicates per-request prefix reuse.*


## Load-Shedding Log (HTTP 503 Backpressure)

**Gateway Rule:** Tier 2 (`X-Tier: 2`) requests receive HTTP 503 when SGLang's `sglang:token_usage` exceeds the configured threshold. Tier 1 requests are never shed. Default threshold: `0.90`.

### Threshold Validation

Ran the demo with `KV_THRESH=0.80`.

Over two attempts with 32 concurrent Tier-1 requests (~3K and ~2K token prompts), observed KV usage peaked at approximately `0.83–0.87`.

SGLang's scheduler holds back admission to reserve capacity for decoding, so the default `0.90` threshold was not crossed on this T4 configuration.

The `0.80` run demonstrates that the load-shedding mechanism works; it does not demonstrate a trigger at `0.90`.

### Flood Test Configuration

- **Tier-1 requests:** 32 unique requests
- **Prompt length:** 3,000 tokens each
- **Forced output:** 600 tokens each
- **KV-cache pool:** 110,154 tokens
- **Tier-2 probes:** One probe every 2 seconds
- **Total Tier-2 probes:** 255
- **KV threshold:** 0.80

### Results

15 Tier-2 probes returned HTTP 503, all after KV usage reached the configured threshold. Probes sent before the threshold was exceeded were admitted.
```text
[INFO]  t=473.4s  kv_usage=0.800  tier2 -> admitted (not above threshold)
[INFO]  t=479.4s  kv_usage=0.800  tier2 -> admitted
[WARN]  t=480.4s  kv_usage=0.820  tier2 -> HTTP 503
[INFO]  t=481.4s  kv_usage=0.800  tier2 -> admitted
[WARN]  t=482.4s  kv_usage=0.820  tier2 -> HTTP 503
[INFO]  t=483.4s  kv_usage=0.800  tier2 -> admitted
[WARN]  t=484.4s  kv_usage=0.820  tier2 -> HTTP 503
[WARN]  t=486.4s .. t=506.4s  kv_usage=0.820  tier2 -> HTTP 503   (11 more)
[WARN]  t=508.4s  kv_usage=0.830  tier2 -> HTTP 503
```

### Gateway Statistics

Gateway `/stats` after the run:

```json
{
  "shed": 15,
  "log": [
    {"kv_usage": 0.82},
    "...",
    {"kv_usage": 0.83}
  ]
}
