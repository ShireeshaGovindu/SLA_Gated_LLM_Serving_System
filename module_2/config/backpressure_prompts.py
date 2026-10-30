"""
Module 2: Backpressure Benchmark Prompts

64 realistic AI inference engineering scenarios.

Every prompt:
- Exactly 1500 input tokens.
- Uses the Qwen2.5 tokenizer.
- Has a unique incident identifier and scenario.
- Uses 128 output tokens.
"""

from transformers import AutoTokenizer


TARGET_PROMPT_TOKENS = 1500
PROMPT_TOKEN_TOLERANCE = 0
OUTPUT_TOKENS = 128

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct-AWQ"


INCIDENT_SCENARIOS = [
    ("TTFT Regression", "Production p95 TTFT increased from 420 ms to 2.8 seconds during a traffic burst. Investigate prefill saturation, queueing delay, batch formation and KV-cache allocation."),
    ("KV Cache Pressure", "KV-cache occupancy is increasing while active request count remains below the configured maximum. Analyze available tokens, evictable tokens, active blocks and admission control."),
    ("Scheduler Saturation", "Continuous batching receives long and short prompts simultaneously. Throughput increases but p99 TTFT deteriorates. Investigate scheduling fairness, prefill interference and queue depth."),
    ("GPU Memory Incident", "A Tesla T4 hosting an AWQ 7B model reports high memory allocation while idle. Separate model weights, CUDA graphs, runtime workspace and KV-cache allocation."),
    ("Gateway Overload", "A FastAPI gateway receives a sudden request burst. Standard requests must be shed at a KV-cache threshold while high-priority requests receive preferential admission."),
    ("Prefix Cache Efficiency", "Repeated requests share technical context. Investigate RadixAttention prefix matching, cached tokens, cache hit rate, eviction and KV-memory consumption."),
    ("Timeout Recovery", "SGLang experiences upstream timeouts during a sudden increase in concurrency. Investigate request deadlines, retry storms, cancellation and recovery behavior."),
    ("Throughput Latency Tradeoff", "Aggregate throughput increases during a concurrency sweep while p99 latency worsens. Identify the operating point balancing throughput, TTFT and ITL."),

    ("Prefill Bottleneck", "Long input prompts cause high prefill latency while decode latency remains stable. Investigate prompt processing, attention computation and prefill chunking."),
    ("Decode Bottleneck", "Generation throughput falls despite moderate prompt lengths. Investigate memory bandwidth, decode scheduling, batch size and KV-cache reads."),
    ("KV Eviction", "The serving engine repeatedly evicts cached prefixes during traffic bursts. Investigate eviction policy, reusable blocks and cache retention."),
    ("Request Queue Growth", "Gateway requests accumulate faster than SGLang can process them. Investigate queueing delay, concurrency limits and admission policies."),
    ("Batch Formation", "Continuous batching produces inconsistent batch sizes under mixed workloads. Investigate scheduling intervals, request arrival patterns and batch efficiency."),
    ("Long Context Requests", "Requests containing several thousand input tokens cause latency spikes. Investigate context length, prefill cost and KV-cache allocation."),
    ("Short Request Starvation", "Short generation requests experience excessive latency behind long-running requests. Investigate scheduling fairness and prefill-decode interference."),
    ("GPU Utilization Drop", "GPU utilization falls during a high request volume. Investigate CPU bottlenecks, synchronization, scheduler behavior and GPU idle periods."),

    ("Memory Fragmentation", "GPU memory availability becomes inconsistent after repeated inference workloads. Investigate allocation patterns, memory fragmentation and runtime workspace."),
    ("KV Capacity Planning", "A serving system must estimate safe concurrency from KV-cache capacity. Calculate memory per token and evaluate request-length distributions."),
    ("Priority Admission", "High-priority traffic must continue during overload while standard traffic is rejected. Evaluate priority handling and physical resource safety."),
    ("Fail Closed Admission", "Prometheus KV-cache metrics become unavailable during a traffic burst. Evaluate fail-closed admission, observability failures and high-priority behavior."),
    ("HTTP 503 Shedding", "A gateway returns HTTP 503 when KV-cache occupancy reaches its threshold. Investigate Retry-After behavior, request accounting and client retries."),
    ("Admission Race", "Many concurrent requests read identical KV occupancy before being forwarded to SGLang. Investigate admission races and capacity reservation."),
    ("Concurrency Limit", "An inference gateway uses an asynchronous semaphore to control in-flight requests. Evaluate semaphore sizing, queueing and engine saturation."),
    ("Backpressure Recovery", "A gateway begins rejecting standard traffic during overload. Define recovery conditions and safe restoration of normal admission."),

    ("FP16 Versus FP8", "A model is evaluated using FP16 and FP8 inference. Compare memory footprint, throughput, numerical accuracy and hardware support."),
    ("AWQ Quantization", "An AWQ 4-bit model reduces weight memory but experiences unexpected latency. Investigate quantized kernels, dequantization and memory bandwidth."),
    ("Quantization Accuracy", "A quantized model shows a small increase in perplexity. Evaluate accuracy trade-offs and benchmark methodology."),
    ("Attention Backend", "Changing the attention backend produces different inference latency. Investigate kernel selection, memory access and hardware compatibility."),
    ("CUDA Graph Overhead", "CUDA graph capture changes GPU memory allocation and startup latency. Investigate graph memory, capture behavior and steady-state performance."),
    ("GPU Memory Bandwidth", "A decode workload achieves low arithmetic intensity and high memory traffic. Analyze HBM bandwidth, weight reads and KV-cache access."),
    ("Roofline Analysis", "Measured inference throughput is below the theoretical compute limit. Apply roofline analysis and distinguish compute-bound from memory-bound execution."),
    ("GPU Occupancy", "A GPU kernel shows unexpectedly low occupancy. Investigate registers, shared memory, wave occupancy and kernel launch configuration."),

    ("Prefix Cache Cold Start", "The first request has high TTFT while repeated requests are faster. Measure cold and warm prefix behavior and cached-token counts."),
    ("Prefix Cache Mixed Traffic", "Requests share some prefixes but differ in later context. Investigate longest-prefix matching and the effect of unique prompt suffixes."),
    ("Cache Hit Rate", "The reported prefix-cache hit rate increases but end-to-end latency does not improve proportionally. Investigate prefill and decode contributions."),
    ("Cache Memory Pressure", "Reusable prefix entries consume KV-cache capacity during sustained traffic. Evaluate the balance between prefix reuse and active request capacity."),
    ("Prompt Length Variance", "A workload contains prompts ranging from 100 to 3000 tokens. Investigate padding, batching efficiency and KV allocation."),
    ("Token Pool Exhaustion", "SGLang approaches its token-pool capacity even though GPU memory appears available. Investigate token allocation and cache accounting."),
    ("Request Cancellation", "Clients disconnect during generation but engine work may continue. Investigate cancellation propagation and resource reclamation."),
    ("Streaming Latency", "Streaming clients report irregular token delivery. Investigate first-token latency, inter-token latency and buffering."),

    ("High Concurrency", "A 64-request burst causes increasing latency. Investigate scheduler saturation, KV-cache pressure and throughput scaling."),
    ("Burst Traffic", "Traffic arrives in sudden bursts rather than a steady arrival rate. Evaluate burst absorption, admission control and recovery."),
    ("Mixed Priority Traffic", "High- and standard-priority requests arrive simultaneously. Investigate fairness, priority bypass and standard-request shedding."),
    ("Request Length Prediction", "Prompt and output lengths vary substantially. Evaluate token-budget estimation and its usefulness for admission control."),
    ("Maximum Context Length", "Requests approach the configured context limit. Investigate memory requirements, truncation and request rejection."),
    ("Engine Health Failure", "SGLang health checks intermittently fail under load. Distinguish transient probe failures from genuine engine failure."),
    ("Metrics Scrape Failure", "Prometheus metrics become temporarily unavailable. Investigate observability reliability and safe gateway behavior."),
    ("Gateway Counter Audit", "Gateway counters show requests accepted and received but do not match completed responses. Investigate counter semantics and accounting."),

    ("TTFT Percentiles", "Average TTFT appears acceptable while p99 TTFT is high. Investigate latency distributions and queueing outliers."),
    ("ITL Regression", "Inter-token latency increases during concurrent generation. Investigate decode scheduling, memory bandwidth and batch composition."),
    ("Throughput Saturation", "Aggregate throughput stops increasing after a concurrency threshold. Identify the saturation point and resource bottleneck."),
    ("Load Shedding Policy", "A service must preserve capacity for critical requests. Evaluate threshold-based shedding and priority admission."),
    ("Retry Storm", "Clients retry every HTTP 503 immediately, worsening overload. Design bounded retries, exponential backoff and Retry-After handling."),
    ("Engine Restart Recovery", "The inference engine restarts and the gateway receives requests during recovery. Investigate readiness checks and traffic restoration."),
    ("Model Loading Memory", "Model loading temporarily consumes additional GPU memory. Investigate loading peaks, weight allocation and KV-cache initialization."),
    ("GPU Thermal Throttling", "Inference throughput declines during sustained GPU workloads. Investigate temperature, clock frequency and power limits."),

    ("Multi Tenant Isolation", "Multiple tenants share one inference engine with different priority levels. Investigate fairness, isolation and resource quotas."),
    ("SLA Enforcement", "A serving platform must maintain a TTFT target during variable traffic. Design admission controls and measurable SLA monitoring."),
    ("Autoscaling Signals", "A serving platform must decide when to add inference replicas. Evaluate queue depth, GPU utilization, latency and KV occupancy."),
    ("Production Benchmark Design", "A benchmark produces inconsistent latency measurements. Investigate warmup, concurrency, repetitions and measurement methodology."),
    ("Inference Capacity Planning", "Estimate safe serving capacity for a quantized 7B model on a 16 GiB GPU. Account for weights, runtime memory and KV cache."),
    ("Resource Monitoring", "GPU memory, GPU utilization and KV-cache occupancy show different trends. Explain what each metric measures and how to correlate them."),
    ("Graceful Degradation", "An inference service must remain useful during resource pressure. Design controlled shedding, priority handling and recovery behavior."),
    ("Production Incident Review", "A production LLM service experiences elevated latency, request failures and memory pressure. Build an evidence-driven root cause analysis and prevention plan."),
]


def load_tokenizer(model_name=MODEL_NAME):
    """Load the tokenizer used by the serving model."""

    return AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )


def build_prompt(request_id, tokenizer):
    """
    Build a prompt with exactly TARGET_PROMPT_TOKENS tokens.

    Token count is verified against the actual text sent to SGLang.
    Unique incident information appears at the beginning.
    """

    title, scenario = INCIDENT_SCENARIOS[
        request_id % len(INCIDENT_SCENARIOS)
    ]

    prompt = f"""
Incident ID: BP-{request_id + 1001}
Unique case identifier: CASE-{request_id:03d}-{(request_id * 37) % 997:03d}
Incident category: {title}

Scenario:
{scenario}

You are a senior production AI inference engineer investigating
a live GPU inference serving incident.

Environment:
- Model: Qwen2.5-7B-Instruct AWQ INT4.
- GPU: NVIDIA Tesla T4, approximately 16 GiB VRAM.
- Serving engine: SGLang.
- Context length: 4096 tokens.
- Gateway: asynchronous FastAPI.
- Request priorities: standard and high.
- KV-cache admission threshold: 90 percent.

Analyze the incident.

Explain the likely root causes and distinguish confirmed evidence
from hypotheses.

Analyze prefill computation, decode computation, continuous batching,
scheduler queueing, KV-cache allocation, prefix caching and GPU memory.

Explain how GPU utilization, memory bandwidth, available VRAM,
KV-cache occupancy and running request count should be interpreted.

Evaluate gateway admission decisions, priority handling,
HTTP 503 responses, Retry-After behavior and upstream timeouts.

Recommend immediate mitigation steps that do not require restarting
the inference engine.

Define a verification plan using measurable performance metrics.

Be technically precise. Do not invent measurements.
Explain operational trade-offs and failure modes.
"""

    def count_tokens(text):
        return len(
            tokenizer.encode(
                text,
                add_special_tokens=False,
            )
        )

    current_count = count_tokens(prompt)

    if current_count > TARGET_PROMPT_TOKENS:
        raise ValueError(
            f"Base prompt too long: {current_count} tokens. "
            f"Target: {TARGET_PROMPT_TOKENS}"
        )

    filler = (
        " Inference monitoring must correlate request arrival time, "
        "prompt processing, token generation, scheduler queueing, "
        "KV-cache allocation, GPU memory usage and completed responses."
    )

    # Add larger chunks while safely below the target.
    while TARGET_PROMPT_TOKENS - current_count > 32:
        candidate = prompt + filler
        candidate_count = count_tokens(candidate)

        if candidate_count > TARGET_PROMPT_TOKENS:
            break

        prompt = candidate
        current_count = candidate_count

    # Finish padding one token at a time.
    padding_units = [
        " x",
        " a",
        " the",
        " and",
        " monitoring",
        " GPU",
        " request",
    ]

    while current_count < TARGET_PROMPT_TOKENS:

        found = False

        for unit in padding_units:
            candidate = prompt + unit
            candidate_count = count_tokens(candidate)

            if candidate_count == TARGET_PROMPT_TOKENS:
                prompt = candidate
                current_count = candidate_count
                found = True
                break

            if candidate_count == current_count + 1:
                prompt = candidate
                current_count = candidate_count
                found = True
                break

        if not found:
            raise ValueError(
                f"Could not pad request {request_id} exactly. "
                f"Current: {current_count}, "
                f"Target: {TARGET_PROMPT_TOKENS}"
            )

    # Final verification.
    actual_count = count_tokens(prompt)

    if actual_count != TARGET_PROMPT_TOKENS:
        raise ValueError(
            f"Token verification failed for request {request_id}: "
            f"expected {TARGET_PROMPT_TOKENS}, got {actual_count}"
        )

    return prompt


def verify_prompts(tokenizer, count=64):
    """Verify all generated prompts before a benchmark."""

    print("\nBackpressure prompt verification")
    print("--------------------------------")

    for request_id in range(count):
        prompt = build_prompt(request_id, tokenizer)

        token_count = len(
            tokenizer.encode(
                prompt,
                add_special_tokens=False,
            )
        )

        if token_count != TARGET_PROMPT_TOKENS:
            raise AssertionError(
                f"Request {request_id}: invalid token count {token_count}"
            )

    print(f"Prompts verified: {count}")
    print(f"Target prompt tokens: {TARGET_PROMPT_TOKENS}")
    print(f"Allowed tolerance: {PROMPT_TOKEN_TOLERANCE}")
    print(f"Output tokens: {OUTPUT_TOKENS}")
    print("Result: PASS")


if __name__ == "__main__":
    tokenizer = load_tokenizer()
    verify_prompts(tokenizer)