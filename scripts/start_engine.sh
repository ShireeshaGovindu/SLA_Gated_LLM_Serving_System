#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct-AWQ}"

# Optional: cap the KV pool (tokens) so the backpressure demo can cross 90%
# usage with ~32 requests. Leave unset for the benchmarks.
EXTRA=()
if [[ -n "${MAX_TOTAL_TOKENS:-}" ]]; then
    EXTRA+=(--max-total-tokens "$MAX_TOTAL_TOKENS")
fi

exec python -m sglang.launch_server \
    --model-path "$MODEL" \
    --quantization awq \
    --dtype half \
    --attention-backend triton \
    --sampling-backend pytorch \
    --mem-fraction-static "${MEM_FRACTION:-0.80}" \
    --context-length 4096 \
    --max-running-requests "${MAX_RUNNING_REQUESTS:-32}" \
    --enable-metrics \
    ${EXTRA[@]+"${EXTRA[@]}"} \
    --host 127.0.0.1 --port "${ENGINE_PORT:-30000}"
