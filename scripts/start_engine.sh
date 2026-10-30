#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct-AWQ}"

exec python -m sglang.launch_server \
    --model-path "$MODEL" \
    --quantization awq \
    --dtype half \
    --attention-backend triton \
    --sampling-backend pytorch \
    --mem-fraction-static 0.80 \
    --context-length 4096 \
    --max-running-requests 32 \
    --enable-metrics \
    --port 30000