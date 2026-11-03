#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

GATEWAY_URL="${GATEWAY_URL:-http://127.0.0.1:8081}"
ENGINE_URL="${SGLANG_URL:-http://127.0.0.1:30000}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/results}"

mkdir -p "$OUTPUT_DIR"

python benchmark.py \
    --gateway "$GATEWAY_URL" \
    --engine-url "$ENGINE_URL" \
    --output-dir "$OUTPUT_DIR" \
    --run-warm-cold
