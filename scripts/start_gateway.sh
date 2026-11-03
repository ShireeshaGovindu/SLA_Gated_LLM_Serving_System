#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

exec python -m uvicorn gateway:app \
    --app-dir "$ROOT" \
    --host 127.0.0.1 \
    --port "${GATEWAY_PORT:-8081}"
