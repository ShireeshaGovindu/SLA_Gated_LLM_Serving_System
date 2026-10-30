#!/usr/bin/env bash
set -euo pipefail

cd /content/LLM_Serving_System

exec python -m uvicorn gateway:app \
    --app-dir /content/LLM_Serving_System \
    --host 127.0.0.1 \
    --port 8081