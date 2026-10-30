#!/bin/bash

BASE_URL="http://127.0.0.1:30000"
FAILED=0

echo "========================================"
echo "       SGLANG ENGINE HEALTH CHECK"
echo "========================================"

# 1. SGLang health endpoint
HEALTH=$(curl -sf --max-time 10 "$BASE_URL/health")
if [ $? -eq 0 ]; then
    echo "[PASS] /health"
    echo "       Engine is healthy"
else
    echo "[FAIL] /health"
    FAILED=1
fi

# 2. Model availability endpoint
MODELS=$(curl -sf --max-time 10 "$BASE_URL/v1/models")
if [ $? -eq 0 ]; then
    echo "[PASS] /v1/models"
    echo "$MODELS" | python3 -c "
import sys, json
d = json.load(sys.stdin)
models = d.get('data', [])
print('       Models loaded:', len(models))
for model in models:
    print('       Model ID:', model.get('id', 'unknown'))
"
    if [ $? -ne 0 ]; then
        echo "       WARNING: Could not parse model response"
    fi
else
    echo "[FAIL] /v1/models"
    FAILED=1
fi

echo "========================================"

if [ "$FAILED" -eq 0 ]; then
    echo "SGLANG HEALTH CHECK: PASSED"
else
    echo "SGLANG HEALTH CHECK: FAILED"
fi

echo "========================================"

exit "$FAILED"
