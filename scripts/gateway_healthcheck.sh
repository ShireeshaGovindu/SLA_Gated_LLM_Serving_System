#!/bin/bash

BASE_URL="http://127.0.0.1:8081"
FAILED=0

echo "========================================"
echo "       GATEWAY HEALTH CHECK"
echo "========================================"

# 1. Health endpoint
HEALTH=$(curl -sf --max-time 10 "$BASE_URL/health")
if [ $? -eq 0 ]; then
    echo "[PASS] /health"
    echo "$HEALTH" | python3 -c "import sys,json; d=json.load(sys.stdin); print('       Status:',d['status']); print('       Engine:',d['engine_url'])"
else
    echo "[FAIL] /health"
    FAILED=1
fi

# 2. Gateway metrics
METRICS=$(curl -sf --max-time 10 "$BASE_URL/metrics")
if [ $? -eq 0 ]; then
    echo "[PASS] /metrics"
    echo "$METRICS" | python3 -c "import sys,json; d=json.load(sys.stdin); print('       Requests:',d['requests_received']); print('       Accepted:',d['requests_accepted']); print('       Shed:',d['requests_shed']); print('       KV Occupancy: %.2f%%' % d['kv_occupancy_percent']); print('       KV Monitoring:',d['kv_metrics_available'])"
else
    echo "[FAIL] /metrics"
    FAILED=1
fi

# 3. Engine metrics
ENGINE=$(curl -sf --max-time 10 "$BASE_URL/engine-metrics")
if [ $? -eq 0 ]; then
    echo "[PASS] /engine-metrics"
    echo "$ENGINE" | python3 -c "import sys,json; d=json.load(sys.stdin); print('       Prometheus metrics available:',bool(d.get('prometheus_text'))); print('       Metric lines:',len(d.get('prometheus_text','').splitlines()))"
else
    echo "[FAIL] /engine-metrics"
    FAILED=1
fi

echo "========================================"

if [ "$FAILED" -eq 0 ]; then
    echo "GATEWAY HEALTH CHECK: PASSED"
else
    echo "GATEWAY HEALTH CHECK: FAILED"
fi

echo "========================================"

exit "$FAILED"
