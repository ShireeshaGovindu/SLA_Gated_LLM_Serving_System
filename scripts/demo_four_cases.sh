#!/usr/bin/env bash
# Four-case backpressure check against a running load (run on the T4 after the
# engine is started with MEM_FRACTION=0.60 and the gateway with ENGINE_TIMEOUT_SECONDS=900).
#   1. Tier 1 while KV <= 90%   -> admitted
#   2. Tier 2 while KV <= 90%   -> admitted
#   3. Tier 1 while KV  > 90%   -> admitted (never shed)
#   4. Tier 2 while KV  > 90%   -> HTTP 503
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
OUT="${OUT:-results/fourcase_060}"
G="${GATEWAY_URL:-http://127.0.0.1:8081}"
mkdir -p "$OUT"
usage() { curl -s "$G/metrics" | python3 -c "import sys,json;print(json.load(sys.stdin)['kv_token_usage_percent'])"; }
probe() {  # tier label extra-args
  python module_2/demo_backpressure.py --mode priority-batch --priority-batch "$1" --requests 8 --max-tokens 16 \
    --timeout 850 --output-dir "$OUT/$2" "${@:3}" > "$OUT/$2.txt" 2>&1
}
stamp() { echo "$(date +%H:%M:%S) $*" | tee -a "$OUT/sequence.log"; }

stamp "starting load: 32 requests in flight, 300-1500 tokens each"
python module_2/demo_backpressure.py --mode load --requests 32 --max-tokens 1500 --max-tokens-min 300 \
  --duration 1500 --output-dir "$OUT/load" > "$OUT/load.txt" 2>&1 &
LOAD_PID=$!

# Cases 1 and 2: wait until usage is clearly rising but still below 90%
while true; do u=$(usage); python3 -c "import sys;sys.exit(0 if 45<=float('$u')<=85 else 1)" && break; sleep 2; done
stamp "usage=$u% (below 90%): sending case 1 (tier1) and case 2 (tier2)"
probe tier1 case1_tier1_below &
P1=$!
probe tier2 case2_tier2_below &
P2=$!
wait $P2
stamp "case 2 finished"

# Cases 3 and 4: wait for usage above 90%
while true; do u=$(usage); python3 -c "import sys;sys.exit(0 if float('$u')>90 else 1)" && break; sleep 2; done
stamp "usage=$u% (above 90%): sending case 3 (tier1) and case 4 (tier2)"
probe tier1 case3_tier1_above &
P3=$!
probe tier2 case4_tier2_above &
P4=$!
wait $P4
stamp "case 4 finished"
wait $P1 $P3
stamp "tier1 probes finished"
kill -TERM $LOAD_PID 2>/dev/null
stamp "load stopped"
