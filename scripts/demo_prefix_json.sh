#!/usr/bin/env bash
# Demo checks for prefix caching (RadixAttention) and the JSON-schema route.
set -uo pipefail
G="${GATEWAY_URL:-http://127.0.0.1:8081}"
E="${SGLANG_URL:-http://127.0.0.1:30000}"
H='content-type: application/json'

gen() {  # id prompt profile
  curl -s "$G/generate" -H "$H" -d "{\"request_id\":\"$1\",\"prompt\":\"$2\",\"prefix_profile\":\"$3\",\"domain\":\"KV-cache optimization\",\"max_tokens\":1,\"temperature\":0}" |
  python3 -c "import sys,json;d=json.load(sys.stdin);m=d['meta_info'];print(f\"  {d['request_id']:<8} prompt_tokens={m['prompt_tokens']:<5} cached_tokens={m['cached_tokens']:<5} latency={d['gateway']['elapsed_seconds']:.3f}s\")"
}

echo "== 1. PREFIX CACHING (RadixAttention): shared 1,500-token system prefix =="
curl -s -X POST "$E/flush_cache" -o /dev/null -w "cache flushed: HTTP %{http_code}\n"
echo "-- cold: first request, prefix not in cache"
gen cold-1 "What is prefix caching?" shared_1500
echo "-- warm: same prefix, different questions (prefix reused)"
gen warm-2 "What is continuous batching?" shared_1500
gen warm-3 "Why does TTFT drop on warm prompts?" shared_1500
echo "-- control: no shared prefix (short unique prompt)"
gen noprefix "Explain RadixAttention in one sentence." none

echo
echo "== 2. JSON SCHEMA CONSTRAINT (/generate-json) =="
SCHEMA='{"type":"object","properties":{"request_id":{"type":"string"},"domain":{"type":"string"},"summary":{"type":"string"},"score":{"type":"integer"},"risks":{"type":"array","items":{"type":"string"}}},"required":["request_id","domain","summary","score","risks"],"additionalProperties":false}'
echo "-- valid request: engine-side constrained decoding + gateway validation"
curl -s "$G/generate-json" -H "$H" -d "{\"request_id\":\"json-1\",\"prompt\":\"Assess GPU memory risks for an inference server. Return JSON.\",\"max_tokens\":200,\"temperature\":0,\"schema\":$SCHEMA}" |
python3 -c "import sys,json;d=json.load(sys.stdin);print('  HTTP ok; schema_validation =',d['schema_validation']);print('  parsed_json =',json.dumps(d['parsed_json'])[:300])"
echo "-- missing schema -> expect HTTP 400"
curl -s -o /dev/null -w "  HTTP %{http_code}\n" "$G/generate-json" -H "$H" -d '{"prompt":"x"}'
echo "-- invalid priority value -> expect HTTP 422"
curl -s -o /dev/null -w "  HTTP %{http_code}\n" "$G/generate" -H "$H" -d '{"prompt":"x","priority":"urgent"}'
