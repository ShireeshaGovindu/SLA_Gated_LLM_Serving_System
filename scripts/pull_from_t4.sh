#!/usr/bin/env bash
# Run on the Mac: copy results, logs and any on-T4 edits back into this folder.
set -uo pipefail
HOST="${T4_HOST:-wd-t4}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p results t4_logs

if ! ssh -n -o ConnectTimeout=40 "$HOST" 'test -d /content/LLM_Serving_System' 2>/dev/null; then
  echo "T4 runtime unreachable or /content/LLM_Serving_System is gone (runtime recycled?). Nothing pulled; local copies untouched."
  exit 1
fi

ssh -n "$HOST" 'cd /content/LLM_Serving_System && tar czf - results* 2>/dev/null' | tar xzf - 2>/dev/null
ssh -n "$HOST" 'cd /content && tar czf - *.log *.txt 2>/dev/null' | tar xzf - -C t4_logs 2>/dev/null
# code possibly edited on the T4: report real differences only, never overwrite
for f in gateway.py benchmark.py module_2/demo_backpressure.py scripts/start_engine.sh startup.sh; do
  remote="$(ssh -n "$HOST" "cat /content/LLM_Serving_System/$f" 2>/dev/null)"
  if [ -z "$remote" ]; then echo "could not read $f on T4 (skipped)"; continue; fi
  if [ "$remote" != "$(cat "$f")" ]; then echo "DIFFERS between T4 and Mac: $f"; fi
done
echo "pulled. local: $(ls results | wc -l | tr -d ' ') items in results/, $(ls t4_logs | wc -l | tr -d ' ') logs in t4_logs/"
