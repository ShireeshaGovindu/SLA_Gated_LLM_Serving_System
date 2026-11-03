#!/usr/bin/env python3
"""Print RESULTS.md tables from results/*.json (no hand-typed numbers)."""
import json
import sys
from pathlib import Path

R = Path(sys.argv[1] if len(sys.argv) > 1 else "results")


def load(name):
    p = R / name
    return json.loads(p.read_text()) if p.exists() else None


def q(hist):
    return ((hist or {}).get("approx_quantiles_seconds")) or {}


def f(v, nd=3):
    return "n/a" if v is None else f"{v:.{nd}f}"


print("## Latency decomposition (seconds; TTFT/ITL from SGLang histogram deltas)\n")
print("| Scenario | ok/total | TTFT p50 | TTFT p95 | TTFT p99 | ITL p50 | ITL p95 | ITL p99 | E2E p50 | E2E p95 | E2E p99 | wall s | out tok/s |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
names = {"a": "A cold (50)", "b": "B warm (100)", "c": "C JSON (50)"}
for k, n in names.items():
    d = load(f"scenario_{k}_results.json")
    if not d:
        continue
    s = d["summary"]
    t, i, e = q(s.get("ttft_histogram_delta")), q(s.get("itl_histogram_delta")), s["request_latency_seconds"]
    print(f"| {n} | {s['success_count']}/{s['request_count']} | {f(t.get('p50'),2)} | {f(t.get('p95'),2)} | {f(t.get('p99'),2)} | "
          f"{f(i.get('p50'))} | {f(i.get('p95'))} | {f(i.get('p99'))} | {f(e['p50'],1)} | {f(e['p95'],1)} | {f(e['p99'],1)} | "
          f"{f(s['wall_seconds'],1)} | {f(s['output_tokens_per_second'],1)} |")

print("\n## Cache efficiency per scenario\n")
print("| Scenario | mean prompt tok | mean cached tok | cached % | cache_hit_rate gauge before -> after |")
print("|---|---|---|---|---|")
for k, n in names.items():
    d = load(f"scenario_{k}_results.json")
    if not d:
        continue
    s = d["summary"]
    cm = s.get("cache_metrics") or {}
    pct = 100 * s["mean_cached_tokens"] / s["mean_prompt_tokens"] if s.get("mean_prompt_tokens") else None
    print(f"| {n} | {f(s['mean_prompt_tokens'],1)} | {f(s['mean_cached_tokens'],1)} | {f(pct,1)} | "
          f"{f(cm.get('cache_hit_rate_before'),3)} -> {f(cm.get('cache_hit_rate_after'),3)} |")

c = load("scenario_c_results.json")
if c:
    s = c["summary"]
    print(f"\nScenario C JSON validity: {s['json_valid_count']} valid / {s['json_invalid_count']} invalid.")

p = load("prefix_warm_cold_probe.json")
if p:
    print("\n## Warm vs cold TTFT (cache flushed before each cold request, max_tokens=1)\n")
    print("| | n | mean s | p50 s | p95 s | mean cached tokens |")
    print("|---|---|---|---|---|---|")
    for label, key, ck in (("cold", "cold_ttft_seconds", "mean_cached_tokens_cold"), ("warm", "warm_ttft_seconds", "mean_cached_tokens_warm")):
        x = p.get(key) or {}
        print(f"| {label} | {x.get('count')} | {f(x.get('mean'))} | {f(x.get('p50'))} | {f(x.get('p95'))} | {p.get(ck)} |")
    print(f"\nMean TTFT reduction (warm vs cold): {p.get('mean_ttft_reduction_percent')}%")
