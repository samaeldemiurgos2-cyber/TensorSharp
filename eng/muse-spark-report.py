#!/usr/bin/env python3
"""Build a TensorSharp vs llama.cpp Muse Spark comparison from benchmark JSON.

Inputs are the JSON files written by eng/muse-spark-bench.py (performance) and,
optionally, eng/muse-spark-validate.py (quality fixtures) for each engine:

  python3 eng/muse-spark-report.py \
      --ts-perf ts.json --llama-perf llama.json \
      [--ts-quality ts-validate.json --llama-quality llama-validate.json] \
      [--extra-perf "label=ts.json;llama.json" ...] \
      --title "6x A40, UD-Q3_K_XL" --out report.md

Every number is a median over the scenario's repeats. A ratio above 1.00 means
TensorSharp is better (higher throughput, or lower latency).
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8")) if path else None


def med(values):
    values = [v for v in values if isinstance(v, (int, float))]
    return statistics.median(values) if values else None


def fmt(v, digits=2):
    return "—" if v is None else f"{v:.{digits}f}"


def ratio(ts, ll, higher_is_better=True):
    if ts is None or ll is None or ts == 0 or ll == 0:
        return "—"
    r = ts / ll if higher_is_better else ll / ts
    return f"**{r:.2f}x**" if r >= 1 else f"{r:.2f}x"


def metric_rows(ts, ll):
    """(scenario, metric, ts, llama, higher_is_better, note)"""
    rows = []
    tr, lr = ts["results"], ll["results"]

    def runs(name):
        return tr.get(name), lr.get(name)

    a, b = runs("short")
    if a and b:
        label = (f"Short prompt (~{med([x['prompt_tokens'] for x in a]):.0f} tok), "
                 f"{med([x['completion_tokens'] for x in a]):.0f} generated")
        rows.append((label, "decode tok/s",
                     med([x["decode_tok_s"] for x in a]), med([x["decode_tok_s"] for x in b]), True, ""))
        rows.append((label, "TTFT s", med([x["ttft_s"] for x in a]), med([x["ttft_s"] for x in b]), False, ""))
    for name in ("long_4k", "long_16k", "long_32k"):
        a, b = runs(name)
        if a and b:
            label = f"Long prompt ({med([x['prompt_tokens'] for x in a]):.0f} tok)"
            rows.append((label, "prefill tok/s", med([x.get("prefill_tok_s") for x in a]),
                         med([x.get("prefill_tok_s") for x in b]), True, ""))
            rows.append((label, "TTFT s", med([x["ttft_s"] for x in a]), med([x["ttft_s"] for x in b]), False, ""))
            ok_a = sum(1 for x in a if x.get("needle_ok")); ok_b = sum(1 for x in b if x.get("needle_ok"))
            rows.append((label, "needle found", f"{ok_a}/{len(a)}", f"{ok_b}/{len(b)}", None, ""))
    for name in sorted(k for k in tr if k.startswith("parallel_")):
        a, b = runs(name)
        if a and b:
            n = name.split("_")[1]
            rows.append((f"{n} concurrent requests", "aggregate tok/s", med([x["aggregate_tok_s"] for x in a]),
                         med([x["aggregate_tok_s"] for x in b]), True, ""))
            rows.append((f"{n} concurrent requests", "per-request decode tok/s",
                         med([v for x in a for v in x["per_request_decode_tok_s"]]),
                         med([v for x in b for v in x["per_request_decode_tok_s"]]), True, ""))
            rows.append((f"{n} concurrent requests", "TTFT s", med([v for x in a for v in x["ttft_s"]]),
                         med([v for x in b for v in x["ttft_s"]]), False, ""))
    a, b = runs("cache_reuse")
    if a and b:
        rows.append(("Multi-turn KV reuse (2.3K prefix)", "turn-1 TTFT s", med([x["turn1"]["ttft_s"] for x in a]),
                     med([x["turn1"]["ttft_s"] for x in b]), False, ""))
        rows.append(("Multi-turn KV reuse (2.3K prefix)", "turn-2 TTFT s", med([x["turn2"]["ttft_s"] for x in a]),
                     med([x["turn2"]["ttft_s"] for x in b]), False, ""))
        rows.append(("Multi-turn KV reuse (2.3K prefix)", "turn-2 cached tokens",
                     med([x["turn2"]["cached_tokens"] for x in a]), med([x["turn2"]["cached_tokens"] for x in b]), None, ""))
    a, b = runs("shared_prefix")
    if a and b:
        rows.append(("New conversations sharing a 2.3K system prompt", "first TTFT s",
                     med([x["turns"][0]["ttft_s"] for x in a]), med([x["turns"][0]["ttft_s"] for x in b]), False, ""))
        rows.append(("New conversations sharing a 2.3K system prompt", "later TTFT s",
                     med([t["ttft_s"] for x in a for t in x["turns"][1:]]),
                     med([t["ttft_s"] for x in b for t in x["turns"][1:]]), False, ""))
    return rows


def perf_table(ts, ll):
    out = ["| Scenario | Metric | TensorSharp | llama.cpp | TensorSharp advantage |", "|---|---|---:|---:|---:|"]
    for scen, metric, a, b, hib, note in metric_rows(ts, ll):
        if hib is None:
            out.append(f"| {scen} | {metric} | {a if isinstance(a, str) else fmt(a, 0)} | {b if isinstance(b, str) else fmt(b, 0)} | |")
        else:
            out.append(f"| {scen} | {metric} | {fmt(a)} | {fmt(b)} | {ratio(a, b, hib)} |")
    return "\n".join(out)


def quality_table(tq, lq):
    def by_name(rep):
        res = {}
        for c in rep.get("cases", []):
            name = c.get("name") or c.get("scenario") or c.get("tag", "").rsplit("-r", 1)[0]
            res.setdefault(name, []).append(c)
        return res
    ta, la = by_name(tq), by_name(lq) if lq else {}
    names = sorted(set(ta) | set(la))
    out = ["| Scenario | TensorSharp | TensorSharp wall ms | llama.cpp | llama.cpp wall ms |", "|---|---|---:|---|---:|"]
    for n in names:
        def cell(cases):
            if not cases: return ("—", "—")
            ok = sum(1 for c in cases if c.get("status") == "ok")
            return (f"{ok}/{len(cases)} ok", fmt(med([c.get("total_wall_ms") for c in cases]), 0))
        a, b = cell(ta.get(n)), cell(la.get(n))
        out.append(f"| {n} | {a[0]} | {a[1]} | {b[0]} | {b[1]} |")
    return "\n".join(out)


def load_merged(paths):
    """Load one or more bench JSON files; later files replace earlier scenarios."""
    merged = None
    for path in paths.split(","):
        data = load(path)
        if merged is None:
            merged = data
        else:
            merged["results"].update(data["results"])
    return merged


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ts-perf", required=True, help="bench JSON; a comma list merges scenarios (later wins)")
    ap.add_argument("--llama-perf", required=True, help="bench JSON; a comma list merges scenarios (later wins)")
    ap.add_argument("--ts-quality")
    ap.add_argument("--llama-quality")
    ap.add_argument("--extra-perf", action="append", default=[], help="label=ts.json;llama.json")
    ap.add_argument("--title", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ts, ll = load_merged(args.ts_perf), load_merged(args.llama_perf)
    parts = [f"# Muse Spark: TensorSharp vs llama.cpp {args.title}".rstrip(), "",
             "Medians over each scenario's repeats; a ratio above 1.00x means TensorSharp is better.", "",
             "## Main profile", "", perf_table(ts, ll), ""]
    for spec in args.extra_perf:
        label, files = spec.split("=", 1)
        a, b = files.split(";", 1)
        parts += [f"## {label}", "", perf_table(load_merged(a), load_merged(b)), ""]
    if args.ts_quality:
        parts += ["## Quality fixtures (eng/muse-spark-validate.py)", "", quality_table(load(args.ts_quality), load(args.llama_quality)), ""]
    Path(args.out).write_text("\n".join(parts), encoding="utf-8")
    print("\n".join(parts))


if __name__ == "__main__":
    main()
