#!/usr/bin/env python3
# Copyright (c) Zhongkai Fu. All rights reserved.
# https://github.com/zhongkaifu/TensorSharp
#
# This file is part of TensorSharp.
#
# TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
"""Interleaved A/B of two CpuDecodeAttentionBench builds.

    python ab.py <baseline bench dir> <candidate bench dir> <out dir> [--rounds N] [-- bench args...]

Runs the two builds alternately in ABBA order (so a slow drift - laptop clocks, thermal state -
lands on both sides equally), one fresh process per run, and writes every run's raw lines plus a
summary: per shape, the median over runs of each run's median per-call latency, the
candidate/baseline ratio, the spread of the per-pair ratios, and whether the two builds' outputs
hashed identically. Build the baseline with -p:TensorSharpRoot=<baseline tree>/.
"""
import json
import os
import statistics
import subprocess
import sys


def run(bench_dir, label, extra):
    dll = os.path.join(bench_dir, "CpuDecodeAttentionBench.dll")
    out = subprocess.run(["dotnet", dll, "--label", label] + extra, capture_output=True, text=True, check=True)
    rows = []
    for line in out.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) < 3 or parts[0] != label:
            continue
        row = {"label": parts[0], "shape": parts[1]}
        for kv in parts[2:]:
            k, v = kv.split("=", 1)
            row[k] = v if k == "hash" else float(v)
        rows.append(row)
    return rows


def main():
    args = sys.argv[1:]
    extra = []
    if "--" in args:
        i = args.index("--")
        args, extra = args[:i], args[i + 1:]
    rounds = 10
    if "--rounds" in args:
        i = args.index("--rounds")
        rounds = int(args[i + 1])
        del args[i:i + 2]
    base_dir, cand_dir, out_dir = args
    os.makedirs(out_dir, exist_ok=True)

    order = []
    for r in range(rounds):
        order += [("baseline", base_dir), ("candidate", cand_dir)] if r % 2 == 0 else [("candidate", cand_dir), ("baseline", base_dir)]

    runs = []
    with open(os.path.join(out_dir, "raw.tsv"), "w") as raw:
        for n, (label, d) in enumerate(order):
            rows = run(d, label, extra)
            for row in rows:
                row["run"] = n
                raw.write(json.dumps(row) + "\n")
            runs.append((label, rows))
            print(f"run {n:2d} {label:9s} " + "  ".join(f"{r['shape']}={r['median_us']:.1f}" for r in rows), flush=True)

    shapes = [r["shape"] for r in runs[0][1]]
    summary = []
    for s in shapes:
        med = {"baseline": [], "candidate": []}
        hashes = {"baseline": set(), "candidate": set()}
        for label, rows in runs:
            for r in rows:
                if r["shape"] == s:
                    med[label].append(r["median_us"])
                    hashes[label].add(r["hash"])
        # Pair each candidate run with the baseline run of the same round.
        ratios = [c / b for b, c in zip(med["baseline"], med["candidate"])]
        b, c = statistics.median(med["baseline"]), statistics.median(med["candidate"])
        summary.append({
            "shape": s,
            "baseline_median_us": round(b, 2),
            "candidate_median_us": round(c, 2),
            "ratio": round(c / b, 4),
            "pair_ratio_median": round(statistics.median(ratios), 4),
            "pair_ratio_min": round(min(ratios), 4),
            "pair_ratio_max": round(max(ratios), 4),
            "runs_per_side": len(ratios),
            "outputs_identical": len(hashes["baseline"]) == 1 and hashes["baseline"] == hashes["candidate"],
        })
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print()
    print(f"{'shape':24s} {'base us':>10s} {'cand us':>10s} {'ratio':>7s} {'pair med':>8s} {'pair min':>8s} {'pair max':>8s}  same output")
    for r in summary:
        print(f"{r['shape']:24s} {r['baseline_median_us']:10.1f} {r['candidate_median_us']:10.1f} {r['ratio']:7.3f} "
              f"{r['pair_ratio_median']:8.3f} {r['pair_ratio_min']:8.3f} {r['pair_ratio_max']:8.3f}  {r['outputs_identical']}")


if __name__ == "__main__":
    main()
