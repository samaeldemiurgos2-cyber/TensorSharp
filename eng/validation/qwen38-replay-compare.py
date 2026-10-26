#!/usr/bin/env python3
"""Compare two qwen38-agent-replay.py result files (reference vs candidate).

Prints, per (scenario, concurrency), the summed end-to-end milliseconds of the
requests both runs completed and the candidate/reference ratio; --steps adds a
per-step table (mean TTFT, mean e2e and mean cached tokens of each side). Only
requests present in both files are summed, so a partial candidate run compares
like with like. Evidence belongs in ignored artifacts/ or docs/validation/.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path


def load(path: Path):
    data = json.loads(path.read_text())
    rows = data.get("results") or data.get("requests") or []
    by_key = {}
    for row in rows:
        concurrency = row.get("concurrency", data.get("concurrency", 1))
        by_key[(row["scenario"], concurrency, row["lane"], row["step"])] = row
    return by_key


def mean(rows, field):
    values = [r.get(field) for r in rows if r.get(field) is not None]
    return statistics.mean(values) if values else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--scenario", help="only scenarios whose name contains this text")
    parser.add_argument("--steps", action="store_true", help="also print the per-step table")
    args = parser.parse_args()

    reference, candidate = load(args.reference), load(args.candidate)
    shared = sorted(k for k in reference.keys() & candidate.keys()
                    if args.scenario is None or args.scenario in k[0])

    totals = collections.defaultdict(lambda: [0.0, 0.0, 0])
    steps = collections.defaultdict(lambda: ([], []))
    for key in shared:
        ref, cand = reference[key], candidate[key]
        group = (key[0], key[1])
        totals[group][0] += ref["e2e_ms"]
        totals[group][1] += cand["e2e_ms"]
        totals[group][2] += 1
        steps[(key[0], key[1], key[3])][0].append(ref)
        steps[(key[0], key[1], key[3])][1].append(cand)

    if args.steps:
        print("%-14s %2s %3s | %8s %8s | %8s %8s | %7s %7s | %s" % (
            "scenario", "c", "st", "ref_ttft", "can_ttft", "ref_e2e", "can_e2e", "ref_hit", "can_hit", "ratio"))
        for (scenario, concurrency, step), (refs, cands) in sorted(steps.items()):
            ref_e2e, can_e2e = mean(refs, "e2e_ms"), mean(cands, "e2e_ms")
            print("%-14s %2d %3d | %8.0f %8.0f | %8.0f %8.0f | %7.0f %7.0f | %.2f" % (
                scenario, concurrency, step, mean(refs, "ttft_ms"), mean(cands, "ttft_ms"),
                ref_e2e, can_e2e, mean(refs, "cached_tokens"), mean(cands, "cached_tokens"),
                can_e2e / ref_e2e if ref_e2e > 0 else float("nan")))
        print()

    for (scenario, concurrency), (ref_sum, can_sum, count) in sorted(totals.items()):
        print("%-14s c=%d  n=%-3d reference %8.0f ms  candidate %8.0f ms  ratio %.2f" % (
            scenario, concurrency, count, ref_sum, can_sum, can_sum / ref_sum if ref_sum else float("nan")))


if __name__ == "__main__":
    main()
