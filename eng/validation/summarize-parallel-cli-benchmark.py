#!/usr/bin/env python3
"""Summarize completed run-parallel-case CLI timings without claiming quality.

Compute rates from the actual start-line token counts and logged milliseconds;
exclude the requested number of initial runs. Preserve every sample and refuse
incomplete runs, missing/duplicate samples, crashes or mutated runtime files.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics


def summarize(directory, discard=1):
    if discard < 0:
        raise ValueError("discard must be nonnegative")
    run_path, log_path = directory / "run.json", directory / "process.log"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    log = log_path.read_text(encoding="utf-8")
    failures = []
    if run.get("status") != "completed": failures.append("Run is not completed")
    if run.get("runtime_file_identity_check", {}).get("status") != "passed":
        failures.append("Runtime identity did not pass")
    shutdown = run.get("shutdown", {})
    if not shutdown.get("started") or shutdown.get("forced") or shutdown.get("exit_code") != 0:
        failures.append("Benchmark process did not exit cleanly")
    starts = re.findall(r"benchmark starting: prefillTokens=(\d+) decodeTokens=(\d+) runs=(\d+).*?decodeMode=([^\s]+)", log)
    samples, configuration = [], {}
    if len(starts) != 1:
        failures.append("Need exactly one benchmark start with actual token/run counts")
    else:
        prefill, decode, runs = (int(x) for x in starts[0][:3])
        mode = starts[0][3]
        configuration = dict(prefill_tokens=prefill, decode_tokens=decode, runs=runs,
                             mode=mode, discard_runs=discard)
        for item in re.finditer(r"benchmark run (\d+)/(\d+): prefillMs=([0-9.]+).*?decodeMs=([0-9.]+).*?decodeMode=([^\s]+)", log):
            index, total, prefill_ms, decode_ms, actual_mode = item.groups()
            p, d = float(prefill_ms), float(decode_ms)
            row = dict(run=int(index), runs=int(total), prefill_ms=p, decode_ms=d, mode=actual_mode)
            if p <= 0 or d <= 0 or actual_mode != mode or int(total) != runs:
                failures.append("Invalid timing/mode/run count for sample " + index)
            else:
                row.update(prefill_tokens_per_second=prefill * 1000 / p,
                           decode_tokens_per_second=decode * 1000 / d)
            samples.append(row)
        if [sample["run"] for sample in samples] != list(range(1, runs + 1)):
            failures.append("Benchmark samples must contain every run once in order")
        if not prefill or not decode or discard >= runs:
            failures.append("Need positive token counts and at least one measured run after discard")
    report = dict(status="failed" if failures else "measured", failures=failures,
        run=str(run_path), run_sha256=hashlib.sha256(run_path.read_bytes()).hexdigest(),
        process_log_sha256=hashlib.sha256(log_path.read_bytes()).hexdigest(),
        configuration=configuration, command=run.get("command"), environment=run.get("environment"),
        shutdown=shutdown, runtime_file_identity_check=run.get("runtime_file_identity_check"), samples=samples,
        limitations=["Short CLI timing windows measure the emitted decode mode, not trained-model answer quality.",
            "Initial-run exclusion does not establish steady state or controlled cold/warm page residency.",
            "Rates use logged milliseconds; original rounded CLI rates remain in process.log.",
            "Different CPU expert placement/context/precision changes the workload and cannot establish topology-only scaling."])
    if not failures:
        measured = samples[discard:]
        report["measured_runs"] = [row["run"] for row in measured]
        report["summary"] = {"median_" + field: statistics.median(row[field] for row in measured)
            for field in ("prefill_tokens_per_second", "decode_tokens_per_second", "prefill_ms", "decode_ms")}
        report["summary"]["prefill_max_min_ratio"] = max(row["prefill_ms"] for row in measured) / min(row["prefill_ms"] for row in measured)
        report["summary"]["decode_max_min_ratio"] = max(row["decode_ms"] for row in measured) / min(row["decode_ms"] for row in measured)
    token_control = re.findall(r"benchmark sampled tokens \(untimed correctness\): prefillTopToken=(\d+) decode=([0-9,]+)", log)
    if len(token_control) == 1:
        report["untimed_greedy_control"] = dict(prefill_top_token=int(token_control[0][0]),
            decode_token_ids=[int(value) for value in token_control[0][1].split(",")])
    effective = directory / "effective-env.json"
    if effective.exists(): report["effective_environment_observation"] = json.loads(effective.read_text(encoding="utf-8"))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--discard-runs", type=int, default=1)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists(): parser.error("Output must be fresh; preserve earlier evidence")
    report = summarize(args.directory, args.discard_runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "failures", "configuration", "summary") if key in report}, indent=2))
    return int(report["status"] != "measured")


if __name__ == "__main__": raise SystemExit(main())
