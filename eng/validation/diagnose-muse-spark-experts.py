#!/usr/bin/env python3
"""Bounded CPU/streamed/resident GPU expert diagnostics, with no parity verdict.

Uses the CPU microbenchmark's checked GGUF mappings and exact native ABI. Saves
actual inputs, routes and outputs for a subsequent independent oracle. Run with
no other GPU jobs. Default is one real layer, 256 rows, 32 CPU threads, and two
calls per path. Timing includes first-use allocations and is diagnostic only.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time

SPEC = importlib.util.spec_from_file_location("spark_expert_bench", Path(__file__).with_name("benchmark-muse-spark-cpu-experts.py"))
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


def compare(reference, actual, np):
    left, right = reference.astype(np.float64).ravel(), actual.astype(np.float64).ravel()
    difference = np.abs(left - right)
    return {"finite": bool(np.isfinite(actual).all()),
        "reference_rms": float(np.sqrt(np.mean(left * left))), "actual_rms": float(np.sqrt(np.mean(right * right))),
        "rmse": float(np.sqrt(np.mean((left - right) ** 2))),
        "max_abs": float(difference.max()), "max_abs_index": int(difference.argmax()),
        "absolute_error_percentiles_50_95_99_99_9": np.percentile(difference, [50, 95, 99, 99.9]).tolist(),
        "cosine": float(np.dot(left, right) / max(np.linalg.norm(left) * np.linalg.norm(right), 1e-300))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=17817)
    args = parser.parse_args()
    if not 128 <= args.rows <= 256 or not 1 <= args.repeats <= 3 or not 1 <= args.threads <= 256:
        parser.error("rows128..256, repeats1..3, threads1..256 required")
    import numpy as np
    from gguf import GGUFReader, GGML_QUANT_SIZES
    plan = bench.plan_model(bench.warm.discover_shards(args.model), [args.layer], GGUFReader, GGML_QUANT_SIZES)
    inputs, ids, routing = bench.make_inputs(plan["metadata"], args.rows, args.seed + args.layer * 1009 + args.rows, np)
    report = {"status": "diagnostic_no_parity_verdict", "plan": plan, "calls": [], "comparisons": {},
        "configuration": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "native_sha256": bench.sha256_file(args.library), "tool_sha256": bench.sha256_file(__file__),
        "microbenchmark_sha256": bench.sha256_file(bench.__file__),
        "environment": {key: os.environ[key] for key in ("TS_HOST_MOE_EXPERT_FILTER", "GGML_CUDA_DISABLE_FUSION") if key in os.environ}}
    saved = {"inputs": inputs, "ids": ids, "routing": routing}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    log = args.output.with_suffix(".native.log")
    os.environ["TS_HOST_MOE_DEVICE_MIN_BATCH"] = "128"
    native = None
    try:
        with bench.capture_native_stderr(log):
            native = bench.NativeMoE(args.library, backend=3)
            native.threads(args.threads)
            with bench.MappedLayer(plan, args.layer, np) as mapped:
                saved["post_expert_norm"] = mapped.norm.copy()
                report["norm_abs_percentiles_0_50_95_99_100"] = np.percentile(np.abs(mapped.norm), [0, 50, 95, 99, 100]).tolist()
                for kind, stream, cpu in (("cpu", False, True), ("stream", True, True), ("resident", False, False), ("stream_after_resident", True, True)):
                    os.environ["TS_MUSE_SPARK_STREAM_EXPERTS"] = "1" if stream else "0"
                    for repeat in range(args.repeats):
                        output = np.full_like(inputs, np.nan)
                        started = time.perf_counter()
                        native.run(inputs, output, ids, routing, mapped, plan["metadata"], run_on_cpu=cpu)
                        duration = (time.perf_counter() - started) * 1000
                        if not np.isfinite(output).all():
                            raise ValueError(f"{kind} produced nonfinite output")
                        key = f"{kind}_{repeat}"
                        saved[key] = output
                        report["calls"].append({"kind": kind, "repeat": repeat, "diagnostic_wall_ms": duration})
                        report["comparisons"][key + "_vs_cpu0"] = compare(saved["cpu_0"], output, np)
                        if kind != "cpu":
                            report["comparisons"][key + "_vs_stream0"] = compare(saved["stream_0"], output, np)
                        bench.save(args.output, report)
                        print(key, report["comparisons"][key + "_vs_cpu0"], flush=True)
    finally:
        if native is not None:
            native.close()
        report["native_log"] = {"path": str(log), "sha256": bench.sha256_file(log)}
        payload = args.output.with_suffix(".npz")
        np.savez_compressed(payload, **saved)
        report["arrays"] = {"path": str(payload), "sha256": bench.sha256_file(payload)}
        bench.save(args.output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
