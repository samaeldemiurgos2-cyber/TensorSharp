#!/usr/bin/env python3
"""Compare every tiny Muse Spark logit against the independent scalar oracle.

Consumes exports from TS_MUSE_SPARK_FIXTURE_OUTPUT and the TensorSharp-owned
muse-spark-llama-reference.cpp driver. Synthetic correctness evidence only;
neither model-quality nor production-throughput claims follow from this test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(expected, actual, atol, rtol):
    if len(expected) != len(actual) or any(len(a) != len(b) for a, b in zip(expected, actual)):
        raise ValueError("Logit shape mismatch")
    if not expected or not expected[0]:
        raise ValueError("Empty logits cannot validate inference")
    output = []
    for position, (a, b) in enumerate(zip(expected, actual)):
        if not all(math.isfinite(x) for x in a + b):
            raise ValueError("Nonfinite logits")
        errors = [abs(x-y) for x, y in zip(a, b)]
        an = math.sqrt(sum(x*x for x in a))
        bn = math.sqrt(sum(x*x for x in b))
        difference = math.sqrt(sum(e*e for e in errors))
        expected_top = max(range(len(a)), key=a.__getitem__)
        actual_top = max(range(len(b)), key=b.__getitem__)
        close = all(error <= atol + rtol * abs(x) for x, error in zip(a, errors))
        output.append({"position": position, "max_absolute_error": max(errors),
                       "relative_l2": difference / an if an else None,
                       "cosine": sum(x*y for x, y in zip(a, b)) / (an * bn) if an * bn else None,
                       "expected_argmax": expected_top, "actual_argmax": actual_top,
                       "allclose": close, "status": "pass" if close and expected_top == actual_top else "fail"})
    return {"status": "pass" if all(r["status"] == "pass" for r in output) else "fail",
            "max_absolute_error": max(r["max_absolute_error"] for r in output), "rows": output}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--expected", type=Path, required=True)
    p.add_argument("--managed", type=Path, required=True)
    p.add_argument("--llama-logits", type=Path, required=True)
    p.add_argument("--llama-metadata", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--atol", type=float, default=3e-5)
    p.add_argument("--rtol", type=float, default=3e-5)
    args = p.parse_args()
    if not math.isfinite(args.atol) or not math.isfinite(args.rtol) or args.atol < 0 or args.rtol < 0:
        p.error("Tolerances must be finite and nonnegative")
    oracle = json.loads(args.expected.read_text(encoding="utf-8"))
    managed = json.loads(args.managed.read_text(encoding="utf-8"))
    metadata = json.loads(args.llama_metadata.read_text(encoding="utf-8"))
    reference = oracle["logits"]
    vocab, count = metadata["vocab"], metadata["tokens"]
    if (oracle["tokens"] != managed["tokens"] or oracle["tokens"] != metadata.get("token_ids")
            or len(oracle["tokens"]) != count):
        raise ValueError("Input token IDs/count differ")
    if len(reference) != count or any(len(row) != vocab for row in reference):
        raise ValueError("Oracle shape does not match llama output metadata")
    raw = args.llama_logits.read_bytes()
    if len(raw) != count * vocab * 4:
        raise ValueError("llama binary logit length is not exact")
    llama = [list(struct.unpack_from("<" + "f" * vocab, raw, i * vocab * 4)) for i in range(count)]
    comparisons = {"managed_vs_oracle": compare(reference, managed["logits"], args.atol, args.rtol),
                   "llama_vs_oracle": compare(reference, llama, args.atol, args.rtol),
                   "managed_vs_llama": compare(llama, managed["logits"], args.atol, args.rtol)}
    report = {"status": "pass" if all(c["status"] == "pass" for c in comparisons.values()) else "fail",
              "scope": "Tiny synthetic F32 model, all logits, exact tokens; not large-model quality/performance evidence.",
              "tokens": oracle["tokens"], "vocab": vocab, "atol": args.atol, "rtol": args.rtol,
              "llama_settings": metadata, "comparisons": comparisons,
              "input_sha256": {str(path.resolve()): sha(path) for path in (args.fixture, args.expected,
                  args.managed, args.llama_logits, args.llama_metadata)}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "comparisons": {
        name: {k: v for k, v in item.items() if k != "rows"} for name, item in comparisons.items()}}))
    return int(report["status"] != "pass")


if __name__ == "__main__":
    raise SystemExit(main())
