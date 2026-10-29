#!/usr/bin/env python3
"""Gate every fixed-token vector from DeepSeek41DistributedProbe against a control.

The caller must separately verify immutable checkpoint bytes and build identities.
This compares matching operation streams, including decode after forced tokens;
it never substitutes a generated-text match for a numerical gate.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path


spec = importlib.util.spec_from_file_location("parallel_logits", Path(__file__).with_name("compare-parallel-logits.py"))
logits = importlib.util.module_from_spec(spec)
spec.loader.exec_module(logits)


def compare(reference_path, candidate_path, *, max_absolute_error, max_rmse,
            min_cosine, max_relative_l2, require_same_argmax=False, max_normalized_error=None):
    if max_normalized_error is not None and (not math.isfinite(max_normalized_error) or max_normalized_error < 0):
        raise ValueError("Normalized error gate must be finite and nonnegative")
    reference = json.loads(reference_path.read_text())
    candidate = json.loads(candidate_path.read_text())
    failures, rows = [], []
    for name, report in (("reference", reference), ("candidate", candidate)):
        if report.get("status") != "completed" or report.get("disposed") is not True:
            failures.append(f"{name} is not a completed, disposed driver run")
    for field in ("schema_version", "weights_id", "tokens", "tokens_sha256", "vocab_size"):
        if reference.get(field) is None or reference.get(field) != candidate.get(field):
            failures.append(f"Missing or different {field}")
    left_steps, right_steps = reference.get("steps", []), candidate.get("steps", [])
    if not left_steps or len(left_steps) != len(right_steps):
        failures.append("Operation counts differ or are empty")
    for index, (left, right) in enumerate(zip(left_steps, right_steps)):
        if any(left.get(field) != right.get(field) for field in ("operation", "tokens", "count")):
            failures.append(f"Operation {index} input, kind or vocabulary count differs")
            continue
        if (left.get("logits") is None) != (right.get("logits") is None):
            failures.append(f"Operation {index} is missing one logit vector")
            continue
        if left.get("logits") is None:
            continue
        def vector(report_path, step):
            path = report_path.parent / step["logits"]
            # Report paths must point to immediate evidence files, not unrelated
            # vectors outside this run or a model directory.
            if path.resolve().parent != report_path.parent.resolve():
                raise ValueError("Logit vector must be inside its run directory")
            value = logits.read(path)
            if len(value) != step["count"]:
                raise ValueError("Vector length differs from the operation's count")
            return value, hashlib.sha256(path.read_bytes()).hexdigest()
        try:
            ref, ref_hash = vector(reference_path, left)
            cand, cand_hash = vector(candidate_path, right)
            row = logits.compare(ref, cand, max_absolute_error, max_rmse, min_cosine,
                                 require_same_argmax, max_relative_l2)
            # Match Glm5NextCudaEngineTests.RelativeError: normalize the maximum
            # absolute error by the reference's largest magnitude, with a unit
            # floor. This is distinct from the relative-L2 gate above.
            scale = max(1.0, max(abs(value) for value in ref))
            row["normalization_scale"] = scale
            row["normalized_max_error"] = row["max_absolute_error"] / scale
            row["thresholds"]["max_normalized_error"] = max_normalized_error
            if max_normalized_error is not None:
                if row["normalized_max_error"] > max_normalized_error:
                    row["failures"].append(
                        f"Normalized maximum error {row['normalized_max_error']} exceeds {max_normalized_error}")
                row["status"] = "failed" if row["failures"] else "passed"
            row.update(index=index, operation=left["operation"], reference_sha256=ref_hash,
                       candidate_sha256=cand_hash)
            rows.append(row)
            if row["status"] != "passed":
                failures.append(f"Operation {index} numerical gate failed")
        except (OSError, ValueError, KeyError) as error:
            failures.append(f"Operation {index}: {error}")
    if not rows:
        failures.append("No logit vectors compared")
    return {"status": "failed" if failures else "passed", "failures": failures, "vectors": rows,
            "reference": str(reference_path), "candidate": str(candidate_path),
            "limitations": "Fixed teacher-forced vectors, not broad answer quality, speculative parity or performance scaling."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-absolute-error", type=float, required=True)
    parser.add_argument("--max-rmse", type=float, required=True)
    parser.add_argument("--max-relative-l2", type=float, required=True)
    parser.add_argument("--min-cosine", type=float, required=True)
    parser.add_argument("--max-normalized-error", type=float,
                        help="Optional max absolute error / max(1, max absolute reference logit), per vector")
    parser.add_argument("--require-same-argmax", action="store_true")
    args = parser.parse_args()
    if any(not math.isfinite(value) or value < 0 for value in
           (args.max_absolute_error, args.max_rmse, args.max_relative_l2, args.max_normalized_error)
           if value is not None):
        parser.error("Error gates must be finite and nonnegative")
    if not math.isfinite(args.min_cosine) or not -1 <= args.min_cosine <= 1:
        parser.error("Cosine gate must be finite and between -1 and 1")
    report = compare(args.reference, args.candidate, max_absolute_error=args.max_absolute_error,
                     max_rmse=args.max_rmse, min_cosine=args.min_cosine,
                     max_relative_l2=args.max_relative_l2, require_same_argmax=args.require_same_argmax,
                     max_normalized_error=args.max_normalized_error)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "vectors"}, indent=2))
    return int(report["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
