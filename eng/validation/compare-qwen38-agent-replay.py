#!/usr/bin/env python3
"""Compare immutable Qwen3.8 replay reports using identical paired samples.

This checks request identity and observed output checks, not weight equivalence
or general model accuracy. Ratios are candidate/baseline (<1 means faster).
"""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
import math
from pathlib import Path
import re
import statistics

SPEC = importlib.util.spec_from_file_location("qwen38_replay", Path(__file__).with_name("qwen38-agent-replay.py"))
REPLAY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPLAY)
SHA256 = re.compile(r"[0-9a-fA-F]{64}\Z")


def request_key(row):
    scenario = row.get("scenario")
    values = [row.get(name) for name in ("concurrency", "lane", "step")]
    if not isinstance(scenario, str) or not scenario or any(type(v) is not int for v in values):
        raise ValueError("request key requires scenario and integer concurrency/lane/step")
    degree, lane, step = values
    if degree < 1 or not 0 <= lane < degree or step < 0:
        raise ValueError("request key contains invalid concurrency/lane/step")
    return scenario, degree, lane, step


def finite_positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def distribution(values):
    values = list(values)
    return {"n": len(values), "p50": statistics.median(values) if values else None,
            "p90": REPLAY.quantile(values, .9), "min": min(values) if values else None,
            "max": max(values) if values else None}


def row_failures(row):
    recorded = row.get("failures")
    failures = list(recorded) if isinstance(recorded, list) and all(isinstance(v, str) for v in recorded) else ["missing/invalid recorded validation"]
    if row.get("done") is not True:
        failures.append("stream not explicitly complete")
    if "expected" not in row:
        failures.append("missing expected-output declaration")
    calls = row.get("tool_calls")
    if not isinstance(calls, list) or any(not isinstance(call, dict) or not isinstance(call.get("function"), dict)
                                         for call in calls):
        failures.append("missing/invalid tool-call evidence")
    try:
        validation = REPLAY.validate_result(row, row.get("expected"))
        if all(isinstance(reason, str) for reason in validation):
            failures.extend(validation)
        else:
            failures.append("malformed validation error evidence")
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        failures.append(f"malformed validation evidence: {type(error).__name__}")
    if not finite_positive(row.get("e2e_ms")):
        failures.append("missing/invalid end-to-end timing")
    ttft = row.get("ttft_ms")
    if not finite_positive(ttft) or (finite_positive(row.get("e2e_ms")) and ttft > row["e2e_ms"]):
        failures.append("missing/invalid first-delta timing")
    return sorted(set(failures))


def index_report(report):
    rows = report.get("results")
    errors = []
    if not isinstance(rows, list) or not rows:
        return {}, ["missing or empty results"]
    indexed = {}
    for i, row in enumerate(rows):
        try:
            if not isinstance(row, dict):
                raise ValueError("result is not an object")
            key = request_key(row)
            if key in indexed:
                raise ValueError(f"duplicate request key {key}")
            indexed[key] = row
        except ValueError as error:
            errors.append(f"row {i}: {error}")
    return indexed, errors


def output_signature(row):
    calls = []
    for call in row.get("tool_calls", []):
        function = call.get("function", {})
        arguments = function.get("arguments", "")
        try:
            arguments = json.loads(arguments)
        except (TypeError, ValueError):
            pass
        # Server-generated call IDs may differ for the same function/arguments.
        calls.append({"name": function.get("name"), "arguments": arguments})
    return {"answer": row.get("answer"), "reasoning": row.get("reasoning"), "calls": calls,
            "completion_tokens": row.get("completion_tokens"), "finish_reason": row.get("finish_reason")}


def summarize_case(rows):
    checked = [(row, row_failures(row)) for row in rows]
    image_tool = lambda row: isinstance(row.get("expected"), dict) and "code" in row["expected"]
    arithmetic = lambda row: isinstance(row.get("expected"), dict) and "answer" in row["expected"]
    return {
        "requests": len(rows),
        "observed_checks_passed": sum(not errors for _, errors in checked),
        "failed": sum(bool(errors) for _, errors in checked),
        "image_or_tool_errors": sum(bool(errors) and image_tool(row) for row, errors in checked),
        "arithmetic_errors": sum(bool(errors) and arithmetic(row) for row, errors in checked),
        "prose_transport_errors": sum(bool(errors) and not row.get("expected") for row, errors in checked),
        "prose_semantics_unscored": sum(not row.get("expected") for row in rows),
        "truncated_prose": sum(not row.get("expected") and row.get("finish_reason") == "length" for row in rows),
        "failure_reasons": dict(Counter(reason for _, errors in checked for reason in errors)),
        "finish_reasons": dict(Counter(str(row.get("finish_reason", "missing")) for row in rows)),
        # These descriptive totals include failed requests, with their failures
        # displayed alongside. Performance ratios below use paired valid rows.
        "all_observed_e2e_ms": distribution(row["e2e_ms"] for row in rows if finite_positive(row.get("e2e_ms"))),
        "all_observed_ttft_ms": distribution(row["ttft_ms"] for row in rows if finite_positive(row.get("ttft_ms"))),
        "all_observed_completion_tokens": distribution(row["completion_tokens"] for row in rows
                                                       if type(row.get("completion_tokens")) is int and row["completion_tokens"] >= 0),
    }


def case_key(key):
    return key[0], key[1], "step0" if key[3] == 0 else "later_steps"


def describe_report(report, index, errors):
    return {"label": report.get("label"), "model": report.get("model"), "complete": report.get("complete") is True,
            "reported_passed": report.get("passed"), "fixture_sha256": report.get("fixture_sha256"),
            "provenance": report.get("provenance"), "index_errors": errors,
            "results": summarize_case(list(index.values()))}


def compare(baseline, candidate):
    left, left_errors = index_report(baseline)
    right, right_errors = index_report(candidate)
    errors = [f"baseline: {error}" for error in left_errors] + [f"candidate: {error}" for error in right_errors]
    a, b = baseline.get("fixture_sha256"), candidate.get("fixture_sha256")
    if not isinstance(a, str) or not SHA256.fullmatch(a) or a != b:
        errors.append("fixture SHA-256 missing or different")
    if baseline.get("complete") is not True or candidate.get("complete") is not True:
        errors.append("one or both reports are incomplete")
    missing = sorted(left.keys() - right.keys())
    extra = sorted(right.keys() - left.keys())
    if missing or extra:
        errors.append("request key sets differ")
    common = left.keys() & right.keys()
    mismatches = []
    for key in sorted(common):
        first, second = left[key], right[key]
        sha = first.get("request_sha256")
        if not isinstance(sha, str) or not SHA256.fullmatch(sha) or sha != second.get("request_sha256"):
            mismatches.append({"key": key, "reason": "request SHA-256 missing or different"})
        if first.get("expected") != second.get("expected"):
            mismatches.append({"key": key, "reason": "expected-output declarations differ"})
    if mismatches:
        errors.append("one or more request identities differ")
    identity_valid = not errors
    cases = []
    for scenario, degree, phase in sorted({case_key(key) for key in left.keys() | right.keys()}):
        group = (scenario, degree, phase)
        left_keys = {key for key in left if case_key(key) == group}
        right_keys = {key for key in right if case_key(key) == group}
        paired = []
        excluded = []
        for key in sorted(left_keys & right_keys):
            first, second = left[key], right[key]
            first_errors, second_errors = row_failures(first), row_failures(second)
            if first_errors or second_errors or not identity_valid:
                excluded.append({"key": key, "baseline_failures": first_errors, "candidate_failures": second_errors,
                                 "identity_qualified": identity_valid})
            else:
                paired.append({"key": key, "baseline_e2e_ms": first["e2e_ms"], "candidate_e2e_ms": second["e2e_ms"],
                               "candidate_over_baseline_e2e": second["e2e_ms"] / first["e2e_ms"],
                               "baseline_ttft_ms": first["ttft_ms"], "candidate_ttft_ms": second["ttft_ms"],
                               "candidate_over_baseline_ttft": second["ttft_ms"] / first["ttft_ms"],
                               "baseline_completion_tokens": first["completion_tokens"],
                               "candidate_completion_tokens": second["completion_tokens"],
                               "output_identity_matched": output_signature(first) == output_signature(second)})
        all_valid = identity_valid and bool(left_keys) and len(paired) == len(left_keys) == len(right_keys)
        cases.append({
            "scenario": scenario, "concurrency": degree, "phase": phase,
            "baseline": summarize_case([left[key] for key in sorted(left_keys)]),
            "candidate": summarize_case([right[key] for key in sorted(right_keys)]),
            "n_max_paired": min(len(left_keys), len(right_keys)),
            "n_identity_matched": len(left_keys & right_keys) if identity_valid else 0,
            "n_paired_valid": len(paired), "n_excluded_common": len(excluded),
            "n_missing_or_extra": len(left_keys ^ right_keys),
            "all_requested_checks_passed": all_valid,
            "ratio_scope": "all requested paired samples" if all_valid else "exploratory valid paired subset; failures/missing coverage prevent qualification",
            "paired_candidate_over_baseline_e2e": distribution(row["candidate_over_baseline_e2e"] for row in paired),
            "paired_candidate_over_baseline_ttft": distribution(row["candidate_over_baseline_ttft"] for row in paired),
            "paired_baseline_e2e_ms": distribution(row["baseline_e2e_ms"] for row in paired),
            "paired_candidate_e2e_ms": distribution(row["candidate_e2e_ms"] for row in paired),
            "paired_output_identity_matches": sum(row["output_identity_matched"] for row in paired),
            "all_paired_outputs_identical": bool(paired) and all(row["output_identity_matched"] for row in paired),
            "paired_samples": paired, "excluded_samples": excluded,
        })
    return {"baseline_label": baseline.get("label"), "candidate_label": candidate.get("label"),
            "identity_and_completion_valid": identity_valid, "identity_errors": errors,
            "missing_candidate_keys": missing, "extra_candidate_keys": extra, "request_mismatches": mismatches,
            "all_requested_checks_passed": bool(cases) and identity_valid and all(case["all_requested_checks_passed"] for case in cases)
                and baseline.get("passed") is True and candidate.get("passed") is True,
            "cases": cases}


def compare_reports(baseline, candidates):
    reports = [baseline, *candidates]
    labels = [report.get("label") for report in reports]
    if any(not isinstance(label, str) or not label for label in labels) or len(labels) != len(set(labels)):
        raise ValueError("reports require distinct nonempty labels")
    if not candidates:
        raise ValueError("at least one candidate report is required")
    comparisons = [compare(baseline, candidate) for candidate in candidates]
    return {
        "ratio_direction": "candidate/baseline; less than 1 means faster",
        "scope": "Observed replay checks only, not a semantic-quality or Velox-equivalence claim. Model weights, quantization and server configuration require separate matching.",
        "phase_scope": "step0 and later_steps are separate; step0 is not guaranteed cold because replay waves do not reset server cache.",
        "prose_scope": "Length-capped/nonempty prose can pass transport checks; semantic correctness is not scored.",
        "pairing_scope": "The same request keys must appear in both reports with matching fixture/request hashes. Both sides use the identical paired-valid sample subset; any excluded request prevents an all-samples qualification.",
        "sample_scope": "Every case records n_max_paired and n_paired_valid. Medians/p90 are descriptive finite-sample statistics, not confidence intervals or a significance test.",
        "reports": [describe_report(report, *index_report(report)) for report in reports],
        "comparisons": comparisons,
        "all_requested_checks_passed": all(comparison["all_requested_checks_passed"] for comparison in comparisons),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--output", type=REPLAY.evidence_path, required=True)
    args = parser.parse_args()
    try:
        def reject_nonfinite(value):
            raise ValueError(f"nonfinite JSON value: {value}")
        baseline = json.loads(args.baseline.read_text(), parse_constant=reject_nonfinite)
        candidates = [json.loads(path.read_text(), parse_constant=reject_nonfinite) for path in args.candidate]
        if not isinstance(baseline, dict) or any(not isinstance(report, dict) for report in candidates):
            raise ValueError("reports must be JSON objects")
        result = compare_reports(baseline, candidates)
    except (ValueError, OSError) as error:
        result = {"all_requested_checks_passed": False, "input_error": str(error)}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "all_requested_checks_passed": result["all_requested_checks_passed"]}))
    return 0 if result["all_requested_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
