#!/usr/bin/env python3
"""Compare TensorSharp TS_DUMP_LOGITS F32 vectors without requiring numpy.

TS_DUMP_LOGITS=<path> writes the logits that choose a run's FIRST GENERATED TOKEN: the last prompt
row once the whole prompt has been processed, whichever path ran it (one forward or several chunks,
per-sequence, fused or batched prefill, --tp / --layer-split, the CLI benchmark's Forward or
ForwardRefill). Warm-up forwards and warm-up requests are never dumped, and neither is a decode step.
Beside the vector it writes <path>.json: the semantics tag, prompt length and token fingerprint,
prefix-cache reuse, every prefill forward with the path that served it, and the first token.

Two vectors are comparable only when they are rows for the SAME prompt. When both sidecars are present
this tool checks that (same semantics, prompt length, prompt token fingerprint, media-span count and
LoRA adapter) and fails the comparison otherwise, whatever the numbers say. It also checks that each
sidecar describes the vector beside it (vector_sha256, count, argmax): an older sidecar left next to a
vector written by another run or build - a first-forward build overwriting only <path>, a run that died
before writing its own pair - is "not_comparable", and so is a sidecar that is not valid JSON.

A vector without a sidecar was written by a build that dumped the first non-warm-up forward instead
(which is not the prompt's last row when the prompt ran as several forwards); the report marks its
comparability "unknown" and --require-sidecar turns that into a failure. Harnesses should delete <path>
and <path>.json before each run (the engine also removes them when it arms the dump).

The sidecar's prefill_steps name the scheduler plan path each prompt forward ran under (a label, not
the kernel; the CLI's direct loops record one step for the whole prompt), so use them to explain a
difference, not to assert forward counts. The prompt fingerprint covers token ids only: media content
is not part of it (media_spans only says media were present), so callers still bind the inputs.
Under concurrent requests the first request to SAMPLE is the one dumped; compare single-request runs.
"""
import argparse
from array import array
import hashlib
import json
import math
from pathlib import Path
import sys


def read(path):
    value = array("f")
    value.frombytes(path.read_bytes())
    if sys.byteorder != "little":
        value.byteswap()
    if not value or not all(math.isfinite(x) for x in value):
        raise ValueError(f"Empty or nonfinite logits: {path}")
    return value


FIRST_TOKEN_SEMANTICS = "first-generated-token"


def load_meta(path):
    """(meta, problem) for the dump's JSON sidecar (<path>.json): (None, None) when the build wrote
    none, (None, reason) when it exists but is not a JSON object (truncated, killed mid-write)."""
    sidecar = Path(str(path) + ".json")
    if not sidecar.is_file():
        return None, None
    try:
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as error:
        return None, f"sidecar {sidecar.name} is unreadable: {error}"
    if not isinstance(meta, dict):
        return None, f"sidecar {sidecar.name} is not a JSON object"
    return meta, None


def read_meta(path):
    """The dump's JSON sidecar (<path>.json), or None when the build wrote none or it is unreadable."""
    return load_meta(path)[0]


def argmax_of(vector):
    return max(range(len(vector)), key=vector.__getitem__) if len(vector) else None


def sidecar_mismatches(name, meta, vector, raw=None):
    """Reasons the sidecar does NOT describe the vector beside it (empty when it does): a stale
    sidecar next to a vector another run or build wrote. vector_sha256 is checked when the sidecar
    has it (and the raw bytes are given); count and argmax always."""
    if meta is None:
        return []
    reasons = []
    if meta.get("count") is not None and meta.get("count") != len(vector):
        reasons.append(f"{name} sidecar count {meta.get('count')!r} is not the vector's {len(vector)} values")
    if meta.get("argmax") is not None and meta.get("argmax") != argmax_of(vector):
        reasons.append(f"{name} sidecar argmax {meta.get('argmax')!r} is not the vector's argmax {argmax_of(vector)}")
    expected = meta.get("vector_sha256")
    if expected is not None and raw is not None and hashlib.sha256(raw).hexdigest() != str(expected).lower():
        reasons.append(f"{name} sidecar vector_sha256 does not match the vector's bytes")
    if reasons:
        reasons.append(f"{name} sidecar is stale: it describes another run's vector")
    return reasons


def comparability(reference_meta, candidate_meta, reference_problem=None, candidate_problem=None):
    """(status, reasons): 'comparable', 'not_comparable' or 'unknown' (a sidecar is missing)."""
    problems = [p for p in (reference_problem, candidate_problem) if p]
    if problems:
        return "not_comparable", problems
    if reference_meta is None or candidate_meta is None:
        missing = [name for name, meta in (("reference", reference_meta), ("candidate", candidate_meta)) if meta is None]
        return "unknown", [f"no sidecar for the {' and '.join(missing)} vector: written by a first-forward build, "
                           "so it may not be the prompt's last row"]
    reasons = []
    for name, meta in (("reference", reference_meta), ("candidate", candidate_meta)):
        if meta.get("semantics") != FIRST_TOKEN_SEMANTICS:
            reasons.append(f"{name} semantics {meta.get('semantics')!r} is not {FIRST_TOKEN_SEMANTICS!r}")
    # media_spans / adapter arrived after the first sidecars: absent means none.
    for field, absent in (("prompt_tokens", None), ("prompt_fingerprint", None), ("media_spans", 0), ("adapter", None)):
        ref_value, cand_value = reference_meta.get(field, absent), candidate_meta.get(field, absent)
        if ref_value != cand_value:
            reasons.append(f"{field} differs: {ref_value!r} vs {cand_value!r}")
    if reference_meta.get("count") != candidate_meta.get("count"):
        reasons.append(f"vocabulary width differs: {reference_meta.get('count')!r} vs {candidate_meta.get('count')!r}")
    return ("not_comparable" if reasons else "comparable"), reasons


def first_token_check(meta, vector):
    """Whether the sidecar's recorded first token is the vector's argmax (informational: a sampled,
    forced or device-pending first token need not be)."""
    if meta is None or meta.get("first_token") is None:
        return None
    return {"first_token": meta.get("first_token"), "first_token_source": meta.get("first_token_source"),
            "argmax": argmax_of(vector), "argmax_is_first_token": meta.get("first_token") == argmax_of(vector)}


def compare(reference, candidate, max_absolute_error=None, max_rmse=None, min_cosine=None,
            require_same_argmax=False, max_relative_l2=None):
    if not reference or len(reference) != len(candidate):
        raise ValueError("Logit vector sizes differ or are empty")
    if not all(math.isfinite(x) for x in (*reference, *candidate)):
        raise ValueError("Nonfinite logits")
    delta = [x - y for x, y in zip(reference, candidate)]
    dot = sum(x * y for x, y in zip(reference, candidate))
    norm = math.sqrt(sum(x * x for x in reference) * sum(y * y for y in candidate))
    reference_l2 = math.sqrt(sum(x * x for x in reference))
    error_l2 = math.sqrt(sum(x * x for x in delta))
    ref_max = max(range(len(reference)), key=reference.__getitem__)
    cand_max = max(range(len(candidate)), key=candidate.__getitem__)
    absolute = sorted(abs(x) for x in delta)
    top_reference = sorted(range(len(reference)), key=reference.__getitem__, reverse=True)[:10]
    top_candidate = sorted(range(len(candidate)), key=candidate.__getitem__, reverse=True)[:10]
    report = {"count": len(reference), "all_finite": True,
              "max_absolute_error": max(abs(x) for x in delta),
              "rmse": math.sqrt(sum(x * x for x in delta) / len(delta)),
              "relative_l2_error": error_l2 / reference_l2 if reference_l2 else (0. if not error_l2 else None),
              "cosine_similarity": dot / norm if norm else None,
              "reference_argmax": ref_max, "candidate_argmax": cand_max,
              "same_argmax": ref_max == cand_max,
              "absolute_error_percentiles": {str(p): absolute[round((len(absolute) - 1) * p / 100)]
                                              for p in (50, 90, 95, 99, 100)},
              "reference_top10": [{"token": i, "logit": reference[i]} for i in top_reference],
              "candidate_top10": [{"token": i, "logit": candidate[i]} for i in top_candidate],
              "reference_top1_margin": reference[top_reference[0]] - reference[top_reference[1]] if len(reference) > 1 else None,
              "candidate_top1_margin": candidate[top_candidate[0]] - candidate[top_candidate[1]] if len(candidate) > 1 else None,
              "top10_overlap": len(set(top_reference) & set(top_candidate)),
              "thresholds": {"max_absolute_error": max_absolute_error, "max_rmse": max_rmse,
                             "min_cosine": min_cosine, "require_same_argmax": require_same_argmax},
              "failures": []}
    for field, threshold in (("max_absolute_error", max_absolute_error), ("rmse", max_rmse)):
        if threshold is not None and report[field] > threshold:
            report["failures"].append(f"{field} {report[field]} exceeds {threshold}")
    if min_cosine is not None and (report["cosine_similarity"] is None or report["cosine_similarity"] < min_cosine):
        report["failures"].append(f"Cosine similarity {report['cosine_similarity']} is below {min_cosine} or undefined")
    if require_same_argmax and not report["same_argmax"]:
        report["failures"].append("Argmax token differs")
    report["thresholds"]["max_relative_l2"] = max_relative_l2
    if max_relative_l2 is not None and (report["relative_l2_error"] is None or report["relative_l2_error"] > max_relative_l2):
        report["failures"].append(f"Relative L2 error {report['relative_l2_error']} exceeds {max_relative_l2} or is undefined")
    gated = any(value is not None for value in (max_absolute_error, max_rmse, min_cosine, max_relative_l2)) or require_same_argmax
    report["status"] = "failed" if report["failures"] else "passed" if gated else "measured_not_gated"
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("reference", type=Path)
    p.add_argument("candidate", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--max-absolute-error", type=float)
    p.add_argument("--max-rmse", type=float)
    p.add_argument("--min-cosine", type=float)
    p.add_argument("--max-relative-l2", type=float)
    p.add_argument("--require-same-argmax", action="store_true")
    p.add_argument("--require-sidecar", action="store_true",
                   help="Fail unless both vectors carry a first-token sidecar (<path>.json) proving they are rows for the same prompt")
    a = p.parse_args()
    for value in (a.max_absolute_error, a.max_rmse, a.max_relative_l2):
        if value is not None and (not math.isfinite(value) or value < 0):
            p.error("Error thresholds must be finite and nonnegative")
    if a.min_cosine is not None and (not math.isfinite(a.min_cosine) or not -1 <= a.min_cosine <= 1):
        p.error("min-cosine must be finite and between -1 and 1")
    reference_raw, candidate_raw = a.reference.read_bytes(), a.candidate.read_bytes()
    reference, candidate = read(a.reference), read(a.candidate)
    report = compare(reference, candidate, a.max_absolute_error, a.max_rmse, a.min_cosine, a.require_same_argmax, a.max_relative_l2)
    (reference_meta, reference_problem), (candidate_meta, candidate_problem) = load_meta(a.reference), load_meta(a.candidate)
    status, reasons = comparability(reference_meta, candidate_meta, reference_problem, candidate_problem)
    stale = (sidecar_mismatches("reference", reference_meta, reference, reference_raw)
             + sidecar_mismatches("candidate", candidate_meta, candidate, candidate_raw))
    if stale:
        status, reasons = "not_comparable", stale + (reasons if status == "not_comparable" else [])
    report.update(comparability=status, comparability_reasons=reasons,
                  reference_meta=reference_meta, candidate_meta=candidate_meta,
                  reference_first_token=first_token_check(reference_meta, reference),
                  candidate_first_token=first_token_check(candidate_meta, candidate))
    if status == "not_comparable" or (status == "unknown" and a.require_sidecar):
        report["failures"].extend("Not comparable: " + reason for reason in reasons)
        report["status"] = "failed"
    report.update(reference=str(a.reference), candidate=str(a.candidate),
                  reference_sha256=hashlib.sha256(reference_raw).hexdigest(),
                  candidate_sha256=hashlib.sha256(candidate_raw).hexdigest(),
                  limitations="One first-generated-token vector (the last prompt row) on fixed input, not full-model quality or autoregressive token parity; callers must bind model and input identity (the sidecar binds the prompt).")
    encoded = json.dumps(report, indent=2) + "\n"
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(encoded)
    print(encoded)
    return int(report["status"] == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
