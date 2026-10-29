#!/usr/bin/env python3
"""Score teacher-forced DeepSeek41DistributedProbe logits as next-token quality.

Compare a candidate run (for example distributed tensor parallel or layer split)
with a reference control run from the same --text token stream. Every saved
last-position vector is mapped to the true next token by replaying the probe's
cache position (reset -> 0; each forward advances by its input count). NLL and
perplexity, KL in both directions and top-k agreement stay meaningful when two
parallel modes are not bit-exact. Artifacts only: no model is loaded, and the
result is not a task-quality, long-context or throughput qualification.

The probe saves only last-position logits, so run both sides with --text and a
large --decode-count (each decoded token adds one scored decode and one scored
refill vector of vocab*4 bytes); the default --decode-count 4 scores a handful.
"""
import argparse
from array import array
import hashlib
import heapq
import json
import math
from pathlib import Path
import sys

try:
    import numpy as np
except ImportError:  # pragma: no cover - exercised only on hosts without numpy
    np = None

TOP_K = 5
DEFAULT_MAX_PPL_RATIO = 1.01
DEFAULT_MAX_MEAN_KL = 0.01
DEFAULT_MIN_TOP1 = 0.98
PROBE_SCHEMA_VERSION = 1
MIN_UNIQUE_TARGETS_WARNING = 32
SCOPES = ("all", "unique_targets")
TARGET_MAPPING = ("Replays DeepSeek41DistributedProbe/Program.cs: a 'reset' control sets the cache position to 0; "
                  "each forward must feed tokens[position:position+n] and returns the last input position's logits, "
                  "which score tokens[position+n] when that index exists. With prefix P = len(tokens) - decode_count: "
                  "prefill and reset-prefill score tokens[P]; decode/refill of tokens[i] score tokens[i+1] (the final "
                  "one is unscored); chunk tokens[s:e] scores tokens[e].")
DISTRIBUTION_METRICS = ("kl_reference_to_candidate", "kl_candidate_to_reference", "max_abs_logit_diff",
                        "relative_l2", "top5_overlap")
SCORED_METRICS = ("reference_nll", "candidate_nll", "nll_delta")


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def zero_normalized(value):
    return value + 0.0  # turn -0.0 into 0.0 so exact self-checks serialize cleanly


def safe_exp(value):
    try:
        return math.exp(value)
    except OverflowError:
        return None


def resolve_run(path):
    """Accept a probe output directory or its probe.json; return (directory, probe path, report, raw bytes)."""
    path = Path(path)
    probe = path / "probe.json" if path.is_dir() else path
    if probe.name != "probe.json":
        raise ValueError(f"Expected a probe output directory or probe.json: {path}")
    raw = probe.read_bytes()
    return probe.parent, probe, json.loads(raw), raw


def teacher_targets(tokens, steps):
    """Map each saved vector to the teacher token it predicts, or None past the stream end.

    Inputs are checked against the token stream so a changed probe schedule fails
    instead of silently scoring the wrong token. After any unknown control or a
    non-continuing input the position is unknown until the next reset.
    """
    rows, failures, position = [], [], None
    for index, step in enumerate(steps):
        operation, inputs = step.get("operation"), step.get("tokens")
        if step.get("logits") is None:
            if operation == "reset" and inputs == []:
                position = 0
            else:
                failures.append(f"Step {index}: unknown control {operation!r}; cache position cannot be tracked")
                position = None
            continue
        if position is None:
            failures.append(f"Step {index}: forward without a preceding reset; its target is unknown")
            continue
        if (not isinstance(inputs, list) or not inputs
                or inputs != tokens[position:position + len(inputs)]):
            failures.append(f"Step {index} ({operation}): input does not continue the teacher stream at position {position}")
            position = None
            continue
        before, position = position, position + len(inputs)
        rows.append(dict(index=index, operation=operation, position_before=before, input_count=len(inputs),
                         target_position=position, target_token=tokens[position] if position < len(tokens) else None))
    return rows, failures


def decode_vector(raw, use_numpy):
    if not raw or len(raw) % 4:
        raise ValueError("Logit file is empty or not a whole number of F32 values")
    if use_numpy:
        return np.frombuffer(raw, dtype="<f4").astype(np.float64)
    value = array("f")
    value.frombytes(raw)
    if sys.byteorder != "little":
        value.byteswap()
    return value.tolist()


def _top_k_numpy(value, k):
    k = min(k, value.size)
    kth = np.partition(value, value.size - k)[value.size - k]
    greater = np.flatnonzero(value > kth)
    greater = greater[np.lexsort((greater, -value[greater]))]
    equal = np.flatnonzero(value == kth)[:k - greater.size]
    return [int(i) for i in np.concatenate((greater, equal))]


def _top_k_python(value, k):
    return heapq.nsmallest(min(k, len(value)), range(len(value)), key=lambda i: (-value[i], i))


def _core_numpy(a, b):
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("Nonfinite logits")
    lse = []
    for value in (a, b):
        peak = float(value.max())
        lse.append(peak + math.log(float(np.exp(value - peak).sum())))
    log_a, log_b = a - lse[0], b - lse[1]
    delta = b - a
    norm = float(np.linalg.norm(a))
    return dict(lse=lse, kl_rc=float(np.dot(np.exp(log_a), log_a - log_b)),
                kl_cr=float(np.dot(np.exp(log_b), log_b - log_a)),
                max_abs=float(np.max(np.abs(delta))), norm=norm, distance=float(np.linalg.norm(delta)),
                top_a=_top_k_numpy(a, TOP_K), top_b=_top_k_numpy(b, TOP_K),
                rank=lambda value, token: 1 + int(np.count_nonzero(value > value[token])))


def _core_python(a, b):
    if not all(math.isfinite(x) for x in a) or not all(math.isfinite(x) for x in b):
        raise ValueError("Nonfinite logits")
    lse = []
    for value in (a, b):
        peak = max(value)
        lse.append(peak + math.log(math.fsum(math.exp(x - peak) for x in value)))
    log_a = [x - lse[0] for x in a]
    log_b = [x - lse[1] for x in b]
    delta = [y - x for x, y in zip(a, b)]
    return dict(lse=lse, kl_rc=math.fsum(math.exp(p) * (p - q) for p, q in zip(log_a, log_b)),
                kl_cr=math.fsum(math.exp(q) * (q - p) for p, q in zip(log_a, log_b)),
                max_abs=max(abs(x) for x in delta), norm=math.sqrt(math.fsum(x * x for x in a)),
                distance=math.sqrt(math.fsum(x * x for x in delta)),
                top_a=_top_k_python(a, TOP_K), top_b=_top_k_python(b, TOP_K),
                rank=lambda value, token: 1 + sum(1 for x in value if x > value[token]))


def vector_metrics(reference, candidate, target, use_numpy):
    """Float64 metrics of two decoded full-vocabulary vectors; NLL only when target is known."""
    size = reference.size if use_numpy else len(reference)
    if size < 2 or size != (candidate.size if use_numpy else len(candidate)):
        raise ValueError("Vector sizes differ or hold fewer than two logits")
    core = (_core_numpy if use_numpy else _core_python)(reference, candidate)
    top_a, top_b = core["top_a"], core["top_b"]
    if core["norm"]:
        relative = core["distance"] / core["norm"]
    else:
        relative = 0.0 if not core["distance"] else None
    row = dict(kl_reference_to_candidate=zero_normalized(core["kl_rc"]),
               kl_candidate_to_reference=zero_normalized(core["kl_cr"]),
               max_abs_logit_diff=core["max_abs"], relative_l2=relative,
               reference_argmax=top_a[0], candidate_argmax=top_b[0], argmax_match=top_a[0] == top_b[0],
               top5_overlap=len(set(top_a) & set(top_b)) / len(top_a),
               reference_top5=top_a, candidate_top5=top_b)
    if target is not None:
        if not 0 <= target < size:
            raise ValueError("Target token outside the vocabulary")
        reference_nll = core["lse"][0] - float(reference[target])
        candidate_nll = core["lse"][1] - float(candidate[target])
        row.update(reference_nll=zero_normalized(reference_nll), candidate_nll=zero_normalized(candidate_nll),
                   nll_delta=zero_normalized(candidate_nll - reference_nll),
                   reference_target_rank=core["rank"](reference, target),
                   candidate_target_rank=core["rank"](candidate, target))
    return row


def percentile(sorted_values, fraction):
    """Linear interpolation, matching numpy's default percentile method."""
    position = (len(sorted_values) - 1) * fraction
    low = math.floor(position)
    high = min(low + 1, len(sorted_values) - 1)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (position - low)


def distribution(rows, metric):
    values = [(row[metric], row["index"]) for row in rows if row.get(metric) is not None]
    if not values:
        return None
    ordered = sorted(value for value, _ in values)
    return dict(count=len(ordered), mean=math.fsum(ordered) / len(ordered), min=ordered[0],
                p50=percentile(ordered, .5), p90=percentile(ordered, .9), p99=percentile(ordered, .99),
                max=ordered[-1], max_step=max(values, key=lambda item: item[0])[1])


def summarize(rows):
    scored = [row for row in rows if row.get("target_token") is not None]
    summary = dict(vectors=len(rows), scored_vectors=len(scored),
                   bitwise_equal_vectors=sum(row["bitwise_equal"] for row in rows))
    if rows:
        summary.update(top1_agreement=sum(row["argmax_match"] for row in rows) / len(rows),
                       mean_top5_overlap=math.fsum(row["top5_overlap"] for row in rows) / len(rows),
                       mean_kl_reference_to_candidate=math.fsum(row["kl_reference_to_candidate"] for row in rows) / len(rows),
                       mean_kl_candidate_to_reference=math.fsum(row["kl_candidate_to_reference"] for row in rows) / len(rows))
    if scored:
        reference_mean = math.fsum(row["reference_nll"] for row in scored) / len(scored)
        candidate_mean = math.fsum(row["candidate_nll"] for row in scored) / len(scored)
        summary.update(reference_mean_nll=reference_mean, candidate_mean_nll=candidate_mean,
                       reference_ppl=safe_exp(reference_mean), candidate_ppl=safe_exp(candidate_mean),
                       ppl_ratio=safe_exp(zero_normalized(candidate_mean - reference_mean)),
                       reference_target_top1=sum(row["reference_target_rank"] == 1 for row in scored) / len(scored),
                       candidate_target_top1=sum(row["candidate_target_rank"] == 1 for row in scored) / len(scored))
    summary["distributions"] = {metric: distribution(rows, metric) for metric in DISTRIBUTION_METRICS + SCORED_METRICS}
    return summary


def evaluate_gates(summary, max_ppl_ratio, max_mean_kl, min_top1, scope):
    def gate(threshold, value, passed, reason=None):
        result = dict(scope=scope, threshold=threshold, value=value, passed=passed)
        if reason:
            result["reason"] = reason
        return result
    gates = {}
    ratio = summary.get("ppl_ratio")
    if not summary.get("scored_vectors"):
        gates["max_ppl_ratio"] = gate(max_ppl_ratio, None, False, "No scored vectors in scope")
    elif ratio is None:
        gates["max_ppl_ratio"] = gate(max_ppl_ratio, None, False, "Perplexity ratio overflowed")
    else:
        gates["max_ppl_ratio"] = gate(max_ppl_ratio, ratio, ratio <= max_ppl_ratio)
    kl = summary.get("mean_kl_reference_to_candidate")
    gates["max_mean_kl"] = (gate(max_mean_kl, None, False, "No vectors in scope") if kl is None
                            else gate(max_mean_kl, kl, kl <= max_mean_kl))
    top1 = summary.get("top1_agreement")
    gates["min_top1"] = (gate(min_top1, None, False, "No vectors in scope") if top1 is None
                         else gate(min_top1, top1, top1 >= min_top1))
    return gates


def describe_run(directory, probe_path, report, raw):
    tokens_json = directory / "tokens.json"
    return dict(directory=str(directory), probe_json_sha256=sha256(raw),
                tokens_json_sha256=sha256(tokens_json.read_bytes()) if tokens_json.is_file() else None,
                status=report.get("status"), disposed=report.get("disposed"), weights_id=report.get("weights_id"),
                token_source="text" if report.get("text_sha256") else "tokens", text_sha256=report.get("text_sha256"),
                tokens_sha256=report.get("tokens_sha256"), vocab_size=report.get("vocab_size"),
                topology=report.get("topology"), arguments=report.get("arguments"),
                load_seconds=report.get("load_seconds"), requested_kv_cache_dtype=report.get("requested_kv_cache_dtype"))


def validate_pair(reference, candidate, ref_dir, cand_dir, allow_different_weights):
    """Return (fatal failures, other failures, warnings) for the two probe reports."""
    fatal, failures, warnings = [], [], []
    for name, report in (("reference", reference), ("candidate", candidate)):
        if report.get("schema_version") != PROBE_SCHEMA_VERSION:
            fatal.append(f"{name} has unsupported probe schema_version {report.get('schema_version')!r}")
        if report.get("status") != "completed" or report.get("disposed") is not True:
            failures.append(f"{name} is not a completed, disposed driver run")
    tokens, vocab = reference.get("tokens"), reference.get("vocab_size")
    if not (isinstance(tokens, list) and len(tokens) >= 2 and all(type(x) is int and x >= 0 for x in tokens)):
        fatal.append("Reference token stream is missing or invalid")
    elif not (type(vocab) is int and vocab > 1 and max(tokens) < vocab):
        fatal.append("Reference vocabulary is missing or smaller than the token stream")
    for field in ("tokens", "tokens_sha256", "vocab_size"):
        if reference.get(field) is None or reference.get(field) != candidate.get(field):
            fatal.append(f"Missing or different {field}")
    if reference.get("text_sha256") != candidate.get("text_sha256"):
        fatal.append("Different text_sha256")
    if reference.get("weights_id") is None or reference.get("weights_id") != candidate.get("weights_id"):
        (warnings if allow_different_weights else failures).append("Missing or different weights_id")
    if not reference.get("steps") or len(reference.get("steps", [])) != len(candidate.get("steps", [])):
        fatal.append("Operation counts differ or are empty")
    for name, report, directory in (("reference", reference, ref_dir), ("candidate", candidate, cand_dir)):
        path = directory / "tokens.json"
        if path.is_file():
            raw = path.read_bytes()
            if sha256(raw) != report.get("tokens_sha256") or json.loads(raw) != report.get("tokens"):
                fatal.append(f"{name} tokens.json does not match probe.json tokens/tokens_sha256")
    if not reference.get("text_sha256"):
        warnings.append("Token stream came from --tokens, not --text; NLL/PPL are meaningful only for real text")
    return fatal, failures, warnings


def read_vector(directory, step, vocab, use_numpy):
    path = directory / step["logits"]
    # Vectors must be immediate evidence files of this run, never unrelated files.
    if path.resolve().parent != directory.resolve():
        raise ValueError("Logit vector must be inside its run directory")
    raw = path.read_bytes()
    if step.get("count") != vocab or len(raw) != vocab * 4:
        raise ValueError("Vector length differs from the operation count or vocabulary")
    return raw, decode_vector(raw, use_numpy)


def self_consistency(directory, rows, file_key, use_numpy):
    """KL(first || later) between one run's vectors that end at the same cache position.

    Prefill, chunks, decode and refill rebuild identical contexts, so a correct cache gives
    the same distribution on every path. A run that disagrees with itself (for example a
    chunked prefill that loses recurrent state) cannot qualify a candidate, even when the
    candidate repeats the same defect and every reference/candidate gate passes.
    """
    groups = {}
    for row in rows:
        groups.setdefault(row["target_position"], []).append(row)
    pairs = []
    for group in groups.values():
        if len(group) < 2:
            continue
        first = decode_vector((directory / group[0][file_key]).read_bytes(), use_numpy)
        for row in group[1:]:
            other = decode_vector((directory / row[file_key]).read_bytes(), use_numpy)
            kl = vector_metrics(first, other, None, use_numpy)["kl_reference_to_candidate"]
            pairs.append(dict(index=row["index"], operation=row["operation"], first_index=group[0]["index"],
                              target_position=row["target_position"], kl=kl))
    pairs.sort(key=lambda pair: pair["index"])
    phases = {}
    for pair in pairs:
        phase = phases.setdefault(pair["operation"], dict(count=0, max_kl=None))
        phase["count"] += 1
        if phase["max_kl"] is None or pair["kl"] > phase["max_kl"]:
            phase.update(max_kl=pair["kl"], max_step=pair["index"], first_step=pair["first_index"])
    return dict(pairs=pairs, phases=phases)


def compare(reference_path, candidate_path, *, max_ppl_ratio=DEFAULT_MAX_PPL_RATIO, max_mean_kl=DEFAULT_MAX_MEAN_KL,
            min_top1=DEFAULT_MIN_TOP1, gate_scope="all", per_phase_gates=False, allow_different_weights=False,
            use_numpy=None, self_check=False):
    if use_numpy is None:
        use_numpy = np is not None
    if use_numpy and np is None:
        raise ValueError("numpy was requested but is not importable")
    if gate_scope not in SCOPES:
        raise ValueError("Unknown gate scope: " + str(gate_scope))
    ref_dir, ref_probe, reference, ref_raw = resolve_run(reference_path)
    cand_dir, cand_probe, candidate, cand_raw = resolve_run(candidate_path)
    fatal, failures, warnings = validate_pair(reference, candidate, ref_dir, cand_dir, allow_different_weights)
    report = dict(schema_version=1, status="running", mode="self-check" if self_check else "compare",
                  backend=dict(name="numpy", version=np.__version__) if use_numpy else dict(name="pure-python"),
                  comparator_sha256=sha256(Path(__file__).read_bytes()), target_mapping=TARGET_MAPPING,
                  reference=describe_run(ref_dir, ref_probe, reference, ref_raw),
                  candidate=describe_run(cand_dir, cand_probe, candidate, cand_raw),
                  thresholds=dict(max_ppl_ratio=max_ppl_ratio, max_mean_kl=max_mean_kl, min_top1=min_top1,
                                  gate_scope=gate_scope, per_phase_gates=per_phase_gates),
                  failures=failures, warnings=warnings, vectors=[],
                  limitations=["Teacher-forced next-token scoring of one fixed text; not task, long-context or throughput qualification.",
                               "Only last-position logits are saved, so a prefill scores one token; raise the probe's --decode-count to score more text.",
                               "Per-phase gates on one-vector phases (prefill, reset-prefill) gate a single token's NLL and are noisy.",
                               "Different phases rescore the same target positions; 'all' weights them by occurrence, 'unique_targets' once each.",
                               "Reference/candidate gates cannot see a defect both runs share; self_consistency warnings flag a run whose "
                               "chunk/refill/reset paths disagree with its own prefill/decode at the same cache position.",
                               "Checkpoint and build identities must be verified separately; weights_id is caller-supplied."])
    if fatal:
        failures[:0] = fatal
        report["status"] = "failed"
        return report
    tokens, vocab = reference["tokens"], reference["vocab_size"]
    ref_steps, cand_steps = reference["steps"], candidate["steps"]
    mismatched = set()
    for index, (left, right) in enumerate(zip(ref_steps, cand_steps)):
        if any(left.get(field) != right.get(field) for field in ("operation", "tokens", "count")) or \
                (left.get("logits") is None) != (right.get("logits") is None):
            mismatched.add(index)
            failures.append(f"Step {index}: operation, input, vocabulary count or vector presence differs")
    targets, mapping_failures = teacher_targets(tokens, ref_steps)
    failures.extend(mapping_failures)
    for target in targets:
        index = target["index"]
        if index in mismatched:
            continue
        left, right = ref_steps[index], cand_steps[index]
        try:
            ref_bytes, ref_vector = read_vector(ref_dir, left, vocab, use_numpy)
            cand_bytes, cand_vector = read_vector(cand_dir, right, vocab, use_numpy)
            row = dict(target, reference_file=left["logits"], candidate_file=right["logits"],
                       reference_sha256=sha256(ref_bytes), candidate_sha256=sha256(cand_bytes),
                       bitwise_equal=ref_bytes == cand_bytes)
            row.update(vector_metrics(ref_vector, cand_vector, target["target_token"], use_numpy))
            for name, step, argmax in (("reference", left, row["reference_argmax"]), ("candidate", right, row["candidate_argmax"])):
                if "argmax" in step and step["argmax"] != argmax:
                    raise ValueError(f"{name} recorded argmax {step['argmax']} differs from its vector file ({argmax})")
            report["vectors"].append(row)
        except (OSError, ValueError, KeyError, TypeError) as error:
            failures.append(f"Step {index} ({target['operation']}): {error}")
    rows = report["vectors"]
    if not rows:
        failures.append("No logit vectors compared")
    first_seen, unique = set(), []
    for row in rows:
        if row["target_token"] is not None and row["target_position"] not in first_seen:
            first_seen.add(row["target_position"])
            unique.append(row)
    report["aggregates"] = dict(all=summarize(rows), unique_targets=summarize(unique))
    report["phases"] = {phase: summarize([row for row in rows if row["operation"] == phase])
                        for phase in dict.fromkeys(row["operation"] for row in rows)}
    if len(unique) < MIN_UNIQUE_TARGETS_WARNING:
        warnings.append(f"Only {len(unique)} unique target tokens were scored; perplexity is a small-sample estimate")
    report["self_consistency"] = {}
    runs = [("reference", ref_dir, "reference_file")]
    if cand_dir.resolve() != ref_dir.resolve():
        runs.append(("candidate", cand_dir, "candidate_file"))
    for name, directory, file_key in runs:
        try:
            consistency = self_consistency(directory, rows, file_key, use_numpy)
        except (OSError, ValueError) as error:
            failures.append(f"{name} self-consistency check failed: {error}")
            continue
        report["self_consistency"][name] = consistency
        for phase, summary in consistency["phases"].items():
            if summary["max_kl"] > max_mean_kl:
                warnings.append(f"{name} run disagrees with itself: {phase} step {summary['max_step']} vs step "
                                f"{summary['first_step']} at the same cache position has KL {summary['max_kl']:.4g} > "
                                f"{max_mean_kl}; that cache path is suspect, and matching runs share the defect")
    report["gates"] = evaluate_gates(report["aggregates"][gate_scope], max_ppl_ratio, max_mean_kl, min_top1, gate_scope)
    if per_phase_gates:
        report["phase_gates"] = {}
        for phase, summary in report["phases"].items():
            gates = evaluate_gates(summary, max_ppl_ratio, max_mean_kl, min_top1, phase)
            if not summary["scored_vectors"]:  # e.g. a phase whose only vector is past the stream end
                gates["max_ppl_ratio"].update(passed=True, reason="Not applicable: no scored vectors in phase")
            report["phase_gates"][phase] = gates
    gate_failures = [f"Gate {name} ({gate['scope']}) failed" for name, gate in report["gates"].items() if not gate["passed"]]
    for phase, gates in report.get("phase_gates", {}).items():
        gate_failures += [f"Gate {name} ({phase}) failed" for name, gate in gates.items() if not gate["passed"]]
    failures.extend(gate_failures)
    if self_check:
        violations = [row["index"] for row in rows if not (
            row["bitwise_equal"] and row["kl_reference_to_candidate"] == 0 and row["kl_candidate_to_reference"] == 0
            and row["max_abs_logit_diff"] == 0 and row["argmax_match"] and row["top5_overlap"] == 1
            and row.get("nll_delta", 0) == 0)]
        overall = report["aggregates"]["all"]
        exact = overall.get("ppl_ratio") == 1.0 and overall.get("mean_kl_reference_to_candidate") == 0
        report["self_check"] = dict(passed=bool(rows) and not violations and exact, violating_steps=violations)
        if not report["self_check"]["passed"]:
            failures.append("Self-check invariant violated (expected KL 0, PPL ratio 1, identical top-k)")
    report["status"] = "failed" if failures else "passed"
    return report


def format_number(value, digits=4):
    if value is None:
        return "-"
    if value == 0 or 1e-3 <= abs(value) < 1e6:
        return f"{value:.{digits}f}"
    return f"{value:.{digits - 1}e}"


def table(report):
    lines = [f"{'scope':<16}{'vecs':>6}{'scored':>8}{'ref_ppl':>12}{'cand_ppl':>12}{'ratio':>10}"
             f"{'mean_kl':>12}{'max_kl':>12}{'top1':>8}{'top5':>8}{'max|dlogit|':>13}"]
    scopes = list(report.get("phases", {}).items()) + [
        (name, summary) for name, summary in report.get("aggregates", {}).items()]
    for name, summary in scopes:
        kl = (summary.get("distributions") or {}).get("kl_reference_to_candidate") or {}
        diff = (summary.get("distributions") or {}).get("max_abs_logit_diff") or {}
        lines.append(f"{name:<16}{summary.get('vectors', 0):>6}{summary.get('scored_vectors', 0):>8}"
                     f"{format_number(summary.get('reference_ppl')):>12}{format_number(summary.get('candidate_ppl')):>12}"
                     f"{format_number(summary.get('ppl_ratio'), 5):>10}"
                     f"{format_number(summary.get('mean_kl_reference_to_candidate')):>12}{format_number(kl.get('max')):>12}"
                     f"{format_number(summary.get('top1_agreement'), 3):>8}{format_number(summary.get('mean_top5_overlap'), 3):>8}"
                     f"{format_number(diff.get('max')):>13}")
    comparison = {"max_ppl_ratio": "<=", "max_mean_kl": "<=", "min_top1": ">="}
    for name, gate in report.get("gates", {}).items():
        lines.append(f"gate {name} ({gate['scope']}): {format_number(gate['value'], 6)} {comparison[name]} "
                     f"{gate['threshold']} {'PASS' if gate['passed'] else 'FAIL'}"
                     + (f" ({gate['reason']})" if gate.get("reason") else ""))
    for phase, gates in report.get("phase_gates", {}).items():
        for name, gate in gates.items():
            lines.append(f"gate {name} ({phase}): {format_number(gate['value'], 6)} {comparison[name]} "
                         f"{gate['threshold']} {'PASS' if gate['passed'] else 'FAIL'}")
    for message in report.get("warnings", []):
        lines.append("warning: " + message)
    for message in report.get("failures", []):
        lines.append("failure: " + message)
    backend = " ".join(filter(None, (report["backend"]["name"], report["backend"].get("version"))))
    lines.append(f"status: {report['status']} ({report['mode']}; {backend})")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reference", nargs="?", type=Path, help="Reference probe directory (or its probe.json)")
    parser.add_argument("candidate", nargs="?", type=Path, help="Candidate probe directory (or its probe.json)")
    parser.add_argument("--self-check", type=Path, metavar="REF",
                        help="Compare REF with itself; requires KL 0, PPL ratio 1 and identical top-k")
    parser.add_argument("--output", type=Path, required=True, help="JSON report path")
    parser.add_argument("--max-ppl-ratio", type=float, default=DEFAULT_MAX_PPL_RATIO,
                        help="Maximum candidate/reference perplexity ratio (default %(default)s)")
    parser.add_argument("--max-mean-kl", type=float, default=DEFAULT_MAX_MEAN_KL,
                        help="Maximum mean KL(reference||candidate) in nats (default %(default)s)")
    parser.add_argument("--min-top1", type=float, default=DEFAULT_MIN_TOP1,
                        help="Minimum argmax agreement rate (default %(default)s)")
    parser.add_argument("--gate-scope", choices=SCOPES, default="all",
                        help="Vectors the gates use: every vector, or each target position once (default %(default)s)")
    parser.add_argument("--per-phase-gates", action="store_true",
                        help="Also apply every gate to each phase (prefill, decode, chunk, refill, reset-prefill); "
                             "prefill and reset-prefill hold one vector, so their ratio is a one-token estimate")
    parser.add_argument("--allow-different-weights", action="store_true",
                        help="Record a weights_id difference as a warning (e.g. quantization studies)")
    parser.add_argument("--no-numpy", action="store_true", help="Force the pure-Python implementation")
    args = parser.parse_args()
    if args.self_check is not None:
        if args.reference is not None or args.candidate is not None:
            parser.error("--self-check takes no positional runs")
        args.reference = args.candidate = args.self_check
    elif args.reference is None or args.candidate is None:
        parser.error("Need REFERENCE and CANDIDATE, or --self-check REF")
    if not math.isfinite(args.max_ppl_ratio) or args.max_ppl_ratio <= 0:
        parser.error("--max-ppl-ratio must be finite and positive")
    if not math.isfinite(args.max_mean_kl) or args.max_mean_kl < 0:
        parser.error("--max-mean-kl must be finite and nonnegative")
    if not math.isfinite(args.min_top1) or not 0 <= args.min_top1 <= 1:
        parser.error("--min-top1 must be between 0 and 1")
    try:
        report = compare(args.reference, args.candidate, max_ppl_ratio=args.max_ppl_ratio, max_mean_kl=args.max_mean_kl,
                         min_top1=args.min_top1, gate_scope=args.gate_scope, per_phase_gates=args.per_phase_gates,
                         allow_different_weights=args.allow_different_weights,
                         use_numpy=False if args.no_numpy else None, self_check=args.self_check is not None)
    except (OSError, ValueError) as error:  # missing, unreadable or malformed probe.json
        parser.error(f"Cannot read probe runs: {error}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(table(report))
    return int(report["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
