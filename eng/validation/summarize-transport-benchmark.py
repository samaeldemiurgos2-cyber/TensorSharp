#!/usr/bin/env python3
"""Validate and summarize paired DistributedTransportProbe reports.

The manifest names every baseline/candidate repeat and node report explicitly,
plus observed route/workload conditions and optional source archive identities.
Input reports and samples remain immutable. Node timings stay separate. Summary
latencies are medians of repeat statistics, not pooled observations or confidence
intervals. Existing output files are never overwritten.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import tarfile


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_report(path, node, expected_cases, iterations, warmup):
    raw = path.read_bytes()
    report = json.loads(raw)
    require(report.get("schema_version") == 1, f"{path}: unsupported schema")
    require(report.get("status") == "passed", f"{path}: run did not pass")
    require(report.get("node_id") == node, f"{path}: node identity mismatch")
    args = report.get("arguments", [])
    require(len(args) % 2 == 0 and len(set(args[::2])) == len(args) // 2,
            f"{path}: malformed or duplicate arguments")
    options = dict(zip(args[::2], args[1::2]))
    require(options.get("--node-id") == str(node), f"{path}: argument node mismatch")
    require(int(options.get("--iterations", 100)) == iterations and
            int(options.get("--warmup", 10)) == warmup, f"{path}: iteration configuration mismatch")
    require(report.get("route") == options.get("--route", "unspecified"), f"{path}: route mismatch")
    require(report.get("started_utc") and report.get("finished_utc"), f"{path}: incomplete timestamps")
    for field in ("transport_sha256", "probe_sha256"):
        require(re.fullmatch(r"[0-9a-f]{64}", report.get(field, "")), f"{path}: invalid {field}")
    cases = {}
    for case in report.get("cases", []):
        key = (case["operation"], case["bytes"])
        require(key not in cases, f"{path}: duplicate case {key}")
        require(case.get("iterations") == iterations and case.get("warmup") == warmup,
                f"{path}: case configuration mismatch")
        samples = case.get("samples_ms", [])
        require(len(samples) == iterations and all(type(x) in (int, float) and math.isfinite(x) and x > 0 for x in samples),
                f"{path}: invalid or missing timing samples")
        median = statistics.median(samples)
        p95 = sorted(samples)[math.ceil(.95 * iterations) - 1]
        throughput = case["bytes"] / 1048576 / (median / 1000)
        for field, value in (("median_ms", median), ("p95_ms", p95), ("payload_mib_per_second", throughput)):
            require(math.isclose(case.get(field, float("nan")), value, rel_tol=1e-10, abs_tol=1e-10),
                    f"{path}: {field} disagrees with raw samples")
        cases[key] = dict(median_ms=median, p95_ms=p95, payload_mib_per_second=throughput)
    require(set(cases) == expected_cases, f"{path}: missing or unexpected benchmark cases")
    identity = {key: report[key] for key in ("node_id", "route", "started_utc", "finished_utc", "transport_sha256", "probe_sha256")}
    identity.update(file=path.name, sha256=sha256(raw), peers=options.get("--peers"))
    return identity, cases


def direction(values):
    """Only describe observed direction; do not imply statistical significance."""
    if all(value < 0 for value in values):
        return "lower_in_every_repeat"
    if all(value > 0 for value in values):
        return "higher_in_every_repeat"
    if all(value == 0 for value in values):
        return "equal_in_every_repeat"
    return "mixed_across_repeats"


def summarize(manifest_path):
    raw_manifest = manifest_path.read_bytes()
    manifest = json.loads(raw_manifest)
    root = manifest_path.parent
    require(manifest.get("schema_version") == 1, "Unsupported manifest schema")
    iterations, warmup = manifest["iterations"], manifest["warmup"]
    require(type(iterations) is int and iterations >= 3 and type(warmup) is int and warmup >= 1,
            "Invalid requested timing counts")
    nodes = manifest["nodes"]
    require(nodes == list(range(len(nodes))) and len(nodes) >= 2, "Need contiguous node IDs starting at zero")
    case_order = [(item["operation"], item["bytes"]) for item in manifest["cases"]]
    expected_cases = set(case_order)
    require(len(expected_cases) == len(case_order) and len(case_order) > 0, "Duplicate or empty cases")
    require(all(op in ("all_reduce", "broadcast_from_last") and type(size) is int and size > 0 and size % 4 == 0
                for op, size in case_order), "Invalid benchmark case")
    variants = manifest["variants"]
    require(len(variants) == 2 and len({v["label"] for v in variants}) == 2, "Need distinct baseline and candidate variants")
    repeats = [run["repeat"] for run in variants[0]["runs"]]
    require(len(repeats) >= 2 and len(set(repeats)) == len(repeats), "Need at least two unique paired repeats")
    inputs, all_cases, identities, used_files = [], {}, [], set()
    reference_route = None
    for variant in variants:
        label = variant["label"]
        require([run["repeat"] for run in variant["runs"]] == repeats, "Repeat IDs/order differ between variants")
        variant_hashes = set()
        for run in variant["runs"]:
            require(set(run["reports"]) == {str(node) for node in nodes}, "Missing or extra node report")
            for node in nodes:
                path = (root / run["reports"][str(node)]).resolve()
                require(path not in used_files, "Input report reused for multiple runs/nodes")
                used_files.add(path)
                identity, cases = load_report(path, node, expected_cases, iterations, warmup)
                route = (identity["route"], identity["peers"])
                if reference_route is None:
                    reference_route = route
                require(route == reference_route and identity["peers"] and len(identity["peers"].split(",")) == len(nodes),
                        "Route, peers or node count differs between reports")
                variant_hashes.add((identity["transport_sha256"], identity["probe_sha256"]))
                identity.update(variant=label, repeat=run["repeat"])
                inputs.append(identity)
                all_cases[(label, run["repeat"], node)] = cases
        require(len(variant_hashes) == 1, f"{label}: transport/probe binary changed between repeats or nodes")
        transport, probe = variant_hashes.pop()
        require(transport == variant["transport_sha256"] and probe == variant["probe_sha256"],
                f"{label}: measured binary identity differs from manifest")
        source = variant.get("source_archive")
        source_identity = None
        if source:
            archive = root / source["file"]
            require(sha256(archive.read_bytes()) == source["sha256"], f"{label}: source archive hash mismatch")
            with tarfile.open(archive) as package:
                for member, expected_hash in source["members"].items():
                    handle = package.extractfile(member)
                    require(handle is not None and sha256(handle.read()) == expected_hash,
                            f"{label}: source member hash mismatch: {member}")
            source_identity = source
        identities.append(dict(label=label, transport_sha256=transport, probe_sha256=probe,
                               source_revision=variant.get("source_revision"), source_archive=source_identity))
    rows = []
    baseline, candidate = [v["label"] for v in variants]
    for node in nodes:
        for operation, size in case_order:
            paired = []
            for repeat in repeats:
                before = all_cases[(baseline, repeat, node)][(operation, size)]
                after = all_cases[(candidate, repeat, node)][(operation, size)]
                paired.append(dict(repeat=repeat, baseline=before, candidate=after,
                    median_change_percent=100 * (after["median_ms"] / before["median_ms"] - 1),
                    p95_change_percent=100 * (after["p95_ms"] / before["p95_ms"] - 1)))
            aggregates = {}
            for metric in ("median_ms", "p95_ms"):
                before = statistics.median(pair["baseline"][metric] for pair in paired)
                after = statistics.median(pair["candidate"][metric] for pair in paired)
                field = "median_change_percent" if metric == "median_ms" else "p95_change_percent"
                aggregates["median_of_repeat_" + metric] = dict(baseline=before, candidate=after,
                    change_percent=100 * (after / before - 1), observed_direction=direction([pair[field] for pair in paired]))
            rows.append(dict(node_id=node, operation=operation, bytes=size, repeats=paired, aggregate=aggregates))
    return dict(schema_version=1, status="validated", manifest=dict(file=manifest_path.name, sha256=sha256(raw_manifest)),
        baseline=baseline, candidate=candidate, iterations_per_case=iterations, warmup_per_case=warmup,
        repeat_count=len(repeats), measured_case_count=len(inputs) * len(case_order),
        retained_timing_count=len(inputs) * len(case_order) * iterations,
        identities=identities, inputs=inputs, context=manifest["context"],
        post_measurement_changes=manifest["post_measurement_changes"],
        limitations=manifest["limitations"],
        statistic_definition="Per-run median and nearest-rank p95 are recomputed from samples; aggregate is the median of repeat statistics. Negative percentage means lower latency. Nodes are not pooled.",
        comparisons=rows)


def markdown(report):
    lines = ["# Paired multi-node transport measurements", "",
        f"Validated {report['repeat_count']} paired repeats, {report['iterations_per_case']} measured iterations and "
        f"{report['warmup_per_case']} warmups per operation/payload/node. Both nodes completed every payload check.", "",
        f"Candidate: **{report['candidate']}**. These measurements apply to the binary identities below, not to later builds.", "",
        report["statistic_definition"], "", "## Conditions and scope", ""]
    lines.extend("- " + value for value in report["context"].values())
    lines += ["", "## Exact measured identities", ""]
    for identity in report["identities"]:
        lines += [f"**{identity['label']}**", "", f"- Transport DLL SHA256: `{identity['transport_sha256']}`",
                  f"- Probe DLL SHA256: `{identity['probe_sha256']}`",
                  f"- Source revision: {identity['source_revision']}"]
        source = identity.get("source_archive")
        if source:
            lines.append(f"- Source archive: `{source['file']}`, SHA256 `{source['sha256']}`")
            lines.extend(f"- Archive member `{name}`: SHA256 `{digest}`" for name, digest in source["members"].items())
        lines.append("")
    lines += ["## Driver (node 0) aggregate", "", "Latency in milliseconds. Each entry is the median of repeat statistics; raw repeat results follow.", "",
              "| Operation | Bytes | Median baseline → candidate | Change | p95 baseline → candidate | Change | Median direction | p95 direction |",
              "|---|---:|---:|---:|---:|---:|---|---|"]
    for row in report["comparisons"]:
        if row["node_id"] != 0:
            continue
        med = row["aggregate"]["median_of_repeat_median_ms"]
        p95 = row["aggregate"]["median_of_repeat_p95_ms"]
        lines.append(f"| {row['operation']} | {row['bytes']} | {med['baseline']:.4f} → {med['candidate']:.4f} | {med['change_percent']:+.1f}% | "
                     f"{p95['baseline']:.4f} → {p95['candidate']:.4f} | {p95['change_percent']:+.1f}% | "
                     f"{med['observed_direction'].replace('_', ' ')} | {p95['observed_direction'].replace('_', ' ')} |")
    for node in sorted({row["node_id"] for row in report["comparisons"]}):
        lines += ["", f"## Node {node}: every repeat", "", "| Operation | Bytes | Repeat | Baseline median | Candidate median | Change | Baseline p95 | Candidate p95 | Change |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for row in report["comparisons"]:
            if row["node_id"] != node:
                continue
            for pair in row["repeats"]:
                before, after = pair["baseline"], pair["candidate"]
                lines.append(f"| {row['operation']} | {row['bytes']} | {pair['repeat']} | {before['median_ms']:.4f} | {after['median_ms']:.4f} | "
                             f"{pair['median_change_percent']:+.1f}% | {before['p95_ms']:.4f} | {after['p95_ms']:.4f} | {pair['p95_change_percent']:+.1f}% |")
    lines += ["", "## Changes after the measured candidate", ""]
    lines.extend("- " + item for item in report["post_measurement_changes"])
    lines += ["", "## Limitations", ""]
    lines.extend("- " + item for item in report["limitations"])
    lines += ["", "## Immutable reports", "", "| Variant | Repeat | Node | Input | SHA256 |", "|---|---:|---:|---|---|"]
    for item in report["inputs"]:
        lines.append(f"| {item['variant']} | {item['repeat']} | {item['node_id']} | `{item['file']}` | `{item['sha256']}` |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.markdown.exists() or args.output.resolve() == args.markdown.resolve():
        parser.error("Outputs must be fresh distinct files; preserve earlier evidence")
    try:
        report = summarize(args.manifest)
    except (ValueError, KeyError, OSError, TypeError) as error:
        parser.error(str(error))
    for path in (args.output, args.markdown):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.markdown.write_text(markdown(report), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "candidate", "repeat_count", "measured_case_count", "retained_timing_count")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
