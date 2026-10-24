#!/usr/bin/env python3
"""Fail-closed local/VM guards for validate-muse-spark-final-candidate.sh.

No server is launched or stopped. Component checks and synthetic actual-layer
timings do not establish full-checkpoint quality or llama.cpp performance parity.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import statistics
import subprocess
import tarfile
import xml.etree.ElementTree as ET

GGML_REVISION = "a05c9922613c6a2230dbb1d0de679519ecafb4f0"
NATIVE_TESTS = {
    "resident-device-copy-cpu", "resident-device-copy-cuda", "attention-device-cache-cuda",
    "muse-spark-moe-cpu", "muse-spark-moe-cpu-reuse", "muse-spark-moe-cuda", "muse-spark-moe-cuda-multi",
    "muse-spark-moe-cuda-reuse", "muse-spark-moe-cuda-multi-reuse",
    "muse-spark-cpu-moe-scratch-off", "muse-spark-cpu-moe-scratch-on",
    "muse-spark-residual-cpu", "muse-spark-residual-cuda", "muse-spark-qk-prep-cpu", "muse-spark-qk-prep-cuda",
    "muse-spark-kv-2d-cpu", "muse-spark-kv-2d-cuda", "muse-spark-kv-2d-cuda-multi",
    "moe-fused-bias-alias-cpu", "moe-fused-bias-alias-cpu-reuse", "moe-fused-bias-alias-cuda",
    "muse-spark-norm-gate-cpu", "muse-spark-norm-gate-cuda", "muse-spark-norm-gate-cuda-multi",
    "paged-attention-lifecycle-cpu", "paged-attention-lifecycle-cuda", "paged-attention-lifecycle-cuda-multi"}
QK_METHODS = ("NativeCudaActualGeometryAndPackedStridesMatchIndependentAndUnfusedQK",
    "NativeCudaAlternatingRanksAndScratchReusePreserveQK",
    "NativeCudaFusedQKPreservesChunkPrefixSpeculativeVisionAndPackedLogits",
    "NativeCudaTwoRankFusedQKPreservesModelLogits")
KV_2D_METHODS = ("NativeCuda2DUploadsPreserveChunkPrefixAndSpeculativeLogits",
    "NativeCudaTwoRank2DUploadsPreserveChunkPrefixAndSpeculativeLogits")


NORM_CPU_METHODS = ("NativeCpuNormGateFreshReuseAndOriginalOpsMatch",
    "NativeCpuNormScratchPreservesChunkPrefixSpeculativeVisionAndPackedLogits",
    "UnsupportedBackendRetainsOriginalNorm")
NORM_CUDA_METHODS = ("NativeCudaNormGateFreshReuseAndOriginalOpsMatch",
    "NativeCudaTwoRankNormGateFreshReuseAndOriginalOpsMatch",
    "NativeCudaNormScratchPreservesChunkPrefixSpeculativeVisionAndPackedLogits",
    "NativeCudaTwoRankNormScratchPreservesModelLogits")
MANAGED_EXPECTED_COUNTS = {"cpu": 95, "cuda": 23}

def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_new(path, data):
    with Path(path).open("x", encoding="utf-8") as output:
        json.dump(data, output, indent=2, allow_nan=False)
        output.write("\n")


def fresh_evidence(raw, base=Path("/workspace/evidence")):
    requested, base = Path(raw), base.resolve()
    require(requested.is_absolute() and not any(c in str(raw) for c in "\r\n"), "Invalid absolute evidence path")
    resolved = requested.resolve()
    require(resolved != base and resolved.is_relative_to(base), "Evidence path must be below /workspace/evidence")
    require(not requested.exists() and not requested.is_symlink(), "Refusing existing evidence path")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.mkdir()  # Exclusive creation; never delete or reuse earlier proof.
    return resolved


def assert_idle(proc=Path("/proc")):
    require(proc.is_dir(), "Linux /proc required; process state unavailable")
    for item in proc.glob("[0-9]*"):
        try:
            argv = (item / "cmdline").read_bytes().split(b"\0")
        except FileNotFoundError:
            continue  # Process exited while enumerating.
        except OSError as error:
            raise ValueError(f"Cannot inspect process {item.name}") from error
        names = {value.replace(b"\\", b"/").rsplit(b"/", 1)[-1].lower() for value in argv}
        require(not names.intersection({b"llama-server", b"llama-server.exe", b"tensorsharp.server.host",
            b"tensorsharp.server.host.dll", b"tensorsharp.server.host.exe"}), "Active full server: " + item.name)


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args])


def verify_upstream(root):
    upstream = root / "ExternalProjects/ggml"
    require(git(upstream, "rev-parse", "HEAD").decode().strip() == GGML_REVISION, "Wrong ggml revision")
    require(not git(upstream, "status", "--porcelain", "--untracked-files=all").strip(), "Modified upstream ggml")


BUILD_INPUT_SUFFIXES = {".cs", ".fs", ".vb", ".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx",
    ".cu", ".cuh", ".m", ".mm", ".metal", ".cmake", ".csproj", ".fsproj", ".vbproj", ".vcxproj",
    ".props", ".targets", ".resx", ".razor", ".cshtml"}
BUILD_INPUT_NAMES = {"cmakelists.txt", "global.json", "nuget.config", "packages.lock.json"}
GENERATED_COMPONENTS = {".git", "bin", "obj", "__pycache__", "node_modules"}
EXCLUDED_SOURCE_TREES = ("ExternalProjects", "artifacts", "docs/validation",
    "TensorSharp.GGML.Native/build", "TensorSharp.GGML.Native/build-windows", "TensorSharp.GGML.Native/build-ios",
    "TensorSharp.Backends.MLX/Native/build", "TensorSharp.Backends.MLX/Native/dist", "TensorAgent/python-runtime")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate manifest key: " + key)
        result[key] = value
    return result


def applied_manifest(path):
    # Read/hash the SAME bytes; the archive checkout intentionally has no .git.
    raw = Path(path).read_bytes()
    data = json.loads(raw, object_pairs_hook=unique_object)
    require(data.get("hash_basis") == "archive_plus_working_changes", "Require --applied-working-tree source manifest")
    require(data.get("original_clean") is False, "Expected reviewed working changes; no clean-source claim")
    revision = data.get("revision", "")
    require(type(revision) is str and len(revision) == 40 and all(c in "0123456789abcdef" for c in revision),
        "Invalid source revision")
    require(type(data.get("files")) is dict and data["files"], "Applied source manifest has no files")
    return data, hashlib.sha256(raw).hexdigest()


def source_path(root, name):
    require(type(name) is str and name and not any(ord(c) < 32 for c in name) and "\\" not in name
        and ":" not in name and all(part not in ("", ".", "..") for part in name.split("/")),
        "Unsafe manifest source path: " + repr(name))
    root = Path(root).resolve()
    path = root.joinpath(*name.split("/"))
    require(path.resolve().is_relative_to(root), "Manifest source path escapes root: " + name)
    cursor = root
    for part in name.split("/"):
        cursor /= part
        require(not cursor.is_symlink(), "Manifest source path is a symlink: " + name)
    require(path.is_file() and stat.S_ISREG(path.stat().st_mode), "Missing/nonregular manifest source file: " + name)
    return path


def excluded_tree(name):
    return (any(part.lower() in GENERATED_COMPONENTS for part in name.split("/")) or
        any(name == prefix or name.startswith(prefix + "/") for prefix in EXCLUDED_SOURCE_TREES))


def check_unlisted_build_inputs(root, expected):
    # This detects stale/unlisted files that common CMake/MSBuild source globs
    # could compile. It is not a claim of hermetic dependency/build coverage.
    def walk_error(error):
        raise ValueError("Cannot inspect source inventory: " + str(error)) from error
    root = Path(root).resolve()
    for directory, children, filenames in os.walk(root, topdown=True, followlinks=False, onerror=walk_error):
        relative = Path(directory).relative_to(root)
        kept = []
        for child in children:
            name = (relative / child).as_posix()
            if excluded_tree(name):
                continue
            require(not (Path(directory) / child).is_symlink(), "Unlisted source directory symlink: " + name)
            kept.append(child)
        children[:] = kept
        for filename in filenames:
            name = (relative / filename).as_posix()
            path = Path(name)
            if path.suffix.lower() in BUILD_INPUT_SUFFIXES or path.name.lower() in BUILD_INPUT_NAMES:
                require(name in expected, "Unlisted compiled source/build input: " + name)


def source_files(root, manifest):
    files = {}
    for name, expected in sorted(manifest["files"].items()):
        require(type(expected) is str and len(expected) == 64 and all(c in "0123456789abcdef" for c in expected),
            "Invalid manifest source hash: " + repr(name))
        path = source_path(root, name)
        require(not excluded_tree(name), "Applied manifest includes generated/dependency tree: " + name)
        require(digest(path) == expected, "Applied manifest source hash mismatch: " + name)
        files[name] = expected
    check_unlisted_build_inputs(root, files)
    return files


def freeze(root, proof, parent_manifest):
    verify_upstream(root)
    parent_manifest = Path(parent_manifest).resolve()
    parent, parent_sha = applied_manifest(parent_manifest)
    files = source_files(root, parent)  # Verify EVERY supplied hash before any build/snapshot.
    coverage = {"inventory_basis": "root_supplied_applied_archive_manifest", "checked_files": len(files),
        "revision": parent["revision"], "hash_basis": parent["hash_basis"], "original_clean": parent["original_clean"],
        "source_clean_claim": False, "unexpected_build_input_check": "passed_outside_declared_generated_and_dependency_exclusions",
        "excluded_trees": list(EXCLUDED_SOURCE_TREES), "excluded_directory_components": sorted(GENERATED_COMPONENTS),
        "scope": "Exact supplied archive/working-change bytes plus unlisted compiled-input scan; TensorSharp Git is not required. ggml verified separately."}
    write_new(proof / "source-manifest.json", {"ggml_revision": GGML_REVISION, "coverage": coverage, "files": files})
    write_new(proof / "parent-manifest.json", {"path": str(parent_manifest), "sha256": parent_sha,
        "candidate_source_manifest_sha256": digest(proof / "source-manifest.json")})
    with tarfile.open(proof / "candidate-source.tar.gz", "x:gz") as archive:
        for name in files:
            archive.add(root / name, arcname=name, recursive=False)
    verify_sources(root, proof)


def verify_sources(root, proof):
    verify_upstream(root)
    pin = json.loads((proof / "parent-manifest.json").read_text())
    parent, parent_sha = applied_manifest(pin["path"])
    require(parent_sha == pin["sha256"], "Root final-source manifest changed")
    candidate_path = proof / "source-manifest.json"
    require(digest(candidate_path) == pin["candidate_source_manifest_sha256"], "Candidate source manifest changed")
    expected = json.loads(candidate_path.read_text())["files"]
    require(parent["files"] == expected, "Candidate inventory differs from root applied manifest")
    require(source_files(root, parent) == expected, "Candidate source files changed after freeze")
    binary_manifest = proof / "candidate-binary-manifest.json"
    if binary_manifest.exists():
        expected = json.loads(binary_manifest.read_text())
        require({str(path): digest(path) for path in candidate_binaries(proof)} == expected,
            "Candidate binary inventory changed after build")


def candidate_binaries(proof):
    native = proof / "native-build"
    return sorted(set([path for path in (proof / "managed").rglob("*") if path.is_file()] +
        [path for path in native.rglob("*") if path.is_file() and (".so" in path.name or
          (path.parent == native and path.name.startswith("GgmlOps")))]))


def pin_binaries(proof):
    require(digest(proof / "native-build/libGgmlOps.so") == digest(proof / "managed/libGgmlOps.so"), "Managed native library differs")
    require((proof / "managed/InferenceWeb.Tests.dll").is_file() and (proof / "managed/TensorSharp.Server.Host.dll").is_file(),
        "Required candidate managed assemblies missing")
    write_new(proof / "candidate-binary-manifest.json", {str(path): digest(path) for path in candidate_binaries(proof)})


def check_native(path):
    cases = ET.parse(path).getroot().findall(".//testcase")
    names = [case.attrib["name"] for case in cases]
    require(len(names) == len(set(names)) and set(names) == NATIVE_TESTS, "Missing/duplicate/unexpected native tests")
    for case in cases:
        require(case.find("failure") is None and case.find("error") is None and case.find("skipped") is None,
            "Native case failed or skipped: " + case.attrib["name"])
        require(case.get("status", "run") in ("run", "passed"), "Native case did not execute")
    return {"passed": len(cases), "failed": 0, "skipped": 0}


def check_managed(path, cuda):
    cases = ET.parse(path).getroot().findall(".//{*}UnitTestResult")
    require(cases and all(c.get("outcome") == "Passed" for c in cases), "Managed failure, skipped case or empty selection")
    names = [c.attrib["testName"] for c in cases]
    require(len(names) == len(set(names)), "Duplicate managed test result")
    required = (["MuseSparkQKPrepTests." + name for name in QK_METHODS] +
        ["MuseSparkKv2DUploadTests." + name for name in KV_2D_METHODS] +
        ["MuseSparkModelTests.NativeCudaCpuExpertsMatchIndependentDecoder",
         "MuseSparkResidualTests.NativeCudaResidualScratchReuseKeepsChangingInputsAndRetainedOutputs"] if cuda else
        ["MuseSparkQKPrepTests.NativeCpuFusedQKPreservesChunkPrefixSpeculativeVisionAndPackedLogits",
         "MuseSparkKv2DUploadTests.NativeCpu2DUploadsPreserveChunkPrefixAndSpeculativeLogits",
         "MuseSparkResidualTests.NativeCpuResidualScratchReuseKeepsChangingInputsAndRetainedOutputs"])
    required += ["MuseSparkNormScratchTests." + name for name in (NORM_CUDA_METHODS if cuda else NORM_CPU_METHODS)]
    require(all(any(name.endswith(suffix) for name in names) for suffix in required), "Required model/actual-QK/actual-2D/actual-norm execution test missing")
    return {"passed": len(cases), "failed": 0, "skipped": 0,
        "actual_qk_execution": "Model tests assert FusedQKPrepCallCount > 0 when packed opt-in executes and zero for default/fallback",
        "actual_2d_execution": "Model tests assert Kv2DUploadCallCount > 0 only for exact opt-in and exact chunk/prefix/speculative logit parity",
        "actual_norm_execution": "Norm tests require successful scratch reuse, original-op/fresh parity, default-off/fallback, per-head gates and complete-model logits"}


def check_tests(proof):
    result = {"native": check_native(proof / "native-regressions.xml"), "managed": {}}
    for flag in (0, 1):
        for lane in ("cpu", "cuda"):
            name = f"{lane}-arena{flag}"
            result["managed"][name] = check_managed(proof / (name + ".trx"), lane == "cuda")
            require(result["managed"][name]["passed"] == MANAGED_EXPECTED_COUNTS[lane],
                f"Unexpected {lane} test count; the frozen complete MuseSpark selection must run")
    result["scope"] = "Component and tiny model validation only; full-checkpoint quality/performance pending"
    write_new(proof / "component-validation.json", result)


def receipt(root, proof):
    verify_sources(root, proof)
    components = json.loads((proof / "component-validation.json").read_text())
    require(components["native"]["passed"] == len(NATIVE_TESTS), "Component validation missing")
    ab = json.loads((proof / "cpu-arena-ab-comparison.json").read_text())
    require(ab["status"] == "exact_outputs_and_disk_quiet_repeats_passed", "Exact A/B validation missing")
    require(len(ab["inputs"]) == 4, "Four verified A/B source reports required")
    for source in ab["inputs"]:
        require(digest(source["path"]) == source["sha256"], "A/B input report changed after comparison")
    library = proof / "native-build/libGgmlOps.so"
    require(digest(library) == digest(proof / "managed/libGgmlOps.so"), "Managed native library differs")
    require(digest(library) == json.loads(Path(ab["inputs"][0]["path"]).read_text())["native_library"]["sha256"],
        "Microbenchmark used a different native library")
    binaries = candidate_binaries(proof)
    require((proof / "managed/TensorSharp.Server.Host.dll") in binaries, "Candidate server assembly not built")
    parent = json.loads((proof / "parent-manifest.json").read_text())
    logs = sorted(proof.glob("*.log")) + sorted(proof.glob("*.trx")) + [proof / "native-regressions.xml"]
    write_new(proof / "build-record.json", {"status": "pass", "source_manifest_sha256": parent["sha256"],
        "source_manifest_path": parent["path"], "candidate_source_manifest_sha256": digest(proof / "source-manifest.json"),
        "source_coverage": json.loads((proof / "source-manifest.json").read_text())["coverage"],
        "server_assembly": str(proof / "managed/TensorSharp.Server.Host.dll"),
        "native_library_directory": str(proof / "native-build"),
        "binary_hashes": {str(path): digest(path) for path in binaries},
        "build_logs": {str(path): digest(path) for path in logs},
        "validation_reports": {str(proof / name): digest(proof / name) for name in ("component-validation.json", "cpu-arena-ab-comparison.json")},
        "scope": "Isolated candidate build/component checks/one-layer CPU A/B; no deployment or full-model quality/performance acceptance"})


def validate_ab_report(report, flag):
    require(report.get("status") == "numerics_passed", "Failed/incomplete microbenchmark")
    require(report["cpu_scratch_reuse"]["requested_value"] == str(flag), "Wrong process-startup flag")
    config = report["configuration"]
    require(config["backend"] == "cpu" and config["layers"] == [0] and config["rows"] == [1, 128, 256], "Wrong microbenchmark scope")
    require(config["threads"] == [32] and config["warmup"] >= 1 and config["repeats"] >= 3, "Wrong timing configuration")
    require(config["atol"] == 0 and config["rtol"] == 0, "Exact-output guard required")
    records = {}
    for result in report["results"]:
        key = (result["layer"], result["rows"], result["threads"])
        require(key not in records and result["status"] == "numerics_passed", "Duplicate/failed microbenchmark result")
        require(len(result["warmup"]) == config["warmup"] and len(result["measured"]) == config["repeats"], "Incomplete repeats")
        hashes = set()
        for call in result["warmup"] + result["measured"]:
            value = call["wall_ms"]
            require(type(value) in (int, float) and math.isfinite(value) and value > 0, "Invalid wall duration")
            sha = call.get("output_f32_sha256", "")
            require(len(sha) == 64 and all(c in "0123456789abcdef" for c in sha), "Missing output byte hash")
            hashes.add(sha)
            state = call["cpu_scratch_reuse"]
            require(state["reused_on_this_call"] is bool(flag) and state["successful_calls_delta"] == flag, "Requested arena path did not execute")
            require(state["before"] is not None and state["after"] is not None, "Missing native arena telemetry")
            require(all(type(snapshot[k]) is int and snapshot[k] >= 0 for snapshot in (state["before"], state["after"])
                for k in ("successful_calls", "allocated_bytes")), "Invalid native counter types")
            require(state["after"]["successful_calls"] - state["before"]["successful_calls"] == flag, "Inconsistent native arena counters")
            require(state["after"]["allocated_bytes"] > 0 if flag else state["after"]["allocated_bytes"] == 0, "Unexpected arena capacity")
        require(len(hashes) == 1, "Outputs differ within repeat sequence")
        for call in result["measured"]:
            require(all(type(call[k]) is int and call[k] >= 0 for k in ("read_bytes", "major_faults", "minor_faults")), "Fault counters unavailable/malformed")
            require(call["read_bytes"] == 0 and call["major_faults"] == 0 and call["cache_quiet"] is True, "Measured call was not disk quiet")
            require(type(call.get("minor_faults")) is int and call["minor_faults"] >= 0, "Minor-fault telemetry missing")
            before, after = (call["process_counters"][x] for x in ("before", "after"))
            require(all(type(snapshot[k]) is int and snapshot[k] >= 0 for snapshot in (before, after)
                for k in ("minor_faults", "major_faults")) and all(type(snapshot["io"]["read_bytes"]) is int and snapshot["io"]["read_bytes"] >= 0
                for snapshot in (before, after)), "Raw fault counters unavailable/malformed")
            require(after["minor_faults"] - before["minor_faults"] == call["minor_faults"] and
                after["major_faults"] == before["major_faults"] and after["io"]["read_bytes"] == before["io"]["read_bytes"], "Raw process counters disagree")
        records[key] = result
    require(set(records) == {(0, rows, 32) for rows in (1, 128, 256)}, "Missing row scenario")
    return records


def compare_ab(paths):
    require(len(paths) == 4, "Require four separate-process reports in flag0,flag1,flag1,flag0 order")
    reports = [json.loads(Path(path).read_text()) for path in paths]
    lanes = [validate_ab_report(report, flag) for report, flag in zip(reports, (0, 1, 1, 0))]
    first = reports[0]
    for report in reports[1:]:
        require(all(report[k] == first[k] for k in ("plan", "native_library", "source_sha256", "reader_source")), "Binary, weights, reader or source changed between processes")
        left, right = (dict(r["configuration"]) for r in (first, report))
        left.pop("output"); right.pop("output")
        require(left == right, "A/B inputs/configuration differ")
        left, right = (json.loads(json.dumps(r["capacity"])) for r in (first, report))
        for capacity in (left, right):
            capacity["environment"].pop("TS_MUSE_SPARK_REUSE_CPU_MOE_SCRATCH", None)
        require(left == right, "A/B CPU capacity or environment differs")
    comparisons = []
    for key in sorted(lanes[0]):
        records = [lane[key] for lane in lanes]
        require(all(record["input"] == records[0]["input"] for record in records), "Synthetic activations/routes differ")
        require(len({call["output_f32_sha256"] for record in records for call in record["warmup"] + record["measured"]}) == 1, "Flag0/flag1 output bytes differ")
        pairs = []
        for off, on in ((records[0], records[1]), (records[3], records[2])):
            medians = [statistics.median(c["wall_ms"] for c in record["measured"]) for record in (off, on)]
            faults = [statistics.median(c["minor_faults"] for c in record["measured"]) for record in (off, on)]
            pairs.append({"off_median_ms": medians[0], "on_median_ms": medians[1], "off_over_on": medians[0] / medians[1],
                          "off_median_minor_faults": faults[0], "on_median_minor_faults": faults[1]})
        comparisons.append({"layer": key[0], "rows": key[1], "threads": key[2], "order_reversed_pairs": pairs})
    return {"status": "exact_outputs_and_disk_quiet_repeats_passed", "comparisons": comparisons,
        "inputs": [{"path": str(path), "sha256": digest(path)} for path in paths],
        "limitations": ["One actual Q3 expert layer with synthetic activations/routes; no full model or llama.cpp performance claim.",
            "Bit-exact arena A/B output hashes; existing quantized CPU/CUDA numerical difference is outside this CPU-only comparison.",
            "Counter sampling overhead is outside native timing; minor faults are not measured memory bandwidth."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fresh", "idle", "freeze", "verify", "pin-binaries", "tests", "ab", "receipt"))
    parser.add_argument("--root", type=Path, default=Path("/workspace/TensorSharp"))
    parser.add_argument("--proof", type=Path)
    parser.add_argument("reports", nargs="*")
    args = parser.parse_intermixed_args()
    if args.command == "fresh": print(fresh_evidence(args.proof))
    elif args.command == "idle": assert_idle()
    elif args.command == "freeze":
        require(len(args.reports) == 1, "freeze requires root final-source manifest")
        freeze(args.root, args.proof, args.reports[0])
    elif args.command == "verify": verify_sources(args.root, args.proof)
    elif args.command == "pin-binaries": pin_binaries(args.proof)
    elif args.command == "tests": check_tests(args.proof)
    elif args.command == "receipt": receipt(args.root, args.proof)
    else: write_new(args.proof / "cpu-arena-ab-comparison.json", compare_ab(args.reports))


if __name__ == "__main__":
    main()
