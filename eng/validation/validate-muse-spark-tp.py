#!/usr/bin/env python3
"""Run explicit Muse Spark component gates against an already built Linux runtime.

This does not build, download a checkpoint, launch a server, or establish model
quality/performance. --list-lanes shows the exact inventories. The default runs
all required lanes (including currently failing regressions); a selected subset
is reported only as a subset. Diagnostic lanes are opt-in and never replace a
required regression. No skip/unavailable result counts as a pass.

Example (paths are supplied by the caller, not tied to a particular VM):
  python eng/validation/validate-muse-spark-tp.py --root /work/TensorSharp \
    --source-manifest /proof/source.json --build-record /proof/build.json \
    --runtime /proof/managed --native-build /proof/native --output /proof/new-tp \
    --dotnet /opt/dotnet/dotnet --devices 0,1,2,3,4,5,6,7,8

source.json contains a nonempty relative-path -> SHA256 "files" map. build.json
must have status "pass", its source_manifest_sha256, and an absolute-path ->
SHA256 "files" map covering the entire managed runtime and, for the native lane,
every selected native test executable and native-build/libGgmlOps.so. Supply
these build-time hashes; this runner does not manufacture a build receipt.
Source/binary bytes,
unchanged upstream ggml, native CTest commands and tool bytes are sealed before
execution and checked before/after each lane. Native and managed libGgmlOps.so
must be identical. New evidence directories are exclusive; failures are retained.
"""
import argparse
from collections import Counter
import contextlib
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import subprocess
import sys
import time
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
    "paged-attention-lifecycle-cpu", "paged-attention-lifecycle-cuda", "paged-attention-lifecycle-cuda-multi",
    "addmm-quant-cache-recovery-cpu", "addmm-quant-cache-recovery-cuda",
    "muse-spark-projection-precision-cpu", "muse-spark-projection-precision-cuda",
    "muse-spark-projection-precision-cuda-multi",
    "muse-spark-attention-precision-cpu", "muse-spark-attention-precision-cuda",
    "muse-spark-attention-precision-cuda-multi",
}
PREFIX = "InferenceWeb.Tests."
GPU_CLASS = PREFIX + "MuseSparkTensorParallelGpuTests."
CPU_CLASS = PREFIX + "MuseSparkTensorParallelTests."
OPTIMIZATIONS = {
    "TS_MUSE_SPARK_REUSE_NORM_SCRATCH": "1", "TS_MUSE_SPARK_KV_UPLOAD_2D": "1",
    "TS_MUSE_SPARK_REUSE_CPU_MOE_SCRATCH": "1", "TS_MUSE_SPARK_RELEASE_GPU_SOURCE_PAGES": "1",
    "TS_MUSE_SPARK_FUSE_RESIDUALS": "1", "TS_MUSE_SPARK_FUSE_QKVG": "0",
    "TS_MUSE_SPARK_FUSE_QK_PREP": "0", "TS_MUSE_SPARK_STREAM_EXPERTS": "0",
    "TS_MUSE_SPARK_TP_HOST_POOL": "1",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def lane(methods, devices=0, expected=None, environment=None, diagnostic=False):
    return {"methods": methods, "expected": expected or methods, "devices": devices,
            "environment": environment or {}, "diagnostic": diagnostic}


def lanes():
    cpu_methods = [CPU_CLASS + name for name in (
        "WholeAndUnevenPartitionsMatchScalarOracleAcrossChunksSnapshotsAndSpeculation",
        "QuantizedAttentionShardsAndExpertSlicesPreserveExactStoredWeights",
        "PublishedGeometryPartitionsCoverEveryHeadAndExpertExactlyOnce",
        "PackedUnevenRanksRetainIndependentPagedSequences",
        "RouteCompactionAvoidsForeignWorkAndDuplicateZeroFillers",
        "DescriptorCapacityCountsUnevenShardsAndRankZeroTrunk",
        "RejectsMoreRanksThanKvGroupsBeforeWeightMaterialization",
        "MixedCheckpointKvDefaultAndOwnedViewsSurviveTensorSharding")]
    contracts = [PREFIX + "GgmlTensorParallelPlanSlotContractTests." + name for name in (
        "EveryNativePlanSlotParameter_IsAnArray_SoNonTpCallersPassNull",
        "PlanSlotParameters_Exist_SoThisGuardCannotSilentlyPass")]
    cpu_cases = cpu_methods[3:] + contracts
    cpu_cases += [cpu_methods[0] + f"(degree: {degree}, halfKv: {half})"
                  for degree, half in [(2, "False"), (3, "False"), (5, "False"), (8, "False"),
                                       (9, "False"), (3, "True"), (9, "True")]]
    cpu_cases += [cpu_methods[1] + f"(cpuExperts: {value})" for value in ("False", "True")]
    cpu_cases += [cpu_methods[2] + f"(degree: {degree})" for degree in (2, 3, 5, 8, 9)]
    result = {"native": lane([], 2, sorted(NATIVE_TESTS)),
              "cpu-tp": lane(cpu_methods + contracts, expected=cpu_cases)}
    for degree, word in ((2, "Two"), (3, "ThreeUneven"), (5, "FiveUneven"), (8, "Eight"), (9, "NineUneven")):
        result[f"q8-ranks-{degree}"] = lane(
            [GPU_CLASS + word + "RanksMatchNativeQuantizedOracleAcrossCacheModes"], degree)
        result[f"f32-ranks-{degree}"] = lane(
            [GPU_CLASS + word + "RanksFloatWeightsMatchIndependentScalarOracle"], degree)
    result.update({
        "vision-ranks-3": lane([GPU_CLASS + "VisionEmbeddingRowsCrossTpPrefillChunks"], 3),
        "expert-residency-ranks-2": lane(
            [GPU_CLASS + "FloatExpertPlaneResidencyReleasesAndReloadsAcrossPlacementModes"], 2),
        "quantized-residency-ranks-2": lane(
            [PREFIX + "MuseSparkQuantizedResidencyTests.QuantizedOpaqueKeysReleaseAndReloadAcrossPlacementModes"], 2),
        "packed-ranks-5": lane([GPU_CLASS + "QuantizedPackedPrefillDecodeAndNewArrivalMatchSameScheduleOnOneGpu"], 5),
        "views-ranks-2": lane([PREFIX + "GgmlTensorParallelViewTests." + name for name in (
            "OffsetViewsPreserveSentinelsWithDifferentBackingStorageSizes",
            "UnequalLogicalCountsAndStridedViewsAreRejectedBeforeWriting")], 2),
        "collective-ranks-2": lane([GPU_CLASS + "LargeCollectivePreservesFloat32ValuesBelowBfloat16Resolution"], 2),
        "collective-ranks-9": lane([GPU_CLASS + "NineRankLargeCollectivePreservesFloat32ValuesBelowBfloat16Resolution"], 9),
        "optimized-ranks-9": lane([GPU_CLASS + "NineRanksOptimizedProfileMatchesNativeQuantizedOracleAcrossCacheModes"],
                                  9, environment=dict(OPTIMIZATIONS)),
        "residual-off-ranks-9": lane([GPU_CLASS + "NineUnevenRanksMatchNativeQuantizedOracleAcrossCacheModes"],
                                     9, environment=dict(OPTIMIZATIONS, TS_MUSE_SPARK_FUSE_RESIDUALS="0")),
    })
    for degree, word in ((5, "Five"), (9, "Nine")):
        result[f"head64-ranks-{degree}"] = lane([PREFIX + "MuseSparkTensorParallelFlashAttentionTests." +
            word + "RanksPublishedHeadGeometryPackedF32MatchesOrdinaryAndScalar"], degree)
    result["cpu-attention-f32-binding"] = lane([
        PREFIX + "MuseSparkAttentionF32BindingTests." + name for name in (
            "OrdinaryRejectsNarrowedStridedAliasedAndWrongBackendBuffers",
            "PagedRejectsUnsafeLengthsAndAliasingBeforeNativeAccess")])
    for rows, word in ((32, "ThirtyTwo"), (256, "TwoHundredFiftySix")):
        result[f"attention-{rows}"] = lane([PREFIX + "MuseSparkPrefillAttentionGeometryTests.NativeCuda" +
            word + "QueryPrefillMatchesPagedAndDoubleReference"], 1)
    result["diagnostic-packed-head"] = lane([PREFIX + "MuseSparkPackedHeadNumericalControlTests." +
        "PackedFloatHeadBatchShapesRetainRawSingleAndParallelControls"], 5, diagnostic=True)
    result["diagnostic-quantization"] = lane([PREFIX + "MuseSparkTensorParallelNumericalControlTests." +
        "SameBackendQuantizedAndDequantizedControlsIsolateTwoRankError"], 2, diagnostic=True)
    packed = PREFIX + "MuseSparkPackedHeadNumericalControlTests."
    result["cpu-packed-controls"] = lane([packed + name for name in (
        "ManagedCpuNarrowPackedMixedSequencesRetainIndependentScalarOracle",
        "ManagedCpuPackedFirstBatchStageReplayMatchesForwardAndScalarProjection")])
    result["narrow-packed-ranks-5"] = lane(
        [packed + "NarrowPackedFloatMixedSequencesRetainIndependentScalarOracle"], 5)
    result["diagnostic-packed-stages"] = lane(
        [packed + "PackedFloatFirstBatchRetainsLayerAndSameInputProjectionControls"], 5, diagnostic=True)
    timing = PREFIX + "MuseSparkPhaseTimingTests."
    timing_methods = [timing + name for name in (
        "CallerIntervalsPartitionWallTimeAndOverlappingRanksStaySeparate",
        "EnabledAndDisabledOrdinaryExecutionHasIdenticalLogitsAndSeparateCallRecords",
        "PackedExecutionKeepsIdenticalCapturedLogitsAndUnknownSingleCachePosition",
        "CleanupExceptionsMarkTheCompletedRecordFailed")]
    timing_cases = [timing_methods[0], timing_methods[2]]
    timing_cases += [timing_methods[1] + f"(degree: {degree})" for degree in (1, 3)]
    timing_cases += [timing_methods[3] + f"(outerFinally: {value})" for value in ("False", "True")]
    result["cpu-phase-timing"] = lane(timing_methods, expected=timing_cases)
    pool = PREFIX + "MuseSparkHostPoolTests."
    pool_methods = [pool + name for name in (
        "OnlyExplicitCudaPoolOptInChangesRetention",
        "NineRankBurstsReuseBoundedBlocksWhileBaselineReturnsOverflow",
        "CapAndPerBlockLimitRemainBounded",
        "BestFitReturnsFullOriginalExtentAndCloseRejectsLateRetention",
        "ConcurrentReturnsAfterCloseLeaveNoRetainedOrOutstandingBlocks")]
    pool_cases = pool_methods[1:] + [
        pool_methods[0] + f"(backend: {backend}, expected: {count})"
        for backend, count in (("Cuda", 72), ("Vulkan", 8), ("Cpu", 32), ("Metal", 32))]
    result["cpu-host-pool"] = lane(pool_methods, expected=pool_cases)
    placement = PREFIX + "MuseSparkExpertPlacementTests."
    placement_methods = [placement + name for name in (
        "UniformFitIsPreservedDespiteDifferentFreeMemory",
        "AsymmetricSaturatedCapacityUsesEveryExpertExactlyOnce",
        "BalancedPlanMaximizesMinimumRemainingBytesAgainstExhaustiveOracle",
        "UnknownRankFallsBackUniformlyAndPreservesKnownRankRefusal",
        "InsufficientCapacityRefusesWithoutReturningPartialOwnership",
        "NoGpuExpertsUsesUniformPlanAndDoesNotInventExpertBytes",
        "PlanCopiesInputsAndExposesReadOnlyCounts",
        "PublishedQ3CpuEightSnapshotKeepsRankZeroHeadroom",
        "AsymmetricCpuPlacementMatchesOracleAndPreservesOffload")]
    placement_cases = placement_methods[:-1] + [
        placement_methods[-1] + f"(quantized: {quantized}, cpuExperts: {offload})"
        for quantized in ("False", "True") for offload in ("False", "True")]
    result["cpu-expert-placement"] = lane(placement_methods, expected=placement_cases)
    result["expert-placement-f32-ranks-3"] = lane(
        [placement + "AsymmetricGpuFloatPlacementMatchesIndependentScalar"], 3)
    result["expert-placement-q8-ranks-3"] = lane(
        [placement + "AsymmetricGpuQuantizedPlacementMatchesNativeAcrossOffloadAndKvModes"], 3)
    qk = PREFIX + "MuseSparkTensorParallelQKPrepTests."
    qk_methods = [qk + name for name in (
        "RankPackingInitializesEveryUploadedByteAndPreservesPaddedInputs",
        "InvalidProjectionGeometryFailsBeforeAnyInputMutation",
        "UnsupportedCpuTensorParallelRetainsIndependentPackedScalarPath")]
    qk_cases = qk_methods[1:] + [
        qk_methods[0] + f"(rows: {rows}, kvHeads: {heads})"
        for rows, heads in ((1, 1), (2, 2), (9, 1), (128, 2), (256, 1))]
    qk_environment = {"TS_MUSE_SPARK_FUSE_QK_PREP": "1"}
    result["cpu-qk-prep"] = lane(qk_methods, expected=qk_cases, environment=dict(qk_environment))
    # Keep the native CPU prerequisite separate from the seven managed packing
    # and fallback cases: unavailable native execution must not disappear.
    result["cpu-qk-prep-native"] = lane(
        [qk + "NativeCpuLocalHeadGroupsRetainPrimitiveOracleAcrossRowsAndPositions"],
        environment=dict(qk_environment))
    result["qk-prep-ranks-2"] = lane(
        [qk + "NativeCudaLocalHeadGroupsRetainPrimitiveOracleAcrossRanksRowsAndPositions"],
        2, environment=dict(qk_environment))
    result["qk-cache-ranks-9"] = lane(
        [qk + "FusedRankPreparationPreservesQuantizedCacheOffloadAndScalarControls"],
        9, environment=dict(qk_environment))
    result["qk-head64-ranks-9"] = lane(
        [qk + "FusedRankPreparationPreservesPublishedHeadGeometryAndPackedSchedules"],
        9, environment=dict(qk_environment))
    return result


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "Duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8-sig"), object_pairs_hook=unique)


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def regular(path):
    path = Path(path)
    require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)),
            "Missing or symlink file: " + str(path))
    return path


def hash_map(files):
    require(isinstance(files, dict) and files, "Missing file hashes")
    require(len({key.casefold() for key in files}) == len(files), "Case-colliding file paths")
    for name, digest in files.items():
        require(isinstance(name, str) and isinstance(digest, str) and re.fullmatch("[0-9a-f]{64}", digest),
                "Invalid file hash")
    return files


def verify_hashes(files):
    for path, expected in hash_map(files).items():
        require(sha(regular(path)) == expected, "Changed pinned file: " + path)


def merge_pins(target, extra):
    for path, digest in extra.items():
        require(path not in target or target[path] == digest, "Conflicting file seal: " + path)
        target[path] = digest


def source_pins(root, manifest):
    files = hash_map(manifest["files"])
    result = {}
    for name, digest in files.items():
        relative = PurePosixPath(name)
        require(not relative.is_absolute() and "\\" not in name and ":" not in name
                and all(part not in ("", ".", "..") for part in name.split("/")),
                "Unsafe source path: " + name)
        result[str(root.joinpath(*relative.parts))] = digest
    verify_hashes(result)
    return result


def runtime_files(runtime):
    require(runtime.is_dir(), "Missing runtime")
    result = {}
    for path in runtime.rglob("*"):
        require(not path.is_symlink(), "Symlink runtime member")
        if path.is_file():
            result[str(regular(path))] = sha(path)
    return result


def bind_build(root, runtime, manifest_path, build_path):
    manifest, build = read_json(manifest_path), read_json(build_path)
    require(build.get("status") == "pass" and build.get("source_manifest_sha256") == sha(manifest_path),
            "Build/source identity mismatch")
    sources = source_pins(root, manifest)
    recorded = hash_map(build["files"])
    verify_hashes(recorded)
    actual = runtime_files(runtime)
    expected = {name: digest for name, digest in recorded.items()
                if Path(name).is_relative_to(runtime)}
    require(actual == expected and str(runtime / "InferenceWeb.Tests.dll") in actual
            and str(runtime / "libGgmlOps.so") in actual, "Runtime inventory differs from build receipt")
    pins = dict(sources)
    merge_pins(pins, recorded)
    for path in (manifest_path, build_path):
        pins[str(path)] = sha(regular(path))
    return pins, actual


def check_cases(cases, expected, returncode):
    names = [case["name"] for case in cases]
    require(returncode == 0, "Test process failed")
    require(len(names) == len(set(names)) and Counter(names) == Counter(expected),
            "Missing, duplicate or unexpected test cases")
    require(all(case["outcome"] == "Passed" for case in cases),
            "Failed or unavailable test; skips are not passes")
    return {"passed": len(cases), "failed": 0, "skipped": 0}


def read_cases(path, native=False):
    doc = ET.parse(path)
    if native:
        return [{"name": row.attrib["name"], "outcome":
                 "Failed" if row.find("failure") is not None or row.find("error") is not None else
                 "NotExecuted" if row.find("skipped") is not None or row.get("status") in ("notrun", "disabled") else "Passed"}
                for row in doc.findall(".//testcase")]
    return [{"name": row.attrib["testName"], "outcome": row.attrib["outcome"]}
            for row in doc.findall(".//{http://microsoft.com/schemas/VisualStudio/TeamTest/2010}UnitTestResult")]


def select_lanes(raw):
    available = lanes()
    selected = [name for name, item in available.items() if not item["diagnostic"]] if raw == "all" else raw.split(",")
    require(selected and len(set(selected)) == len(selected) and all(name in available for name in selected),
            "Unknown or duplicate lane")
    return {name: available[name] for name in selected}


def parse_devices(raw):
    devices = raw.split(",") if raw else []
    require(len(set(devices)) == len(devices) and all(re.fullmatch(r"0|[1-9][0-9]*", d) for d in devices),
            "Devices must be unique physical GPU indices")
    return devices


def device_inventory(raw, devices, selected):
    rows = list(csv.reader(raw.splitlines(), skipinitialspace=True))
    require(all(len(row) == 3 for row in rows), "Invalid nvidia-smi inventory")
    records = {row[0].strip(): {"uuid": row[1].strip(), "name": row[2].strip()} for row in rows}
    require(len(records) == len(rows) and all(d in records for d in devices), "Selected GPU is unavailable")
    needed = max(item["devices"] for item in selected.values())
    require(len(devices) >= needed, f"Selected lanes require {needed} GPUs; unavailable coverage cannot pass")
    return {d: records[d] for d in devices}


def environment(base, runtime, cuda_dir, devices, item, arrays):
    # Keep OS/tool discovery settings; explicitly remove inherited inference/TP
    # tuning so a caller's ambient optimizations cannot silently change a lane.
    result = {key: value for key, value in base.items()
              if not key.startswith(("TS_", "TENSORSHARP_", "NCCL_", "GGML_"))
              and key not in ("LD_PRELOAD", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES")}
    result.update(TS_TEST_GGML_BACKEND="cuda" if item["devices"] else "cpu",
                  CUDA_VISIBLE_DEVICES=",".join(devices[:item["devices"]]),
                  LD_LIBRARY_PATH=str(runtime) + ":" + str(cuda_dir),
                  OMP_NUM_THREADS="4", TS_GGML_CPU_THREADS="4", TS_CPU_MOE_THREADS="4",
                  DOTNET_PROCESSOR_COUNT="8", NCCL_DEBUG="INFO",
                  TS_MUSE_SPARK_REUSE_CPU_MOE_SCRATCH="0",
                  TS_MUSE_SPARK_TP_CASE_OUTPUT_DIRECTORY=str(arrays))
    result.update(item["environment"])
    return result


def verify_upstream(root):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root / "ExternalProjects/ggml"), *args],
                                       text=True, timeout=30).strip()
    require(git("rev-parse", "HEAD") == GGML_REVISION, "Wrong upstream ggml revision")
    require(not git("status", "--porcelain", "--untracked-files=all"), "Modified upstream ggml")


def assert_idle(needs_gpu):
    for entry in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            argv = entry.read_bytes().split(b"\0")
        except FileNotFoundError:
            continue
        names = {part.rsplit(b"/", 1)[-1].lower() for part in argv}
        require(not names.intersection({b"llama-server", b"tensorsharp.server.host", b"tensorsharp.server.host.dll"}),
                "Active inference server: " + str(entry.parent))
    if needs_gpu:
        require(not subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid",
                    "--format=csv,noheader,nounits"], text=True, timeout=30).strip(), "GPU compute already active")


def native_commands(ctest, build):
    regex = "^(" + "|".join(sorted(NATIVE_TESTS)) + ")$"
    command = [ctest, "--test-dir", str(build), "-R", regex]
    data = json.loads(subprocess.check_output(command + ["--show-only=json-v1"], text=True, timeout=30))
    rows = data["tests"]
    require(Counter(row["name"] for row in rows) == Counter(NATIVE_TESTS),
            f"Native CTest inventory is not exactly {len(NATIVE_TESTS)}")
    pins = {}
    for row in rows:
        executable = Path(row["command"][0])
        require(executable.is_absolute() and executable.is_relative_to(build), "Unexpected native executable")
        pins[str(executable)] = sha(regular(executable))
    for path in build.rglob("CTestTestfile.cmake"):
        pins[str(path)] = sha(regular(path))
    pins[str(build / "libGgmlOps.so")] = sha(regular(build / "libGgmlOps.so"))
    return command, rows, pins


def verify_native_build_binding(rows, library, build):
    recorded = hash_map(build["files"])
    required = {str(library), *(row["command"][0] for row in rows)}
    require(required.issubset(recorded), "Build receipt does not pin every native test executable/library")
    verify_hashes({path: recorded[path] for path in required})


def run_child(command, log, cwd, env, timeout):
    def terminate(*_):
        raise KeyboardInterrupt("Component runner received SIGTERM")

    with log.open("xb") as stream:
        child = subprocess.Popen(command, cwd=cwd, env=env, stdout=stream,
                                 stderr=subprocess.STDOUT, close_fds=True, start_new_session=True)
        previous = signal.signal(signal.SIGTERM, terminate)
        try:
            return child.wait(timeout=timeout)
        except BaseException:
            # Only this test process group is owned by the runner. Never stop
            # unrelated inference/build processes to obtain an idle machine.
            with contextlib.suppress(ProcessLookupError):
                os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            raise
        finally:
            signal.signal(signal.SIGTERM, previous)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list-lanes", action="store_true")
    for name in ("root", "source-manifest", "build-record", "runtime", "native-build", "output"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--dotnet", default="dotnet")
    parser.add_argument("--ctest", default="ctest")
    parser.add_argument("--cuda-library-dir", type=Path, default=Path("/usr/local/cuda/lib64"))
    parser.add_argument("--devices", default="0,1,2,3,4,5,6,7,8")
    parser.add_argument("--lanes", default="all", help="Comma-separated labels, or all required lanes")
    parser.add_argument("--timeout", type=int, default=900, help="Per-lane timeout in seconds (1..3600)")
    args = parser.parse_args(argv)
    if args.list_lanes:
        print(json.dumps(lanes(), indent=2))
        return 0
    require(sys.platform == "linux", "Execution requires Linux; parser/fixture tests are portable")
    require(all(getattr(args, name) is not None for name in
                ("root", "source_manifest", "build_record", "runtime", "native_build", "output")), "Missing required paths")
    require(1 <= args.timeout <= 3600, "Invalid timeout")
    for name in ("root", "source_manifest", "build_record", "runtime", "native_build", "output"):
        path = getattr(args, name).absolute()
        require(not any(p.is_symlink() for p in (path, *path.parents)), "Symlink input/output path")
        setattr(args, name, path)
    require(args.output.parent.is_dir() and not args.output.exists(), "Output must be a new directory under an existing parent")
    require(not args.output.is_relative_to(args.runtime) and not args.output.is_relative_to(args.native_build),
            "Evidence must be outside binary directories")
    selected, devices = select_lanes(args.lanes), parse_devices(args.devices)
    needs_gpu = any(item["devices"] for item in selected.values())
    import fcntl  # Linux-only execution; portable tests import the pure guards.
    with (args.output.parent / ".muse-spark-components.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        args.output.mkdir()  # Exclusive; no retries can overwrite a failed run.
        result = {"status": "running", "started_utc": utc(), "lanes": [], "requested_lanes": list(selected),
                  "scope": "Synthetic component validation only; not full-checkpoint quality, serving, or performance."}
        try:
            pins, runtime = bind_build(args.root, args.runtime, args.source_manifest, args.build_record)
            tool = Path(__file__).resolve()
            require(str(tool) in pins and pins[str(tool)] == sha(tool), "Runner absent from supplied source/build snapshot")
            tools = {name: str(Path(shutil.which(getattr(args, name)) or getattr(args, name)).resolve(strict=True))
                     for name in ("dotnet", "ctest")}
            for path in tools.values():
                merge_pins(pins, {path: sha(regular(path))})
            native_command, native_rows = [], []
            if "native" in selected:
                native_command, native_rows, native_pins = native_commands(tools["ctest"], args.native_build)
                verify_native_build_binding(native_rows, args.native_build / "libGgmlOps.so", read_json(args.build_record))
                merge_pins(pins, native_pins)
            else:
                path = args.native_build / "libGgmlOps.so"
                merge_pins(pins, {str(path): sha(regular(path))})
            require(sha(args.native_build / "libGgmlOps.so") == sha(args.runtime / "libGgmlOps.so"),
                    "Native and managed runtime libraries differ")
            inventory = device_inventory(subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name",
                "--format=csv,noheader"], text=True, timeout=30), devices, selected) if needs_gpu else {}
            seal = {"files": pins, "runtime_files": runtime, "native_tests": native_rows, "lanes": selected,
                    "devices": inventory, "ggml_revision": GGML_REVISION, "created_utc": utc()}
            write_new(args.output / "inputs.json", seal)
            pins[str(args.output / "inputs.json")] = sha(args.output / "inputs.json")

            def verify():
                verify_hashes(pins)
                require(runtime_files(args.runtime) == runtime, "Runtime inventory changed")
                verify_upstream(args.root)
                assert_idle(needs_gpu)

            for name, item in selected.items():
                verify()
                arrays = args.output / (name + "-arrays")
                arrays.mkdir()
                env = environment(os.environ, args.runtime, args.cuda_library_dir, devices, item, arrays)
                env["DOTNET_ROOT"] = str(Path(tools["dotnet"]).parent)
                native = name == "native"
                report = args.output / (name + (".xml" if native else ".trx"))
                command = (native_command + ["--output-on-failure", "--output-junit", str(report)] if native else
                    [tools["dotnet"], "vstest", str(args.runtime / "InferenceWeb.Tests.dll"),
                     "/TestCaseFilter:" + "|".join("FullyQualifiedName=" + method for method in item["methods"]),
                     "/Logger:trx;LogFileName=" + report.name, "/ResultsDirectory:" + str(args.output),
                     "/Logger:console;verbosity=detailed"])
                record = {"lane": name, "started_utc": utc(), "command": command,
                          "diagnostic": item["diagnostic"], "environment": {k: v for k, v in env.items()
                              if k.startswith(("TS_", "NCCL_", "OMP_", "CUDA_", "DOTNET_")) or k == "LD_LIBRARY_PATH"}}
                started = time.monotonic()
                try:
                    record["returncode"] = run_child(command, args.output / (name + ".log"),
                                                     args.root, env, args.timeout)
                    record["cases"] = read_cases(report, native)
                    record["outcome_counts"] = dict(Counter(case["outcome"] for case in record["cases"]))
                    check_cases(record["cases"], item["expected"], record["returncode"])
                    record["status"] = "pass"
                except BaseException as error:
                    record.update(status="fail", error=f"{type(error).__name__}: {error}")
                    raise
                finally:
                    record.update(finished_utc=utc(), elapsed_seconds=time.monotonic() - started,
                                  output_hashes={str(p.relative_to(args.output)): sha(p)
                                      for p in [report, args.output / (name + ".log"), *arrays.glob("*.json")] if p.is_file()})
                    result["lanes"].append(record)
                    try:
                        verify()
                    except BaseException as error:
                        record.update(status="fail", verification_error=f"{type(error).__name__}: {error}")
                        raise
                    finally:
                        write_new(args.output / (name + "-result.json"), record)
                print(name + ": passed", flush=True)
            required = {name for name, item in lanes().items() if not item["diagnostic"]}
            result.update(status="pass", all_required_lanes_passed=required.issubset(selected),
                          omitted_required_lanes=sorted(required - selected.keys()))
        except BaseException as error:
            result.update(status="fail", all_required_lanes_passed=False, error=f"{type(error).__name__}: {error}")
        finally:
            result["finished_utc"] = utc()
            result["not_run_lanes"] = [name for name in selected
                                       if name not in {row["lane"] for row in result["lanes"]}]
            write_new(args.output / "result.json", result)
        print(json.dumps({"status": result["status"], "evidence": str(args.output),
                          "all_required_lanes_passed": result.get("all_required_lanes_passed", False)}))
        return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
