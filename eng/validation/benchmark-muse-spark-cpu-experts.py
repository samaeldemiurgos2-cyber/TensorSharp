#!/usr/bin/env python3
r"""Actual-checkpoint Muse Spark CPU expert microbenchmark (no server required).

Example (after stopping other inference/build/download workloads):
  PYTHONPATH=/workspace/llama.cpp/gguf-py python3 \
    eng/validation/benchmark-muse-spark-cpu-experts.py \
    --model /workspace/models/quant/model-00001-of-00010.gguf \
    --library /workspace/TensorSharp/artifacts/native/libGgmlOps.so \
    --layers 0 --rows 1,128,256 --threads 8,16,24,32 --repeats 3 \
    --output /workspace/evidence/cpu-experts-q3.json

Run separately for each published quantization. Default: ONE layer, not every
layer in an 816B checkpoint. --check-only validates headers/bounds without loading
the native library or reading tensor payloads. Needs numpy and unchanged llama.cpp
gguf-py on PYTHONPATH. The exact ABI comes from TensorSharp-owned ggml_ops_moe.cpp.

Maps only the selected gate/up/down/post-expert-norm ranges read-only; there is no
resident expert copy or dense expansion. The tiny norm vector is converted to
F32. Native execution initializes CPU backend 2, forces run_on_cpu=1, and disables
expert GPU streaming. Warmup, threadpool recreation, mapping and output checks
are outside timings. Every output is finite and checked against the first warmup
at the first thread count for that layer/row count. This is thread consistency,
NOT an independent model-quality oracle or a llama.cpp performance comparison.

Optional Linux --backend cuda-stream initializes CUDA on device 0, takes a CPU
reference with streaming disabled, then enables streaming for rows >=128. It
captures native stderr and requires explicit decision=stream evidence; fallback
cannot count as GPU timing. This mode uses temporary GPU expert storage and
compares against CPU outputs with the same recorded --atol/--rtol settings.

Inputs and routes are deterministic synthetic latent activations, not captured
model states. Prefill routes span many experts; decode touches only its selected
experts. Record cache residency and per-call minor/major faults/read_bytes. Only
disk-quiet measured calls qualify for a best-thread summary; minor faults remain
an allocation diagnostic and do not disqualify those timings. Re-run in reversed thread order
to assess drift; do not change production/llama thread counts from one sweep.

For CPU scratch A/B diagnostics, run separate processes with
TS_MUSE_SPARK_REUSE_CPU_MOE_SCRATCH=0 and =1, identical arguments/weights and
the same native binary. The flag is latched by native code at process startup.
Use separate output paths and --rows 1,128,256. Recorded requested enablement is
not proof the native reusable allocator was selected (it can fall back), and
minor_faults multiplied by page size is not measured disk or memory bandwidth.
When available, the native scratch-state query records actual successful arena
dispatches and retained capacity before/after each call, outside both the timed
interval and process-counter interval. Older binaries report unavailable state.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import ctypes
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import mmap
import os
from pathlib import Path
import re
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("spark_expert_warm", Path(__file__).with_name("warm-muse-spark-cpu-experts.py"))
warm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(warm)
PARTS = ("gate", "up", "down")
CPU_BACKEND = 2


def csv_positive(value, *, allow_zero=False):
    values = [int(v) for v in value.split(",")]
    if not values or len(values) != len(set(values)) or min(values) < (0 if allow_zero else 1):
        raise ValueError("Expected unique nonnegative layers or positive row/thread counts")
    return values


def tensor_names(layer):
    return [f"blk.{layer}.ffn_{part}_exps.weight" for part in PARTS] + [f"blk.{layer}.ffn_post_exp_norm.weight"]


def plan_model(paths, layers, reader_factory, quant_sizes):
    """Capture scalar metadata, never retain GGUFReader's whole-file views."""
    needed = {name for layer in layers for name in tensor_names(layer)}
    tensors, metadata = {}, None
    for path in paths:
        # The reader maps a shard transiently to parse its header. No tensor data
        # is accessed here; all persisted fields below are Python scalars.
        reader = reader_factory(path)
        try:
            if getattr(reader, "byte_order", "I") == "S" or sys.byteorder != "little":
                raise ValueError("Only native little-endian GGUF weights are supported")
            if metadata is None:
                arch = reader.fields["general.architecture"].contents()
                if arch not in ("muse-spark", "muse_spark"):
                    raise ValueError(f"Unexpected architecture: {arch}")
                metadata = {"architecture": arch, **{key: reader.fields[arch + "." + key].contents()
                    for key in ("block_count", "expert_count", "expert_used_count", "moe_latent_size",
                                "expert_feed_forward_length", "post_norm_epsilon")}}
                for key in metadata:
                    if key != "architecture":
                        metadata[key] = float(metadata[key]) if key == "post_norm_epsilon" else int(metadata[key])
                if not layers or min(layers) < 0 or max(layers) >= metadata["block_count"]:
                    raise ValueError("Selected layer exceeds model block count")
                if (not 0 < metadata["expert_used_count"] <= metadata["expert_count"] or
                        min(metadata["moe_latent_size"], metadata["expert_feed_forward_length"]) <= 0 or
                        not math.isfinite(metadata["post_norm_epsilon"]) or metadata["post_norm_epsilon"] <= 0):
                    raise ValueError("Invalid expert dimensions or normalization epsilon")
            if "split.count" in reader.fields and int(reader.fields["split.count"].contents()) != len(paths):
                raise ValueError("GGUF split.count does not match the complete shard set")
            stat = path.stat()
            for tensor in reader.tensors:
                if tensor.name not in needed:
                    continue
                if tensor.name in tensors:
                    raise ValueError(f"Duplicate tensor: {tensor.name}")
                shape = [int(v) for v in tensor.shape]
                offset, size = int(tensor.data_offset), int(tensor.n_bytes)
                block, type_bytes = quant_sizes[tensor.tensor_type]
                if not shape or min(shape) <= 0 or shape[0] % block or math.prod(shape) // block * type_bytes != size:
                    raise ValueError(f"Invalid quantized tensor shape/size: {tensor.name}")
                if offset < int(reader.data_offset) or offset > stat.st_size or size <= 0 or size > stat.st_size - offset:
                    raise ValueError(f"Tensor range exceeds its shard: {tensor.name}")
                tensors[tensor.name] = dict(tensor=tensor.name, path=str(path.resolve()), offset=offset,
                    bytes=size, shape=shape, ggml_type=int(tensor.tensor_type), type_name=tensor.tensor_type.name,
                    file_size=stat.st_size, mtime_ns=stat.st_mtime_ns)
        finally:
            # Remove the last tensor local before disposing the reader so its
            # memmap does not outlive this shard's metadata scan.
            tensor = None
            del reader
    if tensors.keys() != needed:
        raise ValueError(f"Missing selected tensors: {sorted(needed - tensors.keys())}")
    h, f, e = (metadata[k] for k in ("moe_latent_size", "expert_feed_forward_length", "expert_count"))
    for layer in layers:
        gate, up, down, norm = [tensors[n] for n in tensor_names(layer)]
        if gate["shape"] != [h, f, e] or up["shape"] != [h, f, e] or down["shape"] != [f, h, e]:
            raise ValueError(f"Layer {layer}: expert stack dimensions disagree with model metadata")
        if norm["shape"] != [h] or norm["ggml_type"] not in (0, 1):
            raise ValueError("Post-expert norm must be a channel vector in F32 or F16")
    ordered = sorted(tensors.values(), key=lambda item: (item["path"], item["offset"]))
    for left, right in zip(ordered, ordered[1:]):
        if left["path"] == right["path"] and left["offset"] + left["bytes"] > right["offset"]:
            raise ValueError("Selected tensor ranges overlap")
    return {"metadata": metadata, "layers": layers, "tensors": tensors,
            "selected_bytes": sum(t["bytes"] for t in tensors.values())}


class MappedLayer:
    def __init__(self, plan, layer, np):
        self.items = [plan["tensors"][name] for name in tensor_names(layer)]
        self.np, self.entries, self.pointers = np, [], {}

    def __enter__(self):
        try:
            for item in self.items:
                fd = os.open(item["path"], os.O_RDONLY)
                try:
                    warm.verify_file(fd, item)
                    aligned = item["offset"] // mmap.ALLOCATIONGRANULARITY * mmap.ALLOCATIONGRANULARITY
                    lead = item["offset"] - aligned
                    mapped = mmap.mmap(fd, lead + item["bytes"], access=mmap.ACCESS_READ, offset=aligned)
                finally:
                    os.close(fd)
                view = self.np.frombuffer(mapped, dtype=self.np.uint8, count=item["bytes"], offset=lead)
                self.entries.append((mapped, view))
                self.pointers[item["tensor"]] = int(view.ctypes.data)
            norm_type = "<f4" if self.items[-1]["ggml_type"] == 0 else "<f2"
            self.norm = self.entries[-1][1].view(norm_type).astype(self.np.float32, copy=True)
            if not self.np.isfinite(self.norm).all():
                raise ValueError("Nonfinite post-expert normalization weights")
            return self
        except BaseException:
            # The current local view otherwise keeps a mapping exported while
            # cleanup closes it after a validation failure.
            view = None
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *_):
        self.pointers.clear()
        self.norm = None
        while self.entries:
            mapped, view = self.entries.pop()
            del view
            mapped.close()


class NativeMoE:
    """ctypes declaration matches TSGgml_MuseSparkMoEFFNQuantF32 exactly."""
    def __init__(self, path, loader=ctypes.CDLL, backend=CPU_BACKEND):
        self.lib = loader(str(path))
        pointer, integer, wide = ctypes.c_void_p, ctypes.c_int, ctypes.c_int64
        weight = [pointer, integer, wide, wide, wide]
        self.invoke = self.lib.TSGgml_MuseSparkMoEFFNQuantF32
        self.invoke.argtypes = [pointer, pointer] + [integer] * 5 + [pointer, pointer] + weight * 3 + \
            [pointer] * 3 + [integer, ctypes.c_float, ctypes.c_float, integer, pointer, ctypes.c_float]
        self.invoke.restype = integer
        self.lib.TSGgml_IsBackendAvailable.argtypes = [integer]
        self.lib.TSGgml_IsBackendAvailable.restype = integer
        self.lib.TSGgml_GetLastError.argtypes = []
        self.lib.TSGgml_GetLastError.restype = ctypes.c_char_p
        self.lib.TSGgml_SetHostMoeThreads.argtypes = [integer]
        self.lib.TSGgml_SetHostMoeThreads.restype = None
        self.lib.TSGgml_Shutdown.argtypes = []
        self.lib.TSGgml_Shutdown.restype = None
        self.scratch_query = getattr(self.lib, "TSGgml_GetMuseSparkCpuMoeScratchState", None)
        if self.scratch_query is not None:
            self.scratch_query.argtypes = [ctypes.POINTER(ctypes.c_uint64)] * 2
            self.scratch_query.restype = integer
        if not self.lib.TSGgml_IsBackendAvailable(backend):
            raise RuntimeError(f"Native backend {backend} unavailable: " + self.error())

    def error(self):
        return (self.lib.TSGgml_GetLastError() or b"unknown native error").decode("utf-8", "replace")

    def threads(self, count):
        self.lib.TSGgml_SetHostMoeThreads(count)

    def scratch_state(self):
        if self.scratch_query is None:
            return None  # Older baseline binaries do not expose this telemetry.
        allocated, calls = ctypes.c_uint64(), ctypes.c_uint64()
        if not self.scratch_query(ctypes.byref(allocated), ctypes.byref(calls)):
            raise RuntimeError("Native CPU scratch-state query failed")
        return {"allocated_bytes": allocated.value, "successful_calls": calls.value}

    def run(self, inputs, output, ids, routing, mapped, metadata, run_on_cpu=True):
        weights = []
        for item in mapped.items[:3]:
            weights.extend([mapped.pointers[item["tensor"]], item["ggml_type"],
                            item["shape"][0], item["shape"][1], item["bytes"]])
        ok = self.invoke(inputs.ctypes.data, output.ctypes.data, inputs.shape[0], inputs.shape[1],
            metadata["expert_feed_forward_length"], metadata["expert_count"], metadata["expert_used_count"],
            ids.ctypes.data, routing.ctypes.data, *weights, None, None, None, 0, 1.702, 7.0, int(run_on_cpu),
            mapped.norm.ctypes.data, metadata["post_norm_epsilon"])
        if not ok:
            raise RuntimeError("Native expert graph failed: " + self.error())

    def close(self):
        self.lib.TSGgml_Shutdown()


def make_inputs(metadata, rows, seed, np):
    rng = np.random.Generator(np.random.PCG64(seed))
    inputs = rng.standard_normal((rows, metadata["moe_latent_size"]), dtype=np.float32) * np.float32(0.25)
    ids = np.stack([rng.choice(metadata["expert_count"], metadata["expert_used_count"], replace=False)
                    for _ in range(rows)]).astype(np.int32)
    routing = rng.uniform(0.1, 1.0, ids.shape).astype(np.float32)
    routing /= routing.sum(axis=1, keepdims=True)
    return inputs, ids, routing


def check_output(output, baseline, atol, rtol, np):
    if not np.isfinite(output).all():
        raise ValueError("Native output contains NaN/Inf or unwritten entries")
    if baseline is None:
        return 0.0
    error = np.abs(output - baseline)
    maximum = float(error.max())
    if not np.all(error <= atol + rtol * np.abs(baseline)):
        raise ValueError(f"Thread-count output mismatch: maximum absolute error {maximum}")
    return maximum


def counters():
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        major_faults = int(usage.ru_majflt)
        minor_faults = int(usage.ru_minflt)
    except ImportError:
        major_faults = minor_faults = None
    return {"io": warm.process_io(), "major_faults": major_faults, "minor_faults": minor_faults}


def counter_delta(before, after):
    def delta(first, last):
        if first is None or last is None:
            return None
        if type(first) is not int or type(last) is not int or not 0 <= first <= last:
            raise ValueError("Process counters must be nonnegative monotonic integers")
        return last - first
    reads = delta((before.get("io") or {}).get("read_bytes"), (after.get("io") or {}).get("read_bytes"))
    major = delta(before.get("major_faults"), after.get("major_faults"))
    minor = delta(before.get("minor_faults"), after.get("minor_faults"))
    return {"read_bytes": reads, "major_faults": major, "minor_faults": minor,
            "cache_quiet": reads == 0 and major == 0,
            "process_counters": {"before": before, "after": after}}


def cpu_scratch_request():
    name = "TS_MUSE_SPARK_REUSE_CPU_MOE_SCRATCH"
    raw = os.environ.get(name)
    return {"environment_variable": name, "requested_value": raw,
            "enabled_requested": raw == "1",
            "scope": "Native process-startup request only; allocator use/fallback needs native evidence."}


def scratch_state(native):
    query = getattr(native, "scratch_state", None)
    return query() if query is not None else None


def scratch_evidence(before, after):
    # A benchmark process dispatches exactly one synchronous expert operation
    # between observations. More than one call or a reset means attribution is
    # no longer sound; an unavailable export cannot be reported as zero use.
    change = None
    if before is not None and after is not None:
        for state in (before, after):
            if any(type(state[k]) is not int or state[k] < 0 for k in ("allocated_bytes", "successful_calls")):
                raise ValueError("Invalid native CPU scratch-state telemetry")
        change = after["successful_calls"] - before["successful_calls"]
        if change not in (0, 1) or (change == 1 and after["allocated_bytes"] == 0):
            raise ValueError("Native CPU scratch-state counter cannot be attributed to one call")
    return {"before": before, "after": after, "successful_calls_delta": change,
            "reused_on_this_call": change == 1 if change is not None else None}


def capacity_evidence():
    paths = ("/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory.max",
             "/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "/sys/fs/cgroup/cpu/cpu.cfs_period_us",
             "/sys/fs/cgroup/cpu.max")
    evidence = {"logical_cpu_count": os.cpu_count(),
        "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "cgroup": {}, "environment": {key: os.environ[key] for key in
            ("TS_CPU_MOE_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "TS_MUSE_SPARK_REUSE_CPU_MOE_SCRATCH") if key in os.environ}}
    for name in paths:
        try:
            evidence["cgroup"][name] = Path(name).read_text().strip()
        except OSError:
            pass
    return evidence


@contextmanager
def capture_native_stderr(path):
    """Capture C/C++ stderr, not just Python's sys.stderr wrapper (Linux)."""
    saved = os.dup(2)
    with path.open("wb", buffering=0) as log:
        try:
            os.dup2(log.fileno(), 2)
            yield
        finally:
            os.dup2(saved, 2)
            os.close(saved)


def stream_decisions(path, expected_rows):
    pattern = re.compile(r"\[muse-spark-stream\] rank=(\d+) rows=(\d+) decision=(stream|cpu) reason=(\S+) required=(\d+) free=(\d+) reusable=(\d+)")
    text = path.read_text(encoding="utf-8", errors="replace")
    rows = [dict(rank=int(m[0]), rows=int(m[1]), decision=m[2], reason=m[3],
                 required_bytes=int(m[4]), free_bytes=int(m[5]), reusable_bytes=int(m[6]))
            for m in pattern.findall(text)]
    return {"log": str(path), "sha256": sha256_file(path), "decisions": rows,
            "stream_confirmed": bool(rows) and all(row["decision"] == "stream" for row in rows)
                                and set(expected_rows) <= {row["rows"] for row in rows}}


def sweep_layer(plan, layer, mapped, native, args, np, emit):
    results = []
    for rows in args.rows:
        seed = args.seed + layer * 1009 + rows
        inputs, ids, routing = make_inputs(plan["metadata"], rows, seed, np)
        output = np.empty_like(inputs)
        baseline = None
        input_evidence = {"seed": seed, "activation_sha256": hashlib.sha256(inputs.tobytes()).hexdigest(),
            "ids_sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
            "routing_sha256": hashlib.sha256(routing.tobytes()).hexdigest(),
            "distinct_experts": int(np.unique(ids).size),
            "expert_token_counts": np.bincount(ids.ravel(), minlength=plan["metadata"]["expert_count"]).tolist()}
        if getattr(args, "backend", "cpu") == "cuda-stream":
            # Same process, same pointers/data and same native ABI: only the
            # explicitly dynamic Spark opt-in changes after a CPU reference.
            os.environ["TS_MUSE_SPARK_STREAM_EXPERTS"] = "0"
            native.threads(args.threads[0])
            reference = {"kind": "cpu_reference", "layer": layer, "rows": rows,
                "threads": args.threads[0], "input": input_evidence, "warmup": [], "measured": []}
            for kind, count in (("warmup", args.warmup), ("measured", args.repeats)):
                for repeat in range(count):
                    output.fill(np.nan)
                    scratch_before = scratch_state(native)
                    before = counters()
                    start = time.perf_counter_ns()
                    native.run(inputs, output, ids, routing, mapped, plan["metadata"])
                    duration = time.perf_counter_ns() - start
                    after = counters()
                    scratch_after = scratch_state(native)
                    error = check_output(output, baseline, args.atol, args.rtol, np)
                    if baseline is None:
                        baseline = output.copy()
                    reference[kind].append({"repeat": repeat, "wall_ms": duration / 1e6,
                        "output_f32_sha256": hashlib.sha256(output.tobytes(order="C")).hexdigest(),
                        "max_abs_error_vs_first_thread_warmup": error, **counter_delta(before, after),
                        "cpu_scratch_reuse": scratch_evidence(scratch_before, scratch_after)})
            reference["median_wall_ms"] = statistics.median(r["wall_ms"] for r in reference["measured"])
            reference["timing_eligible"] = all(r["cache_quiet"] for r in reference["measured"])
            reference["status"] = "numerics_passed"
            emit(reference)
            os.environ["TS_MUSE_SPARK_STREAM_EXPERTS"] = "1"
        for threads in args.threads:
            native.threads(threads)  # Rebuild/latch team outside measured calls.
            result = {"layer": layer, "rows": rows, "threads": threads, "input": input_evidence,
                      "warmup": [], "measured": []}
            for kind, count in (("warmup", args.warmup), ("measured", args.repeats)):
                for repeat in range(count):
                    output.fill(np.nan)
                    scratch_before = scratch_state(native)
                    before = counters()
                    start = time.perf_counter_ns()
                    native.run(inputs, output, ids, routing, mapped, plan["metadata"])
                    duration = time.perf_counter_ns() - start
                    after = counters()
                    scratch_after = scratch_state(native)
                    error = check_output(output, baseline, args.atol, args.rtol, np)
                    if baseline is None:
                        baseline = output.copy()
                    result[kind].append({"repeat": repeat, "wall_ms": duration / 1e6,
                        "output_f32_sha256": hashlib.sha256(output.tobytes(order="C")).hexdigest(),
                        "max_abs_error_vs_first_thread_warmup": error, **counter_delta(before, after),
                        "cpu_scratch_reuse": scratch_evidence(scratch_before, scratch_after)})
            result["median_wall_ms"] = statistics.median(r["wall_ms"] for r in result["measured"])
            result["timing_eligible"] = all(r["cache_quiet"] for r in result["measured"])
            result["status"] = "numerics_passed"
            results.append(result)
            emit(result)
    return results


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    tool_started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--backend", choices=("cpu", "cuda-stream"), default="cpu",
                        help="cuda-stream first measures a CPU reference, then requires native streaming-decision evidence")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", default="0")
    parser.add_argument("--rows", default="1,128,256")
    parser.add_argument("--threads", default="8,16,24,32")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=17817)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    try:
        args.layers = csv_positive(args.layers, allow_zero=True)
        args.rows, args.threads = csv_positive(args.rows), csv_positive(args.threads)
        if max(args.rows) > 256 or max(args.threads) > 256 or args.warmup < 1 or args.repeats < 3 or args.seed < 0:
            raise ValueError("rows/threads <=256, warmup >=1, repeats >=3 and seed >=0 required")
        if not all(math.isfinite(v) and v >= 0 for v in (args.atol, args.rtol)):
            raise ValueError("Tolerances must be finite and nonnegative")
        if args.backend == "cuda-stream" and (sys.platform != "linux" or min(args.rows) < 128):
            raise ValueError("cuda-stream requires Linux native stderr capture and rows >=128")
        if not args.check_only and (args.library is None or not args.library.is_file()):
            raise ValueError("Execution requires --library pointing to the actual native binary")
    except ValueError as error:
        parser.error(str(error))
    import numpy as np
    from gguf import GGUFReader, GGML_QUANT_SIZES
    plan = plan_model(warm.discover_shards(args.model), args.layers, GGUFReader, GGML_QUANT_SIZES)
    report = {"format_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "planned", "plan": plan, "configuration": vars(args).copy(), "results": [], "cpu_reference": [],
        "source_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in
            (Path(__file__), ROOT / "TensorSharp.GGML.Native/ggml_ops_moe.cpp", ROOT / "TensorSharp.GGML.Native/ggml_ops_core.cpp")},
        "reader_source": {"path": str(Path(sys.modules[GGUFReader.__module__].__file__).resolve())},
        "capacity": capacity_evidence(),
        "cpu_scratch_reuse": cpu_scratch_request(),
        "limitations": ["CPU expert graph only; excludes router, latent projections, attention, HTTP and scheduling.",
            "Deterministic synthetic activations/routes are not a captured production workload.",
            "Thread-baseline agreement is not an independent numerical/model-quality oracle.",
            "No llama.cpp timings are measured. A production thread change needs paired end-to-end validation.",
            "First warmup may fault selected weights from disk. Only cache-quiet measured repeats enter the best-thread summary.",
            "Cache-quiet counters do not establish absence of competing workloads or CPU throttling.",
            "Minor faults are process-wide first-touch/COW/NUMA diagnostics, not disk traffic or transferred bytes; their deltas include counter sampling overhead outside the timed native call.",
            "CPU scratch reuse is a process-startup request, not allocator-use proof. Compare separate flag0/1 processes with identical inputs and the same binary; fallback needs native evidence.",
            "Optional native scratch-state queries run outside timings/counter snapshots; per-call successful dispatch deltas distinguish arena use from fallback, while old binaries report null telemetry.",
            "Threadpool construction and warmup are excluded. Repeat in reversed thread order to assess drift."]}
    for key in ("model", "library", "output"):
        report["configuration"][key] = str(getattr(args, key)) if getattr(args, key) is not None else None
    report["reader_source"]["sha256"] = sha256_file(report["reader_source"]["path"])
    save(args.output, report)
    if args.check_only:
        report["status"] = "headers_validated_no_native_execution"
        save(args.output, report)
        return 0
    report["native_library"] = {"path": str(args.library.resolve()), "sha256": sha256_file(args.library)}
    # Set before initialization: CPU expert streaming must never satisfy a
    # supposedly CPU-only measurement on a hidden/default GPU backend.
    os.environ["TS_MUSE_SPARK_STREAM_EXPERTS"] = "0"
    report["forced_environment"] = {"TS_MUSE_SPARK_STREAM_EXPERTS": "0"}
    if args.backend == "cuda-stream":
        os.environ["TS_HOST_MOE_DEVICE_MIN_BATCH"] = "128"
        report["forced_environment"].update(TS_HOST_MOE_DEVICE_MIN_BATCH="128",
            TS_MUSE_SPARK_STREAM_EXPERTS="0 for CPU reference, then 1 for CUDA streaming")
        report["limitations"].append("CUDA results compare against the same-process CPU reference under the recorded tolerances; this does not establish full-model quality parity.")
    native = None
    native_log = args.output.with_suffix(".native.log")
    started = time.monotonic()
    try:
        with capture_native_stderr(native_log) if args.backend == "cuda-stream" else nullcontext():
            native = NativeMoE(args.library.resolve(), backend=3 if args.backend == "cuda-stream" else CPU_BACKEND)
            for layer in args.layers:
                items = [plan["tensors"][n] for n in tensor_names(layer)]
                report.setdefault("residency", []).append({"layer": layer, "before": warm.residency(items)})
                with MappedLayer(plan, layer, np) as mapped:
                    def emit(result):
                        report["cpu_reference" if result.get("kind") == "cpu_reference" else "results"].append(result)
                        save(args.output, report)
                        print(f"kind={result.get('kind', args.backend)} layer={layer} rows={result['rows']} threads={result['threads']} median_ms={result['median_wall_ms']:.3f} cache_quiet={result['timing_eligible']}", flush=True)
                    sweep_layer(plan, layer, mapped, native, args, np, emit)
                report["residency"][-1]["after_unmap"] = warm.residency(items)
        if args.backend == "cuda-stream":
            report["streaming_evidence"] = stream_decisions(native_log, args.rows)
            if not report["streaming_evidence"]["stream_confirmed"]:
                raise ValueError("Native telemetry did not confirm CUDA streaming for every selected row count")
        report["status"] = "numerics_passed"
        report["best_threads_by_layer_rows"] = []
        for layer in args.layers:
            for rows in args.rows:
                eligible = [r for r in report["results"] if r["layer"] == layer and r["rows"] == rows and r["timing_eligible"]]
                best = min(eligible, key=lambda r: r["median_wall_ms"]) if eligible else None
                report["best_threads_by_layer_rows"].append({"layer": layer, "rows": rows,
                    "threads": best["threads"] if best else None, "median_wall_ms": best["median_wall_ms"] if best else None,
                    "eligible_configurations": len(eligible)})
    except BaseException as error:
        report.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                      error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if native is not None:
            native.close()
        report["native_initialization_mapping_and_measurement_seconds"] = time.monotonic() - started
        report["total_tool_seconds"] = time.monotonic() - tool_started
        save(args.output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
