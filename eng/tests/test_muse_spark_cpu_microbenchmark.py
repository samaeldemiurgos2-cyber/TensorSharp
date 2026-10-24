"""No checkpoint, server or native backend needed for the benchmark guard tests."""
import copy
import ctypes
from enum import IntEnum
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

SPEC = importlib.util.spec_from_file_location("spark_cpu_microbenchmark",
    Path(__file__).parents[1] / "validation/benchmark-muse-spark-cpu-experts.py")
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)
ORACLE_SPEC = importlib.util.spec_from_file_location("spark_expert_row_oracle",
    Path(__file__).parents[1] / "validation/oracle-muse-spark-expert-row.py")
ORACLE = importlib.util.module_from_spec(ORACLE_SPEC)
ORACLE_SPEC.loader.exec_module(ORACLE)


class Quant(IntEnum):
    F32 = 0
    F16 = 1


class Function:
    def __init__(self, implementation):
        self.implementation = implementation
        self.calls = []

    def __call__(self, *args):
        self.calls.append(args)
        return self.implementation(*args)


class CpuExpertMicrobenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "spark-00001-of-00001.gguf"
        self.path.write_bytes(bytes(512))
        self.metadata = {"block_count": 2, "expert_count": 4, "expert_used_count": 2,
            "moe_latent_size": 2, "expert_feed_forward_length": 3, "post_norm_epsilon": 1e-5}
        names = BENCH.tensor_names(0)
        self.tensors = [SimpleNamespace(name=name, shape=shape, data_offset=offset, n_bytes=size, tensor_type=Quant.F32)
            for name, shape, offset, size in zip(names,
                ([2, 3, 4], [2, 3, 4], [3, 2, 4], [2]), (64, 160, 256, 352), (96, 96, 96, 8))]
        with self.path.open("r+b") as file:
            file.seek(352)
            file.write(np.array([0.75, -1.25], dtype=np.float32).tobytes())

    def tearDown(self):
        self.temp.cleanup()

    def reader(self, _):
        values = {"general.architecture": "muse-spark", "split.count": 1,
                  **{"muse-spark." + k: v for k, v in self.metadata.items()}}
        fields = {k: SimpleNamespace(contents=lambda value=v: value) for k, v in values.items()}
        return SimpleNamespace(tensors=self.tensors, fields=fields, data_offset=64, byte_order="I")

    def plan(self):
        return BENCH.plan_model([self.path], [0], self.reader, {Quant.F32: (1, 4), Quant.F16: (1, 2)})

    def test_plan_reads_actual_shapes_offsets_types_and_no_payload(self):
        plan = self.plan()
        self.assertEqual(296, plan["selected_bytes"])
        self.assertEqual([2, 3, 4], plan["tensors"][BENCH.tensor_names(0)[0]]["shape"])
        self.assertEqual(352, plan["tensors"][BENCH.tensor_names(0)[3]]["offset"])
        self.assertEqual("F32", plan["tensors"][BENCH.tensor_names(0)[3]]["type_name"])
        # Fake header tensors intentionally have no .data member: planning
        # cannot accidentally expand or access the actual expert payload.

    def test_invalid_ranges_duplicates_shapes_and_norm_types_fail_closed(self):
        original = copy.deepcopy(self.tensors)
        mutations = (lambda: setattr(self.tensors[0], "data_offset", 32),
                     lambda: setattr(self.tensors[0], "data_offset", 500),
                     lambda: setattr(self.tensors[0], "n_bytes", 95),
                     lambda: setattr(self.tensors[0], "shape", [3, 2, 4]),
                     lambda: setattr(self.tensors[-1], "shape", [1, 2]),
                     lambda: setattr(self.tensors[1], "data_offset", 128),
                     lambda: self.tensors.append(self.tensors[0]),
                     lambda: self.tensors.pop())
        for mutate in mutations:
            self.tensors = copy.deepcopy(original)
            mutate()
            with self.assertRaises(ValueError):
                self.plan()

    def test_invalid_metadata_and_endianness_are_rejected(self):
        for key, value in (("expert_used_count", 5), ("moe_latent_size", 0),
                           ("post_norm_epsilon", float("nan")), ("post_norm_epsilon", 0)):
            with self.subTest(key=key, value=value):
                previous = self.metadata[key]
                self.metadata[key] = value
                with self.assertRaises(ValueError):
                    self.plan()
                self.metadata[key] = previous
        with self.assertRaisesRegex(ValueError, "layer"):
            BENCH.plan_model([self.path], [2], self.reader, {Quant.F32: (1, 4)})
        reader = self.reader(None)
        reader.byte_order = "S"
        with self.assertRaisesRegex(ValueError, "endian"):
            BENCH.plan_model([self.path], [0], lambda _: reader, {Quant.F32: (1, 4)})

    def test_maps_only_exact_selected_ranges_readonly_and_releases_them(self):
        plan, original = self.plan(), self.path.read_bytes()
        mapped = BENCH.MappedLayer(plan, 0, np)
        with mapped:
            self.assertEqual(4, len(mapped.entries))
            self.assertEqual([96, 96, 96, 8], [view.size for _, view in mapped.entries])
            self.assertTrue(all(not view.flags.writeable for _, view in mapped.entries))
            self.assertEqual([0.75, -1.25], mapped.norm.tolist())
            pointers = [view.ctypes.data for _, view in mapped.entries]
            self.assertEqual(pointers, list(mapped.pointers.values()))
        self.assertEqual([], mapped.entries)
        self.assertEqual({}, mapped.pointers)
        self.assertEqual(original, self.path.read_bytes())

    def test_file_change_and_mapping_failure_clean_up(self):
        plan = self.plan()
        self.path.write_bytes(b"short")
        mapped = BENCH.MappedLayer(plan, 0, np)
        with self.assertRaisesRegex(ValueError, "changed"):
            with mapped:
                pass
        self.assertEqual([], mapped.entries)
        self.setUp_second_file(plan)

    def setUp_second_file(self, _):
        # A second-map failure must release the first exported numpy view.
        self.path.write_bytes(bytes(512))
        plan = self.plan()
        mapped = BENCH.MappedLayer(plan, 0, np)
        original = BENCH.mmap.mmap
        calls = 0
        def allocate(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("fixture mapping failure")
            return original(*args, **kwargs)
        with patch.object(BENCH.mmap, "mmap", side_effect=allocate):
            with self.assertRaisesRegex(OSError, "fixture mapping failure"):
                with mapped:
                    pass
        self.assertEqual([], mapped.entries)

    def test_norm_half_conversion_does_not_expand_expert_weights(self):
        self.tensors[-1].tensor_type = Quant.F16
        self.tensors[-1].n_bytes = 4
        with self.path.open("r+b") as file:
            file.seek(352)
            file.write(np.array([0.75, -1.25], dtype=np.float16).tobytes())
        with BENCH.MappedLayer(self.plan(), 0, np) as mapped:
            self.assertEqual(np.float32, mapped.norm.dtype)
            self.assertEqual([0.75, -1.25], mapped.norm.tolist())
            self.assertEqual([96, 96, 96, 4], [view.size for _, view in mapped.entries])

    def test_ctypes_abi_exact_cpu_arguments_and_mapped_pointer_identity(self):
        fake = SimpleNamespace(**{name: Function(function) for name, function in {
            "TSGgml_MuseSparkMoEFFNQuantF32": lambda *args: 1,
            "TSGgml_IsBackendAvailable": lambda backend: int(backend == 2),
            "TSGgml_GetLastError": lambda: b"fixture failure",
            "TSGgml_SetHostMoeThreads": lambda count: None,
            "TSGgml_Shutdown": lambda: None}.items()})
        native = BENCH.NativeMoE("unused", loader=lambda _: fake)
        self.assertIsNone(native.scratch_state())
        native.threads(24)
        plan = self.plan()
        inputs, ids, routing = BENCH.make_inputs(plan["metadata"], 3, 5, np)
        output = np.empty_like(inputs)
        with BENCH.MappedLayer(plan, 0, np) as mapped:
            native.run(inputs, output, ids, routing, mapped, plan["metadata"])
            args = fake.TSGgml_MuseSparkMoEFFNQuantF32.calls[0]
            self.assertEqual(33, len(args))
            self.assertEqual(33, len(fake.TSGgml_MuseSparkMoEFFNQuantF32.argtypes))
            self.assertEqual((3, 2, 3, 4, 2), args[2:7])
            self.assertEqual((inputs.ctypes.data, output.ctypes.data), args[:2])
            self.assertEqual((ids.ctypes.data, routing.ctypes.data), args[7:9])
            for offset, item in zip((9, 14, 19), mapped.items[:3]):
                self.assertEqual((mapped.pointers[item["tensor"]], 0, *item["shape"][:2], item["bytes"]), args[offset:offset + 5])
            self.assertEqual((None, None, None, 0, 1.702, 7.0, 1), args[24:31])
            self.assertEqual(mapped.norm.ctypes.data, args[31])
            self.assertEqual(1e-5, args[32])
            native.run(inputs, output, ids, routing, mapped, plan["metadata"], run_on_cpu=False)
            self.assertEqual(0, fake.TSGgml_MuseSparkMoEFFNQuantF32.calls[-1][30])
        native.close()
        self.assertEqual([(24,)], fake.TSGgml_SetHostMoeThreads.calls)
        self.assertEqual([()], fake.TSGgml_Shutdown.calls)

    def test_deterministic_unique_routes_and_normalized_weights(self):
        first = BENCH.make_inputs(self.metadata, 128, 7, np)
        second = BENCH.make_inputs(self.metadata, 128, 7, np)
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left, right)
        inputs, ids, weights = first
        self.assertEqual(np.float32, inputs.dtype)
        self.assertEqual(np.int32, ids.dtype)
        self.assertTrue(all(len(set(row)) == self.metadata["expert_used_count"] for row in ids))
        np.testing.assert_allclose(weights.sum(axis=1), np.ones(128), atol=1e-6)
        self.assertFalse(np.array_equal(inputs, BENCH.make_inputs(self.metadata, 128, 8, np)[0]))

    def test_nonfinite_and_thread_drift_fail_and_unknown_cache_is_not_quiet(self):
        baseline = np.array([1, 2], dtype=np.float32)
        for output in (np.array([np.nan, 2]), np.array([1, np.inf]), np.array([1.1, 2])):
            with self.assertRaises(ValueError):
                BENCH.check_output(output, baseline, 1e-4, 1e-4, np)
        self.assertEqual(0, BENCH.check_output(baseline, baseline, 0, 0, np))
        self.assertFalse(BENCH.counter_delta({"io": None, "major_faults": None}, {"io": None, "major_faults": None})["cache_quiet"])

    def test_counters_capture_process_minor_faults_and_preserve_unavailable_values(self):
        calls = []
        resource = SimpleNamespace(RUSAGE_SELF=17, getrusage=lambda who:
            (calls.append(who) or SimpleNamespace(ru_majflt=31, ru_minflt=12345)))
        with patch.dict(sys.modules, {"resource": resource}), patch.object(BENCH.warm, "process_io", return_value={"read_bytes": 8192}):
            self.assertEqual({"io": {"read_bytes": 8192}, "major_faults": 31, "minor_faults": 12345}, BENCH.counters())
        self.assertEqual([17], calls)
        # Windows cannot supply resource counters. Missing telemetry must not
        # become zero faults or silently satisfy the disk-quiet guard.
        with patch.dict(sys.modules, {"resource": None}), patch.object(BENCH.warm, "process_io", return_value=None):
            unavailable = BENCH.counters()
        self.assertEqual({"io": None, "major_faults": None, "minor_faults": None}, unavailable)
        self.assertFalse(BENCH.counter_delta(unavailable, unavailable)["cache_quiet"])

    def test_minor_fault_delta_is_allocation_evidence_not_a_disk_cache_veto(self):
        before = {"io": {"read_bytes": 8192}, "major_faults": 5, "minor_faults": 100}
        after = {"io": {"read_bytes": 8192}, "major_faults": 5, "minor_faults": 321}
        result = BENCH.counter_delta(before, after)
        self.assertEqual((0, 0, 221), (result["read_bytes"], result["major_faults"], result["minor_faults"]))
        self.assertTrue(result["cache_quiet"])
        self.assertEqual({"before": before, "after": after}, result["process_counters"])
        legacy = {"io": {"read_bytes": 8192}, "major_faults": 5}
        result = BENCH.counter_delta(legacy, after)
        self.assertIsNone(result["minor_faults"])
        self.assertTrue(result["cache_quiet"])

    def test_counter_regressions_negative_values_and_boolean_telemetry_fail_closed(self):
        baseline = {"io": {"read_bytes": 3}, "major_faults": 3, "minor_faults": 3}
        for counter in ("read_bytes", "major_faults", "minor_faults"):
            for bad in (-1, 2, True, 4.0):
                with self.subTest(counter=counter, bad=bad):
                    after = copy.deepcopy(baseline)
                    (after["io"] if counter == "read_bytes" else after)[counter] = bad
                    with self.assertRaisesRegex(ValueError, "monotonic integers"):
                        BENCH.counter_delta(baseline, after)

    def test_cpu_scratch_flag_records_exact_startup_request_without_mutating_environment(self):
        name = "TS_MUSE_SPARK_REUSE_CPU_MOE_SCRATCH"
        for raw in (None, "0", "1", "true", "01", "1 "):
            with self.subTest(raw=raw), patch.dict(BENCH.os.environ, {}, clear=True):
                if raw is not None:
                    BENCH.os.environ[name] = raw
                before = dict(BENCH.os.environ)
                result = BENCH.cpu_scratch_request()
                self.assertEqual(raw, result["requested_value"])
                self.assertEqual(raw == "1", result["enabled_requested"])
                self.assertEqual(name, result["environment_variable"])
                self.assertIn("fallback", result["scope"])
                self.assertEqual(before, dict(BENCH.os.environ))

    def test_rows128_sweep_keeps_warmup_and_measured_fault_deltas_separate(self):
        args = SimpleNamespace(rows=[128], threads=[16, 32], warmup=1, repeats=3, seed=7, atol=1e-4, rtol=1e-4)
        shapes, emitted, samples = [], [], []
        minor = 5000
        for change in (1000, 23, 23, 23) * 2:
            samples.append({"io": {"read_bytes": 0}, "major_faults": 0, "minor_faults": minor})
            minor += change
            samples.append({"io": {"read_bytes": 0}, "major_faults": 0, "minor_faults": minor})
        class FakeNative:
            count = 0
            def threads(self, _):
                pass
            def scratch_state(self):
                return {"allocated_bytes": 4096 if self.count else 0, "successful_calls": self.count}
            def run(self, inputs, output, *_):
                shapes.append(inputs.shape)
                output[:] = inputs * np.float32(0.5)
                self.count += 1
        with patch.object(BENCH, "counters", side_effect=samples):
            results = BENCH.sweep_layer(self.plan(), 0, None, FakeNative(), args, np, emitted.append)
        self.assertEqual([(128, 2)] * 8, shapes)
        self.assertEqual(results, emitted)
        for result in results:
            self.assertEqual([1000], [r["minor_faults"] for r in result["warmup"]])
            self.assertEqual([23, 23, 23], [r["minor_faults"] for r in result["measured"]])
            self.assertTrue(result["timing_eligible"])
            self.assertEqual("numerics_passed", result["status"])
            self.assertTrue(all("process_counters" in r for r in result["warmup"] + result["measured"]))
            self.assertTrue(all(r["cpu_scratch_reuse"]["reused_on_this_call"] for r in result["warmup"] + result["measured"]))
            expected = BENCH.make_inputs(self.metadata, 128, args.seed + 128, np)[0] * np.float32(0.5)
            digest = BENCH.hashlib.sha256(expected.tobytes(order="C")).hexdigest()
            self.assertTrue(all(r["output_f32_sha256"] == digest for r in result["warmup"] + result["measured"]))

    def test_optional_scratch_query_abi_uses_uint64_and_reports_query_failure(self):
        def query(allocated, calls):
            ctypes.cast(allocated, ctypes.POINTER(ctypes.c_uint64))[0] = 1 << 35
            ctypes.cast(calls, ctypes.POINTER(ctypes.c_uint64))[0] = 29
            return 1
        fake = SimpleNamespace(**{name: Function(function) for name, function in {
            "TSGgml_MuseSparkMoEFFNQuantF32": lambda *args: 1,
            "TSGgml_IsBackendAvailable": lambda backend: 1,
            "TSGgml_GetLastError": lambda: b"fixture",
            "TSGgml_SetHostMoeThreads": lambda count: None,
            "TSGgml_Shutdown": lambda: None,
            "TSGgml_GetMuseSparkCpuMoeScratchState": query}.items()})
        native = BENCH.NativeMoE("unused", loader=lambda _: fake)
        self.assertEqual([ctypes.POINTER(ctypes.c_uint64)] * 2, fake.TSGgml_GetMuseSparkCpuMoeScratchState.argtypes)
        self.assertEqual({"allocated_bytes": 1 << 35, "successful_calls": 29}, native.scratch_state())
        fake.TSGgml_GetMuseSparkCpuMoeScratchState.implementation = lambda *_: 0
        with self.assertRaisesRegex(RuntimeError, "query failed"):
            native.scratch_state()

    def test_scratch_evidence_distinguishes_fallback_unavailable_and_successful_reuse(self):
        before = {"allocated_bytes": 4096, "successful_calls": 9}
        self.assertIsNone(BENCH.scratch_evidence(None, None)["reused_on_this_call"])
        self.assertFalse(BENCH.scratch_evidence(before, before)["reused_on_this_call"])
        # A failed reallocation may discard the arena and use fresh scratch.
        self.assertFalse(BENCH.scratch_evidence(before, {"allocated_bytes": 0, "successful_calls": 9})["reused_on_this_call"])
        self.assertTrue(BENCH.scratch_evidence(before, {"allocated_bytes": 8192, "successful_calls": 10})["reused_on_this_call"])
        for after in ({"allocated_bytes": 8192, "successful_calls": 8},
                      {"allocated_bytes": 8192, "successful_calls": 11},
                      {"allocated_bytes": 0, "successful_calls": 10},
                      {"allocated_bytes": True, "successful_calls": 10}):
            with self.subTest(after=after), self.assertRaises(ValueError):
                BENCH.scratch_evidence(before, after)

    def test_sweep_separates_warmup_checks_all_repeats_and_invalidates_disk_fault_timings(self):
        args = SimpleNamespace(rows=[1, 3], threads=[8, 16], warmup=1, repeats=3, seed=7, atol=1e-4, rtol=1e-4)
        class FakeNative:
            def __init__(self):
                self.count = 0
                self.teams = []
            def threads(self, count):
                self.teams.append(count)
            def run(self, inputs, output, *_):
                self.count += 1
                output[:] = inputs * np.float32(0.5)
        native, emitted = FakeNative(), []
        # Every call gets one read byte: outputs still pass, timings cannot
        # enter a warm-cache best-thread recommendation.
        sequence = iter({"io": {"read_bytes": index}, "major_faults": 0} for index in range(32))
        with patch.object(BENCH, "counters", side_effect=lambda: next(sequence)):
            rows = BENCH.sweep_layer(self.plan(), 0, None, native, args, np, emitted.append)
        self.assertEqual(16, native.count)
        self.assertEqual([8, 16, 8, 16], native.teams)
        self.assertEqual(4, len(emitted))
        self.assertTrue(all(len(row["warmup"]) == 1 and len(row["measured"]) == 3 for row in rows))
        self.assertTrue(all(row["status"] == "numerics_passed" and not row["timing_eligible"] for row in rows))

    def test_cuda_mode_takes_same_input_cpu_reference_before_streaming(self):
        args = SimpleNamespace(rows=[128], threads=[32], warmup=1, repeats=3, seed=7,
                               atol=1e-4, rtol=1e-4, backend="cuda-stream")
        flags, inputs_seen, emitted = [], [], []
        class FakeNative:
            def threads(self, _):
                pass
            def run(self, inputs, output, *_):
                flags.append(BENCH.os.environ["TS_MUSE_SPARK_STREAM_EXPERTS"])
                inputs_seen.append(inputs.copy())
                output[:] = inputs * np.float32(0.5)
        quiet = {"io": {"read_bytes": 0}, "major_faults": 0}
        with patch.dict(BENCH.os.environ), patch.object(BENCH, "counters", return_value=quiet):
            BENCH.sweep_layer(self.plan(), 0, None, FakeNative(), args, np, emitted.append)
        self.assertEqual(["0"] * 4 + ["1"] * 4, flags)
        self.assertEqual("cpu_reference", emitted[0]["kind"])
        self.assertEqual(2, len(emitted))
        self.assertTrue(all(np.array_equal(inputs_seen[0], value) for value in inputs_seen))
        self.assertEqual(emitted[0]["input"], emitted[1]["input"])

    def test_streaming_requires_positive_native_evidence_and_rejects_any_fallback(self):
        path = Path(self.temp.name) / "native.log"
        template = "[muse-spark-stream] rank=0 rows=256 decision={} reason=fixture required=600 free=800 reusable=0\n"
        for data, expected in (("", False), (template.format("cpu"), False),
                               (template.format("stream"), True),
                               (template.format("stream") + template.format("cpu"), False)):
            path.write_text(data)
            self.assertIs(expected, BENCH.stream_decisions(path, [256])["stream_confirmed"])
        path.write_text(template.format("stream"))
        self.assertFalse(BENCH.stream_decisions(path, [128, 256])["stream_confirmed"])

    def test_oracle_quantization_uses_distinct_block_sizes_and_tie_rules(self):
        source = np.zeros(256, dtype=np.float32)
        source[:3] = [127, .5, -.5]
        source[32:35] = [12.7, 1.0, -1.0]
        cpu = ORACLE.quantized_activation(source, "cpu_q8k", False, np)
        gpu = ORACLE.quantized_activation(source, "cuda_q8_1", False, np)
        np.testing.assert_array_equal(cpu[:3], [127, 0, 0])
        np.testing.assert_array_equal(gpu[:3], [127, 1, -1])
        self.assertEqual(13, cpu[32])
        self.assertAlmostEqual(float(source[32]), gpu[32], places=5)
        np.testing.assert_array_equal(source.astype(np.float64), ORACLE.quantized_activation(source, "exact", False, np))

    def test_oracle_cuda_half_scale_and_zero_blocks_remain_finite(self):
        source = np.zeros(256, dtype=np.float32)
        source[:2] = [1.0, .25]
        gpu = ORACLE.quantized_activation(source, "cuda_q8_1", True, np)
        expected_scale = float(np.float16(np.float32(1) / np.float32(127)))
        self.assertEqual(127 * expected_scale, gpu[0])
        self.assertEqual(32 * expected_scale, gpu[1])
        self.assertTrue(np.isfinite(gpu).all())
        np.testing.assert_array_equal(gpu[32:], np.zeros(224))


if __name__ == "__main__":
    unittest.main()
