"""Portable fixture guards for the component runner; no GPU/model/build access."""
import copy
import importlib.util
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import Mock, patch
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("spark_tp_validation",
    ROOT / "eng/validation/validate-muse-spark-tp.py")
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


class TpValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()

    def tearDown(self):
        self.temp.cleanup()

    def put(self, relative, content=b"fixture"):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def json(self, relative, value):
        return self.put(relative, json.dumps(value).encode())

    def build_fixture(self):
        source = self.put("src/model.cs")
        manifest = self.json("source.json", {"files": {"src/model.cs": TOOL.sha(source)}})
        runtime = self.root / "runtime"
        dll = self.put("runtime/InferenceWeb.Tests.dll", b"assembly")
        native = self.put("runtime/libGgmlOps.so", b"native")
        build = self.json("build.json", {"status": "pass", "source_manifest_sha256": TOOL.sha(manifest),
            "files": {str(path): TOOL.sha(path) for path in (dll, native)}})
        return runtime, manifest, build

    def trx(self, names, outcomes=None):
        ns = "{http://microsoft.com/schemas/VisualStudio/TeamTest/2010}"
        doc = ET.Element(ns + "TestRun")
        results = ET.SubElement(doc, ns + "Results")
        for name, outcome in zip(names, outcomes or ["Passed"] * len(names)):
            ET.SubElement(results, ns + "UnitTestResult", testName=name, outcome=outcome)
        path = self.root / "test.trx"
        ET.ElementTree(doc).write(path, encoding="utf-8", xml_declaration=True)
        return path

    def test_exact_managed_inventory_rejects_fail_skip_missing_extra_and_duplicate(self):
        expected = TOOL.lanes()["views-ranks-2"]["expected"]
        good = TOOL.read_cases(self.trx(expected))
        self.assertEqual(2, TOOL.check_cases(good, expected, 0)["passed"])
        negatives = [good[:-1], good + [good[0]], [good[0], good[0]],
                     good + [{"name": "unexpected", "outcome": "Passed"}]]
        for outcome in ("Failed", "NotExecuted", "Inconclusive"):
            bad = copy.deepcopy(good); bad[0]["outcome"] = outcome; negatives.append(bad)
        for cases in negatives:
            with self.subTest(cases=cases), self.assertRaises(ValueError):
                TOOL.check_cases(cases, expected, 0)
        with self.assertRaisesRegex(ValueError, "process failed"):
            TOOL.check_cases(good, expected, 1)

    def test_all_35_native_lanes_required_including_cache_recovery_and_precision(self):
        self.assertEqual(35, len(TOOL.NATIVE_TESTS))
        rows = [{"name": name, "outcome": "Passed"} for name in sorted(TOOL.NATIVE_TESTS)]
        self.assertEqual(35, TOOL.check_cases(rows, TOOL.NATIVE_TESTS, 0)["passed"])
        required = ["addmm-quant-cache-recovery-" + suffix for suffix in ("cpu", "cuda")]
        required += ["muse-spark-projection-precision-" + suffix for suffix in ("cpu", "cuda", "cuda-multi")]
        required += ["muse-spark-attention-precision-" + suffix for suffix in ("cpu", "cuda", "cuda-multi")]
        for missing in required:
            with self.assertRaises(ValueError):
                TOOL.check_cases([row for row in rows if row["name"] != missing], TOOL.NATIVE_TESTS, 0)
            skipped = [dict(row, outcome="NotExecuted") if row["name"] == missing else row for row in rows]
            with self.assertRaises(ValueError):
                TOOL.check_cases(skipped, TOOL.NATIVE_TESTS, 0)
        cmake = (ROOT / "TensorSharp.GGML.Native/CMakeLists.txt").read_text()
        for name in TOOL.NATIVE_TESTS:
            self.assertIn("add_test(NAME " + name + " ", cmake)

    def test_native_xml_failure_error_skip_and_notrun_never_pass(self):
        for child, status in (("failure", None), ("error", None), ("skipped", None), (None, "notrun")):
            doc = ET.Element("testsuite")
            case = ET.SubElement(doc, "testcase", name="native")
            if child: ET.SubElement(case, child)
            if status: case.set("status", status)
            path = self.root / "native.xml"
            ET.ElementTree(doc).write(path)
            with self.subTest(child=child, status=status), self.assertRaises(ValueError):
                TOOL.check_cases(TOOL.read_cases(path, native=True), ["native"], 0)

    def test_owned_attention_requires_binding_guards_and_current_head64_cases(self):
        selected = TOOL.select_lanes("all")
        binding = selected["cpu-attention-f32-binding"]
        self.assertEqual(0, binding["devices"])
        self.assertEqual(2, len(binding["expected"]))
        rows = TOOL.read_cases(self.trx(binding["expected"]))
        self.assertEqual(2, TOOL.check_cases(rows, binding["expected"], 0)["passed"])
        for index in range(2):
            with self.assertRaises(ValueError):
                TOOL.check_cases(rows[:index] + rows[index + 1:], binding["expected"], 0)
            unavailable = copy.deepcopy(rows)
            unavailable[index]["outcome"] = "NotExecuted"
            with self.assertRaises(ValueError):
                TOOL.check_cases(unavailable, binding["expected"], 0)
        for degree, word in ((5, "Five"), (9, "Nine")):
            item = selected[f"head64-ranks-{degree}"]
            self.assertEqual(degree, item["devices"])
            expected = TOOL.PREFIX + "MuseSparkTensorParallelFlashAttentionTests." + word + \
                "RanksPublishedHeadGeometryPackedF32MatchesOrdinaryAndScalar"
            self.assertEqual([expected], item["expected"])
            # A historical Flash-named result cannot satisfy the repaired path.
            old = expected.replace("PackedF32", "PackedFlash")
            with self.assertRaises(ValueError):
                TOOL.check_cases([{"name": old, "outcome": "Passed"}], item["expected"], 0)

    def test_cpu_exact_21_cases_include_theory_rows_and_contracts(self):
        item = TOOL.lanes()["cpu-tp"]
        self.assertEqual(21, len(item["expected"]))
        self.assertEqual(21, len(set(item["expected"])))
        self.assertEqual(0, item["devices"])
        self.assertTrue(any("(degree: 9, halfKv: True)" in name for name in item["expected"]))
        self.assertEqual(2, sum("PlanSlotContractTests" in name for name in item["expected"]))
        TOOL.check_cases(TOOL.read_cases(self.trx(item["expected"])), item["expected"], 0)
        # A collapsed theory placeholder cannot replace the seven exact rows.
        bad = [row for row in item["expected"] if "halfKv:" not in row]
        bad.append(item["methods"][0])
        with self.assertRaises(ValueError):
            TOOL.check_cases(TOOL.read_cases(self.trx(bad)), item["expected"], 0)

    def test_host_pool_and_asymmetric_placement_lanes_require_every_case(self):
        selected = TOOL.lanes()
        for name, count in (("cpu-host-pool", 8), ("cpu-expert-placement", 12)):
            item = selected[name]
            self.assertEqual(0, item["devices"])
            self.assertFalse(item["diagnostic"])
            self.assertEqual(count, len(set(item["expected"])))
            cases = TOOL.read_cases(self.trx(item["expected"]))
            self.assertEqual(count, TOOL.check_cases(cases, item["expected"], 0)["passed"])
            for index in range(count):
                with self.assertRaises(ValueError):
                    TOOL.check_cases(cases[:index] + cases[index + 1:], item["expected"], 0)
                skipped = copy.deepcopy(cases)
                skipped[index]["outcome"] = "NotExecuted"
                with self.assertRaises(ValueError):
                    TOOL.check_cases(skipped, item["expected"], 0)
        for name in ("expert-placement-f32-ranks-3", "expert-placement-q8-ranks-3"):
            item = selected[name]
            self.assertEqual(3, item["devices"])
            self.assertFalse(item["diagnostic"])
            self.assertEqual(1, len(item["expected"]))
        self.assertEqual("1", selected["optimized-ranks-9"]["environment"]["TS_MUSE_SPARK_TP_HOST_POOL"])

    def test_new_packed_and_timing_controls_are_explicit_and_preserve_wide_gate(self):
        selected = TOOL.lanes()
        self.assertEqual(2, len(selected["cpu-packed-controls"]["expected"]))
        timing = selected["cpu-phase-timing"]
        self.assertEqual(6, len(timing["expected"]))
        self.assertEqual(6, len(set(timing["expected"])))
        TOOL.check_cases(TOOL.read_cases(self.trx(timing["expected"])), timing["expected"], 0)
        for missing in timing["expected"]:
            with self.subTest(missing=missing), self.assertRaises(ValueError):
                TOOL.check_cases(TOOL.read_cases(self.trx([n for n in timing["expected"] if n != missing])),
                                 timing["expected"], 0)
        for name in ("narrow-packed-ranks-5", "diagnostic-packed-stages"):
            self.assertEqual(5, selected[name]["devices"])
            with self.subTest(name=name), self.assertRaises(ValueError):
                TOOL.check_cases(TOOL.read_cases(self.trx(selected[name]["expected"], ["NotExecuted"])),
                                 selected[name]["expected"], 0)
        self.assertFalse(selected["narrow-packed-ranks-5"]["diagnostic"])
        self.assertTrue(selected["diagnostic-packed-stages"]["diagnostic"])
        self.assertFalse(selected["packed-ranks-5"]["diagnostic"])
        self.assertIn("packed-ranks-5", TOOL.select_lanes("all"))
        self.assertNotIn("diagnostic-packed-stages", TOOL.select_lanes("all"))

    def test_qk_required_lanes_keep_managed_native_and_gpu_coverage_distinct(self):
        selected = TOOL.select_lanes("all")
        prefix = TOOL.PREFIX + "MuseSparkTensorParallelQKPrepTests."
        facts = {
            "cpu-qk-prep-native": (0, "NativeCpuLocalHeadGroupsRetainPrimitiveOracleAcrossRowsAndPositions"),
            "qk-prep-ranks-2": (2, "NativeCudaLocalHeadGroupsRetainPrimitiveOracleAcrossRanksRowsAndPositions"),
            "qk-cache-ranks-9": (9, "FusedRankPreparationPreservesQuantizedCacheOffloadAndScalarControls"),
            "qk-head64-ranks-9": (9, "FusedRankPreparationPreservesPublishedHeadGeometryAndPackedSchedules"),
        }
        for name, (devices, method) in facts.items():
            item = selected[name]
            self.assertEqual(devices, item["devices"])
            self.assertEqual([prefix + method], item["methods"])
            self.assertEqual(item["methods"], item["expected"])
            self.assertFalse(item["diagnostic"])
            self.assertEqual(1, TOOL.check_cases(
                TOOL.read_cases(self.trx(item["expected"])), item["expected"], 0)["passed"])
            for outcomes in (["Failed"], ["NotExecuted"]):
                with self.subTest(lane=name, outcomes=outcomes), self.assertRaises(ValueError):
                    TOOL.check_cases(TOOL.read_cases(self.trx(item["expected"], outcomes)), item["expected"], 0)
            with self.subTest(lane=name), self.assertRaises(ValueError):
                TOOL.check_cases([], item["expected"], 0)
        for name in ("cpu-qk-prep", *facts):
            item = selected[name]
            environment = TOOL.environment(
                {"TS_MUSE_SPARK_FUSE_QK_PREP": "0", "CUDA_VISIBLE_DEVICES": "old"},
                Path("/runtime"), Path("/cuda"), [str(i) for i in range(9)], item, Path("/arrays"))
            self.assertEqual("1", environment["TS_MUSE_SPARK_FUSE_QK_PREP"])
            self.assertEqual(",".join(str(i) for i in range(item["devices"])),
                             environment["CUDA_VISIBLE_DEVICES"])

    def test_qk_packing_theory_requires_all_five_rows_and_both_fallback_guards(self):
        item = TOOL.select_lanes("all")["cpu-qk-prep"]
        self.assertEqual(0, item["devices"])
        self.assertEqual(7, len(set(item["expected"])))
        theory = item["methods"][0]
        expected_rows = [theory + f"(rows: {rows}, kvHeads: {heads})"
                         for rows, heads in ((1, 1), (2, 2), (9, 1), (128, 2), (256, 1))]
        self.assertCountEqual(item["expected"], item["methods"][1:] + expected_rows)
        good = TOOL.read_cases(self.trx(item["expected"]))
        self.assertEqual(7, TOOL.check_cases(good, item["expected"], 0)["passed"])
        for index in range(7):
            with self.subTest(missing=index), self.assertRaises(ValueError):
                TOOL.check_cases(good[:index] + good[index + 1:], item["expected"], 0)
            skipped = copy.deepcopy(good)
            skipped[index]["outcome"] = "NotExecuted"
            with self.subTest(skipped=index), self.assertRaises(ValueError):
                TOOL.check_cases(skipped, item["expected"], 0)
        collapsed = TOOL.read_cases(self.trx(item["methods"]))
        with self.assertRaises(ValueError):
            TOOL.check_cases(collapsed, item["expected"], 0)

    def test_expert_plane_residency_is_required_and_unavailable_is_not_pass(self):
        selected = TOOL.select_lanes("all")
        item = selected["expert-residency-ranks-2"]
        self.assertEqual(2, item["devices"])
        expected = [TOOL.GPU_CLASS + "FloatExpertPlaneResidencyReleasesAndReloadsAcrossPlacementModes"]
        self.assertEqual(expected, item["methods"])
        self.assertEqual(expected, item["expected"])
        self.assertFalse(item["diagnostic"])
        for outcomes in (["NotExecuted"], ["Failed"]):
            report = self.trx(expected, outcomes)
            with self.assertRaises(ValueError):
                TOOL.check_cases(TOOL.read_cases(report, False), expected, 0)
        with self.assertRaises(ValueError):
            TOOL.check_cases([], expected, 0)

    def test_each_selected_method_exists_in_actual_test_source(self):
        sources = [(path, path.read_text(encoding="utf-8-sig"))
                   for path in (ROOT / "InferenceWeb.Tests").glob("*.cs")]
        for label, item in TOOL.lanes().items():
            for method in item["methods"]:
                cls, name = method.rsplit(".", 1)
                class_name = cls.rsplit(".", 1)[1]
                declarations = [text for _, text in sources
                                if re.search(r"\bclass\s+" + re.escape(class_name) + r"\b", text)]
                with self.subTest(lane=label, method=method):
                    self.assertTrue(declarations, class_name)
                    self.assertRegex("\n".join(declarations),
                                     r"public (?:async )?(?:void|Task) " + re.escape(name) + r"\(")

    def test_opaque_quantized_residency_requires_actual_two_gpu_pass(self):
        item = TOOL.select_lanes("all")["quantized-residency-ranks-2"]
        expected = [TOOL.PREFIX + "MuseSparkQuantizedResidencyTests.QuantizedOpaqueKeysReleaseAndReloadAcrossPlacementModes"]
        self.assertEqual(2, item["devices"])
        self.assertEqual(expected, item["methods"])
        self.assertEqual(expected, item["expected"])
        self.assertFalse(item["diagnostic"])
        passed = TOOL.read_cases(self.trx(expected))
        self.assertEqual(1, TOOL.check_cases(passed, expected, 0)["passed"])
        for outcomes in (["NotExecuted"], ["Failed"]):
            with self.subTest(outcome=outcomes), self.assertRaises(ValueError):
                TOOL.check_cases(TOOL.read_cases(self.trx(expected, outcomes), False), expected, 0)
        with self.assertRaises(ValueError):
            TOOL.check_cases([], expected, 0)

    def test_required_inventory_keeps_known_failing_regressions_and_separates_diagnostics(self):
        all_lanes = TOOL.select_lanes("all")
        self.assertIn("packed-ranks-5", all_lanes)
        self.assertIn("optimized-ranks-9", all_lanes)
        self.assertIn("head64-ranks-9", all_lanes)
        self.assertIn("residual-off-ranks-9", all_lanes)
        self.assertFalse(any(lane["diagnostic"] for lane in all_lanes.values()))
        self.assertTrue(TOOL.select_lanes("diagnostic-packed-head")["diagnostic-packed-head"]["diagnostic"])
        for raw in ("", "not-a-lane", "cpu-tp,cpu-tp"):
            with self.assertRaises(ValueError): TOOL.select_lanes(raw)

    def test_actual_physical_device_inventory_and_rank_requirements(self):
        raw = "\n".join(f"{i}, GPU-{i}, Test GPU" for i in range(9))
        devices = TOOL.parse_devices("0,1,2,3,4,5,6,7,8")
        selected = TOOL.select_lanes("q8-ranks-9")
        self.assertEqual(9, len(TOOL.device_inventory(raw, devices, selected)))
        for chosen in (devices[:8], devices[:-1] + ["99"]):
            with self.assertRaises(ValueError): TOOL.device_inventory(raw, chosen, selected)
        for raw_devices in ("0,0", "0,-1", "0,GPU-id", "0, 1"):
            with self.assertRaises(ValueError): TOOL.parse_devices(raw_devices)
        with self.assertRaises(ValueError): TOOL.device_inventory(raw + "\n0, other, GPU", devices, selected)
        self.assertEqual(2, TOOL.lanes()["native"]["devices"])

    def test_environment_removes_ambient_tuning_and_sets_library_load_latched_flag(self):
        base = {"PATH": "/bin", "TS_MUSE_SPARK_FUSE_RESIDUALS": "1", "TS_UNKNOWN_TUNING": "1",
                "TENSORSHARP_TP_DEGREE": "9", "NCCL_P2P_DISABLE": "1", "LD_PRELOAD": "wrong.so",
                "GGML_CUDA_DISABLE_FUSION": "1", "GGML_CUDA_CUBLAS_COMPUTE_TYPE": "16",
                "GGML_VK_FORCE_MAX_ALLOCATION_SIZE": "32"}
        ordinary = TOOL.environment(base, Path("/runtime"), Path("/cuda"), ["3", "4"],
                                    TOOL.lanes()["q8-ranks-2"], Path("/arrays"))
        self.assertEqual("3,4", ordinary["CUDA_VISIBLE_DEVICES"])
        for key in ("TS_MUSE_SPARK_FUSE_RESIDUALS", "TS_UNKNOWN_TUNING", "TENSORSHARP_TP_DEGREE",
                    "LD_PRELOAD", "NCCL_P2P_DISABLE", "GGML_CUDA_DISABLE_FUSION",
                    "GGML_CUDA_CUBLAS_COMPUTE_TYPE", "GGML_VK_FORCE_MAX_ALLOCATION_SIZE"):
            self.assertNotIn(key, ordinary)
        self.assertEqual("0", ordinary["TS_MUSE_SPARK_REUSE_CPU_MOE_SCRATCH"])
        optimized = TOOL.environment(base, Path("/runtime"), Path("/cuda"), [str(i) for i in range(9)],
                                     TOOL.lanes()["optimized-ranks-9"], Path("/arrays"))
        self.assertEqual("1", optimized["TS_MUSE_SPARK_REUSE_CPU_MOE_SCRATCH"])
        cpu = TOOL.environment(base, Path("/runtime"), Path("/cuda"), ["3"], TOOL.lanes()["cpu-tp"], Path("/arrays"))
        self.assertEqual("", cpu["CUDA_VISIBLE_DEVICES"])

    def test_build_must_bind_exact_source_and_complete_runtime_inventory(self):
        runtime, manifest, build = self.build_fixture()
        pins, inventory = TOOL.bind_build(self.root, runtime, manifest, build)
        self.assertEqual(2, len(inventory))
        self.assertIn(str(manifest), pins)
        extra = self.put("runtime/unexpected.dll", b"stale")
        with self.assertRaisesRegex(ValueError, "Runtime inventory"):
            TOOL.bind_build(self.root, runtime, manifest, build)
        extra.unlink()
        (runtime / "InferenceWeb.Tests.dll").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Changed pinned"):
            TOOL.bind_build(self.root, runtime, manifest, build)

    def test_changed_source_or_mismatched_build_manifest_rejected(self):
        runtime, manifest, build = self.build_fixture()
        (self.root / "src/model.cs").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Changed pinned"):
            TOOL.bind_build(self.root, runtime, manifest, build)
        data = TOOL.read_json(build); data["source_manifest_sha256"] = "0" * 64
        build.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "Build/source"):
            TOOL.bind_build(self.root, runtime, manifest, build)

    def test_unsafe_source_paths_hashes_and_duplicate_json_fail_closed(self):
        for name in ("../outside.cs", "/outside.cs", "C:/outside.cs", "src\\bad.cs", "src//bad.cs"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                TOOL.source_pins(self.root, {"files": {name: "a" * 64}})
        with self.assertRaises(ValueError): TOOL.hash_map({"one": True})
        with self.assertRaises(ValueError): TOOL.hash_map({"one": "a" * 64, "ONE": "a" * 64})
        path = self.put("duplicate.json", b'{"files": {}, "files": {}}')
        with self.assertRaisesRegex(ValueError, "Duplicate"): TOOL.read_json(path)

    def test_merging_new_pins_cannot_replace_prior_expected_hash(self):
        pins = {"same": "a" * 64}
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            TOOL.merge_pins(pins, {"same": "b" * 64})
        self.assertEqual("a" * 64, pins["same"])
        TOOL.merge_pins(pins, {"same": "a" * 64, "new": "c" * 64})

    def test_exclusive_outputs_preserve_original_failure(self):
        path = self.root / "result.json"
        TOOL.write_new(path, {"status": "fail", "error": "original"})
        old = path.read_bytes()
        with self.assertRaises(FileExistsError): TOOL.write_new(path, {"status": "pass"})
        self.assertEqual(old, path.read_bytes())

    def test_native_discovery_requires_every_exact_case_and_in_build_executables(self):
        build = self.root / "native"
        exe = self.put("native/test")
        self.put("native/libGgmlOps.so")
        self.put("native/CTestTestfile.cmake")
        rows = [{"name": name, "command": [str(exe), "cpu"]} for name in sorted(TOOL.NATIVE_TESTS)]
        with patch.object(TOOL.subprocess, "check_output", return_value=json.dumps({"tests": rows})):
            command, _, pins = TOOL.native_commands("ctest", build)
        self.assertIn(str(exe), pins)
        self.assertIn("addmm-quant-cache-recovery-cuda", command[-1])
        for bad in (rows[:-1], rows + [rows[0]],
                    [dict(rows[0], command=["/outside/test"])] + rows[1:]):
            with patch.object(TOOL.subprocess, "check_output", return_value=json.dumps({"tests": bad})):
                with self.assertRaises(ValueError): TOOL.native_commands("ctest", build)

    def test_native_executables_must_match_prior_build_receipt(self):
        exe = self.put("native/test", b"executable")
        lib = self.put("native/libGgmlOps.so", b"library")
        rows = [{"command": [str(exe)]}]
        record = {"files": {str(p): TOOL.sha(p) for p in (exe, lib)}}
        TOOL.verify_native_build_binding(rows, lib, record)
        with self.assertRaisesRegex(ValueError, "does not pin"):
            TOOL.verify_native_build_binding(rows, lib, {"files": {str(lib): TOOL.sha(lib)}})
        exe.write_bytes(b"unrecorded rebuilt test")
        with self.assertRaisesRegex(ValueError, "Changed pinned"):
            TOOL.verify_native_build_binding(rows, lib, record)

    def test_child_timeout_kills_only_owned_group_and_restores_signal_handler(self):
        child = Mock(pid=123)
        child.wait.side_effect = [TOOL.subprocess.TimeoutExpired(["test"], 1), 0]
        previous = object()
        with patch.object(TOOL.subprocess, "Popen", return_value=child) as start, \
             patch.object(TOOL.signal, "signal", return_value=previous) as handler, \
             patch.object(TOOL.os, "killpg", create=True) as kill, \
             self.assertRaises(TOOL.subprocess.TimeoutExpired):
            TOOL.run_child(["test"], self.root / "timeout.log", self.root, {}, 1)
        kill.assert_called_once_with(123, TOOL.signal.SIGTERM)
        self.assertEqual((TOOL.signal.SIGTERM, previous), handler.call_args.args)
        self.assertTrue(start.call_args.kwargs["start_new_session"])
        self.assertTrue(start.call_args.kwargs["close_fds"])

    def test_upstream_must_be_the_recorded_unmodified_revision(self):
        for outputs in ([TOOL.GGML_REVISION, ""], ["wrong", ""], [TOOL.GGML_REVISION, " M src/ggml.c"]):
            with patch.object(TOOL.subprocess, "check_output", side_effect=outputs):
                if outputs == [TOOL.GGML_REVISION, ""]:
                    TOOL.verify_upstream(self.root)
                else:
                    with self.assertRaises(ValueError): TOOL.verify_upstream(self.root)


if __name__ == "__main__":
    unittest.main()
