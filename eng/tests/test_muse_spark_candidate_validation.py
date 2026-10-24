"""Local mocks only: no server, GPU, model access, builds or VM operations."""
import copy
import importlib.util
import json
import re
from pathlib import Path
import tempfile
import tarfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("spark_candidate",
    Path(__file__).parents[1] / "validation/muse-spark-candidate-validation.py")
GUARD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GUARD)


class CandidateValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def report(self, flag):
        report = {"status": "numerics_passed", "cpu_scratch_reuse": {"requested_value": str(flag)},
            "configuration": {"backend": "cpu", "layers": [0], "rows": [1, 128, 256], "threads": [32],
                "warmup": 1, "repeats": 3, "atol": 0, "rtol": 0, "output": f"flag{flag}.json"},
            "plan": {"tensors": {"a": {"offset": 32, "bytes": 512, "mtime_ns": 123}}},
            "native_library": {"path": "same.so", "sha256": "b" * 64}, "source_sha256": {"tool.py": "c" * 64},
            "reader_source": {"sha256": "d" * 64}, "capacity": {"environment": {"TS_MUSE_SPARK_REUSE_CPU_MOE_SCRATCH": str(flag)}, "cpu": 40},
            "results": []}
        count = 0
        for rows in (1, 128, 256):
            calls = []
            for repeat in range(4):
                before = {"io": {"read_bytes": 0}, "major_faults": 0, "minor_faults": repeat * 100}
                after = copy.deepcopy(before); after["minor_faults"] += 2 if flag else 83
                state = {"allocated_bytes": 4096 if flag and count else 0, "successful_calls": count}
                count += flag
                calls.append({"wall_ms": 4 if flag else 8, "output_f32_sha256": (str(rows)[0]) * 64,
                    "read_bytes": 0, "major_faults": 0, "minor_faults": after["minor_faults"] - before["minor_faults"], "cache_quiet": True,
                    "process_counters": {"before": before, "after": after},
                    "cpu_scratch_reuse": {"before": state, "after": {"allocated_bytes": 4096 if flag else 0, "successful_calls": count},
                        "successful_calls_delta": flag, "reused_on_this_call": bool(flag)}})
            report["results"].append({"layer": 0, "rows": rows, "threads": 32, "status": "numerics_passed",
                "input": {"hash": "a" * 64}, "warmup": calls[:1], "measured": calls[1:]})
        return report

    def paths(self, reports):
        paths = [self.root / f"report{i}.json" for i in range(len(reports))]
        for path, report in zip(paths, reports):
            path.write_text(json.dumps(report))
        return paths

    def source_manifest(self, files=None):
        if files is None:
            files = {"project/model.cs": b"source", "fixtures/input.bin": b"fixture"}
        for name, data in files.items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        parent = self.root / "final-manifest.json"
        data = {"revision": "a" * 40, "original_clean": False, "hash_basis": "archive_plus_working_changes",
            "files": {name: GUARD.digest(self.root / name) for name in files}}
        parent.write_text(json.dumps(data))
        return parent

    def test_fresh_directory_is_exclusive_bounded_and_preserves_existing(self):
        base = self.root / "evidence"; base.mkdir()
        good = GUARD.fresh_evidence(base / "new" / "proof", base)
        (good / "keep").write_text("old evidence")
        for path in (good, base, self.root / "escape", Path("relative")):
            with self.subTest(path=path), self.assertRaises(ValueError):
                GUARD.fresh_evidence(path, base)
        self.assertEqual("old evidence", (good / "keep").read_text())

    def test_shell_uses_existing_validation_venv_for_every_python_phase(self):
        script = (Path(__file__).parents[1] / "validation/validate-muse-spark-final-candidate.sh").read_text()
        self.assertIn('python=/workspace/venv/bin/python', script)
        self.assertIn('[[ -x "$python" ]]', script)
        self.assertNotIn('python3', script)
        for tool in ('guard', 'warm', 'micro'):
            self.assertIn('"$python" "$' + tool + '"', script)

    def test_shell_builds_norm_target_and_does_not_inherit_norm_optin(self):
        script = (Path(__file__).parents[1] / "validation/validate-muse-spark-final-candidate.sh").read_text()
        self.assertIn("GgmlOpsMuseSparkNormGateTest", script)
        self.assertIn("|muse-spark-norm-gate|", script)
        unsets = [line for line in script.splitlines() if line.startswith("unset ")]
        self.assertTrue(any("TS_MUSE_SPARK_REUSE_NORM_SCRATCH" in line.split() for line in unsets))

    def test_idle_guard_detects_servers_without_model_flag_but_not_mentions(self):
        proc = self.root / "proc"; proc.mkdir()
        one = proc / "123"; one.mkdir()
        for argv in (b"dotnet\0/work/TensorSharp.Server.Host.dll\0", b"/work/llama-server\0", b"/work/TensorSharp.Server.Host\0"):
            (one / "cmdline").write_bytes(argv)
            with self.subTest(argv=argv), self.assertRaisesRegex(ValueError, "Active full server"):
                GUARD.assert_idle(proc)
        (one / "cmdline").write_bytes(b"python3\0script.py\0an explanation mentioning llama-server\0")
        GUARD.assert_idle(proc)

    def test_missing_proc_and_unreadable_process_fail_closed(self):
        with self.assertRaises(ValueError): GUARD.assert_idle(self.root / "missing")
        proc = self.root / "proc"; proc.mkdir(); (proc / "123").mkdir()
        with patch.object(Path, "read_bytes", side_effect=PermissionError("denied")), self.assertRaisesRegex(ValueError, "Cannot inspect"):
            GUARD.assert_idle(proc)

    def test_upstream_revision_and_dirty_tree_are_rejected(self):
        for answers in ((b"bad\n",), (GUARD.GGML_REVISION.encode(), b" M src/ggml.c")):
            with patch.object(GUARD, "git", side_effect=answers), self.assertRaises(ValueError): GUARD.verify_upstream(self.root)
        with patch.object(GUARD, "git", side_effect=(GUARD.GGML_REVISION.encode(), b"")): GUARD.verify_upstream(self.root)

    def test_archive_without_git_verifies_every_file_and_pins_parent_and_snapshot(self):
        proof = self.root / "proof"; proof.mkdir()
        parent = self.source_manifest()
        original = parent.read_bytes()
        def upstream_only(root, *args):
            self.assertEqual(self.root / "ExternalProjects/ggml", root)
            return GUARD.GGML_REVISION.encode() if args == ("rev-parse", "HEAD") else b""
        self.assertFalse((self.root / ".git").exists())
        with patch.object(GUARD, "git", side_effect=upstream_only):
            GUARD.freeze(self.root, proof, parent)
            data = json.loads((proof / "source-manifest.json").read_text())
            self.assertEqual(json.loads(original)["files"], data["files"])
            self.assertEqual(2, data["coverage"]["checked_files"])
            self.assertFalse(data["coverage"]["source_clean_claim"])
            self.assertFalse(data["coverage"]["original_clean"])
            with tarfile.open(proof / "candidate-source.tar.gz") as archive:
                self.assertEqual(sorted(data["files"]), archive.getnames())
            (self.root / "fixtures/input.bin").write_text("mutation")
            with self.assertRaisesRegex(ValueError, "source hash mismatch"): GUARD.verify_sources(self.root, proof)
            (self.root / "fixtures/input.bin").write_bytes(b"fixture")
            parent.write_bytes(original + b"\n")
            with self.assertRaisesRegex(ValueError, "manifest changed"): GUARD.verify_sources(self.root, proof)
            parent.write_bytes(original)
            (proof / "source-manifest.json").write_text(json.dumps(dict(data, files={})))
            with self.assertRaisesRegex(ValueError, "Candidate source manifest changed"): GUARD.verify_sources(self.root, proof)

    def test_freeze_rejects_wrong_parent_file_hash_or_missing_file_before_any_snapshot(self):
        proof = self.root / "proof"; proof.mkdir()
        parent = self.source_manifest()
        with patch.object(GUARD, "verify_upstream"):
            (self.root / "fixtures/input.bin").write_bytes(b"wrong archive")
            with self.assertRaisesRegex(ValueError, "source hash mismatch"): GUARD.freeze(self.root, proof, parent)
            self.assertEqual([], list(proof.iterdir()))
            (self.root / "fixtures/input.bin").unlink()
            with self.assertRaisesRegex(ValueError, "Missing/nonregular"): GUARD.freeze(self.root, proof, parent)
            self.assertEqual([], list(proof.iterdir()))

    def test_manifest_requires_applied_dirty_basis_valid_hashes_and_unique_json_keys(self):
        parent = self.source_manifest(); original = json.loads(parent.read_text())
        for key, value in (("hash_basis", "git_archive"), ("original_clean", True), ("files", {}), ("revision", "bad")):
            parent.write_text(json.dumps(dict(original, **{key: value})))
            with self.subTest(key=key), self.assertRaises(ValueError): GUARD.applied_manifest(parent)
        parent.write_text('{"files": {}, "files": {}}')
        with self.assertRaisesRegex(ValueError, "Duplicate"): GUARD.applied_manifest(parent)
        original["files"]["project/model.cs"] = True
        with self.assertRaisesRegex(ValueError, "Invalid manifest source hash"): GUARD.source_files(self.root, original)

    def test_manifest_paths_reject_absolute_escape_alias_and_dependency_trees(self):
        parent = self.source_manifest(); original = json.loads(parent.read_text())
        for name in ("/etc/passwd", "../escape.cs", "project/../project/model.cs", "project//model.cs", "./project/model.cs",
                     "C:/escape.cs", "project\\model.cs", "project/model.cs:stream", "project/\nmodel.cs"):
            data = dict(original, files={name: "a" * 64})
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "Unsafe"): GUARD.source_files(self.root, data)
        path = self.root / "ExternalProjects/ggml/source.c"; path.parent.mkdir(parents=True); path.write_text("upstream")
        with self.assertRaisesRegex(ValueError, "generated/dependency"):
            GUARD.source_files(self.root, dict(original, files={"ExternalProjects/ggml/source.c": GUARD.digest(path)}))

    def test_manifest_rejects_file_and_directory_symlinks_even_with_matching_bytes(self):
        self.source_manifest()
        # Mock the classification only: this also exercises the guard on Windows
        # hosts without permission to create real symbolic links.
        actual = Path.is_symlink
        for target in (self.root / "project", self.root / "project/model.cs"):
            with self.subTest(target=target), patch.object(Path, "is_symlink", lambda p: p == target or actual(p)):
                with self.assertRaisesRegex(ValueError, "symlink"): GUARD.source_path(self.root, "project/model.cs")

    def test_unlisted_compiled_sources_and_build_controls_rejected_before_and_after_freeze(self):
        proof = self.root / "proof"; proof.mkdir(); parent = self.source_manifest()
        with patch.object(GUARD, "verify_upstream"):
            for name in ("project/stale.cs", "project/extra.targets", "unexpected/CMakeLists.txt"):
                path = self.root / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text("unlisted")
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, "Unlisted compiled"):
                    GUARD.freeze(self.root, proof, parent)
                self.assertEqual([], list(proof.iterdir()))
                path.unlink()
            for name in ("project/obj/linux/generated.cs", "TensorSharp.GGML.Native/build/CMakeFiles/generated.c",
                         "docs/validation/generated.cs", "ExternalProjects/ggml/src/ggml.c"):
                path = self.root / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text("excluded")
            GUARD.freeze(self.root, proof, parent)
            (self.root / "project/stale.cs").write_text("added later")
            with self.assertRaisesRegex(ValueError, "Unlisted compiled"): GUARD.verify_sources(self.root, proof)

    def test_native_required_cases_must_execute_and_not_skip(self):
        path = self.root / "native.xml"
        cases = [f'<testcase name="{name}" status="run"/>' for name in sorted(GUARD.NATIVE_TESTS)]
        def write(values): path.write_text("<testsuite>" + "".join(values) + "</testsuite>")
        write(cases); self.assertEqual(27, GUARD.check_native(path)["passed"])
        for mutation in (cases[:-1], cases + cases[:1], [cases[0].replace('/>', '><skipped/></testcase>')] + cases[1:]):
            write(mutation)
            with self.assertRaises(ValueError): GUARD.check_native(path)

    def test_native_gate_rejects_missing_2d_cpu_single_and_multi_gpu_cases(self):
        path = self.root / "native.xml"
        for missing in ("muse-spark-kv-2d-cpu", "muse-spark-kv-2d-cuda", "muse-spark-kv-2d-cuda-multi"):
            self.assertIn(missing, GUARD.NATIVE_TESTS)
            path.write_text("<testsuite>" + "".join(f'<testcase name="{name}" status="run"/>'
                for name in sorted(GUARD.NATIVE_TESTS - {missing})) + "</testsuite>")
            with self.subTest(missing=missing), self.assertRaisesRegex(ValueError, "Missing"):
                GUARD.check_native(path)

    def test_native_gate_rejects_missing_norm_cpu_single_and_multi_gpu_cases(self):
        path = self.root / "native.xml"
        for missing in ("muse-spark-norm-gate-cpu", "muse-spark-norm-gate-cuda", "muse-spark-norm-gate-cuda-multi"):
            self.assertIn(missing, GUARD.NATIVE_TESTS)
            path.write_text("<testsuite>" + "".join(f'<testcase name="{name}" status="run"/>'
                for name in sorted(GUARD.NATIVE_TESTS - {missing})) + "</testsuite>")
            with self.subTest(missing=missing), self.assertRaisesRegex(ValueError, "Missing"):
                GUARD.check_native(path)

    def test_native_gate_rejects_missing_or_unavailable_paged_lifecycle_lanes(self):
        path = self.root / "native.xml"
        for target in ("paged-attention-lifecycle-cpu", "paged-attention-lifecycle-cuda",
                       "paged-attention-lifecycle-cuda-multi"):
            self.assertIn(target, GUARD.NATIVE_TESTS)
            for unavailable in (False, True):
                cases = []
                for name in sorted(GUARD.NATIVE_TESTS):
                    if name != target:
                        cases.append(f'<testcase name="{name}" status="run"/>')
                    elif unavailable:
                        cases.append(f'<testcase name="{name}" status="run"><skipped/></testcase>')
                path.write_text("<testsuite>" + "".join(cases) + "</testsuite>")
                with self.subTest(target=target, unavailable=unavailable), self.assertRaises(ValueError):
                    GUARD.check_native(path)

    def test_native_shell_builds_lifecycle_target_and_selects_all_required_cases(self):
        shell = (Path(__file__).parents[1] / "validation/validate-muse-spark-final-candidate.sh").read_text()
        build = shell.split("run native-build.log", 1)[1].split("run managed-rebuild.log", 1)[0]
        self.assertIn("GgmlOpsPagedAttentionLifecycleTest", build)
        selection = re.search(r"-R '([^']+)'", shell).group(1)
        self.assertEqual(27, len(GUARD.NATIVE_TESTS))
        self.assertTrue(all(re.search(selection, name) for name in GUARD.NATIVE_TESTS))

    def test_complete_component_gate_requires_exact_measured_cpu_and_expected_cuda_counts(self):
        def lane(path, cuda):
            return {"passed": 23 if cuda else 95, "failed": 0, "skipped": 0}
        with patch.object(GUARD, "check_native", return_value={"passed": 27}), \
             patch.object(GUARD, "check_managed", side_effect=lane):
            GUARD.check_tests(self.root)
        accepted = json.loads((self.root / "component-validation.json").read_text())
        self.assertEqual([95, 23, 95, 23], [v["passed"] for v in accepted["managed"].values()])
        for bad_lane, count in (("cpu", 94), ("cpu", 96), ("cuda", 22), ("cuda", 24)):
            with self.subTest(lane=bad_lane, count=count), \
                 patch.object(GUARD, "check_native", return_value={"passed": 27}), \
                 patch.object(GUARD, "check_managed", side_effect=lambda path, cuda: {
                     "passed": count if ("cuda" if cuda else "cpu") == bad_lane else (23 if cuda else 95)}), \
                 self.assertRaisesRegex(ValueError, "Unexpected .* test count"):
                GUARD.check_tests(self.root)

    def test_managed_gpu_gate_requires_actual_qk_and_2d_execution_tests_passed(self):
        path = self.root / "cuda.trx"
        names = (["InferenceWeb.Tests.MuseSparkQKPrepTests." + name for name in GUARD.QK_METHODS] +
            ["InferenceWeb.Tests.MuseSparkKv2DUploadTests." + name for name in GUARD.KV_2D_METHODS] + [
            "InferenceWeb.Tests.MuseSparkModelTests.NativeCudaCpuExpertsMatchIndependentDecoder",
            "InferenceWeb.Tests.MuseSparkResidualTests.NativeCudaResidualScratchReuseKeepsChangingInputsAndRetainedOutputs"] +
            ["InferenceWeb.Tests.MuseSparkNormScratchTests." + name for name in GUARD.NORM_CUDA_METHODS])
        def write(items, outcome="Passed"):
            path.write_text('<TestRun xmlns="urn:fixture"><Results>' + ''.join(
                f'<UnitTestResult testName="{name}" outcome="{outcome}"/>' for name in items) + '</Results></TestRun>')
        write(names); self.assertEqual(12, GUARD.check_managed(path, True)["passed"])
        write(names + names[:1])
        with self.assertRaisesRegex(ValueError, "Duplicate managed test result"):
            GUARD.check_managed(path, True)
        for missing in names:
            write([name for name in names if name != missing])
            with self.subTest(missing=missing), self.assertRaisesRegex(ValueError, "execution test missing"):
                GUARD.check_managed(path, True)
        for items, outcome in ((names[:-1], "Passed"), (names, "NotExecuted"), (names, "Failed"), ([], "Passed")):
            write(items, outcome)
            with self.assertRaises(ValueError): GUARD.check_managed(path, True)

    def test_managed_cpu_gate_requires_2d_model_execution(self):
        path = self.root / "cpu.trx"
        names = ["InferenceWeb.Tests.MuseSparkQKPrepTests.NativeCpuFusedQKPreservesChunkPrefixSpeculativeVisionAndPackedLogits",
            "InferenceWeb.Tests.MuseSparkResidualTests.NativeCpuResidualScratchReuseKeepsChangingInputsAndRetainedOutputs"] + [
            "InferenceWeb.Tests.MuseSparkNormScratchTests." + name for name in GUARD.NORM_CPU_METHODS]
        def write(items):
            path.write_text('<TestRun><Results>' + ''.join(
                f'<UnitTestResult testName="{name}" outcome="Passed"/>' for name in items) + '</Results></TestRun>')
        write(names)
        with self.assertRaisesRegex(ValueError, "actual-2D/actual-norm execution test missing"):
            GUARD.check_managed(path, False)
        names.append("InferenceWeb.Tests.MuseSparkKv2DUploadTests.NativeCpu2DUploadsPreserveChunkPrefixAndSpeculativeLogits")
        write(names); self.assertEqual(6, GUARD.check_managed(path, False)["passed"])

    def test_ab_accepts_exact_counterbalanced_process_reports_and_derives_timings(self):
        result = GUARD.compare_ab(self.paths([self.report(flag) for flag in (0, 1, 1, 0)]))
        self.assertEqual("exact_outputs_and_disk_quiet_repeats_passed", result["status"])
        self.assertEqual(3, len(result["comparisons"]))
        self.assertEqual([2, 2], [x["off_over_on"] for x in result["comparisons"][0]["order_reversed_pairs"]])

    def test_ab_rejects_different_weights_binary_inputs_environment_and_output_bytes(self):
        mutations = [lambda r: r["native_library"].update(sha256="e" * 64),
            lambda r: r["plan"]["tensors"]["a"].update(offset=33),
            lambda r: r["capacity"].update(cpu=32),
            lambda r: r["results"][0]["input"].update(hash="b" * 64),
            lambda r: r["results"][0]["measured"][0].update(output_f32_sha256="f" * 64)]
        for mutate in mutations:
            reports = [self.report(flag) for flag in (0, 1, 1, 0)]; mutate(reports[2])
            with self.subTest(mutate=mutate), self.assertRaises(ValueError): GUARD.compare_ab(self.paths(reports))

    def test_ab_rejects_missing_or_fallback_arena_and_nonquiet_or_unknown_counters(self):
        mutations = [lambda c: c["cpu_scratch_reuse"].update(reused_on_this_call=False, successful_calls_delta=0),
            lambda c: c["cpu_scratch_reuse"].update(after=None),
            lambda c: c.update(read_bytes=4096), lambda c: c.update(major_faults=1),
            lambda c: c.update(minor_faults=None), lambda c: c.update(minor_faults=True),
            lambda c: c["process_counters"]["after"].update(minor_faults=9999),
            lambda c: c.update(wall_ms=float("nan"))]
        for mutate in mutations:
            report = self.report(1); mutate(report["results"][0]["measured"][0])
            with self.subTest(mutate=mutate), self.assertRaises(ValueError): GUARD.validate_ab_report(report, 1)

    def test_ab_requires_all_rows_repeats_exact_tolerances_and_order(self):
        for mutate in (lambda r: r["results"].pop(), lambda r: r["results"][0]["measured"].pop(),
                       lambda r: r["configuration"].update(atol=1e-4), lambda r: r.update(status="failed")):
            report = self.report(0); mutate(report)
            with self.assertRaises(ValueError): GUARD.validate_ab_report(report, 0)
        with self.assertRaises(ValueError): GUARD.compare_ab(self.paths([self.report(flag) for flag in (1, 0, 0, 1)]))

    def test_receipt_refuses_changed_native_copy_and_preserves_prior_receipt(self):
        proof = self.root / "proof"; (proof / "native-build").mkdir(parents=True); (proof / "managed").mkdir()
        (proof / "component-validation.json").write_text(json.dumps({"native": {"passed": len(GUARD.NATIVE_TESTS)}}))
        ab_sources = self.paths([self.report(flag) for flag in (0, 1, 1, 0)])
        (proof / "cpu-arena-ab-comparison.json").write_text(json.dumps(GUARD.compare_ab(ab_sources)))
        (proof / "native-build/libGgmlOps.so").write_bytes(b"new")
        (proof / "managed/libGgmlOps.so").write_bytes(b"old")
        with patch.object(GUARD, "verify_sources"), self.assertRaisesRegex(ValueError, "library differs"):
            GUARD.receipt(self.root, proof)
        path = proof / "existing.json"; GUARD.write_new(path, {"old": True})
        with self.assertRaises(FileExistsError): GUARD.write_new(path, {"new": True})
        self.assertEqual({"old": True}, json.loads(path.read_text()))

    def test_binary_pin_covers_transitive_managed_files_and_native_test_executables(self):
        proof = self.root / "proof"; native = proof / "native-build"; managed = proof / "managed"
        native.mkdir(parents=True); (managed / "runtimes/linux/native").mkdir(parents=True)
        for path in (native / "libGgmlOps.so", managed / "libGgmlOps.so", managed / "InferenceWeb.Tests.dll",
                     managed / "TensorSharp.Server.Host.dll", managed / "runtimes/linux/native/libdep.so", native / "GgmlOpsMuseSparkQKPrepTest"):
            path.write_bytes(b"fixture")
        GUARD.pin_binaries(proof)
        manifest = json.loads((proof / "candidate-binary-manifest.json").read_text())
        self.assertEqual(6, len(manifest))
        self.assertIn(str(managed / "runtimes/linux/native/libdep.so"), manifest)
        parent = self.source_manifest()
        with patch.object(GUARD, "verify_upstream"):
            GUARD.freeze(self.root, proof, parent)
            GUARD.verify_sources(self.root, proof)
            (managed / "runtimes/linux/native/libdep.so").write_bytes(b"different")
            with self.assertRaisesRegex(ValueError, "binary inventory changed"): GUARD.verify_sources(self.root, proof)

    def test_accepted_receipt_identifies_exact_candidate_launch_paths_and_recursive_inventory(self):
        proof = self.root / "proof"; native = proof / "native-build"; managed = proof / "managed"
        native.mkdir(parents=True); (managed / "runtimes/linux/native").mkdir(parents=True)
        for path in (native / "libGgmlOps.so", managed / "libGgmlOps.so", managed / "TensorSharp.Server.Host.dll",
                     managed / "runtimes/linux/native/libdep.so"):
            path.write_bytes(b"candidate")
        (proof / "component-validation.json").write_text(json.dumps({"native": {"passed": len(GUARD.NATIVE_TESTS)}}))
        reports = [self.report(flag) for flag in (0, 1, 1, 0)]
        for report in reports:
            report["native_library"] = {"path": str(native / "libGgmlOps.so"), "sha256": GUARD.digest(native / "libGgmlOps.so")}
        (proof / "cpu-arena-ab-comparison.json").write_text(json.dumps(GUARD.compare_ab(self.paths(reports))))
        coverage = {"checked_files": 3723, "original_clean": False, "source_clean_claim": False,
                    "inventory_basis": "root_supplied_applied_archive_manifest"}
        (proof / "source-manifest.json").write_text(json.dumps({"coverage": coverage}))
        (proof / "parent-manifest.json").write_text(json.dumps({"path": "root-manifest.json", "sha256": "a" * 64}))
        (proof / "native-build.log").write_text("completed")
        (proof / "native-regressions.xml").write_text("completed")
        with patch.object(GUARD, "verify_sources"):
            GUARD.receipt(self.root, proof)
        result = json.loads((proof / "build-record.json").read_text())
        self.assertEqual(str(managed / "TensorSharp.Server.Host.dll"), result["server_assembly"])
        self.assertEqual(str(native), result["native_library_directory"])
        self.assertEqual("a" * 64, result["source_manifest_sha256"])
        self.assertEqual(coverage, result["source_coverage"])
        self.assertIn(str(managed / "runtimes/linux/native/libdep.so"), result["binary_hashes"])


if __name__ == "__main__": unittest.main()
