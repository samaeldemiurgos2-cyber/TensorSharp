"""A failed, mismatched or incomplete numerical control cannot pass distribution."""
from array import array
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("dsv41_distributed_compare",
    Path(__file__).parents[1] / "compare-deepseek41-distributed.py")
comparator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparator)


class DistributedComparatorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.report = dict(schema_version=1, status="completed", disposed=True, weights_id="fixture-sha",
            tokens=[1, 2, 3], tokens_sha256="input-sha", vocab_size=3,
            steps=[dict(operation="reset", tokens=[], logits=None),
                   dict(operation="prefill", tokens=[1, 2], logits="step.f32", count=3)])
        self.reference = self.write("reference", self.report)
        self.candidate = self.write("candidate", self.report)

    def write(self, name, report, vector=(1., -2., 3.)):
        directory = self.root / name
        directory.mkdir(exist_ok=True)
        data = array("f", vector)
        if sys.byteorder != "little": data.byteswap()
        (directory / "step.f32").write_bytes(data.tobytes())
        path = directory / "probe.json"
        path.write_text(json.dumps(report))
        return path

    def compare(self, **overrides):
        gates = dict(max_absolute_error=1e-5, max_rmse=1e-5, min_cosine=.999999,
                     max_relative_l2=1e-5, require_same_argmax=True)
        gates.update(overrides)
        return comparator.compare(self.reference, self.candidate, **gates)

    def test_every_matching_vector_is_gated(self):
        report = self.compare()
        self.assertEqual("passed", report["status"])
        self.assertEqual(1, len(report["vectors"]))

    def test_failed_worker_or_undisposed_run_does_not_establish_a_control(self):
        for field, value in (("status", "failed"), ("status", "worker_loop_returned"), ("disposed", False)):
            with self.subTest(field=field, value=value):
                self.write("candidate", {**self.report, field: value})
                self.assertEqual("failed", self.compare()["status"])

    def test_changed_weights_tokens_or_operation_cannot_pass_identical_vectors(self):
        for mutation in ("weights_id", "tokens", "tokens_sha256", "vocab_size", "steps"):
            report = copy.deepcopy(self.report)
            if mutation == "steps": report[mutation][1]["tokens"] = [2, 1]
            elif mutation == "tokens": report[mutation] = [1, 2, 4]
            else: report[mutation] = "different"
            with self.subTest(mutation=mutation):
                self.write("candidate", report)
                self.assertEqual("failed", self.compare()["status"])

    def test_numerical_difference_and_nonfinite_vectors_fail(self):
        for vector in ((1., -2., 30.), (1., float("nan"), 3.), (1., 2.)):
            with self.subTest(vector=vector):
                self.write("candidate", self.report, vector)
                self.assertEqual("failed", self.compare()["status"])

    def test_no_vectors_and_missing_operation_fail(self):
        for steps in ([], self.report["steps"][:1]):
            with self.subTest(steps=steps):
                self.write("candidate", {**self.report, "steps": steps})
                self.assertEqual("failed", self.compare()["status"])

    def test_vector_outside_evidence_directory_fails(self):
        report = copy.deepcopy(self.report)
        report["steps"][1]["logits"] = "../reference/step.f32"
        self.write("candidate", report)
        self.assertEqual("failed", self.compare()["status"])

    def test_normalized_gate_uses_reference_magnitude_and_inclusive_boundary(self):
        self.write("reference", self.report, (-100., 0., 1.))
        self.write("candidate", self.report, (-98., 0., 1.))
        loose = dict(max_absolute_error=3., max_rmse=3., min_cosine=.99, max_relative_l2=.1)
        report = self.compare(**loose, max_normalized_error=.02)
        self.assertEqual("passed", report["status"])
        row = report["vectors"][0]
        self.assertEqual(100., row["normalization_scale"])
        self.assertEqual(.02, row["normalized_max_error"])
        self.assertEqual(.02, row["thresholds"]["max_normalized_error"])
        failed = self.compare(**loose, max_normalized_error=.019)
        self.assertEqual("failed", failed["status"])
        self.assertIn("Normalized maximum error", failed["vectors"][0]["failures"][0])
        self.assertEqual("passed", self.compare(**loose)["status"])  # opt-in only

    def test_normalization_has_unit_floor_including_zero_reference(self):
        for reference in ((.5, -.5, 0.), (0., 0., 0.)):
            with self.subTest(reference=reference):
                self.write("reference", self.report, reference)
                self.write("candidate", self.report, (reference[0] + .125, *reference[1:]))
                report = self.compare(max_absolute_error=None, max_rmse=None, min_cosine=None,
                                      max_relative_l2=None, require_same_argmax=False, max_normalized_error=.125)
                self.assertEqual("passed", report["status"])
                self.assertEqual(1., report["vectors"][0]["normalization_scale"])
                self.assertEqual(.125, report["vectors"][0]["normalized_max_error"])

    def test_normalized_gate_never_overrides_existing_absolute_or_argmax_failure(self):
        self.write("candidate", self.report, (1., -2., 3.125))
        report = self.compare(max_normalized_error=.1)
        self.assertEqual("failed", report["status"])
        self.assertLess(report["vectors"][0]["normalized_max_error"], .1)
        self.write("reference", self.report, (1., 2., 2.125))
        self.write("candidate", self.report, (1., 2.25, 2.125))
        report = self.compare(max_absolute_error=1., max_rmse=1., min_cosine=.9,
                              max_relative_l2=1., max_normalized_error=.2)
        self.assertEqual("failed", report["status"])
        self.assertIn("Argmax token differs", report["vectors"][0]["failures"])

    def test_each_vector_uses_its_own_scale_and_one_failure_fails_the_run(self):
        report = copy.deepcopy(self.report)
        report["steps"].append(dict(operation="decode", tokens=[3], logits="second.f32", count=3))
        for name, second in (("reference", (.5, -.5, 0.)), ("candidate", (.625, -.5, 0.))):
            self.write(name, report, (-100., 0., 1.))
            data = array("f", second)
            if sys.byteorder != "little": data.byteswap()
            (self.root / name / "second.f32").write_bytes(data.tobytes())
        result = self.compare(max_absolute_error=1., max_rmse=1., min_cosine=.9,
                              max_relative_l2=1., max_normalized_error=.02)
        self.assertEqual("failed", result["status"])
        self.assertEqual([100., 1.], [row["normalization_scale"] for row in result["vectors"]])
        self.assertEqual(["passed", "failed"], [row["status"] for row in result["vectors"]])

    def test_cli_records_normalized_gate_and_rejects_invalid_thresholds(self):
        output = self.root / "comparison.json"
        args = ["compare", str(self.reference), str(self.candidate), "--output", str(output),
                "--max-absolute-error", "1", "--max-rmse", "1", "--max-relative-l2", "1",
                "--min-cosine", ".9", "--max-normalized-error"]
        with patch.object(sys, "argv", args + [".02"]), patch("builtins.print"):
            self.assertEqual(0, comparator.main())
        self.assertEqual(.02, json.loads(output.read_text())["vectors"][0]["thresholds"]["max_normalized_error"])
        for value in ("-1", "nan", "inf"):
            with self.subTest(value=value), patch.object(sys, "argv", args + [value]), patch("sys.stderr"):
                with self.assertRaises(SystemExit) as failure:
                    comparator.main()
                self.assertEqual(2, failure.exception.code)


if __name__ == "__main__": unittest.main()
