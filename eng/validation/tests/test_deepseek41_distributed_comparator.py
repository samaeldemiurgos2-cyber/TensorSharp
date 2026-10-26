"""A failed, mismatched or incomplete numerical control cannot pass distribution."""
from array import array
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

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

    def compare(self):
        return comparator.compare(self.reference, self.candidate,
            max_absolute_error=1e-5, max_rmse=1e-5, min_cosine=.999999,
            max_relative_l2=1e-5, require_same_argmax=True)

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


if __name__ == "__main__": unittest.main()
