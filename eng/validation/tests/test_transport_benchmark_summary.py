"""Integrity checks for paired transport evidence, without running a benchmark."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "summarize-transport-benchmark.py"
SPEC = importlib.util.spec_from_file_location("transport_summary", SCRIPT)
SUMMARY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARY)


class TransportSummaryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.manifest = dict(schema_version=1, nodes=[0, 1], iterations=4, warmup=1,
            cases=[dict(operation="all_reduce", bytes=4)], variants=[], context={},
            post_measurement_changes=[], limitations=[])
        for variant, samples in (("baseline", [1., 2., 3., 4.]), ("candidate", [.5, 1., 1.5, 2.])):
            item = dict(label=variant, transport_sha256="a" * 64, probe_sha256="b" * 64, runs=[])
            for repeat in (1, 2):
                run = dict(repeat=repeat, reports={})
                for node in (0, 1):
                    name = f"{variant}-{repeat}-{node}.json"
                    run["reports"][str(node)] = name
                    median = (samples[1] + samples[2]) / 2
                    report = dict(schema_version=1, status="passed", node_id=node, route="test",
                        arguments=["--node-id", str(node), "--iterations", "4", "--warmup", "1", "--route", "test", "--peers", "a:1,b:2"],
                        transport_sha256="a" * 64, probe_sha256="b" * 64,
                        started_utc="2026-01-01T00:00:00Z", finished_utc="2026-01-01T00:00:01Z",
                        cases=[dict(operation="all_reduce", bytes=4, iterations=4, warmup=1,
                            median_ms=median, p95_ms=samples[-1], samples_ms=samples,
                            payload_mib_per_second=4 / 1048576 / (median / 1000))])
                    (self.root / name).write_text(json.dumps(report), encoding="utf-8")
                item["runs"].append(run)
            self.manifest["variants"].append(item)

    def summarize(self):
        path = self.root / "manifest.json"
        path.write_text(json.dumps(self.manifest), encoding="utf-8")
        return SUMMARY.summarize(path)

    def mutate(self, fn):
        path = self.root / "candidate-2-1.json"
        report = json.loads(path.read_text())
        fn(report)
        path.write_text(json.dumps(report), encoding="utf-8")

    def test_valid_pairs_keep_nodes_separate_and_recompute_statistics(self):
        report = self.summarize()
        self.assertEqual(8, report["measured_case_count"])
        self.assertEqual(32, report["retained_timing_count"])
        self.assertEqual([0, 1], [row["node_id"] for row in report["comparisons"]])
        self.assertEqual(-50, report["comparisons"][0]["aggregate"]["median_of_repeat_median_ms"]["change_percent"])
        self.assertIn("lower in every repeat", SUMMARY.markdown(report))

    def test_failed_partner_rejected(self):
        self.mutate(lambda report: report.update(status="failed"))
        with self.assertRaisesRegex(ValueError, "did not pass"):
            self.summarize()

    def test_missing_partner_rejected(self):
        del self.manifest["variants"][1]["runs"][1]["reports"]["1"]
        with self.assertRaisesRegex(ValueError, "Missing or extra node"):
            self.summarize()

    def test_tampered_summary_rejected(self):
        self.mutate(lambda report: report["cases"][0].update(p95_ms=1))
        with self.assertRaisesRegex(ValueError, "disagrees with raw samples"):
            self.summarize()

    def test_missing_samples_rejected(self):
        self.mutate(lambda report: report["cases"][0]["samples_ms"].pop())
        with self.assertRaisesRegex(ValueError, "invalid or missing timing"):
            self.summarize()

    def test_binary_mismatch_rejected(self):
        self.mutate(lambda report: report.update(transport_sha256="c" * 64))
        with self.assertRaisesRegex(ValueError, "binary changed"):
            self.summarize()

    def test_node_mismatch_rejected(self):
        self.mutate(lambda report: report.update(node_id=0))
        with self.assertRaisesRegex(ValueError, "node identity mismatch"):
            self.summarize()

    def test_case_loss_rejected(self):
        self.mutate(lambda report: report.update(cases=[]))
        with self.assertRaisesRegex(ValueError, "missing or unexpected benchmark cases"):
            self.summarize()

    def test_direction_reports_mixed_and_higher_results(self):
        self.assertEqual("mixed_across_repeats", SUMMARY.direction([-1, 100, -1]))
        self.assertEqual("higher_in_every_repeat", SUMMARY.direction([1, 2, 3]))


if __name__ == "__main__":
    unittest.main()
