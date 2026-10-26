import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("cli_benchmark_summary",
    Path(__file__).parents[1] / "summarize-parallel-cli-benchmark.py")
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


class CliBenchmarkSummaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run = dict(status="completed", runtime_file_identity_check={"status": "passed"},
            shutdown=dict(started=True, forced=False, exit_code=0))
        self.lines = ["benchmark starting: prefillTokens=64 decodeTokens=8 runs=4 chunked=False decodeMode=random-inference"]
        self.lines += [f"benchmark run {i}/4: prefillMs={1000 * 2 ** (i-1)} prefillTps=999 decodeMs={100 * 2 ** (i-1)} decodeTps=999 decodeMode=random-inference"
                       for i in range(1, 5)]

    def summarize(self, discard=1):
        (self.root / "run.json").write_text(json.dumps(self.run))
        (self.root / "process.log").write_text("\n".join(self.lines), encoding="utf-8")
        return summary.summarize(self.root, discard)

    def test_uses_actual_counts_and_milliseconds_not_rounded_cli_rate(self):
        result = self.summarize()
        self.assertEqual("measured", result["status"])
        self.assertEqual([2, 3, 4], result["measured_runs"])
        self.assertEqual(16, result["summary"]["median_prefill_tokens_per_second"])
        self.assertEqual(20, result["summary"]["median_decode_tokens_per_second"])

    def test_partial_duplicate_or_wrong_order_sample_does_not_measure(self):
        original = list(self.lines)
        for lines in (original[:-1], original + original[-1:], original[:1] + list(reversed(original[1:]))):
            self.lines = lines
            self.assertEqual("failed", self.summarize()["status"])

    def test_crash_or_runtime_mutation_cannot_be_hidden_by_good_timings(self):
        for field, value in (("status", "failed"), ("runtime_file_identity_check", {"status": "failed"}),
                             ("shutdown", dict(started=True, forced=False, exit_code=134))):
            original = self.run[field]
            self.run[field] = value
            self.assertEqual("failed", self.summarize()["status"])
            self.run[field] = original

    def test_missing_start_or_discarding_every_run_fails(self):
        self.assertEqual("failed", self.summarize(discard=4)["status"])
        self.lines = self.lines[1:]
        self.assertEqual("failed", self.summarize()["status"])

    def test_zero_duration_or_changed_decode_mode_fails(self):
        self.lines[2] = self.lines[2].replace("decodeMs=200", "decodeMs=0")
        self.assertEqual("failed", self.summarize()["status"])
        self.lines[2] = self.lines[2].replace("decodeMs=0", "decodeMs=200").replace("decodeMode=random", "decodeMode=greedy")
        self.assertEqual("failed", self.summarize()["status"])


if __name__ == "__main__": unittest.main()
