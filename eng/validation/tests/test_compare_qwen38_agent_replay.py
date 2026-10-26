import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("compare_agent_replay", Path(__file__).parents[1] / "compare-qwen38-agent-replay.py")
COMPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMPARE)


def row(step=0, e2e=1000, scenario="text", expected=None):
    return {"scenario": scenario, "concurrency": 1, "lane": 0, "step": step,
            "request_sha256": f"{step + 1:064x}", "expected": expected,
            "done": True, "http_status": 200, "finish_reason": "stop", "answer": "A bicycle has wheels.", "reasoning": "",
            "tool_calls": [], "errors": [], "failures": [], "prompt_tokens": 19000, "completion_tokens": 10,
            "ttft_ms": 100, "e2e_ms": e2e}


def report(label, rows):
    return {"label": label, "fixture_sha256": "a" * 64, "model": "test", "complete": True,
            "passed": all(not item["failures"] for item in rows), "results": rows}


class CompareReplayTests(unittest.TestCase):
    def test_same_keys_are_paired_even_if_results_are_reordered(self):
        baseline = report("baseline", [row(0, 1000), row(1, 2000), row(2, 4000)])
        candidate = report("candidate", [row(2, 2000), row(0, 500), row(1, 1000)])
        result = COMPARE.compare(baseline, candidate)
        self.assertTrue(result["all_requested_checks_passed"])
        cases = {case["phase"]: case for case in result["cases"]}
        self.assertEqual(1, cases["step0"]["n_paired_valid"])
        self.assertEqual(2, cases["later_steps"]["n_paired_valid"])
        self.assertEqual(.5, cases["later_steps"]["paired_candidate_over_baseline_e2e"]["p50"])
        self.assertTrue(cases["later_steps"]["all_paired_outputs_identical"])

    def test_fixture_and_request_hash_mismatches_block_ratios(self):
        for field in ("fixture", "request"):
            baseline, candidate = report("base", [row()]), report("candidate", [row()])
            if field == "fixture":
                candidate["fixture_sha256"] = "b" * 64
            else:
                candidate["results"][0]["request_sha256"] = "b" * 64
            result = COMPARE.compare(baseline, candidate)
            self.assertFalse(result["identity_and_completion_valid"])
            self.assertFalse(result["all_requested_checks_passed"])
            self.assertEqual(0, result["cases"][0]["n_paired_valid"])
            self.assertIsNone(result["cases"][0]["paired_candidate_over_baseline_e2e"]["p50"])

    def test_missing_request_does_not_qualify_a_fast_subset(self):
        result = COMPARE.compare(report("base", [row(0), row(1)]), report("candidate", [row(0, 200)]))
        self.assertEqual(1, len(result["missing_candidate_keys"]))
        self.assertFalse(result["all_requested_checks_passed"])
        self.assertTrue(all(case["n_paired_valid"] == 0 for case in result["cases"]))

    def test_incomplete_report_blocks_ratios_despite_passing_rows(self):
        candidate = report("candidate", [row()])
        candidate["complete"] = False
        result = COMPARE.compare(report("base", [row()]), candidate)
        self.assertFalse(result["all_requested_checks_passed"])
        self.assertEqual(0, result["cases"][0]["n_paired_valid"])

    def test_duplicate_request_key_is_not_counted_twice(self):
        result = COMPARE.compare(report("base", [row()]), report("candidate", [row(), row()]))
        self.assertFalse(result["identity_and_completion_valid"])
        self.assertTrue(any("duplicate" in error for error in result["identity_errors"]))

    def test_opposite_failures_cannot_compare_different_survivors(self):
        first, second = row(1, 1000), row(2, 2000)
        first["failures"] = ["baseline-only failure"]
        left = report("base", [first, second])
        first, second = row(1, 100), row(2, 100)
        second["failures"] = ["candidate-only failure"]
        right = report("candidate", [first, second])
        result = COMPARE.compare(left, right)
        case = result["cases"][0]
        self.assertEqual(1, case["baseline"]["observed_checks_passed"])
        self.assertEqual(1, case["candidate"]["observed_checks_passed"])
        self.assertEqual(0, case["n_paired_valid"])
        self.assertEqual(2, case["n_excluded_common"])
        self.assertFalse(result["all_requested_checks_passed"])

    def test_common_valid_subset_is_explicitly_exploratory(self):
        left = report("base", [row(1, 1000), row(2, 2000)])
        right = report("candidate", [row(1, 500), row(2, 10)])
        right["results"][1]["failures"] = ["wrong output"]
        right["passed"] = False
        case = COMPARE.compare(left, right)["cases"][0]
        self.assertEqual(2, case["n_max_paired"])
        self.assertEqual(1, case["n_paired_valid"])
        self.assertEqual(.5, case["paired_candidate_over_baseline_e2e"]["p50"])
        self.assertEqual(1000, case["paired_baseline_e2e_ms"]["p50"])
        self.assertEqual(500, case["paired_candidate_e2e_ms"]["p50"])
        self.assertFalse(case["all_requested_checks_passed"])
        self.assertIn("exploratory", case["ratio_scope"])

    def test_prose_length_passes_transport_only_and_tracks_output_drift(self):
        first, second = row(), row(e2e=500)
        first["finish_reason"] = second["finish_reason"] = "length"
        second["answer"] = "A different capped answer."
        second["completion_tokens"] = 20
        case = COMPARE.compare(report("base", [first]), report("candidate", [second]))["cases"][0]
        self.assertTrue(case["all_requested_checks_passed"])
        self.assertEqual(1, case["candidate"]["truncated_prose"])
        self.assertEqual(1, case["candidate"]["prose_semantics_unscored"])
        self.assertFalse(case["all_paired_outputs_identical"])
        self.assertEqual(20, case["paired_samples"][0]["candidate_completion_tokens"])

    def test_image_error_is_revalidated_even_if_report_says_passed(self):
        first = row(scenario="agent34", expected={"code": "4821", "color": "red"})
        first.update(answer="", finish_reason="tool_calls", tool_calls=[{"id": "a", "function": {
            "name": "submit_observation", "arguments": '{"code":"4821","color":"red"}'}}])
        second = copy.deepcopy(first)
        second["tool_calls"][0]["function"]["arguments"] = '{"code":"0000","color":"red"}'
        result = COMPARE.compare(report("base", [first]), report("candidate", [second]))
        case = result["cases"][0]
        self.assertEqual(1, case["candidate"]["image_or_tool_errors"])
        self.assertEqual(0, case["n_paired_valid"])
        self.assertFalse(result["all_requested_checks_passed"])

    def test_tool_call_ids_are_not_output_drift(self):
        first = row(scenario="agent34", expected={"code": "4821", "color": "red"})
        first.update(answer="", finish_reason="tool_calls", tool_calls=[{"id": "a", "function": {
            "name": "submit_observation", "arguments": '{"code":"4821","color":"red"}'}}])
        second = copy.deepcopy(first)
        second["tool_calls"][0]["id"] = "different"
        second["tool_calls"][0]["function"]["arguments"] = '{"color":"red","code":"4821"}'
        case = COMPARE.compare(report("base", [first]), report("candidate", [second]))["cases"][0]
        self.assertTrue(case["all_paired_outputs_identical"])

    def test_invalid_or_missing_timing_and_validation_are_not_passes(self):
        for field, value in (("e2e_ms", float("nan")), ("ttft_ms", 2000), ("failures", None),
                             ("tool_calls", None), ("completion_tokens", True)):
            second = row()
            second[field] = value
            result = COMPARE.compare(report("base", [row()]), report("candidate", [second]))
            self.assertEqual(0, result["cases"][0]["n_paired_valid"], field)
            self.assertFalse(result["all_requested_checks_passed"], field)
            json.dumps(result, allow_nan=False)

    def test_changed_expected_values_block_comparison(self):
        first, second = row(), row()
        second["expected"] = {"answer": "579"}
        result = COMPARE.compare(report("base", [first]), report("candidate", [second]))
        self.assertFalse(result["identity_and_completion_valid"])

    def test_multiple_labels_and_unique_label_requirement(self):
        result = COMPARE.compare_reports(report("base", [row()]),
                                         [report("fast", [row(e2e=500)]), report("slow", [row(e2e=2000)])])
        self.assertEqual(2, len(result["comparisons"]))
        self.assertTrue(result["all_requested_checks_passed"])
        with self.assertRaises(ValueError):
            COMPARE.compare_reports(report("base", [row()]), [report("base", [row()])])

    def test_empty_reports_never_pass(self):
        result = COMPARE.compare(report("base", []), report("candidate", []))
        self.assertFalse(result["all_requested_checks_passed"])
        self.assertFalse(result["identity_and_completion_valid"])

    def test_cli_writes_failure_evidence_and_nonzero_exit_for_incomplete_or_invalid_input(self):
        artifacts = COMPARE.REPLAY.ROOT / "artifacts"
        artifacts.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=artifacts, prefix="compare-replay-test-") as directory:
            directory = Path(directory)
            baseline = directory / "baseline.json"
            candidate = directory / "candidate.json"
            output = directory / "comparison.json"
            baseline.write_text(json.dumps(report("base", [row()])))
            good = report("candidate", [row(e2e=500)])
            incomplete = copy.deepcopy(good)
            incomplete["complete"] = False
            for contents, expected_exit in ((json.dumps(good), 0), (json.dumps(incomplete), 1), ("not-json", 1)):
                candidate.write_text(contents)
                result = subprocess.run([sys.executable, str(Path(COMPARE.__file__)),
                                         "--baseline", str(baseline), "--candidate", str(candidate),
                                         "--output", str(output)], capture_output=True, text=True)
                self.assertEqual(expected_exit, result.returncode, result.stderr)
                self.assertEqual(expected_exit == 0, json.loads(output.read_text())["all_requested_checks_passed"])


if __name__ == "__main__":
    unittest.main()
