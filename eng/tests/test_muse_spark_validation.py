"""Guard against false passes in the reusable Muse Spark comparison harness."""
import copy
import importlib.util
import json
import io
from pathlib import Path
import struct
import tempfile
import types
import unittest
from unittest.mock import patch
import zlib

SCRIPT = Path(__file__).resolve().parents[1] / "muse-spark-validate.py"
SPEC = importlib.util.spec_from_file_location("muse_spark_validate", SCRIPT)
harness = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(harness)
LOGITS_SPEC = importlib.util.spec_from_file_location("compare_muse_spark_logits",
    SCRIPT.parent / "validation/compare-muse-spark-logits.py")
logits_comparison = importlib.util.module_from_spec(LOGITS_SPEC)
LOGITS_SPEC.loader.exec_module(logits_comparison)


def turn(answer, cached=None, prompt_tokens=30):
    usage = {"prompt_tokens": prompt_tokens, "completion_tokens": 5}
    if cached is not None:
        usage["prompt_tokens_details"] = {"cached_tokens": cached}
    return {"request": {}, "metrics": {"assistant_message": {"role": "assistant", "content": json.dumps(answer)},
        "finish_reason": "stop", "raw_usage": usage, "prompt_tokens": prompt_tokens,
        "completion_tokens": 5, "total_wall_ms": 5, "decode_timing_source": "server",
        "t_start_abs": 1, "t_first_abs": 2, "t_end_abs": 3}}


class MuseSparkValidationTests(unittest.TestCase):
    TOOL_ARGUMENTS = {"null_text": "null", "bool_text": "true", "number_text": "123",
                      "quoted_text": '"Paris"', "json_text": '{"n":1}',
                      "spaced_text": "  keep whitespace  ",
                      "null_value": None, "number_value": 123, "bool_value": True}

    def setUp(self):
        self.args = types.SimpleNamespace(engine="tensorsharp", model="spark", url="http://unused",
            thinking=False, timeout=1, concurrency=2)

    def test_json_types_and_reasoning_only_do_not_pass(self):
        with self.assertRaises(ValueError):
            harness.validate_answer(turn({"ready": 1}), {"ready": True})
        sample = turn(None)
        sample["metrics"]["assistant_message"] = {"content": None, "reasoning_content": '{"ready":true}'}
        with self.assertRaises(ValueError):
            harness.validate_answer(sample, {"ready": True})

    def test_truncated_correct_json_is_failure(self):
        sample = turn({"answer": 42})
        sample["metrics"]["finish_reason"] = "length"
        with self.assertRaises(ValueError):
            harness.validate_answer(sample, {"answer": 42})

    def run_cache(self, cached):
        tag = "cache-r0"
        code = harness.hashlib.sha256(tag.encode()).hexdigest()[:16]
        with patch.object(harness, "request", side_effect=[turn({"ready": True}), turn({"code": code}, cached, 50)]):
            return harness.extra_case(self.args, "cache_reuse", tag)

    def test_cache_no_telemetry_is_unavailable_even_with_correct_answers(self):
        result = self.run_cache(None)
        self.assertEqual("unavailable", result["status"])
        self.assertTrue(result["answer_correct"])

    def test_zero_partial_and_invalid_cache_reuse_fail(self):
        for cached in (0, 29, True, 51):
            with self.subTest(cached=cached):
                self.assertEqual("fail", self.run_cache(cached)["status"])

    def test_positive_full_prefix_reuse_passes(self):
        self.assertEqual("ok", self.run_cache(30)["status"])

    def test_vision_requires_both_opposite_layouts(self):
        with patch.object(harness, "request", side_effect=[turn({"left": "red", "right": "blue"})] * 2):
            result = harness.extra_case(self.args, "image", "image-r0")
        self.assertEqual("fail", result["status"])
        self.assertEqual(2, len(result["turns"]))
        with patch.object(harness, "request", side_effect=[turn({"left": "red", "right": "blue"}), turn({"left": "blue", "right": "red"})]):
            self.assertEqual("ok", harness.extra_case(self.args, "image", "image-r0")["status"])

    def test_png_layout_pixels_are_reversed(self):
        def scanlines(data):
            self.assertEqual(b"\x89PNG\r\n\x1a\n", data[:8])
            offset, compressed = 8, b""
            while offset < len(data):
                size = struct.unpack(">I", data[offset:offset + 4])[0]
                kind, payload = data[offset + 4:offset + 8], data[offset + 8:offset + 8 + size]
                crc = struct.unpack(">I", data[offset + 8 + size:offset + 12 + size])[0]
                self.assertEqual(zlib.crc32(kind + payload) & 0xffffffff, crc)
                if kind == b"IDAT":
                    compressed += payload
                offset += size + 12
            return zlib.decompress(compressed)
        normal, reversed_ = scanlines(harness.image_png()), scanlines(harness.image_png(True))
        self.assertEqual(128 * (1 + 256 * 3), len(normal))
        self.assertEqual(normal[1:385], reversed_[385:769])
        self.assertNotEqual(normal, reversed_)
        self.assertEqual(harness.image_png(), (harness.VISION_FIXTURES / "vision-left-red.png").read_bytes())
        self.assertEqual(harness.image_png(True), (harness.VISION_FIXTURES / "vision-left-blue.png").read_bytes())

    def test_chart_requires_correct_ranking_and_ocr_on_both_images(self):
        first = {'code': '7319', 'tallest': 'C', 'shortest': 'B', 'bars': 3}
        second = {'code': '4826', 'tallest': 'B', 'shortest': 'A', 'bars': 3}
        for answers, expected in (([first, second], 'ok'), ([first, first], 'fail'),
                                  ([dict(first, code='7318'), second], 'fail')):
            with patch.object(harness, 'request', side_effect=[turn(a) for a in answers]):
                self.assertEqual(expected, harness.extra_case(self.args, 'image_chart', 'chart-r0')['status'])
        self.assertNotIn('image_chart', harness.DEFAULT_SCENARIOS)
        self.assertEqual(harness.chart_png(), (harness.VISION_FIXTURES / 'vision-chart-a.png').read_bytes())
        self.assertEqual(harness.chart_png(True), (harness.VISION_FIXTURES / 'vision-chart-b.png').read_bytes())
        self.assertNotEqual(harness.chart_png(), harness.chart_png(True))

    def tool_turn(self, arguments=None):
        sample = turn(None)
        sample['metrics'].update(finish_reason='tool_calls', usage_present=True,
            tool_call_details=[{'id': 'fixture-call', 'type': 'function', 'function': {
                'name': 'record_values', 'arguments': json.dumps(
                    self.TOOL_ARGUMENTS if arguments is None else arguments)}}])
        return sample

    def test_tool_string_arguments_rejects_value_and_type_coercions(self):
        with patch.object(harness, 'request', return_value=self.tool_turn()):
            self.assertEqual('ok', harness.extra_case(self.args, 'tool_string_arguments', 'typed-r0')['status'])
        for key, wrong in (('null_text', None), ('bool_text', True), ('number_text', 123),
                           ('quoted_text', 'Paris'), ('json_text', {'n': 1}),
                           ('spaced_text', 'keep whitespace'), ('null_value', 'null'),
                           ('number_value', '123'), ('bool_value', 1)):
            with self.subTest(key=key):
                arguments = dict(self.TOOL_ARGUMENTS, **{key: wrong})
                with patch.object(harness, 'request', return_value=self.tool_turn(arguments)):
                    result = harness.extra_case(self.args, 'tool_string_arguments', 'typed-r0')
                    self.assertEqual('fail', result['status'])
        self.assertIn('tool_string_arguments', harness.EXTRA_SCENARIOS)
        self.assertNotIn('tool_string_arguments', harness.DEFAULT_SCENARIOS)
        self.assertEqual(14, len(harness.DEFAULT_SCENARIOS))

    def test_tool_argument_probe_rejects_missing_malformed_or_extra_calls(self):
        for mutate in (
                lambda m: m.update(finish_reason='length'),
                lambda m: m.update(tool_call_details=[]),
                lambda m: m['tool_call_details'].append(copy.deepcopy(m['tool_call_details'][0])),
                lambda m: m['tool_call_details'][0].update(id=''),
                lambda m: m['tool_call_details'][0].update(type='text'),
                lambda m: m['tool_call_details'][0]['function'].update(name='other'),
                lambda m: m['tool_call_details'][0]['function'].update(arguments=self.TOOL_ARGUMENTS),
                lambda m: m['tool_call_details'][0]['function'].update(arguments='not json')):
            sample = self.tool_turn()
            mutate(sample['metrics'])
            with patch.object(harness, 'request', return_value=sample):
                self.assertEqual('fail', harness.extra_case(self.args, 'tool_string_arguments', 'typed-r0')['status'])
        with patch.object(harness, 'request', return_value=turn(self.TOOL_ARGUMENTS)):
            self.assertEqual('fail', harness.extra_case(self.args, 'tool_string_arguments', 'typed-r0')['status'])

    def test_tool_argument_requests_match_and_reference_keeps_routed_parser(self):
        sent = []
        def stream(url, body, timeout, wire):
            sent.append(copy.deepcopy(body))
            wire['sse_lines'] = []
            # WireLog wraps assignments so append through its actual list.
            wire['sse_lines'].append('data: [DONE]')
            return self.tool_turn()['metrics']
        for engine in ('tensorsharp', 'llamacpp'):
            self.args.engine = engine
            with patch.object(harness.engines, '_run_streaming', side_effect=stream):
                result = harness.extra_case(self.args, 'tool_string_arguments', 'typed-r0')
                self.assertEqual('ok', result['status'], result.get('detail'))
            self.assertEqual('required', sent[-1]['tool_choice'])
            self.assertIs(False, sent[-1]['parallel_tool_calls'])
            self.assertNotIn('reasoning_format', sent[-1])
            self.assertEqual(1, len(sent[-1]['tools']))
            self.assertEqual('string', sent[-1]['tools'][0]['function']['parameters']['properties']['null_text']['type'])
            if engine == 'tensorsharp':
                self.assertEqual([], sent[-1]['skills'])
                self.assertIs(False, sent[-1]['skills_discovery'])
                self.assertIs(False, sent[-1]['multi_agent'])
            else:
                self.assertNotIn('skills', sent[-1])
                self.assertNotIn('multi_agent', sent[-1])
        for field in ('messages', 'tools', 'tool_choice', 'parallel_tool_calls', 'max_tokens'):
            self.assertEqual(sent[0][field], sent[1][field])

    @staticmethod
    def report():
        return {"engine": "tensorsharp", "profile": "same", "weights_id": "revision/quant",
            "thinking": False, "sampling": {}, "scenarios": ["short"], "repeats": 1,
            "concurrency": 2, "harness_sha256": {"script": "a"}, "run_complete": True,
            "provenance": {"hardware_id": "machine", "weights_sha256": {"part": "a" * 64},
                "settings": {"context": 4096}, "upstream_ggml_revision": "a" * 40, "upstream_ggml_clean": True,
                "launch_observed": True, "server_revision": "b" * 40,
                "server_command": "observed-server --context 4096", "server_log": "/evidence/server.log"},
            "cases": [{"scenario": "short", "tag": "short-r0", "status": "ok", "input_sha256": "same",
                       "turns": [turn({"answer": 42}, cached=0)], "total_wall_ms": 100}]}

    def test_comparison_needs_provenance_and_complete_execution_plan(self):
        for mutate in (lambda r: r["provenance"].pop("weights_sha256"),
                       lambda r: r["provenance"].update(upstream_ggml_clean=False),
                       lambda r: r.update(run_complete=False),
                       lambda r: r.update(repeats=2),
                       lambda r: r.update(cases=[]),
                       lambda r: r["harness_sha256"].update(script="different")):
            a, b = self.report(), self.report()
            mutate(a)
            self.assertEqual("inconclusive", harness.comparison(a, b, 0.05)["status"])

    def test_failed_unavailable_mismatched_tokens_cannot_establish_parity(self):
        for mutate in (lambda c: c.update(status="fail"), lambda c: c.update(status="unavailable"),
                       lambda c: c["turns"][0]["metrics"].update(prompt_tokens=31),
                       lambda c: c["turns"][0]["metrics"].update(completion_tokens=7)):
            a, b = self.report(), self.report()
            mutate(a["cases"][0])
            self.assertEqual("inconclusive", harness.comparison(a, b, 0.05)["status"])

    def test_planned_launch_cannot_establish_parity(self):
        for side in (0, 1):
            for field, invalid in (("launch_observed", False), ("launch_observed", "true"),
                                   ("server_revision", None), ("server_command", ""),
                                   ("server_log", "  ")):
                with self.subTest(side=side, field=field, invalid=invalid):
                    reports = [self.report(), self.report()]
                    reports[side]["provenance"][field] = invalid
                    result = harness.comparison(*reports, 0.05)
                    self.assertEqual("inconclusive", result["status"])
                    self.assertEqual({}, result["ratios"])

    def test_cache_work_must_be_measured_and_equal_for_wall_time_comparison(self):
        for left, right in ((None, None), (None, 0), (0, 1), (3, 4),
                            (-1, -1), (31, 31), (True, True), ("0", "0")):
            with self.subTest(left=left, right=right):
                a, b = self.report(), self.report()
                a["cases"][0]["turns"] = [turn({"answer": 42}, cached=left)]
                b["cases"][0]["turns"] = [turn({"answer": 42}, cached=right)]
                result = harness.comparison(a, b, 0.05)
                self.assertEqual("inconclusive", result["status"])
                self.assertEqual({}, result["ratios"])
        for count in (0, 12, 30):
            a, b = self.report(), self.report()
            a["cases"][0]["turns"] = [turn({"answer": 42}, cached=count)]
            b["cases"][0]["turns"] = [turn({"answer": 42}, cached=count)]
            self.assertEqual("pass", harness.comparison(a, b, 0.05)["status"])

    def test_regression_and_matching_fixture_verdicts(self):
        a, b = self.report(), self.report()
        a["cases"][0]["total_wall_ms"] = 110
        self.assertEqual("fail", harness.comparison(a, b, 0.05)["status"])
        a["cases"][0]["total_wall_ms"] = 99
        self.assertEqual("pass", harness.comparison(a, b, 0.05)["status"])

    def test_wall_comparison_uses_client_clock_not_engine_decode_timers(self):
        a, b = self.report(), self.report()
        a["cases"][0]["turns"][0]["metrics"].update(
            decode_timing_source="stream_window", decode_tps=1000)
        b["cases"][0]["turns"][0]["metrics"].update(
            decode_timing_source="server", decode_tps=1)
        result = harness.comparison(a, b, 0.05)
        self.assertEqual("pass", result["status"])
        self.assertEqual({"conversation_wall_speedup": 1}, result["ratios"]["short"])
        self.assertIn("decode timers are not compared", result["timing_scope"])
        a["cases"][0]["total_wall_ms"] = 200
        self.assertEqual("fail", harness.comparison(a, b, 0.05)["status"])

    def test_duplicate_case_keys_rejected(self):
        a = self.report()
        a["repeats"] = 2
        a["cases"].append(copy.deepcopy(a["cases"][0]))
        self.assertEqual("inconclusive", harness.comparison(a, a, 0.05)["status"])

    @classmethod
    def no_cache_report(cls, engine="tensorsharp"):
        report = cls.report()
        report.update(engine=engine, cache_policy="disabled")
        report["provenance"]["settings"]["prefix_cache"] = False
        flag = "--no-prefix-cache" if engine == "tensorsharp" else "--no-cache-prompt"
        report["provenance"]["server_command"] += " " + flag
        report["cases"][0]["turns"][0]["request"]["extra_body"] = {"cache_prompt": False}
        report["warmup"] = copy.deepcopy(report["cases"][0])
        return report

    def test_no_cache_launch_requires_observed_exact_flag_and_false_setting(self):
        for engine in ("tensorsharp", "llamacpp"):
            report = self.no_cache_report(engine)
            self.assertEqual([], harness.no_cache_profile_errors(engine, report["provenance"], ["short"]))
            for mutate in (lambda p: p.update(launch_observed=False),
                           lambda p: p.update(launch_observed="true"),
                           lambda p: p["settings"].pop("prefix_cache"),
                           lambda p: p["settings"].update(prefix_cache=0),
                           lambda p: p["settings"].update(prefix_cache=True),
                           lambda p: p.update(server_command="server --no-prefix-cache-wrong --no-cache-prompt-wrong"),
                           lambda p: p.update(server_command="server 'unterminated")):
                provenance = copy.deepcopy(report["provenance"])
                mutate(provenance)
                self.assertTrue(harness.no_cache_profile_errors(engine, provenance, ["short"]))
            self.assertTrue(harness.no_cache_profile_errors(engine, report["provenance"], ["cache_reuse"]))

    def test_no_cache_sends_opt_out_in_both_client_paths_and_engines(self):
        self.args.cache_policy = "disabled"
        for engine in ("tensorsharp", "llamacpp"):
            self.args.engine = engine
            self.assertNotIn("cache_prompt", harness.request_adapter_body(engine, False))
            self.assertIs(False, harness.request_adapter_body(engine, False, True, "disabled")["cache_prompt"])
            def standard(*args, **kwargs):
                sample = turn({"answer": 42}, cached=0)
                sample["request"]["extra_body"] = kwargs["extra_body"]
                return {"status": "ok", "turns": [sample]}
            with patch.object(harness.validation, "run_case", side_effect=standard) as mock:
                self.assertEqual("ok", harness.run_standard_case(self.args, "short", "short-r0")["status"])
                self.assertIs(False, mock.call_args.kwargs["extra_body"]["cache_prompt"])
            def stream(url, body, timeout, wire):
                self.assertIs(False, body["cache_prompt"])
                wire["sse_lines"] = []
                wire["sse_lines"].append("data: [DONE]")
                metrics = turn({"answer": 42}, cached=0)["metrics"]
                metrics["usage_present"] = True
                return metrics
            with patch.object(harness.engines, "_run_streaming", side_effect=stream):
                sent = harness.request(self.args, [{"role": "user", "content": "test"}])
                self.assertIs(False, sent["request"]["cache_prompt"])
                self.assertEqual([], harness.no_cache_case_errors({"turns": [sent]}))

    def test_no_cache_audits_every_turn_without_losing_quality_failure(self):
        self.args.cache_policy = "disabled"
        base = self.no_cache_report()["cases"][0]
        base["turns"].append(copy.deepcopy(base["turns"][0]))
        for invalid in (None, 1, True, "0", -1):
            case = copy.deepcopy(base)
            case["turns"][1]["metrics"]["raw_usage"]["prompt_tokens_details"]["cached_tokens"] = invalid
            result = harness.enforce_cache_policy(self.args, case)
            self.assertEqual("fail", result["status"])
            self.assertIn("Turn 1", result["detail"])
        case = copy.deepcopy(base)
        case["turns"][1]["request"] = {}
        case.update(status="fail", detail="wrong answer")
        self.assertEqual("fail", harness.enforce_cache_policy(self.args, case)["status"])
        self.assertTrue(case["detail"].startswith("wrong answer;"))
        self.assertTrue(harness.no_cache_case_errors({"turns": []}))

    def test_no_cache_extra_fixture_enforces_opt_out_and_zero_cache(self):
        self.args.cache_policy = "disabled"
        sample = self.tool_turn()
        sample["request"]["cache_prompt"] = False
        for cached, expected in ((0, "ok"), (1, "fail"), (None, "fail")):
            sample["metrics"]["raw_usage"]["prompt_tokens_details"] = {"cached_tokens": cached}
            with patch.object(harness, "request", return_value=copy.deepcopy(sample)):
                result = harness.extra_case(self.args, "tool_string_arguments", "typed-r0")
                self.assertEqual(expected, result["status"], result.get("detail"))

    def test_no_cache_comparison_requires_valid_warmup_and_zero_reuse(self):
        left, right = self.no_cache_report(), self.no_cache_report("llamacpp")
        # Engine-specific implementations are allowed; common settings must match.
        left["provenance"]["engine_options"] = {"fuse_residuals": True, "stream_experts": True}
        right["provenance"]["engine_options"] = {"different_internal_kernels": True}
        self.assertEqual("pass", harness.comparison(left, right, .05)["status"])
        mutations = (
            lambda r: r.update(cache_policy="profile"),
            lambda r: r.pop("warmup"),
            lambda r: r["warmup"].update(status="fail"),
            lambda r: r["warmup"]["turns"][0]["request"].clear(),
            lambda r: r["cases"][0]["turns"][0]["request"].clear(),
            lambda r: r["warmup"]["turns"][0]["metrics"]["raw_usage"]["prompt_tokens_details"].update(cached_tokens=1),
            lambda r: r["provenance"]["settings"].update(context=8192))
        for mutate in mutations:
            changed = copy.deepcopy(left)
            mutate(changed)
            result = harness.comparison(changed, right, .05)
            self.assertEqual("inconclusive", result["status"])
            self.assertEqual({}, result["ratios"])
        # Equal nonzero reuse still fails this profile, though ordinary strict
        # comparison permits equal observed reuse under its launch policy.
        for report in (left, right):
            report["cases"][0]["turns"][0]["metrics"]["raw_usage"]["prompt_tokens_details"]["cached_tokens"] = 1
        self.assertEqual("inconclusive", harness.comparison(left, right, .05)["status"])

    def test_no_cache_preflight_and_warmup_abort_before_measured_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            provenance = Path(directory) / "provenance.json"
            argv = [str(SCRIPT), "--engine", "tensorsharp", "--url", "http://unused", "--model", "spark",
                    "--weights-id", "same", "--profile", "same", "--output", str(output),
                    "--scenarios", "short", "--repeats", "1", "--cache-policy", "disabled"]
            with patch.object(harness.sys, "argv", argv), patch.object(harness, "run_standard_case") as runner, \
                    patch("sys.stderr", new_callable=io.StringIO):
                with self.assertRaises(SystemExit) as error:
                    harness.main()
                self.assertEqual(2, error.exception.code)
                runner.assert_not_called()
            provenance.write_text(json.dumps(self.no_cache_report()["provenance"]), encoding="utf-8")
            failed = copy.deepcopy(self.no_cache_report()["warmup"])
            failed.update(status="fail", detail="nonzero cached tokens")
            with patch.object(harness.sys, "argv", argv + ["--provenance", str(provenance)]), \
                    patch.object(harness, "run_standard_case", return_value=failed) as runner:
                self.assertEqual(1, harness.main())
                self.assertEqual(1, runner.call_count)
                self.assertEqual("warmup", runner.call_args.args[2])
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertFalse(report["run_complete"])
            self.assertEqual([], report["cases"])
            self.assertIn("aborted", report)

    def test_only_generated_content_wakes_staggered_admission(self):
        event = harness.threading.Event()
        wire = harness.WireLog(event)
        wire["sse_lines"] = []
        wire["sse_lines"].append('data: {"choices":[{"delta":{"role":"assistant"}}]}')
        self.assertFalse(event.is_set())
        wire["sse_lines"].append('data: {"choices":[{"delta":{"content":"hello"}}]}')
        self.assertTrue(event.is_set())

    def test_logits_all_elements_and_argmax_are_checked(self):
        self.assertEqual("pass", logits_comparison.compare([[0.1, 0.2]], [[0.100001, 0.200001]], 1e-5, 1e-5)["status"])
        self.assertEqual("fail", logits_comparison.compare([[0.1, 0.2]], [[0.1, 0.21]], 1e-5, 1e-5)["status"])
        # Both rows are within tolerance but the greedy decision changed.
        self.assertEqual("fail", logits_comparison.compare([[0.1, 0.100001]], [[0.100001, 0.1]], 1e-5, 1e-5)["status"])

    def test_nonfinite_empty_and_misaligned_logits_are_rejected(self):
        for a, b in (([], []), ([[]], [[]]), ([[1, 2]], [[1]]), ([[1]], [[float("nan")]])):
            with self.assertRaises(ValueError):
                logits_comparison.compare(a, b, 1e-5, 1e-5)


if __name__ == "__main__":
    unittest.main()
