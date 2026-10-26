import importlib.util
import base64
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("agent_replay", Path(__file__).parents[1] / "qwen38-agent-replay.py")
REPLAY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPLAY)


class Response:
    status = 200

    def __init__(self, events):
        self.events = events

    def getheader(self, name, default=None):
        return "text/event-stream"

    def __iter__(self):
        for event in self.events:
            yield b"data: " + (event.encode() if isinstance(event, str) else json.dumps(event).encode()) + b"\n"


class Connection:
    def __init__(self, events):
        self.response = Response(events)

    def request(self, *args):
        pass

    def getresponse(self):
        return self.response

    def close(self):
        pass


def event(delta, finish=None):
    return {"choices": [{"delta": delta, "finish_reason": finish}]}


class AgentReplayTests(unittest.TestCase):
    def measure(self, events, expected=None):
        row = {"scenario": "test", "lane": 0, "step": 0, "request_sha256": "x", "body": {}, "expected": expected}
        with patch.object(REPLAY.http.client, "HTTPConnection", return_value=Connection(events)):
            return REPLAY.request("http://localhost:5088", "model", row, 60)

    def test_tool_fragments_and_usage_are_collected_before_done(self):
        measured = self.measure([
            event({"role": "assistant"}),
            event({"tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "submit_observation", "arguments": '{"code": "48'}}]}),
            event({"tool_calls": [{"index": 0, "function": {"arguments": '21", "color": "red"}'}}]}, "tool_calls"),
            {"usage": {"prompt_tokens": 19000, "completion_tokens": 32, "prompt_tokens_details": {"cached_tokens": 16500}}},
            "[DONE]"], {"code": "4821", "color": "red"})
        self.assertEqual([], measured["failures"])
        self.assertIsNotNone(measured["ttft_ms"])
        self.assertEqual(2500, measured["prefilled_tokens"])
        self.assertIsNone(measured["post_first_delta_tokens_per_second"])
        self.assertEqual(2, measured["nonempty_delta_count"])
        self.assertGreaterEqual(measured["last_model_delta_ms"], measured["ttft_ms"])

    def test_headers_and_role_are_not_first_token(self):
        measured = self.measure([event({"role": "assistant"}), "[DONE]"])
        self.assertIsNone(measured["ttft_ms"])
        self.assertIn("incomplete stream", measured["failures"])

    def test_truncated_transport_and_server_error_fail(self):
        measured = self.measure([event({"content": "579"}, "stop"), {"error": {"message": "lost"}},
                                {"usage": {"completion_tokens": 3}}], {"answer": "579"})
        self.assertIn("incomplete stream", measured["failures"])
        self.assertTrue(measured["errors"])

    def test_wrong_image_answer_and_invalid_json_fail(self):
        for arguments in ('{"code":"9999","color":"red"}', "broken"):
            measured = self.measure([event({"tool_calls": [{"index": 0, "function": {
                "name": "submit_observation", "arguments": arguments}}]}, "tool_calls"),
                {"usage": {"completion_tokens": 32}}, "[DONE]"], {"code": "4821", "color": "red"})
            self.assertTrue(measured["failures"])

    def test_reasoning_is_rejected(self):
        measured = self.measure([event({"reasoning_content": "thinking"}), event({"content": "579"}, "stop"),
                                {"usage": {"completion_tokens": 3}}, "[DONE]"], {"answer": "579"})
        self.assertIn("unexpected reasoning output", measured["failures"])

    def test_failed_requests_do_not_boost_throughput(self):
        summary = REPLAY.summarize([{"failures": [], "completion_tokens": 20, "e2e_ms": 200},
                                    {"failures": ["incorrect"], "completion_tokens": 100, "e2e_ms": 1}], 2)
        self.assertEqual(10, summary["throughput_tokens_per_second"])
        self.assertEqual(200, summary["e2e_ms"]["p50"])
        self.assertEqual(1, summary["failed"])

    def test_fixture_digest_detects_tool_or_image_changes(self):
        body = {"messages": [{"role": "user", "content": "image1"}], "tools": REPLAY.tools34()}
        original = REPLAY.digest(body)
        body["messages"][0]["content"] = "image2"
        self.assertNotEqual(original, REPLAY.digest(body))
        self.assertEqual(34, len(body["tools"]))

    def test_tool_call_metadata_is_not_a_first_token(self):
        measured = self.measure([event({"tool_calls": [{"index": 0, "id": "call_1", "type": "function"}]}), "[DONE]"])
        self.assertIsNone(measured["ttft_ms"])
        self.assertIsNone(measured["last_model_delta_ms"])
        self.assertEqual(0, measured["nonempty_delta_count"])

    def test_buffered_prose_does_not_invent_a_decode_rate(self):
        measured = self.measure([event({"content": "579"}, "stop"),
                                {"usage": {"prompt_tokens": 20, "completion_tokens": 3}}, "[DONE]"], {"answer": "579"})
        self.assertEqual([], measured["failures"])
        self.assertEqual(1, measured["content_delta_count"])
        self.assertIsNone(measured["post_first_delta_tokens_per_second"])

    def test_invalid_usage_and_finish_reason_fail_without_crashing(self):
        for usage in ({"prompt_tokens": 10, "completion_tokens": True},
                      {"completion_tokens": 3},
                      {"prompt_tokens": 10, "completion_tokens": 3, "prompt_tokens_details": "bad"},
                      {"prompt_tokens": 10, "completion_tokens": 3, "prompt_tokens_details": {"cached_tokens": 11}}):
            measured = self.measure([event({"content": "579"}, "cancelled"), {"usage": usage}, "[DONE]"], {"answer": "579"})
            self.assertTrue(measured["failures"])
            self.assertIn("unexpected finish reason", measured["failures"])

    def test_usage_cannot_overwrite_transport_evidence(self):
        measured = self.measure([event({"content": "579"}, "stop"),
                                {"usage": {"prompt_tokens": 20, "completion_tokens": 3, "done": True}}], {"answer": "579"})
        self.assertFalse(measured["done"])
        self.assertIn("incomplete stream", measured["failures"])

    def test_length_capped_prose_is_recorded_without_a_semantic_correctness_claim(self):
        measured = self.measure([event({"content": "A bicycle uses"}, "length"),
                                {"usage": {"prompt_tokens": 20, "completion_tokens": 3}}, "[DONE]"])
        self.assertEqual([], measured["failures"])
        self.assertTrue(measured["output_truncated"])
        self.assertIn("semantic answer quality not scored", measured["validation_scope"])
        summary = REPLAY.summarize([measured], 2)
        self.assertEqual(1, summary["truncated"])
        self.assertEqual(0, summary["exact_output_checks"])

    def test_tool_call_prose_is_a_workload_failure(self):
        measured = self.measure([event({"content": "I will call a tool."}),
                                event({"tool_calls": [{"index": 0, "function": {
                                    "name": "submit_observation", "arguments": '{"code":"4821","color":"red"}'}}]}, "tool_calls"),
                                {"usage": {"prompt_tokens": 20, "completion_tokens": 32}}, "[DONE]"], {"code": "4821", "color": "red"})
        self.assertIn("unexpected prose before tool call", measured["failures"])

    def test_prose_task_cannot_pass_with_an_unrequested_tool_call(self):
        measured = self.measure([event({"content": "A bicycle has wheels.", "tool_calls": [{"index": 0, "function": {
            "name": "unexpected", "arguments": "{}"}}]}, "tool_calls"),
            {"usage": {"prompt_tokens": 20, "completion_tokens": 20}}, "[DONE]"])
        self.assertIn("unexpected tool call in prose task", measured["failures"])

    def test_retained_images_extend_history_and_replacement_changes_only_image(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "fixture.jsonl"
            args = SimpleNamespace(output=output, steps=8, lanes=1, max_tokens=128, image_width=1344, image_height=768)
            def fake_card(code, color, width, height):
                return "data:image/png;base64," + base64.b64encode(f"image:{code}:{color}:{width}:{height}".encode()).decode()
            with patch.object(REPLAY, "card", side_effect=fake_card), contextlib.redirect_stdout(io.StringIO()):
                REPLAY.prepare(args)
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            manifest = json.loads(output.with_suffix(".manifest.json").read_text())
            self.assertEqual(1008, manifest["image_token_target"])
            self.assertEqual((1344, 768), (manifest["image_width"], manifest["image_height"]))
            self.assertEqual(11, len(manifest["images"]))
            self.assertTrue(all(len(image["sha256"]) == 64 for image in manifest["images"]))
        agent = [row for row in rows if row["scenario"] == "agent34"]
        for step in range(1, 7):
            before = agent[step - 1]["body"]["messages"]
            after = agent[step]["body"]["messages"]
            self.assertEqual(before, after[:len(before)])
        # At the rolling boundary, the initial screenshot is intentionally removed.
        self.assertEqual(agent[1]["body"]["messages"][-1], agent[7]["body"]["messages"][1])
        replacements = [row for row in rows if row["scenario"] == "image_replace"]
        self.assertEqual(3, len(replacements))
        stripped = []
        images = []
        for row in replacements:
            self.assertEqual(REPLAY.digest(row["body"]), row["request_sha256"])
            image = row["body"]["messages"][-1]["content"][-1]
            images.append(image["image_url"]["url"])
            image["image_url"]["url"] = "placeholder"
            stripped.append(row["body"])
        self.assertEqual(3, len(set(images)))
        self.assertTrue(all(body == stripped[0] for body in stripped))

    def test_latency_only_group_has_no_fabricated_wall_throughput(self):
        summary = REPLAY.summarize([{"failures": [], "completion_tokens": 20, "e2e_ms": 200}])
        self.assertNotIn("throughput_tokens_per_second", summary)
        self.assertNotIn("wall_seconds", summary)


if __name__ == "__main__":
    unittest.main()
