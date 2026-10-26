"""Ensure incomplete synthetic HTTP evidence cannot establish a passing run."""
import copy
import contextlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("http_probe", Path(__file__).parents[1] / "deepseek41-distributed-http-probe.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def completed():
    return {"case": "request", "kind": "chat", "request": {"model": "fixture"}, "failures": [],
            "http_status": 200, "wall_seconds": 0.1,
            "response": {"done": True, "done_reason": "length", "eval_count": 4, "prompt_eval_count": 8,
                         "total_duration": 100, "eval_duration": 70, "prompt_eval_duration": 20,
                         "message": {"role": "assistant", "content": "fixture"}}}


class HttpProbeTests(unittest.TestCase):
    def test_stream_requires_one_terminal_and_reconstructs_text(self):
        with self.assertRaises(ValueError):
            probe.combine_stream([{"message": {"content": "unfinished"}}])
        terminal = completed()["response"] | {"message": {"role": "assistant", "content": ""}}
        events = [{"message": {"content": "fi"}}, {"message": {"content": "xture"}}, terminal]
        self.assertEqual("fixture", probe.combine_stream(events)["message"]["content"])
        with self.assertRaises(ValueError):
            probe.combine_stream(events + [terminal])

    def test_completion_rejects_error_zero_counts_and_nonfinite_timings(self):
        self.assertEqual([], probe.completion_failures(completed()))
        for update in ({"done": False}, {"eval_count": 0}, {"prompt_eval_count": -1},
                       {"eval_duration": float("nan")}, {"done_reason": "error"}):
            row = completed()
            row["response"].update(update)
            self.assertTrue(probe.completion_failures(row), update)

    def test_synthetic_empty_text_does_not_claim_language_quality(self):
        row = completed()
        row["response"]["message"]["content"] = ""
        self.assertEqual([], probe.completion_failures(row))

    def test_baseline_requires_same_completed_local_case_set_and_outputs(self):
        baseline = {"mode": "local", "status": "completed", "rows": [completed()]}
        self.assertEqual([], probe.compare_baseline([completed()], baseline))
        for change in ("missing", "output", "request", "failed"):
            candidate = [completed()]
            if change == "missing": candidate = []
            if change == "output": candidate[0]["response"]["message"]["content"] = "different"
            if change == "request": candidate[0]["request"] = {"model": "different"}
            if change == "failed": candidate[0]["failures"] = ["request failed"]
            self.assertTrue(probe.compare_baseline(candidate, copy.deepcopy(baseline)), change)

    def test_nonfinite_wire_json_is_rejected_before_evidence_serialization(self):
        with self.assertRaises(ValueError):
            probe.strict_json('{"eval_duration": NaN}')

    def test_complete_local_and_distributed_main_flows_with_real_http(self):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                status = 200
                if self.path == "/api/models/load":
                    if body["backend"] == "ggml_cuda":
                        response = {"ok": True, "model": "fixture.gguf"}
                    else:
                        status = 500
                        response = {"ok": False, "refused": True, "loadedModel": "fixture.gguf",
                                    "error": "Distributed tensor parallelism requires a startup --model."}
                    wire = json.dumps(response)
                else:
                    self.assert_chat_path()
                    response = completed()["response"] | {"model": "fixture.gguf"}
                    if body.get("stream"):
                        terminal = response | {"message": {"role": "assistant", "content": ""}}
                        wire = "\n".join(json.dumps(event) for event in (
                            {"done": False, "message": {"role": "assistant", "content": "fixture"}}, terminal)) + "\n"
                    else:
                        wire = json.dumps(response)
                encoded = wire.encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def assert_chat_path(self):
                if self.path != "/api/chat/ollama":
                    raise AssertionError(self.path)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                local_path, distributed_path = Path(directory) / "local.json", Path(directory) / "distributed.json"
                common = ["http-probe", "--url", f"http://127.0.0.1:{server.server_port}",
                          "--model", "fixture.gguf", "--timeout", "5"]
                with contextlib.redirect_stdout(io.StringIO()):
                    with patch.object(probe.sys, "argv", common + ["--mode", "local", "--output", str(local_path)]):
                        self.assertEqual(0, probe.main())
                    with patch.object(probe.sys, "argv", common + ["--mode", "distributed", "--output", str(distributed_path),
                                                                "--baseline", str(local_path)]):
                        self.assertEqual(0, probe.main())
                report = json.loads(distributed_path.read_text())
                self.assertEqual("completed", report["status"])
                self.assertEqual([], report["failures"])
                self.assertEqual(11, len(report["rows"]))
                self.assertEqual(9, sum(row["kind"] == "chat" for row in report["rows"]))
                for row in report["rows"]:
                    self.assertGreater(row["wall_seconds"], 0)
                    self.assertTrue(row["raw_response"])
                by_case = {row["case"]: row for row in report["rows"]}
                self.assertTrue(by_case["stream-a"]["stream_events"][-1]["done"])
                self.assertEqual(500, by_case["refuse-backend-switch"]["http_status"])
                self.assertEqual(200, by_case["after-load-guard"]["http_status"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
