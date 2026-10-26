import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import signal
import socket
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("run_qwen38_perf", Path(__file__).parents[1] / "run-qwen38-perf.py")
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def fixture_row(lane=0, step=0):
    body = {"messages": [{"role": "user", "content": "123 + 456"}], "temperature": 0}
    return {"scenario": "arithmetic", "lane": lane, "step": step, "body": body,
            "request_sha256": RUNNER.REPLAY.digest(body), "expected": {"answer": "579"}}


def result_row(row):
    return {key: row[key] for key in ("scenario", "lane", "step", "request_sha256", "expected")} | {
        "concurrency": 1, "wire_request_sha256": RUNNER.REPLAY.digest(dict(row["body"], model="model")),
        "answer": "579", "reasoning": "", "tool_calls": [], "errors": [], "failures": [],
        "http_status": 200, "done": True, "finish_reason": "stop", "prompt_tokens": 20,
        "completion_tokens": 3, "ttft_ms": 100, "e2e_ms": 1000}


class FakeServer:
    pid = 12345
    returncode = None

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = -15
        return self.returncode


class RunnerTests(unittest.TestCase):
    def setUp(self):
        artifacts = RUNNER.ROOT / "artifacts"
        artifacts.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=artifacts, prefix="perf-runner-test-")
        self.directory = Path(self.temporary.name)
        self.output = self.directory / "output"
        self.output.mkdir()
        self.fixture = self.directory / "fixture.jsonl"
        self.fixture.write_text(json.dumps(fixture_row()) + "\n")
        self.config = {"argv": ["owned-server"], "cwd": str(self.directory), "fixture": str(self.fixture),
                       "url": "http://127.0.0.1:5088", "model": "model", "label": "test", "engine": "tensorsharp",
                       "scenarios": "arithmetic", "concurrency": "1", "source_revision": "declared-revision"}
        self.config_path = self.directory / "config.json"
        self.config_path.write_text(json.dumps(self.config))

    def tearDown(self):
        self.temporary.cleanup()

    def report(self):
        return {"model": "model", "label": "test", "complete": True, "passed": True,
                "fixture_sha256": RUNNER.sha(self.fixture), "results": [result_row(fixture_row())]}

    def execute_fake(self, report=None, create_report=True, preflight_error=None, mutate_fixture=False):
        original = self.fixture.read_bytes()
        report = self.report() if report is None else report
        def capture(argv, cwd=None):
            return {"exit_code": 0, "output": "10, 0, 20" if "--query-gpu=memory.used,utilization.gpu,power.draw" in argv else ""}
        def urlopen(url, timeout=None):
            return io.BytesIO(json.dumps({"data": [{"id": "model"}]}).encode() if url.endswith("/v1/models") else b"metric 1\n")
        def replay(command, **kwargs):
            snapshot = Path(command[command.index("--fixture") + 1])
            self.assertNotEqual(self.fixture, snapshot)
            if mutate_fixture:
                self.fixture.write_text("changed after launch")
            self.assertEqual(original, snapshot.read_bytes())
            self.assertEqual(7200, kwargs["timeout"])
            if create_report:
                (self.output / "replay.json").write_text(json.dumps(report))
            return SimpleNamespace(returncode=0)
        def killpg(pid, sig):
            self.assertEqual(12345, pid)
            if sig == 0:
                raise ProcessLookupError()
        server = FakeServer()
        with patch.object(RUNNER, "capture", side_effect=capture), \
             patch.object(RUNNER, "ensure_endpoint_unused", side_effect=preflight_error), \
             patch.object(RUNNER.subprocess, "Popen", return_value=server) as launch, \
             patch.object(RUNNER.subprocess, "run", side_effect=replay), \
             patch.object(RUNNER.urllib.request, "urlopen", side_effect=urlopen), \
             patch.object(RUNNER.os, "killpg", side_effect=killpg):
            code = RUNNER.execute(self.config_path, self.output)
            if preflight_error:
                launch.assert_not_called()
            else:
                self.assertTrue(launch.call_args.kwargs["start_new_session"])
        return code, json.loads((self.output / "provenance.json").read_text())

    def test_frozen_fixture_is_used_and_effective_environment_is_recorded(self):
        code, provenance = self.execute_fake(mutate_fixture=True)
        self.assertEqual(0, code)
        self.assertTrue(provenance["passed"])
        self.assertTrue(provenance["complete"])
        self.assertTrue(provenance["shutdown_requested"])
        self.assertEqual(1, provenance["planned_requests"])
        self.assertIn("served_models", provenance)
        self.assertEqual(str(self.output / "prefix-cache"), provenance["effective_environment_overrides"]["TENSORSHARP_PREFIX_CACHE_DIR"])

    def test_launch_environment_records_inheritance_override_and_unset_without_secrets(self):
        self.config["environment"] = {"TS_PREFIX_CACHE_ISOLATION": "lineage", "TS_GGML_HOST_HUGEPAGES": "0"}
        self.config_path.write_text(json.dumps(self.config))
        inherited = {"TS_PREFIX_CACHE_ISOLATION": "shared", "TS_GGML_HOST_HUGEPAGES": "1",
                     "GGML_CUDA_PDL": "1", "CUDA_VISIBLE_DEVICES": "2",
                     "TENSORSHARP_API_KEY": "private-api-key", "TS_JEV_TRANSCRIPTION_API_KEY": "private-key",
                     "TENSORSHARP_ACCESS_TOKEN": "private-token", "GGML_UNKNOWN_SECRET": "private-secret"}
        with patch.dict(RUNNER.os.environ, inherited, clear=True):
            code, provenance = self.execute_fake()
        self.assertEqual(0, code)
        actual = provenance["benchmark_launch_environment"]
        self.assertEqual({"value": "lineage", "source": "override"}, actual["TS_PREFIX_CACHE_ISOLATION"])
        self.assertEqual({"value": "0", "source": "override"}, actual["TS_GGML_HOST_HUGEPAGES"])
        self.assertEqual({"value": "1", "source": "inherited"}, actual["GGML_CUDA_PDL"])
        self.assertEqual({"value": "2", "source": "inherited"}, actual["CUDA_VISIBLE_DEVICES"])
        self.assertEqual({"value": None, "source": "unset"}, actual["GGML_CUDA_DISABLE_GRAPHS"])
        self.assertEqual({"value": str(self.output / "prefix-cache"), "source": "override"},
                         actual["TENSORSHARP_PREFIX_CACHE_DIR"])
        serialized = json.dumps(provenance)
        for secret in ("private-api-key", "private-key", "private-token", "private-secret"):
            self.assertNotIn(secret, serialized)
        for key in ("TENSORSHARP_API_KEY", "TS_JEV_TRANSCRIPTION_API_KEY", "TENSORSHARP_ACCESS_TOKEN", "GGML_UNKNOWN_SECRET"):
            self.assertNotIn(key, actual)

    def test_preflight_failure_preserves_failed_provenance_without_starting_server(self):
        code, provenance = self.execute_fake(preflight_error=ValueError("listener occupied"))
        self.assertEqual(1, code)
        self.assertFalse(provenance["complete"])
        self.assertFalse(provenance["passed"])
        self.assertTrue(any("listener occupied" in error for error in provenance["errors"]))
        self.assertTrue((self.output / "gpu-samples.json").exists())

    def test_zero_exit_without_report_is_not_success(self):
        code, provenance = self.execute_fake(create_report=False)
        self.assertEqual(1, code)
        self.assertIn("replay report missing", provenance["errors"])
        self.assertFalse(provenance["passed"])

    def test_zero_exit_with_incomplete_report_is_not_success(self):
        report = self.report()
        report["complete"] = False
        code, provenance = self.execute_fake(report=report)
        self.assertEqual(1, code)
        self.assertFalse(provenance["passed"])

    def test_expected_request_set_and_wire_hash_are_both_required(self):
        expected = RUNNER.expected_requests(self.fixture.read_bytes(), self.config)
        valid = self.report()
        path = self.output / "replay.json"
        for mutate in (lambda report: report["results"].clear(),
                       lambda report: report["results"][0].update(wire_request_sha256="b" * 64),
                       lambda report: report["results"].append(copy.deepcopy(report["results"][0])),
                       lambda report: report["results"][0].update(answer="999")):
            report = copy.deepcopy(valid)
            mutate(report)
            path.write_text(json.dumps(report))
            self.assertTrue(RUNNER.verify_replay(path, valid["fixture_sha256"], expected, self.config))

    def test_fixture_must_cover_all_requested_lanes_and_contiguous_steps(self):
        cfg = self.config | {"concurrency": "1,2"}
        with self.assertRaises(ValueError):
            RUNNER.expected_requests(self.fixture.read_bytes(), cfg)
        fixture = b"\n".join(json.dumps(fixture_row(step=step)).encode() for step in (0, 2))
        with self.assertRaises(ValueError):
            RUNNER.expected_requests(fixture, self.config)

    def test_mutated_fixture_request_hash_is_rejected(self):
        row = fixture_row()
        row["body"]["temperature"] = 1
        with self.assertRaises(ValueError):
            RUNNER.expected_requests(json.dumps(row).encode(), self.config)

    def test_existing_listener_is_refused(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            with self.assertRaisesRegex(ValueError, "unowned"):
                RUNNER.ensure_endpoint_unused(f"http://127.0.0.1:{listener.getsockname()[1]}")

    def test_cleanup_kills_children_remaining_after_parent_exits(self):
        server = FakeServer()
        with patch.object(RUNNER.os, "killpg") as killpg:
            details = RUNNER.stop_server(server)
        self.assertTrue(details["forced_shutdown"])
        self.assertEqual([(12345, signal.SIGTERM), (12345, 0), (12345, signal.SIGKILL)],
                         [call.args for call in killpg.call_args_list])

    def test_unavailable_monitor_is_not_reported_as_idle(self):
        with patch.object(RUNNER, "capture", return_value={"exit_code": -1, "output": "unavailable"}):
            state = RUNNER.wait_for_idle_gpu(["nvidia-smi"], timeout=0)
        self.assertEqual(-1, state["exit_code"])

    def test_cuda_context_release_is_given_a_bounded_cleanup_window(self):
        with patch.object(RUNNER, "capture", side_effect=[{"exit_code": 0, "output": "123, owned-server, 100"},
                                                         {"exit_code": 0, "output": ""}]), \
             patch.object(RUNNER.time, "sleep"):
            state = RUNNER.wait_for_idle_gpu(["nvidia-smi"])
        self.assertEqual("", state["output"])


if __name__ == "__main__":
    unittest.main()
