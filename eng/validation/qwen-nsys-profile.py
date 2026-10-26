#!/usr/bin/env python3
"""Capture one prescribed HTTP request after optional unprofiled cache priming.

Nsight Systems interactive launch/start/stop keeps model loading and priming out
of the report. This creates a NEW server and shuts down only that profiling
session. Run only when the GPU is otherwise idle. Reports contain the full
request, including decode; --max-tokens 1 is a prefill diagnostic, not a quality
or throughput benchmark. Requires nsys 2026 CLI and the standard Python library.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit
import uuid

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("qwen_replay", HERE / "qwen38-agent-replay.py")
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


def select_row(fixture, scenario, lane, step, max_tokens=None):
    matches = [row for row in fixture if (row["scenario"], row["lane"], row["step"]) == (scenario, lane, step)]
    if len(matches) != 1:
        raise ValueError(f"Expected one fixture row for {(scenario, lane, step)}, got {len(matches)}")
    row = json.loads(json.dumps(matches[0]))
    if replay.digest(row["body"]) != row["request_sha256"]:
        raise ValueError("Fixture digest mismatch")
    if max_tokens is not None:
        row["body"]["max_tokens"] = max_tokens
        row["request_sha256"] = replay.digest(row["body"])
    return row


def process_counters(pid):
    # Linux /proc stat indices after the '(comm)' field. These counters sum all
    # threads; unlike CUDA API durations they exclude time blocked on the GPU.
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    tick_ms = 1000 / os.sysconf("SC_CLK_TCK")
    return {"user_ms": int(fields[11]) * tick_ms, "system_ms": int(fields[12]) * tick_ms,
            "minor_faults": int(fields[7]), "major_faults": int(fields[9])}


def models(url):
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/v1/models", timeout=2) as response:
            data = json.load(response)
        return data["data"][0]["id"]
    except (OSError, ValueError, KeyError, IndexError):
        return None


def exec_server(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-spec", required=True, type=Path)
    parser.add_argument("--pid-file", required=True, type=Path)
    args = parser.parse_args(argv)
    cfg = json.loads(args.server_spec.read_text())
    command = cfg["argv"]
    if not command or not all(isinstance(x, str) for x in command):
        raise ValueError("server-spec.argv must be a nonempty string array")
    env = dict(os.environ, **cfg.get("env", {}))
    if cfg.get("cwd"):
        os.chdir(cfg["cwd"])
    args.pid_file.write_text(str(os.getpid()))
    os.execvpe(command[0], command, env)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nsys", default="nsys")
    parser.add_argument("--server-spec", type=Path, required=True, help="JSON: argv array, optional env object and cwd")
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:5089")
    parser.add_argument("--scenario", choices=("repeat19k", "incremental", "image_replace", "agent34"), required=True)
    parser.add_argument("--lane", type=int, default=0)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--prime-steps", default="0", help="Comma-separated unprofiled steps, or empty for a cold request")
    parser.add_argument("--max-tokens", type=int, help="Optional cap applied to both prime and measured requests")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--startup-timeout", type=float, default=600)
    parser.add_argument("--cpu-sampling", choices=("none", "process-tree"), default="none")
    parser.add_argument("--trace", default="cuda,nvtx,osrt", help="Use osrt for a CPU-only harness smoke test")
    parser.add_argument("--output", type=replay.evidence_path, required=True, help="Fresh evidence filename prefix")
    args = parser.parse_args(argv)
    if args.max_tokens is not None and args.max_tokens < 1:
        parser.error("max-tokens must be positive")
    if args.timeout <= 0 or args.startup_timeout <= 0:
        parser.error("timeouts must be positive")
    if urlsplit(args.url).hostname not in ("127.0.0.1", "localhost", "::1"):
        parser.error("Use an isolated loopback server")
    if models(args.url) is not None:
        parser.error("The selected port already serves a model; select a free port")
    prefix = args.output
    if any(prefix.parent.glob(prefix.name + ".*")):
        parser.error("Output prefix already exists")
    fixture = [json.loads(line) for line in args.fixture.read_text().splitlines() if line]
    measured = select_row(fixture, args.scenario, args.lane, args.step, args.max_tokens)
    primes = [select_row(fixture, args.scenario, args.lane, int(s), args.max_tokens)
              for s in args.prime_steps.split(",") if s.strip()]
    cfg = json.loads(args.server_spec.read_text())
    session = "qwen-profile-" + uuid.uuid4().hex[:12]
    pid_file = Path(str(prefix) + ".pid")
    report = {"session": session, "server_spec": cfg, "scenario": args.scenario,
              "fixture": str(args.fixture.resolve()), "max_tokens_override": args.max_tokens,
              "cpu_sampling": args.cpu_sampling, "trace": args.trace, "prime_results": [],
              "scope": "One isolated request, including decode. Profiling perturbs timings; compare unprofiled runs separately.",
              "gpu_gap_limit": "Wall time minus GPU activity is host/queue/idle time, not measured CPU execution time. Process CPU counters are separate, summed over threads.",
              "complete": False}
    profiler_log = open(str(prefix) + ".nsys.log", "w")
    server_log = open(str(prefix) + ".server.log", "w")
    launched = collecting = False
    launcher = None

    def nsys(*command, timeout=180):
        subprocess.run([args.nsys, *command], check=True, stdout=profiler_log,
                       stderr=subprocess.STDOUT, timeout=timeout)

    try:
        report["nsys_version"] = subprocess.check_output([args.nsys, "--version"], text=True).strip()
        launch = [args.nsys, "launch", "--session-new=" + session, "--trace=" + args.trace,
                  "--cuda-graph-trace=node", sys.executable, str(Path(__file__).resolve()), "_exec",
                  "--server-spec", str(args.server_spec.resolve()), "--pid-file", str(pid_file)]
        launcher = subprocess.Popen(launch, stdout=server_log, stderr=subprocess.STDOUT)
        launched = True
        deadline = time.monotonic() + args.startup_timeout
        while True:
            model = models(args.url)
            if model is not None:
                break
            if launcher.poll() not in (None, 0):
                raise RuntimeError("nsys launch failed; inspect server log")
            if time.monotonic() >= deadline:
                raise TimeoutError("Server readiness deadline exceeded")
            time.sleep(.25)
        report["model"] = model
        pid = int(pid_file.read_text())
        report["server_pid"] = pid
        for row in primes:
            result = replay.request(args.url, model, row, args.timeout)
            report["prime_results"].append(result)
            if result["errors"] or not result["done"]:
                raise RuntimeError("Cache priming failed; inspect metadata")
        nsys("start", "--session=" + session, "--output=" + str(prefix),
             "--sample=" + args.cpu_sampling, "--cpuctxsw=" + args.cpu_sampling,
             "--discard-environment=true", "--force-overwrite=false")
        collecting = True
        cpu_before = process_counters(pid)
        report["request_started_unix_ns"] = time.time_ns()
        report["result"] = replay.request(args.url, model, measured, args.timeout)
        report["request_finished_unix_ns"] = time.time_ns()
        cpu_after = process_counters(pid)
        report["server_process_cpu"] = {k: cpu_after[k] - cpu_before[k] for k in ("user_ms", "system_ms")}
        fault_keys = ("minor_faults", "major_faults")
        report["server_process_faults"] = {"before": {k: cpu_before[k] for k in fault_keys},
            "after": {k: cpu_after[k] for k in fault_keys},
            "delta": {k: cpu_after[k] - cpu_before[k] for k in fault_keys}}
        nsys("stop", "--session=" + session)
        collecting = False
        report["request_complete"] = (not report["result"]["errors"] and report["result"]["done"]
            and report["result"].get("finish_reason") in ("stop", "length", "tool_calls"))
        if not report["request_complete"]:
            raise RuntimeError("Measured request did not finish correctly")
        nsys("export", "--type=sqlite", "--output=" + str(prefix) + ".sqlite", str(prefix) + ".nsys-rep")
        report["complete"] = True
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        if collecting:
            try:
                nsys("stop", "--session=" + session)
            except Exception as error:
                report["stop_error"] = repr(error)
        if launched:
            try:
                nsys("shutdown", "--session=" + session, "--kill=sigterm", timeout=30)
            except Exception as error:
                report["shutdown_error"] = repr(error)
        if launcher is not None:
            try:
                launcher.wait(timeout=15)
            except subprocess.TimeoutExpired:
                report["launcher_wait_timeout"] = True
        Path(str(prefix) + ".json").write_text(json.dumps(report, indent=2) + "\n")
        profiler_log.close()
        server_log.close()
    subprocess.run([sys.executable, str(HERE / "qwen-nsys-summary.py"),
                    "--sqlite", str(prefix) + ".sqlite", "--request", str(prefix) + ".json",
                    "--output", str(prefix) + ".summary.json"], check=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "_exec":
        exec_server(sys.argv[2:])
    else:
        main()
