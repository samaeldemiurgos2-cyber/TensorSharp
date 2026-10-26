#!/usr/bin/env python3
"""Exercise an already running fixture server's distributed HTTP control path.

Run once with --mode local to save a control, then --mode distributed --baseline
LOCAL.json against a Server.Host driver with CLI workers. This tests protocol,
reset/history/concurrency/streaming controls; synthetic output is NOT language
quality and these tiny requests are NOT a throughput/scaling benchmark.
"""
import argparse
import concurrent.futures
import json
import math
from pathlib import Path
import sys
import threading
import time
import urllib.error
import urllib.request


def strict_json(text):
    def reject(value):
        raise ValueError("Nonfinite JSON value: " + value)
    return json.loads(text, parse_constant=reject)


def combine_stream(events):
    terminal = None
    message = {"role": "assistant", "content": "", "thinking": ""}
    for event in events:
        if terminal is not None:
            raise ValueError("Data after terminal done event")
        if event.get("error"):
            raise ValueError("Stream error: " + str(event["error"]))
        delta = event.get("message", {})
        for field in ("content", "thinking"):
            piece = delta.get(field) or ""
            if not isinstance(piece, str):
                raise ValueError("Non-text stream delta: " + field)
            message[field] += piece
        if delta.get("tool_calls"):
            message["tool_calls"] = delta["tool_calls"]
        if event.get("done") is True:
            terminal = event
    if terminal is None:
        raise ValueError("Stream ended without a terminal done event")
    return {**terminal, "message": message}


def completion_failures(row):
    response = row.get("response", {})
    errors = []
    if row.get("error") or row.get("http_status") != 200:
        errors.append("HTTP/request error")
    if response.get("error") or response.get("done") is not True:
        errors.append("Missing successful terminal completion")
    if response.get("done_reason") not in ("stop", "length"):
        errors.append("Unexpected finish reason")
    message = response.get("message", {})
    if message.get("role") != "assistant" or not isinstance(message.get("content"), str):
        errors.append("Malformed assistant message")
    for name, minimum in (("eval_count", 1), ("prompt_eval_count", 1)):
        value = response.get(name)
        if type(value) is not int or value < minimum:
            errors.append("Invalid " + name)
    for name in ("total_duration", "prompt_eval_duration", "eval_duration"):
        value = response.get(name)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            errors.append("Invalid " + name)
    if not math.isfinite(row["wall_seconds"]) or row["wall_seconds"] <= 0:
        errors.append("Invalid wall duration")
    return errors


def signature(row):
    response = row.get("response", {})
    message = response.get("message", {})
    return {"content": message.get("content", ""), "thinking": message.get("thinking") or "",
            "tool_calls": message.get("tool_calls") or [], "finish": response.get("done_reason"),
            "eval_count": response.get("eval_count")}


def compare_baseline(rows, baseline):
    failures = []
    if baseline.get("status") != "completed" or baseline.get("mode") != "local":
        failures.append("Baseline must be a completed local control")
    left = {r["case"]: r for r in baseline.get("rows", []) if r.get("kind") == "chat"}
    right = {r["case"]: r for r in rows if r.get("kind") == "chat"}
    if not left or set(left) != set(right):
        failures.append("Baseline and candidate chat case sets differ or are empty")
    for name in sorted(set(left) & set(right)):
        if left[name].get("failures") or right[name].get("failures"):
            failures.append(name + ": incomplete/failed request")
        if left[name].get("request") != right[name].get("request"):
            failures.append(name + ": request differs from local control")
        if signature(left[name]) != signature(right[name]):
            failures.append(name + ": visible output/count/finish differs from local control")
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True, help="Hosted fixture GGUF basename or model ID")
    parser.add_argument("--mode", choices=("local", "distributed"), required=True)
    parser.add_argument("--backend", default="ggml_cuda")
    parser.add_argument("--switch-backend", default="ggml_cpu")
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.max_tokens < 1 or args.timeout <= 0 or not math.isfinite(args.timeout):
        parser.error("max-tokens and timeout must be positive and finite")
    if args.output.exists():
        parser.error("output must be fresh; preserve prior evidence")
    if args.switch_backend == args.backend:
        parser.error("switch-backend must differ from backend")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows, failures = [], []
    report = {"schema_version": 1, "status": "running", "mode": args.mode,
              "arguments": {**vars(args), "output": str(args.output),
                            "baseline": str(args.baseline) if args.baseline else None},
              "limitations": ["Synthetic HTTP/control integration only; no language-quality or throughput claim.",
                              "Exact parity covers visible output/count/finish, not complete logit vectors.",
                              "Server startup/worker exit and logs must be verified separately.",
                              "Empty synthetic text is allowed if the API completes with positive token counts."],
              "rows": rows, "failures": failures}

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    def request(name, path, body, kind="chat"):
        row = {"case": name, "kind": kind, "path": path, "request": body}
        started = time.perf_counter()
        raw, events, event_seconds = [], [], []
        try:
            req = urllib.request.Request(args.url.rstrip("/") + path, data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=args.timeout) as response:
                row["http_status"] = response.status
                if kind == "chat" and body.get("stream"):
                    for line in response:
                        decoded = line.decode("utf-8")
                        raw.append(decoded)
                        if decoded.strip():
                            events.append(strict_json(decoded))
                            event_seconds.append(time.perf_counter() - started)
                    row["response"] = combine_stream(events)
                else:
                    raw.append(response.read().decode("utf-8"))
                    row["response"] = strict_json(raw[-1])
        except urllib.error.HTTPError as error:
            row["http_status"] = error.code
            raw.append(error.read().decode("utf-8", errors="replace"))
            try:
                row["response"] = strict_json(raw[-1])
            except ValueError:
                row["error"] = "Non-JSON HTTP error response"
        except Exception as error:
            row["error"] = f"{type(error).__name__}: {error}"
        row["wall_seconds"] = time.perf_counter() - started
        row["raw_response"] = "".join(raw)
        if body.get("stream"):
            row.update(stream_events=events, stream_event_seconds=event_seconds)
        row["failures"] = completion_failures(row) if kind == "chat" else []
        return row

    def record(row):
        rows.append(row)
        failures.extend(row["case"] + ": " + failure for failure in row["failures"])
        save()
        print(row["case"], "PASS" if not row["failures"] else "FAIL", f"{row['wall_seconds']:.3f}s", flush=True)
        return row

    def chat(name, messages, stream=False):
        body = {"model": args.model, "messages": messages, "stream": stream, "think": False,
                "multi_agent": False, "skills": [], "skills_discovery": False,
                "options": {"temperature": 0, "seed": 1017, "num_predict": args.max_tokens}}
        return request(name, "/api/chat/ollama", body)

    prompt_a = [{"role": "user", "content": "Fixture control A: 1 2 3."}]
    save()
    try:
        first = record(chat("sequential-a", prompt_a))
        record(chat("sequential-b", [{"role": "user", "content": "Fixture control B: 4 5 6."}]))
        repeated = record(chat("reset-a", prompt_a))
        if signature(first) != signature(repeated):
            failures.append("Identical request changed after the intervening reset/request")
        history = [{"role": "user", "content": "Fixture history: remember 17."}]
        turn1 = record(chat("multiturn-1", history))
        assistant = turn1.get("response", {}).get("message", {})
        history = history + [{"role": "assistant", "content": assistant.get("content", "")}]
        history.append({"role": "user", "content": "Continue the fixture history with 18."})
        record(chat("multiturn-2", history))

        barrier = threading.Barrier(2)
        def concurrent_case(index):
            barrier.wait(timeout=args.timeout)
            return chat(f"concurrent-{index}", [{"role": "user", "content": f"Concurrent fixture {index}: 7 8 9."}])
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            pending = [pool.submit(concurrent_case, i) for i in range(2)]
            for future in concurrent.futures.as_completed(pending):
                record(future.result())

        streamed = record(chat("stream-a", prompt_a, stream=True))
        if signature(first) != signature(streamed):
            failures.append("Streaming reconstruction differs from the identical nonstreaming request")

        same = request("load-same", "/api/models/load", {"model": args.model, "backend": args.backend}, "load")
        if same.get("http_status") != 200 or same.get("response", {}).get("ok") is not True:
            same["failures"].append("Already-loaded checkpoint/backend request was not accepted")
        record(same)
        if args.mode == "distributed":
            switched = request("refuse-backend-switch", "/api/models/load",
                               {"model": args.model, "backend": args.switch_backend}, "load")
            response = switched.get("response", {})
            if (switched.get("http_status", 0) < 400 or response.get("refused") is not True
                    or "startup --model" not in response.get("error", "")
                    or response.get("loadedModel") != same.get("response", {}).get("model")
                    or not response.get("loadedModel")):
                switched["failures"].append("Backend switch did not exercise the distributed startup-load refusal")
            record(switched)
        recovered = record(chat("after-load-guard", prompt_a))
        if signature(first) != signature(recovered):
            failures.append("Loaded model changed after load/switch guard requests")
        if first.get("response", {}).get("model") != recovered.get("response", {}).get("model"):
            failures.append("Response model identity changed after load/switch guard requests")
        if args.baseline:
            failures.extend(compare_baseline(rows, strict_json(args.baseline.read_text(encoding="utf-8"))))
        else:
            report["baseline_comparison"] = "not_run; no exact local parity claim"
        report["status"] = "failed" if failures else "completed"
    except Exception as error:
        failures.append(f"{type(error).__name__}: {error}")
        report["status"] = "failed"
    finally:
        save()
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
