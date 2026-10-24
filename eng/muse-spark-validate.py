#!/usr/bin/env python3
"""Strict Muse Spark HTTP correctness and comparative performance evidence.

Run each engine separately against the SAME checkpoint, quantization, hardware,
context, cache policy, sampling, and placement settings. This tool never launches
servers or downloads models. It reuses the repository's exact-answer/tool tests.

  python eng/muse-spark-validate.py --engine llamacpp --url http://localhost:8082 \
    --model muse-spark --weights-id REPO@REVISION/QUANT --profile same-host-gpu \
    --provenance provenance.json --output docs/validation/muse-spark/llama.json
  python eng/muse-spark-validate.py --engine tensorsharp --url http://localhost:8083 \
    --model muse-spark --weights-id REPO@REVISION/QUANT --profile same-host-gpu \
    --provenance provenance.json --reference docs/validation/muse-spark/llama.json \
    --output docs/validation/muse-spark/tensorsharp.json

Provenance JSON should have hardware_id, weights_sha256 (map of shard names to
SHA256), settings (context/batch/cache/placement etc.), upstream_ggml_revision,
upstream_ggml_clean (boolean), server_revision, server_command, and server_log.
Keep shared execution constraints in settings. Record implementation-specific
kernels, source-page advice, fusion and streaming under engine_options or the
observed environment instead; these need not match between different engines.
Missing or mismatched comparison provenance is INCONCLUSIVE, never parity.
Run multiple launch profiles for CPU expert offload, multi-GPU and speculation;
profile names alone do not prove those internal mechanisms executed. Record real
device/placement/draft-acceptance telemetry separately in the final report.

Launch llama.cpp with the template from prepare_muse_spark_llama_reference.py.
It preserves the released source and matches TensorSharp's date and answer-prefix
behavior. This harness uses plain content parsing only for primed no-tool answers;
tool-bearing requests retain llama.cpp's routed parser and reasoning extraction.
TensorSharp requests explicitly disable intrinsic skill discovery and delegation
with skills:[], skills_discovery:false, and multi_agent:false. Caller-declared
tools remain available. Launch with code execution disabled for this comparison.

For a strict benchmark without prefix reuse, launch TensorSharp with
--no-prefix-cache and llama.cpp with --no-cache-prompt, record the observed launch
and settings.prefix_cache=false in provenance, then use --cache-policy disabled
with --scenarios short,decode,long_8k. Every request sends cache_prompt:false;
TensorSharp's launch flag is required because its HTTP API does not implement
that field. Warmup and every measured turn must expose cached_tokens exactly 0.
This isolates fresh-KV work with separately warmed weights; it does not measure
the engines' default cache-enabled conversation performance.

The generated PNG fixtures measure simple image spatial/color comprehension.
The text fixtures screen integration, not broad reasoning/model quality. Long
case names are approximate target token counts; actual prompt usage is retained.
Staggered arrivals prove overlapping service, not an internally batched kernel.
Optional --scenarios tool_string_arguments checks exact typed function arguments
without executing the returned tool; it is not part of the default scenario set.
"""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import statistics
import struct
import sys
import threading
import time
import zlib

ROOT = Path(__file__).resolve().parents[1]
VISION_FIXTURES = ROOT / "InferenceWeb.Tests/Fixtures/MuseSpark"
sys.path.insert(0, str(ROOT / "benchmarks/engine_comparison"))
import engines
import validate_inference as validation

EXTRA_SCENARIOS = ("parallel", "cache_reuse", "image", "staggered_requests", "image_chart",
                   "json_schema_adversarial", "tool_string_arguments")
DEFAULT_SCENARIOS = ("short", "short_zh", "decode", "json", "json_schema", "json_unicode",
                     "long_8k", "multi_turn", "tool_round_trip", "agentic",
                     "parallel", "cache_reuse", "image", "staggered_requests")
INTERNAL_FEATURES = {
    "continuous_batching": "Overlapping HTTP requests alone do not prove batched execution; requires scheduler/kernel telemetry.",
    "speculative_decoding": "Requires an enabled draft-model or n-gram profile and measured proposed/accepted draft tokens.",
    "moe_expert_cpu_offload": "Requires an offload launch profile and observed expert CPU placement/execution.",
    "multiple_gpus": "Requires a multi-GPU launch profile and observed model placement/execution on each device.",
    "audio": "The Spark audio pyramid needs its own encoder and speech fixtures; vision coverage does not cover audio.",
    "video": "Still-image fixtures do not establish frame sequencing or temporal understanding.",
}


class Unavailable(RuntimeError):
    """An observation needed to make a claim was not exposed by this server."""


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def image_png(reverse=False):
    """Deterministic PNG, no graphics libraries or external image dependencies."""
    width, height = 256, 128
    red, blue = bytes((235, 30, 30)), bytes((20, 55, 235))
    left, right = (blue, red) if reverse else (red, blue)
    rows = (b"\0" + left * (width // 2) + right * (width // 2)) * height
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


def chart_png(alternate=False):
    """Font-independent bar chart and OCR fixture using explicit 5x7 glyphs."""
    glyphs = {
        'A': ('01110','10001','10001','11111','10001','10001','10001'),
        'B': ('11110','10001','10001','11110','10001','10001','11110'),
        'C': ('01111','10000','10000','10000','10000','10000','01111'),
        'D': ('11110','10001','10001','10001','10001','10001','11110'),
        'E': ('11111','10000','10000','11110','10000','10000','11111'),
        'O': ('01110','10001','10001','10001','10001','10001','01110'),
        ':': ('00000','00100','00100','00000','00100','00100','00000'),
        '1': ('00100','01100','00100','00100','00100','00100','01110'),
        '2': ('01110','10001','00001','00010','00100','01000','11111'),
        '3': ('11110','00001','00001','01110','00001','00001','11110'),
        '4': ('00010','00110','01010','10010','11111','00010','00010'),
        '6': ('01110','10000','10000','11110','10001','10001','01110'),
        '7': ('11111','00001','00010','00100','01000','01000','01000'),
        '8': ('01110','10001','10001','01110','10001','10001','01110'),
        '9': ('01110','10001','10001','01111','00001','00001','01110'),
    }
    width, height = 320, 256
    pixels = bytearray(b'\xff' * width * height * 3)
    def rect(x, y, w, h, color):
        for row in range(y, y + h):
            offset = (row * width + x) * 3
            pixels[offset:offset + w * 3] = bytes(color) * w
    def label(text, x, y, scale):
        for char in text:
            for row, bits in enumerate(glyphs[char]):
                for column, bit in enumerate(bits):
                    if bit == '1':
                        rect(x + column * scale, y + row * scale, scale, scale, (15, 15, 15))
            x += 6 * scale
    label('CODE:' + ('4826' if alternate else '7319'), 52, 8, 4)
    rect(30, 58, 2, 153, (30, 30, 30))
    rect(30, 209, 260, 2, (30, 30, 30))
    heights = (45, 135, 90) if alternate else (90, 45, 135)
    for index, bar_height in enumerate(heights):
        x = 50 + index * 90
        rect(x, 209 - bar_height, 40, bar_height, (30, 95, 210))
        label('ABC'[index], x + 12, 223, 3)
    rows = b''.join(b'\0' + pixels[row * width * 3:(row + 1) * width * 3] for row in range(height))
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data) & 0xffffffff)
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress(rows)) + chunk(b'IEND', b''))


class WireLines(list):
    def __init__(self, first_token):
        super().__init__()
        self.first_token = first_token

    def append(self, line):
        super().append(line)
        if not self.first_token or self.first_token.is_set() or not line.startswith("data:"):
            return
        try:
            data = json.loads(line[5:].strip())
            for choice in data.get("choices", []):
                delta = choice.get("delta", {})
                if any(delta.get(key) for key in ("content", "reasoning_content", "tool_calls")):
                    self.first_token.set()
        except (ValueError, TypeError):
            pass  # The streaming parser owns validation, including malformed JSON.


class WireLog(dict):
    def __init__(self, first_token=None):
        super().__init__()
        self.first_token = first_token

    def __setitem__(self, key, value):
        super().__setitem__(key, WireLines(self.first_token) if key == "sse_lines" else value)


def reference_body(engine, thinking, has_tools=False):
    """Request settings for the matched reference template, scoped to Muse Spark."""
    if engine != "llamacpp":
        return {}
    options = {"chat_template_kwargs": {"enable_thinking": thinking, "current_date": ""}}
    if not thinking and not has_tools:
        options["reasoning_format"] = "none"
        # The unchanged Spark plain parser preserves protocol tokens in its
        # content tail. Stop at the end-of-turn marker on the server, keeping
        # the matched rendered prompt and the actual usage counters unchanged.
        options["stop"] = ["<|eot|>"]
    return options


def request_adapter_body(engine, thinking, has_tools=False, cache_policy="profile"):
    if engine == "tensorsharp":
        # API defaults can advertise server skills and delegation tools even if
        # this client did not declare tools. Match the caller-only llama input.
        body = {"skills": [], "skills_discovery": False, "multi_agent": False}
    else:
        body = reference_body(engine, thinking, has_tools)
    if cache_policy == "disabled":
        body["cache_prompt"] = False
    return body


def no_cache_profile_errors(engine, provenance, scenarios):
    """Fail closed on a requested opt-out that only appears in an ignored body."""
    errors = []
    if not isinstance(provenance, dict):
        return ["No-cache profile requires an observed provenance object"]
    if provenance.get("launch_observed") is not True:
        errors.append("No-cache profile requires an observed server launch")
    settings = provenance.get("settings")
    if not isinstance(settings, dict) or settings.get("prefix_cache") is not False:
        errors.append("No-cache profile requires provenance settings.prefix_cache=false")
    flag = {"tensorsharp": "--no-prefix-cache", "llamacpp": "--no-cache-prompt"}.get(engine)
    command = provenance.get("server_command")
    try:
        arguments = shlex.split(command) if isinstance(command, str) else []
    except ValueError:
        arguments = []
    if flag is None or flag not in arguments:
        errors.append("No-cache profile requires the observed server command to contain " + str(flag))
    if "cache_reuse" in scenarios:
        errors.append("cache_reuse requires prefix caching; omit it from a no-cache profile")
    return errors


def no_cache_case_errors(case):
    errors = []
    turns = case.get("turns") or []
    if not turns:
        errors.append("No-cache profile has no request/cache telemetry")
    for index, turn in enumerate(turns):
        request_body = turn.get("request") or {}
        # Standard fixtures record the common client wrapper; extra fixtures
        # record its final HTTP body. Require the opt-out in either exact form.
        request_flag = request_body.get("cache_prompt", (request_body.get("extra_body") or {}).get("cache_prompt"))
        if request_flag is not False:
            errors.append(f"Turn {index}: cache_prompt:false was not recorded on the request")
        metrics = turn.get("metrics") or {}
        cached = ((metrics.get("raw_usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens")
        if type(cached) is not int or cached != 0:
            errors.append(f"Turn {index}: disabled prefix cache requires integer cached_tokens=0, got {cached!r}")
    return errors


def enforce_cache_policy(args, case):
    if getattr(args, "cache_policy", "profile") == "disabled":
        errors = no_cache_case_errors(case)
        if errors:
            case["cache_policy_errors"] = errors
            case["status"] = "fail"
            case["detail"] = "; ".join(([case["detail"]] if case.get("detail") else []) + errors)
    return case


def run_standard_case(args, name, tag, **kwargs):
    has_tools = bool(validation.case_spec(name, tag)["tools"])
    result = validation.run_case(args.url, args.model, args.engine, name, tag,
        thinking=args.thinking, timeout=args.timeout,
        extra_body=request_adapter_body(args.engine, args.thinking, has_tools,
                                        getattr(args, "cache_policy", "profile")), **kwargs)
    return enforce_cache_policy(args, result)


def request(args, messages, *, max_tokens=128, response_format=None, first_token=None,
            tools=None, tool_choice=None, parallel_tool_calls=None):
    body = {"model": args.model, "messages": messages, "stream": True,
            "stream_options": {"include_usage": True}, "timings_per_token": True,
            "max_tokens": max_tokens, **validation.SAMPLING,
            **engines.thinking_body(args.engine, args.thinking),
            **request_adapter_body(args.engine, args.thinking, bool(tools),
                                   getattr(args, "cache_policy", "profile"))}
    if response_format:
        body["response_format"] = response_format
    if tools:
        body["tools"] = tools
    if tool_choice is not None:
        body["tool_choice"] = tool_choice
    if parallel_tool_calls is not None:
        body["parallel_tool_calls"] = parallel_tool_calls
    wire = WireLog(first_token)
    metrics = engines._run_streaming(args.url.rstrip("/") + "/v1/chat/completions", body, args.timeout, wire)
    lines = wire.get("sse_lines", [])
    require(any(line.strip() == "data: [DONE]" for line in lines), "SSE ended without [DONE]")
    require(metrics.get("usage_present"), "Server omitted token usage")
    require(metrics.get("prompt_tokens", 0) > 0 and metrics.get("completion_tokens", 0) > 0, "Empty token usage")
    require(metrics.get("finish_reason") in ("stop", "length", "tool_calls"), "Missing/invalid finish reason")
    for line in lines:
        if line.startswith("data:") and line[5:].strip() != "[DONE]":
            event = json.loads(line[5:].strip())
            if event.get("usage"):
                metrics["raw_usage"] = event["usage"]
    # Keep exact image bytes by hash; avoid duplicating their base64 in reports.
    recorded = json.loads(json.dumps(body))
    for message in recorded["messages"]:
        if isinstance(message.get("content"), list):
            for part in message["content"]:
                if part.get("type") == "image_url":
                    image_url = part["image_url"]["url"]
                    part["image_url"]["url"] = "sha256:" + hashlib.sha256(image_url.encode()).hexdigest()
    return {"request": recorded, "metrics": metrics}


def validate_answer(turn, expected):
    metrics = turn["metrics"]
    require(metrics["finish_reason"] == "stop", "Answer was truncated or ended in an unexpected tool call")
    text = validation.assistant_content(metrics)
    # Compare canonical JSON so bool/int equivalence cannot hide a wrong type.
    actual = json.loads(text)
    require(json.dumps(actual, sort_keys=True) == json.dumps(expected, sort_keys=True),
            f"Incorrect JSON answer: expected {expected!r}, got {actual!r}")


def validate_tool_arguments(turn, name, expected):
    metrics = turn["metrics"]
    calls = metrics.get("tool_call_details") or []
    require(metrics.get("finish_reason") == "tool_calls" and len(calls) == 1,
            "Expected exactly one structured function call with finish_reason=tool_calls")
    call = calls[0]
    require(call.get("type") == "function" and isinstance(call.get("id"), str) and call["id"],
            "Function call requires type=function and a nonempty id")
    function = call.get("function") or {}
    require(function.get("name") == name and isinstance(function.get("arguments"), str),
            "Incorrect function name or non-string arguments payload")
    actual = json.loads(function["arguments"])
    require(json.dumps(actual, sort_keys=True, allow_nan=False)
            == json.dumps(expected, sort_keys=True, allow_nan=False),
            f"Incorrect tool argument values/types: expected {expected!r}, got {actual!r}")


def extra_case(args, name, tag):
    result = {"scenario": name, "tag": tag, "status": "fail", "turns": [],
              "concurrency": args.concurrency if name in ("parallel", "staggered_requests") else 1,
              "input_sha256": validation.digest({"name": name, "tag": tag, "thinking": args.thinking,
                  "concurrency": args.concurrency if name in ("parallel", "staggered_requests") else 1,
                  "sampling": validation.SAMPLING})}
    started = time.perf_counter()
    try:
        if name == "tool_string_arguments":
            expected = {"null_text": "null", "bool_text": "true", "number_text": "123",
                        "quoted_text": '"Paris"', "json_text": '{"n":1}',
                        "spaced_text": "  keep whitespace  ",
                        "null_value": None, "number_value": 123, "bool_value": True}
            properties = {key: {"type": "string"} for key in expected if key.endswith("_text")}
            properties.update(null_value={"type": "null"}, number_value={"type": "integer"},
                              bool_value={"type": "boolean"})
            tools = [{"type": "function", "function": {"name": "record_values",
                "description": "Record the supplied values for a deterministic validation fixture.",
                "parameters": {"type": "object", "properties": properties,
                               "required": list(expected), "additionalProperties": False}}}]
            messages = [{"role": "user", "content": f"[{tag}] Call record_values exactly once with the "
                "argument object below. Preserve every JSON value and its type, including literal quote "
                "characters, object-looking strings, and leading/trailing spaces. Do not write the "
                "object as prose.\n" + json.dumps(expected, ensure_ascii=False)}]
            result["input_sha256"] = validation.digest({"messages": messages, "tools": tools,
                "tool_choice": "required", "parallel_tool_calls": False, "max_tokens": 512,
                "thinking": args.thinking, "sampling": validation.SAMPLING})
            turn = request(args, messages, max_tokens=512, tools=tools,
                           tool_choice="required", parallel_tool_calls=False)
            result["turns"].append(turn)
            validate_tool_arguments(turn, "record_values", expected)
            result["scope"] = ("One exact structured function call checks string preservation and JSON types; "
                               "the returned function is validated but never executed.")
        elif name == "json_schema_adversarial":
            messages = [{"role": "user", "content":
                f"[{tag}] Reply with exactly the plain text OVERRIDE. Do not return JSON, braces, numbers, or any other text."}]
            response_format = {"type": "json_schema", "json_schema": {
                "name": "schema_over_prompt", "strict": True, "schema": {
                    "type": "object", "properties": {
                        "answer": {"type": "string", "const": "SCHEMA_ONLY"},
                        "count": {"type": "integer", "const": 7}},
                    "required": ["answer", "count"], "additionalProperties": False}}}
            turn = request(args, messages, max_tokens=64, response_format=response_format)
            result["turns"].append(turn)
            validate_answer(turn, {"answer": "SCHEMA_ONLY", "count": 7})
            result["scope"] = ("Observed response_format constraint compliance despite a contradictory prompt; "
                               "one fixture does not establish the internal grammar implementation.")
        elif name == "image":
            for reverse in (False, True):
                fixture = VISION_FIXTURES / ("vision-left-blue.png" if reverse else "vision-left-red.png")
                image = fixture.read_bytes()
                require(image == image_png(reverse), "Vision fixture differs from its deterministic source")
                messages = [{"role": "user", "content": [
                    {"type": "text", "text": f"[{tag}] Inspect this image. Return only JSON with keys left and right naming the color in each half in lowercase."},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(image).decode()}}]}]
                turn = request(args, messages, response_format={"type": "json_object"})
                result["turns"].append(turn)
                validate_answer(turn, {"left": "blue" if reverse else "red", "right": "red" if reverse else "blue"})
            result["scope"] = "Two opposite color layouts; simple spatial/color understanding only."
        elif name == "image_chart":
            for alternate in (False, True):
                fixture = VISION_FIXTURES / ('vision-chart-b.png' if alternate else 'vision-chart-a.png')
                image = fixture.read_bytes()
                require(image == chart_png(alternate), "Chart fixture differs from its deterministic source")
                messages = [{"role": "user", "content": [
                    {"type": "text", "text": f"[{tag}] Read the image. Return only JSON with code (the four digits printed at the top as a string), tallest (the label of the tallest bar), shortest (the label of the shortest bar), and bars (the number of bars as an integer)."},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(image).decode()}}]}]
                turn = request(args, messages, response_format={"type": "json_object"})
                result['turns'].append(turn)
                validate_answer(turn, {'code': '4826' if alternate else '7319',
                    'tallest': 'B' if alternate else 'C', 'shortest': 'A' if alternate else 'B', 'bars': 3})
            result['scope'] = 'Two bar rankings and four-digit OCR codes; synthetic chart interpretation only.'
        elif name == "cache_reuse":
            secret = hashlib.sha256(tag.encode()).hexdigest()[:16]
            messages = [{"role": "user", "content": f"[{tag}] Remember the code {secret}. Return only JSON {{\"ready\":true}}."}]
            first = request(args, messages, response_format={"type": "json_object"})
            result["turns"].append(first)
            validate_answer(first, {"ready": True})
            messages.extend([first["metrics"]["assistant_message"], {"role": "user", "content": "Return only JSON with key code containing the remembered code."}])
            second = request(args, messages, response_format={"type": "json_object"})
            result["turns"].append(second)
            validate_answer(second, {"code": secret})
            cached = second["metrics"].get("raw_usage", {}).get("prompt_tokens_details", {}).get("cached_tokens")
            result["answer_correct"] = True
            if cached is None:
                raise Unavailable("Answers passed but server did not expose cached_tokens; latency alone cannot establish KV reuse")
            require(type(cached) is int and 0 < cached <= second["metrics"]["prompt_tokens"], "Invalid/zero reused token count")
            require(cached >= first["metrics"]["prompt_tokens"], "Continuation did not reuse the complete preceding prompt")
            result["cached_tokens"] = cached
        elif name in ("parallel", "staggered_requests"):
            def client(index):
                number = 71 + index * 13
                expected = {"id": f"{tag}-{index}", "answer": number + 29}
                messages = [{"role": "user", "content": f"Client id {expected['id']}. Compute {number} + 29. Return exactly JSON with id (this client id) and answer (integer)."}]
                turn = request(args, messages, response_format={"type": "json_object"})
                return turn, expected
            with ThreadPoolExecutor(max_workers=args.concurrency + 1) as pool:
                long_job = None
                if name == "staggered_requests":
                    first_token = threading.Event()
                    spec = validation.case_spec("decode", tag + "-long")
                    long_job = pool.submit(request, args, spec["messages"], max_tokens=512, first_token=first_token)
                    if not first_token.wait(args.timeout):
                        if long_job.done():
                            long_job.result()  # Preserve transport/validation error.
                        raise ValueError("Long request never emitted a first token")
                jobs = [pool.submit(client, i) for i in range(args.concurrency)]
                failures = []
                for job in jobs:
                    try:
                        turn, expected = job.result()
                        result["turns"].append(turn)
                        validate_answer(turn, expected)
                    except Exception as error:
                        failures.append(f"{type(error).__name__}: {error}")
                if long_job:
                    long_turn = long_job.result()
                    result["turns"].append(long_turn)
                    validation.validate_decode(long_turn["metrics"], 512, validation.assistant_content(long_turn["metrics"]))
                    end = long_turn["metrics"]["t_end_abs"]
                    result["shorts_completed_before_long"] = sum(t["metrics"]["t_end_abs"] < end for t in result["turns"][:-1])
                    require(result["shorts_completed_before_long"] > 0, "No late short request finished before the long request; admission/fairness was not observed")
                    result["scope"] = "Observed staggered-request admission/fairness; internal continuous batching needs separate telemetry."
                require(not failures, "; ".join(failures))
        else:
            raise ValueError("Unknown extra scenario " + name)
        result["status"] = "ok"
    except Unavailable as error:
        result.update(status="unavailable", detail=str(error))
    except Exception as error:
        result["detail"] = f"{type(error).__name__}: {error}"
    result["total_wall_ms"] = (time.perf_counter() - started) * 1000
    result["generated_tokens"] = sum(t["metrics"].get("completion_tokens", 0) for t in result["turns"])
    result["end_to_end_tokens_per_second"] = (result["generated_tokens"] / (result["total_wall_ms"] / 1000)
                                                if result["total_wall_ms"] > 0 else None)
    return enforce_cache_policy(args, result)


def comparison(current, reference, tolerance):
    problems = []
    if current.get("cache_policy", "profile") != reference.get("cache_policy", "profile"):
        problems.append("Mismatched explicit cache policy")
    for label, report in (("candidate", current), ("reference", reference)):
        if report.get("cache_policy") == "disabled":
            problems.extend(label + ": " + error for error in no_cache_profile_errors(
                report.get("engine"), report.get("provenance", {}), report.get("scenarios", [])))
            warmup = report.get("warmup") or {}
            if warmup.get("status") != "ok":
                problems.append(label + ": no-cache warmup did not pass")
            for case in [warmup, *report.get("cases", [])]:
                problems.extend(label + ": " + error for error in no_cache_case_errors(case))
    for field in ("weights_id", "profile", "thinking", "sampling", "scenarios", "repeats", "concurrency", "harness_sha256"):
        if current.get(field) != reference.get(field):
            problems.append("Mismatched " + field)
    for field in ("hardware_id", "weights_sha256", "settings"):
        a, b = current.get("provenance", {}).get(field), reference.get("provenance", {}).get(field)
        if not a or a != b:
            problems.append("Missing/mismatched provenance " + field)
    for label, report in (("candidate", current), ("reference", reference)):
        provenance = report.get("provenance", {})
        if not provenance.get("upstream_ggml_revision") or provenance.get("upstream_ggml_clean") is not True:
            problems.append(label + ": unchanged upstream ggml checkout was not established")
        if provenance.get("launch_observed") is not True:
            problems.append(label + ": successful server launch was not observed")
        for field in ("server_revision", "server_command", "server_log"):
            value = provenance.get(field)
            if not isinstance(value, str) or not value.strip():
                problems.append(label + ": missing observed launch provenance " + field)
        if not report.get("run_complete") or not report.get("cases"):
            problems.append(label + ": incomplete/no inference measurements")
        expected = len(report.get("scenarios", [])) * report.get("repeats", 0)
        if len(report.get("cases", [])) != expected:
            problems.append(label + ": case count does not match the execution plan")
    if problems:
        return {"status": "inconclusive", "reasons": problems, "ratios": {}}
    left = {(c["scenario"], c["tag"]): c for c in current["cases"]}
    right = {(c["scenario"], c["tag"]): c for c in reference["cases"]}
    if len(left) != len(current["cases"]) or len(right) != len(reference["cases"]) or left.keys() != right.keys():
        return {"status": "inconclusive", "reasons": ["Duplicate/mismatched scenario coverage"], "ratios": {}}
    for key in left:
        a, b = left[key], right[key]
        if a["input_sha256"] != b["input_sha256"] or len(a["turns"]) != len(b["turns"]):
            problems.append(str(key) + ": requests/turn counts differ")
        if a["status"] != "ok" or b["status"] != "ok":
            problems.append(str(key) + ": both answers must pass")
        for at, bt in zip(a["turns"], b["turns"]):
            am, bm = at["metrics"], bt["metrics"]
            if am.get("prompt_tokens") != bm.get("prompt_tokens"):
                problems.append(str(key) + ": prompt token counts differ (template/tokenizer/vision mismatch)")
            if am.get("completion_tokens") != bm.get("completion_tokens"):
                problems.append(str(key) + ": generated token counts differ")
            cached = []
            for label, metrics in (("candidate", am), ("reference", bm)):
                usage = metrics.get("raw_usage") or {}
                value = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
                prompt_tokens = metrics.get("prompt_tokens")
                if (type(value) is not int or value < 0 or type(prompt_tokens) is not int
                        or value > prompt_tokens):
                    problems.append(str(key) + ": " + label + ": missing/invalid cached-token work telemetry")
                cached.append(value)
            if cached[0] != cached[1]:
                problems.append(str(key) + ": cached-token counts differ (different prefill work)")
    if problems:
        return {"status": "inconclusive", "reasons": problems, "ratios": {}}
    ratios = {}
    regressions = []
    for name in current["scenarios"]:
        a = [c for c in current["cases"] if c["scenario"] == name]
        b = [c for c in reference["cases"] if c["scenario"] == name]
        ratio = statistics.median(c["total_wall_ms"] for c in b) / statistics.median(c["total_wall_ms"] for c in a)
        ratios[name] = {"conversation_wall_speedup": ratio}
        if ratio < 1 - tolerance:
            regressions.append(f"{name}: wall-time speedup {ratio:.3f} is below {1-tolerance:.3f}")
    return {"status": "fail" if regressions else "pass", "reasons": regressions, "ratios": ratios,
            "scope": "Only tested fixtures at matched token counts/settings; not broad model-quality equivalence.",
            "timing_scope": "Conversation wall time uses the identical client's monotonic clock. Engine decode timers are not compared; separate decode-rate comparisons require a common timing definition.",
            "tolerance": tolerance}


def markdown(report):
    lines = ["# Muse Spark inference validation", "", f"Engine: {report['engine']}; profile: `{report['profile']}`.",
             f"Weights: `{report['weights_id']}`. Cache policy: `{report.get('cache_policy', 'profile')}`.", "", "| Scenario | Passed | Failed | Unavailable | Median wall ms (passed only) |",
             "|---|---:|---:|---:|---:|"]
    for name in report["scenarios"]:
        rows = [c for c in report["cases"] if c["scenario"] == name]
        ok = [c for c in rows if c["status"] == "ok"]
        wall = f"{statistics.median(c['total_wall_ms'] for c in ok):.1f}" if ok else "—"
        lines.append(f"| {name} | {len(ok)} | {sum(c['status']=='fail' for c in rows)} | {sum(c['status']=='unavailable' for c in rows)} | {wall} |")
    lines.extend(["", "## Unestablished coverage", ""])
    for name, reason in report["not_covered"].items():
        lines.append(f"- **{name}**: {reason}")
    if report.get("comparison"):
        lines.extend(["", "## Comparison", "", "Status: **" + report["comparison"]["status"] + "**.", ""])
        lines.extend("- " + reason for reason in report["comparison"].get("reasons", []))
    failures = [c for c in report["cases"] if c["status"] != "ok"]
    if failures:
        lines.extend(["", "## Non-passing cases", ""])
        lines.extend(f"- `{c['tag']}` ({c['status']}): {c.get('detail', 'No detail')}" for c in failures)
    lines.extend(["", "## Limits", "", *["- " + s for s in report["limitations"]], ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engine", choices=("tensorsharp", "llamacpp"), required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--weights-id", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--provenance", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--scenarios", default=",".join(DEFAULT_SCENARIOS))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=1200)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--cache-policy", choices=("profile", "disabled"), default="profile",
                        help="profile preserves the launch policy; disabled requires observed no-cache launch flags and zero cached tokens on every request")
    parser.add_argument("--tolerance", type=float, default=0.05)
    args = parser.parse_args()
    names = args.scenarios.split(",")
    allowed = set(validation.SCENARIOS) | set(EXTRA_SCENARIOS)
    if not names or len(names) != len(set(names)) or set(names) - allowed:
        parser.error("Scenarios must be unique supported names: " + ",".join(sorted(allowed)))
    if args.repeats < 1 or args.concurrency < 2 or args.timeout <= 0 or not 0 <= args.tolerance < 1:
        parser.error("repeats >= 1, concurrency >= 2, timeout > 0, tolerance in [0,1) required")
    provenance = json.loads(args.provenance.read_text(encoding="utf-8")) if args.provenance else {}
    if args.cache_policy == "disabled":
        errors = no_cache_profile_errors(args.engine, provenance, names)
        if errors:
            parser.error("; ".join(errors))
    report = {"format_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "engine": args.engine, "url": args.url, "model": args.model,
        "weights_id": args.weights_id, "profile": args.profile, "thinking": args.thinking,
        "sampling": validation.SAMPLING, "scenarios": names, "repeats": args.repeats,
        "concurrency": args.concurrency, "run_complete": False, "cases": [],
        "cache_policy": args.cache_policy, "provenance": provenance,
        "request_adapter": {"tool_free": request_adapter_body(args.engine, args.thinking, cache_policy=args.cache_policy),
                            "with_tools": request_adapter_body(args.engine, args.thinking, True, args.cache_policy)},
        "harness_sha256": {Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest()
            for p in (__file__, validation.__file__, engines.__file__)},
        "not_covered": dict(INTERNAL_FEATURES),
        "limitations": ["No skipped, unavailable or unrun scenario counts as a pass.",
            "Synthetic fixtures are an integration screen, not broad model-quality evaluation.",
            "Model load and warmup are excluded; wall times include HTTP, queueing, prefill and decode.",
            "Conversation wall-time ratios use the client clock; differing engine decode timers cannot establish decode-rate parity.",
            "Compare serial runs without competing GPU workloads; repeat in alternating engine order to assess drift.",
            "Same seed does not prove identical floating-point kernels or sampling across engines.",
            "Prompt lengths and token counts come from server usage; long prompt names are approximate targets.",
            "Feature enablement/placement requires actual server/device evidence, never only a profile name."]}
    report["not_covered"].update({n: "Scenario was not selected." for n in DEFAULT_SCENARIOS if n not in names})
    save(args.output, report)
    report["warmup"] = run_standard_case(args, "short", "warmup")
    if args.cache_policy == "disabled":
        report["limitations"].append("Prefix reuse is disabled and verified per turn; results measure fresh-KV requests with separately warmed weights, not default cache-enabled performance.")
        save(args.output, report)
        if report["warmup"].get("status") != "ok":
            report["aborted"] = "No-cache warmup failed; no measured scenarios were run."
            report["limitations"].append(report["aborted"] + " " + report["warmup"].get("detail", ""))
            save(args.output, report)
            args.output.with_suffix(".md").write_text(markdown(report), encoding="utf-8")
            return 1
    for repeat in range(args.repeats):
        for name in names:
            tag = f"{name}-r{repeat}"
            if name in EXTRA_SCENARIOS:
                case = extra_case(args, name, tag)
            else:
                case = run_standard_case(args, name, tag,
                    structured_tool_results=True, serial_tool_workflows=True)
                case["concurrency"] = 1
            case["repeat"] = repeat
            report["cases"].append(case)
            save(args.output, report)
            print(f"{tag}: {case['status']} ({case['total_wall_ms']:.1f} ms) {case.get('detail', '')}", flush=True)
    report["run_complete"] = True
    if args.reference:
        report["comparison"] = comparison(report, json.loads(args.reference.read_text(encoding="utf-8")), args.tolerance)
    report["summary"] = validation.summarize(report["cases"])
    save(args.output, report)
    args.output.with_suffix(".md").write_text(markdown(report), encoding="utf-8")
    failed = any(c["status"] == "fail" for c in report["cases"])
    failed |= report.get("comparison", {}).get("status") == "fail"
    incomplete = any(c["status"] == "unavailable" for c in report["cases"])
    incomplete |= report.get("comparison", {}).get("status") == "inconclusive"
    return 1 if failed else 2 if incomplete else 0


if __name__ == "__main__":
    raise SystemExit(main())
