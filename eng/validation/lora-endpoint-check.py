#!/usr/bin/env python3
"""Check /v1/loras confinement on a live TensorSharp.Server.Host (PR #282 review item 5).

Two steps, against a server you start yourself:

  # 1. Fixture adapters, broken files, files outside the directory and (where the OS
  #    allows) links that leave it. Siblings <dir>-outside and <dir>-evil are created too.
  python3 eng/validation/lora-endpoint-check.py fixtures --dir /tmp/loras

  # 2a. Server started WITHOUT --loras-dir: POST answers 403 and nothing loads.
  python3 eng/validation/lora-endpoint-check.py check --url http://127.0.0.1:5000 \
      --dir /tmp/loras --disabled --output artifacts/lora-endpoints/disabled.json

  # 2b. Server started with --loras-dir /tmp/loras. --expect load: no model, or one that
  #     applies a 1x1 adapter on --target; --expect refusal: a model family that applies
  #     no adapters (every POST of a valid file is a 400 naming the reason).
  python3 eng/validation/lora-endpoint-check.py check --url http://127.0.0.1:5000 \
      --dir /tmp/loras --expect load --chat-model qwen.gguf --output artifacts/lora-endpoints/enabled.json

Checks: every path that does not resolve inside the directory (missing, outside, '..',
sibling prefix, a directory, wrong extension, rooted/drive/UNC/stream paths, link escapes)
gets one byte-identical 400; broken files get a 400 that names the reason and no server
path; malformed bodies get the OpenAI error shape; load/list/delete/404 and the 32-adapter
409 limit (with --expect load); and, with --chat-model, that greedy base output is the same
before and after the LoRA calls and that a bad "lora" is refused before any stream output.
Exit status 1 on any failed check. This covers the endpoints only: it does not show that an
adapter changes what a model generates.
"""

import argparse
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
from urllib.error import HTTPError
from urllib.request import Request, urlopen

TARGET = "blk.0.attn_output.weight"
CAPACITY = 32


def write_safetensors(path, header, data):
    raw = json.dumps(header, separators=(",", ":")).encode() if isinstance(header, dict) else header.encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + data)


def write_adapter(path, target=TARGET, dtype="F32", size=4):
    header = {f"{target}.lora_a": {"dtype": dtype, "shape": [1, 1], "data_offsets": [0, size]},
              f"{target}.lora_b": {"dtype": dtype, "shape": [1, 1], "data_offsets": [size, 2 * size]}}
    data = struct.pack("<2f", 1.0, 1.0) if dtype == "F32" else bytes(2 * size)
    write_safetensors(path, header, data)


def fixtures(args):
    root = Path(args.dir).resolve()
    outside, evil = Path(f"{root}-outside"), Path(f"{root}-evil")
    for d in (root / "sub", outside, evil):
        d.mkdir(parents=True, exist_ok=True)
    write_adapter(root / "good.safetensors", args.target)
    write_adapter(root / "sub" / "adapter_model.safetensors", args.target)
    (root / "sub" / "adapter_config.json").write_text('{"r": 1, "lora_alpha": 4}')
    write_adapter(outside / "secret.safetensors", args.target)
    write_adapter(evil / "good.safetensors", args.target)
    (root / "truncated.safetensors").write_bytes(b"\x01\x02\x03\x04")
    write_safetensors(root / "header-not-json.safetensors", '{"w.lora_a": oops', bytes(8))
    write_adapter(root / "f8.safetensors", "w", "F8_E4M3", 1)
    write_adapter(root / "i32.safetensors", "w", "I32", 4)
    write_adapter(root / "q4k.safetensors", "w", "Q4_K", 1)
    write_safetensors(root / "b-only.safetensors", {"w.lora_b": {"dtype": "F32", "shape": [1, 1], "data_offsets": [0, 4]}}, bytes(4))
    (root / "stub.gguf").write_bytes(b"GGUF" + bytes(60))
    links = []
    for link, target, is_dir in ((root / "leak.safetensors", outside / "secret.safetensors", False),
                                 (root / "escape", outside, True),
                                 (root / "passwd.safetensors", Path("/etc/passwd"), False),
                                 (root / "etc", Path("/etc"), True)):
        if not target.exists() or link.exists() or link.is_symlink():
            continue
        try:
            os.symlink(target, link, target_is_directory=is_dir)
            links.append(link.name)
        except OSError as ex:
            # Windows without Developer Mode: a junction needs no privilege, and the
            # server's guard has to follow it just the same.
            if is_dir and os.name == "nt" and subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)],
                                                             capture_output=True).returncode == 0:
                links.append(link.name)
            else:
                print(f"link {link.name} not created: {ex}", file=sys.stderr)
    print(json.dumps({"dir": str(root), "outside": str(outside), "evil": str(evil), "links": links}, indent=2))


class Server:
    def __init__(self, url, timeout):
        self.url, self.timeout = url.rstrip("/"), timeout

    def call(self, method, route, body=None, raw=None):
        data = raw if raw is not None else (None if body is None else json.dumps(body).encode())
        request = Request(self.url + route, data=data, method=method,
                          headers={"Content-Type": "application/json"} if data is not None else {})
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return response.status, response.read().decode()
        except HTTPError as error:
            return error.code, error.read().decode()


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)


def leaks(body, paths):
    texts = [body] + list(strings(json.loads(body))) if body.strip().startswith(("{", "[")) else [body]
    forms = {f for p in paths for f in (p, p.replace("\\", "/"), p.replace("/", "\\"))}
    return sorted({f for f in forms for t in texts if f.lower() in t.lower()})


def check(args):
    root = Path(args.dir).resolve()
    outside, evil = f"{root}-outside", f"{root}-evil"
    server = Server(args.url, args.timeout)
    results = []

    def record(name, ok, detail=None):
        results.append({"check": name, "ok": bool(ok), "detail": detail})
        print(("PASS " if ok else "FAIL ") + name + ("" if ok or detail is None else f": {detail}"))

    def error_of(body):
        try:
            return json.loads(body).get("error", {})
        except ValueError:
            return {}

    status, body = server.call("GET", "/v1/loras")
    record("GET /v1/loras answers 200", status == 200, body[:300])
    initial = [a["id"] for a in json.loads(body).get("data", [])] if status == 200 else []

    if args.disabled:
        status, body = server.call("POST", "/v1/loras", {"name": "e2e-a", "path": str(root / "good.safetensors")})
        err = error_of(body)
        record("POST without --loras-dir is 403 lora_loading_disabled",
               status == 403 and err.get("code") == "lora_loading_disabled" and "--loras-dir" in err.get("message", ""), body)
        record("nothing is loaded without --loras-dir", initial == [], initial)
    else:
        record("startup load: good is loaded" if args.expect == "load" else "startup load: the model refused every file",
               ("good" in initial) if args.expect == "load" else initial == [], initial)

        bad_paths = ["missing.safetensors", os.path.join(outside, "secret.safetensors"), os.path.join(outside, "missing.safetensors"),
                     f"../{Path(outside).name}/secret.safetensors", os.path.join(evil, "good.safetensors"), "sub",
                     "sub/adapter_config.json", "/etc/passwd", "../../../../../../etc/passwd", "/proc/self/environ",
                     "C:\\Windows\\win.ini", "\\\\server\\share\\x.safetensors", "good.safetensors:s", "good.safetensors."]
        bad_paths += [p for p in ("leak.safetensors", "escape/secret.safetensors", "passwd.safetensors", "etc/passwd")
                      if os.path.lexists(root / p.split("/")[0])]
        bodies = {}
        for path in bad_paths:
            status, body = server.call("POST", "/v1/loras", {"name": "e2e-x", "path": path})
            bodies[path] = (status, body)
        distinct = {b for _, b in bodies.values()}
        record(f"{len(bad_paths)} unresolvable paths all answer 400", all(s == 400 for s, _ in bodies.values()),
               {p: s for p, (s, _) in bodies.items() if s != 400})
        record("unresolvable paths share one byte-identical body", len(distinct) == 1, sorted(distinct))
        sample = next(iter(distinct))
        record("that body names no server path and says nothing about existence",
               not leaks(sample, [str(root), outside, evil]) and "not found" not in sample.lower() and "bytes" not in sample,
               sample)

        corrupt = {"truncated.safetensors": "truncated", "header-not-json.safetensors": "not valid JSON",
                   "f8.safetensors": "LoRA factors must be", "i32.safetensors": "LoRA factors must be",
                   "q4k.safetensors": "Q4_K", "b-only.safetensors": "lora_a", "stub.gguf": "GGUF adapters are not supported yet"}
        for name, reason in corrupt.items():
            status, body = server.call("POST", "/v1/loras", {"name": "e2e-bad", "path": name})
            message = error_of(body).get("message", "")
            record(f"broken {name}: 400 naming the reason, no server path",
                   status == 400 and message.startswith(f"LoRA adapter '{name}' could not be loaded: ") and reason in message
                   and "Could not read the request body" not in body and not leaks(body, [str(root)]), body)

        for raw, param in ((b"[]", None), (b"not json", None), (b"{}", "name"),
                           (b'{"name":"none","path":"good.safetensors"}', "name"),
                           (b'{"name":"a","path":"good.safetensors","scale":1e60}', "scale")):
            status, body = server.call("POST", "/v1/loras", raw=raw)
            err = error_of(body)
            record(f"malformed body {raw.decode()}: 400 in the OpenAI shape",
                   status == 400 and err.get("type") == "invalid_request_error" and err.get("param") == param
                   and all(k in err for k in ("message", "param", "code")), body)

        status, body = server.call("POST", "/v1/loras", {"name": "e2e-a", "path": "good.safetensors"})
        if args.expect == "load":
            record("POST of a valid relative path loads it", status == 200 and json.loads(body).get("id") == "e2e-a", body)
            status, body = server.call("POST", "/v1/loras", {"name": "e2e-cfg", "path": "sub/adapter_model.safetensors"})
            record("sibling adapter_config.json is read (alpha 4)", status == 200 and json.loads(body).get("alpha") == 4, body)
            status, body = server.call("GET", "/v1/loras")
            ids = [a["id"] for a in json.loads(body)["data"]]
            record("GET lists the loaded adapters", {"e2e-a", "e2e-cfg"} <= set(ids), ids)
            loaded, conflict, i = len(ids), None, 0
            while conflict is None and i < CAPACITY + 2:
                status, body = server.call("POST", "/v1/loras", {"name": f"e2e-fill{i}", "path": "good.safetensors"})
                if status == 409:
                    conflict = body
                elif status == 200:
                    loaded += 1
                i += 1
            record(f"the {CAPACITY + 1}st adapter is a 409 lora_limit_reached",
                   conflict is not None and loaded == CAPACITY and error_of(conflict).get("code") == "lora_limit_reached",
                   {"held": loaded, "body": conflict})
            status, _ = server.call("POST", "/v1/loras", {"name": "e2e-a", "path": "good.safetensors"})
            record("a reload is still allowed at the limit", status == 200, status)
            for name in [n for n in ids + [f"e2e-fill{k}" for k in range(i)] if n.startswith("e2e-")]:
                server.call("DELETE", f"/v1/loras/{name}")
            status, body = server.call("DELETE", "/v1/loras/e2e-a")
            record("DELETE of an unloaded name is 404 lora_not_found",
                   status == 404 and error_of(body).get("code") == "lora_not_found", body)
        else:
            message = error_of(body).get("message", "")
            record("POST of a valid file the model cannot apply is a 400 naming the reason",
                   status == 400 and message.startswith("LoRA adapter 'good.safetensors' could not be loaded: ")
                   and not leaks(body, [str(root)]), body)
        status, body = server.call("GET", "/v1/loras")
        record("the e2e adapters are gone again", sorted(a["id"] for a in json.loads(body)["data"]) == sorted(initial), body)

    if args.chat_model:
        def chat(lora=None, stream=False):
            request = {"model": args.chat_model, "messages": [{"role": "user", "content": "Count from one to five."}],
                       "max_tokens": 24, "temperature": 0, "stream": stream}
            if lora is not None:
                request["lora"] = lora
            return server.call("POST", "/v1/chat/completions", request)

        status, body = chat()
        record("base chat answers 200", status == 200, body[:300])
        text = json.loads(body)["choices"][0]["message"]["content"] if status == 200 else None
        for stream in (False, True):
            status, body = chat("e2e-missing", stream)
            record(f"unknown lora (stream={stream}) is a 400 before any output",
                   status == 400 and "data:" not in body and error_of(body).get("param") == "lora", body[:300])
        status, body = chat("none")
        record('"lora": "none" is the base model', status == 200 and json.loads(body)["choices"][0]["message"]["content"] == text,
               body[:300])
        status, body = chat()
        record("greedy base output is unchanged after the LoRA calls",
               status == 200 and json.loads(body)["choices"][0]["message"]["content"] == text, body[:300])

    failed = [r for r in results if not r["ok"]]
    report = {"url": args.url, "dir": str(root), "mode": "disabled" if args.disabled else args.expect,
              "chat_model": args.chat_model, "passed": len(results) - len(failed), "failed": len(failed), "checks": results}
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2))
    print(f"{len(results) - len(failed)} passed, {len(failed)} failed")
    return 1 if failed else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("fixtures", help="write the fixture directory")
    make.add_argument("--dir", required=True)
    make.add_argument("--target", default=TARGET, help="weight name the valid adapters patch (1x1 factors)")
    run = sub.add_parser("check", help="run the checks against a live server")
    run.add_argument("--url", required=True)
    run.add_argument("--dir", required=True, help="the fixture directory (the server's --loras-dir, unless --disabled)")
    run.add_argument("--disabled", action="store_true", help="the server was started without --loras-dir")
    run.add_argument("--expect", choices=("load", "refusal"), default="load")
    run.add_argument("--chat-model", help="model name for the greedy chat regression checks")
    run.add_argument("--timeout", type=float, default=300)
    run.add_argument("--output", help="JSON report path (keep it under the ignored artifacts/)")
    args = parser.parse_args()
    if args.command == "fixtures":
        fixtures(args)
        return 0
    return check(args)


if __name__ == "__main__":
    sys.exit(main())
