#!/usr/bin/env python3
"""Exercise an isolated actual MAUI app's native Use action and LTX HTTP jobs.

Launch the app with TENSORAGENT_VALIDATION_ROOT containing the five pinned cache
assets, then pass its private connection.json. This tool never exports the cookie.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import time
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--connection", type=Path, required=True)
    parser.add_argument("--app", type=Path, required=True, help="Actual app bundle for binary provenance.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--chat-turn", action="store_true", help="Also exercise the normal chat UI route with model defaults.")
    args = parser.parse_args()
    if args.output.exists() or args.repeats < 1:
        raise ValueError("Use a fresh report path and positive repeat count.")
    connection = json.loads(args.connection.read_text())
    base = connection["baseUrl"].rstrip("/")
    if urlsplit(base).hostname not in ("localhost", "127.0.0.1", "::1"):
        raise ValueError("Only the isolated loopback app is supported.")
    def open_route(route, body=None, timeout=1800):
        data = json.dumps(body).encode() if body is not None else None
        return urlopen(Request(base + route, data=data, headers={"Cookie": connection["cookie"],
            "Content-Type": "application/json"}), timeout=timeout)
    def read_json(route, body=None):
        with open_route(route, body, timeout=30) as response:
            return json.load(response)
    report = {"completed": False, "scope": "Actual packaged Mac Catalyst app; native Models page action, real HTTP/SSE full AV generation, AVFoundation MP4 export and downloaded WAV. No substitute model/runtime.",
              "limitations": "Short single-prompt functional/performance samples; filesystem caches may already be warm. Inference, HTTP/export and app startup are different timing scopes.",
              "process_id": connection["processId"], "executions": []}
    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    save()
    try:
        binaries = args.app / "Contents/MonoBundle"
        report["binaries"] = {name: sha(binaries / name) for name in
            ("TensorSharp.Models.dll", "TensorSharp.Backends.MLX.dll", "TensorAgent.Core.dll", "libmlxc.dylib", "libmlx.dylib", "mlx.metallib")}
        report["native_use_action"] = read_json("/api/agent/validation/tap-model", {"id": "ltx-2.5-distilled-q4kxl"})
        if not report["native_use_action"].get("ok"):
            raise RuntimeError("The native Models page did not accept Use: " + str(report["native_use_action"]))
        deadline = time.monotonic() + 120
        while True:
            info = read_json("/api/models")
            if info.get("architecture") == "ltx-2.5":
                if info.get("loadedBackend") != "ggml_metal":
                    raise RuntimeError("The actual app did not use its native GPU backend: " + str(info.get("loadedBackend")))
                report["model_info"] = info
                break
            if time.monotonic() > deadline:
                raise RuntimeError("The actual app did not activate LTX.")
            time.sleep(.5)
        body = {"prompt": "A red toy car drives slowly along a quiet road in warm afternoon light.",
                "width": 1024, "height": 1024, "frames": 17, "fps": 24, "steps": 8,
                "cfg": 1, "sampler": "euler_ancestral", "flowShift": 0, "seed": 42, "generateAudio": True}
        report["request"] = body
        save()
        for repeat in range(args.repeats):
            before, events, terminal = time.monotonic(), [], None
            execution = {"repeat": repeat, "completed": False, "events": events}
            report["executions"].append(execution)
            save()
            with open_route("/api/video-generate/stream", body) as response:
                if response.headers.get_content_type() != "text/event-stream":
                    raise RuntimeError("Expected actual SSE response.")
                for line in response:
                    if not line.startswith(b"data: "):
                        continue
                    event = json.loads(line[6:])
                    events.append(event)
                    if event.get("error"):
                        raise RuntimeError(str(event["error"]))
                    if event.get("done"):
                        terminal = event
            execution["whole_http_job_ms"] = (time.monotonic() - before) * 1000
            if terminal is None or any(terminal.get(key) != value for key, value in
                {"width": 1024, "height": 1024, "frames": 17, "fps": 24, "seed": 42, "codec": "h264", "audioMuxed": True}.items()):
                raise RuntimeError("Invalid generated AV result: " + str(terminal))
            execution["generation_ms"] = next(e["elapsedSeconds"] * 1000 for e in reversed(events) if e.get("phase") == "done")
            execution["terminal"] = terminal
            for role, key, suffix in (("video", "url", ".mp4"), ("audio", "audioUrl", ".wav")):
                path = Path(str(args.output) + f".{repeat}" + suffix)
                with open_route(terminal[key]) as response:
                    path.write_bytes(response.read())
                execution[role] = {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}
            wav = Path(execution["audio"]["path"]).read_bytes()
            if len(wav) != 44 + 33120 * 4 or struct.unpack_from("<I", wav, 24)[0] != 48000 or struct.unpack_from("<H", wav, 22)[0] != 2:
                raise RuntimeError("Generated sidecar is not the original-length 48k stereo WAV.")
            execution["completed"] = True
            save()
        if args.chat_turn:
            session = read_json("/api/sessions", {})
            chat_body = {"sessionId": session["sessionId"], "messages": [
                {"role": "user", "content": body["prompt"]}]}
            chat = {"completed": False, "scope": "Normal user chat route with the catalog model's recipe and a random seed; excluded from matched benchmark samples.",
                    "request": chat_body, "events": []}
            report["chat_turn"] = chat
            save()
            before, video, done = time.monotonic(), None, False
            with open_route("/api/chat", chat_body) as response:
                if response.headers.get_content_type() != "text/event-stream":
                    raise RuntimeError("Normal chat route did not return SSE.")
                for line in response:
                    if not line.startswith(b"data: "):
                        continue
                    event = json.loads(line[6:])
                    chat["events"].append(event)
                    if event.get("error"):
                        raise RuntimeError("Normal chat failed: " + str(event["error"]))
                    if event.get("videoUrl"):
                        video = event
                    if event.get("done"):
                        done = True
                        break
            chat["whole_http_job_ms"] = (time.monotonic() - before) * 1000
            if not done or video is None or any(video.get(key) != value for key, value in
                {"width": 1024, "height": 1024, "frames": 17, "fps": 24, "hasAudio": True}.items()):
                raise RuntimeError("Normal chat did not produce the expected AV preview: " + str(video))
            chat["video_result"] = video
            if not video["videoUrl"].endswith(".mp4"):
                raise RuntimeError("Expected normal chat MP4 route.")
            for role, route, suffix in (("video", video["videoUrl"], ".mp4"),
                ("audio", video["videoUrl"][:-4] + ".wav", ".wav")):
                path = Path(str(args.output) + ".chat" + suffix)
                with open_route(route) as response:
                    path.write_bytes(response.read())
                chat[role] = {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}
            wav = Path(chat["audio"]["path"]).read_bytes()
            if len(wav) != 44 + 33120 * 4 or struct.unpack_from("<I", wav, 24)[0] != 48000 or struct.unpack_from("<H", wav, 22)[0] != 2:
                raise RuntimeError("Normal chat produced an invalid audio sidecar.")
            deadline = time.monotonic() + 10
            while True:
                conversation = read_json("/api/agent/conversations/" + session["conversationId"])
                if any(message.get("role") == "assistant" and message.get("videoUrl") == video["videoUrl"]
                    for message in conversation.get("messages", [])):
                    chat["saved_assistant_video_url"] = True
                    break
                if time.monotonic() > deadline:
                    raise RuntimeError("Normal chat did not persist the generated clip.")
                time.sleep(.2)
            chat["completed"] = True
            save()
        report["completed"] = True
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        save()
    print(json.dumps({"completed": True, "generation_ms": [e["generation_ms"] for e in report["executions"]]}))


if __name__ == "__main__":
    main()
