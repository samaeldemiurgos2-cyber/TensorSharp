#!/usr/bin/env python3
"""Validate Studio in an isolated, running TensorAgent Mac Catalyst app.

Launch with TENSORAGENT_VALIDATION_ROOT and the pinned LTX catalog files linked
into Cache/models/ltx-2.5-distilled-q4kxl. The private connection file is never
copied into the report. Requires installed ffmpeg/ffprobe; no model doubles.
Evidence must be written below ignored artifacts/ or docs/validation/.
Use --image-conditioning to run image-to-video with the imported synthetic frame.
"""
import argparse
import array
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
import uuid


def run(*args):
    return subprocess.run(args, check=True, capture_output=True).stdout


def inspect_video(path, frames):
    metadata = json.loads(run("ffprobe", "-v", "error", "-count_frames", "-show_streams",
                              "-show_format", "-of", "json", str(path)))
    video = next(s for s in metadata["streams"] if s["codec_type"] == "video")
    audio = next(s for s in metadata["streams"] if s["codec_type"] == "audio")
    if (video["codec_name"], video["width"], video["height"], int(video["nb_read_frames"])) != ("h264", 1024, 1024, frames):
        raise RuntimeError("Unexpected decoded video geometry/frame count: " + str(video))
    if audio["codec_name"] != "aac" or int(audio["sample_rate"]) != 48000 or audio["channels"] != 2:
        raise RuntimeError("Expected 48 kHz stereo AAC: " + str(audio))
    if abs(float(metadata["format"]["duration"]) - frames / 24) > 2 / 24 + .05:
        raise RuntimeError("Unexpected output duration")
    run("ffmpeg", "-v", "error", "-xerror", "-i", str(path), "-f", "null", "-")
    pixels = run("ffmpeg", "-v", "error", "-i", str(path), "-vf", "scale=32:32", "-pix_fmt", "rgb24", "-f", "rawvideo", "-")
    samples = array.array("f", run("ffmpeg", "-v", "error", "-i", str(path), "-vn", "-f", "f32le", "-"))
    if sys.byteorder != "little":
        samples.byteswap()
    if not samples or not all(math.isfinite(s) for s in samples):
        raise RuntimeError("Empty or nonfinite decoded audio")
    rms = math.sqrt(sum(s * s for s in samples) / len(samples))
    if rms <= 1e-6 or max(pixels) - min(pixels) <= 8:
        raise RuntimeError("Silent audio or blank video")
    return {"video": video, "audio": audio, "duration": metadata["format"]["duration"],
            "audio_rms": rms, "pixel_range": [min(pixels), max(pixels)],
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--connection", type=Path, required=True)
    parser.add_argument("--app", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--frames", type=int, default=17)
    parser.add_argument("--image-conditioning", action="store_true", help="Animate the imported synthetic image as the first frame.")
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    if not any(output.is_relative_to(repository / folder) for folder in ("artifacts", "docs/validation")):
        raise ValueError("--output must be inside artifacts/ or docs/validation/")
    if output.exists() or args.repeats < 1 or args.frames < 9 or (args.frames - 1) % 8:
        raise ValueError("Use a fresh output directory, positive repeats, and 8n+1 frames")
    output.mkdir(parents=True)
    connection = json.loads(args.connection.read_text())
    base = connection["baseUrl"].rstrip("/")
    if urlsplit(base).hostname not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError("Only an isolated loopback app is supported")
    report = {"completed": False, "process_id": connection["processId"], "app": str(args.app.resolve()),
              "scope": "Actual packaged Mac Catalyst app: native model Use, Studio capabilities, media import, joint LTX generation, thumbnail/waveform, timeline edit and H.264/AAC export.",
              "limitations": "Short single-prompt functional samples; warmed filesystem caches. Pixel/audio sanity and full decode do not establish perceptual quality or long-video performance. No Windows/CUDA validation.",
              "executions": []}

    def save():
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    def request(route, body=None, content_type="application/json"):
        data = json.dumps(body).encode() if body is not None and not isinstance(body, bytes) else body
        with urlopen(Request(base + route, data=data, headers={"Cookie": connection["cookie"],
                     "X-TensorSharp-Studio": "1", "Content-Type": content_type}), timeout=120) as response:
            return response.read()

    def api(action, payload=None):
        return json.loads(request("/api/studio", {"action": action, "payload": payload or {}}))

    def wait_job(project_id, job):
        terminal = ("completed", "failed", "cancelled", "interrupted")

        def current_job():
            return next(j for j in api("jobs/list", {"projectId": project_id}) if j["id"] == job["id"])

        try:
            deadline = time.monotonic() + 1800
            while time.monotonic() < deadline:
                current = current_job()
                if current["state"] == "completed":
                    return current
                if current["state"] in terminal:
                    raise RuntimeError("Studio job failed: " + str(current))
                time.sleep(1)
            raise TimeoutError("Studio job timed out")
        except BaseException:
            # Do not abandon native inference or encoding when the harness fails or is interrupted.
            try:
                api("jobs/cancel", {"jobId": job["id"]})
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline:
                    if current_job()["state"] in terminal:
                        break
                    time.sleep(.25)
                else:
                    report["cleanup_warning"] = "Cancellation requested; job did not join within 30 seconds: " + job["id"]
            except Exception as cleanup_error:
                report["cleanup_warning"] = str(cleanup_error)
            raise

    save()
    try:
        report["ffmpeg"] = run("ffmpeg", "-version").decode().splitlines()[0]
        report["ggml_revision"] = run("git", "-C", str(repository / "ExternalProjects/ggml"), "rev-parse", "HEAD").decode().strip()
        report["ggml_clean"] = not run("git", "-C", str(repository / "ExternalProjects/ggml"), "status", "--porcelain").strip()
        binaries = args.app / "Contents/MonoBundle"
        report["binary_sha256"] = {name: hashlib.sha256((binaries / name).read_bytes()).hexdigest()
                                   for name in ("TensorSharp.Studio.dll", "TensorSharp.Chat.dll", "TensorSharp.Models.dll", "TensorAgent.Core.dll", "libmlxc.dylib")}
        report["native_use"] = json.loads(request("/api/agent/validation/tap-model", {"id": "ltx-2.5-distilled-q4kxl"}))
        if not report["native_use"].get("ok"):
            raise RuntimeError("Native model Use was rejected")
        deadline = time.monotonic() + 180
        while True:
            info = json.loads(request("/api/models"))
            if info.get("architecture") == "ltx-2.5":
                report["model_info"] = info
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("LTX model load timed out")
            time.sleep(.5)
        capabilities = api("capabilities")
        report["capabilities"] = capabilities
        if not all(capabilities.get(k) for k in ("externalProcesses", "importAvailable", "renderAvailable", "generationAvailable")):
            raise RuntimeError("Studio capabilities remain gated: " + str(capabilities))
        model = next(m for m in capabilities["models"] if m["available"] and m["family"] == "ltx-2.5")
        image = output / "import.png"
        run("ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=1024x1024", "-frames:v", "1", str(image))
        for repeat in range(args.repeats):
            project = api("projects/create", {"name": "Mac Studio validation", "width": 1024, "height": 1024})
            project = json.loads(request(f"/api/studio/projects/{project['id']}/import?fileName=import.png", image.read_bytes(), "image/png"))
            if len(project["assets"]) != 1 or project["assets"][0]["width"] != 1024:
                raise RuntimeError("Studio image import did not preserve geometry")
            execution = {"repeat": repeat, "project_id": project["id"], "imported_asset": project["assets"][0]}
            report["executions"].append(execution)
            save()
            before = time.monotonic()
            generation = {"projectId": project["id"], "modelId": model["modelId"],
                      "requestId": uuid.uuid4().hex, "prompt": "A red toy car drives slowly along a quiet road in warm afternoon light.",
                      "width": 1024, "height": 1024, "frameCount": args.frames, "fps": 24,
                      "steps": 8, "cfgScale": 1, "seed": 42, "generateAudio": True}
            if args.image_conditioning:
                generation.update(mode="i2v", conditioningAssetId=project["assets"][0]["id"])
            execution["generation_request"] = generation
            job = api("jobs/generate", generation)
            execution["generation_job"] = wait_job(project["id"], job)
            execution["whole_generation_job_seconds"] = time.monotonic() - before
            project = api("projects/get", {"projectId": project["id"]})
            asset = next(a for a in project["assets"] if a.get("generation"))
            execution["generated_asset"] = asset
            video = output / f"take-{repeat}.mp4"
            video.write_bytes(request(f"/api/studio/projects/{project['id']}/assets/{asset['id']}"))
            execution["take_validation"] = inspect_video(video, args.frames)
            thumbnail = request(f"/api/studio/thumbnail/{project['id']}/{asset['id']}")
            if not thumbnail.startswith(b"\xff\xd8"):
                raise RuntimeError("Invalid generated thumbnail")
            (output / f"take-{repeat}.jpg").write_bytes(thumbnail)
            waveform = json.loads(request(f"/api/studio/waveform/{project['id']}/{asset['id']}"))
            if len(waveform["peaks"]) != 512 or max(waveform["peaks"]) <= 0:
                raise RuntimeError("Invalid generated waveform")
            execution["waveform_max"] = max(waveform["peaks"])
            project = api("commands/apply", {"projectId": project["id"], "expectedRevision": project["revision"],
                          "commandId": uuid.uuid4().hex, "commands": [{"type": "insertClip", "trackId": project["tracks"][0]["id"],
                          "assetId": asset["id"], "durationFrames": args.frames}]})
            before = time.monotonic()
            job = api("jobs/render", {"projectId": project["id"], "expectedRevision": project["revision"], "requestId": uuid.uuid4().hex})
            execution["render_job"] = wait_job(project["id"], job)
            execution["whole_render_job_seconds"] = time.monotonic() - before
            rendered = output / f"timeline-{repeat}.mp4"
            rendered.write_bytes(request(f"/api/studio/projects/{project['id']}/jobs/{job['id']}/output"))
            execution["export_validation"] = inspect_video(rendered, args.frames)
            execution["completed"] = True
            save()
            print(json.dumps({"repeat": repeat, "generation_seconds": execution["whole_generation_job_seconds"],
                              "render_seconds": execution["whole_render_job_seconds"]}), flush=True)
        report["completed"] = True
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
