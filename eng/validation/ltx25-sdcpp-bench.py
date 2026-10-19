#!/usr/bin/env python3
"""Benchmark unchanged SD.cpp LTX-2.5 and validate decoded audiovisual output.

Requires a source export manifest derived from a verified clean checkout and its
build manifest. Whole-process timings include loading and codec work. Structural
media checks do not establish perceptual quality or TensorSharp parity.
"""
import argparse
import array
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_source(root, manifest):
    rows = json.loads(manifest.read_text())
    mismatches = []
    for item in rows:
        path = root / item["path"]
        if not path.is_file() or path.stat().st_size != item["bytes"] or sha(path) != item["sha256"]:
            mismatches.append(item["path"])
    if mismatches:
        raise ValueError("Source export differs from pinned clean checkout: " + repr(mismatches[:10]))
    return {"manifest_sha256": sha(manifest), "verified_files": len(rows), "mismatches": []}


def query(argv):
    result = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise RuntimeError("Read-only query failed: " + argv[0])
    return result.stdout


def cgroup_memory():
    paths = {"v2_limit_bytes": "/sys/fs/cgroup/memory.max", "v2_current_bytes": "/sys/fs/cgroup/memory.current",
             "v1_limit_bytes": "/sys/fs/cgroup/memory/memory.limit_in_bytes",
             "v1_current_bytes": "/sys/fs/cgroup/memory/memory.usage_in_bytes",
             "v1_failcnt": "/sys/fs/cgroup/memory/memory.failcnt"}
    return {name: Path(path).read_text().strip() for name, path in paths.items() if Path(path).is_file()}


def media(path, args):
    def probe(target):
        return json.loads(query(["ffprobe", "-v", "error", "-count_frames", "-show_streams", "-show_format", "-of", "json", str(target)]))
    details = probe(path)
    video = next(row for row in details["streams"] if row["codec_type"] == "video")
    frames = int(video.get("nb_read_frames", video.get("nb_frames", -1)))
    if (int(video["width"]), int(video["height"]), frames) != (args.width, args.height, args.frames):
        raise ValueError("Unexpected decoded video dimensions/frame count")
    rgb = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0", "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"], capture_output=True, timeout=180, check=True).stdout
    if len(rgb) != args.width * args.height * args.frames * 3:
        raise ValueError("Decoded RGB length mismatch")
    audio_path = path.with_suffix(".wav")
    if not audio_path.is_file():
        audio_path = path
    audio = next(row for row in probe(audio_path)["streams"] if row["codec_type"] == "audio")
    if int(audio["sample_rate"]) != 48000 or int(audio["channels"]) != 2:
        raise ValueError("Expected official LTX-2.5 48 kHz stereo audio")
    raw_audio = subprocess.run(["ffmpeg", "-v", "error", "-i", str(audio_path), "-map", "0:a:0", "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1"], capture_output=True, timeout=180, check=True).stdout
    waveform = array.array("f")
    waveform.frombytes(raw_audio)
    if sys.byteorder != "little":
        waveform.byteswap()
    if not waveform or not all(math.isfinite(value) for value in waveform):
        raise ValueError("Empty/nonfinite decoded audio")
    audio_seconds = len(waveform) / (int(audio["sample_rate"]) * int(audio["channels"]))
    # The native audio VAE works on a latent grid, so retain its measured duration
    # and bound gross truncation without requiring codec-specific exact trimming.
    if abs(audio_seconds - args.frames / args.fps) > .25:
        raise ValueError("Audio duration differs from video by more than 250 ms")
    return {"video": video, "audio": audio, "audio_path": str(audio_path), "audio_seconds": audio_seconds,
            "rgb_min": min(rgb), "rgb_max": max(rgb), "rgb_mean": sum(rgb) / len(rgb),
            "rgb_sha256": hashlib.sha256(rgb).hexdigest(), "audio_sha256": hashlib.sha256(raw_audio).hexdigest(),
            "audio_peak": max(abs(value) for value in waveform),
            "audio_rms": math.sqrt(sum(value * value for value in waveform) / len(waveform)),
            "files": [{"path": str(target), "bytes": target.stat().st_size, "sha256": sha(target)} for target in dict.fromkeys([path, audio_path])]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--build-manifest", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--denoiser", default="ltx-2.5-22b-distilled-UD-Q4_K_XL.gguf",
                        help="Distilled denoiser GGUF under --models (config/ltx-2.5-distilled.json's file by default)")
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--frames", type=int, default=9)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--backend", default="cuda0")
    parser.add_argument("--prompt", default="A red toy car drives slowly along a quiet road in warm afternoon light.")
    parser.add_argument("--sigmas", help="Explicit comma-separated sigmas, including terminal zero")
    parser.add_argument("--identity-only", action="store_true")
    parser.add_argument("--image", type=Path, help="First-frame image; conditioning strength is fixed to one")
    parser.add_argument("--end-image", type=Path, help="Last-frame image; conditioning strength is fixed to one")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    pins = json.loads(args.build_manifest.read_text())
    report = {"started_unix": time.time(), "passed": False, "perceptual_quality_parity_established": False,
              "performance_parity_established": False, "executions": [], "tool_sha256": sha(__file__),
              "source_revisions": {"sdcpp": pins["sdc_revision"], "sdcpp_ggml": pins["ggml_revision"]},
              "timing_scope": "whole CLI process including model loading and output codec; new process per repetition",
              "limitations": ["Structural media validity is not perceptual quality parity.", "Cold file cache and warm-model latency are not established.", "Sampled GPU telemetry is not an allocator peak measurement."],
              "hardware": {"platform": platform.platform(), "logical_cpus": os.cpu_count()},
              "reference_inputs": [{"path": str(path.resolve()), "frame_index": frame, "strength": 1,
                                    "sha256": sha(path), "bytes": path.stat().st_size}
                                   for path, frame in ((args.image, 0), (args.end_image, args.frames - 1)) if path],
              "configuration": {key: getattr(args, key) for key in ["width", "height", "frames", "steps", "fps", "seed", "rounds", "threads", "backend", "prompt", "sigmas"]}}
    save = lambda: (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    process = None
    try:
        report["source_before"] = verify_source(args.source, args.source_manifest)
        report["binary_sha256"] = sha(args.cli)
        if report["binary_sha256"] != pins["remote_cuda_build" if platform.system() == "Linux" else "local_metal_build"]["sha256"]:
            raise ValueError("Baseline binary differs from verified build manifest")
        report["hardware"]["gpu"] = query(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]).strip() if platform.system() == "Linux" else query(["sysctl", "-n", "machdep.cpu.brand_string"]).strip()
        report["hardware"]["cgroup_memory_before"] = cgroup_memory()
        if args.identity_only:
            report["identity_verified"] = True
            report["limitations"].append("Identity only; inference was not executed.")
            return 0
        assets = json.loads(args.model_manifest.read_text())
        if assets["status"] != "verified" or not all(item["completed"] for item in assets["files"]):
            raise ValueError("All pinned model downloads must be verified")
        report["model_manifest_sha256"] = sha(args.model_manifest)
        report["models"] = assets["files"]
        for item in report["models"]:
            path = args.models / item["path"]
            if path.stat().st_size != item["bytes"] or sha(path) != item["sha256"]:
                raise ValueError("Pinned model differs: " + item["path"])
        if platform.system() == "Linux" and query(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"]).strip():
            raise RuntimeError("GPU benchmark lane is occupied")
        for index in range(args.rounds):
            output = (args.output / ("sdcpp-" + str(index) + ".webm")).resolve()
            command = [str(args.cli.resolve()), "-M", "vid_gen", "--diffusion-model", str(args.models / args.denoiser),
                       "--llm", str(args.models / "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"),
                       "--vae", str(args.models / "vae/ltx-2.5-video-vae-conv-bf16.safetensors"),
                       "--audio-vae", str(args.models / "vae/ltx-2.5-audio-vae-bf16.safetensors"),
                       "-p", args.prompt, "-W", str(args.width), "-H", str(args.height), "--video-frames", str(args.frames),
                       "--steps", str(args.steps), "--fps", str(args.fps), "--cfg-scale", "1", "--sampling-method", "euler",
                       "--rng", "cuda", "-s", str(args.seed), "--backend", args.backend, "-t", str(args.threads),
                       "--mmap", "--offload-to-cpu", "--diffusion-fa", "-o", str(output)]
            if args.sigmas:
                command += ["--sigmas", args.sigmas]
            if args.image or args.end_image:
                command += ["--strength", "1"]
            if args.image:
                command += ["--init-img", str(args.image.resolve())]
            if args.end_image:
                command += ["--end-img", str(args.end_image.resolve())]
            record = {"round": index, "argv": command, "started_unix": time.time(), "gpu_samples": []}
            report["executions"].append(record)
            save()
            started = time.monotonic()
            with (args.output / ("sdcpp-" + str(index) + ".log")).open("x") as stream:
                process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
                                           env=dict(os.environ, OMP_NUM_THREADS=str(args.threads)))
                record["pid"] = process.pid
                while process.poll() is None:
                    if time.monotonic() - started > args.timeout:
                        raise TimeoutError("Owned baseline timed out")
                    if platform.system() == "Linux":
                        sample = {"elapsed_seconds": time.monotonic() - started, "gpu": query(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu,power.draw", "--format=csv,noheader"]).strip(), "cgroup_memory": cgroup_memory()}
                        status = Path("/proc") / str(process.pid) / "status"
                        try:
                            sample["process_memory"] = [line for line in status.read_text().splitlines() if line.startswith(("VmRSS:", "VmHWM:"))]
                        except FileNotFoundError:
                            pass
                        record["gpu_samples"].append(sample)
                    save()
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        pass
                record.update(exit_code=process.returncode, wall_seconds=time.monotonic() - started)
            record["passed"] = False
            if process.returncode == 0:
                record["media"] = media(output, args)
                record["passed"] = True
            save()
            print(json.dumps({key: record[key] for key in ["round", "exit_code", "wall_seconds", "passed"]}), flush=True)
        report["source_after"] = verify_source(args.source, args.source_manifest)
        report["hardware"]["cgroup_memory_after"] = cgroup_memory()
        report["passed"] = bool(report["executions"]) and all(item["passed"] for item in report["executions"])
        return 0 if report["passed"] else 1
    except Exception as error:
        report["error"] = repr(error)
        return 1
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        save()


if __name__ == "__main__":
    sys.exit(main())
