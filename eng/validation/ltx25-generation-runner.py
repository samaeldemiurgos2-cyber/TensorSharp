#!/usr/bin/env python3
"""Run an explicit inference argv with bounded process/device memory sampling."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time


def file_cache(paths):
    """Observe Linux file page residency without faulting pages or evicting caches."""
    if not sys.platform.startswith("linux"):
        return {"available": False, "reason": "mincore observation implemented for Linux only"}
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_longlong]
    libc.mmap.restype = ctypes.c_void_p
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
    libc.mincore.restype = ctypes.c_int
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    page = os.sysconf("SC_PAGE_SIZE")
    rows = []
    for path in paths:
        path = Path(path)
        if not path.is_file():
            continue
        size = path.stat().st_size
        fd = os.open(path, os.O_RDONLY)
        address = None
        try:
            address = libc.mmap(None, size, 0, 2, fd, 0)  # PROT_NONE, MAP_PRIVATE: no page readback
            if address == ctypes.c_void_p(-1).value:
                raise OSError(ctypes.get_errno(), "mmap failed")
            count = (size + page - 1) // page
            resident = (ctypes.c_ubyte * count)()
            if libc.mincore(address, size, resident) != 0:
                raise OSError(ctypes.get_errno(), "mincore failed")
            present = sum(bool(value & 1) for value in resident)
            rows.append({"path": str(path), "bytes": size, "pages": count,
                         "resident_pages": present, "resident_fraction": present / count})
        except OSError as error:
            rows.append({"path": str(path), "bytes": size, "error": str(error)})
        finally:
            if address is not None and address != ctypes.c_void_p(-1).value:
                libc.munmap(address, size)
            os.close(fd)
    return {"available": True, "scope": "Read-only mincore immediately outside launched process; no eviction or cache warming; residency can change during inference.", "page_bytes": page, "files": rows}


def model_paths(command):
    paths = {Path(item).resolve() for item in command if item.endswith((".gguf", ".safetensors")) and Path(item).is_file()}
    for model in list(paths):
        if model.suffix == ".gguf" and "ltx" in model.name.lower():
            root = model.parent
            paths.update([root / "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
                          root / "vae/ltx-2.5-video-vae-conv-bf16.safetensors", root / "vae/ltx-2.5-audio-vae-bf16.safetensors"])
    return sorted(paths)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()
    command = json.loads(args.command_json.read_text())
    if not isinstance(command, list) or not command or not all(isinstance(item, str) for item in command):
        raise ValueError("Command must be a nonempty JSON array of argv strings.")
    if args.output.exists() or Path(str(args.output) + ".log").exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cache_paths = model_paths(command)
    cache_before = file_cache(cache_paths)
    started = time.monotonic()
    samples, timeout = [], False
    with Path(str(args.output) + ".log").open("xb") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        while process.poll() is None:
            sample = {"elapsed_seconds": time.monotonic() - started}
            if sys.platform.startswith("linux"):
                try:
                    rows = dict(line.split(":", 1) for line in Path(f"/proc/{process.pid}/status").read_text().splitlines() if ":" in line)
                    sample["process_rss_bytes"] = int(rows["VmRSS"].split()[0]) * 1024
                except (OSError, KeyError):
                    pass
                for name in ("memory.usage_in_bytes", "memory.max_usage_in_bytes", "memory.limit_in_bytes", "memory.failcnt"):
                    path = Path("/sys/fs/cgroup/memory") / name
                    if path.exists():
                        sample["cgroup_v1_" + name] = int(path.read_text())
                try:
                    result = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                                            capture_output=True, text=True, timeout=3)
                    if result.returncode == 0:
                        sample["gpu"] = [{"device_index": i, "used_bytes": int(row.split(",")[0]) * 1024 ** 2,
                                          "utilization_percent": int(row.split(",")[1])}
                                         for i, row in enumerate(result.stdout.strip().splitlines())]
                except (OSError, subprocess.TimeoutExpired):
                    pass
            else:
                result = subprocess.run(["ps", "-o", "rss=", "-p", str(process.pid)], capture_output=True, text=True)
                if result.returncode == 0 and result.stdout.strip():
                    sample["process_rss_bytes"] = int(result.stdout.strip()) * 1024
            samples.append(sample)
            if sample["elapsed_seconds"] > args.timeout:
                timeout = True
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                break
            time.sleep(1)
        code = process.wait()
    wall_seconds = time.monotonic() - started
    log_path = Path(str(args.output) + ".log")
    max_rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    record = {"completed": code == 0 and not timeout, "exit_code": code, "timed_out": timeout,
              "file_cache_before": cache_before, "file_cache_after": file_cache(cache_paths),
              "argv": command, "command_sha256": hashlib.sha256(args.command_json.read_bytes()).hexdigest(),
              "wall_seconds": wall_seconds, "sample_interval_seconds": 1,
              "max_child_rss_bytes": max_rss if sys.platform == "darwin" else max_rss * 1024,
              "scope": "Whole launched process, including setup and any codecs/file writes; use inner generation reports for paired inference timings. GPU/cgroup values are device/container-wide sampled values, not exclusive process allocations. File pages may be warm.",
              "log": str(log_path), "log_sha256": hashlib.sha256(log_path.read_bytes()).hexdigest(), "samples": samples}
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({key: record[key] for key in ("completed", "exit_code", "wall_seconds", "max_child_rss_bytes")}))
    raise SystemExit(code or (1 if timeout else 0))


if __name__ == "__main__":
    main()
