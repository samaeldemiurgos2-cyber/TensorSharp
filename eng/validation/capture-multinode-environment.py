#!/usr/bin/env python3
"""Capture bounded, non-secret Linux multi-node validation environment evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--node", required=True)
    parser.add_argument("--route", required=True)
    parser.add_argument("--dotnet", default="dotnet")
    parser.add_argument("--native", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must be fresh; preserve previous evidence")
    report = {"schema_version": 1, "node": args.node, "route": args.route,
              "captured_unix": time.time(), "platform": platform.platform(), "commands": {},
              "limitations": ["Host RAM and CPUs may exceed the captured cgroup allowance.",
                              "Topology and route capture does not establish throughput or model quality."]}
    commands = {
        "gpu": ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,driver_version", "--format=csv"],
        "gpu_topology": ["nvidia-smi", "topo", "-m"],
        "cpu": ["lscpu"], "memory": ["free", "-b"],
        "storage": ["df", "-B1", str(args.repo), "/", "/dev/shm"],
        "addresses": ["ip", "-br", "address"], "routes": ["ip", "route"],
        "dotnet": [args.dotnet, "--info"],
        "source_revision": ["git", "-C", str(args.repo), "rev-parse", "HEAD"],
        "source_status": ["git", "-C", str(args.repo), "status", "--porcelain"],
        "ggml_revision": ["git", "-C", str(args.repo / "ExternalProjects/ggml"), "rev-parse", "HEAD"],
        "ggml_status": ["git", "-C", str(args.repo / "ExternalProjects/ggml"), "status", "--porcelain", "--untracked-files=all"],
        # Process names only: command arguments could contain credentials.
        "processes": ["ps", "-eo", "pid,pcpu,pmem,comm", "--sort=-pcpu"],
    }
    for key, command in commands.items():
        try:
            value = subprocess.run(command, capture_output=True, text=True, timeout=45)
            report["commands"][key] = {"command": command, "exit_code": value.returncode,
                                       "stdout": value.stdout, "stderr": value.stderr}
        except (OSError, subprocess.TimeoutExpired) as error:
            report["commands"][key] = {"command": command, "error": str(error)}
    report["cgroup"] = {}
    for name in ("memory.max", "memory.current", "cpu.max", "cpuset.cpus.effective",
                 "memory/memory.limit_in_bytes", "memory/memory.usage_in_bytes",
                 "cpu/cpu.cfs_quota_us", "cpu/cpu.cfs_period_us", "cpuset/cpuset.cpus"):
        path = Path("/sys/fs/cgroup") / name
        if path.is_file():
            report["cgroup"][name] = path.read_text().strip()
    if args.native:
        with args.native.open("rb") as source:
            report["native"] = {"path": str(args.native), "bytes": args.native.stat().st_size,
                                "sha256": hashlib.file_digest(source, "sha256").hexdigest()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
