#!/usr/bin/env python3
"""Stream every model shard against a pinned manifest, outside timed inference.

Input: {repo, revision, quant, files:[{path:relative, size:int, sha256:hex}]}.
Output is compatible with run-parallel-case.py --model-verification-report.
Previously recorded verified/actual_sha256 fields are never trusted as evidence.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import time


def save(path, report):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(path)


def verify(manifest_path, root, output, block_size=4 * 1024 * 1024):
    if output.exists():
        raise ValueError("Output must be fresh; preserve earlier verification: " + str(output))
    if block_size < 1:
        raise ValueError("block_size must be positive")
    manifest = json.loads(manifest_path.read_text())
    for field in ("repo", "revision", "quant"):
        if not isinstance(manifest.get(field), str) or not manifest[field].strip():
            raise ValueError("Manifest requires a nonempty " + field)
    if not re.fullmatch(r"[0-9a-fA-F]{40,64}", manifest["revision"]):
        raise ValueError("Manifest revision must be an immutable full commit digest")
    if not isinstance(manifest.get("files"), list) or not manifest["files"]:
        raise ValueError("Manifest must contain a nonempty files list")
    root = root.resolve()
    entries, seen = [], set()
    for entry in manifest["files"]:
        relative = Path(entry["path"])
        path = (root / relative).resolve()
        if relative.is_absolute() or path == root or root not in path.parents:
            raise ValueError("Shard path must remain inside model root: " + str(relative))
        if path in seen:
            raise ValueError("Duplicate model shard: " + str(relative))
        seen.add(path)
        if type(entry.get("size")) is not int or entry["size"] < 1:
            raise ValueError("Shard size must be a positive integer")
        if not re.fullmatch(r"[0-9a-fA-F]{64}", str(entry.get("sha256", ""))):
            raise ValueError("Shard sha256 must be a full hex digest")
        entries.append((path, entry))
    report = {"status": "running", "repo": manifest["repo"], "revision": manifest["revision"],
        "quant": manifest["quant"], "root": str(root), "manifest": str(manifest_path.resolve()),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "started_unix": time.time(), "shards": [], "failures": [],
        "limitations": ["Full-file reads may warm storage/page caches; run outside measured requests.",
            "Expected hashes are supplied by the caller's pinned manifest; this utility does not authenticate its publisher.",
            "Verified files must remain unchanged; run-parallel-case rechecks sizes, not every byte, before loading."]}
    output.parent.mkdir(parents=True, exist_ok=True)
    save(output, report)
    for path, entry in entries:
        shard = {"path": str(path), "bytes": entry["size"], "sha256": entry["sha256"].lower(),
                 "status": "running", "verified": False, "bytes_hashed": 0}
        report["shards"].append(shard)
        save(output, report)
        started = time.monotonic()
        try:
            before = path.stat()
            if before.st_size != entry["size"]:
                raise ValueError(f"Size differs: expected {entry['size']}, actual {before.st_size}")
            identity = lambda stat: (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
            digest, progress = hashlib.sha256(), 0
            with path.open("rb") as source:
                for block in iter(lambda: source.read(block_size), b""):
                    digest.update(block)
                    shard["bytes_hashed"] += len(block)
                    if shard["bytes_hashed"] - progress >= 256 * 1024 * 1024:
                        progress = shard["bytes_hashed"]
                        save(output, report)
            shard["actual_sha256"] = digest.hexdigest()
            if identity(before) != identity(path.stat()):
                raise ValueError("Model shard changed while it was being hashed")
            if shard["bytes_hashed"] != entry["size"] or shard["actual_sha256"] != shard["sha256"]:
                raise ValueError("Model shard SHA256 or bytes read differs from pinned manifest")
            shard.update(status="verified", verified=True)
        except (OSError, ValueError) as error:
            shard.update(status="failed", error=str(error))
            report["failures"].append({"path": str(path), "error": str(error)})
        finally:
            shard["seconds"] = time.monotonic() - started
            save(output, report)
        print(shard["status"], path.name, shard["bytes_hashed"], flush=True)
    report.update(status="failed" if report["failures"] else "verified", finished_unix=time.time())
    save(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = verify(args.manifest, args.root, args.output)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, str(error) + "\n")
    return int(report["status"] != "verified")


if __name__ == "__main__":
    raise SystemExit(main())
