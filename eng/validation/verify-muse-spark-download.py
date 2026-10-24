#!/usr/bin/env python3
"""Verify a pinned Muse Spark quantization, including its small metadata shard.

The publisher intentionally stores no tensors in shard 00001. Its 13 MB size is
not evidence of an incomplete download. All expected sizes come from the pinned
HF manifest. Full hashing is optional because this checkpoint exceeds 500 GB.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor
from muse_spark_artifacts import shard_entries


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--revision", default="c00670568b6ce5e56a247f67091247984df2a01e")
    p.add_argument("--quantization", default="UD-Q4_K_XL")
    p.add_argument("--force-first", action="store_true")
    p.add_argument("--hash-all", action="store_true")
    p.add_argument("--workers", type=int, default=2)
    args = p.parse_args()
    if not 1 <= args.workers <= 4:
        p.error("workers must be between 1 and 4")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("sha") != args.revision or manifest.get("id") != "unsloth/Muse-Spark-Open-816B-A42B-GGUF":
        p.error("Manifest repository or pinned revision mismatch")
    try:
        entries = shard_entries(manifest, args.quantization)
    except ValueError as error:
        p.error(str(error))
    report = {"repo": manifest["id"], "revision": args.revision,
              "quantization": args.quantization, "shard_count": len(entries),
              "manifest_sha256": sha256(args.manifest), "force_first": args.force_first,
              "hash_all": args.hash_all, "scope": "All selected shard sizes and first-shard SHA256; full hashes only if hash_all=true.",
              "shards": []}
    started = time.monotonic()
    if args.force_first:
        from huggingface_hub import hf_hub_download
        hf_hub_download(manifest["id"], entries[0]["rfilename"], revision=args.revision,
                        local_dir=args.directory, force_download=True)
    def verify(item):
        i, entry = item
        path = args.directory / entry["rfilename"]
        expected = entry.get("size", entry.get("lfs", {}).get("size"))
        actual = path.stat().st_size if path.exists() else None
        row = {"file": entry["rfilename"], "expected_bytes": expected, "actual_bytes": actual,
               "size_status": "pass" if actual == expected and expected is not None else "fail",
               "sha256_status": "not_checked"}
        if actual is not None and (i == 0 or args.hash_all):
            expected_hash = entry.get("lfs", {}).get("sha256")
            actual_hash = sha256(path)
            row.update(expected_sha256=expected_hash, actual_sha256=actual_hash,
                       sha256_status="pass" if expected_hash and expected_hash == actual_hash else "fail")
        print(json.dumps(row), flush=True)
        return row
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        report["shards"] = list(pool.map(verify, enumerate(entries)))
    report["status"] = "pass" if all(r["size_status"] == "pass" and r["sha256_status"] != "fail" for r in report["shards"]) else "fail"
    report["seconds"] = time.monotonic() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(report["status"], flush=True)
    return int(report["status"] != "pass")


if __name__ == "__main__":
    raise SystemExit(main())
