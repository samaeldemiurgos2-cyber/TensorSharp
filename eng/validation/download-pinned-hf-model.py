#!/usr/bin/env python3
"""Download named HF model files at one immutable revision and save its LFS manifest.

Requires huggingface_hub. Run verify-pinned-model-manifest.py afterward: the
download cache and publisher's manifest alone do not establish local byte identity.
"""
import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--file", action="append", required=True)
    parser.add_argument("--quant", required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.manifest.exists():
        parser.error("Manifest must be fresh; preserve the earlier pinned identity")
    api = HfApi()
    revision = api.model_info(args.repo, revision=args.revision).sha
    metadata = api.model_info(args.repo, revision=revision, files_metadata=True)
    files = {entry.rfilename: entry for entry in metadata.siblings}
    manifest = {"repo": args.repo, "revision": revision, "quant": args.quant, "files": []}
    for name in args.file:
        entry = files[name]
        if not entry.lfs or not entry.lfs.sha256 or not entry.size:
            raise ValueError("File lacks publisher LFS SHA256 or size: " + name)
        manifest["files"].append({"path": name, "size": entry.size, "sha256": entry.lfs.sha256})
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    for name in args.file:
        print(hf_hub_download(args.repo, filename=name, revision=revision, local_dir=args.directory), flush=True)


if __name__ == "__main__":
    main()
