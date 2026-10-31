#!/usr/bin/env python3
"""Digest a DeepSeek V4.1 trace directory so two runs can be checked for bit-identity without keeping both.

    dsv41-trace-digest.py digest TRACE_DIR --probe PROBE_DIR --model MODEL.gguf > A.digest
    dsv41-trace-digest.py compare A.digest B.digest

digest prints one line per trace file: its sha256 and name. Every file is hashed byte for byte except
the direct-CUDA engine's sparse-selection dumps, which hold memory the engine never reads:
  * p<p0>_blkLL_index_scores.f32 [nt, cache rows]: only each query's visible rows, (p + 1) // ratio,
    are written (ratio = the layer's deepseek41.attention.compress_ratios entry); the rest is stale.
  * p<p0>_blkLL_index_top.i32 [nt, k]: only the first index_count entries of each row are written.
Those are hashed over the written part only. The native (ggml) trace has no such regions.
PROBE_DIR (probe.json) gives the prefix length, which gives each file's token count.

compare reports the files whose digests differ and the files present in one digest only, and exits
1 unless both digests list the same files with the same hashes.
"""
import argparse
import hashlib
import json
import os
import re
import struct
import sys

import numpy as np

MANAGED = re.compile(r"p(\d+)_blk(\d+)_(index_scores|index_top|index_count)\.(f32|i32)$")


def gguf_array(path, key):
    """One array value from a GGUF header (little-endian, version 2 or 3)."""
    scalar = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
    with open(path, "rb") as f:
        def read(fmt):
            size = struct.calcsize("<" + fmt)
            return struct.unpack("<" + fmt, f.read(size))[0]

        def string():
            return f.read(read("Q")).decode("utf-8", "replace")

        def value(kind):
            if kind == 8:
                return string()
            if kind == 9:
                inner, count = read("I"), read("Q")
                return [value(inner) for _ in range(count)]
            return read(scalar[kind])

        if f.read(4) != b"GGUF":
            raise SystemExit(f"{path}: not a GGUF file")
        read("I")
        read("Q")
        for _ in range(read("Q")):
            name = string()
            v = value(read("I"))
            if name == key:
                return v
    raise SystemExit(f"{path}: no {key}")


def digest(args):
    with open(os.path.join(args.probe, "probe.json")) as f:
        probe = json.load(f)
    steps = [s for s in probe["steps"] if s.get("logits")]
    prefix = len(probe["tokens"]) - sum(1 for s in steps if s["operation"] == "decode")
    ratios = gguf_array(args.model, "deepseek41.attention.compress_ratios")
    names = sorted(os.listdir(args.trace))
    managed = {}
    for name in names:
        m = MANAGED.match(name)
        if m:
            managed[(int(m.group(1)), int(m.group(2)), m.group(3))] = name
    p0s = {}
    for (p0, layer, kind) in managed:
        if kind == "index_scores":
            p0s.setdefault(layer, []).append(p0)

    def nt_of(layer, p0):
        ordered = sorted(p0s[layer])
        if p0 >= prefix:
            return 1
        later = [p for p in ordered if p > p0]
        return min(later[0] if later else prefix, prefix) - p0

    for name in names:
        path = os.path.join(args.trace, name)
        m = MANAGED.match(name)
        h = hashlib.sha256()
        if m and m.group(3) in ("index_scores", "index_top"):
            p0, layer, kind = int(m.group(1)), int(m.group(2)), m.group(3)
            nt = nt_of(layer, p0)
            ratio = max(int(ratios[layer]), 1)
            if kind == "index_scores":
                v = np.fromfile(path, np.float32).reshape(nt, -1)
                for q in range(nt):
                    h.update(v[q, :(p0 + q + 1) // ratio].tobytes())
            else:
                v = np.fromfile(path, np.int32).reshape(nt, -1)
                count = managed.get((p0, layer, "index_count"))
                count = np.fromfile(os.path.join(args.trace, count), np.int32) if count else None
                for q in range(nt):
                    h.update(v[q, :count[q]].tobytes() if count is not None else v[q].tobytes())
            tag = "written-part"
        else:
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 22), b""):
                    h.update(chunk)
            tag = "bytes"
        print(f"{h.hexdigest()}  {tag:12s} {name}")
    return 0


def compare(args):
    def read(path):
        out = {}
        with open(path) as f:
            for line in f:
                digest_hex, _tag, name = line.rstrip("\n").split(None, 2)
                out[name] = digest_hex
        return out

    a, b = read(args.a), read(args.b)
    differing = sorted(n for n in set(a) & set(b) if a[n] != b[n])
    only_a, only_b = sorted(set(a) - set(b)), sorted(set(b) - set(a))
    print(f"files: {len(a)} vs {len(b)}; common {len(set(a) & set(b))}; differing {len(differing)};"
          f" only in A {len(only_a)}; only in B {len(only_b)}")
    for n in differing[:20]:
        print("  differs:", n)
    for n in only_a[:10]:
        print("  only A:", n)
    for n in only_b[:10]:
        print("  only B:", n)
    identical = not differing and not only_a and not only_b
    print("BIT-IDENTICAL" if identical else "NOT IDENTICAL")
    return 0 if identical else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    d = sub.add_parser("digest")
    d.add_argument("trace")
    d.add_argument("--probe", required=True)
    d.add_argument("--model", required=True)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    args = ap.parse_args()
    return digest(args) if args.command == "digest" else compare(args)


if __name__ == "__main__":
    sys.exit(main())
