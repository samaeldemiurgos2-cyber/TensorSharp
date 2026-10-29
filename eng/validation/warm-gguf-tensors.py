#!/usr/bin/env python3
"""Read selected tensors of a (split) GGUF once so their pages sit in the page cache.

Host-mapped tables that are gathered row by row at run time (Qwen3.8's PLE n-gram
table, DeepSeek V4.1's Engram) otherwise fault pages in from storage during timed
decode. On network filesystems each fault can cost milliseconds, which then shows
up as decode time and differs between nodes. Run this before benchmarks on every
node that owns such a table, and record that the cache was warmed.

    warm-gguf-tensors.py FIRST_SHARD.gguf --match per_layer_token_embd [--threads 8]

Only headers are parsed; no tensor data is interpreted. Exits non-zero when no
tensor matches.
"""
import argparse
import concurrent.futures
import json
import os
import re
import struct
import sys
import time

_SCALARS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


def _string(f):
    (n,) = struct.unpack("<Q", f.read(8))
    return f.read(n).decode("utf-8", errors="replace")


def _skip_value(f, kind):
    if kind in _SCALARS:
        f.read(struct.calcsize("<" + _SCALARS[kind]))
    elif kind == 8:
        _string(f)
    elif kind == 9:
        (item, count) = struct.unpack("<IQ", f.read(12))
        for _ in range(count):
            _skip_value(f, item)
    else:
        raise ValueError(f"unknown GGUF metadata type {kind}")


def read_tensors(path):
    """Return [(name, absolute_offset, byte_count)] for one shard, sorted by offset."""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError(f"{path} is not a GGUF file")
        (version,) = struct.unpack("<I", f.read(4))
        (n_tensors, n_kv) = struct.unpack("<QQ", f.read(16))
        alignment = 32
        for _ in range(n_kv):
            key = _string(f)
            (kind,) = struct.unpack("<I", f.read(4))
            if key == "general.alignment" and kind == 4:
                (alignment,) = struct.unpack("<I", f.read(4))
            else:
                _skip_value(f, kind)
        infos = []
        for _ in range(n_tensors):
            name = _string(f)
            (n_dims,) = struct.unpack("<I", f.read(4))
            f.read(8 * n_dims)
            (_type, offset) = struct.unpack("<IQ", f.read(12))
            infos.append((name, offset))
        start = f.tell()
        start += (alignment - start % alignment) % alignment
    infos.sort(key=lambda item: item[1])
    tensors = []
    for index, (name, offset) in enumerate(infos):
        end = infos[index + 1][1] if index + 1 < len(infos) else size - start
        tensors.append((name, start + offset, end - offset))
    return version, tensors


def shards(first):
    match = re.match(r"(.*)-00001-of-(\d{5})\.gguf$", first)
    if not match:
        return [first]
    count = int(match.group(2))
    return [f"{match.group(1)}-{i:05d}-of-{count:05d}.gguf" for i in range(1, count + 1)]


def warm(path, offset, length, chunk):
    done = 0
    with open(path, "rb", buffering=0) as f:
        while done < length:
            n = min(chunk, length - done)
            data = os.pread(f.fileno(), n, offset + done)
            if not data:
                raise OSError(f"{path}: unexpected end at {offset + done}")
            done += len(data)
    return done


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", help="first shard of the GGUF")
    parser.add_argument("--match", action="append", required=True, help="substring of tensor names to warm (repeatable)")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--chunk-mib", type=int, default=8)
    parser.add_argument("--report", help="optional JSON report path")
    args = parser.parse_args()

    jobs = []
    for shard in shards(args.model):
        _, tensors = read_tensors(shard)
        for name, offset, length in tensors:
            if any(m in name for m in args.match):
                jobs.append((shard, name, offset, length))
    if not jobs:
        print("no tensor matches " + ", ".join(args.match), file=sys.stderr)
        return 1

    chunk = args.chunk_mib * 1024 * 1024
    pieces = []
    for shard, _, offset, length in jobs:
        for start in range(0, length, chunk * 16):
            pieces.append((shard, offset + start, min(chunk * 16, length - start)))
    total = sum(p[2] for p in pieces)
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.threads) as pool:
        read = sum(pool.map(lambda p: warm(p[0], p[1], p[2], chunk), pieces))
    seconds = time.monotonic() - started
    report = {
        "model": args.model, "match": args.match, "threads": args.threads,
        "tensors": [{"shard": os.path.basename(s), "name": n, "offset": o, "bytes": b} for s, n, o, b in jobs],
        "bytes": total, "read_bytes": read, "seconds": round(seconds, 3),
        "MiB_per_second": round(read / max(seconds, 1e-9) / 1024**2, 1),
    }
    print(f"warmed {len(jobs)} tensor(s), {read / 1024**3:.2f} GiB in {seconds:.1f} s "
          f"({report['MiB_per_second']} MiB/s)")
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    return 0 if read == total else 1


if __name__ == "__main__":
    sys.exit(main())
