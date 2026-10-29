#!/usr/bin/env python3
"""Stage the tensors one node needs from a pinned (split) GGUF onto fast local storage.

Each shard is streamed from its pinned Hugging Face URL with parallel range
requests and hashed in order, so every byte written locally belongs to a file
whose publisher SHA-256 was verified. Only the header and the selected tensors
are written; the rest of the file stays a hole, so the local copy has the
published size but allocates only what this node reads. A loader that touches
an unselected tensor would read zeros: select by the same layer ownership the
node's placement uses, and keep the run's numerical checks.

Stores are filled in order. Typical: a RAM-backed tmpfs first, then local disk.

    stage-gguf-local.py MANIFEST.json --dest /root/models/qwen38 \\
        --store /dev/shm/ts-models:60 --store /root/model-store:38 \\
        --layers 0:24 --exclude '^output' [--connections 8]

MANIFEST is the {"shards": [{"url", "bytes", "sha256"}]} format written for
download-validated-model.py. --layers keeps blk.N tensors with A <= N < B
(repeatable); tensors outside any block are kept unless --exclude matches;
--include always keeps a matching tensor. --all keeps every tensor.
--rows REGEX:FIRST:END keeps only rows FIRST..END-1 (along ne1) of every expert
slab of a matching tensor, for tensor-parallel nodes that slice output rows;
its other rows stay holes. It applies after the layer/name selection.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import struct
import sys
import threading
import time
import urllib.error
import urllib.request
from urllib.parse import unquote, urlsplit

CHUNK = 64 * 1024 * 1024
_SCALARS = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


class Reader:
    def __init__(self, data):
        self.data, self.pos = data, 0

    def take(self, n):
        if self.pos + n > len(self.data):
            raise EOFError
        value = self.data[self.pos:self.pos + n]
        self.pos += n
        return value

    def u32(self):
        return struct.unpack("<I", self.take(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.take(8))[0]

    def string(self):
        return self.take(self.u64()).decode("utf-8", errors="replace")

    def skip_value(self, kind):
        if kind in _SCALARS:
            self.take(_SCALARS[kind])
        elif kind == 8:
            self.string()
        elif kind == 9:
            item, count = self.u32(), self.u64()
            if item in _SCALARS:
                self.take(_SCALARS[item] * count)
            else:
                for _ in range(count):
                    self.skip_value(item)
        else:
            raise ValueError(f"unknown GGUF metadata type {kind}")


def parse_header(data, size):
    """Return (data_start, [(name, absolute_offset, bytes, dims)]) or raise EOFError if data is short."""
    r = Reader(data)
    if r.take(4) != b"GGUF":
        raise ValueError("not a GGUF file")
    r.u32()
    n_tensors, n_kv = r.u64(), r.u64()
    alignment = 32
    for _ in range(n_kv):
        key, kind = r.string(), r.u32()
        if key == "general.alignment" and kind == 4:
            alignment = r.u32()
        else:
            r.skip_value(kind)
    infos = []
    for _ in range(n_tensors):
        name = r.string()
        n_dims = r.u32()
        dims = struct.unpack(f"<{n_dims}Q", r.take(8 * n_dims))
        r.u32()
        infos.append((name, r.u64(), dims))
    start = r.pos + (alignment - r.pos % alignment) % alignment
    infos.sort(key=lambda item: item[1])
    tensors = []
    for i, (name, offset, dims) in enumerate(infos):
        end = infos[i + 1][1] if i + 1 < len(infos) else size - start
        tensors.append((name, start + offset, end - offset, dims))
    return start, tensors


def resolve(url):
    """Follow redirects once and return a URL that answers range requests."""
    request = urllib.request.Request(url, headers={"Range": "bytes=0-0", "Accept-Encoding": "identity"})
    with urllib.request.urlopen(request, timeout=60) as response:
        if response.status != 206:
            raise ValueError(f"{url}: server ignored the range request ({response.status})")
        return response.geturl()


def fetch(url, start, end, retries=6):
    """Bytes [start, end] inclusive."""
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"})
            with urllib.request.urlopen(request, timeout=120) as response:
                if response.status != 206:
                    raise ValueError(f"unexpected HTTP status {response.status}")
                data = response.read()
            if len(data) != end - start + 1:
                raise OSError(f"short range: {len(data)} of {end - start + 1} bytes")
            return data
        except (OSError, urllib.error.URLError, ValueError) as error:
            if attempt == retries - 1:
                raise
            time.sleep(min(2 ** attempt, 15))
            if isinstance(error, urllib.error.HTTPError) and error.code in (403, 410):
                raise  # an expired signed URL; the caller re-resolves


def link_into_dest(target, dest, name):
    """Expose a staged shard under DEST; a store that is DEST itself needs no link."""
    if os.path.realpath(os.path.dirname(target)) == os.path.realpath(dest):
        return
    link = os.path.join(dest, name)
    if os.path.lexists(link):
        os.unlink(link)
    os.symlink(os.path.abspath(target), link)


def pwrite_all(fd, data, offset):
    view = memoryview(data)
    while view:
        written = os.pwrite(fd, view, offset)
        if written <= 0:
            raise OSError("pwrite wrote no bytes")
        view, offset = view[written:], offset + written


def merge(ranges):
    out = []
    for start, end in sorted(ranges):
        if out and start <= out[-1][1]:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end])
    return out


def select(tensors, args):
    keep = []
    for name, offset, length, dims in tensors:
        if args.all or any(re.search(p, name) for p in args.include):
            keep.append((name, offset, length, dims))
            continue
        if any(re.search(p, name) for p in args.exclude):
            continue
        match = re.match(r"blk\.(\d+)\.", name)
        if match:
            layer = int(match.group(1))
            if any(a <= layer < b for a, b in args.layers):
                keep.append((name, offset, length, dims))
        else:
            keep.append((name, offset, length, dims))
    return keep


def byte_ranges(keep, args):
    """Byte ranges of the kept tensors, narrowed to --rows slices where they match."""
    out = []
    for name, offset, length, dims in keep:
        rows = next(((a, b) for pattern, a, b in args.rows if re.search(pattern, name)), None)
        if rows is None or len(dims) < 2:
            out.append((offset, offset + length))
            continue
        ne1 = dims[1]
        slabs = 1
        for d in dims[2:]:
            slabs *= d
        # Rows are fixed-size and slabs contiguous; the tensor's byte count only
        # adds alignment padding (< one row), so floor division is exact.
        row_bytes = length // (ne1 * slabs)
        first, end = rows[0], min(rows[1], ne1)
        if not 0 <= first < end:
            raise ValueError(f"{name}: invalid --rows {rows} for {ne1} rows")
        slab = row_bytes * ne1
        for e in range(slabs):
            out.append((offset + e * slab + first * row_bytes, offset + e * slab + end * row_bytes))
    return out


def stage_shard(shard, args, stores, log):
    name = unquote(urlsplit(shard["url"]).path.rsplit("/", 1)[-1])
    size, digest = shard["bytes"], shard["sha256"]
    url = resolve(shard["url"])
    head = b""
    while True:
        head += fetch(url, len(head), min(size, len(head) + 16 * 1024 * 1024) - 1)
        try:
            data_start, tensors = parse_header(head, size)
            break
        except EOFError:
            if len(head) >= size:
                raise
    keep = select(tensors, args)
    if args.plan:
        selected = sum(e - s for s, e in merge([(0, data_start)] + byte_ranges(keep, args)))
        print(f"{name}: plan keeps {len(keep)}/{len(tensors)} tensors, {selected / 1e9:.2f} of {size / 1e9:.2f} GB")
        return {"file": name, "bytes": size, "selected_bytes": selected, "selected_tensors": len(keep), "plan": True}
    if not keep:
        # The loader still opens every shard of a split file, but reads no data
        # from this one: write only its header. Nothing here is hash-verified.
        os.makedirs(stores[-1]["path"], exist_ok=True)
        target = os.path.join(stores[-1]["path"], name)
        with open(target + ".staging", "wb") as f:
            f.truncate(size)
            f.write(head[:size] if len(head) >= size else head[:data_start])
        os.replace(target + ".staging", target)
        link_into_dest(target, args.dest, name)
        # A metadata-only shard is all header: then the whole file was fetched
        # and can be checked against the publisher hash.
        whole = len(head) >= size
        if whole and hashlib.sha256(head[:size]).hexdigest() != digest:
            os.unlink(target)
            raise RuntimeError(f"{name}: SHA-256 differs from pinned {digest}")
        row = {"file": name, "path": target, "bytes": size, "selected_bytes": data_start,
               "tensors": len(tensors), "selected_tensors": 0, "header_only": True, "verified": whole}
        print(json.dumps(row), file=log, flush=True)
        print(f"{name}: header only (no selected tensors)", flush=True)
        return row
    ranges = merge([(0, data_start)] + byte_ranges(keep, args))
    selected = sum(e - s for s, e in ranges)
    store = next((s for s in stores if s["free"] >= selected), None)
    if store is None:
        raise RuntimeError(f"{name}: no store has {selected / 1e9:.1f} GB free")
    store["free"] -= selected
    os.makedirs(store["path"], exist_ok=True)
    target = os.path.join(store["path"], name)
    # geesefs implements rename as an object copy; a 135 GB staged shard
    # vanished after its rename. --in-place writes the final name directly and
    # deletes it if streaming or verification fails.
    partial = target if args.in_place else target + ".staging"
    with open(partial, "wb") as f:
        # Object-store filesystems (e.g. geesefs) write a new file sequentially
        # far faster than they modify a preallocated one; whole-file copies are
        # written in order anyway, so the hole-preserving truncate is optional.
        if not args.no_preallocate:
            f.truncate(size)
    started = time.monotonic()
    hasher = hashlib.sha256()
    lock = threading.Lock()
    offsets = list(range(0, size, CHUNK))
    fd = os.open(partial, os.O_WRONLY)
    try:
        def job(offset):
            nonlocal url
            end = min(size, offset + CHUNK) - 1
            try:
                return offset, fetch(url, offset, end)
            except urllib.error.HTTPError as error:
                if error.code not in (403, 410):
                    raise
                with lock:
                    url = resolve(shard["url"])
                return offset, fetch(url, offset, end)

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.connections) as pool:
            # Bounded look-ahead: hash in order while later chunks download.
            pending = {}
            futures = {}
            next_submit = 0
            for _ in range(min(len(offsets), args.connections * 2)):
                futures[offsets[next_submit]] = pool.submit(job, offsets[next_submit])
                next_submit += 1
            for offset in offsets:
                _, data = futures.pop(offset).result()
                hasher.update(data)
                end = offset + len(data)
                for s, e in ranges:
                    if s >= end or e <= offset:
                        continue
                    lo, hi = max(s, offset), min(e, end)
                    pwrite_all(fd, data[lo - offset:hi - offset], lo)
                if next_submit < len(offsets):
                    futures[offsets[next_submit]] = pool.submit(job, offsets[next_submit])
                    next_submit += 1
                if offset // CHUNK % 64 == 0:
                    rate = (end / max(time.monotonic() - started, 1e-6)) / 1024**2
                    print(f"  {name}: {end / 1e9:.1f}/{size / 1e9:.1f} GB streamed ({rate:.0f} MiB/s)", flush=True)
        os.fsync(fd)
    except BaseException:
        # An interrupted in-place copy must not remain under the final name.
        if args.in_place:
            os.close(fd)
            fd = -1
            os.unlink(partial)
        raise
    finally:
        if fd >= 0:
            os.close(fd)
    actual = hasher.hexdigest()
    if actual != digest:
        os.unlink(partial)
        raise RuntimeError(f"{name}: SHA-256 {actual} differs from pinned {digest}")
    if partial != target:
        os.replace(partial, target)
    link_into_dest(target, args.dest, name)
    seconds = time.monotonic() - started
    row = {"file": name, "path": target, "bytes": size, "selected_bytes": selected,
           "tensors": len(tensors), "selected_tensors": len(keep), "sha256": actual,
           "verified": True, "seconds": round(seconds, 1),
           "MiB_per_second": round(size / max(seconds, 1e-6) / 1024**2, 1)}
    print(json.dumps(row), file=log, flush=True)
    print(f"{name}: verified; kept {len(keep)}/{len(tensors)} tensors, "
          f"{selected / 1e9:.1f} GB in {store['path']} ({row['MiB_per_second']} MiB/s)", flush=True)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifest")
    parser.add_argument("--dest", required=True, help="model directory the loader opens (symlinks)")
    parser.add_argument("--store", action="append", required=True, help="PATH:GB capacity to use, in fill order")
    parser.add_argument("--layers", action="append", default=[], help="A:B keeps blk.A..blk.(B-1)")
    parser.add_argument("--include", action="append", default=[], help="regex of tensor names always kept")
    parser.add_argument("--exclude", action="append", default=[], help="regex of non-layer tensors dropped")
    parser.add_argument("--all", action="store_true", help="keep every tensor")
    parser.add_argument("--rows", action="append", default=[], help="REGEX:FIRST:END row range per expert slab")
    parser.add_argument("--plan", action="store_true", help="read headers only and print what would be staged")
    parser.add_argument("--no-preallocate", action="store_true",
                        help="do not truncate to full size first (use with --all on object-store filesystems)")
    parser.add_argument("--in-place", action="store_true",
                        help="write the final file name directly instead of renaming a .staging file "
                             "(object-store filesystems copy on rename); deleted again on failure")
    parser.add_argument("--connections", type=int, default=8)
    parser.add_argument("--report", help="JSON report path (default DEST/stage-report.json)")
    args = parser.parse_args()
    args.layers = [tuple(int(x) for x in item.split(":")) for item in args.layers]
    if not (args.all or args.layers or args.include):
        parser.error("select tensors with --all, --layers or --include (block tensors are otherwise skipped)")
    if args.no_preallocate and not args.all:
        parser.error("--no-preallocate writes whole files in order; combine it with --all")
    args.rows = [(item.rsplit(":", 2)[0], int(item.rsplit(":", 2)[1]), int(item.rsplit(":", 2)[2])) for item in args.rows]
    stores = []
    for item in args.store:
        path, _, gb = item.rpartition(":")
        stores.append({"path": os.path.abspath(path), "free": int(float(gb) * 1e9)})
    manifest = json.load(open(args.manifest, encoding="utf-8-sig"))
    if args.plan:
        # A preview reads headers only and must not touch DEST or its report.
        for shard in manifest["shards"]:
            stage_shard(shard, args, stores, sys.stdout)
        return 0
    os.makedirs(args.dest, exist_ok=True)
    report = {"manifest": os.path.abspath(args.manifest), "repo": manifest.get("repo"),
              "revision": manifest.get("revision"), "layers": args.layers, "include": args.include,
              "exclude": args.exclude, "all": args.all, "status": "running", "shards": []}
    report_path = args.report or os.path.join(args.dest, "stage-report.json")
    started = time.time()
    with open(report_path + ".log", "a", encoding="utf-8") as log:
        try:
            for shard in manifest["shards"]:
                report["shards"].append(stage_shard(shard, args, stores, log))
            report["status"] = "verified"
        except Exception as error:
            report.update(status="failed", error=repr(error))
            raise
        finally:
            report["wall_seconds"] = round(time.time() - started, 1)
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
