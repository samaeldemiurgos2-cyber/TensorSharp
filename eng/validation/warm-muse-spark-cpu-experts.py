#!/usr/bin/env python3
"""Warm only the first N CPU-offloaded expert layers in the shared file cache.

Run AFTER each engine has uploaded its GPU weights, before measured requests.
Startup and this explicit prewarm must be reported separately from throughput.
Requires unchanged llama.cpp's gguf-py on PYTHONPATH. --check-only reads headers
and queries mincore without touching tensor data. No model files are modified.
The default pread-mmap-touch method bulk-reads bounded chunks, then references
their pages through a temporary read-only mapping. This avoids sparse NFS page
faults while encouraging retention over recently uploaded GPU source pages.
Always inspect post-unmap residency: completing a read does not prove that all
expert pages remain resident under the process/container memory limit.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes
import json
import mmap
import os
from pathlib import Path
import re
import sys
import threading
import time


def discover_shards(first):
    first = Path(first).resolve()
    match = re.fullmatch(r"(.+)-00001-of-(\d{5})\.gguf", first.name)
    if not match or int(match[2]) < 1:
        raise ValueError("--model must name the first numbered GGUF shard")
    paths = [first.with_name(f"{match[1]}-{i:05d}-of-{int(match[2]):05d}.gguf")
             for i in range(1, int(match[2]) + 1)]
    for path in paths:
        if not path.is_file():
            raise ValueError(f"Missing shard: {path}")
    return paths


def plan_ranges(paths, cpu_moe, reader_factory):
    ranges, seen = [], set()
    layers = None
    for path in paths:
        reader = reader_factory(path)
        if layers is None:
            arch = reader.fields["general.architecture"].contents()
            if arch not in ("muse-spark", "muse_spark"):
                raise ValueError(f"Unexpected architecture: {arch}")
            layers = int(reader.fields[arch + ".block_count"].contents())
            if not 0 <= cpu_moe <= layers:
                raise ValueError("CPU expert count exceeds the model layer count")
        if "split.count" in reader.fields and int(reader.fields["split.count"].contents()) != len(paths):
            raise ValueError("GGUF split.count does not match the complete shard set")
        stat = path.stat()
        for tensor in reader.tensors:
            match = re.fullmatch(r"blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight", tensor.name)
            if not match or int(match[1]) >= cpu_moe:
                continue
            if tensor.name in seen:
                raise ValueError(f"Duplicate tensor: {tensor.name}")
            seen.add(tensor.name)
            offset, size = int(tensor.data_offset), int(tensor.n_bytes)
            if offset < int(reader.data_offset) or size <= 0 or offset > stat.st_size or size > stat.st_size - offset:
                raise ValueError(f"Tensor range exceeds its shard: {tensor.name}")
            ranges.append(dict(path=str(path), tensor=tensor.name, layer=int(match[1]),
                               projection=match[2], offset=offset, bytes=size,
                               ggml_type=tensor.tensor_type.name, file_size=stat.st_size,
                               mtime_ns=stat.st_mtime_ns))
    expected = {f"blk.{layer}.ffn_{part}_exps.weight" for layer in range(cpu_moe)
                for part in ("gate", "up", "down")}
    if seen != expected:
        raise ValueError(f"Missing selected expert tensors: {sorted(expected - seen)}")
    ranges.sort(key=lambda item: (item["path"], item["offset"]))
    for left, right in zip(ranges, ranges[1:]):
        if left["path"] == right["path"] and left["offset"] + left["bytes"] > right["offset"]:
            raise ValueError("Selected tensor ranges overlap")
    return ranges


def verify_file(fd, item):
    stat = os.fstat(fd)
    if stat.st_size != item["file_size"] or stat.st_mtime_ns != item["mtime_ns"]:
        raise ValueError(f"Shard changed after header inspection: {item['path']}")


def residency(ranges):
    if sys.platform != "linux":
        return {"status": "unavailable", "reason": "mincore requires Linux"}
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                          ctypes.c_int, ctypes.c_int, ctypes.c_long]
    libc.mmap.restype = ctypes.c_void_p
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
    libc.mincore.restype = ctypes.c_int
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    libc.munmap.restype = ctypes.c_int
    page_size = os.sysconf("SC_PAGE_SIZE")
    rows = []
    for item in ranges:
        offset = item["offset"] // page_size * page_size
        length = item["offset"] - offset + item["bytes"]
        pages = (length + page_size - 1) // page_size
        fd = os.open(item["path"], os.O_RDONLY)
        try:
            verify_file(fd, item)
            # PROT_NONE, MAP_SHARED: mincore inspects residency without a read,
            # write, page fault or anonymous copy of the mapped tensor bytes.
            address = libc.mmap(None, length, 0, 1, fd, offset)
            if address == ctypes.c_void_p(-1).value:
                raise OSError(ctypes.get_errno(), "mmap for mincore failed")
            try:
                bits = (ctypes.c_ubyte * pages)()
                if libc.mincore(address, length, bits):
                    raise OSError(ctypes.get_errno(), "mincore failed")
                resident = sum(value & 1 for value in bits)
            finally:
                libc.munmap(address, length)
            rows.append(dict(tensor=item["tensor"], pages=pages, resident_pages=resident))
        finally:
            os.close(fd)
    pages = sum(row["pages"] for row in rows)
    resident = sum(row["resident_pages"] for row in rows)
    return dict(status="measured", page_size=page_size, pages=pages,
                resident_pages=resident, resident_fraction=resident / pages if pages else 1.0,
                ranges=rows,
                note="Page-aligned counts include boundary padding; residency is a point-in-time observation.")


def process_io():
    try:
        return {key: int(value) for key, value in
                (line.split(":", 1) for line in Path("/proc/self/io").read_text().splitlines())}
    except (OSError, ValueError):
        return None


def touch_chunk(fd, offset, count, np):
    aligned = offset // mmap.ALLOCATIONGRANULARITY * mmap.ALLOCATIONGRANULARITY
    prefix = offset - aligned
    with mmap.mmap(fd, prefix + count, access=mmap.ACCESS_READ, offset=aligned) as mapped:
        values = np.frombuffer(mapped, dtype=np.uint8)
        try:
            # The first sample is inside the tensor; subsequent samples start
            # at OS page boundaries. This also covers a partial final page
            # when an unaligned tensor/chunk crosses a page boundary.
            first_boundary = (prefix // mmap.PAGESIZE + 1) * mmap.PAGESIZE
            checksum = int(values[prefix]) + int(values[first_boundary::mmap.PAGESIZE].sum(dtype=np.uint64))
            samples = 1 + max(0, (len(values) - first_boundary + mmap.PAGESIZE - 1) // mmap.PAGESIZE)
            return samples, checksum
        finally:
            # Release every NumPy view before closing the read-only mapping.
            del values


def warm_ranges(ranges, chunk_bytes, progress=None, workers=1, read_mode="pread"):
    if chunk_bytes <= 0 or not 1 <= workers <= 8:
        raise ValueError("Chunk size must be positive and workers must be between 1 and 8")
    if read_mode not in ("pread", "mmap-touch", "pread-mmap-touch"):
        raise ValueError("Unknown read mode")
    np = None
    if read_mode != "pread":
        import numpy as np
    before_io = process_io()
    start = time.monotonic()
    stop = threading.Event()
    def read_range(item):
        if stop.is_set():
            raise InterruptedError("Preparation cancelled")
        fd = os.open(item["path"], os.O_RDONLY)
        try:
            verify_file(fd, item)
            offset, remaining = item["offset"], item["bytes"]
            bytes_read = checksum = sampled_bytes = 0
            while remaining:
                if stop.is_set():
                    raise InterruptedError("Preparation cancelled")
                count = min(remaining, chunk_bytes)
                # pread avoids reading intervening dense/GPU tensors. Buffers
                # are discarded each iteration; memory usage is O(chunk size).
                if read_mode != "mmap-touch":
                    data = os.pread(fd, count, offset)
                    if not data:
                        raise IOError(f"Unexpected EOF in {item['tensor']}")
                    count = len(data)
                    bytes_read += count
                    del data
                if read_mode != "pread":
                    # Bulk-read first in hybrid mode so sparse NFS holes do
                    # not each incur a synchronous mapped-page fault. The
                    # temporary read buffer is gone before the mapping opens.
                    samples, sample_sum = touch_chunk(fd, offset, count, np)
                    sampled_bytes += samples
                    checksum += sample_sum
                offset += count
                remaining -= count
            verify_file(fd, item)
        finally:
            os.close(fd)
        return item["bytes"], bytes_read, checksum, sampled_bytes
    total = bytes_read = checksum = sampled_bytes = 0
    # Results contain only counts. Progress and accounting run on this thread;
    # each worker owns one FD and at most one bounded byte buffer at a time.
    with ThreadPoolExecutor(max_workers=workers) as executor:
        try:
            for index, (count, read_count, sample_sum, samples) in enumerate(executor.map(read_range, ranges)):
                total += count
                bytes_read += read_count
                checksum += sample_sum
                sampled_bytes += samples
                if progress:
                    progress(index + 1, len(ranges), ranges[index], total, time.monotonic() - start)
        except BaseException:
            # Ctrl-C/failure must not drain all queued expert tensors. Running
            # workers finish at most their current chunk and close their FD/map.
            stop.set()
            executor.shutdown(wait=True, cancel_futures=True)
            raise
    after_io = process_io()
    return dict(bytes_covered=total, bytes_read=bytes_read, bytes_sampled=sampled_bytes, read_mode=read_mode,
                bytes_read_definition="pread returned bytes; bytes_sampled counts one scalar byte per covered OS page per mapped chunk; neither equals physical I/O",
                sample_checksum=checksum if read_mode != "pread" else None,
                process_io_before=before_io, process_io_after=after_io,
                process_io_delta={key: after_io[key] - before_io[key] for key in before_io} if before_io and after_io else None,
                duration_seconds=time.monotonic() - start, completed_ranges=len(ranges),
                chunk_bytes=chunk_bytes, workers=workers)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="First numbered GGUF shard")
    parser.add_argument("--cpu-moe", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true", help="Query residency without warming tensor bytes")
    parser.add_argument("--chunk-mib", type=int, default=8)
    parser.add_argument("--workers", type=int, default=1, help="Independent range readers (1..8); memory is workers times chunk size")
    parser.add_argument("--read-mode", choices=("pread", "mmap-touch", "pread-mmap-touch"), default="pread-mmap-touch", help="Default: pread-mmap-touch. Bulk-read then touch each chunk; pread and mmap-touch are diagnostic alternatives")
    args = parser.parse_args()
    if not 1 <= args.chunk_mib <= 64:
        parser.error("--chunk-mib must be between 1 and 64")
    if not 1 <= args.workers <= 8:
        parser.error("--workers must be between 1 and 8")
    import gguf
    report = dict(mode="check-only" if args.check_only else "prewarm", model=str(args.model),
                  cpu_moe_layers=args.cpu_moe, timing_scope="Explicit file-cache preparation; excluded from inference timing",
                  started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    exit_code = 0
    try:
        ranges = plan_ranges(discover_shards(args.model), args.cpu_moe, gguf.GGUFReader)
        report.update(ranges=ranges, selected_bytes=sum(item["bytes"] for item in ranges),
                      before=residency(ranges))
        if not args.check_only:
            def progress(done, count, item, total, elapsed):
                print(f"{done}/{count} {item['tensor']} total={total / 1024**3:.3f}GiB elapsed={elapsed:.1f}s", file=sys.stderr, flush=True)
            report["warmup"] = warm_ranges(ranges, args.chunk_mib * 1024**2, progress, args.workers, args.read_mode)
            report["after"] = residency(ranges)
        report["status"] = "checked" if args.check_only else "completed"
    except (Exception, KeyboardInterrupt) as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        exit_code = 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "selected_bytes", "warmup", "error") if key in report}))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
