#!/usr/bin/env python3
"""Report, evict or read model files through the OS file cache, for cold/warm benchmark arms.

  file-cache.py report FILE...            cached bytes per file (mincore)
  file-cache.py evict FILE...             drop each file's cached pages, then report
  file-cache.py read --readers N FILE...  cold-or-warm read throughput with N concurrent 8 MiB readers

Eviction uses msync(MS_INVALIDATE) on macOS and posix_fadvise(POSIX_FADV_DONTNEED) on
Linux; both only discard clean pages of read-only files. Network filesystems may keep
their own client caches, so check `report` after `evict` before calling an arm cold.
"""
import argparse
import ctypes
import ctypes.util
import mmap
import os
import sys
import threading
import time

PAGE = os.sysconf("SC_PAGE_SIZE")
libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
libc.mmap.restype = ctypes.c_void_p
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_longlong]
libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]
libc.msync.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]


def resident_bytes(path):
    size = os.path.getsize(path)
    if size == 0:
        return 0, 0
    fd = os.open(path, os.O_RDONLY)
    try:
        base = libc.mmap(None, size, mmap.PROT_READ, mmap.MAP_SHARED, fd, 0)
        if base in (None, ctypes.c_void_p(-1).value):
            raise OSError(ctypes.get_errno(), "mmap", path)
        try:
            vector = ctypes.create_string_buffer((size + PAGE - 1) // PAGE)
            if libc.mincore(base, size, vector) != 0:
                raise OSError(ctypes.get_errno(), "mincore", path)
            return size, sum(1 for b in vector.raw if b & 1) * PAGE
        finally:
            libc.munmap(base, size)
    finally:
        os.close(fd)


def evict(path):
    size = os.path.getsize(path)
    fd = os.open(path, os.O_RDONLY)
    try:
        if sys.platform == "darwin":
            base = libc.mmap(None, size, mmap.PROT_READ, mmap.MAP_SHARED, fd, 0)
            try:
                if libc.msync(base, size, 0x2) != 0:  # MS_INVALIDATE
                    raise OSError(ctypes.get_errno(), "msync", path)
            finally:
                libc.munmap(base, size)
        else:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def read(path, readers, chunk=8 << 20):
    size = os.path.getsize(path)
    fd = os.open(path, os.O_RDONLY)
    position = [0]
    lock = threading.Lock()

    def worker():
        buffer = bytearray(chunk)
        while True:
            with lock:
                offset = position[0]
                position[0] += chunk
            if offset >= size:
                return
            os.preadv(fd, [memoryview(buffer)[:min(chunk, size - offset)]], offset)

    started = time.perf_counter()
    threads = [threading.Thread(target=worker) for _ in range(readers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    os.close(fd)
    return size, time.perf_counter() - started


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["report", "evict", "read"])
    parser.add_argument("files", nargs="+")
    parser.add_argument("--readers", type=int, default=1)
    args = parser.parse_args()
    for path in args.files:
        if args.action == "evict":
            evict(path)
        if args.action == "read":
            size, seconds = read(path, max(1, args.readers))
            print(f"{size / 2**30:8.2f} GiB in {seconds:7.2f} s = {size / 2**30 / seconds:6.2f} GiB/s"
                  f"  readers={args.readers}  {path}")
        else:
            size, cached = resident_bytes(path)
            print(f"{cached / 2**30:8.2f} / {size / 2**30:8.2f} GiB cached  {path}")


if __name__ == "__main__":
    main()
