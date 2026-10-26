#!/usr/bin/env python3
"""Linux-only, CPU-only fresh/reused anonymous mapping copy probe.

Never changes system THP settings. Each mode maps owned destination buffers,
optionally requests MADV_HUGEPAGE, copies distinct source blocks, validates every
byte with memcmp, then repeats the copy on already faulted destination pages.
This is not a CUDA transfer or end-to-end inference benchmark.
"""
import argparse
import ctypes
import json
import mmap
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    try:
        return Path(path).read_text().strip()
    except OSError as error:
        return "unavailable: " + str(error)


def address(mapping):
    return ctypes.addressof(ctypes.c_char.from_buffer(mapping))


def huge_kib(regions):
    selected = False
    total = 0
    try:
        for line in Path('/proc/self/smaps').read_text().splitlines():
            first = line.split()[0]
            if '-' in first:
                try:
                    begin, end = (int(value, 16) for value in first.split('-'))
                except ValueError:
                    continue
                selected = any(begin < start + size and end > start for start, size in regions)
            elif selected and line.startswith('AnonHugePages:'):
                total += int(line.split()[1])
        return total
    except OSError:
        return None


def measure_copy(destinations, source, size):
    before = resource.getrusage(resource.RUSAGE_SELF)
    started = time.perf_counter()
    for i, destination in enumerate(destinations):
        ctypes.memmove(destination, source + i * size, size)
    elapsed = time.perf_counter() - started
    after = resource.getrusage(resource.RUSAGE_SELF)
    return {'wall_ms': elapsed * 1000, 'user_ms': (after.ru_utime - before.ru_utime) * 1000,
            'system_ms': (after.ru_stime - before.ru_stime) * 1000,
            'minor_faults': after.ru_minflt - before.ru_minflt,
            'major_faults': after.ru_majflt - before.ru_majflt}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--blocks', type=int, default=36)
    parser.add_argument('--block-mib', type=int, default=40)
    parser.add_argument('--iterations', type=int, default=3)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != 'linux':
        parser.error('Linux is required; other platforms are not passing coverage')
    if not 1 <= args.blocks <= 128 or not 1 <= args.block_mib <= 128 or not 1 <= args.iterations <= 20 or args.blocks * args.block_mib > 2048:
        parser.error('Use 1..128 blocks of 1..128 MiB, <=2048 MiB total, and 1..20 iterations')
    output = args.output.resolve()
    if not any(output.is_relative_to(ROOT / folder) for folder in ('artifacts', 'docs/validation')):
        parser.error('Output must be in ignored artifacts/ or docs/validation/')
    output.parent.mkdir(parents=True, exist_ok=True)
    size = args.block_mib << 20
    total = size * args.blocks
    libc = ctypes.CDLL(None)
    libc.memcmp.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    libc.memcmp.restype = ctypes.c_int
    report = {'complete': False, 'passed': False, 'blocks': args.blocks, 'block_bytes': size,
              'total_bytes': total, 'iterations': args.iterations, 'rows': [],
              'thp_enabled': read('/sys/kernel/mm/transparent_hugepage/enabled'),
              'thp_defrag': read('/sys/kernel/mm/transparent_hugepage/defrag'),
              'pmd_bytes': read('/sys/kernel/mm/transparent_hugepage/hpage_pmd_size'),
              'loadavg_before': read('/proc/loadavg'), 'kernel': read('/proc/sys/kernel/osrelease'),
              'scope': 'Single-process CPU memcpy with full memcmp, no GPU. Source is explicitly NOHUGEPAGE. No global OS settings modified.'}
    source = None
    started = time.perf_counter()
    try:
        source = mmap.mmap(-1, total, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS,
                           prot=mmap.PROT_READ | mmap.PROT_WRITE)
        source.madvise(mmap.MADV_NOHUGEPAGE)
        src = address(source)
        for i in range(args.blocks):
            ctypes.memset(src + i * size, 1 + i % 254, size)
        for repetition in range(args.iterations):
            modes = ('default', 'hugepage') if repetition % 2 == 0 else ('hugepage', 'default')
            for mode in modes:
                mappings = []
                pointers = []
                try:
                    allocate_start = time.perf_counter()
                    for _ in range(args.blocks):
                        mapped = mmap.mmap(-1, size, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS,
                                           prot=mmap.PROT_READ | mmap.PROT_WRITE)
                        mappings.append(mapped)
                        if mode == 'hugepage':
                            mapped.madvise(mmap.MADV_HUGEPAGE)
                        pointers.append(address(mapped))
                    allocation_ms = (time.perf_counter() - allocate_start) * 1000
                    fresh = measure_copy(pointers, src, size)
                    fresh_correct = all(libc.memcmp(dst, src + i * size, size) == 0 for i, dst in enumerate(pointers))
                    huge = huge_kib([(dst, size) for dst in pointers])
                    reused = measure_copy(pointers, src, size)
                    reused_correct = all(libc.memcmp(dst, src + i * size, size) == 0 for i, dst in enumerate(pointers))
                    report['rows'].append({'iteration': repetition, 'mode': mode, 'allocation_ms': allocation_ms,
                                           'fresh_copy': fresh, 'reused_copy': reused,
                                           'anon_huge_pages_kib': huge, 'aligned_2m_buffers': sum(ptr % (2 << 20) == 0 for ptr in pointers),
                                           'fresh_copy_correct': fresh_correct, 'reused_copy_correct': reused_correct})
                    if not fresh_correct or not reused_correct:
                        raise ValueError('memcmp found corrupt copied bytes')
                finally:
                    for mapped in mappings:
                        mapped.close()
        report.update(complete=True, passed=True)
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
    finally:
        if source is not None:
            source.close()
        report['wall_seconds'] = time.perf_counter() - started
        report['loadavg_after'] = read('/proc/loadavg')
        output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'output': str(output), 'passed': report['passed'], 'wall_seconds': report['wall_seconds']}))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
