#!/usr/bin/env python3
"""Trace sink: keep only the trace files an oracle window needs, hash every file version the engine writes.
Linux only (inotify, SIGSTOP/SIGCONT).

    dsv41-trace-sink.py --stage TMPFS_DIR --dest KEPT_DIR --keep REGEX [--keep REGEX ...]
                        --digest OUT.digest --log OUT.log --stop-file F [--pid-file F]
                        [--max-stage-bytes N] [--max-dest-bytes N] [--pause-bytes N] [--resume-bytes N]
                        [--index-ratio LAYER:RATIO ... --prefix N --chunk N --topk N] [--writers N]

Neither engine's trace can be filtered by layer or stage (TS_DSV4_CUDA_TRACE_DIR, TS_DSV41_TRACE_DIR write every
trace point; ggml also writes its view tensors), and a 4K-token prompt traces ~25 GB per run. The engine writes into
--stage (tmpfs); this process takes each file as soon as the writer closes it (inotify IN_CLOSE_WRITE, plus a
rescan for files quiet for 3 s): it renames it to a private name, hashes the bytes (sha256), copies it to --dest
when its name matches a --keep pattern (atomic replace, so a later pass's version overwrites an earlier one), and
deletes it. Nothing in the engine changes.

--digest lists the sha256 of the LAST version of every file that was NOT kept ("<sha256>  <tag>  <name>", the
dsv41-trace-digest.py format); the kept files are digested from --dest afterwards. The direct engine's sparse-selection
dumps hold memory it never wrote (index_scores is [nt, cache rows], only each query's visible (p + 1) // ratio rows
are written; index_top holds index_count = min(k, visible) ids per row): with --index-ratio LAYER:RATIO, --prefix
and --chunk the sink hashes a dropped dump over its written part exactly as dsv41-trace-digest.py hashes a kept one
(nt = min(chunk, prefix - p0) below the prefix, 1 above; index_top over min(--topk, visible) ids per row). Together
the two digests cover every file of the run, so two runs can be compared for bit-identity although neither was
stored.

Flow control: copying kept files to a network filesystem is slower than an engine can trace, so when --stage holds
more than --pause-bytes the process in --pid-file is stopped (SIGSTOP) until the sink has drained it below
--resume-bytes (SIGCONT); the engine's arithmetic does not depend on its timing. Kept files are written by
--writers threads, each file name always to the same thread, so the versions of one file land in order.
Guard: when --stage holds more than --max-stage-bytes or --dest more than --max-dest-bytes, the process in
--pid-file is killed (SIGKILL) and the sink exits 3. It exits when --stop-file appears, after a final sweep.
"""
import argparse
import ctypes
import hashlib
import os
import re
import select
import shutil
import signal
import queue
import struct
import sys
import threading
import time

IN_CLOSE_WRITE, IN_OPEN, IN_MOVED_TO, IN_Q_OVERFLOW = 0x8, 0x20, 0x80, 0x4000
PRIVATE = ".sink."


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--keep", action="append", default=[])
    ap.add_argument("--digest", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--stop-file", required=True)
    ap.add_argument("--pid-file")
    ap.add_argument("--max-stage-bytes", type=float, default=6e9)
    ap.add_argument("--max-dest-bytes", type=float, default=11e9)
    ap.add_argument("--pause-bytes", type=float, default=4e9)
    ap.add_argument("--resume-bytes", type=float, default=1.5e9)
    ap.add_argument("--writers", type=int, default=4)
    ap.add_argument("--index-ratio", action="append", default=[], help="LAYER:RATIO of a direct-engine index layer")
    ap.add_argument("--prefix", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--topk", type=int, default=512)
    args = ap.parse_args()
    if not sys.platform.startswith("linux"):
        raise SystemExit("dsv41-trace-sink.py needs Linux (inotify and SIGSTOP/SIGCONT flow control)")
    ratios = {int(a): int(b) for a, b in (x.split(":") for x in args.index_ratio)}
    managed = re.compile(r"p(\d+)_blk(\d+)_(index_scores|index_top)\.(f32|i32)$")
    keep = [re.compile(k) for k in args.keep]
    os.makedirs(args.stage, exist_ok=True)
    os.makedirs(args.dest, exist_ok=True)
    log = open(args.log, "a", buffering=1)

    def say(msg):
        log.write(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}\n")

    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    fd = libc.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
    if fd < 0 or libc.inotify_add_watch(fd, args.stage.encode(), IN_CLOSE_WRITE | IN_OPEN | IN_MOVED_TO) < 0:
        raise SystemExit(f"inotify failed: errno {ctypes.get_errno()}")
    final, versions = {}, {}
    opens, closes = {}, {}            # per name: a newer version may be open when an older close is handled
    stats = {"files": 0, "versions": 0, "bytes": 0, "kept_versions": 0, "kept_bytes_written": 0, "warnings": 0,
             "max_stage": 0, "overflows": 0}
    dest_sizes = {}
    counter = [0]
    paused = [False, 0, 0.0]          # paused now, pauses so far, seconds paused
    queues = [queue.Queue() for _ in range(max(1, args.writers))]
    errors = []

    def writer(qu):
        while True:
            item = qu.get()
            if item is None:
                qu.task_done()
                return
            name, priv = item
            try:
                tmp = os.path.join(args.dest, f".{name}.tmp")
                shutil.copyfile(priv, tmp)
                os.replace(tmp, os.path.join(args.dest, name))
                os.unlink(priv)
            except Exception as ex:      # noqa: BLE001 - reported, and the sink exits nonzero
                errors.append(f"{name}: {ex}")
            qu.task_done()

    threads = [threading.Thread(target=writer, args=(qu,), daemon=True) for qu in queues]
    for t in threads:
        t.start()

    def probe_pid():
        try:
            return int(open(args.pid_file).read().split()[0]) if args.pid_file else None
        except (OSError, ValueError, IndexError):
            return None

    def flow(total):
        pid = probe_pid()
        if pid is None:
            return
        try:
            if not paused[0] and total > args.pause_bytes:
                os.kill(pid, signal.SIGSTOP)
                paused[0], paused[1], paused[2] = True, paused[1] + 1, paused[2] - time.time()
            elif paused[0] and total < args.resume_bytes:
                os.kill(pid, signal.SIGCONT)
                paused[0], paused[2] = False, paused[2] + time.time()
        except ProcessLookupError:
            paused[0] = False

    def process(name):
        if name.startswith(PRIVATE) or name.endswith(".tmp"):
            return
        src = os.path.join(args.stage, name)
        counter[0] += 1
        priv = os.path.join(args.stage, f"{PRIVATE}{counter[0]}")
        try:
            os.rename(src, priv)
        except FileNotFoundError:
            return
        st0 = os.stat(priv)
        h = hashlib.sha256()
        with open(priv, "rb") as f:
            data = f.read()
        mm = managed.match(name)
        kept = any(k.search(name) for k in keep)
        tag = "bytes"
        if mm and not kept and int(mm.group(2)) in ratios and args.prefix > 0:
            p0, ratio = int(mm.group(1)), ratios[int(mm.group(2))]
            nt = min(args.chunk, args.prefix - p0) if p0 < args.prefix else 1
            row = len(data) // nt
            for q in range(nt):
                vis = (p0 + q + 1) // ratio
                n = vis if mm.group(3) == "index_scores" else min(args.topk, vis)
                h.update(data[q * row:q * row + 4 * n])
            tag = "written-part"
        else:
            h.update(data)
        st1 = os.stat(priv)
        if (st0.st_size, st0.st_mtime_ns) != (st1.st_size, st1.st_mtime_ns) or len(data) != st1.st_size:
            # a newer version is being written into this inode: give the name back and let its close finalize it
            stats["restored"] = stats.get("restored", 0) + 1
            try:
                os.rename(priv, src)
                say(f"RESTORED {name}: still being written ({st0.st_size} -> {st1.st_size} B)")
                return
            except OSError as ex:
                stats["warnings"] += 1
                say(f"WARNING {name}: changed while read and not restorable ({ex})")
        digest = h.hexdigest()
        versions[name] = versions.get(name, 0) + 1
        stats["versions"] += 1
        stats["bytes"] += len(data)
        if kept:
            dest_sizes[name] = len(data)
            final.pop(name, None)
            stats["kept_versions"] += 1
            stats["kept_bytes_written"] += len(data)
            queues[hash(name) % len(queues)].put((name, priv))   # the writer copies, then removes priv
        else:
            final[name] = (digest, tag)
            os.unlink(priv)

    def kill(reason):
        say(f"GUARD: {reason}")
        if args.pid_file and os.path.exists(args.pid_file):
            try:
                pid = int(open(args.pid_file).read().split()[0])
                os.kill(pid, signal.SIGCONT)
                os.kill(pid, signal.SIGKILL)
                say(f"killed pid {pid}")
            except (ValueError, ProcessLookupError, PermissionError) as ex:
                say(f"kill failed: {ex}")
        finish(3)

    def sweep(min_age):
        now = time.time()
        total = 0
        with os.scandir(args.stage) as it:
            entries = [e for e in it if e.is_file(follow_symlinks=False)]
        for e in entries:
            try:
                s = e.stat()
            except FileNotFoundError:
                continue
            total += s.st_size
            if not e.name.startswith(PRIVATE) and now - s.st_mtime >= min_age and \
                    (min_age == 0 or opens.get(e.name, 0) <= closes.get(e.name, 0)):
                process(e.name)
        stats["max_stage"] = max(stats["max_stage"], total)
        flow(total)
        if total > args.max_stage_bytes:
            kill(f"stage holds {total} B > {args.max_stage_bytes:.0f}")
        dest_total = sum(dest_sizes.values())
        if dest_total > args.max_dest_bytes:
            kill(f"dest holds {dest_total} B > {args.max_dest_bytes:.0f}")

    def finish(rc):
        for qu in queues:
            qu.put(None)
        for qu in queues:
            qu.join()
        if paused[0]:
            pid = probe_pid()
            if pid:
                try:
                    os.kill(pid, signal.SIGCONT)
                except ProcessLookupError:
                    pass
            paused[2] += time.time()
        if errors:
            say(f"WRITE ERRORS ({len(errors)}): {errors[:3]}")
            rc = rc or 4
        stats["files"] = len(versions)
        stats["pauses"], stats["paused_seconds"] = paused[1], round(paused[2], 1)
        with open(args.digest, "w") as f:
            for name in sorted(final):
                f.write(f"{final[name][0]}  {final[name][1]:12s} {name}\n")
        vc = {}
        for v in versions.values():
            vc[v] = vc.get(v, 0) + 1
        say(f"done rc {rc}: {stats}; kept files {len(dest_sizes)} ({sum(dest_sizes.values())} B); "
            f"versions per file {dict(sorted(vc.items()))}")
        log.close()
        sys.exit(rc)

    say(f"start: stage {args.stage} dest {args.dest} keep {args.keep}")
    buf_size = 1 << 20
    last_sweep = 0.0
    while True:
        r, _, _ = select.select([fd], [], [], 0.5)
        if r:
            try:
                buf = os.read(fd, buf_size)
            except BlockingIOError:
                buf = b""
            off = 0
            ready = []
            while off + 16 <= len(buf):
                _wd, mask, _cookie, ln = struct.unpack_from("iIII", buf, off)
                name = buf[off + 16:off + 16 + ln].split(b"\0", 1)[0].decode("utf-8", "surrogateescape")
                off += 16 + ln
                if mask & IN_Q_OVERFLOW:
                    stats["overflows"] += 1
                    continue
                if not name or name.startswith(PRIVATE):
                    continue
                if mask & IN_OPEN:
                    opens[name] = opens.get(name, 0) + 1
                if mask & (IN_CLOSE_WRITE | IN_MOVED_TO):
                    closes[name] = closes.get(name, 0) + 1
                    ready.append(name)
            for name in dict.fromkeys(ready):
                if opens.get(name, 0) <= closes.get(name, 0):   # no newer version of this name is open
                    process(name)
        if time.time() - last_sweep > 2.0:
            sweep(3.0)
            last_sweep = time.time()
        if os.path.exists(args.stop_file):
            sweep(0.0)
            for qu in queues:
                qu.join()
            leftover = [n for n in os.listdir(args.stage) if not n.startswith(PRIVATE)]
            if leftover:
                say(f"WARNING leftover after final sweep: {leftover[:5]}")
            shutil.rmtree(args.stage, ignore_errors=True)
            finish(0)


if __name__ == "__main__":
    main()
