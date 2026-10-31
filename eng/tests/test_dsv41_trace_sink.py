#!/usr/bin/env python3
"""Tests for eng/validation/dsv41-trace-sink.py and its digest companion eng/validation/dsv41-trace-digest.py.

    python -m unittest eng/tests/test_dsv41_trace_sink.py      (from the repository root)
    python -I eng/tests/test_dsv41_trace_sink.py -v

The sink is Linux only (inotify, /proc, SIGSTOP/SIGCONT); its tests are skipped elsewhere. A synthetic writer
stands in for the engine. The digest tool's tests run on every platform; the agreement test (a dropped file's sink
digest equals the digest tool's digest of the kept file) needs the sink, so it is Linux only too. The tests that
write synthetic index dumps need numpy and are skipped without it; the tools themselves are stdlib only.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import unittest

VALIDATION = Path(__file__).resolve().parents[1] / "validation"
SINK = str(VALIDATION / "dsv41-trace-sink.py")
DIGEST = str(VALIDATION / "dsv41-trace-digest.py")
LINUX = sys.platform.startswith("linux")
try:
    import numpy  # noqa: F401  (the synthetic index dumps; the sink and the digest tool themselves are stdlib only)
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False


def write(path, data):
    with open(path, "wb") as f:
        f.write(data)


def read(path, mode="r"):
    with open(path, mode) as f:
        return f.read()


def wait_for(condition, timeout=20.0, what="condition"):
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out after {timeout:g} s waiting for {what}")
        time.sleep(0.05)


def write_ratio_gguf(path, ratios):
    """A header-only GGUF (v3, no tensors) holding deepseek41.attention.compress_ratios as an int32 array."""
    key = b"deepseek41.attention.compress_ratios"
    with open(path, "wb") as f:
        f.write(b"GGUF" + struct.pack("<IQQ", 3, 0, 1))
        f.write(struct.pack("<Q", len(key)) + key)
        f.write(struct.pack("<IIQ", 9, 5, len(ratios)) + struct.pack(f"<{len(ratios)}i", *ratios))


def synthetic_index_trace(directory, *, prefix, chunk, decode, rows, width, topk, ratio, stale_seed, layer=2, bump=0.0):
    """Direct-engine sparse-selection dumps for one index layer (index_scores [nt, rows], index_top [nt, width],
    index_count [nt]): the written part is deterministic, everything past it stale memory seeded by stale_seed.

    Returns {file name: sha256 of its written part} for index_scores / index_top (index_count is hashed whole)."""
    import numpy as np
    stale = np.random.default_rng(stale_seed)
    calls = [(p0, min(chunk, prefix - p0)) for p0 in range(0, prefix, chunk)] + [(prefix + i, 1) for i in range(decode)]
    expect = {}
    for p0, nt in calls:
        sc = stale.normal(size=(nt, rows)).astype(np.float32)
        top = stale.integers(0, 99, size=(nt, width)).astype(np.int32)
        count = np.zeros(nt, np.int32)
        h_sc, h_top = hashlib.sha256(), hashlib.sha256()
        for q in range(nt):
            vis = (p0 + q + 1) // ratio
            n = min(topk, vis)
            sc[q, :vis] = np.arange(vis, dtype=np.float32) * 0.5 + p0 + bump
            top[q, :n] = np.arange(n)[::-1]
            count[q] = n
            h_sc.update(sc[q, :vis].tobytes())
            h_top.update(top[q, :n].tobytes())
        stem = f"p{p0:06d}_blk{layer:02d}_index"
        write(os.path.join(directory, f"{stem}_scores.f32"), sc.tobytes())
        write(os.path.join(directory, f"{stem}_top.i32"), top.tobytes())
        write(os.path.join(directory, f"{stem}_count.i32"), count.tobytes())
        expect[f"{stem}_scores.f32"] = h_sc.hexdigest()
        expect[f"{stem}_top.i32"] = h_top.hexdigest()
    return expect, calls


def write_probe(directory, prefix, decode):
    steps = [{"operation": "prefill", "logits": "step-000-prefill.f32"}] + \
        [{"operation": "decode", "logits": f"step-{i + 1:03d}-decode.f32"} for i in range(decode)]
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "probe.json"), "w") as f:
        json.dump({"tokens": list(range(prefix + decode)), "steps": steps}, f)


def run_digest(trace, probe, model):
    r = subprocess.run([sys.executable, "-I", DIGEST, "digest", trace, "--probe", probe, "--model", model],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise AssertionError(f"digest failed ({r.returncode}): {r.stderr}")
    return r.stdout


def parse_digest(text):
    return {line.split(None, 2)[2]: (line.split()[0], line.split()[1]) for line in text.splitlines() if line.strip()}


class Digest(unittest.TestCase):
    """dsv41-trace-digest.py: written-part hashing of the direct engine's index dumps, and compare's verdict."""

    def setUp(self):
        if not HAVE_NUMPY:
            self.skipTest("numpy is not installed")
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = self.tmp.name
        self.model = os.path.join(self.d, "ratios.gguf")
        write_ratio_gguf(self.model, [0, 0, 2, 1])
        self.probe = os.path.join(self.d, "probe")
        write_probe(self.probe, prefix=6, decode=2)

    def trace(self, name, stale_seed, bump=0.0, extra=b"x" * 64):
        path = os.path.join(self.d, name)
        os.makedirs(path)
        expect, _ = synthetic_index_trace(path, prefix=6, chunk=4, decode=2, rows=10, width=8, topk=3, ratio=2,
                                          stale_seed=stale_seed, bump=bump)
        write(os.path.join(path, "p000000_blk00_q.f32"), extra)
        return path, expect

    def compare(self, a, b):
        pa, pb = os.path.join(self.d, "a.digest"), os.path.join(self.d, "b.digest")
        write(pa, a.encode())
        write(pb, b.encode())
        return subprocess.run([sys.executable, "-I", DIGEST, "compare", pa, pb], capture_output=True, text=True)

    def test_written_part_ignores_stale_memory(self):
        a, expect = self.trace("a", stale_seed=1)
        b, _ = self.trace("b", stale_seed=2)
        da, db = run_digest(a, self.probe, self.model), run_digest(b, self.probe, self.model)
        dig = parse_digest(da)
        for n, h in expect.items():
            self.assertEqual(dig[n], (h, "written-part"), n)
        self.assertEqual(dig["p000000_blk00_q.f32"], (hashlib.sha256(b"x" * 64).hexdigest(), "bytes"))
        self.assertEqual(da, db)                                   # stale bytes differ, the digests do not
        r = self.compare(da, db)
        self.assertEqual(r.returncode, 0, r.stdout)
        self.assertIn("BIT-IDENTICAL", r.stdout)

    def test_written_part_and_file_set_changes_are_reported(self):
        a, _ = self.trace("a", stale_seed=1)
        c, _ = self.trace("c", stale_seed=1, bump=1000.0)
        da, dc = run_digest(a, self.probe, self.model), run_digest(c, self.probe, self.model)
        r = self.compare(da, dc)
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn("differs: p000000_blk02_index_scores.f32", r.stdout)
        self.assertNotIn("differs: p000000_blk00_q.f32", r.stdout)
        r = self.compare(da, da.replace("p000000_blk00_q.f32", "renamed.f32"))
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn("only A: p000000_blk00_q.f32", r.stdout)
        self.assertIn("only B: renamed.f32", r.stdout)


@unittest.skipUnless(LINUX, "dsv41-trace-sink.py uses inotify, /proc and SIGSTOP/SIGCONT (Linux only)")
class Sink(unittest.TestCase):
    def start(self, d, *extra, keep=(r"_blk0[01]_", r"index_top")):
        self.stage, self.dest = os.path.join(d, "stage"), os.path.join(d, "dest")
        self.digest, self.log, self.stop = (os.path.join(d, n) for n in ("x.digest", "x.log", "stop"))
        os.makedirs(self.stage, exist_ok=True)
        keep_args = [a for k in keep for a in ("--keep", k)]
        p = subprocess.Popen([sys.executable, "-I", SINK, "--stage", self.stage, "--dest", self.dest, *keep_args,
                              "--digest", self.digest, "--log", self.log, "--stop-file", self.stop, *extra])
        self.addCleanup(lambda: p.poll() is None and (p.kill(), p.wait(5)))
        # the sink logs "start:" after its inotify watch is in place; files written before that would be missed
        # by the event path and only found by the 3 s rescan
        wait_for(lambda: p.poll() is not None or (os.path.exists(self.log) and "start:" in read(self.log)),
                 what="the sink's start line")
        self.assertIsNone(p.poll(), "the sink exited during startup")
        return p

    def wait_drained(self):
        """Every file the writer closed has been taken: dropped ones deleted, kept ones copied and deleted."""
        wait_for(lambda: not os.listdir(self.stage), what="the stage to drain")

    def test_keep_drop_digest_and_versions(self):
        with tempfile.TemporaryDirectory() as d:
            p = self.start(d)
            names = {"p000000_blk00_q.f32": b"a" * 1000, "p000000_blk05_q.f32": b"b" * 2000,
                     "p000000_v41.05.query (permuted).f32": b"c" * 10, "p000000_blk07_index_top.i32": b"d" * 8}
            for n, v in names.items():
                write(os.path.join(self.stage, n), v)
            self.wait_drained()
            # a later pass rewrites a kept and a dropped file: the last version must win in both places
            write(os.path.join(self.stage, "p000000_blk00_q.f32"), b"A" * 1000)
            write(os.path.join(self.stage, "p000000_blk05_q.f32"), b"B" * 2000)
            self.wait_drained()
            open(self.stop, "w").close()
            self.assertEqual(p.wait(20), 0)
            self.assertEqual(sorted(os.listdir(self.dest)), ["p000000_blk00_q.f32", "p000000_blk07_index_top.i32"])
            self.assertEqual(read(os.path.join(self.dest, "p000000_blk00_q.f32"), "rb"), b"A" * 1000)
            dig = {line.split(None, 2)[2]: line.split()[0] for line in read(self.digest).splitlines()}
            self.assertEqual(set(dig), {"p000000_blk05_q.f32", "p000000_v41.05.query (permuted).f32"})
            self.assertEqual(dig["p000000_blk05_q.f32"], hashlib.sha256(b"B" * 2000).hexdigest())
            self.assertFalse(os.path.exists(self.stage))
            self.assertIn("versions per file {1: 2, 2: 2}", read(self.log))

    @unittest.skipUnless(HAVE_NUMPY, "numpy is not installed")
    def test_written_part_hash_of_dropped_index_dumps(self):
        import numpy as np
        prefix, chunk, rows, topk, ratio = 6, 4, 10, 3, 2
        chunks = [(0, 4), (4, 2), (6, 1)]

        def run(seed_stale, bump_written=False):
            rng = np.random.default_rng(seed_stale)
            with tempfile.TemporaryDirectory() as d:
                p = self.start(d, "--index-ratio", "2:2", "--prefix", str(prefix), "--chunk", str(chunk), "--topk", str(topk),
                               keep=(r"_blk0[01]_",))
                expect = {}
                for p0, nt in chunks:
                    sc = rng.normal(size=(nt, rows)).astype(np.float32)          # stale everywhere ...
                    top = rng.integers(0, 99, size=(nt, 8)).astype(np.int32)
                    h1, h2 = hashlib.sha256(), hashlib.sha256()
                    for q in range(nt):                                            # ... except the written part
                        vis = (p0 + q + 1) // ratio
                        sc[q, :vis] = np.arange(vis) + p0 + (1000 if bump_written else 0)
                        top[q, :min(topk, vis)] = np.arange(min(topk, vis))
                        h1.update(sc[q, :vis].tobytes())
                        h2.update(top[q, :min(topk, vis)].tobytes())
                    write(os.path.join(self.stage, f"p{p0:06d}_blk02_index_scores.f32"), sc.tobytes())
                    write(os.path.join(self.stage, f"p{p0:06d}_blk02_index_top.i32"), top.tobytes())
                    expect[f"p{p0:06d}_blk02_index_scores.f32"] = h1.hexdigest()
                    expect[f"p{p0:06d}_blk02_index_top.i32"] = h2.hexdigest()
                open(self.stop, "w").close()
                self.assertEqual(p.wait(20), 0)
                dig = {line.split(None, 2)[2]: (line.split()[0], line.split()[1]) for line in read(self.digest).splitlines()}
            for n, h in expect.items():
                self.assertEqual(dig[n], (h, "written-part"), n)
            return dig
        a, b, c = run(1), run(2), run(1, bump_written=True)
        self.assertEqual(a, b)                  # stale memory differs, the digests do not
        self.assertNotEqual(a, c)               # the written part differs

    @unittest.skipUnless(HAVE_NUMPY, "numpy is not installed")
    def test_dropped_digest_equals_the_digest_tool(self):
        """A run digest is the digest tool over the kept files plus the sink's digest of the dropped ones: the two
        must hash a file identically, so the same trace dropped by the sink and digested from disk must agree."""
        with tempfile.TemporaryDirectory() as d:
            source = os.path.join(d, "source")
            os.makedirs(source)
            synthetic_index_trace(source, prefix=6, chunk=4, decode=2, rows=10, width=8, topk=3, ratio=2, stale_seed=5)
            write(os.path.join(source, "p000000_blk05_q.f32"), b"q" * 4000)
            model, probe = os.path.join(d, "ratios.gguf"), os.path.join(d, "probe")
            write_ratio_gguf(model, [0, 0, 2, 1])
            write_probe(probe, prefix=6, decode=2)
            from_disk = run_digest(source, probe, model)
            p = self.start(d, "--index-ratio", "2:2", "--prefix", "6", "--chunk", "4", "--topk", "3", keep=(r"^$",))
            for name in sorted(os.listdir(source)):
                shutil.copyfile(os.path.join(source, name), os.path.join(self.stage, name))
            open(self.stop, "w").close()
            self.assertEqual(p.wait(20), 0)
            self.assertEqual(os.listdir(self.dest), [])
            self.assertEqual(parse_digest(read(self.digest)), parse_digest(from_disk))
            self.assertEqual(sum(t == "written-part" for _, t in parse_digest(from_disk).values()), 8)

    def test_flow_control_pauses_and_resumes_the_writer(self):
        with tempfile.TemporaryDirectory() as d:
            victim = subprocess.Popen(["sleep", "60"])
            self.addCleanup(lambda: victim.poll() is None and (victim.kill(), victim.wait(5)))
            pid_file = os.path.join(d, "pid")
            write(pid_file, str(victim.pid).encode())
            os.makedirs(os.path.join(d, "stage"))
            old = time.time() - 30
            for i in range(3):
                path = os.path.join(d, "stage", f"p00000{i}_blk05_q.f32")
                write(path, b"x" * 1000)
                os.utime(path, (old, old))
            p = self.start(d, "--pid-file", pid_file, "--pause-bytes", "2000", "--resume-bytes", "500")

            def stopped():
                status = [ln for ln in read(f"/proc/{victim.pid}/status").splitlines() if ln.startswith("State:")][0]
                return "T" in status.split()[1]
            wait_for(stopped, what="SIGSTOP while the stage held 3000 B")
            wait_for(lambda: not stopped(), what="SIGCONT once the stage drained")
            open(self.stop, "w").close()
            self.assertEqual(p.wait(20), 0)
            self.assertIn("'pauses': 1", read(self.log))

    def test_guard_kills_the_writer(self):
        with tempfile.TemporaryDirectory() as d:
            victim = subprocess.Popen(["sleep", "60"])
            self.addCleanup(lambda: victim.poll() is None and (victim.kill(), victim.wait(5)))
            pid_file = os.path.join(d, "pid")
            write(pid_file, str(victim.pid).encode())
            p = self.start(d, "--pid-file", pid_file, "--max-dest-bytes", "5000")
            for i in range(4):
                write(os.path.join(self.stage, f"p00000{i}_blk00_q.f32"), b"x" * 2000)
            self.assertEqual(p.wait(20), 3)
            self.assertEqual(victim.wait(5), -9)
            self.assertIn("GUARD", read(self.log))


@unittest.skipIf(LINUX, "the refusal is for platforms without inotify")
class SinkElsewhere(unittest.TestCase):
    def test_refuses_without_linux(self):
        with tempfile.TemporaryDirectory() as d:
            r = subprocess.run([sys.executable, "-I", SINK, "--stage", os.path.join(d, "s"), "--dest", os.path.join(d, "t"),
                                "--digest", os.path.join(d, "x.digest"), "--log", os.path.join(d, "x.log"),
                                "--stop-file", os.path.join(d, "stop")], capture_output=True, text=True)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("needs Linux", r.stderr)
            self.assertFalse(os.path.exists(os.path.join(d, "s")))


if __name__ == "__main__":
    unittest.main()
