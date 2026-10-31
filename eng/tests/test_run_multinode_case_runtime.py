"""run-multinode-case.py must hash, and name, the libGgmlOps.so each node's process loads.

GgmlNative.ImportResolver (TensorSharp.Backends.GGML/GgmlNative.cs, GetCandidatePaths) loads the first
existing file of: the app directory, then TensorSharp.GGML.Native/{build,build/Release,build-windows,
build-windows/Release} under each repo root from the app directory upwards, then the runtime's default
probing (LD_LIBRARY_PATH). The probe's bin directory has no libGgmlOps.so, so before this check the
launcher's runtime() hashed only managed DLLs for probe runs. These tests run the generated launcher's
shell functions against a fake tree.

The launcher also hashes the sources under SOURCE_DIRS into source-files.sha256 and source-fingerprint.txt.
That fingerprint must cover the direct CUDA engine that --backend cuda runs: Dsv4CudaEngine, its .cu
kernels and the committed PTX live in TensorSharp.Backends.Cuda, which the list used to leave out.
"""
import fnmatch
import hashlib
import importlib.util
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest


spec = importlib.util.spec_from_file_location(
    "run_multinode_case", Path(__file__).parents[1] / "validation" / "run-multinode-case.py")
case = importlib.util.module_from_spec(spec)
spec.loader.exec_module(case)

REPO = Path(__file__).resolve().parents[2]
PROBE_BIN = "eng/validation/DeepSeek41DistributedProbe/bin/Release/net10.0"
FUNCTIONS = ("ggml_candidates()", "runtime()", "loaded_native()")
# Files of the direct CUDA engine (--backend cuda) that every node's source fingerprint must cover.
DIRECT_ENGINE_SOURCES = (
    "TensorSharp.Backends.Cuda/TensorSharp.Backends.Cuda.csproj",
    "TensorSharp.Backends.Cuda/Dsv4/Dsv4CudaEngine.cs",
    "TensorSharp.Backends.Cuda/Dsv4/Dsv4CudaEngine.Distributed.cs",
    "TensorSharp.Backends.Cuda/Dsv4/Dsv4CudaEngine.LayerSplit.cs",
    "TensorSharp.Backends.Cuda/Dsv4/Dsv4Kernels.cs",
    "TensorSharp.Backends.Cuda/Dsv4/IDsv41EngramSource.cs",
    "TensorSharp.Backends.Cuda/native/kernels/tensorsharp_dsv4_kernels.cu",
    "TensorSharp.Backends.Cuda/native/kernels/tensorsharp_iq1s_tables.cuh",
    "TensorSharp.Backends.Cuda/native/ptx/tensorsharp_dsv4_kernels.ptx",
    "TensorSharp.Models/Models/DeepSeek4/DeepSeek4CudaExecutor.cs",
)
PRUNED_DIRS = ("bin", "obj", "node_modules", ".git")


def launcher(root, dll, native_dir=None, fingerprint=False, kind="probe"):
    a = SimpleNamespace(label="unit", remote_root=root, dotnet="/opt/dotnet/dotnet", native_dir=native_dir,
                        model="/m/model-00001-of-00002.gguf", no_source_fingerprint=not fingerprint, kind=kind,
                        gpu_telemetry_ms=0)
    node = SimpleNamespace(index=0, role="driver")
    return case.launcher_text(a, node, root + "/runs/unit-node0", ["true"], root + "/" + dll, {})


def fingerprint_block(text):
    """The launcher lines that write source-files.sha256 and source-fingerprint.txt."""
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith('( cd "$root"'))
    end = next(i for i, line in enumerate(lines) if line.startswith("sha256sum < source-files.sha256"))
    return "\n".join(lines[start:end + 1])


def sha_lines(text):
    """{path: digest} from sha256sum output. GNU sha256sum on Linux prints 'digest  path'; the Git for
    Windows build prints 'digest *path' (binary mode), so the parser here accepts both."""
    result = {}
    for line in text.splitlines():
        match = re.fullmatch(r"([0-9a-f]{64}) [ *](.+)", line)
        if match:
            result[match[2]] = match[1]
    return result


def shell_functions(text):
    lines = [line for line in text.splitlines() if line.startswith(FUNCTIONS)]
    assert len(lines) == len(FUNCTIONS), lines
    return "\n".join(lines)


@unittest.skipUnless(shutil.which("sh") and shutil.which("sha256sum"), "needs a POSIX sh and sha256sum")
class LoadedNativeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def put(self, relative, payload=b"x"):
        path = self.base / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return path

    def sh(self, script):
        result = subprocess.run(["sh", "-c", script], capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout

    def shell_path(self, relative):
        """The path as the shell spells it after cd + pwd -P (POSIX on Linux, /c/... under MSYS)."""
        return self.sh(f"cd {shlex.quote(str(self.base / relative))} && pwd -P").strip()

    def run_functions(self, root, dll, ld_dirs=()):
        root_q = shlex.quote(str(self.base / root))
        bin_dir, main = dll.rsplit("/", 1)
        ld = ":".join(f"$(cd {shlex.quote(str(self.base / d))} && pwd -P)" for d in ld_dirs)
        script = "\n".join([
            shell_functions(launcher("/r", dll)),
            f"root=$(cd {root_q} && pwd -P); bin=\"$root/{bin_dir}\" main={shlex.quote(main)}",
            f"LD_LIBRARY_PATH=\"{ld}\"; export LD_LIBRARY_PATH",
            "echo '--- runtime'; runtime; echo '--- loaded'; loaded_native"])
        out = self.sh(script)
        runtime, loaded = out.split("--- loaded\n")
        return sha_lines(runtime.split("--- runtime\n")[1]), sha_lines(loaded)

    def test_probe_layout_loads_the_repo_root_build(self):
        self.put("ts/TensorSharp.slnx")
        self.put(f"ts/{PROBE_BIN}/DeepSeek41DistributedProbe.dll")
        self.put(f"ts/{PROBE_BIN}/TensorSharp.Models.dll")
        built = self.put("ts/TensorSharp.GGML.Native/build/libGgmlOps.so", b"repo-build")
        self.put("ld/libGgmlOps.so", b"ld-path")
        runtime, loaded = self.run_functions("ts", f"{PROBE_BIN}/DeepSeek41DistributedProbe.dll", ["ld"])
        root = self.shell_path("ts")
        expected = root + "/TensorSharp.GGML.Native/build/libGgmlOps.so"
        self.assertEqual(loaded, {expected: hashlib.sha256(built.read_bytes()).hexdigest()})
        self.assertEqual(runtime[expected], loaded[expected])
        self.assertIn(self.shell_path("ld") + "/libGgmlOps.so", runtime)
        self.assertIn(f"{root}/{PROBE_BIN}/TensorSharp.Models.dll", runtime)

    def test_app_directory_copy_wins_over_the_repo_build(self):
        self.put("ts/TensorSharp.slnx")
        self.put("ts/TensorSharp.Cli/bin/TensorSharp.Cli.dll")
        local = self.put("ts/TensorSharp.Cli/bin/libGgmlOps.so", b"bin-copy")
        self.put("ts/TensorSharp.GGML.Native/build/libGgmlOps.so", b"repo-build")
        runtime, loaded = self.run_functions("ts", "TensorSharp.Cli/bin/TensorSharp.Cli.dll")
        root = self.shell_path("ts")
        self.assertEqual(loaded, {root + "/TensorSharp.Cli/bin/libGgmlOps.so": hashlib.sha256(local.read_bytes()).hexdigest()})
        self.assertIn(root + "/TensorSharp.GGML.Native/build/libGgmlOps.so", runtime)

    def test_search_order_within_a_root_and_across_nested_roots(self):
        self.put("outer/.git/HEAD")
        self.put("outer/TensorSharp.GGML.Native/build/libGgmlOps.so", b"outer")
        self.put("outer/inner/TensorSharp.sln")
        self.put("outer/inner/app/bin/App.dll")
        self.put("outer/inner/TensorSharp.GGML.Native/build-windows/Release/libGgmlOps.so", b"inner-bwr")
        _, loaded = self.run_functions("outer/inner", "app/bin/App.dll")
        self.assertEqual(list(loaded), [self.shell_path("outer/inner") + "/TensorSharp.GGML.Native/build-windows/Release/libGgmlOps.so"])
        self.put("outer/inner/TensorSharp.GGML.Native/build/Release/libGgmlOps.so", b"inner-br")
        _, loaded = self.run_functions("outer/inner", "app/bin/App.dll")
        self.assertEqual(list(loaded), [self.shell_path("outer/inner") + "/TensorSharp.GGML.Native/build/Release/libGgmlOps.so"])

    def test_a_git_file_is_not_a_repo_root_and_ld_library_path_is_the_fallback(self):
        # A git worktree has a .git FILE; the resolver only accepts a .git directory or a solution file.
        self.put("wt/.git", b"gitdir: /elsewhere\n")
        self.put("wt/TensorSharp.GGML.Native/build/libGgmlOps.so", b"not-a-root")
        self.put("wt/app/bin/App.dll")
        fallback = self.put("ld2/libGgmlOps.so", b"ld")
        self.put("ld1/other.so")
        runtime, loaded = self.run_functions("wt", "app/bin/App.dll", ["ld1", "ld2"])
        self.assertEqual(loaded, {self.shell_path("ld2") + "/libGgmlOps.so": hashlib.sha256(fallback.read_bytes()).hexdigest()})
        self.assertNotIn(self.shell_path("wt") + "/TensorSharp.GGML.Native/build/libGgmlOps.so", runtime)

    def test_no_library_anywhere_leaves_native_loaded_empty(self):
        self.put("ts/TensorSharp.slnx")
        self.put("ts/app/bin/App.dll")
        runtime, loaded = self.run_functions("ts", "app/bin/App.dll")
        self.assertEqual(loaded, {})
        self.assertEqual(list(runtime), [self.shell_path("ts") + "/app/bin/App.dll"])

    def test_generated_launcher_parses_and_records_before_the_command(self):
        text = launcher("/root/ts", PROBE_BIN + "/DeepSeek41DistributedProbe.dll", native_dir="/opt/native")
        result = subprocess.run(["sh", "-n"], input=text, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = text.splitlines()
        record = lines.index("loaded_native > native-loaded.sha256")
        self.assertEqual(lines[record - 1], "runtime > runtime-before.sha256")
        command = next(i for i, line in enumerate(lines) if line.endswith('> "$run/process.log" 2>&1 < /dev/null'))
        self.assertLess(record, command)


class LoadedNativeParsingTests(unittest.TestCase):
    def test_reads_the_single_line_or_none(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.assertIsNone(case.loaded_native(directory))
            (directory / "native-loaded.sha256").write_text("", encoding="utf-8")
            self.assertIsNone(case.loaded_native(directory))
            digest = "ab" * 32
            (directory / "native-loaded.sha256").write_text(
                f"{digest}  /root/ts/TensorSharp.GGML.Native/build/libGgmlOps.so\n", encoding="utf-8")
            self.assertEqual(case.loaded_native(directory),
                             {"path": "/root/ts/TensorSharp.GGML.Native/build/libGgmlOps.so", "sha256": digest})


class SourceFingerprintScopeTests(unittest.TestCase):
    def test_direct_cuda_engine_sources_are_in_scope(self):
        """Each direct-engine file in this checkout is under a SOURCE_DIRS entry, matches a SOURCE_PATTERNS
        glob, and sits outside the directories the launcher prunes."""
        for relative in DIRECT_ENGINE_SOURCES:
            with self.subTest(relative):
                self.assertTrue((REPO / relative).is_file(),
                                "moved? update DIRECT_ENGINE_SOURCES and SOURCE_DIRS together")
                top, *middle, name = relative.split("/")
                self.assertIn(top, case.SOURCE_DIRS)
                self.assertTrue(any(fnmatch.fnmatchcase(name, pattern) for pattern in case.SOURCE_PATTERNS))
                self.assertFalse([part for part in middle if part in PRUNED_DIRS or part.startswith("build")])


@unittest.skipUnless(all(shutil.which(tool) for tool in ("sh", "sha256sum", "find", "sort", "xargs")),
                     "needs a POSIX sh, find, sort, xargs and sha256sum")
class SourceFingerprintTests(unittest.TestCase):
    """Runs the generated launcher's fingerprint lines against a fake tree."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        for relative in DIRECT_ENGINE_SOURCES:
            self.put("ts/" + relative, relative.encode())
        self.put("ts/Directory.Build.props")
        self.put("ts/eng/validation/DeepSeek41DistributedProbe/Program.cs")
        # Build outputs: never part of the fingerprint.
        self.put("ts/TensorSharp.Backends.Cuda/obj/cuda_ptx/ptx/tensorsharp_dsv4_kernels.ptx", b"nvcc")
        self.put("ts/TensorSharp.Backends.Cuda/bin/Release/cuda_kernels/tensorsharp_dsv4_kernels.ptx", b"copy")
        self.put("ts/TensorSharp.GGML.Native/build-linux/generated.h", b"cmake")

    def tearDown(self):
        self.temporary.cleanup()

    def put(self, relative, payload=b"x"):
        path = self.base / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return path

    def fingerprint(self, kind="probe"):
        """(source-files.sha256 as {path: digest}, source-fingerprint.txt) for one launcher run."""
        run = self.base / "run"
        run.mkdir(exist_ok=True)
        text = launcher("/root/ts", case.BINARIES[kind], fingerprint=True, kind=kind)
        script = "\n".join([f"cd {shlex.quote(str(run))} || exit 1",
                            f"root=$(cd {shlex.quote(str(self.base / 'ts'))} && pwd -P) || exit 1",
                            fingerprint_block(text)])
        result = subprocess.run(["sh", "-c", script], capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        files = sha_lines((run / "source-files.sha256").read_text(encoding="utf-8"))
        return files, (run / "source-fingerprint.txt").read_text(encoding="utf-8").strip()

    def test_launcher_hashes_the_direct_engine_and_skips_build_outputs(self):
        text = launcher("/root/ts", case.BINARIES["probe"], fingerprint=True)
        parsed = subprocess.run(["sh", "-n"], input=text, capture_output=True, text=True, timeout=60)
        self.assertEqual(parsed.returncode, 0, parsed.stderr)
        files, fingerprint = self.fingerprint()
        for relative in DIRECT_ENGINE_SOURCES:
            self.assertEqual(files.get(relative), hashlib.sha256(relative.encode()).hexdigest(), relative)
        self.assertIn("Directory.Build.props", files)
        self.assertIn("eng/validation/DeepSeek41DistributedProbe/Program.cs", files)
        self.assertEqual([path for path in files if "/obj/" in path or "/bin/" in path or "/build-" in path], [])
        self.assertRegex(fingerprint, r"^[0-9a-f]{64}$")
        server_files, _ = self.fingerprint(kind="server")
        self.assertIn("TensorSharp.Backends.Cuda/Dsv4/Dsv4CudaEngine.cs", server_files)
        self.assertNotIn("eng/validation/DeepSeek41DistributedProbe/Program.cs", server_files)

    def test_an_engine_or_committed_ptx_edit_changes_the_fingerprint(self):
        _, before = self.fingerprint()
        self.put("ts/TensorSharp.Backends.Cuda/obj/cuda_ptx/ptx/tensorsharp_dsv4_kernels.ptx", b"rebuilt")
        _, after_obj = self.fingerprint()
        self.assertEqual(after_obj, before)
        self.put("ts/TensorSharp.Backends.Cuda/Dsv4/Dsv4CudaEngine.cs", b"edited")
        _, after_engine = self.fingerprint()
        self.assertNotEqual(after_engine, before)
        self.put("ts/TensorSharp.Backends.Cuda/native/ptx/tensorsharp_dsv4_kernels.ptx", b"refreshed")
        _, after_ptx = self.fingerprint()
        self.assertNotEqual(after_ptx, after_engine)


if __name__ == "__main__":
    unittest.main()
