"""run-multinode-case.py must hash, and name, the libGgmlOps.so each node's process loads.

GgmlNative.ImportResolver (TensorSharp.Backends.GGML/GgmlNative.cs, GetCandidatePaths) loads the first
existing file of: the app directory, then TensorSharp.GGML.Native/{build,build/Release,build-windows,
build-windows/Release} under each repo root from the app directory upwards, then the runtime's default
probing (LD_LIBRARY_PATH). The probe's bin directory has no libGgmlOps.so, so before this check the
launcher's runtime() hashed only managed DLLs for probe runs. These tests run the generated launcher's
shell functions against a fake tree.
"""
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

PROBE_BIN = "eng/validation/DeepSeek41DistributedProbe/bin/Release/net10.0"
FUNCTIONS = ("ggml_candidates()", "runtime()", "loaded_native()")


def launcher(root, dll, native_dir=None):
    a = SimpleNamespace(label="unit", remote_root=root, dotnet="/opt/dotnet/dotnet", native_dir=native_dir,
                        model="/m/model-00001-of-00002.gguf", no_source_fingerprint=True, kind="probe",
                        gpu_telemetry_ms=0)
    node = SimpleNamespace(index=0, role="driver")
    return case.launcher_text(a, node, root + "/runs/unit-node0", ["true"], root + "/" + dll, {})


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


if __name__ == "__main__":
    unittest.main()
