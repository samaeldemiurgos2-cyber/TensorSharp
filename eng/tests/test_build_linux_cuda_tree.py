"""Checks when TensorSharp.GGML.Native/build-linux.sh starts a CUDA build tree over.

Runs the real script from a temporary copy, with fake nvcc, cmake, fetch-ggml and
fetch-cudnn, so it needs no toolkit, GPU or network. Linux only: it relies on bash,
symlinks and readlink -f, as the script itself does.
"""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "TensorSharp.GGML.Native" / "build-linux.sh"


@unittest.skipUnless(sys.platform.startswith("linux"), "build-linux.sh is a Linux build script")
class BuildLinuxCudaTreeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="tsg-build-linux-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        native = self.root / "TensorSharp.GGML.Native"
        native.mkdir()
        self.script = native / "build-linux.sh"
        shutil.copy2(SCRIPT, self.script)
        eng = self.root / "eng"
        eng.mkdir()
        for name in ("fetch-ggml.sh", "fetch-cudnn.sh"):
            (eng / name).write_text("exit 0\n")
        self.tools = self.root / "tools"
        self.tools.mkdir()
        self.cmake_log = self.root / "cmake.log"
        self.executable(self.tools / "cmake", f'#!/bin/sh\necho "$*" >> "{self.cmake_log}"\nexit 0\n')
        self.build = native / "build"
        self.cuda = self.root / "cuda"  # the stable path update-alternatives manages

    @staticmethod
    def executable(path, text):
        path.write_text(text)
        path.chmod(0o755)

    def toolkit(self, release):
        bin_dir = self.root / f"cuda-{release}" / "bin"
        bin_dir.mkdir(parents=True, exist_ok=True)
        self.executable(bin_dir / "nvcc",
                        f"#!/bin/sh\necho 'Cuda compilation tools, release {release}, V{release}.90'\n")
        return bin_dir / "nvcc"

    def point_cuda_at(self, release):
        self.toolkit(release)
        if self.cuda.is_symlink():
            self.cuda.unlink()
        self.cuda.symlink_to(self.root / f"cuda-{release}")

    def configured_tree(self, compiler, identified_version):
        """A tree whose CMake run identified `compiler` as CUDA `identified_version`."""
        identified = self.build / "CMakeFiles" / "4.1.0"
        identified.mkdir(parents=True)
        (self.build / "CMakeCache.txt").write_text(
            f"CMAKE_CUDA_COMPILER:FILEPATH={compiler}\nTENSORSHARP_GGML_NATIVE_ENABLE_CUDA:BOOL=ON\n")
        (identified / "CMakeCUDACompiler.cmake").write_text(
            f'set(CMAKE_CUDA_COMPILER "{compiler}")\n'
            f'set(CMAKE_CUDA_COMPILER_VERSION "{identified_version}")\n'
            f'set(CMAKE_CUDA_COMPILER_TOOLKIT_VERSION "{identified_version}")\n')
        (self.build / "kept.marker").write_text("")

    def run_script(self):
        env = {
            "PATH": os.pathsep.join([str(self.cuda / "bin"), str(self.tools), "/usr/bin", "/bin"]),
            "HOME": str(self.root),
            "TENSORSHARP_GGML_NATIVE_ENABLE_CUDA": "ON",
            "TENSORSHARP_GGML_NATIVE_ENABLE_VULKAN": "OFF",
            "TENSORSHARP_GGML_NATIVE_BUILD_PARALLEL_LEVEL": "1",
        }
        return subprocess.run(["bash", str(self.script)], env=env, capture_output=True, text=True, timeout=120)

    def assert_started_over(self, result, reason):
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)
        self.assertIn(reason, result.stdout)
        self.assertIn("removing", result.stdout)
        self.assertFalse((self.build / "kept.marker").exists(), "the stale tree was kept")

    def test_cuda12_swapped_for_13_behind_the_same_path_starts_the_tree_over(self):
        self.point_cuda_at("12.8")
        self.configured_tree(self.cuda / "bin" / "nvcc", "12.8.93")
        self.point_cuda_at("13.2")  # update-alternatives --set cuda /usr/local/cuda-13.2
        result = self.run_script()
        self.assert_started_over(result, "configured with CUDA 12.8, but")
        self.assertIn("is now CUDA 13.2", result.stdout)

    def test_minor_toolkit_swap_behind_the_same_path_starts_the_tree_over(self):
        self.point_cuda_at("13.2")
        self.configured_tree(self.cuda / "bin" / "nvcc", "13.2.78")
        self.point_cuda_at("13.4")
        self.assert_started_over(self.run_script(), "configured with CUDA 13.2, but")

    def test_unchanged_toolkit_keeps_the_tree(self):
        self.point_cuda_at("13.2")
        self.configured_tree(self.cuda / "bin" / "nvcc", "13.2.78")
        result = self.run_script()
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)
        self.assertNotIn("removing", result.stdout)
        self.assertTrue((self.build / "kept.marker").exists(), "an up-to-date tree was discarded")
        self.assertIn("-S", self.cmake_log.read_text())

    def test_a_different_compiler_path_still_starts_the_tree_over(self):
        self.configured_tree(self.toolkit("13.2"), "13.2.78")
        self.point_cuda_at("13.4")
        self.assert_started_over(self.run_script(), "CUDA compiler changed")

    def test_cuda12_nvcc_is_an_error(self):
        self.point_cuda_at("12.8")
        result = self.run_script()
        self.assertEqual(1, result.returncode, result.stdout + result.stderr)
        self.assertIn("CUDA 13+ required (found 12.8", result.stderr)
        self.assertFalse(self.cmake_log.exists(), "cmake ran with a CUDA 12 nvcc")


if __name__ == "__main__":
    unittest.main()
