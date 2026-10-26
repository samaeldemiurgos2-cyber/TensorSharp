"""Check the GB10 build gate and real Docker context filtering, without CUDA or a GPU."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = ROOT / "eng" / "Dockerfile.gb10"
dockerfile_bytes = DOCKERFILE.read_bytes()
assert b"\r\n" not in dockerfile_bytes, "Docker RUN heredocs must retain LF line endings"
for name in ("package-gb10.sh", "verify-gb10-release.sh"):
    assert b"\r\n" not in (ROOT / "eng" / name).read_bytes(), f"{name}: shell scripts must retain LF line endings"
bash = shutil.which("bash")
if bash is None:
    raise SystemExit("The GB10 container check requires Bash on PATH.")
guard = "\n".join(
    dockerfile_bytes.decode("utf-8").split(f"RUN <<'{marker}'\n", 1)[1].split(f"\n{marker}", 1)[0]
    for marker in ("CHECK_GB10_PLATFORM", "CHECK_GB10_CUDA")
)
probes = """
uname() { printf '%s\\n' "$TEST_UNAME"; }
nvcc() { printf '%s\\n' "Cuda compilation tools, release $TEST_CUDA, V$TEST_CUDA.0"; }
nvidia-smi() { echo UNEXPECTED_GPU_PROBE; return 1; }
"""

for build, target, system, cuda, error in [
    ("linux/arm64", "linux/arm64", "Linux aarch64", "13.0", None),
    ("linux/arm64", "linux/arm64", "Linux aarch64", "13.4", None),
    ("linux/arm64", "linux/arm64", "Linux aarch64", "12.9", "require CUDA 13"),
    ("linux/arm64", "linux/arm64", "Linux aarch64", "14.0", "require CUDA 13"),
    ("linux/amd64", "linux/arm64", "Linux aarch64", "13.0", "native Linux ARM64"),
    ("linux/arm64", "linux/amd64", "Linux aarch64", "13.0", "native Linux ARM64"),
    ("linux/arm64", "linux/arm64", "Linux x86_64", "13.0", "native Linux ARM64"),
]:
    result = subprocess.run(
        [bash, "-euo", "pipefail"], input=probes + guard + "\n",
        env={**os.environ, "BUILDPLATFORM": build, "TARGETPLATFORM": target,
             "TEST_UNAME": system, "TEST_CUDA": cuda},
        capture_output=True, text=True,
    )
    assert "UNEXPECTED_GPU_PROBE" not in result.stdout + result.stderr
    if error is None:
        assert result.returncode == 0, result.stderr
    else:
        assert result.returncode != 0 and error in result.stderr, result.stderr
print("PASS: native ARM64/CUDA 13 gate; no GPU probe")

included = [
    "TensorSharp.Models/Models/LocalEdit.cs",
    "TensorSharp.GGML.Native/build-linux.sh",
    "TensorSharp.GGML.Native/build-windows.ps1",
    "TensorSharp.Backends.Cuda/native/kernels/tensorsharp_kernels.cu",
    "TensorSharp.Backends.Cuda/native/ptx/tensorsharp_kernels.ptx",
    "eng/fetch-ggml.sh",
]
excluded = [
    ".git", "nested/.git/config", ".env", "nested/.env.local", "nested/secrets.json",
    "nested/private.pem", "nested/private.key", "nested/cert.pfx", "nested/cert.p12",
    "nested/.ssh/config", "nested/.vscode/launch.json", "models/checkpoint.bin", "downloads/model.gguf",
    "downloads/model.safetensors", "ExternalProjects/ggml/CMakeLists.txt",
    "TensorSharp.Core/bin/stale.dll", "TensorSharp.Core/obj/project.assets.json",
    "TensorSharp.GGML.Native/build/libGgmlOps.so",
    "TensorSharp.GGML.Native/build-windows/GgmlOps.dll",
    "TensorSharp.Backends.MLX/Native/dist/libmlx.dylib",
]
with tempfile.TemporaryDirectory(prefix="tensorsharp-gb10-") as temporary:
    context = Path(temporary) / "context"
    output = Path(temporary) / "output"
    for name in included + excluded:
        path = context / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("local checkout fixture\n", encoding="utf-8")
    fixture = context / "eng" / "Dockerfile.gb10"
    fixture.write_bytes(DOCKERFILE.read_bytes())
    fixture.with_name(fixture.name + ".dockerignore").write_bytes(
        DOCKERFILE.with_name(DOCKERFILE.name + ".dockerignore").read_bytes()
    )
    subprocess.run(
        ["docker", "build", "--file", str(fixture), "--target", "source",
         "--output", f"type=local,dest={output}", str(context)],
        check=True,
    )
    for name in included:
        assert (output / name).read_text() == "local checkout fixture\n", name
    for name in excluded:
        assert not (output / name).exists(), name
print("PASS: Docker keeps local sources and excludes credentials, models, and stale builds")
