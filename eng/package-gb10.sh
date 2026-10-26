#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
OUTPUT_DIR="${1:?Usage: package-gb10.sh OUTPUT_DIR [VERSION]}"
VERSION="${2:-$(dotnet msbuild TensorSharp.Cli/TensorSharp.Cli.csproj -nologo -getProperty:Version)}"
[[ "$VERSION" =~ ^[0-9][0-9A-Za-z.+-]*$ ]] || { echo "Invalid release version: $VERSION" >&2; exit 1; }
[[ "$(uname -sm)" == "Linux aarch64" ]] || { echo "GB10 packaging requires Linux ARM64." >&2; exit 1; }
nvcc --version | grep -q 'release 13\.' || { echo "GB10 packaging requires CUDA 13." >&2; exit 1; }

NATIVE=TensorSharp.GGML.Native/build/libGgmlOps.so
readelf -d "$NATIVE" | grep -Fq '[$ORIGIN]' || { echo "GGML must use an origin-relative runtime path." >&2; exit 1; }
if readelf -d "$NATIVE" | grep -q 'libnccl'; then
    echo "Rebuild the single-device GB10 backend with NCCL disabled." >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(realpath "$OUTPUT_DIR")"
PUBLISH_ROOT="$(mktemp -d)"
trap 'rm -rf "$PUBLISH_ROOT"' EXIT
CUDA_LIB_DIR="${CUDA_HOME:-/usr/local/cuda}/lib64"
NUGET_ROOT="${NUGET_PACKAGES:-$HOME/.nuget/packages}"
archives=()

for entry in cli:TensorSharp.Cli server:TensorSharp.Server.Host; do
    app="${entry%%:*}"
    project="${entry#*:}"
    publish="$PUBLISH_ROOT/$app"
    dotnet publish "$project/$project.csproj" -c Release -r linux-arm64 \
        --self-contained true -p:Version="$VERSION" -p:CudaArch=compute_121 \
        -p:TensorSharpSkipGgmlNative=true -o "$publish"

    cp "$NATIVE" "$publish/"
    for library in libcudart.so.13 libcublas.so.13 libcublasLt.so.13; do
        [[ -s "$CUDA_LIB_DIR/$library" ]] || { echo "Required CUDA library missing: $CUDA_LIB_DIR/$library" >&2; exit 1; }
        cp -L "$CUDA_LIB_DIR/$library" "$publish/"
    done
    mkdir -p "$publish/cuda_kernels"
    for kernel in TensorSharp.Backends.Cuda/native/kernels/*.cu; do
        ptx="TensorSharp.Backends.Cuda/obj/cuda_ptx/ptx/$(basename "${kernel%.cu}").ptx"
        grep -Eq '^\.target[[:space:]]+sm_121([[:space:],]|$)' "$ptx" \
            || { echo "Missing or non-GB10 PTX: $ptx" >&2; exit 1; }
        # The sm_121 SASS loads on any CUDA 13 driver; the CUDA 13.4 PTX beside it
        # only JIT-compiles on a 13.4 or newer driver (DGX OS ships R580, CUDA 13.0).
        fatbin="${ptx%.ptx}.fatbin"
        [[ -s "$fatbin" && -s "$fatbin.sha256" ]] || { echo "Missing GB10 fatbin: $fatbin" >&2; exit 1; }
        cp "$ptx" "$fatbin" "$fatbin.sha256" "$publish/cuda_kernels/"
    done
    for library in libOpenCvSharpExtern.so libx264.so.164 Magick.Native-Q8-arm64.dll.so; do
        [[ -s "$publish/$library" ]] || { echo "Required ARM64 media library missing: $library" >&2; exit 1; }
    done
    [[ ! -e "$publish/libcuda.so" && ! -e "$publish/libcuda.so.1" ]] \
        || { echo "Do not bundle the host driver or CUDA driver stubs." >&2; exit 1; }

    mkdir -p "$publish/licenses"
    cp LICENSE "$publish/licenses/TensorSharp-LICENSE.txt"
    cp ExternalProjects/ggml/LICENSE "$publish/licenses/GGML-LICENSE.txt"
    cp /usr/share/doc/cuda-cudart-13-4/copyright "$publish/licenses/NVIDIA-cudart.txt"
    cp /usr/share/doc/libcublas-13-4/copyright "$publish/licenses/NVIDIA-cuBLAS.txt"
    cp "$NUGET_ROOT"/magick.net-q8-anycpu/*/Notice.txt "$publish/licenses/Magick.NET-NOTICE.txt"
    cp "$NUGET_ROOT"/oneware.opencvsharp4.runtime.ubuntu.24.04-arm64/*/*.nuspec "$publish/licenses/OpenCV-runtime.nuspec"
    cp /usr/share/common-licenses/Apache-2.0 /usr/share/common-licenses/GPL-2 \
        /usr/share/common-licenses/LGPL-2.1 "$publish/licenses/"
    cat > "$publish/README-GB10.txt" <<EOF
TensorSharp $VERSION - Linux ARM64, NVIDIA GB10, CUDA 13
Run ./$project --help from this directory, or use its absolute path.
No .NET SDK or CUDA toolkit installation is required. Built with CUDA 13.4.3;
any NVIDIA driver of the CUDA 13 family (R580 or newer) runs it.
Ubuntu 24.04 OS packages: ca-certificates libgomp1 libgssapi-krb5-2 libicu74
libssl3t64 libstdc++6 zlib1g.
NCCL, Vulkan and optional cuDNN acceleration are disabled in this build.
The native media package includes OpenCV/FFmpeg/x264; preserve its licensing
terms and corresponding-source obligations when redistributing.
Native media source: https://github.com/hendrikmennen/opencvsharp-mini-runtime
Source revision: ${SOURCE_REVISION:-unspecified}
GGML revision: $(git -C ExternalProjects/ggml rev-parse HEAD)
EOF
    archive="tensorsharp-$app-$VERSION-linux-arm64-cuda13-GB10.tar.gz"
    tar -C "$publish" -czf "$OUTPUT_DIR/$archive" .
    archives+=("$archive")
    rm -rf "$publish"
done
(cd "$OUTPUT_DIR" && sha256sum "${archives[@]}" > SHA256SUMS)
