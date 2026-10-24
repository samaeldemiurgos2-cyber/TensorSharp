#!/usr/bin/env bash
# Reproducible remote build/download setup; never modifies upstream sources.
set -euo pipefail
root="${1:-/workspace}"
ggml_revision="${2:-a05c9922613c6a2230dbb1d0de679519ecafb4f0}"
mkdir -p "$root/TensorSharp" "$root/llama.cpp" "$root/evidence" "$root/dotnet" "$root/models/muse-spark"
tar -xzf "$root/tensorsharp-baseline.tar.gz" -C "$root/TensorSharp"
tar -xzf "$root/llama-source.tar.gz" -C "$root/llama.cpp"
export PATH="$root/dotnet:/usr/local/cuda/bin:$PATH"
export DOTNET_ROOT="$root/dotnet" DOTNET_CLI_TELEMETRY_OPTOUT=1
if ! test -x "$root/dotnet/dotnet"; then
    curl -fsSL https://dot.net/v1/dotnet-install.sh -o "$root/dotnet-install.sh"
    bash "$root/dotnet-install.sh" --channel 10.0 --install-dir "$root/dotnet"
fi
python3 -m venv "$root/venv"
"$root/venv/bin/pip" install 'huggingface_hub>=0.34' numpy requests pillow
mkdir -p "$root/TensorSharp/ExternalProjects/ggml"
if ! test -d "$root/TensorSharp/ExternalProjects/ggml/.git"; then
    git -C "$root/TensorSharp/ExternalProjects/ggml" init
    git -C "$root/TensorSharp/ExternalProjects/ggml" remote add origin https://github.com/ggml-org/ggml.git
    git -C "$root/TensorSharp/ExternalProjects/ggml" fetch --depth 1 origin "$ggml_revision"
    git -C "$root/TensorSharp/ExternalProjects/ggml" checkout --detach FETCH_HEAD
fi
test "$(git -C "$root/TensorSharp/ExternalProjects/ggml" rev-parse HEAD)" = "$ggml_revision"
test -z "$(git -C "$root/TensorSharp/ExternalProjects/ggml" status --porcelain --untracked-files=all)"
cmake -S "$root/TensorSharp/TensorSharp.GGML.Native" -B "$root/TensorSharp/TensorSharp.GGML.Native/build" \
    -DCMAKE_BUILD_TYPE=Release -DTENSORSHARP_GGML_NATIVE_ENABLE_CUDA=ON \
    -DTENSORSHARP_GGML_NATIVE_ENABLE_VULKAN=OFF -DTENSORSHARP_GGML_NATIVE_BUILD_TESTS=ON \
    -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build "$root/TensorSharp/TensorSharp.GGML.Native/build" -j 12
cmake -S "$root/llama.cpp" -B "$root/llama.cpp/build" -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON -DGGML_VULKAN=OFF -DCMAKE_CUDA_ARCHITECTURES=86 \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DLLAMA_CURL=ON
cmake --build "$root/llama.cpp/build" -j 12 --target llama-server llama-bench llama-cli
test -z "$(git -C "$root/TensorSharp/ExternalProjects/ggml" status --porcelain --untracked-files=all)"
