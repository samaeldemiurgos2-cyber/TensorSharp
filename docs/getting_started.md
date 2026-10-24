<a id="getting-started"></a>
# Getting started

[English](getting_started.md) | [中文](getting_started_zh-cn.md)


Get one text reply first: choose an interface and platform → install → download one model → send a prompt → check the result. Add images, video, multiple GPUs or skills after that path works.

<a id="choose"></a>
<a id="when-to-try-tensorsharp"></a>
## 1 · Choose your interface

| Your goal | Start here |
| --- | --- |
| Install and chat | [Desktop install and first chat](tensoragent_desktop.md). Mac/Windows packages include their runtime; skip the source-build steps below. |
| Use a terminal, browser or compatible HTTP client | Continue with the CLI/server source path below; both use the same GGUF text model. |
| Embed the engine in C# | Get one text run working, then read [package boundaries and integration](../DEVELOPMENT.md#project--nuget-package-boundaries). |

Desktop path: choose the Mac Apple Silicon DMG or Windows x64 MSI on the release page, install and open it, then **☰ → Models → Download → Use** and send a message. PKG/ZIP alternatives exist; Windows may need WebView2. The built-in catalog starts at 12 GB system RAM and downloads weights separately. Older releases may lack desktop assets; [Desktop install and first chat](tensoragent_desktop.md) covers package verification and limits.

The CLI/server also have [self-contained release archives](https://github.com/zhongkaifu/TensorSharp/releases) for Windows/Linux x64 CPU/CUDA and macOS arm64. Packages reflect their own tag, not every current-source feature. The commands below build source; they do not validate downloaded release archives.

<a id="prerequisites"></a>
<a id="install-and-first-run"></a>
## 2 · Install and match your platform

<a id="install-dotnet"></a>
### Source builds need the .NET 10 SDK

| Platform | Install and check |
| --- | --- |
| Windows | PowerShell: `winget install Microsoft.DotNet.SDK.10`. |
| macOS | Install the .NET 10 SDK from [Microsoft](https://learn.microsoft.com/en-us/dotnet/core/install/macos): Arm64 for Apple Silicon, x64 for Intel. |
| Linux | Follow the [distribution instructions](https://learn.microsoft.com/en-us/dotnet/core/install/linux), configure its package feed and install the .NET 10 SDK (commonly `dotnet-sdk-10.0`). |

Install the full SDK; the .NET Runtime alone cannot build this source. Open a new terminal and confirm a `10.0.x` SDK is listed:

```bash
dotnet --list-sdks
```

<a id="e4b-fast-lane"></a>
<a id="pick-a-backend"></a>
### Choose the backend for this machine

| Machine | Value to use below | Native source-build prerequisites |
| --- | --- | --- |
| Apple Silicon Mac | `ggml_metal` | Git, curl, CMake 3.20+ and Xcode command-line tools. |
| Windows / Linux + NVIDIA | `ggml_cuda` | CMake, a C++ toolchain, NVIDIA driver and matching CUDA Toolkit; Windows needs Visual Studio C++ build tools. |
| Windows / Linux + Vulkan GPU | `ggml_vulkan` | CMake, C++/Vulkan build tools and a Vulkan 1.3 driver. |
| CPU, no GPU | `ggml_cpu` | CMake 3.20+ and a C++ compiler; this is still a native build. |

Examples use `ggml_cpu`; replace the backend with one value from the table. Before a CUDA build set `TENSORSHARP_GGML_NATIVE_ENABLE_CUDA=ON`; for Vulkan use `TENSORSHARP_GGML_NATIVE_ENABLE_VULKAN=ON`. In Windows PowerShell the CUDA form is `$env:TENSORSHARP_GGML_NATIVE_ENABLE_CUDA="ON"`; on macOS/Linux it is `export TENSORSHARP_GGML_NATIVE_ENABLE_CUDA=ON`. Metal needs no CUDA switch. See the [backend and native-build guide](../USAGE.md#compute-backends) for complete installation, automatic Vulkan tooling and MLX prerequisites.

<a id="build"></a>
### Get the source

```bash
git clone https://github.com/zhongkaifu/TensorSharp.git
cd TensorSharp
```

Run the application project below; its first invocation restores and builds it. You do not need to build the whole solution and desktop apps first. Git, curl and network access fetch source, dependencies and the model; the first native build fetches upstream ggml.

<a id="download"></a>
## 3 · Download one text model

This example uses the public [Gemma 4 E4B Q8_0](https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF/blob/main/gemma-4-E4B-it-Q8_0.gguf), a **7.48 GiB** file. Disk size is not the runtime memory requirement: leave room for KV cache and execution. A smaller Q4_K_M is in the same repository; check the [Gemma 4 model card](models/gemma4.md) and [other model downloads](../MODEL_DOWNLOADS.md) before changing model or quantization. **Text input needs no mmproj.**

<details>
<summary>Windows PowerShell download command</summary>

```powershell
curl.exe --create-dirs --fail -L "https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q8_0.gguf?download=true" -o models/gemma-4-E4B-it-Q8_0.gguf
```


</details>

<details>
<summary>macOS / Linux download command</summary>

```bash
curl --create-dirs --fail -L "https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q8_0.gguf?download=true" -o models/gemma-4-E4B-it-Q8_0.gguf
```


</details>

<a id="first-run"></a>
## 4 · Get your first text reply

Run from the repository root and replace `ggml_cpu` using the platform table. A one-shot CLI text prompt comes from a file via `--input`; `--prompt` belongs to image/video generation. This example uses greedy sampling and at most 32 output tokens. The first native build can take several minutes.

<details>
<summary>Terminal: Windows PowerShell</summary>

```powershell
'What is 2+2? Reply with only the number.' | Set-Content -Encoding utf8 prompt.txt
dotnet run --project TensorSharp.Cli -c Release -p:TensorSharpSkipMlxNative=true -- --model models/gemma-4-E4B-it-Q8_0.gguf --input prompt.txt --backend ggml_cpu --temperature 0 --max-tokens 32
```


</details>

<details>
<summary>Terminal: macOS / Linux</summary>

```bash
printf '%s\n' 'What is 2+2? Reply with only the number.' > prompt.txt
dotnet run --project TensorSharp.Cli -c Release -p:TensorSharpSkipMlxNative=true -- --model models/gemma-4-E4B-it-Q8_0.gguf --input prompt.txt --backend ggml_cpu --temperature 0 --max-tokens 32
```


</details>

Prefer a browser or HTTP? Start the server with the same model and selected backend instead, and leave this terminal running:

```bash
dotnet run --project TensorSharp.Server.Host -c Release -p:TensorSharpSkipMlxNative=true -- --model models/gemma-4-E4B-it-Q8_0.gguf --backend ggml_cpu --host 127.0.0.1 --port 8080 --max-tokens 32
```

Open [http://127.0.0.1:8080/](http://127.0.0.1:8080/) and send the same arithmetic prompt. These examples explicitly use loopback and port 8080; the actual default is `0.0.0.0:5000`, and macOS commonly reserves 5000 for AirPlay. The server has no built-in authentication or TLS; remote deployments need their own authenticated HTTPS boundary.

<a id="verify"></a>
## 5 · Verify this path

| Check | Expected result |
| --- | --- |
| CLI or browser reply | Nonempty assistant text; the arithmetic example should answer `4`. Log messages alone are not a model reply. |
| Process and errors | The CLI exits normally; the server stays running. Resolve download, build or model-load errors at that stage. |
| Server liveness | `GET /health` returns `TensorSharp.Server is running`. This checks liveness, not model readiness or quality. |
| Model and API | `GET /v1/models` lists the configured model ID. For a direct request, use the JSON below. |

Check from another terminal; in Windows PowerShell use `curl.exe` instead of `curl`:

```bash
curl --fail http://127.0.0.1:8080/health
curl --fail http://127.0.0.1:8080/v1/models
```

Save this as `request.json`; if the model ID differs, use the ID returned by `/v1/models`:

```json
{
  "model": "gemma-4-E4B-it-Q8_0",
  "messages": [
    {
      "role": "user",
      "content": "What is 2+2? Reply with only the number."
    }
  ],
  "temperature": 0,
  "max_tokens": 32,
  "stream": false
}
```

```bash
curl --fail http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" --data-binary "@request.json"
```

Check that `choices[0].message.content` contains text. With the Web UI available, `/` opens chat; with `--no-webui` or no Web UI files, `/` returns liveness text. **The stable liveness route is `/health` in either case.** For terminal conversations use CLI `--interactive`; see the [CLI reference](../USAGE.md#console-application) and [HTTP API examples](../TensorSharp.Server.Host/API_EXAMPLES.md) for full formats.

<a id="troubleshooting"></a>
## If a step fails

| Symptom | Next check |
| --- | --- |
| No `10.0.x` SDK | Install the SDK and open a new terminal; the runtime alone is insufficient. |
| CMake, compiler or GPU library error | Check the [backend and native-build guide](../USAGE.md#compute-backends); when native builds are unavailable, try the managed CLI path below. |
| Model path or memory error | Confirm a complete download, matching filename and room beyond weight size; choose a supported quantization. |
| Server cannot listen | Keep `--host 127.0.0.1`, choose a free `--port`, and update request URLs together. |
| Liveness works but no model reply | Read load logs and `/v1/models`, then check the request against [HTTP API examples](../TensorSharp.Server.Host/API_EXAMPLES.md). Liveness is not an inference check. |

<details>
<summary>No native toolchain: managed CPU CLI path</summary>

`cpu` is pure C#, distinct from native `ggml_cpu`. It emphasizes portability and debugging; check model/feature coverage. Explicitly skip both native builds:

```bash
dotnet run --project TensorSharp.Cli -c Release -p:TensorSharpSkipGgmlNative=true -p:TensorSharpSkipMlxNative=true -- --model models/gemma-4-E4B-it-Q8_0.gguf --input prompt.txt --backend cpu --temperature 0 --max-tokens 32
```


</details>

<a id="next"></a>
## After the reply, choose your next task

| Task | Existing guide and key conditions |
| --- | --- |
| Image generation/editing | [Qwen-Image-2.1](models/qwenimage21.md#lora-plug-ins) covers matching 4–8-step LoRA recipes, dimensions, CFG and licensing; some plug-ins are non-commercial, so check each configuration's source and license. |
| Video and audio | [MiniMax-H3](models/minimax-h3.md) covers FL2VA/Ref2VA, 32 kHz audio, CFG 1 and Turbo parameters; [Wan](models/wan.md) covers checkpoint-dependent base/distilled recipes. |
| LTX 2.5 | [Two-stage recipe and conditioning](models/ltx-2.5.md) covers five base files (43.08 GB), the experimental 48 GB tier, a 1024×1024×17 preview and 8+3 steps. Base files are ungated; the optional Ingredients IC-LoRA (~1.31 GB) needs approved access. |
| Embeddings, knowledge and skills | [Embeddings](embeddings.md) · [knowledge library](knowledge-system.md) · [Agent Skills](agent_skills.md). Check their models, extraction coverage and execution permissions first. |
| More GPUs and serving throughput | [Multi-GPU](../USAGE.md#tensor-parallelism--distributed-inference) · [cache and speculation](PAGED_ATTENTION_AND_CONTINUOUS_BATCHING.md). `--tp` and `--layer-split` are mutually exclusive and model-specific. |

Reuse options through [JSON configuration files](../config/README.md); configurations can download resources, so inspect the file first. Source may lead published packages; check [project status](PROJECT_STATUS.md) before choosing a feature.

<a id="reference-links"></a>
<a id="build-tensoragent-from-the-same-solution"></a>
<a id="text-and-code-embeddings"></a>
<a id="new-video-embedding-and-skill-workflows"></a>
<a id="reuse-your-files-as-a-knowledge-library"></a>
<a id="make-it-fast"></a>
### Advanced setups and references

- Desktop source builds: [TensorAgent build and validation](../TensorAgent/README.md) retains MAUI workloads, platform app heads, `TensorSharpSkipTensorAgentApp` and missing-prerequisite skip behavior. Building a test project does not run its tests.
- Embeddings: the [separate service guide](embeddings.md) retains MiniLM/Snowflake downloads, `--embeddings`, managed/native backends and `/v1/embeddings` / `/api/embed` requests. Run chat and embeddings on different ports.
- Knowledge: the [knowledge guide](knowledge-system.md) retains `TS_KNOWLEDGE_DIRECTORY`, `knowledgeCollections`, BM25/optional embeddings, Reindex and media-extraction limits. Uploading still requires selecting “Use in this chat”.
- Multimodal input: the [Gemma 4 model card](models/gemma4.md) retains matching projector filenames and explicit `--mmproj`; the server does not auto-detect a projector. The [MiniMax-H3 guide](models/minimax-h3.md) retains the ~46 GB config, 1.38 GB Turbo factors, sampling constraints and `TS_VIDEO_TOKENIZER`.
- DGX Spark / GB10: the [experimental CUDA 13 / Linux ARM64 build](../DEVELOPMENT.md#gb10--dgx-spark-build-container-experimental) targets a single GB10, not x64 CUDA archives or generic ARM64 GPUs. Its historical hardware check predates upstream reintegration; hosted CI checks CPU/archive paths.

<details>
<summary>Compatibility notes for existing advanced deployments</summary>

For native ggml source builds, `TENSORSHARP_GGML_NO_UPDATE=1` skips subsequent network updates. The [cache guide](PAGED_ATTENTION_AND_CONTINUOUS_BATCHING.md#radix-prefix-cache) retains default Radix behavior, `TS_SCHED_PREFIX_CACHE=0`, `--no-prefix-cache` and server disk checkpoints. The earlier guide's environment-specific Ubuntu settings remain below for matching CUDA/package-source environments; they are not part of the normal first run:

```bash
# RunPod Ubuntu 24.04 images using this CUDA compatibility directory:
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/compat:$LD_LIBRARY_PATH
# Older Ubuntu releases needing the .NET backports package source:
add-apt-repository ppa:dotnet/backports
apt update && apt install dotnet-sdk-10.0
```


</details>
