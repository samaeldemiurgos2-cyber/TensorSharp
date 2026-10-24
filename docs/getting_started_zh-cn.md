<a id="getting-started"></a>
<a id="快速上手指南"></a>

[English](getting_started.md) | [中文](getting_started_zh-cn.md)

# 快速上手

先得到一次文本回复：选择界面与平台 → 安装 → 下载一个模型 → 发出提示 → 检查结果。图像、视频、多 GPU 与技能在完成这一步后再选。

<a id="choose"></a>
<a id="when-to-try-tensorsharp"></a>
<a id="什么时候适合尝试-tensorsharp"></a>
## 1 · 选择界面

| 想完成什么 | 从哪里开始 |
| --- | --- |
| 安装后直接聊天 | [桌面安装与首次聊天](tensoragent_desktop_zh-cn.md)。Mac/Windows 安装包自带运行时；无需下文的源码构建。 |
| 使用终端、浏览器或兼容 HTTP 客户端 | 继续下文的 CLI/服务端源码路径；使用同一个 GGUF 文本模型。 |
| 把引擎嵌入 C# 应用 | 先完成一次文本运行，再看[包分层与集成](../DEVELOPMENT_zh-cn.md#项目--nuget-包分层)。 |

桌面路径：从发布页选择 Mac Apple Silicon DMG 或 Windows x64 MSI，安装并打开，在 **☰ → 模型 → 下载 → 使用** 后发送消息。也有 PKG/ZIP；Windows 可能需要 WebView2。内置目录从 12 GB 系统内存起步，权重另下；较早发布可能没有桌面资产。具体包校验与限制见[桌面安装与首次聊天](tensoragent_desktop_zh-cn.md)。

CLI/服务端也有 [自包含发布归档](https://github.com/zhongkaifu/TensorSharp/releases)，适用于 Windows/Linux x64 CPU/CUDA 和 macOS arm64。发布包对应自己的标签，不能据此假定包含当前源码的全部功能；下文命令是源码构建，不是对下载归档的验证。

<a id="prerequisites"></a>
<a id="install-and-first-run"></a>
<a id="安装与首次运行"></a>
## 2 · 安装并匹配平台

<a id="install-dotnet"></a>
### 源码构建需要 .NET 10 SDK

| 平台 | 安装与检查 |
| --- | --- |
| Windows | PowerShell：`winget install Microsoft.DotNet.SDK.10`。 |
| macOS | 从 [Microsoft](https://learn.microsoft.com/zh-cn/dotnet/core/install/macos) 安装 .NET 10 SDK；Apple 芯片选 Arm64，Intel 选 x64。 |
| Linux | 按 [发行版说明](https://learn.microsoft.com/zh-cn/dotnet/core/install/linux) 配置软件源并安装 .NET 10 SDK（通常为 `dotnet-sdk-10.0`）。 |

需要完整 SDK，只有 .NET Runtime 不足以构建。安装后打开新终端，确认列表中有 `10.0.x`：

```bash
dotnet --list-sdks
```

<a id="e4b-fast-lane"></a>
<a id="pick-a-backend"></a>
<a id="选择后端"></a>
### 为这台机器选择后端

| 机器 | 下文使用的值 | 原生源码构建还需要 |
| --- | --- | --- |
| Apple Silicon Mac | `ggml_metal` | Git、curl、CMake 3.20+、Xcode 命令行工具。 |
| Windows / Linux + NVIDIA | `ggml_cuda` | CMake、C++ 工具链、NVIDIA 驱动与匹配的 CUDA Toolkit；Windows 需要 Visual Studio C++ 构建工具。 |
| Windows / Linux + Vulkan GPU | `ggml_vulkan` | CMake、C++/Vulkan 构建工具与 Vulkan 1.3 驱动。 |
| CPU，无 GPU | `ggml_cpu` | CMake 3.20+ 与 C++ 编译器；它仍是原生构建。 |

示例默认使用 `ggml_cpu`；把运行命令的后端改成表中的一个值即可。CUDA 构建前设置 `TENSORSHARP_GGML_NATIVE_ENABLE_CUDA=ON`；Vulkan 使用 `TENSORSHARP_GGML_NATIVE_ENABLE_VULKAN=ON`。Windows PowerShell 的写法是 `$env:TENSORSHARP_GGML_NATIVE_ENABLE_CUDA="ON"`；macOS/Linux 是 `export TENSORSHARP_GGML_NATIVE_ENABLE_CUDA=ON`。Metal 不需要 CUDA 开关。完整安装、Vulkan 工具链自动配置与 MLX 的额外依赖见[后端与原生构建指南](../USAGE_zh-cn.md#计算后端)。

<a id="build"></a>
### 取得源码

```bash
git clone https://github.com/zhongkaifu/TensorSharp.git
cd TensorSharp
```

只运行下面的应用项目即可；首次运行会还原与构建，不必先构建包含桌面应用的整个解决方案。Git、curl 与网络访问用于下载源码、依赖及模型；首次原生构建会取得上游 ggml。

<a id="download"></a>
## 3 · 下载一个文本模型

本例使用公开的 [Gemma 4 E4B Q8_0](https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF/blob/main/gemma-4-E4B-it-Q8_0.gguf)，文件为 **7.48 GiB**。磁盘文件大小不是运行内存需求：还要为 KV 缓存与执行预留内存。同仓库有较小的 Q4_K_M；换量化或模型前核对[Gemma 4 模型卡](models/gemma4_zh-cn.md)与[其他模型下载](../MODEL_DOWNLOADS_zh-cn.md)。**这次只输入文本，不需要 mmproj。**

<details>
<summary>Windows PowerShell 下载命令</summary>

```powershell
curl.exe --create-dirs --fail -L "https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q8_0.gguf?download=true" -o models/gemma-4-E4B-it-Q8_0.gguf
```


</details>

<details>
<summary>macOS / Linux 下载命令</summary>

```bash
curl --create-dirs --fail -L "https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q8_0.gguf?download=true" -o models/gemma-4-E4B-it-Q8_0.gguf
```


</details>

<a id="first-run"></a>
## 4 · 得到第一条文本回复

在仓库根目录执行，按上表替换 `ggml_cpu`。CLI 的一次性文本提示来自 `--input` 文件；`--prompt` 是生成图像/视频的参数。下例使用贪心采样，最多输出 32 token，首次构建可能需要数分钟。

<details>
<summary>终端：Windows PowerShell</summary>

```powershell
'What is 2+2? Reply with only the number.' | Set-Content -Encoding utf8 prompt.txt
dotnet run --project TensorSharp.Cli -c Release -p:TensorSharpSkipMlxNative=true -- --model models/gemma-4-E4B-it-Q8_0.gguf --input prompt.txt --backend ggml_cpu --temperature 0 --max-tokens 32
```


</details>

<details>
<summary>终端：macOS / Linux</summary>

```bash
printf '%s\n' 'What is 2+2? Reply with only the number.' > prompt.txt
dotnet run --project TensorSharp.Cli -c Release -p:TensorSharpSkipMlxNative=true -- --model models/gemma-4-E4B-it-Q8_0.gguf --input prompt.txt --backend ggml_cpu --temperature 0 --max-tokens 32
```


</details>

想用浏览器或 HTTP？改为启动同一模型的服务端，在运行命令中选择相同后端，并让这个终端保持运行：

```bash
dotnet run --project TensorSharp.Server.Host -c Release -p:TensorSharpSkipMlxNative=true -- --model models/gemma-4-E4B-it-Q8_0.gguf --backend ggml_cpu --host 127.0.0.1 --port 8080 --max-tokens 32
```

打开 [http://127.0.0.1:8080/](http://127.0.0.1:8080/)，发送同一个算术问题。这里显式使用本机回环与 8080 端口；默认其实是 `0.0.0.0:5000`，macOS 的 5000 常被 AirPlay 占用。服务端没有内置鉴权或 TLS；对外部署需自己的鉴权 HTTPS 边界。

<a id="verify"></a>
## 5 · 验证这条路径

| 检查 | 预期结果 |
| --- | --- |
| CLI 或浏览器回复 | 出现非空的助手文本；本例算术应回答 `4`。日志本身不算模型回复。 |
| 进程与错误 | CLI 正常结束；服务端保持运行。下载、构建或模型加载报错时，先解决该阶段。 |
| 服务端存活 | `GET /health` 返回 `TensorSharp.Server is running`；这只是存活检查，不证明模型已就绪或回复质量。 |
| 模型与 API | `GET /v1/models` 列出配置的模型 ID。需要直接请求时，使用下方 JSON。 |

在另一个终端检查；Windows PowerShell 将 `curl` 改为 `curl.exe`：

```bash
curl --fail http://127.0.0.1:8080/health
curl --fail http://127.0.0.1:8080/v1/models
```

把下面内容保存为 `request.json`；若模型 ID 不同，使用 `/v1/models` 返回的值：

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

检查 JSON 中的 `choices[0].message.content` 有文本。正常 Web UI 部署中 `/` 打开聊天；加 `--no-webui` 或没有 Web UI 文件时，`/` 才返回存活文本。**稳定的存活路由始终是 `/health`。** 想继续终端多轮聊天，可用 CLI 的 `--interactive`；完整格式见[CLI 参考](../USAGE_zh-cn.md#控制台应用)与[HTTP API 示例](../TensorSharp.Server.Host/API_EXAMPLES_zh-cn.md)。

<a id="troubleshooting"></a>
## 卡住时，按阶段排查

| 症状 | 下一步 |
| --- | --- |
| 没有 `10.0.x` SDK | 重新安装 SDK 后打开新终端，不要只安装 Runtime。 |
| CMake、编译器或 GPU 库错误 | 核对[后端与原生构建指南](../USAGE_zh-cn.md#计算后端)；不能构建原生代码时先使用下面的托管 CLI 路径。 |
| 模型路径或内存错误 | 确认下载完成、文件名匹配、内存不止覆盖权重大小；使用模型卡支持的量化。 |
| 服务端无法监听 | 保持 `--host 127.0.0.1`，换一个空闲 `--port`，并同时更新请求 URL。 |
| 有存活响应但没有模型回复 | 查看加载日志与 `/v1/models`，再按[HTTP API 示例](../TensorSharp.Server.Host/API_EXAMPLES_zh-cn.md)检查请求；存活不是推理检查。 |

<details>
<summary>没有原生工具链：托管 CPU CLI 路径</summary>

`cpu` 是纯 C# 路径，与原生 `ggml_cpu` 不同。它更侧重可移植性与调试，模型/功能覆盖需核对；显式跳过两个原生构建：

```bash
dotnet run --project TensorSharp.Cli -c Release -p:TensorSharpSkipGgmlNative=true -p:TensorSharpSkipMlxNative=true -- --model models/gemma-4-E4B-it-Q8_0.gguf --input prompt.txt --backend cpu --temperature 0 --max-tokens 32
```


</details>

<a id="next"></a>
## 回复成功后，再选择下一项任务

| 任务 | 现有指南与保留的关键条件 |
| --- | --- |
| 图像生成与编辑 | [Qwen-Image-2.1](models/qwenimage21_zh-cn.md#lora-插件)：匹配 LoRA 的 4–8 步配方、分辨率、CFG 与许可证；部分插件不可商用，检查每个配置的来源与许可。 |
| 视频与音频 | [MiniMax-H3](models/minimax-h3_zh-cn.md)（FL2VA/Ref2VA、32 kHz 音频、CFG 1、Turbo 参数）与 [Wan](models/wan_zh-cn.md)（检查点决定基础/蒸馏步数）。 |
| LTX 2.5 | [两阶段配方与条件](models/ltx-2.5_zh-cn.md)：五个基础文件 43.08 GB、实验性 48 GB 层级、1024×1024×17 起步、8+3 步；基础仓库不设访问门槛，额外 Ingredients IC-LoRA 约 1.31 GB 需批准访问。 |
| 嵌入、知识与技能 | [嵌入](embeddings_zh-cn.md) · [知识库](knowledge-system_zh-cn.md) · [Agent Skills](agent_skills.md)。先检查各自模型、提取覆盖与执行权限。 |
| 多 GPU 与更快的服务 | [多 GPU](../USAGE_zh-cn.md#张量并行与分布式推理) · [缓存与投机解码](PAGED_ATTENTION_AND_CONTINUOUS_BATCHING_zh-cn.md)。`--tp` 与 `--layer-split` 互斥，支持范围由模型决定。 |

可用 [JSON 配置文件](../config/README.md)复用参数；文件可以下载资源，所以先检查配置内容。分支源码可能领先于发布包，选择功能前查看[项目状态](PROJECT_STATUS_zh-cn.md)。

<a id="reference-links"></a>
<a id="build-tensoragent-from-the-same-solution"></a>
<a id="从同一解决方案构建-tensoragent"></a>
<a id="text-and-code-embeddings"></a>
<a id="文本与代码嵌入"></a>
<a id="new-video-embedding-and-skill-workflows"></a>
<a id="新视频嵌入与技能工作流"></a>
<a id="reuse-your-files-as-a-knowledge-library"></a>
<a id="将文件复用为知识库"></a>
<a id="make-it-fast"></a>
<a id="让它跑得更快"></a>
### 高级设置与参考

- 桌面源码构建：[TensorAgent 构建与验证](../TensorAgent/README.md)保留 MAUI 工作负载、平台应用头、`TensorSharpSkipTensorAgentApp` 与缺失依赖跳过规则；构建测试项目不会运行测试。
- 嵌入：[独立服务指南](embeddings_zh-cn.md)保留 MiniLM/Snowflake 下载、`--embeddings`、纯 C#/原生后端及 `/v1/embeddings`、`/api/embed` 示例；与聊天服务使用不同端口。
- 知识：[知识库指南](knowledge-system_zh-cn.md)保留 `TS_KNOWLEDGE_DIRECTORY`、`knowledgeCollections`、BM25/可选嵌入、Reindex 与媒体提取限制；上传后仍需选择“在此聊天使用”。
- 多模态输入：[Gemma 4 模型卡](models/gemma4_zh-cn.md)保留匹配 mmproj 文件名与显式 `--mmproj` 用法；服务端不会自动发现投影器。[MiniMax-H3 的专门指南](models/minimax-h3_zh-cn.md)保留约 46 GB 配置下载、Turbo 1.38 GB、采样器约束与 `TS_VIDEO_TOKENIZER`。
- DGX Spark / GB10：[实验性 CUDA 13 / Linux ARM64 构建](../DEVELOPMENT_zh-cn.md#gb10--dgx-spark-构建容器实验性)面向单 GB10，不能使用 x64 CUDA 包；历史硬件检查不能证明当前上游重集成，托管 CI 只检查 CPU/归档路径。

<details>
<summary>已有高级部署的兼容提示</summary>

使用原生 ggml 源码构建时，`TENSORSHARP_GGML_NO_UPDATE=1` 跳过后续网络更新。默认 Radix 前缀缓存、`TS_SCHED_PREFIX_CACHE=0`、`--no-prefix-cache` 与服务端磁盘检查点的关系见[缓存指南](PAGED_ATTENTION_AND_CONTINUOUS_BATCHING_zh-cn.md#radix-前缀缓存)。下面保留原指南的特定 Ubuntu 环境设置，仅在匹配的 CUDA/软件源环境使用，常规首次运行不需要：

```bash
# RunPod Ubuntu 24.04 images using this CUDA compatibility directory:
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/compat:$LD_LIBRARY_PATH
# Older Ubuntu releases needing the .NET backports package source:
add-apt-repository ppa:dotnet/backports
apt update && apt install dotnet-sdk-10.0
```


</details>
