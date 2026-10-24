# MiniMax-M3.1-Flash（`minimax-m3`）

[← 返回模型索引](README_zh-cn.md) | [English](minimax-m3.md)

TensorSharp 支持 [MiniMax-M3.1-Flash](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash)
的 `minimax-m3` GGUF 架构（由 [Unsloth](https://huggingface.co/unsloth/MiniMax-M3.1-Flash-GGUF) 转换）。
它是总参数 428B、每 token 激活约 23B 的稀疏 MoE。解码器、MiniMax 稀疏注意力、NVFP4 专家路径、
张量并行、DSpark 草稿器与视觉塔都在 TensorSharp 自有代码中实现；拉取的 ggml 依赖（上游 `da8e2a5`）
保持不变。这是一个**多 GPU 服务器模型**：UD-Q4_K_XL 检查点为 232.3 GiB，实测环境是八张 46 GB 的 A40。

| 属性 | 实现 |
|---|---|
| 架构 | `minimax-m3`，别名 `minimax_m3` |
| 模型 | 60 层，隐藏宽度 6144：前 3 层为稠密层（SwiGLU-OAI FFN，宽 12288），其后 57 个 MoE 层，128 个路由专家（sigmoid 路由 top-4，带 F32 选择偏置，权重重新归一化后乘 2.0）加 1 个共享专家（宽 3072）。64 个 query 头 / 4 个 KV 头，头维度 128，逐头 Q/K RMSNorm，NeoX RoPE 只作用于 128 维中的前 64 维（base 1e7）。词表 200,064；GGUF 声明 1,048,576 token 上下文 |
| 注意力 | **每一层**都是 MiniMax 稀疏注意力（MSA）：索引器（4 个 128 维索引头，每个 KV 头一个）为 128 行的块打分，每个 KV 头关注得分最高的 16 个块，且始终包含自身所在块。位置 2048 之前所有可见块都会被选中，因此那里的稠密因果注意力是精确的。稀疏区间由 TensorSharp 的 CUDA MSA 内核（[ggml_ops_minimax_msa.cu](../../TensorSharp.GGML.Native/ggml_ops_minimax_msa.cu)）运行，上游 ggml 算子作为后备（[详情](#注意力模式)） |
| 解码器 | [MiniMaxM3Model](../../TensorSharp.Models/Models/MiniMaxM3/MiniMaxM3Model.cs) 解析并校验检查点；[原生整模型执行器](../../TensorSharp.GGML.Native/ggml_ops_minimax_m3_exec.cpp)自行加载分片 GGUF，每个微批提交一张 GGML 图 |
| 专家 | NVFP4 路由专家，带逐专家的 F32 输出缩放（在激活之前作用于 gate 与 up，在路由权重之前作用于 down）；注意力、共享专家与稠密 FFN 为 Q8_0；索引器与 router 为 F32；嵌入与输出头为 BF16 |
| 后端 | 服务后端是 `ggml_cuda`。`ggml_cpu` 在单个 CPU 设备上运行同一执行器，是正确性路径。`mlx`、`ggml_metal`、`ggml_vulkan`、`cuda`（直连 CUDA）与 `cpu`（纯 C#）会拒绝加载，退出码 2 |
| GPU 放置 | `--layer-split N` 把整层放到 N 张 GPU 上（最多 16 张）。`--tp N` 通过上游 ggml 的 meta 后端把每一层切分到前 N 张 CUDA GPU（2 到 16 张）。**两者都不给时执行器只用一张 GPU**，232 GiB 的检查点放不下（见[运行](#运行)） |
| 专家卸载 | `--n-cpu-moe N` 让第 0..N-1 层的路由专家留在系统内存中；`TS_MINIMAX_M3_CPU_MOE=auto` 只把放不下的专家栈留在主机上。仅限按层切分 |
| KV 缓存 | 逐序列 F16 槽位，位于读取它们的 GPU 上：K、V 与 MSA 索引键，60 层合计每 token 135 KiB。每一行都保留，因此回退到任意 token 都是精确的。块量化的 `--kv-cache-dtype` 会被拒绝；`f32` 会带警告地以 F16 运行 |
| 连续批处理 | 跨序列槽位的批量 decode，以及并发文本提示在共享图中的批量 prefill |
| 前缀缓存 | 从已完成槽位续写对话、精确回退，以及克隆到新对话中的共享系统提示 / 工具检查点 |
| 图像 | 需要匹配的 `mmproj`（`clip.projector_type = minimax_m3`），需自行转换（[见下文](#生成视觉投影器与-dspark-草稿器)）；视频以抽样帧输入；拒绝音频 |
| 投机解码 | 通过 `--draft-model` 使用 DSpark 块草稿器（仅按层切分），以及无需权重的 n-gram 草稿器（`--spec --spec-type ngram`）。`--tp` 下两者都不运行 |
| 聊天协议 | 自有渲染器，与 Hugging Face 模板逐字节一致；自适应思考；带 schema 类型推断的命名空间 XML 工具调用；由语法约束的 `tool_choice` 与 `response_format`（[详情](#聊天格式推理工具与-json)） |

## 下载

### 文本模型

```bash
# UD-Q4_K_XL：七个分片，249,402,105,310 字节（232.3 GiB）。所有分片放在同一目录，
# --model 指向 -00001-of- 分片。
hf download unsloth/MiniMax-M3.1-Flash-GGUF --include "UD-Q4_K_XL/*" --local-dir models/minimax-m3
```

Unsloth 仓库标注为仍在制作中。撰写本文时它只有文本模型：没有 `mmproj`，也没有 DSpark
草稿器。请按下文自行生成这两个文件，或只运行文本。

### 生成视觉投影器与 DSpark 草稿器

两个伴随文件都用 [unslothai/llama.cpp PR #320](https://github.com/unslothai/llama.cpp/pull/320)
（分支 `minimax-m31`，提交 `ef8f94cc`）的转换器从 Hugging Face 发布版转换。只需要配置、分词器与模板文件、
包含视觉塔的两个 safetensors 分片（48 个中的第 13 与第 48 个）以及 `dspark/` 目录：

```bash
hf download MiniMaxAI/MiniMax-M3.1-Flash \
    --include "*.json" "*.py" "*.jinja" "merges.txt" \
              "model-00013-of-00048.safetensors" "model-00048-of-00048.safetensors" "dspark/*" \
    --local-dir models/MiniMax-M3.1-Flash-hf

git clone https://github.com/unslothai/llama.cpp llama.cpp-m31
git -C llama.cpp-m31 fetch origin pull/320/head:minimax-m31
git -C llama.cpp-m31 checkout ef8f94cc
python3 -m venv convert-venv
convert-venv/bin/pip install -r llama.cpp-m31/requirements/requirements-convert_hf_to_gguf.txt

# mmproj 步骤会读取索引中列出的每个分片，所以给它一个索引只列出第 13、48 分片的目录。
HF=$PWD/models/MiniMax-M3.1-Flash-hf; VIS=$PWD/models/MiniMax-M3.1-Flash-hf-vision
mkdir -p "$VIS"
for f in "$HF"/*.json "$HF"/*.py "$HF"/*.jinja "$HF"/merges.txt "$HF"/model-000{13,48}-of-00048.safetensors; do
  ln -sf "$f" "$VIS/"
done
rm "$VIS/model.safetensors.index.json"
convert-venv/bin/python -I - "$HF" "$VIS" <<'PY'
import json, sys
hf, vis = sys.argv[1:]
index = json.load(open(f"{hf}/model.safetensors.index.json"))
keep = {"model-00013-of-00048.safetensors", "model-00048-of-00048.safetensors"}
index["weight_map"] = {k: v for k, v in index["weight_map"].items() if v in keep}
json.dump(index, open(f"{vis}/model.safetensors.index.json", "w"))
PY

cd llama.cpp-m31
# 视觉投影器，1,735,294,400 字节
../convert-venv/bin/python convert_hf_to_gguf.py "$VIS" --mmproj --outtype bf16 \
    --outfile ../models/minimax-m3/mmproj-MiniMax-M3.1-Flash-BF16.gguf
# DSpark 草稿器（GGUF 架构 dflash），3,925,395,968 字节
../convert-venv/bin/python convert_hf_to_gguf.py "$HF/dspark" --target-model-dir "$HF" --outtype bf16 \
    --outfile ../models/minimax-m3/MiniMax-M3.1-Flash-DSpark-BF16.gguf
```

TensorSharp 在映射 232 GiB 模型之前就检查这两个文件：投影器必须声明
`clip.projector_type = minimax_m3` 并投影到 6144 维；草稿器必须是带 Markov 头的 `dflash` GGUF，
采用逐行预测（`sample_from_anchor`）的块布局、`swigluoai`、每个草稿层都有滑动窗口，且目标层、
词表与 mask token 与本模型匹配。其他情况都会被拒绝，并在消息中指出不匹配之处。

## 运行

232 GiB 的检查点需要多张 GPU。TensorSharp 默认只用**一张** GPU，
所以请显式给出放置方式：在八张 46 GB GPU 上用 `--layer-split 8` 或 `--tp 8`。不给时，
加载会因模型放不进一张 GPU 而被拒绝，消息中会提到 `--n-cpu-moe` 与 `TS_MINIMAX_M3_CPU_MOE=auto`。
请始终传入 `--backend ggml_cuda`：服务器在 macOS 以外的默认后端是 `ggml_cpu`，而本模型在它上面只是正确性路径。

```bash
# 实测的按层切分配置：八张 A40，72K 上下文供一个请求加共享前缀检查点使用
#（TS_MINIMAX_M3_PLAN_CTX_TOKENS = 上下文 x 2）。
MAX_CONTEXT=73728 TS_MINIMAX_M3_PLAN_CTX_TOKENS=147456 \
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll \
    --model models/minimax-m3/UD-Q4_K_XL/MiniMax-M3.1-Flash-UD-Q4_K_XL-00001-of-00007.gguf \
    --mmproj models/minimax-m3/mmproj-MiniMax-M3.1-Flash-BF16.gguf \
    --backend ggml_cuda --layer-split 8 --port 5000

# 同样八张 GPU 上的张量并行：decode 更快，浅深度时尤其明显
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll \
    --model models/minimax-m3/UD-Q4_K_XL/MiniMax-M3.1-Flash-UD-Q4_K_XL-00001-of-00007.gguf \
    --backend ggml_cuda --tp 8 --port 5000

# DSpark 投机解码（仅按层切分）
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll \
    --model models/minimax-m3/UD-Q4_K_XL/MiniMax-M3.1-Flash-UD-Q4_K_XL-00001-of-00007.gguf \
    --backend ggml_cuda --layer-split 8 \
    --draft-model models/minimax-m3/MiniMax-M3.1-Flash-DSpark-BF16.gguf --port 5000

# GPU 较少时（未在完整检查点上测量）：第 0..19 层（其中 17 个 MoE 层，约 65 GiB）的路由专家放在系统内存
MAX_CONTEXT=16384 CUDA_VISIBLE_DEVICES=0,1,2,3 \
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll \
    --model models/minimax-m3/UD-Q4_K_XL/MiniMax-M3.1-Flash-UD-Q4_K_XL-00001-of-00007.gguf \
    --backend ggml_cuda --layer-split 4 --n-cpu-moe 20 --port 5000

# CLI 单次运行
echo "Give me three facts about the Moon." > prompt.txt
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll \
    --model models/minimax-m3/UD-Q4_K_XL/MiniMax-M3.1-Flash-UD-Q4_K_XL-00001-of-00007.gguf \
    --backend ggml_cuda --layer-split 8 --input prompt.txt --max-tokens 512
```

`--model` 可以指向分片集中的任意一个分片；TensorSharp 从第 1 个分片读取元数据，从所有分片读取张量。
服务器从不在模型旁边查找聊天模型的投影器，图像输入请传 `--mmproj`。启动日志会打印放置结果
（`MiniMax-M3.1 native executor: 60 layers over 8 GPU(s), context ..., ubatch 2048, layers with host experts ...`），
`--tp` 下还会打印每张 GPU 上的权重、K/V 与预留。

### 上下文长度与内存

GGUF 声明 1,048,576 token，任何实测环境都放不下。不设 `MAX_CONTEXT` 时，执行器把上下文限制为能放下的最大值并记录
`[minimax-m3] context capped to N tokens (asked for 1048576) ...; set MAX_CONTEXT to choose a size`。设置了
`MAX_CONTEXT` 时，放不下的上下文会拒绝加载。放置规划为所有槽位共预留 `TS_MINIMAX_M3_PLAN_CTX_TOKENS` 个
token 的 K/V 行，默认是一个完整上下文；如上面的启动命令，应设为上下文乘以（并发请求数加一，留给共享前缀检查点）。
槽位初始为 4,096 行，按需翻倍。

K、V 与索引键每 token 占 135 KiB：按层切分时一个 73,728 token 的序列为 9.5 GiB，每张 GPU 只保存自己那些层的行。
`--tp` 下每张 GPU 都保存完整副本，因此在八张 A40 上不设 `MAX_CONTEXT` 时，上下文被限制为 45,824 token。
每个 MoE 层的路由专家约 3.8 GiB，也就是 `--n-cpu-moe` 每多一层移到系统内存的量。使用 `TS_MINIMAX_M3_CPU_MOE=auto` 时，请同时设置 `MAX_CONTEXT`（或 `TS_MINIMAX_M3_PLAN_CTX_TOKENS`）：自动方案总能把更多专家移到主机来腾出空间，因此不会限制上下文，不设上限时它会为声明的 1,048,576 token 预留 K/V。

## 运行方式

- **每个微批一张图。** `ggml_backend_sched` 把每张图分到各 GPU 上（有主机专家时还有 CPU）。prefill 使用
  2048 行的微批（`TS_MINIMAX_M3_UBATCH`）；放不下的图会把这次调用拆成更小的微批运行。按形状缓存的
  LRU 图让 decode 逐 token 重放同一张图及其捕获的 CUDA graph。
- **层放置。** `--layer-split N` 把连续的整层分配给 N 张 GPU。在主机专家字节最少的方案中，选择最忙 GPU
  每 token 计算量最小的方案（流水线 prefill 的速度取决于最慢的一级），再选择 GPU 占用最均匀的方案。
  `TS_MINIMAX_M3_LAYER_SPLIT` 则显式给出每张 GPU 的层数。执行器无法满足的 GPU 数（超过 16 张或超过可见数量）
  会被拒绝，而不是在更少的 GPU 上运行。
- **流水线 prefill。** 按层切分到两张及以上 CUDA GPU 且没有主机专家时，每次提示调用被切成多个微批，
  GPU k 处理一个微批的同时 GPU k+1 处理前一个。微批数按调用选择（八张 A40 上 2,048 token 为 2 个、
  4,096 为 3 个、8,192 为 4 个），大小在 256 行到 prefill ubatch 之间，除非 `TS_MINIMAX_M3_PIPELINE_UBATCH`
  固定它。加载日志会打印 `pipelined prefill: on (...)` 或 `off (reason)`；无法流水线化的调用对每种原因记录一次
  `pipelined prefill declined (reason)`。启用流水线时，服务器中只有一个提示时按 8,192 token 一步 prefill，较短的余数并入前一步，
  除非设置了 `TS_SCHED_SOLO_PREFILL_CHUNK`。
- **逐序列稀疏注意力。** 位置等于缓存行，因此一个块就是槽位缓存的视图，槽位从不平移。批量 decode
  在一张图中为多个槽位各算一个 token，每个槽位有自己的稀疏选择；llama.cpp 只有在逐序列 KV
  （`--no-kv-unified`）下才运行 MSA，而 TensorSharp 的槽位始终是逐序列的。
- **主机专家。** `--n-cpu-moe N` 从 GGUF 映射提供第 0..N-1 层的路由专家，只把这些 `mul_mat_id` 节点放在 CPU 上运行。
  在有两个及以上 NUMA 节点的主机上，它们会被复制到绑核工作线程本地的内存中（`TS_MINIMAX_M3_HOST_NUMA=0`
  保留映射），2-16 个 token 的批次通过使用 ggml 自身点积的静态行内核运行。主机工作线程默认使用 CPU 配额的一半：主机专家的 decode 受内存带宽限制，空出的一半让服务器自己的线程不占用工作线程的核心（用满全部配额时，服务器 decode 只有执行器自身速度的一半）。prompt 较重的场景可用 `--cpu-moe-threads N` 调高。第 0-2 层是稠密层，因此
  `--n-cpu-moe 3` 及以下不会移动任何专家；执行器会给出提示。
- **前缀缓存。** 已完成请求的槽位会被保留，并交给继续该对话的请求，必要时先回退。系统块、默认 developer
  轮与工具声明出现在每个对话开头，因此会被复制一次到检查点槽位，再克隆进每个新对话。前缀复用止于图像跨度的起点。
  无关对话只共享这段公共前缀。
- **内存准入。** 只有当每张 GPU 还能保留 `TS_MINIMAX_M3_SLOT_HEADROOM_MB` 加上迄今最大图所需空间（上限为
  `TS_MINIMAX_M3_ADMIT_GRAPH_MB`）时，新槽位或 KV 增长才会被准入；否则先释放保留的槽位。加载时运行一次满尺寸
  prefill（`TS_MINIMAX_M3_WARM_POOLS`），让 ggml-cuda 的内核暂存池在槽位占用内存之前达到峰值。

### 注意力模式

`TS_MINIMAX_M3_MSA` 选择注意力的运行方式：

| 取值 | 运行内容 |
|---|---|
| `auto`（默认） | 最后一个位置低于 2048 的微批运行稠密 flash attention，在那里它与稀疏结果完全一致。其余微批运行稀疏注意力：CUDA GPU 上用 TensorSharp 的 MSA 内核（每段一个自定义节点完成打分、选择与块稀疏注意力），其他设备上用上游 ggml 算子。`--tp` 下稀疏区间始终使用 ggml 算子 |
| `dense` | 处处稠密注意力。仅用于调试：从位置 2048 起会改变结果，其缓存也从不与稀疏模型共享 |
| `ggml` | 每个微批都通过上游 ggml 算子运行稀疏注意力（llama.cpp 的算法） |
| `kernel` | 每个微批都使用 TensorSharp 的 MSA 内核，通过每张 CUDA GPU 的融合算子包装运行（纯 CPU 加载时使用其 CPU 参考实现）。无法运行它们的 GPU 上会被拒绝；`--tp` 下也会被拒绝，因为其 meta 后端无法容纳该内核节点 |

其他取值会拒绝加载。所有模式都会缓存索引键。

### 张量并行

`--tp N` 在上游 ggml 的 meta 后端（即 llama.cpp `-sm tensor` 背后的机制）上，用前 N 张 CUDA GPU
运行同一执行器；用 `CUDA_VISIBLE_DEVICES` 选择 GPU，且各 GPU 型号必须相同。MiniMax-M3.1 只有四个 KV 头，
其稀疏注意力也无法在 meta 后端上按头切分，因此注意力、索引器以及每个槽位的 K、V 与索引键缓存都**镜像**到每张 GPU 上。
稠密、路由与共享专家 FFN 沿隐藏维按 128 个值的单位切分，输出头按词表行切分；每层以一次 all-reduce 结束。
TensorSharp 会先探测对等拷贝，在声明了 P2P 但实际不可用的主机上让 NCCL 走共享内存。切分方案见
[ggml_ops_minimax_m3_tp.cpp](../../TensorSharp.GGML.Native/ggml_ops_minimax_m3_tp.cpp)（ctest `minimax-m3-tp-split-plan`）。

以下情况会拒绝加载（退出码 2）：`--layer-split` 或 `TS_MINIMAX_M3_LAYER_SPLIT` 与 `--tp` 同时使用；
`--n-cpu-moe`（所有专家都留在显存）；`--draft-model`；`TS_MINIMAX_M3_MSA=kernel`；`TS_MINIMAX_M3_PIPELINE=1`；
`TS_MINIMAX_M3_PROFILE`；`TS_MINIMAX_M3_GRAPH_CACHE` 大于 1；`TS_MINIMAX_M3_CPU_REPACK=1`；本路径不读取的
`TENSORSHARP_TP_DEVICES`；`ggml_cuda` 以外的后端；分布式 `--tp-node-id` / `--tp-peers` 组；以及让某张 GPU 放不下的
`MAX_CONTEXT` 或 `TS_MINIMAX_M3_PLAN_CTX_TOKENS`（消息中会给出能放下的最大上下文）。流水线 prefill 与投机解码关闭，
图缓存只保留一张图，因为 meta 后端只为最后运行的那张图保留各 GPU 视图。

TP decode 在浅深度时最快：在八张 A40 上，深度 0 时 decode 为 35.7 tok/s，深度 2,048 到 16,384 时为 26.0-26.5 tok/s，此时稀疏区间运行在 ggml 算子上；按层切分为 23.4-24.8 tok/s（TP 数字来自合并前张量并行分支的构建）。它在 2K / 8K / 16K token 时的 prefill 为 1,147 / 928 / 762 tok/s，因此长提示请使用按层切分。

## DSpark 投机解码

MiniMax-M3.1-Flash 的 DSpark 草稿器（GGUF 架构 `dflash`，五个滑动窗口层，一个 Markov 头与一个置信度头）读取目标模型
第 3、17、31、45、59 层的输出。用 `--draft-model`（或 `TS_SPEC_DRAFT_MODEL`）指定它就是启用请求：除非给出 `--no-spec`，
投机解码会自动开启。原生执行器把草稿器与目标模型一起加载到持有其借用的输出头的那张 GPU 上（在那里约占 4.4 GiB），
并在每张主干图内部提交草稿器的上下文，因此 prefill 与验证都不需要主机参与，一次草稿就是一次原生调用，提出最多 7 个
token 的块。加载之后再挂接草稿器会被拒绝，通用的托管 DFlash 宿主也不运行它。

默认门限是块草稿器的累积置信度门限：只要置信度头在第 i 个位置之前（含）的估计值之积不低于 0.35，就保留该位置
（`--spec-pmin` / `TS_SPEC_PMIN` 可覆盖；`--spec-pmin 0` 起草每个位置，与 llama.cpp `draft-dspark` 在 `p-min 0` 时相同）。
`--spec-draft` 默认为 8，并受草稿器 7 个 token 的块大小限制。投机解码只对单独（无并发）的请求生效，可用于图像提示，
`--tp` 下会被拒绝。在八张 A40 上，默认门限在代码、散文与段落复制三类生成上的平均 decode 速度为 32.3 tok/s，
不用草稿器为 22.9 tok/s，同样提示下 llama.cpp 的 `draft-dspark` 为 25.8 tok/s（[测量](../benchmarks_zh-cn.md#八张-a40-上的-minimax-m31-flash)）。

验证与单 token decode 的舍入不同，因此在真实模型上，投机与普通贪心文本会在 50-130 个 token 后出现分歧，
每次分歧的位置上普通 decode 的前两个 token 相差都在 0.035 logit 以内；llama.cpp 也是如此。在合成夹具上，
投机贪心输出与普通贪心输出一致（`minimax-m3-dspark-{cpu,cuda,cuda-multi}`）；在两个夹具提示上，TensorSharp
的投机输出与 llama.cpp 的 200 个 token 全部一致，草稿 token 接受率为 72.3% 对 72.2%。

## 聊天格式、推理、工具与 JSON

- **模板。** TensorSharp 用自有渲染器
  （[ChatTemplate.MiniMaxM3.cs](../../TensorSharp.Runtime/ChatTemplate.MiniMaxM3.cs)）渲染 MiniMax-M3.1 的
  `chat_template.jinja`，并与 Hugging Face 的渲染结果逐字节比对
  （`InferenceWeb.Tests/Fixtures/MiniMaxM3/chat_reference.json`），因为模板的递归宏超出了轻量 Jinja 引擎的能力。
  开头的 `root` 消息替换身份行，第一条 system 或 developer 消息作为 developer 轮，连续的工具结果共用一个 tool 轮。
  `current_date`、`current_location` 与 `model_identity` 没有对应的请求字段。
- **推理。** 模板没有思考开关：它告诉模型 "Current thinking mode: adaptive"，因此默认由模型逐轮决定是否打开
  `<mm:think>`。`"think": true`（CLI 为 `--think`）会在生成提示中打开该块。无法关闭推理。推理内容与回答分开返回，
  每个历史助手轮都会重新渲染其推理，因此后续轮次会延续已缓存的前缀。`reasoning_effort`（`low` / `medium` / `high`；
  Responses API 为 `reasoning.effort`）会渲染为系统块中的 `<effort>...</effort>`。
- **工具。** 模型以命名空间 XML（`]<]minimax[>[<tool_call>` ... `<invoke name="f">` ... `</tool_call>`）调用工具，
  [MiniMaxM3OutputParser](../../TensorSharp.Runtime/MiniMaxM3OutputParser.cs) 把它转换为结构化的 `tool_calls`，
  并像 llama.cpp 的 `peg-minimax-m3` 解析器一样按声明的 JSON Schema 推断每个参数的类型。在 `/v1/chat/completions` 上，
  `tool_choice: "required"` 或指定函数由一个在回复的 `</mm:think>` 之后生效的语法强制；`parallel_tool_calls: false`
  只允许一次调用。`auto` 与 `none` 不受约束。必选工具调用不能与 `response_format` 同时使用。
- **JSON 模式。** `response_format` 的 `json_object` / `json_schema` 语法在 `</mm:think>` 之后生效，因此模型仍可先推理；
  回答是不带代码围栏的原始 JSON。
- **分词器。** 使用 `minimax-m2` 预分词器与 NFC 规范化的 BPE，并与 Hugging Face 的分词结果比对
  （`InferenceWeb.Tests/Fixtures/MiniMaxM3/tokenizer_reference.json`）。不添加 BOS：模板自己写出 `]~!b[`，`[e~[` 结束一轮。

## 图像

图像可通过 OpenAI `image_url` data URI、Ollama `images`、Web UI 上传或 CLI `--image` 输入，需要转换好的 `mmproj`。
预处理移植自 Hugging Face 处理器（智能缩放到 28 像素的倍数，每张图 4 到 576 个 token，最多相当于 672×672 像素）。
32 个 ViT 块通过共享的 Qwen-VL 编码器图在 GPU 0 上作为一张设备图运行，投影器与 2×2 patch 合并是第二张图；
图像行在原生执行器内按绝对位置替换图像占位符，因此图像跨度在 prefill 分块后依然完整。若 GPU 0 内存不足，可用
`TS_MINIMAX_M3_DEV0_RESERVE_MB` 预留空间（实测的 `--tp 8` 图像运行使用 2560 与 8,192 token 上下文）。有待处理的图像行时，
批量 decode 与批量 prefill 会让行，前缀复用止于图像跨度。

没有视频编码器。`video_url` 部分会被抽成带时间的帧（每个部分的 `fps` 与 `max_frames`，或 `VIDEO_SAMPLE_FPS` /
`VIDEO_MAX_FRAMES`），每帧带时间标记走图像路径；视频输入没有在真实模型上测量。拒绝音频：该发布没有音频塔。

编码器在真实 `mmproj` 上与 Hugging Face transformers 的 MiniMax-M3 视觉塔和投影器比对（`MiniMaxM3VisionOracleTests`，
需设置 `TS_TEST_MINIMAX_M3_MMPROJ` 与 `TS_TEST_MINIMAX_M3_VISION_ORACLE`：平均余弦至少 0.99999，相对 L2 至多 0.005）。

## 选项

执行器的设置都是环境变量；下表列出大多数部署会用到的。所有 `TS_MINIMAX_M3_*` 变量及其默认值的完整列表见
[环境变量矩阵](../env_var_feature_matrix_zh-cn.md#矩阵外的-minimax-m31-开关)。

| 设置 | 默认值 | 作用 |
|---|---|---|
| `TS_MINIMAX_M3_MSA` | `auto` | 注意力路径：`auto`、`dense`、`ggml` 或 `kernel`（[模式](#注意力模式)） |
| `TS_MINIMAX_M3_PLAN_CTX_TOKENS` | 上下文长度 | 放置规划为所有槽位预留的 K/V token 数 |
| `TS_MINIMAX_M3_CPU_MOE=auto` | 关闭 | 只把放不下的专家栈留在主机；覆盖 `--n-cpu-moe`（仅按层切分） |
| `TS_MINIMAX_M3_LAYER_SPLIT` | 自动 | 每张 GPU 的层数，逗号分隔，总和为 60；与 `--tp` 同用会被拒绝 |
| `TS_MINIMAX_M3_UBATCH` | `2048` | prefill 微批行数 |
| `TS_MINIMAX_M3_DEV0_RESERVE_MB` | `0` | 在 GPU 0 上额外保留的内存，例如留给视觉编码器 |
| `TS_MINIMAX_M3_PIPELINE` | 自动 | 按层切分时的流水线 prefill：`1` 强制开启，`0` 关闭 |
| `TS_MINIMAX_M3_INDEXER_F16` | 有 GPU 时开启 | 为 decode 形状的图使用 F32 索引器投影的 F16 副本 |

## 实测结果

2026-10-24 在一台有八张 A40 46 GB GPU 的主机上测量，对照 unslothai `minimax-m31`（`ef8f94cc`）的 llama.cpp，
测量时主机无其他负载，各重复三次取中位数。比值大于 1.00× 表示 TensorSharp 更好。

| 工作负载 | TensorSharp | llama.cpp | 比值 |
|---|---:|---:|---:|
| prefill，4K token 提示，服务器，按层切分（tok/s） | 1,263 | 283 | **4.46×** |
| prefill，32K token 提示，服务器，按层切分（tok/s） | 1,544 | 240 | **6.43×** |
| 4K 深度的 decode，服务器，按层切分（tok/s） | 23.9 | 19.2 | **1.24×** |
| 深度 0 的 decode，执行器基准对 `llama-bench`（tok/s） | 24.8 | 22.5 | **1.10×** |
| 首 token 时间，短对话（s） | 0.41 | 0.39 | 0.95× |
| 八个并发请求，聚合输出（tok/s） | 53.7 | 40.6 | **1.32×** |
| DSpark 投机 decode，默认门限（tok/s） | 32.3 | 25.8 | **1.25×** |
| 深度 0 的 decode，`--tp 8`（tok/s） | 35.7 | 该模型没有张量切分 | |

所有单元格、方法与限制见[基准测试页](../benchmarks_zh-cn.md#八张-a40-上的-minimax-m31-flash)。完整对比报告：
[MiniMax-M3.1-Flash：TensorSharp 与 llama.cpp 对比](../perf/minimax-m3-vs-llama-cpp.md)（英文）。

## 验证

- **单元与夹具测试。** `InferenceWeb.Tests/MiniMaxM3*.cs` 覆盖协议、分词器、输出解析器、工具语法、图像预处理与提示接线、
  加载拒绝，以及在由 `MiniMaxM3SyntheticModelBuilder` 与 `eng/minimax-m3-fixture.py` 写出的合成 GGUF 上运行模型。
  受门控的测试需要 `TS_TEST_MINIMAX_M3_FIXTURE`、`TS_TEST_MINIMAX_M3_GGUF`、`TS_TEST_MINIMAX_M3_TP`、
  `TS_TEST_MINIMAX_M3_MMPROJ` 或 `TS_TEST_MINIMAX_M3_VISION_ORACLE`，并配合 `TS_TEST_GGML_BACKEND`；被跳过的门控测试不算通过。
- **原生测试。** `ctest -R "minimax-m3|minimax-msa"` 运行执行器测试（`minimax-m3-exec-*`，覆盖 CPU、单张与多张 CUDA GPU、主机专家与 `--tp`：
  一个序列的 logits 与其输入方式和放置方式无关）、稀疏注意力图对照参考数学（`minimax-m3-msa-graph-*`）、
  MSA 内核（`minimax-msa-kernels-cuda`）、TP 切分方案（`minimax-m3-tp-split-plan`）以及 DSpark
  （`minimax-m3-dspark-*`）。
- **与 llama.cpp 的一致性。** [eng/validation/minimax-m3-parity-suite.sh](../../eng/validation/minimax-m3-parity-suite.sh)
  在合成的 `small` 与 `real-block` 夹具（随机权重；`real-block` 使用发布版的 MSA 几何）上比较 TensorSharp 与 llama.cpp 的 logits。
  在 `real-block` 上，TensorSharp 相对 llama.cpp 的平均 KL 散度为 0.00127，低于 llama.cpp 自身在微批 512 与 1 之间的下限 0.00139；
  `--tp 2` / `--tp 4` 分别为 0.00123 / 0.00126。这是夹具级的一致性，不是完整检查点的质量测量。
- **真实模型。** [eng/minimax-m3-bench.py](../../eng/minimax-m3-bench.py) 的功能场景（可核对答案的问题、最长 64K token 的
  大海捞针提示、JSON 对象与 schema、单个与并行工具调用、智能体循环、三轮 KV 复用、图像、并发请求）在按层切分下全部通过，
  文本与图像场景在 `--tp 8` 下也通过。CPU 专家卸载与视频没有在真实模型上测量。原始记录是测量主机上的本地验证证据，未提交
  （`/workspace/b1/`、`/workspace/b2/`、`/workspace/b3/real/`）。

## 当前限制

- 只有 `ggml_cuda` 用于服务；`ggml_cpu` 是正确性路径，其他后端都会被拒绝。
- `--tp` 镜像注意力与缓存，因此每张 GPU 能容纳的上下文更少（八张 A40 上为 45,824 token），深度超过 2K 后 decode 优势缩小，
  长提示 prefill 也慢于按层切分；它既不运行投机解码，也不支持主机专家。
- DSpark 只在按层切分下、对单独请求运行。
- 关闭思考的提示（`</mm:think>` 前缀）没有请求字段。
- 真实模型的质量通过功能场景与夹具级 logit 一致性检查，而不是完整检查点的困惑度或 KL 比较。
