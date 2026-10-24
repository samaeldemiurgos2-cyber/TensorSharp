# 性能数据

[English](benchmarks.md) | [中文](benchmarks_zh-cn.md)

> 2026-10-23 核对的最新已记录对比。DeepSeek 与 Muse Spark 部分没有进行新一轮基准；[MiniMax-M3.1-Flash 部分](#八张-a40-上的-minimax-m31-flash)来自 2026-10-24 的测量。[README](../README_zh-cn.md#性能数据)摘录了 DeepSeek 与 Muse Spark 的测试单元。

## DeepSeek V4.1 Flash 1017 — 五张 A40

**测量日期：2026-10-22/23 UTC。** Unsloth UD-IQ2_XXS，六个文本分片，Hugging Face 版本 `60ad3651641b5c97679c807e064ab1d7a3ca1455`；五张 46 GB A40，CUDA 12.8 / 驱动 595.91。主机映射的 Q5_K Engram 表已预热。两个服务均使用 32,768 token 上下文、贪心采样、关闭思考，单请求生成 256 token。使用相同的 wikitext-2 摘要提示集，约 1.8K 和 7.3K token，每次有唯一首行。下表均为一次预热后三次请求的中位数。

TensorSharp 较快的放置方式为路由 MoE `--tp 5`。参照为 Unsloth 的 llama.cpp `be8365722ea432809818fa21a4ecebfac2e7d298`，使用层切分；该检查点在其 row/tensor split 上无法运行。这里是**服务内部 prefill 计时**，不是流式 TTFT 或客户端墙钟耗时。

| 提示长度 | TensorSharp 精确 TP5 tok/s | 精确比值 | TensorSharp 默认**近似 YOCO** TP5 tok/s | 近似比值 | llama.cpp tok/s |
|---|---:|---:|---:|---:|---:|
| 约 1.8K token | 563.9 | **3.23×** | 960.5 | **5.50×** | 174.6 |
| 约 7.3K token | 549.7 | **2.44×** | 1,000.7 | **4.43×** | 225.7 |

**近似计算与质量。** 精确 prefill 设置 `TS_DSV41_YOCO_KEEP=0`；默认值使用有界 YOCO 重放。在十个 teacher-forced wikitext-2 块上（4,608 token 窗口、2,304 token prefill），TP 困惑度为：**精确 2.4811**、**默认 2.4978**、**参照 2.4697**。在此样本上，默认值比精确值约高 0.67%。相对参照的平均 KL 散度分别为 0.01605 和 0.017208。这不能证明数值等价或通用质量保证。**5.50×** 仅属于默认近似计算的约 1.8K prefill 单元，并非 decode 加速比。

**来源与局限。** 实测 TensorSharp 是基于 `ae60abf6` 的工作树加当轮修改，原生 SHA-256 前缀为 `04bc9dd748378492`；修改后来合入 `7f26e2d1`。最终评审原生构建（`8b910b29e88d6891…`）只复核了 2K 合成单元，并未重跑整个服务端矩阵。未修改的 ggml 为 `da8e2a54b919557b816b1c2beb432ccb766f7ce0`。参照服务记录来自较早会话，含一次原因未明的慢 prefill；每单元三个样本不足以给出置信区间。两个引擎通过不同协议路径报告内部计时。TensorSharp 的合成测试三次最佳值与 llama-bench 的均值不列入此核心表格。

[模型卡片](models/deepseek41_zh-cn.md#unsloth-的-deepseek-v41-flash-1017-gguf)提供分片哈希、启动配置、decode、DSpark 与并发结果。原始记录位于未提交的本地 `docs/validation/deepseek41-1017/`；此次文档更新从可用 JSONL 独立重算了服务端中位数，但没有重新计算完整检查点的哈希。数据不能作为另一套修复版 Q2_K/Q5_K 包的验证结果。

## 十块 A40 上的 Muse Spark 816B

**报告更新于 2026-10-22；未记录确切测量日期。** Muse-Spark-Open-816B-A42B，UD-IQ3_XXS，八分片 / 315.7 GiB 权重，十张 A40 46 GB，所有专家均驻留显存。两个服务均使用 F16 KV、flash attention、32,768 token 上下文、贪心采样并关闭思考。长提示 prefill 请求的输出上限为 32 token；短提示 decode 与并发请求为 256 token。Decode 使用 167 token 短提示。Muse prefill 按未缓存提示 token 数除以流式 TTFT 计算，与上面的 DeepSeek 内部 prefill 计时不同。除标明四请求的单元外，并发为一。

按工作负载选择放置方式：本次对比中，层切分的绝对 prefill 吞吐最高，张量并行的 decode 与四请求聚合吞吐最高。下表每个比值都比较两个引擎的对应模式。

| 工作负载 / 放置方式 | TensorSharp tok/s | llama.cpp tok/s | 比值 |
|---|---:|---:|---:|
| prefill，4,075 token，层切分 10 | 999.7 | 483.2 | **2.07×** |
| prefill，15,767 token，层切分 10 | 1,346.8 | 826.1 | **1.63×** |
| decode，单请求，TP10 / 张量切分 | 22.63 | 16.19 | **1.40×** |
| 四请求聚合输出，TP10 / 张量切分 | 45.73 | 24.49 | **1.87×** |
| decode，单请求，层切分 10 | 16.69 | 16.79 | 0.99× |

吞吐与延迟为三次重复的中位数，llama.cpp 张量切分行为两次；校验用例各运行一次。llama.cpp 为 `7170cad`，使用其自带的 ggml；TensorSharp 使用未修改的上游 ggml `da8e2a5`；TensorSharp 报告记录的是基于 `e2a84893` 的工作树，含 `192295a9` 与后续修改，而非冻结的完整基准提交。此报告不能证明当前发布二进制的性能。

**传输与覆盖限制。** 驱动声明 P2P 可用但实际不工作，且没有 NVLink。llama.cpp 的 NCCL 构建在十张 GPU 上中止，因此张量切分**未启用 NCCL**；TensorSharp 使用经共享内存的 NCCL。TensorSharp TP 的大规模 prefill 归约使用 BF16，这些条件会影响 TP 性能与质量。单主机结果没有置信区间。未测量图像、CPU 专家卸载、投机解码、其他 GPU 数量或后端，以及超过 16K token 的实际提示。四请求层切分的逐请求 decode 较慢（8.54 对 9.37 tok/s），尽管聚合吞吐略高。

完整的两种模式表格、teacher-forced 质量检查与启动命令见[完整对比](perf/muse-spark-vs-llama-cpp.md)。当前十卡 A40 原始记录在本机不可用，因此本次只核对了已提交报告的抄录和算术，没有重新审核原始运行日志。

## 八张 A40 上的 MiniMax-M3.1-Flash

**2026-10-24 测量。** Unsloth MiniMax-M3.1-Flash UD-Q4_K_XL（七个分片，232.3 GiB），一台有八张 A40 46 GB GPU 的主机。TensorSharp 使用未修改的上游 ggml `da8e2a5`；对照为 unslothai `minimax-m31` 分支（`ef8f94cc`，PR #320）的 llama.cpp。两个引擎在同一台机器上轮流运行，运行时独占 GPU，主机无其他负载。除特别说明外，各单元为三次重复的中位数。比值大于 1.0× 表示 TensorSharp 更好。

**执行器吞吐。** TensorSharp 的执行器基准（[eng/validation/minimax-m3-native-bench.cpp](../eng/validation/minimax-m3-native-bench.cpp)：一次加载、均匀随机 token、每个单元一次不计时的预热加若干次计时重复、深度 prefill 不计时、调用与托管模型相同）对照 `llama-bench -fa 1`，两者都在八张 GPU 上按层切分。

| 单元 | TensorSharp tok/s | llama.cpp tok/s | 比值 |
|---|---:|---:|---:|
| prefill，512 token | 448 | 344 | **1.30×** |
| prefill，2,048 token | 948 | 784 | **1.21×** |
| prefill，8,192 token | 1,699 | 1,312 | **1.29×** |
| prefill，32,768 token | 2,449 | 681 | **3.60×** |
| prefill，65,536 token | 2,567 | 262 | **9.80×** |
| decode，深度 0 处 128 token | 24.8 | 22.5 | **1.10×** |
| decode，深度 2,048 处 128 token | 23.9 | 19.6 | **1.22×** |
| decode，深度 8,192 处 128 token | 23.8 | 19.1 | **1.25×** |
| decode，深度 32,768 处 128 token | 23.4 | 17.5 | **1.34×** |

llama.cpp 的每个 prefill 单元取 `-ub 512` 与 `-ub 2048` 中较快者。32K 时 `-ub 2048` 显存不足；64K 单元是 `-ub 2048` 的单次运行（`-ub 512` 为 181 tok/s）。位置 2,048 之后 MiniMax 稀疏注意力进入稀疏区间，TensorSharp 用自己的 CUDA 内核运行该区间。

**服务器场景。** 用 [eng/minimax-m3-bench.py](../eng/minimax-m3-bench.py) 测试 TensorSharp.Server.Host 与 llama-server，两者分别用 [eng/minimax-m3-ts-server.sh](../eng/minimax-m3-ts-server.sh) 与 [eng/minimax-m3-llama-server.sh](../eng/minimax-m3-llama-server.sh) 的匹配设置启动：八张 GPU 按层切分、单序列（llama-server 用 `-np 1 --no-kv-unified`，否则它以稠密方式运行 MSA）、flash attention、贪心流式请求，思考保持模板默认的自适应模式。长提示是基于真实文本的大海捞针问题，答案会被核对。每个对话以新的随机标记开头，因此两个引擎都无法复用之前运行的提示。引擎按 ABBA 轮换（llama.cpp、TensorSharp、TensorSharp、llama.cpp）；两次启动之间未清除页缓存，因此不比较加载时间。prefill 按未缓存提示 token 数除以首 token 时间计算。

| 提示 | prefill，TensorSharp | prefill，llama.cpp | 比值 | decode，TensorSharp | decode，llama.cpp | 比值 |
|---|---:|---:|---:|---:|---:|---:|
| 4K token | 1,263 | 283 | **4.46×** | 23.9 | 19.2 | **1.24×** |
| 16K token | 1,478 | 273 | **5.41×** | 23.7 | 18.5 | **1.28×** |
| 32K token | 1,544 | 240 | **6.43×** | 23.5 | 17.7 | **1.33×** |
| 64K token | 1,545 | 211 | **7.32×** | 23.0 | 16.1 | **1.43×** |

短对话的首 token 时间为 0.41 s，llama.cpp 为 0.39 s（0.95×）：TensorSharp 在这里略慢。四个与八个不同的并发请求下，聚合输出分别为 42.1 对 31.8 tok/s（1.32×）与 53.7 对 40.6（1.32×）；该并发下的首 token 时间为 1.62 对 1.56 s 与 2.39 对 2.29 s，同样略慢。这些场景中 TensorSharp 的答案通过 12/12 与 24/24 项检查，llama.cpp 为 10/12 与 21/24。

**张量并行。** llama.cpp 对该模型没有张量切分，因此 `--tp 8` 没有对照。TensorSharp 的 TP 基准（[eng/validation/minimax-m3-tp-bench.cpp](../eng/validation/minimax-m3-tp-bench.cpp)，ubatch 2048，三轮，使用合并前张量并行分支的构建）在深度 0 时 decode 35.7 tok/s，深度 2,048 / 8,192 / 16,384 时为 26.5 / 26.3 / 26.0 tok/s；2K / 8K / 16K 的 prefill 为 1,147 / 928 / 762 tok/s。未设 `MAX_CONTEXT` 时上下文被限制为 45,824 token（每张 GPU 使用 38.1 GiB），因为注意力与缓存在每张 GPU 上镜像。`--tp 8` 下服务器的文本与图像场景都通过（各一次）。

**DSpark 投机解码。** 基准的 `speculative` 场景（代码、散文与段落复制三类生成，各三次），按层切分，使用转换得到的 `MiniMax-M3.1-Flash-DSpark-BF16.gguf`。TensorSharp 使用默认的累积置信度门限（0.35）；llama-server 使用 `--spec-type draft-dspark`，最多 7 个草稿 token，`p-min 0`。

| 生成 | DSpark，TensorSharp | DSpark，llama.cpp | 比值 | 普通，TensorSharp | 普通，llama.cpp |
|---|---:|---:|---:|---:|---:|
| 平均（tok/s） | 32.3 | 25.8 | **1.25×** | 22.9 | 20.5 |
| 代码（中位数） | 35.3 | 26.7 | **1.32×** | 23.9 | 21.1 |
| 段落复制（中位数） | 35.0 | 32.0 | **1.10×** | 20.8 | 19.3 |
| 散文（中位数） | 26.6 | 18.7 | **1.42×** | 24.0 | 21.1 |

差异主要来自门限：使用 llama.cpp 的设置（`--spec-draft 7 --spec-pmin 0`，并关闭成本调节器 `TS_SPEC_ADAPTIVE=0`）时，TensorSharp 平均为 26.6 tok/s；没有门限时，两个引擎在散文上都比普通 decode 更慢。在真实模型上，两个引擎的投机与普通贪心文本都会在 50-130 个 token 后出现分歧，每次都发生在普通 decode 前两个 token 几乎持平的位置；见[模型卡片](models/minimax-m3_zh-cn.md#dspark-投机解码)。

**来源与限制。** TensorSharp 的数字来自 2026-10-24 MiniMax-M3.1 分支未提交的工作树构建，而非冻结的提交；张量并行与 DSpark 的数字来自这两项功能合并前的构建。只有一台主机与一个检查点，没有置信区间。执行器基准不含 HTTP，服务器场景包含。未测量视频输入、其他 GPU 数量与后端，以及完整检查点的困惑度或 KL 散度；MoE CPU 卸载每个引擎测量了一次（两次重复），见完整报告。与 llama.cpp 的 logit 一致性只在合成夹具上测量（[模型卡片](models/minimax-m3_zh-cn.md#验证)）。原始记录是测量主机上的本地验证证据，未提交：`/workspace/b1/`（执行器基准、llama-bench、服务器与并发场景）、`/workspace/b2/`（张量并行）与 `/workspace/b3/real/`（DSpark）。完整对比报告（含 MoE CPU 卸载）：[perf/minimax-m3-vs-llama-cpp.md](perf/minimax-m3-vs-llama-cpp.md)（英文）。

## 阅读与复现

比值大于 1.0× 表示该单元的吞吐更高。不能把不同模型、放置方式、提示长度、近似设置或并发级别合并为通用加速比。本概览已移除旧笔记本、早期检查点及内部后端的核心表格。

可在自己的匹配模型与后端上使用[引擎对比框架](../benchmarks/engine_comparison)或模型专用工具，记录准确构建、权重哈希、提示与输出长度、上下文、并发、缓存策略、预热和重复次数。生成报告应保存在忽略的验证目录中。没有执行的场景不算通过。
