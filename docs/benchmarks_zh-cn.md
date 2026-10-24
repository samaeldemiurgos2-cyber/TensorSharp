# 性能数据

[English](benchmarks.md) | [中文](benchmarks_zh-cn.md)

> 2026-10-23 核对的最新已记录对比；本次更新没有进行新一轮基准。[README](../README_zh-cn.md#性能数据)摘录了这些测试单元。

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

## 阅读与复现

比值大于 1.0× 表示该单元的吞吐更高。不能把不同模型、放置方式、提示长度、近似设置或并发级别合并为通用加速比。本概览已移除旧笔记本、早期检查点及内部后端的核心表格。

可在自己的匹配模型与后端上使用[引擎对比框架](../benchmarks/engine_comparison)或模型专用工具，记录准确构建、权重哈希、提示与输出长度、上下文、并发、缓存策略、预热和重复次数。生成报告应保存在忽略的验证目录中。没有执行的场景不算通过。
