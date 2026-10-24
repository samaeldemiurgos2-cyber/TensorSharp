# Muse Spark

[← 返回模型索引](README_zh-cn.md) | [English](muse-spark.md)

TensorSharp 支持 [Unsloth 的 Muse-Spark-Open-816B-A42B 发布版](https://huggingface.co/unsloth/Muse-Spark-Open-816B-A42B-GGUF)所使用的 `muse-spark` GGUF 架构。
解码器、MoE 集成与多模态接线都在 TensorSharp 自有代码中实现；拉取的 ggml 依赖保持不变。

| 属性 | 实现 |
|---|---|
| 架构 | `muse-spark`，别名 `muse_spark` |
| 解码器 | [MuseSparkModel](../../TensorSharp.Models/Models/MuseSpark/MuseSparkModel.cs)；在 `ggml_cuda`/`ggml_vulkan` 上默认由[原生整模型执行器](#原生整模型执行器)运行，`ggml_cuda` 上的 `--tp N` 也是如此 |
| 专家路径 | 原生执行器：在整模型图内每层一个 `mul_mat_id` 图。逐算子路径：[MuseSparkModel.Moe.cs](../../TensorSharp.Models/Models/MuseSpark/MuseSparkModel.Moe.cs)，原生 `TSGgml_MuseSparkMoEFFNQuantF32` |
| 图像 | 匹配的 Spark mmproj，共用 Muse-Glimmer 视觉编码器 |
| 音频 | 未实现；投影器中带有音频塔并不会启用音频推理 |
| GPU 放置 | 原生执行器，最多 16 张 GPU：`--layer-split N` 把各层分配到 N 张 GPU，按字节最小化方案放置专家栈，并在各 GPU 之间[流水线化 prefill](#运行方式)；`ggml_cuda` 上的 `--tp N` 通过 ggml 的 meta 后端（llama.cpp `-sm tensor` 背后的机制）把每一层切分到 N 张 GPU（[详情](#原生执行器上的张量并行)）。逐算子路径（`TS_MUSE_SPARK_NATIVE=0`，以及 `ggml_vulkan` 上的 `--tp N`）：`--tp N` 切分注意力并划分专家；`--layer-split N` 分配完整的层 |
| 专家卸载 | 原生执行器（按层切分；原生 `--tp` 把所有专家留在显存中）：`TS_MUSE_SPARK_CPU_MOE=auto` 把显存放不下的专家栈留在按 NUMA 放置的主机内存中；`--n-cpu-moe N` 强制把前 N 层的专家放到主机。逐算子路径：`--n-cpu-moe N` 在 CPU 上运行前 N 层保留的原生专家栈 |
| KV 缓存 | 原生执行器：F16 逐序列槽位，滑动层使用环，全局层的行惰性增长。逐算子路径：F32/F16 线性 GQA 缓存；块量化 KV 会被拒绝 |
| 连续批处理 | 原生执行器：跨序列槽位的批量 decode（文本与图像请求）以及新文本提示的批量 prefill。逐算子路径：使用 F32 KV 的打包分页文本前向 |
| 前缀缓存 | 原生执行器：对话续写、已完成对话的回退，以及克隆到新对话中的共享系统提示检查点 |
| 投机解码 | 通过 `--spec --spec-type ngram` 使用通用 n-gram 草稿器；不假定存在学习型 Muse Spark 草稿模型。原生 `--tp` 下不使用（[原因](#原生执行器上的张量并行)） |

## 原生整模型执行器

在 `ggml_cuda` 与 `ggml_vulkan` 上，Muse Spark 通过 TensorSharp 自有的原生执行器
[ggml_ops_muse_spark_exec.cpp](../../TensorSharp.GGML.Native/ggml_ops_muse_spark_exec.cpp)
运行，托管侧胶水代码位于
[MuseSparkModel.Native.cs](../../TensorSharp.Models/Models/MuseSpark/MuseSparkModel.Native.cs)。
它自行加载分片 GGUF，并为每个 micro-batch 提交一张横跨所有 GPU 与 CPU 的 GGML 图。它使用未修改的上游 ggml。
后文各节描述的逐算子路径仍是参考实现：`TS_MUSE_SPARK_NATIVE=0` 选择它，
`TS_MUSE_SPARK_NATIVE=1` 让 GGML CPU 后端也启用原生执行器。`--layer-split N` 以及 `ggml_cuda` 上的 `--tp N`
都运行在这个执行器上，见[原生执行器上的张量并行](#原生执行器上的张量并行)；在 `ggml_vulkan` 上，
`--tp N` 走[逐算子路径](#逐算子路径上的张量并行)。

### 运行方式

- **每个 micro-batch 一张图。** `ggml_backend_sched` 把每张图拆分到各 GPU 与 CPU 后端。
  图按形状缓存，因此 decode 步会复用它的图。prefill 使用 2048 token 的 micro-batch
  （`TS_MUSE_SPARK_UBATCH`）。行数从 512 增加到 2048 时，每 token 每层的主机专家开销从 0.305 ms 降到 0.175 ms。
  图放不下时，执行器把 micro-batch 减半后重试。
- **权重放置。** `--layer-split N` 把 62 层分布到 N 张 GPU。
  每层的稠密权重（注意力、路由器、潜空间投影、norm）留在该层所在的 GPU。
  随后由一个动态规划分别决定每层 gate、up、down 三个专家栈的位置：在 GPU 上放下尽可能多的专家栈，
  并让留在主机上的字节尽可能少。主机上的专家字节数决定 decode 开销。
  `TS_MUSE_SPARK_CPU_MOE=auto` 启用这一方案；`--n-cpu-moe N` 则强制把前 N 层的专家放到主机。
  两者都不设置时，放不下的模型会被拒绝，提示信息会指明该选项。
  在留在主机上的专家字节数最少的方案中，选出把各 GPU 填得最均匀的那一个。
  启动日志会打印每张 GPU 上的层以及主机上的专家栈（`6,7,...,18:gu,...,45:d`；`g`、`u`、`d` 各指一个专家栈）。
- **主机专家。** 有两个或更多 NUMA 节点时，主机专家栈由绑定到各节点的线程拷贝进匿名内存，
  并由一个严格绑核的 CPU 线程池运行。ggml 的 CPU `mul_mat_id` 为每个工作线程分配固定的行区间，
  因此每个工作线程都读取自己节点上的内存。在双路测试主机上，这让主机专家带宽从约 100 GB/s 提高到 165–174 GB/s。
  `numactl --interleave` 无法使用，因为容器禁止 `set_mempolicy`。
  `TS_MUSE_SPARK_HOST_NUMA=0` 则保留 GGUF 内存映射。
  当批量 decode 步中有两个 token 选中同一个专家时，ggml 的内核会把该专家切换为动态领取的分块，
  工作线程于是会读到另一路 CPU 上的行。因此 2–16 token 的批次改由 TensorSharp 内核（`host_moe_matmul`）运行主机专家。
  它保持静态的行归属，每个权重行只读一次、供所有选中它的 token 使用，并使用 ggml 自己的量化与点积，
  因此结果逐位相同。4 个 token 的一层耗时 3.5–3.8 ms，原先为 3.8–3.9 ms。
- **KV 缓存。** 每个序列槽位在每层所在的 GPU 上拥有 F16 K/V 行。
  46 个滑动窗口层保留一个环，容量为 2048 token 窗口加上 micro-batch，再加 512 行的回退余量；
  每行都带有自己的绝对位置。16 个全局层起步为 4096 行，按需加倍。
  环已无法满足的回退会被拒绝，调度器改为重新 prefill。
- **批处理。** 所有运行中请求的 decode 步共享一张图，使用逐序列的注意力掩码。
  同时到达的新文本提示在一张图中完成 prefill。图像 embedding 在图内替换各自的占位行。
- **流水线 prefill。** 按层切分时，一次提示调用被切成等长的 micro-batch，GPU k 处理 micro-batch j 的同时，
  GPU k+1 处理 j−1，与 llama.cpp 的流水线并行相同。调用运行在一个常驻的 ggml 调度器上，
  使用四份轮换的输入副本、跨 GPU 事件和一个锁页暂存环。最后一个 micro-batch 用最后一个 token 的副本补齐；
  补齐的行会被重写或清除标签，因此不会被读取。除非 `TS_MUSE_SPARK_PIPELINE_UBATCH` 固定了大小，
  每次调用的 micro-batch 约为调用长度的四分之一，介于 512 行与 prefill ubatch（`TS_MUSE_SPARK_UBATCH`，2048）之间；
  长上下文的掩码超出 96 MiB 暂存预算时会缩小（最小 128 行）。在十张 A40 上，一次 15,641 token 的调用
  使用固定 512 行 micro-batch 耗时 15.3 s，自适应时为 10.2 s；一次 4K 调用分别为 4.5 s 和 3.7 s。
  在有两张或更多 CUDA 家族（CUDA、ROCm、MUSA）GPU、没有主机专家、未设置 `TS_MUSE_SPARK_PROFILE` 且不是张量并行时，
  流水线自动开启（`TS_MUSE_SPARK_PIPELINE`）；加载日志会显示 `[muse-spark] pipelined prefill: on (...)` 或 `off (原因)`。
  无法流水线化的调用会按原因各记录一次 `pipelined prefill declined (原因)`，并逐个 micro-batch 运行：
  即使最小的 micro-batch 掩码也超出预算、计算缓冲区放不下（在某个槽位释放内存之前，不再对该形状重试）、
  没有设备内存容纳该调用的 K/V 行、没有锁页主机内存，或调用中含有图像行（图像 embedding 在图内替换 token 行）。
  流水线调度器的缓冲区在内存准入中计为可回收，
  并会在逐出保留槽位之前先被释放。
- **单独提示的 prefill 步长。** 服务器中只有一个请求时，调度器会询问模型每个 prefill 步应有多大
  （`ISoloPrefillBudgetModel`），答案可以超过 `TS_SCHED_MAX_BATCHED_TOKENS`。流水线 prefill 开启时，
  Muse Spark 回答 8,192 个 token，并把不超过 2,048 的余数并入前一步，使任何短调用都不会单独填满并排空流水线；
  调用无法流水线化时它回答 0，即沿用调度器自己的步长；会带有图像行的步也沿用调度器自己的步长。
  在十张 A40 上，一次 8K 调用保持 16K 调用约 90% 的吞吐，
  单独步进行期间到达的请求要等这一步完成（8K 时约 6 s）。设置 `TS_SCHED_SOLO_PREFILL_CHUNK`
  会改用调度器的步长（受 `TS_SCHED_MAX_BATCHED_TOKENS` 限制）并关闭模型自选的步长；
  只设置 `TS_SCHED_MAX_BATCHED_TOKENS` 不会。引擎日志会说明采用的是哪一种。
- **前缀缓存。** 已完成请求的槽位会被保留，并交给续写该对话的请求。
  它可以先被回退，因此重新生成的回答也能复用提示。
  聊天层共享前缀边界之前的 token（系统提示与工具声明）会被拷贝一次到检查点槽位，再克隆进每个新对话。
  Muse Spark 的模板用相同的系统头（知识截止时间、日期、推理等级）开启每个对话，
  因此即使没有系统消息，这个头部也是共享前缀。内存紧张时，先释放保留的结束状态，再释放检查点。
  与 TensorSharp 的所有模型一样，无关的对话只共享这段公开前缀，
  因此内容相同但彼此独立的请求不会复用另一个请求的私有 token。
- **内存准入。** 只有当每个设备都能保留 `TS_MUSE_SPARK_SLOT_HEADROOM_MB`，
  外加迄今最大的图所需的空间（上限为 `TS_MUSE_SPARK_ADMIT_GRAPH_MB`）时，新槽位或 KV 增长才会被准入。
  否则先释放保留的槽位；已缓存的图计为可回收。只要物理内存确实存在，运行中序列的 KV 增长永远不会被拒绝。
  没有这个上限时，单次 16K token prefill 的 1.2 GiB 图曾让之后的并发请求逐出共享提示检查点。
  放不下的图会先释放保留的槽位，然后把那次调用拆成减半的 micro-batch 运行。
  ggml-cuda 的内核 scratch 池不在这套记账之内：它在首次使用时增长、从不收缩，无法增长时会中止进程。
  因此加载时会运行一次满尺寸 prefill（`TS_MUSE_SPARK_WARM_POOLS`），让该池先达到峰值；
  没有这一步时，一连串并发请求之后的第一个 4K token 提示曾让服务器崩溃。

### 原生执行器上的张量并行

`ggml_cuda` 上的 `--tp N` 运行原生执行器，把每一层切分到前 N 张 CUDA GPU 上；用 `CUDA_VISIBLE_DEVICES` 选择 GPU。
它基于上游 ggml 的 meta 后端（`ggml_backend_meta_device`），即 llama.cpp `-sm tensor` 背后的机制，不打 ggml 补丁。
TensorSharp 在 [ggml_ops_muse_spark_tp.cpp](../../TensorSharp.GGML.Native/ggml_ops_muse_spark_tp.cpp)
中提供切分方案（ctest `muse-spark-tp-split-plan`）。`TS_MUSE_SPARK_NATIVE=0` 则保留[逐算子路径](#逐算子路径上的张量并行)，
`ggml_vulkan` 上的 `--tp N` 也走这条路径。

- **切分方式。** 注意力按 KV 组切分：每张 GPU 持有其各组的 q 与 gate 行、k 与 v 行、K/V 缓存列，
  以及注意力输出投影中对应的输入列。路由专家沿隐藏维度切分（gate 与 up 的行、down 的列），
  单位是 down 投影的量化块向上取整到 128 元素的倍数，UD-IQ3_XXS 检查点为 256 元素；输出头按词表行切分。norm、路由器和潜空间投影在每张 GPU 上各有一份副本。
  各份额沿用 llama.cpp 的分配方式并逐层轮换，因此 16 个 KV 组分到十张 GPU 上，每张 GPU 得到一到两个。
  每层以两次 all-reduce 结束。
- **集合通信。** 在后端创建通信器之前，TensorSharp 会探测 GPU 之间的对等拷贝。
  如果对等访问被宣称可用但实际不工作（十张 A40 的测试主机即是如此），NCCL 以 `NCCL_P2P_DISABLE=1` 走共享内存传输。
  显式设置的 `GGML_CUDA_ALLREDUCE` 优先；没有 NCCL 时，两张 GPU 使用 ggml 自带的 all-reduce，
  更多 GPU 运行 meta 后端的对等拷贝 butterfly，与不带 NCCL 构建的 llama.cpp 相同。ggml 的 NCCL 路径以 BF16 归约超过 262,144 个元素的张量，
  这覆盖 prefill 的 MoE 部分和，不涉及 decode。
- **拒绝加载。** 以下情况会拒绝加载（退出码 2）：`TS_MUSE_SPARK_LAYER_SPLIT`；
  `--n-cpu-moe N`（`TS_MUSE_SPARK_CPU_MOE=auto` 会把所有专家留在显存中）；`TS_MUSE_SPARK_PIPELINE=1`；
  `TS_MUSE_SPARK_PROFILE`；大于 1 的 `TS_MUSE_SPARK_GRAPH_CACHE`；`TS_MUSE_SPARK_FUSE_QKVG=1`；
  `TS_MUSE_SPARK_CPU_REPACK=1`；`TENSORSHARP_TP_DEVICES`（该路径不读取它）；N 超过模型的 16 个 KV 头、
  超过 16 张 GPU 或超过可见的 GPU 数；GPU 型号不一致；以及某张 GPU 分不到 KV 组或专家行的方案。
  GPU 在某个形状下无法运行的图算子会让该次调用带着原因失败，而不是中止进程，因为张量并行的层无法回退到 CPU。逐算子路径自己的设置
  （`TS_MUSE_SPARK_PHASE_TIMING`、`TS_MUSE_SPARK_FUSE_QK_PREP`、`TS_MUSE_SPARK_REUSE_NORM_SCRATCH`、
  `TS_MUSE_SPARK_KV_UPLOAD_2D`、`TS_MUSE_SPARK_FUSE_RESIDUALS`、`TS_MUSE_SPARK_TP_HOST_POOL`、
  `TS_MUSE_SPARK_PREFILL_CHUNK`、`TS_MUSE_SPARK_SCRATCH_RESERVE_MIB`、`TS_MUSE_SPARK_RELEASE_GPU_SOURCE_PAGES`）
  会在日志中统一列出一次，说明被忽略。
- **内存。** 每张 GPU 持有自己的切片、镜像张量的一份副本（约 4.4 GB 的潜空间投影）以及整张图的计算缓冲区。
  上下文按每张 GPU 封顶，与按层切分相同（见[十张及以上 GPU](#十张及以上-gpu)）。在十张 A40 上，
  UD-IQ3_XXS 检查点在 32K 上下文、五个计划槽位下每张 GPU 占用 35.4–36.0 GiB 权重和 0.6 GiB K/V。
  K/V 行增长和前缀检查点拷贝在 GPU 上以一张拷贝图完成，不经主机往返。
- **关闭的功能。** 流水线 prefill 关闭，因为每张 GPU 都参与每一层。图缓存只保留一张图，
  因为 meta 后端只为最后一张图保留各 GPU 的视图；出于同一原因不使用投机解码，
  否则草稿验证图与 decode 图交替时每一步都要重建。

[InferenceWeb.Tests/MuseSparkNativeTensorParallelTests.cs](../../InferenceWeb.Tests/MuseSparkNativeTensorParallelTests.cs)
在 2、4、10 张 GPU 上与双精度参考解码器对照检查：K/V 行在滑动环回绕时增长的 prefill、decode、回退、
三个槽位的批量 decode 和前缀检查点克隆，以及各项拒绝。[基准报告](../perf/muse-spark-vs-llama-cpp.md)（英文）
解释了为什么在该主机上 `--tp` 的 prefill 受限于 all-reduce 传输，以及为什么长提示用按层切分更快。

### 十张及以上 GPU

- **对等连接上限。** 在带 NCCL 的 ggml 构建中（安装了 NCCL 时 TensorSharp 的 CMake 会启用它），
  ggml-cuda 用一次 `cuMemSetAccess` 调用把每个内核 scratch 池分块授权给所有支持对等访问的 GPU，
  而 CUDA 只允许每张 GPU 至多 8 个对等连接。在十张 A40 上，十张 GPU 都可见时每次 `ggml_cuda` 运行
  都以 `peer mapping resources exhausted` 中止：`--tp 10` 直接中止，`--layer-split 10` 则在执行器的八 GPU 上限（见下）
  不再先拒绝它之后中止。TensorSharp 的包装层
  [ggml_ops_cuda_vmm_peer_guard.cpp](../../TensorSharp.GGML.Native/ggml_ops_cuda_vmm_peer_guard.cpp)
  （以 `-Wl,--wrap=cuMemSetAccess` 链接，仅限 Linux；ctest `cuda-vmm-peer-guard`）捕获这一拒绝，撤销部分授权，
  只授权给所属 GPU，并警告一次（`[CUDA] GPU N cannot share its kernel-scratch memory with ...`）。
  GPU 之间的张量拷贝不使用该池。随后 NCCL 会在第一个通信器之前得到 `NCCL_P2P_DIRECT_DISABLE=1`，
  除非操作者已设置它或开启了 `NCCL_P2P_DISABLE`。llama.cpp `7170cad` 在这样的主机上也会同样中止，
  除非以 `-DGGML_CUDA_NCCL=OFF` 构建。
- **GPU 数量。** 两种模式下原生执行器最多跨 16 张 GPU。请求更多，或多于可见的 GPU 数，会拒绝加载（退出码 2），
  而不是在更少的 GPU 上运行。
- **上下文封顶。** 未设置 `MAX_CONTEXT`（以及 `TS_MUSE_SPARK_PLAN_CTX_TOKENS`）且宣称的上下文（1,048,576 token）放不下时，
  上下文被封顶为能放下的最大值（KV 对齐的倍数，至少 4,096 token），日志显示
  `[muse-spark] context capped to N tokens (the GGUF advertises 1048576) ...; set MAX_CONTEXT to choose a size`。
  设置了 `MAX_CONTEXT` 时，放不下的上下文会拒绝加载。

### 选项

| 设置 | 默认值 | 作用 |
|---|---|---|
| `TS_MUSE_SPARK_NATIVE` | CUDA/Vulkan 上开启，CUDA 上的 `--tp` 也开启 | `0` 选择逐算子路径（`--tp` 也是）；`1` 在 GGML CPU 后端上启用执行器 |
| `TS_MUSE_SPARK_CPU_MOE=auto` | 关闭 | 按字节最小化的逐专家栈放置（覆盖 `--n-cpu-moe`） |
| `TS_MUSE_SPARK_LAYER_SPLIT` | 自动 | 按层切分时每张 GPU 的层数，以逗号分隔的非负数、总和为 62；严格解析，格式错误会拒绝加载。与 `--tp` 同时使用会被拒绝 |
| `TS_MUSE_SPARK_UBATCH` | 2048 | prefill micro-batch 行数 |
| `TS_MUSE_SPARK_PIPELINE` | 自动 | 流水线 prefill。未设置：有两张或更多 CUDA 家族 GPU（CUDA、ROCm、MUSA）、没有主机专家、未设置 `TS_MUSE_SPARK_PROFILE` 且没有 `--tp` 时开启。`1` 强制开启（在 CUDA 家族以外的多 GPU 后端上以及 `--tp` 下会被拒绝）；`0` 关闭；其他值会拒绝加载 |
| `TS_MUSE_SPARK_PIPELINE_UBATCH` | 自适应 | 固定的流水线 micro-batch 行数，取 1 到 `TS_MUSE_SPARK_UBATCH` 之间的整数。未设置：约为每次调用的四分之一，从 512 行到 prefill ubatch，长上下文的掩码需要时缩小到 128 |
| `TS_MUSE_SPARK_PLAN_SLOTS` | 1 | 放置方案中预留的滑动窗口环数；有并发请求时调高 |
| `TS_MUSE_SPARK_PLAN_CTX_TOKENS` | 上下文长度 | 放置方案中为全局层 KV 预留的 token 数 |
| `TS_MUSE_SPARK_VRAM_RESERVE_MB` | 2048 + ubatch/4 | 每张 GPU 为计算缓冲区保留的空间 |
| `TS_MUSE_SPARK_DEV0_RESERVE_MB` | 0 | GPU 0 上额外保留的空间，例如留给视觉编码器 |
| `TS_MUSE_SPARK_SLOT_INIT_ROWS` | 4096 | 每个槽位全局层 KV 的初始行数 |
| `TS_MUSE_SPARK_SLOT_HEADROOM_MB` | 256 | 新槽位或 KV 增长之后必须保留的空闲内存 |
| `TS_MUSE_SPARK_ADMIT_GRAPH_MB` | 512 | 新槽位或 KV 增长时，在余量之外最多为图保留的空闲内存 |
| `TS_MUSE_SPARK_SWA_REWIND` | 512 | 使短距离回退成为可能的额外滑动环行数 |
| `TS_MUSE_SPARK_HOST_NUMA` | ≥2 个节点时开启 | 按 NUMA 放置的主机专家与绑核工作线程 |
| `TS_CPU_MOE_THREADS` | 允许使用的 CPU 数；NUMA 模式下约留出 1/6 | 主机专家工作线程数 |
| `TS_MUSE_SPARK_HOST_MOE_TOKENS` | 16 | 2 到 N 个 token 的批次通过静态行内核运行主机专家；`0` 保留 ggml 的 `mul_mat_id` |
| `TS_MUSE_SPARK_CPU_REPACK` | 关闭 | 对主机专家启用 ggml CPU 重打包；在这里没有加快 decode |
| `TS_MUSE_SPARK_OP_OFFLOAD` | 仅在没有主机专家时开启 | 允许调度器在大批次时把主机权重拷贝到 GPU |
| `TS_MUSE_SPARK_WARM_POOLS` | 有 GPU 时开启 | 加载时运行一次满尺寸 prefill，让 ggml-cuda 的内核 scratch 池在槽位占用内存之前达到峰值。流水线 prefill 开启时，还会运行一次简短的流水线 prefill（两个固定或四个最小的自适应 micro-batch），构建流水线的调度器并确认它确实生效 |
| `TS_MUSE_SPARK_EXEC_DEBUG` | 关闭 | 记录放置、图构建与内存决策 |
| `TS_MUSE_SPARK_PROFILE` | 0 | 打印 N 张图的逐节点耗时；会关闭流水线 prefill，在 `--tp` 下会被拒绝 |

### 启动

实测配置使用十张 A40 GPU（每张 46 GB）和 8 个分片的 UD-IQ3_XXS 检查点（315.7 GiB 权重），所有专家都在显存中。
32K 上下文和五个计划槽位为四个并发请求和共享前缀检查点留出了空间。
同一条命令运行按层切分；把 `--layer-split 10` 换成 `--tp 10` 即为张量并行：

```bash
MAX_CONTEXT=32768 TS_MUSE_SPARK_PLAN_SLOTS=5 TS_CODE_EXEC=0 \
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll \
  --model /workspace/Muse-Spark-Open-816B-A42B-GGUF/UD-IQ3_XXS/Muse-Spark-Open-816B-A42B-UD-IQ3_XXS-00001-of-00008.gguf \
  --backend ggml_cuda --layer-split 10 \
  --host 127.0.0.1 --port 8083 --no-webui --no-skills --no-multi-agent
```

按层切分在每张 GPU 上放置约 36 GiB，开启流水线 prefill，加载耗时 6.0 分钟。
`--tp 10` 下每张 GPU 持有 35.4–36.0 GiB 权重和 0.6 GiB K/V，加载耗时 6.4 分钟。
在该主机上两种运行都依赖[对等连接上限保护](#十张及以上-gpu)。

### 实测结果

使用上面的十张 A40 配置，在同一台机器、同一检查点上与 llama.cpp `7170cad` 对比：驱动 595.91、CUDA 12.8、
未修改的上游 ggml `da8e2a5`；llama.cpp 以 `-DGGML_CUDA_NCCL=OFF` 构建，因为它的 NCCL 构建在该主机上会遇到
[对等连接上限导致的中止](#十张及以上-gpu)。该主机上 GPU 之间的对等访问被宣称可用但实际不工作，因此两个引擎都没有使用 P2P。
吞吐与延迟数据为三次重复的中位数，llama.cpp 张量切分各行除外（两次）；每个夹具运行一次。比值大于 1.00x 表示 TensorSharp 更好。标为"见报告"的单元格请查看
[完整报告](../perf/muse-spark-vs-llama-cpp.md)（英文），其中还有测量方法、各场景细节以及这次对比的局限。

| 场景 | TensorSharp `--layer-split 10` | llama.cpp `-sm layer` | 比值 | TensorSharp `--tp 10` | llama.cpp `-sm tensor` | 比值 |
|---|---:|---:|---:|---:|---:|---:|
| decode，单请求（tok/s） | 16.69 | 16.79 | 0.99x | 22.63 | 16.19 | 1.40x |
| 首 token 时间，短提示（s） | 0.64 | 0.75 | 1.17x | 0.58 | 1.29 | 2.22x |
| prefill，4K token 提示（tok/s） | 999.7 | 483.2 | 2.07x | 294.4 | 75.9 | 3.88x |
| prefill，16K token 提示（tok/s） | 1346.8 | 826.1 | 1.63x | 295.2 | 见报告 | |
| 四个并发请求，总输出（tok/s） | 33.65 | 33.04 | 1.02x | 45.73 | 见报告 | |
| 2.3K token 对话的第二轮，首 token（s） | 0.45 | 0.91 | 2.00x | 0.46 | 见报告 | |
| 共享 2.3K token 系统提示的新对话，首 token（s） | 0.48 | 0.64 | 1.34x | 0.51 | 见报告 | |
| 答案校验夹具（`eng/muse-spark-validate.py`） | 13 / 14 | 12 / 14 | | 14 / 14 | 见报告 | |
| 加载时间 | 6.0 分钟 | 14.4 分钟 | 2.4x | 6.4 分钟 | 14.7 分钟 | 2.3x |

按层切分适合长提示：流水线 prefill 让十张 GPU 都在处理同一个提示。它的单请求 decode 与 llama.cpp 持平
（0.99x，在运行间波动范围内），因为两个引擎都让一个 token 依次经过十张 GPU。张量并行适合生成：
它的 decode 速度是 llama.cpp 张量切分的 1.40 倍，是任一引擎按层切分的 1.35 倍。它的 prefill 停留在约 295 tok/s，
因为该主机上的 all-reduce 走 NCCL 的共享内存传输，而 Muse Spark 在求和之前先对每个专家的输出做归一化，
所以 MoE 的 all-reduce 要携带每个被选中专家的输出。

在 teacher-forced 的 wikitext-2 上（相同的 token ID，12,276 个计分位置），按层切分的下一 token 分布与 llama.cpp 的
平均 KL 散度为 0.0026，低于 llama.cpp 只改变 micro-batch 大小时与自身之间的 0.0028；由流水线 prefill 写入的前缀结果相同。
`--tp 10` 处于这一噪声水平：其平均值 0.0031 比它高 9%，最大值也更高，而 p99 和 top-1 一致率更好；
ggml 的 NCCL 路径以 BF16 对其 prefill 的 MoE 部分和求和。按层切分唯一失败的夹具是交错到达用例：一个请求在 512 个 token 中的第 322 个处以通顺的回答
和文本结束 token 结束，这是贪心解码的分支，不是引擎故障；llama.cpp 的按层切分也没有通过这个用例。
在上述工作之前，TensorSharp 在该主机上两条命令都无法运行，它早先的 `--tp` 路径 decode 速度为 1.9–2.4 tok/s。

## 逐算子路径上的张量并行

本节描述托管逐算子路径的张量并行：设置 `TS_MUSE_SPARK_NATIVE=0` 或后端为 `ggml_vulkan` 时，`--tp N` 使用它。
在 `ggml_cuda` 上，`--tp N` 改为运行[原生执行器](#原生执行器上的张量并行)。

`--tp N` 按完整的 GQA 头组切分 Q/K/V/gate 投影，并按输入列切分注意力输出投影。
每张 GPU 并发计算自己的头，然后在 post-attention 归一化之前把各部分输出投影相加。
路由专家以整个专家为单位划分到同一组 GPU 上。每个专家完整的潜空间输出先做归一化，再施加路由权重和跨 GPU 求和。
路由器、潜空间下/上投影、残差流和输出头在 rank 0 上执行。

这支持不均匀的头/专家划分，包括为已发布的 128 查询头 / 16 KV 头检查点使用九张 GPU。
每个 rank 必须至少拥有一个 KV 组和一个专家，量化注意力输出的切分必须保持完整的量化块。
不支持的几何形状和多节点 TP 会被拒绝。
`--n-cpu-moe N` 仍会在主机上对所选前缀中保留的原生专家栈只计算一次；它不会在各 rank 之间重复这些专家计算。
这覆盖了已发布的量化检查点。使用普通张量运算回退的 F32 专家张量仍留在所选后端（TP 中为 rank 0），
即使它们位于该前缀之内。容量检查把这些 F32 权重计为 GPU 权重。

GGML 逐算子路径把部分结果返回主机内存。小规模归约在主机上进行；较大的归约可以使用后端传输。
Spark 要求保留 F32 的归约，包括 TensorSharp 自有的 F32 NCCL 传输或精确回退，
因此 prefill 集合通信不会悄悄把部分和收窄为 BF16。PCIe 拓扑和 rank 0 上的工作可能限制扩展性。
因此 TP 需要在目标硬件上实测；它既不承诺线性加速，也不承诺降低模型的总内存。

TP 路径保留各自独立的 Q/K/V/gate 权重投影。可选的 `TS_MUSE_SPARK_FUSE_QK_PREP=1` 会打包这些投影已初始化的主机输出，
并在每个 CUDA rank 上按该 rank 实际的头数一并完成 Q/K 归一化和可选的 RoPE。
它会增加主机打包开销，默认关闭；它在 TP 下的性能需要受控对比。norm scratch 复用和 F16 2D KV 上传仍然可用。
按层切分运行中的优化成功使用记录，并不能证明它们在 TP 运行中也被使用。

Spark 针对 F32 权重的 CUDA 投影显式保留 F32 操作数，包括不均匀的注意力分片、专家平面和输出头。
这条限定范围的路径为每个 rank 复用一个后端包装器；量化权重和其他模型保留原有的矩阵乘策略。
它针对的是观察到的、完整投影与不均匀分片投影之间随形状变化的 TF32 运算差异。
这项修正的设备回归和整模型验证尚待完成；此前失败的数值检查仍然保留记录。

Spark 的 CUDA F32 KV 路径还使用 TensorSharp 自有的、保留 F32 Q/K/V 操作数的注意力。
上游 CUDA flash attention 可能把 F32 K/V 收窄为 F16；已发布模型的头几何在打包标量检查中暴露了这一差异。
普通与分页的 Spark 调用现在都要求显式的自有 CUDA 成功状态，并分别报告成功使用计数。这不是上游的 flash 执行。
F16 路径和其他模型的注意力策略不变。严格的打包标量界限保持不变；这项修正的设备验证尚待完成。

容量检查在加载权重数据之前根据 GGUF 描述符运行。TP 检查实际的不均匀分片加上 rank 0 的权重，
按层切分则检查完整的层分配。其余的内存估算取决于工作负载；见下文[启动分片模型](#启动分片模型)中的指南。
两种模式都保留 F32/F16 线性 KV 快照和投机回退；打包连续批处理需要 F32 KV。

## 早期五 GPU 按层切分测量

> 本节是摘要。完整记录（各阶段细节、证据与审计说明）见[英文卡片](muse-spark.md#previous-five-gpu-layer-split-measurements)。

这些测量早于原生执行器：它们使用逐算子路径，`TS_MUSE_SPARK_NATIVE=0` 仍会选择该路径。

**F16（v4r2）活动。** Q3 检查点上已完成的 v4r2 F16 活动通过了 13 个选定的质量夹具、一项单独的思维链检查，
以及全部九个计时用例的内容检查。夹具覆盖简短的英文/中文回答、JSON/schema/Unicode、工具与智能体工作流、
图像/图表 OCR、对话记忆和并行请求隔离，共 22 个计量轮次外加一次预热，并不是广泛的模型质量评估。
三个阶段的全部 35 个请求（含被排除的预热）报告的缓存 token 数均为整数零。

**未达到所要求的性能目标。** 在等工作量的 512 token decode 对比中，TensorSharp 的客户端墙钟时间中位数为
**111.743 秒**，llama.cpp 为 **80.631 秒**。参考/候选的墙钟比值中位数为 **0.721574**，
低于冻结的验收阈值 **0.95**（5% 的速度比容差）。这是一种硬件/配置组合上的三次重复，不是统计等价性结论。

| 场景，三次重复 | TensorSharp 墙钟中位数（秒） | llama.cpp 墙钟中位数（秒） | 提示 token；输出 TensorSharp/参考 | 解读 |
|---|---:|---:|---|---|
| 简短回答 | 6.598 | 13.738 | 156; 1/2 | 仅为描述性任务延迟：原始输出工作量不同 |
| 固定 decode | 111.743 | 80.631 | 171; 512/512 | 等工作量性能目标未通过 |
| 长检索 | 294.907 | 441.550 | 7812; 31/32 | 仅为描述性任务延迟：原始输出工作量不同 |

由于短/长输出的数量不同，整体严格对比没有定论；这并不抹去符合条件的 decode 失败。
模型加载、显式 CPU 专家页准备和预热不计入；两个计量的性能阶段在服务端的物理读字节数和主缺页次数均为零。
F16 走的是普通序列路径，不是打包 F32 批处理。

**F32 打包批处理活动存在未解决的质量失败。** 交错到达的长回复返回了起草性的评论，而不是哈希表的解释，
并在 189 个 token 处停止，未满足其恰好 512 token 的要求。同一 F32 进程上的精确串行重放也返回了非答案，
在 179 个 token 处停止。因此仅凭并发无法解释这一失败；这些观察也没有把原因定位到精度、某个内核或某项优化。
成功的 F16 decode 请求在验证标签和提示 token 数上也不同，因此它不是受控的 F16/F32 因果对比。
新的调度器记录证明确实以 `BatchedPaged` 执行（三个拷贝请求一起服务；交错准入产生了五个混合 prefill/decode 步，
最多同时服务四个请求）。其他若干夹具和三个无投机回归夹具通过，但这些成功不改变交错用例的失败判定；
显式的失败活动清理保留了失败的质量结论。

**其他机制检查。** 经独立审计的四请求 n-gram 续跑与未修改、已通过的无投机基线输出一致，
记录了 225 个草稿中接受 138 个 token、34 个验证步和 17 次回滚。它使用 `TS_SPEC_ADAPTIVE=0` 和单独的 F32 机制配置，
只确立这些机制检查，不代表单独的投机加速，也不修复失败的交错回答。
单独的 F16 被挤出前缀（displaced-prefix）探测通过了独立的活动审计，四个请求（含预热）都返回了精确答案：
第一个请求为 222/5 个提示/输出 token、无复用，无关的挤出请求为 156/1、无复用，续写请求为 248/16，其中 224 个是缓存 token。
新的遥测确认从 radix 页面恢复，首个 prefill 位置为 224，只剩 24 个新提示 token；恢复的前缀超过挤出请求的 157 个总 token。它证明了所测的跨请求缓存复用，不声称缓存加速，也不改变失败的 F32 结果。

**参考引擎。** llama.cpp 通过了 13 个有界质量夹具中的 12 个；其对抗性 schema 回复是字面量 `OVERRIDE`，未通过。
配合式的 JSON 输出不能证明 schema 语法被强制执行。两个引擎的夹具通过数都不能确立通用模型质量的持平或优劣。

生成的证据包括
F16 活动与未修改的配对分析（`docs/validation/muse-spark/q3-final-v4r2-performance-complete/independent-audit.json`）、
资源与实际使用摘要（`docs/validation/muse-spark/q3-final-v4r2-performance-complete/completed-campaign-summary.json`）、
失败的 F32 活动、串行重放与清理（`docs/validation/muse-spark/q3-f32-nospec-failed-cleanup-v4r2-complete/SUMMARY.md`）、
独立的 n-gram 续跑（`docs/validation/muse-spark/recovery-ngram-offline-audit.json`）
以及被挤出前缀的复用（`docs/validation/muse-spark/recovery-prefix-offline-audit.json`）（本地验证记录，未提交到 Git）。
这些被忽略的报告由验证过程生成，新的源码检出中可能不存在。记录的源码清单为
`f8615e93fb90286076ce21e4db15c516735d722b457fb562beaaa7a46f940f56`；
原生库 SHA256 为
`9e7e7153a65a3fdf8ef4499a39c8c9a46a15d2a8b83c313dcdd1c19684fb8e71`。
之后一项经独立验证的修复会在后端拆除/重建之前清理调用线程拥有的分页注意力会话；它不在这些整模型基准二进制中，
而且这些新进程模型运行没有重建后端，因此这一生命周期发现不能解释 F32 非答案。
其他线程上的工作者仍须在后端拆除之前排空；该修复不声称能清理每个线程的缓存。见
隔离的生命周期审计 `docs/validation/muse-spark/lifecycle-completed-review/accepted-audit.json` 和
最终已应用源码回执 `docs/validation/muse-spark/final-applied-source/receipt.json`（本地验证记录，未提交到 Git）。
修复后的库上没有重跑整模型推理和计时。

## 启动分片模型

把所选量化的所有分片放在同一目录中，并传入其中任意一个分片（`UD-Q4_K_XL` 有十二个；`UD-Q3_K_XL` 有十个；`UD-IQ3_XXS` 有八个）。
TensorSharp 从第 1 个分片解析架构、分词器和模型设置，然后从完整的分片集合加载张量。
后续分片只包含切分元数据，不是独立的模型。
下面的 Q3 配置是五张 A40 GPU 的显式 F16 评估配置：每张 GPU 总内存 46,068 MiB、初始可用 45,490 MiB，
容器内存上限 233.762 GiB，CPU 配额 40.8 核。它把前 32 个专家层放到 CPU（182 GiB 专家权重），
使用 32 个 CPU 工作线程，并把解码器层分配为 `35,7,7,7,6`。
共享 KV 池总共容纳 16,384 个 token，最多四个活动请求，每个请求上限 16,384 token；它并不预留四个同时存在的 16K 上下文。
下面六个优化开关是这套配置显式开启的选项，它们在实现中的默认值仍为关闭。
前缀缓存和投机解码被禁用，以控制对齐的性能负载。
前文已完成的 F16 结果使用这套配置；它们不能确立通用的质量持平，而且所测的等工作量 decode 目标未通过。

```bash
MAX_CONTEXT=16384 TS_KV_INITIAL_TOKENS=2048 \
TS_SCHED_NUM_BLOCKS=1024 TS_SCHED_BLOCK_SIZE=16 TS_SCHED_MAX_RUNNING_SEQS=4 \
TS_SCHED_MAX_BATCHED_TOKENS=256 TS_SCHED_PREFILL_CHUNK=256 \
TS_SCHED_SOLO_PREFILL_CHUNK=256 TS_MUSE_SPARK_PREFILL_CHUNK=256 \
TS_MUSE_SPARK_LAYER_SPLIT=35,7,7,7,6 TS_MUSE_SPARK_STREAM_EXPERTS=0 \
TS_MUSE_SPARK_RELEASE_GPU_SOURCE_PAGES=1 TS_MUSE_SPARK_FUSE_RESIDUALS=1 \
TS_MUSE_SPARK_FUSE_QK_PREP=1 TS_MUSE_SPARK_REUSE_CPU_MOE_SCRATCH=1 \
TS_MUSE_SPARK_KV_UPLOAD_2D=1 TS_MUSE_SPARK_REUSE_NORM_SCRATCH=1 \
TS_GGML_CPU_THREADS=32 TS_CPU_MOE_THREADS=32 OMP_NUM_THREADS=32 \
DOTNET_PROCESSOR_COUNT=40 TS_CODE_EXEC=0 TS_HOST_MOE_TIMING=0 \
dotnet run --project TensorSharp.Server.Host -c Release -- \
  --model /workspace/models/muse-spark/UD-Q3_K_XL/Muse-Spark-Open-816B-A42B-UD-Q3_K_XL-00001-of-00010.gguf \
  --mmproj /workspace/models/muse-spark/mmproj-Q8_0.gguf \
  --backend ggml_cuda --layer-split 5 --n-cpu-moe 32 --cpu-moe-threads 32 \
  --kv-cache-dtype f16 --host 127.0.0.1 --port 8083 --max-tokens 512 \
  --no-webui --no-skills --no-multi-agent --sampling-precedence request \
  --no-prefix-cache --no-spec
```

这条命令是配置示例。可复现的活动会从隔离构建中启动经过测试的那份服务端 DLL 和原生库，并保留它们的哈希；
用 `dotnet run` 重新构建无法复现那份二进制身份。F16 可以通过普通序列调度器服务相互重叠的 HTTP 请求。
这并不执行打包的 F32 `ForwardBatch` 路径，仅有请求重叠也不能证明打包连续批处理。

做对齐评估时，HTTP 测试工具还会向 TensorSharp 发送 `skills: []`、`skills_discovery: false` 和 `multi_agent: false`，
同时保留调用方声明的工具。这些请求设置防止服务端额外提供的技能/委派指令改变提示；它们不改变解码器。
无缓存测量还会发送 `cache_prompt: false`，像上面那样在服务端启动时禁用前缀缓存，并核对观察到的复用计数。

评估打包分页文本路径时使用 `--kv-cache-dtype f32 --continuous-batching`，并在有界 F32 验证配置中
把上面四个批/prefill 行数上限从 256 降到 128。该机制测试配置还设置 `MAX_CONTEXT=8192` 和 `TS_SCHED_NUM_BLOCKS=512`：
8192 token 的上限和 8192 token 的共享池在这台机器上为计算 scratch 留出更多空间；它们并不提供四个同时存在的 8K 上下文。
请显式设置 `TS_MUSE_SPARK_KV_UPLOAD_2D=0`，因为该选项作用于普通 F16 注意力路径。
基础配置中保持专家流式传输（`TS_MUSE_SPARK_STREAM_EXPERTS`）禁用。
打包批次需要 F32 KV 且只接受纯文本请求；有待注入的图像 embedding 时会选择普通路径。
这套确切的 F32 机制配置产生了前文所述的打包调度器记录，以及未解决的交错/串行质量失败。
不得把它当作 F16 配置的已验证替代。单独的 n-gram 续跑通过了其有界机制检查；
单独的 F16 被挤出前缀探测也通过了独立验收。

F32 KV 的存储是 F16 的两倍。每次对比都要让两个引擎的缓存精度、上下文和调度器配置保持一致。
对于上面的 F16 配置，llama.cpp 的 `--ctx-size 16384 --kv-unified
--kv-unified-per-slot 16384 --parallel 4` 保持相同的总容量和请求上限。
它的整层切分包括输出投影，因此对应设置为 `--split-mode layer --tensor-split 35,7,7,7,7`。
这些设置描述的是此前的整层对比。TensorSharp 单独的 `--tp N` 模式执行前文所述的注意力切分。

规划主机专家时请以进程/容器内存上限为准。卸载 42 层的 Q4 配置仅这些专家权重就需要约 334 GiB；
即使主机报告 504 GiB 内存，它们也无法常驻在这个 233.762 GiB 的容器中。
它的整模型尝试在 HTTP 就绪之前因严重换页而被中止，不算通过的推理测试。
`eng/validation/sample-inference-hardware.py` 把 cgroup 限制与主机容量分开记录。
`eng/validation/muse-spark-launch-plan.py` 根据完整的分片头以及显式的 KV、投影器、运行时和流式传输预留来估算放置；
估算能放下仍需要实测的内存峰值来确认。

在 GGML CUDA/Vulkan 上做整层放置时，TensorSharp 现在会在读取权重数据之前，
把 GGUF 张量描述符与每个所选设备当前的空闲内存进行比对。
拒绝信息会区分"权重本身已超出显存"与"估算的计算/活动 KV 余量不足"。
当更高的 CPU 卸载数能满足同一份基于头部的估算时，它会报告这个数；设置永远不会被自动更改。
主机内存也必须容得下这些被卸载的专家。

逐算子路径的张量并行同样在加载权重之前根据描述符做规划。如果均匀的整专家数量能放进每个实测设备，它就保持均匀。
否则它会分配连续的专家区间，以平衡扣除固定注意力/主干权重和同样的预留之后的剩余内存；
rank 0 可以拥有更少的专家来容纳它的主干。预检和加载器使用同一份不可变的方案，启动日志会记录其数量和策略。
如果任何设备的空闲内存无法获取，它会回退到均匀数量，并报告哪些检查不可用。
这永远不会更改 `--n-cpu-moe`。它不会为视觉塔预留空间，也不能证明配置的最大上下文能放得下。

在逐算子路径上，Muse Spark CUDA TP 上下文每个 rank 至多保留八个可复用的主机 scratch 块，上限为 128 块、每块至多 8 MiB（共 1 GiB），按需分配。
单设备、按层放置以及其他模型的池保持原有策略。`TS_MUSE_SPARK_TP_HOST_POOL=0` 选择原来的八块池，用于受控对比。
设置 `TS_MUSE_SPARK_PHASE_TIMING=1` 时，前向报告会在阶段计时旁附带不可变的前/后主机池计数。
这些计数衡量的是主机分配/复用，不是设备内存或内核时间。增加的保留量是有界的；对整模型性能的收益仍未测量。

余量估算为：对这个完整模型、最多 256 个 prefill 行时，每张 GPU 2560 MiB；它随模型几何和更大的分块而增大，
不包括视觉塔和额外保留的请求。它不是能放下的保证。
`TS_MUSE_SPARK_SCRATCH_RESERVE_MIB` 可显式覆盖该估算；调低它无法绕过仅针对权重的检查。
请像示例中那样显式设置 `MAX_CONTEXT` 和 `TS_KV_INITIAL_TOKENS`：模型宣称的上下文上限并不能证明一台机器装得下那么多 KV 内存。
GGML 线性 KV 张量以主机内存为后备，除非配置的初始分配预留更多，否则会惰性增长；
检查报告的是它们的初始和最大主机大小，而不是把完整的宣称上下文计入每张 GPU。

使用图像时，像上面那样传入匹配 Spark 发布版的 `--mmproj`。
它的投影器类型是 `muse-glimmer`，但输出宽度必须与 Spark 解码器一致（8192）。
Muse-Glimmer-30B 的投影器输出宽度不同，会被拒绝。OpenAI 兼容请求可以包含 `image_url` 内容段。
图像区间会跨内部 prefill 分块保留，在 embedding RMSNorm 之前注入，并通过正常的注意力路径处理。

使用 `--spec --spec-type ngram --spec-draft 8` 请求无权重的投机解码，它替换示例中的 `--no-spec`。
有界的 F32 机制探测可以设置 `TS_SPEC_ADAPTIVE=0`，防止自适应调控器关闭投机；
这是测试配置，并不验证默认的自适应策略。接受率取决于提示/输出中的重复文本。
在声称投机加速了某个负载之前，请核实实际的草稿提议和被接受的 token；仅仅启用该标志并不能证明加速。

在逐算子路径上，`TS_MUSE_SPARK_LAYER_SPLIT` 可选地为每张 GPU 指定一个正的层数，各数之和为 62
（原生执行器也接受 0；见[选项](#选项)）。
没有它时，TensorSharp 按常驻权重字节数平衡连续的层组，并计入 CPU 专家卸载以及 embedding/输出矩阵。
KV 缓存、图像编码器和计算 scratch 需要额外内存。
层放置是在不同设备上依次执行相继的层组；它不会把一次矩阵乘切分到多张 GPU 上。

## 架构与数值顺序

发布的元数据声明了 62 个块、隐藏宽度 8192、128 个查询头、16 个 KV 头、头维度 64、256 个专家（top-8 路由），
以及 4096 的专家 latent/中间宽度。它宣称的最大上下文为 1,048,576 个 token；
这一元数据值并不能证明当前实现或某台特定机器已在该上下文下验证过。配置的实际运行上下文应与真实的内存预算相符。

每个块按以下顺序执行：

1. pre-attention RMSNorm，Q/K/V 与注意力门控投影，逐头 Q/K RMSNorm。
   只在 GGUF 布尔模式中标记为滑动的层上施加相邻成对的 RoPE；全局层使用 NoPE。
2. GQA 因果注意力，滑动层使用 2048 token 窗口。对注意力输出逐头归一化，乘以 sigmoid 门控，
   投影回残差宽度，再施加无权重的 post-attention RMSNorm。
3. 用 GGUF 的逐通道 alpha/beta 权重合并残差与分支。这些系数已经从源模型的门控转换而来。
4. 对 FFN 输入做归一化。路由器读取这条全宽度的流，而一个单独的投影和 norm 把专家输入移入潜空间。
5. 用 `sigmoid(router_logits) + correction_bias` 选择专家。修正偏置只影响选择。
   对所选专家未加偏置的 sigmoid 分数做归一化（下界与参考实现相同），并施加配置的路由缩放。
6. 运行每个所选专家的 SwiGLU 投影。把每个专家的输出乘以 post-expert 缩放并做 RMS 归一化，
   **然后**再乘以其路由权重并对各专家求和。把这个 norm 移到求和之后会改变模型。
7. 投影出潜空间，乘以 FFN post-norm 缩放，施加 RMSNorm，并用其 alpha/beta 权重与 FFN 残差合并。

输入的 token/图像 embedding 经过一次无权重的 RMSNorm。最终的流经过一个可学习的 RMSNorm 和输出投影，
随后是 logit 缩放（本发布版为 `0.17677669`）以及上限为 20 的 tanh 软上限。
块的 post-norm 使用 GGUF 中单独给出的 post-norm epsilon。
Delta norm 权重和 Q MetaP 缩放已经折叠进 GGUF 张量，不得再次施加。

## 优化与当前限制

原生专家图用 `mul_mat_id` 批量处理所选专家，并在一张图中完成对顺序敏感的专家 norm、路由乘法和归约。
被卸载的保留原生专家栈默认在 CPU 上运行；前文所述的 F32 张量回退限制仍然适用。
实验性的 `TS_MUSE_SPARK_STREAM_EXPERTS=1` 选项允许至少 `TS_HOST_MOE_DEVICE_MIN_BATCH` 行（默认 128）的 prefill
把所选专家流式传到该层所在的 GPU。它支持按层切分，并在选择 GPU 之前，
把实测的完整专家栈图分配量与实际空闲显存加上其可复用缓冲区进行比较。
`[muse-spark-stream]` 诊断会记录该决策和各项内存大小；余量不足时工作留在 CPU 上。
合成的原生测试覆盖了流式机制，但完整检查点下流式运行的质量和性能仍未确立。
上面的 F16 配置禁用了该选项。托管回退路径按专家对路由后的 token 分组，以批量执行 GEMM。

`TS_MUSE_SPARK_REUSE_CPU_MOE_SCRATCH=1` 选择为 Spark 被卸载的专家在各层和各分块之间保留一个 CPU 图 arena。
请在加载原生库之前在进程环境中设置它。调用方拥有的权重和激活仍以外部方式绑定；
每次调用都会上传新的路由、偏置和 norm 权重。该 arena 受主机计算互斥锁保护，并在后端拆除之前释放。
其他模型保持原有的分配路径。首次成功诊断会报告实际分配的字节数；
原生查询 `TSGgml_GetMuseSparkCpuMoeScratchState` 报告保留的字节数和成功的 arena 调度次数。该选项默认关闭。
已完成的 CPU/CUDA 组件测试，以及一次使用合成激活/路由的真实 Q3 第 0 层 CPU 对比，检查了它的数值行为；
后者不是整解码器吞吐测试。合并的 F16 活动中也记录了实际使用；其结果无法单独体现该选项的端到端贡献。

在 Linux CUDA 上，`TS_MUSE_SPARK_RELEASE_GPU_SOURCE_PAGES=1` 选择在一次成功上传并通过与常驻设备副本的精确比对之后，
对源文件页面发出释放建议（advise away）。它保留 GGUF 映射、主机指针和缓存键，以便回退或重新上传。
CPU/流式专家、CPU embedding gather 以及拷贝出来的融合缓冲区不在此列。
该建议只覆盖每个张量内的完整页面，并使用原始的分片句柄。
诊断对新被建议的权重和整页字节按每个权重只计一次；发出建议并不保证物理逐出。该选项默认仍关闭。
F16 活动记录了累计 277 个权重、193,806,282,752 字节被建议，并附有实测的内存与 I/O 证据。
这些观察不能证明每个被建议的页面都被逐出，也不能单独体现该选项带来的吞吐增益。

`TS_MUSE_SPARK_FUSE_RESIDUALS=1` 选择在 CPU 和 CUDA 上（包括打包批次）为每个 post-normalization 门控残差使用单张 GGML 图。
FFN 通道缩放仍在归一化之前；alpha 和 beta 仍分别作用在两条残差分支上。诊断会确认首次成功调度。
该图现在把归一化分支和 beta 缩放分支保留为各自独立存储的 F32 中间结果，防止 CUDA 表达式融合改变它们的舍入边界。
需要新的、与独立原生运算的同输入对比；这项修正的设备验证尚待完成。
其他后端保留原始运算。该选项默认仍关闭。已完成的合并 F16 配置记录了成功使用；这不是单独的残差融合速度对比。

`TS_MUSE_SPARK_FUSE_QK_PREP=1` 在 CPU 和 CUDA 上把逐头 Q/K 归一化与滑动层的 RoPE 合并到一张 GGML 图中。
普通路径使用融合的 Q/K/V/gate 投影；CUDA TP 打包其各自独立的投影输出，而不改变权重归属。
其他后端保留现有路径。全局层保持 NoPE。Q/K 准备图和融合残差图复用设备本地的 scratch，每次调用都使用新的输入。
诊断会确认首次成功的 Q/K 调度。该选项默认关闭。
已完成的合并 F16 活动记录了成功使用；要把性能增益归因于它，需要单独的受控对比。

`TS_MUSE_SPARK_KV_UPLOAD_2D=1` 对 Spark 的 F16 K/V 注意力窗口使用两次带步长的 GGML 传输，取代逐 KV 头的单独传输。
它保持同一份以主机为准的缓存、传输字节数、窗口偏移和缓存生命周期。
GGML CPU 和 CUDA 支持该选项；其他模型和分页 F32 路径保留现有的传输。诊断会确认首次成功调度。该选项默认关闭。
已完成的组件检查包括 CPU、CUDA 和双设备用例，以及微型模型的分块、前缀恢复和投机回滚。
实际的 F16 使用和整个活动的计时已在前文[早期五 GPU 按层切分测量](#早期五-gpu-按层切分测量)中记录，
但没有针对该选项的单独性能结论。

`TS_MUSE_SPARK_REUSE_NORM_SCRATCH=1` 选择在 GGML CPU 和 CUDA 上启用 Spark 专用的 RMS 归一化 scratch 复用，
并把注意力输出的逐头归一化与其 sigmoid 门控合并。可学习缩放仍在 RMSNorm 之后；
其他后端和默认配置保留原始运算。共享 arena 遵循现有的串行计算约定，所有结果都会在下一个运算可能复用其存储之前拷回。
成功复用和成功执行头 norm/门控各有独立的首次使用诊断。
如果复用分配失败，会改用新的分配，并记录为回退，而不是记录为已执行的复用。该选项默认关闭。
已完成的 CPU、CUDA 和双设备数值/生命周期测试以及微型模型等价检查验证了所测路径，包括从 rank 1 重建后端。
合并的 F16 活动也记录了两条成功使用诊断，并通过了其有界质量夹具。
这些夹具和组件检查都不能确立通用质量持平或单独的归一化加速。

在六个已启用的优化选项中，已完成的 F16 日志包含七条成功使用记录：Q/K 准备、残差融合、F16 2D KV 上传、
CPU 专家 scratch、源页面建议、加权 RMS scratch 以及头 norm/门控 scratch。最后两条属于同一个选项。
流式传输单独禁用；它不是第七个启用的优化。首次使用记录只能确立至少一次成功使用，不能说明调度频率或单独的加速。

注意力使用融合的 GGML 注意力运算；F16 和分页的滑动层只提交存活的 key 窗口。prefill 分块限制了激活内存。
前缀快照和投机回滚保留已接受的线性 KV 前缀。该模型尚无捕获的整解码器 CUDA 图。
主机路由器传输、逐算子调度和以主机为后备的分页路径仍然是性能开销。

## 复现验证

> 本节是摘要。完整的步骤、通道细节与各项限定见[英文卡片](muse-spark.md#reproducing-validation)。

已完成的冻结 v4r2 组件运行使用未修改的上游 ggml 修订 `a05c9922613c6a2230dbb1d0de679519ecafb4f0`，
并在组件、微基准和回归检查中使用同一套隔离的原生/托管二进制：

| 检查 | 观察结果 | 范围 |
|---|---|---|
| 原生 CTest 套件 | 24 个通过，零失败/零跳过 | CPU、CUDA 和双设备用例，包括 norm/门控的拆除与重建 |
| 选定的托管 CPU 通道 | CPU 专家 arena 关闭时 95 个通过；开启时同样 95 个通过 | 组件/微型模型检查；断言了实际的显式开启执行 |
| 选定的托管 CUDA 通道 | CPU 专家 arena 关闭时 23 个通过；开启时同样 23 个通过 | 组件/微型模型检查，含双设备用例；零跳过 |
| 广泛的托管回归 | 1,524 个通过，零失败，七个显式跳过 | 六个不可用的模型/分词器夹具和一个仅限 Windows 的平台测试不算通过 |
| 后端/CLI/预热补充 | CPU/CUDA 每个通道 25 个用例通过，零失败/零跳过 | 25 个不同用例各执行两次，不是 50 个不同测试 |

- 四个选定托管通道均为零失败/零跳过，共 118 个不同的结果名称和 236 次执行；不要把重复的 arena 模式计为额外的不同测试，
  也不要把重叠的组件套件与广泛套件加总成一个覆盖总数。Q3 专家微基准在独立进程中以两种顺序比较 arena 关/开
  （第 0 层、32 线程、1/128/256 行），输出哈希完全一致；它使用合成激活/路由，不能确立整模型或相对 llama.cpp 的性能。
- 之后的隔离生命周期证明在 CPU、CUDA 和双 GPU 通道上复现了被保留的会话，然后验证了每个固定通道三次重建和三次关闭之后的清理
  （CPU 与双 GPU 通道各通过 15 张数值图和 60 个损坏对照，CUDA 通道通过 9 张图和 36 个对照）。
  更广的固定库套件通过了 27 个原生用例，以及新的托管通道 CPU Muse 95、CPU 后端/生命周期 25、CUDA Muse 23、
  CUDA 后端/生命周期 25，无失败或跳过。这些是重复的已有组件用例，不计入前面的覆盖总数。
  原始的绝对误差失败、有界的 CUDA 舍入诊断和仅出现在控制台的加载器审计失败仍保留在
  隔离证明 `docs/validation/muse-spark/lifecycle-completed-review/accepted-audit.json`（本地验证记录，未提交到 Git）中。
  这是组件验证，不是修复后的整模型运行。
- `eng/validation/validate-muse-spark-final-candidate.sh` 根据提供的已应用源码清单复现隔离构建、原生/组件门禁和 CPU arena 对比；
  它拒绝正在运行的完整服务器和复用的证据目录。仅调用这个脚本并不包含单独的广泛回归和后端回归结果。
  组件通过不能替代独立记录的运行时检查：已完成的五 GPU F16 活动、有运行时证据但存在未解决回答质量失败的打包 F32 调度，
  以及已通过的 n-gram 和被挤出前缀检查。
- 逐算子路径的 TP 实现请使用 [`eng/validation/validate-muse-spark-tp.py`](../../eng/validation/validate-muse-spark-tp.py)；
  原生 `--tp` 则由 [MuseSparkNativeTensorParallelTests.cs](../../InferenceWeb.Tests/MuseSparkNativeTensorParallelTests.cs)
  和 ctest `muse-spark-tp-split-plan` 覆盖。
  它运行已构建的 Linux 运行时，有显式的 CPU、rank 数、注意力、缓存和集合通信通道；`--list-lanes` 列出确切的清单和设备要求。
  默认包含所有必需通道；诊断通道是单独的，只跑子集不能确立整套通过。原生清单现在要求 32 个用例，
  包括缓存失败恢复和新的 F32 投影精度检查；这一要求并不表示新的设备检查已经通过。
  运行器要求源码清单和成功构建回执的构建时哈希覆盖整个托管运行时、所选的每个原生测试可执行文件和
  `native-build/libGgmlOps.so`，托管库与原生库必须匹配；它在每个通道前后核对这些哈希和未修改的上游 ggml，
  拒绝正在运行的模型负载，把失败输出保留在新的证据目录中，并拒绝跳过或缺失的必需用例。例如：

  ```bash
  python eng/validation/validate-muse-spark-tp.py \
    --root /work/TensorSharp --source-manifest /proof/source.json \
    --build-record /proof/build.json --runtime /proof/managed \
    --native-build /proof/native --output /proof/new-tp \
    --dotnet /opt/dotnet/dotnet --devices 0,1,2,3,4,5,6,7,8
  ```

  这些合成组件门禁不能替代检查点质量测试、服务机制证据或对齐的性能测量。
- 测试项目构建一个微型确定性模型，包含滑动层与全局层、GQA、路由修正、latent 专家、非均匀的 post-expert 缩放、
  残差门控和输出软上限。它的独立标量解码器检查每一个输出 logit、分块 prefill/decode、KV 恢复和图像 embedding 区间。
  这证明了所测的数值路径；它不衡量 816B 模型的语义质量或生产吞吐。
- `eng/validation/muse-spark-llama-reference.cpp` 链接未修改的 llama.cpp 构建，并按确切的 token ID 导出全部 logits；
  用 `eng/validation/compare-muse-spark-logits.py` 与托管实现和标量参照的导出进行比较。
  最初的 F32 CPU 对比覆盖 8 个位置 × 256 个 logits，参考分块大小为 1 和 5；
  TensorSharp 与参考之间的最大差值低于 `6.4e-7`，每个位置的贪心选择都一致。
- `eng/muse-spark-validate.py` 运行严格的 HTTP 夹具，覆盖短/长提示、JSON/schema/Unicode 输出、工具往返、
  有依赖的智能体工作流、图像布局、请求隔离、缓存 token 复用和交错到达；可选的 `image_chart` 场景还用确定性夹具
  检查两个柱状图排名和可见的四位 OCR 代码。它输出 JSON 和 Markdown，区分不可用的场景与通过，
  并在模型哈希、硬件/设置、请求覆盖或 token 数不一致，或尚未确认实际启动了引擎时拒绝给出持平结论。
  对话延迟使用同一客户端的单调墙钟；引擎内部计时器不同并不妨碍这一对比。引擎 decode 速率的对比需要统一的计时定义。
  来源清单与对应引擎的命令见其 `--help`。内部批处理、投机接受、CPU 卸载和多 GPU 执行需要额外的运行时证据；
  仅有 HTTP 重叠不能证明它们的实现。

生成的日志/报告应放在被忽略的 `docs/validation/muse-spark/` 或 `artifacts/` 中。
已记录的证据不能确立通用模型质量的持平。逐算子路径的等工作量 decode 目标未达到。
原生执行器的上述测量只覆盖一台机器和一种量化。
架构支持、合成数值测试和通过的有界夹具不得被当作通用质量或性能优势的证明。
