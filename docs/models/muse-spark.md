# Muse Spark

[← back to model index](README.md) | [中文](muse-spark_zh-cn.md)

TensorSharp recognizes the `muse-spark` GGUF architecture used by
[Unsloth's Muse-Spark-Open-816B-A42B release](https://huggingface.co/unsloth/Muse-Spark-Open-816B-A42B-GGUF).
The decoder, MoE integration and multimodal wiring live in TensorSharp-owned
code; the fetched ggml dependency remains unchanged.

| Property | Implementation |
|---|---|
| Architecture | `muse-spark`, alias `muse_spark` |
| Decoder | [MuseSparkModel](../../TensorSharp.Models/Models/MuseSpark/MuseSparkModel.cs); on `ggml_cuda`/`ggml_vulkan` the [native whole-model executor](#native-whole-model-executor) runs it by default, including `--tp N` on `ggml_cuda` |
| Expert path | Native executor: one `mul_mat_id` graph per layer inside the whole-model graph. Per-op path: [MuseSparkModel.Moe.cs](../../TensorSharp.Models/Models/MuseSpark/MuseSparkModel.Moe.cs), native `TSGgml_MuseSparkMoEFFNQuantF32` |
| Images | Matching Spark mmproj, shared Muse-Glimmer vision encoder |
| Audio | Not implemented; an audio tower in the projector does not enable audio inference |
| GPU placement | Native executor, up to 16 GPUs: `--layer-split N` assigns layers to N GPUs, places expert stacks by a byte-minimizing plan and [pipelines prefill](#how-it-runs) across the GPUs; `--tp N` on `ggml_cuda` splits every layer over N GPUs through ggml's meta backend, the mechanism behind llama.cpp's `-sm tensor` ([details](#tensor-parallelism-on-the-native-executor)). Per-op path (`TS_MUSE_SPARK_NATIVE=0`, and `--tp N` on `ggml_vulkan`): `--tp N` shards attention and partitions experts; `--layer-split N` assigns complete layers |
| Expert offload | Native executor (layer split; native `--tp` keeps every expert in VRAM): `TS_MUSE_SPARK_CPU_MOE=auto` keeps the expert stacks that do not fit VRAM on NUMA-placed host memory; `--n-cpu-moe N` forces the first N layers there. Per-op path: `--n-cpu-moe N` runs the first N layers' preserved native expert stacks on the CPU |
| KV cache | Native executor: F16 per-sequence slots, a ring for sliding layers and lazily grown rows for global layers. Per-op path: F32/F16 linear GQA caches; block-quantized KV is rejected |
| Continuous batching | Native executor: batched decode across sequence slots (text and image requests) and batched prefill of new text prompts. Per-op path: packed, paged text forward with F32 KV |
| Prefix cache | Native executor: conversation continuation, rewind of a finished conversation, and shared system-prompt checkpoints cloned into new conversations |
| Speculation | Generic n-gram drafter through `--spec --spec-type ngram`; no learned Muse Spark draft model is assumed. Not used under native `--tp` ([why](#tensor-parallelism-on-the-native-executor)) |

## Native whole-model executor

On `ggml_cuda` and `ggml_vulkan`, Muse Spark runs through a TensorSharp-owned
native executor,
[ggml_ops_muse_spark_exec.cpp](../../TensorSharp.GGML.Native/ggml_ops_muse_spark_exec.cpp),
with managed glue in
[MuseSparkModel.Native.cs](../../TensorSharp.Models/Models/MuseSpark/MuseSparkModel.Native.cs).
It loads the split GGUF itself and submits one GGML graph per micro-batch
across every GPU and the CPU. It uses the unchanged upstream ggml. The per-op
path described in the later sections remains the reference implementation:
`TS_MUSE_SPARK_NATIVE=0` selects it, and `TS_MUSE_SPARK_NATIVE=1` opts the GGML
CPU backend into the native executor. `--layer-split N` and, on `ggml_cuda`,
`--tp N` both run on this executor; see
[Tensor parallelism on the native executor](#tensor-parallelism-on-the-native-executor).
On `ggml_vulkan`, `--tp N` uses the
[per-op path](#tensor-parallelism-on-the-per-op-path).

### How it runs

- **One graph per micro-batch.** `ggml_backend_sched` splits each graph
  between the GPUs and the CPU backend. Graphs are cached by shape, so a decode
  step reuses its graph. Prefill uses 2048-token micro-batches
  (`TS_MUSE_SPARK_UBATCH`). Host expert cost per token falls from 0.305 ms to
  0.175 ms per layer between 512 and 2048 rows. When a graph does not fit, the
  executor halves the micro-batch and retries.
- **Weight placement.** `--layer-split N` spreads the 62 layers over N GPUs.
  Each layer's dense weights (attention, router, latent projections, norms)
  stay on its GPU. A dynamic program then places each layer's gate, up and
  down expert stacks independently: it keeps as many on the GPUs as fit and
  puts as few bytes as possible on the host. The expert bytes on the host set
  the decode cost. `TS_MUSE_SPARK_CPU_MOE=auto` enables this plan.
  `--n-cpu-moe N` instead forces the first N layers' experts to the host.
  Without either, a model that does not fit is refused with a message that
  names the option. Among plans that leave the fewest expert bytes on the
  host, the one that fills the GPUs most evenly wins. The startup log prints
  each GPU's layers and the host stacks (`6,7,...,18:gu,...,45:d`; `g`, `u`
  and `d` name single stacks).
- **Host experts.** With two or more NUMA nodes, host expert stacks are copied
  into anonymous memory by threads pinned to each node, and a strict, pinned
  CPU thread pool runs them. ggml's CPU `mul_mat_id` gives each worker a fixed
  row range, so every worker reads memory on its own node. On the 2-socket
  test host this raised host expert bandwidth from about 100 to 165–174 GB/s.
  `numactl --interleave` was unavailable because the container forbids
  `set_mempolicy`. `TS_MUSE_SPARK_HOST_NUMA=0` keeps the GGUF memory mapping
  instead. When two tokens of a batched decode step pick the same expert,
  ggml's kernel switches that expert to dynamically claimed chunks, so workers
  read the other socket's rows. Batches of 2–16 tokens therefore run host
  experts through a TensorSharp kernel (`host_moe_matmul`). It keeps the
  static row ownership and reads each weight row once for every token that
  selected it, using ggml's own quantization and dot product, so results are
  bit-identical. One layer with 4 tokens took 3.5–3.8 ms instead of 3.8–3.9 ms.
- **KV cache.** Each sequence slot owns F16 K/V rows on the GPU of each layer.
  The 46 sliding-window layers keep a ring of the 2048-token window plus the
  micro-batch and a 512-row rewind margin; each row carries its absolute
  position. The 16 global layers start at 4096 rows and double on demand.
  A rewind that the ring can no longer serve is refused, and the scheduler
  re-prefills instead.
- **Batching.** Decode steps of every running request share one graph with
  per-sequence attention masks. New text prompts that arrive together are
  prefilled in one graph. Image embeddings replace their placeholder rows
  inside the graph.
- **Pipelined prefill.** With a layer split, one prompt call is cut into
  equal micro-batches, and GPU k works on micro-batch j while GPU k+1 works
  on j−1, as llama.cpp's pipeline parallelism does. The call runs on one
  persistent ggml scheduler with four rotating input copies, cross-GPU events
  and a pinned staging ring. The last micro-batch is padded with copies of
  the final token; the padded rows are rewritten or unlabeled, so nothing
  reads them. Unless `TS_MUSE_SPARK_PIPELINE_UBATCH` fixes the size, each
  call's micro-batches are about a quarter of the call, between 512 rows and
  the prefill ubatch (`TS_MUSE_SPARK_UBATCH`, 2048), and shrink (down to 128
  rows) where a long context's masks would exceed a 96 MiB staging budget. On
  ten A40s one 15,641-token call took 15.3 s with fixed 512-row micro-batches
  and 10.2 s adaptive; one 4K call took 4.5 s and 3.7 s. The pipeline turns
  on by itself with two or more GPUs of the CUDA family (CUDA, ROCm, MUSA), no
  host experts, no `TS_MUSE_SPARK_PROFILE` and no tensor parallelism
  (`TS_MUSE_SPARK_PIPELINE`); the load log says
  `[muse-spark] pipelined prefill: on (...)` or `off (reason)`. A call that
  cannot pipeline logs `pipelined prefill declined (reason)` once per reason
  and runs one micro-batch at a time: masks over budget even at the smallest
  micro-batch, compute buffers that do not fit (not retried for that shape
  until a slot frees memory), no device memory for the call's K/V rows, no
  pinned host memory, or image rows in the call (image embeddings replace
  token rows inside the graph). The pipeline's scheduler buffers count as reclaimable
  in memory admission and are released before retained slots are evicted.
- **Lone-prompt steps.** When one request is alone in the server, the
  scheduler asks the model how large each prefill step should be
  (`ISoloPrefillBudgetModel`), and the answer may exceed
  `TS_SCHED_MAX_BATCHED_TOKENS`. With pipelined prefill on, Muse Spark answers
  8,192 tokens and joins a remainder of at most 2,048 to the step before it,
  so no short call fills and drains the pipeline alone; when the call would
  not pipeline it answers 0, which keeps the scheduler's own step. A step
  that would carry image rows also keeps the scheduler's own step. On ten A40s
  one 8K call keeps about 90% of a 16K call's throughput, and a request that
  arrives during a lone step waits for that step (about 6 s at 8K). Setting
  `TS_SCHED_SOLO_PREFILL_CHUNK` chooses the scheduler's step instead (bounded
  by `TS_SCHED_MAX_BATCHED_TOKENS`) and turns the model's steps off;
  `TS_SCHED_MAX_BATCHED_TOKENS` alone does not. The engine log says which
  applies.
- **Prefix cache.** A finished request's slot is retained and donated to the
  request that continues the conversation. It may be rewound first, so a
  regenerated answer reuses the prompt. The tokens before the chat layer's
  shared-prefix boundary (system prompt and tool declarations) are copied once
  into a checkpoint slot and cloned into each new conversation. Muse Spark's
  template opens every conversation with the same system header (knowledge
  cutoff, date, reasoning level), so that header is a shared prefix even
  without a system message. Under memory
  pressure, retained end states are freed before checkpoints. As in every
  TensorSharp model, unrelated conversations share only that public prefix,
  so an identical but separate request does not reuse another request's
  private tokens.
- **Memory admission.** A new slot or KV growth is admitted only while each
  device keeps `TS_MUSE_SPARK_SLOT_HEADROOM_MB` plus room for the largest
  graph seen so far, capped at `TS_MUSE_SPARK_ADMIT_GRAPH_MB`. Otherwise
  retained slots are freed first; cached graphs count as reclaimable. A
  running sequence's KV growth is never refused while the memory physically
  exists. Without the cap, a single 16K-token prefill's 1.2 GiB graph made
  later concurrent requests evict the shared-prompt checkpoint. A graph that
  does not fit frees retained slots, then runs that call in halved
  micro-batches. ggml-cuda's kernel-scratch pool sits outside this
  accounting: it grows on first use, never shrinks, and aborts the process
  when it cannot grow. Loading therefore runs one full-size prefill
  (`TS_MUSE_SPARK_WARM_POOLS`) so the pool reaches its peak first; without it,
  the first 4K-token prompt after a run of concurrent requests crashed the
  server.

### Tensor parallelism on the native executor

`--tp N` on `ggml_cuda` runs the native executor with every layer split over
the first N CUDA GPUs; select them with `CUDA_VISIBLE_DEVICES`. It is built on
upstream ggml's meta backend (`ggml_backend_meta_device`), the mechanism behind
llama.cpp's `-sm tensor`, with no ggml patch. TensorSharp supplies the split
plan in
[ggml_ops_muse_spark_tp.cpp](../../TensorSharp.GGML.Native/ggml_ops_muse_spark_tp.cpp)
(ctest `muse-spark-tp-split-plan`). `TS_MUSE_SPARK_NATIVE=0` keeps the
[per-op path](#tensor-parallelism-on-the-per-op-path) instead, which is also
what `--tp N` uses on `ggml_vulkan`.

- **How it splits.** Attention splits by KV group: each GPU holds its groups'
  q and gate rows, k and v rows and K/V cache columns, and the matching input
  columns of the attention output projection. Routed experts split along their
  hidden dimension (gate and up rows, down columns) in units of the down
  projection's quantization block rounded up to a multiple of 128 elements,
  256 for the UD-IQ3_XXS checkpoint, and the output head splits by vocabulary
  rows. Norms, the router and the latent
  projections are copied to every GPU. The shares follow llama.cpp's
  distribution and rotate per layer, so the 16 KV groups over ten GPUs give
  each GPU one or two. Each layer ends in two all-reduces.
- **Collective.** Before the backend creates its communicator, TensorSharp
  probes peer copies between the GPUs. Where peer access is advertised but
  does not work, as on the ten-A40 test host, NCCL runs with
  `NCCL_P2P_DISABLE=1` over its shared-memory transport. An explicit
  `GGML_CUDA_ALLREDUCE` wins. Without NCCL, ggml's own all-reduce covers two
  GPUs, and more GPUs use the meta backend's peer-copy butterfly, as llama.cpp
  built without NCCL does. ggml's NCCL path reduces
  tensors above 262,144 elements in BF16, which covers prefill's MoE partials
  but not decode.
- **Refusals.** These refuse the load (exit 2):
  `TS_MUSE_SPARK_LAYER_SPLIT`; `--n-cpu-moe N` (`TS_MUSE_SPARK_CPU_MOE=auto`
  keeps every expert in VRAM); `TS_MUSE_SPARK_PIPELINE=1`;
  `TS_MUSE_SPARK_PROFILE`; `TS_MUSE_SPARK_GRAPH_CACHE` above 1;
  `TS_MUSE_SPARK_FUSE_QKVG=1`; `TS_MUSE_SPARK_CPU_REPACK=1`;
  `TENSORSHARP_TP_DEVICES`, which this path does not read; N above the
  model's 16 KV heads, above 16 GPUs or above the visible GPUs; unlike GPUs;
  and a plan that leaves a GPU without a KV group or expert rows. A graph op
  the GPUs cannot run at some shape fails that call with a reason rather than
  aborting, because a tensor-parallel layer cannot fall back to the CPU. The
  per-op path's own settings (`TS_MUSE_SPARK_PHASE_TIMING`,
  `TS_MUSE_SPARK_FUSE_QK_PREP`, `TS_MUSE_SPARK_REUSE_NORM_SCRATCH`,
  `TS_MUSE_SPARK_KV_UPLOAD_2D`, `TS_MUSE_SPARK_FUSE_RESIDUALS`,
  `TS_MUSE_SPARK_TP_HOST_POOL`, `TS_MUSE_SPARK_PREFILL_CHUNK`,
  `TS_MUSE_SPARK_SCRATCH_RESERVE_MIB`,
  `TS_MUSE_SPARK_RELEASE_GPU_SOURCE_PAGES`) are named once in the log as
  ignored.
- **Memory.** Each GPU holds its slices, a copy of the mirrored tensors (about
  4.4 GB of latent projections) and the whole graph's compute buffer. The
  context is capped per GPU as for a layer split (see
  [Ten or more GPUs](#ten-or-more-gpus)). On ten A40s the UD-IQ3_XXS
  checkpoint took 35.4–36.0 GiB of weights and 0.6 GiB of K/V per GPU at a 32K
  context with five planned slots. K/V row growth and prefix-checkpoint copies
  run as one copy graph on the GPUs, without a host round trip.
- **What is off.** Pipelined prefill is off, since every GPU works on every
  layer. The graph cache keeps one graph, because the meta backend keeps
  per-GPU views only for its last graph; for the same reason speculative
  decoding is not used, since alternating draft-verify and decode graphs
  would rebuild every step.

[InferenceWeb.Tests/MuseSparkNativeTensorParallelTests.cs](../../InferenceWeb.Tests/MuseSparkNativeTensorParallelTests.cs)
checks 2, 4 and 10 GPUs against a double-precision reference decoder: a
prefill whose K/V rows grow while the sliding ring wraps, decode, a rewind,
batched decode over three slots and a prefix-checkpoint clone, plus the
refusals. The
[benchmark report](../perf/muse-spark-vs-llama-cpp.md) explains why prefill
under `--tp` is bound by the all-reduce transport on that host and why the
layer split is the faster mode for long prompts.

### Ten or more GPUs

- **Peer limit.** In an NCCL build of ggml (TensorSharp's CMake enables NCCL
  when it is installed), ggml-cuda grants each kernel-scratch pool chunk to
  every peer-capable GPU in one `cuMemSetAccess` call, and CUDA allows a GPU
  at most 8 peer connections. On ten A40s every `ggml_cuda` run with the ten
  GPUs visible aborted with `peer mapping resources exhausted`: `--tp 10`
  directly, and `--layer-split 10` once the executor's eight-GPU limit (below)
  no longer refused it first. A TensorSharp wrapper,
  [ggml_ops_cuda_vmm_peer_guard.cpp](../../TensorSharp.GGML.Native/ggml_ops_cuda_vmm_peer_guard.cpp)
  (linked with `-Wl,--wrap=cuMemSetAccess`, Linux only; ctest
  `cuda-vmm-peer-guard`), catches the refusal, revokes the partial grant,
  grants the owning GPU only and warns once (`[CUDA] GPU N cannot share its
  kernel-scratch memory with ...`). Tensor copies between GPUs do not use that
  pool. NCCL then gets `NCCL_P2P_DIRECT_DISABLE=1` before its first
  communicator, unless the operator set it or `NCCL_P2P_DISABLE` is on.
  llama.cpp `7170cad` has the same abort on such a host unless it is built
  with `-DGGML_CUDA_NCCL=OFF`.
- **GPU count.** The native executor spans up to 16 GPUs in either mode.
  Asking for more, or for more than are visible, refuses the load (exit 2)
  rather than running on fewer GPUs.
- **Context cap.** When `MAX_CONTEXT` (and `TS_MUSE_SPARK_PLAN_CTX_TOKENS`)
  is unset and the advertised context
  (1,048,576 tokens) does not fit, the context is capped to the largest size
  that fits, a multiple of the KV pad and at least 4,096 tokens, and the log
  says `[muse-spark] context capped to N tokens (the GGUF advertises 1048576)
  ...; set MAX_CONTEXT to choose a size`. With `MAX_CONTEXT` set, a context
  that does not fit refuses the load.

### Options

| Setting | Default | Effect |
|---|---|---|
| `TS_MUSE_SPARK_NATIVE` | on for CUDA/Vulkan, and for `--tp` on CUDA | `0` selects the per-op path (also for `--tp`); `1` enables the executor on the GGML CPU backend |
| `TS_MUSE_SPARK_CPU_MOE=auto` | off | Byte-minimizing per-stack expert placement (overrides `--n-cpu-moe`) |
| `TS_MUSE_SPARK_LAYER_SPLIT` | automatic | Layers per GPU for a layer split, as comma-separated non-negative counts summing to 62; parsed strictly, so a malformed list refuses the load. Refused with `--tp` |
| `TS_MUSE_SPARK_UBATCH` | 2048 | Prefill micro-batch rows |
| `TS_MUSE_SPARK_PIPELINE` | automatic | Pipelined prefill. Unset: on with two or more CUDA-family GPUs (CUDA, ROCm, MUSA), no host experts, `TS_MUSE_SPARK_PROFILE` unset and no `--tp`. `1` forces it (refused on a multi-GPU backend outside the CUDA family and under `--tp`); `0` turns it off; any other value refuses the load |
| `TS_MUSE_SPARK_PIPELINE_UBATCH` | adaptive | Fixed pipelined micro-batch rows, an integer from 1 to `TS_MUSE_SPARK_UBATCH`. Unset: about a quarter of each call, from 512 rows up to the prefill ubatch, shrinking to 128 where a long context's masks need it |
| `TS_MUSE_SPARK_PLAN_SLOTS` | 1 | Sliding-window rings reserved in the placement plan; raise for concurrent requests |
| `TS_MUSE_SPARK_PLAN_CTX_TOKENS` | context | Global-layer KV tokens reserved in the placement plan |
| `TS_MUSE_SPARK_VRAM_RESERVE_MB` | 2048 + ubatch/4 | Per-GPU room kept for compute buffers |
| `TS_MUSE_SPARK_DEV0_RESERVE_MB` | 0 | Extra room on GPU 0, e.g. for the vision encoder |
| `TS_MUSE_SPARK_SLOT_INIT_ROWS` | 4096 | Initial global-layer KV rows per slot |
| `TS_MUSE_SPARK_SLOT_HEADROOM_MB` | 256 | Free memory a new slot or KV growth must leave |
| `TS_MUSE_SPARK_ADMIT_GRAPH_MB` | 512 | Most graph memory a new slot or KV growth keeps free, on top of the headroom |
| `TS_MUSE_SPARK_SWA_REWIND` | 512 | Extra sliding-ring rows that make short rewinds possible |
| `TS_MUSE_SPARK_HOST_NUMA` | on with ≥2 nodes | NUMA-placed host experts and pinned workers |
| `TS_CPU_MOE_THREADS` | allowed CPUs; NUMA mode leaves about 1/6 free | Host expert worker count |
| `TS_MUSE_SPARK_HOST_MOE_TOKENS` | 16 | Batches of 2 to N tokens run host experts through the static-row kernel; `0` keeps ggml's `mul_mat_id` |
| `TS_MUSE_SPARK_CPU_REPACK` | off | ggml CPU repacking of host experts; it did not speed up decode here |
| `TS_MUSE_SPARK_OP_OFFLOAD` | on only without host experts | Let the scheduler copy host weights to a GPU for large batches |
| `TS_MUSE_SPARK_WARM_POOLS` | on with GPUs | Run one full-size prefill at load so ggml-cuda's kernel-scratch pools reach their peak before slots take the memory. With pipelined prefill on, also run a short pipelined prefill (two fixed or four of the smallest adaptive micro-batches) that builds the pipeline's scheduler and checks that it engages |
| `TS_MUSE_SPARK_EXEC_DEBUG` | off | Log placement, graph builds and memory decisions |
| `TS_MUSE_SPARK_PROFILE` | 0 | Print per-node timings of N graphs; turns pipelined prefill off and is refused under `--tp` |

### Launching

The measured profile uses ten A40 GPUs (46 GB each) and the 8-shard
UD-IQ3_XXS checkpoint (315.7 GiB of weights) with every expert in GPU memory.
Its 32K context and five planned slots leave room for four concurrent requests
and the shared-prefix checkpoint. The same command runs the layer split and,
with `--tp 10` in place of `--layer-split 10`, tensor parallelism:

```bash
MAX_CONTEXT=32768 TS_MUSE_SPARK_PLAN_SLOTS=5 TS_CODE_EXEC=0 \
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll \
  --model /workspace/Muse-Spark-Open-816B-A42B-GGUF/UD-IQ3_XXS/Muse-Spark-Open-816B-A42B-UD-IQ3_XXS-00001-of-00008.gguf \
  --backend ggml_cuda --layer-split 10 \
  --host 127.0.0.1 --port 8083 --no-webui --no-skills --no-multi-agent
```

The layer split puts about 36 GiB on each GPU, turns pipelined prefill on and
loaded in 6.0 minutes. Under `--tp 10` each GPU held 35.4–36.0 GiB of weights
and 0.6 GiB of K/V, and the load took 6.4 minutes. On that host both runs
rely on the [peer-limit guard](#ten-or-more-gpus).

### Measured results

Measured with the ten-A40 profile above against llama.cpp `7170cad` on the
same machine and checkpoint: driver 595.91, CUDA 12.8, unchanged upstream ggml
`da8e2a5`, and llama.cpp built with `-DGGML_CUDA_NCCL=OFF` because its NCCL
build hits the [peer-limit abort](#ten-or-more-gpus) on that host. Peer access
between the GPUs is advertised but does not work there, so neither engine used
P2P. Throughput and latency figures are medians of three repeats, except
llama.cpp tensor-split rows, which use two; each fixture ran once. Ratios above 1.00x mean
TensorSharp is better. For cells marked "see report", see the
[full report](../perf/muse-spark-vs-llama-cpp.md), which also has the method,
the per-scenario detail and the limits of the comparison.

| Scenario | TensorSharp `--layer-split 10` | llama.cpp `-sm layer` | Ratio | TensorSharp `--tp 10` | llama.cpp `-sm tensor` | Ratio |
|---|---:|---:|---:|---:|---:|---:|
| Decode, one request (tok/s) | 16.69 | 16.79 | 0.99x | 22.63 | 16.19 | 1.40x |
| Time to first token, short prompt (s) | 0.64 | 0.75 | 1.17x | 0.58 | 1.29 | 2.22x |
| Prefill, 4K-token prompt (tok/s) | 999.7 | 483.2 | 2.07x | 294.4 | 75.9 | 3.88x |
| Prefill, 16K-token prompt (tok/s) | 1346.8 | 826.1 | 1.63x | 295.2 | see report | |
| Four concurrent requests, total output (tok/s) | 33.65 | 33.04 | 1.02x | 45.73 | see report | |
| Second turn of a 2.3K-token conversation, first token (s) | 0.45 | 0.91 | 2.00x | 0.46 | see report | |
| New conversation sharing a 2.3K-token system prompt, first token (s) | 0.48 | 0.64 | 1.34x | 0.51 | see report | |
| Answer-checking fixtures (`eng/muse-spark-validate.py`) | 13 / 14 | 12 / 14 | | 14 / 14 | see report | |
| Load time | 6.0 min | 14.4 min | 2.4x | 6.4 min | 14.7 min | 2.3x |

The layer split is the mode for long prompts: pipelined prefill keeps all ten
GPUs busy on one prompt. Its single-request decode is level with llama.cpp's
(0.99x, within run-to-run noise), since one token passes through the ten GPUs
one after another on both engines. Tensor parallelism is the mode for
generation: it decodes 1.40x as fast as llama.cpp's tensor split and 1.35x as
fast as either engine's layer split. Its prefill stays near 295 tok/s because
the all-reduces run over NCCL's shared-memory transport on that host, and
Muse Spark normalizes each expert's output before summing, so the MoE
all-reduce carries every selected expert's output.

On teacher-forced wikitext-2 (identical token IDs, 12,276 scored positions),
the layer split's next-token distributions differ from llama.cpp's by a mean
KL divergence of 0.0026, below the 0.0028 that llama.cpp shows against itself
when only its micro-batch size changes; prefixes written by pipelined prefill
give the same result. `--tp 10` sits at that noise floor: its mean, 0.0031,
is 9% above it and its maximum is higher, while its p99 and top-1 agreement
are better; ggml's NCCL path sums its prefill MoE partials in BF16. The layer
split's one fixture failure is the staggered-arrivals case: a request ended
with a coherent answer and an end-of-text token after 322 of its 512 tokens,
a greedy-decoding branch rather than an engine fault, and llama.cpp's layer
split failed the same case. Before the work described above, TensorSharp
could not run either command on this host, and its earlier `--tp` path decoded
at 1.9–2.4 tok/s.

## Tensor parallelism on the per-op path

This section describes the managed per-op path's tensor parallelism, which
`--tp N` uses when `TS_MUSE_SPARK_NATIVE=0` is set or the backend is
`ggml_vulkan`. On `ggml_cuda`, `--tp N` runs the
[native executor](#tensor-parallelism-on-the-native-executor) instead.

`--tp N` splits Q/K/V/gate projections by complete GQA head groups and splits
the attention output projection by input columns. Each GPU computes its heads
concurrently, then the partial output projections are summed before the
post-attention normalization. Routed experts are partitioned whole across the
same GPUs. Each expert's complete latent output is normalized before routing
weights and the cross-GPU sum are applied. The router, latent down/up
projections, residual stream and output head execute on rank 0.

This supports uneven head/expert partitions, including nine GPUs for the
published 128-query/16-KV-head checkpoint. Each rank must own at least one KV
group and expert, and quantized attention output cuts must preserve complete
quantization blocks. Unsupported geometry and multi-node TP are refused.
`--n-cpu-moe N` still evaluates the selected prefix of preserved native expert
stacks once on the host; it does not duplicate those expert computations across
ranks. This covers the published quantized checkpoint. F32 expert tensors that
use the ordinary tensor-operation fallback remain on the selected backend
(rank 0 in TP), even when they lie in that prefix. Capacity checks count those
F32 weights as GPU weights.

The GGML per-operation path returns partial results to host memory. Small
reductions run there; larger reductions can use the backend transport.
Spark requests F32-preserving reduction, including TensorSharp's own F32 NCCL
transport or exact fallback, so prefill collectives do not silently narrow
partials to BF16. PCIe topology and rank-0 work can limit scaling. TP therefore
requires measurement on the intended hardware; it is not a promise of linear
speedup or lower aggregate model memory.

The TP path retains separate Q/K/V/gate weight projections. The optional
`TS_MUSE_SPARK_FUSE_QK_PREP=1` packs their initialized host outputs and performs
Q/K normalization and optional RoPE together on each CUDA rank, using that
rank's actual head counts. It adds a host packing cost and is off by default;
its TP performance requires a controlled comparison. Norm scratch reuse and
F16 2D KV uploads remain available. Successful
optimization-use records from a layer-split run do not establish their use in
a TP run.

Spark's CUDA projections for F32 weights explicitly preserve F32 operands,
including uneven attention shards, expert planes and the output head. This
scoped path reuses a backend wrapper per rank; quantized weights and other
models retain their existing matmul policies. It addresses the observed
shape-dependent TF32 arithmetic difference between full and uneven-shard
projections. Device regression and full-model validation of this correction
are pending; the earlier failed numerical checks remain recorded.

Spark's CUDA F32 KV paths also use TensorSharp-owned attention that preserves
F32 Q/K/V operands. Upstream CUDA flash attention can narrow F32 K/V to F16;
the published head geometry exposed this difference in packed scalar checks.
Ordinary and paged Spark calls now require an explicit owned-CUDA success
status and report separate successful-use counters. This is not upstream
flash execution. The F16 path and other models' attention policies are unchanged.
The strict packed scalar bounds remain unchanged; device validation of this
correction is pending.

Capacity checks run from GGUF descriptors before payload loading. TP checks
the actual uneven shards plus rank-0 weights, while layer split checks complete
layer assignments. The remaining memory estimate is workload-dependent; see
the startup guidance below. Both modes preserve F32/F16 linear KV snapshots
and speculative rewind; packed continuous batching requires F32 KV.

## Previous five-GPU layer-split measurements

These measurements predate the native executor: they used the per-op path,
which `TS_MUSE_SPARK_NATIVE=0` still selects. The completed v4r2 F16 campaign on the Q3 checkpoint passed 13 selected quality
fixtures, a separate thinking check and all nine timing-case content checks.
The quality fixtures cover short English/Chinese answers, JSON/schema/Unicode,
tool and agent workflows, images/chart OCR, conversation memory and parallel
request isolation. They comprise 22 measured turns plus a warmup, not a broad
model-quality evaluation. Thinking captured a separate reasoning stream as well
as the expected final answer. All 35 requests across the three phases, including
their excluded warmups, reported integer zero cached tokens.

**The requested performance target is not met.** In the equal-work 512-token
decode comparison, TensorSharp's median client wall time was **111.743 seconds**
versus **80.631 seconds** for llama.cpp. The reference/candidate median wall ratio
is **0.721574**, below the frozen acceptance threshold of **0.95** (a 5%
speed-ratio tolerance). These are three repeats on one hardware/profile
combination, not a statistical equivalence claim.

| Scenario, three repeats | TensorSharp median wall seconds | llama.cpp median wall seconds | Prompt tokens; output TensorSharp/reference | Interpretation |
|---|---:|---:|---|---|
| Short answer | 6.598 | 13.738 | 156; 1/2 | Descriptive task latency only: raw output work differs |
| Fixed decode | 111.743 | 80.631 | 171; 512/512 | Equal-work performance target failed |
| Long retrieval | 294.907 | 441.550 | 7812; 31/32 | Descriptive task latency only: raw output work differs |

Raw completion counts are preserved, including different EOS accounting.
The overall strict comparison is inconclusive because short/long output counts
differ; that does not erase the eligible decode failure. Both reference
long-request KV-space retries remain in its timings. Model load, explicit CPU
expert-page preparation and warmups are excluded. Both measured performance
phases had zero server physical-read bytes and major faults. F16 used the
ordinary sequence path, not packed F32 batching.

The separate F32 packed-batching campaign has an **unresolved quality failure**:
the staggered long response returned drafting commentary instead of a hash-table
explanation and stopped at 189 tokens, failing its exact 512-token requirement.
An exact serial replay on the same F32 process also returned a non-answer,
stopping at 179 tokens. Concurrency alone therefore does not explain the failure;
these observations do not isolate precision, a kernel, or an optimization as
its cause. Saved traces show the actual end-of-turn token after the planning
prose. The serial SSE stream contains content deltas and a normal stop, with no
hidden final answer or separate reasoning delta. The successful F16 decode request also differs in its validation tag
and prompt-token count, so it is not a controlled F16/F32 causal comparison.

Fresh scheduler records prove actual `BatchedPaged` execution: three copy
requests were served together, and staggered admission produced five mixed
prefill/decode steps with up to four served requests. Plain/parallel copies and
all three late JSON responses passed. Three separate no-speculation regression
fixtures also passed: edited copy, a following exact answer and an exact output
budget. These successes do not regrade the failed staggered case. The original
nine requests plus the serial replay and three regression fixtures all reported
zero cached tokens. An explicit failed-campaign cleanup preserves failed quality.
The independently audited four-request n-gram continuation passed output parity
against the unchanged successful no-speculation baselines. It recorded 138
accepted tokens from 225 drafts, 34 verification steps and 17 rollbacks. The
edited-copy case exercised 15 rollbacks; the following request returned the
expected one-token answer, and the final budget case stopped at exactly 17
tokens with `finish_reason=length`. This used `TS_SPEC_ADAPTIVE=0` and a separate
F32 mechanism configuration. It establishes those mechanism checks, not an
isolated speculation speedup or a repair of the failed staggered answer.
The separate F16 displaced-prefix probe passed independent stopped-campaign
audit. All four requests, including the warmup, returned exact answers. The first
request had 222 prompt/5 output tokens and no reuse, an unrelated displacement
had 156/1 and no reuse, and the continuation had 248/16 with 224 cached tokens.
Fresh telemetry confirmed restoration from radix pages and a first prefill
position of 224, leaving 24 new prompt tokens. The restored prefix exceeds the
displacement request's 157 total tokens. This proves the tested cross-request
cache reuse, without claiming a cache speedup or changing the failed F32 result.

The reference passed 12 of the 13 bounded quality fixtures; its adversarial
schema response was literal `OVERRIDE` and failed. Cooperative JSON output is
not evidence of enforced schema grammar. Neither engine's fixture count
establishes general model-quality parity or superiority.

The generated evidence includes the F16 campaign and unchanged paired analysis
(`docs/validation/muse-spark/q3-final-v4r2-performance-complete/independent-audit.json`),
the resource and actual-use summary
(`docs/validation/muse-spark/q3-final-v4r2-performance-complete/completed-campaign-summary.json`),
the failed F32 campaign, serial replay and cleanup
(`docs/validation/muse-spark/q3-f32-nospec-failed-cleanup-v4r2-complete/SUMMARY.md`),
the independent n-gram continuation
(`docs/validation/muse-spark/recovery-ngram-offline-audit.json`), and
displaced-prefix reuse
(`docs/validation/muse-spark/recovery-prefix-offline-audit.json`) (local
validation evidence, not committed). These ignored reports are produced by
validation and may be absent from a fresh source checkout. The recorded source manifest is
`f8615e93fb90286076ce21e4db15c516735d722b457fb562beaaa7a46f940f56`;
the native library SHA256 is
`9e7e7153a65a3fdf8ef4499a39c8c9a46a15d2a8b83c313dcdd1c19684fb8e71`.
A later, independently validated repair clears paged-attention sessions owned
by the calling thread before backend teardown/recreation. It is not part of
these full-model benchmark binaries. Those fresh-process model runs did not
recreate their backend, so this lifecycle finding does not explain the F32
non-answer. Workers on other threads must still drain before backend teardown;
the repair does not claim to clear every thread's cache. See the isolated
lifecycle audit
(`docs/validation/muse-spark/lifecycle-completed-review/accepted-audit.json`)
and the final applied-source receipt
(`docs/validation/muse-spark/final-applied-source/receipt.json`) (local
validation evidence, not committed).
Full-model inference and timing were not rerun on the repaired library.
Paged-attention sessions have since moved to a process-wide registry
(`ThreadCacheRegistry` in `ggml_ops_internal.h`): a model unload and
`TSGgml_Shutdown` free every thread's sessions, and a thread that exits hands
its sessions over instead of freeing them during its exit.

## Launching the sharded model

Keep every shard of the selected quantization in one directory and pass any
shard (`UD-Q4_K_XL` has twelve; `UD-Q3_K_XL` has ten; `UD-IQ3_XXS` has
eight). TensorSharp resolves
shard 1 for the architecture, tokenizer and model settings, then loads the
tensors from the complete set. Later shards contain only split metadata;
they are not standalone models.
The following Q3 configuration is the explicit F16 evaluation profile for five
A40 GPUs, each with 46,068 MiB total memory and 45,490 MiB initially usable,
a 233.762 GiB container memory limit and a CPU quota of
40.8 cores. It places the first 32 expert layers on the CPU (182 GiB of expert
weights), uses 32 CPU worker threads and assigns decoder layers as
`35,7,7,7,6`. The shared KV pool holds 16,384 tokens total, with at most four
active requests and a 16,384-token ceiling per request. It does not reserve
four simultaneous 16K contexts. The six optimization switches below are opt-ins
for this profile; their implementation defaults remain off. Prefix caching and
speculation are disabled to control the matched performance workload. The
completed F16 results above use this profile; they do not establish general
quality parity, and the measured equal-work decode target failed.

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

This command is a configuration example. Reproducible campaigns launch the
exact tested server DLL and native library from an isolated build and retain
their hashes; rebuilding with `dotnet run` does not reproduce that binary
identity. F16 can serve overlapping HTTP requests through the ordinary sequence
scheduler. That does not execute the packed F32 `ForwardBatch` path, and request
overlap alone is not proof of packed continuous batching.

For matched evaluation, the HTTP harness also sends `skills: []`,
`skills_discovery: false` and `multi_agent: false` to TensorSharp, while retaining
caller-declared tools. These request settings prevent extra server-provided
skill/delegation instructions from changing the prompt; they do not change the
decoder. No-cache measurements additionally send `cache_prompt: false`, disable
prefix caching at server launch as above, and verify the observed reuse count.

Use `--kv-cache-dtype f32 --continuous-batching` when evaluating the packed
paged text path, and reduce all four batch/prefill row limits above from 256 to
128 for the bounded F32 validation profile. That mechanism-test profile also
sets `MAX_CONTEXT=8192` and `TS_SCHED_NUM_BLOCKS=512`: an 8192-token ceiling and
8192-token shared pool leave more space for compute scratch on this machine;
they do not provide four simultaneous 8K contexts. Explicitly set
`TS_MUSE_SPARK_KV_UPLOAD_2D=0`, because that option applies to the ordinary F16
attention path. Keep streaming disabled for the base profile. Packed batches
require F32 KV and text-only requests; pending image embeddings select the
ordinary path. This exact F32 mechanism profile produced the packed scheduler
records and the unresolved staggered/serial quality failures described above.
It must not be treated as a validated replacement for the F16 profile. The separate n-gram continuation passed its bounded mechanism checks;
the separate F16 displaced-prefix probe also passed independent acceptance.

F32 KV uses twice the storage of F16. Keep each comparison's cache precision,
context and scheduler configuration identical across engines. For the F16
profile above, llama.cpp's `--ctx-size 16384 --kv-unified
--kv-unified-per-slot 16384 --parallel 4` preserves the same aggregate capacity
and request ceiling. Its whole-layer split includes the output projection, so
the corresponding settings are `--split-mode layer --tensor-split 35,7,7,7,7`.
These settings describe the previous whole-layer comparison. TensorSharp's
separate `--tp N` mode performs the attention sharding described above.

Use the process/container memory limit when budgeting host experts. The Q4
profile with 42 offloaded layers needs approximately 334 GiB for those expert
weights alone; it cannot keep them resident in this 233.762 GiB container even
if the host reports 504 GiB RAM. Its full-model attempt was stopped before HTTP
readiness after severe paging and is not a passing inference test.
`eng/validation/sample-inference-hardware.py` records cgroup limits separately
from host capacity. `eng/validation/muse-spark-launch-plan.py` estimates placement
from complete shard headers and explicit KV, projector, runtime and streaming
reserves; an estimated fit still requires measured memory peaks.

For whole-layer placement on GGML CUDA/Vulkan, TensorSharp now checks the
GGUF tensor descriptors against each selected device's current free memory
before reading weight payloads. A refusal distinguishes weights that already
exceed VRAM from insufficient estimated compute/active-KV headroom. It reports
a higher CPU-offload count when one fits the same header-based estimate;
settings are never changed automatically. Host memory must also accommodate
those offloaded experts.

The per-op path's tensor parallelism also plans from descriptors before
loading weights. It keeps
uniform whole-expert counts when they fit every measured device. Otherwise it
assigns contiguous expert ranges to balance remaining memory after fixed
attention/trunk weights and the same reserve; rank 0 can own fewer experts to
accommodate its trunk. The preflight and loader use one immutable plan, and the
startup log records its counts and policy. If any device's free memory is
unavailable, it falls back to uniform counts and reports which checks were
unavailable. This never changes `--n-cpu-moe`. It does not reserve the vision
tower or prove that the maximum configured context will fit.

On the per-op path, Muse Spark CUDA TP contexts retain up to eight reusable
host scratch blocks per rank, capped at 128 blocks of at most 8 MiB each (1 GiB total), allocated on
demand. Single-device, layer-placement and other-model pools keep their existing
policy. `TS_MUSE_SPARK_TP_HOST_POOL=0` selects the original eight-block pool for
a controlled comparison. With `TS_MUSE_SPARK_PHASE_TIMING=1`, forward reports
include immutable before/after host-pool counters alongside phase timings.
These counters measure host allocation/reuse, not device memory or kernel time.
The increased retention is bounded; a whole-model performance benefit remains
unmeasured.

The headroom estimate is 2560 MiB per GPU for this full model at up to 256
prefill rows, scales with model geometry and larger chunks, and excludes the
vision tower and additional retained requests. It is not a fit guarantee.
`TS_MUSE_SPARK_SCRATCH_RESERVE_MIB` explicitly overrides that estimate; lowering
it cannot bypass the weight-only check. Keep `MAX_CONTEXT` and
`TS_KV_INITIAL_TOKENS` explicit as in the example: the model's advertised
context limit is not evidence that a machine can hold that much KV memory.
GGML linear KV tensors are host-backed and grow lazily unless the configured
initial allocation reserves more; the check reports their initial and
maximum host sizes rather than charging the full advertised context to every
GPU.

For images, pass the matching Spark release's `--mmproj` as shown above.
Its projector type is `muse-glimmer`, but its output width must match the Spark
decoder (8192). A Muse-Glimmer-30B projector has a different output width and is
rejected. OpenAI-compatible requests can include `image_url` content parts.
Image spans are retained across internal prefill chunks, injected before the
embedding RMSNorm, and consumed through the normal attention path.

Use `--spec --spec-type ngram --spec-draft 8` to request weight-free speculation.
This replaces `--no-spec` in the example. A bounded F32 mechanism probe may set
`TS_SPEC_ADAPTIVE=0` to keep the adaptive governor from disabling speculation;
that is a test configuration and does not validate the default adaptive policy.
Acceptance depends on repeated text in the prompt/output. Verify actual draft
proposals and accepted tokens before claiming that speculation accelerated a
workload; an enabled flag alone does not establish a speedup.

On the per-op path, `TS_MUSE_SPARK_LAYER_SPLIT` optionally supplies a positive
layer count for each GPU, with counts summing to 62 (the native executor also
accepts zero; see [Options](#options)). Without it, TensorSharp balances contiguous
layer groups by resident weight bytes, accounting for CPU expert offload and
the embedding/output matrices. KV cache, image encoder and compute scratch
need additional memory. Layer placement executes successive layer groups on
different devices; it does not shard one matrix multiplication across GPUs.

## Architecture and numerical ordering

The released metadata declares 62 blocks, hidden width 8192, 128 query heads,
16 KV heads, head dimension 64, 256 experts with top-8 routing, and expert
latent/intermediate widths of 4096. Its maximum advertised context is
1,048,576 tokens; that metadata value is not evidence that the current
implementation or a particular machine has been validated at that context.
The configured operational context should fit the actual memory budget.

Each block follows this order:

1. Pre-attention RMSNorm, Q/K/V and attention-gate projections, per-head Q/K
   RMSNorm. Apply adjacent-pair RoPE only on layers marked sliding in the GGUF
   boolean pattern; global layers use NoPE.
2. GQA causal attention with a 2048-token window on sliding layers. Normalize
   the attention output per head, multiply by the sigmoid gate, project back
   to the residual width, and apply the unweighted post-attention RMSNorm.
3. Combine the residual and branch using the GGUF's per-channel alpha/beta
   weights. These coefficients are already converted from the source gates.
4. Normalize the FFN input. The router reads this full-width stream while a
   separate projection and norm move expert inputs into the latent space.
5. Select experts using `sigmoid(router_logits) + correction_bias`. The
   correction bias affects selection only. Normalize the selected unbiased
   sigmoid scores, with the same lower bound as the reference implementation,
   and apply the configured routing scale.
6. Run each selected expert's SwiGLU projections. Multiply each expert's output
   by the post-expert scale and RMS-normalize it **before** multiplying by its
   routing weight and summing experts. Moving this norm after the sum changes
   the model.
7. Project out of the latent space, multiply by the FFN post-norm scale, apply
   RMSNorm, and combine with the FFN residual using its alpha/beta weights.

Input token/image embeddings receive an unweighted RMSNorm. The final stream
receives a learned RMSNorm and output projection, followed by logit scaling
(`0.17677669` for the release) and tanh softcapping at 20. Block post-norms use
the separate GGUF post-norm epsilon. Delta norm weights and Q MetaP scaling
are already folded into the GGUF tensors and must not be applied again.

## Optimizations and current limits

The native expert graph batches selected experts with `mul_mat_id` and performs
the order-sensitive expert norm, routing multiply and reduction in one graph.
Offloaded preserved native expert stacks run on the CPU by default; the F32
tensor fallback limitation described above still applies. The experimental
`TS_MUSE_SPARK_STREAM_EXPERTS=1` option lets prefills of at least
`TS_HOST_MOE_DEVICE_MIN_BATCH` rows (default 128) stream selected experts to
the layer's GPU. It supports layer splitting and checks the measured full
expert-stack graph allocation against actual free VRAM plus its reusable
buffer before choosing the GPU. A `[muse-spark-stream]` diagnostic records the
decision and memory sizes; insufficient headroom keeps the work on the CPU.
Synthetic native tests cover the streaming mechanism, but full-checkpoint
streamed quality and performance remain unestablished. The F16 profile above
keeps this option disabled. The managed fallback groups routed tokens by expert
to batch GEMMs.

`TS_MUSE_SPARK_REUSE_CPU_MOE_SCRATCH=1` opts into retaining one CPU graph arena
for Spark's offloaded experts across layers and chunks. Set it in the process
environment before loading the native library. Caller-owned weights and
activations remain externally bound; each call uploads fresh routes, biases
and norm weights. The arena is protected by the host compute mutex and freed
before backend teardown. Other models keep their existing allocation path.
A first-success diagnostic reports actual allocated bytes; the native
`TSGgml_GetMuseSparkCpuMoeScratchState` query reports retained bytes and successful
arena dispatches. This option is off by default. Completed CPU/CUDA component
tests and an actual Q3 layer-0 CPU comparison with synthetic activations/routes
check its numerical behavior; the latter is not a full-decoder throughput test.
Actual use was also recorded in the combined F16 campaign; its result does not
isolate this option's end-to-end contribution.

On Linux CUDA, `TS_MUSE_SPARK_RELEASE_GPU_SOURCE_PAGES=1` opts into advising
away source-file pages after a successful upload and an exact resident-device
copy check. It retains the GGUF mappings, host pointers and cache keys for
fallback or re-upload. CPU/streamed experts, CPU embedding gathers and copied
fusion buffers are excluded. The advice covers only complete pages within
each tensor, using the original shard handle. Diagnostics count newly advised
weights and whole-page bytes once per weight; advice is not a guarantee of
physical eviction. This option remains off by default. The F16 campaign recorded
277 weights and 193,806,282,752 bytes cumulatively advised, along with measured
memory and I/O evidence. Those observations do not prove that every advised page
was evicted or isolate a throughput gain from this option.

`TS_MUSE_SPARK_FUSE_RESIDUALS=1` opts into a single GGML graph for each
post-normalization gated residual on CPU and CUDA, including packed batches.
The FFN channel scale stays before normalization; alpha and beta stay on the
two residual branches. A diagnostic confirms the first successful dispatch.
The graph now retains the normalized branch and beta-scaled branch as separate
stored-F32 intermediates, preventing CUDA expression fusion from changing
their rounding boundaries. New same-input comparisons against separate native
operations are required; device validation of this correction is pending.
Other backends retain the original operations. This option remains off by
default. The completed combined F16 profile recorded successful use; it is not
an isolated residual-fusion speed comparison.

`TS_MUSE_SPARK_FUSE_QK_PREP=1` combines per-head Q/K normalization and the
sliding layers' RoPE in one GGML graph on CPU and CUDA. The ordinary path uses
the fused Q/K/V/gate projection; CUDA TP packs its separate projection outputs
without changing weight ownership. Other backends retain their existing path.
Global layers preserve NoPE. Q/K preparation and fused residual
graphs reuse device-local scratch with fresh inputs on every call. A diagnostic
confirms the first successful Q/K dispatch. This option is off by default.
Successful use was recorded in the completed combined F16 campaign; a separate
controlled comparison would be needed to attribute a performance gain to it.

`TS_MUSE_SPARK_KV_UPLOAD_2D=1` uses two strided GGML transfers for Spark's F16
K/V attention window, replacing separate transfers for each KV head. It keeps
the same host-authoritative cache, transferred bytes, window offsets and cache
lifecycle. GGML CPU and CUDA support this option; other models and the paged
F32 path retain their existing transfers. A diagnostic confirms the first
successful dispatch. This option is off by default. Completed component checks
include CPU, CUDA and two-device cases, plus tiny-model chunking, prefix
restoration and speculative rollback. Actual F16 use and whole-campaign timing
are recorded above, without an isolated performance claim for this option.

`TS_MUSE_SPARK_REUSE_NORM_SCRATCH=1` opts into Spark-only RMS normalization
scratch reuse on GGML CPU and CUDA, and combines attention-output per-head
normalization with its sigmoid gate. The learned scales remain after RMSNorm;
other backends and the default configuration retain the original operations.
The shared arena follows the existing serialized compute contract, and all
results are copied back before the next operation can reuse its storage.
Successful reuse and successful head norm/gate execution have separate first-use
diagnostics. If reuse allocation fails, a fresh allocation is used and logged
as fallback, not as executed reuse. This option is off by default. Completed
CPU, CUDA and two-device numerical/lifecycle tests and tiny-model equivalence
checks validate the tested paths, including backend recreation from rank 1.
The combined F16 campaign also recorded both successful-use diagnostics and
passed its bounded quality fixtures. Neither those fixtures nor the component
checks establish general quality parity or an isolated normalization speedup.

Across the six enabled optimization options, the completed F16 log contains
seven successful-use records: Q/K preparation, residual fusion, F16 2D KV upload,
CPU expert scratch, source-page advice, weighted RMS scratch and head norm/gate
scratch. The last two belong to one option. Streaming is separately disabled;
it is not a seventh enabled optimization. A first-use record establishes at
least one successful use, not dispatch frequency or an isolated speedup.

Attention uses fused GGML attention operations; F16 and paged sliding layers
submit only the live key window. Prefill chunks bound activation memory. Prefix snapshots
and speculative rollback preserve the accepted linear KV prefix. The model
does not yet have a captured whole-decoder CUDA graph. Host router transfers,
per-operation dispatch and the host-backed paged path remain performance costs.

## Reproducing validation

The completed frozen v4r2 component run used unchanged upstream ggml revision
`a05c9922613c6a2230dbb1d0de679519ecafb4f0` and the same isolated native/managed
binaries across its component, microbenchmark and regression checks:

| Check | Observed result | Scope |
|---|---|---|
| Native CTest suite | 24 passed, zero failed/skipped | CPU, CUDA and two-device cases, including norm/gate teardown and recreation |
| Selected managed CPU lane | 95 passed with CPU expert arena off; the same 95 passed with it on | Component/tiny-model checks; actual opt-in execution is asserted |
| Selected managed CUDA lane | 23 passed with CPU expert arena off; the same 23 passed with it on | Component/tiny-model checks including two-device cases; zero skips |
| Broad managed regression | 1,524 passed, zero failed, seven explicitly skipped | Six unavailable model/tokenizer fixtures and one Windows-only platform test are not passes |
| Backend/CLI/warmup supplement | 25 cases passed in each CPU/CUDA lane, zero failed/skipped | 25 distinct cases executed twice, not 50 distinct tests |

All four selected managed lanes had zero failures/skips. They contain 118
distinct result names and 236 executions; do not count repeated arena modes as
additional distinct tests or add overlapping component and broad suites into
one coverage total. The Q3 expert
microbenchmark compared arena off/on in separate processes in both orders for
layer 0, 32 threads and 1/128/256 rows. Output hashes matched exactly and measured
repeats had zero process read bytes and major faults. It used synthetic
activations/routes and does not establish whole-model or llama.cpp performance.

The later isolated lifecycle proof reproduced retained sessions in the original
CPU/CUDA/two-GPU lanes, then verified cleanup after three recreations and three
shutdowns per fixed lane. Fixed CPU and two-GPU lanes each passed 15 numerical
graphs and 60 corruption controls; the CUDA lane passed 9 graphs and 36 controls.
The broader fixed-library suite passed 27 native cases and fresh managed lanes
of CPU Muse 95, CPU backend/lifecycle 25, CUDA Muse 23 and CUDA backend/lifecycle
25, with no failures or skips. These repeat existing component cases and are
not added to the earlier coverage totals. The original absolute-error failure,
bounded CUDA rounding diagnosis and console-only loader-audit failure remain
preserved in the isolated proof,
`docs/validation/muse-spark/lifecycle-completed-review/accepted-audit.json`
(local validation evidence, not committed).
The native libraries used a scoped partial relink; continuations rebuilt only
the fixture and reused those exact libraries. This was component validation,
not a post-repair full-model run.
That lifecycle proof's validation tool required all 27 native cases, including
the three lifecycle lanes; 27 local Python guard tests separately verified that
selection. Those tooling tests are not additional device tests.

`eng/validation/validate-muse-spark-final-candidate.sh` reproduces the isolated
build, native/component gates and CPU arena comparison from a supplied applied
source manifest. It refuses active full servers and reused evidence directories.
Use its matching source revision and preserve its raw logs and receipts; the
separate broad and backend regression results are not implied by invoking this
script alone. The separate completed F16 campaign covers this checkpoint on
five GPUs, its selected quality fixtures and matched timing cases. Actual packed
F32 scheduling has runtime evidence but an unresolved answer-quality failure.
The separately continued n-gram full-model checks passed, including verified
acceptance and rollback. The separate F16 displaced-prefix probe also passed.
Component passes cannot substitute for these independently recorded runtime
checks.

For the per-op path's TP implementation, use
[`eng/validation/validate-muse-spark-tp.py`](../../eng/validation/validate-muse-spark-tp.py).
Native `--tp` is covered by
[MuseSparkNativeTensorParallelTests.cs](../../InferenceWeb.Tests/MuseSparkNativeTensorParallelTests.cs)
and the `muse-spark-tp-split-plan` ctest instead.
It runs an already-built Linux runtime, with explicit CPU, rank-count,
attention, cache and collective lanes. `--list-lanes` lists the exact inventory
and device requirements. The default includes every required lane; diagnostic
lanes are separate and a subset does not establish a full-suite pass.
The native inventory now requires 32 cases, including cache-failure recovery
and the new F32 projection precision checks. This requirement is not a claim
that the new device checks have passed.

Supply a source manifest and a successful build receipt whose build-time hashes
cover the entire managed runtime, every selected native test executable and
`native-build/libGgmlOps.so`. The managed and native libraries must match.
The runner verifies those hashes and unchanged upstream ggml before and after
each lane, refuses active model workloads, preserves failed output in a fresh
evidence directory, and rejects skipped or missing required cases. For example:

```bash
python eng/validation/validate-muse-spark-tp.py \
  --root /work/TensorSharp --source-manifest /proof/source.json \
  --build-record /proof/build.json --runtime /proof/managed \
  --native-build /proof/native --output /proof/new-tp \
  --dotnet /opt/dotnet/dotnet --devices 0,1,2,3,4,5,6,7,8
```

These synthetic component gates do not replace checkpoint quality tests,
serving-mechanism evidence or matched performance measurements.

The test project builds a tiny deterministic model with both sliding and global
layers, GQA, routing correction, latent experts, nonuniform post-expert scales,
residual gates and output softcapping. Its independent scalar decoder checks
every output logit, chunked prefill/decode, KV restoration, and image embedding
spans. This proves those tested numerical paths; it does not measure the 816B
model's semantic quality or production throughput.

`eng/validation/muse-spark-llama-reference.cpp` links to an unchanged llama.cpp
build and exports all logits for exact token IDs. Compare it with the managed
and scalar-oracle exports using `eng/validation/compare-muse-spark-logits.py`.
The initial F32 CPU comparison covered 8 positions × 256 logits with reference
chunk sizes 1 and 5; the maximum TensorSharp/reference difference was below
`6.4e-7`, with matching greedy choices at every position.

`eng/muse-spark-validate.py` runs strict HTTP fixtures for short/long prompts,
JSON/schema/Unicode output, tool round trips, dependent agent workflows,
image layouts, request isolation, cache-token reuse and staggered arrivals.
The optional `image_chart` scenario also checks two bar rankings and visible
four-digit OCR codes using deterministic fixtures.
It emits JSON plus Markdown, distinguishes unavailable scenarios from passes,
and refuses a parity verdict when model hashes, hardware/settings, request
coverage or token counts differ, or an actual engine launch
has not been established. Conversation latency uses the same client's monotonic
wall clock; different engine-internal timers do not prevent that comparison.
Engine decode-rate comparisons require a common timing definition. Inspect its `--help` for the
provenance manifest and matching-engine commands. Internal batching,
speculative acceptance, CPU offload and multi-GPU execution need additional
runtime evidence; HTTP overlap alone does not prove their implementation.
The completed campaign evidence above distinguishes actual packed execution,
failed answer quality and passing independent n-gram and displaced-prefix
checks.

Generated logs/reports belong in ignored `docs/validation/muse-spark/` or
`artifacts/`. The recorded evidence does not establish general model-quality
parity. The per-op path's equal-work decode target was not met. The native
executor's measurements above cover one machine and one quantization.
Architecture support, synthetic numerical tests and passing bounded fixtures
must not be presented as proof of general quality or performance superiority.
