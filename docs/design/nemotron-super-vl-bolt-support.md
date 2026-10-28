# Nemotron 3.5 Super VL / Bolt support - Design & gap analysis

**Status:** Text path addressed: G1/G2/G6 implemented; G3/G4 verified already covered. G5 (vision) needed a fix for
Bolt's split-projection mmproj, see G5 below.
**Date:** 2026-10-19 / 2026-10-20
**Target models:**
- `NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4` (ggml-org GGUF, ~79GB)
- `NVIDIA-Nemotron-3.5-Bolt-74B-A9B-GGUF` (unsloth)

## 1. What the model actually is (from the downloaded GGUF header)

Parsed the first 8 MB of `NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4.gguf` (the header carries all
metadata + the full tensor-info array; tensor data is later in the file).

| Key | Value |
|---|---|
| `general.architecture` | **`nemotron_h_moe`** -- already a registered TensorSharp architecture alias |
| `block_count` | 88 |
| `embedding_length` | 4096 |
| `vocab_size` | 131072 |
| `context_length` | 1048576 (1M) |
| `ssm.inner_size` / `state_size` / `conv_kernel` / `time_step_rank` | 8192 / 128 / 4 / 128 |
| `expert_count` | **512** |
| `expert_used_count` | **22** |
| `expert_group_count` / `expert_group_used_count` | 1 / 1 |
| `expert_feed_forward_length` | 2688 |
| `expert_shared_count` | 1 |
| `expert_shared_feed_forward_length` | 5376 |
| `moe_latent_size` | **1024** |
| `expert_weights_norm` / `expert_weights_scale` | true / 5.0 |

Layer pattern is interleaved Mamba2 / Attention / MoE, encoded the same way Nemotron-H already does:
per-layer `feed_forward_length` (`0` = non-FFN, `2688` = MoE) and `attention.head_count_kv` (`0` = Mamba2,
`2` = attention).

## 2. Confirmed tensor layout (from the header)

**MoE layer** (e.g. `blk.1`):
```
blk.1.ffn_latent_down.weight    [4096, 1024]    BF16(30)   hidden -> latent
blk.1.ffn_latent_up.weight      [1024, 4096]    BF16(30)   latent -> hidden
blk.1.ffn_gate_inp.weight       [4096, 512]     F32        router: hidden -> 512 experts
blk.1.exp_probs_b.bias          [512]           F32        routing logit bias
blk.1.ffn_up_exps.weight        [1024, 2688, 512]  NVFP4(40)  [latent, n_ff, experts]
blk.1.ffn_down_exps.weight      [2688, 1024, 512]  NVFP4(40)  [n_ff, latent, experts]
blk.1.ffn_up_exps.scale         [512]           F32        per-expert NVFP4 weight scale
blk.1.ffn_up_exps.input_scale   [512]           F32        per-expert NVFP4 input scale
blk.1.ffn_up_shexp.weight       [4096, 5376]    NVFP4(40)  shared expert (gated, 2x n_ff)
blk.1.ffn_down_shexp.weight     [5376, 4096]    NVFP4(40)  shared expert down
blk.1.attn_norm.weight          [4096]          F32
```

**Mamba2 layer** (e.g. `blk.0`):
```
blk.0.ssm_in.weight    [4096, 18560]  BF16(30)
blk.0.ssm_out.weight   [8192, 4096]   BF16(30)
blk.0.ssm_conv1d.weight [4, 10240]    F32
blk.0.ssm_conv1d.bias   [10240]       F32
blk.0.ssm_a / ssm_d     [1, 128]      F32
blk.0.ssm_dt.bias       [128]         F32
blk.0.ssm_norm.weight   [1024, 8]     F32
```

**Attention layer** (e.g. `blk.7`): separate `attn_q` / `attn_k` / `attn_v` / `attn_output` (BF16), not a
fused `attn_qkv`.

## 3. What TensorSharp already supports

`NemotronModel` (architecture id `nemotron_h`, aliases `nemotron_h`/`nemotron_h_moe`/`nemotron_h_omni`)
already implements the hard parts of this family:

- Hybrid `Mamba2` / `Attention` / `FFN(MoE)` layer dispatch from the two per-layer arrays.
- Latent-space MoE (applies a latent projection, routes in latent space, projects back out).
- Shared experts (`ffn_up_shexp` / `ffn_down_shexp`, gated `ReLU^2`), see `moeInfo.HasSharedExperts`.
- Routing logit bias (`exp_probs_b` / `exp_probs_b.bias`).
- NVFP4 weights + per-tensor `scale` (scale2) sidecars (`qw.Scale`).
- MTP / speculative (DFlash) and RADIO vision (`NemotronVisionEncoder`).

## 4. Gaps

### G1 -- Latent tensor naming (the load-blocker) -- **FIXED**

TensorSharp's `NemotronModel` looks up the latent projections as
`ffn_latent_in.weight` / `ffn_latent_out.weight`, but the Super-VL/Bolt GGUF uses
**`ffn_latent_down.weight` / `ffn_latent_up.weight`** (llama.cpp `MOE_LATENT_DOWN`/`MOE_LATENT_UP`, the
"nemotron 3 super" naming).

Semantic mapping (confirmed by dims):
- `ffn_latent_down` [4096,1024] = `ffn_latent_in` (hidden -> latent)
- `ffn_latent_up` [1024,4096] = `ffn_latent_out` (latent -> hidden)

**Fix (in tree):** `NemotronModel.NormalizeLatentWeightNames()` rewrites the dictionary keys
(`down->in`, `up->out`) once, right after `LoadWeights()`, in both `_quantWeights` and `_weights`. Weight
objects are untouched (their `CacheKey`/device buffers are independent of the dictionary key), so the
legacy `ffn_latent_in/out` path and the new `ffn_latent_down/up` checkpoint both serve the same downstream
code.

### G2 -- Per-expert NVFP4 scale2 (`<base>.scale`) -- **IMPLEMENTED, then fixed twice**

Routed/non-shared experts carry a **per-expert** sidecar in addition to the NVFP4 blocks:
- `ffn_*_exps.weight` [.., 512] NVFP4(40) -- self-contained per-block scale (36-byte/64-elem block
  format embeds the block scale).
- `ffn_*_exps.scale` [512] F32 -- the **`weight_scale_2`** (per llama.cpp `_generate_nvfp4_tensors`, written
  separately and applied inside the up/down matmuls at inference).
- `ffn_*_exps.input_scale` [512] F32 -- `input_global_scale` calibration metadata.

`AttachSidecarWeightScales` only attaches a **1-element** `.scale` (covering the shared experts' scalar
`.scale[1]`); the routed experts' vector `.scale[512]` stays in `_weights` with `Scale = 1.0f` on the views.

**Fix (in tree):** `NemotronModel.InitExpertScales()` reads `ffn_*_exps.scale` and applies the per-expert
multipliers in both `MoEForward` and `TryMoEPrefillBatchedByExpert`. Experts without sidecars
(IQ-quantized, e.g. Bolt) are a no-op.

**Validated on the DGX Spark, and the first implementation was wrong.** It originally folded a single
multiplier `downScale * upScale^2` into the *routing weight*. That is algebraically correct --
`ReLU^2(us*x) = us^2 * ReLU^2(x)` for positive scale2 -- but numerically fatal: it defers the up
correction until *after* the squaring, so the kernel squares an `up_out` still `1/us` too large. At
NVFP4 `us ~ 1e-5` that overflows to `+Inf`, and the Inf met a `~1e-15` route weight to produce NaN.
Super VL aborted warmup with "non-finite router logits" at its first MoE layer.

Now (`6be5296e`, `a0941ceb`) the scales are kept **separate**: `us` is applied to the up projection
immediately before `ReluSquaredInPlace`, and only `ds` is folded into the routing weight (a single
multiply on correctly-magnituded values, which is safe). The batched GGML kernel gained an optional
trailing `up_scales` array so it can apply `us` between the up matmul and the ReLU.

**Separately**, the fused residual-add path dropped the *per-tensor* 1-element scale entirely
(`b3aba946`): `TryLinearAddInto` -> `FusedMatMulQuantAdd` takes no scale argument, so
`ssm_out.weight` on all 40 Mamba2 layers was added to the residual unscaled. `LinearForward` and
`LinearForwardInto` both apply `qw.Scale`, so only the fused decode path was wrong -- which is why the
native and CPU Mamba2 paths produced *identical* garbage and kernel-level debugging was a dead end.
Bolt's symptom was fluent text for 3-4 tokens, then collapse. Gate every fused quant fast path on
`qw.Scale == 1.0f` (the Qwen3.5 pattern); note two MoE call sites `throw` on fallback rather than
degrading, so `CanMoEDecodeResidualAdd` needs the same gate.

### G6 -- Per-block MoE config ("Puzzle"/Bolt) -- **IMPLEMENTED**

Bolt-74B-A9B (the "Puzzle" / `block_configs` variant) gives each MoE layer its own expert count,
FF width and top-k, unlike the uniform Super-VL:

```
blk.1  ffn_up_exps [1024, 2048, 240]  # 240 experts, n_ff 2048
blk.3  ffn_up_exps [1024, 1536, 240]  # n_ff 1536
blk.21 ffn_up_exps [1024, 2560, 432]  # 432 experts, n_ff 2560
```

`expert_used_count` and `expert_feed_forward_length` are per-layer arrays `[88]`, not scalars.

**Fix (in tree):** `NemotronModel.InitPerLayerMoEConfig()` derives `_numExpertsPerLayer[]` (from the
stacked expert tensor's last dim) and `_numExpertsUsedPerLayer[]` (from the `expert_used_count` array,
scalar fallback). `_numExperts`/`_numExpertsUsed` are grown to the observed maximum so every pre-allocated
routing buffer still fits; `MoEForward` and `TryMoEPrefillBatchedByExpert` route against the per-layer
counts. Uniform checkpoints are binary-identical to before (per-layer == global). Per-block under **tensor
parallelism is rejected with a clear error** for now (the TP MoE paths are not per-layer aware yet).

Note: Bolt's experts are `IQ2_XXS` (type 16), so G2 (NVFP4 per-expert scale2) does **not** apply to Bolt --
it only affects the 120B NVFP4 file.

### G3 -- 512 experts / top-22 -- **was NOT covered; fixed**

`expert_count=512`, `expert_used_count=22`. The stacked-expert storage, per-expert view creation,
`SelectTopKInPlace` (delegates to `TensorComputePrimitives`, no pool-size limit), and the per-block
config infrastructure do support 512 experts, and routing buffers are correctly sized
(`_moeProbs`[512], `_moeTopExperts`[22]).

**But this analysis missed the GGML kernel limit.** `GgmlNative.MoEExpertsForward` (native
`TSGgml_MoEExpertsForwardF32`) rejects `numExperts` outside **[1, 16]**, so top-22 aborted on the first
MoE forward with `MoE: num_experts must be 1..16`. The sibling SwiGLU variant allows 1..32. Fixed in
`57a20f89` by splitting wider routing into chunks of <=16 and accumulating with `VecScaleAdd`; models
at or below 16 take a single chunk and are unchanged. The tensor-parallel equivalent got the same
treatment in `69201abb`.

Lesson for the next shape review: "the managed routing data structures support N" is not the same as
"every native kernel on the path supports N". Check the kernel's own argument validation.

The outstanding verification is still CPU-offload memory for 512 NVFP4 experts per MoE layer
(~2.8 GB/layer) -- `--n-cpu-moe` has not been validated.

### G4 -- BF16 non-expert weights -- **verified already covered (confirmed on Spark)**

`ssm_in/ssm_out/attn_*/ffn_latent_*` are BF16 (type 30). BF16 is explicitly listed in
`CanStoreDirectCudaCompressedWeight`, `ManagedQuantizedOps.SupportsCpuQuantizedStorage`, and the
`ManagedQuantGemm.Float` cheap-convert path. The Nemotron DSpark drafter already exercises BF16
attention weights in production. No gap.

**Confirmed empirically, and worth stating because the opposite looks plausible.** Super VL's Mamba2
`ssm_in/ssm_out` are BF16 and its NaN appeared right after a Mamba2 layer, which makes BF16 an
attractive suspect. It is not the cause. `addmm_quant_f32_impl` passes the type straight to
`ggml_new_tensor_2d` with **no** block-quant guard, and BF16 `mul_mat` works; the in-tree comment in
`ModelBase.DFlash.cs` claiming the AddmmQuant family "silently returns a zeroed output" for F16/BF16
is drafter-specific and stale, not a general constraint.

Do **not** "fix" this by dequantizing BF16 to F32 at load. That grew Super VL from ~73 GB to ~85 GB of
weights and OOM-killed it on a 121 GB Spark during prefill warmup. Run models at the quantization they
ship in -- these GGUFs are sized deliberately for the platform. If a dtype genuinely is unsupported,
fix it in TensorSharp-owned native code rather than inflating the footprint.

### G5 -- Vision -- **fixed for Bolt's split projections**

Super-VL / Bolt are `image-text-to-text` with a RADIO-family vision tower in the mmproj.
`NemotronVisionEncoder` reads the `clip` mmproj + `nemotron_omni` projector format: `clip.vision.*`
config keys, `v.blk.*` ViT blocks (with bias and GELU) and `mm.model.mlp.*` projector tensors.

An earlier revision of this document said the mmproj had been verified against its header and needed no
work. That was wrong: only the language-model GGUF header had been read. The mmproj files differ:

| mmproj | attention projections |
|---|---|
| Super VL (ggml-org), Nano Omni | fused `v.blk.N.attn_qkv.{weight,bias}` |
| Bolt (`mmproj-NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf`, `nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF`) | separate `attn_q` / `attn_k` / `attn_v` weights and biases |

The encoder read only `attn_qkv`, so every Bolt image request threw `NullReferenceException` (HTTP 500)
and the startup multimodal warm-up was skipped. The encoder now concatenates split `attn_q/k/v` (and
their biases; a missing one is zero) into the fused `[Q | K | V]` layout at load, which is exactly the
fused projection, and refuses at load - naming every tensor - an mmproj missing a tensor it reads, or
whose projector width does not match the language model. `NemotronVisionSplitQkvTests` checks a split
file encodes identically to its fused twin. The Bolt mmproj itself has not been run here; the reported
layout comes from the review of the PR on 2x A40.

### G7 -- Tensor parallelism for Super VL and Bolt -- **IMPLEMENTED, NOT YET RUN**

`--tp` used to be refused for both files: they carry `.scale` sidecars and Nemotron did not set
`SupportsTensorParallelWeightScales`; Bolt's per-block MoE config was refused under TP outright; and the fast
expert-parallel MoE path refused latent MoE (both models) and NVFP4 up-scaled experts (Super VL), leaving a
per-expert loop of thousands of tiny matmuls per layer on `ggml_cuda`.

What changed:

| Piece | Change |
|---|---|
| Per-tensor `.scale` | Audited every TP path: attention QKV/output use the TP linears, which carry each shard's scale; Mamba2, router, latent in/out, shared expert and LM head run replicated through `LinearForward`, which applies it. `SupportsTensorParallelWeightScales => true`. `FuseQKVWeights` now keeps the scale and refuses to fuse Q/K/V with different scales; `ShardSeparateColumnParallel` (which builds `[Q|K|V]` per rank from separate sources) used to drop each source's scale despite a comment saying otherwise and now bakes it into the rows. |
| Per-expert NVFP4 scale2 | Down scale folded into the TP routing weights (all TP expert paths); up scale applied before ReLU^2 on every path. The batched TP decode kernel takes the up scales instead of falling back to the per-expert loop. |
| Expert-parallel (EP) MoE | Now serves latent MoE (each rank's kernel reads the latent projection) and per-block expert counts (each layer is partitioned on its own). New native entry `TSGgml_MoEFFNPrefillRouteScaledF32` (TensorSharp-owned `ggml_ops_moe.cpp`; ggml is unchanged) multiplies the ReLU^2 projection by one scale per route before the activation, so Super VL's NVFP4 experts run EP too. A route the kernel neutralises (weight 0) takes the scale of the expert it is pointed at, so it cannot overflow into `Inf * 0 = NaN`. |
| Per-block MoE (Bolt) | Router stride, top-k, expert loops and decode tables use each layer's own counts. The refusal in `InitPerLayerMoEConfig` is gone. The per-expert (Megatron) split now refuses, naming the tensor, an expert width the TP degree cannot split between quant blocks instead of silently dropping the remainder. |
| Across nodes | Image/audio embeddings exist only on the driver. A forward whose tokens hold `<image>`/`<so_embedding>` now syncs the driver's embedding rows to every node with one AllReduce (the driver contributes them, every other rank zeros); the decision is taken from the tokens, which every node sees. The multimodal warm-up is skipped across nodes, since only nodes given the mmproj would join its collectives. |

Tests: `NemotronExpertParallelTests` runs the new kernel on the GGML CPU backend against a C# reference
and checks the per-rank route table; `NemotronPerLayerMoEConfigTests` covers the per-block counts. There is
no CPU harness for the Nemotron TP forward itself, so **none of the TP forward has executed**.

Limits that remain:
- The server hosts Nemotron under TP through KV snapshots (`NemotronModel.TensorParallelSnapshot.cs`): each
  rank's K/V heads are a contiguous run of the single-device block layout, so the block format is unchanged;
  across nodes the driver broadcasts each extract/inject (`ModelBase.BroadcastModelControl`) and the bytes move
  between nodes as raw bytes over the TCP mesh (`ICrossNodeByteGroup`): an inject sends the driver's block to
  every node, an extract gathers each node's head shards on the driver. Each owner swap on the per-sequence
  route re-injects the incoming sequence's pages, its ~50 MB Mamba2 state included (Lightning), so the
  interconnect bandwidth bounds concurrent serving. Run the server as node 0 and `TensorSharp.Cli` as
  every other node. The engine serves it on the per-sequence route (one sequence's forward at a time); prompt
  reuse uses 256-token pages, as the single-device batched route does (`InferenceEngine.PreferredBlockSize`).
- MTP/DFlash speculation is not mirrored to worker nodes.
- `--n-cpu-moe` with TP has not been run.

#### DGX Spark test plan (two nodes, one GB10 each)

Same model, backend and flags on both nodes; node 0 is the driver and prints the output. For the server,
start `TensorSharp.Server.Host` with the same `--tp 1 --tp-node-id 0 --tp-peers ...` on node 0 instead of the
CLI driver, and the CLI worker on node 1.

```bash
# node 0 (driver)
TensorSharp.Cli --model <model.gguf> [--mmproj <mmproj.gguf>] --backend ggml_cuda --tp 1 \
  --tp-node-id 0 --tp-peers <spark0>:9500,<spark1>:9500 --temperature 0 --max-tokens 64 --prompt "<prompt>"
# node 1 (worker)
TensorSharp.Cli --model <model.gguf> --backend ggml_cuda --tp 1 \
  --tp-node-id 1 --tp-peers <spark0>:9500,<spark1>:9500
```

1. **Load banner.** Expect `Nemotron MoE: expert-parallel across 2 GPU(s), N experts per GPU per layer` (Super VL:
   256; Bolt: a range) and, for Bolt, `MoE per block: ...`. `per-expert tensor parallelism (<reason>)` means EP was
   refused and gives the reason.
2. **Logits parity, the real check.** Run the same prompt once single-device and once across the two nodes with
   `TS_DUMP_LOGITS=<file>` on node 0, then compare the two float32 dumps (cosine ~1, same argmax). Text alone can
   differ at near-ties.
3. **Greedy text** on a few prompts, both models, `--temperature 0`.
4. **Images** (Super VL, Bolt): one `--image` prompt across nodes; the worker does not need `--mmproj`.
5. **Debugging aids:** `TS_TP_SCALE_DEBUG=1` prints each sharded weight's scale as it is applied;
   `CUDA_LAUNCH_BLOCKING=1` for an exact failing kernel.

## 5. Summary

**Validated on a DGX Spark (GB10) on 2026-10-24/25. Both models now produce correct output.** The
original gap analysis was directionally right but declared three things covered that were not, and all
three were real bugs. "Loads and reports the right shapes" turned out to be weak evidence: both models
loaded cleanly, printed correct layer/expert counts, and then produced garbage or NaN.

What was actually broken:

| Gap | Claimed | Actual | Fix |
|---|---|---|---|
| G2 per-tensor scale2 | implemented | fused residual-add dropped it entirely | `b3aba946` |
| G2 per-expert scale2 | implemented | folded past ReLU^2, overflowed to Inf/NaN | `6be5296e`, `a0941ceb` |
| G3 top-22 | "verified already covered" | GGML kernel rejects >16 experts | `57a20f89`, `69201abb` |
| G4 BF16 | "no gap" | **correct as written** | none needed |

G4 is the one claim that held up, and it is worth trusting: BF16 works natively through AddmmQuant.

## 6. Verification results (DGX Spark, GB10, aarch64, CUDA 13.0)

Both models, `--temperature 0`, native quantization, no weight expansion:

- **Bolt 74B** on `ggml_cuda` and on `cuda`: correct answers, `finishReason=eos`. Byte-identical
  output across the pre- and post-merge trees.
- **Super VL 120B** on `ggml_cuda`: `"The capital of France is Paris."` and a correct multi-step
  reasoning answer, both `finishReason=eos`. 72 GB of weights loads and runs in the 121 GB pool.

Symptoms before the fixes, for anyone who hits something similar: Bolt was fluent for 3-4 tokens then
collapsed into `>>(>>(` and file-extension soup; Super VL aborted warmup with "non-finite router
logits" at MoE layer 3. A per-layer probe located it -- `after L0 (Mamba2): finite absmax=0.85`,
`after L1 (MoE): NON-FINITE` -- which is what separated the MoE scale bug from the Mamba2 one.

**Benchmarking on this host:** end-to-end model timing is useless for kernel-level work. Two runs of
the *identical binary* gave Attention = 63553 ms and 81341 ms, a 28% swing wider than the spread
across four different code paths. Use `benchmarks/FlashAttnBench` (median of N, isolated kernel, plus
a CPU-reference `maxErr`). Note the profiler's "Attention" bucket also accumulates Mamba2 time
(`_attnTicks` is incremented inside `Mamba2Forward` and `TryMamba2NativeDecode`), so on this hybrid
model that label is mostly Mamba2 -- which is why an attention-kernel win is invisible end-to-end.

**Backend choice:** prefer `ggml_cuda` for Nemotron. Bolt decoded at ~8.8 tok/s there versus 0.7 tok/s
on the direct `cuda` backend, whose attention is the bottleneck. The direct backend's matmuls are ~11x
faster, so it is worth optimizing, but Mamba2 -- not attention -- is where its time goes.

**A `model load refused ... CUDA device has 5486 MiB free of 124546 MiB` is memory pressure, not a
platform limit.** Leftover `dotnet` runs and Ollama hold the unified pool. Clear them and retry; check
holders with `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv`.

### Building on a Spark

The supported path is `eng/Dockerfile.gb10` (see `DEVELOPMENT.md`), which pins CUDA 13.0.2, compiles
`compute_121` PTX for the direct CUDA backend, and **disables Vulkan**. Committing `.cu` source is
therefore how a kernel reaches Spark users: `SelectCudaPtxContent` swaps locally compiled PTX in
whenever nvcc is present, so the committed baseline is not on their path.

Two traps if you build on the host instead of in the container:

- The GGML native build fails in the **Vulkan** target, because
  `ExternalProjects/vulkan-toolchain/shaderc-linux/bin/glslc` is an **x86-64 ELF** on an aarch64 host.
  The kernel returns ENOEXEC and `sh` parses the binary as a script, giving the misleading
  `glslc: 1: Syntax error: "(" unexpected`. Build with
  `TENSORSHARP_GGML_NATIVE_ENABLE_VULKAN=false bash TensorSharp.GGML.Native/build-linux.sh`. The
  `curl: (22) ... 404` alongside it is that toolchain fetch failing, not upstream flakiness.
- Do not then run `dotnet build` without a skip flag: the csproj re-invokes `build-linux.sh`, where the
  same 404 is fatal (`exited with code -1` -> MSB3073). Use `-p:TensorSharpSkipGgmlNative=true`;
  `CopyGgmlNativeBinary` has no skip condition, so it still copies the fresh `.so`.

The committed PTX baseline targets `compute_61`, and **nvcc 13 dropped `compute_61`** (it lists
`compute_75`..`compute_121`), so the baseline cannot be refreshed on a Spark. Any newly added kernel
must therefore be resolved with `GetOptionalFunction`, not `GetFunction`: the latter calls
`ThrowOnError` and handles are resolved eagerly in the `CudaKernels` constructor, so a missing symbol
would take the whole direct-CUDA backend down on no-nvcc builds. Gate the feature on the handle being
non-zero and fall back.

### Still unverified

- **Tensor parallelism.** `69201abb` is compile-verified only; a single GB10 cannot execute any TP
  path. Expert-parallel MoE throws `NotSupportedException` for up-scaled experts rather than routing
  garbage, because `MoEFFNPrefill` has no per-expert scale hook and building EP shards removes the
  per-expert weights the fallback needs.
- **Tensor parallelism (G7, below) is implemented but has not run.** Nothing here has a GPU; see G7 for
  what changed and the DGX Spark test plan.
- **Super VL on a single GPU still needs more than ~73 GB.** `NemotronModel` logs `--n-cpu-moe` but still
  preloads every expert to the GPU (as `main` did); on a 48 GB card it runs on `ggml_cpu`, or split it with
  `--tp` across GPUs/nodes (G7).
- **Vision (G5)** -- Bolt's split projections are covered by a synthetic parity test only; no Bolt image
  has been encoded end-to-end here.
- **Direct `cuda` prefill attention changed.** Every Nemotron prefill on the direct `cuda` backend now goes
  through `TryGqaPrefillAttention`. The kernel matches the CPU reference (max error 1.21e-4) and is
  1.7-3.6x faster at 2K-8K tokens, but greedy text can differ at near-ties (2 of 7 Nemotron-H 8B answers
  were reworded, both still correct), and the gain did not show end to end in Lightning's ~6K-token
  prefill.
- **CPU offload** with 512 NVFP4 experts per layer (`--n-cpu-moe`).
- The new group-16/head_dim-128 flash prefill kernel is restricted to `windowSize == 0`; it implements
  sliding windows but only full attention has been run.