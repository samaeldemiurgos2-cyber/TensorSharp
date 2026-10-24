# MiniMax-M3.1-Flash (`minimax-m3`)

[← back to model index](README.md) | [中文](minimax-m3_zh-cn.md)

TensorSharp recognizes the `minimax-m3` GGUF architecture of
[MiniMax-M3.1-Flash](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash), a
428B-parameter sparse MoE with about 23B parameters active per token, as
converted by [Unsloth](https://huggingface.co/unsloth/MiniMax-M3.1-Flash-GGUF).
The decoder, MiniMax Sparse Attention, the NVFP4 expert path, tensor
parallelism, the DSpark drafter and the vision tower all live in
TensorSharp-owned code; the fetched ggml dependency (upstream `da8e2a5`) is
unchanged. This is a **multi-GPU server model**: the UD-Q4_K_XL checkpoint is
232.3 GiB, and the measured setup is eight 46 GB A40 GPUs.

| Property | Implementation |
|---|---|
| Architecture | `minimax-m3`, alias `minimax_m3` |
| Model | 60 layers at hidden width 6144: 3 leading dense layers (SwiGLU-OAI FFN, 12288 wide), then 57 MoE layers with 128 routed experts (top-4 sigmoid routing with an F32 selection bias, weights renormalized and scaled by 2.0) plus one shared expert (3072 wide). 64 query / 4 KV heads at head dim 128, per-head Q/K RMSNorm, NeoX RoPE over the first 64 of 128 dims (base 1e7). Vocabulary 200,064; the GGUF advertises a 1,048,576-token context |
| Attention | MiniMax Sparse Attention (MSA) on **every** layer: an indexer (4 index heads of 128, one per KV head) scores 128-row blocks and each KV head attends to its top-16 blocks, its own block always included. Below position 2048 every visible block is selected, so dense causal attention there is exact. TensorSharp's CUDA MSA kernels ([ggml_ops_minimax_msa.cu](../../TensorSharp.GGML.Native/ggml_ops_minimax_msa.cu)) run the sparse regime; upstream ggml ops are the fallback ([details](#attention-modes)) |
| Decoder | [MiniMaxM3Model](../../TensorSharp.Models/Models/MiniMaxM3/MiniMaxM3Model.cs) parses and validates the checkpoint; the [native whole-model executor](../../TensorSharp.GGML.Native/ggml_ops_minimax_m3_exec.cpp) loads the split GGUF itself and submits one GGML graph per micro-batch |
| Experts | NVFP4 routed experts with per-expert F32 output scales (applied to gate and up before the activation and to down before the routing weight); Q8_0 attention, shared expert and dense FFN; F32 indexer and router; BF16 embeddings and output head |
| Backends | `ggml_cuda` serves. `ggml_cpu` runs the same executor on one CPU device as a correctness path. `mlx`, `ggml_metal`, `ggml_vulkan`, `cuda` (direct CUDA) and `cpu` (pure C#) refuse the load with exit code 2 |
| GPU placement | `--layer-split N` places whole layers on N GPUs (up to 16). `--tp N` splits every layer over the first N CUDA GPUs (2 to 16) through upstream ggml's meta backend. **Without either, the executor uses one GPU**, which the 232 GiB checkpoint does not fit ([running](#running-it)) |
| Expert offload | `--n-cpu-moe N` serves the routed experts of layers 0..N-1 from system RAM; `TS_MINIMAX_M3_CPU_MOE=auto` places only the expert stacks that do not fit. Layer split only |
| KV cache | F16 per-sequence slots on the GPUs that read them: K, V and the MSA index keys, 135 KiB per token over the 60 layers. Every row is kept, so a rewind to any token is exact. Block-quantized `--kv-cache-dtype` values are refused; `f32` runs as F16 with a warning |
| Continuous batching | Batched decode across sequence slots and batched prefill of concurrent text prompts in shared graphs |
| Prefix cache | Conversation continuation from a finished slot, exact rewind, and shared system-prompt / tool checkpoints cloned into new conversations |
| Images | Matching `mmproj` (`clip.projector_type = minimax_m3`), which you convert yourself ([below](#make-the-vision-projector-and-the-dspark-drafter)); video arrives as sampled frames; audio is refused |
| Speculation | DSpark block drafter through `--draft-model` (layer split only), and the weight-free n-gram drafter (`--spec --spec-type ngram`). Neither runs under `--tp` |
| Chat protocol | Own renderer, byte-exact against Hugging Face's template; adaptive thinking; namespaced-XML tool calls with schema typing; grammar-enforced `tool_choice` and `response_format` ([details](#chat-format-reasoning-tools-and-json)) |

## Download

### The text model

```bash
# UD-Q4_K_XL: seven shards, 249,402,105,310 bytes (232.3 GiB). Keep every shard in one
# directory and point --model at the -00001-of- shard.
hf download unsloth/MiniMax-M3.1-Flash-GGUF --include "UD-Q4_K_XL/*" --local-dir models/minimax-m3
```

The Unsloth repository is marked as work in progress. At the time of writing it
holds the text model only: no `mmproj` and no DSpark drafter. Make those two
files yourself as described next, or run text only.

### Make the vision projector and the DSpark drafter

Both companions are converted from the Hugging Face release with the converter
of [unslothai/llama.cpp PR #320](https://github.com/unslothai/llama.cpp/pull/320)
(branch `minimax-m31`, commit `ef8f94cc`). Only the configuration, tokenizer and
template files, the two safetensors shards that hold the vision tower (13 and
48 of 48) and the `dspark/` directory are needed:

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

# The mmproj step reads every shard its index names, so give it a directory whose
# index lists only shards 13 and 48.
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
# Vision projector, 1,735,294,400 bytes
../convert-venv/bin/python convert_hf_to_gguf.py "$VIS" --mmproj --outtype bf16 \
    --outfile ../models/minimax-m3/mmproj-MiniMax-M3.1-Flash-BF16.gguf
# DSpark drafter (GGUF arch dflash), 3,925,395,968 bytes
../convert-venv/bin/python convert_hf_to_gguf.py "$HF/dspark" --target-model-dir "$HF" --outtype bf16 \
    --outfile ../models/minimax-m3/MiniMax-M3.1-Flash-DSpark-BF16.gguf
```

TensorSharp checks both files before it maps the 232 GiB model: the projector
must declare `clip.projector_type = minimax_m3` and project to 6144 dims, and
the drafter must be a `dflash` GGUF with a Markov head, the every-row
(`sample_from_anchor`) block layout, `swigluoai`, a sliding window on every
draft layer, and target layers, vocabulary and mask token that match this
model. Anything else is refused with a message that names the mismatch.

## Running it

The 232 GiB checkpoint needs several GPUs. TensorSharp defaults to **one**
GPU, so pass the placement explicitly: `--layer-split 8`
or `--tp 8` on eight 46 GB GPUs. A launch without one is refused because the
model does not fit one GPU, and the message names `--n-cpu-moe` and
`TS_MINIMAX_M3_CPU_MOE=auto`. Always pass `--backend ggml_cuda`: the server's
default backend off macOS is `ggml_cpu`, which this model runs only as a
correctness path.

```bash
# The measured layer-split profile: eight A40s, a 72K context for one request plus
# the shared-prefix checkpoint (TS_MINIMAX_M3_PLAN_CTX_TOKENS = context x 2).
MAX_CONTEXT=73728 TS_MINIMAX_M3_PLAN_CTX_TOKENS=147456 \
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll \
    --model models/minimax-m3/UD-Q4_K_XL/MiniMax-M3.1-Flash-UD-Q4_K_XL-00001-of-00007.gguf \
    --mmproj models/minimax-m3/mmproj-MiniMax-M3.1-Flash-BF16.gguf \
    --backend ggml_cuda --layer-split 8 --port 5000

# Tensor parallelism over the same eight GPUs: faster decode, most of all at short depths
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll \
    --model models/minimax-m3/UD-Q4_K_XL/MiniMax-M3.1-Flash-UD-Q4_K_XL-00001-of-00007.gguf \
    --backend ggml_cuda --tp 8 --port 5000

# DSpark speculative decoding (layer split only)
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll \
    --model models/minimax-m3/UD-Q4_K_XL/MiniMax-M3.1-Flash-UD-Q4_K_XL-00001-of-00007.gguf \
    --backend ggml_cuda --layer-split 8 \
    --draft-model models/minimax-m3/MiniMax-M3.1-Flash-DSpark-BF16.gguf --port 5000

# Fewer GPUs (not measured on the full checkpoint): the routed experts of layers 0..19
# (17 MoE layers, about 65 GiB) in system RAM
MAX_CONTEXT=16384 CUDA_VISIBLE_DEVICES=0,1,2,3 \
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll \
    --model models/minimax-m3/UD-Q4_K_XL/MiniMax-M3.1-Flash-UD-Q4_K_XL-00001-of-00007.gguf \
    --backend ggml_cuda --layer-split 4 --n-cpu-moe 20 --port 5000

# CLI one-shot
echo "Give me three facts about the Moon." > prompt.txt
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll \
    --model models/minimax-m3/UD-Q4_K_XL/MiniMax-M3.1-Flash-UD-Q4_K_XL-00001-of-00007.gguf \
    --backend ggml_cuda --layer-split 8 --input prompt.txt --max-tokens 512
```

`--model` may name any shard of the set; TensorSharp reads the metadata from
shard 1 and the tensors from every shard. The server never looks for a chat
model's projector beside it, so pass `--mmproj` for image input. The startup
log prints the placement (`MiniMax-M3.1 native executor: 60 layers over 8
GPU(s), context ..., ubatch 2048, layers with host experts ...`) and, under
`--tp`, the weights, K/V and reserve on each GPU.

### Context length and memory

The GGUF advertises 1,048,576 tokens, which no measured setup holds. Without
`MAX_CONTEXT`, the executor caps the context to the largest size that fits and
logs `[minimax-m3] context capped to N tokens (asked for 1048576) ...; set
MAX_CONTEXT to choose a size`. With `MAX_CONTEXT` set, a context that does not
fit refuses the load. The placement reserves K/V rows for
`TS_MINIMAX_M3_PLAN_CTX_TOKENS` tokens across all slots, one full context by
default; set it to the context times the number of concurrent requests plus
one (for the shared-prefix checkpoint), as the launch above does. Slots start
at 4,096 rows and double on demand.

K, V and the index keys take 135 KiB per token: 9.5 GiB for a 73,728-token
sequence on a layer split, where each GPU holds its own layers' rows. Under
`--tp` every GPU holds a full copy, so on eight A40s with no `MAX_CONTEXT` the
context is capped to 45,824 tokens. Each MoE layer's routed experts are about
3.8 GiB, which is what one `--n-cpu-moe` layer moves to system RAM. With
`TS_MINIMAX_M3_CPU_MOE=auto`, set `MAX_CONTEXT` (or
`TS_MINIMAX_M3_PLAN_CTX_TOKENS`) too: the automatic plan can always make room
by moving more experts to the host, so it does not cap the context, and
without a limit it reserves K/V for the advertised 1,048,576 tokens.

## How it runs

- **One graph per micro-batch.** `ggml_backend_sched` splits each graph across
  the GPUs (and the CPU for host experts). Prefill uses 2048-row micro-batches
  (`TS_MINIMAX_M3_UBATCH`); a graph that does not fit runs that call in smaller
  micro-batches. An LRU of built graphs keyed by shape lets decode replay the
  same graph, and its captured CUDA graphs, token after token.
- **Layer placement.** `--layer-split N` assigns contiguous runs of whole
  layers to N GPUs. Among the plans that leave the fewest expert bytes on the
  host, it takes the one whose busiest GPU does the least per-token work (a
  pipelined prefill runs at the pace of its slowest stage), then the one that
  fills the GPUs most evenly. `TS_MINIMAX_M3_LAYER_SPLIT` gives explicit
  per-GPU layer counts instead. A GPU count the executor cannot honour (more
  than 16, more than are visible) is refused rather than run on fewer GPUs.
- **Pipelined prefill.** With a layer split over two or more CUDA GPUs and no
  host experts, each prompt call is cut into micro-batches and GPU k works on
  one while GPU k+1 works on the one before. The number of micro-batches is
  chosen per call (2 for 2,048 tokens, 3 for 4,096, 4 for 8,192 on eight
  A40s), between 256 rows and the prefill ubatch, unless
  `TS_MINIMAX_M3_PIPELINE_UBATCH` fixes it. The load log says
  `pipelined prefill: on (...)` or `off (reason)`; a call that cannot pipeline
  logs `pipelined prefill declined (reason)` once per reason. With pipelining
  on, a prompt alone in the server is prefilled in 8,192-token steps, a short remainder joined to the
  step before, unless `TS_SCHED_SOLO_PREFILL_CHUNK` is set.
- **Sparse attention per sequence.** Positions equal cache rows, so a block is
  a view of the slot's cache and a slot is never shifted. A batched decode step
  evaluates one token of several slots in one graph, each with its own sparse
  selection; llama.cpp runs MSA only with per-sequence KV (`--no-kv-unified`),
  which TensorSharp's slots always are.
- **Host experts.** `--n-cpu-moe N` serves layers 0..N-1's routed experts from
  the GGUF mapping and runs exactly those `mul_mat_id` nodes on the CPU. On a
  host with two or more NUMA nodes they are copied into memory local to pinned
  worker threads (`TS_MINIMAX_M3_HOST_NUMA=0` keeps the mapping), and batches of
  2-16 tokens run through a static-row kernel with ggml's own dot products.
  The host workers default to half the CPU quota: host-expert decode is
  memory-bandwidth bound, and the free half keeps the server's own threads off
  the workers' cores (with all of them, server decode measured half the
  executor's rate). `--cpu-moe-threads N` raises it for prompt-heavy work.
  Layers 0-2 are dense, so `--n-cpu-moe 3` or less moves nothing; the executor
  says so.
- **Prefix cache.** A finished request's slot is retained and donated to the
  request that continues the conversation, rewound first if needed. The system
  block, the default developer turn and the tool declarations open every
  conversation, so they are copied once into a checkpoint slot and cloned into
  each new conversation. Prefix reuse stops at the start of an image span.
  Unrelated conversations share only that public prefix.
- **Memory admission.** A new slot or KV growth is admitted only while each GPU
  keeps `TS_MINIMAX_M3_SLOT_HEADROOM_MB` plus room for the largest graph seen
  so far (capped at `TS_MINIMAX_M3_ADMIT_GRAPH_MB`); retained slots are freed
  first. Loading runs one full-size prefill (`TS_MINIMAX_M3_WARM_POOLS`) so
  ggml-cuda's kernel-scratch pools reach their peak before slots take the
  memory.

### Attention modes

`TS_MINIMAX_M3_MSA` selects how attention runs:

| Value | What runs |
|---|---|
| `auto` (default) | A micro-batch whose last position is below 2048 runs dense flash attention, which is exactly the sparse result there. Any other runs sparse attention through TensorSharp's MSA kernels on a CUDA GPU (scoring, selection and block-sparse attention in one custom node per segment) and through upstream ggml ops on any other device. Under `--tp` the sparse regime always uses the ggml ops |
| `dense` | Dense attention everywhere. Debugging only: it changes results from position 2048 on, and its caches are never shared with the sparse model's |
| `ggml` | Sparse attention for every micro-batch through upstream ggml ops (llama.cpp's algorithm) |
| `kernel` | TensorSharp's MSA kernels for every micro-batch, through each CUDA GPU's fused-op wrapper (on a CPU-only load, through their CPU reference). Refused on a GPU that cannot run them, and under `--tp`, whose meta backend cannot hold the kernel node |

Any other value refuses the load. Index keys are cached in every mode.

### Tensor parallelism

`--tp N` runs the same executor on upstream ggml's meta backend, the mechanism
behind llama.cpp's `-sm tensor`, over the first N CUDA GPUs; choose them with
`CUDA_VISIBLE_DEVICES`. The GPUs must be identical. MiniMax-M3.1 has only four
KV heads, and its sparse attention cannot be split by head on the meta backend,
so attention, the indexer and every slot's K, V and index-key caches are
**mirrored** on each GPU. The dense, routed and shared-expert FFNs are split
along their hidden dimension in units of 128 values, and the output head by
vocabulary rows; each layer ends in one all-reduce. TensorSharp probes peer
copies first and runs NCCL over shared memory where P2P is advertised but
broken. The split plan is
[ggml_ops_minimax_m3_tp.cpp](../../TensorSharp.GGML.Native/ggml_ops_minimax_m3_tp.cpp)
(ctest `minimax-m3-tp-split-plan`).

These refuse the load (exit 2): `--layer-split` or `TS_MINIMAX_M3_LAYER_SPLIT`
together with `--tp`; `--n-cpu-moe` (every expert stays in GPU memory);
`--draft-model`; `TS_MINIMAX_M3_MSA=kernel`; `TS_MINIMAX_M3_PIPELINE=1`;
`TS_MINIMAX_M3_PROFILE`; `TS_MINIMAX_M3_GRAPH_CACHE` above 1;
`TS_MINIMAX_M3_CPU_REPACK=1`; `TENSORSHARP_TP_DEVICES`, which this path does not
read; a backend other than `ggml_cuda`; distributed `--tp-node-id` /
`--tp-peers` groups; and a `MAX_CONTEXT` or `TS_MINIMAX_M3_PLAN_CTX_TOKENS`
that leaves any GPU short, with the largest context that would fit in the
message. Pipelined prefill and speculation are
off, and the graph cache keeps one graph, because the meta backend keeps
per-GPU views only for the graph it ran last.

TP decode is fastest at short depths: on eight A40s it decoded 35.7 tok/s at
depth 0 and 26.0-26.5 tok/s at depths 2,048 to 16,384, where the sparse
regime runs on the ggml ops, against the layer split's 23.4-24.8 tok/s (the TP
figures come from the tensor-parallel branch's build before integration). Its
prefill was 1,147 / 928 / 762 tok/s at 2K / 8K / 16K tokens, so the layer split
is the mode for long prompts.

## DSpark speculative decoding

The MiniMax-M3.1-Flash DSpark drafter (GGUF arch `dflash`, five sliding-window
layers, a Markov head and a confidence head) reads the outputs of target
layers 3, 17, 31, 45 and 59. Naming it with `--draft-model` (or
`TS_SPEC_DRAFT_MODEL`) is the request: speculation turns on by itself unless
`--no-spec` is given. The native executor loads the drafter with the target,
on the GPU that holds the output head it borrows (about 4.4 GiB there), and
commits the drafter's context inside every trunk graph, so prefill and
verification need nothing from the host and a draft is one native call that
proposes a block of up to 7 tokens. A drafter attached after the load is refused, and the
generic managed DFlash host does not run it.

The default gate is the block drafter's cumulative confidence gate: draft
position i is kept while the product of the confidence head's estimates up to
it stays at or above 0.35 (`--spec-pmin` / `TS_SPEC_PMIN` overrides it;
`--spec-pmin 0` drafts every position, as llama.cpp's `draft-dspark` with
`p-min 0` does). `--spec-draft` defaults to 8 and is capped at the drafter's
block of 7. Speculation engages for solo (non-concurrent) requests, works with
image prompts, and is refused under `--tp`. On eight A40s the default gate
decoded 32.3 tok/s on average over code, prose and passage-copy generations,
against 22.9 tok/s without the drafter and 25.8 tok/s for llama.cpp's
`draft-dspark` on the same prompts ([measurements](../benchmarks.md#minimax-m31-flash-on-eight-a40-gpus)).

Verification and one-token decode round differently, so on the real model the
speculative and plain greedy texts diverge after 50-130 tokens, each time at a
position where plain decoding's top two tokens were within 0.035 logits; the
same happens in llama.cpp. On the synthetic fixture, speculative greedy output
equals plain greedy output (`minimax-m3-dspark-{cpu,cuda,cuda-multi}`), and
on two fixture prompts TensorSharp's speculative output matched llama.cpp's
for 200 of 200 tokens, with 72.3% against 72.2% of drafted tokens accepted.

## Chat format, reasoning, tools and JSON

- **Template.** TensorSharp renders MiniMax-M3.1's `chat_template.jinja` with
  its own renderer
  ([ChatTemplate.MiniMaxM3.cs](../../TensorSharp.Runtime/ChatTemplate.MiniMaxM3.cs)),
  byte for byte against Hugging Face renders
  (`InferenceWeb.Tests/Fixtures/MiniMaxM3/chat_reference.json`), because the
  template's recursive macros are beyond the lightweight Jinja engine. A
  leading `root` message replaces the identity line, the first system or
  developer message is the developer turn, and consecutive tool results share
  one tool turn. `current_date`, `current_location` and `model_identity` have
  no request field.
- **Reasoning.** The template has no thinking switch: it tells the model
  "Current thinking mode: adaptive", so by default the model decides per turn
  whether to open `<mm:think>`. `"think": true` (or `--think` on the CLI) opens
  the block in the generation prompt. There is no way to turn reasoning off.
  Reasoning is returned separately from the answer, and every past assistant
  turn re-renders its reasoning, so a follow-up turn extends the cached prefix.
  `reasoning_effort` (`low` / `medium` / `high`; `reasoning.effort` on the
  Responses API) is rendered as `<effort>...</effort>` in the system block.
- **Tools.** The model calls tools as namespaced XML
  (`]<]minimax[>[<tool_call>` ... `<invoke name="f">` ... `</tool_call>`), which
  [MiniMaxM3OutputParser](../../TensorSharp.Runtime/MiniMaxM3OutputParser.cs)
  turns into structured `tool_calls`, typing each argument from the declared
  JSON Schema as llama.cpp's `peg-minimax-m3` parser does. On
  `/v1/chat/completions`, `tool_choice: "required"` or a named function is
  enforced by a grammar that arms after the reply's `</mm:think>`;
  `parallel_tool_calls: false` allows one call. `auto` and `none` are
  unconstrained. A required tool call cannot be combined with
  `response_format`.
- **JSON mode.** `response_format` `json_object` / `json_schema` arms its
  grammar after `</mm:think>`, so the model can still reason first; the answer
  is raw JSON without a code fence.
- **Tokenizer.** BPE with the `minimax-m2` pre-tokenizer and NFC
  normalization, checked against Hugging Face tokenizations
  (`InferenceWeb.Tests/Fixtures/MiniMaxM3/tokenizer_reference.json`). No BOS is
  added: the template writes `]~!b[` itself, and `[e~[` ends a turn.

## Images

Images arrive as OpenAI `image_url` data URIs, Ollama `images`, Web UI uploads
or CLI `--image`, and need the converted `mmproj`. Preprocessing is a port of
the Hugging Face processor (smart resize to multiples of 28 pixels, 4 to 576
tokens per image, at most 672×672 pixels' worth). The 32 ViT blocks run on GPU
0 as one device graph through the shared Qwen-VL encoder graph, and the
projector and 2×2 patch merge as a second one; the rows replace the image
placeholders at their absolute positions inside the native executor, so an
image span survives prefill chunking. If GPU 0 runs short, reserve room with
`TS_MINIMAX_M3_DEV0_RESERVE_MB` (the measured `--tp 8` image run used 2560 and
an 8,192-token context). Batched decode and batched prefill step aside while
image rows are pending, and prefix reuse stops at an image span.

There is no video encoder. A `video_url` part is sampled into timed frames
(`fps` and `max_frames` per part, or `VIDEO_SAMPLE_FPS` / `VIDEO_MAX_FRAMES`)
that each go through the image path with a time marker; video input was not
measured on the real model. Audio is refused: the release has no audio tower.

The encoder is checked against Hugging Face transformers' MiniMax-M3 vision
tower and projector on the real `mmproj` (`MiniMaxM3VisionOracleTests`, gated
on `TS_TEST_MINIMAX_M3_MMPROJ` and `TS_TEST_MINIMAX_M3_VISION_ORACLE`: mean
cosine at least 0.99999, relative L2 at most 0.005).

## Options

The executor's settings are environment variables; the table lists the ones
most deployments touch. The full list of every `TS_MINIMAX_M3_*` variable, with
defaults, is in the
[environment variable matrix](../env_var_feature_matrix.md#out-of-matrix-minimax-m31-knobs).

| Setting | Default | Effect |
|---|---|---|
| `TS_MINIMAX_M3_MSA` | `auto` | Attention path: `auto`, `dense`, `ggml` or `kernel` ([modes](#attention-modes)) |
| `TS_MINIMAX_M3_PLAN_CTX_TOKENS` | the context | K/V tokens the placement reserves across all slots |
| `TS_MINIMAX_M3_CPU_MOE=auto` | off | Place only the expert stacks that do not fit on the host; overrides `--n-cpu-moe` (layer split only) |
| `TS_MINIMAX_M3_LAYER_SPLIT` | automatic | Layers per GPU, comma-separated, summing to 60; refused with `--tp` |
| `TS_MINIMAX_M3_UBATCH` | `2048` | Prefill micro-batch rows |
| `TS_MINIMAX_M3_DEV0_RESERVE_MB` | `0` | Extra memory kept free on GPU 0, e.g. for the vision encoder |
| `TS_MINIMAX_M3_PIPELINE` | automatic | Pipelined prefill on a layer split: `1` forces it, `0` turns it off |
| `TS_MINIMAX_M3_INDEXER_F16` | on with GPUs | F16 copies of the F32 indexer projections for decode-shaped graphs |

## Measured results

Measured on 2026-10-24 on one host with eight A40 46 GB GPUs, against
llama.cpp from unslothai `minimax-m31` (`ef8f94cc`), with the host otherwise
idle, three repeats and medians. Ratios above 1.00× mean TensorSharp is better.

| Workload | TensorSharp | llama.cpp | Ratio |
|---|---:|---:|---:|
| Prefill, 4K-token prompt, server, layer split (tok/s) | 1,263 | 283 | **4.46×** |
| Prefill, 32K-token prompt, server, layer split (tok/s) | 1,544 | 240 | **6.43×** |
| Decode at 4K depth, server, layer split (tok/s) | 23.9 | 19.2 | **1.24×** |
| Decode at depth 0, executor bench vs `llama-bench` (tok/s) | 24.8 | 22.5 | **1.10×** |
| Time to first token, short chat (s) | 0.41 | 0.39 | 0.95× |
| Eight concurrent requests, aggregate output (tok/s) | 53.7 | 40.6 | **1.32×** |
| DSpark speculative decode, default gate (tok/s) | 32.3 | 25.8 | **1.25×** |
| Decode at depth 0, `--tp 8` (tok/s) | 35.7 | no tensor split for this model | |

The [benchmarks page](../benchmarks.md#minimax-m31-flash-on-eight-a40-gpus) has
every cell, the method and the limits. Full comparison report:
[MiniMax-M3.1-Flash: TensorSharp vs llama.cpp](../perf/minimax-m3-vs-llama-cpp.md).

## Validation

- **Unit and fixture tests.** `InferenceWeb.Tests/MiniMaxM3*.cs` cover the
  protocol, tokenizer, output parser, tool grammar, image preprocessing and
  prompt wiring, load refusals, and the model on synthetic GGUFs written by
  `MiniMaxM3SyntheticModelBuilder` and `eng/minimax-m3-fixture.py`. The gated
  ones need `TS_TEST_MINIMAX_M3_FIXTURE`, `TS_TEST_MINIMAX_M3_GGUF`,
  `TS_TEST_MINIMAX_M3_TP`, `TS_TEST_MINIMAX_M3_MMPROJ` or
  `TS_TEST_MINIMAX_M3_VISION_ORACLE` with `TS_TEST_GGML_BACKEND`; a skipped
  gated test is not a pass.
- **Native tests.** `ctest -R "minimax-m3|minimax-msa"` runs the executor
  (`minimax-m3-exec-*`, on the CPU, one and several CUDA GPUs, host experts and `--tp`: a sequence's
  logits do not depend on how it is fed or placed), the sparse-attention graphs
  against the reference math (`minimax-m3-msa-graph-*`), the MSA
  kernels (`minimax-msa-kernels-cuda`), the TP split plan
  (`minimax-m3-tp-split-plan`) and DSpark
  (`minimax-m3-dspark-*`).
- **Parity with llama.cpp.**
  [eng/validation/minimax-m3-parity-suite.sh](../../eng/validation/minimax-m3-parity-suite.sh)
  compares TensorSharp's logits with llama.cpp's on the synthetic `small` and
  `real-block` fixtures (random weights; `real-block` has the released MSA
  geometry). On `real-block`, TensorSharp's mean KL divergence from llama.cpp
  was 0.00127, below llama.cpp's own floor of 0.00139 between micro-batch 512
  and 1; `--tp 2` / `--tp 4` gave 0.00123 / 0.00126. This is fixture-level
  agreement, not a full-checkpoint quality measurement.
- **Real model.** The functional scenarios of
  [eng/minimax-m3-bench.py](../../eng/minimax-m3-bench.py) (checkable answers,
  needle-in-a-haystack prompts up to 64K tokens, JSON object and schema, single
  and parallel tool calls, an agentic loop, three-turn KV reuse, images,
  concurrent requests) passed under the layer split and, for text and images,
  under `--tp 8`. CPU expert offload and video were not measured on the real
  model. The raw records are local validation evidence on the measurement host,
  not committed (`/workspace/b1/`, `/workspace/b2/`, `/workspace/b3/real/`).

## Current limits

- Only `ggml_cuda` serves; `ggml_cpu` is a correctness path, and every other
  backend is refused.
- `--tp` mirrors attention and the caches, so it holds less context per GPU
  (45,824 tokens on eight A40s), its decode lead shrinks past 2K depth, and its
  long-prompt prefill is slower than the layer split's; it runs neither
  speculation nor host experts.
- DSpark runs only on a layer split, for solo requests.
- The thinking-disabled prompt (`</mm:think>` prefix) has no request field.
- Real-model quality was checked by functional scenarios and fixture-level
  logit parity, not by a full-checkpoint perplexity or KL comparison.
