# Muse Spark 816B: TensorSharp vs llama.cpp on ten A40 GPUs

This report compares TensorSharp with llama.cpp on Unsloth's
`Muse-Spark-Open-816B-A42B` (UD-IQ3_XXS, 317 GiB) across ten A40 GPUs, in both
multi-GPU modes each engine offers:

- **Layer split.** Whole layers per GPU: TensorSharp's `--layer-split 10`
  against llama.cpp's `-sm layer` (its default).
- **Tensor parallelism.** Every layer split across all ten GPUs: TensorSharp's
  `--tp 10` against llama.cpp's `-sm tensor`.

**Provenance:** this report was refreshed on 2026-10-22. The measurement date
and a frozen TensorSharp benchmark commit were not recorded; the setup below
identifies the working-tree base and changes that were measured.

The same checkpoint, machine and requests were used for every run. Each engine
ran alone on the machine, one after the other. Ratios above 1.00x mean
TensorSharp is better: higher throughput or lower latency.

## Summary

Results depend on the workload and multi-GPU mode. Layer-split single-request
decode is close (0.99x), and its four-request aggregate throughput is 1.02x
despite lower per-request decode throughput (0.91x). The prefill and
tensor-parallel results below show larger differences on this host.

- **Layer split** is the mode for long prompts. Prefill runs at 1,000 tok/s on
  a 4K prompt and 1,347 tok/s on a 16K prompt, 2.07x and 1.63x llama.cpp,
  because TensorSharp keeps all ten GPUs busy on one prompt (pipelined
  prefill, with micro-batches sized per call).
- **Tensor parallelism** is the mode for generation. It decodes at
  22.6 tok/s, 1.40x llama.cpp's tensor split and 1.35x either engine's layer
  split, and serves four concurrent requests at 45.7 tok/s (1.87x). Its
  prefill (295 tok/s) is 3.8–3.9x llama.cpp's tensor split but below either
  engine's layer split.
- **Quality.** On teacher-forced wikitext-2, TensorSharp's layer-split
  next-token distributions differ from llama.cpp's less than llama.cpp's own
  differ between two micro-batch sizes; tensor parallelism is at that noise
  floor.

| Scenario | TensorSharp layer split | llama.cpp layer split | Ratio | TensorSharp `--tp 10` | llama.cpp `-sm tensor` | Ratio |
|---|---:|---:|---:|---:|---:|---:|
| Decode, one request (tok/s) | 16.69 | 16.79 | 0.99x | **22.63** | 16.19 | **1.40x** |
| Time to first token, short prompt (s) | 0.64 | 0.75 | **1.17x** | 0.58 | 1.29 | **2.22x** |
| Prefill, 4K-token prompt (tok/s) | **999.7** | 483.2 | **2.07x** | 294.4 | 75.9 | **3.88x** |
| Prefill, 16K-token prompt (tok/s) | **1346.8** | 826.1 | **1.63x** | 295.2 | 77.9 | **3.79x** |
| Four concurrent requests, total output (tok/s) | 33.65 | 33.04 | **1.02x** | **45.73** | 24.49 | **1.87x** |
| Second turn of a 2.3K-token conversation, first token (s) | 0.45 | 0.91 | **2.00x** | 0.46 | 2.15 | **4.67x** |
| New conversation sharing a 2.3K-token system prompt, first token (s) | 0.48 | 0.64 | **1.34x** | 0.51 | 1.43 | **2.80x** |
| Answer-checking fixtures | 13 / 14 | 12 / 14 | | **14 / 14** | 12 / 14 | |
| Load time | 6.0 min | 14.4 min | **2.4x** | 6.4 min | 14.7 min | **2.3x** |

Before this work, TensorSharp could not run either command on this machine
(see [What changed in TensorSharp](#what-changed-in-tensorsharp)), and its
earlier `--tp` path decoded at 1.9–2.4 tok/s.

## Setup

| | TensorSharp | llama.cpp |
|---|---|---|
| Source | working tree on top of main `e2a84893`; the changes are on branch `feature/muse_spark_tp_bug_fix` (`192295a9` and later edits) | `7170cad` (`version: 0.7.0-dev`), unmodified |
| ggml | upstream `da8e2a5`, unmodified | in-tree |
| Build | CUDA 12.8, `sm_86`, NCCL 2.25.1 | CUDA 12.8, `-DGGML_CUDA_NCCL=OFF` (see [Limits](#limits-of-this-comparison)) |
| Model | `unsloth/Muse-Spark-Open-816B-A42B-GGUF`, UD-IQ3_XXS, 8 shards (340 GB), 315.7 GiB of weights | same files |
| Context | 32,768 tokens; five planned slots (four requests plus the shared-prefix checkpoint) | `-c 32768 -np 4 --kv-unified` |
| KV cache | F16 | F16 |
| Attention | flash attention | `-fa on` |
| Experts | all in GPU memory | all in GPU memory (`-ngl 999`) |

Hardware: ten NVIDIA A40 (46 GB, PCIe 4.0 x16, driver 595.91.07) in one
container, five GPUs under each of two Xeon Gold 6342 sockets (96 threads),
503 GB of RAM. The model files sit on a network filesystem. Peer-to-peer
access between the GPUs is advertised by the driver but does not deliver data
(a probe copy returns zeros), so no engine used P2P.

Launch commands:

```bash
# TensorSharp (add --tp 10 instead of --layer-split 10 for tensor parallelism)
MAX_CONTEXT=32768 TS_MUSE_SPARK_PLAN_SLOTS=5 TS_CODE_EXEC=0 \
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll --model <shard 1> \
  --backend ggml_cuda --layer-split 10 --no-webui --no-skills --no-multi-agent

# llama.cpp (add -sm tensor for tensor parallelism)
llama-server -m <shard 1> --jinja -fa on -c 32768 -np 4 --kv-unified -ngl 999 \
  --chat-template-file llama-matched-template.jinja
```

The llama.cpp template comes from
`eng/validation/prepare_muse_spark_llama_reference.py`. It preserves the
released template and matches TensorSharp's date and answer-prefix handling.

## Method

[`eng/muse-spark-bench.py`](../../eng/muse-spark-bench.py) sends the same
OpenAI-compatible streaming requests to each server. Requests are greedy with
thinking disabled. Decode rate excludes the first token:
`(completion_tokens - 1) / (last_token_time - first_token_time)`. Prefill rate
is uncached prompt tokens divided by time to first token (TTFT). Every number
is the median of three repeats, except llama.cpp's tensor-split rows, which
are medians of two (each of its 16K prompts takes over three minutes).

| Scenario | What it measures |
|---|---|
| Short prompt | A ~167-token prompt and 256 generated tokens: decode speed |
| Long prompt 4K / 16K | A document with one fact in the middle, then a question about it; output capped at 32 tokens: prefill speed and the answer |
| 4 concurrent requests | Four short prompts at once, 256 tokens each: aggregate throughput |
| Multi-turn KV reuse | A ~2.3K-token system prompt, then a second turn of the same conversation |
| Shared system prompt | Four new conversations that share one ~2.3K-token system prompt (an agent pattern) |

[`eng/muse-spark-validate.py`](../../eng/muse-spark-validate.py) checks
answers in 14 scenarios: short English and Chinese answers, a fixed 512-token
decode, JSON, JSON schema, Unicode JSON, an 8K-token retrieval, multi-turn
memory, a tool round trip, typed tool arguments, a dependent agent workflow,
parallel isolation, cache reuse and staggered arrivals. No projector was
loaded, so the two image scenarios of the earlier report were not run.

Teacher-forced quality uses
[`eng/validation/muse-spark-ppl-kld.cpp`](../../eng/validation/muse-spark-ppl-kld.cpp)
on identical token IDs (no tokenizer, template or sampling differences): the
first 24,564 tokens of the wikitext-2 test set, cut into 12 chunks of BOS plus
2,047 tokens, each evaluated from an empty cache, scoring positions 1,024 to
2,046 (12,276 next-token predictions). It computes perplexity, the KL
divergence of each engine's distribution from llama.cpp's, and top-1
agreement, in the manner of llama-perplexity's `--kl-divergence`.

The raw files are kept in the ignored `docs/validation/` directory (local
validation evidence, not committed).

## Performance

### Layer split (`--layer-split 10` against `-sm layer`)

| Scenario | Metric | TensorSharp | llama.cpp | Ratio |
|---|---|---:|---:|---:|
| Short prompt (167 tokens), 256 generated | decode tok/s | 16.69 | 16.79 | 0.99x |
| | TTFT s | 0.64 | 0.75 | **1.17x** |
| Long prompt (4,075 tokens) | prefill tok/s | 999.7 | 483.2 | **2.07x** |
| | TTFT s | 3.95 | 8.43 | **2.13x** |
| | fact found | 3/3 | 3/3 | |
| Long prompt (15,767 tokens) | prefill tok/s | 1346.8 | 826.1 | **1.63x** |
| | TTFT s | 11.61 | 19.09 | **1.64x** |
| | fact found | 3/3 | 3/3 | |
| 4 concurrent requests, 256 tokens each | aggregate tok/s | 33.65 | 33.04 | **1.02x** |
| | per-request decode tok/s | 8.54 | 9.37 | 0.91x |
| | TTFT s | 1.35 | 3.78 | **2.80x** |
| Multi-turn reuse (2.3K-token prefix) | turn-1 TTFT s | 3.47 | 6.76 | **1.95x** |
| | turn-2 TTFT s | 0.45 | 0.91 | **2.00x** |
| | turn-2 cached tokens | 2,343 | 2,342 | |
| New conversations sharing a 2.3K-token system prompt | first TTFT s | 3.43 | 6.33 | **1.85x** |
| | later TTFT s | 0.48 | 0.64 | **1.34x** |

Layer-split decode is the same on both engines: one token passes through the
ten GPUs one after another, and both spend about 60 ms per token. In the
four-request scenario llama.cpp decodes each request slightly faster, but
TensorSharp's requests start 2.4 s earlier, so the totals are level.

### Tensor parallelism (`--tp 10` against `-sm tensor`)

| Scenario | Metric | TensorSharp | llama.cpp | Ratio |
|---|---|---:|---:|---:|
| Short prompt (167 tokens), 256 generated | decode tok/s | 22.63 | 16.19 | **1.40x** |
| | TTFT s | 0.58 | 1.29 | **2.22x** |
| Long prompt (4,075 tokens) | prefill tok/s | 294.4 | 75.9 | **3.88x** |
| | TTFT s | 13.41 | 53.69 | **4.00x** |
| | fact found | 3/3 | 2/2 | |
| Long prompt (15,767 tokens) | prefill tok/s | 295.2 | 77.9 | **3.79x** |
| | TTFT s | 52.98 | 202.50 | **3.82x** |
| | fact found | 3/3 | 2/2 | |
| 4 concurrent requests, 256 tokens each | aggregate tok/s | 45.73 | 24.49 | **1.87x** |
| | per-request decode tok/s | 12.22 | 9.00 | **1.36x** |
| | TTFT s | 1.52 | 13.84 | **9.11x** |
| Multi-turn reuse (2.3K-token prefix) | turn-1 TTFT s | 8.40 | 37.87 | **4.51x** |
| | turn-2 TTFT s | 0.46 | 2.15 | **4.67x** |
| New conversations sharing a 2.3K-token system prompt | first TTFT s | 8.48 | 32.81 | **3.87x** |
| | later TTFT s | 0.51 | 1.43 | **2.80x** |

Both engines split each layer the same way, the way llama.cpp's `-sm tensor`
does: attention by KV group (16 groups over 10 GPUs gives each GPU one or
two, rotating per layer), routed experts along their hidden dimension, and the
rest copied to every GPU. Each layer then ends in two all-reduces. The
difference is the all-reduce. llama.cpp was built without NCCL (an NCCL build
aborts on this machine, see the limits), so it combines partial results
through its peer-copy butterfly: 28 driver-staged copies and 26 single-node
add graphs per all-reduce. TensorSharp runs NCCL over shared memory
(`NCCL_P2P_DISABLE=1`, chosen automatically because P2P is broken here): about
175 µs for a decode-sized all-reduce. Prefill is bound by that transport's
bandwidth (about 1.8 GB/s for a large all-reduce, measured with a standalone
NCCL benchmark). Muse Spark normalizes each expert's output before summing
the experts, so the MoE all-reduce carries every selected expert's output, 8
times the size of the summed result. That holds TensorSharp's tensor-parallel
prefill near 295 tok/s, and layer split is the faster mode for long prompts.
llama.cpp's four-request TTFT varied between its two repeats (9.7 s and
18.0 s); both are well above TensorSharp's 1.5 s.

## Answer quality

### Teacher-forced next-token distributions

All rows score the same 12,276 positions. KL divergence is KL(llama.cpp ‖
engine) over the full 202,048-token vocabulary, against llama.cpp's
512-token-micro-batch run.

| Run | Perplexity | KLD mean | KLD median | KLD p99 | KLD max | Top-1 agreement |
|---|---:|---:|---:|---:|---:|---:|
| llama.cpp, 512-token micro-batches (reference) | 1.847196 | | | | | |
| llama.cpp, 2048-token micro-batches (noise floor) | 1.847222 | 0.002818 | 0.000154 | 0.0445 | 0.764 | 98.62% |
| TensorSharp layer split | 1.848177 | 0.002590 | 0.000151 | 0.0377 | 0.299 | 98.84% |
| TensorSharp layer split, prefix written by pipelined prefill | 1.848177 | 0.002590 | 0.000151 | 0.0377 | 0.299 | 98.84% |
| TensorSharp `--tp 10` | 1.848249 | 0.003061 | 0.000170 | 0.0407 | 1.641 | 98.82% |

The noise floor row is llama.cpp against itself: the same model and tokens
with only the micro-batch size changed. TensorSharp's layer split stays below
it on every divergence statistic. The pipelined row prefills each chunk's
unscored first half through the pipelined prefill (12 calls, 24
micro-batches), so the scored rows read the K/V that path wrote. It matches the sequential row
exactly. Tensor parallelism sits at the noise floor: its mean divergence is
9% above it and its maximum is higher (one position at 1.64), while its p99
and top-1 agreement are better. ggml's NCCL path sums the MoE partials of
prefill-sized micro-batches in BF16 (see the limits); against TensorSharp's
own layer split, the tensor-parallel run differs by a mean KLD of 0.0029.

### Answer-checking fixtures

Each scenario ran once per engine.

| Scenario | TensorSharp layer split | llama.cpp layer split | TensorSharp `--tp 10` | llama.cpp `-sm tensor` |
|---|---|---|---|---|
| Short answer (English, Chinese) | pass, pass | pass, pass | pass, pass | pass, pass |
| Fixed 512-token decode | pass | pass | pass | pass |
| JSON, JSON schema, Unicode JSON | pass, pass, pass | pass, pass, pass | pass, pass, pass | pass, pass, pass |
| 8K-token retrieval | pass | pass | pass | pass |
| Multi-turn memory | pass | pass | pass | pass |
| Tool round trip, typed tool arguments | pass, pass | pass, pass | pass, pass | pass, pass |
| Dependent agent workflow | pass | pass | pass | pass |
| Parallel request isolation | pass | pass | pass | pass |
| Cache reuse | pass | **fail**: the second turn did not reuse the whole first prompt | pass | **fail**: the second turn did not reuse the whole first prompt |
| Staggered arrivals | **fail**: one request stopped after 322 of 512 tokens | **fail**: stopped after 409 of 512 tokens | pass | **fail**: stopped after 402 of 512 tokens |
| **Total** | **13 / 14** | **12 / 14** | **14 / 14** | **12 / 14** |

The staggered-arrivals fixture requires each of several requests that start
at different times to generate exactly its 512-token budget. Three of the
four runs produced a coherent answer that ended with an end-of-text token
before the budget, a greedy-decoding branch rather than an engine fault; the
same prompt passed under TensorSharp's tensor parallelism, whose arithmetic
order differs. The cache-reuse fixture checks that the second turn's cached
tokens cover the whole first turn; llama.cpp's covered less in both modes.
The fixtures check integration, not general model quality.

## What changed in TensorSharp

The two commands that prompted this work,
`TensorSharp.Cli --backend ggml_cuda --model <shard 1> --tp 10` and the same
with `--layer-split 10`, both failed on this machine before it:
`--tp 10` aborted while allocating GPU scratch memory, and `--layer-split 10`
was refused because the model "did not fit 8 GPUs" (the executor spanned at
most eight), and with that limit raised it hit the same abort. The fixes and the
optimizations behind the tables above, all in TensorSharp-owned code with
ggml unmodified:

- **Ten or more GPUs (both modes).** In an NCCL build of ggml, ggml-cuda
  shares each kernel-scratch pool chunk with every peer-capable GPU in one
  `cuMemSetAccess` call, and CUDA allows a GPU at most 8 peer connections. A
  TensorSharp wrapper (`ggml_ops_cuda_vmm_peer_guard.cpp`, linked with
  `--wrap=cuMemSetAccess`) catches the refusal, revokes the partial grant,
  keeps that pool private to its GPU, and tells NCCL not to use its in-process
  P2P direct mode. llama.cpp's NCCL build has the same abort on this machine.
- **More than 8 GPUs in the native executor.** It now spans up to 16 GPUs and
  refuses a count it cannot honour instead of running on fewer. An
  advertised context that does not fit (the GGUF advertises 1,048,576 tokens)
  is capped to the largest that fits unless `MAX_CONTEXT` is set, and the
  layer placement fills the GPUs evenly.
- **Pipelined prefill (layer split).** One prompt call is cut into equal
  micro-batches, and GPU k works on micro-batch j while GPU k+1 works on j−1,
  as llama.cpp's pipeline parallelism does. TensorSharp sizes the
  micro-batches per call: about a quarter of the call, between 512 rows and
  the 2,048-row prefill ubatch. A 16K prompt in one call took 15.3 s with
  llama.cpp's fixed 512 rows and 10.2 s this way; a 4K call took 4.5 s and
  3.7 s.
- **Lone-prompt steps.** When one request is alone, the scheduler lets the
  model size its prefill steps: 8,192 tokens for Muse Spark, with a short
  remainder joined to the step before it. Every call fills and drains the
  pipeline once, so a 4,096-token call ran at about 70% of a 16K call's
  throughput and an 8,192-token call at about 90%.
- **Native tensor parallelism.** `--tp N` on `ggml_cuda` now runs the native
  whole-model executor on upstream ggml's meta backend, the mechanism behind
  llama.cpp's `-sm tensor`. TensorSharp supplies the split plan
  (`ggml_ops_muse_spark_tp.cpp`), chooses the all-reduce transport for the
  host, copies K/V rows on the GPUs, and refuses settings it cannot honour.
  The earlier managed per-operation path, which decoded at 1.9–2.4 tok/s,
  remains available with `TS_MUSE_SPARK_NATIVE=0`.

## Limits of this comparison

- **One machine.** All numbers come from one container on one host. The
  model files live on a network filesystem, which slows loads and makes their
  times vary. No confidence intervals are claimed.
- **No P2P.** The driver advertises peer access between these GPUs but peer
  copies deliver nothing, so both engines exchange data through host memory.
  On a machine with working P2P or NVLink, tensor-parallel results would be
  very different for both engines.
- **llama.cpp without NCCL.** llama.cpp 7170cad built with NCCL aborts at the
  first scratch-pool growth on ten GPUs, the same failure TensorSharp's guard
  fixes. Its tensor split therefore used its NCCL-free peer-copy butterfly.
  An NCCL build of llama.cpp with this fix applied could narrow the
  tensor-parallel decode gap; that configuration does not exist upstream
  today.
- **Precision of tensor-parallel reductions.** ggml's NCCL path reduces
  tensors above 262,144 elements in BF16. That covers TensorSharp's
  tensor-parallel prefill (the MoE partials of a micro-batch), not decode. The
  teacher-forced row above measures its effect.
- **llama.cpp tensor-split repeats.** Its tensor-split prefill runs at 76
  tok/s, so its rows are medians of two repeats instead of three, and its
  short and 4K rows come from an earlier run of the same build than its other
  rows.
- **Prompt cache policy.** Both engines reuse the chat template's fixed
  preamble (126–127 tokens). TensorSharp keeps unrelated conversations
  isolated except for public system-prompt and template prefixes, so the
  short and four-request scenarios number each repeat's prompts differently.
  Prefill rates count only uncached tokens.
- **Not covered.** Images (no projector loaded), speculative decoding (off
  under TensorSharp's tensor parallelism, which keeps one graph), CPU expert
  offload, contexts above 16K tokens, other GPU counts, Vulkan, ROCm and
  Windows. Thinking was disabled for every request.

## Reproducing

```bash
# Performance (same scenarios against each server)
python3 eng/muse-spark-bench.py --engine tensorsharp --url http://127.0.0.1:8083 \
  --out ts.json --scenarios short,long_4k,long_16k,parallel_4,cache_reuse,shared_prefix \
  --repeats 3 --decode-tokens 256
python3 eng/muse-spark-bench.py --engine llamacpp --url http://127.0.0.1:8082 \
  --out llama.json --scenarios short,long_4k,long_16k,parallel_4,cache_reuse,shared_prefix \
  --repeats 3 --decode-tokens 256

# Answer checks
python3 eng/muse-spark-validate.py --engine tensorsharp --url http://127.0.0.1:8083 \
  --model <served model id> --weights-id unsloth/Muse-Spark-Open-816B-A42B-GGUF/UD-IQ3_XXS \
  --profile <name> --output ts-validate.json --repeats 1 \
  --scenarios short,short_zh,decode,json,json_schema,json_unicode,long_8k,multi_turn,tool_round_trip,tool_string_arguments,agentic,parallel,cache_reuse,staggered_requests

# Teacher-forced quality (build instructions in the file header)
ppl-kld-llama tokenize <shard 1> wiki.test.raw wt2.tokens 24564
ppl-kld-llama llama <shard 1> wt2.tokens 2048 llama.lp16
PPL_UBATCH=2048 ppl-kld-llama llama <shard 1> wt2.tokens 2048 llama-ub2048.lp16
ppl-kld-llama compare llama.lp16 llama-ub2048.lp16 wt2.tokens
ppl-kld-ts ts libGgmlOps.so <shard 1> wt2.tokens 2048 llama.lp16 10                    # layer split
PPL_PIPELINED=1 ppl-kld-ts ts libGgmlOps.so <shard 1> wt2.tokens 2048 llama.lp16 10    # pipelined prefix
PPL_TP=10 ppl-kld-ts ts libGgmlOps.so <shard 1> wt2.tokens 2048 llama.lp16 10          # tensor parallel
```

## Test coverage of this build

On the ten-A40 machine, against the unchanged ggml checkout:

- **Managed tests**, CUDA lane (Muse Spark, scheduler, prefix-cache
  configuration, tensor-parallel support gate and iOS export manifest
  suites): 307 passed, 20 skipped, none failed. The skips are CPU-lane tests;
  they pass on a CPU run.
- **Native tensor parallelism** against a double-precision reference decoder
  on 2, 4 and 10 real GPUs: a prefill whose K/V rows grow on the GPUs while
  the sliding-window ring wraps, decode, a rewind, batched decode over three
  sequences, and a prefix-checkpoint clone. At 10 GPUs the fixture's 16 KV
  groups split unevenly and rotate per layer, as the real model's do.
  Unsupported combinations are refused at load.
- **Pipelined prefill** against the same reference on one and two GPUs, with
  K/V windows that grow inside a call, a first GPU that holds only a global
  layer, and fixed and adaptive micro-batches.
- **Native unit tests:** the split plan (`muse-spark-tp-split-plan`) and the
  peer-limit guard (`cuda-vmm-peer-guard`, on 10 GPUs and on 8).
- **Not run on this build:** the full native ctest suite and the CPU-lane
  managed tests on the machine. An earlier run of the native suite on this
  branch passed 107 of 116; the 8 failures it had reproduce identically on the
  unmodified base commit.
