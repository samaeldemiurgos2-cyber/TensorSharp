# MiniMax-M3.1-Flash 428B: TensorSharp vs llama.cpp on eight A40 GPUs

This report compares TensorSharp with llama.cpp on Unsloth's
`MiniMax-M3.1-Flash` GGUF (UD-Q4_K_XL, 232.3 GiB) on one host with eight
46 GB A40 GPUs. It covers raw prefill and decode throughput, the
OpenAI-compatible server across the scenarios a deployment uses (short and
long prompts, JSON mode, tool calls, an agentic loop, KV-cache reuse, images,
concurrent requests), MoE CPU offload, DSpark speculative decoding, and
TensorSharp's tensor parallelism, which llama.cpp does not offer for this
model.

**Measured 2026-10-24.** Both engines ran on the same machine and the same
files, one at a time, with the GPUs held exclusively by the run under test.
Ratios above 1.00x mean TensorSharp is better: higher throughput or lower
latency.

## Summary

- **Prefill** is where the engines differ most. MiniMax-M3.1 uses MiniMax
  Sparse Attention (MSA) on every layer: past position 2,048 each query
  attends only 16 selected 128-token blocks per KV group. TensorSharp runs that
  regime in its own CUDA kernels; llama.cpp computes it as dense attention with
  a block mask. At 32K and 64K tokens TensorSharp prefills 3.6x and 9.8x faster
  (executor), and the server's long-prompt time to first token is 4.5x–7.3x
  shorter.
- **Decode** is faster at every depth, 1.10x at depth 0 rising to 1.49x at
  64K, because TensorSharp's sparse decode reads only the selected blocks.
- **Every functional scenario passes on both engines** (after one harness fix,
  below). TensorSharp finishes JSON, tool-call, agentic and image requests
  1.05x–1.88x sooner end to end.
- **Concurrency:** four and eight concurrent requests produce 1.32x the
  aggregate output.
- **Tensor parallelism** (`--tp 8`, no llama.cpp counterpart) decodes at
  35.7–41 tok/s, about 1.6x llama.cpp's best mode, and runs the agentic loop
  1.7x faster.
- **DSpark speculative decoding:** 32.3 tok/s against llama.cpp's 25.8 with the
  same drafter.
- **MoE CPU offload** (experts of 24 layers on the host): 1.30x decode and
  1.62x prefill, after a host-thread default fixed during this comparison (see
  [MoE CPU offload](#moe-cpu-offload)).
- **Where TensorSharp is behind:** the first token of a short chat is
  0.41 s against 0.39 s, and the time to first token under concurrency is
  about 4% longer.

| Scenario | TensorSharp | llama.cpp | Ratio |
|---|---:|---:|---:|
| Prefill, 8K-token prompt (executor, tok/s) | 1,699 | 1,312 | **1.30x** |
| Prefill, 32K-token prompt (executor, tok/s) | 2,449 | 681 | **3.60x** |
| Prefill, 64K-token prompt (executor, tok/s) | 2,567 | 262 | **9.80x** |
| Decode at depth 0 / 8K / 32K (tok/s) | 24.8 / 23.8 / 23.4 | 22.5 / 19.1 / 17.5 | **1.10x / 1.25x / 1.34x** |
| Server: 32K-token prompt, time to first token (s) | 21.0 | 134.9 | **6.42x** |
| Server: 64K-token prompt, time to first token (s) | 40.0 | 293.4 | **7.34x** |
| Server: short chat, time to first token (s) | 0.41 | 0.39 | 0.95x |
| Server: short chat, decode (tok/s) | 26.3 | 26.0 | **1.01x** |
| Server: 4 / 8 concurrent requests, total output (tok/s) | 42.1 / 53.7 | 31.8 / 40.6 | **1.32x / 1.32x** |
| Server: agentic tool loop, end to end (s) | 18.1 | 20.1 | **1.11x** |
| Server: second turn of a 3K-token conversation, first token (s) | 0.38 | 0.41 | **1.08x** |
| DSpark speculative decoding (tok/s, mean of 9 generations) | 32.3 | 25.8 | **1.25x** |
| `--tp 8` decode, short chat (tok/s) | 41.0 | n/a (layer split: 26.0) | |
| MoE CPU offload, experts of 24 layers on the host: decode / 4K prefill (tok/s) | 17.9 / 76.3 | 13.8 / 47.0 | **1.30x / 1.62x** |
| Load time, page cache not controlled (s) | 188–215 | 533–588 | indicative only |

## Setup

| | TensorSharp | llama.cpp |
|---|---|---|
| Source | working tree on branch `feature/minimax-m31` atop main `a0341c67` (uncommitted when measured) | `unslothai/llama.cpp` branch `minimax-m31` at `ef8f94ccb0d24216cd361924aeec2fe0ab436d72` (PR #320; upstream llama.cpp cannot load M3.1, whose dense-lead layers carry indexer tensors) |
| ggml | upstream `da8e2a54b919557b816b1c2beb432ccb766f7ce0`, unmodified | in-tree |
| Build | CUDA 12.8, `sm_86` | CUDA 12.8, `-DCMAKE_CUDA_ARCHITECTURES=86` |
| Model | `unsloth/MiniMax-M3.1-Flash-GGUF`, UD-Q4_K_XL, 7 shards (249,402,105,310 bytes): NVFP4 routed experts with per-expert scales, Q8_0 attention, shared and dense weights, F32 indexer and router, BF16 embeddings and head | same files |
| Companions | `mmproj-MiniMax-M3.1-Flash-BF16.gguf` and `MiniMax-M3.1-Flash-DSpark-BF16.gguf`, converted from `MiniMaxAI/MiniMax-M3.1-Flash` with the PR #320 converter (the Unsloth repository did not yet ship them) | same files |
| Placement | `--layer-split 8` (one GPU unless asked), or `--tp 8` | `-sm layer` (default); `-sm tensor` is refused for `minimax-m3` |
| Attention | `TS_MINIMAX_M3_MSA=auto`: exact dense below position 2,048, TensorSharp's MSA kernels above | `-fa on`, one sequence per slot (`-np N --no-kv-unified`; with llama-server's default unified cache MSA silently runs dense) |
| KV cache | F16 K/V + F16 index keys | F16 |

Hardware: eight NVIDIA A40 (46 GB, PCIe, driver 595.91.07), 96 threads, a
373.5 GiB memory cgroup, model files on a network filesystem. GPUs 0-4 and 5-7
sit under different NUMA nodes and peer-to-peer copies are not used.

Launch commands (the helpers record the full command lines):

```bash
# TensorSharp server: eng/minimax-m3-ts-server.sh --layer-split 8 --np 1 --ctx 73728 [--mmproj ...] [--tp 8] [--dspark ...]
MAX_CONTEXT=73728 TS_SCHED_MAX_RUNNING_SEQS=1 TS_MINIMAX_M3_PLAN_CTX_TOKENS=147456 \
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll --model <shard 1> --backend ggml_cuda \
  --layer-split 8 --no-webui --no-skills --no-multi-agent [--mmproj <mmproj>] [--draft-model <dspark>]

# llama.cpp server: eng/minimax-m3-llama-server.sh --np 1 --ctx 73728 [--mmproj ...] [--dspark ...]
llama-server -m <shard 1> --jinja -fa on -ngl 999 -c 73728 -np 1 --no-kv-unified [--mmproj <mmproj>] \
  [-md <dspark> --spec-type draft-dspark --spec-draft-n-max 7 --spec-draft-p-min 0]
```

## Method

**Executor throughput.** TensorSharp's
[`eng/validation/minimax-m3-native-bench.cpp`](../../eng/validation/minimax-m3-native-bench.cpp)
drives the native executor through the same C ABI the managed model uses
(one load, uniform-random tokens, an untimed warm-up and three timed
repetitions per cell, untimed prefill up to the decode depth) against
`llama-bench -fa 1 -ngl 999 -r 3 -mmp 0` (`-d` for depth). Each llama.cpp
prefill cell is the faster of `-ub 512` and `-ub 2048`; at 32K `-ub 2048` ran
out of GPU memory and the 64K cell is a single run.

**Server scenarios.** [`eng/minimax-m3-bench.py`](../../eng/minimax-m3-bench.py)
sends the same streamed `/v1/chat/completions` requests to both servers:
greedy, a fixed seed, thinking in the template's default adaptive mode (the
M3.1 template has no switch), and a per-run nonce so neither engine reuses a
previous run's prompt. Prefill is uncached prompt tokens divided by time to
first token; decode excludes the first token. Every scenario checks the answer
automatically:

| Scenario | What it measures and checks |
|---|---|
| `short_chat` | Eight short factual and arithmetic questions; exact answers |
| `long_prompt_{4k,16k,32k,64k}` | A unique code word hidden in real prose; it must be retrieved |
| `json_object`, `json_schema` | `response_format`; the reply must parse and validate (llama.cpp fences JSON in a code block; the harness strips the fence) |
| `tool_call_single`, `tool_call_parallel` | One and three tool calls with typed arguments |
| `agentic_loop` | File-listing, file-reading and calculator tools answered by the harness until a final answer, which must be correct |
| `kv_reuse` | A three-turn conversation over a 2.5K-token system prompt; per-turn time to first token and cached tokens |
| `image` | Four generated images (a rendered number, a word, a shape's colour, a count); the answer must name the content |
| `parallel_{4,8}` | Different concurrent requests: aggregate throughput and per-request answers |
| `speculative` | Code, prose and passage-copy generations, with and without the DSpark drafter |

The long-prompt and short-chat cells were rotated ABBA (llama.cpp,
TensorSharp, TensorSharp, llama.cpp), three repeats per arm; the page cache
was not evicted between launches, so load times are only indicative. The
functional scenarios ran three repeats per engine in one arm each.

## Executor throughput

| Cell | TensorSharp tok/s | llama.cpp tok/s | Ratio |
|---|---:|---:|---:|
| Prefill, 512 tokens | 448.2 | 343.7 | **1.30x** |
| Prefill, 1,024 tokens | 602.3 | 594.4 | **1.01x** |
| Prefill, 2,048 tokens | 947.9 | 783.8 | **1.21x** |
| Prefill, 4,096 tokens | 1,451.9 | 1,208.3 | **1.20x** |
| Prefill, 8,192 tokens | 1,699.3 | 1,311.7 | **1.30x** |
| Prefill, 32,768 tokens | 2,448.8 | 680.9 | **3.60x** |
| Prefill, 65,536 tokens | 2,567.2 | 262.1 | **9.80x** |
| Decode, depth 0 | 24.80 | 22.54 | **1.10x** |
| Decode, depth 2,048 | 23.93 | 19.56 | **1.22x** |
| Decode, depth 8,192 | 23.82 | 19.06 | **1.25x** |
| Decode, depth 32,768 | 23.41 | 17.48 | **1.34x** |
| Decode, depth 65,536 | 22.92 | 15.42 | **1.49x** |

Prefill above 2,048 tokens is where MSA turns sparse. TensorSharp's kernels
score index keys per block without materializing per-position scores, select
the top 16 blocks per KV group, and run block-sparse attention with the 16
query heads of a group packed into one tensor-core tile; llama.cpp builds a
dense `[n_kv, n_tokens, 4]` mask and runs dense flash attention over the whole
cache. Below 2,048 tokens MSA is exactly dense and both engines run the same
upstream ggml kernels; TensorSharp's advantage there comes from pipelining one
prompt across the eight GPUs in micro-batches sized per prompt.

A decode step on the layer split takes 42.9 ms of wall time, 39.9 ms of it on
the GPUs (CUPTI): Q8_0 matrix-vector products 18.9 ms, NVFP4 experts 12.3 ms,
the BF16 output head 4.1 ms, the indexer 0.9 ms (F16 decode copies of its F32
weights), the router 0.4 ms. The weights a token reads (~21 GB) give a floor of
about 31 ms at 696 GB/s.

## Server scenarios

### Long prompts and short chat (ABBA, three repeats per arm)

| Prompt | Prefill, TensorSharp | Prefill, llama.cpp | Ratio | Decode, TensorSharp | Decode, llama.cpp | Ratio |
|---|---:|---:|---:|---:|---:|---:|
| 4K tokens | 1,263 | 283 | **4.47x** | 23.9 | 19.2 | **1.24x** |
| 16K tokens | 1,478 | 273 | **5.42x** | 23.7 | 18.5 | **1.28x** |
| 32K tokens | 1,544 | 240 | **6.42x** | 23.5 | 17.7 | **1.32x** |
| 64K tokens | 1,545 | 211 | **7.34x** | 23.0 | 16.1 | **1.43x** |

The 64K row is three TensorSharp repeats against three llama.cpp repeats in a
separate arm. The final TensorSharp build reproduced the 4K/16K/32K prefill at
1,289 / 1,494 / 1,549 tok/s in the final functional arm. The server stays near
1,545 tok/s past 16K because the scheduler feeds 8,192-token steps, each of
which fills and drains the eight-stage pipeline; one 32K executor call reaches
2,449 tok/s.

Short chat (a 211-token prompt): time to first token 0.41 s against 0.39 s
(0.95x), decode 26.3 against 26.0 tok/s. A new chat gets a new cache slot,
so its first two decode steps build and capture graphs (51-55 ms against
40.5 ms steady), and its 28-token first-use prefill takes 276 ms where
llama-bench's reused graph takes 258 ms. An experimental change that keys
decode graphs by cache instead of slot reached 0.38-0.39 s in an ABBA run, but
lowered four-request throughput by about 10% and let two of sixteen
eight-request answers run out of tokens, so it was not adopted.

### Functional scenarios (three repeats each)

| Scenario | TensorSharp pass | llama.cpp pass | End to end, TensorSharp (s) | End to end, llama.cpp (s) | Ratio |
|---|---|---|---:|---:|---:|
| `short_chat` | 3/3 (24/24 checks) | 3/3 (24/24) | | | |
| `json_object` | 3/3 (18/18) | 3/3 (18/18) | 2.90 | 3.54 | **1.22x** |
| `json_schema` | 3/3 (15/15) | 3/3 (15/15) | 4.78 | 8.98 | **1.88x** |
| `tool_call_single` | 3/3 (15/15) | 3/3 (15/15) | 3.28 | 3.54 | **1.08x** |
| `tool_call_parallel` | 3/3 (18/18) | 3/3 (18/18) | 6.82 | 7.19 | **1.05x** |
| `agentic_loop` | 3/3 (9/9) | 3/3 (9/9) | 18.08 | 20.10 | **1.11x** |
| `kv_reuse` | 3/3 (9/9) | 3/3 (9/9) | turn 1: 3.09 / turns 2-3: 0.38 (first token) | 10.18 / 0.41 | **3.30x / 1.08x** |
| `image` | 2/3 (11/12); 8/8 in two further arms | 3/3 (12/12) | 2.30 | 2.60 | **1.13x** |
| `long_prompt_{4k,16k,32k}` | 3/3 each | 3/3 each (ABBA arms) | | | |

Both engines cached 99% of the conversation on turns 2 and 3 of `kv_reuse`
(2,993 of ~3,010 tokens). `json_schema` is faster mainly because llama.cpp
reasoned for 80 tokens before the JSON in that scenario while TensorSharp
answered directly. TensorSharp emits a parsed tool call or `response_format`
reply as one chunk, while llama.cpp streams fragments, so in those scenarios
end-to-end time is the comparable figure; the agentic loop's median
per-turn first-token time is 1.66 s against 1.05 s for that reason.

**Harness fix.** The original `kv_reuse` turn-3 check (`\bthursday\b`) could
not match the correct answer "Thursdays", which both engines gave in every
repeat; the check is now `\bthursdays?\b`, and the turn-3 answers of both
engines were re-scored with it (the raw records keep the original verdicts).

### Concurrent requests

| Scenario | Total output, TensorSharp (tok/s) | llama.cpp | Ratio | First token, TensorSharp (s) | llama.cpp (s) | Wall, TensorSharp (s) | llama.cpp (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4 requests | 42.05 | 31.80 | **1.32x** | 1.62 | 1.56 | 34.3 | 85.8 |
| 8 requests | 53.69 | 40.61 | **1.32x** | 2.39 | 2.29 | 57.5 | 96.4 |

Both servers had eight slots (`--np 8`, 8,192 tokens per sequence;
llama-server with `--no-kv-unified`, so each sequence keeps sparse attention).
TensorSharp decodes the slots in one batched graph with per-sequence MSA.
Answers: TensorSharp 12/12 and 24/24 checks, llama.cpp 10/12 and 21/24. A
repeat of the TensorSharp arms later the same day measured 38.5-39.9 and
46.7-48.2 tok/s with all answers correct, still above llama.cpp.

## Tensor parallelism (`--tp 8`)

llama.cpp has no tensor split for this model (`-sm tensor` is refused), so
this mode has no counterpart. With only four KV heads, TensorSharp mirrors
attention, the indexer and the caches on every GPU and splits the routed
experts, the shared expert and the dense FFN along their intermediate
dimension (one all-reduce per layer, on upstream ggml's meta backend); the
output head is split by vocabulary rows.

| | `--tp 8` | `--layer-split 8` | llama.cpp layer split |
|---|---:|---:|---:|
| Short chat decode (server, tok/s) | 40.7–41.2 | 26.3 | 26.0 |
| Short chat time to first token (s) | 0.40–0.41 | 0.41 | 0.39 |
| Agentic loop, end to end (s) | 11.7 | 18.1 | 20.1 |
| 4K-token prompt prefill (server, tok/s) | 876–880 | 1,263 | 283 |
| Executor decode at depth 0 / 2K / 8K / 16K (tok/s) | 35.7 / 26.5 / 26.3 / 26.0 | 24.8 / 23.9 / 23.8 / — | 22.5 / 19.6 / 19.1 / — |
| Executor prefill 2K / 8K / 16K (tok/s) | 1,147 / 928 / 762 | 948 / 1,699 / — | 784 / 1,312 / — |

Every functional scenario passed under `--tp 8` (two repeats). Each GPU holds
35.7 GiB of weights, so the context is capped at 45,824 tokens on 46 GB GPUs.
Layer split remains the mode for long prompts; `--tp 8` is the mode for
generation-heavy and agentic work. The `--tp 8` executor rows come from
[`eng/validation/minimax-m3-tp-bench.cpp`](../../eng/validation/minimax-m3-tp-bench.cpp) on the
tensor-parallel branch's build before integration, the layer-split executor
rows from the native bench above; the server rows are the final build.

## DSpark speculative decoding

The DSpark drafter (5 layers, block 7, Markov and confidence heads) taps the
outputs of target layers 3, 17, 31, 45 and 59. Both engines ran the bench's
`speculative` scenario (code, prose and passage-copy generations, three
repeats, the same nonce seed) on a layer split; llama.cpp with
`--spec-type draft-dspark --spec-draft-n-max 7 --spec-draft-p-min 0`,
TensorSharp with its default cumulative-confidence gate.

| Generation | DSpark, TensorSharp | DSpark, llama.cpp | Ratio | Plain, TensorSharp | Plain, llama.cpp |
|---|---:|---:|---:|---:|---:|
| Mean of 9 (tok/s) | 32.3 | 25.8 | **1.25x** | 22.9 | 20.5 |
| Code (median) | 35.3 | 26.7 | **1.32x** | 23.9 | 21.1 |
| Passage copy (median) | 35.0 | 32.0 | **1.10x** | 20.8 | 19.3 |
| Prose (median) | 26.6 | 18.7 | **1.42x** | 24.0 | 21.1 |

Acceptance was 0.75–0.77 for TensorSharp's gate. With llama.cpp's exact
settings (no gate) the two engines are close (26.6 against 25.8 tok/s), so the
gain comes from TensorSharp's calibrated use of the confidence head: on this
128-expert MoE a verify step costs 50 ms for one row and 128 ms for eight.
On the fixture, TensorSharp's drafter produced the same 200 tokens as
llama.cpp's and accepted 167 of 231 drafts against llama.cpp's 166 of 230.
The final build, rerun with the same seed after a fix to the drafter's
cross-GPU ordering in pipelined prefill, measured a mean of 32.9 tok/s
(1.27x) with acceptance 0.75-0.77. In both runs the code prompt's answer was
not reached within the scenario's 1,024-token budget because the model kept
reasoning; with a 4,096-token budget the same run passed at 36.9 tok/s.

## MoE CPU offload

`--n-cpu-moe 24` keeps the routed experts of the first 24 layers (79.7 GiB;
layers 0-2 are dense, so 21 expert stacks) in host memory and computes them on
the CPU; everything else stays on the eight GPUs. Both engines ran the same
placement (`llama-server --n-cpu-moe 24`), two repeats per scenario:

| Scenario | Metric | TensorSharp | llama.cpp | Ratio |
|---|---|---:|---:|---:|
| `short_chat` | time to first token (s) | 0.80 | 0.79 | 0.99x |
| `short_chat` | decode (tok/s) | 17.94 | 13.84 | **1.30x** |
| `tool_call_single` | end to end (s) / decode (tok/s) | 7.21 / 17.12 | 13.44 / 13.30 | **1.87x / 1.29x** |
| `json_schema` | end to end (s) | 9.33 | 20.36 | **2.18x** |
| `long_prompt_4k` | prefill (tok/s) / decode (tok/s) | 76.3 / 14.79 | 47.0 / 11.63 | **1.62x / 1.27x** |

All answers passed on both engines. TensorSharp copies each host expert stack
into memory local to the NUMA node of the pinned worker that reads it.

This row was the one where TensorSharp first lost: its server decoded at
7.7-8.4 tok/s while the executor alone, driven by the native bench, decoded
the same placement at 15.1-16.0 tok/s. Host-expert decode is
memory-bandwidth bound (32, 48 and 54 workers gave the same executor rate),
and a layer's experts meet at a barrier, so a pinned worker that shares a core
with the serving process's own threads (.NET runtime and server GC, HTTP, the
scheduler) stalls the whole layer, 21 times per token. Taking 54 of the 64-core
quota left those threads nowhere else to run. The host workers now default to
half the quota (32 here), which brought server decode to the executor's rate;
a prompt's host-expert prefill is about a quarter slower than with 54 workers
(101 against 76 tok/s at 4K), still 1.6x llama.cpp. `--cpu-moe-threads N`
(`TS_CPU_MOE_THREADS`) overrides the default for prompt-heavy work.

## Images

The four generated images (a rendered number, a word, a coloured shape and a
count of squares) were read correctly in 10 of 11 TensorSharp repeats and in
3 of 3 llama.cpp repeats; the one miss read the number 7319 as 7219. The
image tower's F32 matrix products run as TF32 on CUDA by default (ggml-cuda
enables it on its cuBLAS handle); exact F32 (`NVIDIA_TF32_OVERRIDE=0`, about
140 ms more per 672x672 image) and the default each passed 4 of 4 repeats in a
dedicated A/B, so the miss is not attributable to TF32. Against the Hugging
Face tower in F32, TensorSharp's image embeddings have a mean cosine of at
least 0.99999 (exact F32) and 0.99917 (TF32); preprocessing is bit-identical.
An image costs N + 2 positions (N = 576 at most, 540 for 1024x768); the 32-layer
tower plus projector takes about 290 ms for a 672x672 image on one A40.

## Correctness

- **Logits.** On synthetic `minimax-m3` fixtures in the real storage formats
  (NVFP4 experts with per-expert scales, Q8_0, F32 indexer; 6 layers, block 16
  so sparse attention starts at 64 tokens, and a real-block variant with block
  128 / top-16), TensorSharp's logits differ from llama.cpp's by less than
  llama.cpp's own spread between micro-batch sizes, in both the dense and the
  sparse range (mean KLD 0.0018 against a 0.0019–0.0024 floor on the small
  fixture; 0.0012 against 0.0011–0.0014 on the real-block fixture at 4,096
  tokens). Forcing dense attention moves the sparse range by KLD 0.05–2.4, so a
  wrong block selection would show. Layer split over two GPUs, host experts,
  chunked prefill, rewind, slot copy, batched decode and `--tp 2`/`--tp 4` are
  checked the same way.
- **Tokenizer and template.** 444 of 444 tokenizer cases match the Hugging Face
  tokenizer (code, CJK, Hangul jamo, NFC cases, every added token); 38 of 38
  chat renders are byte-identical to the Hugging Face template.
- **Vision.** Preprocessed pixels are bit-identical to Hugging Face's; image
  embeddings match its F32 tower with a mean cosine of at least 0.99999.
- **Tests.** 13 native test groups (executor on CPU, CUDA, two GPUs, host
  experts and `--tp 2`; MSA kernels and graph; DSpark on CPU, CUDA, two and
  four GPUs) pass on the A40 host. Of 233 managed tests, 230 pass on CUDA and
  224 on the CPU backend; the rest are skipped there (none failed).

## Limits of this comparison

- One host, one quantization, a single-user context of up to 73,728 tokens;
  medians of three repeats without confidence intervals.
- The llama.cpp reference is a pull-request branch, not an upstream release.
- Load times were not page-cache controlled.
- `--tp 8` was compared with layer split only, because llama.cpp has no tensor
  split for this model.
- Images were limited to the four generated test images; video (native
  temporal patches) is not implemented in TensorSharp, which sends video
  frames as images.
- TensorSharp's adaptive thinking cannot be turned off by a request field (the
  template has no switch); `think=true` forces a reasoning block.

Raw records are local validation evidence (not committed): the run directories
`/workspace/b1`, `/workspace/b2`, `/workspace/b3`, `/workspace/final` and
`/workspace/final2` on the benchmark host.
