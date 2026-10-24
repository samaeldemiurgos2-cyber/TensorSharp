# Benchmarks

[English](benchmarks.md) | [中文](benchmarks_zh-cn.md)

> Latest recorded comparison selection, audited 2026-10-23. No new benchmark runs were performed for the DeepSeek and Muse Spark sections; the [MiniMax-M3.1-Flash section](#minimax-m31-flash-on-eight-a40-gpus) was added from runs on 2026-10-24. The [README](../README.md#benchmarks) summarizes the DeepSeek and Muse Spark cells.

## DeepSeek V4.1 Flash 1017 — five A40 GPUs

**Measured 2026-10-22/23 UTC.** Unsloth UD-IQ2_XXS, six text shards at Hugging Face revision `60ad3651641b5c97679c807e064ab1d7a3ca1455`; five 46 GB A40s, CUDA 12.8 / driver 595.91. The host-mapped Q5_K Engram table was warmed. Both servers used a 32,768-token context, greedy sampling, thinking off, and one request generating 256 tokens. The same wikitext-2 summarization prompt sets were used: approximately 1.8K and 7.3K tokens, with unique first lines. Each number below is a median of three requests after one warmup.

The faster TensorSharp placement was routed-MoE `--tp 5`. The reference was Unsloth's llama.cpp `be8365722ea432809818fa21a4ecebfac2e7d298` with layer split; its row/tensor split could not run this checkpoint. These are **internal server prefill timings**, not streamed TTFT or client wall time.

| Prompt | TensorSharp exact TP5 tok/s | Exact ratio | TensorSharp default **approximate YOCO** TP5 tok/s | Approximate ratio | llama.cpp tok/s |
|---|---:|---:|---:|---:|---:|
| ~1.8K tokens | 563.9 | **3.23×** | 960.5 | **5.50×** | 174.6 |
| ~7.3K tokens | 549.7 | **2.44×** | 1,000.7 | **4.43×** | 225.7 |

**Approximation and quality.** Exact prefill sets `TS_DSV41_YOCO_KEEP=0`; the default uses bounded YOCO replay. On ten teacher-forced wikitext-2 chunks (4,608-token windows, 2,304 prefill tokens), TP perplexity was **2.4811 exact**, **2.4978 default**, and **2.4697 reference**. Default was approximately 0.67% above exact on this sample. Mean KL divergence from the reference was 0.01605 exact and 0.017208 default. These do not establish numerical equivalence or a general quality guarantee. The **5.50×** ratio belongs only to the default approximate ~1.8K prefill cell; it is not a decode speedup.

**Provenance and limits.** The measured TensorSharp working tree was based on `ae60abf6` plus campaign changes, with native SHA-256 prefix `04bc9dd748378492`; the changes later landed in `7f26e2d1`. The final review native build (`8b910b29e88d6891…`) confirmed a 2K synthetic cell, not a rerun of the entire server matrix. Unmodified ggml was `da8e2a54b919557b816b1c2beb432ccb766f7ce0`. Reference server observations came from an earlier session and include an unexplained slow prefill observation; three samples per cell do not establish a confidence interval. TensorSharp and llama.cpp report internal timings through different protocol routes. Synthetic best-of-three TensorSharp timings and llama-bench means are excluded from this headline table.

The [model card](models/deepseek41.md#unsloths-deepseek-v41-flash-1017-ggufs) records shard hashes, launch configuration, decode, DSpark and concurrency results. Raw campaign records are local, uncommitted evidence under `docs/validation/deepseek41-1017/`; the server medians were independently recomputed from the available local JSONL for this documentation update. Full checkpoint files were not rehashed here. These results do not qualify the different repaired Q2_K/Q5_K package.

## Muse Spark 816B on ten A40 GPUs

**Report refreshed 2026-10-22; exact measurement date not recorded.** Muse-Spark-Open-816B-A42B, UD-IQ3_XXS, eight shards / 315.7 GiB weights, on ten A40 46 GB GPUs. All experts stayed in GPU memory. Both servers used F16 KV, flash attention, a 32,768-token context and greedy sampling with thinking off. Long-prompt prefill requests capped output at 32 tokens; short decode and concurrent requests capped it at 256. Decode used a short 167-token prompt. Muse prefill is uncached prompt tokens divided by streamed TTFT, unlike the internal prefill timers used for DeepSeek above. Concurrency is one unless the row says four.

Choose placement per workload: layer split gives the highest absolute prefill throughput in this comparison; tensor parallelism gives the highest decode and four-request aggregate throughput. Every ratio below compares the corresponding mode in both engines.

| Workload / placement | TensorSharp tok/s | llama.cpp tok/s | Ratio |
|---|---:|---:|---:|
| Prefill, 4,075 tokens, layer split 10 | 999.7 | 483.2 | **2.07×** |
| Prefill, 15,767 tokens, layer split 10 | 1,346.8 | 826.1 | **1.63×** |
| Decode, one request, TP10 / tensor split | 22.63 | 16.19 | **1.40×** |
| Four requests, aggregate output, TP10 / tensor split | 45.73 | 24.49 | **1.87×** |
| Decode, one request, layer split 10 | 16.69 | 16.79 | 0.99× |

Throughput/latency cells are medians of three repeats, except llama.cpp tensor-split rows, which have two; fixtures ran once. llama.cpp was `7170cad` with its in-tree ggml; TensorSharp used unchanged upstream ggml `da8e2a5`; TensorSharp's report identifies a working tree atop `e2a84893` with `192295a9` and later edits, rather than an exact frozen benchmark commit. The report does not establish the performance of today's release binaries.

**Transport and coverage limits.** P2P was advertised but did not work; there was no NVLink. llama.cpp tensor split was built **without NCCL** because its NCCL build aborted on ten GPUs; TensorSharp used NCCL through shared memory. Large TensorSharp TP prefill reductions used BF16. These conditions can materially affect TP performance and quality. Single-host results have no confidence intervals. Images, expert CPU offload, speculative decoding, other GPU counts/backends and actual prompts beyond 16K were not measured. Four-request layer-split per-request decode was slower (8.54 vs 9.37 tok/s), despite a small aggregate throughput lead.

See the [full comparison](perf/muse-spark-vs-llama-cpp.md) for both complete mode tables, teacher-forced quality checks and launch commands. The raw current ten-A40 records were not available on this Mac, so this update verifies the committed report's transcription and arithmetic, not the original run logs.

## MiniMax-M3.1-Flash on eight A40 GPUs

**Measured 2026-10-24.** Unsloth MiniMax-M3.1-Flash UD-Q4_K_XL (seven shards, 232.3 GiB) on one host with eight A40 46 GB GPUs. TensorSharp ran on unchanged upstream ggml `da8e2a5`; the reference was llama.cpp from unslothai's `minimax-m31` branch (`ef8f94cc`, PR #320). Both engines ran on the same box, one at a time with the GPUs held exclusively, and the host was otherwise idle. Cells are medians of three repeats unless noted. Ratios above 1.0× mean TensorSharp is better.

**Executor throughput.** TensorSharp's executor bench ([eng/validation/minimax-m3-native-bench.cpp](../eng/validation/minimax-m3-native-bench.cpp): one load, uniform-random tokens, an untimed warm-up and timed repetitions per cell, untimed depth prefill, the calls the managed model makes) against `llama-bench -fa 1`, both as a layer split over the eight GPUs.

| Cell | TensorSharp tok/s | llama.cpp tok/s | Ratio |
|---|---:|---:|---:|
| Prefill, 512 tokens | 448 | 344 | **1.30×** |
| Prefill, 2,048 tokens | 948 | 784 | **1.21×** |
| Prefill, 8,192 tokens | 1,699 | 1,312 | **1.29×** |
| Prefill, 32,768 tokens | 2,449 | 681 | **3.60×** |
| Prefill, 65,536 tokens | 2,567 | 262 | **9.80×** |
| Decode, 128 tokens at depth 0 | 24.8 | 22.5 | **1.10×** |
| Decode, 128 tokens at depth 2,048 | 23.9 | 19.6 | **1.22×** |
| Decode, 128 tokens at depth 8,192 | 23.8 | 19.1 | **1.25×** |
| Decode, 128 tokens at depth 32,768 | 23.4 | 17.5 | **1.34×** |

Each llama.cpp prefill cell is the faster of `-ub 512` and `-ub 2048`. At 32K, `-ub 2048` ran out of GPU memory, and the 64K cell is a single run at `-ub 2048` (181 tok/s at `-ub 512`). Past position 2,048 MiniMax Sparse Attention is sparse; TensorSharp runs that regime in its own CUDA kernels.

**Server scenarios.** [eng/minimax-m3-bench.py](../eng/minimax-m3-bench.py) against TensorSharp.Server.Host and llama-server, launched with the matched settings of [eng/minimax-m3-ts-server.sh](../eng/minimax-m3-ts-server.sh) and [eng/minimax-m3-llama-server.sh](../eng/minimax-m3-llama-server.sh): layer split over eight GPUs, one sequence (llama-server with `-np 1 --no-kv-unified`, without which it runs MSA dense), flash attention, greedy streamed requests, thinking in the template's default adaptive mode. Long prompts are needle-in-a-haystack questions over real prose whose answer is checked. Every conversation opens with a fresh nonce, so neither engine reuses a previous run's prompt. The engines were rotated ABBA (llama.cpp, TensorSharp, TensorSharp, llama.cpp); the page cache was not evicted between launches, so load times are not compared. Prefill is uncached prompt tokens divided by time to first token.

| Prompt | Prefill, TensorSharp | Prefill, llama.cpp | Ratio | Decode, TensorSharp | Decode, llama.cpp | Ratio |
|---|---:|---:|---:|---:|---:|---:|
| 4K tokens | 1,263 | 283 | **4.46×** | 23.9 | 19.2 | **1.24×** |
| 16K tokens | 1,478 | 273 | **5.41×** | 23.7 | 18.5 | **1.28×** |
| 32K tokens | 1,544 | 240 | **6.43×** | 23.5 | 17.7 | **1.33×** |
| 64K tokens | 1,545 | 211 | **7.32×** | 23.0 | 16.1 | **1.43×** |

The short-chat time to first token is 0.41 s against llama.cpp's 0.39 s (0.95×): TensorSharp is slightly behind there. With four and eight different concurrent requests, aggregate output was 42.1 against 31.8 tok/s (1.32×) and 53.7 against 40.6 (1.32×); time to first token under that concurrency was 1.62 against 1.56 s and 2.39 against 2.29 s, again slightly behind. TensorSharp's answers passed 12 of 12 and 24 of 24 checks there, llama.cpp's 10 of 12 and 21 of 24.

**Tensor parallelism.** llama.cpp has no tensor split for this model, so `--tp 8` has no counterpart. TensorSharp's TP bench ([eng/validation/minimax-m3-tp-bench.cpp](../eng/validation/minimax-m3-tp-bench.cpp), ubatch 2048, three passes, on the tensor-parallel branch's build before integration) decoded 35.7 tok/s at depth 0 and 26.5 / 26.3 / 26.0 tok/s at depths 2,048 / 8,192 / 16,384, and prefilled 1,147 / 928 / 762 tok/s at 2K / 8K / 16K. With no `MAX_CONTEXT` the context was capped to 45,824 tokens (38.1 GiB in use per GPU), because attention and the caches are mirrored on every GPU. The server's text and image scenarios passed under `--tp 8` (one repeat each).

**DSpark speculative decoding.** The bench's `speculative` scenario (code, prose and passage-copy generations, three repeats each) on a layer split, with the converted `MiniMax-M3.1-Flash-DSpark-BF16.gguf`. TensorSharp used its default cumulative-confidence gate (0.35); llama-server used `--spec-type draft-dspark` with at most 7 draft tokens and `p-min 0`.

| Generation | DSpark, TensorSharp | DSpark, llama.cpp | Ratio | Plain, TensorSharp | Plain, llama.cpp |
|---|---:|---:|---:|---:|---:|
| Mean (tok/s) | 32.3 | 25.8 | **1.25×** | 22.9 | 20.5 |
| Code (median) | 35.3 | 26.7 | **1.32×** | 23.9 | 21.1 |
| Passage copy (median) | 35.0 | 32.0 | **1.10×** | 20.8 | 19.3 |
| Prose (median) | 26.6 | 18.7 | **1.42×** | 24.0 | 21.1 |

The gate is most of the difference: with llama.cpp's settings (`--spec-draft 7 --spec-pmin 0` and the cost governor off, `TS_SPEC_ADAPTIVE=0`) TensorSharp averaged 26.6 tok/s, and without a gate both engines decoded prose more slowly than plain decoding. On the real model, speculative and plain greedy texts diverge after 50-130 tokens on both engines, each time at a near-tie of plain decoding's top two tokens; see the [model card](models/minimax-m3.md#dspark-speculative-decoding).

**Provenance and limits.** The TensorSharp figures come from uncommitted working-tree builds of the MiniMax-M3.1 branch on 2026-10-24, not a frozen commit; the tensor-parallel and DSpark figures come from those features' builds before they were merged. One host and one checkpoint, with no confidence intervals. The executor benches exclude HTTP; the server scenarios include it. Video input, other GPU counts and backends, and full-checkpoint perplexity or KL divergence were not measured; MoE CPU offload was measured once per engine (two repeats) and is in the full report. Logit parity with llama.cpp was measured on synthetic fixtures only ([model card](models/minimax-m3.md#validation)). The raw records are local validation evidence on the measurement host, not committed: `/workspace/b1/` (executor bench, llama-bench, server and concurrency scenarios), `/workspace/b2/` (tensor parallelism) and `/workspace/b3/real/` (DSpark). Full comparison report, including MoE CPU offload: [perf/minimax-m3-vs-llama-cpp.md](perf/minimax-m3-vs-llama-cpp.md).

## Reading and reproducing results

Ratios above 1.0× mean higher throughput for the stated cell. Do not combine the models, placements, prompt lengths, approximation settings or concurrency levels into a universal speedup. Old laptop, earlier checkpoint and internal-backend headline tables have been removed from this overview.

Use the [engine-comparison harness](../benchmarks/engine_comparison) or model-specific tools with your own matched model/backend and record the exact builds, weight hashes, prompt/output lengths, context, concurrency, cache policy, warmups and repeats. Generated reports belong in ignored validation storage. A missing scenario is not a passing result.
