# Benchmarks

[English](benchmarks.md) | [中文](benchmarks_zh-cn.md)

> Latest recorded comparison selection, audited 2026-10-23. No new benchmark runs were performed for this update. The [README](../README.md#benchmarks) summarizes these cells.

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

## Reading and reproducing results

Ratios above 1.0× mean higher throughput for the stated cell. Do not combine the two models, placements, prompt lengths, approximation settings or concurrency levels into a universal speedup. Old laptop, earlier checkpoint and internal-backend headline tables have been removed from this overview.

Use the [engine-comparison harness](../benchmarks/engine_comparison) or model-specific tools with your own matched model/backend and record the exact builds, weight hashes, prompt/output lengths, context, concurrency, cache policy, warmups and repeats. Generated reports belong in ignored validation storage. A missing scenario is not a passing result.
