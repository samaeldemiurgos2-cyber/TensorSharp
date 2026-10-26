# Qwen checkpoint copy correctness probe

`Qwen35CheckpointCopyProbe` separates checkpoint copying from prefill partition
changes. It uses the production CUDA Qwen model, holder capture/materialization,
and exact-capacity pool paths. No arena batch is called.

```sh
dotnet build eng/validation/Qwen35CheckpointCopyProbe -c Release \
  -p:TensorSharpSkipGgmlNative=true -p:TensorSharpSkipMlxNative=true
MAX_CONTEXT=32768 TS_KV_INITIAL_TOKENS=128 KV_CACHE_DTYPE=f16 \
  dotnet eng/validation/Qwen35CheckpointCopyProbe/bin/Release/net10.0/Qwen35CheckpointCopyProbe.dll \
  --model /workspace/models/Qwen3.8-27B-NVFP4-MTP-N4_0.gguf \
  --fixture artifacts/qwen38-perf/prefix-pool-functional-f16.json \
  --chunk 2048 --steps 32 --out artifacts/qwen38-perf/checkpoint-copy-f16.json
```

The report records the resolved cache dtype. Build/copy the unchanged upstream
native dependency as for the other model probes. The program records the loaded
native and managed hashes. `TS_VALIDATION_GGML_REVISION` and
`TS_VALIDATION_DEVICE` add operator-verified dependency/device metadata.

The optional fixture is a `Qwen35PrefixReuseProbe` report, from which only each
fixture's name, token IDs and shared-prefix boundary are read as data. Without
it, a deterministic 2,231-token synthetic text fixture is used. Each fixture is
checked at its public prefix and immediately before its final prompt token.
For a long-context copy check, omit `--fixture` and set
`--prompt-tokens 19001 --prefix-tokens 16502` with `MAX_CONTEXT=32768`.
Run F16 and Q8 in separate processes and retain both reports; a dtype that cannot
allocate the requested state is unavailable coverage, not a passing check.

For each case the probe performs one prefill, captures its checkpoint, then uses
the original full host synchronization to independently hash the live source's
valid KV rows and GDN state. The checkpoint must match that hash. The live source,
a fresh clone and a clone obtained from the released fresh holder then consume
identical suffix chunks and identical teacher-forced greedy-reference tokens.
All checkpoint and clone padding must be zero, capacities must match, and pooled
reuse requires the same holder and K storage objects as the preceding released
clone, as well as the same allocation pointer. Pointer equality alone would not
exclude a new mmap at a recently freed address.

The numerical gates match `Qwen35DecodeProbe`: finite logits, cosine at least
0.995, softmax KL at most 0.02, and no argmax change when the reference logit
margin exceeds 0.25. Every step records its metrics; step zero is the first
output after the prompt. Exact argmax equality is also reported separately.
Missing pool reuse fails this probe rather than silently qualifying an untested
allocation path. Hashing and synchronization perturb execution, so results are
correctness evidence only. They do not establish equivalence between differently
partitioned prefills, arena correctness, complete task quality or HTTP latency.
