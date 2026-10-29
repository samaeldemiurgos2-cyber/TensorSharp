# GLM distributed native fixture

Generate a deterministic six-layer `glm5-next` GGUF with KDA recurrence,
pooled sparse MLA, hyper-connections and Q8 expert/head partitions. The hidden
sizes permit four global TP ranks. These seeded weights test numerical state
and placement; they cannot establish language quality or production throughput.

```sh
dotnet build eng/validation/GlmDistributedFixture -c Release
dotnet eng/validation/GlmDistributedFixture/bin/Release/net10.0/GlmDistributedFixture.dll artifacts/glm-fixture
dotnet build eng/validation/DeepSeek41DistributedProbe -c Release -p:TensorSharpSkipGgmlNative=true -p:TensorSharpSkipMlxNative=true
```

Use a fresh output directory for each run and the matching TensorSharp native
library. Keep upstream ggml unchanged. The generic `DeepSeek41DistributedProbe`
also accepts this GLM fixture through the public model factory. For example,
set these identically on both CUDA nodes:

```sh
export CUDA_VISIBLE_DEVICES=0,1 MAX_CONTEXT=128 KV_CACHE_DTYPE=f16
export TS_N_CPU_MOE=0 TS_GLM_UBATCH=16 TS_GLM_THREADS=2
export TS_PREFILL_WARMUP=0
```

Run the probe on node 1 and then node 0, substituting reachable peer addresses,
the node ID and a separate output directory:

```sh
dotnet DeepSeek41DistributedProbe.dll \
  --model artifacts/glm-fixture/glm53-mixed-q8.gguf \
  --tokens artifacts/glm-fixture/tokens.json \
  --weights-id FIXTURE_SHA256 --output artifacts/glm-node1 \
  --backend ggml_cuda --layer-split 2 --tp-node-id 1 \
  --tp-peers NODE0:29500,NODE1:29500 --chunk 13 --decode-count 4
```

Replace `--layer-split 2` with `--tp 2` for two nodes with two tensor-parallel
ranks each. Omit the three placement/network options for an unsplit control.
Use the SHA-256 in `fixture.json` as `FIXTURE_SHA256`, and verify the same file
on each node. Compare all driver vectors with the independent control using
`eng/validation/compare-deepseek41-distributed.py`; preserve failures and both
process exit statuses. A worker returning is a lifecycle result, not a numeric
comparison. Store generated evidence under ignored `artifacts/` or
`docs/validation/`.

The tiny fixture can select different attention/indexer kernels from the real
checkpoint. Record the selected kernels, physical GPUs and any overrides in
the evidence. Separate-process loopback on one GPU does not qualify a physical
two-node/four-GPU deployment. Native CPU adapter tests are in
`Glm5NextDistributedTests`; public distributed deployment currently requires
`ggml_cuda`.
