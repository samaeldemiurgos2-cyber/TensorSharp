# DeepSeek V4.1 fixed-token distributed validation

This probe uses TensorSharp's tensor-parallel or distributed layer-split groups, TCP transport and
driver/worker controls. Every driver forward writes its actual input tokens,
elapsed time and complete finite F32 logit vector. It exercises a full prefill,
teacher-forced decode, reset, chunked prefill, refill and another reset/prefill.
Teacher forcing keeps inputs identical after a greedy near-tie. Its timings are
diagnostics, not a throughput benchmark. A worker returning from its loop is
recorded separately from a completed driver numerical run.

For whole-layer placement, replace `--tp N` with `--layer-split N` on **both**
nodes; retain the same `--tp-node-id` and `--tp-peers` arguments. `N` is the GPU
count per node, including an explicit `--layer-split 1`. Run a matching local
`--layer-split N` control and compare every saved logit vector. The same probe
accepts other supported model families, such as Mistral 3, with a verified
checkpoint identity and valid token IDs. Fixture inputs demonstrate numerical
execution only; run real text/structured-output serving tests separately.
For real checkpoints, `--text UTF8_TEXT_FILE` can replace `--tokens TOKENS.json`;
the probe uses that model's tokenizer and saves the exact IDs and text SHA256.
Do not reuse a synthetic fixture's small token IDs for another vocabulary: they
can denote unused special tokens and yield an uninformative zero-vector case.

`--backend cuda` selects the direct-CUDA engine: single-process, or, with
`--layer-split N --tp-node-id K --tp-peers ...`, one pipeline stage per node
(DeepSeek V4.1 only; distributed `--tp` stays `ggml_cuda`). The driver (node 0)
must run the last stage, the default `TENSORSHARP_LAYER_SPLIT_DRIVER_STAGE`.
Both stages run the same kernels as the single-process engine and exchange
their boundary state losslessly, so the driver's vectors are expected to be
**bit-identical** to a single-process `--backend cuda` control with the same
settings (set `TS_DSV41_ENGRAM_DEVICE=0` on the control: distributed stages
keep the Engram tables host-mapped unless it is 1). Every node must use the
same `MAX_CONTEXT`, `TS_DSV4_UBATCH` and CUDA routing variables; a node whose
settings differ refuses at load.

Retain native per-device ownership diagnostics: seeing two visible devices
alone does not establish that both execute model layers. When local controls
need CPU expert offload while distributed stages fit on GPUs, report the
comparison as a capacity configuration change, not isolated network scaling.

Build on both nodes from identical source, against an unchanged upstream ggml
checkout. Save `git -C ExternalProjects/ggml rev-parse HEAD` and
`git -C ExternalProjects/ggml status --porcelain --untracked-files=all` with the
build logs. Retain source and native/managed binary hashes on both nodes. Use
the same native library on both nodes when their CPU/CUDA requirements match.

```bash
dotnet build eng/validation/DeepSeek41DistributedProbe/DeepSeek41DistributedProbe.csproj \
  -c Release -p:TensorSharpSkipGgmlNative=true -p:TensorSharpSkipMlxNative=true
```

Generate the existing independent F32 fixture; the second command adds only the
typed empty BPE merges required by managed tokenizer loading. Its manifest
asserts every tensor is unchanged. This synthetic model does not produce useful
language and cannot establish checkpoint quality or production performance.

```bash
python eng/dsv41-fixture.py artifacts/dsv41-fixture --f32 --cuda-index
fixture_sha=$(sha256sum artifacts/dsv41-fixture/deepseek41-fixture.gguf | cut -d' ' -f1)
python eng/validation/prepare-dsv41-managed-fixture.py \
  --source artifacts/dsv41-fixture/deepseek41-fixture.gguf \
  --source-sha256 "$fixture_sha" --output-dir artifacts/dsv41-managed-fixture
```

For quantized cross-node coverage, generate a separate fixture with
`--k-mix --cuda-index --moe-intermediate-size 2304` and prepare its managed
copy the same way. Its Q2_K gate/up and Q3_K down tensors have nine hidden
quantization blocks, so four global ranks receive uneven strips that rotate
across layers. The embedding width, layer/expert counts and tokenizer remain
synthetic; this is not the full checkpoint's geometry or quality. The default
256-wide fixture is unchanged and has only one down quantization block, which
cannot be split across four global ranks.

Use a fresh output directory on every attempt. Set `LD_LIBRARY_PATH` to the
native build directory and give both nodes the identical model and token files.
`PEERS` contains the reachable endpoint for node 0 followed by node 1. The
existing transport opens each node's endpoint; direct private networking or
explicit SSH forwards may be used. Record the route and measure it with
`eng/validation/parallel-network-probe.py`. SSH round-trip measurements do not
qualify an RDMA or low-latency fabric.

```bash
probe=eng/validation/DeepSeek41DistributedProbe/bin/Release/net10.0/DeepSeek41DistributedProbe.dll
model=artifacts/dsv41-managed-fixture/deepseek41-fixture.gguf
tokens=artifacts/dsv41-managed-fixture/tokens.json
weights_id=$(sha256sum "$model" | cut -d' ' -f1)
export TS_DSV41_YOCO_KEEP=0 TS_DSV41_SPARSE_FA=0 TS_DSV41_COMPACT_RAW_GATHER=0
export MAX_CONTEXT=256 TS_DSV4_UBATCH=32

# First collect local controls, with no distributed configuration in the environment.
dotnet "$probe" --model "$model" --tokens "$tokens" --weights-id "$weights_id" \
  --output artifacts/dsv41-probe-local2 --tp 2

# Start on node 1, then node 0, with the same PEERS value on both.
dotnet "$probe" --model "$model" --tokens "$tokens" --weights-id "$weights_id" \
  --output artifacts/dsv41-probe-node1 --tp 2 --tp-node-id 1 --tp-peers "$PEERS"
dotnet "$probe" --model "$model" --tokens "$tokens" --weights-id "$weights_id" \
  --output artifacts/dsv41-probe-node0 --tp 2 --tp-node-id 0 --tp-peers "$PEERS"

# Compare every vector and its operation/input identity. Choose gates before
# examining results; these tight values are intended for the F32 fixture.
python eng/validation/compare-deepseek41-distributed.py \
  artifacts/dsv41-probe-local2/probe.json artifacts/dsv41-probe-node0/probe.json \
  --max-absolute-error 0.00002 --max-rmse 0.00002 \
  --max-relative-l2 0.00002 --min-cosine 0.999999 --require-same-argmax \
  --output artifacts/dsv41-probe-comparison.json
```

Also run a one-GPU control and two nodes with `--tp 1` to cover two global
ranks. For independent architectural validation, run
`eng/tests/dsv41-inference.py` against the original F32 fixture and native
library with `--backend CUDA --gpus 2 --tp 2 --atol 2e-5 --rtol 2e-5`; its
PyTorch oracle covers prefill, decode, chunking, reset and native cache state.
`ctest -R '^deepseek41-distributed-moe-numerical$'` separately tests global
weight-strip coverage, both-node outputs and collective failure recovery with
CPU devices and a simulated transport. Neither replaces real TCP/GPU tests.

For the published 1017 checkpoint, verify **all** shard hashes against its
immutable repository revision outside timed inference. Repeat fixed-token
local/distributed controls with the same tokens and precision/settings. Record
any quantized numerical differences without weakening the F32 gate. Then use
`benchmarks/engine_comparison/validate_inference.py` for text, Chinese, strict
JSON/schema/Unicode, generated multi-turn history and dependent tool workflows;
use `validate_deepseek41_tools.py` for tool policies. Retain every response and
failure, including truncated answers. Run sustained decode and long-context
cases only with a sufficient token/context budget; short eight-token samples
are diagnostic latency measurements, not quality or sustained-decode passes.

`eng/validation/verify-pinned-model-manifest.py MANIFEST.json --root MODEL_ROOT
--output artifacts/model-verified.json` streams every shard and produces the
`--model-verification-report` format consumed by `run-parallel-case.py`. Its
input records `repo`, a full immutable commit `revision`, `quant`, and
`files: [{"path": "relative.gguf", "size": 123, "sha256": "full digest"}]`.
Existing `verified` flags in an input manifest are ignored; bytes are reread.
The output is written incrementally, and an interrupted `running` report or a
failed hash never qualifies as verified.

For HTTP fixture controls, make another derived copy with
`prepare-dsv41-managed-fixture.py --byte-tokenizer`. This replaces only the
256 vocabulary strings with GPT-2's reversible byte alphabet (ID `b` encodes
byte `b`) and adds the empty merges table; the manifest verifies all 142 model
tensor hashes. Keep the original numerical fixture and its token IDs separate.
The original `t0`–`t255` vocabulary cannot encode arbitrary English prompts;
nonempty HTTP prompts can therefore tokenize to zero tokens. Byte vocabulary
enables protocol testing, not language quality.

Use `eng/validation/run-parallel-case.py` around CLI/server launches to retain
source/binary identity, telemetry and shutdown results. Record both nodes'
CPU/GPU topology, RAM and cgroup limits, CPU expert placement, page residency,
storage, peer route and competing processes. Compare against a matched local
control, with warmup outside steady-state timing and at least three measured
repetitions. Report skipped device counts and unavailable reference/model
scenarios as uncovered. Keep generated reports in ignored `artifacts/` or
`docs/validation/`; do not force-add them.

Summarize completed CLI timing runs with
`eng/validation/summarize-parallel-cli-benchmark.py RUN_DIRECTORY --discard-runs
1 --output artifacts/cli-summary.json`. It computes medians from actual start
counts and logged milliseconds, retains all samples, and rejects incomplete
runs, crashes and failed runtime identity checks. Initial-run exclusion does
not establish steady state; preserve changes in timing across measured runs.

For the synthetic server/control lane, use
`eng/validation/deepseek41-distributed-http-probe.py` against the local and
distributed endpoints with identical arguments. It retains complete HTTP/stream
bodies and compares visible output, token counts and finish reasons across
reset, history, overlapping requests, streaming and guarded model-switch cases.
This exercises the real server/CLI worker protocol; the fixture's text does not
establish language quality or throughput scaling.
