# Multi-node layer splitting and tensor parallelism

Both execution modes use the same peer configuration. `--layer-split N` assigns
whole layers to N local GPUs on each node; `--tp N` shards supported weights
across N local ranks on each node. Node 0 owns tokenization, sampling and output.
Other nodes run `TensorSharp.Cli` as workers. Every node must use the same build,
model bytes, backend, mode, local degree and execution settings.

## Launch a layer pipeline

Start workers and then the driver. Replace the addresses with endpoints reachable
between every pair of nodes. IPv6 endpoints use `[address]:port`.

```bash
# Set the same context budget on every node; increase after checking GPU headroom.
export MAX_CONTEXT=2048

# Node 1: two local GPUs, worker
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll \
  --model /models/model.gguf --backend ggml_cuda --layer-split 2 \
  --tp-node-id 1 --tp-peers 10.0.0.10:9500,10.0.0.11:9500

# Node 0: two local GPUs, driver
printf '%s\n' 'Explain why the sky is blue.' > prompt.txt
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll \
  --model /models/model.gguf --backend ggml_cuda --layer-split 2 \
  --tp-node-id 0 --tp-peers 10.0.0.10:9500,10.0.0.11:9500 \
  --input prompt.txt --chat --max-tokens 128
```

For HTTP serving, replace the driver executable with
`TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll` and omit `--input` and `--chat`.
Workers still use the CLI. The server accepts only node 0 and requires a startup
model; changing a running distributed cluster's model/backend is refused.

Use `--layer-split 1` for one GPU on each of several nodes. With neither placement
mode configured, the default is one local GPU. `--tp-node-id` and `--tp-peers`
without an explicit layer selection retain their tensor-parallel meaning.
`--tp N` and `--layer-split M` cannot both exceed one. Environment equivalents are
`TENSORSHARP_LAYER_SPLIT_DEGREE`, `TENSORSHARP_TP_DEGREE`,
`TENSORSHARP_TP_NODE_ID` and `TENSORSHARP_TP_PEERS`.

For native DeepSeek, select physical CUDA devices with `CUDA_VISIBLE_DEVICES`
before launching each process. The managed dense layer executor also supports
`TENSORSHARP_LAYER_SPLIT_DEVICES` for multi-GPU device ordinals. Model files and
all shards must be available on every node; each stage loads its owned weights.

## Supported stage executors

| Family | Multi-node layer backend | State and limits |
| --- | --- | --- |
| DeepSeek V4.1 Flash, including 1017 | `ggml_cuda` | Native whole-layer graphs transfer residual streams, delayed gates, sparse selections, candidate masks and newly written shared cache rows. Exact prefill; approximate YOCO, vision and DSpark are unavailable in this mode. |
| Mistral 3 dense text tower | `ggml_cuda`, `ggml_vulkan` | Shared dense pipeline transfers F32 hidden activations. Each node loads only its layer weights/KV state; the first node embeds and the final node computes logits. |

The transport, ownership plan and dense-stage execution are independent of model
names. Architecture descriptors explicitly opt in after implementing their full
boundary state. Other families are refused before GPU allocation or network
connection; support for a local layer split alone does not imply cross-node
support. Backend availability is distinct from measured device/model coverage.

Placement balances estimated resident bytes while keeping each node's layers
contiguous and assigning at least one layer to every selected GPU. The dense
pipeline accounts for layer weights and embedding/output endpoints; DeepSeek
also accounts for KV state, cross-stage cache replicas and explicitly offloaded
experts. This avoids the large memory imbalance that equal layer counts can
produce in models with differently sized layers. Estimates do not guarantee a
fit on heterogeneous GPUs: runtime buffers and other device allocations still
require headroom. New architecture adapters must account for their own replicated
weights and boundary state.

Distributed requests currently use serialized forward/refill/reset/truncate
controls. Paged multi-sequence batching, KV snapshots and speculative paths that
bypass those controls are disabled for these stage executors. Layer splitting
increases aggregate model capacity. Each token still traverses the stages in
order, so more devices need not make single-request decode faster.

## Adding a model family

`ILayerSplitGroup` supplies ordered, lossless point-to-point transfers independently
of tensor sharding. `LayerSplitPlan` supplies contiguous ownership and weighted
partitioning. Dense adapters can use `ModelBase.ConfigureLayerPlacement` and
`ForwardLayerPipeline`, route weights and KV allocations through the selected
layer allocator, and leave embedding/output work at the endpoints. Mistral 3 is
the complete managed example. Native executors can use the same group through
a callback; DeepSeek demonstrates a boundary containing more than hidden
activations.

An adapter must transfer every state value needed by later stages, retain cache
ownership across refill/reset/truncate, and agree on numerical settings before
loading stage weights. Enable the architecture descriptor only after testing
those operations against an unsplit control. In particular, a family with
cross-layer attention, recurrent state or delayed residuals cannot safely reuse
a hidden-activation-only boundary without accounting for that extra state.

## Transport and tuning

TP and layer splitting share persistent TCP connections with `TCP_NODELAY`,
keepalive, a versioned mesh handshake and mode/degree agreement. Tensor frames
transfer lossless 32-bit words directly into destination buffers; small frames
combine their header and payload into one write. TP reuses its receive scratch
and keeps deterministic node-order reduction through node 0. Stage transfers perform no
floating-point reduction or compression.

Incorrect frame lengths/types, incomplete reads, timeouts and configuration
mismatches fail the operation. Transport failures close the mesh so peers cannot
continue consuming a partially read tensor. Every node must run the same wire
protocol version; restart the whole cluster when updating binaries.
Startup also checks the model's execution settings and the resolved warmup
schedule. Use the same `TS_PREFILL_WARMUP` and `TS_PREFILL_WARMUP_LEN` settings on
every node. Mistral selects its default KV precision from the entire checkpoint
before filtering stage weights, so mixed-precision layers keep the same cache
precision on every node.

| Variable | Default | Purpose |
| --- | --- | --- |
| `TENSORSHARP_TP_CONNECT_TIMEOUT_SECONDS` | `120` | Bound connection setup, inbound accepts and handshakes. |
| `TENSORSHARP_TP_RECV_TIMEOUT_SECONDS` | `300` | Bound socket reads and writes. Include the time a peer spends computing before its next transfer. |
| `TENSORSHARP_TP_OPERATION_TIMEOUT_SECONDS` | Receive timeout (`300` by default) | Deadline for one complete activation, broadcast-data or reduction frame, including heartbeats and partial transfers. Increase for long forwards. Idle worker controls and load barriers are exempt; their socket inactivity timeout still applies. |
| `TENSORSHARP_TP_SOCKET_BUFFER_BYTES` | OS default | Optional send/receive buffer budget, minimum 4096 bytes. Measure the route before overriding OS tuning. |

Peer sockets have no built-in authentication or encryption. Use trusted private
networking or authenticated tunnels. A tunnel's latency and encryption cost are
part of the measured deployment; results on it do not qualify RDMA, GPUDirect or
a low-latency fabric. Compare small decode collectives and large prefill payloads
separately using `eng/validation/DistributedTransportProbe`.

## Reproducible validation

Keep ggml upstream unchanged. Record its commit and clean status, both nodes'
source/binary hashes, topology, RAM limits, model revision and full shard hashes,
route, execution settings and competing workloads. Verify weights before timing.
Use `eng/validation/capture-multinode-environment.py` for host evidence.

`eng/validation/DeepSeek41DistributedProbe` accepts either placement mode and
records complete finite logits for fixed-token full/chunked prefill, decode,
refill and resets. Compare local and distributed runs with
`eng/validation/compare-deepseek41-distributed.py`. The native diagnostic
`eng/validation/deepseek-layer-native-probe.py` separately isolates native stage
state from managed transport. Synthetic fixtures test numerical plumbing; they
do not establish checkpoint language quality.

For the full 1017 checkpoint, also run text/JSON/tool workflows and sustained
decode with matched prompts, contexts, precision and token budgets. Exclude
warmup, retain at least three measured repetitions and compare matched local
controls. Report failures, skipped configurations, host paging and short-output
limits. Generated logs/reports belong in ignored `artifacts/` or
`docs/validation/`; reusable tools belong in `eng/`.
