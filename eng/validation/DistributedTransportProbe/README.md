# Multi-node transport microbenchmark

This probe exercises the actual `TcpCommunicator` with exact F32 payload checks
on **every node**. It measures AllReduce and broadcast from the last node at
4 B, 4 KiB, 64 KiB, 1 MiB and 4 MiB. Reports retain all samples, warmup counts,
median/p95 latency, transport/probe assembly hashes and the declared route.

```bash
dotnet build eng/validation/DistributedTransportProbe/DistributedTransportProbe.csproj -c Release
probe=eng/validation/DistributedTransportProbe/bin/Release/net10.0/DistributedTransportProbe.dll

# Start on node 1 before starting node 0; use the same reachable peer list.
dotnet "$probe" --node-id 1 --peers NODE0:9500,NODE1:9500 \
  --output artifacts/transport-node1.json --route direct-private-tcp \
  --iterations 100 --warmup 10
dotnet "$probe" --node-id 0 --peers NODE0:9500,NODE1:9500 \
  --output artifacts/transport-node0.json --route direct-private-tcp \
  --iterations 100 --warmup 10
```

Use fresh outputs for every attempt. For before/after comparisons, retain
separate immutable runtimes and run at least three matched repetitions on the
same nodes and network route without concurrent compilation, downloads or
model workloads. A protocol-version change requires matching binaries on all
nodes in each run. Retain failures alongside successful runs.

The payload rate is `payload bytes / elapsed seconds`; it does not multiply by
wire copies or claim fabric saturation. A barrier separates iterations and is
outside the timed operation. GPU copies and inference are **not** measured.
An SSH-forward result qualifies only that tunnel, not a direct or RDMA fabric.
See `parallel-network-probe.py` for an independent TCP echo path check.

Supporting evidence tools in `eng/validation/`:

- `capture-multinode-environment.py` records GPU/CPU topology, cgroup limits,
  routes, processes, compiler runtime and unchanged upstream ggml identity.
- `download-pinned-hf-model.py` resolves the requested revision once and saves
  publisher file sizes/SHA256s before downloading named files. Independently run
  `verify-pinned-model-manifest.py` to verify all downloaded bytes.
- `transfer-validation-artifact.py` copies one fresh artifact over an already
  authenticated SSH forward. Both ends bind/connect loopback; the receiver
  verifies SHA256 before promoting the partial file. It establishes neither
  SSH credentials nor a public network listener.

Keep generated evidence under ignored `artifacts/` or `docs/validation/`.
