# Multi-node transport microbenchmark

This probe exercises the actual `TcpCommunicator` with exact F32 payload checks
on **every node**. It measures AllReduce and broadcast from the last node at
4 B, 4 KiB, 10 KiB, 16 KiB, 28,160 B, 64 KiB, 1 MiB and 4 MiB. The intermediate
sizes represent Qwen and GLM decode reductions. Reports retain all samples, warmup counts,
median/p95 latency, transport/probe assembly hashes and the declared route.

Three operations are timed per size:

- `all_reduce`: one collective after a barrier. Node 0 releases the barrier,
  so it starts with up to one one-way latency of skew; this hides the
  difference between the `star` and `symmetric` algorithms.
- `chained_all_reduce`: `--chain` (default 32) collectives back to back after
  each barrier, reported per collective. Every node reaches the next collective
  together, as in a tensor-parallel decode, and both nodes' figures agree. Use
  this one to compare algorithms and tunnels.
- `broadcast_from_last`.

On two RunPod A40 VMs reachable only over SSH, chained medians for 10/28 KiB
were 0.75/1.48 ms with protocol 2 (`star`) over an `ssh -N` chacha20 forward and
0.26/0.44 ms with protocol 3 (`symmetric`) over the interactive aes128-gcm
forward from `multinode-ssh-tunnel.sh`; 4 MiB went from 18.9 to 9.1 ms. One
route and one quiet period; measure your own.

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
The barrier does not release every process simultaneously: one node can send
data before another starts its timer. Compare both nodes' reports and validate
any proposed optimization in full inference; per-node timings alone do not
measure collective completion across the cluster.
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
