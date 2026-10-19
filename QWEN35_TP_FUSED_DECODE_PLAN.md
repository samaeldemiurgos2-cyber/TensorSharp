# Qwen3.5/3.6 — Fused CUDA-Graph Tensor-Parallel Decode (Plan)

**Status:** Planning (not implemented)
**Owner:** TBD
**Related:** `TENSOR_PARALLELISM_PLAN.md`
**Applies to:** `qwen35` (dense, e.g. Qwen3.5-9B) and `qwen35moe` (e.g. Qwen3.5/3.6-35B-A3B) on `--backend cuda` with `--tp N`.

---

## 1. Problem statement

On the **direct-CUDA backend** (`--backend cuda`, the pure-C# path), tensor-parallel
decode is dramatically slower than both single-GPU and the GGML backend:

| Configuration (Qwen3.5-35B-A3B-UD-IQ4_XS, 2× L4) | Decode | Prefill |
|---|---|---|
| `cuda` **tp=1** (fused CUDA-graph decode) | **36.0 tok/s** | 76.4 tok/s |
| `cuda` **tp=2** (op-by-op `ForwardTP`) | **6.1 tok/s** | 12.9 tok/s |
| `ggml_cuda` tp=2 (fused) | **61.3 tok/s** | 482 tok/s |

Adding a second GPU makes the direct-CUDA backend **~6× slower** than running on one
GPU, and ~10× slower than `ggml_cuda`. The timing breakdown for tp=2 decode shows
**~95–96% of time in "Other"** and only ~4–5% in matmul:

```
Timing (151 forward calls, 22819 ms total, 151 ms/token):
  Linear (matmul): 816 ms (3.6%)
  Attention:       0 ms (0.0%)
  Norm:            1 ms (0.0%)
  (Logits copy:    455 ms, in Other)
  Other:           22002 ms (96.4%)
```

For a 3B-active MoE the actual compute per token is tiny; ~143 ms/token is overhead.

---

## 2. Root cause (confirmed)

`Qwen35Model.ForwardTP` (`TensorSharp.Models/Models/Qwen35/Qwen35Model.TensorParallel.cs`)
runs the 40-layer hybrid model **op-by-op**:

- Each layer dispatches `AttentionBlockTP` / `RecurrentBlockTP` / `MoEBlockTP` as many
  separate kernel launches.
- Every attention output and FFN/MoE output does a **`_tpGroup.AllReduce`** — ~2 per
  layer × 40 layers ≈ **80 AllReduces per token**.
- `CudaP2PCommunicator.AllReduce` (`TensorSharp.Backends.Cuda/CudaP2PCommunicator.cs`)
  performs multiple **host-side `_allocators[i].Synchronize()`** calls per AllReduce on
  tiny `[1, hidden]` (≈8 KB) tensors, so each reduction is latency-bound by device
  synchronization rather than bandwidth.

By contrast, the **single-GPU** path captures the *entire* decode (attention + GDN +
MoE + RoPE position fill) into **one CUDA graph** and replays it in a single launch:

- `Qwen35Model.RunCudaDecodeLayerLoop` (`Qwen35Model.cs`, ~line 1663)
- `CudaPrefillGraphCache` — capture/replay cache keyed by buffer pointers + capacity
- `CudaDecodeDynParams` — a device block holding the per-token dynamic parameters
  (`attendLen`, `startPos`, conv-ring phase) that kernels read on replay; updated via
  `dyn.Write(...)` + `dyn.EnqueueUpload()` + `dyn.Activate()`.

The graph reads the KV cache and GDN recurrent-state buffers **in place** (their device
pointers are baked into the graph key), so a KV-cache reallocation simply changes the
key and triggers a re-capture.

### Why MoE alone is not the fix

A fused per-rank on-device MoE was prototyped (reusing
`CudaFusedOps.TryMoEExpertFFNDecodeSwiGLU` with per-rank pointer tables over the
TP-sharded experts). Result: **correct output, but only ~30% decode improvement
(6.1 → 8.0 tok/s)**, because the MoE per-expert loop was only ~30% of the overhead.
It also required holding all experts device-resident (~8 GB/rank) and added ~6 s of
startup, reversing the deliberate "keep experts on host for KV-cache headroom" design in
`ModelBase.PrepareCudaQuantizedWeightsForInferenceTP`. **That prototype was reverted.**
It is reusable later as Phase 3 (MoE *inside* the graph), where its costs are amortized.

**Conclusion:** the dominant cost is the op-by-op layer loop + ~80 host-synchronized
AllReduces/token. The real fix is a **fused CUDA-graph TP decode**.

---

## 3. The blocker: AllReduce is not graph-capturable

CUDA graph capture requires all work to be **stream-ordered with no host
synchronization**. `CudaP2PCommunicator.AllReduce` currently does:

```csharp
for (int i = 0; i < _worldSize; i++) ((CudaStorage)...).EnsureDeviceCurrent();
for (int i = 0; i < _worldSize; i++) _allocators[i].Synchronize();   // host sync — illegal in capture
... cuMemcpyPeerAsync + LaunchBinaryF32 ...
_allocators[0].Synchronize();                                         // host sync
... broadcast ...
for (int i = 0; i < _worldSize; i++) _allocators[i].Synchronize();   // host sync
```

Those `Synchronize()` calls cannot appear inside a captured graph. A **graph-capturable
AllReduce** (stream-ordered, event-synchronized, no host syncs) is therefore the
critical-path enabler.

---

## 4. Phased implementation plan

### Phase 0 — Profile to confirm the prize (do first; low risk)

Add temporary per-category timing inside `ForwardTP` for a single decode run:
- AllReduce time (sum of `_tpGroup.AllReduce` calls)
- GDN recurrent time
- attention time
- MoE time
- remaining dispatch/other

**Goal:** confirm AllReduce-sync + launch overhead dominate and size the expected win
before committing to Phases 1–3. (Strongly expected to confirm, given the 95% "Other".)

### Phase 1 — Graph-capturable 2-GPU AllReduce (enabler; highest risk)

Implement a stream-ordered reduce + broadcast for `N == 2` (generalize later) using
cross-device peer copies and **CUDA events** (`cuEventRecord` / `cuStreamWaitEvent`)
instead of host `Synchronize()`:

1. Each rank's partial is already produced on its own stream.
2. Record an event on rank1's stream after its partial is written; make rank0's stream
   wait on it.
3. `cuMemcpyPeerAsync` rank1 → rank0 staging **on rank0's stream**.
4. `LaunchBinaryF32` (Add) into rank0's tensor on rank0's stream.
5. Record an event on rank0's stream; make rank1's stream wait; `cuMemcpyPeerAsync`
   rank0 → rank1 (broadcast) on rank1's stream.
6. Mark both tensors device-modified (mirror the existing dirty-tracking handling).

Deliverables:
- New method (e.g. `AllReduceGraphCapturable(Tensor[] tensors)`) in
  `CudaP2PCommunicator` / `TensorParallelGroup`, used only inside graph capture.
- Keep the existing eager `AllReduce` for non-graph paths.
- **Unit test: bit-exact equivalence** vs the eager AllReduce on random tensors
  (see `InferenceWeb.Tests/CudaBackendTests.cs` for existing AllReduce/MoE test patterns)
  — gate integration on this passing.
- Verify L4 supports **graph-captured peer copies**; if not, provide a host-staged
  variant that is still stream-ordered (pinned host staging, async HtoD/DtoH on streams,
  event-synced) and capturable.

### Phase 2 — TP decode graph capture/replay

Mirror `RunCudaDecodeLayerLoop` as `RunCudaDecodeLayerLoopTP`:

- Capture the per-rank `ForwardTP` layer loop on **each rank's stream**, weaving in the
  Phase-1 graph-capturable AllReduce at each reduction point.
- **Graph key:** per-rank KV-cache + GDN delta/conv-state device pointers + KV capacity
  (so a KV reallocation invalidates/rebuilds the graph, exactly like single-GPU).
- **TP dynamic params:** a TP variant of `CudaDecodeDynParams` that feeds
  `attendLen` / `startPos` / conv-ring phase to **both ranks'** kernels on replay
  (either per-rank dyn blocks or one uploaded block read by both).
- **Replay path:** copy input → broadcast to ranks → `dyn.Write(...)` →
  `graphs.Replay(key)` → `AdvanceGdnConvPhases(...)` → `MarkCudaGraphStateModified()`.
- **Fallback:** on any capture abort / unsupported shape / non-graphable op, run the
  existing op-by-op `ForwardTP` (zero regression risk).
- Gate behind an env var (e.g. `TS_Q35_TP_DECODE_GRAPH=1`, opt-in) until proven.

### Phase 3 — MoE inside the graph

Reuse the reverted fused per-rank on-device MoE prototype:
- Build per-rank device pointer tables over the TP-sharded experts
  (`_tpQuantWeights["blk.L.ffn_{gate,up,down}_exps.E.weight"][rank]`), preloading expert
  shards to their rank's device.
- Call `CudaFusedOps.TryMoEExpertFFNDecodeSwiGLU` per-rank with `nFf / GlobalTpDegree`,
  producing a partial `[1, hidden]` that the in-graph AllReduce sums.
- Shared expert: TP column/row-parallel gate/up/down + gated accumulate (the prototype's
  approach), or fold into the kernel with a per-rank `sharedDown`.
- Now the expert-residency memory cost and table-build time are amortized inside the
  captured graph (no separate 6 s startup, no per-token launch storm).

### Phase 4 — Verification & guardrails

- **Correctness:** logit/bit equivalence of graph-TP decode vs op-by-op `ForwardTP` on
  fixed prompts; end-to-end output equivalence; run existing TP / MTP / CUDA backend
  tests (`InferenceWeb.Tests/CudaBackendTests.cs`, `MtpSpeculativeExecutionTests.cs`).
- **Determinism:** repeated runs identical (rules out race/event bugs).
- **Long-context:** force KV-cache growth past the initial 2048 to exercise graph
  re-capture on reallocation.
- **Perf target:** decode ≥ single-GPU (~36 tok/s), ideally toward `ggml_cuda` (~61).
- Promote from opt-in env gate to default once stable.

---

## 5. Key code references

| Concern | Location |
|---|---|
| TP forward (op-by-op) | `TensorSharp.Models/Models/Qwen35/Qwen35Model.TensorParallel.cs` — `ForwardTP`, `AttentionBlockTP`, `RecurrentBlockTP`, `MoEBlockTP`, `TpColumnParallelLinear`, `TpRowParallelLinear`, `TpExpertLinear` |
| Single-GPU fused decode graph | `Qwen35Model.cs` — `RunCudaDecodeLayerLoop`, `RunPerOpLayerLoop`, `AdvanceGdnConvPhases`, `EnsureDecodeGraphInputsResident`, `MarkCudaGraphStateModified` |
| Graph cache + dyn params | `CudaPrefillGraphCache`, `CudaDecodeDynParams` (Qwen35 model dir) |
| AllReduce (eager, host-sync) | `TensorSharp.Backends.Cuda/CudaP2PCommunicator.cs` — `AllReduce`, `AllReduceViaHost`, `BroadcastViaHost`; `TensorSharp.Backends.Cuda/TensorParallelGroup.cs` — `AllReduce` |
| On-device fused MoE (single-GPU) | `Qwen35Model.CudaMoE.cs` — `EnsureQwenCudaMoETablesBuilt`, `TryCudaMoEForwardOnDevice`; kernel `CudaFusedOps.TryMoEExpertFFNDecodeSwiGLU` |
| TP expert sharding layout | `Qwen35Model.TensorParallel.cs` — `ShardMoeWeightsForTP`, `ShardExpertColumnParallel` (gate/up: `[hidden, nFf/tp]`), `ShardExpertRowParallel` (down: `[nFf/tp, hidden]`) |
| TP weight device residency | `ModelBase.cs` — `PrepareCudaQuantizedWeightsForInferenceTP` (note: deliberately skips `_exps.` weights) |
| Residency primitives | `TensorSharp.Backends.Cuda/CudaQuantizedOps.cs` — `PreloadQuantizedWeight`, `TryGetResidentDevicePtr`, `CreateDevicePointerTable` |

---

## 6. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Multi-device CUDA graph capture with cross-stream events is unsupported/flaky on some drivers/topologies | Verify graph-captured peer copy on L4 first; provide a host-staged-but-stream-ordered AllReduce fallback; keep eager path + opt-in gate |
| GDN recurrent state (conv ring + delta) updated in place under capture diverges from eager bookkeeping → output drift | Mirror `AdvanceGdnConvPhases` exactly; key graph on conv phase; validate logits vs eager path |
| Graph pins buffers; KV/state reallocation invalidates pointers | Key the graph on per-rank buffer pointers + capacity (re-capture on grow), as single-GPU does |
| AllReduce correctness regression | Bit-exact unit test vs eager AllReduce before integration |
| Expert device-residency memory pressure (Phase 3) | Opt-in; measure KV-cache headroom; fall back to host-served experts if near-full |

---

## 7. Effort estimate

A multi-day feature. Suggested order: **Phase 0** (quick, de-risks the rest) → **Phase 1**
(critical-path risk, unit-tested in isolation) → **Phase 2** → **Phase 3** → **Phase 4**.
Do not start Phase 2 until Phase 1's AllReduce is proven bit-exact and graph-capturable on
the target hardware.

---

## 8. Appendix — environment & reproduction notes

- The test host required `LD_LIBRARY_PATH` **without** `/usr/local/cuda-12.6/compat`
  (the forward-compat `libcuda` stub conflicts with the newer installed driver and yields
  `CUDA error 803`). Use:
  `LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64`.
- Repro command (decode speed):
  `./TensorSharp.Cli/bin/TensorSharp.Cli --model models/Qwen3.5-35B-A3B-UD-IQ4_XS.gguf --backend cuda --max-tokens 300 --tp 2 --input <prompt>`.
- Toggle the single-GPU on-device MoE with `TS_CUDA_MOE_ONDEVICE=0` to isolate its
  contribution (single-GPU: 36.3 → 25.2 tok/s when disabled).
- Separate already-fixed issue noted during investigation: MTP was wrongly reported
  "incomplete" under TP because the fused Q/K/V moves into the `_tpQuantWeights` shard
  tables; fixed in `Qwen35Model.Mtp.cs` (`HasMtp` now consults the TP shard tables).
