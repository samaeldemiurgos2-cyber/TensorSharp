// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace tsg_dsv41_tp
{
struct source
{
    std::string path;
    size_t offset = 0;
    ggml_type type = GGML_TYPE_F32;
    std::array<int64_t, 4> ne = {1, 1, 1, 1};
};

struct strip
{
    int64_t first = 0;
    int64_t count = 0;
};

// A contiguous, complete tiling in aligned blocks.
// Rotating the extra blocks by layer balances uneven V4.1 strips.
// Unquantized strips also honor ggml CUDA's two-element vector alignment.
std::vector<strip> split(int64_t width, int64_t block, int ranks, int layer);

// Prefer 64-channel floating-point strips where possible. This keeps BF16/F16
// gate/up rows eligible for the same matrix path as the full tensor, avoiding
// shape-dependent activation rounding. Use this for
// both upload layout and loader memory pricing. Quantized/F32 layouts retain
// their original block policy; small floating tensors retain vector alignment.
std::vector<strip> split_weights(int64_t width, ggml_type down_type, int ranks, int layer);
std::vector<strip> split_outputs(int64_t width, int ranks, int layer);

class executor
{
public:
    // Each rank owns a separate backend queue for the supplied device. CPU
    // devices are accepted for the independent numerical regression test.
    // Sequential pread warms only the current layer before its strided copies.
    // Zero preserves direct reads; the native loader supplies its warm policy.
    executor(const std::vector<ggml_backend_dev_t> & devices, int used_experts, int load_threads = 16);
    ~executor();
    executor(const executor &) = delete;
    executor & operator=(const executor &) = delete;

    // Shard gate/up intermediate rows and down output rows. Gather activations
    // between projections to retain the complete down reduction dimension.
    // No full expert tensor is allocated on any participating rank.
    void add_layer(int layer, const source & gate, const source & up,
                   const source & down, float clamp_limit);
    // Release each layer's source pages once its strips are uploaded (the
    // loader's TS_DSV4_LOAD_DROP_CACHE rule). Set before add_layer.
    void set_drop_source_cache(bool drop);
    bool has_layer(int layer) const;
    size_t rank_weight_bytes(int rank) const;

    // F32 x[embedding,tokens], routing weights[1,used,tokens], and I32 expert
    // IDs[used,tokens] -> routed output[embedding,tokens], as a
    // TSG_DSV41_TP_MOE fused node. Pin it to the layer device's fused CUDA
    // backend when device_resident(): the node then enqueues every rank behind
    // that device's stream with no host synchronization. Pinned to the CPU
    // backend it drives the ranks synchronously from the host instead.
    //
    // With `after` (device-resident only), build a launch node and a gather
    // node instead: ggml then orders `after` (the shared expert) between
    // them, so the layer device computes it while the ranks run. The returned
    // gather's src[0] is the launch node; pin both to the layer device.
    ggml_tensor * build(ggml_context * ctx, int layer, ggml_tensor * x,
                        ggml_tensor * weights, ggml_tensor * ids, ggml_tensor * after = nullptr);
    // Every rank is a CUDA device, so build()'s node can run on a fused
    // CUDA backend (TensorSharp's ggml_ops_dsv4_fused).
    bool device_resident() const;

    // Native graph callbacks cannot return a status. They fill NaNs on error;
    // callers must inspect this after graph execution and refuse generation.
    // Start once per validated request, never once per layer: a later healthy
    // layer must not conceal an earlier rank failure in the same request.
    void begin_forward();
    std::string error() const;

#if defined(TSG_GGML_TEST_HOOKS)
    void test_set_position(int64_t position);
    void test_single_fanout(bool enabled);
    void test_pinned_staging(bool enabled);
    void test_pipelined_upload(bool enabled);
    void test_device_gather(bool enabled);
    void test_gate_fence(bool enabled);
#endif

private:
    struct impl;
    std::unique_ptr<impl> state;
};
}
