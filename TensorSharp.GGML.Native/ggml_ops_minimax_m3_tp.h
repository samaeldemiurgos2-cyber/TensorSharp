// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Tensor parallelism for the MiniMax-M3 native executor
// (ggml_ops_minimax_m3_exec.cpp). The executor runs its ordinary graph on
// upstream ggml's meta backend (the mechanism behind llama.cpp's -sm tensor)
// over the first N CUDA GPUs, and this file decides how every statically
// allocated tensor is divided among them.
//
// MiniMax-M3 has 4 KV heads, and its sparse attention (MSA) cannot be split by
// head on the meta backend at all: the block scores' max-pool (pool_2d) and
// TensorSharp's custom nodes must have every input mirrored, ggml_arange has
// no source the meta backend could take a split from, and flash attention with
// the KV groups on ne[3] (MSA's layout) is not one of its split patterns. So:
//   * attention, the indexer and every slot's K, V and index-key caches are
//     MIRRORED: each GPU holds them whole and computes the layer's attention
//     itself (dense below the dense span, upstream ggml's sparse ops past it);
//   * the FFN is split along n_ff: the leading dense layers' gate/up rows and
//     down columns, every routed expert's and the shared expert's, in
//     multiples of lcm(down block, 128) (NVFP4 64 and Q8_0 32 give 128), so a
//     layer ends in ONE all-reduce: the meta backend delays the routed
//     experts' partial sum through the scale and weight multiplies and the
//     n_used views, and merges it with the shared expert's (llama.cpp's node
//     order);
//   * the output head by vocabulary rows (the logits are gathered on read);
//   * everything else (norms, router, per-expert scales) is mirrored.
// The distribution is llama.cpp's (src/llama-model.cpp, 7170cad): per tensor,
// device j's share ends at floor(ne * (j + 1) / n) rounded down to the
// granularity, the last device takes the rest, and layer il rotates the shares
// by il % n so per-GPU totals even out.
//
// The meta backend asserts (aborts the process) when one GPU cannot allocate
// its slice of a buffer, so under --tp the executor never relies on a failed
// allocation to recover: the load checks every GPU against the budget below,
// slot admission keeps room for the largest graph, an unadmitted slot growth
// is refused instead of attempted, and a new graph's compute buffers (plus any
// growth of the reduction scratch) are measured against the tightest GPU
// before anything is allocated: retained slots are freed while it does not
// fit, and a graph that still does not fit is declined (the call retries in
// smaller ubatches) instead of reaching the allocation.
//
// The plan, the split states and the memory budget are pure (no backend
// calls) and live in ggml_ops_minimax_m3_tp.cpp, testable on any host; the
// backend helpers (opening the meta device, checking a graph, copying cache
// rows) live in ggml_ops_minimax_m3_tp_backend.cpp.
#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace tsg_mm3_tp
{
    constexpr int MAX_DEVICES = GGML_BACKEND_META_MAX_DEVICES;

    /// Names of the executor's own constant tensors (mirrored): 0..block-1 and
    /// 0..n_head_kv-1 as F32, for the sparse decode's row arithmetic.
    constexpr const char * ARANGE_ROWS = "msa.arange_rows";
    constexpr const char * ARANGE_HEADS = "msa.arange_heads";

    struct shape
    {
        int n_layer = 0;
        int n_lead = 0;                             // leading dense layers (dense FFN, no experts)
        int64_t n_ff = 0;                           // dense FFN length
        int64_t n_ff_exp = 0;                       // routed-expert FFN length
        int64_t n_ff_shexp = 0;                     // shared-expert FFN length
        int64_t n_vocab = 0;
        ggml_type ffn_down_type = GGML_TYPE_F32;    // ffn_down: the dense granularity
        ggml_type exp_down_type = GGML_TYPE_F32;    // ffn_down_exps: the routed granularity
        ggml_type shexp_down_type = GGML_TYPE_F32;  // ffn_down_shexp: the shared granularity
    };

    using row = std::array<int64_t, MAX_DEVICES>;

    struct plan
    {
        int n = 0;
        std::vector<row> ffn;       // [layer][device] n_ff elements: the dense FFN (leading layers) or the routed experts
        std::vector<row> shexp;     // [layer][device] shared-expert n_ff elements (MoE layers; zero on dense ones)
        row out_rows{};             // [device] vocabulary rows of output.weight
        int64_t g_dense = 0, g_exp = 0, g_shexp = 0;   // granularity of each FFN family
    };

    /// The plan for n devices, or false with the reason in *error.
    bool make_plan(const shape & s, int n, plan * out, std::string * error);

    struct userdata
    {
        shape s;
        plan p;
        std::atomic<bool> unplanned{ false };   // a tensor the plan does not know reached a meta buffer
        std::mutex mu;
        std::string unplanned_name;
    };

    /// The split state of a statically allocated tensor, by its name:
    /// "blk.N.<weight>" (the fused "blk.N.attn_qkv.weight" and
    /// "blk.N.indexer.qk_proj.weight" included), "output.weight",
    /// "output_norm.weight", "slotS_k.N" / "slotS_v.N" / "slotS_ik.N" (a
    /// slot's caches: mirrored) and ARANGE_ROWS / ARANGE_HEADS. Any other name,
    /// or a shape the plan does not expect, is mirrored and recorded in
    /// ud->unplanned.
    ggml_backend_meta_split_state split_state(const ggml_tensor * t, userdata & ud);

    /// ggml_backend_meta_get_split_state_t over a userdata*.
    ggml_backend_meta_split_state split_state_cb(const ggml_tensor * t, void * ud);

    /// Bytes device j holds of a tensor with this name, type and shape (the
    /// whole tensor when mirrored). Rows are not padded.
    size_t device_bytes(const userdata & ud, const char * name, ggml_type type, const int64_t ne[4], int j);

    /// Bytes per upload call into tensor t: the meta buffer writes a split
    /// tensor only in whole slabs along its split axis (rows for axis 0, whole
    /// matrices for axis 1), so `chunk` is rounded to a multiple of the slab
    /// (at least one). Mirrored tensors take `chunk` as it is.
    size_t upload_step(const ggml_tensor * t, userdata & ud, size_t chunk);

    /// What each GPU must hold.
    struct budget
    {
        int n = 0;
        size_t weights[MAX_DEVICES] = {};   // this GPU's slices plus every mirrored tensor
        size_t free[MAX_DEVICES] = {};      // free device memory before the load
        size_t reserve = 0;                 // compute buffers, scratch pools, reduction scratch (every GPU)
        size_t dev0_extra = 0;              // GPU 0 only (the managed side's vision encoder)
        size_t kv_per_token = 0;            // every GPU: the caches are mirrored
    };

    /// The meta backend's all-reduce scratch on every GPU for graphs of up to
    /// `rows` tokens: ceil(log2 n) buffers (its n_reduce_steps), each the size
    /// of the graph's largest PARTIAL node, kept for the backend's lifetime
    /// (ggml-backend-meta.cpp, graph_compute). MiniMax-M3's largest is the
    /// routed experts' down projection, F32 [n_embd, n_expert_used, rows] (the
    /// shared expert's and the dense FFN's are [n_embd, rows]).
    size_t reduction_scratch_bytes(int n, int64_t n_embd, int64_t n_expert_used, int64_t rows);

    /// Bytes GPU j needs for its weights, its reserve and kv_tokens rows of every slot cache.
    size_t need_bytes(const budget & b, int j, int64_t kv_tokens);

    /// The first GPU that cannot hold kv_tokens cache rows next to its weights and reserve, or -1.
    int misfit(const budget & b, int64_t kv_tokens);

    /// The most cache rows (a multiple of pad, at most cap) that every GPU can
    /// hold; 0 when not even one pad does.
    int64_t max_kv_tokens(const budget & b, int64_t pad, int64_t cap);

    // ---- backend side (ggml_ops_minimax_m3_tp_backend.cpp) ------------------

    struct devices
    {
        int n = 0;
        ggml_backend_dev_t devs[MAX_DEVICES] = {};
        ggml_backend_dev_t meta = nullptr;
    };

    /// One ggml meta backend over the first `tp` CUDA GPUs (CUDA_VISIBLE_DEVICES
    /// selects them), which must be identical (every GPU routes the experts
    /// itself from mirrored activations: unlike GPUs could pick differently and
    /// sum mismatched partials without an error). Prepares the collective
    /// first (NCCL without P2P on a host whose P2P is broken). Null with the
    /// reason in *why.
    ggml_backend_t open(int tp, userdata * ud, devices * out, std::string * why);

    /// Free and total memory of the tightest GPU (the meta device reports sums,
    /// but each GPU holds its own slices).
    void memory(const devices & d, size_t * free_min, size_t * total_min);

    /// Every layer op runs on the meta backend: an op the GPUs cannot run would
    /// be scheduled on the CPU and come back as a whole (mirrored) tensor, which
    /// the meta backend cannot combine with its GPU slices (it aborts). Checked
    /// before allocation so the call fails with a reason instead. Nodes reading
    /// `host_table` (the token-row gather) belong on the CPU.
    bool check_graph(ggml_backend_t meta, ggml_cgraph * gf, const ggml_tensor * host_table, std::string * why);

    /// The first `rows` rows of src copied into dst.
    struct row_copy { ggml_tensor * src; ggml_tensor * dst; int64_t rows; };

    /// Copy cache rows on the meta backend: the slices of a tensor live on
    /// every GPU and the meta buffer cannot copy, so one graph of copies runs
    /// on the meta backend (each GPU copies its own rows, nothing crosses the
    /// host). The meta backend keeps per-GPU views only for the graph it ran
    /// last: the caller drops its cached graphs afterwards.
    bool copy_rows(ggml_backend_t meta, const std::vector<row_copy> & copies);
}
