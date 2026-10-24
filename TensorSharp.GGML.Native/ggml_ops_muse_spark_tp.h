// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Tensor-parallel split plan for the Muse Spark native executor. The executor
// runs its ordinary graph on upstream ggml's meta backend (the mechanism behind
// llama.cpp's -sm tensor), and this file decides how every statically allocated
// tensor (weights and K/V slots) is divided among the GPUs:
//   * attention by KV group: q/gate rows, k/v rows and the K/V cache columns of
//     a group stay on one GPU, and attn_output is split along its input, so the
//     layer's attention ends in one all-reduce;
//   * routed experts along n_ff: gate/up rows and down columns of every expert,
//     so the MoE ends in a second all-reduce;
//   * the output head by vocabulary rows (the logits are gathered on read);
//   * everything else (norms, router, latent projections) on every GPU.
// The distribution is llama.cpp's (src/llama-model.cpp, 7170cad): per tensor,
// device j's share ends at floor(ne * (j + 1) / n) rounded down to the tensor's
// granularity, and each layer rotates which device takes which share, counting
// only earlier layers of the same kind (sliding or global), so that per-GPU
// totals even out. Pure: no ggml backend calls, so it is testable on any host.
#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace tsg_mspark_tp
{
    constexpr int MAX_DEVICES = GGML_BACKEND_META_MAX_DEVICES;

    struct shape
    {
        int n_layer = 0;
        int n_head = 0;
        int n_head_kv = 0;
        int head_dim = 0;
        int64_t n_ff_exp = 0;
        int64_t n_vocab = 0;
        std::vector<uint8_t> is_swa;            // [n_layer]
        ggml_type wo_type = GGML_TYPE_F32;      // attn_output (sets the attention granularity)
        ggml_type down_type = GGML_TYPE_F32;    // ffn_down_exps (sets the expert granularity)
    };

    using row = std::array<int64_t, MAX_DEVICES>;

    struct plan
    {
        int n = 0;
        std::vector<row> groups;    // [layer][device] KV groups (heads of K/V; n_gqa query heads each)
        std::vector<row> ffn;       // [layer][device] expert n_ff elements
        row out_rows{};             // [device] vocabulary rows of output.weight
        int64_t ffn_granularity = 0;
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
    /// "blk.N.<weight>", "output.weight", "slotS_k.N" / "slotS_v.N" (K/V of
    /// slot S, layer N). Any other name is mirrored and, unless it is a known
    /// mirrored weight, recorded in ud->unplanned. A shape that does not match
    /// the plan is recorded the same way.
    ggml_backend_meta_split_state split_state(const ggml_tensor * t, userdata & ud);

    /// ggml_backend_meta_get_split_state_t over a userdata*.
    ggml_backend_meta_split_state split_state_cb(const ggml_tensor * t, void * ud);

    /// Bytes device j holds of a tensor with this name, type and shape
    /// (the whole tensor when mirrored). Rows are not padded.
    size_t device_bytes(const userdata & ud, const char * name, ggml_type type, const int64_t ne[4], int j);
}
