// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// DSpark speculative drafter for the MiniMax-M3 whole-model executor: the
// state the executor carries and the hooks it calls. Only
// ggml_ops_minimax_m3_exec.cpp includes this header; the implementation is
// ggml_ops_minimax_m3_dspark.inc, which the executor includes after its own
// definitions (it needs the model, slot and graph-builder types).
//
// The drafter (GGUF arch "dflash" with a Markov head: MiniMax-M3.1-Flash's
// DSpark) reads the OUTPUTS of the target layers its dflash.target_layers
// name (the "input of layer k" convention: k = n_layer is the final pre-norm
// stream). Its sliding-window context is committed INSIDE the trunk graph:
// every trunk forward taps those layers, projects the concatenated rows
// through fc and enc.output_norm, and writes each draft layer's K/V for every
// row it processed into the slot's ring. A draft is then one graph over
// [anchor, mask x (block - 1)] at positions n..n+block-1 and needs nothing from
// the host but the anchor token.
#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace tsg_mm3
{

struct model;
struct slot;
struct builder;
struct tensor_source;

struct dspark_layer
{
    ggml_tensor * attn_norm = nullptr;
    ggml_tensor * wq = nullptr;
    ggml_tensor * wk = nullptr;
    ggml_tensor * wv = nullptr;
    ggml_tensor * wo = nullptr;
    ggml_tensor * q_norm = nullptr;
    ggml_tensor * k_norm = nullptr;
    ggml_tensor * ffn_norm = nullptr;
    ggml_tensor * ffn_gate = nullptr;
    ggml_tensor * ffn_up = nullptr;
    ggml_tensor * ffn_down = nullptr;
};

/// One sequence slot's draft context: per draft layer, F16 K and V rows
/// [n_head_kv * head_dim, rows] on the drafter's GPU. Row r holds position
/// pos[r] (r == pos[r] mod rows), or nothing (-1). The draft masks every row
/// by the position it HOLDS, so a row that a rejected verify, a rewind or an
/// earlier draft block left behind is never read as context.
struct dspark_ring
{
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    std::vector<ggml_tensor *> k, v;
    std::vector<int64_t> pos;

    ~dspark_ring()
    {
        if (buf) ggml_backend_buffer_free(buf);
        if (ctx) ggml_free(ctx);
    }
};

/// The draft graph of one slot (its ring is baked in). Fixed shape: built and
/// allocated once, then replayed (and on CUDA, its captured graph with it).
struct dspark_graph
{
    int slot_id = -1;
    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_backend_sched_t sched = nullptr;
    ggml_tensor * emb = nullptr;      // F32 [n_embd, block]
    ggml_tensor * pos = nullptr;      // I32 [block]
    ggml_tensor * ridx = nullptr;     // I32 [block] ring rows of the block
    ggml_tensor * mask = nullptr;     // F16 [rows, block]
    ggml_tensor * anchor = nullptr;   // I32 [1]
    ggml_tensor * toks = nullptr;     // F32 [block] drafted token ids
    ggml_tensor * conf = nullptr;     // F32 [block] confidence (null without the head)

    ~dspark_graph()
    {
        if (sched) ggml_backend_sched_free(sched);
        if (ctx) ggml_free(ctx);
    }
};

struct dspark_state
{
    bool loaded = false;
    std::string path;
    int dev = 0;                      // the output head's device: the drafter reads the target's head
    int32_t n_layer = 0, n_embd = 0, n_head = 0, n_head_kv = 0, head_dim = 0, n_rot = 0, n_ff = 0;
    int32_t block = 0;                // rows of one draft (anchor + masks), every row a prediction
    int32_t window = 0;               // sliding window of the context attention
    int32_t mask_token = 0;
    int32_t rank = 0;                 // Markov head rank
    float rms_eps = 1e-6f;
    float rope_base = 10000.0f;
    bool causal = false;              // dflash.attention.causal: causal inside the block (M3.1: absent, non-causal)
    std::vector<int32_t> target_layers;   // feature order; target layer k-1's output is feature row k
    int64_t rows = 0;                 // ring rows per slot (a power of two)
    size_t weight_bytes = 0;

    ggml_tensor * fc = nullptr;       // [n_taps * n_embd, n_embd]
    ggml_tensor * enc_norm = nullptr;
    ggml_tensor * out_norm = nullptr;
    ggml_tensor * markov_w1 = nullptr;   // [rank, n_vocab]
    ggml_tensor * markov_w2 = nullptr;   // [rank, n_vocab]
    ggml_tensor * conf_w = nullptr;      // [n_embd + rank, 1]
    ggml_tensor * conf_b = nullptr;      // [1]
    std::vector<dspark_layer> layers;
    std::vector<float> mask_row;      // embedding row of mask_token (F32)

    std::map<int, std::unique_ptr<dspark_ring>> rings;   // by slot id
    std::list<std::unique_ptr<dspark_graph>> graphs;      // most recently used first
    int graph_cap = 4;
    bool warned_no_ring = false;

    uint64_t n_drafts = 0;
    uint64_t n_graph_builds = 0;
    double draft_ms = 0.0;

    void release()
    {
        graphs.clear();
        rings.clear();
    }
};

// ---- hooks the executor calls (defined in ggml_ops_minimax_m3_dspark.inc) ----

/// Open the drafter (none when path is null or empty), check it against the
/// target, append its file to the model's shards and register its tensors as
/// "dspark.<name>". *placement_bytes receives what the output head's device
/// must also hold for it. False after a load refusal.
static bool dspark_scan(model & m, const char * path, std::map<std::string, tensor_source> & sources,
                        size_t * placement_bytes);
/// Create the drafter's weight tensors on the output head's device.
static bool dspark_make_weights(model & m, const std::function<ggml_tensor * (int, const char *, bool)> & make);
/// After the weights are uploaded, before the first slot: ring geometry, the mask row.
static bool dspark_finish_load(model & m);
static void dspark_slot_added(model & m, const slot & s);
static void dspark_slot_removed(model & m, int slot_id);
static bool dspark_slot_copy(model & m, const slot & src, const slot & dst);
/// A trunk graph wrote the drafter context of positions p0..p0+n-1 of slot s.
static void dspark_rows_written(model & m, const slot & s, int64_t p0, int64_t n);
/// The drafter taps the final pre-norm stream: the last layer runs every row.
static bool dspark_taps_final(const model & m);
/// Called with the output of layer il (the residual after its FFN); the
/// drafter's context commit is built once every tap is in.
static void dspark_tap(builder & b, int il, ggml_tensor * x);
/// Extra graph nodes a trunk graph of n_segments segments needs for the commit.
static size_t dspark_graph_nodes(const model & m, size_t n_segments);

} // namespace tsg_mm3
