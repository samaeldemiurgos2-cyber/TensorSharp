// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// The MiniMax-M3 tensor-parallel split plan (ggml_ops_minimax_m3_tp.cpp) on the
// real checkpoint's shapes (MiniMax-M3.1-Flash UD-Q4_K_XL: 60 layers, 3 dense,
// NVFP4 experts, Q8_0 attention / shared expert / dense FFN, BF16 head):
// llama.cpp's distribution and rotation, block alignment, whole tensors
// accounted for, every name the executor allocates classified (attention and
// caches mirrored), the upload slabs, the per-GPU memory budget and every
// refusal. Pure planning: no GPU.
#include "ggml_ops_minimax_m3_tp.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace
{
    int g_failures = 0;

#define CHECK(cond) do { if (!(cond)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); g_failures++; } } while (0)

    constexpr int N_LAYER = 60, N_LEAD = 3;
    constexpr int64_t N_EMBD = 6144, N_FF = 12288, N_FF_EXP = 3072, N_EXPERT = 128, N_USED = 4, N_VOCAB = 200064;

    tsg_mm3_tp::shape minimax_m31_flash()
    {
        tsg_mm3_tp::shape s;
        s.n_layer = N_LAYER;
        s.n_lead = N_LEAD;
        s.n_ff = N_FF;
        s.n_ff_exp = N_FF_EXP;
        s.n_ff_shexp = N_FF_EXP;
        s.n_vocab = N_VOCAB;
        s.ffn_down_type = GGML_TYPE_Q8_0;
        s.exp_down_type = GGML_TYPE_NVFP4;
        s.shexp_down_type = GGML_TYPE_Q8_0;
        return s;
    }

    struct tensor_desc { std::string name; ggml_type type; int64_t ne[4]; };

    /// Every tensor of the real checkpoint (as the GGUF lists them), plus the
    /// executor's fused attention/indexer projections, the F16 decode copies of
    /// the F32 indexer projections (TS_MINIMAX_M3_INDEXER_F16, on by default on
    /// GPUs) and its constants.
    std::vector<tensor_desc> real_tensors(bool fused)
    {
        std::vector<tensor_desc> v;
        auto add = [&](std::string n, ggml_type t, int64_t a, int64_t b = 1, int64_t c = 1) { v.push_back({ n, t, { a, b, c, 1 } }); };
        add("output.weight", GGML_TYPE_BF16, N_EMBD, N_VOCAB);
        add("output_norm.weight", GGML_TYPE_F32, N_EMBD);
        for (int il = 0; il < N_LAYER; il++)
        {
            const std::string p = "blk." + std::to_string(il) + ".";
            add(p + "attn_norm.weight", GGML_TYPE_F32, N_EMBD);
            add(p + "ffn_norm.weight", GGML_TYPE_F32, N_EMBD);
            add(p + "attn_q_norm.weight", GGML_TYPE_F32, 128);
            add(p + "attn_k_norm.weight", GGML_TYPE_F32, 128);
            if (fused) add(p + "attn_qkv.weight", GGML_TYPE_Q8_0, N_EMBD, 8192 + 2 * 512);
            else
            {
                add(p + "attn_q.weight", GGML_TYPE_Q8_0, N_EMBD, 8192);
                add(p + "attn_k.weight", GGML_TYPE_Q8_0, N_EMBD, 512);
                add(p + "attn_v.weight", GGML_TYPE_Q8_0, N_EMBD, 512);
            }
            add(p + "attn_output.weight", GGML_TYPE_Q8_0, 8192, N_EMBD);
            add(p + "indexer.q_norm.weight", GGML_TYPE_F32, 128);
            add(p + "indexer.k_norm.weight", GGML_TYPE_F32, 128);
            if (fused)
            {
                add(p + "indexer.qk_proj.weight", GGML_TYPE_F32, N_EMBD, 512 + 128);
                add(p + "indexer.qk_proj.f16", GGML_TYPE_F16, N_EMBD, 512 + 128);
            }
            else
            {
                add(p + "indexer.q_proj.weight", GGML_TYPE_F32, N_EMBD, 512);
                add(p + "indexer.k_proj.weight", GGML_TYPE_F32, N_EMBD, 128);
                add(p + "indexer.q_proj.weight.f16", GGML_TYPE_F16, N_EMBD, 512);
                add(p + "indexer.k_proj.weight.f16", GGML_TYPE_F16, N_EMBD, 128);
            }
            if (il < N_LEAD)
            {
                add(p + "ffn_gate.weight", GGML_TYPE_Q8_0, N_EMBD, N_FF);
                add(p + "ffn_up.weight", GGML_TYPE_Q8_0, N_EMBD, N_FF);
                add(p + "ffn_down.weight", GGML_TYPE_Q8_0, N_FF, N_EMBD);
                continue;
            }
            add(p + "ffn_gate_inp.weight", GGML_TYPE_F32, N_EMBD, N_EXPERT);
            add(p + "exp_probs_b.bias", GGML_TYPE_F32, N_EXPERT);
            add(p + "ffn_gate_exps.weight", GGML_TYPE_NVFP4, N_EMBD, N_FF_EXP, N_EXPERT);
            add(p + "ffn_up_exps.weight", GGML_TYPE_NVFP4, N_EMBD, N_FF_EXP, N_EXPERT);
            add(p + "ffn_down_exps.weight", GGML_TYPE_NVFP4, N_FF_EXP, N_EMBD, N_EXPERT);
            add(p + "ffn_gate_exps.scale", GGML_TYPE_F32, N_EXPERT);
            add(p + "ffn_up_exps.scale", GGML_TYPE_F32, N_EXPERT);
            add(p + "ffn_down_exps.scale", GGML_TYPE_F32, N_EXPERT);
            add(p + "ffn_gate_shexp.weight", GGML_TYPE_Q8_0, N_EMBD, N_FF_EXP);
            add(p + "ffn_up_shexp.weight", GGML_TYPE_Q8_0, N_EMBD, N_FF_EXP);
            add(p + "ffn_down_shexp.weight", GGML_TYPE_Q8_0, N_FF_EXP, N_EMBD);
        }
        add(tsg_mm3_tp::ARANGE_ROWS, GGML_TYPE_F32, 128);
        add(tsg_mm3_tp::ARANGE_HEADS, GGML_TYPE_F32, 4);
        return v;
    }

    ggml_backend_meta_split_state state_of(tsg_mm3_tp::userdata & ud, const char * name, ggml_type type,
                                           int64_t ne0, int64_t ne1 = 1, int64_t ne2 = 1)
    {
        ggml_tensor t;
        std::memset(&t, 0, sizeof(t));
        t.type = type;
        t.ne[0] = ne0; t.ne[1] = ne1; t.ne[2] = ne2; t.ne[3] = 1;
        std::snprintf(t.name, sizeof(t.name), "%s", name);
        return tsg_mm3_tp::split_state(&t, ud);
    }

    void test_real_model_plan()
    {
        tsg_mm3_tp::userdata ud;
        ud.s = minimax_m31_flash();
        std::string err;
        CHECK(tsg_mm3_tp::make_plan(ud.s, 8, &ud.p, &err));
        const auto & p = ud.p;
        // lcm(64 NVFP4, 128) = 128 = lcm(32 Q8_0, 128): 24 routed / shared units, 96 dense ones.
        CHECK(p.g_exp == 128 && p.g_shexp == 128 && p.g_dense == 128);
        for (int j = 0; j < 8; j++)
        {
            CHECK(p.ffn[0][(size_t) j] == N_FF / 8);
            CHECK(p.ffn[3][(size_t) j] == N_FF_EXP / 8);
            CHECK(p.shexp[3][(size_t) j] == N_FF_EXP / 8);
            CHECK(p.shexp[0][(size_t) j] == 0);   // dense layers have no shared expert
            CHECK(p.out_rows[(size_t) j] == N_VOCAB / 8);
        }
        // Every degree that leaves each GPU some rows: sums, alignment, rotation.
        for (int n = 2; n <= 16; n++)
        {
            tsg_mm3_tp::plan q;
            const bool ok = tsg_mm3_tp::make_plan(ud.s, n, &q, &err);
            CHECK(ok);   // 24 routed units of 128: every GPU of up to 16 gets some
            if (!ok) continue;
            for (int il = 0; il < N_LAYER; il++)
            {
                int64_t f = 0, s = 0;
                for (int j = 0; j < n; j++)
                {
                    const int64_t fj = q.ffn[(size_t) il][(size_t) j], sj = q.shexp[(size_t) il][(size_t) j];
                    f += fj;
                    s += sj;
                    CHECK(fj > 0 && fj % (il < N_LEAD ? q.g_dense : q.g_exp) == 0);
                    CHECK(il < N_LEAD ? sj == 0 : (sj > 0 && sj % q.g_shexp == 0));
                }
                CHECK(f == (il < N_LEAD ? N_FF : N_FF_EXP));
                CHECK(s == (il < N_LEAD ? 0 : N_FF_EXP));
                // layer il takes layer il - 1's shares one device further on
                if (il > N_LEAD)
                    for (int j = 0; j < n; j++)
                        CHECK(q.ffn[(size_t) il][(size_t) ((j + 1) % n)] == q.ffn[(size_t) il - 1][(size_t) j]);
            }
            int64_t rows = 0;
            for (int j = 0; j < n; j++) rows += q.out_rows[(size_t) j];
            CHECK(rows == N_VOCAB);
        }
        // 5 GPUs: 24 units do not divide, so the rotation spreads the larger shares.
        tsg_mm3_tp::plan q5;
        CHECK(tsg_mm3_tp::make_plan(ud.s, 5, &q5, &err));
        int64_t total5[5] = {};
        for (int il = N_LEAD; il < N_LAYER; il++)
            for (int j = 0; j < 5; j++) total5[j] += q5.ffn[(size_t) il][(size_t) j];
        for (int j = 1; j < 5; j++) CHECK(llabs(total5[j] - total5[0]) <= 128);
    }

    void test_split_states()
    {
        tsg_mm3_tp::userdata ud;
        ud.s = minimax_m31_flash();
        std::string err;
        CHECK(tsg_mm3_tp::make_plan(ud.s, 4, &ud.p, &err));
        using A = ggml_backend_meta_split_axis;
        // Attention, indexer, router, scales and caches: mirrored.
        for (const char * n : { "blk.5.attn_q.weight", "blk.5.attn_qkv.weight", "blk.5.attn_output.weight",
                                "blk.5.indexer.qk_proj.weight", "blk.5.indexer.q_proj.weight", "blk.5.indexer.k_norm.weight",
                                "blk.5.indexer.qk_proj.f16", "blk.5.indexer.q_proj.weight.f16", "blk.5.indexer.k_proj.weight.f16",
                                "blk.5.ffn_gate_inp.weight", "blk.5.exp_probs_b.bias", "blk.5.ffn_down_exps.scale",
                                "output_norm.weight", "slot0_k.5", "slot3_v.59", "slot12_ik.0", tsg_mm3_tp::ARANGE_ROWS })
            CHECK(state_of(ud, n, GGML_TYPE_F32, 128).axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        CHECK(!ud.unplanned.load());
        // Routed experts: gate/up rows (axis 1), down columns (axis 0), the layer's shares.
        auto st = state_of(ud, "blk.10.ffn_gate_exps.weight", GGML_TYPE_NVFP4, N_EMBD, N_FF_EXP, N_EXPERT);
        CHECK(st.axis == (A) 1);
        for (int j = 0; j < 4; j++) CHECK(st.ne[j] == ud.p.ffn[10][(size_t) j]);
        st = state_of(ud, "blk.10.ffn_down_exps.weight", GGML_TYPE_NVFP4, N_FF_EXP, N_EMBD, N_EXPERT);
        CHECK(st.axis == (A) 0);
        for (int j = 0; j < 4; j++) CHECK(st.ne[j] == ud.p.ffn[10][(size_t) j]);
        st = state_of(ud, "blk.10.ffn_up_shexp.weight", GGML_TYPE_Q8_0, N_EMBD, N_FF_EXP);
        CHECK(st.axis == (A) 1);
        for (int j = 0; j < 4; j++) CHECK(st.ne[j] == ud.p.shexp[10][(size_t) j]);
        st = state_of(ud, "blk.10.ffn_down_shexp.weight", GGML_TYPE_Q8_0, N_FF_EXP, N_EMBD);
        CHECK(st.axis == (A) 0);
        // Dense FFN (layer 1) and the head.
        st = state_of(ud, "blk.1.ffn_up.weight", GGML_TYPE_Q8_0, N_EMBD, N_FF);
        CHECK(st.axis == (A) 1 && st.ne[0] == ud.p.ffn[1][0]);
        st = state_of(ud, "blk.1.ffn_down.weight", GGML_TYPE_Q8_0, N_FF, N_EMBD);
        CHECK(st.axis == (A) 0 && st.ne[1] == ud.p.ffn[1][1]);
        st = state_of(ud, "output.weight", GGML_TYPE_BF16, N_EMBD, N_VOCAB);
        CHECK(st.axis == (A) 1 && st.ne[2] == ud.p.out_rows[2]);
        CHECK(!ud.unplanned.load());

        // What the plan does not know is mirrored AND recorded (the load refuses it).
        const char * unknown[] = { "blk.1.ffn_gate_exps.weight" /* dense layer */, "blk.10.ffn_gate.weight" /* MoE layer */,
                                   "blk.10.attn_gate.weight", "token_embd_extra.weight", "slot0_k.60", "blk.60.attn_q.weight" };
        for (const char * n : unknown)
        {
            tsg_mm3_tp::userdata u2;
            u2.s = ud.s;
            u2.p = ud.p;
            CHECK(state_of(u2, n, GGML_TYPE_F32, 4096, 3072).axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
            CHECK(u2.unplanned.load() && u2.unplanned_name == n);
        }
        // A split tensor with the wrong shape is recorded too.
        tsg_mm3_tp::userdata u3;
        u3.s = ud.s;
        u3.p = ud.p;
        CHECK(state_of(u3, "blk.10.ffn_down_exps.weight", GGML_TYPE_NVFP4, 2048, N_EMBD, N_EXPERT).axis ==
              GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        CHECK(u3.unplanned.load());
    }

    void test_bytes_and_upload()
    {
        tsg_mm3_tp::userdata ud;
        ud.s = minimax_m31_flash();
        std::string err;
        CHECK(tsg_mm3_tp::make_plan(ud.s, 8, &ud.p, &err));
        // Whole tensors accounted for: per-GPU slices of every tensor sum to it, mirrored ones are whole everywhere.
        for (bool fused : { false, true })
            for (const tensor_desc & t : real_tensors(fused))
            {
                const size_t whole = ggml_row_size(t.type, t.ne[0]) * (size_t) (t.ne[1] * t.ne[2] * t.ne[3]);
                size_t sum = 0, mx = 0;
                for (int j = 0; j < 8; j++)
                {
                    const size_t b = tsg_mm3_tp::device_bytes(ud, t.name.c_str(), t.type, t.ne, j);
                    sum += b;
                    mx = std::max(mx, b);
                }
                const auto st = state_of(ud, t.name.c_str(), t.type, t.ne[0], t.ne[1], t.ne[2]);
                if (st.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) CHECK(sum == 8 * whole && mx == whole);
                else CHECK(sum == whole);
            }
        CHECK(!ud.unplanned.load());

        // Uploads into a split tensor come in whole slabs along its split axis.
        ggml_init_params ip = { 16 * ggml_tensor_overhead(), nullptr, true };
        ggml_context * c = ggml_init(ip);
        ggml_tensor * gate = ggml_new_tensor_3d(c, GGML_TYPE_NVFP4, N_EMBD, N_FF_EXP, N_EXPERT);
        ggml_set_name(gate, "blk.7.ffn_gate_exps.weight");
        ggml_tensor * down = ggml_new_tensor_3d(c, GGML_TYPE_NVFP4, N_FF_EXP, N_EMBD, N_EXPERT);
        ggml_set_name(down, "blk.7.ffn_down_exps.weight");
        ggml_tensor * head = ggml_new_tensor_2d(c, GGML_TYPE_BF16, N_EMBD, N_VOCAB);
        ggml_set_name(head, "output.weight");
        ggml_tensor * wq = ggml_new_tensor_2d(c, GGML_TYPE_Q8_0, N_EMBD, 8192 + 1024);
        ggml_set_name(wq, "blk.7.attn_qkv.weight");
        const size_t chunk = (size_t) 64 << 20;
        CHECK(tsg_mm3_tp::upload_step(gate, ud, chunk) == chunk / gate->nb[2] * gate->nb[2]);   // whole expert matrices
        CHECK(tsg_mm3_tp::upload_step(down, ud, chunk) == chunk / down->nb[1] * down->nb[1]);   // whole rows
        CHECK(tsg_mm3_tp::upload_step(head, ud, chunk) == ggml_nbytes(head));                   // one slab: the whole head
        CHECK(tsg_mm3_tp::upload_step(wq, ud, chunk) == chunk);                                 // mirrored
        ggml_free(c);
        CHECK(!ud.unplanned.load());
    }

    /// The per-GPU budget the executor checks, with its default reserve
    /// (2048 + ubatch/4 MiB, the 256 MiB sparse-chunk budget and the meta
    /// backend's reduction scratch) at ubatch 2048 on 44.5 GiB-free A40s. The
    /// weights include the F16 decode copies of the indexer projections, which
    /// every GPU holds whole.
    tsg_mm3_tp::budget real_budget(int n, size_t free_gib_x10)
    {
        tsg_mm3_tp::userdata ud;
        ud.s = minimax_m31_flash();
        std::string err;
        tsg_mm3_tp::budget b;
        if (!tsg_mm3_tp::make_plan(ud.s, n, &ud.p, &err)) return b;
        b.n = n;
        for (const tensor_desc & t : real_tensors(true))
            for (int j = 0; j < n; j++) b.weights[j] += tsg_mm3_tp::device_bytes(ud, t.name.c_str(), t.type, t.ne, j) + 4096;
        b.reserve = (size_t) (2048 + 2048 / 4) * 1024 * 1024 + ((size_t) 256 << 20) +
                    tsg_mm3_tp::reduction_scratch_bytes(n, N_EMBD, N_USED, 2048);
        b.kv_per_token = (size_t) N_LAYER * (512 + 512 + 128) * 2;   // K, V and index keys, F16, every layer
        for (int j = 0; j < n; j++) b.free[j] = free_gib_x10 * ((size_t) 1 << 30) / 10;
        return b;
    }

    /// The meta backend sizes its all-reduce scratch by the graph's largest
    /// PARTIAL node and keeps ceil(log2 n) buffers of it on every GPU
    /// (ggml-backend-meta.cpp graph_compute). MiniMax-M3's largest is the routed
    /// experts' down projection, F32 [n_embd, n_expert_used, T]: 192 MiB at a
    /// 2048-row ubatch, three of them at --tp 8 (not the [n_embd, T] of a dense
    /// partial sum: 48 MiB).
    void test_reduction_scratch()
    {
        const size_t mib = (size_t) 1 << 20;
        CHECK(tsg_mm3_tp::reduction_scratch_bytes(8, N_EMBD, N_USED, 2048) == 3 * 192 * mib);
        CHECK(tsg_mm3_tp::reduction_scratch_bytes(2, N_EMBD, N_USED, 2048) == 1 * 192 * mib);
        CHECK(tsg_mm3_tp::reduction_scratch_bytes(5, N_EMBD, N_USED, 2048) == 3 * 192 * mib);   // ceil(log2 5) = 3
        CHECK(tsg_mm3_tp::reduction_scratch_bytes(16, N_EMBD, N_USED, 8192) == 4 * 768 * mib);
        CHECK(tsg_mm3_tp::reduction_scratch_bytes(8, N_EMBD, N_USED, 1) == 3 * (size_t) N_EMBD * N_USED * sizeof(float));
        CHECK(tsg_mm3_tp::reduction_scratch_bytes(8, N_EMBD, N_USED, 0) == 0);
    }

    void test_budget()
    {
        // 8 GPUs: ~35 GiB of weights each (27 GiB of experts + 7.6 GiB of mirrored attention and indexer).
        tsg_mm3_tp::budget b8 = real_budget(8, 445);
        CHECK(b8.n == 8);
        const double w8 = b8.weights[0] / 1073741824.0;
        std::printf("tp 8: %.2f GiB weights per GPU, K/V %zu B per token\n", w8, b8.kv_per_token);
        CHECK(w8 > 34.5 && w8 < 36.0);
        const int64_t ctx8 = tsg_mm3_tp::max_kv_tokens(b8, 256, 1 << 20);
        std::printf("tp 8: the largest context that fits is %lld tokens\n", (long long) ctx8);
        CHECK(ctx8 >= 40960 && ctx8 <= 57344 && ctx8 % 256 == 0);
        CHECK(tsg_mm3_tp::misfit(b8, ctx8) < 0);
        CHECK(tsg_mm3_tp::misfit(b8, ctx8 + 256) == 0);
        // GPU 0 also holds the vision encoder when asked: it is the one that misfits first.
        b8.dev0_extra = (size_t) 2 << 30;
        CHECK(tsg_mm3_tp::misfit(b8, ctx8) == 0);
        CHECK(tsg_mm3_tp::max_kv_tokens(b8, 256, 1 << 20) < ctx8);
        // One GPU with less memory bounds the context.
        tsg_mm3_tp::budget b8b = real_budget(8, 445);
        b8b.free[5] -= (size_t) 1 << 30;
        CHECK(tsg_mm3_tp::max_kv_tokens(b8b, 256, 1 << 20) < ctx8);
        CHECK(tsg_mm3_tp::misfit(b8b, ctx8) == 5);
        // 7 GPUs fit a smaller context; 6 and fewer do not fit at all.
        const int64_t ctx7 = tsg_mm3_tp::max_kv_tokens(real_budget(7, 445), 256, 1 << 20);
        std::printf("tp 7: the largest context that fits is %lld tokens\n", (long long) ctx7);
        CHECK(ctx7 > 0 && ctx7 < ctx8);
        for (int n : { 2, 4, 5, 6 }) CHECK(tsg_mm3_tp::max_kv_tokens(real_budget(n, 445), 256, 1 << 20) == 0);
        // The cap is respected.
        CHECK(tsg_mm3_tp::max_kv_tokens(real_budget(8, 445), 256, 8192) == 8192);
    }

    void test_fixture_shapes_and_refusals()
    {
        std::string err;
        tsg_mm3_tp::plan p;
        // The executor test's fixture: 4 layers, 1 dense of 512, NVFP4 experts and Q8_0 shared expert of 128.
        tsg_mm3_tp::shape fx;
        fx.n_layer = 4; fx.n_lead = 1; fx.n_ff = 512; fx.n_ff_exp = 128; fx.n_ff_shexp = 128; fx.n_vocab = 512;
        fx.ffn_down_type = GGML_TYPE_Q8_0; fx.exp_down_type = GGML_TYPE_NVFP4; fx.shexp_down_type = GGML_TYPE_Q8_0;
        CHECK(tsg_mm3_tp::make_plan(fx, 2, &p, &err));
        CHECK(p.g_exp == 64 && p.ffn[1][0] == 64 && p.ffn[1][1] == 64);   // too small for 128-element units
        CHECK(!tsg_mm3_tp::make_plan(fx, 4, &p, &err) && err.find("without FFN rows") != std::string::npos);
        // eng/minimax-m3-fixture.py: 6 layers, 2 dense of 1024, experts and shared expert of 256.
        tsg_mm3_tp::shape a5 = fx;
        a5.n_layer = 6; a5.n_lead = 2; a5.n_ff = 1024; a5.n_ff_exp = 256; a5.n_ff_shexp = 256;
        CHECK(tsg_mm3_tp::make_plan(a5, 2, &p, &err) && p.g_exp == 128);
        CHECK(tsg_mm3_tp::make_plan(a5, 4, &p, &err) && p.g_exp == 64);
        CHECK(!tsg_mm3_tp::make_plan(a5, 8, &p, &err));

        const tsg_mm3_tp::shape s = minimax_m31_flash();
        CHECK(!tsg_mm3_tp::make_plan(s, 1, &p, &err) && err.find("2 to") != std::string::npos);
        CHECK(!tsg_mm3_tp::make_plan(s, 17, &p, &err));
        tsg_mm3_tp::shape odd = s;
        odd.n_ff_exp = 3000;   // not a multiple of NVFP4's 64
        CHECK(!tsg_mm3_tp::make_plan(odd, 8, &p, &err) && err.find("ffn_down_exps") != std::string::npos);
        tsg_mm3_tp::shape no_ff = s;
        no_ff.n_ff_shexp = 0;
        CHECK(!tsg_mm3_tp::make_plan(no_ff, 8, &p, &err));
    }
}

int main()
{
    test_real_model_plan();
    test_split_states();
    test_bytes_and_upload();
    test_reduction_scratch();
    test_budget();
    test_fixture_shapes_and_refusals();
    std::printf("minimax-m3 tp split plan: %d failure(s)\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
