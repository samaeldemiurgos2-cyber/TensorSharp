// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// The Muse Spark tensor-parallel split plan (ggml_ops_muse_spark_tp.cpp) on the
// real checkpoint's shapes: llama.cpp's distribution and rotation, quantization
// block alignment, the attention ratio flash attention needs, whole tensors
// accounted for, and every refusal. Pure planning: no GPU.
#include "ggml_ops_muse_spark_tp.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace
{
    int g_failures = 0;

#define CHECK(cond) do { if (!(cond)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); g_failures++; } } while (0)

    tsg_mspark_tp::shape muse_spark_816b()
    {
        tsg_mspark_tp::shape s;
        s.n_layer = 62;
        s.n_head = 128;
        s.n_head_kv = 16;
        s.head_dim = 64;
        s.n_ff_exp = 4096;
        s.n_vocab = 202048;
        s.wo_type = GGML_TYPE_Q8_0;
        s.down_type = GGML_TYPE_Q3_K;
        // Every fourth layer global, starting with three sliding ones (15 global, 47 sliding).
        for (int il = 0; il < s.n_layer; il++) s.is_swa.push_back(il % 4 == 3 ? 0 : 1);
        return s;
    }

    ggml_backend_meta_split_state state_of(tsg_mspark_tp::userdata & ud, const char * name, ggml_type type,
                                           int64_t ne0, int64_t ne1, int64_t ne2 = 1)
    {
        ggml_tensor t;
        std::memset(&t, 0, sizeof(t));
        t.type = type;
        t.ne[0] = ne0; t.ne[1] = ne1; t.ne[2] = ne2; t.ne[3] = 1;
        std::snprintf(t.name, sizeof(t.name), "%s", name);
        return tsg_mspark_tp::split_state(&t, ud);
    }

    void test_real_model_plan()
    {
        tsg_mspark_tp::userdata ud;
        ud.s = muse_spark_816b();
        std::string err;
        CHECK(tsg_mspark_tp::make_plan(ud.s, 10, &ud.p, &err));
        const auto & p = ud.p;
        // llama.cpp 7170cad: floor(16 (j + 1) / 10) groups end at device j.
        const int64_t base[10] = { 1, 2, 1, 2, 2, 1, 2, 1, 2, 2 };
        for (int j = 0; j < 10; j++) CHECK(p.groups[0][(size_t) j] == base[j]);   // layer 0: rotation 0
        // Layer 1 is the second sliding layer: rotated by one device.
        for (int j = 0; j < 10; j++) CHECK(p.groups[1][(size_t) ((j + 1) % 10)] == base[j]);
        // Layer 3 is the first global layer: rotation 0 again.
        for (int j = 0; j < 10; j++) CHECK(p.groups[3][(size_t) j] == base[j]);
        // Experts: 16 units of 256 (lcm(Q3_K block, 128)), the same pattern.
        CHECK(p.ffn_granularity == 256);
        for (int j = 0; j < 10; j++) CHECK(p.ffn[0][(size_t) j] == base[j] * 256);
        // Output rows: rotation n_layer % n = 2.
        int64_t rows = 0;
        for (int j = 0; j < 10; j++) rows += p.out_rows[(size_t) j];
        CHECK(rows == 202048);
        CHECK(p.out_rows[2] == 20204);

        // Per-layer sums and alignment, for every layer.
        for (int il = 0; il < 62; il++)
        {
            int64_t g = 0, f = 0;
            for (int j = 0; j < 10; j++)
            {
                g += p.groups[(size_t) il][(size_t) j];
                f += p.ffn[(size_t) il][(size_t) j];
                CHECK(p.groups[(size_t) il][(size_t) j] >= 1);
                CHECK(p.ffn[(size_t) il][(size_t) j] % 256 == 0);
            }
            CHECK(g == 16);
            CHECK(f == 4096);
        }

        // Tensor states, by name.
        auto q = state_of(ud, "blk.7.attn_q.weight", GGML_TYPE_Q8_0, 8192, 8192);
        auto k = state_of(ud, "blk.7.attn_k.weight", GGML_TYPE_Q8_0, 8192, 1024);
        auto o = state_of(ud, "blk.7.attn_output.weight", GGML_TYPE_Q8_0, 8192, 8192);
        auto c = state_of(ud, "slot3_k.7", GGML_TYPE_F16, 1024, 4096);
        auto up = state_of(ud, "blk.7.ffn_up_exps.weight", GGML_TYPE_IQ3_XXS, 4096, 4096, 256);
        auto dn = state_of(ud, "blk.7.ffn_down_exps.weight", GGML_TYPE_Q3_K, 4096, 4096, 256);
        auto lat = state_of(ud, "blk.7.ffn_latent_down.weight", GGML_TYPE_Q8_0, 8192, 4096);
        CHECK(q.axis == GGML_BACKEND_SPLIT_AXIS_1 && k.axis == GGML_BACKEND_SPLIT_AXIS_1);
        CHECK(o.axis == GGML_BACKEND_SPLIT_AXIS_0 && c.axis == GGML_BACKEND_SPLIT_AXIS_0);
        CHECK(up.axis == GGML_BACKEND_SPLIT_AXIS_1 && dn.axis == GGML_BACKEND_SPLIT_AXIS_0);
        CHECK(lat.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        for (int j = 0; j < 10; j++)
        {
            // Flash attention needs 8 query heads per KV head on every GPU.
            CHECK(q.ne[j] == 8 * k.ne[j]);
            CHECK(o.ne[j] == q.ne[j]);
            CHECK(c.ne[j] == k.ne[j]);
            CHECK(o.ne[j] % 32 == 0);    // Q8_0 blocks along attn_output's input
            CHECK(up.ne[j] == dn.ne[j]);
        }
        CHECK(!ud.unplanned.load());
        // A tensor the plan does not know is mirrored and recorded.
        state_of(ud, "blk.7.attn_qkvg.weight", GGML_TYPE_Q8_0, 8192, 18432);
        CHECK(ud.unplanned.load() && ud.unplanned_name == "blk.7.attn_qkvg.weight");

        // Device bytes add up to the whole tensor.
        const int64_t ne[4] = { 4096, 4096, 256, 1 };
        size_t total = 0;
        for (int j = 0; j < 10; j++) total += tsg_mspark_tp::device_bytes(ud, "blk.7.ffn_down_exps.weight", GGML_TYPE_Q3_K, ne, j);
        CHECK(total == ggml_row_size(GGML_TYPE_Q3_K, 4096) * 4096 * 256);
    }

    void test_refusals()
    {
        tsg_mspark_tp::plan p;
        std::string err;
        auto s = muse_spark_816b();
        CHECK(!tsg_mspark_tp::make_plan(s, 1, &p, &err));
        CHECK(!tsg_mspark_tp::make_plan(s, 17, &p, &err));
        // 17 GPUs would also leave one without a KV group; 16 is the most this model splits over.
        CHECK(tsg_mspark_tp::make_plan(s, 16, &p, &err));
        auto small = s;
        small.n_head_kv = 4;
        small.n_head = 32;
        CHECK(!tsg_mspark_tp::make_plan(small, 5, &p, &err));
        CHECK(err.find("KV group") != std::string::npos);
        auto odd = s;
        odd.n_ff_exp = 4000;   // not a multiple of Q3_K's 256-element block
        CHECK(!tsg_mspark_tp::make_plan(odd, 10, &p, &err));
        for (int n = 2; n <= 16; n++) CHECK(tsg_mspark_tp::make_plan(s, n, &p, &err));
    }
}

int main()
{
    test_real_model_plan();
    test_refusals();
    if (g_failures) { std::fprintf(stderr, "%d check(s) failed\n", g_failures); return 1; }
    std::printf("muse spark tensor-parallel split plan: ok\n");
    return 0;
}
