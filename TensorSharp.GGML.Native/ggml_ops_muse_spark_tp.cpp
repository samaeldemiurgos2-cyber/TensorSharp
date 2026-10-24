// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
#include "ggml_ops_muse_spark_tp.h"

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <numeric>

namespace tsg_mspark_tp
{
    namespace
    {
        constexpr int MIRRORED = -1;

        /// llama.cpp's shares of ne elements over n devices in multiples of g,
        /// device (j + rot) % n taking the j-th share; the last takes the rest.
        row distribute(int64_t ne, int64_t g, int n, int rot)
        {
            row out{};
            int64_t low = 0;
            for (int j = 0; j < n - 1; j++)
            {
                int64_t high = ne * (j + 1) / n;
                high -= high % g;
                out[(size_t) ((j + rot) % n)] = high - low;
                low = high;
            }
            out[(size_t) ((n - 1 + rot) % n)] = ne - low;
            return out;
        }

        bool starts_with(const char * s, const char * prefix) { return strncmp(s, prefix, strlen(prefix)) == 0; }

        bool is_mirrored_layer_weight(const char * suffix)
        {
            static const char * const names[] = {
                "attn_norm.weight", "attn_q_norm.weight", "attn_k_norm.weight", "attn_resid_alpha.weight",
                "attn_resid_beta.weight", "ffn_norm.weight", "post_ffw_norm.weight", "ffn_resid_alpha.weight",
                "ffn_resid_beta.weight", "ffn_gate_inp.weight", "exp_probs_b.bias", "ffn_latent_down.weight",
                "ffn_latent_up.weight", "ffn_pre_exp_norm.weight", "ffn_post_exp_norm.weight",
            };
            for (const char * n : names)
                if (strcmp(suffix, n) == 0) return true;
            return false;
        }

        /// The split axis of a tensor (MIRRORED: on every device) and, for a split,
        /// each device's elements along it. `known` is false for a name the plan
        /// does not cover or a shape it does not expect.
        int classify(const shape & s, const plan & p, const char * name, const int64_t ne[4], row & shares, bool & known)
        {
            known = true;
            shares = row{};
            const int n_gqa = s.n_head / s.n_head_kv;
            const int64_t q_rows = (int64_t) s.n_head * s.head_dim, kv_rows = (int64_t) s.n_head_kv * s.head_dim;
            auto per_group = [&](int il, int64_t width)
            {
                for (int j = 0; j < p.n; j++) shares[(size_t) j] = p.groups[(size_t) il][(size_t) j] * width;
            };
            int il = -1, slot = -1, consumed = 0;
            if (sscanf(name, "blk.%d.%n", &il, &consumed) == 1 && consumed > 0 && il >= 0 && il < s.n_layer)
            {
                const char * suffix = name + consumed;
                if (strcmp(suffix, "attn_q.weight") == 0 || strcmp(suffix, "attn_gate.weight") == 0)
                {
                    known = ne[1] == q_rows;
                    per_group(il, (int64_t) n_gqa * s.head_dim);
                    return 1;
                }
                if (strcmp(suffix, "attn_k.weight") == 0 || strcmp(suffix, "attn_v.weight") == 0)
                {
                    known = ne[1] == kv_rows;
                    per_group(il, s.head_dim);
                    return 1;
                }
                if (strcmp(suffix, "attn_output.weight") == 0)
                {
                    known = ne[0] == q_rows;
                    per_group(il, (int64_t) n_gqa * s.head_dim);
                    return 0;
                }
                if (strcmp(suffix, "ffn_gate_exps.weight") == 0 || strcmp(suffix, "ffn_up_exps.weight") == 0)
                {
                    known = ne[1] == s.n_ff_exp;
                    shares = p.ffn[(size_t) il];
                    return 1;
                }
                if (strcmp(suffix, "ffn_down_exps.weight") == 0)
                {
                    known = ne[0] == s.n_ff_exp;
                    shares = p.ffn[(size_t) il];
                    return 0;
                }
                known = is_mirrored_layer_weight(suffix);
                return MIRRORED;
            }
            if (strcmp(name, "output.weight") == 0)
            {
                known = ne[1] == s.n_vocab;
                shares = p.out_rows;
                return 1;
            }
            if (strcmp(name, "output_norm.weight") == 0 || strcmp(name, "token_embd.weight") == 0) return MIRRORED;
            // K/V of a sequence slot: [n_kv_dim, rows], split by KV group like attn_k/attn_v.
            consumed = 0;
            if ((starts_with(name, "slot") && (sscanf(name, "slot%d_k.%d%n", &slot, &il, &consumed) == 2 ||
                                               sscanf(name, "slot%d_v.%d%n", &slot, &il, &consumed) == 2)) &&
                name[consumed] == '\0' && il >= 0 && il < s.n_layer)
            {
                known = ne[0] == kv_rows;
                per_group(il, s.head_dim);
                return 0;
            }
            known = false;
            return MIRRORED;
        }
    }

    bool make_plan(const shape & s, int n, plan * out, std::string * error)
    {
        char msg[512];
        auto fail = [&](const char * text) { if (error) *error = text; return false; };
        if (n < 2 || n > MAX_DEVICES)
        {
            snprintf(msg, sizeof(msg), "tensor parallelism spans 2 to %d GPUs, not %d", MAX_DEVICES, n);
            return fail(msg);
        }
        if (s.n_layer <= 0 || s.n_head <= 0 || s.n_head_kv <= 0 || s.head_dim <= 0 || s.n_head % s.n_head_kv != 0 ||
            (int) s.is_swa.size() < s.n_layer || s.n_ff_exp <= 0 || s.n_vocab <= 0)
            return fail("the model's attention or expert shape cannot be split");

        // Attention, in KV groups. As llama.cpp, a share is a multiple of the
        // query rows of one group and of attn_output's quantization block, the
        // block doubled (to at most 128) while every GPU still gets a group.
        const int n_gqa = s.n_head / s.n_head_kv;
        const int64_t group_q = (int64_t) n_gqa * s.head_dim, q_rows = (int64_t) s.n_head * s.head_dim;
        const int64_t blck_o = ggml_blck_size(s.wo_type);
        // Shares are multiples of lcm(group, block); the last one is the rest, which
        // stays block-aligned when the whole is.
        if (q_rows % blck_o != 0)
        {
            snprintf(msg, sizeof(msg), "attn_output (%s) cannot be split by KV group: %" PRId64 " query rows per group "
                     "do not align with its %" PRId64 "-element blocks", ggml_type_name(s.wo_type), group_q, blck_o);
            return fail(msg);
        }
        int64_t blck_perf = blck_o;
        while (blck_perf < 128 && blck_perf * n < group_q) blck_perf *= 2;
        const int64_t g_q = std::lcm(group_q, blck_perf);

        // Experts, along n_ff, in multiples of ffn_down_exps' block rounded up to
        // 128 (llama.cpp's choice); a model too small for that uses the block.
        const int64_t blck_d = ggml_blck_size(s.down_type);
        if (s.n_ff_exp % blck_d != 0)
        {
            snprintf(msg, sizeof(msg), "ffn_down_exps (%s) cannot be split: n_ff %" PRId64 " is not a multiple of its "
                     "%" PRId64 "-element blocks", ggml_type_name(s.down_type), s.n_ff_exp, blck_d);
            return fail(msg);
        }
        int64_t g_ff = std::lcm(blck_d, (int64_t) 128);
        if (s.n_ff_exp / g_ff < n) g_ff = blck_d;

        plan p;
        p.n = n;
        p.ffn_granularity = g_ff;
        p.groups.resize((size_t) s.n_layer);
        p.ffn.resize((size_t) s.n_layer);
        int seen[2] = { 0, 0 };   // earlier global / sliding layers
        for (int il = 0; il < s.n_layer; il++)
        {
            const int kind = s.is_swa[(size_t) il] ? 1 : 0;
            const int rot = seen[kind]++ % n;
            const row q = distribute(q_rows, g_q, n, rot);
            const row f = distribute(s.n_ff_exp, g_ff, n, rot);
            for (int j = 0; j < n; j++)
            {
                if (q[(size_t) j] % group_q != 0 || q[(size_t) j] == 0)
                {
                    snprintf(msg, sizeof(msg), "tensor parallelism over %d GPUs leaves GPU %d without a whole KV group "
                             "in layer %d (the model has %d KV heads)", n, j, il, s.n_head_kv);
                    return fail(msg);
                }
                if (f[(size_t) j] == 0)
                {
                    snprintf(msg, sizeof(msg), "tensor parallelism over %d GPUs leaves GPU %d without expert rows "
                             "in layer %d (n_ff %" PRId64 " in %" PRId64 "-element units)", n, j, il, s.n_ff_exp, g_ff);
                    return fail(msg);
                }
                p.groups[(size_t) il][(size_t) j] = q[(size_t) j] / group_q;
            }
            p.ffn[(size_t) il] = f;
        }
        p.out_rows = distribute(s.n_vocab, 1, n, s.n_layer % n);
        *out = std::move(p);
        return true;
    }

    ggml_backend_meta_split_state split_state(const ggml_tensor * t, userdata & ud)
    {
        ggml_backend_meta_split_state st;
        memset(&st, 0, sizeof(st));
        st.nr[0] = 1;
        st.n_segments = 1;
        row shares;
        bool known = true;
        const int axis = classify(ud.s, ud.p, t->name, t->ne, shares, known);
        if (!known && !ud.unplanned.exchange(true))
        {
            std::lock_guard<std::mutex> lk(ud.mu);
            ud.unplanned_name = t->name;
        }
        if (axis == MIRRORED || !known)
        {
            st.axis = GGML_BACKEND_SPLIT_AXIS_MIRRORED;
            return st;
        }
        st.axis = (ggml_backend_meta_split_axis) axis;
        for (int j = 0; j < ud.p.n; j++) st.ne[j] = shares[(size_t) j];
        return st;
    }

    ggml_backend_meta_split_state split_state_cb(const ggml_tensor * t, void * ud)
    {
        return split_state(t, *(userdata *) ud);
    }

    size_t device_bytes(const userdata & ud, const char * name, ggml_type type, const int64_t ne[4], int j)
    {
        row shares;
        bool known = true;
        const int axis = classify(ud.s, ud.p, name, ne, shares, known);
        int64_t e[4] = { ne[0], ne[1], ne[2], ne[3] };
        if (axis != MIRRORED && known) e[axis] = shares[(size_t) j];
        return ggml_row_size(type, e[0]) * (size_t) e[1] * (size_t) e[2] * (size_t) e[3];
    }
}
