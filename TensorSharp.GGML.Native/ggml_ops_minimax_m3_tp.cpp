// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// The MiniMax-M3 tensor-parallel split plan (see ggml_ops_minimax_m3_tp.h).
// Pure: no ggml backend calls.
#include "ggml_ops_minimax_m3_tp.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <numeric>

namespace tsg_mm3_tp
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
                "attn_norm.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_qkv.weight",
                "attn_output.weight", "attn_q_norm.weight", "attn_k_norm.weight", "indexer.q_proj.weight",
                "indexer.k_proj.weight", "indexer.qk_proj.weight", "indexer.q_norm.weight", "indexer.k_norm.weight",
                // the executor's F16 decode copies of the F32 indexer projections (IDX_F16_MAX_ROWS)
                "indexer.q_proj.weight.f16", "indexer.k_proj.weight.f16", "indexer.qk_proj.f16",
                "ffn_norm.weight", "ffn_gate_inp.weight", "exp_probs_b.bias", "ffn_gate_exps.scale",
                "ffn_up_exps.scale", "ffn_down_exps.scale",
            };
            for (const char * n : names)
                if (strcmp(suffix, n) == 0) return true;
            return false;
        }

        /// One FFN family's split: gate/up rows (axis 1) and down columns
        /// (axis 0) along n_ff with the layer's shares.
        int ffn_axis(const char * suffix, const char * gate, const char * up, const char * down, int64_t n_ff,
                     const row & layer_shares, const int64_t ne[4], row & shares, bool & known)
        {
            if (strcmp(suffix, gate) == 0 || strcmp(suffix, up) == 0)
            {
                known = ne[1] == n_ff;
                shares = layer_shares;
                return 1;
            }
            if (strcmp(suffix, down) == 0)
            {
                known = ne[0] == n_ff;
                shares = layer_shares;
                return 0;
            }
            return -2;   // not this family
        }

        /// The split axis of a tensor (MIRRORED: on every device) and, for a
        /// split, each device's elements along it. `known` is false for a name
        /// the plan does not cover or a shape it does not expect.
        int classify(const shape & s, const plan & p, const char * name, const int64_t ne[4], row & shares, bool & known)
        {
            known = true;
            shares = row{};
            int il = -1, slot = -1, consumed = 0;
            if (sscanf(name, "blk.%d.%n", &il, &consumed) == 1 && consumed > 0 && il >= 0 && il < s.n_layer)
            {
                const char * suffix = name + consumed;
                const bool dense = il < s.n_lead;
                int axis = dense ? ffn_axis(suffix, "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight", s.n_ff,
                                            p.ffn[(size_t) il], ne, shares, known)
                                 : ffn_axis(suffix, "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight",
                                            s.n_ff_exp, p.ffn[(size_t) il], ne, shares, known);
                if (axis == -2 && !dense)
                    axis = ffn_axis(suffix, "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight",
                                    s.n_ff_shexp, p.shexp[(size_t) il], ne, shares, known);
                if (axis != -2) return axis;
                known = is_mirrored_layer_weight(suffix);
                return MIRRORED;
            }
            if (strcmp(name, "output.weight") == 0)
            {
                known = ne[1] == s.n_vocab;
                shares = p.out_rows;
                return 1;
            }
            if (strcmp(name, "output_norm.weight") == 0 || strcmp(name, "token_embd.weight") == 0 ||
                strcmp(name, ARANGE_ROWS) == 0 || strcmp(name, ARANGE_HEADS) == 0)
                return MIRRORED;
            // The caches of a sequence slot: mirrored, like the attention that reads them.
            consumed = 0;
            if (starts_with(name, "slot") && (sscanf(name, "slot%d_k.%d%n", &slot, &il, &consumed) == 2 ||
                                              sscanf(name, "slot%d_v.%d%n", &slot, &il, &consumed) == 2 ||
                                              sscanf(name, "slot%d_ik.%d%n", &slot, &il, &consumed) == 2) &&
                name[consumed] == '\0' && il >= 0 && il < s.n_layer)
                return MIRRORED;
            known = false;
            return MIRRORED;
        }

        /// Granularity of one FFN family: multiples of the down projection's
        /// block rounded up to 128 (llama.cpp's choice); a model too small for
        /// that uses the block. 0 with the reason in msg when n_ff is not a
        /// multiple of the block.
        int64_t granularity(const char * what, ggml_type down, int64_t n_ff, int n, char * msg, size_t msg_size)
        {
            const int64_t blck = ggml_blck_size(down);
            if (n_ff <= 0 || n_ff % blck != 0)
            {
                snprintf(msg, msg_size, "%s (%s) cannot be split: n_ff %" PRId64 " is not a multiple of its %" PRId64
                         "-element blocks", what, ggml_type_name(down), n_ff, blck);
                return 0;
            }
            int64_t g = std::lcm(blck, (int64_t) 128);
            if (n_ff / g < n) g = blck;
            return g;
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
        const bool has_moe = s.n_lead < s.n_layer;
        if (s.n_layer <= 0 || s.n_lead < 0 || s.n_lead > s.n_layer || s.n_vocab <= 0 || (s.n_lead > 0 && s.n_ff <= 0) ||
            (has_moe && (s.n_ff_exp <= 0 || s.n_ff_shexp <= 0)))
            return fail("the model's FFN shape cannot be split");

        plan p;
        p.n = n;
        if (s.n_lead > 0 && !(p.g_dense = granularity("ffn_down", s.ffn_down_type, s.n_ff, n, msg, sizeof(msg)))) return fail(msg);
        if (has_moe && !(p.g_exp = granularity("ffn_down_exps", s.exp_down_type, s.n_ff_exp, n, msg, sizeof(msg)))) return fail(msg);
        if (has_moe && !(p.g_shexp = granularity("ffn_down_shexp", s.shexp_down_type, s.n_ff_shexp, n, msg, sizeof(msg))))
            return fail(msg);
        p.ffn.resize((size_t) s.n_layer);
        p.shexp.resize((size_t) s.n_layer);
        for (int il = 0; il < s.n_layer; il++)
        {
            const int rot = il % n;
            const bool dense = il < s.n_lead;
            p.ffn[(size_t) il] = dense ? distribute(s.n_ff, p.g_dense, n, rot) : distribute(s.n_ff_exp, p.g_exp, n, rot);
            if (!dense) p.shexp[(size_t) il] = distribute(s.n_ff_shexp, p.g_shexp, n, rot);
            for (int j = 0; j < n; j++)
            {
                const bool empty = p.ffn[(size_t) il][(size_t) j] == 0 || (!dense && p.shexp[(size_t) il][(size_t) j] == 0);
                if (!empty) continue;
                const int64_t ff = dense ? s.n_ff : std::min(s.n_ff_exp, s.n_ff_shexp);
                const int64_t g = dense ? p.g_dense : std::max(p.g_exp, p.g_shexp);
                snprintf(msg, sizeof(msg), "tensor parallelism over %d GPUs leaves GPU %d without FFN rows in layer %d "
                         "(n_ff %" PRId64 " in %" PRId64 "-element units: at most %" PRId64 " GPUs)", n, j, il, ff, g, ff / g);
                return fail(msg);
            }
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

    size_t upload_step(const ggml_tensor * t, userdata & ud, size_t chunk)
    {
        const ggml_backend_meta_split_state st = split_state(t, ud);
        if ((int) st.axis < 0 || (int) st.axis >= GGML_MAX_DIMS) return chunk;
        const size_t slab = t->nb[(int) st.axis + 1];
        return std::max<size_t>(1, chunk / std::max<size_t>(1, slab)) * slab;
    }

    size_t reduction_scratch_bytes(int n, int64_t n_embd, int64_t n_expert_used, int64_t rows)
    {
        int steps = 0;
        while ((1 << steps) < n) steps++;
        return (size_t) steps * (size_t) std::max<int64_t>(0, n_embd) * (size_t) std::max<int64_t>(1, n_expert_used) *
               (size_t) std::max<int64_t>(0, rows) * sizeof(float);
    }

    size_t need_bytes(const budget & b, int j, int64_t kv_tokens)
    {
        return b.weights[j] + b.reserve + (j == 0 ? b.dev0_extra : 0) + (size_t) std::max<int64_t>(0, kv_tokens) * b.kv_per_token;
    }

    int misfit(const budget & b, int64_t kv_tokens)
    {
        for (int j = 0; j < b.n; j++)
            if (need_bytes(b, j, kv_tokens) > b.free[j]) return j;
        return -1;
    }

    int64_t max_kv_tokens(const budget & b, int64_t pad, int64_t cap)
    {
        if (b.n <= 0 || pad <= 0 || b.kv_per_token == 0) return 0;
        int64_t best = cap;
        for (int j = 0; j < b.n; j++)
        {
            const size_t fixed = need_bytes(b, j, 0);
            if (fixed >= b.free[j]) return 0;
            best = std::min<int64_t>(best, (int64_t) ((b.free[j] - fixed) / b.kv_per_token));
        }
        return best / pad * pad;
    }
}
