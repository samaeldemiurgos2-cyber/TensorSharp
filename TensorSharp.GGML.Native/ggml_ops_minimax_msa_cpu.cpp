// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// ---------------------------------------------------------------------------
// MiniMax Sparse Attention: parameter validation, the launch plan, the CPU
// reference (double accumulation, same selection rules as the CUDA kernels)
// and the ggml graph builder. Semantics are in ggml_ops_minimax_msa.h.
// ---------------------------------------------------------------------------

#include "ggml_ops_minimax_msa.h"
#include "ggml_ops_dsv4_fused.h"

#include "ggml.h"
#include "ggml-impl.h"   // ggml_custom_op_params

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <new>
#include <vector>

namespace {

// Bounds the score buffer of one score + top-k pass; the launcher chunks the
// tokens of longer calls (1M rows at block 128 is 128 KiB per token).
constexpr size_t kScoreBudget = size_t(64) << 20;
// Split-K target for small launches: about two CTAs per SM on an 84-SM A40.
constexpr int kSplitTargetCtas = 160;

size_t align256(size_t bytes)
{
    return (bytes + 255) & ~size_t(255);
}

const float * f16_table()
{
    static const std::vector<float> table = []
    {
        std::vector<float> t(65536);
        for (int i = 0; i < 65536; ++i) t[i] = ggml_fp16_to_fp32((ggml_fp16_t) i);
        return t;
    }();
    return table.data();
}

int msa_select_range(const tsg_msa_params & p, const float * iq, const uint16_t * idx, const int64_t idx_pitch,
                     const int32_t * pos, int32_t * sel, float * scores, const int t0, const int t1)
{
    const int H = p.n_head_kv, B = p.block, K = p.topk, D = p.head_dim;
    const int64_t nblk = ((int64_t) p.rows_stride + B - 1) / B;
    const float * lut = f16_table();
    std::vector<float> key(D), sc;
    std::vector<int> order;
    for (int t = t0; t < t1; ++t)
    {
        const int pt = pos[t];
        if (pt < 0 || pt >= p.rows_stride) return -1;
        const int own = pt / B, nv = own + 1;
        const int nf = std::min(p.local_blocks, nv), ce = nv <= K ? 0 : nv - nf, r = K - nf;
        // Candidate block maxima for every index head; candidates precede the
        // forced blocks, so with local_blocks >= 1 they are fully visible.
        sc.assign((size_t) H * ce, -INFINITY);
        for (int b = 0; b < ce; ++b)
        {
            for (int j = b * B; j < std::min((b + 1) * B, pt + 1); ++j)
            {
                const uint16_t * row = idx + (int64_t) j * idx_pitch;
                for (int d = 0; d < D; ++d) key[d] = lut[row[d]];
                for (int h = 0; h < H; ++h)
                {
                    const float * qh = iq + ((int64_t) t * H + h) * D;
                    double dot = 0.0;
                    for (int d = 0; d < D; ++d) dot += (double) qh[d] * key[d];
                    float & best = sc[(size_t) h * ce + b];
                    if ((float) dot > best) best = (float) dot;
                }
            }
        }
        for (int h = 0; h < H; ++h)
        {
            const float * hs = sc.data() + (size_t) h * ce;
            if (scores)
            {
                float * srow = scores + ((int64_t) t * H + h) * nblk;
                std::fill(srow, srow + nblk, -INFINITY);
                std::copy(hs, hs + ce, srow);
            }
            int32_t * o = sel + ((int64_t) t * H + h) * K;
            if (nv <= K)
            {
                for (int j = 0; j < K; ++j) o[j] = j < nv ? j : -1;
                continue;
            }
            // Highest score first, ties to the lower block id; NaN never wins.
            order.resize(ce);
            for (int b = 0; b < ce; ++b) order[b] = b;
            std::partial_sort(order.begin(), order.begin() + r, order.end(), [&](int a, int b)
            {
                const float sa = hs[a], sb = hs[b];
                const bool na = std::isnan(sa), nb = std::isnan(sb);
                if (na || nb) return na == nb ? a < b : nb;
                return sa > sb || (sa == sb && a < b);
            });
            int n = 0;
            for (int j = 0; j < r; ++j)
                if (!std::isnan(hs[order[j]])) o[n++] = order[j];
            std::sort(o, o + n);
            for (int f = 0; f < nf; ++f) o[n++] = own - nf + 1 + f;
            for (; n < K; ++n) o[n] = -1;
        }
    }
    return 0;
}

int msa_attend_range(const tsg_msa_params & p, const float * q, const uint16_t * k, const uint16_t * v,
                     const int64_t k_pitch, const int64_t v_pitch, const int32_t * pos, const int32_t * sel,
                     float * out, const int t0, const int t1)
{
    const int H = p.n_head_kv, B = p.block, K = p.topk, D = p.head_dim, G = p.n_head / p.n_head_kv;
    const float * lut = f16_table();
    std::vector<int> keys;
    std::vector<float> kf, vf;
    std::vector<double> w, acc(D);
    for (int t = t0; t < t1; ++t)
    {
        const int pt = pos[t];
        if (pt < 0 || pt >= p.rows_stride) return -1;
        for (int g = 0; g < H; ++g)
        {
            keys.clear();
            const int32_t * s = sel + ((int64_t) t * H + g) * K;
            for (int j = 0; j < K; ++j)
            {
                if (s[j] < 0) continue;
                for (int key = s[j] * B; key < std::min((s[j] + 1) * B, pt + 1); ++key) keys.push_back(key);
            }
            const size_t n = keys.size();
            kf.resize(n * D);
            vf.resize(n * D);
            for (size_t i = 0; i < n; ++i)
            {
                const uint16_t * kr = k + (int64_t) keys[i] * k_pitch + (int64_t) g * D;
                const uint16_t * vr = v + (int64_t) keys[i] * v_pitch + (int64_t) g * D;
                for (int d = 0; d < D; ++d)
                {
                    kf[i * D + d] = lut[kr[d]];
                    vf[i * D + d] = lut[vr[d]];
                }
            }
            w.resize(n);
            for (int hh = 0; hh < G; ++hh)
            {
                const int64_t hq = (int64_t) t * p.n_head + (int64_t) g * G + hh;
                const float * qr = q + hq * D;
                float * dst = out + hq * D;
                double mx = -INFINITY;
                for (size_t i = 0; i < n; ++i)
                {
                    double dot = 0.0;
                    for (int d = 0; d < D; ++d) dot += (double) qr[d] * kf[i * D + d];
                    w[i] = dot * p.kq_scale;
                    mx = std::max(mx, w[i]);
                }
                double sum = 0.0;
                for (size_t i = 0; i < n; ++i)
                {
                    w[i] = std::exp(w[i] - mx);
                    sum += w[i];
                }
                std::fill(acc.begin(), acc.end(), 0.0);
                for (size_t i = 0; i < n; ++i)
                    for (int d = 0; d < D; ++d) acc[d] += w[i] * vf[i * D + d];
                for (int d = 0; d < D; ++d) dst[d] = sum > 0.0 ? (float) (acc[d] / sum) : 0.0f;
            }
        }
    }
    return 0;
}

// ggml_custom_op_t entry for the CPU backend: tokens split across threads.
void msa_cpu_custom(ggml_tensor * dst, int ith, int nth, void * userdata)
{
    const auto * d = static_cast<const tsg_dsv4_fused_desc *>(userdata);
    GGML_ASSERT(d && d->magic == TSG_DSV4_FUSED_MAGIC && d->kind == TSG_MSA_ATTENTION);
    tsg_msa_params p;
    tsg_msa_params_from_node(dst, &p);
    const int t0 = (int) ((int64_t) p.n_tokens * ith / nth), t1 = (int) ((int64_t) p.n_tokens * (ith + 1) / nth);
    if (t0 >= t1) return;
    const ggml_tensor * q = dst->src[0], * iq = dst->src[1], * k = dst->src[2];
    const ggml_tensor * v = dst->src[3], * idx = dst->src[4], * pos = dst->src[5];
    std::vector<int32_t> sel((size_t) p.n_tokens * p.n_head_kv * p.topk);
    const auto * positions = static_cast<const int32_t *>(pos->data);
    int status = msa_select_range(p, static_cast<const float *>(iq->data), static_cast<const uint16_t *>(idx->data),
        (int64_t) (idx->nb[1] / 2), positions, sel.data(), nullptr, t0, t1);
    if (status == 0)
        status = msa_attend_range(p, static_cast<const float *>(q->data), static_cast<const uint16_t *>(k->data),
            static_cast<const uint16_t *>(v->data), (int64_t) (k->nb[1] / 2), (int64_t) (v->nb[1] / 2), positions,
            sel.data(), static_cast<float *>(dst->data), t0, t1);
    if (status != 0) GGML_ABORT("minimax-msa: a position is outside the cache rows (%d rows)", p.rows_stride);
}

} // namespace

const char * tsg_msa_status_string(int status)
{
    switch (status)
    {
        case 0:  return "ok";
        case -1: return "invalid parameters";
        case -2: return "workspace too small";
        case -3: return "CUDA launch failed";
        case -4: return "the CUDA device is older than compute capability 8.0 (the kernels need sm_80)";
        default: return "unknown status";
    }
}

int tsg_msa_validate(const tsg_msa_params * p)
{
    if (!p) return -1;
    const bool ok = p->n_tokens >= 1 && p->head_dim == 128 && p->n_head_kv >= 1 && p->n_head_kv <= 16 &&
        p->n_head >= p->n_head_kv && p->n_head % p->n_head_kv == 0 && p->n_head / p->n_head_kv <= 16 &&
        p->block >= 16 && p->block % 16 == 0 && p->topk >= 1 && p->topk <= 64 &&
        p->local_blocks >= 0 && p->local_blocks <= p->topk && p->rows_stride >= 1 &&
        // block ids are packed with a 4-bit token index into an int
        (p->rows_stride + (int64_t) p->block - 1) / p->block < (int64_t(1) << 26) &&
        std::isfinite(p->kq_scale) && (p->index_precision == 0 || p->index_precision == 1);
    return ok ? 0 : -1;
}

int tsg_msa_make_plan(const tsg_msa_params * p, tsg_msa_plan * plan)
{
    const int status = tsg_msa_validate(p);
    if (status != 0 || !plan) return status != 0 ? status : -1;
    const int H = p->n_head_kv, nt = p->n_tokens;
    plan->score_kt = p->block % 64 == 0 ? 64 : p->block % 32 == 0 ? 32 : 16;
    plan->attn_kt = p->block % 32 == 0 ? 32 : 16;
    plan->nblk = (int) (((int64_t) p->rows_stride + p->block - 1) / p->block);

    const size_t per_token = (size_t) H * plan->nblk * sizeof(float);
    const int tpc = tsg_msa_score_warps * (16 / H);
    int chunk = (int) std::min<size_t>((size_t) nt, std::max<size_t>(1, kScoreBudget / per_token));
    if (chunk > tpc) chunk = chunk / tpc * tpc;
    plan->score_chunk = chunk;

    plan->n_tiles = (nt + tsg_msa_attn_tokens - 1) / tsg_msa_attn_tokens;
    const int ctas = plan->n_tiles * H;
    const int max_split = std::max(1, p->topk * (p->block / plan->attn_kt) / 2);
    plan->nsplit = ctas >= kSplitTargetCtas / 2 ? 1 : std::min({max_split, (kSplitTargetCtas + ctas - 1) / ctas,
                                                                tsg_msa_max_split});

    const size_t sel_bytes = align256((size_t) nt * H * p->topk * sizeof(int32_t));
    const size_t score_bytes = align256((size_t) chunk * per_token);
    const size_t part_bytes = plan->nsplit > 1
        ? align256((size_t) nt * p->n_head * plan->nsplit * (p->head_dim + 2) * sizeof(float)) : 0;
    plan->off_scores = sel_bytes;
    plan->off_partials = sel_bytes + score_bytes;
    plan->bytes = sel_bytes + score_bytes + part_bytes;
    return 0;
}

size_t tsg_msa_workspace_bytes(const tsg_msa_params * p)
{
    tsg_msa_plan plan;
    return tsg_msa_make_plan(p, &plan) == 0 ? plan.bytes : 0;
}

int tsg_msa_select_cpu(const tsg_msa_params * p, const float * iq, const uint16_t * idx, const int32_t * pos,
                       int32_t * sel, float * scores, int t0, int t1)
{
    if (tsg_msa_validate(p) != 0 || t0 < 0 || t1 > p->n_tokens || t0 > t1) return -1;
    return msa_select_range(*p, iq, idx, p->head_dim, pos, sel, scores, t0, t1);
}

int tsg_msa_attend_cpu(const tsg_msa_params * p, const float * q, const uint16_t * k, const uint16_t * v,
                       const int32_t * pos, const int32_t * sel, float * out, int t0, int t1)
{
    if (tsg_msa_validate(p) != 0 || t0 < 0 || t1 > p->n_tokens || t0 > t1) return -1;
    const int64_t pitch = (int64_t) p->n_head_kv * p->head_dim;
    return msa_attend_range(*p, q, k, v, pitch, pitch, pos, sel, out, t0, t1);
}

int tsg_msa_attention_cpu(const tsg_msa_params * p, const float * q, const float * iq, const uint16_t * k,
                          const uint16_t * v, const uint16_t * idx, const int32_t * pos, float * out)
{
    if (tsg_msa_validate(p) != 0) return -1;
    std::vector<int32_t> sel((size_t) p->n_tokens * p->n_head_kv * p->topk);
    const int status = msa_select_range(*p, iq, idx, p->head_dim, pos, sel.data(), nullptr, 0, p->n_tokens);
    if (status != 0) return status;
    const int64_t pitch = (int64_t) p->n_head_kv * p->head_dim;
    return msa_attend_range(*p, q, k, v, pitch, pitch, pos, sel.data(), out, 0, p->n_tokens);
}

void tsg_msa_params_from_node(const ggml_tensor * node, tsg_msa_params * p)
{
    // tsg_msa_build returns a view of the node's result.
    if (node->op == GGML_OP_VIEW && node->view_src) node = node->view_src;
    GGML_ASSERT(node->op == GGML_OP_CUSTOM);
    ggml_custom_op_params cp;
    memcpy(&cp, node->op_params, sizeof(cp));
    const auto * d = static_cast<const tsg_dsv4_fused_desc *>(cp.userdata);
    GGML_ASSERT(d && d->magic == TSG_DSV4_FUSED_MAGIC && d->kind == TSG_MSA_ATTENTION);
    const ggml_tensor * q = node->src[0], * iq = node->src[1], * idx = node->src[4];
    p->n_tokens = (int) q->ne[2];
    p->n_head = (int) q->ne[1];
    p->n_head_kv = (int) iq->ne[1];
    p->head_dim = (int) q->ne[0];
    p->block = d->i0;
    p->topk = d->i1;
    p->local_blocks = d->i2;
    p->rows_stride = (int) idx->ne[1];
    p->kq_scale = d->f0;
    p->index_precision = d->i3;
}

ggml_tensor * tsg_msa_build(ggml_context * ctx, const tsg_msa_params & p, ggml_tensor * q, ggml_tensor * iq,
                            ggml_tensor * k_cache, ggml_tensor * v_cache, ggml_tensor * idx_cache, ggml_tensor * pos)
{
    GGML_ASSERT(ctx && q && iq && k_cache && v_cache && idx_cache && pos);
    const int64_t D = q->ne[0], n_head = q->ne[1], nt = q->ne[2], hkv = iq->ne[1], rows = idx_cache->ne[1];
    GGML_ASSERT(q->type == GGML_TYPE_F32 && ggml_is_contiguous(q) && q->ne[3] == 1);
    GGML_ASSERT(iq->type == GGML_TYPE_F32 && ggml_is_contiguous(iq) && iq->ne[0] == D && iq->ne[2] == nt &&
                iq->ne[3] == 1);
    for (const ggml_tensor * cache : {k_cache, v_cache, idx_cache})
    {
        // Contiguous rows at any row pitch; the CUDA path copies 16-byte pieces.
        GGML_ASSERT(cache->type == GGML_TYPE_F16 && cache->nb[0] == 2 && cache->ne[1] >= rows &&
                    cache->ne[2] == 1 && cache->ne[3] == 1 && cache->nb[1] % 16 == 0);
    }
    GGML_ASSERT(k_cache->ne[0] == hkv * D && v_cache->ne[0] == hkv * D && idx_cache->ne[0] == D);
    GGML_ASSERT(pos->type == GGML_TYPE_I32 && ggml_is_contiguous(pos) && ggml_nelements(pos) == nt);
    GGML_ASSERT(rows <= INT32_MAX && nt <= INT32_MAX);

    tsg_msa_params full = p;
    full.n_tokens = (int) nt;
    full.n_head = (int) n_head;
    full.n_head_kv = (int) hkv;
    full.head_dim = (int) D;
    full.rows_stride = (int) rows;
    GGML_ASSERT(tsg_msa_validate(&full) == 0 && "tsg_msa_build: unsupported MSA shape or parameters");

    auto * desc = new (ggml_new_buffer(ctx, sizeof(tsg_dsv4_fused_desc))) tsg_dsv4_fused_desc;
    desc->kind = TSG_MSA_ATTENTION;
    desc->i0 = full.block;
    desc->i1 = full.topk;
    desc->i2 = full.local_blocks;
    desc->i3 = full.index_precision;
    desc->f0 = full.kq_scale;
#ifdef TSG_GGML_USE_CUDA
    desc->cuda_compute = tsg_msa_cuda_compute;
#endif
    // The node is the result followed by the CUDA workspace (both multiples of
    // 256 bytes), so the graph allocator owns the scratch and reuses it once
    // the result's last reader is done; the CPU backend leaves it untouched.
    const size_t out_bytes = (size_t) D * n_head * nt * sizeof(float);
    const size_t ws_bytes = tsg_msa_workspace_bytes(&full);
    GGML_ASSERT(out_bytes % 256 == 0 && ws_bytes % 256 == 0);
    ggml_tensor * args[] = {q, iq, k_cache, v_cache, idx_cache, pos};
    ggml_tensor * node = ggml_custom_4d(ctx, GGML_TYPE_F32, (int64_t) ((out_bytes + ws_bytes) / sizeof(float)), 1, 1, 1,
                                        args, 6, msa_cpu_custom, GGML_N_TASKS_MAX, desc);
    return ggml_view_3d(ctx, node, D, n_head, nt, (size_t) D * sizeof(float), (size_t) D * n_head * sizeof(float), 0);
}
