// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// MiniMax-M3 sparse attention (MSA) graphs of the native executor
// (ggml_ops_minimax_m3_exec.cpp) against the reference semantics, one layer,
// no model: the executor's own attend_dense / msa_decode / msa_prompt run
// through the TSGgml_MiniMaxM3TestAttention hook over caches this test fills.
//
// The index keys carry a per-block, per-index-head offset (dims 124..127; a
// head's index query reads only its own one), spaced 8 apart against score
// noise below 0.2, so every block choice is decided by a wide margin and no
// rounding can flip one: what differs between two paths here is the path.
//
// Reference: ref-math section 2 / cross-check section 3, written out in double
// below -- s[h,t,j] = <iq[t,h], idx[j]> for j <= pos(t); block score = max over
// the block's visible rows; the own block forced; the top min(topk, visible
// blocks) per index head h, used by KV head h (query heads h*G .. h*G+G-1);
// causal attention inside the selected blocks, scale 1/sqrt(head_dim). Built
// next to TensorSharp's MSA kernels (ggml_ops_minimax_msa_cpu.cpp), their CPU
// reference tsg_msa_attention_cpu is checked too.
//
// Checks per geometry (fixture-sized blocks: 16 rows, top-4; real: 128, top-16):
//   * one-token decode, a multi-row verification and a chunked prompt each
//     equal the reference (F16 K/V; flash attention rounds Q, and on the CPU
//     its V accumulator, to F16, hence the bound);
//   * the sparse result differs from dense attention by far more than that
//     bound (the selection is really exercised);
//   * below the dense span the sparse path equals dense attention;
//   * on the CPU a verification row is bit-identical to the one-token decode
//     of the same position (speculative decoding must not change output),
//     sparse and dense, also over more than 512 rows (where ggml-cpu would
//     split a lone [hd, 1, n_head] query's rows across threads).
//
// Usage: GgmlOpsMiniMaxM3MsaGraphTest cpu|cuda      (exit 77: backend unavailable)
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "ggml.h"

#if defined(TSG_MM3_TEST_MSA_REFERENCE)
#include "ggml_ops_minimax_msa.h"
#endif

extern "C" {
int TSGgml_IsBackendAvailable(int);
int TSGgml_MiniMaxM3TestAttention(const char *, int, const int32_t *, const float *, const float *, const uint16_t *,
                                  const uint16_t *, const uint16_t *, int64_t, int64_t, int64_t, int64_t, float *);
}

namespace
{
constexpr int N_HEAD = 64, N_HEAD_KV = 4, HEAD_DIM = 128, IDX_DIM = 128;
constexpr int PATH_DENSE = 0, PATH_DECODE = 1, PATH_PROMPT = 2;
constexpr float BLOCK_STEP = 8.0f;   // spacing of the per-block index offsets

int g_failures = 0;

struct rng
{
    uint64_t s;
    explicit rng(uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ull + 1) {}
    uint32_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return (uint32_t) (s >> 32); }
    float uniform() { return (next() + 0.5f) / 4294967296.0f; }
    float normal()
    {
        const float u1 = uniform(), u2 = uniform();
        return sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
    }
};

struct geometry
{
    const char * name;
    int block, topk, kv_pad;
    int64_t rows;
};

/// One synthetic layer: caches of `rows` rows and the queries of every row.
struct data
{
    geometry g;
    std::vector<uint16_t> k, v, idx;     // F16 [rows][N_HEAD_KV * HEAD_DIM], [rows][IDX_DIM]
    std::vector<float> q, iq;            // F32 [rows][N_HEAD][HEAD_DIM], [rows][N_HEAD_KV][IDX_DIM]
    std::vector<float> kf, vf, idxf;     // the F16 values as F32
};

static data make_data(const geometry & g, uint64_t seed)
{
    data d;
    d.g = g;
    rng r(seed);
    const int64_t R = g.rows, nblk = R / g.block, KV = (int64_t) N_HEAD_KV * HEAD_DIM;
    auto f16 = [](float x) { return (uint16_t) ggml_fp32_to_fp16(x); };
    d.k.resize((size_t) (R * KV));
    d.v.resize((size_t) (R * KV));
    for (auto & x : d.k) x = f16(r.normal());
    for (auto & x : d.v) x = f16(r.normal());
    // Each index head ranks the blocks by its own random permutation.
    std::vector<std::vector<int>> rank(N_HEAD_KV, std::vector<int>((size_t) nblk));
    for (int h = 0; h < N_HEAD_KV; h++)
    {
        for (int64_t b = 0; b < nblk; b++) rank[(size_t) h][(size_t) b] = (int) b;
        for (int64_t b = nblk - 1; b > 0; b--) std::swap(rank[(size_t) h][(size_t) b], rank[(size_t) h][r.next() % (uint32_t) (b + 1)]);
    }
    d.idx.resize((size_t) (R * IDX_DIM));
    for (int64_t j = 0; j < R; j++)
        for (int c = 0; c < IDX_DIM; c++)
        {
            const int h = c - (IDX_DIM - N_HEAD_KV);
            const float x = h >= 0 ? BLOCK_STEP * (float) rank[(size_t) h][(size_t) (j / g.block)] : 0.05f * r.normal();
            d.idx[(size_t) (j * IDX_DIM + c)] = f16(x);
        }
    d.q.resize((size_t) R * N_HEAD * HEAD_DIM);
    for (auto & x : d.q) x = r.normal();
    d.iq.resize((size_t) R * N_HEAD_KV * IDX_DIM);
    for (int64_t t = 0; t < R; t++)
        for (int h = 0; h < N_HEAD_KV; h++)
            for (int c = 0; c < IDX_DIM; c++)
            {
                const int hc = c - (IDX_DIM - N_HEAD_KV);
                d.iq[(size_t) ((t * N_HEAD_KV + h) * IDX_DIM + c)] = hc >= 0 ? (hc == h ? 1.0f : 0.0f) : 0.05f * r.normal();
            }
    auto widen = [](const std::vector<uint16_t> & a)
    {
        std::vector<float> o(a.size());
        for (size_t i = 0; i < a.size(); i++) o[i] = ggml_fp16_to_fp32((ggml_fp16_t) a[i]);
        return o;
    };
    d.kf = widen(d.k);
    d.vf = widen(d.v);
    d.idxf = widen(d.idx);
    return d;
}

/// Selected blocks of the query at position p for index head h (reference
/// semantics, double scores); `margin` gets the gap between the last block
/// taken and the best block left out (inf when every visible block is taken).
static std::vector<int64_t> select_blocks(const data & d, int64_t p, int h, double * margin)
{
    const int64_t B = d.g.block, own = p / B, nv = own + 1;
    std::vector<double> bs((size_t) nv, -INFINITY);
    const float * iq = d.iq.data() + (size_t) ((p * N_HEAD_KV + h) * IDX_DIM);
    for (int64_t j = 0; j <= p; j++)
    {
        double s = 0.0;
        for (int c = 0; c < IDX_DIM; c++) s += (double) iq[c] * d.idxf[(size_t) (j * IDX_DIM + c)];
        bs[(size_t) (j / B)] = std::max(bs[(size_t) (j / B)], s);
    }
    bs[(size_t) own] = INFINITY;   // local_blocks = 1: the query's own block
    std::vector<int64_t> order((size_t) nv);
    for (int64_t b = 0; b < nv; b++) order[(size_t) b] = b;
    std::stable_sort(order.begin(), order.end(), [&](int64_t a, int64_t b) { return bs[(size_t) a] > bs[(size_t) b]; });
    const int64_t K = std::min<int64_t>(d.g.topk, nv);
    *margin = K < nv ? bs[(size_t) order[(size_t) (K - 1)]] - bs[(size_t) order[(size_t) K]] : INFINITY;
    order.resize((size_t) K);
    std::sort(order.begin(), order.end());
    return order;
}

/// Attention of the queries at p0..p0+nt-1, double precision: sparse (the
/// selected blocks) or dense causal.
static std::vector<float> reference(const data & d, int64_t p0, int64_t nt, bool sparse, double * min_margin)
{
    const int64_t G = N_HEAD / N_HEAD_KV, B = d.g.block, KV = (int64_t) N_HEAD_KV * HEAD_DIM;
    const double scale = 1.0 / sqrt((double) HEAD_DIM);
    std::vector<float> out((size_t) nt * N_HEAD * HEAD_DIM);
    std::vector<double> w, acc(HEAD_DIM);
    for (int64_t t = 0; t < nt; t++)
    {
        const int64_t p = p0 + t;
        for (int h = 0; h < N_HEAD_KV; h++)
        {
            std::vector<int64_t> keys;
            if (sparse)
            {
                double margin = 0.0;
                for (int64_t b : select_blocks(d, p, h, &margin))
                    for (int64_t j = b * B; j < std::min((b + 1) * B, p + 1); j++) keys.push_back(j);
                if (min_margin) *min_margin = std::min(*min_margin, margin);
            }
            else
                for (int64_t j = 0; j <= p; j++) keys.push_back(j);
            w.resize(keys.size());
            for (int64_t qh = h * G; qh < (h + 1) * G; qh++)
            {
                const float * qr = d.q.data() + (size_t) ((p * N_HEAD + qh) * HEAD_DIM);
                double mx = -INFINITY, sum = 0.0;
                for (size_t i = 0; i < keys.size(); i++)
                {
                    const float * kr = d.kf.data() + (size_t) (keys[i] * KV + h * HEAD_DIM);
                    double s = 0.0;
                    for (int c = 0; c < HEAD_DIM; c++) s += (double) qr[c] * kr[c];
                    w[i] = s * scale;
                    mx = std::max(mx, w[i]);
                }
                std::fill(acc.begin(), acc.end(), 0.0);
                for (size_t i = 0; i < keys.size(); i++)
                {
                    w[i] = exp(w[i] - mx);
                    sum += w[i];
                    const float * vr = d.vf.data() + (size_t) (keys[i] * KV + h * HEAD_DIM);
                    for (int c = 0; c < HEAD_DIM; c++) acc[(size_t) c] += w[i] * vr[c];
                }
                float * o = out.data() + (size_t) ((t * N_HEAD + qh) * HEAD_DIM);
                for (int c = 0; c < HEAD_DIM; c++) o[c] = (float) (acc[(size_t) c] / sum);
            }
        }
    }
    return out;
}

static bool executor(const std::string & backend, const data & d, int path, int64_t p0, int64_t nt, int64_t chunk,
                     std::vector<float> & out)
{
    const int32_t geom[9] = { N_HEAD, N_HEAD_KV, HEAD_DIM, N_HEAD_KV, IDX_DIM, d.g.block, d.g.topk, 1, d.g.kv_pad };
    out.assign((size_t) nt * N_HEAD * HEAD_DIM, NAN);
    const size_t qo = (size_t) p0 * N_HEAD * HEAD_DIM, io = (size_t) p0 * N_HEAD_KV * IDX_DIM;
    if (!TSGgml_MiniMaxM3TestAttention(backend.c_str(), path, geom, d.q.data() + qo, d.iq.data() + io, d.k.data(), d.v.data(),
                                       d.idx.data(), d.g.rows, p0, nt, chunk, out.data()))
    {
        printf("%s: the executor's attention failed (path %d, p0 %lld, nt %lld)\n", d.g.name, path, (long long) p0, (long long) nt);
        g_failures++;
        return false;
    }
    return true;
}

static float max_diff(const std::vector<float> & a, const std::vector<float> & b)
{
    float m = 0.0f;
    for (size_t i = 0; i < a.size() && i < b.size(); i++)
    {
        const float x = fabsf(a[i] - b[i]);
        if (!(x <= m)) m = std::isnan(x) ? INFINITY : x;
    }
    return a.size() == b.size() ? m : INFINITY;
}

/// `want_at_least`: the check passes when max|a - b| >= bound instead.
static void check(const char * geo, const char * what, const std::vector<float> & a, const std::vector<float> & b,
                  float bound, bool want_at_least = false)
{
    const float m = max_diff(a, b);
    const bool ok = want_at_least ? m >= bound : m <= bound;
    printf("%-11s %-62s max|d| %-10.3g (%s %.3g) %s\n", geo, what, m, want_at_least ? ">=" : "<=", bound, ok ? "ok" : "FAIL");
    if (!ok) g_failures++;
}

static void run_geometry(const std::string & backend, const geometry & g, int64_t p_verify, int64_t p_prompt, int64_t n_prompt)
{
    const bool cpu = backend == "CPU";
    // F16 Q in flash attention (and F16 V accumulation on the CPU) against a
    // double reference over values of unit scale.
    const float tol = 1e-2f;
    const data d = make_data(g, 7 + (uint64_t) g.block);
    const int64_t span = (int64_t) g.topk * g.block;

    // A verification of 8 rows past the dense span, as speculative decoding runs it.
    {
        const int64_t nt = 8;
        double margin = INFINITY;
        const auto ref = reference(d, p_verify, nt, true, &margin);
        const auto dense = reference(d, p_verify, nt, false, nullptr);
        printf("%-11s queries %lld..%lld: block-choice margin %.2f (score noise < 0.2)\n", g.name, (long long) p_verify,
               (long long) (p_verify + nt - 1), margin);
        if (!(margin > 4.0)) { printf("%-11s synthetic block scores are not separated: FAIL\n", g.name); g_failures++; }
        check(g.name, "reference: sparse differs from dense attention", ref, dense, 5 * tol, true);
        std::vector<float> dec, ver, ver3;
        if (!executor(backend, d, PATH_DECODE, p_verify, nt, 0, dec) || !executor(backend, d, PATH_PROMPT, p_verify, nt, 0, ver)
            || !executor(backend, d, PATH_PROMPT, p_verify, nt, 3, ver3))
            return;
        check(g.name, "one-token decode == reference", dec, ref, tol);
        check(g.name, "8-row verification == reference", ver, ref, tol);
        check(g.name, "8-row verification, 3-token mask chunks == reference", ver3, ref, tol);
        // The CPU computes each query row of a flash attention on its own, over
        // the visible rows in ascending order, at any query count below its
        // 64-row tile: the decode's gathered blocks must come in that order.
        check(g.name, "verification == one-token decode", ver, dec, cpu ? 0.0f : tol);
        check(g.name, "chunked verification == verification", ver3, ver, cpu ? 0.0f : tol);
#if defined(TSG_MM3_TEST_MSA_REFERENCE)
        {
            tsg_msa_params p = {};
            p.n_tokens = (int) nt;
            p.n_head = N_HEAD;
            p.n_head_kv = N_HEAD_KV;
            p.head_dim = HEAD_DIM;
            p.block = g.block;
            p.topk = g.topk;
            p.local_blocks = 1;
            p.rows_stride = (int) g.rows;
            p.kq_scale = 1.0f / sqrtf((float) HEAD_DIM);
            std::vector<int32_t> pos((size_t) nt);
            for (int64_t t = 0; t < nt; t++) pos[(size_t) t] = (int32_t) (p_verify + t);
            std::vector<float> kern((size_t) nt * N_HEAD * HEAD_DIM);
            const int st = tsg_msa_attention_cpu(&p, d.q.data() + (size_t) p_verify * N_HEAD * HEAD_DIM,
                                                 d.iq.data() + (size_t) p_verify * N_HEAD_KV * IDX_DIM, d.k.data(), d.v.data(),
                                                 d.idx.data(), pos.data(), kern.data());
            if (st != 0) { printf("%-11s tsg_msa_attention_cpu failed (%d): FAIL\n", g.name, st); g_failures++; }
            else
            {
                check(g.name, "tsg_msa_attention_cpu == reference", kern, ref, 1e-4f);
                check(g.name, "one-token decode == tsg_msa_attention_cpu", dec, kern, tol);
                check(g.name, "8-row verification == tsg_msa_attention_cpu", ver, kern, tol);
            }
        }
#endif
    }

    // The same 8-row verification below the dense span, where auto mode runs
    // dense attention: dense and forced-sparse, per token and as one graph.
    {
        const int64_t nt = 8, p = span - 24;
        const auto dense = reference(d, p, nt, false, nullptr);
        std::vector<float> dver, ddec((size_t) nt * N_HEAD * HEAD_DIM), row, sver, sdec;
        if (!executor(backend, d, PATH_DENSE, p, nt, 0, dver) || !executor(backend, d, PATH_PROMPT, p, nt, 0, sver)
            || !executor(backend, d, PATH_DECODE, p, nt, 0, sdec))
            return;
        for (int64_t t = 0; t < nt; t++)
        {
            if (!executor(backend, d, PATH_DENSE, p + t, 1, 0, row)) return;
            std::copy(row.begin(), row.end(), ddec.begin() + (size_t) t * N_HEAD * HEAD_DIM);
        }
        char what[96];
        snprintf(what, sizeof(what), "dense 8-row verification at %lld == dense reference", (long long) p);
        check(g.name, what, dver, dense, tol);
        check(g.name, "dense verification == dense one-token decode", dver, ddec, cpu ? 0.0f : tol);
        check(g.name, "sparse verification == dense verification (below the span)", sver, dver, cpu ? 0.0f : tol);
        check(g.name, "sparse one-token decode == dense one-token decode (below the span)", sdec, ddec, cpu ? 0.0f : tol);
    }

    // A prompt chunk across the dense span (rows before it already cached):
    // one graph, 37-token mask chunks, and every position decoded on its own.
    {
        const int64_t p0 = p_prompt, nt = n_prompt;
        const auto ref = reference(d, p0, nt, true, nullptr);
        const auto dense = reference(d, p0, nt, false, nullptr);
        std::vector<float> one, ch, dec, ex_dense;
        if (!executor(backend, d, PATH_PROMPT, p0, nt, 0, one) || !executor(backend, d, PATH_PROMPT, p0, nt, 37, ch)
            || !executor(backend, d, PATH_DECODE, p0, nt, 0, dec) || !executor(backend, d, PATH_DENSE, p0, nt, 0, ex_dense))
            return;
        char what[96];
        snprintf(what, sizeof(what), "prompt %lld..%lld, one graph == reference", (long long) p0, (long long) (p0 + nt - 1));
        check(g.name, what, one, ref, tol);
        snprintf(what, sizeof(what), "prompt %lld..%lld, 37-token mask chunks == reference", (long long) p0, (long long) (p0 + nt - 1));
        check(g.name, what, ch, ref, tol);
        check(g.name, "every position decoded alone == reference", dec, ref, tol);
        check(g.name, "executor dense attention == dense reference", ex_dense, dense, tol);
        {
            // dense attention one token per graph, as the dense regime decodes
            std::vector<float> row, dd((size_t) nt * N_HEAD * HEAD_DIM);
            for (int64_t t = 0; t < nt; t++)
            {
                if (!executor(backend, d, PATH_DENSE, p0 + t, 1, 0, row)) return;
                std::copy(row.begin(), row.end(), dd.begin() + (size_t) t * N_HEAD * HEAD_DIM);
            }
            check(g.name, "executor dense, one token per graph == dense reference", dd, dense, tol);
        }
        // Below the dense span every visible block is selected.
        const size_t below = (size_t) std::max<int64_t>(0, std::min(span - p0, nt)) * N_HEAD * HEAD_DIM;
        const std::vector<float> dense_lo(dense.begin(), dense.begin() + below), one_lo(one.begin(), one.begin() + below);
        const std::vector<float> dec_lo(dec.begin(), dec.begin() + below);
        snprintf(what, sizeof(what), "sparse prompt == dense attention below position %lld", (long long) span);
        check(g.name, what, one_lo, dense_lo, tol);
        snprintf(what, sizeof(what), "sparse decode == dense attention below position %lld", (long long) span);
        check(g.name, what, dec_lo, dense_lo, tol);
        if (p0 + nt > span)
        {
            const std::vector<float> dense_hi(dense.begin() + below, dense.end()), ref_hi(ref.begin() + below, ref.end());
            check(g.name, "reference: sparse differs from dense past the span", ref_hi, dense_hi, 5 * tol, true);
        }
    }
}
} // namespace

int main(int argc, char ** argv)
{
    const std::string mode = argc > 1 ? argv[1] : "cpu";
    const bool cuda = mode == "cuda";
    if (!cuda && mode != "cpu") { fprintf(stderr, "usage: %s cpu|cuda\n", argv[0]); return 2; }
    if (!TSGgml_IsBackendAvailable(cuda ? 3 : 2)) { printf("SKIP: backend unavailable\n"); return 77; }
    const std::string backend = cuda ? "CUDA" : "CPU";
#if !defined(TSG_MM3_TEST_MSA_REFERENCE)
    printf("note: built without ggml_ops_minimax_msa_cpu.cpp; tsg_msa_attention_cpu is not compared\n");
#endif
    // The fixture's geometry (sparse from position 64) and the real model's
    // (sparse from position 2048), each with 64 query and 4 KV heads of 128.
    run_geometry(backend, geometry{ "block16", 16, 4, 256, 256 }, 150, 0, 200);
    run_geometry(backend, geometry{ "block128", 128, 16, 256, 2560 }, 2300, 1980, 120);
    printf("%s: %d failure(s)\n", mode.c_str(), g_failures);
    return g_failures == 0 ? 0 : 1;
}
