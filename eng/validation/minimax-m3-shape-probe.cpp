// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// How far one ggml backend computes a row differently at one row and at several rows per
// graph, for the ops the MiniMax-M3 executor's decode, verification and prefill graphs run
// (ggml_ops_minimax_m3_exec.cpp; the synthetic fixtures' shapes and types: BF16 head, Q8_0
// fused QKV and output projection, F32 router, NVFP4 routed experts, dense flash attention
// over F16 K/V with F32 accumulation as attend_dense builds it). The same 8 input columns are
// computed once as ONE graph of N columns and once as 8 graphs of one column; reported is
// max |out_N[:, j] - out_1[:, j]| over those columns, absolute and relative to the output RMS.
// An op whose kernel does not depend on the column count reports 0. N = 8 is a speculative
// verification; N = 512 a prefill ubatch (64 queries for attention, inside its KV window).
//
// It explains logit differences between a one-token decode and a multi-row graph of the same
// positions (kernel choice by batch size in unchanged upstream ggml), which the parity tool
// (minimax-m3-parity.cpp) measures end to end.
//
// Build against a TensorSharp native build (libGgmlOps exports ggml):
//   g++ -std=c++17 -O2 -I<repo>/ExternalProjects/ggml/include minimax-m3-shape-probe.cpp \
//       -L<build> -l:libGgmlOps.so -Wl,-rpath,<build> -o minimax-m3-shape-probe        (Linux)
//   clang++ -std=c++17 -O2 -I<repo>/ExternalProjects/ggml/include minimax-m3-shape-probe.cpp \
//       -L<build> -lGgmlOps -Wl,-rpath,<build> -Wl,-undefined,dynamic_lookup -o ...  (macOS, CPU)
// Usage: minimax-m3-shape-probe cuda|cpu      (CUDA device 0; exit 77 without the backend)
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

extern "C" __attribute__((weak)) ggml_backend_t ggml_backend_cuda_init(int device);

namespace
{
struct rng
{
    uint64_t s = 0x9E3779B97F4A7C15ull;
    explicit rng(uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ull + 1) {}
    uint32_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return (uint32_t) (s >> 32); }
    float uniform() { return (next() + 0.5f) / 4294967296.0f; }
    float normal() { const float u1 = uniform(), u2 = uniform(); return sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2); }
};

std::vector<float> gauss(rng & r, size_t n, float sd)
{
    std::vector<float> v(n);
    for (auto & x : v) x = sd * r.normal();
    return v;
}

ggml_backend_t g_be = nullptr;

/// Host bytes of `v` in `type` (quantized row by row).
std::vector<uint8_t> encode(ggml_type type, const std::vector<float> & v, int64_t ne0)
{
    const int64_t rows = (int64_t) v.size() / ne0;
    std::vector<uint8_t> out(ggml_row_size(type, ne0) * rows);
    if (type == GGML_TYPE_F32) memcpy(out.data(), v.data(), v.size() * sizeof(float));
    else if (type == GGML_TYPE_F16) ggml_fp32_to_fp16_row(v.data(), (ggml_fp16_t *) out.data(), (int64_t) v.size());
    else if (type == GGML_TYPE_BF16) ggml_fp32_to_bf16_row(v.data(), (ggml_bf16_t *) out.data(), (int64_t) v.size());
    else ggml_quantize_chunk(type, v.data(), out.data(), 0, rows, ne0, nullptr);
    return out;
}

/// One op instance: `build` makes the output from inputs it creates (no_alloc context,
/// filled after allocation through `fill`), for `n` columns starting at column `c0`.
struct op_case
{
    std::string name;
    std::function<ggml_tensor *(ggml_context *, int64_t n, int64_t c0, std::vector<std::function<void()>> & fill)> build;
};

std::vector<float> run(const op_case & oc, int64_t n, int64_t c0)
{
    ggml_init_params ip = { (size_t) 64 << 20, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    std::vector<std::function<void()>> fill;
    ggml_tensor * out = oc.build(ctx, n, c0, fill);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 4096, false);
    ggml_build_forward_expand(gf, out);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, g_be);
    for (auto & f : fill) f();
    if (ggml_backend_graph_compute(g_be, gf) != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute failed\n"); exit(1); }
    std::vector<float> r((size_t) ggml_nelements(out));
    ggml_backend_tensor_get(out, r.data(), 0, ggml_nbytes(out));
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return r;
}

/// out is [per_col..., cols] with the column the slowest dimension.
void probe(const op_case & oc, int64_t n_many)
{
    const int64_t ncmp = 8;
    std::vector<float> many = run(oc, n_many, 0);
    const size_t per = many.size() / (size_t) n_many;
    double ss = 0, maxd = 0;
    for (int64_t j = 0; j < ncmp; j++)
    {
        std::vector<float> one = run(oc, 1, j);
        for (size_t i = 0; i < per; i++)
        {
            const double a = many[(size_t) j * per + i], b = one[i];
            ss += a * a;
            maxd = std::max(maxd, fabs(a - b));
        }
    }
    const double rms = sqrt(ss / (double) (per * ncmp));
    printf("%-44s N=%-4lld vs 1: max|d| %-10.3g rms(out) %-9.3g rel %.3g\n", oc.name.c_str(), (long long) n_many, maxd, rms, maxd / rms);
}
}

int main(int argc, char ** argv)
{
    const std::string mode = argc > 1 ? argv[1] : "cuda";

    g_be = mode == "cpu" ? ggml_backend_cpu_init() : (ggml_backend_cuda_init ? ggml_backend_cuda_init(0) : nullptr);
    if (!g_be) { fprintf(stderr, "no %s backend\n", mode.c_str()); return 77; }
    if (mode == "cpu") ggml_backend_cpu_set_n_threads(g_be, 16);

    const int64_t E = 512, HD = 128, NH = 64, NKV = 4, VOCAB = 200064, FF_EXP = 256, NEXP = 16, NUSED = 4, KV = 256, P0 = 40;
    const int64_t COLS = 512;
    rng r(11);
    // shared host data (column-major activations: column j = token j)
    const std::vector<float> x = gauss(r, (size_t) (E * COLS), 1.0f);
    const std::vector<float> xo = gauss(r, (size_t) (NH * HD * COLS), 1.0f);
    const std::vector<float> w_head = gauss(r, (size_t) (E * VOCAB), 2.0f / sqrtf((float) E));
    const std::vector<float> w_qkv = gauss(r, (size_t) (E * (NH + 2 * NKV) * HD), 1.0f / sqrtf((float) E));
    const std::vector<float> w_o = gauss(r, (size_t) (NH * HD * E), 1.0f / sqrtf((float) (NH * HD)));
    const std::vector<float> w_r = gauss(r, (size_t) (E * NEXP), 1.0f / sqrtf((float) E));
    const std::vector<float> w_exp = gauss(r, (size_t) (E * FF_EXP * NEXP), 1.0f / sqrtf((float) E));
    std::vector<int32_t> ids((size_t) (NUSED * COLS));
    for (int64_t t = 0; t < COLS; t++)
        for (int64_t k = 0; k < NUSED; k++) ids[(size_t) (t * NUSED + k)] = (int32_t) ((t * 7 + k * 5) % NEXP);
    const std::vector<float> q = gauss(r, (size_t) (HD * NH * COLS), 1.0f);
    const std::vector<float> kc = gauss(r, (size_t) (HD * NKV * 1024), 1.0f);
    const std::vector<float> vc = gauss(r, (size_t) (HD * NKV * 1024), 1.0f);
    auto kc16 = encode(GGML_TYPE_F16, kc, HD), vc16 = encode(GGML_TYPE_F16, vc, HD);

    auto matmul = [&](const char * name, ggml_type wt, const std::vector<float> & w, int64_t k, int64_t m,
                      const std::vector<float> & act, bool prec_f32) {
        auto wenc = std::make_shared<std::vector<uint8_t>>(encode(wt, w, k));
        return op_case{ name, [=, &act](ggml_context * ctx, int64_t n, int64_t c0, std::vector<std::function<void()>> & fill) {
            ggml_tensor * W = ggml_new_tensor_2d(ctx, wt, k, m);
            ggml_tensor * X = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n);
            fill.push_back([=] { ggml_backend_tensor_set(W, wenc->data(), 0, wenc->size()); });
            fill.push_back([=, &act] { ggml_backend_tensor_set(X, act.data() + (size_t) (c0 * k), 0, (size_t) (n * k) * sizeof(float)); });
            ggml_tensor * y = ggml_mul_mat(ctx, W, X);
            if (prec_f32) ggml_prec_set_acc(y, GGML_PREC_F32);
            return y;
        } };
    };

    std::vector<op_case> cases;
    cases.push_back(matmul("output head BF16 [512 -> 200064]", GGML_TYPE_BF16, w_head, E, VOCAB, x, false));
    cases.push_back(matmul("output head BF16, GGML_PREC_F32", GGML_TYPE_BF16, w_head, E, VOCAB, x, true));
    cases.push_back(matmul("fused qkv Q8_0 [512 -> 9216]", GGML_TYPE_Q8_0, w_qkv, E, (NH + 2 * NKV) * HD, x, false));
    cases.push_back(matmul("attn out Q8_0 [8192 -> 512]", GGML_TYPE_Q8_0, w_o, NH * HD, E, xo, false));
    cases.push_back(matmul("router F32 [512 -> 16]", GGML_TYPE_F32, w_r, E, NEXP, x, true));
    {
        auto wenc = std::make_shared<std::vector<uint8_t>>(encode(GGML_TYPE_NVFP4, w_exp, E));
        cases.push_back(op_case{ "routed experts NVFP4 mul_mat_id [512 -> 256] x4", [=, &x, &ids](ggml_context * ctx, int64_t n, int64_t c0,
                                                                                     std::vector<std::function<void()>> & fill) {
            ggml_tensor * W = ggml_new_tensor_3d(ctx, GGML_TYPE_NVFP4, E, FF_EXP, NEXP);
            ggml_tensor * X = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, E, 1, n);
            ggml_tensor * I = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, NUSED, n);
            fill.push_back([=] { ggml_backend_tensor_set(W, wenc->data(), 0, wenc->size()); });
            fill.push_back([=, &x] { ggml_backend_tensor_set(X, x.data() + (size_t) (c0 * E), 0, (size_t) (n * E) * sizeof(float)); });
            fill.push_back([=, &ids] { ggml_backend_tensor_set(I, ids.data() + (size_t) (c0 * NUSED), 0, (size_t) (n * NUSED) * sizeof(int32_t)); });
            return ggml_mul_mat_id(ctx, W, X, I);
        } });
    }
    // dense causal flash attention exactly as attend_dense builds it: queries at P0 + j over
    // the first KV cache rows (n_kv = 256 below position 256), F16 K/V, F32 accumulation.
    auto fa_case = [&](const char * name, int64_t p0, int64_t n_kv) {
        return op_case{ name, [=, &q, &kc16, &vc16](ggml_context * ctx, int64_t n, int64_t c0, std::vector<std::function<void()>> & fill) {
            ggml_tensor * Q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, HD, NH, n);
            ggml_tensor * Kc = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, HD * NKV, 1024);
            ggml_tensor * Vc = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, HD * NKV, 1024);
            ggml_tensor * M = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_kv, n);
            fill.push_back([=, &q] { ggml_backend_tensor_set(Q, q.data() + (size_t) (c0 * HD * NH), 0, (size_t) (n * HD * NH) * sizeof(float)); });
            fill.push_back([=, &kc16] { ggml_backend_tensor_set(Kc, kc16.data(), 0, kc16.size()); });
            fill.push_back([=, &vc16] { ggml_backend_tensor_set(Vc, vc16.data(), 0, vc16.size()); });
            fill.push_back([=] {
                std::vector<ggml_fp16_t> m((size_t) (n_kv * n));
                const ggml_fp16_t z = ggml_fp32_to_fp16(0.0f), ninf = ggml_fp32_to_fp16(-INFINITY);
                for (int64_t t = 0; t < n; t++)
                    for (int64_t j = 0; j < n_kv; j++) m[(size_t) (t * n_kv + j)] = j <= p0 + c0 + t ? z : ninf;
                ggml_backend_tensor_set(M, m.data(), 0, m.size() * sizeof(ggml_fp16_t));
            });
            const size_t row = Kc->nb[1];
            ggml_tensor * k = ggml_view_3d(ctx, Kc, HD, NKV, n_kv, ggml_row_size(GGML_TYPE_F16, HD), row, 0);
            ggml_tensor * v = ggml_view_3d(ctx, Vc, HD, NKV, n_kv, ggml_row_size(GGML_TYPE_F16, HD), row, 0);
            ggml_tensor * fa = ggml_flash_attn_ext(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3), ggml_permute(ctx, k, 0, 2, 1, 3),
                                                   ggml_permute(ctx, v, 0, 2, 1, 3), M, 1.0f / sqrtf((float) HD), 0.0f, 0.0f);
            ggml_prec_set_acc(fa, GGML_PREC_F32);
            return fa;                                   // [HD, NH, n]
        } };
    };
    cases.push_back(fa_case("dense flash attention, n_kv 256 (pos 40..)", P0, KV));
    cases.push_back(fa_case("dense flash attention, n_kv 1024 (pos 500..)", 500, 1024));

    for (const op_case & oc : cases)
    {
        probe(oc, 8);
        // a 512-query FA would need positions past the 256-row window; 64 queries stay inside
        probe(oc, oc.name.find("flash") != std::string::npos ? 64 : COLS);
    }
    ggml_backend_free(g_be);
    return 0;
}
