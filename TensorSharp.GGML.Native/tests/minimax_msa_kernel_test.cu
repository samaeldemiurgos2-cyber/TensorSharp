// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// ---------------------------------------------------------------------------
// MiniMax sparse attention kernels (ggml_ops_minimax_msa.h).
//
//   default    random data over a shape matrix: CUDA vs the CPU reference
//              (selection as sets with tie tolerance, attention given the same
//              selection) and the CPU reference vs a literal dense
//              HF-semantics reference on sampled tokens; then the
//              tsg_msa_build node on the fused CUDA backend vs the CPU backend.
//   --quick    a reduced matrix (CTest)
//   --bench    real M3.1 shapes: kernel phase timings and upstream ggml
//              flash_attn_ext dense causal attention over the same rows.
// Exit 77 when no CUDA device is present.
// ---------------------------------------------------------------------------

#include "ggml_ops_minimax_msa.h"
#include "ggml_ops_dsv4_fused.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

extern cudaEvent_t tsg_msa_test_phase_events[5];

namespace {

#define CK(x) do { const cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::exit(1); } } while (0)

constexpr uint16_t kF16NaN = 0x7E00;

int g_threads = 8;
// Benchmark index queries that share most of their direction within a
// 64-token window, so neighbouring tokens select overlapping blocks (random
// queries select independent blocks: the worst case for K/V reuse).
bool g_correlated = false;

template <class F>
void parallel_for(int n, F f)
{
    const int nth = std::max(1, std::min(n, g_threads));
    std::vector<std::thread> pool;
    for (int w = 0; w < nth; ++w)
        pool.emplace_back([&, w] { for (int i = w; i < n; i += nth) f(i); });
    for (auto & th : pool) th.join();
}

uint16_t to_f16(float x) { return ggml_fp32_to_fp16(x); }
float from_f16(uint16_t x) { return ggml_fp16_to_fp32(x); }

struct Case
{
    int nt, rows, block, topk, local = 1, n_head = 64, n_head_kv = 4, precision = 0;
    const char * mode;      // "end" | "mid" | "rand"
    uint64_t seed;
    int live = 0;           // "end" only: the last position is live - 1 (0: rows - 1)
    bool tie_blocks = false;    // every block holds the same index keys: exact score ties
};

struct Data
{
    tsg_msa_params p{};
    std::vector<float> q, iq;
    std::vector<uint16_t> k, v, idx;
    std::vector<int32_t> pos;
    int max_pos = 0;
};

Data make_data(const Case & c)
{
    Data d;
    const int D = 128, H = c.n_head_kv;
    d.p.n_tokens = c.nt;
    d.p.n_head = c.n_head;
    d.p.n_head_kv = H;
    d.p.head_dim = D;
    d.p.block = c.block;
    d.p.topk = c.topk;
    d.p.local_blocks = c.local;
    d.p.rows_stride = c.rows;
    d.p.kq_scale = 1.0f / std::sqrt(128.0f);
    d.p.index_precision = c.precision;
    std::mt19937_64 rng(c.seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    d.pos.resize(c.nt);
    if (!std::strcmp(c.mode, "end"))
        for (int t = 0; t < c.nt; ++t) d.pos[t] = (c.live > 0 ? c.live : c.rows) - c.nt + t;
    else if (!std::strcmp(c.mode, "mid"))
    {
        const int s0 = (c.rows - c.nt) / 2;
        for (int t = 0; t < c.nt; ++t) d.pos[t] = s0 + t;
    }
    else
    {
        std::uniform_int_distribution<int> ud(0, c.rows - 1);
        for (int t = 0; t < c.nt; ++t) d.pos[t] = ud(rng);
    }
    d.max_pos = *std::max_element(d.pos.begin(), d.pos.end());
    d.q.resize((size_t) c.nt * c.n_head * D);
    d.iq.resize((size_t) c.nt * H * D);
    for (auto & x : d.q) x = nd(rng);
    for (auto & x : d.iq) x = nd(rng);
    if (g_correlated)
    {
        std::vector<float> common((size_t) H * D);
        for (int t = 0; t < c.nt; ++t)
        {
            if (t % 64 == 0) for (auto & x : common) x = nd(rng);
            for (int i = 0; i < H * D; ++i) d.iq[(size_t) t * H * D + i] = common[i] + 0.35f * d.iq[(size_t) t * H * D + i];
        }
    }
    d.k.resize((size_t) c.rows * H * D);
    d.v.resize(d.k.size());
    d.idx.resize((size_t) c.rows * D);
    // Rows past the last position are stale cache: NaN must never be read.
    for (size_t i = 0; i < d.k.size(); ++i)
    {
        const bool live = (int) (i / (H * D)) <= d.max_pos;
        d.k[i] = live ? to_f16(nd(rng)) : kF16NaN;
        d.v[i] = live ? to_f16(nd(rng)) : kF16NaN;
    }
    for (size_t i = 0; i < d.idx.size(); ++i)
        d.idx[i] = (int) (i / D) <= d.max_pos ? to_f16(nd(rng)) : kF16NaN;
    if (c.tie_blocks)
        for (size_t i = (size_t) c.block * D; i < d.idx.size(); ++i)
            if ((int) (i / D) <= d.max_pos) d.idx[i] = d.idx[i % ((size_t) c.block * D)];
    return d;
}

// HF modeling_minimax_m3_vl.py (MSA forward), literally, for one token:
// index scores for every cached key, -inf for future keys, max per block,
// +inf at the local blocks, top min(topk, nb) by full sort, -inf picks
// dropped, then dense softmax over all keys with the block/causal keep mask.
void naive_token(const Data & d, int t, std::vector<int> & sel, std::vector<double> & out)
{
    const auto & p = d.p;
    const int D = 128, H = p.n_head_kv, G = p.n_head / H, B = p.block;
    const int k_len = d.max_pos + 1, nb = (k_len + B - 1) / B, pt = d.pos[t], qb = pt / B;
    const int kk = std::min(p.topk, nb);
    sel.assign((size_t) H * p.topk, -1);
    std::vector<double> blk(nb);
    std::vector<int> order(nb);
    for (int h = 0; h < H; ++h)
    {
        std::fill(blk.begin(), blk.end(), -INFINITY);
        const float * qh = d.iq.data() + ((size_t) t * H + h) * D;
        for (int j = 0; j <= std::min(pt, k_len - 1); ++j)
        {
            double s = 0.0;
            for (int x = 0; x < D; ++x) s += (double) qh[x] * from_f16(d.idx[(size_t) j * D + x]);
            s = (float) s;
            blk[j / B] = std::max(blk[j / B], s);
        }
        for (int l = 0; l < p.local_blocks; ++l) blk[std::max(0, qb - l)] = INFINITY;
        for (int b = 0; b < nb; ++b) order[b] = b;
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return blk[a] > blk[b]; });
        std::vector<int> chosen;
        for (int i = 0; i < kk; ++i)
            if (blk[order[i]] != -INFINITY) chosen.push_back(order[i]);
        std::sort(chosen.begin(), chosen.end());
        for (size_t i = 0; i < chosen.size(); ++i) sel[(size_t) h * p.topk + i] = chosen[i];
    }
    out.assign((size_t) p.n_head * D, 0.0);
    std::vector<double> w(k_len);
    for (int hq = 0; hq < p.n_head; ++hq)
    {
        const int g = hq / G;
        const int * s = sel.data() + (size_t) g * p.topk;
        const float * qr = d.q.data() + ((size_t) t * p.n_head + hq) * D;
        double mx = -INFINITY;
        for (int j = 0; j < k_len; ++j)
        {
            const bool keep = j <= pt && std::find(s, s + p.topk, j / B) != s + p.topk;
            if (!keep) { w[j] = -INFINITY; continue; }
            double dot = 0.0;
            for (int x = 0; x < D; ++x) dot += (double) qr[x] * from_f16(d.k[((size_t) j * H + g) * D + x]);
            w[j] = dot * p.kq_scale;
            mx = std::max(mx, w[j]);
        }
        double sum = 0.0;
        for (int j = 0; j < k_len; ++j)
        {
            w[j] = w[j] == -INFINITY ? 0.0 : std::exp(w[j] - mx);
            sum += w[j];
        }
        for (int j = 0; j < k_len; ++j)
        {
            if (w[j] == 0.0) continue;
            for (int x = 0; x < D; ++x) out[(size_t) hq * D + x] += w[j] / sum * from_f16(d.v[((size_t) j * H + g) * D + x]);
        }
    }
}

// Selection sets differ only at near ties: every block in the symmetric
// difference scores (reference) within tol of the lowest selected candidate.
// Returns 0 equal, 1 tie, 2 mismatch.
int compare_row(const int32_t * x, const int32_t * y, const float * sc, int K, int64_t nblk, double tol,
                double & worst)
{
    worst = 0;
    if (std::equal(x, x + K, y)) return 0;
    double thr = INFINITY;   // lowest reference score among the selected candidates
    for (int j = 0; j < K; ++j)
        if (y[j] >= 0 && y[j] < nblk && std::isfinite(sc[y[j]])) thr = std::min(thr, (double) sc[y[j]]);
    bool tie = std::isfinite(thr);
    auto check = [&](const int32_t * u, const int32_t * w)
    {
        for (int j = 0; j < K; ++j)
        {
            if (u[j] < 0 || std::find(w, w + K, u[j]) != w + K) continue;
            if (u[j] >= nblk || !std::isfinite(sc[u[j]])) { tie = false; continue; }
            worst = std::max(worst, std::fabs((double) sc[u[j]] - thr));
        }
    };
    check(x, y);
    check(y, x);
    return tie && worst <= tol ? 1 : 2;
}

struct SelStats { long rows = 0, equal = 0, ties = 0, hard = 0; double worst = 0; };

void compare_selection(const Data & d, const std::vector<int32_t> & a, const std::vector<int32_t> & ref,
                       const std::vector<float> & ref_scores, double tol, SelStats & st)
{
    const auto & p = d.p;
    const int H = p.n_head_kv, K = p.topk;
    const int64_t nblk = (p.rows_stride + p.block - 1) / p.block;
    for (int t = 0; t < p.n_tokens; ++t)
    {
        for (int h = 0; h < H; ++h)
        {
            ++st.rows;
            const int32_t * x = a.data() + ((size_t) t * H + h) * K;
            const int32_t * y = ref.data() + ((size_t) t * H + h) * K;
            double worst = 0;
            const int r = compare_row(x, y, ref_scores.data() + ((size_t) t * H + h) * nblk, K, nblk, tol, worst);
            st.worst = std::max(st.worst, worst);
            if (r == 0) { ++st.equal; continue; }
            if (r == 1) { ++st.ties; continue; }
            if (++st.hard <= 3)
            {
                std::fprintf(stderr, "    selection mismatch t=%d pos=%d h=%d (off by %.3g):\n      got ",
                             t, d.pos[t], h, worst);
                for (int j = 0; j < K; ++j) std::fprintf(stderr, "%d ", x[j]);
                std::fprintf(stderr, "\n      ref ");
                for (int j = 0; j < K; ++j) std::fprintf(stderr, "%d ", y[j]);
                std::fprintf(stderr, "\n");
            }
        }
    }
}

struct Err { double abs = 0, ref = 0; };

void accumulate_err(const float * a, const float * b, size_t n, Err & e)
{
    for (size_t i = 0; i < n; ++i)
    {
        const double diff = std::fabs((double) a[i] - b[i]);
        if (!(diff <= e.abs)) e.abs = std::isnan(diff) ? INFINITY : diff;
        e.ref = std::max(e.ref, std::fabs((double) b[i]));
    }
}

struct DeviceData
{
    float * q = nullptr, * iq = nullptr, * out = nullptr;
    void * k = nullptr, * v = nullptr, * idx = nullptr, * ws = nullptr;
    int32_t * pos = nullptr;
    size_t ws_bytes = 0;

    void upload(const Data & d)
    {
        CK(cudaMalloc(&q, d.q.size() * 4));
        CK(cudaMalloc(&iq, d.iq.size() * 4));
        CK(cudaMalloc(&out, d.q.size() * 4));
        CK(cudaMalloc(&k, d.k.size() * 2));
        CK(cudaMalloc(&v, d.v.size() * 2));
        CK(cudaMalloc(&idx, d.idx.size() * 2));
        CK(cudaMalloc(&pos, d.pos.size() * 4));
        ws_bytes = tsg_msa_workspace_bytes(&d.p);
        CK(cudaMalloc(&ws, ws_bytes));
        CK(cudaMemcpy(q, d.q.data(), d.q.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(iq, d.iq.data(), d.iq.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(k, d.k.data(), d.k.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(v, d.v.data(), d.v.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(idx, d.idx.data(), d.idx.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(pos, d.pos.data(), d.pos.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemset(out, 0xff, d.q.size() * 4));   // NaN until written
    }
    int run(const Data & d, cudaStream_t s)
    {
        return tsg_msa_attention_cuda(&d.p, q, iq, k, v, idx, pos, out, ws, ws_bytes, s);
    }
    ~DeviceData()
    {
        for (void * x : {(void *) q, (void *) iq, (void *) out, k, v, idx, (void *) pos, ws}) if (x) cudaFree(x);
    }
};

bool run_case(const Case & c, int naive_samples)
{
    const Data d = make_data(c);
    const auto & p = d.p;
    const int H = p.n_head_kv, K = p.topk, D = 128;
    const int64_t nblk = (p.rows_stride + p.block - 1) / p.block;

    DeviceData dev;
    dev.upload(d);
    const int status = dev.run(d, nullptr);
    CK(cudaDeviceSynchronize());
    if (status != 0)
    {
        std::printf("FAIL %s: tsg_msa_attention_cuda returned %d (%s)\n", c.mode, status, tsg_msa_status_string(status));
        return false;
    }
    std::vector<float> gpu_out(d.q.size());
    std::vector<int32_t> gpu_sel((size_t) p.n_tokens * H * K);
    CK(cudaMemcpy(gpu_out.data(), dev.out, gpu_out.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(gpu_sel.data(), dev.ws, gpu_sel.size() * 4, cudaMemcpyDeviceToHost));

    // CPU reference selection + scores, and attention over the GPU selection.
    std::vector<int32_t> cpu_sel(gpu_sel.size());
    std::vector<float> cpu_scores((size_t) p.n_tokens * H * nblk);
    std::vector<float> cpu_out_gsel(d.q.size()), cpu_out(d.q.size());
    const int chunk = std::max(1, p.n_tokens / (g_threads * 4));
    const int n_chunks = (p.n_tokens + chunk - 1) / chunk;
    std::atomic<int> cpu_bad{0};
    parallel_for(n_chunks, [&](int i)
    {
        const int t0 = i * chunk, t1 = std::min(p.n_tokens, t0 + chunk);
        cpu_bad += tsg_msa_select_cpu(&p, d.iq.data(), d.idx.data(), d.pos.data(), cpu_sel.data(),
                                      cpu_scores.data(), t0, t1) != 0;
        cpu_bad += tsg_msa_attend_cpu(&p, d.q.data(), d.k.data(), d.v.data(), d.pos.data(), gpu_sel.data(),
                                      cpu_out_gsel.data(), t0, t1) != 0;
        cpu_bad += tsg_msa_attend_cpu(&p, d.q.data(), d.k.data(), d.v.data(), d.pos.data(), cpu_sel.data(),
                                      cpu_out.data(), t0, t1) != 0;
    });
    if (cpu_bad) { std::printf("FAIL: CPU reference rejected the case\n"); return false; }

    SelStats sel_st;
    const double tol = c.precision ? 2e-3 : 5e-2;
    compare_selection(d, gpu_sel, cpu_sel, cpu_scores, tol, sel_st);

    Err gpu_err;
    accumulate_err(gpu_out.data(), cpu_out_gsel.data(), gpu_out.size(), gpu_err);

    // Literal HF semantics on sampled tokens vs the CPU reference.
    std::vector<int> samples;
    for (int s = 0; s < naive_samples && s < p.n_tokens; ++s)
        samples.push_back(s == 0 ? p.n_tokens - 1 : s == 1 ? 0 : (int) ((uint64_t) c.seed * 2654435761u * s % p.n_tokens));
    Err naive_err;
    SelStats naive_sel;
    std::vector<std::vector<int>> nsel(samples.size());
    std::vector<std::vector<double>> nout(samples.size());
    parallel_for((int) samples.size(), [&](int i) { naive_token(d, samples[i], nsel[i], nout[i]); });
    for (size_t i = 0; i < samples.size(); ++i)
    {
        const int t = samples[i];
        std::vector<float> o(nout[i].begin(), nout[i].end());
        for (int h = 0; h < H; ++h)
        {
            const int32_t * y = cpu_sel.data() + ((size_t) t * H + h) * K;
            double worst = 0;
            const int r = compare_row(nsel[i].data() + (size_t) h * K, y,
                                      cpu_scores.data() + ((size_t) t * H + h) * nblk, K, nblk, 1e-4, worst);
            ++naive_sel.rows;
            if (r == 0) ++naive_sel.equal; else if (r == 1) ++naive_sel.ties; else ++naive_sel.hard;
        }
        accumulate_err(cpu_out.data() + (size_t) t * p.n_head * D, o.data(), o.size(), naive_err);
    }

    const double rel = gpu_err.abs / std::max(1e-30, gpu_err.ref);
    // Exact ties must resolve to the lower block ids everywhere: no tolerance.
    const bool ties_exact = !c.tie_blocks || (sel_st.ties == 0 && naive_sel.ties == 0);
    const bool ok = sel_st.hard == 0 && gpu_err.abs <= 1.5e-2 && naive_sel.hard == 0 && naive_err.abs <= 2e-5 &&
                    ties_exact;
    std::printf("%s nt=%-4d rows=%-6d B=%-3d k=%-2d L=%d heads=%d/%d prec=%d %-4s%s | sel %ld/%ld equal, %ld ties "
                "(max dscore %.2g), %ld bad | gpu-cpu max|d| %.2e rel %.2e | naive: sel %ld/%ld (+%ld ties), max|d| %.1e\n",
                ok ? "ok  " : "FAIL", c.nt, c.rows, c.block, c.topk, c.local, c.n_head, c.n_head_kv, c.precision,
                c.mode, c.tie_blocks ? " ties" : (c.live ? " live" : ""), sel_st.equal, sel_st.rows, sel_st.ties, sel_st.worst, sel_st.hard, gpu_err.abs, rel,
                naive_sel.equal, naive_sel.rows, naive_sel.ties, naive_err.abs);
    return ok;
}

// ---------------------------------------------------------------------------
// Graph node: fused CUDA backend vs the CPU backend
// ---------------------------------------------------------------------------

bool run_graph(const Case & c)
{
    const Data d = make_data(c);
    const auto & p = d.p;
    const int D = 128, H = p.n_head_kv;
    const int rows_alloc = c.rows + 37;   // the node reads a view of the first `rows`

    auto run_on = [&](ggml_backend_t compute, ggml_backend_t alloc_backend, std::vector<float> & result)
    {
        ggml_init_params ip = { 64 * ggml_tensor_overhead() + ggml_graph_overhead() + 4096, nullptr, true };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, p.n_head, p.n_tokens);
        ggml_tensor * iq = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, H, p.n_tokens);
        ggml_tensor * kf = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, H * D, rows_alloc);
        ggml_tensor * vf = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, H * D, rows_alloc);
        ggml_tensor * xf = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, D, rows_alloc);
        ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, p.n_tokens);
        ggml_tensor * kv = ggml_view_2d(ctx, kf, H * D, c.rows, kf->nb[1], 0);
        ggml_tensor * vv = ggml_view_2d(ctx, vf, H * D, c.rows, vf->nb[1], 0);
        ggml_tensor * xv = ggml_view_2d(ctx, xf, D, c.rows, xf->nb[1], 0);
        ggml_tensor * out = tsg_msa_build(ctx, p, q, iq, kv, vv, xv, pos);
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, alloc_backend);
        ggml_backend_tensor_set(q, d.q.data(), 0, d.q.size() * 4);
        ggml_backend_tensor_set(iq, d.iq.data(), 0, d.iq.size() * 4);
        ggml_backend_tensor_set(kf, d.k.data(), 0, d.k.size() * 2);
        ggml_backend_tensor_set(vf, d.v.data(), 0, d.v.size() * 2);
        ggml_backend_tensor_set(xf, d.idx.data(), 0, d.idx.size() * 2);
        ggml_backend_tensor_set(pos, d.pos.data(), 0, d.pos.size() * 4);
        const bool ok = ggml_backend_graph_compute(compute, gf) == GGML_STATUS_SUCCESS;
        ggml_backend_synchronize(compute);
        result.resize(d.q.size());
        ggml_backend_tensor_get(out, result.data(), 0, result.size() * 4);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return ok;
    };

    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(cpu, std::min(g_threads, 16));
    ggml_backend_t cuda = ggml_backend_cuda_init(0);
    ggml_backend_t fused = tsg_dsv4_fused_backend_init(cuda);
    std::vector<float> on_cpu, on_cuda, ref(d.q.size());
    const bool ok_cpu = run_on(cpu, cpu, on_cpu);
    const bool ok_cuda = run_on(fused, cuda, on_cuda);
    ggml_backend_free(fused);
    ggml_backend_free(cuda);
    ggml_backend_free(cpu);

    // The CPU backend must equal the reference API bit for bit (same code).
    // Rows the stale tail never reaches: the tensors above hold rows_alloc
    // rows but the node only sees c.rows of them.
    tsg_msa_params pr = p;
    const bool ok_ref = tsg_msa_attention_cpu(&pr, d.q.data(), d.iq.data(), d.k.data(), d.v.data(), d.idx.data(),
                                             d.pos.data(), ref.data()) == 0;
    Err e_cpu, e_cuda;
    accumulate_err(on_cpu.data(), ref.data(), ref.size(), e_cpu);
    accumulate_err(on_cuda.data(), on_cpu.data(), ref.size(), e_cuda);
    const bool ok = ok_cpu && ok_cuda && ok_ref && e_cpu.abs == 0.0 && e_cuda.abs <= 1.5e-2;
    std::printf("%s graph nt=%-4d rows=%-6d B=%-3d k=%-2d prec=%d %-4s | cpu-backend vs reference max|d| %.1e | "
                "fused-cuda vs cpu-backend max|d| %.2e\n", ok ? "ok  " : "FAIL", c.nt, c.rows, c.block, c.topk,
                c.precision, c.mode, e_cpu.abs, e_cuda.abs);
    return ok;
}

// A host custom op whose userdata is a plain value, as the executors' host
// expert quantize op passes a ggml type: the wrapper must not take it for a
// descriptor when the scheduler asks it about the node.
void host_copy(ggml_tensor * dst, int ith, int nth, void * userdata)
{
    (void) nth;
    if (ith == 0 && (intptr_t) userdata == GGML_TYPE_Q8_0) std::memcpy(dst->data, dst->src[0]->data, ggml_nbytes(dst));
}

// The node inside a scheduled graph, as an executor runs it: caches and inputs
// in a CUDA buffer, ggml_backend_sched over {fused CUDA wrapper, CPU} with the
// CUDA buffer type, ordinary ops between two chained nodes (so the fused
// backend has runs to flush, and the allocator reuses the first node's
// workspace tail while the second runs), and a host custom op with a plain
// userdata value at the end. Both nodes must land on the wrapper, in one GPU
// split before the host one, and match the CPU reference.
bool run_graph_sched(const Case & c)
{
    const Data d = make_data(c);
    const auto & p = d.p;
    const int D = 128, H = p.n_head_kv;

    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(cpu, std::min(g_threads, 16));
    ggml_backend_t cuda = ggml_backend_cuda_init(0);
    ggml_backend_t fused = tsg_dsv4_fused_backend_init(cuda);

    ggml_init_params ip = { 64 * ggml_tensor_overhead() + ggml_graph_overhead() + 4096, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * q_in = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, p.n_head, p.n_tokens);
    ggml_tensor * iq = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, H, p.n_tokens);
    ggml_tensor * kc = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, H * D, c.rows);
    ggml_tensor * vc = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, H * D, c.rows);
    ggml_tensor * xc = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, D, c.rows);
    ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, p.n_tokens);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, cuda);
    ggml_backend_tensor_set(q_in, d.q.data(), 0, d.q.size() * 4);
    ggml_backend_tensor_set(iq, d.iq.data(), 0, d.iq.size() * 4);
    ggml_backend_tensor_set(kc, d.k.data(), 0, d.k.size() * 2);
    ggml_backend_tensor_set(vc, d.v.data(), 0, d.v.size() * 2);
    ggml_backend_tensor_set(xc, d.idx.data(), 0, d.idx.size() * 2);
    ggml_backend_tensor_set(pos, d.pos.data(), 0, d.pos.size() * 4);

    ggml_tensor * q = ggml_scale(ctx, q_in, 2.0f);
    ggml_tensor * node = tsg_msa_build(ctx, p, q, iq, kc, vc, xc, pos);
    ggml_tensor * node2 = tsg_msa_build(ctx, p, ggml_scale(ctx, node, 3.0f), iq, kc, vc, xc, pos);
    ggml_tensor * scaled = ggml_scale(ctx, node2, 0.5f);
    ggml_tensor * out = ggml_custom_4d(ctx, GGML_TYPE_F32, D, p.n_head, p.n_tokens, 1, &scaled, 1, host_copy, 1,
                                       (void *) (intptr_t) GGML_TYPE_Q8_0);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_backend_t backends[2] = { fused, cpu };
    ggml_backend_buffer_type_t bufts[2] = { ggml_backend_get_default_buffer_type(cuda),
                                            ggml_backend_get_default_buffer_type(cpu) };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, bufts, 2, GGML_DEFAULT_GRAPH_SIZE, false, true);
    bool ok = ggml_backend_sched_alloc_graph(sched, gf);
    const bool on_fused = ok && ggml_backend_sched_get_tensor_backend(sched, node) == fused &&
                          ggml_backend_sched_get_tensor_backend(sched, node2) == fused &&
                          ggml_backend_sched_get_tensor_backend(sched, out) == cpu;
    const int splits = ggml_backend_sched_get_n_splits(sched);
    ok = ok && ggml_backend_sched_graph_compute(sched, gf) == GGML_STATUS_SUCCESS;
    ggml_backend_sched_synchronize(sched);
    std::vector<float> got(d.q.size());
    ggml_backend_tensor_get(out, got.data(), 0, got.size() * 4);
    ggml_backend_sched_free(sched);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(fused);
    ggml_backend_free(cuda);
    ggml_backend_free(cpu);

    // 0.5 * attention(3 * attention(2q)).
    std::vector<float> q2(d.q), mid(d.q.size()), ref(d.q.size());
    for (auto & x : q2) x *= 2.0f;
    bool ok_ref = tsg_msa_attention_cpu(&p, q2.data(), d.iq.data(), d.k.data(), d.v.data(), d.idx.data(),
                                        d.pos.data(), mid.data()) == 0;
    for (auto & x : mid) x *= 3.0f;
    ok_ref = ok_ref && tsg_msa_attention_cpu(&p, mid.data(), d.iq.data(), d.k.data(), d.v.data(), d.idx.data(),
                                             d.pos.data(), ref.data()) == 0;
    for (auto & x : ref) x *= 0.5f;
    Err e;
    accumulate_err(got.data(), ref.data(), ref.size(), e);
    const bool pass = ok && ok_ref && on_fused && splits == 2 && e.abs <= 1.5e-2;
    std::printf("%s sched nt=%-4d rows=%-6d B=%-3d k=%-2d prec=%d %-4s | 2 nodes on %s + host op, %d split(s) | "
                "vs reference max|d| %.2e\n", pass ? "ok  " : "FAIL", c.nt, c.rows, c.block, c.topk, c.precision,
                c.mode, on_fused ? "fused CUDA" : "another backend", splits, e.abs);
    // The same data through the kernel API tells a near-tie selection flip
    // (ties, small gpu-cpu error given the GPU's selection) from a defect.
    if (!pass) run_case(c, 0);
    return pass;
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

double time_ms(cudaEvent_t a, cudaEvent_t b)
{
    float ms = 0;
    CK(cudaEventElapsedTime(&ms, a, b));
    return ms;
}

// Upstream ggml flash_attn_ext over all `rows` keys with a causal mask: the
// dense attention a non-sparse engine (and llama.cpp's MSA prefill) pays.
double bench_dense_fa(const Data & d, int iters)
{
    const auto & p = d.p;
    const int D = 128, H = p.n_head_kv, nt = p.n_tokens, n_kv = p.rows_stride;
    const int nt_pad = nt;
    ggml_backend_t cuda = ggml_backend_cuda_init(0);
    ggml_init_params ip = { 16 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, nt, p.n_head);
    ggml_tensor * k = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, D, n_kv, H);
    ggml_tensor * v = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, D, n_kv, H);
    ggml_tensor * m = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_kv, nt_pad);
    ggml_tensor * o = ggml_flash_attn_ext(ctx, q, k, v, m, p.kq_scale, 0.0f, 0.0f);
    ggml_prec_set_acc(o, GGML_PREC_F32);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, o);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, cuda);
    // Content does not change the FA cost; fill K/V and Q once, mask causal.
    std::vector<uint16_t> kh((size_t) D * n_kv * H);
    for (size_t i = 0; i < kh.size(); ++i) kh[i] = to_f16(0.01f * (float) ((i * 2654435761u) % 200) - 1.0f);
    ggml_backend_tensor_set(k, kh.data(), 0, kh.size() * 2);
    ggml_backend_tensor_set(v, kh.data(), 0, kh.size() * 2);
    std::vector<float> qh((size_t) D * nt * p.n_head, 0.5f);
    ggml_backend_tensor_set(q, qh.data(), 0, qh.size() * 4);
    std::vector<uint16_t> mh((size_t) n_kv * nt_pad);
    const uint16_t zero = to_f16(0.0f), ninf = to_f16(-INFINITY);
    for (int t = 0; t < nt_pad; ++t)
        for (int j = 0; j < n_kv; ++j) mh[(size_t) t * n_kv + j] = t < nt && j <= d.pos[t] ? zero : ninf;
    ggml_backend_tensor_set(m, mh.data(), 0, mh.size() * 2);
    for (int i = 0; i < 2; ++i) ggml_backend_graph_compute(cuda, gf);
    ggml_backend_synchronize(cuda);
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) ggml_backend_graph_compute(cuda, gf);
    ggml_backend_synchronize(cuda);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / iters;
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(cuda);
    return ms;
}

// `live` > 0: the sequence ends at row `live` of a cache of `rows` rows (an
// executor that hands the node its whole slot cache); 0: it ends at `rows`.
void bench_case(int nt, int rows, int precision, bool dense, int live = 0)
{
    Case c{nt, rows, 128, 16, 1, 64, 4, precision, "end", 7, live};
    const Data d = make_data(c);
    DeviceData dev;
    dev.upload(d);
    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    for (auto & e : tsg_msa_test_phase_events) CK(cudaEventCreate(&e));
    const int iters = nt >= 512 ? 10 : 50;
    for (int i = 0; i < 3; ++i) dev.run(d, s);
    CK(cudaStreamSynchronize(s));
    double total = 0, score = 0, select = 0, attn = 0, merge = 0;
    for (int i = 0; i < iters; ++i)
    {
        if (dev.run(d, s) != 0) { std::printf("bench launch failed\n"); std::exit(1); }
        CK(cudaStreamSynchronize(s));
        score += time_ms(tsg_msa_test_phase_events[0], tsg_msa_test_phase_events[1]);
        select += time_ms(tsg_msa_test_phase_events[0], tsg_msa_test_phase_events[2]);
        attn += time_ms(tsg_msa_test_phase_events[2], tsg_msa_test_phase_events[3]);
        merge += time_ms(tsg_msa_test_phase_events[3], tsg_msa_test_phase_events[4]);
        total += time_ms(tsg_msa_test_phase_events[0], tsg_msa_test_phase_events[4]);
    }
    for (auto & e : tsg_msa_test_phase_events) { CK(cudaEventDestroy(e)); e = nullptr; }
    // Back-to-back calls with no host synchronization, as a forward's layers
    // issue them: launch gaps overlap the previous kernels.
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    const int reps = nt >= 512 ? 10 : 240;
    CK(cudaEventRecord(e0, s));
    for (int i = 0; i < reps; ++i) dev.run(d, s);
    CK(cudaEventRecord(e1, s));
    CK(cudaStreamSynchronize(s));
    const double pipelined = time_ms(e0, e1) / reps;
    CK(cudaEventDestroy(e0));
    CK(cudaEventDestroy(e1));
    CK(cudaStreamDestroy(s));
    // Distinct blocks per (4-token attention tile, KV group): the K/V reuse
    // the union kernel can find (topk = no overlap lost, 4*topk = none shared).
    std::vector<int32_t> sel((size_t) nt * 4 * 16);
    CK(cudaMemcpy(sel.data(), dev.ws, sel.size() * 4, cudaMemcpyDeviceToHost));
    double union_sum = 0;
    int tiles = 0;
    for (int t0 = 0; t0 < nt; t0 += tsg_msa_attn_tokens)
        for (int g = 0; g < 4; ++g, ++tiles)
        {
            std::vector<int> u;
            for (int t = t0; t < std::min(nt, t0 + tsg_msa_attn_tokens); ++t)
                for (int j = 0; j < 16; ++j) if (sel[((size_t) t * 4 + g) * 16 + j] >= 0) u.push_back(sel[((size_t) t * 4 + g) * 16 + j]);
            std::sort(u.begin(), u.end());
            union_sum += std::unique(u.begin(), u.end()) - u.begin();
        }
    tsg_msa_plan plan;
    tsg_msa_make_plan(&d.p, &plan);
    // Dense causal attention work over the same rows, 4 FLOP per (head, key, dim).
    double keys = 0;
    for (int t = 0; t < nt; ++t) keys += d.pos[t] + 1;
    const double dense_tflop = 4.0 * 64 * 128 * keys / 1e12;
    double sparse_keys = 0;
    for (int t = 0; t < nt; ++t) sparse_keys += std::min(d.pos[t] + 1, 16 * 128);
    const double sparse_tflop = 4.0 * 64 * 128 * sparse_keys / 1e12;
    const double idx_tflop = 2.0 * 4 * 128 * keys / 1e12 * (precision ? 2 : 1);
    char rows_txt[32];
    if (live > 0) std::snprintf(rows_txt, sizeof(rows_txt), "%d/%d", live, rows);
    else std::snprintf(rows_txt, sizeof(rows_txt), "%d", rows);
    std::printf("bench%s nt=%-4d rows=%-6s prec=%d | total %8.3f ms = select %8.3f (score %.3f) + attn %8.3f + merge %6.3f "
                "(nsplit %d, union %.1f) | back-to-back %8.3f ms/call | sparse attn %.1f TFLOP/s, index %.1f TFLOP/s | "
                "dense would be %.2f TFLOP",
                g_correlated ? "(corr)" : "", nt, rows_txt, precision, total / iters, select / iters, score / iters,
                attn / iters, merge / iters, plan.nsplit, union_sum / tiles, pipelined,
                sparse_tflop / (attn / iters) * 1e3, idx_tflop / (select / iters) * 1e3, dense_tflop);
    if (dense)
    {
        const double fa = bench_dense_fa(d, nt >= 512 ? 3 : 20);
        std::printf(" | ggml FA dense %8.3f ms (%.1f TFLOP/s) -> %.1fx", fa, dense_tflop / fa * 1e3, fa / (total / iters));
    }
    std::printf("\n");
    std::fflush(stdout);
}

} // namespace

int main(int argc, char ** argv)
{
    bool quick = false, bench = false, no_dense = false;
    int one_nt = 0, one_rows = 0, one_prec = 0, one_live = 0;
    for (int i = 1; i < argc; ++i)
    {
        if (!std::strcmp(argv[i], "--quick")) quick = true;
        else if (!std::strcmp(argv[i], "--bench")) bench = true;
        else if (!std::strcmp(argv[i], "--no-dense")) no_dense = true;
        else if (!std::strcmp(argv[i], "--correlated")) g_correlated = true;
        else if (!std::strcmp(argv[i], "--bench-one") && i + 3 < argc)
        {
            one_nt = std::atoi(argv[++i]);
            one_rows = std::atoi(argv[++i]);
            one_prec = std::atoi(argv[++i]);
        }
        else if (!std::strcmp(argv[i], "--live") && i + 1 < argc) one_live = std::atoi(argv[++i]);
        else
        {
            std::fprintf(stderr, "usage: %s [--quick] [--bench [--no-dense] [--correlated]] "
                                 "[--bench-one NT ROWS PREC [--live LIVE_ROWS]]\n", argv[0]);
            return 2;
        }
    }
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev == 0) { std::printf("no CUDA device: skipped\n"); return 77; }
    int cc_major = 0;
    cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, 0);
    if (cc_major < tsg_msa_min_cuda_major)
    {
        // The kernels refuse such a device (status -4) rather than run sm_80 code on it.
        std::printf("CUDA device 0 is compute capability %d.x; the MSA kernels need %d.0: skipped\n", cc_major,
                    tsg_msa_min_cuda_major);
        return 77;
    }
    g_threads = (int) std::max(1u, std::min(64u, std::thread::hardware_concurrency()));
    ggml_cpu_init();

    if (one_nt > 0)
    {
        bench_case(one_nt, one_rows, one_prec, false, one_live);
        return 0;
    }
    if (bench)
    {
        for (const int prec : {0, 1})
        {
            // rows 2048: every query is in the dense regime (exact dense causal).
            for (const int rows : {2048, 4096, 32768, 131072}) bench_case(1, rows, prec, !no_dense && prec == 0);
            for (const int nt : {512, 2048})
                for (const int rows : {2048, 8192, 32768, 131072}) bench_case(nt, rows, prec, !no_dense && prec == 0);
        }
        return 0;
    }

    int failed = 0, total = 0;
    uint64_t seed = 1;
    const std::vector<int> nts = quick ? std::vector<int>{1, 7, 64} : std::vector<int>{1, 7, 64, 512};
    const std::vector<int> rowss = quick ? std::vector<int>{64, 2049, 5000} : std::vector<int>{64, 2047, 2048, 2049, 5000, 33000};
    for (const auto & cfg : {std::pair<int, int>{16, 4}, std::pair<int, int>{128, 16}})
        for (const int rows : rowss)
            for (const int nt : nts)
                for (const char * mode : {"end", "mid"})
                {
                    if (nt > rows) continue;
                    if (!std::strcmp(mode, "mid") && nt == rows) continue;
                    Case c{nt, rows, cfg.first, cfg.second, 1, 64, 4, 0, mode, seed++};
                    ++total;
                    failed += !run_case(c, rows > 10000 ? 3 : 6);
                }
    // Generality: other GQA group sizes / index-head counts, local blocks,
    // the split-F16 index precision and scattered positions.
    std::vector<Case> extra = {
        {64, 3000, 16, 4, 1, 24, 4, 0, "end", seed++},     // G = 6
        {64, 3000, 16, 4, 1, 64, 8, 0, "end", seed++},     // H = 8, G = 8
        {7, 3000, 32, 8, 1, 32, 2, 0, "end", seed++},      // H = 2, G = 16, block 32
        {64, 4000, 16, 8, 2, 64, 4, 0, "mid", seed++},     // two local blocks
        {64, 4000, 16, 8, 0, 64, 4, 0, "mid", seed++},     // no forced block
        {64, 9000, 128, 16, 1, 64, 4, 1, "end", seed++},   // split-F16 index query
        {512, 33000, 16, 4, 1, 64, 4, 1, "mid", seed++},
        {37, 6000, 16, 4, 1, 64, 4, 0, "rand", seed++},    // scattered positions
        {37, 20000, 128, 16, 1, 64, 4, 0, "rand", seed++},
        {64, 3000, 64, 64, 1, 64, 4, 0, "end", seed++},    // topk 64
        // A whole slot cache handed over: rows_stride far past the live rows.
        {1, 200000, 128, 16, 1, 64, 4, 0, "end", seed++, 9000},
        {5, 200000, 16, 4, 1, 64, 4, 0, "end", seed++, 3000},
        // Identical blocks: every candidate ties, the lowest ids must win.
        {64, 3000, 16, 4, 1, 64, 4, 0, "end", seed++, 0, true},
        {7, 9000, 128, 16, 1, 64, 4, 1, "mid", seed++, 0, true},
        {16, 3000, 16, 4, 4, 64, 4, 0, "end", seed++},     // local == topk: nothing ranked
        // More than 8192 candidates: ranked in global memory, by the wide
        // (<= 64 rows) and the narrow top-k CTAs.
        {3, 140000, 16, 4, 1, 64, 4, 0, "end", seed++},
        {17, 140000, 16, 4, 1, 64, 4, 0, "end", seed++},
    };
    for (const auto & c : extra)
    {
        if (quick && c.rows > 10000) continue;
        ++total;
        failed += !run_case(c, 4);
    }

    // The graph node on both backends.
    std::vector<Case> graph = {
        {7, 2100, 16, 4, 1, 64, 4, 1, "end", seed++},
        {64, 5000, 128, 16, 1, 64, 4, 1, "end", seed++},
        {1, 40000, 128, 16, 1, 64, 4, 1, "end", seed++},
    };
    for (const auto & c : graph)
    {
        ++total;
        failed += !run_graph(c);
    }
    // End to end against the CPU's selection, so the split-F16 index query
    // (what the executor passes): an F16 one may pick the other block of a
    // near tie, which the per-kernel cases above tolerate and this cannot.
    std::vector<Case> sched = {
        {7, 2100, 16, 4, 1, 64, 4, 1, "mid", seed++},
        {1, 40000, 128, 16, 1, 64, 4, 1, "end", seed++},
    };
    for (const auto & c : sched)
    {
        ++total;
        failed += !run_graph_sched(c);
    }
    std::printf("%s: %d of %d cases failed\n", failed ? "FAILED" : "PASSED", failed, total);
    return failed ? 1 : 0;
}
