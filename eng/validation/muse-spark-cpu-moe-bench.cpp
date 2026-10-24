// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// Microbenchmark for Muse Spark's host-resident routed experts: one layer's
// gate/up/down stacks evaluated for a decode-sized batch on the GGML CPU
// backend, exactly as the native executor's graph does (mul_mat_id x3 + SwiGLU),
// with fresh random expert selections every repetition. It isolates CPU expert
// throughput from the rest of the model so thread counts, memory placement and
// the repacked layout can be compared in seconds instead of model reloads.
//
// Build against an unchanged ggml as linked into libGgmlOps.so:
//   g++ -O2 -std=c++17 eng/validation/muse-spark-cpu-moe-bench.cpp \
//     -I ExternalProjects/ggml/include -L TensorSharp.GGML.Native/build -lGgmlOps \
//     -Wl,-rpath,$PWD/TensorSharp.GGML.Native/build -o /tmp/moe-bench
// Run:
//   /tmp/moe-bench <shard.gguf> <layer> <mode: mmap|anon|repack|numa> <threads,...> [tokens=1] [reps=40]
// BENCH_SHARE=k gives every token the same first k experts (correlated routing);
// BENCH_PIN=1 pins the workers in the non-numa modes too.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#include <thread>
#include <pthread.h>
#include <sched.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

// CPUs of NUMA node n from sysfs ("0-23,48-71").
static std::vector<int> node_cpus(int node)
{
    std::vector<int> out;
    char path[128];
    snprintf(path, sizeof(path), "/sys/devices/system/node/node%d/cpulist", node);
    FILE * f = fopen(path, "r");
    if (!f) return out;
    char buf[1024] = { 0 };
    if (fgets(buf, sizeof(buf), f))
        for (char * p = buf; *p && *p != '\n';)
        {
            int a = (int) strtol(p, &p, 10), b = a;
            if (*p == '-') b = (int) strtol(p + 1, &p, 10);
            for (int c = a; c <= b; c++) out.push_back(c);
            if (*p == ',') p++;
        }
    fclose(f);
    return out;
}

// ---- the candidate small-batch expert kernel (mode "numa-static") ----------
// Same arithmetic as ggml's CPU mul_mat_id (activations quantized with the
// weight type's from_float, rows reduced with its vec_dot), but worker c always
// owns rows [c*dr0, (c+1)*dr0) of every expert, whatever the batch, and a
// weight row is read once for every token that selected its expert.

// args: x [K, R, T] F32 -> dst I8 rows of vec_dot_type, one per (r, t)
static void quantize_rows(ggml_tensor * dst, int ith, int nth, void * userdata)
{
    const ggml_tensor * x = dst->src[0];
    const ggml_type vdt = (ggml_type) (intptr_t) userdata;
    const auto * tr = ggml_get_type_traits_cpu(vdt);
    const size_t row = ggml_row_size(vdt, x->ne[0]);
    const int64_t rows = x->ne[1] * x->ne[2];
    for (int64_t i = ith; i < rows; i += nth)
    {
        const int64_t r = i % x->ne[1], t = i / x->ne[1];
        tr->from_float((const float *) ((const char *) x->data + r * x->nb[1] + t * x->nb[2]),
                       (char *) dst->data + (size_t) i * row, x->ne[0]);
    }
}

// args: w [K, N, E], xq (quantize_rows of x [K, R, T], R = 1 or n_used), ids [n_used, T] -> dst [N, n_used, T]
static void mmid_static(ggml_tensor * dst, int ith, int nth, void * userdata)
{
    (void) userdata;
    const ggml_tensor * w = dst->src[0];
    const ggml_tensor * xq = dst->src[1];
    const ggml_tensor * ids = dst->src[2];
    const auto * wt = ggml_get_type_traits_cpu(w->type);
    const ggml_type vdt = wt->vec_dot_type;
    const int64_t K = w->ne[0], N = w->ne[1], n_used = ids->ne[0], T = ids->ne[1];
    const size_t xrow = ggml_row_size(vdt, K);
    const int64_t rows_per_tok = (int64_t) (ggml_nbytes(xq) / xrow) / T;    // 1 (shared input) or n_used
    const int64_t dr0 = (N + nth - 1) / nth;
    const int64_t r0 = ith * dr0, r1 = std::min<int64_t>(N, r0 + dr0);
    if (r0 >= r1) return;
    // (expert, slot, token) triples grouped by expert
    struct pick { int32_t e, s, t; };
    std::vector<pick> picks;
    picks.reserve((size_t) (n_used * T));
    for (int64_t t = 0; t < T; t++)
        for (int64_t s = 0; s < n_used; s++)
            picks.push_back({ *(const int32_t *) ((const char *) ids->data + s * ids->nb[0] + t * ids->nb[1]),
                              (int32_t) s, (int32_t) t });
    std::sort(picks.begin(), picks.end(), [](const pick & a, const pick & b) { return a.e < b.e; });
    for (size_t i = 0; i < picks.size();)
    {
        size_t j = i;
        while (j < picks.size() && picks[j].e == picks[i].e) j++;
        const char * we = (const char *) w->data + (size_t) picks[i].e * w->nb[2];
        for (int64_t r = r0; r < r1; r++)
        {
            const char * wr = we + (size_t) r * w->nb[1];
            for (size_t p = i; p < j; p++)
            {
                const int64_t s = picks[p].s, t = picks[p].t;
                const char * xr = (const char *) xq->data + (size_t) (t * rows_per_tok + (rows_per_tok == 1 ? 0 : s)) * xrow;
                float * out = (float *) ((char *) dst->data + s * dst->nb[1] + t * dst->nb[2]);
                wt->vec_dot((int) K, out + r, 0, wr, 0, xr, 0, 1);
            }
        }
        i = j;
    }
}

// Worker i -> a CPU on node 0 for the first half, node 1 for the second, in
// ascending CPU order (what ggml's strict cpumask assigns).
static std::vector<int> numa_worker_cpus(int nth)
{
    std::vector<int> n0 = node_cpus(0), n1 = node_cpus(1), out;
    if (n1.empty()) { for (int i = 0; i < nth; i++) out.push_back(n0[(size_t) i % n0.size()]); return out; }
    const int h0 = (nth + 1) / 2;
    for (int i = 0; i < h0; i++) out.push_back(n0[(size_t) i % n0.size()]);
    for (int i = 0; i < nth - h0; i++) out.push_back(n1[(size_t) i % n1.size()]);
    std::sort(out.begin(), out.end());
    return out;
}

int main(int argc, char ** argv)
{
    if (argc < 5)
    {
        fprintf(stderr, "usage: %s <shard.gguf> <layer> <mmap|anon|repack> <threads,...> [tokens] [reps]\n", argv[0]);
        return 2;
    }
    const char * path = argv[1];
    const int layer = atoi(argv[2]);
    const std::string mode = argv[3];
    const bool numa_mode = mode == "numa" || mode == "numa-static";
    std::vector<int> threads;
    for (const char * p = argv[4]; *p;) { threads.push_back(atoi(p)); while (*p && *p != ',') p++; if (*p) p++; }
    const int ntok = argc > 5 ? atoi(argv[5]) : 1;
    const int reps = argc > 6 ? atoi(argv[6]) : 40;

    ggml_context * meta = nullptr;
    gguf_init_params ip = { true, &meta };
    gguf_context * g = gguf_init_from_file(path, ip);
    if (!g) { fprintf(stderr, "cannot open %s\n", path); return 1; }
    const char * names[3] = { "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps" };
    size_t off[3] = {}; ggml_tensor * mt[3] = {};
    for (int k = 0; k < 3; k++)
    {
        char nm[128];
        snprintf(nm, sizeof(nm), "blk.%d.%s.weight", layer, names[k]);
        const int64_t ti = gguf_find_tensor(g, nm);
        if (ti < 0) { fprintf(stderr, "%s not in this shard\n", nm); return 1; }
        off[k] = gguf_get_data_offset(g) + gguf_get_tensor_offset(g, ti);
        mt[k] = ggml_get_tensor(meta, nm);
    }
    const int fd = open(path, O_RDONLY);
    struct stat st; fstat(fd, &st);
    char * map = (char *) mmap(nullptr, (size_t) st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);

    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_dev_t cpu_dev = ggml_backend_get_device(cpu);
    ggml_backend_buffer_type_t repack = nullptr;
    {
        auto fn = (ggml_backend_dev_get_extra_bufts_t) ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(cpu_dev), "ggml_backend_dev_get_extra_bufts");
        if (fn) for (auto * b = fn(cpu_dev); b && *b; ++b) if (!strcmp(ggml_backend_buft_name(*b), "CPU_REPACK")) repack = *b;
    }

    ggml_init_params wp = { 8 * ggml_tensor_overhead(), nullptr, true };
    ggml_context * wctx = ggml_init(wp);
    ggml_context * rctx = ggml_init(wp);
    ggml_tensor * w[3];
    for (int k = 0; k < 3; k++)
    {
        const bool rp = mode == "repack" && repack && mt[k]->type == GGML_TYPE_Q4_K;
        w[k] = ggml_new_tensor_3d(rp ? rctx : wctx, mt[k]->type, mt[k]->ne[0], mt[k]->ne[1], mt[k]->ne[2]);
    }
    ggml_backend_buffer_t mbuf = nullptr, abuf = nullptr, rbuf = nullptr;
    if (mode == "mmap")
    {
        mbuf = ggml_backend_cpu_buffer_from_ptr(map, (size_t) st.st_size);
        ggml_backend_buffer_set_usage(mbuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        for (int k = 0; k < 3; k++) ggml_backend_tensor_alloc(mbuf, w[k], map + off[k]);
    }
    else if (numa_mode)
    {
        // First-touch NUMA placement for a pinned pool of threads[0] workers:
        // ggml's decode mul_mat_id gives worker c rows [c*dr0, (c+1)*dr0) of
        // every expert, so a helper pinned beside worker c copies exactly those.
        abuf = ggml_backend_alloc_ctx_tensors(wctx, cpu);
        const int nth = threads[0];
        std::vector<int> cpus = numa_worker_cpus(nth);
        std::vector<std::thread> helpers;
        for (int c = 0; c < nth; c++)
            helpers.emplace_back([&, c]()
            {
                cpu_set_t set; CPU_ZERO(&set); CPU_SET(cpus[(size_t) c], &set);
                pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
                for (int k = 0; k < 3; k++)
                {
                    const int64_t nr0 = w[k]->ne[1];
                    const int64_t dr0 = (nr0 + nth - 1) / nth;
                    const int64_t r0 = c * dr0, r1 = std::min<int64_t>(nr0, r0 + dr0);
                    if (r0 >= r1) continue;
                    for (int64_t e = 0; e < w[k]->ne[2]; e++)
                    {
                        const size_t o = (size_t) e * w[k]->nb[2] + (size_t) r0 * w[k]->nb[1];
                        memcpy((char *) w[k]->data + o, map + off[k] + o, (size_t) (r1 - r0) * w[k]->nb[1]);
                    }
                }
            });
        for (auto & h : helpers) h.join();
    }
    else
    {
        if (ggml_get_first_tensor(wctx)) abuf = ggml_backend_alloc_ctx_tensors(wctx, cpu);
        if (ggml_get_first_tensor(rctx)) rbuf = ggml_backend_alloc_ctx_tensors_from_buft(rctx, repack);
        for (int k = 0; k < 3; k++) ggml_backend_tensor_set(w[k], map + off[k], 0, ggml_nbytes(w[k]));
    }
    const double bytes_per_expert = (ggml_nbytes(w[0]) + ggml_nbytes(w[1]) + ggml_nbytes(w[2])) / (double) mt[0]->ne[2];
    if (mode == "mmap")
    {
        // Fault every page in first: the timed loop measures resident memory, not storage.
        volatile uint64_t sink = 0;
        for (int k = 0; k < 3; k++)
            for (size_t i = 0; i < ggml_nbytes(w[k]); i += 4096) sink += (uint8_t) map[off[k] + i];
        (void) sink;
    }

    const int n_used = 8;
    ggml_init_params cp = { 64 * ggml_tensor_overhead() + 2 * ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(cp);
    ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, mt[0]->ne[0], 1, ntok);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, ntok);
    ggml_set_input(x); ggml_set_input(ids);
    ggml_tensor * up = ggml_mul_mat_id(ctx, w[1], x, ids);
    ggml_tensor * gt = ggml_mul_mat_id(ctx, w[0], x, ids);
    ggml_tensor * h = ggml_swiglu_split(ctx, gt, up);
    ggml_tensor * dn = ggml_mul_mat_id(ctx, w[2], h, ids);
    ggml_set_output(dn);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, dn);
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
    ggml_gallocr_alloc_graph(ga, gf);

    // The same layer through the candidate kernel (inputs shared with gf).
    auto quant = [&](ggml_tensor * a, ggml_type vdt)
    {
        ggml_tensor * args[1] = { a };
        return ggml_custom_4d(ctx, GGML_TYPE_I8, (int64_t) ggml_row_size(vdt, a->ne[0]), a->ne[1], a->ne[2], 1,
                              args, 1, quantize_rows, GGML_N_TASKS_MAX, (void *) (intptr_t) vdt);
    };
    auto mmid = [&](ggml_tensor * wk, ggml_tensor * aq)
    {
        ggml_tensor * args[3] = { wk, aq, ids };
        return ggml_custom_4d(ctx, GGML_TYPE_F32, wk->ne[1], n_used, ntok, 1, args, 3, mmid_static, GGML_N_TASKS_MAX, nullptr);
    };
    ggml_tensor * xq = quant(x, ggml_get_type_traits_cpu(w[1]->type)->vec_dot_type);
    ggml_tensor * up2 = mmid(w[1], xq);
    ggml_tensor * gt2 = mmid(w[0], xq);
    ggml_tensor * h2 = ggml_swiglu_split(ctx, gt2, up2);
    ggml_tensor * dn2 = mmid(w[2], quant(h2, ggml_get_type_traits_cpu(w[2]->type)->vec_dot_type));
    ggml_set_output(dn2);
    ggml_cgraph * gf2 = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf2, dn2);
    ggml_gallocr_t ga2 = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
    ggml_gallocr_alloc_graph(ga2, gf2);
    ggml_cgraph * timed = mode == "numa-static" ? gf2 : gf;

    std::mt19937 rng(1234);
    std::vector<float> xv((size_t) mt[0]->ne[0] * ntok);
    for (auto & v : xv) v = std::uniform_real_distribution<float>(-1, 1)(rng);
    ggml_backend_tensor_set(x, xv.data(), 0, xv.size() * sizeof(float));

    for (int nt : threads)
    {
        ggml_threadpool_params tpp = ggml_threadpool_params_default(nt);
        if (numa_mode || getenv("BENCH_PIN"))
        {
            // Strict mask: worker i runs on the i-th listed CPU (ascending ids),
            // first half on node 0, second half on node 1.
            for (int c : numa_worker_cpus(nt)) tpp.cpumask[c] = true;
            tpp.strict_cpu = true;
        }
        ggml_threadpool_t tp = ggml_threadpool_new(&tpp);
        ggml_backend_cpu_set_n_threads(cpu, nt);
        ggml_backend_cpu_set_threadpool(cpu, tp);
        // BENCH_SHARE=k: every token's first k experts are the same k experts, as
        // when the tokens of a batched decode step route alike (k=0: independent).
        const int share = std::min(n_used, std::max(0, getenv("BENCH_SHARE") ? atoi(getenv("BENCH_SHARE")) : 0));
        double total = 0, distinct_total = 0;
        std::vector<int32_t> last_iv;
        for (int r = 0; r < reps + 3; r++)
        {
            std::vector<int32_t> iv((size_t) n_used * ntok);
            std::vector<int> pool((size_t) mt[0]->ne[2]);
            for (size_t i = 0; i < pool.size(); i++) pool[i] = (int) i;
            std::shuffle(pool.begin(), pool.end(), rng);
            const std::vector<int> common(pool.begin(), pool.begin() + share);
            for (int t = 0; t < ntok; t++)
            {
                std::vector<int> rest(pool.begin() + share, pool.end());
                std::shuffle(rest.begin(), rest.end(), rng);
                for (int e = 0; e < n_used; e++)
                    iv[(size_t) t * n_used + e] = e < share ? common[(size_t) e] : rest[(size_t) (e - share)];
            }
            std::vector<int32_t> uniq(iv);
            std::sort(uniq.begin(), uniq.end());
            const double distinct = (double) (std::unique(uniq.begin(), uniq.end()) - uniq.begin());
            ggml_backend_tensor_set(ids, iv.data(), 0, iv.size() * sizeof(int32_t));
            ggml_backend_tensor_set(x, xv.data(), 0, xv.size() * sizeof(float));
            last_iv = iv;
            const auto t0 = std::chrono::steady_clock::now();
            ggml_backend_graph_compute(cpu, timed);
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            if (r >= 3) { total += ms; distinct_total += distinct; }
        }
        const double ms = total / reps;
        const double gb = bytes_per_expert * (distinct_total / reps) / 1e9;
        printf("mode=%s threads=%d tokens=%d share=%d  %.3f ms/layer  %.1f distinct experts  %.1f GB/s\n",
               mode.c_str(), nt, ntok, share, ms, distinct_total / reps, gb / (ms / 1000.0));
        fflush(stdout);
        if (mode == "numa-static")
        {
            // Exactness: both graphs on the last routes must agree bit for bit.
            std::vector<float> a((size_t) ggml_nelements(dn)), b(a.size());
            for (ggml_cgraph * g2 : { gf, gf2 })
            {
                ggml_backend_tensor_set(ids, last_iv.data(), 0, last_iv.size() * sizeof(int32_t));
                ggml_backend_tensor_set(x, xv.data(), 0, xv.size() * sizeof(float));
                ggml_backend_graph_compute(cpu, g2);
                if (g2 == gf) ggml_backend_tensor_get(dn, a.data(), 0, ggml_nbytes(dn));
                else ggml_backend_tensor_get(dn2, b.data(), 0, ggml_nbytes(dn2));
            }
            double maxd = 0;
            for (size_t i = 0; i < a.size(); i++) maxd = std::max(maxd, (double) std::fabs(a[i] - b[i]));
            printf("  check vs ggml mul_mat_id: max |diff| = %g\n", maxd);
        }
        ggml_backend_cpu_set_threadpool(cpu, nullptr);
        ggml_threadpool_free(tp);
    }
    (void) mbuf; (void) abuf; (void) rbuf;
    return 0;
}
