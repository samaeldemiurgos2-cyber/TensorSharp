// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// llama-bench-equivalent throughput of TensorSharp's MiniMax-M3 native executor
// (TSGgml_MiniMaxM3*, contract C1): ONE model load, then every prefill (pp<N>),
// decode-at-depth (tg<M> @ d<D>) and concurrent-decode (np<K>) cell, each with a
// warm-up run and R timed repetitions on fresh uniform-random token ids, the way
// llama-bench measures (its depth prefill is untimed; every decode step returns the
// last token's full-vocabulary logits to the host, as llama_decode does). The calls
// are the ones TensorSharp's managed model makes (Forward, ForwardBatchedDecode), so
// the numbers are the executor's; the CLI's --benchmark adds only the P/Invoke.
//
// Build (no TensorSharp header or library is linked; libGgmlOps is dlopen'ed):
//   g++ -std=c++17 -O2 minimax-m3-native-bench.cpp -ldl -lpthread -o minimax-m3-native-bench
// With per-kernel GPU timing from CUPTI activity records (--cupti; CUDA 13 CUPTI headers,
// whose current KERNEL / MEMCPY records are CUpti_ActivityKernel13 / CUpti_ActivityMemcpy7):
//   g++ -std=c++17 -O2 -DMM3_BENCH_CUPTI -I/usr/local/cuda/include minimax-m3-native-bench.cpp \
//       -ldl -lpthread -o minimax-m3-native-bench      (libcupti is dlopen'ed: MM3_CUPTI_LIB)
//
// Usage:
//   minimax-m3-native-bench LIBGGMLOPS MODEL.gguf N_GPU [options]
//     --ctx N          context (default: the largest cell + 512, rounded to 256)
//     --ubatch N       prefill ubatch handed to the executor (default 2048, the managed default)
//     --pp LIST        prefill cells, e.g. 512,2048,8192,32768
//     --tg M           decode steps per decode cell (default 128)
//     --depth LIST     decode cells' depths, e.g. 0,2048,8192,32768
//     --np LIST        concurrent-decode cells: K slots decoding together (ForwardBatchedDecode)
//     --np-depth D     prompt length of every slot in a concurrent cell (default 512)
//     --ttft LIST      first-call prefill latency (ms) of prompts of about L tokens whose length is new to
//                      the executor (L+1, L+8, ...), so the graph build is part of what is timed
//     --ttft-at LIST   D+L items: latency (ms) of an L-token prefill on top of an untimed D-token one (a chat
//                      turn whose first D tokens the prefix cache already holds), once with lengths new to the
//                      executor (L, L+1, ...) and once more with the same lengths (graphs cached): two cells
//     --reps R         timed repetitions per cell (default 3), after one untimed warm-up
//     --cupti          per-kernel GPU time of one extra, untimed decode run per cell (needs -DMM3_BENCH_CUPTI)
//     --tg-after-pp    also time M decode steps right after every timed pp<N> run (tg<M> @ d<N> without a
//                      second prefill; for long contexts)
//     --no-warmup      skip the untimed warm-up run of each pp cell
//     --seed S         token seed (default 1234)
// Every cell prints one JSON line: {"test":"pp512","reps":[...tok/s],"median":..,"mean":..,"stdev":..}.
// TS_MINIMAX_M3_* options come from the process environment, which the executor reads.
#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(MM3_BENCH_CUPTI)
#include <cupti.h>
#endif

namespace
{
    using clk = std::chrono::steady_clock;
    double seconds(clk::time_point a, clk::time_point b) { return std::chrono::duration<double>(b - a).count(); }

    std::vector<int> parse_list(const char * s)
    {
        std::vector<int> v;
        std::stringstream ss(s);
        std::string item;
        while (std::getline(ss, item, ','))
            if (!item.empty()) v.push_back(std::atoi(item.c_str()));
        return v;
    }

    struct api
    {
        using load_fn = void * (*) (const char *, int, int, int, int, int, const char *, int, const char *, int);
        using fwd_fn = int (*) (void *, const int32_t *, int, float *, int);
        using bd_fn = int (*) (void *, int, const int32_t *, const int32_t *, const int32_t *, float *);
        using h_fn = int (*) (void *);
        using hi_fn = int (*) (void *, int);
        using stats_fn = int (*) (void *, int64_t *, int);
        using free_fn = void (*) (void *);
        using err_fn = const char * (*) ();
        load_fn load = nullptr;
        fwd_fn forward = nullptr;
        bd_fn batched = nullptr;
        h_fn vocab = nullptr, reset = nullptr, n_past = nullptr, ctx = nullptr, slot_alloc = nullptr;
        hi_fn set_active = nullptr, slot_free = nullptr, slot_n_past = nullptr;
        stats_fn stats = nullptr;
        free_fn release = nullptr;
        err_fn last_error = nullptr;

        explicit api(const char * path)
        {
            void * lib = dlopen(path, RTLD_NOW | RTLD_LOCAL);
            if (!lib) throw std::runtime_error(std::string("dlopen ") + path + ": " + dlerror());
            auto sym = [&](const char * n) { return dlsym(lib, n); };
            load = (load_fn) sym("TSGgml_MiniMaxM3LoadModelTp");
            forward = (fwd_fn) sym("TSGgml_MiniMaxM3Forward");
            batched = (bd_fn) sym("TSGgml_MiniMaxM3ForwardBatchedDecode");
            vocab = (h_fn) sym("TSGgml_MiniMaxM3VocabSize");
            reset = (h_fn) sym("TSGgml_MiniMaxM3ResetChecked");
            n_past = (h_fn) sym("TSGgml_MiniMaxM3NPast");
            ctx = (h_fn) sym("TSGgml_MiniMaxM3CtxSize");
            slot_alloc = (h_fn) sym("TSGgml_MiniMaxM3SlotAlloc");
            set_active = (hi_fn) sym("TSGgml_MiniMaxM3SetActiveSlot");
            slot_free = (hi_fn) sym("TSGgml_MiniMaxM3SlotFree");
            slot_n_past = (hi_fn) sym("TSGgml_MiniMaxM3SlotNPast");
            stats = (stats_fn) sym("TSGgml_MiniMaxM3GetStats");
            release = (free_fn) sym("TSGgml_MiniMaxM3Free");
            last_error = (err_fn) sym("TSGgml_GetLastError");
            if (!load || !forward || !batched || !vocab || !reset || !n_past || !ctx || !slot_alloc || !set_active ||
                !slot_free || !slot_n_past || !stats || !release)
                throw std::runtime_error(std::string(path) + " does not export the TSGgml_MiniMaxM3* C ABI");
        }
    };

    struct summary { double median = 0, mean = 0, stdev = 0; };
    summary summarize(std::vector<double> v)
    {
        summary s;
        if (v.empty()) return s;
        s.mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
        double acc = 0;
        for (double x : v) acc += (x - s.mean) * (x - s.mean);
        s.stdev = v.size() > 1 ? std::sqrt(acc / (v.size() - 1)) : 0.0;
        std::sort(v.begin(), v.end());
        s.median = v.size() % 2 ? v[v.size() / 2] : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]);
        return s;
    }

    void print_cell(const std::string & test, const std::vector<double> & tps, const std::string & extra)
    {
        const summary s = summarize(tps);
        std::string reps;
        for (size_t i = 0; i < tps.size(); i++) { char b[32]; snprintf(b, sizeof(b), "%s%.2f", i ? "," : "", tps[i]); reps += b; }
        std::printf("{\"test\":\"%s\",\"reps\":[%s],\"median\":%.2f,\"mean\":%.2f,\"stdev\":%.2f%s}\n", test.c_str(), reps.c_str(),
                    s.median, s.mean, s.stdev, extra.c_str());
        std::fflush(stdout);
    }

#if defined(MM3_BENCH_CUPTI)
    // ---- CUPTI activity tracing (per-kernel GPU time of the timed decode steps) ----
    struct cupti_api
    {
        using reg_fn = CUptiResult (*) (CUpti_BuffersCallbackRequestFunc, CUpti_BuffersCallbackCompleteFunc);
        using kind_fn = CUptiResult (*) (CUpti_ActivityKind);
        using flush_fn = CUptiResult (*) ();
        using next_fn = CUptiResult (*) (uint8_t *, size_t, CUpti_Activity **);
        reg_fn reg = nullptr;
        kind_fn enable = nullptr, disable = nullptr;
        flush_fn flush = nullptr;
        next_fn next = nullptr;
        bool ok = false;
    };
    cupti_api g_cupti;
    struct kernel_rec { uint64_t start, end; uint32_t dev; std::string name; };
    std::vector<kernel_rec> g_recs;
    struct copy_rec { uint64_t start, end; uint32_t dev; uint64_t bytes; uint8_t kind; };
    std::vector<copy_rec> g_copies;

    void CUPTIAPI buf_requested(uint8_t ** buffer, size_t * size, size_t * max_records)
    {
        *size = 16 << 20;
        *buffer = (uint8_t *) aligned_alloc(8, *size);
        *max_records = 0;
    }

    void CUPTIAPI buf_completed(CUcontext, uint32_t, uint8_t * buffer, size_t, size_t valid)
    {
        CUpti_Activity * rec = nullptr;
        while (g_cupti.next(buffer, valid, &rec) == CUPTI_SUCCESS)
        {
            if (rec->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL || rec->kind == CUPTI_ACTIVITY_KIND_KERNEL)
            {
                auto * k = (CUpti_ActivityKernel13 *) rec;
                g_recs.push_back({ k->start, k->end, k->deviceId, k->name ? k->name : "?" });
            }
            else if (rec->kind == CUPTI_ACTIVITY_KIND_MEMCPY)
            {
                auto * c = (CUpti_ActivityMemcpy7 *) rec;
                g_copies.push_back({ c->start, c->end, c->deviceId, c->bytes, c->copyKind });
            }
            else if (rec->kind == CUPTI_ACTIVITY_KIND_MEMCPY2)
            {
                auto * c = (CUpti_ActivityMemcpyPtoP4 *) rec;
                g_copies.push_back({ c->start, c->end, c->srcDeviceId, c->bytes, 255 });
            }
        }
        free(buffer);
    }

    bool cupti_init()
    {
        const char * path = std::getenv("MM3_CUPTI_LIB");
        void * lib = dlopen(path && *path ? path : "libcupti.so", RTLD_NOW | RTLD_GLOBAL);
        if (!lib) { std::fprintf(stderr, "[bench] CUPTI unavailable (%s); --cupti ignored\n", dlerror()); return false; }
        g_cupti.reg = (cupti_api::reg_fn) dlsym(lib, "cuptiActivityRegisterCallbacks");
        g_cupti.enable = (cupti_api::kind_fn) dlsym(lib, "cuptiActivityEnable");
        g_cupti.disable = (cupti_api::kind_fn) dlsym(lib, "cuptiActivityDisable");
        g_cupti.flush = (cupti_api::flush_fn) dlsym(lib, "cuptiActivityFlushAll");
        g_cupti.next = (cupti_api::next_fn) dlsym(lib, "cuptiActivityGetNextRecord");
        g_cupti.ok = g_cupti.reg && g_cupti.enable && g_cupti.disable && g_cupti.flush && g_cupti.next &&
                     g_cupti.reg(buf_requested, buf_completed) == CUPTI_SUCCESS;
        if (!g_cupti.ok) std::fprintf(stderr, "[bench] CUPTI symbols missing; --cupti ignored\n");
        return g_cupti.ok;
    }

    void cupti_start()
    {
        g_recs.clear();
        g_copies.clear();
        g_cupti.enable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
        g_cupti.enable(CUPTI_ACTIVITY_KIND_MEMCPY);
        g_cupti.enable(CUPTI_ACTIVITY_KIND_MEMCPY2);
    }

    // Per device: busy time (union of kernel intervals) per step, kernel count per
    // step; overall: the kernels that take the most time per step, and the copies.
    void cupti_stop_and_report(const std::string & cell, int steps, double wall_s)
    {
        g_cupti.flush();
        g_cupti.disable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
        g_cupti.disable(CUPTI_ACTIVITY_KIND_MEMCPY);
        g_cupti.disable(CUPTI_ACTIVITY_KIND_MEMCPY2);
        g_cupti.flush();
        std::map<uint32_t, std::vector<std::pair<uint64_t, uint64_t>>> by_dev;
        std::map<std::string, std::pair<double, int64_t>> by_name;
        uint64_t t_min = UINT64_MAX, t_max = 0;
        for (const kernel_rec & r : g_recs)
        {
            by_dev[r.dev].push_back({ r.start, r.end });
            std::string n = r.name;
            if (n.size() > 90) n = n.substr(0, 90);
            by_name[n].first += (r.end - r.start) / 1e3;
            by_name[n].second++;
            t_min = std::min(t_min, r.start);
            t_max = std::max(t_max, r.end);
        }
        // Time any GPU is busy (the union over devices): with layers split, decode runs one GPU at
        // a time, so wall minus this is the step's idle time (stage hand-offs, uploads, host work).
        double any_busy_ms = 0;
        {
            std::vector<std::pair<uint64_t, uint64_t>> all;
            for (const kernel_rec & r : g_recs) all.push_back({ r.start, r.end });
            std::sort(all.begin(), all.end());
            uint64_t busy = 0, cs = 0, ce = 0;
            for (auto & p : all)
            {
                if (p.first > ce) { busy += ce - cs; cs = p.first; ce = p.second; }
                else ce = std::max(ce, p.second);
            }
            busy += ce - cs;
            any_busy_ms = busy / 1e6 / steps;
        }
        double busy_total = 0;
        std::string devs;
        for (auto & kv : by_dev)
        {
            auto & iv = kv.second;
            std::sort(iv.begin(), iv.end());
            uint64_t busy = 0, cs = 0, ce = 0;
            for (auto & p : iv)
            {
                if (p.first > ce) { busy += ce - cs; cs = p.first; ce = p.second; }
                else ce = std::max(ce, p.second);
            }
            busy += ce - cs;
            const double ms = busy / 1e6 / steps;
            busy_total += ms;
            char b[96];
            snprintf(b, sizeof(b), "%s\"%u\":{\"busy_ms\":%.3f,\"kernels\":%.1f}", devs.empty() ? "" : ",", kv.first, ms,
                     (double) iv.size() / steps);
            devs += b;
        }
        double copy_ms = 0, copy_mb = 0;
        int64_t n_copies = 0;
        for (const copy_rec & c : g_copies) { copy_ms += (c.end - c.start) / 1e6; copy_mb += c.bytes / 1048576.0; n_copies++; }
        std::printf("{\"cupti\":\"%s\",\"steps\":%d,\"wall_ms_per_step\":%.3f,\"busy_ms_per_step_sum\":%.3f,\"any_gpu_busy_ms_per_step\":%.3f,"
                    "\"copies_per_step\":%.1f,\"copy_ms_per_step\":%.3f,\"copy_mb_per_step\":%.3f,\"devices\":{%s}}\n", cell.c_str(), steps,
                    wall_s * 1e3 / steps, busy_total, any_busy_ms, (double) n_copies / steps, copy_ms / steps, copy_mb / steps, devs.c_str());
        std::vector<std::pair<double, std::string>> rows;
        for (auto & kv : by_name) rows.push_back({ kv.second.first, kv.first });
        std::sort(rows.rbegin(), rows.rend());
        for (size_t i = 0; i < rows.size() && i < 30; i++)
            std::printf("  %9.1f us/step %7.1f calls/step  %s\n", rows[i].first / steps,
                        (double) by_name[rows[i].second].second / steps, rows[i].second.c_str());
        std::fflush(stdout);
    }
#endif
}

int main(int argc, char ** argv)
{
    try
    {
        if (argc < 4)
        {
            std::fprintf(stderr, "usage: minimax-m3-native-bench LIBGGMLOPS MODEL.gguf N_GPU [--pp LIST] [--depth LIST] [--tg M] "
                                 "[--np LIST] [--np-depth D] [--ttft LIST] [--ttft-at D+L,...] [--reps R] [--ubatch N] [--ctx N] [--cpu-moe N] [--cupti] [--seed S]\n");
            return 2;
        }
        const char * lib = argv[1];
        const char * model_path = argv[2];
        const int n_gpu = std::atoi(argv[3]);
        std::vector<int> pp, depth, np, ttft;
        std::vector<std::pair<int, int>> ttft_at;   // (D, L)
        int tg = 128, reps = 3, ubatch = 2048, ctx = 0, np_depth = 512, cpu_moe = 0;
        bool cupti = false, tg_after_pp = false, warmup = true;
        unsigned seed = 1234;
        for (int i = 4; i < argc; i++)
        {
            const std::string a = argv[i];
            auto val = [&]() -> const char * { if (i + 1 >= argc) throw std::runtime_error(a + " needs a value"); return argv[++i]; };
            if (a == "--pp") pp = parse_list(val());
            else if (a == "--depth") depth = parse_list(val());
            else if (a == "--np") np = parse_list(val());
            else if (a == "--ttft") ttft = parse_list(val());
            else if (a == "--ttft-at")
            {
                std::stringstream ss(val());
                std::string item;
                while (std::getline(ss, item, ','))
                {
                    const size_t plus = item.find('+');
                    if (plus == std::string::npos) throw std::runtime_error("--ttft-at takes D+L items, not " + item);
                    ttft_at.emplace_back(std::atoi(item.substr(0, plus).c_str()), std::atoi(item.substr(plus + 1).c_str()));
                }
            }
            else if (a == "--np-depth") np_depth = std::atoi(val());
            else if (a == "--tg") tg = std::atoi(val());
            else if (a == "--reps") reps = std::max(1, std::atoi(val()));
            else if (a == "--ubatch") ubatch = std::atoi(val());
            else if (a == "--ctx") ctx = std::atoi(val());
            else if (a == "--seed") seed = (unsigned) std::atoi(val());
            else if (a == "--cpu-moe") cpu_moe = std::atoi(val());   // routed experts of the first N layers on the host (-1 = auto)
            else if (a == "--cupti") cupti = true;
            else if (a == "--tg-after-pp") tg_after_pp = true;
            else if (a == "--no-warmup") warmup = false;
            else throw std::runtime_error("unknown option " + a);
        }
#if defined(MM3_BENCH_CUPTI)
        if (cupti) cupti = cupti_init();
#else
        if (cupti) { std::fprintf(stderr, "[bench] built without -DMM3_BENCH_CUPTI; --cupti ignored\n"); cupti = false; }
#endif
        int need = 1;
        for (int p : pp) need = std::max(need, p + (tg_after_pp ? tg : 0));
        for (int d : depth) need = std::max(need, d + tg);
        if (!np.empty()) need = std::max(need, np_depth + tg);
        for (const auto & dl : ttft_at) need = std::max(need, dl.first + dl.second + reps);
        if (ctx <= 0) ctx = (need + 512 + 255) / 256 * 256;

        api T(lib);
        const auto t0 = clk::now();
        void * m = T.load(model_path, n_gpu, ctx, ubatch, 0, cpu_moe, n_gpu > 0 ? "CUDA" : "CPU", 1, nullptr, 0);
        if (!m) throw std::runtime_error(std::string("load refused: ") + (T.last_error ? T.last_error() : "?"));
        const double load_s = seconds(t0, clk::now());
        const int vocab = T.vocab(m);
        std::printf("{\"load_s\":%.1f,\"ctx\":%d,\"ubatch\":%d,\"vocab\":%d,\"n_gpu\":%d}\n", load_s, T.ctx(m), ubatch, vocab, n_gpu);
        std::fflush(stdout);

        std::mt19937 rng(seed);
        std::uniform_int_distribution<int32_t> tok(0, 199999);   // no special token
        auto random_tokens = [&](int n) { std::vector<int32_t> v((size_t) n); for (auto & x : v) x = tok(rng); return v; };
        const int max_np = np.empty() ? 1 : std::max(1, *std::max_element(np.begin(), np.end()));
        std::vector<float> logits((size_t) vocab * (size_t) max_np);

        auto prefill = [&](int n) -> double
        {
            if (n <= 0) return 0.0;
            const std::vector<int32_t> t = random_tokens(n);
            const auto a = clk::now();
            if (!T.forward(m, t.data(), n, logits.data(), 0))
                throw std::runtime_error("Forward of " + std::to_string(n) + " tokens failed" +
                                         (T.last_error ? std::string(": ") + T.last_error() : std::string()));
            return seconds(a, clk::now());
        };

        // ---- pp<N>: prefill of N tokens on an empty cache (all but the last row's logits pruned) ----
        for (int n : pp)
        {
            std::vector<double> tps, tg_tps;
            for (int r = (warmup ? -1 : 0); r < reps; r++)
            {
                if (!T.reset(m)) throw std::runtime_error("reset failed");
                const double s = prefill(n);
                if (r >= 0) tps.push_back(n / s);
                if (tg_after_pp && r >= 0)
                {
                    // --tg-after-pp: the decode cell at depth n reuses this prefill (long contexts)
                    const std::vector<int32_t> t = random_tokens(tg);
                    const auto a = clk::now();
                    for (int i = 0; i < tg; i++)
                        if (!T.forward(m, t.data() + i, 1, logits.data(), 0))
                            throw std::runtime_error("decode step " + std::to_string(i) + " at depth " + std::to_string(n) + " failed");
                    tg_tps.push_back(tg / seconds(a, clk::now()));
                }
            }
            int64_t st[16] = {};
            T.stats(m, st, 16);
            char extra[160];
            snprintf(extra, sizeof(extra), ",\"pipelined_calls\":%" PRId64 ",\"sparse_segments\":%" PRId64 ",\"dense_segments\":%" PRId64,
                     st[6], st[13], st[14]);
            print_cell("pp" + std::to_string(n), tps, extra);
            if (tg_after_pp) print_cell("tg" + std::to_string(tg) + "@d" + std::to_string(n), tg_tps, ",\"after_pp\":true");
        }

        // ---- ttft<L>: a short prompt whose length the executor has not seen yet (rep r uses
        // L + r tokens, so every timed call builds its graph, as a new chat request does) ----
        for (int n : ttft)
        {
            std::vector<double> ms;
            for (int r = 0; r < reps; r++)
            {
                if (!T.reset(m)) throw std::runtime_error("reset failed");
                ms.push_back(prefill(n + 1 + r * 7) * 1e3);
            }
            print_cell("ttft" + std::to_string(n) + "_ms", ms, ",\"unit\":\"ms\"");
        }

        // ---- ttft<D>+<L>: an L-token prefill at depth D (untimed D-token prefill first); pass 0 uses
        // lengths L + r the executor has not seen, pass 1 repeats them (its graphs are cached) ----
        for (const auto & dl : ttft_at)
        {
            std::vector<double> ms[2];
            for (int pass = 0; pass < 2; pass++)
                for (int r = 0; r < reps; r++)
                {
                    if (!T.reset(m)) throw std::runtime_error("reset failed");
                    prefill(dl.first);
                    ms[pass].push_back(prefill(dl.second + r) * 1e3);
                }
            const std::string name = "ttft" + std::to_string(dl.first) + "+" + std::to_string(dl.second);
            print_cell(name + "_ms", ms[0], ",\"unit\":\"ms\",\"shape\":\"new\"");
            print_cell(name + "_cached_ms", ms[1], ",\"unit\":\"ms\",\"shape\":\"cached\"");
        }

        // ---- tg<M> @ d<D>: M single-token decodes after an untimed D-token prefill ----
        for (int d : depth)
        {
            std::vector<double> tps;
            // r == -1 warms up; r == reps is the CUPTI-profiled run (not timed: tracing slows it).
            for (int r = -1; r < reps + (cupti ? 1 : 0); r++)
            {
                if (!T.reset(m)) throw std::runtime_error("reset failed");
                prefill(d);
                const std::vector<int32_t> t = random_tokens(tg);
#if defined(MM3_BENCH_CUPTI)
                const bool prof = cupti && r == reps;
                if (prof) cupti_start();
#endif
                const auto a = clk::now();
                for (int i = 0; i < tg; i++)
                    if (!T.forward(m, t.data() + i, 1, logits.data(), 0))
                        throw std::runtime_error("decode step " + std::to_string(i) + " at depth " + std::to_string(d) + " failed");
                const double s = seconds(a, clk::now());
#if defined(MM3_BENCH_CUPTI)
                if (prof) cupti_stop_and_report("tg" + std::to_string(tg) + "@d" + std::to_string(d), tg, s);
#endif
                if (r >= 0 && r < reps) tps.push_back(tg / s);
            }
            int64_t st[16] = {};
            T.stats(m, st, 16);
            char extra[160];
            snprintf(extra, sizeof(extra), ",\"graph_builds\":%" PRId64 ",\"graph_reuses\":%" PRId64, st[0], st[1]);
            print_cell("tg" + std::to_string(tg) + "@d" + std::to_string(d), tps, extra);
        }

        // ---- np<K>: K slots, each prefilled with np_depth tokens, decoding together ----
        for (int k : np)
        {
            std::vector<double> tps;
            for (int r = -1; r < reps + (cupti ? 1 : 0); r++)
            {
                if (!T.reset(m)) throw std::runtime_error("reset failed");
                std::vector<int32_t> slots((size_t) k), toks((size_t) k), pos((size_t) k);
                for (int i = 0; i < k; i++)
                {
                    slots[(size_t) i] = i == 0 ? 0 : T.slot_alloc(m);
                    if (slots[(size_t) i] < 0) throw std::runtime_error("slot alloc failed");
                    if (!T.set_active(m, slots[(size_t) i])) throw std::runtime_error("set active slot failed");
                    if (i > 0 && !T.reset(m)) throw std::runtime_error("reset failed");
                    prefill(np_depth);
                }
#if defined(MM3_BENCH_CUPTI)
                const bool prof = cupti && r == reps;
                if (prof) cupti_start();
#endif
                const auto a = clk::now();
                for (int s = 0; s < tg; s++)
                {
                    for (int i = 0; i < k; i++)
                    {
                        toks[(size_t) i] = tok(rng);
                        pos[(size_t) i] = T.slot_n_past(m, slots[(size_t) i]);
                    }
                    if (!T.batched(m, k, slots.data(), toks.data(), pos.data(), logits.data()))
                        throw std::runtime_error("batched decode of " + std::to_string(k) + " slots failed at step " + std::to_string(s));
                }
                const double sec = seconds(a, clk::now());
#if defined(MM3_BENCH_CUPTI)
                if (prof) cupti_stop_and_report("np" + std::to_string(k), tg, sec);
#endif
                if (r >= 0 && r < reps) tps.push_back((double) k * tg / sec);
                T.set_active(m, 0);
                for (int i = 1; i < k; i++) T.slot_free(m, slots[(size_t) i]);
            }
            print_cell("np" + std::to_string(k) + "@d" + std::to_string(np_depth) + "x" + std::to_string(tg), tps, ",\"aggregate\":true");
        }
        T.release(m);
        return 0;
    }
    catch (const std::exception & e)
    {
        std::fprintf(stderr, "minimax-m3-native-bench: %s\n", e.what());
        return 1;
    }
}
