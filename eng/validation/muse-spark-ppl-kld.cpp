// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Teacher-forced quality comparison of TensorSharp's Muse Spark native executor
// against an unchanged llama.cpp, on IDENTICAL token IDs: perplexity of each
// engine, and the KL divergence / top-1 agreement of TensorSharp's next-token
// distributions against llama.cpp's (the method of llama-perplexity's
// --kl-divergence). No tokenizer, chat template or sampling differences enter.
//
// Scoring follows llama-perplexity: each chunk is evaluated from an empty cache
// and positions [ctx/2, ctx - 1) are scored, their logits predicting the next
// chunk token. The chunk layout does not: here every chunk is BOS + (ctx - 1)
// consecutive text tokens at a stride of ctx - 1 (llama-perplexity strides ctx
// and overwrites token 0 with BOS only when the vocab adds one). Both engines
// share this layout, so their PPL/KLD compare with each other, but the PPLs do
// not compare with a llama-perplexity run on the same text.
//
// Build (llama.cpp modes; link an unchanged llama.cpp build):
//   g++ -std=c++17 -O2 -DWITH_LLAMA muse-spark-ppl-kld.cpp -I<llama>/include -I<llama>/ggml/include \
//       -L<llama>/build/bin -Wl,-rpath,<llama>/build/bin -lllama -o ppl-kld-llama
// Build (TensorSharp mode; loads libGgmlOps.so at run time):
//   g++ -std=c++17 -O2 muse-spark-ppl-kld.cpp -ldl -o ppl-kld-ts
//
// Usage:
//   ppl-kld-llama tokenize MODEL.gguf TEXT.txt TOKENS.bin MAX_TOKENS
//   ppl-kld-llama llama    MODEL.gguf TOKENS.bin CTX REF.lp16           -> writes F16 log-probs of scored rows
//   ppl-kld-ts    ts       libGgmlOps.so MODEL.gguf TOKENS.bin CTX REF.lp16 N_GPU
//   ppl-kld-llama toplast  MODEL.gguf TOKENS.bin                         -> top-10 next tokens after TOKENS
//   ppl-kld-ts    toplast  libGgmlOps.so MODEL.gguf TOKENS.bin N_GPU SPLIT[,SPLIT...][;SPLITS...]
//                 (prefills TOKENS in pieces ending at each SPLIT, then the rest: chunking sensitivity;
//                  each ';'-separated spec runs on a fresh cache and prints its time)
//   either        compare  A.lp16 B.lp16 TOKENS.bin   -> PPL of each, KL(A || B), top-1 agreement
//   ppl-kld-ts    ts-dsv4  libGgmlOps.so MODEL.gguf TOKENS.bin CTX REF.lp16 N_GPU
//                 (DeepSeek V4.1: the same comparison through TSGgml_Dsv4*; scored rows are decode steps.
//                  PPL_TP must equal N_GPU when set. TS_DSV41_YOCO_KEEP is inherited - unset means the
//                  loader's default bounded replay - and is recorded in the JSON line with the TP ranks,
//                  KLD/NLL by offset after the prefill (0-127, 128-511, 512+). stderr adds exact
//                  per-chunk sums, "[ts-dsv4] chunk-sums K nll A nll_ref B kld C scored N top1 M",
//                  for paired per-chunk tests.)
// PPL_UBATCH sets the micro-batch (default 512 in llama, ts and llama toplast;
// 2048 in ts toplast). PPL_TP=N runs TensorSharp tensor-parallel over N GPUs
// (its --tp) instead of the layer split over N_GPU. PPL_OUT makes the ts mode also write its own
// log-probabilities, so two runs of one engine can be compared.
// The ts mode scores rows of one all-rows call per chunk, which TensorSharp
// runs on its sequential path. PPL_PIPELINED=1 instead prefills each chunk's
// unscored first half as an ordinary prompt (pipelined across the GPUs when
// TensorSharp's pipelined prefill is on and the half has at least two of its
// micro-batches; the run fails if it did not pipeline) and scores the second
// half on the K/V that prefill wrote.
// Every mode prints one JSON line. REF.lp16 is [scored rows][vocab] of F16
// log-probabilities preceded by a 24-byte header (magic "LPD2", vocab, rows,
// ctx, BOS id, FNV-1a of the text tokens the chunks consume), so the ts and
// compare modes rebuild exactly the reference's chunks or refuse.
#include <algorithm>
#include <cinttypes>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <string>
#include <vector>

#if defined(WITH_LLAMA)
#include "llama.h"
#else
#include <dlfcn.h>
#include <execinfo.h>
#include <pthread.h>
#include <signal.h>
#include <atomic>
#include <map>
#include <thread>
#endif

namespace
{
    constexpr uint32_t kMagic = 0x3244504Cu;   // "LPD2"

    uint16_t to_half(float f)
    {
        uint32_t x;
        std::memcpy(&x, &f, 4);
        const uint32_t sign = (x >> 16) & 0x8000u;
        int32_t exp = (int32_t) ((x >> 23) & 0xff) - 127 + 15;
        uint32_t mant = x & 0x7fffffu;
        if (exp <= 0) return (uint16_t) sign;                      // log-probs never need subnormals
        if (exp >= 31) return (uint16_t) (sign | 0x7bffu);          // clamp to the largest finite half
        uint32_t h = sign | ((uint32_t) exp << 10) | (mant >> 13);
        if ((mant >> 12) & 1u) h++;                                 // round half up
        return (uint16_t) h;
    }

    float from_half(uint16_t h)
    {
        const uint32_t sign = (uint32_t) (h & 0x8000u) << 16;
        const uint32_t exp = (h >> 10) & 0x1fu;
        const uint32_t mant = h & 0x3ffu;
        uint32_t x = sign;
        if (exp != 0) x |= ((exp - 15 + 127) << 23) | (mant << 13);
        float f;
        std::memcpy(&f, &x, 4);
        return f;
    }

    // log-softmax of one row, in double for the normalizer.
    void log_softmax(const float * logits, int vocab, std::vector<double> & out)
    {
        double mx = -INFINITY;
        for (int i = 0; i < vocab; i++) mx = std::max(mx, (double) logits[i]);
        double sum = 0.0;
        for (int i = 0; i < vocab; i++) sum += std::exp((double) logits[i] - mx);
        const double lse = mx + std::log(sum);
        out.resize((size_t) vocab);
        for (int i = 0; i < vocab; i++) out[(size_t) i] = (double) logits[i] - lse;
    }

    std::vector<int32_t> read_tokens(const char * path)
    {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("cannot open tokens");
        std::vector<int32_t> t;
        int32_t v;
        while (f.read(reinterpret_cast<char *>(&v), 4)) t.push_back(v);
        return t;
    }

    struct chunks
    {
        int ctx = 0, n = 0, first = 0;
        int32_t bos = -1;
        const std::vector<int32_t> * text = nullptr;
        // Chunk c: BOS + text[c * (ctx - 1) ...]
        void fill(int c, std::vector<int32_t> & out) const
        {
            out.resize((size_t) ctx);
            out[0] = bos;
            for (int i = 1; i < ctx; i++) out[(size_t) i] = (*text)[(size_t) c * (ctx - 1) + (i - 1)];
        }
        int scored_per_chunk() const { return ctx - 1 - first; }
        // FNV-1a over the text tokens the chunks consume (the reference's identity).
        uint32_t hash() const
        {
            uint32_t h = 2166136261u;
            const uint8_t * p = reinterpret_cast<const uint8_t *>(text->data());
            const size_t bytes = (size_t) n * (size_t) (ctx - 1) * sizeof(int32_t);
            for (size_t i = 0; i < bytes; i++) { h ^= p[i]; h *= 16777619u; }
            return h;
        }
    };

    chunks plan(const std::vector<int32_t> & text, int ctx, int32_t bos)
    {
        if (ctx < 16) throw std::runtime_error("ctx too small");
        chunks c;
        c.ctx = ctx;
        c.first = ctx / 2;
        c.bos = bos;
        c.text = &text;
        c.n = (int) (text.size() / (size_t) (ctx - 1));
        if (c.n < 1) throw std::runtime_error("not enough tokens for one chunk");
        return c;
    }
}


namespace
{
    // {magic, vocab, rows, ctx, bos, token hash}
    struct lp_header
    {
        uint32_t w[6] = {};
        void read(std::ifstream & f, const char * path)
        {
            if (!f.read(reinterpret_cast<char *>(w), sizeof(w)) || w[0] != kMagic)
                throw std::runtime_error(std::string("not an LPD2 log-prob file (regenerate the reference with this tool version): ") + path);
        }
        // The chunks the file's rows were scored on, from TOKENS.bin, or a refusal.
        chunks rebuild(const std::vector<int32_t> & text) const
        {
            const chunks c = plan(text, (int) w[3], (int32_t) w[4]);
            if ((uint32_t) (c.n * c.scored_per_chunk()) != w[2]) throw std::runtime_error("scored row count differs from the reference");
            if (c.hash() != w[5]) throw std::runtime_error("TOKENS.bin differs from the reference's token stream");
            return c;
        }
    };

    struct lp_file
    {
        std::ifstream f;
        lp_header hdr;
        explicit lp_file(const char * path) : f(path, std::ios::binary) { hdr.read(f, path); }
    };

    // KL(A || B) and agreement over two files of F16 log-probabilities of the same rows.
    int run_compare(int argc, char ** argv)
    {
        if (argc != 5) throw std::runtime_error("compare A.lp16 B.lp16 TOKENS.bin");
        lp_file a(argv[2]), b(argv[3]);
        if (std::memcmp(a.hdr.w, b.hdr.w, sizeof(a.hdr.w)) != 0) throw std::runtime_error("files differ in shape, BOS or tokens");
        const std::vector<int32_t> text = read_tokens(argv[4]);
        const int ctx = (int) a.hdr.w[3], n_vocab = (int) a.hdr.w[1];
        const chunks c = a.hdr.rebuild(text);
        std::vector<uint16_t> ra((size_t) n_vocab), rb((size_t) n_vocab);
        std::vector<int32_t> chunk;
        double nll_a = 0, nll_b = 0, kld = 0, kmax = 0;
        long rows = 0, same = 0;
        std::vector<double> klds;
        for (int ci = 0; ci < c.n; ci++)
        {
            c.fill(ci, chunk);
            for (int i = c.first; i < ctx - 1; i++)
            {
                a.f.read(reinterpret_cast<char *>(ra.data()), (std::streamsize) (ra.size() * 2));
                b.f.read(reinterpret_cast<char *>(rb.data()), (std::streamsize) (rb.size() * 2));
                if (!a.f || !b.f) throw std::runtime_error("file too short");
                double za = 0, zb = 0;
                for (int k = 0; k < n_vocab; k++) { za += std::exp((double) from_half(ra[(size_t) k])); zb += std::exp((double) from_half(rb[(size_t) k])); }
                const double la = std::log(za), lb = std::log(zb);
                double kl = 0;
                int ta = 0, tb = 0;
                for (int k = 0; k < n_vocab; k++)
                {
                    const double pa = (double) from_half(ra[(size_t) k]) - la, pb = (double) from_half(rb[(size_t) k]) - lb;
                    kl += std::exp(pa) * (pa - pb);
                    if (from_half(ra[(size_t) k]) > from_half(ra[(size_t) ta])) ta = k;
                    if (from_half(rb[(size_t) k]) > from_half(rb[(size_t) tb])) tb = k;
                }
                const int32_t next = chunk[(size_t) i + 1];
                nll_a -= (double) from_half(ra[(size_t) next]) - la;
                nll_b -= (double) from_half(rb[(size_t) next]) - lb;
                kld += kl; kmax = std::max(kmax, kl); klds.push_back(kl);
                same += ta == tb ? 1 : 0;
                rows++;
            }
        }
        std::sort(klds.begin(), klds.end());
        auto pct = [&](double q) { return klds[(size_t) std::min<double>(klds.size() - 1, q * (klds.size() - 1))]; };
        std::printf("{\"mode\":\"compare\",\"rows\":%ld,\"ppl_a\":%.6f,\"ppl_b\":%.6f,\"kld_mean\":%.6f,\"kld_median\":%.6f,"
                    "\"kld_p99\":%.6f,\"kld_max\":%.6f,\"top1_agreement\":%.6f}\n", rows, std::exp(nll_a / rows), std::exp(nll_b / rows),
                    kld / rows, pct(0.5), pct(0.99), kmax, (double) same / rows);
        return 0;
    }

    int env_ubatch(int fallback = 512) { const char * v = std::getenv("PPL_UBATCH"); return v ? std::max(1, std::atoi(v)) : fallback; }
}
#if defined(WITH_LLAMA)

static int run_tokenize(int argc, char ** argv)
{
    if (argc != 6) throw std::runtime_error("tokenize MODEL TEXT OUT MAX_TOKENS");
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.vocab_only = true;
    llama_model * model = llama_model_load_from_file(argv[2], mp);
    if (!model) throw std::runtime_error("model load failed");
    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::ifstream f(argv[3], std::ios::binary);
    std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    std::vector<llama_token> toks(text.size() + 16);
    int n = llama_tokenize(vocab, text.data(), (int32_t) text.size(), toks.data(), (int32_t) toks.size(), false, false);
    if (n < 0) throw std::runtime_error("tokenize failed");
    toks.resize((size_t) std::min<long>(n, std::atol(argv[5])));
    std::ofstream o(argv[4], std::ios::binary | std::ios::trunc);
    o.write(reinterpret_cast<const char *>(toks.data()), (std::streamsize) (toks.size() * 4));
    std::printf("{\"mode\":\"tokenize\",\"tokens\":%zu,\"bos\":%d,\"vocab\":%d}\n", toks.size(),
                llama_vocab_bos(vocab), llama_vocab_n_tokens(vocab));
    llama_model_free(model);
    return 0;
}

static int run_llama(int argc, char ** argv)
{
    if (argc != 6) throw std::runtime_error("llama MODEL TOKENS CTX OUT");
    const std::vector<int32_t> text = read_tokens(argv[3]);
    const int ctx = std::atoi(argv[4]);
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    mp.split_mode = LLAMA_SPLIT_MODE_LAYER;
    const auto t0 = std::chrono::steady_clock::now();
    llama_model * model = llama_model_load_from_file(argv[2], mp);
    if (!model) throw std::runtime_error("model load failed");
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    auto cp = llama_context_default_params();
    cp.n_ctx = (uint32_t) ctx;
    cp.n_batch = (uint32_t) ctx;
    cp.n_ubatch = (uint32_t) env_ubatch();
    cp.n_seq_max = 1;
    cp.type_k = GGML_TYPE_F16;
    cp.type_v = GGML_TYPE_F16;
    llama_context * lctx = llama_init_from_model(model, cp);
    if (!lctx) throw std::runtime_error("context init failed");
    const auto t1 = std::chrono::steady_clock::now();
    const llama_token bos = llama_vocab_bos(vocab);
    if (bos == LLAMA_TOKEN_NULL) throw std::runtime_error("the vocab has no BOS; the chunk layout needs one");
    const chunks c = plan(text, ctx, bos);
    std::ofstream o(argv[5], std::ios::binary | std::ios::trunc);
    const uint32_t hdr[6] = { kMagic, (uint32_t) n_vocab, (uint32_t) (c.n * c.scored_per_chunk()), (uint32_t) ctx, (uint32_t) bos, c.hash() };
    o.write(reinterpret_cast<const char *>(hdr), sizeof(hdr));
    llama_batch batch = llama_batch_init(ctx, 0, 1);
    std::vector<int32_t> chunk;
    std::vector<double> lp;
    std::vector<uint16_t> row((size_t) n_vocab);
    double nll = 0.0;
    long scored = 0;
    for (int ci = 0; ci < c.n; ci++)
    {
        c.fill(ci, chunk);
        llama_memory_clear(llama_get_memory(lctx), true);
        batch.n_tokens = ctx;
        for (int i = 0; i < ctx; i++)
        {
            batch.token[i] = chunk[(size_t) i];
            batch.pos[i] = i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = i >= c.first && i < ctx - 1;
        }
        if (llama_decode(lctx, batch) != 0) throw std::runtime_error("llama_decode failed");
        for (int i = c.first; i < ctx - 1; i++)
        {
            log_softmax(llama_get_logits_ith(lctx, i), n_vocab, lp);
            nll -= lp[(size_t) chunk[(size_t) i + 1]];
            scored++;
            for (int k = 0; k < n_vocab; k++) row[(size_t) k] = to_half((float) lp[(size_t) k]);
            o.write(reinterpret_cast<const char *>(row.data()), (std::streamsize) (row.size() * 2));
        }
        std::fprintf(stderr, "[llama] chunk %d/%d ppl so far %.4f\n", ci + 1, c.n, std::exp(nll / scored));
    }
    const auto t2 = std::chrono::steady_clock::now();
    std::printf("{\"mode\":\"llama\",\"chunks\":%d,\"ctx\":%d,\"scored\":%ld,\"ppl\":%.6f,\"load_s\":%.1f,\"eval_s\":%.1f}\n",
                c.n, ctx, scored, std::exp(nll / scored),
                std::chrono::duration<double>(t1 - t0).count(), std::chrono::duration<double>(t2 - t1).count());
    llama_batch_free(batch);
    llama_free(lctx);
    llama_model_free(model);
    return 0;
}

static void print_top(const char * mode, const std::vector<double> & lp)
{
    std::vector<int> idx(lp.size());
    for (size_t i = 0; i < idx.size(); i++) idx[i] = (int) i;
    std::partial_sort(idx.begin(), idx.begin() + 10, idx.end(), [&](int a, int b) { return lp[(size_t) a] > lp[(size_t) b]; });
    std::printf("{\"mode\":\"%s\",\"top\":[", mode);
    for (int i = 0; i < 10; i++) std::printf("%s[%d,%.6f]", i ? "," : "", idx[(size_t) i], lp[(size_t) idx[(size_t) i]]);
    std::printf("]}\n");
}

static int run_toplast(int argc, char ** argv)
{
    if (argc != 4) throw std::runtime_error("toplast MODEL TOKENS");
    const std::vector<int32_t> toks = read_tokens(argv[3]);
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    mp.split_mode = LLAMA_SPLIT_MODE_LAYER;
    llama_model * model = llama_model_load_from_file(argv[2], mp);
    if (!model) throw std::runtime_error("model load failed");
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    auto cp = llama_context_default_params();
    cp.n_ctx = (uint32_t) std::max<size_t>(512, toks.size() + 1);
    cp.n_batch = cp.n_ctx;
    cp.n_ubatch = (uint32_t) env_ubatch(512);
    cp.type_k = GGML_TYPE_F16;
    cp.type_v = GGML_TYPE_F16;
    llama_context * lctx = llama_init_from_model(model, cp);
    llama_batch batch = llama_batch_init((int32_t) toks.size(), 0, 1);
    batch.n_tokens = (int32_t) toks.size();
    for (size_t i = 0; i < toks.size(); i++)
    {
        batch.token[i] = toks[i]; batch.pos[i] = (int32_t) i; batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 0;
        batch.logits[i] = i + 1 == toks.size();
    }
    if (llama_decode(lctx, batch) != 0) throw std::runtime_error("llama_decode failed");
    std::vector<double> lp;
    log_softmax(llama_get_logits_ith(lctx, (int32_t) toks.size() - 1), n_vocab, lp);
    print_top("llama-toplast", lp);
    llama_batch_free(batch);
    llama_free(lctx);
    llama_model_free(model);
    return 0;
}

int main(int argc, char ** argv)
{
    try
    {
        if (argc >= 2 && std::strcmp(argv[1], "toplast") == 0) return run_toplast(argc, argv);
        if (argc >= 2 && std::strcmp(argv[1], "compare") == 0) return run_compare(argc, argv);
        if (argc >= 2 && std::strcmp(argv[1], "tokenize") == 0) return run_tokenize(argc, argv);
        if (argc >= 2 && std::strcmp(argv[1], "llama") == 0) return run_llama(argc, argv);
        std::fprintf(stderr, "usage: tokenize|llama ... (see the file header)\n");
        return 2;
    }
    catch (const std::exception & e)
    {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}

#else   // TensorSharp mode

// PPL_SAMPLE=1: a poor man's profiler for the calling thread (ptrace is often
// unavailable in containers). A helper thread signals it every 20 ms and the
// handler records its stack; the most common frames print at exit.
namespace sampler
{
    constexpr int kMax = 20000, kDepth = 48;
    void * frames[kMax][kDepth];
    int depth[kMax];
    std::atomic<int> n{0};
    std::atomic<bool> stop{false};
    pthread_t target;

    void on_signal(int)
    {
        const int i = n.load();
        if (i >= kMax) return;
        depth[i] = backtrace(frames[i], kDepth);
        n.store(i + 1);
    }

    std::thread start()
    {
        target = pthread_self();
        struct sigaction sa = {};
        sa.sa_handler = on_signal;
        sa.sa_flags = SA_RESTART;
        sigaction(SIGUSR2, &sa, nullptr);
        return std::thread([] {
            while (!stop.load()) { pthread_kill(target, SIGUSR2); std::this_thread::sleep_for(std::chrono::milliseconds(20)); }
        });
    }

    // A frame as "name" when the symbol is exported, else "file(+0xoffset)" so addr2line can
    // resolve a static function later.
    std::string frame_name(const char * sym)
    {
        std::string f = sym;
        const size_t a = f.find('('), b = f.find('+', a == std::string::npos ? 0 : a);
        if (a != std::string::npos && b != std::string::npos && b > a + 1) return f.substr(a + 1, b - a - 1);
        const size_t e = f.find(')');
        const size_t slash = f.rfind('/', a);
        return f.substr(slash == std::string::npos ? 0 : slash + 1, e == std::string::npos ? f.size() : e + 1 - (slash == std::string::npos ? 0 : slash + 1));
    }

    bool driver_frame(const std::string & f)
    {
        return f.find("libcuda") != std::string::npos || f.find("libc.so") != std::string::npos || f.find("vdso") != std::string::npos ||
               f == "__sched_yield" || f == "ioctl" || f == "clock_gettime" || f.rfind("cu", 0) == 0;
    }

    void report()
    {
        // Key: the CUDA API the thread is in (if any), then the first callers outside the driver.
        std::map<std::string, int> top;
        for (int i = 0; i < n.load(); i++)
        {
            char ** sym = backtrace_symbols(frames[i], depth[i]);
            std::string api, key;
            int d = 2;
            for (; d < depth[i]; d++)
            {
                const std::string f = frame_name(sym[d]);
                if (f.rfind("cuda", 0) == 0) { api = f; d++; break; }
                if (!driver_frame(f)) break;
                if (f.rfind("cu", 0) == 0) api = f;
            }
            int kept = 0;
            for (; d < depth[i] && kept < 8; d++, kept++) key += (kept ? " <- " : "") + frame_name(sym[d]);
            free(sym);
            top["[" + (api.empty() ? std::string("-") : api) + "] " + key]++;
        }
        std::vector<std::pair<int, std::string>> v;
        for (auto & kv : top) v.push_back({ kv.second, kv.first });
        std::sort(v.rbegin(), v.rend());
        std::fprintf(stderr, "[sample] %d samples\n", n.load());
        for (size_t i = 0; i < v.size() && i < 25; i++) std::fprintf(stderr, "[sample] %5d  %s\n", v[i].first, v[i].second.c_str());
    }
}

using load_tp_fn = void * (*)(const char *, int, int, int, int, int, const char *, int, const char *, int);
using forward_fn = int (*)(void *, const int32_t *, int, float *, int);
using handle_fn = int (*)(void *);
using free_fn = void (*)(void *);
using err_fn = const char * (*)();
using stats_fn = int (*)(void *, int64_t *, int);

// PPL_TP=N (N > 1) loads TensorSharp with tensor parallelism over N GPUs
// instead of a layer split over N_GPU.
static void * load_ts(void * lib, const char * model, int n_gpu, int ctx, int ubatch)
{
    auto load = (load_tp_fn) dlsym(lib, "TSGgml_MuseSparkLoadModelTp");
    if (!load) throw std::runtime_error("missing TSGgml_MuseSparkLoadModelTp");
    const char * tp = std::getenv("PPL_TP");
    return load(model, n_gpu, ctx, ubatch, 0, 0, "CUDA", 1, nullptr, tp ? std::atoi(tp) : 0);
}

static void print_top_ts(const char * mode, const std::vector<double> & lp)
{
    std::vector<int> idx(lp.size());
    for (size_t i = 0; i < idx.size(); i++) idx[i] = (int) i;
    std::partial_sort(idx.begin(), idx.begin() + 10, idx.end(), [&](int a, int b) { return lp[(size_t) a] > lp[(size_t) b]; });
    std::printf("{\"mode\":\"%s\",\"top\":[", mode);
    for (int i = 0; i < 10; i++) std::printf("%s[%d,%.6f]", i ? "," : "", idx[(size_t) i], lp[(size_t) idx[(size_t) i]]);
    std::printf("]}\n");
}

static int run_ts_toplast(int argc, char ** argv)
{
    if (argc != 7) throw std::runtime_error("toplast LIB MODEL TOKENS N_GPU SPLITS");
    void * lib = dlopen(argv[2], RTLD_NOW | RTLD_LOCAL);
    if (!lib) throw std::runtime_error(std::string("dlopen: ") + dlerror());
    auto forward = (forward_fn) dlsym(lib, "TSGgml_MuseSparkForward");
    auto reset = (handle_fn) dlsym(lib, "TSGgml_MuseSparkResetChecked");
    auto n_vocab_of = (handle_fn) dlsym(lib, "TSGgml_MuseSparkVocabSize");
    auto release = (free_fn) dlsym(lib, "TSGgml_MuseSparkFree");
    auto last_error = (err_fn) dlsym(lib, "TSGgml_GetLastError");
    if (!forward || !reset || !n_vocab_of || !release) throw std::runtime_error("missing TSGgml exports");
    const std::vector<int32_t> toks = read_tokens(argv[4]);
    const int ub = env_ubatch(2048);
    const int ctx = (int) std::max<size_t>(4096, (toks.size() + (size_t) ub + 255) / 256 * 256);
    void * m = load_ts(lib, argv[3], std::atoi(argv[5]), ctx, ub);
    if (!m) throw std::runtime_error(std::string("load refused: ") + (last_error ? last_error() : "?"));
    const int n_vocab = n_vocab_of(m);
    std::vector<float> logits((size_t) n_vocab);
    std::vector<double> lp;
    // One line per chunking: "0" = one pass; "22" = 22 tokens then the rest; "22,100" = three pieces.
    std::string specs = argv[6];
    size_t at = 0;
    while (at <= specs.size())
    {
        size_t end = specs.find(';', at);
        if (end == std::string::npos) end = specs.size();
        const std::string spec = specs.substr(at, end - at);
        at = end + 1;
        std::vector<int> cuts;
        for (size_t p = 0; p < spec.size();)
        {
            size_t q = spec.find(',', p);
            if (q == std::string::npos) q = spec.size();
            const int v = std::atoi(spec.substr(p, q - p).c_str());
            if (v > 0 && v < (int) toks.size()) cuts.push_back(v);
            p = q + 1;
        }
        cuts.push_back((int) toks.size());
        if (!reset(m)) throw std::runtime_error("reset failed");
        const auto t_spec = std::chrono::steady_clock::now();
        const bool sample = std::getenv("PPL_SAMPLE") != nullptr;
        std::thread sampling;
        if (sample) sampling = sampler::start();
        int done = 0;
        for (int cut : cuts)
        {
            const bool last = cut == (int) toks.size();
            if (!forward(m, toks.data() + done, cut - done, last ? logits.data() : nullptr, 0))
                throw std::runtime_error("forward failed");
            done = cut;
        }
        if (sample)
        {
            sampler::stop.store(true);
            sampling.join();
            sampler::report();
            sampler::stop.store(false);
            sampler::n.store(0);
        }
        log_softmax(logits.data(), n_vocab, lp);
        const std::string mode = "ts-toplast split=" + spec + " seconds=" +
            std::to_string(std::chrono::duration<double>(std::chrono::steady_clock::now() - t_spec).count());
        print_top_ts(mode.c_str(), lp);
    }
    release(m);
    return 0;
}

// ts-dsv4: the same comparison for DeepSeek V4.1 (TSGgml_Dsv4*), whose Forward
// returns only the last row's logits. Each chunk's unscored first half is one
// prompt call; every scored row is then one decode step, i.e. what generation
// runs. The llama and compare modes are model-independent and are shared.
using dsv4_load_fn = void * (*)(const char *, int, int, int, int, const char *, int, const char *, int);
using dsv4_forward_fn = int (*)(void *, const int32_t *, int, float *);

static int run_ts_dsv4(int argc, char ** argv)
{
    if (argc != 8) throw std::runtime_error("ts-dsv4 libGgmlOps.so MODEL.gguf TOKENS.bin CTX REF.lp16 N_GPU");
    void * lib = dlopen(argv[2], RTLD_NOW | RTLD_LOCAL);
    if (!lib) throw std::runtime_error(std::string("dlopen: ") + dlerror());
    auto load = (dsv4_load_fn) dlsym(lib, "TSGgml_Dsv4LoadModel");
    auto forward = (dsv4_forward_fn) dlsym(lib, "TSGgml_Dsv4Forward");
    auto reset = (handle_fn) dlsym(lib, "TSGgml_Dsv4ResetChecked");
    auto n_vocab_of = (handle_fn) dlsym(lib, "TSGgml_Dsv4VocabSize");
    auto release = (free_fn) dlsym(lib, "TSGgml_Dsv4Free");
    auto last_error = (err_fn) dlsym(lib, "TSGgml_GetLastError");
    if (!load || !forward || !reset || !n_vocab_of || !release) throw std::runtime_error("missing TSGgml_Dsv4 exports");

    const std::vector<int32_t> text = read_tokens(argv[4]);
    const int ctx = std::atoi(argv[5]);
    std::ifstream ref(argv[6], std::ios::binary);
    lp_header hdr;
    hdr.read(ref, argv[6]);
    if ((int) hdr.w[3] != ctx) throw std::runtime_error("CTX differs from the reference's");
    const chunks c = hdr.rebuild(text);
    const char * tp = std::getenv("PPL_TP");
    const auto t0 = std::chrono::steady_clock::now();
    // n_cpu_moe 0: no routed-expert offload, the CLI's default (CpuMoeNone).
    void * m = load(argv[3], std::atoi(argv[7]), ctx, env_ubatch(), 0, nullptr, 0, "CUDA", tp ? std::atoi(tp) : 0);
    if (!m) throw std::runtime_error(std::string("load refused: ") + (last_error ? last_error() : "?"));
    const auto t1 = std::chrono::steady_clock::now();
    const int n_vocab = n_vocab_of(m);
    if ((uint32_t) n_vocab != hdr.w[1]) throw std::runtime_error("vocab differs from the reference");

    std::vector<float> logits((size_t) n_vocab);
    std::vector<int32_t> chunk;
    std::vector<double> lp;
    std::vector<uint16_t> row((size_t) n_vocab), own((size_t) n_vocab);
    std::ofstream out;
    if (const char * path = std::getenv("PPL_OUT"))
    {
        out.open(path, std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char *>(hdr.w), sizeof(hdr.w));
    }
    double nll = 0.0, nll_ref = 0.0, kld = 0.0, kld_max = 0.0;
    long scored = 0, top1_same = 0;
    std::vector<double> klds;
    // Offset bins after the prefilled half: YOCO bounded replay narrows only
    // the prompt's last positions' windows, so its cost sits in the first
    // generated tokens, which a whole-chunk mean dilutes.
    const int bin_edges[] = {0, 128, 512, 1 << 30};
    double bin_kld[3] = {}, bin_nll[3] = {}, bin_nll_ref[3] = {};
    long bin_count[3] = {};
    for (int ci = 0; ci < c.n; ci++)
    {
        c.fill(ci, chunk);
        const double chunk_nll = nll, chunk_nll_ref = nll_ref, chunk_kld = kld;
        const long chunk_scored = scored, chunk_top1 = top1_same;
        if (!reset(m)) throw std::runtime_error("reset failed");
        if (forward(m, chunk.data(), c.first, logits.data()) != 0) throw std::runtime_error("prefill failed");
        for (int i = c.first; i < ctx - 1; i++)
        {
            if (forward(m, chunk.data() + i, 1, logits.data()) != 0) throw std::runtime_error("decode failed");
            log_softmax(logits.data(), n_vocab, lp);
            if (out.is_open())
            {
                for (int k = 0; k < n_vocab; k++) own[(size_t) k] = to_half((float) lp[(size_t) k]);
                out.write(reinterpret_cast<const char *>(own.data()), (std::streamsize) (own.size() * 2));
            }
            if (!ref.read(reinterpret_cast<char *>(row.data()), (std::streamsize) (row.size() * 2)))
                throw std::runtime_error("reference file too short");
            const int32_t next = chunk[(size_t) i + 1];
            nll -= lp[(size_t) next];
            nll_ref -= from_half(row[(size_t) next]);
            double zr = 0.0;
            for (int k = 0; k < n_vocab; k++) zr += std::exp((double) from_half(row[(size_t) k]));
            const double lzr = std::log(zr);
            double kl = 0.0;
            int top_ref = 0, top_ts = 0;
            for (int k = 0; k < n_vocab; k++)
            {
                const double lr = (double) from_half(row[(size_t) k]) - lzr;
                kl += std::exp(lr) * (lr - lp[(size_t) k]);
                if (lr > (double) from_half(row[(size_t) top_ref]) - lzr) top_ref = k;
                if (lp[(size_t) k] > lp[(size_t) top_ts]) top_ts = k;
            }
            kld += kl;
            kld_max = std::max(kld_max, kl);
            klds.push_back(kl);
            top1_same += top_ref == top_ts ? 1 : 0;
            scored++;
            const int offset = i - c.first;
            for (int b = 0; b < 3; ++b)
                if (offset >= bin_edges[b] && offset < bin_edges[b + 1])
                {
                    bin_kld[b] += kl;
                    bin_nll[b] -= lp[(size_t) next];
                    bin_nll_ref[b] -= from_half(row[(size_t) next]);
                    ++bin_count[b];
                }
        }
        std::fprintf(stderr, "[ts-dsv4] chunk %d/%d ppl %.4f (ref %.4f) mean KLD %.6f top-1 %.4f\n", ci + 1, c.n,
                     std::exp(nll / scored), std::exp(nll_ref / scored), kld / scored, (double) top1_same / scored);
        // Exact per-chunk sums for paired tests (the running line above rounds).
        std::fprintf(stderr, "[ts-dsv4] chunk-sums %d nll %.9f nll_ref %.9f kld %.9f scored %ld top1 %ld\n", ci + 1,
                     nll - chunk_nll, nll_ref - chunk_nll_ref, kld - chunk_kld, scored - chunk_scored,
                     top1_same - chunk_top1);
    }
    const auto t2 = std::chrono::steady_clock::now();
    std::sort(klds.begin(), klds.end());
    auto pct = [&](double q) { return klds[(size_t) std::min<double>(klds.size() - 1, q * (klds.size() - 1))]; };
    std::string bins;
    for (int b = 0; b < 3; ++b)
    {
        if (!bin_count[b]) continue;
        char text[256];
        std::snprintf(text, sizeof(text), "%s{\"from\":%d,\"to\":%d,\"scored\":%ld,\"kld_mean\":%.6f,"
                      "\"nll_mean\":%.6f,\"nll_ref_mean\":%.6f}", bins.empty() ? "" : ",", bin_edges[b],
                      b == 2 ? -1 : bin_edges[b + 1], bin_count[b], bin_kld[b] / bin_count[b],
                      bin_nll[b] / bin_count[b], bin_nll_ref[b] / bin_count[b]);
        bins += text;
    }
    // The run's own settings, so a JSON line says what it measured.
    const char * yoco = std::getenv("TS_DSV41_YOCO_KEEP");
    std::printf("{\"mode\":\"ts-dsv4\",\"chunks\":%d,\"ctx\":%d,\"scored\":%ld,\"ppl\":%.6f,\"ppl_reference\":%.6f,"
                "\"kld_mean\":%.6f,\"kld_median\":%.6f,\"kld_p99\":%.6f,\"kld_max\":%.6f,\"top1_agreement\":%.6f,"
                "\"tp\":%d,\"yoco_keep\":\"%s\",\"ubatch\":%d,\"kld_bins\":[%s],\"load_s\":%.1f,\"eval_s\":%.1f}\n",
                c.n, ctx, scored, std::exp(nll / scored), std::exp(nll_ref / scored), kld / scored, pct(0.5), pct(0.99),
                kld_max, (double) top1_same / scored, tp ? std::atoi(tp) : 0, yoco ? yoco : "default",
                env_ubatch(), bins.c_str(), std::chrono::duration<double>(t1 - t0).count(),
                std::chrono::duration<double>(t2 - t1).count());
    release(m);
    return 0;
}

int main(int argc, char ** argv)
{
    try
    {
        if (argc >= 2 && std::strcmp(argv[1], "toplast") == 0) return run_ts_toplast(argc, argv);
        if (argc >= 2 && std::strcmp(argv[1], "compare") == 0) return run_compare(argc, argv);
        if (argc >= 2 && std::strcmp(argv[1], "ts-dsv4") == 0) return run_ts_dsv4(argc, argv);
        if (argc != 8 || std::strcmp(argv[1], "ts") != 0)
        {
            std::fprintf(stderr, "usage: ts libGgmlOps.so MODEL.gguf TOKENS.bin CTX REF.lp16 N_GPU\n");
            return 2;
        }
        void * lib = dlopen(argv[2], RTLD_NOW | RTLD_LOCAL);
        if (!lib) throw std::runtime_error(std::string("dlopen: ") + dlerror());
        auto forward = (forward_fn) dlsym(lib, "TSGgml_MuseSparkForward");
        auto reset = (handle_fn) dlsym(lib, "TSGgml_MuseSparkResetChecked");
        auto n_vocab_of = (handle_fn) dlsym(lib, "TSGgml_MuseSparkVocabSize");
        auto release = (free_fn) dlsym(lib, "TSGgml_MuseSparkFree");
        auto last_error = (err_fn) dlsym(lib, "TSGgml_GetLastError");
        auto get_stats = (stats_fn) dlsym(lib, "TSGgml_MuseSparkGetStats");
        if (!forward || !reset || !n_vocab_of || !release || !get_stats) throw std::runtime_error("missing TSGgml exports");

        const std::vector<int32_t> text = read_tokens(argv[4]);
        const int ctx = std::atoi(argv[5]);
        std::ifstream ref(argv[6], std::ios::binary);
        lp_header hdr;
        hdr.read(ref, argv[6]);
        if ((int) hdr.w[3] != ctx) throw std::runtime_error("CTX differs from the reference's");
        const chunks c = hdr.rebuild(text);
        const int n_gpu = std::atoi(argv[7]);
        const bool pipelined = std::getenv("PPL_PIPELINED") && std::strcmp(std::getenv("PPL_PIPELINED"), "1") == 0;

        // A 512-row ubatch keeps the all-row output head's buffer small.
        const auto t0 = std::chrono::steady_clock::now();
        void * m = load_ts(lib, argv[3], n_gpu, ctx, env_ubatch());
        if (!m) throw std::runtime_error(std::string("load refused: ") + (last_error ? last_error() : "?"));
        const auto t1 = std::chrono::steady_clock::now();
        const int n_vocab = n_vocab_of(m);
        if ((uint32_t) n_vocab != hdr.w[1]) throw std::runtime_error("vocab differs from the reference");

        std::vector<float> logits((size_t) ctx * n_vocab);
        std::vector<int32_t> chunk;
        std::vector<double> lp;
        std::vector<uint16_t> row((size_t) n_vocab), own((size_t) n_vocab);
        std::ofstream out;
        if (const char * path = std::getenv("PPL_OUT"))
        {
            out.open(path, std::ios::binary | std::ios::trunc);
            out.write(reinterpret_cast<const char *>(hdr.w), sizeof(hdr.w));
        }
        double nll = 0.0, nll_ref = 0.0, kld = 0.0, kld_max = 0.0, dp2 = 0.0;
        long scored = 0, top1_same = 0;
        int64_t piped_calls = 0, piped_ubatches = 0;
        std::vector<double> klds;
        for (int ci = 0; ci < c.n; ci++)
        {
            c.fill(ci, chunk);
            if (!reset(m)) throw std::runtime_error("reset failed");
            // Row r of `logits` holds chunk position row0 + r.
            int row0 = 0;
            if (pipelined)
            {
                int64_t before[11] = {}, after[11] = {};
                if (get_stats(m, before, 11) < 10) throw std::runtime_error("this libGgmlOps reports no pipeline stats");
                if (!forward(m, chunk.data(), c.first, nullptr, 0)) throw std::runtime_error("forward failed");
                get_stats(m, after, 11);
                // Deltas: the load-time warm-up already counts one pipelined call.
                if (after[6] - before[6] != 1 || after[8] != before[8])
                    throw std::runtime_error("PPL_PIPELINED: the unscored half did not run pipelined (pipeline off, or fewer than two micro-batches)");
                piped_calls += after[6] - before[6];
                piped_ubatches += after[7] - before[7];
                if (!forward(m, chunk.data() + c.first, ctx - c.first, logits.data(), 1)) throw std::runtime_error("forward failed");
                row0 = c.first;
            }
            else if (!forward(m, chunk.data(), ctx, logits.data(), 1)) throw std::runtime_error("forward failed");
            for (int i = c.first; i < ctx - 1; i++)
            {
                log_softmax(logits.data() + (size_t) (i - row0) * n_vocab, n_vocab, lp);
                if (out.is_open())
                {
                    for (int k = 0; k < n_vocab; k++) own[(size_t) k] = to_half((float) lp[(size_t) k]);
                    out.write(reinterpret_cast<const char *>(own.data()), (std::streamsize) (own.size() * 2));
                }
                if (!ref.read(reinterpret_cast<char *>(row.data()), (std::streamsize) (row.size() * 2)))
                    throw std::runtime_error("reference file too short");
                const int32_t next = chunk[(size_t) i + 1];
                nll -= lp[(size_t) next];
                nll_ref -= from_half(row[(size_t) next]);
                // KL(reference || TensorSharp) over the full vocabulary; the
                // F16-stored reference is renormalized so it sums to one.
                double zr = 0.0;
                for (int k = 0; k < n_vocab; k++) zr += std::exp((double) from_half(row[(size_t) k]));
                const double lzr = std::log(zr);
                double kl = 0.0;
                int top_ref = 0, top_ts = 0;
                for (int k = 0; k < n_vocab; k++)
                {
                    const double lr = (double) from_half(row[(size_t) k]) - lzr;
                    kl += std::exp(lr) * (lr - lp[(size_t) k]);
                    if (lr > (double) from_half(row[(size_t) top_ref]) - lzr) top_ref = k;
                    if (lp[(size_t) k] > lp[(size_t) top_ts]) top_ts = k;
                }
                const double d = std::exp(lp[(size_t) next]) - std::exp((double) from_half(row[(size_t) next]) - lzr);
                dp2 += d * d;
                kld += kl;
                kld_max = std::max(kld_max, kl);
                klds.push_back(kl);
                top1_same += top_ref == top_ts ? 1 : 0;
                scored++;
            }
            std::fprintf(stderr, "[ts] chunk %d/%d ppl %.4f (ref %.4f) mean KLD %.6f top-1 %.4f\n", ci + 1, c.n,
                         std::exp(nll / scored), std::exp(nll_ref / scored), kld / scored, (double) top1_same / scored);
        }
        const auto t2 = std::chrono::steady_clock::now();
        std::sort(klds.begin(), klds.end());
        auto pct = [&](double q) { return klds[(size_t) std::min<double>(klds.size() - 1, q * (klds.size() - 1))]; };
        std::printf("{\"mode\":\"%s\",\"chunks\":%d,\"ctx\":%d,\"scored\":%ld,\"ppl\":%.6f,\"ppl_reference\":%.6f,"
                    "\"kld_mean\":%.6f,\"kld_median\":%.6f,\"kld_p99\":%.6f,\"kld_max\":%.6f,\"top1_agreement\":%.6f,"
                    "\"rms_dp_next\":%.6f,\"pipelined_calls\":%" PRId64 ",\"pipelined_ubatches\":%" PRId64 ",\"load_s\":%.1f,\"eval_s\":%.1f}\n",
                    pipelined ? "ts-pipelined" : "ts", c.n, ctx, scored, std::exp(nll / scored), std::exp(nll_ref / scored), kld / scored, pct(0.5),
                    pct(0.99), kld_max, (double) top1_same / scored, std::sqrt(dp2 / scored), piped_calls, piped_ubatches,
                    std::chrono::duration<double>(t1 - t0).count(), std::chrono::duration<double>(t2 - t1).count());
        release(m);
        return 0;
    }
    catch (const std::exception & e)
    {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}

#endif
