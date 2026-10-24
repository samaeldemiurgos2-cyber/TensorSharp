// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Teacher-forced logit parity of TensorSharp's MiniMax-M3 native executor
// (TSGgml_MiniMaxM3*) against an unchanged llama.cpp of the MiniMax-M3.1 branch,
// on IDENTICAL token IDs: every engine dumps the full-vocabulary logits of every
// position of one sequence, and `compare` reports per-position KL divergence,
// top-1 agreement and the largest logit difference. No tokenizer, template or
// sampling enters. Both engines are loaded at run time with dlopen, one engine per
// process (libllama and libGgmlOps each carry their own ggml).
//
// Build (one binary; the llama.cpp headers give the structs passed by value, no
// library is linked):
//   g++ -std=c++17 -O2 -DMM3_WITH_LLAMA_H -I<llama>/include -I<llama>/ggml/include
//       minimax-m3-parity.cpp -ldl -lpthread -o minimax-m3-parity
// Without -DMM3_WITH_LLAMA_H the llama, stats, msa and `tokens text` modes are left out.
//
// Usage (every mode prints one JSON line on stdout):
//   llama     LIBLLAMA MODEL.gguf TOKENS.bin OUT.f32 [N_GPU] [UBATCH]
//             libllama (the .so or its directory), flash attention on, one sequence
//             (n_seq_max 1, so MSA runs), F16 K/V, one llama_decode of all tokens split
//             into UBATCH-row micro-batches (default 512). UBATCH 1 runs every position
//             through the decode path. N_GPU 0 runs on the CPU (default: all layers on
//             the GPUs CUDA_VISIBLE_DEVICES shows).
//   ts        LIBGGMLOPS MODEL.gguf TOKENS.bin OUT.f32 N_GPU [UBATCH] [CHUNK]
//             TSGgml_MiniMaxM3LoadModelTp (contract C1), then Forward(all_rows = 1) over
//             CHUNK-token pieces (default UBATCH, default 512), so chunked prefill on an
//             existing cache is part of what is measured.
//   ts-decode LIBGGMLOPS MODEL.gguf TOKENS.bin OUT.f32 N_GPU [UBATCH] [PREFILL]
//             prefills the first PREFILL tokens (default half) with all rows, then decodes
//             the rest one token per Forward call.
//   compare   A.f32 B.f32 [EDGES|-] [PER_POS.tsv]
//             KL(A || B) per position over the full vocabulary, top-1 agreement, max and
//             RMS |logit_A - logit_B|; EDGES ("64,2048") adds the same summary per position
//             range [0,64) [64,2048) [2048,end); PER_POS.tsv gets one line per position.
//   stats     LIBLLAMA MODEL.gguf TOKENS.bin [N_GPU] [UBATCH]
//             llama.cpp's eval callback on the same run as `llama`: per layer the RMS of the
//             attention output, FFN output and residual stream, and the routed-expert
//             histogram (distinct experts, normalized entropy, busiest expert's share).
//   msa       LIBLLAMA MODEL.gguf TOKENS.bin [N_GPU] [UBATCH] [SEL.tsv]
//             llama.cpp's MSA block choice on the same run as `llama`: for every layer,
//             index head and query that has more competing past blocks than free top-k
//             slots, the margin between the last kept and the first dropped block score
//             (absolute, and relative to the largest |block score| of that row), with the
//             smallest margins per layer and head and the count of near-ties. SEL.tsv gets
//             every decision with its kept blocks, to diff another engine's choice against.
//   greedy    LOGITS.f32 TOKENS.bin FIRST
//             whether every token from position FIRST on is the argmax of the previous
//             position's logits (a speculative run's output checked against teacher-forced
//             target logits); a mismatch reports the logit gap it lost by, so rounding-level
//             near-ties are told apart from wrong tokens.
//   tokens random N SEED OUT.bin [HI]   uniform ids in [0, HI) (default 200000: no special token)
//   tokens ids    IDS.txt OUT.bin       whitespace- or comma-separated ids from any tokenizer
//   tokens text   LIBLLAMA MODEL.gguf TEXT.txt OUT.bin [MAX]   llama.cpp's tokenization
//             (special tokens parsed, nothing added), truncated to MAX ids
//
// OUT.f32 = 32-byte header {magic "MM3L", version 1, vocab, rows, FNV-1a of the token
// ids, ubatch, producer (1 llama, 2 ts, 3 ts-decode), prefill} + rows x vocab F32 logits;
// row r holds position r's logits (the prediction of token r + 1). `compare` refuses
// two files of different vocab, row count or token stream.
//
// Environment: MM3_TP=N loads TensorSharp with tensor parallelism over N GPUs instead of
// the layer split over N_GPU; MM3_CPU_MOE=N passes N as n_cpu_moe (default 0, no
// routed-expert offload); MM3_TS_BACKEND names TensorSharp's GPU backend ("CUDA" by
// default, "Metal"; N_GPU 0 is the CPU). TensorSharp options (TS_MINIMAX_M3_*) are read from the
// process environment by the executor. MM3_LLAMA_DENSE=1 turns llama.cpp's flash
// attention off, which makes it run dense attention instead of MSA (it warns): a
// reference for how far sparse attention moves a fixture's logits, never for parity.
#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(MM3_WITH_LLAMA_H)
#include "llama.h"
#endif

namespace
{
    constexpr uint32_t kMagic = 0x4C334D4Du;   // "MM3L"
    constexpr uint32_t kVersion = 1;
    enum producer : uint32_t { kLlama = 1, kTs = 2, kTsDecode = 3 };

    using clk = std::chrono::steady_clock;
    double seconds(clk::time_point a, clk::time_point b) { return std::chrono::duration<double>(b - a).count(); }

    std::vector<int32_t> read_tokens(const char * path)
    {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error(std::string("cannot open ") + path);
        std::vector<int32_t> t;
        int32_t v;
        while (f.read(reinterpret_cast<char *>(&v), 4)) t.push_back(v);
        if (t.empty()) throw std::runtime_error(std::string("no tokens in ") + path);
        return t;
    }

    void write_tokens(const char * path, const std::vector<int32_t> & t)
    {
        std::ofstream o(path, std::ios::binary | std::ios::trunc);
        o.write(reinterpret_cast<const char *>(t.data()), (std::streamsize) (t.size() * 4));
        if (!o) throw std::runtime_error(std::string("cannot write ") + path);
    }

    uint32_t fnv1a(const std::vector<int32_t> & t)
    {
        uint32_t h = 2166136261u;
        const uint8_t * p = reinterpret_cast<const uint8_t *>(t.data());
        for (size_t i = 0; i < t.size() * 4; i++) { h ^= p[i]; h *= 16777619u; }
        return h;
    }

    struct logits_header
    {
        uint32_t w[8] = {};
        uint32_t vocab() const { return w[2]; }
        uint32_t rows() const { return w[3]; }
    };

    // Rows are written as they are produced, so a multi-GB dump never sits in memory twice.
    struct logits_writer
    {
        std::ofstream o;
        logits_writer(const char * path, uint32_t vocab, uint32_t rows, uint32_t hash, uint32_t ubatch, producer p, uint32_t prefill)
            : o(path, std::ios::binary | std::ios::trunc)
        {
            if (!o) throw std::runtime_error(std::string("cannot write ") + path);
            const uint32_t w[8] = { kMagic, kVersion, vocab, rows, hash, ubatch, (uint32_t) p, prefill };
            o.write(reinterpret_cast<const char *>(w), sizeof(w));
        }
        void row(const float * x, size_t vocab)
        {
            o.write(reinterpret_cast<const char *>(x), (std::streamsize) (vocab * sizeof(float)));
            if (!o) throw std::runtime_error("short write of a logits row");
        }
    };

    struct logits_reader
    {
        std::ifstream f;
        logits_header h;
        explicit logits_reader(const char * path) : f(path, std::ios::binary)
        {
            if (!f.read(reinterpret_cast<char *>(h.w), sizeof(h.w)) || h.w[0] != kMagic || h.w[1] != kVersion)
                throw std::runtime_error(std::string("not an MM3L v1 logits file: ") + path);
        }
        void row(std::vector<float> & x)
        {
            x.resize(h.vocab());
            if (!f.read(reinterpret_cast<char *>(x.data()), (std::streamsize) (x.size() * sizeof(float))))
                throw std::runtime_error("logits file too short");
        }
    };

    // log-softmax in double for the normalizer
    void log_softmax(const std::vector<float> & x, std::vector<double> & out)
    {
        double mx = -INFINITY;
        for (float v : x) mx = std::max(mx, (double) v);
        double sum = 0.0;
        for (float v : x) sum += std::exp((double) v - mx);
        const double lse = mx + std::log(sum);
        out.resize(x.size());
        for (size_t i = 0; i < x.size(); i++) out[i] = (double) x[i] - lse;
    }

    int argmax(const std::vector<float> & x) { return (int) (std::max_element(x.begin(), x.end()) - x.begin()); }

    bool all_finite(const std::vector<float> & x)
    {
        for (float v : x) if (!std::isfinite(v)) return false;
        return true;
    }

    std::vector<int> parse_ints(const std::string & s)
    {
        std::vector<int> v;
        std::string cur;
        for (char c : s + ",")
        {
            if (c == ',' || c == ' ' || c == '\n' || c == '\t' || c == '\r')
            {
                if (!cur.empty()) { v.push_back(std::atoi(cur.c_str())); cur.clear(); }
            }
            else cur += c;
        }
        return v;
    }

    struct bin_stats
    {
        long rows = 0, top1 = 0;
        double kld = 0, kld_max = 0, dmax = 0, d2 = 0;
        long d2n = 0;
        void add(double kl, bool same, double dm, double d2row, long n)
        {
            rows++; top1 += same ? 1 : 0; kld += kl; kld_max = std::max(kld_max, kl);
            dmax = std::max(dmax, dm); d2 += d2row; d2n += n;
        }
    };

    int run_compare(int argc, char ** argv)
    {
        if (argc < 4 || argc > 6) throw std::runtime_error("compare A.f32 B.f32 [EDGES|-] [PER_POS.tsv]");
        logits_reader a(argv[2]), b(argv[3]);
        if (a.h.vocab() != b.h.vocab() || a.h.rows() != b.h.rows())
            throw std::runtime_error("the files differ in vocab or row count");
        if (a.h.w[4] != b.h.w[4]) throw std::runtime_error("the files were produced from different token streams");
        std::vector<int> edges;
        if (argc >= 5 && std::strcmp(argv[4], "-") != 0) edges = parse_ints(argv[4]);
        std::sort(edges.begin(), edges.end());
        std::ofstream per_pos;
        if (argc == 6)
        {
            per_pos.open(argv[5], std::ios::trunc);
            per_pos << "pos\tkld\ttop1_a\ttop1_b\tgap_a\tmax_abs_dlogit\trms_dlogit\n";
        }
        const int rows = (int) a.h.rows();
        std::vector<float> ra, rb;
        std::vector<double> la, lb;
        std::vector<double> klds;
        bin_stats all;
        std::vector<bin_stats> bins(edges.size() + 1);
        int first_mismatch = -1, kld_max_pos = 0, dmax_pos = 0;
        double worst_gap = 0.0;   // largest A-logit gap between A's and B's top-1 over the mismatches
        for (int r = 0; r < rows; r++)
        {
            a.row(ra);
            b.row(rb);
            if (!all_finite(ra) || !all_finite(rb)) throw std::runtime_error("non-finite logit at position " + std::to_string(r));
            log_softmax(ra, la);
            log_softmax(rb, lb);
            double kl = 0.0, dm = 0.0, d2 = 0.0;
            for (size_t k = 0; k < ra.size(); k++)
            {
                kl += std::exp(la[k]) * (la[k] - lb[k]);
                const double d = (double) ra[k] - (double) rb[k];
                dm = std::max(dm, std::fabs(d));
                d2 += d * d;
            }
            kl = std::max(kl, 0.0);
            const int ta = argmax(ra), tb = argmax(rb);
            const bool same = ta == tb;
            const double gap = (double) ra[(size_t) ta] - (double) ra[(size_t) tb];
            if (!same)
            {
                if (first_mismatch < 0) first_mismatch = r;
                worst_gap = std::max(worst_gap, gap);
            }
            if (kl > all.kld_max) kld_max_pos = r;
            if (dm > all.dmax) dmax_pos = r;
            all.add(kl, same, dm, d2, (long) ra.size());
            const int bi = (int) (std::upper_bound(edges.begin(), edges.end(), r) - edges.begin());
            bins[(size_t) bi].add(kl, same, dm, d2, (long) ra.size());
            klds.push_back(kl);
            if (per_pos.is_open())
                per_pos << r << '\t' << kl << '\t' << ta << '\t' << tb << '\t' << gap << '\t' << dm << '\t'
                        << std::sqrt(d2 / (double) ra.size()) << '\n';
        }
        std::sort(klds.begin(), klds.end());
        auto pct = [&](double q) { return klds[(size_t) std::min<double>((double) klds.size() - 1, q * (double) (klds.size() - 1))]; };
        std::string bin_json;
        for (size_t i = 0; i < bins.size() && !edges.empty(); i++)
        {
            const bin_stats & s = bins[i];
            if (!s.rows) continue;
            char text[320];
            std::snprintf(text, sizeof(text), "%s{\"from\":%d,\"to\":%d,\"rows\":%ld,\"kld_mean\":%.6g,\"kld_max\":%.6g,"
                          "\"top1_agreement\":%.6f,\"max_abs_dlogit\":%.6g,\"rms_dlogit\":%.6g}",
                          bin_json.empty() ? "" : ",", i == 0 ? 0 : edges[i - 1], i < edges.size() ? edges[i] : rows,
                          s.rows, s.kld / (double) s.rows, s.kld_max, (double) s.top1 / (double) s.rows, s.dmax,
                          std::sqrt(s.d2 / (double) s.d2n));
            bin_json += text;
        }
        std::printf("{\"mode\":\"compare\",\"rows\":%d,\"vocab\":%u,\"producers\":[%u,%u],\"ubatch\":[%u,%u],"
                    "\"kld_mean\":%.6g,\"kld_median\":%.6g,\"kld_p99\":%.6g,\"kld_max\":%.6g,\"kld_max_pos\":%d,"
                    "\"top1_agreement\":%.6f,\"top1_first_mismatch\":%d,\"top1_mismatch_max_gap\":%.6g,"
                    "\"max_abs_dlogit\":%.6g,\"max_abs_dlogit_pos\":%d,\"rms_dlogit\":%.6g,\"bins\":[%s]}\n",
                    rows, a.h.vocab(), a.h.w[6], b.h.w[6], a.h.w[5], b.h.w[5], all.kld / rows, pct(0.5), pct(0.99),
                    all.kld_max, kld_max_pos, (double) all.top1 / rows, first_mismatch, worst_gap, all.dmax, dmax_pos,
                    std::sqrt(all.d2 / (double) all.d2n), bin_json.c_str());
        return 0;
    }

    // Tokens [FIRST, n) against the argmax of the logits row before each of them.
    int run_greedy(int argc, char ** argv)
    {
        if (argc != 5) throw std::runtime_error("greedy LOGITS.f32 TOKENS.bin FIRST");
        logits_reader a(argv[2]);
        const std::vector<int32_t> toks = read_tokens(argv[3]);
        const int n = (int) toks.size(), first = std::atoi(argv[4]);
        if (a.h.rows() != (uint32_t) n || a.h.w[4] != fnv1a(toks))
            throw std::runtime_error("the logits were not produced from TOKENS.bin");
        if (first < 1 || first >= n) throw std::runtime_error("FIRST must be in [1, tokens)");
        long checked = 0, same = 0;
        int first_bad = -1;
        double worst_gap = 0.0, min_margin = INFINITY;   // min_margin: top-1 over top-2 where they agree
        std::vector<float> r;
        for (int i = 0; i + 1 < n; i++)   // row i predicts token i + 1
        {
            a.row(r);
            if (i + 1 < first) continue;
            const int top = argmax(r), tok = toks[(size_t) i + 1];
            checked++;
            if (top == tok)
            {
                same++;
                float second = -INFINITY;
                for (size_t k = 0; k < r.size(); k++)
                    if ((int) k != top) second = std::max(second, r[k]);
                min_margin = std::min(min_margin, (double) r[(size_t) top] - (double) second);
                continue;
            }
            if (first_bad < 0) first_bad = i + 1;
            worst_gap = std::max(worst_gap, (double) r[(size_t) top] - (double) r[(size_t) tok]);
        }
        std::printf("{\"mode\":\"greedy\",\"checked\":%ld,\"argmax_matches\":%ld,\"first_mismatch\":%d,\"mismatch_max_gap\":%.6g,"
                    "\"match_min_margin\":%.6g}\n", checked, same, first_bad, worst_gap, same ? min_margin : 0.0);
        return 0;
    }

    // splitmix64: a fixed generator, so a seed names the same ids on every platform
    uint64_t splitmix64(uint64_t & s)
    {
        uint64_t z = (s += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
}

#if defined(MM3_WITH_LLAMA_H)
namespace
{
#if defined(__APPLE__)
    constexpr const char * kLibExt = ".dylib";
#else
    constexpr const char * kLibExt = ".so";
#endif

    // The libllama entry points this tool calls, resolved from the library at run time.
    struct llama_api
    {
        void * lib = nullptr;
        void * base = nullptr;   // libggml-base, for the eval callback's tensor reads
        decltype(&::llama_backend_init) backend_init = nullptr;
        decltype(&::llama_model_default_params) model_default_params = nullptr;
        decltype(&::llama_context_default_params) context_default_params = nullptr;
        decltype(&::llama_model_load_from_file) model_load = nullptr;
        decltype(&::llama_model_free) model_free = nullptr;
        decltype(&::llama_model_get_vocab) model_get_vocab = nullptr;
        decltype(&::llama_vocab_n_tokens) vocab_n_tokens = nullptr;
        decltype(&::llama_init_from_model) init_from_model = nullptr;
        decltype(&::llama_free) free_ctx = nullptr;
        decltype(&::llama_batch_init) batch_init = nullptr;
        decltype(&::llama_batch_free) batch_free = nullptr;
        decltype(&::llama_decode) decode = nullptr;
        decltype(&::llama_get_logits_ith) get_logits_ith = nullptr;
        decltype(&::llama_tokenize) tokenize = nullptr;
        decltype(&::llama_model_meta_val_str) meta_val_str = nullptr;
        decltype(&::ggml_backend_tensor_get) tensor_get = nullptr;
        decltype(&::ggml_nbytes) nbytes = nullptr;

        template <typename T> void sym(void * h, T & f, const char * name)
        {
            f = (T) dlsym(h, name);
            if (!f) throw std::runtime_error(std::string("libllama lacks ") + name);
        }

        explicit llama_api(const std::string & where)
        {
            std::string dir = where, path = where;
            const bool is_lib = where.size() > 3 && (where.find(".so") != std::string::npos || where.find(".dylib") != std::string::npos);
            if (is_lib) dir = where.substr(0, where.find_last_of('/') == std::string::npos ? 0 : where.find_last_of('/'));
            else path = where + "/libllama" + kLibExt;
            lib = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
            if (!lib) throw std::runtime_error(std::string("dlopen libllama: ") + dlerror());
            // already loaded as libllama's dependency; this only returns its handle
            const std::string base_path = (dir.empty() ? std::string(".") : dir) + "/libggml-base" + kLibExt;
            base = dlopen(base_path.c_str(), RTLD_NOW | RTLD_LOCAL);
            if (!base) throw std::runtime_error(std::string("dlopen libggml-base: ") + dlerror());
            sym(lib, backend_init, "llama_backend_init");
            sym(lib, model_default_params, "llama_model_default_params");
            sym(lib, context_default_params, "llama_context_default_params");
            sym(lib, model_load, "llama_model_load_from_file");
            sym(lib, model_free, "llama_model_free");
            sym(lib, model_get_vocab, "llama_model_get_vocab");
            sym(lib, vocab_n_tokens, "llama_vocab_n_tokens");
            sym(lib, init_from_model, "llama_init_from_model");
            sym(lib, free_ctx, "llama_free");
            sym(lib, batch_init, "llama_batch_init");
            sym(lib, batch_free, "llama_batch_free");
            sym(lib, decode, "llama_decode");
            sym(lib, get_logits_ith, "llama_get_logits_ith");
            sym(lib, tokenize, "llama_tokenize");
            sym(lib, meta_val_str, "llama_model_meta_val_str");
            sym(base, tensor_get, "ggml_backend_tensor_get");
            sym(base, nbytes, "ggml_nbytes");
            backend_init();
        }
    };

    // Per-layer statistics gathered by llama.cpp's eval callback (stats mode).
    struct layer_stats
    {
        std::map<std::string, std::pair<double, long>> sumsq;   // tensor name -> (sum of squares, count)
        std::map<int, std::vector<long>> experts;               // layer -> routed-expert histogram
        std::vector<uint8_t> buf;
        llama_api * api = nullptr;
    };

    bool wanted(const char * name)
    {
        static const char * prefixes[] = { "inp_embd", "kqv_out-", "attn_out-", "ffn_out-", "ffn_moe_out-", "ffn_shexp-",
                                           "l_out-", "result_norm", "result_output", "ffn_moe_topk-" };
        for (const char * p : prefixes)
            if (std::strncmp(name, p, std::strlen(p)) == 0 && std::strstr(name, " (") == nullptr) return true;
        return false;
    }

    bool stats_cb(struct ggml_tensor * t, bool ask, void * user)
    {
        if (ask) return wanted(t->name);
        auto * s = (layer_stats *) user;
        const size_t n = s->api->nbytes(t);
        s->buf.resize(n);
        s->api->tensor_get(t, s->buf.data(), 0, n);
        const uint8_t * d = s->buf.data();
        if (std::strncmp(t->name, "ffn_moe_topk-", 13) == 0 && t->type == GGML_TYPE_I32)
        {
            const int il = std::atoi(t->name + 13);
            for (int64_t i1 = 0; i1 < t->ne[1]; i1++)
                for (int64_t i0 = 0; i0 < t->ne[0]; i0++)
                {
                    const int e = *(const int32_t *) (d + i0 * t->nb[0] + i1 * t->nb[1]);
                    auto & h = s->experts[il];
                    if (e >= 0)
                    {
                        if ((size_t) e >= h.size()) h.resize((size_t) e + 1, 0);
                        h[(size_t) e]++;
                    }
                }
            return true;
        }
        if (t->type != GGML_TYPE_F32) return true;
        double ss = 0.0;
        long cnt = 0;
        for (int64_t i3 = 0; i3 < t->ne[3]; i3++)
            for (int64_t i2 = 0; i2 < t->ne[2]; i2++)
                for (int64_t i1 = 0; i1 < t->ne[1]; i1++)
                    for (int64_t i0 = 0; i0 < t->ne[0]; i0++)
                    {
                        const double v = *(const float *) (d + i0 * t->nb[0] + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3]);
                        ss += v * v;
                        cnt++;
                    }
        auto & e = s->sumsq[t->name];
        e.first += ss;
        e.second += cnt;
        return true;
    }

    // MSA block-choice margins gathered by llama.cpp's eval callback (msa mode). `msa_bs-L` is
    // layer L's max-pooled index score of every position block, per index head and query,
    // before the own block is forced and the top-k taken; the prefill path (several queries per
    // ubatch, in token chunks) and the decode path (one query) both name it so. Queries arrive
    // in position order, one sequence, so a per-layer counter gives each row its position.
    struct msa_decision
    {
        int layer, head, pos;
        double margin, rel;   // last kept minus first dropped past-block score; margin / max |score|
    };

    struct msa_stats
    {
        llama_api * api = nullptr;
        int blk = 0, topk = 0, local = 0;
        std::map<int, long> next_pos;      // layer -> position of the next query row
        std::map<int, long> no_choice;     // layer -> (query, head) rows that kept every past block
        long nonfinite = 0;                // decisions whose competing scores were not all finite
        std::vector<msa_decision> d;
        std::vector<uint8_t> buf;
        std::vector<std::pair<double, int>> row;
        std::ofstream sel;                 // optional: one line per decision with the kept blocks
    };

    bool msa_cb(struct ggml_tensor * t, bool ask, void * user)
    {
        const bool mine = std::strncmp(t->name, "msa_bs-", 7) == 0 && std::strstr(t->name, " (") == nullptr;
        if (ask) return mine;
        if (!mine || t->type != GGML_TYPE_F32) return true;
        auto * s = (msa_stats *) user;
        const int il = std::atoi(t->name + 7);
        const size_t n = s->api->nbytes(t);
        s->buf.resize(n);
        s->api->tensor_get(t, s->buf.data(), 0, n);
        const uint8_t * d = s->buf.data();
        const int64_t nblk = t->ne[0], heads = t->ne[1], rows = t->ne[2] * t->ne[3];
        long & pos = s->next_pos[il];
        for (int64_t r = 0; r < rows; r++, pos++)
        {
            const int64_t i2 = r % t->ne[2], i3 = r / t->ne[2];
            const long own = pos / s->blk;
            // forced: the own block and the local - 1 before it; the rest compete for the others
            const long forced = std::min<long>(s->local, own + 1);
            const long past = own + 1 - forced;            // competing blocks [0, past)
            const long kfree = (long) s->topk - forced;
            for (int64_t h = 0; h < heads; h++)
            {
                if (kfree <= 0 || past <= kfree) { s->no_choice[il]++; continue; }
                s->row.clear();
                double amax = 0.0;
                for (long b = 0; b < past && b < nblk; b++)
                {
                    const double v = *(const float *) (d + b * t->nb[0] + h * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3]);
                    s->row.emplace_back(v, (int) b);
                    if (std::isfinite(v)) amax = std::max(amax, std::fabs(v));
                }
                // descending score, lower block first on equal scores (ggml_top_k's order is unspecified there)
                std::sort(s->row.begin(), s->row.end(), [](const std::pair<double, int> & a, const std::pair<double, int> & b) {
                    return a.first != b.first ? a.first > b.first : a.second < b.second;
                });
                const double margin = s->row[(size_t) kfree - 1].first - s->row[(size_t) kfree].first;
                // a past block is fully visible, so a non-finite score means the row's position is wrong
                if (!std::isfinite(margin)) { s->nonfinite++; continue; }
                s->d.push_back({ il, (int) h, (int) pos, margin, amax > 0 ? margin / amax : INFINITY });
                if (s->sel.is_open())
                {
                    std::vector<int> kept;
                    for (long l = 0; l < forced; l++) kept.push_back((int) (own - l));
                    for (long k = 0; k < kfree; k++) kept.push_back(s->row[(size_t) k].second);
                    std::sort(kept.begin(), kept.end());
                    s->sel << il << '\t' << pos << '\t' << h << '\t' << margin << '\t' << (amax > 0 ? margin / amax : INFINITY) << '\t';
                    for (size_t k = 0; k < kept.size(); k++) s->sel << (k ? "," : "") << kept[k];
                    s->sel << '\n';
                }
            }
        }
        return true;
    }

    // One sequence through llama_decode; returns the logits of every position through `sink`.
    // `on_load` sees the model before the context exists (msa mode reads its MSA geometry there).
    template <typename Sink>
    void llama_forward(llama_api & L, const char * model_path, const std::vector<int32_t> & toks, int n_gpu, int ubatch,
                       ggml_backend_sched_eval_callback cb, void * cb_user, Sink sink, double & load_s, double & eval_s,
                       const std::function<void(llama_model *)> & on_load = nullptr)
    {
        const auto t0 = clk::now();
        auto mp = L.model_default_params();
        mp.n_gpu_layers = n_gpu > 0 ? 999 : 0;
        mp.split_mode = LLAMA_SPLIT_MODE_LAYER;
        llama_model * model = L.model_load(model_path, mp);
        if (!model) throw std::runtime_error("llama model load failed");
        if (on_load) on_load(model);
        const int n = (int) toks.size();
        auto cp = L.context_default_params();
        cp.n_ctx = (uint32_t) ((n + 255) / 256 * 256);
        cp.n_batch = cp.n_ctx;
        cp.n_ubatch = (uint32_t) ubatch;
        cp.n_seq_max = 1;   // one stream: MSA runs (it falls back to dense on a unified multi-sequence cache)
        const char * dense = std::getenv("MM3_LLAMA_DENSE");
        const bool fa = !(dense && std::strcmp(dense, "1") == 0);
        if (!fa) std::fprintf(stderr, "MM3_LLAMA_DENSE=1: flash attention off, llama.cpp runs DENSE attention, not MSA\n");
        cp.flash_attn_type = fa ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;   // MSA needs FA
        cp.type_k = GGML_TYPE_F16;
        cp.type_v = GGML_TYPE_F16;
        if (cb)
        {
            cp.cb_eval = cb;
            cp.cb_eval_user_data = cb_user;
        }
        llama_context * ctx = L.init_from_model(model, cp);
        if (!ctx) throw std::runtime_error("llama context init failed");
        const int n_vocab = L.vocab_n_tokens(L.model_get_vocab(model));
        const auto t1 = clk::now();
        llama_batch batch = L.batch_init(n, 0, 1);
        batch.n_tokens = n;
        for (int i = 0; i < n; i++)
        {
            batch.token[i] = toks[(size_t) i];
            batch.pos[i] = i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = 1;
        }
        if (L.decode(ctx, batch) != 0) throw std::runtime_error("llama_decode failed");
        for (int i = 0; i < n; i++) sink(L.get_logits_ith(ctx, i), n_vocab);
        const auto t2 = clk::now();
        L.batch_free(batch);
        L.free_ctx(ctx);
        L.model_free(model);
        load_s = seconds(t0, t1);
        eval_s = seconds(t1, t2);
    }

    int run_llama(int argc, char ** argv)
    {
        if (argc < 6 || argc > 8) throw std::runtime_error("llama LIBLLAMA MODEL.gguf TOKENS.bin OUT.f32 [N_GPU] [UBATCH]");
        const std::vector<int32_t> toks = read_tokens(argv[4]);
        const int n_gpu = argc >= 7 ? std::atoi(argv[6]) : 1;
        const int ubatch = argc >= 8 ? std::max(1, std::atoi(argv[7])) : 512;
        llama_api L(argv[2]);
        std::unique_ptr<logits_writer> out;
        double load_s = 0, eval_s = 0;
        llama_forward(L, argv[3], toks, n_gpu, ubatch, nullptr, nullptr, [&](const float * row, int vocab) {
            if (!out) out.reset(new logits_writer(argv[5], (uint32_t) vocab, (uint32_t) toks.size(), fnv1a(toks), (uint32_t) ubatch, kLlama, 0));
            out->row(row, (size_t) vocab);
        }, load_s, eval_s);
        const char * dense = std::getenv("MM3_LLAMA_DENSE");
        std::printf("{\"mode\":\"llama\",\"tokens\":%zu,\"ubatch\":%d,\"n_gpu\":%d,\"attention\":\"%s\",\"load_s\":%.2f,\"eval_s\":%.2f}\n",
                    toks.size(), ubatch, n_gpu, dense && std::strcmp(dense, "1") == 0 ? "dense" : "msa", load_s, eval_s);
        return 0;
    }

    int run_stats(int argc, char ** argv)
    {
        if (argc < 5 || argc > 7) throw std::runtime_error("stats LIBLLAMA MODEL.gguf TOKENS.bin [N_GPU] [UBATCH]");
        const std::vector<int32_t> toks = read_tokens(argv[4]);
        const int n_gpu = argc >= 6 ? std::atoi(argv[5]) : 1;
        const int ubatch = argc >= 7 ? std::max(1, std::atoi(argv[6])) : 512;
        llama_api L(argv[2]);
        layer_stats s;
        s.api = &L;
        double load_s = 0, eval_s = 0, lsum = 0, lsq = 0;
        long lcnt = 0;
        llama_forward(L, argv[3], toks, n_gpu, ubatch, stats_cb, &s, [&](const float * row, int vocab) {
            for (int k = 0; k < vocab; k++) { lsum += row[k]; lsq += (double) row[k] * row[k]; }
            lcnt += vocab;
        }, load_s, eval_s);
        std::string rms;
        for (auto & kv : s.sumsq)
        {
            char text[160];
            std::snprintf(text, sizeof(text), "%s\"%s\":%.5g", rms.empty() ? "" : ",", kv.first.c_str(),
                          std::sqrt(kv.second.first / (double) std::max(1L, kv.second.second)));
            rms += text;
        }
        std::string routing;
        for (auto & kv : s.experts)
        {
            long total = 0, top = 0;
            int used = 0;
            for (long c : kv.second) { total += c; top = std::max(top, c); used += c > 0 ? 1 : 0; }
            double ent = 0.0;
            for (long c : kv.second)
                if (c > 0) { const double p = (double) c / (double) total; ent -= p * std::log(p); }
            const double max_ent = kv.second.size() > 1 ? std::log((double) kv.second.size()) : 1.0;
            std::string hist;
            for (long c : kv.second) hist += (hist.empty() ? "" : ",") + std::to_string(c);
            char text[256];
            std::snprintf(text, sizeof(text), "%s{\"layer\":%d,\"picks\":%ld,\"experts_used\":%d,\"entropy_norm\":%.4f,"
                          "\"top_share\":%.4f,\"hist\":[", routing.empty() ? "" : ",", kv.first, total, used,
                          ent / max_ent, (double) top / (double) total);
            routing += text + hist + "]}";
        }
        const double mean = lsum / (double) lcnt;
        std::printf("{\"mode\":\"stats\",\"tokens\":%zu,\"ubatch\":%d,\"logit_mean\":%.5g,\"logit_std\":%.5g,\"rms\":{%s},"
                    "\"routing\":[%s]}\n", toks.size(), ubatch, mean, std::sqrt(std::max(0.0, lsq / (double) lcnt - mean * mean)),
                    rms.c_str(), routing.c_str());
        return 0;
    }

    int meta_int(llama_api & L, llama_model * model, const char * key)
    {
        char buf[64] = {};
        if (L.meta_val_str(model, key, buf, sizeof(buf)) < 0) throw std::runtime_error(std::string("the model has no ") + key);
        return std::atoi(buf);
    }

    int run_msa(int argc, char ** argv)
    {
        if (argc < 5 || argc > 8) throw std::runtime_error("msa LIBLLAMA MODEL.gguf TOKENS.bin [N_GPU] [UBATCH] [SEL.tsv]");
        const std::vector<int32_t> toks = read_tokens(argv[4]);
        const int n_gpu = argc >= 6 ? std::atoi(argv[5]) : 1;
        const int ubatch = argc >= 7 ? std::max(1, std::atoi(argv[6])) : 512;
        if (std::getenv("MM3_LLAMA_DENSE")) throw std::runtime_error("msa mode needs llama.cpp's MSA; unset MM3_LLAMA_DENSE");
        llama_api L(argv[2]);
        msa_stats s;
        s.api = &L;
        if (argc == 8)
        {
            s.sel.open(argv[7], std::ios::trunc);
            if (!s.sel) throw std::runtime_error(std::string("cannot write ") + argv[7]);
            s.sel << "layer\tpos\thead\tmargin\trel_margin\tkept_blocks\n";
        }
        double load_s = 0, eval_s = 0;
        llama_forward(L, argv[3], toks, n_gpu, ubatch, msa_cb, &s, [](const float *, int) {}, load_s, eval_s,
                      [&](llama_model * model) {
                          s.blk = meta_int(L, model, "minimax-m3.attention.indexer.block_size");
                          s.topk = meta_int(L, model, "minimax-m3.attention.indexer.top_k");
                          s.local = meta_int(L, model, "minimax-m3.attention.indexer.local_blocks");
                          if (s.blk < 1 || s.topk < 1 || s.local < 0) throw std::runtime_error("bad MSA geometry in the model");
                      });
        if (s.next_pos.empty()) throw std::runtime_error("llama.cpp produced no msa_bs tensor (MSA off? flash attention off?)");
        for (auto & kv : s.next_pos)
            if (kv.second != (long) toks.size())
                throw std::runtime_error("layer " + std::to_string(kv.first) + " scored " + std::to_string(kv.second) +
                                         " queries, not " + std::to_string(toks.size()));
        // per layer: decisions, smallest relative and absolute margin, smallest relative margin per head
        struct agg { long n = 0; double rel = INFINITY, abs = INFINITY; std::map<int, double> head; };
        std::map<int, agg> per;
        std::vector<double> rels;
        long lt[3] = {};
        const double thr[3] = { 1e-3, 5e-3, 1e-2 };
        for (const auto & x : s.d)
        {
            agg & a = per[x.layer];
            a.n++;
            a.rel = std::min(a.rel, x.rel);
            a.abs = std::min(a.abs, x.margin);
            auto it = a.head.find(x.head);
            a.head[x.head] = it == a.head.end() ? x.rel : std::min(it->second, x.rel);
            rels.push_back(x.rel);
            for (int k = 0; k < 3; k++) lt[k] += x.rel < thr[k] ? 1 : 0;
        }
        std::sort(rels.begin(), rels.end());
        auto pct = [&](double q) { return rels.empty() ? 0.0 : rels[(size_t) std::min<double>((double) rels.size() - 1, q * (double) (rels.size() - 1))]; };
        std::string layers;
        for (auto & kv : per)
        {
            std::string heads;
            for (auto & h : kv.second.head)
            {
                char t[48];
                std::snprintf(t, sizeof(t), "%s%.4g", heads.empty() ? "" : ",", h.second);
                heads += t;
            }
            char text[256];
            std::snprintf(text, sizeof(text), "%s{\"layer\":%d,\"decisions\":%ld,\"no_choice\":%ld,\"rel_min\":%.4g,\"abs_min\":%.4g,\"head_rel_min\":[",
                          layers.empty() ? "" : ",", kv.first, kv.second.n, s.no_choice[kv.first], kv.second.rel, kv.second.abs);
            layers += text + heads + "]}";
        }
        std::vector<msa_decision> worst = s.d;
        std::sort(worst.begin(), worst.end(), [](const msa_decision & a, const msa_decision & b) { return a.rel < b.rel; });
        std::string w;
        for (size_t i = 0; i < worst.size() && i < 5; i++)
        {
            char text[160];
            std::snprintf(text, sizeof(text), "%s{\"layer\":%d,\"head\":%d,\"pos\":%d,\"margin\":%.4g,\"rel\":%.4g}",
                          w.empty() ? "" : ",", worst[i].layer, worst[i].head, worst[i].pos, worst[i].margin, worst[i].rel);
            w += text;
        }
        std::printf("{\"mode\":\"msa\",\"tokens\":%zu,\"ubatch\":%d,\"block\":%d,\"top_k\":%d,\"local\":%d,\"decisions\":%zu,"
                    "\"nonfinite\":%ld,\"rel_min\":%.4g,\"rel_p001\":%.4g,\"rel_p01\":%.4g,\"rel_median\":%.4g,"
                    "\"rel_lt_1e-3\":%ld,\"rel_lt_5e-3\":%ld,\"rel_lt_1e-2\":%ld,\"eval_s\":%.2f,\"layers\":[%s],\"worst\":[%s]}\n",
                    toks.size(), ubatch, s.blk, s.topk, s.local, s.d.size(), s.nonfinite, pct(0.0), pct(0.001), pct(0.01),
                    pct(0.5), lt[0], lt[1], lt[2], eval_s, layers.c_str(), w.c_str());
        return s.nonfinite ? 1 : 0;
    }

    int run_tokens_text(int argc, char ** argv)
    {
        if (argc < 7 || argc > 8) throw std::runtime_error("tokens text LIBLLAMA MODEL.gguf TEXT.txt OUT.bin [MAX]");
        llama_api L(argv[3]);
        auto mp = L.model_default_params();
        mp.n_gpu_layers = 0;
        mp.vocab_only = true;
        llama_model * model = L.model_load(argv[4], mp);
        if (!model) throw std::runtime_error("vocab load failed");
        std::ifstream f(argv[5], std::ios::binary);
        if (!f) throw std::runtime_error(std::string("cannot open ") + argv[5]);
        const std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        std::vector<llama_token> toks(text.size() + 16);
        const int n = L.tokenize(L.model_get_vocab(model), text.data(), (int32_t) text.size(), toks.data(),
                                 (int32_t) toks.size(), /*add_special*/ false, /*parse_special*/ true);
        if (n < 0) throw std::runtime_error("tokenize failed");
        toks.resize((size_t) n);
        if (argc == 8) toks.resize(std::min(toks.size(), (size_t) std::max(0, std::atoi(argv[7]))));
        write_tokens(argv[6], std::vector<int32_t>(toks.begin(), toks.end()));
        std::printf("{\"mode\":\"tokens\",\"source\":\"text\",\"tokens\":%zu}\n", toks.size());
        L.model_free(model);
        return 0;
    }
}
#endif

namespace
{
    // TensorSharp's MiniMax-M3 executor (contract C1), resolved from libGgmlOps at run time.
    using load_tp_fn = void * (*)(const char *, int, int, int, int, int, const char *, int, const char *, int);
    using forward_fn = int (*)(void *, const int32_t *, int, float *, int);
    using handle_fn = int (*)(void *);
    using free_fn = void (*)(void *);
    using err_fn = const char * (*)();

    struct ts_api
    {
        void * lib = nullptr;
        load_tp_fn load = nullptr;
        forward_fn forward = nullptr;
        handle_fn vocab = nullptr, reset = nullptr, n_past = nullptr;
        free_fn release = nullptr;
        err_fn last_error = nullptr;

        explicit ts_api(const char * path)
        {
            lib = dlopen(path, RTLD_NOW | RTLD_LOCAL);
            if (!lib) throw std::runtime_error(std::string("dlopen libGgmlOps: ") + dlerror());
            load = (load_tp_fn) dlsym(lib, "TSGgml_MiniMaxM3LoadModelTp");
            forward = (forward_fn) dlsym(lib, "TSGgml_MiniMaxM3Forward");
            vocab = (handle_fn) dlsym(lib, "TSGgml_MiniMaxM3VocabSize");
            reset = (handle_fn) dlsym(lib, "TSGgml_MiniMaxM3ResetChecked");
            n_past = (handle_fn) dlsym(lib, "TSGgml_MiniMaxM3NPast");
            release = (free_fn) dlsym(lib, "TSGgml_MiniMaxM3Free");
            last_error = (err_fn) dlsym(lib, "TSGgml_GetLastError");
            if (!load || !forward || !vocab || !reset || !n_past || !release)
                throw std::runtime_error(std::string(path) + " does not export the TSGgml_MiniMaxM3* C ABI");
        }

        void * open(const char * model, int n_gpu, int n_ctx, int ubatch)
        {
            const char * tp = std::getenv("MM3_TP");
            const char * cpu_moe = std::getenv("MM3_CPU_MOE");
            const char * be = std::getenv("MM3_TS_BACKEND");
            void * m = load(model, n_gpu, n_ctx, ubatch, 0, cpu_moe ? std::atoi(cpu_moe) : 0, be && *be ? be : n_gpu > 0 ? "CUDA" : "CPU",
                            1, nullptr, tp ? std::atoi(tp) : 0);
            if (!m) throw std::runtime_error(std::string("TensorSharp load refused: ") + (last_error ? last_error() : "?"));
            return m;
        }
    };

    // ts and ts-decode: rows [0, prefill) from all-row Forward calls of `chunk` tokens, the
    // rest (ts-decode only) from one Forward call per token.
    int run_ts(int argc, char ** argv, bool decode)
    {
        if (argc < 7 || argc > 9)
            throw std::runtime_error(decode ? "ts-decode LIBGGMLOPS MODEL.gguf TOKENS.bin OUT.f32 N_GPU [UBATCH] [PREFILL]"
                                            : "ts LIBGGMLOPS MODEL.gguf TOKENS.bin OUT.f32 N_GPU [UBATCH] [CHUNK]");
        const std::vector<int32_t> toks = read_tokens(argv[4]);
        const int n = (int) toks.size();
        const int n_gpu = std::atoi(argv[6]);
        const int ubatch = argc >= 8 ? std::max(1, std::atoi(argv[7])) : 512;
        int chunk = ubatch, prefill = n;
        if (decode) prefill = argc >= 9 ? std::atoi(argv[8]) : n / 2;
        else if (argc >= 9) chunk = std::max(1, std::atoi(argv[8]));
        if (prefill < 1 || prefill > n) throw std::runtime_error("PREFILL must be in [1, tokens]");
        if (decode) chunk = prefill;   // the prefill is one call; the executor splits it into micro-batches

        ts_api T(argv[2]);
        const auto t0 = clk::now();
        void * m = T.open(argv[3], n_gpu, (n + 1 + 255) / 256 * 256, ubatch);
        const auto t1 = clk::now();
        const int vocab = T.vocab(m);
        if (vocab <= 0) throw std::runtime_error("TensorSharp reports no vocab");
        if (!T.reset(m)) throw std::runtime_error("TSGgml_MiniMaxM3ResetChecked failed");
        logits_writer out(argv[5], (uint32_t) vocab, (uint32_t) n, fnv1a(toks), (uint32_t) ubatch, decode ? kTsDecode : kTs,
                          decode ? (uint32_t) prefill : 0);
        std::vector<float> logits;
        double prefill_s = 0;
        for (int done = 0; done < prefill;)
        {
            const int k = std::min(chunk, prefill - done);
            logits.resize((size_t) k * vocab);
            if (!T.forward(m, toks.data() + done, k, logits.data(), 1))
                throw std::runtime_error("TSGgml_MiniMaxM3Forward failed at position " + std::to_string(done) +
                                         (T.last_error ? std::string(": ") + T.last_error() : std::string()));
            for (int r = 0; r < k; r++) out.row(logits.data() + (size_t) r * vocab, (size_t) vocab);
            done += k;
        }
        const auto t2 = clk::now();
        prefill_s = seconds(t1, t2);
        logits.resize((size_t) vocab);
        for (int i = prefill; i < n; i++)
        {
            if (!T.forward(m, toks.data() + i, 1, logits.data(), 0))
                throw std::runtime_error("TSGgml_MiniMaxM3Forward (decode) failed at position " + std::to_string(i));
            out.row(logits.data(), (size_t) vocab);
        }
        const auto t3 = clk::now();
        const int past = T.n_past(m);
        if (past != n) throw std::runtime_error("TensorSharp n_past " + std::to_string(past) + " != " + std::to_string(n));
        T.release(m);
        const char * tp = std::getenv("MM3_TP");
        std::printf("{\"mode\":\"%s\",\"tokens\":%d,\"ubatch\":%d,\"chunk\":%d,\"prefill\":%d,\"n_gpu\":%d,\"tp\":%d,"
                    "\"load_s\":%.2f,\"prefill_s\":%.2f,\"decode_s\":%.2f}\n", decode ? "ts-decode" : "ts", n, ubatch, chunk,
                    prefill, n_gpu, tp ? std::atoi(tp) : 0, seconds(t0, t1), prefill_s, seconds(t2, t3));
        return 0;
    }

    int run_tokens(int argc, char ** argv)
    {
        if (argc >= 3 && std::strcmp(argv[2], "random") == 0)
        {
            if (argc < 6 || argc > 7) throw std::runtime_error("tokens random N SEED OUT.bin [HI]");
            const long n = std::atol(argv[3]);
            uint64_t s = std::strtoull(argv[4], nullptr, 10);
            const uint64_t hi = argc == 7 ? std::strtoull(argv[6], nullptr, 10) : 200000ull;
            if (n < 1 || hi < 1) throw std::runtime_error("N and HI must be positive");
            std::vector<int32_t> t((size_t) n);
            for (auto & v : t) v = (int32_t) (splitmix64(s) % hi);
            write_tokens(argv[5], t);
            std::printf("{\"mode\":\"tokens\",\"source\":\"random\",\"tokens\":%ld,\"hi\":%" PRIu64 ",\"fnv1a\":%u}\n", n, hi, fnv1a(t));
            return 0;
        }
        if (argc >= 3 && std::strcmp(argv[2], "ids") == 0)
        {
            if (argc != 5) throw std::runtime_error("tokens ids IDS.txt OUT.bin");
            std::ifstream f(argv[3]);
            if (!f) throw std::runtime_error(std::string("cannot open ") + argv[3]);
            std::stringstream ss;
            ss << f.rdbuf();
            const std::vector<int> v = parse_ints(ss.str());
            if (v.empty()) throw std::runtime_error("no ids");
            const std::vector<int32_t> t(v.begin(), v.end());
            write_tokens(argv[4], t);
            std::printf("{\"mode\":\"tokens\",\"source\":\"ids\",\"tokens\":%zu,\"fnv1a\":%u}\n", t.size(), fnv1a(t));
            return 0;
        }
#if defined(MM3_WITH_LLAMA_H)
        if (argc >= 3 && std::strcmp(argv[2], "text") == 0) return run_tokens_text(argc, argv);
#endif
        throw std::runtime_error("tokens random|ids|text ... (see the file header)");
    }
}

int main(int argc, char ** argv)
{
    try
    {
        const std::string mode = argc >= 2 ? argv[1] : "";
        if (mode == "compare") return run_compare(argc, argv);
        if (mode == "greedy") return run_greedy(argc, argv);
        if (mode == "tokens") return run_tokens(argc, argv);
        if (mode == "ts") return run_ts(argc, argv, false);
        if (mode == "ts-decode") return run_ts(argc, argv, true);
#if defined(MM3_WITH_LLAMA_H)
        if (mode == "llama") return run_llama(argc, argv);
        if (mode == "stats") return run_stats(argc, argv);
        if (mode == "msa") return run_msa(argc, argv);
#else
        if (mode == "llama" || mode == "stats" || mode == "msa") throw std::runtime_error("built without -DMM3_WITH_LLAMA_H; rebuild with the llama.cpp headers");
#endif
        std::fprintf(stderr, "usage: minimax-m3-parity llama|ts|ts-decode|compare|greedy|stats|msa|tokens ... (see the file header)\n");
        return 2;
    }
    catch (const std::exception & e)
    {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
