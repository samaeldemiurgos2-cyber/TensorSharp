// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// MiniMax-M3 native-executor throughput in ONE model load (prefill of a prompt
// from an empty sequence, then one-token decode steps on top of it), so a
// tensor-parallel and a layer-split arm each cost a single load of the 232 GiB
// checkpoint. llama-bench's pp N / tg M @ dN shape, through the C ABI the
// managed host uses (TSGgml_MiniMaxM3*, contract C1), logits downloaded every
// step as the host does.
//
//   minimax-m3-tp-bench LIBGGMLOPS MODEL.gguf N_GPU TP UBATCH CTX DEPTHS DECODE REPEATS [SEED]
//     N_GPU    GPUs of a layer split (ignored when TP > 1)
//     TP       tensor-parallel degree (0 or 1: the layer split)
//     CTX      context; negative asks for -CTX without making it a hard limit (MAX_CONTEXT
//              unset), so the executor caps it to what fits
//     DEPTHS   comma-separated prompt lengths, e.g. 1,2048,8192,16384: each is prefilled into
//              a reset sequence (timed: prefill tok/s), then DECODE tokens are decoded one per
//              call (timed: decode tok/s at that depth)
//     REPEATS  passes over DEPTHS; a first untimed pass of every shape warms the graphs
// Tokens are uniform ids in [1000, 199000) (no special token). Prints one JSON line per
// (pass, depth) and one with the load. Build: g++ -O2 -std=c++17 minimax-m3-tp-bench.cpp -ldl
#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    using clk = std::chrono::steady_clock;
    double seconds(clk::time_point a, clk::time_point b) { return std::chrono::duration<double>(b - a).count(); }

    using load_tp_fn = void * (*)(const char *, int, int, int, int, int, const char *, int, const char *, int);
    using forward_fn = int (*)(void *, const int32_t *, int, float *, int);
    using handle_fn = int (*)(void *);
    using free_fn = void (*)(void *);
    using stats_fn = int (*)(void *, int64_t *, int);
    using err_fn = const char * (*)();

    std::vector<int> parse_ints(const char * s)
    {
        std::vector<int> v;
        for (const char * p = s; *p;)
        {
            char * end = nullptr;
            v.push_back((int) std::strtol(p, &end, 10));
            if (end == p) throw std::runtime_error(std::string("bad list: ") + s);
            p = *end == ',' ? end + 1 : end;
        }
        return v;
    }

    int run(int argc, char ** argv)
    {
        if (argc < 10)
            throw std::runtime_error("minimax-m3-tp-bench LIBGGMLOPS MODEL.gguf N_GPU TP UBATCH CTX DEPTHS DECODE REPEATS [SEED]");
        void * lib = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
        if (!lib) throw std::runtime_error(std::string("dlopen: ") + dlerror());
        auto load = (load_tp_fn) dlsym(lib, "TSGgml_MiniMaxM3LoadModelTp");
        auto forward = (forward_fn) dlsym(lib, "TSGgml_MiniMaxM3Forward");
        auto vocab_of = (handle_fn) dlsym(lib, "TSGgml_MiniMaxM3VocabSize");
        auto ctx_of = (handle_fn) dlsym(lib, "TSGgml_MiniMaxM3CtxSize");
        auto reset = (handle_fn) dlsym(lib, "TSGgml_MiniMaxM3ResetChecked");
        auto release = (free_fn) dlsym(lib, "TSGgml_MiniMaxM3Free");
        auto stats = (stats_fn) dlsym(lib, "TSGgml_MiniMaxM3GetStats");
        auto last_error = (err_fn) dlsym(lib, "TSGgml_GetLastError");
        if (!load || !forward || !vocab_of || !ctx_of || !reset || !release || !stats)
            throw std::runtime_error("the library does not export the TSGgml_MiniMaxM3* C ABI");

        const int n_gpu = std::atoi(argv[3]), tp = std::atoi(argv[4]), ubatch = std::atoi(argv[5]), ctx_arg = std::atoi(argv[6]);
        const int n_ctx = std::abs(ctx_arg), hard = ctx_arg > 0 ? 1 : 0;
        const std::vector<int> depths = parse_ints(argv[7]);
        const int n_decode = std::atoi(argv[8]), repeats = std::atoi(argv[9]);
        uint64_t s = argc > 10 ? std::strtoull(argv[10], nullptr, 10) : 7;

        const auto t0 = clk::now();
        void * m = load(argv[2], n_gpu, n_ctx, ubatch, 0, 0, "CUDA", hard, nullptr, tp);
        if (!m) throw std::runtime_error(std::string("load refused: ") + (last_error ? last_error() : "?"));
        const double load_s = seconds(t0, clk::now());
        const int vocab = vocab_of(m);
        int64_t st[16] = {};
        stats(m, st, 16);
        std::printf("{\"mode\":\"load\",\"tp\":%d,\"n_gpu\":%d,\"ubatch\":%d,\"ctx_asked\":%d,\"hard\":%d,\"ctx\":%d,\"load_s\":%.1f,\"tp_reported\":%" PRId64
                    ",\"gpus_reported\":%" PRId64 ",\"pipeline_ubatch\":%" PRId64 ",\"attention_mode\":%" PRId64 "}\n",
                    tp, n_gpu, ubatch, n_ctx, hard, ctx_of(m), load_s, st[11], st[4], st[9], st[12]);
        std::fflush(stdout);

        int max_depth = 1;
        for (int d : depths) max_depth = std::max(max_depth, d);
        std::vector<int32_t> toks((size_t) (max_depth + n_decode));
        for (auto & t : toks)
        {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            t = (int32_t) (1000 + (s >> 33) % 198000);
        }
        std::vector<float> logits((size_t) vocab);
        for (int pass = -1; pass < repeats; pass++)   // pass -1: warm-up, not printed
            for (int d : depths)
            {
                if (d < 1 || d + n_decode > ctx_of(m)) throw std::runtime_error("depth + decode exceeds the context");
                if (!reset(m)) throw std::runtime_error("reset failed");
                const auto a = clk::now();
                if (!forward(m, toks.data(), d, logits.data(), 0)) throw std::runtime_error("prefill failed");
                const auto b = clk::now();
                const int nd = pass < 0 ? std::min(n_decode, 4) : n_decode;
                for (int i = 0; i < nd; i++)
                    if (!forward(m, toks.data() + d + i, 1, logits.data(), 0)) throw std::runtime_error("decode failed");
                const auto c = clk::now();
                if (pass < 0) continue;
                std::printf("{\"mode\":\"bench\",\"tp\":%d,\"n_gpu\":%d,\"ubatch\":%d,\"pass\":%d,\"depth\":%d,\"prefill_s\":%.3f,"
                            "\"prefill_tok_s\":%.1f,\"decode_n\":%d,\"decode_s\":%.3f,\"decode_tok_s\":%.2f}\n",
                            tp, n_gpu, ubatch, pass, d, seconds(a, b), d / seconds(a, b), nd, seconds(b, c), nd / seconds(b, c));
                std::fflush(stdout);
            }
        release(m);
        return 0;
    }
}

int main(int argc, char ** argv)
{
    try { return run(argc, argv); }
    catch (const std::exception & e) { std::fprintf(stderr, "minimax-m3-tp-bench: %s\n", e.what()); return 1; }
}
