// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Encodes the same patch grid with two DeepSeek V4.1 vision companions and
// compares the complete image spans: for example TensorSharp's own
// deepseek41.vision.gguf (eng/dsv41-prepare-vision.py, from the official
// checkpoint) against llama.cpp's deepseek41v mmproj (unsloth's mmproj-F16 or
// mmproj-BF16). Patches are seeded random BF16 values in [-1, 1], the range the
// image processor produces, so the comparison covers every encoder weight.
//
// Build (loads libGgmlOps at run time):
//   g++ -std=c++17 -O2 dsv41-vision-companion-diff.cpp -ldl -o dsv41-vision-companion-diff
// Usage:
//   GGMLOPS=/path/libGgmlOps.so dsv41-vision-companion-diff A.gguf B.gguf [backend=CUDA] [rows=32] [cols=48]
// Prints one JSON line: span rows, identical floats, max |a-b|, relative L2.
#include <dlfcn.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using load_fn = void * (*)(const char *, const char *, int, int);
using free_fn = void (*)(void *);
using info_fn = int (*)(void *, int32_t *, int);
using encode_fn = int (*)(void *, const float *, int, int, float *, int);
using error_fn = const char * (*)();

static float bf16_round(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    bits = (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000u;
    std::memcpy(&value, &bits, sizeof(bits));
    return value;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: GGMLOPS=libGgmlOps.so %s A.gguf B.gguf [backend=CUDA] [rows=32] [cols=48]\n", argv[0]);
        return 2;
    }
    const char * library = std::getenv("GGMLOPS") ? std::getenv("GGMLOPS") : "libGgmlOps.so";
    void * lib = dlopen(library, RTLD_NOW | RTLD_LOCAL);
    if (!lib) { std::fprintf(stderr, "cannot load %s: %s\n", library, dlerror()); return 2; }
    auto load = reinterpret_cast<load_fn>(dlsym(lib, "TSGgml_Dsv41VisionLoad"));
    auto release = reinterpret_cast<free_fn>(dlsym(lib, "TSGgml_Dsv41VisionFree"));
    auto info = reinterpret_cast<info_fn>(dlsym(lib, "TSGgml_Dsv41VisionInfo"));
    auto encode = reinterpret_cast<encode_fn>(dlsym(lib, "TSGgml_Dsv41VisionEncode"));
    auto last_error = reinterpret_cast<error_fn>(dlsym(lib, "TSGgml_GetLastError"));
    if (!load || !release || !info || !encode) { std::fprintf(stderr, "missing V4.1 vision exports\n"); return 2; }
    const char * backend = argc > 3 ? argv[3] : "CUDA";
    const int rows = argc > 4 ? std::atoi(argv[4]) : 32, cols = argc > 5 ? std::atoi(argv[5]) : 48;

    std::vector<int32_t> geometry[2];
    std::vector<float> spans[2];
    for (int i = 0; i < 2; ++i) {
        void * handle = load(argv[1 + i], backend, 0, 16);
        if (!handle) {
            std::fprintf(stderr, "cannot load %s: %s\n", argv[1 + i], last_error ? last_error() : "(see stderr)");
            return 1;
        }
        geometry[i].assign(8, 0);
        if (info(handle, geometry[i].data(), 8) != 0) { std::fprintf(stderr, "info failed\n"); return 1; }
        const int patch = geometry[i][0], text_dim = geometry[i][2];
        std::mt19937 rng(20261022);
        std::uniform_real_distribution<float> pixel(-1.0f, 1.0f);
        std::vector<float> patches(size_t(rows) * cols * 3 * patch * patch);
        for (auto & v : patches) v = bf16_round(pixel(rng));
        spans[i].assign(size_t(geometry[i][3] + 2) * text_dim, 0.0f);
        const int n = encode(handle, patches.data(), rows, cols, spans[i].data(), int(spans[i].size()));
        if (n <= 0) { std::fprintf(stderr, "encode failed: %s\n", last_error ? last_error() : "(see stderr)"); return 1; }
        spans[i].resize(size_t(n) * text_dim);
        release(handle);
    }
    if (geometry[0] != geometry[1] || spans[0].size() != spans[1].size()) {
        std::fprintf(stderr, "the companions disagree on geometry\n");
        return 1;
    }
    size_t identical = 0;
    double max_abs = 0, diff2 = 0, ref2 = 0;
    for (size_t i = 0; i < spans[0].size(); ++i) {
        const double a = spans[0][i], b = spans[1][i];
        identical += std::memcmp(&spans[0][i], &spans[1][i], sizeof(float)) == 0;
        max_abs = std::fmax(max_abs, std::fabs(a - b));
        diff2 += (a - b) * (a - b);
        ref2 += a * a;
    }
    std::printf("{\"rows\":%zu,\"floats\":%zu,\"identical\":%zu,\"max_abs\":%.6g,\"rel_l2\":%.6g}\n",
                spans[0].size() / size_t(geometry[0][2]), spans[0].size(), identical, max_abs,
                ref2 > 0 ? std::sqrt(diff2 / ref2) : 0.0);
    return 0;
}
