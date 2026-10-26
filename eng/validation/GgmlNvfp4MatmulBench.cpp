// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

// Times an unchanged ggml's quantized matmul on weights that do NOT fit in L2:
// a graph multiplies one activation batch by several distinct weight matrices
// (Qwen3.8-27B's packed GDN input, gate+up and down shapes), so every weight is
// streamed from device memory once per graph, as in a decode or verify forward.
// test-backend-ops reuses one L2-resident matrix and overstates small-batch speed.
//
// Build against an unchanged ggml build tree, e.g.
//   g++ -O2 -std=c++17 GgmlNvfp4MatmulBench.cpp -I<ggml>/include -L<build>/bin \
//       -lggml -lggml-base -lggml-cuda -Wl,-rpath,<build>/bin -o nvfp4-mm-bench
// Usage: nvfp4-mm-bench [type=nvfp4] [rows csv=1,2,4,8,12,16,24,32,64,128]

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sstream>
#include <string>
#include <vector>

struct Shape { const char* name; int64_t k; int64_t m; };

int main(int argc, char** argv)
{
    std::string type_name = argc > 1 ? argv[1] : "nvfp4";
    std::vector<int> rows_list = { 1, 2, 4, 8, 12, 16, 24, 32, 64, 128 };
    if (argc > 2)
    {
        rows_list.clear();
        std::stringstream ss(argv[2]);
        std::string item;
        while (std::getline(ss, item, ',')) rows_list.push_back(std::atoi(item.c_str()));
    }
    ggml_type wtype = GGML_TYPE_COUNT;
    for (int t = 0; t < GGML_TYPE_COUNT; t++)
        if (ggml_type_name((ggml_type)t) != nullptr && type_name == ggml_type_name((ggml_type)t)) wtype = (ggml_type)t;
    if (wtype == GGML_TYPE_COUNT) { std::fprintf(stderr, "unknown type %s\n", type_name.c_str()); return 1; }

    // One Qwen3.8-27B dense layer's large projections, repeated so the set exceeds L2.
    const Shape layer[] = {
        { "gdn_in",  5120, 10240 + 6144 + 48 + 48 },
        { "gate_up", 5120, 2 * 17408 },
        { "down",   17408, 5120 },
        { "ssm_out", 6144, 5120 },
    };
    const int repeats = 6;

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) { std::fprintf(stderr, "no CUDA backend\n"); return 1; }

    ggml_init_params wp = { ggml_tensor_overhead() * 64, nullptr, true };
    ggml_context* wctx = ggml_init(wp);
    std::vector<ggml_tensor*> weights;
    std::vector<const Shape*> shapes;
    for (int r = 0; r < repeats; r++)
        for (const Shape& s : layer)
        {
            weights.push_back(ggml_new_tensor_2d(wctx, wtype, s.k, s.m));
            shapes.push_back(&s);
        }
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(wctx, backend);
    std::size_t weight_bytes = 0;
    {
        std::mt19937 rng(1234);
        for (ggml_tensor* w : weights)
        {
            // Random bytes are valid NVFP4/Q blocks except for NaN-ish scales; clear the
            // top bit of every byte so UE4M3 scales stay finite.
            std::vector<uint8_t> bytes(ggml_nbytes(w));
            for (auto& b : bytes) b = static_cast<uint8_t>(rng() & 0x3f);
            ggml_backend_tensor_set(w, bytes.data(), 0, bytes.size());
            weight_bytes += bytes.size();
        }
    }
    std::printf("type=%s weights=%zu MiB in %zu matrices\n", type_name.c_str(), weight_bytes >> 20, weights.size());

    for (int rows : rows_list)
    {
        ggml_init_params gp = { ggml_tensor_overhead() * 256 + ggml_graph_overhead(), nullptr, true };
        ggml_context* gctx = ggml_init(gp);
        ggml_tensor* x5120 = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, 5120, rows);
        ggml_tensor* x17408 = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, 17408, rows);
        ggml_tensor* x6144 = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, 6144, rows);
        ggml_set_input(x5120); ggml_set_input(x17408); ggml_set_input(x6144);
        ggml_cgraph* graph = ggml_new_graph(gctx);
        for (std::size_t i = 0; i < weights.size(); i++)
        {
            ggml_tensor* x = shapes[i]->k == 5120 ? x5120 : shapes[i]->k == 17408 ? x17408 : x6144;
            ggml_tensor* y = ggml_mul_mat(gctx, weights[i], x);
            ggml_set_output(y);
            ggml_build_forward_expand(graph, y);
        }
        ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        ggml_gallocr_alloc_graph(galloc, graph);
        for (ggml_tensor* x : { x5120, x17408, x6144 })
        {
            std::vector<float> v(ggml_nelements(x));
            for (std::size_t i = 0; i < v.size(); i++) v[i] = 0.01f * static_cast<float>((i * 7919) % 200) - 1.0f;
            ggml_backend_tensor_set(x, v.data(), 0, ggml_nbytes(x));
        }
        for (int w = 0; w < 3; w++) ggml_backend_graph_compute(backend, graph);
        const int iters = 20;
        auto t0 = std::chrono::steady_clock::now();
        for (int it = 0; it < iters; it++) ggml_backend_graph_compute(backend, graph);
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
        double gbps = static_cast<double>(weight_bytes) / (ms * 1e-3) / 1e9;
        std::printf("rows=%4d  %8.3f ms/graph  %7.1f GB/s weight stream  %6.3f ms per GB\n",
            rows, ms, gbps, ms / (static_cast<double>(weight_bytes) / 1e9));
        ggml_gallocr_free(galloc);
        ggml_free(gctx);
    }
    ggml_backend_buffer_free(wbuf);
    ggml_free(wctx);
    ggml_backend_free(backend);
    return 0;
}
