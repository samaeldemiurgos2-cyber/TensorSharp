// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
// Exploratory benchmark only; does not change inference or upstream ggml.
// Compare the normal NVFP4 <=8-column MMVQ with a >=9-column padded MMQ,
// explicitly retaining Q8 activation quantization in BOTH paths. A graph
// slice discards the padded outputs. Reports full-output numeric differences.
//
// Build against a built TensorSharp native library and its unchanged ggml:
// c++ -O2 -std=c++17 -I ExternalProjects/ggml/include \
//   eng/validation/qwen-nvfp4-matmul-probe.cpp \
//   -L TensorSharp.GGML.Native/build -lGgmlOps \
//   -Wl,-rpath,$PWD/TensorSharp.GGML.Native/build -o artifacts/nvfp4-probe
// Run artifacts/nvfp4-probe > artifacts/nvfp4-probe.csv with no competing GPU
// work. These synthetic projections do not qualify model output or E2E speed.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cuda.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#include <utility>

struct Graph {
    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_cgraph* graph = nullptr;
    ggml_tensor *weights = nullptr, *input = nullptr, *output = nullptr;
    ~Graph() {
        if (buffer) ggml_backend_buffer_free(buffer);
        if (ctx) ggml_free(ctx);
    }
    bool create(ggml_backend_t backend, int k, int m, int n, bool padded,
                const std::vector<unsigned char>& raw, const std::vector<float>& x) {
        ctx = ggml_init({1024*1024,nullptr,true});
        if (!ctx) return false;
        weights = ggml_new_tensor_2d(ctx,GGML_TYPE_NVFP4,k,m);
        input = ggml_new_tensor_2d(ctx,GGML_TYPE_F32,k,n);
        ggml_set_input(input);
        ggml_tensor* columns = padded ? ggml_pad(ctx,input,0,9-n,0,0) : input;
        ggml_tensor* product = ggml_mul_mat(ctx,weights,columns);
        ggml_prec_set_src(product,GGML_PREC_Q8,1);
        output = padded ? ggml_cont(ctx,ggml_view_2d(ctx,product,m,n,product->nb[1],0)) : product;
        ggml_set_output(output);
        graph = ggml_new_graph_custom(ctx,64,false);
        ggml_build_forward_expand(graph,output);
        buffer = ggml_backend_alloc_ctx_tensors(ctx,backend);
        if (!buffer) return false;
        ggml_backend_buffer_set_usage(buffer,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        ggml_backend_tensor_set(weights,raw.data(),0,raw.size());
        ggml_backend_tensor_set(input,x.data(),0,x.size()*sizeof(float));
        return true;
    }
    double measure(ggml_backend_t backend) {
        for (int i=0;i<8;++i) if (ggml_backend_graph_compute(backend,graph)!=GGML_STATUS_SUCCESS) return -1;
        ggml_backend_synchronize(backend);
        auto start=std::chrono::steady_clock::now();
        constexpr int iterations=100;
        for (int i=0;i<iterations;++i) if (ggml_backend_graph_compute(backend,graph)!=GGML_STATUS_SUCCESS) return -1;
        ggml_backend_synchronize(backend);
        return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()/iterations;
    }
};

int main() {
    ggml_backend_t backend=ggml_backend_cuda_init(0);
    if (!backend) return 77;
    std::puts("k,m,n,mmvq_ms,padded_mmq_q8_ms,speedup,max_abs_diff,rms_diff");
    for (const auto shape : {std::pair<int,int>{5120,17408},{17408,5120},{5120,10240}}) {
        const int k=shape.first,m=shape.second;
        std::vector<unsigned char> raw(size_t(k)*m/64*36);
        uint32_t rng=1234567;
        for (size_t block=0;block<raw.size()/36;++block) {
            for (int i=0;i<36;++i) {
                rng=rng*1664525u+1013904223u;
                // NVFP4 block: four UE4M3 subscales then 32 E2M1 bytes.
                raw[block*36+i]=i<4 ? 0x20+(rng>>30) : (rng>>24);
            }
        }
        for (int n : {1,2,4,8}) {
            std::vector<float> x(k*n),a(m*n),b(m*n);
            for (float& f:x) { rng=rng*1664525u+1013904223u; f=(int(rng>>16)-32768)/32768.f; }
            Graph normal,padded;
            if (!normal.create(backend,k,m,n,false,raw,x) || !padded.create(backend,k,m,n,true,raw,x)) return 1;
            double t0=normal.measure(backend),t1=padded.measure(backend);
            ggml_backend_tensor_get(normal.output,a.data(),0,a.size()*sizeof(float));
            ggml_backend_tensor_get(padded.output,b.data(),0,b.size()*sizeof(float));
            double square=0;float maxerr=0;
            for (size_t i=0;i<a.size();++i) {
                if (!std::isfinite(a[i])||!std::isfinite(b[i])) return 1;
                float error=std::fabs(a[i]-b[i]);maxerr=std::max(maxerr,error);square+=double(error)*error;
            }
            std::printf("%d,%d,%d,%.6f,%.6f,%.4f,%.9g,%.9g\n",k,m,n,t0,t1,t0/t1,maxerr,std::sqrt(square/a.size()));
            if (t0<0||t1<0) return 1;
        }
    }
    ggml_backend_free(backend);
    return 0;
}
