// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// Checks and times the direct-CUDA engine's NVFP4 projections (TensorSharp.Backends.Cuda/native/
// kernels/tensorsharp_q35_kernels.cu) on Qwen3.8-27B's dense projection shapes: the q8_1 GEMV for
// n <= 8 activation rows and the block-scaled FP4 GEMM above. Weights are random bytes in ggml's
// block_nvfp4 layout, repacked into the engine's planes; one distinct matrix per layer and shape,
// so a pass streams them from device memory as a forward does rather than from L2. Each n reports
// one pass over every matrix (activation quantization included), its weight bandwidth and TFLOPS,
// the largest error relative to the output's magnitude against an exact reference over the same
// quantized activations ("kernel"), and against the unquantized activations ("quant", the
// rounding the path itself adds).
//
//   nvcc -O3 -std=c++17 -arch=sm_120a eng/validation/tsg-q35-matmul-bench.cu -o artifacts/tsg-q35-matmul-bench
//   artifacts/tsg-q35-matmul-bench [layers=6] [rows csv=1,2,4,8,16,32,64,128,512,2048]
#include "../../TensorSharp.Backends.Cuda/native/kernels/tensorsharp_q35_kernels.cu"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#define CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); std::exit(1); } } while (0)

namespace
{
    struct Shape { const char* name; int k; int m; };

    __device__ float e2m1_value(int c)
    {
        const float mag[8] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };
        return (c & 8) ? -mag[c & 7] : mag[c & 7];
    }

    // Value k of a plane row (ggml nibble order: byte j of a 16-value group holds values j, j + 8).
    __device__ float plane_value(const uint8_t* codes, const uint8_t* scales, int k)
    {
        const int g = k >> 4, w = k & 15;
        const uint8_t byte = codes[g * 8 + (w & 7)];
        const int c = w < 8 ? (byte & 0xF) : (byte >> 4);
        return e2m1_value(c) * 2.0f * ts_q35_ue4m3(scales[g]);
    }

    enum Mode { kGemv = 0, kGemm = 1, kExact = 2 };

    // y[t, r] = alpha * dot(W[r], activation[t]) in double. kGemv: the q8_1-rounded activations;
    // kGemm: the NVFP4-rounded activations (codes, scales, row scale); kExact: F32 x.
    __global__ void reference(Mode mode, const uint8_t* wc, const uint8_t* ws, int m, int k,
                              const float* x, const int8_t* xq, const float* xd,
                              const uint8_t* ac, const uint8_t* as, const float* ars,
                              int n, float alpha, float* y)
    {
        const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
        if (i >= (long long)n * m) return;
        const int r = (int)(i % m), t = (int)(i / m);
        const uint8_t* rc = wc + (long long)r * (k / 2);
        const uint8_t* rs = ws + (long long)r * (k / 16);
        double acc = 0.0;
        for (int kk = 0; kk < k; ++kk)
        {
            const double w = plane_value(rc, rs, kk);
            double a;
            if (mode == kGemv) a = (double)xq[(long long)t * k + kk] * xd[(long long)t * (k / 32) + kk / 32];
            else if (mode == kGemm) a = plane_value(ac + (long long)t * (k / 2), as + (long long)t * (k / 16), kk) * (double)ars[t];
            else a = x[(long long)t * k + kk];
            acc += w * a;
        }
        y[i] = (float)(alpha * acc);
    }

    void gemv(int n, const uint8_t* wc, const uint8_t* ws, int m, int k, const int8_t* xq, const float* xd,
              float alpha, float* y, cudaStream_t s)
    {
        const int warps = 4, rows = TS_Q35_GEMV_ROWS(n), per = warps * rows;
        const dim3 grid((m + per - 1) / per), block(32 * warps);
        switch (n)
        {
#define TS_BENCH_CASE(N) case N: ts_q35_gemv_nvfp4_##N<<<grid, block, 0, s>>>(wc, ws, m, k, xq, xd, alpha, y); break;
            TS_BENCH_CASE(1) TS_BENCH_CASE(2) TS_BENCH_CASE(3) TS_BENCH_CASE(4)
            TS_BENCH_CASE(5) TS_BENCH_CASE(6) TS_BENCH_CASE(7)
            default: ts_q35_gemv_nvfp4_8<<<grid, block, 0, s>>>(wc, ws, m, k, xq, xd, alpha, y); break;
#undef TS_BENCH_CASE
        }
    }

    int sms = 0;

    // Tile width and split count: verify-size n keeps one tile column and splits K until the
    // grid fills the device; prefill-size n uses wide tiles and no split.
    void gemm(int n, const uint8_t* wc, const uint8_t* ws, int m, int k, const uint8_t* ac, const uint8_t* as,
              const float* ars, float alpha, float* y, float* part, int* counters, cudaStream_t s)
    {
        // TS_BENCH_BN=64 forces the 64-row activation tile above 64 rows.
        static const int forced = std::getenv("TS_BENCH_BN") ? std::atoi(std::getenv("TS_BENCH_BN")) : 0;
        const int bn = n <= 32 ? 32 : n <= 64 || forced == 64 ? 64 : 128;
        const int tilesX = (m + TS_Q35_BM - 1) / TS_Q35_BM, tilesY = (n + bn - 1) / bn;
        int splits = 1;
        const int stages = k / TS_Q35_BK;
        while (tilesX * tilesY * splits < sms && splits * 2 <= stages / 2 && splits < 8)
            splits *= 2;
        if (bn == 32)
            ts_q35_gemm_w4a4_32<<<dim3(tilesX, tilesY, splits), 256, sizeof(TsQ35GemmSmem<32, 8, 1, 3>), s>>>(wc, ws, m, k, ac, as, ars, n, alpha, y, part, counters);
        else if (bn == 64)
            ts_q35_gemm_w4a4_64<<<dim3(tilesX, tilesY, splits), 256, sizeof(TsQ35GemmSmem<64, 4, 2, 3>), s>>>(wc, ws, m, k, ac, as, ars, n, alpha, y, part, counters);
        else   // activation tiles fastest
            ts_q35_gemm_w4a4_128<<<dim3(tilesY, tilesX, splits), 256, sizeof(TsQ35GemmSmem<128, 4, 2, 2>), s>>>(wc, ws, m, k, ac, as, ars, n, alpha, y, part, counters);
    }
}

int main(int argc, char** argv)
{
    const int layers = argc > 1 ? std::atoi(argv[1]) : 6;
    std::vector<int> rows_list = { 1, 2, 4, 8, 16, 32, 64, 128, 512, 2048 };
    if (argc > 2)
    {
        rows_list.clear();
        std::stringstream ss(argv[2]);
        std::string item;
        while (std::getline(ss, item, ',')) rows_list.push_back(std::atoi(item.c_str()));
    }
    CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    CHECK(cudaFuncSetAttribute(ts_q35_gemm_w4a4_32, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sizeof(TsQ35GemmSmem<32, 8, 1, 3>)));
    CHECK(cudaFuncSetAttribute(ts_q35_gemm_w4a4_64, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sizeof(TsQ35GemmSmem<64, 4, 2, 3>)));
    CHECK(cudaFuncSetAttribute(ts_q35_gemm_w4a4_128, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sizeof(TsQ35GemmSmem<128, 4, 2, 2>)));

    // One Qwen3.8-27B layer's large dense projections (GDN in-proj, gate+up, down, GDN out).
    const Shape layer[] = {
        { "gdn_in", 5120, 10240 + 6144 + 48 + 48 },
        { "gate_up", 5120, 2 * 17408 },
        { "down", 17408, 5120 },
        { "ssm_out", 6144, 5120 },
    };
    struct Mat { const Shape* s; uint8_t* codes; uint8_t* scales; };
    std::vector<Mat> mats;
    std::mt19937 rng(1234);
    double total_bytes = 0, total_flop_per_row = 0;
    for (int l = 0; l < layers; ++l)
        for (const Shape& s : layer)
        {
            const long long rb = (long long)s.k / 64 * 36;
            const size_t bytes = (size_t)rb * s.m;
            std::vector<uint8_t> host(bytes);
            for (size_t i = 0; i < bytes; ++i)
                host[i] = i % 36 < 4 ? (uint8_t)(0x30 + rng() % 0x10) : (uint8_t)rng();
            uint8_t *src = nullptr, *codes = nullptr, *scales = nullptr;
            CHECK(cudaMalloc(&src, bytes));
            CHECK(cudaMalloc(&codes, (size_t)s.m * s.k / 2));
            CHECK(cudaMalloc(&scales, (size_t)s.m * s.k / 16));
            CHECK(cudaMemcpy(src, host.data(), bytes, cudaMemcpyHostToDevice));
            const long long blocks = (long long)s.m * (s.k / 64);
            ts_q35_repack_nvfp4<<<(unsigned)((blocks + 255) / 256), 256>>>(src, rb, codes, scales, s.m, s.k, 0, 1, 0);
            CHECK(cudaDeviceSynchronize());
            CHECK(cudaFree(src));
            mats.push_back({ &s, codes, scales });
            total_bytes += (double)bytes;
            total_flop_per_row += 2.0 * s.k * s.m;
        }

    const int kmax = 17408, mmax = 2 * 17408, nmax = 2048;
    float *x = nullptr, *xd = nullptr, *y = nullptr, *yref = nullptr, *ars = nullptr, *part = nullptr;
    int8_t* xq = nullptr;
    uint8_t *ac = nullptr, *as = nullptr;
    int* counters = nullptr;
    CHECK(cudaMalloc(&x, sizeof(float) * (size_t)nmax * kmax));
    CHECK(cudaMalloc(&xq, (size_t)8 * kmax));
    CHECK(cudaMalloc(&xd, sizeof(float) * 8 * (kmax / 32)));
    CHECK(cudaMalloc(&ac, (size_t)nmax * kmax / 2));
    CHECK(cudaMalloc(&as, (size_t)nmax * kmax / 16));
    CHECK(cudaMalloc(&ars, sizeof(float) * nmax));
    CHECK(cudaMalloc(&y, sizeof(float) * (size_t)nmax * mmax));
    CHECK(cudaMalloc(&yref, sizeof(float) * (size_t)64 * mmax));
    CHECK(cudaMalloc(&part, sizeof(float) * (size_t)8 * 64 * mmax));
    CHECK(cudaMalloc(&counters, sizeof(int) * 65536));
    CHECK(cudaMemset(counters, 0, sizeof(int) * 65536));
    {
        std::vector<float> hx((size_t)nmax * kmax);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        for (auto& v : hx) v = nd(rng);
        CHECK(cudaMemcpy(x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice));
    }
    CHECK(cudaFuncSetAttribute(ts_q35_quantize_nvfp4_f32, cudaFuncAttributeMaxDynamicSharedMemorySize, kmax * 4));
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    cudaEvent_t e0, e1;
    CHECK(cudaEventCreate(&e0));
    CHECK(cudaEventCreate(&e1));
    const float alpha = 0.75f;

    auto run = [&](int n, const Mat& mt) {
        const int k = mt.s->k, m = mt.s->m;
        if (n <= 8)
        {
            const long long threads = (long long)n * (k / 32) * 32;
            ts_q35_quantize_q81_f32<<<(unsigned)((threads + 255) / 256), 256, 0, stream>>>(x, xq, xd, k, n);
            gemv(n, mt.codes, mt.scales, m, k, xq, xd, alpha, y, stream);
        }
        else
        {
            ts_q35_quantize_nvfp4_f32<<<n, 512, k * 4, stream>>>(x, ac, as, ars, k);
            gemm(n, mt.codes, mt.scales, m, k, ac, as, ars, alpha, y, part, counters, stream);
        }
    };

    std::printf("%d layers, %.2f GB of weights per pass, %d SMs\n", layers, total_bytes / 1e9, sms);
    std::printf("rows,ms_per_pass,GB_per_s,TFLOPS,max_rel_err_kernel,max_rel_err_quant\n");
    for (int n : rows_list)
    {
        double max_rel = 0.0, max_rel_q = 0.0;
        const int ncheck = n < 64 ? n : 64;   // check the first rows (the reference is slow)
        for (int i = 0; i < 4; ++i)
        {
            const Mat& mt = mats[i];
            const int k = mt.s->k, m = mt.s->m;
            run(n, mt);
            const Mode mode = n <= 8 ? kGemv : kGemm;
            for (int pass = 0; pass < 2; ++pass)
            {
                reference<<<(int)(((long long)ncheck * m + 255) / 256), 256, 0, stream>>>(pass == 0 ? mode : kExact,
                    mt.codes, mt.scales, m, k, x, xq, xd, ac, as, ars, ncheck, alpha, yref);
                CHECK(cudaStreamSynchronize(stream));
                std::vector<float> a((size_t)ncheck * m), b((size_t)ncheck * m);
                CHECK(cudaMemcpy(a.data(), y, a.size() * 4, cudaMemcpyDeviceToHost));
                CHECK(cudaMemcpy(b.data(), yref, b.size() * 4, cudaMemcpyDeviceToHost));
                double maxabs = 0.0;
                for (float v : b) maxabs = std::fmax(maxabs, std::fabs(v));
                double& target = pass == 0 ? max_rel : max_rel_q;
                for (size_t j = 0; j < a.size(); ++j)
                    target = std::fmax(target, std::fabs((double)a[j] - b[j]) / (maxabs > 0 ? maxabs : 1.0));
            }
        }
        auto pass = [&]() { for (const Mat& mt : mats) run(n, mt); };
        for (int w = 0; w < 2; ++w) pass();
        CHECK(cudaStreamSynchronize(stream));
        const int reps = n >= 512 ? 3 : 10;
        CHECK(cudaEventRecord(e0, stream));
        for (int r = 0; r < reps; ++r) pass();
        CHECK(cudaEventRecord(e1, stream));
        CHECK(cudaEventSynchronize(e1));
        float ms = 0.0f;
        CHECK(cudaEventElapsedTime(&ms, e0, e1));
        ms /= reps;
        std::printf("%d,%.3f,%.1f,%.1f,%.3g,%.3g\n", n, ms, total_bytes / (ms * 1e6),
                    total_flop_per_row * n / (ms * 1e9), max_rel, max_rel_q);
        // TS_BENCH_SHAPES=1: each shape on its own (TFLOPS), first layer's matrices.
        if (std::getenv("TS_BENCH_SHAPES"))
            for (int i = 0; i < 4; ++i)
            {
                const Mat& mt = mats[i];
                CHECK(cudaEventRecord(e0, stream));
                for (int r = 0; r < reps * 4; ++r) run(n, mt);
                CHECK(cudaEventRecord(e1, stream));
                CHECK(cudaEventSynchronize(e1));
                float sms_ = 0.0f;
                CHECK(cudaEventElapsedTime(&sms_, e0, e1));
                sms_ /= reps * 4;
                std::printf("  %-8s %8.3f ms %7.1f TFLOPS\n", mt.s->name, sms_, 2.0 * mt.s->k * mt.s->m * n / (sms_ * 1e9));
            }
        std::fflush(stdout);
    }
    return 0;
}
