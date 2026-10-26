// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// Checks and times the direct-CUDA engine's prefill attention (TensorSharp.Backends.Cuda/native/
// kernels/tensorsharp_q35_kernels.cu) on Qwen3.8-27B's shape (24 query heads, 4 KV heads, head
// width 256): ts_q35_attn_q8_f32 (F32 accumulation) and ts_q35_attn_prefill_f32 (F16 accumulation,
// pipelined), over the same
// paged q8_0 cache of a prompt chunk that follows a cached prefix. Both are measured against a
// double-precision CPU reference over the cache's exact q8_0 values for a sample of rows: largest
// error relative to the reference's largest magnitude, and relative L2 error.
//
//   nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a eng/validation/tsg-q35-attn-bench.cu -o artifacts/tsg-q35-attn-bench
//   artifacts/tsg-q35-attn-bench [prefix=17000] [chunk=2048] [sequences=1] [checked rows=96] [key splits=1]
#include "../../TensorSharp.Backends.Cuda/native/kernels/tensorsharp_q35_kernels.cu"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); std::exit(1); } } while (0)

namespace
{
    constexpr int D = TS_Q35_HD, NH = 24, NKV = 4, G = NH / NKV, B = TS_Q35_KV_BLOCK;

    // q8_0 row in the engine's layout: codes, then the 8 F16 scales.
    void Quantize(const float* x, uint8_t* row)
    {
        for (int j = 0; j < 8; ++j)
        {
            float amax = 0.0f;
            for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(x[32 * j + i]));
            const float d = amax / 127.0f, id = d != 0.0f ? 1.0f / d : 0.0f;
            for (int i = 0; i < 32; ++i) row[32 * j + i] = (uint8_t)(int8_t)(int)std::round(x[32 * j + i] * id);
            const half h = __float2half(d);
            std::memcpy(row + TS_Q35_Q8_SCALES + 2 * j, &h, 2);
        }
    }

    void Dequantize(const uint8_t* row, double* x)
    {
        for (int j = 0; j < 8; ++j)
        {
            half h;
            std::memcpy(&h, row + TS_Q35_Q8_SCALES + 2 * j, 2);
            const double d = __half2float(h);
            for (int i = 0; i < 32; ++i) x[32 * j + i] = d * (int8_t)row[32 * j + i];
        }
    }
}

int main(int argc, char** argv)
{
    const int prefix = argc > 1 ? std::atoi(argv[1]) : 17000;
    const int chunk = argc > 2 ? std::atoi(argv[2]) : 2048;
    const int nSeq = argc > 3 ? std::atoi(argv[3]) : 1;
    const int checked = argc > 4 ? std::atoi(argv[4]) : 96;
    const int splits = argc > 5 ? std::atoi(argv[5]) : 1;
    const int keys = prefix + chunk;
    const int maxBlocks = (keys + B - 1) / B;
    const long long headBytes = (long long)B * TS_Q35_Q8_ROW, blockBytes = 2LL * NKV * headBytes;
    const long long kOff = 0, vOff = NKV * headBytes;
    const float scale = 1.0f / std::sqrt((float)D);

    // Q and K with uneven channels (norm weights) and a few outliers; V plain.
    std::mt19937 rng(7);
    std::normal_distribution<float> n01(0.0f, 1.0f);
    std::lognormal_distribution<float> chan(0.0f, 0.6f);
    std::vector<float> qw(D), kw(D);
    for (int i = 0; i < D; ++i) { qw[i] = chan(rng); kw[i] = chan(rng); }
    for (int i = 0; i < D; i += 37) { qw[i] *= 4.0f; kw[i] *= 4.0f; }

    std::vector<uint8_t> cache((size_t)nSeq * maxBlocks * blockBytes, 0);
    // The values before q8_0 rounding (first sequence), for the cache's own error.
    std::vector<float> rawK((size_t)keys * NKV * D), rawV((size_t)keys * NKV * D);
    std::vector<float> x(D);
    for (int s = 0; s < nSeq; ++s)
        for (int k = 0; k < keys; ++k)
            for (int h = 0; h < NKV; ++h)
            {
                uint8_t* blk = cache.data() + ((size_t)s * maxBlocks + k / B) * blockBytes;
                for (int i = 0; i < D; ++i) x[i] = n01(rng) * kw[i];
                Quantize(x.data(), blk + kOff + ((long long)h * B + k % B) * TS_Q35_Q8_ROW);
                if (s == 0) std::memcpy(&rawK[((size_t)k * NKV + h) * D], x.data(), D * 4);
                for (int i = 0; i < D; ++i) x[i] = n01(rng);
                Quantize(x.data(), blk + vOff + ((long long)h * B + k % B) * TS_Q35_Q8_ROW);
                if (s == 0) std::memcpy(&rawV[((size_t)k * NKV + h) * D], x.data(), D * 4);
            }
    const size_t qn = (size_t)nSeq * chunk * NH * D;
    std::vector<float> q(qn), gate(qn);
    for (size_t i = 0; i < qn; ++i) { q[i] = n01(rng) * qw[i % D]; gate[i] = n01(rng); }
    std::vector<int> ends((size_t)nSeq * chunk);
    for (int s = 0; s < nSeq; ++s)
        for (int t = 0; t < chunk; ++t) ends[(size_t)s * chunk + t] = prefix + t + 1;

    uint8_t* dCache; float *dQ, *dGate, *dOut; int* dEnds; uint8_t** dTable;
    CHECK(cudaMalloc(&dCache, cache.size()));
    CHECK(cudaMalloc(&dQ, qn * 4));
    CHECK(cudaMalloc(&dGate, qn * 4));
    CHECK(cudaMalloc(&dOut, qn * 4));
    CHECK(cudaMalloc(&dEnds, ends.size() * 4));
    CHECK(cudaMalloc(&dTable, (size_t)nSeq * maxBlocks * sizeof(uint8_t*)));
    float* dPartial;
    CHECK(cudaMalloc(&dPartial, qn / TS_Q35_HD * splits * (TS_Q35_HD + 2) * 4));
    std::vector<uint8_t*> table((size_t)nSeq * maxBlocks);
    for (size_t i = 0; i < table.size(); ++i) table[i] = dCache + i * blockBytes;
    CHECK(cudaMemcpy(dCache, cache.data(), cache.size(), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dQ, q.data(), qn * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dGate, gate.data(), qn * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dEnds, ends.data(), ends.size() * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dTable, table.data(), table.size() * sizeof(uint8_t*), cudaMemcpyHostToDevice));
    CHECK(cudaFuncSetAttribute(ts_q35_attn_prefill_f32, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sizeof(TsQ35AttnPrefill)));

    // Reference rows: (sequence, token, head) samples spread over the chunk.
    struct Row { int s, t, h; };
    std::vector<Row> rows;
    std::uniform_int_distribution<int> pickT(0, chunk - 1), pickH(0, NH - 1), pickS(0, nSeq - 1);
    for (int i = 0; i < checked; ++i) rows.push_back({ pickS(rng), i == 0 ? 0 : i == 1 ? chunk - 1 : pickT(rng), pickH(rng) });
    // Reference over the cache's q8_0 values, and (exact = true) over the values before rounding.
    auto reference = [&](bool exact)
    {
        std::vector<std::vector<double>> ref(rows.size(), std::vector<double>(D));
        std::vector<double> kv(D), sc;
        auto fetch = [&](const Row& r, int k, bool isV)
        {
            const int kvh = r.h / G;
            if (exact)
            {
                const float* src = (isV ? rawV : rawK).data() + ((size_t)k * NKV + kvh) * D;
                for (int j = 0; j < D; ++j) kv[j] = src[j];
                return;
            }
            const uint8_t* blk = cache.data() + ((size_t)r.s * maxBlocks + k / B) * blockBytes;
            Dequantize(blk + (isV ? vOff : kOff) + ((long long)kvh * B + k % B) * TS_Q35_Q8_ROW, kv.data());
        };
        for (size_t i = 0; i < rows.size(); ++i)
        {
            const Row r = rows[i];
            const int n = prefix + r.t + 1;
            const float* qr = q.data() + (((size_t)r.s * chunk + r.t) * NH + r.h) * D;
            sc.assign(n, 0.0);
            double mx = -1e300;
            for (int k = 0; k < n; ++k)
            {
                fetch(r, k, false);
                double dot = 0.0;
                for (int j = 0; j < D; ++j) dot += (double)qr[j] * kv[j];
                sc[k] = dot * scale;
                mx = std::max(mx, sc[k]);
            }
            double sum = 0.0;
            std::vector<double> o(D, 0.0);
            for (int k = 0; k < n; ++k)
            {
                const double p = std::exp(sc[k] - mx);
                sum += p;
                fetch(r, k, true);
                for (int j = 0; j < D; ++j) o[j] += p * kv[j];
            }
            const float* gr = gate.data() + (((size_t)r.s * chunk + r.t) * NH + r.h) * D;
            for (int j = 0; j < D; ++j) ref[i][j] = o[j] / sum / (1.0 + std::exp(-(double)gr[j]));
        }
        return ref;
    };
    // Rows of the first sequence only, where the values before rounding were kept.
    for (Row& r : rows) r.s = 0;
    const auto ref = reference(false);
    {
        const auto exact = reference(true);
        double maxErr = 0.0, maxRef = 0.0, e2 = 0.0, r2 = 0.0;
        for (size_t i = 0; i < rows.size(); ++i)
            for (int j = 0; j < D; ++j)
            {
                const double e = std::fabs(ref[i][j] - exact[i][j]);
                maxErr = std::max(maxErr, e);
                maxRef = std::max(maxRef, std::fabs(exact[i][j]));
                e2 += e * e;
                r2 += exact[i][j] * exact[i][j];
            }
        std::printf("%-24s %8s     %6s          max err %.2e of %.3f  rel L2 %.2e  (exact q8_0 vs unrounded K/V)\n",
                    "q8_0 cache itself", "", "", maxErr / maxRef, maxRef, std::sqrt(e2 / r2));
    }

    const dim3 grid((chunk * G + 127) / 128, NKV, nSeq);
    auto run = [&](bool prefill)
    {
        if (prefill)
        {
            ts_q35_attn_prefill_f32<<<dim3(grid.x * splits, grid.y, grid.z), 256, sizeof(TsQ35AttnPrefill)>>>(dQ, dGate, dEnds, dTable,
                maxBlocks, kOff, vOff, chunk, NH, NKV, scale, splits, dPartial, dOut);
            if (splits > 1)
                ts_q35_attn_combine_f32<<<nSeq * chunk * NH, TS_Q35_HD>>>(dPartial, dGate, splits, dOut);
        }
        else
            ts_q35_attn_q8_f32<<<grid, 256>>>(dQ, dGate, dEnds, dTable, maxBlocks, kOff, vOff, chunk, NH, NKV, scale, dOut);
    };
    // Causal FLOPs actually needed (QK and PV): 4 * D per (row, visible key).
    const double flops = 4.0 * D * NH * nSeq * ((double)chunk * prefix + (double)chunk * (chunk + 1) / 2);
    std::vector<float> got(qn);
    for (int k = 0; k < 2; ++k)
    {
        CHECK(cudaMemset(dOut, 0, qn * 4));
        run(k);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(got.data(), dOut, qn * 4, cudaMemcpyDeviceToHost));
        double maxErr = 0.0, maxRef = 0.0, e2 = 0.0, r2 = 0.0;
        for (size_t i = 0; i < rows.size(); ++i)
        {
            const Row r = rows[i];
            const float* gr = got.data() + (((size_t)r.s * chunk + r.t) * NH + r.h) * D;
            for (int j = 0; j < D; ++j)
            {
                const double e = std::fabs(gr[j] - ref[i][j]);
                maxErr = std::max(maxErr, e);
                maxRef = std::max(maxRef, std::fabs(ref[i][j]));
                e2 += e * e;
                r2 += ref[i][j] * ref[i][j];
            }
        }
        cudaEvent_t e0, e1;
        CHECK(cudaEventCreate(&e0));
        CHECK(cudaEventCreate(&e1));
        const int reps = 10;
        CHECK(cudaEventRecord(e0));
        for (int i = 0; i < reps; ++i) run(k);
        CHECK(cudaEventRecord(e1));
        CHECK(cudaEventSynchronize(e1));
        float ms;
        CHECK(cudaEventElapsedTime(&ms, e0, e1));
        ms /= reps;
        std::printf("%-24s %8.3f ms  %6.1f TFLOPS  max err %.2e of %.3f  rel L2 %.2e\n",
                    k ? "ts_q35_attn_prefill_f32" : "ts_q35_attn_q8_f32", ms, flops / ms / 1e9, maxErr / maxRef, maxRef,
                    std::sqrt(e2 / r2));
    }
    return 0;
}
