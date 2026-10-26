// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// Checks and times the direct-CUDA engine's gated delta rule (TensorSharp.Backends.Cuda/native/
// kernels/tensorsharp_q35_kernels.cu) on Qwen3.8-27B's Gated DeltaNet shape (16 key heads, 48
// value heads, head width 128): the token-by-token ts_q35_gdn_scan_f32 and the chunked
// ts_q35_gdn_chunk_f32, from the same random state over the same inputs (q and k L2-normalized
// as the prep kernel leaves them, log decays and betas in the model's ranges). Both are measured
// against a double-precision CPU reference of the first value heads: outputs and final state,
// largest error relative to the reference's largest magnitude.
//
//   nvcc -O3 -std=c++17 -arch=sm_120a eng/validation/tsg-q35-gdn-bench.cu -o artifacts/tsg-q35-gdn-bench
//   artifacts/tsg-q35-gdn-bench [rows=4096] [sequences=1] [checked heads=2]
#include "../../TensorSharp.Backends.Cuda/native/kernels/tensorsharp_q35_kernels.cu"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); std::exit(1); } } while (0)

namespace
{
    constexpr int S = TS_Q35_GDN_HD, HK = 16, HV = 48, CONV = 2 * HK * S + HV * S;

    struct Result { std::vector<float> out, state; float ms; };

    Result Run(bool chunked, const std::vector<float>& qkv, const std::vector<float>& g, const std::vector<float>& beta,
               const std::vector<float>& state0, int rows, int nSeq)
    {
        float *dQkv, *dG, *dBeta, *dState, *dOut;
        float** dPtrs;
        CHECK(cudaMalloc(&dQkv, qkv.size() * 4));
        CHECK(cudaMalloc(&dG, g.size() * 4));
        CHECK(cudaMalloc(&dBeta, beta.size() * 4));
        CHECK(cudaMalloc(&dState, state0.size() * 4));
        CHECK(cudaMalloc(&dOut, (size_t)rows * nSeq * HV * S * 4));
        CHECK(cudaMalloc(&dPtrs, nSeq * sizeof(float*)));
        std::vector<float*> ptrs(nSeq);
        for (int i = 0; i < nSeq; ++i)
            ptrs[i] = dState + (size_t)i * HV * S * S;
        CHECK(cudaMemcpy(dPtrs, ptrs.data(), nSeq * sizeof(float*), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(dQkv, qkv.data(), qkv.size() * 4, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(dG, g.data(), g.size() * 4, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(dBeta, beta.data(), beta.size() * 4, cudaMemcpyHostToDevice));
        auto launch = [&]()
        {
            if (chunked)
                ts_q35_gdn_chunk_f32<<<dim3(HV * (S / TS_Q35_GDN_V), nSeq), 256>>>(dQkv, CONV, HK, HV, rows, dG, dBeta, dPtrs, dOut);
            else
                ts_q35_gdn_scan_f32<<<dim3(HV, nSeq, S / 16), 128>>>(dQkv, CONV, HK, HV, rows, dG, dBeta, dPtrs, dOut);
        };
        // Timed passes evolve the state; the checked pass starts from state0 again.
        CHECK(cudaMemcpy(dState, state0.data(), state0.size() * 4, cudaMemcpyHostToDevice));
        launch();
        cudaEvent_t e0, e1;
        CHECK(cudaEventCreate(&e0));
        CHECK(cudaEventCreate(&e1));
        const int reps = 10;
        CHECK(cudaEventRecord(e0));
        for (int i = 0; i < reps; ++i)
            launch();
        CHECK(cudaEventRecord(e1));
        CHECK(cudaEventSynchronize(e1));
        float ms = 0;
        CHECK(cudaEventElapsedTime(&ms, e0, e1));
        CHECK(cudaMemcpy(dState, state0.data(), state0.size() * 4, cudaMemcpyHostToDevice));
        launch();
        CHECK(cudaGetLastError());
        Result r{ std::vector<float>((size_t)rows * nSeq * HV * S), std::vector<float>(state0.size()), ms / reps };
        CHECK(cudaMemcpy(r.out.data(), dOut, r.out.size() * 4, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(r.state.data(), dState, r.state.size() * 4, cudaMemcpyDeviceToHost));
        cudaFree(dQkv); cudaFree(dG); cudaFree(dBeta); cudaFree(dState); cudaFree(dOut); cudaFree(dPtrs);
        return r;
    }
}

int main(int argc, char** argv)
{
    const int rows = argc > 1 ? std::atoi(argv[1]) : 4096;
    const int nSeq = argc > 2 ? std::atoi(argv[2]) : 1;
    const int checked = argc > 3 ? std::atoi(argv[3]) : 2;
    const int total = rows * nSeq;
    std::mt19937 rng(11);
    std::normal_distribution<float> n01(0.0f, 1.0f);
    std::vector<float> qkv((size_t)total * CONV), g((size_t)total * HV), beta((size_t)total * HV);
    for (int r = 0; r < total; ++r)
    {
        float* row = qkv.data() + (size_t)r * CONV;
        for (int i = 0; i < CONV; ++i)
            row[i] = n01(rng);
        for (int head = 0; head < 2 * HK; ++head)   // q and k heads unit-norm
        {
            double ss = 0;
            for (int i = 0; i < S; ++i)
                ss += (double)row[head * S + i] * row[head * S + i];
            const float inv = (float)(1.0 / std::sqrt(ss));
            for (int i = 0; i < S; ++i)
                row[head * S + i] *= inv;
        }
        for (int h = 0; h < HV; ++h)
        {
            const float a = n01(rng), b = n01(rng);
            g[(size_t)r * HV + h] = -std::log1p(std::exp(a)) * std::exp(0.5f * n01(rng) - 1.0f);   // softplus * -e^A
            beta[(size_t)r * HV + h] = 1.0f / (1.0f + std::exp(-b));
        }
    }
    std::vector<float> state0((size_t)nSeq * HV * S * S);
    for (auto& x : state0)
        x = 0.05f * n01(rng);

    const Result seq = Run(false, qkv, g, beta, state0, rows, nSeq);
    const Result chk = Run(true, qkv, g, beta, state0, rows, nSeq);

    // Double-precision reference of sequence 0, value heads [0, checked).
    double errOut[2] = {}, errState[2] = {}, refOut = 0, refState = 0;
    const Result* res[2] = { &seq, &chk };
    for (int h = 0; h < checked; ++h)
    {
        const int kh = h % HK;
        std::vector<double> st(S * S);   // st[col * S + i] = S[i][col]
        for (int i = 0; i < S * S; ++i)
            st[i] = state0[(size_t)h * S * S + i];
        for (int t = 0; t < rows; ++t)
        {
            const float* row = qkv.data() + (size_t)t * CONV;
            const float* q = row + kh * S;
            const float* k = row + (HK + kh) * S;
            const float* v = row + 2 * HK * S + h * S;
            const double decay = std::exp((double)g[(size_t)t * HV + h]), b = beta[(size_t)t * HV + h];
            for (int col = 0; col < S; ++col)
            {
                double* m = st.data() + (size_t)col * S;
                double kv = 0;
                for (int i = 0; i < S; ++i)
                    kv += m[i] * k[i];
                const double delta = (v[col] - decay * kv) * b;
                double o = 0;
                for (int i = 0; i < S; ++i)
                {
                    m[i] = decay * m[i] + k[i] * delta;
                    o += m[i] * q[i];
                }
                o /= std::sqrt((double)S);
                refOut = std::max(refOut, std::fabs(o));
                for (int which = 0; which < 2; ++which)
                    errOut[which] = std::max(errOut[which], std::fabs(o - res[which]->out[((size_t)t * HV + h) * S + col]));
            }
        }
        for (int i = 0; i < S * S; ++i)
        {
            refState = std::max(refState, std::fabs(st[i]));
            for (int which = 0; which < 2; ++which)
                errState[which] = std::max(errState[which], std::fabs(st[i] - res[which]->state[(size_t)h * S * S + i]));
        }
    }
    double diff = 0, mag = 0;
    for (size_t i = 0; i < seq.out.size(); ++i)
    {
        diff = std::max(diff, (double)std::fabs(seq.out[i] - chk.out[i]));
        mag = std::max(mag, (double)std::fabs(seq.out[i]));
    }
    std::printf("rows %d x %d sequences, %d value heads checked against the F64 reference\n", rows, nSeq, checked);
    std::printf("  sequential: %8.3f ms   out err %.3e   state err %.3e\n", seq.ms, errOut[0] / refOut, errState[0] / refState);
    std::printf("  chunked:    %8.3f ms   out err %.3e   state err %.3e\n", chk.ms, errOut[1] / refOut, errState[1] / refState);
    std::printf("  chunked vs sequential, every head: max |diff| / max |out| = %.3e\n", diff / mag);
    return 0;
}
