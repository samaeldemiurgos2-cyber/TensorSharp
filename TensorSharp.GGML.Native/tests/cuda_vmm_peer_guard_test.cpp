// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// ggml-cuda's scratch pool on every visible GPU, through GgmlOps (and so through
// ggml_ops_cuda_vmm_peer_guard.cpp). Each device runs a Q4_0 matrix-vector
// product, which quantizes the vector into pool memory and so grows the pool.
// All backends stay alive together, as tensor-parallel ranks do.
//
// On a host within CUDA's 8-peers-per-GPU limit this checks the pass-through
// path. On a host past it (10+ mutually peer-capable GPUs with an NCCL build),
// ggml's grant is refused and the guard must (1) keep the compute alive and
// correct and (2) leave no peer slot behind: the refused grant had already
// mapped peers, so without the revoke GPU 0 would hold 8 peers and the last GPU
// could not enable peer access to it.
//
// Exit codes: 0 pass, 1 fail, 77 no CUDA device.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace tsg
{
    bool cuda_vmm_pool_grant_exceeds_peer_limit(const int * device_indices, int count);
    std::uint64_t cuda_vmm_peer_guard_handled_mask();
}

namespace
{
    constexpr int K = 4096;
    constexpr int N = 256;

    struct device_run
    {
        ggml_backend_t backend = nullptr;
        ggml_context * ctx = nullptr;
        ggml_backend_buffer_t buf = nullptr;
    };

    bool run_on(int device, const std::vector<uint8_t> & wq, const std::vector<float> & x,
                const std::vector<float> & want, device_run & run)
    {
        run.backend = ggml_backend_cuda_init(device);
        if (!run.backend) { std::printf("device %d: ggml_backend_cuda_init failed\n", device); return false; }
        ggml_init_params ip = { 8 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
        run.ctx = ggml_init(ip);
        ggml_tensor * w = ggml_new_tensor_2d(run.ctx, GGML_TYPE_Q4_0, K, N);
        ggml_tensor * v = ggml_new_tensor_2d(run.ctx, GGML_TYPE_F32, K, 1);
        ggml_tensor * y = ggml_mul_mat(run.ctx, w, v);
        ggml_cgraph * gf = ggml_new_graph(run.ctx);
        ggml_build_forward_expand(gf, y);
        run.buf = ggml_backend_alloc_ctx_tensors(run.ctx, run.backend);
        if (!run.buf) { std::printf("device %d: weight allocation failed\n", device); return false; }
        ggml_backend_tensor_set(w, wq.data(), 0, wq.size());
        ggml_backend_tensor_set(v, x.data(), 0, x.size() * sizeof(float));
        if (ggml_backend_graph_compute(run.backend, gf) != GGML_STATUS_SUCCESS)
        {
            std::printf("device %d: graph compute failed\n", device);
            return false;
        }
        std::vector<float> got(N);
        ggml_backend_tensor_get(y, got.data(), 0, got.size() * sizeof(float));
        double worst = 0.0, scale = 0.0;
        for (int i = 0; i < N; i++)
        {
            worst = std::fmax(worst, std::fabs((double) got[(size_t) i] - want[(size_t) i]));
            scale = std::fmax(scale, std::fabs((double) want[(size_t) i]));
        }
        // Exact inputs leave only F32 summation order.
        const bool ok = worst <= 1e-3 * std::fmax(scale, 1.0);
        std::printf("device %d: max |error| %.3g of max |y| %.3g -> %s\n", device, worst, scale, ok ? "ok" : "WRONG");
        return ok;
    }
}

int main()
{
    const int n = ggml_backend_cuda_get_device_count();
    if (n <= 0)
    {
        std::printf("SKIP: no CUDA device\n");
        return 77;
    }

    std::vector<float> wf((size_t) K * N), x(K);
    for (size_t i = 0; i < wf.size(); i++) wf[i] = (float) std::sin(0.37 * (double) i) * 0.5f;
    // The kernel quantizes the vector to Q8_1 (scale and block sum in F16).
    // Make that lossless: every 32-value block holds +-k * 2^-7 pairs with one
    // k = 127, so the scale is exactly 2^-7 and the block sum is zero.
    for (int b = 0; b < K / 32; b++)
        for (int j = 0; j < 16; j++)
        {
            const int k = j == 0 ? 127 : (b * 7 + j * 13) % 127;
            x[(size_t) (32 * b + 2 * j)] = (float) k / 128.0f;
            x[(size_t) (32 * b + 2 * j + 1)] = -(float) k / 128.0f;
        }
    std::vector<uint8_t> wq(ggml_row_size(GGML_TYPE_Q4_0, K) * N);
    ggml_quantize_chunk(GGML_TYPE_Q4_0, wf.data(), wq.data(), 0, N, K, nullptr);
    // Reference from the quantized weights, so only the kernel's own rounding remains.
    std::vector<float> row(K), want(N);
    const ggml_type_traits * tt = ggml_get_type_traits(GGML_TYPE_Q4_0);
    for (int r = 0; r < N; r++)
    {
        tt->to_float(wq.data() + (size_t) r * ggml_row_size(GGML_TYPE_Q4_0, K), row.data(), K);
        double acc = 0.0;
        for (int i = 0; i < K; i++) acc += (double) row[(size_t) i] * x[(size_t) i];
        want[(size_t) r] = (float) acc;
    }

    std::vector<device_run> runs((size_t) n);
    bool ok = true;
    for (int d = 0; d < n; d++)
        ok = run_on(d, wq, x, want, runs[(size_t) d]) && ok;

    // Read before the probe below, which marks devices without the wrapper.
    const std::uint64_t handled = tsg::cuda_vmm_peer_guard_handled_mask();
    // Asked after the computes, so the guard met the refusal itself (and had to
    // revoke the partial grant) instead of learning it from this probe first.
    std::vector<int> all((size_t) n);
    for (int d = 0; d < n; d++) all[(size_t) d] = d;
    const bool limited = tsg::cuda_vmm_pool_grant_exceeds_peer_limit(all.data(), n);
    std::printf("%d CUDA device(s); ggml's pool grant %s CUDA's per-GPU peer limit\n",
                n, limited ? "exceeds" : "fits");

    if (limited && handled == 0)
    {
#if defined(TSG_GGML_CUDA_NCCL)
        std::printf("FAIL: ggml asked to share pool memory past the limit but the wrapper never handled a refusal\n");
        ok = false;
#else
        // Without NCCL ggml grants each chunk to its own GPU only; nothing to downgrade.
        std::printf("SKIP: this build's ggml never shares pool memory, so the downgrade path did not run\n");
        return ok ? 77 : 1;
#endif
    }
    else if (limited)
    {
        std::printf("the guard downgraded refused grants on device mask 0x%llx\n", (unsigned long long) handled);
    }

    if (limited && n >= 2)
    {
        int can = 0;
        cudaDeviceCanAccessPeer(&can, n - 1, 0);
        if (can)
        {
            cudaSetDevice(n - 1);
            const cudaError_t rc = cudaDeviceEnablePeerAccess(0, 0);
            const bool slot_free = rc == cudaSuccess || rc == cudaErrorPeerAccessAlreadyEnabled;
            std::printf("peer access %d -> 0 after the refused grants: %s\n", n - 1, cudaGetErrorString(rc));
            if (rc == cudaSuccess) cudaDeviceDisablePeerAccess(0);
            (void) cudaGetLastError();
            if (!slot_free)
            {
                std::printf("FAIL: a refused pool grant left peer slots mapped\n");
                ok = false;
            }
        }
    }

    for (auto & r : runs)
    {
        if (r.buf) ggml_backend_buffer_free(r.buf);
        if (r.ctx) ggml_free(r.ctx);
        if (r.backend) ggml_backend_free(r.backend);
    }
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
