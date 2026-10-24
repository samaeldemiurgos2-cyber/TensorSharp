// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// ---------------------------------------------------------------------------
// Keeps ggml-cuda's kernel-scratch pool working on hosts with more peer-capable
// GPUs than CUDA's per-GPU peer limit.
//
// ggml-cuda grows each device's scratch pool with the VMM API. In an NCCL build
// (GGML_USE_NCCL, which TensorSharp turns on whenever NCCL is installed) or with
// GGML_CUDA_P2P set, ggml_cuda_pool_vmm::alloc grants every new pool chunk
// read/write access for the owning GPU and for EVERY visible device that
// reports peer capability, in one cuMemSetAccess call. A GPU can hold at most
// eight peer connections. On a host with ten mutually peer-capable GPUs the
// call fails with CUDA_ERROR_TOO_MANY_PEERS ("peer mapping resources
// exhausted") and ggml aborts at the first pool growth, so every ggml_cuda
// compute dies: tensor parallelism, layer split, even one GPU.
//
// ggml stays unchanged. GgmlOps is linked with --wrap=cuMemSetAccess, so ggml's
// call sites reach __wrap_cuMemSetAccess below, and a grant the driver accepts
// passes straight through. When the driver refuses one with TOO_MANY_PEERS:
//   * The refused call is NOT atomic. It has already granted the peers listed
//     before the one that ran out of slots, and those grants keep holding peer
//     slots. Every non-owner descriptor is revoked with PROT_NONE, which also
//     returns the slots.
//   * The owning GPU keeps read/write access. That is exactly what ggml grants
//     in a build without NCCL.
//   * The device is remembered, so its later pool growth goes straight to the
//     owner-only grant, and one stderr line reports the switch.
// Measured on 10x A40 (driver 595.91): each GPU accepts self + 8 peers; a
// refused grant had mapped peers 1-8; revoking them (or freeing the chunk) let
// GPU 9 peer with GPU 0 again.
//
// Pool memory is kernel scratch that only its own GPU reads, with one
// exception: ggml's NCCL AllReduce converts large tensors to BF16 temporaries
// in the pool, and NCCL's in-process "P2P direct" mode lets peer GPUs access
// those. tp_cuda_respect_vmm_peer_limit() turns that mode off before the first
// communicator exists on such hosts. Cross-device tensor copies use ordinary
// cudaMalloc buffers and need no grant.
// ---------------------------------------------------------------------------
#include "ggml_ops_internal.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(TSG_CUDA_VMM_PEER_GUARD)

#include <cuda.h>
#include <cuda_runtime.h>

#include <atomic>
#include <cstdint>
#include <mutex>
#include <vector>

#ifdef cuMemSetAccess
#error "cuda.h maps cuMemSetAccess to a versioned entry point; --wrap=cuMemSetAccess would no longer reach ggml's call"
#endif

extern "C" CUresult CUDAAPI __real_cuMemSetAccess(CUdeviceptr ptr, size_t size, const CUmemAccessDesc * desc, size_t count);
// Hidden inside the GgmlOps shared library: ggml's call sites bind to it at
// link time and nothing outside should. An executable that links the static
// ggml-cuda keeps default visibility: when it also links GgmlOps, --wrap
// applies to that library's pass-through reference, and ld refuses a hidden
// definition for it (the library still calls libcuda at run time).
#if defined(TSG_CUDA_VMM_PEER_GUARD_HIDDEN)
extern "C" __attribute__((visibility("hidden")))
#else
extern "C"
#endif
CUresult CUDAAPI __wrap_cuMemSetAccess(CUdeviceptr ptr, size_t size, const CUmemAccessDesc * desc, size_t count);

namespace
{
    // Bit d: device d cannot share its pool memory with every peer ggml lists.
    std::atomic<std::uint64_t> g_owner_only{0};
    // Bit d: the wrapper itself downgraded a refused grant on device d.
    std::atomic<std::uint64_t> g_handled{0};
    std::once_flag g_switch_logged;

    bool limited(int device)
    {
        return device >= 0 && device < 64 && ((g_owner_only.load(std::memory_order_acquire) >> device) & 1u) != 0;
    }

    void mark_limited(int device)
    {
        if (device >= 0 && device < 64)
            g_owner_only.fetch_or(std::uint64_t(1) << device, std::memory_order_acq_rel);
    }

    // The GPU whose physical memory backs `ptr`, read from the allocation
    // itself rather than from whichever context happens to be current.
    bool owner_of(CUdeviceptr ptr, int * owner)
    {
        CUmemGenericAllocationHandle handle;
        if (cuMemRetainAllocationHandle(&handle, reinterpret_cast<void *>(ptr)) != CUDA_SUCCESS)
            return false;
        CUmemAllocationProp prop = {};
        const CUresult rc = cuMemGetAllocationPropertiesFromHandle(&prop, handle);
        cuMemRelease(handle);
        if (rc != CUDA_SUCCESS || prop.location.type != CU_MEM_LOCATION_TYPE_DEVICE)
            return false;
        *owner = prop.location.id;
        return true;
    }

    const CUmemAccessDesc * find_device(const CUmemAccessDesc * desc, size_t count, int device)
    {
        for (size_t i = 0; i < count; i++)
            if (desc[i].location.type == CU_MEM_LOCATION_TYPE_DEVICE && desc[i].location.id == device)
                return &desc[i];
        return nullptr;
    }

    // Revoke every listed location except the owner, then grant the owner alone.
    CUresult grant_owner_only(CUdeviceptr ptr, size_t size, const CUmemAccessDesc * desc, size_t count,
                              const CUmemAccessDesc & owner, bool revoke)
    {
        if (revoke)
        {
            std::vector<CUmemAccessDesc> none;
            for (size_t i = 0; i < count; i++)
            {
                if (&desc[i] == &owner) continue;
                CUmemAccessDesc d = desc[i];
                d.flags = CU_MEM_ACCESS_FLAGS_PROT_NONE;
                none.push_back(d);
            }
            if (!none.empty())
            {
                const CUresult rc = __real_cuMemSetAccess(ptr, size, none.data(), none.size());
                if (rc != CUDA_SUCCESS) return rc;
            }
        }
        return __real_cuMemSetAccess(ptr, size, &owner, 1);
    }

    void log_switch(int device, size_t requested)
    {
        std::call_once(g_switch_logged, [&]
        {
            std::fprintf(stderr,
                "[CUDA] GPU %d cannot share its kernel-scratch memory with %zu peer GPU(s): CUDA allows each GPU "
                "at most 8 peer connections. ggml-cuda's scratch pool on this host stays private to each GPU, "
                "as in a build without NCCL; tensor copies between GPUs do not use it, and NCCL's in-process "
                "P2P direct mode is turned off for it before tensor parallelism starts.\n",
                device, requested - 1);
            std::fflush(stderr);
        });
    }
}

extern "C" CUresult CUDAAPI __wrap_cuMemSetAccess(CUdeviceptr ptr, size_t size, const CUmemAccessDesc * desc, size_t count)
{
    if (count <= 1 || desc == nullptr)
        return __real_cuMemSetAccess(ptr, size, desc, count);

    int owner = -1;
    if (g_owner_only.load(std::memory_order_acquire) != 0 && owner_of(ptr, &owner) && limited(owner))
    {
        // Nothing is granted yet on this chunk, so there is nothing to revoke.
        if (const CUmemAccessDesc * own = find_device(desc, count, owner))
            return grant_owner_only(ptr, size, desc, count, *own, false);
    }

    const CUresult rc = __real_cuMemSetAccess(ptr, size, desc, count);
    if (rc != CUDA_ERROR_TOO_MANY_PEERS)
        return rc;
    if (owner < 0 && !owner_of(ptr, &owner))
        return rc;
    const CUmemAccessDesc * own = find_device(desc, count, owner);
    if (own == nullptr || grant_owner_only(ptr, size, desc, count, *own, true) != CUDA_SUCCESS)
        return rc;
    mark_limited(owner);
    if (owner >= 0 && owner < 64)
        g_handled.fetch_or(std::uint64_t(1) << owner, std::memory_order_acq_rel);
    log_switch(owner, count);
    return CUDA_SUCCESS;
}

namespace tsg
{
    std::uint64_t cuda_vmm_peer_guard_handled_mask()
    {
        return g_handled.load(std::memory_order_acquire);
    }

    bool cuda_vmm_pool_grant_exceeds_peer_limit(const int * device_indices, int count)
    {
        int n = 0;
        if (device_indices == nullptr || count <= 0 || cudaGetDeviceCount(&n) != cudaSuccess)
        {
            (void) cudaGetLastError();
            return false;
        }
        int previous = -1;
        (void) cudaGetDevice(&previous);
        bool exceeds = false;
        for (int r = 0; r < count; r++)
        {
            const int device = device_indices[r];
            if (device < 0 || device >= n) continue;
            if (limited(device)) { exceeds = true; continue; }

            // The set ggml_cuda_pool_vmm::alloc grants when it shares pool memory.
            std::vector<CUmemAccessDesc> desc;
            for (int id = 0; id < n; id++)
            {
                if (id != device)
                {
                    int can = 0;
                    if (cudaDeviceCanAccessPeer(&can, id, device) != cudaSuccess || !can) continue;
                }
                CUmemAccessDesc a = {};
                a.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                a.location.id = id;
                a.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
                desc.push_back(a);
            }
            if (desc.size() <= 1) continue;

            // Try that grant on a scratch chunk. Freeing the chunk returns
            // whatever a refused, partially applied grant took.
            if (cudaSetDevice(device) != cudaSuccess || cudaFree(nullptr) != cudaSuccess) continue;
            CUmemAllocationProp prop = {};
            prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            prop.location.id = device;
            size_t granularity = 0;
            if (cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM) != CUDA_SUCCESS ||
                granularity == 0)
                continue;
            CUmemGenericAllocationHandle handle;
            if (cuMemCreate(&handle, granularity, &prop, 0) != CUDA_SUCCESS) continue;
            CUdeviceptr ptr = 0;
            if (cuMemAddressReserve(&ptr, granularity, 0, 0, 0) != CUDA_SUCCESS)
            {
                cuMemRelease(handle);
                continue;
            }
            if (cuMemMap(ptr, granularity, 0, handle, 0) == CUDA_SUCCESS)
            {
                if (__real_cuMemSetAccess(ptr, granularity, desc.data(), desc.size()) == CUDA_ERROR_TOO_MANY_PEERS)
                {
                    mark_limited(device);
                    exceeds = true;
                }
                cuMemUnmap(ptr, granularity);
            }
            cuMemRelease(handle);
            cuMemAddressFree(ptr, granularity);
        }
        if (previous >= 0) (void) cudaSetDevice(previous);
        (void) cudaGetLastError();
        return exceeds;
    }
}

#endif // TSG_CUDA_VMM_PEER_GUARD

namespace tsg
{
    void tp_cuda_respect_vmm_peer_limit(const int * device_indices, int count)
    {
#if defined(TSG_CUDA_VMM_PEER_GUARD) && defined(TSG_GGML_CUDA_NCCL)
        // An operator's explicit choice wins, NCCL without P2P never reads a
        // peer's memory directly, and an explicit non-NCCL AllReduce makes
        // ggml create no communicator.
        if (std::getenv("NCCL_P2P_DIRECT_DISABLE") != nullptr)
            return;
        const char * transport = std::getenv("GGML_CUDA_ALLREDUCE");
        if (transport != nullptr && std::strcmp(transport, "nccl") != 0)
            return;
        const char * p2p_off = std::getenv("NCCL_P2P_DISABLE");
        if (p2p_off != nullptr && std::atoi(p2p_off) != 0)
            return;
        if (!cuda_vmm_pool_grant_exceeds_peer_limit(device_indices, count))
            return;
        setenv("NCCL_P2P_DIRECT_DISABLE", "1", 1);
        std::fprintf(stderr,
            "[TP] these GPUs have more peers than CUDA's 8-per-GPU limit, so ggml's scratch memory stays private "
            "to each GPU; running NCCL with NCCL_P2P_DIRECT_DISABLE=1 (P2P through NCCL's own buffers).\n");
        std::fflush(stderr);
#else
        (void) device_indices;
        (void) count;
#endif
    }
}
