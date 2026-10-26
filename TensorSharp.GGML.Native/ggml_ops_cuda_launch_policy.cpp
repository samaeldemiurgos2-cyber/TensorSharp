// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#include "ggml_ops_cuda_launch_policy.h"
#include <cstdio>
#include <cstdlib>
#include <mutex>

#ifdef TSG_GGML_USE_CUDA
#include <cuda.h>
#endif

namespace tsg {
CudaPdlDefaultResult apply_cuda_pdl_default(bool has_sm12_device) {
    if (!has_sm12_device || std::getenv("GGML_CUDA_PDL") != nullptr)
        return CudaPdlDefaultResult::Unchanged;
#ifdef _WIN32
    const int status = _putenv_s("GGML_CUDA_PDL", "0");
#else
    const int status = setenv("GGML_CUDA_PDL", "0", 0);
#endif
    return status == 0 ? CudaPdlDefaultResult::Disabled : CudaPdlDefaultResult::Error;
}

bool configure_cuda_launch_policy() {
#ifdef TSG_GGML_USE_CUDA
    static std::once_flag once;
    static bool success = true;
    std::call_once(once, [] {
        if (std::getenv("GGML_CUDA_PDL") != nullptr) return;
        // Driver queries neither create a CUDA context nor initialize ggml's
        // device registry. Keep the workaround scoped to visible SM12 devices.
        // On the tested RTX 5090, PDL with CUDA graph capture corrupts the
        // 2-lane -> 4-lane F32 arena fixture starting at its first capture;
        // disabling PDL alone restores exact logits and recurrent state.
        if (cuInit(0) != CUDA_SUCCESS) return; // normal backend discovery reports unavailable devices
        int count = 0;
        if (cuDeviceGetCount(&count) != CUDA_SUCCESS) return;
        bool affected = false;
        for (int i = 0; i < count; ++i) {
            CUdevice device;
            int major = 0;
            if (cuDeviceGet(&device, i) == CUDA_SUCCESS &&
                cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device) == CUDA_SUCCESS &&
                major == 12) {
                affected = true;
                break;
            }
        }
        const auto result = apply_cuda_pdl_default(affected);
        success = result != CudaPdlDefaultResult::Error;
        if (result == CudaPdlDefaultResult::Disabled) {
            std::fprintf(stderr, "[CUDA] SM12 graph compatibility: default GGML_CUDA_PDL=0; CUDA graphs remain enabled.\n");
        }
    });
    return success;
#else
    return true;
#endif
}
}
