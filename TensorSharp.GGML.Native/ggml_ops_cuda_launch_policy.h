// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#pragma once

namespace tsg {
enum class CudaPdlDefaultResult { Unchanged, Disabled, Error };

// Separate environment policy from hardware discovery so explicit settings
// and unaffected architectures can be checked without a CUDA device.
CudaPdlDefaultResult apply_cuda_pdl_default(bool has_sm12_device);

// Call before ggml's first CUDA device registration/enumeration. The upstream
// launch helpers cache their environment setting on first use.
bool configure_cuda_launch_policy();
}
