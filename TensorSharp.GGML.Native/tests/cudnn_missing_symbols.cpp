// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
// A cuDNN stand-in with an incomplete export table: it reports the CUDA major this
// build targets, so the loader's CUDA-major check passes, then exports none of the
// cuDNN functions the loader resolves next. The regression places this DLL and
// sibling-name copies in an isolated build fixture to exercise that failure path.
#include <cuda_runtime_api.h>   // CUDART_VERSION of the toolkit GgmlOps is built with
#include <cstddef>

static int g_version_queries = 0;

extern "C" __declspec(dllexport) int TensorSharpMissingCudnnFixture() { return 1; }
extern "C" __declspec(dllexport) int TensorSharpCudnnFixtureVersionQueries() { return g_version_queries; }
extern "C" __declspec(dllexport) std::size_t cudnnGetCudartVersion()
{
    ++g_version_queries;
    return CUDART_VERSION;
}
