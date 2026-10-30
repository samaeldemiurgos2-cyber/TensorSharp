// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
// A cuDNN stand-in that reports a CUDA 12.8 build (cudnnGetCudartVersion() == 12080)
// but otherwise exports every function the loader resolves. cuda12 and cuda13 builds
// of cuDNN 9 share every file name, so the loader must decline this one on the CUDA
// major alone, before calling anything. Each stub counts its calls and returns a
// failure status, so the regression can prove that none of them ran.
#include <cstddef>

static int g_version_queries = 0;
static int g_entry_calls = 0;

extern "C" __declspec(dllexport) int TensorSharpCuda12CudnnFixture() { return 1; }
extern "C" __declspec(dllexport) int TensorSharpCudnnFixtureVersionQueries() { return g_version_queries; }
extern "C" __declspec(dllexport) int TensorSharpCudnnFixtureEntryCalls() { return g_entry_calls; }
extern "C" __declspec(dllexport) std::size_t cudnnGetCudartVersion()
{
    ++g_version_queries;
    return 12080;
}

// The real signatures do not matter: these are resolved by name only, and on x64 a
// function that ignores its register arguments is safe to call with any of them.
#define TSG_FIXTURE_STUB(name) \
    extern "C" __declspec(dllexport) int name() { ++g_entry_calls; return 1; /* not CUDNN_STATUS_SUCCESS */ }
TSG_FIXTURE_STUB(cudnnCreate)
TSG_FIXTURE_STUB(cudnnDestroy)
TSG_FIXTURE_STUB(cudnnCreateTensorDescriptor)
TSG_FIXTURE_STUB(cudnnCreateFilterDescriptor)
TSG_FIXTURE_STUB(cudnnCreateConvolutionDescriptor)
TSG_FIXTURE_STUB(cudnnDestroyTensorDescriptor)
TSG_FIXTURE_STUB(cudnnDestroyFilterDescriptor)
TSG_FIXTURE_STUB(cudnnDestroyConvolutionDescriptor)
TSG_FIXTURE_STUB(cudnnSetTensor4dDescriptor)
TSG_FIXTURE_STUB(cudnnSetFilter4dDescriptor)
TSG_FIXTURE_STUB(cudnnSetConvolution2dDescriptor)
TSG_FIXTURE_STUB(cudnnSetConvolutionMathType)
TSG_FIXTURE_STUB(cudnnGetConvolution2dForwardOutputDim)
TSG_FIXTURE_STUB(cudnnGetConvolutionForwardWorkspaceSize)
TSG_FIXTURE_STUB(cudnnConvolutionForward)
#undef TSG_FIXTURE_STUB
