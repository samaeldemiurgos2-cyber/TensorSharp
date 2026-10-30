// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Runtime.CompilerServices;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda;

/// <summary>Embedding-specific CUDA GEMM dispatch with F32 tensor storage,
/// F32 accumulation/output, and explicitly selected pedantic F32 or TF32 inputs.
/// These calls preserve the F32 exponent range and do not change global cuBLAS
/// math mode or unrelated model dispatch.</summary>
public static class CudaEmbeddingOps
{
    // Verified against CUDA 13.4 cublas_api.h:208-212. cuBLAS compute type is per call.
    private const int Compute32FastTf32 = 77;
    private const int Compute32Pedantic = 69;
    private sealed class Capability
    {
        public bool Supported { get; }
        public Capability(CudaAllocator allocator) => Supported = CudaDevice.GetDevice(allocator.DeviceId).ComputeCapabilityMajor >= 8;
    }
    private static readonly ConditionalWeakTable<CudaAllocator, Capability> Capabilities = new();

    /// <summary>Compute result[rows,out] = input[rows,in] × weight[in,out].
    /// Returns false for unsupported GPUs, layouts, or tensor types.</summary>
    public static bool TryLinearTf32(Tensor result, Tensor input, Tensor weight)
        => TryLinear(result, input, weight, tf32: true);

    /// <summary>Compute an embedding projection with pedantic F32 operands and
    /// accumulation selected per call, without changing cuBLAS math mode.</summary>
    public static bool TryLinearFloat32(Tensor result, Tensor input, Tensor weight)
        => TryLinear(result, input, weight, tf32: false);

    private static bool TryLinear(Tensor result, Tensor input, Tensor weight, bool tf32)
    {
        if (result == null || input == null || weight == null ||
            result.ElementType != DType.Float32 || input.ElementType != DType.Float32 || weight.ElementType != DType.Float32 ||
            result.DimensionCount != 2 || input.DimensionCount != 2 || weight.DimensionCount != 2 ||
            !result.IsContiguous() || !input.IsContiguous() || !weight.IsContiguous() ||
            result.Storage is not CudaStorage resultStorage || input.Storage is not CudaStorage inputStorage ||
            weight.Storage is not CudaStorage weightStorage)
            return false;
        long rows = input.Sizes[0], shared = input.Sizes[1], columns = weight.Sizes[1];
        if (rows <= 0 || shared <= 0 || columns <= 0 || rows > int.MaxValue || shared > int.MaxValue || columns > int.MaxValue ||
            weight.Sizes[0] != shared || result.Sizes[0] != rows || result.Sizes[1] != columns)
            return false;
        CudaAllocator allocator = resultStorage.AllocatorImpl;
        if (!ReferenceEquals(allocator, inputStorage.AllocatorImpl) || !ReferenceEquals(allocator, weightStorage.AllocatorImpl))
            return false;
        if (ReferenceEquals(resultStorage, inputStorage) || ReferenceEquals(resultStorage, weightStorage))
            return false;
        allocator.Context.MakeCurrent();
        if (tf32 && !Capabilities.GetValue(allocator, static value => new Capability(value)).Supported)
            return false;
        inputStorage.EnsureDeviceCurrent(); weightStorage.EnsureDeviceCurrent(); resultStorage.EnsureDeviceCurrent();
        allocator.Blas.SetStream(allocator.Stream.Handle);
        float alpha = 1, beta = 0;
        // Row-major operands are transposed column-major views: W[out,in],
        // X[in,rows], Y[out,rows]. No copies or F16 casts are required.
        CublasApi.cublasGemmEx(allocator.Blas.Handle, CublasApi.CUBLAS_OP_N, CublasApi.CUBLAS_OP_N,
            (int)columns, (int)rows, (int)shared, ref alpha,
            weightStorage.DevicePtrAtElement(weight.StorageOffset), CublasApi.CUDA_R_32F, (int)columns,
            inputStorage.DevicePtrAtElement(input.StorageOffset), CublasApi.CUDA_R_32F, (int)shared, ref beta,
            resultStorage.DevicePtrAtElement(result.StorageOffset), CublasApi.CUDA_R_32F, (int)columns,
            tf32 ? Compute32FastTf32 : Compute32Pedantic, CublasApi.CUBLAS_GEMM_DEFAULT).ThrowOnCublasError();
        resultStorage.MarkDeviceModified();
        return true;
    }
}
