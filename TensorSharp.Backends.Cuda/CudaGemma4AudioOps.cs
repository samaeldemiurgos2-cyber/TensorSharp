using System;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda;

/// <summary>F32 device attention and causal depthwise convolution for Gemma4 audio.</summary>
public static class CudaGemma4AudioOps
{
    /// <summary>Pedantic F32 GEMM for audio precision diagnostics. All operands,
    /// accumulation, and output remain F32; cuBLAS compute type is per call.</summary>
    public static bool TryProjectionFloat32(Tensor result, Tensor input, Tensor weight)
    {
        if (result == null || input == null || weight == null ||
            result.DimensionCount != 2 || input.DimensionCount != 2 || weight.DimensionCount != 2 ||
            !CudaKernelOps.TryGetContiguousFloat(result, out var ys, out var y, out _) ||
            !CudaKernelOps.TryGetContiguousFloat(input, out var xs, out var x, out _) ||
            !CudaKernelOps.TryGetContiguousFloat(weight, out var ws, out var w, out _) ||
            input.Sizes[0] <= 0 || input.Sizes[1] <= 0 || weight.Sizes[1] <= 0 ||
            input.Sizes[0] > int.MaxValue || input.Sizes[1] > int.MaxValue || weight.Sizes[1] > int.MaxValue ||
            weight.Sizes[0] != input.Sizes[1] || result.Sizes[0] != input.Sizes[0] || result.Sizes[1] != weight.Sizes[1] ||
            !ReferenceEquals(ys.AllocatorImpl, xs.AllocatorImpl) || !ReferenceEquals(ys.AllocatorImpl, ws.AllocatorImpl) ||
            ReferenceEquals(ys, xs) || ReferenceEquals(ys, ws)) return false;
        var allocator = ys.AllocatorImpl;
        allocator.Context.MakeCurrent();
        xs.EnsureDeviceCurrent(); ws.EnsureDeviceCurrent(); ys.EnsureDeviceCurrent();
        allocator.Blas.SetStream(allocator.Stream.Handle);
        float alpha = 1, beta = 0;
        // Row-major Y = X*W is column-major Y^T = W^T*X^T.
        CublasApi.cublasGemmEx(allocator.Blas.Handle, CublasApi.CUBLAS_OP_N, CublasApi.CUBLAS_OP_N,
            checked((int)weight.Sizes[1]), checked((int)input.Sizes[0]), checked((int)input.Sizes[1]),
            ref alpha, w, CublasApi.CUDA_R_32F, checked((int)weight.Sizes[1]),
            x, CublasApi.CUDA_R_32F, checked((int)input.Sizes[1]), ref beta,
            y, CublasApi.CUDA_R_32F, checked((int)weight.Sizes[1]),
            69 /* CUBLAS_COMPUTE_32F_PEDANTIC, CUDA 13.4 cublas_api.h */, CublasApi.CUBLAS_GEMM_DEFAULT).ThrowOnCublasError();
        ys.MarkDeviceModified(); return true;
    }

    public static bool TryAttention(Tensor result, Tensor query, Tensor key, Tensor value, Tensor relative,
        int sequence, int heads, int head, int past, int future, float cap)
    {
        if (sequence <= 0 || heads <= 0 || head <= 0 || head > 512 || past < 0 || future < 0 || past + future > 256 ||
            !float.IsFinite(cap) || cap <= 0 || (long)sequence * heads * head > int.MaxValue ||
            !CudaKernelOps.TryGetContiguousFloat(result, out var ys, out var y, out _) ||
            !CudaKernelOps.TryGetContiguousFloat(query, out var qs, out var q, out _) ||
            !CudaKernelOps.TryGetContiguousFloat(key, out var ks, out var k, out _) ||
            !CudaKernelOps.TryGetContiguousFloat(value, out var vs, out var v, out _) ||
            !CudaKernelOps.TryGetContiguousFloat(relative, out var rs, out var r, out _) ||
            result.DimensionCount != 2 || query.DimensionCount != 2 || key.DimensionCount != 2 || value.DimensionCount != 2 || relative.DimensionCount != 2 ||
            query.Sizes[0] != sequence || query.Sizes[1] != (long)heads * head || result.Sizes[0] != sequence || result.Sizes[1] != query.Sizes[1] ||
            key.Sizes[0] != sequence || key.Sizes[1] != query.Sizes[1] || value.Sizes[0] != sequence || value.Sizes[1] != query.Sizes[1] ||
            relative.Sizes[0] != past + future + 1 || relative.Sizes[1] != query.Sizes[1] ||
            !ReferenceEquals(ys.AllocatorImpl, qs.AllocatorImpl) || !ReferenceEquals(ys.AllocatorImpl, ks.AllocatorImpl) ||
            !ReferenceEquals(ys.AllocatorImpl, vs.AllocatorImpl) || !ReferenceEquals(ys.AllocatorImpl, rs.AllocatorImpl) ||
            ReferenceEquals(ys, qs) || ReferenceEquals(ys, ks) || ReferenceEquals(ys, vs) || ReferenceEquals(ys, rs)) return false;
        var allocator = ys.AllocatorImpl;
        if (!CudaKernelOps.TryGetKernels(allocator, out var kernels) || !kernels.HasGemma4AudioKernels) return false;
        allocator.Context.MakeCurrent(); qs.EnsureDeviceCurrent(); ks.EnsureDeviceCurrent(); vs.EnsureDeviceCurrent(); rs.EnsureDeviceCurrent(); ys.EnsureDeviceCurrent();
        kernels.LaunchGemma4AudioAttention(q, k, v, r, y, sequence, heads, head, past, future, cap, allocator.Stream.Handle);
        ys.MarkDeviceModified(); return true;
    }

    public static bool TryDepthwise(Tensor result, Tensor input, Tensor weight, int sequence, int channels, int kernel)
    {
        if (sequence <= 0 || channels <= 0 || kernel <= 0 || kernel > 256 || (long)sequence * channels > int.MaxValue ||
            !CudaKernelOps.TryGetContiguousFloat(result, out var ys, out var y, out _) ||
            !CudaKernelOps.TryGetContiguousFloat(input, out var xs, out var x, out _) ||
            !CudaKernelOps.TryGetContiguousFloat(weight, out var ws, out var w, out _) ||
            result.DimensionCount != 2 || input.DimensionCount != 2 || weight.DimensionCount != 2 ||
            result.Sizes[0] != sequence || result.Sizes[1] != channels || input.Sizes[0] != sequence || input.Sizes[1] != channels ||
            weight.Sizes[0] != channels || weight.Sizes[1] != kernel ||
            !ReferenceEquals(ys.AllocatorImpl, xs.AllocatorImpl) || !ReferenceEquals(ys.AllocatorImpl, ws.AllocatorImpl) ||
            ReferenceEquals(ys, xs) || ReferenceEquals(ys, ws)) return false;
        var allocator = ys.AllocatorImpl;
        if (!CudaKernelOps.TryGetKernels(allocator, out var kernels) || !kernels.HasGemma4AudioKernels) return false;
        allocator.Context.MakeCurrent(); xs.EnsureDeviceCurrent(); ws.EnsureDeviceCurrent(); ys.EnsureDeviceCurrent();
        kernels.LaunchGemma4AudioDepthwise(x, w, y, sequence, channels, kernel, allocator.Stream.Handle);
        ys.MarkDeviceModified(); return true;
    }
}
