// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    /// <summary>
    /// F32 GEMMs that accept IEEE-half operands, by name: <c>CUBLAS_COMPUTE_32F_FAST_16F</c> lets cuBLAS round
    /// both operands to half (10-bit mantissa, inf past 65504) on the shapes it sends to tensor cores, with
    /// F32 accumulation and F32 output. Every F32 GEMM got this implicitly while the shared handle was in
    /// <c>CUBLAS_TENSOR_OP_MATH</c> (see <see cref="CudaCublasHandle.Create"/>); now only a call site that
    /// names it does, and cuBLAS picks the same kernels for it as before.
    /// </summary>
    /// <remarks>
    /// For bulk GEMMs that ggml also runs on half operands and that cost 1.3-3.7x as exact F32: attention
    /// scores and outputs (ggml's flash attention reads F16 K/V and converts Q and P to F16) and vision-encoder
    /// and VAE projections over F16 weights (ggml converts the activations to F16 for an F16 weight). TF32 does
    /// not recover that time on these shapes (1.3-2.3x on sm_86). Never for a GEMM whose output picks (routers,
    /// gates, hyper-connection mixers, indexer inputs) or whose operands can pass 65504. Returns false, having
    /// done nothing, for tensors that are not on CUDA or a layout cuBLAS cannot take directly; the caller then
    /// runs the exact <c>Ops</c> call. Regression tests: InferenceWeb.Tests/CudaF32GemmPrecisionTests.cs.
    /// </remarks>
    public static class CudaHalfOperandGemm
    {
        /// <summary><c>Ops.Addmm</c> with half operands: result = beta * src + alpha * (m1 @ m2).</summary>
        public static bool TryAddmm(Tensor result, float beta, Tensor src, float alpha, Tensor m1, Tensor m2)
            => CudaBlas.TryAddmm(result, beta, src, alpha, m1, m2, CublasApi.CUBLAS_COMPUTE_32F_FAST_16F);

        /// <summary><c>Ops.AddmmBatch</c> with half operands.</summary>
        public static bool TryAddmmBatch(Tensor result, float beta, Tensor src, float alpha, Tensor m1, Tensor m2)
            => CudaBlas.TryAddmmBatch(result, beta, src, alpha, m1, m2, CublasApi.CUBLAS_COMPUTE_32F_FAST_16F);
    }
}
