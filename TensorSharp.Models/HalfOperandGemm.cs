// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using TensorSharp.Cuda;

namespace TensorSharp.Models
{
    /// <summary>
    /// <c>Ops.Addmm</c> / <c>Ops.AddmmBatch</c> for the bulk GEMMs that accept IEEE-half operands on direct CUDA
    /// (<see cref="CudaHalfOperandGemm"/>): attention scores and outputs, and vision-encoder and VAE
    /// projections. These ran on half operands before (the old shared cuBLAS handle rounded every F32 GEMM it
    /// sent to tensor cores) and cost 1.3-3.7x as exact F32; naming it here keeps that speed and those numerics
    /// at these sites only. Every other backend, and any layout the cuBLAS call cannot take directly, runs the
    /// plain (exact) <c>Ops</c> call.
    /// </summary>
    internal static class HalfOperandGemm
    {
        public static void Addmm(Tensor result, float beta, Tensor src, float alpha, Tensor m1, Tensor m2)
        {
            if (!CudaHalfOperandGemm.TryAddmm(result, beta, src, alpha, m1, m2))
                Ops.Addmm(result, beta, src, alpha, m1, m2);
        }

        public static void AddmmBatch(Tensor result, float beta, Tensor src, float alpha, Tensor m1, Tensor m2)
        {
            if (!CudaHalfOperandGemm.TryAddmmBatch(result, beta, src, alpha, m1, m2))
                Ops.AddmmBatch(result, beta, src, alpha, m1, m2);
        }
    }
}
