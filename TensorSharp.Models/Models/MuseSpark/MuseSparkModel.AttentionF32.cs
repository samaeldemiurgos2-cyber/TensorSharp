// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Threading;
using TensorSharp.GGML;

namespace TensorSharp.Models;

public partial class MuseSparkModel
{
    private long _ownedF32AttentionCalls, _ownedF32PagedAttentionCalls;
    internal long OwnedF32AttentionCalls => Interlocked.Read(ref _ownedF32AttentionCalls);
    internal long OwnedF32PagedAttentionCalls => Interlocked.Read(ref _ownedF32PagedAttentionCalls);

    private void SparkPrefillAttentionF32(Tensor q, Tensor k, Tensor v, Tensor output,
        int heads, int kvHeads, int rows, int total, int start, int window, float scale)
    {
        if (_backend == BackendType.GgmlCuda)
        {
            GgmlBasicOps.MuseSparkPrefillAttentionF32(q, k, v, output, heads, kvHeads, _headDim,
                rows, total, start, window, scale);
            // The binding only returns after native code reports owned CUDA
            // execution (status 2), not a generic successful fallback.
            if (Interlocked.Increment(ref _ownedF32AttentionCalls) == 1)
                Console.WriteLine("Muse Spark owned F32 ordinary attention active: CUDA, F32 Q/K/V operands.");
        }
        else
            GgmlBasicOps.FusedPrefillAttention(q, k, v, output, heads, kvHeads, _headDim,
                rows, total, start, window, scale);
    }

    private void SparkPagedAttentionF32(float[] q, float[] k, float[] v, float[] output,
        int[] starts, int[] lengths, int[] positions, int[] blocks, int[] offsets,
        int sequences, int rows, int heads, int kvHeads, int window)
    {
        float scale = 1f / MathF.Sqrt(_headDim);
        if (_backend == BackendType.GgmlCuda)
        {
            GgmlBasicOps.MuseSparkPagedAttentionF32(q, k, v, output, starts, lengths, positions, blocks, offsets,
                sequences, rows, heads, kvHeads, _headDim, _pagedBlockSize, scale, window);
            if (Interlocked.Increment(ref _ownedF32PagedAttentionCalls) == 1)
                Console.WriteLine("Muse Spark owned F32 paged attention active: CUDA, F32 Q/K/V operands.");
        }
        else
            GgmlBasicOps.PagedAttentionForward(q, k, v, output, starts, lengths, positions, blocks, offsets,
                sequences, rows, heads, kvHeads, _headDim, _pagedBlockSize, scale, window);
    }
}
