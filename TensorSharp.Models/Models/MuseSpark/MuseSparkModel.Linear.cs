// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Diagnostics;
using TensorSharp.GGML;

namespace TensorSharp.Models;

public partial class MuseSparkModel
{
    // CUDA's default F32 matmul can narrow operands to TF32 according to the
    // output width and row count. Uneven head shards then use different math
    // from the original matrix. Keep Spark F32 projections explicitly F32;
    // quantized checkpoints retain their native quantized arithmetic.
    private new Tensor LinearForward(Tensor input, string weightName)
    {
        if (_backend != BackendType.GgmlCuda || _quantWeights.ContainsKey(weightName) ||
            !_weights.TryGetValue(weightName, out var weight))
            return base.LinearForward(input, weightName);

        long started = Stopwatch.GetTimestamp();
        var output = new Tensor(_allocator, DType.Float32, input.Sizes[0], weight.Sizes[0]);
        try
        {
            SparkFloatProjection(output, input, weight);
            _linearTicks += Stopwatch.GetTimestamp() - started;
            return output;
        }
        catch { output.Dispose(); throw; }
    }

    private void SparkFloatProjection(Tensor output, Tensor input, Tensor weight)
    {
        if (_backend == BackendType.GgmlCuda)
            GgmlBasicOps.MuseSparkProjectionFloat32(output, input, weight);
        else
        {
            using var transpose = weight.Transpose();
            Ops.Addmm(output, 0f, output, 1f, input, transpose);
        }
    }
}
