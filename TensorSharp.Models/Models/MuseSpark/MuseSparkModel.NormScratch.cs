// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using TensorSharp.GGML;

namespace TensorSharp.Models;

public partial class MuseSparkModel
{
    private readonly bool _reuseNormScratch = Environment.GetEnvironmentVariable("TS_MUSE_SPARK_REUSE_NORM_SCRATCH") == "1";
    private bool UseNormScratch => _reuseNormScratch && _backend is BackendType.GgmlCpu or BackendType.GgmlCuda;
    internal long NormScratchReuseCallCount { get; private set; }
    internal long NormScratchFallbackCallCount { get; private set; }
    internal long HeadNormGateCallCount { get; private set; }

    private Tensor ScratchNorm(Tensor input, Tensor gamma, float eps, Tensor gate = null)
    {
        var result = new Tensor(_allocator, DType.Float32, input.Sizes);
        try
        {
            bool reused = GgmlBasicOps.MuseSparkNormGate(result, input, gamma, gate, eps, reuseScratch: true);
            if (gate != null && ++HeadNormGateCallCount == 1)
                Console.WriteLine($"Muse Spark head norm/gate active: {_backend}, head dimension {_headDim}, scratch={(reused ? "reused" : "fresh")}.");
            if (reused)
            {
                if (++NormScratchReuseCallCount == 1)
                    Console.WriteLine($"Muse Spark norm scratch reuse active: {_backend}, width {input.Sizes[1]}, rows {input.Sizes[0]}.");
            }
            else if (++NormScratchFallbackCallCount == 1)
                Console.WriteLine($"Muse Spark norm scratch fallback: {_backend}, fresh allocation; reuse not executed.");
            return result;
        }
        catch { result.Dispose(); throw; }
    }

    private Tensor HeadNormGate(Tensor input, Tensor gate, int rows)
    {
        if (UseNormScratch) return ScratchNorm(input, Ones(_headDim), Config.Eps, gate);
        var normed = HeadNorm(input, rows, Config.NumHeads);
        try { Ops.SigmoidMul(normed, normed, gate); return normed; }
        catch { normed.Dispose(); throw; }
    }
}
