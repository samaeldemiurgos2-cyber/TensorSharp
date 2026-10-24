// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using TensorSharp.GGML;

namespace TensorSharp.Models;

public partial class MuseSparkModel
{
    private readonly bool _fuseResiduals = Environment.GetEnvironmentVariable("TS_MUSE_SPARK_FUSE_RESIDUALS") == "1";
    internal long FusedResidualCallCount { get; private set; }

    private Tensor PostNormResidual(Tensor stream, Tensor branch, string alpha, string beta, string preScale = null)
    {
        if (_fuseResiduals && _backend is BackendType.GgmlCpu or BackendType.GgmlCuda)
        {
            var result = new Tensor(_allocator, DType.Float32, stream.Sizes);
            try
            {
                GgmlBasicOps.MuseSparkPostNormResidual(result, stream, branch,
                    _weights[alpha], _weights[beta], preScale == null ? null : _weights[preScale], _postNormEps);
                FusedResidualCallCount++;
                if (FusedResidualCallCount == 1)
                    Console.WriteLine($"Muse Spark fused residual active: {_backend}, width {stream.Sizes[1]}, rows {stream.Sizes[0]}.");
                return result;
            }
            catch { result.Dispose(); throw; }
        }
        // Other backends and the default configuration retain the original
        // operations. In particular, post_ffw_norm scales BEFORE the RMS norm.
        using var scaled = preScale == null ? null : ScaleChannels(branch, preScale);
        using var normalized = Norm(scaled ?? branch, _postNormEps);
        return Residual(stream, normalized, alpha, beta);
    }
}
