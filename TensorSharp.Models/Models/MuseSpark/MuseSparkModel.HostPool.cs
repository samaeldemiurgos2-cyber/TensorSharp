// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using TensorSharp.GGML;

namespace TensorSharp.Models;

public partial class MuseSparkModel
{
    private GgmlContext _sparkHostPoolContext;

    private void ConfigureSparkTensorParallelHostPool()
    {
        if (!IsTensorParallel || _backend != BackendType.GgmlCuda ||
            _allocator is not GgmlAllocator allocator)
            return;
        _sparkHostPoolContext = allocator.Context;
        // Keep both profiles observable; zero is the controlled baseline.
        bool enabled = Environment.GetEnvironmentVariable("TS_MUSE_SPARK_TP_HOST_POOL") != "0";
        bool configured = enabled && _sparkHostPoolContext.ConfigureTensorParallelHostPool();
        var stats = _sparkHostPoolContext.GetHostPoolStatistics();
        Console.WriteLine($"Muse Spark CUDA TP host pool: {(configured ? "scaled" : "baseline")}, " +
            $"up to {stats.MaxPooledBlocks} blocks of {stats.MaxRetainedBlockBytes / (1L << 20)} MiB; allocated on demand.");
    }
}
