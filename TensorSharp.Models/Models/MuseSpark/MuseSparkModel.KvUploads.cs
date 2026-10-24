// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;

namespace TensorSharp.Models;

public partial class MuseSparkModel
{
    private readonly bool _kvUpload2D = Environment.GetEnvironmentVariable("TS_MUSE_SPARK_KV_UPLOAD_2D") == "1";
    private bool Use2DKvUploads => _kvUpload2D && _backend is BackendType.GgmlCpu or BackendType.GgmlCuda;
    internal long Kv2DUploadCallCount { get; private set; }
    private void Record2DKvUpload(int rows)
    {
        if (!Use2DKvUploads) return;
        if (++Kv2DUploadCallCount == 1)
            Console.WriteLine($"Muse Spark 2D K/V uploads active: {_backend}, F16, rows {rows}.");
    }
}
