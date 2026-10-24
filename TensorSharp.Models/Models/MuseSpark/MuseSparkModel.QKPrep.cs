// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Threading;
using TensorSharp.GGML;

namespace TensorSharp.Models;

public partial class MuseSparkModel
{
    private readonly bool _fuseQKPrep = Environment.GetEnvironmentVariable("TS_MUSE_SPARK_FUSE_QK_PREP") == "1";
    private long _fusedQKPrepCallCount;
    internal long FusedQKPrepCallCount => Interlocked.Read(ref _fusedQKPrepCallCount);
    private bool UseFusedQKPrep => _fuseQKPrep && _backend is BackendType.GgmlCpu or BackendType.GgmlCuda;

    private void PrepareFusedQK(Tensor query, Tensor key, Tensor packed, int layer, int start, int[] positions)
    {
        bool rope = _isSwa[layer];
        if (rope && positions == null)
        {
            positions = new int[checked((int)packed.Sizes[0])];
            for (int row = 0; row < positions.Length; row++) positions[row] = checked(start + row);
        }
        string p = _prefixes[layer];
        GgmlBasicOps.MuseSparkPrepareQK(query, key, packed,
            _weights[p + "attn_q_norm.weight"], _weights[p + "attn_k_norm.weight"], rope ? positions : null,
            Config.NumHeads, Config.NumKVHeads, _headDim, Config.Eps, rope,
            _ropeDim, Config.RopeBase, 1f / Config.RopeScale);
        RecordFusedQKPrep();
    }

    private void RecordFusedQKPrep()
    {
        // TP workers prepare disjoint rank outputs concurrently. Count only
        // successful synchronous calls and print one actual-use diagnostic.
        if (Interlocked.Increment(ref _fusedQKPrepCallCount) == 1)
            Console.WriteLine($"Muse Spark fused Q/K preparation active: {_backend}, head dimension {_headDim}.");
    }
}
