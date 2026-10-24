// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using TensorSharp.GGML;

namespace TensorSharp.Models;

public partial class MuseSparkModel
{
    private (Tensor Query, Tensor Key) PrepareSparkRankQK(Tensor query, Tensor key, Tensor value,
        Tensor gate, int layer, int[] positions, int rank)
    {
        var allocator = _tpGroup.GetAllocator(rank);
        using var packed = PackSparkRankQKVG(allocator, query, key, value, gate);
        Tensor preparedQuery = null, preparedKey = null;
        try
        {
            preparedQuery = new Tensor(allocator, DType.Float32, query.Sizes);
            preparedKey = new Tensor(allocator, DType.Float32, key.Sizes);
            string prefix = _prefixes[layer];
            bool rope = _isSwa[layer];
            GgmlBasicOps.MuseSparkPrepareQK(preparedQuery, preparedKey, packed,
                _sparkTpNorms[prefix + "attn_q_norm.weight"][rank],
                _sparkTpNorms[prefix + "attn_k_norm.weight"][rank], rope ? positions : null,
                _tpQueryCount[rank], _tpKvCount[rank], _headDim, Config.Eps, rope,
                _ropeDim, Config.RopeBase, 1f / Config.RopeScale);
            RecordFusedQKPrep();
            return (preparedQuery, preparedKey);
        }
        catch { preparedQuery?.Dispose(); preparedKey?.Dispose(); throw; }
    }

    // The existing native ABI uploads the entire Q,K,V,gate span. Copy all four
    // initialized projections, even though preparation only consumes Q and K.
    // These are host copies; projection arithmetic and weight ownership stay
    // unchanged. The ordinary Tensor allocator provides bounded pool reuse.
    internal static Tensor PackSparkRankQKVG(IAllocator allocator, Tensor query, Tensor key, Tensor value, Tensor gate)
    {
        if (query == null || key == null || value == null || gate == null ||
            query.DimensionCount != 2 || key.DimensionCount != 2 || value.DimensionCount != 2 || gate.DimensionCount != 2 ||
            query.ElementType != DType.Float32 || key.ElementType != DType.Float32 ||
            value.ElementType != DType.Float32 || gate.ElementType != DType.Float32 ||
            query.Sizes[0] <= 0 || query.Sizes[1] <= 0 || key.Sizes[1] <= 0 ||
            key.Sizes[0] != query.Sizes[0] || value.Sizes[0] != query.Sizes[0] || gate.Sizes[0] != query.Sizes[0] ||
            value.Sizes[1] != key.Sizes[1] || gate.Sizes[1] != query.Sizes[1])
            throw new ArgumentException("Muse Spark Q/K preparation needs matching row-major Q,K,V,gate projections.");
        long queryWidth = query.Sizes[1], keyWidth = key.Sizes[1];
        var packed = new Tensor(allocator, DType.Float32, query.Sizes[0], checked(2 * (queryWidth + keyWidth)));
        try
        {
            CopySparkProjection(packed, query, 0);
            CopySparkProjection(packed, key, queryWidth);
            CopySparkProjection(packed, value, queryWidth + keyWidth);
            CopySparkProjection(packed, gate, queryWidth + 2 * keyWidth);
            return packed;
        }
        catch { packed.Dispose(); throw; }
    }

    private static void CopySparkProjection(Tensor packed, Tensor source, long offset)
    {
        using var destination = packed.Narrow(1, offset, source.Sizes[1]);
        Ops.Copy(destination, source);
    }
}
