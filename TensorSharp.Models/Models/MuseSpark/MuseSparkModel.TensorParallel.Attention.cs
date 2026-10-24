// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using TensorSharp.GGML;
using TensorSharp.Runtime.Paged;

namespace TensorSharp.Models;

public partial class MuseSparkModel
{
    private Tensor SparkRankHeadNorm(Tensor input, int rows, int heads, int rank, string weight = null, Tensor gate = null)
    {
        var output = new Tensor(_tpGroup.GetAllocator(rank), DType.Float32, rows, heads * _headDim);
        Tensor gamma = weight == null ? _sparkTpHeadOnes[rank] : _sparkTpNorms[weight][rank];
        try
        {
            if (UseNormScratch)
                GgmlBasicOps.MuseSparkNormGate(output, input, gamma, gate, Config.Eps, reuseScratch: true);
            else
            {
                using var source = input.View(rows * heads, _headDim);
                using var result = output.View(rows * heads, _headDim);
                Ops.RMSNorm(result, source, gamma, null, Config.Eps);
                if (gate != null) Ops.SigmoidMul(output, output, gate);
            }
            return output;
        }
        catch { output.Dispose(); throw; }
    }

    private Tensor SparkRankRope(Tensor input, int[] positions, int rows, int heads, int rank)
    {
        var repeated = new int[checked(rows * heads)];
        for (int row = 0; row < rows; row++) repeated.AsSpan(row * heads, heads).Fill(positions[row]);
        using var pos = CreateIntTensorOn(_tpGroup.GetAllocator(rank), repeated, repeated.Length);
        using var view = input.View(1, rows, heads, _headDim);
        using var result = new Tensor(_tpGroup.GetAllocator(rank), DType.Float32, 1, rows, heads, _headDim);
        Ops.RoPEEx(result, view, pos, _ropeDim, 0, 0, Config.RopeBase, 1f / Config.RopeScale, 0f, 1f, 0f, 0f);
        return result.View(rows, heads * _headDim);
    }

    private Tensor TensorParallelAttention(Tensor input, int layer, int rows, int start)
    {
        var positions = new int[rows];
        for (int row = 0; row < rows; row++) positions[row] = start + row;
        var partials = new Tensor[TpDegree];
        try
        {
            _sparkPhaseFrame?.Next(MuseSparkPhase.AttentionFanout);
            _tpGroup.RunPerRank(rank =>
            {
                using var timing = _sparkPhaseFrame?.BeginRank(rank, MuseSparkRankPhase.AttentionInputCopy);
                string p = _prefixes[layer];
                int qHeads = _tpQueryCount[rank], kvHeads = _tpKvCount[rank];
                using var local = SparkRankInput(input, rank);
                timing?.Next(MuseSparkRankPhase.AttentionProjections);
                using var q0 = SparkRankLinear(local, p + "attn_q.weight", rank);
                using var k0 = SparkRankLinear(local, p + "attn_k.weight", rank);
                using var value = SparkRankLinear(local, p + "attn_v.weight", rank);
                using var gate = SparkRankLinear(local, p + "attn_gate.weight", rank);
                timing?.Next(MuseSparkRankPhase.AttentionQkNormRope);
                bool prepare = UseFusedQKPrep;
                using var qn = prepare ? null : SparkRankHeadNorm(q0, rows, qHeads, rank, p + "attn_q_norm.weight");
                using var kn = prepare ? null : SparkRankHeadNorm(k0, rows, kvHeads, rank, p + "attn_k_norm.weight");
                var prepared = prepare ? PrepareSparkRankQK(q0, k0, value, gate, layer, positions, rank) : default;
                using var q = prepare ? prepared.Query : _isSwa[layer] ? SparkRankRope(qn, positions, rows, qHeads, rank) : qn.CopyRef();
                using var k = prepare ? prepared.Key : _isSwa[layer] ? SparkRankRope(kn, positions, rows, kvHeads, rank) : kn.CopyRef();
                timing?.Next(MuseSparkRankPhase.AttentionKvAndAttention);
                using var qh = ReshapeToHeads(q, qHeads, rows, _headDim);
                using var kh = ReshapeToHeads(k, kvHeads, rows, _headDim);
                using var vh = ReshapeToHeads(value, kvHeads, rows, _headDim);
                // Disjoint whole-head views preserve canonical host KV snapshots.
                // Their second stride remains this layer's actual physical capacity.
                using var kc = _kvCacheK[layer].Narrow(0, _tpKvFirst[rank], kvHeads);
                using var vc = _kvCacheV[layer].Narrow(0, _tpKvFirst[rank], kvHeads);
                CopyToCache(kc, kh, start, rows); CopyToCache(vc, vh, start, rows);
                using var attention = SparkRankAttentionHeads(qh, kc, vc, layer, rank, rows, start);
                using var flat = ReshapeFromHeads(attention, qHeads, rows, _headDim);
                timing?.Next(MuseSparkRankPhase.AttentionGateOutput);
                using var gated = SparkRankHeadNorm(flat, rows, qHeads, rank, gate: gate);
                partials[rank] = SparkRankLinear(gated, p + "attn_output.weight", rank);
                timing?.Next(MuseSparkRankPhase.Cleanup);
            });
            // The full output projection is reconstructed before the nonlinear
            // post-attention RMSNorm/residual in the common decoder.
            _sparkPhaseFrame?.Next(MuseSparkPhase.AttentionReduce);
            return SparkReduce(partials);
        }
        catch { foreach (var partial in partials) partial?.Dispose(); throw; }
    }

    private unsafe Tensor SparkRankCompactCache(Tensor cache, int rank, int total)
    {
        var compact = new Tensor(_tpGroup.GetAllocator(rank), DType.Float32, cache.Sizes[0], total, _headDim);
        try
        {
            if (cache.ElementType == DType.Float32)
            {
                using var view = cache.Narrow(1, 0, total);
                Ops.Copy(compact, view);
            }
            else
            {
                ushort* source = TensorComputePrimitives.GetHalfPointer(cache);
                float* destination = GetFloatPtr(compact);
                int capacity = (int)cache.Sizes[1];
                for (int head = 0; head < cache.Sizes[0]; head++)
                    for (int index = 0; index < total * _headDim; index++)
                        destination[head * total * _headDim + index] =
                            (float)BitConverter.UInt16BitsToHalf(source[head * capacity * _headDim + index]);
                InvalidateTensorDeviceCache(compact);
            }
            return compact;
        }
        catch { compact.Dispose(); throw; }
    }

    private unsafe Tensor SparkRankAttentionHeads(Tensor q, Tensor kc, Tensor vc, int layer, int rank, int rows, int start)
    {
        int qHeads = _tpQueryCount[rank], kvHeads = _tpKvCount[rank];
        int total = start + rows, window = _isSwa[layer] ? _slidingWindow : 0;
        var output = new Tensor(_tpGroup.GetAllocator(rank), DType.Float32, qHeads, rows, _headDim);
        try
        {
            float scale = 1f / MathF.Sqrt(_headDim);
            if (IsGgmlBackend && kc.ElementType == DType.Float16)
            {
                int first = window > 0 ? Math.Max(0, start - window + 1) : 0;
                using var kw = kc.Narrow(1, first, total - first);
                using var vw = vc.Narrow(1, first, total - first);
                GgmlBasicOps.FusedPrefillAttentionF16KV(q, kw, vw, output, qHeads, kvHeads, _headDim,
                    rows, total - first, (int)kc.Sizes[1], start - first, window, scale, museSpark2DUploads: Use2DKvUploads);
                InvalidateTensorDeviceCache(output);
                using var flat = output.View(rows, qHeads * _headDim);
                var reshaped = ReshapeToHeads(flat, qHeads, rows, _headDim);
                output.Dispose();
                return reshaped;
            }
            using var k = SparkRankCompactCache(kc, rank, total);
            using var v = SparkRankCompactCache(vc, rank, total);
            if (IsGgmlBackend)
            {
                SparkPrefillAttentionF32(q, k, v, output, qHeads, kvHeads,
                    rows, total, start, window, scale);
                InvalidateTensorDeviceCache(output);
                using var flat = output.View(rows, qHeads * _headDim);
                var reshaped = ReshapeToHeads(flat, qHeads, rows, _headDim);
                output.Dispose();
                return reshaped;
            }
            float* qp = GetFloatPtr(q), kp = GetFloatPtr(k), vp = GetFloatPtr(v), op = GetFloatPtr(output);
            var scores = new float[total];
            int group = qHeads / kvHeads;
            for (int head = 0; head < qHeads; head++)
                for (int row = 0; row < rows; row++)
                {
                    int first = window > 0 ? Math.Max(0, start + row - window + 1) : 0;
                    int end = start + row + 1;
                    float max = float.NegativeInfinity;
                    for (int key = first; key < end; key++)
                    {
                        float dot = 0;
                        for (int d = 0; d < _headDim; d++)
                            dot += qp[(head * rows + row) * _headDim + d] * kp[((head / group) * total + key) * _headDim + d];
                        scores[key] = dot * scale; max = MathF.Max(max, scores[key]);
                    }
                    float sum = 0;
                    for (int key = first; key < end; key++) { scores[key] = MathF.Exp(scores[key] - max); sum += scores[key]; }
                    for (int d = 0; d < _headDim; d++)
                    {
                        float value = 0;
                        for (int key = first; key < end; key++)
                            value += scores[key] / sum * vp[((head / group) * total + key) * _headDim + d];
                        op[(head * rows + row) * _headDim + d] = value;
                    }
                }
            return output;
        }
        catch { output.Dispose(); throw; }
    }

    private void EnsureSparkTpPagedBuffers(int blocks, int blockSize)
    {
        if (_sparkTpPagedKeys != null && _pagedBlocks >= blocks && _pagedBlockSize == blockSize) return;
        int capacity = Math.Max(blocks, checked(_pagedBlocks * 2));
        var keys = new float[Config.NumLayers][][];
        var values = new float[Config.NumLayers][][];
        for (int layer = 0; layer < Config.NumLayers; layer++)
        {
            keys[layer] = new float[TpDegree][]; values[layer] = new float[TpDegree][];
            for (int rank = 0; rank < TpDegree; rank++)
            {
                keys[layer][rank] = PagedKvBatchOps.AllocateLayerBuffer(capacity, blockSize, _tpKvCount[rank], _headDim);
                values[layer][rank] = PagedKvBatchOps.AllocateLayerBuffer(capacity, blockSize, _tpKvCount[rank], _headDim);
                if (_sparkTpPagedKeys != null && blockSize == _pagedBlockSize)
                {
                    _sparkTpPagedKeys[layer][rank].CopyTo(keys[layer][rank], 0);
                    _sparkTpPagedValues[layer][rank].CopyTo(values[layer][rank], 0);
                }
            }
        }
        _sparkTpPagedKeys = keys; _sparkTpPagedValues = values;
        _pagedBlocks = capacity; _pagedBlockSize = blockSize;
    }

    private Tensor SparkTpBatchAttention(Tensor input, int layer, int rows, int sequences,
        int[] positions, int[] starts, int[] slots, int[] lengths, int[][] tables, int[] blocks, int[] offsets)
    {
        var partials = new Tensor[TpDegree];
        try
        {
            _sparkPhaseFrame?.Next(MuseSparkPhase.AttentionFanout);
            _tpGroup.RunPerRank(rank =>
            {
                using var timing = _sparkPhaseFrame?.BeginRank(rank, MuseSparkRankPhase.AttentionInputCopy);
                string p = _prefixes[layer];
                int qHeads = _tpQueryCount[rank], kvHeads = _tpKvCount[rank];
                using var local = SparkRankInput(input, rank);
                timing?.Next(MuseSparkRankPhase.AttentionProjections);
                using var q0 = SparkRankLinear(local, p + "attn_q.weight", rank);
                using var k0 = SparkRankLinear(local, p + "attn_k.weight", rank);
                using var value = SparkRankLinear(local, p + "attn_v.weight", rank);
                using var gate = SparkRankLinear(local, p + "attn_gate.weight", rank);
                timing?.Next(MuseSparkRankPhase.AttentionQkNormRope);
                bool prepare = UseFusedQKPrep;
                using var qn = prepare ? null : SparkRankHeadNorm(q0, rows, qHeads, rank, p + "attn_q_norm.weight");
                using var kn = prepare ? null : SparkRankHeadNorm(k0, rows, kvHeads, rank, p + "attn_k_norm.weight");
                var prepared = prepare ? PrepareSparkRankQK(q0, k0, value, gate, layer, positions, rank) : default;
                using var q = prepare ? prepared.Query : _isSwa[layer] ? SparkRankRope(qn, positions, rows, qHeads, rank) : qn.CopyRef();
                using var k = prepare ? prepared.Key : _isSwa[layer] ? SparkRankRope(kn, positions, rows, kvHeads, rank) : kn.CopyRef();
                timing?.Next(MuseSparkRankPhase.AttentionKvAndAttention);
                float[] keys = _sparkTpPagedKeys[layer][rank], values = _sparkTpPagedValues[layer][rank];
                PagedKvBatchOps.ScatterKv(TensorToFloatArray(k), TensorToFloatArray(value), keys, values,
                    slots, rows, kvHeads, _headDim, _pagedBlockSize);
                float[] qFlat = TensorToFloatArray(q), attention = new float[checked(rows * qHeads * _headDim)];
                int window = _isSwa[layer] ? _slidingWindow : 0;
                if (IsGgmlBackend)
                    SparkPagedAttentionF32(qFlat, keys, values, attention, starts, lengths, positions, blocks,
                        offsets, sequences, rows, qHeads, kvHeads, window);
                else
                    ManagedPagedAttention.Forward(qFlat, keys, values, attention, rows, qHeads, kvHeads, _headDim,
                        _pagedBlockSize, starts, lengths, positions, tables, sequences, 1f / MathF.Sqrt(_headDim),
                        causal: true, slidingWindow: window);
                using var flat = new Tensor(_tpGroup.GetAllocator(rank), DType.Float32, rows, qHeads * _headDim);
                flat.CopyFrom(attention);
                timing?.Next(MuseSparkRankPhase.AttentionGateOutput);
                using var gated = SparkRankHeadNorm(flat, rows, qHeads, rank, gate: gate);
                partials[rank] = SparkRankLinear(gated, p + "attn_output.weight", rank);
                timing?.Next(MuseSparkRankPhase.Cleanup);
            });
            _sparkPhaseFrame?.Next(MuseSparkPhase.AttentionReduce);
            return SparkReduce(partials);
        }
        catch { foreach (var partial in partials) partial?.Dispose(); throw; }
    }
}
