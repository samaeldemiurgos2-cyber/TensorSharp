// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using TensorSharp.GGML;
using TensorSharp.Runtime.Paged;
using TensorSharp.Runtime.Scheduling;

namespace TensorSharp.Models;

public partial class MuseSparkModel : IBatchedPagedModel
{
    private float[][] _pagedKeys, _pagedValues;
    private int _pagedBlocks, _pagedBlockSize;

    // These buffers contain F32 values. Quantized/F16 caches retain the ordinary
    // snapshot scheduler instead of silently changing their requested precision.
    // The native executor serves concurrency through its own sequence slots.
    public bool BatchedForwardAvailable => !UsesNativeExecutor && _kvCacheDtype.ToDType() == DType.Float32 && !HasPendingVisionEmbeddings;

    public IReadOnlyList<float[]> ForwardBatch(BatchedForwardContext ctx)
    {
        if (!_sparkPhaseTimingEnabled) return ForwardBatchCore(ctx);
        return TimedForwardBatch(ctx);
    }

    private IReadOnlyList<float[]> TimedForwardBatch(BatchedForwardContext ctx)
    {
        return RecordSparkForward("packed", ctx?.Positions.Count ?? 0, -1, () => ForwardBatchCore(ctx));
    }

    private IReadOnlyList<float[]> ForwardBatchCore(BatchedForwardContext ctx)
    {
        ArgumentNullException.ThrowIfNull(ctx);
        if (!BatchedForwardAvailable)
            throw new NotSupportedException("Muse Spark paged batching requires F32 KV and text-only input.");
        int sequences = ctx.Sequences.Count;
        if (sequences == 0) return Array.Empty<float[]>();
        int rows = ctx.Positions.Count;
        if (ctx.QueryStartLoc.Count != sequences + 1 || ctx.QueryStartLoc[0] != 0 ||
            ctx.QueryStartLoc[^1] != rows || ctx.SlotMapping.Count != rows ||
            ctx.NumScheduledTokens.Count != sequences || ctx.BlockTables.Length != sequences)
            throw new ArgumentException("Invalid Muse Spark packed batch dimensions.", nameof(ctx));
        int blockSize = ctx.Sequences[0].BlockTable.BlockSize;
        int requiredBlocks = 0;
        var lengths = new int[sequences];
        var offsets = new int[sequences];
        int blockCount = 0;
        for (int s = 0; s < sequences; s++)
        {
            int scheduled = ctx.NumScheduledTokens[s];
            lengths[s] = checked(ctx.Sequences[s].NumComputedTokens + scheduled);
            if (scheduled <= 0 || ctx.QueryStartLoc[s + 1] - ctx.QueryStartLoc[s] != scheduled ||
                lengths[s] > _maxContextLength || ctx.Sequences[s].BlockTable.BlockSize != blockSize)
                throw new ArgumentException("Invalid Muse Spark sequence length or block size.", nameof(ctx));
            offsets[s] = blockCount;
            blockCount = checked(blockCount + ctx.BlockTables[s].Length);
            foreach (int block in ctx.BlockTables[s])
            {
                if (block < 0) throw new ArgumentException("Negative physical block ID.", nameof(ctx));
                requiredBlocks = Math.Max(requiredBlocks, checked(block + 1));
            }
        }
        EnsureSparkPagedBuffers(requiredBlocks, blockSize);
        var blocks = new int[blockCount];
        for (int s = 0; s < sequences; s++) ctx.BlockTables[s].CopyTo(blocks, offsets[s]);
        int[] tokens = ctx.OverrideFlatTokens ?? new int[rows];
        if (tokens.Length != rows) throw new ArgumentException("Packed token length differs from positions.", nameof(ctx));
        if (ctx.OverrideFlatTokens == null)
            for (int s = 0; s < sequences; s++)
                for (int i = 0; i < ctx.NumScheduledTokens[s]; i++)
                    tokens[ctx.QueryStartLoc[s] + i] = ctx.Sequences[s].TokenAt(ctx.Sequences[s].NumComputedTokens + i);
        int[] positions = ctx.Positions.ToArray(), starts = ctx.QueryStartLoc.ToArray(), slots = ctx.SlotMapping.ToArray();
        foreach (int slot in slots)
            if (slot < 0 || slot >= checked(requiredBlocks * blockSize))
                throw new ArgumentException("Invalid Muse Spark KV slot.", nameof(ctx));

        bool pinExecution = IsGgmlBackend && (LayerSplitDegree > 1 || IsTensorParallel);
        int priorRank = pinExecution ? GgmlBasicOps.GetActiveRank() : 0;
        bool priorPin = pinExecution && GgmlBasicOps.PinRank(0);
        Tensor hidden = null;
        _forwardSw.Start();
        try
        {
            _sparkPhaseFrame?.Next(MuseSparkPhase.Embedding);
            using (var embedding = Embedding(tokens)) hidden = Norm(embedding, Config.Eps);
            using var qPositions = SparkBatchPositions(positions, Config.NumHeads);
            using var kPositions = SparkBatchPositions(positions, Config.NumKVHeads);
            for (int layer = 0; layer < Config.NumLayers; layer++)
            {
                _sparkPhaseFrame?.Next(MuseSparkPhase.LayerNorms);
                if (LayerSplitDegree > 1 && GgmlBasicOps.GetActiveRank() != LayerRank(layer))
                {
                    SyncTensorHostCache(hidden);
                    GgmlBasicOps.SetActiveRank(LayerRank(layer));
                }
                string p = _prefixes[layer];
                using var preAttention = Norm(hidden, Config.Eps, p + "attn_norm.weight");
                _sparkPhaseFrame?.Next(MuseSparkPhase.AttentionSingle);
                using var attention = SparkBatchAttention(preAttention, layer, rows, sequences,
                    qPositions, kPositions, positions, starts, slots, lengths, ctx.BlockTables, blocks, offsets);
                _sparkPhaseFrame?.Next(MuseSparkPhase.Residual);
                using var ffnInput = PostNormResidual(hidden, attention, p + "attn_resid_alpha.weight", p + "attn_resid_beta.weight");
                hidden.Dispose(); hidden = null;
                _sparkPhaseFrame?.Next(MuseSparkPhase.LayerNorms);
                using var ffnNorm = Norm(ffnInput, Config.Eps, p + "ffn_norm.weight");
                using var moe = MoE(ffnNorm, layer, rows);
                _sparkPhaseFrame?.Next(MuseSparkPhase.Residual);
                hidden = PostNormResidual(ffnInput, moe, p + "ffn_resid_alpha.weight", p + "ffn_resid_beta.weight",
                    p + "post_ffw_norm.weight");
                _sparkPhaseFrame?.Next(MuseSparkPhase.Cleanup);
            }
            _sparkPhaseFrame?.Next(MuseSparkPhase.OutputNorm);
            using var final = Norm(hidden, Config.Eps, "output_norm.weight");
            float[] finalFlat = TensorToFloatArray(final);
            if (ctx.CaptureHiddenAll != null)
                finalFlat.CopyTo(ctx.CaptureHiddenAll, 0);
            if (ctx.CaptureLogitsAll != null)
            {
                _sparkPhaseFrame?.Next(MuseSparkPhase.OutputHead);
                using var everyLogit = LinearForward(final, "output.weight");
                float[] all = TensorToFloatArray(everyLogit);
                ApplySparkLogitScale(all);
                all.CopyTo(ctx.CaptureLogitsAll, 0);
            }
            float[] lastRows = PagedKvBatchOps.GatherLastTokenPerSeq(finalFlat, Config.HiddenSize, starts, sequences);
            using var last = CreateFloatTensor(lastRows, sequences, Config.HiddenSize);
            _sparkPhaseFrame?.Next(MuseSparkPhase.OutputHead);
            using var output = LinearForward(last, "output.weight");
            float[] logits = TensorToFloatArray(output);
            ApplySparkLogitScale(logits);
            var result = new float[sequences][];
            for (int s = 0; s < sequences; s++) result[s] = logits.AsSpan(s * Config.VocabSize, Config.VocabSize).ToArray();
            _sparkPhaseFrame?.Next(MuseSparkPhase.Cleanup);
            return result;
        }
        finally
        {
            _sparkPhaseFrame?.Next(MuseSparkPhase.Cleanup);
            hidden?.Dispose();
            if (pinExecution) GgmlBasicOps.RestoreRankPin(priorPin, priorRank);
            _forwardCount++;
            _forwardSw.Stop();
        }
    }

    private void ApplySparkLogitScale(float[] logits)
    {
        for (int i = 0; i < logits.Length; i++)
        {
            float value = logits[i] * _logitScale;
            logits[i] = _softcap > 0 ? _softcap * MathF.Tanh(value / _softcap) : value;
        }
    }

    private Tensor SparkBatchAttention(Tensor input, int layer, int rows, int sequences,
        Tensor qPositions, Tensor kPositions, int[] positions, int[] starts, int[] slots,
        int[] lengths, int[][] tables, int[] blocks, int[] offsets)
    {
        if (IsTensorParallel)
            return SparkTpBatchAttention(input, layer, rows, sequences, positions, starts, slots, lengths, tables, blocks, offsets);
        string p = _prefixes[layer];
        var projections = AttentionProjections(input, layer, 0, positions, out bool preparedQK);
        using var q0 = projections.Query;
        using var k0 = projections.Key;
        using var v = projections.Value;
        using var gate = projections.Gate;
        using var qn = preparedQK ? q0.CopyRef() : HeadNorm(q0, rows, Config.NumHeads, p + "attn_q_norm.weight");
        using var kn = preparedQK ? k0.CopyRef() : HeadNorm(k0, rows, Config.NumKVHeads, p + "attn_k_norm.weight");
        using var q = !preparedQK && _isSwa[layer] ? SparkBatchRope(qn, qPositions, rows, Config.NumHeads) : qn.CopyRef();
        using var k = !preparedQK && _isSwa[layer] ? SparkBatchRope(kn, kPositions, rows, Config.NumKVHeads) : kn.CopyRef();
        PagedKvBatchOps.ScatterKv(TensorToFloatArray(k), TensorToFloatArray(v), _pagedKeys[layer], _pagedValues[layer],
            slots, rows, Config.NumKVHeads, _headDim, _pagedBlockSize);
        float[] qFlat = TensorToFloatArray(q), attention = new float[checked(rows * Config.NumHeads * _headDim)];
        int window = _isSwa[layer] ? _slidingWindow : 0;
        if (IsGgmlBackend)
            SparkPagedAttentionF32(qFlat, _pagedKeys[layer], _pagedValues[layer], attention,
                starts, lengths, positions, blocks, offsets, sequences, rows, Config.NumHeads, Config.NumKVHeads,
                window);
        else
            ManagedPagedAttention.Forward(qFlat, _pagedKeys[layer], _pagedValues[layer], attention, rows,
                Config.NumHeads, Config.NumKVHeads, _headDim, _pagedBlockSize, starts, lengths, positions,
                tables, sequences, 1f / MathF.Sqrt(_headDim), causal: true, slidingWindow: window);
        using var flat = CreateFloatTensor(attention, rows, Config.NumHeads * _headDim);
        using var normed = HeadNormGate(flat, gate, rows);
        return LinearForward(normed, p + "attn_output.weight");
    }

    private Tensor SparkBatchPositions(int[] positions, int heads)
    {
        var expanded = new int[checked(positions.Length * heads)];
        for (int row = 0; row < positions.Length; row++)
            expanded.AsSpan(row * heads, heads).Fill(positions[row]);
        return CreateIntTensorOn(_allocator, expanded, expanded.Length);
    }

    private Tensor SparkBatchRope(Tensor input, Tensor positions, int rows, int heads)
    {
        using var view = input.View(1, rows, heads, _headDim);
        using var rotated = Ops.RoPEEx(null, view, positions, _ropeDim, 0, 0,
            Config.RopeBase, 1f / Config.RopeScale, 0f, 1f, 0f, 0f);
        return rotated.View(rows, heads * _headDim);
    }

    private void EnsureSparkPagedBuffers(int blocks, int blockSize)
    {
        if (IsTensorParallel) { EnsureSparkTpPagedBuffers(blocks, blockSize); return; }
        if (_pagedKeys != null && _pagedBlocks >= blocks && _pagedBlockSize == blockSize) return;
        int capacity = Math.Max(blocks, checked(_pagedBlocks * 2));
        var keys = new float[Config.NumLayers][];
        var values = new float[Config.NumLayers][];
        for (int layer = 0; layer < Config.NumLayers; layer++)
        {
            keys[layer] = PagedKvBatchOps.AllocateLayerBuffer(capacity, blockSize, Config.NumKVHeads, _headDim);
            values[layer] = PagedKvBatchOps.AllocateLayerBuffer(capacity, blockSize, Config.NumKVHeads, _headDim);
            if (_pagedKeys != null && blockSize == _pagedBlockSize)
            {
                _pagedKeys[layer].CopyTo(keys[layer], 0);
                _pagedValues[layer].CopyTo(values[layer], 0);
            }
        }
        _pagedKeys = keys; _pagedValues = values; _pagedBlocks = capacity; _pagedBlockSize = blockSize;
    }
}
