// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.Linq;

namespace TensorSharp.Models;

public partial class MuseSparkModel
{
    private bool[] _fusedAttention;
    internal int FusedAttentionLayerCount => _fusedAttention?.Count(x => x) ?? 0;

    private void FuseAttentionProjections()
    {
        _fusedAttention = new bool[Config.NumLayers];
        if (IsTensorParallel) return; // Retain exact independent shard boundaries and quantization types.
        if (Environment.GetEnvironmentVariable("TS_MUSE_SPARK_FUSE_QKVG") == "0") return;
        for (int layer = 0; layer < Config.NumLayers; layer++)
        {
            string p = _prefixes[layer], fusedName = p + "attn_qkvg.weight";
            string[] names = [p + "attn_q.weight", p + "attn_k.weight", p + "attn_v.weight", p + "attn_gate.weight"];
            var quant = names.Select(name => _quantWeights.GetValueOrDefault(name)).ToArray();
            if (quant.All(w => w != null) && quant.All(w => w.GgmlType == quant[0].GgmlType &&
                    w.Ne0 == quant[0].Ne0 && w.Scale == quant[0].Scale))
            {
                // Respect the shared memory policy when GGUF tensors are not
                // adjacent. Do this before device preload, then remove sources
                // so only one copy of each matrix occupies GPU memory.
                if (!TryCreateFusedQuantizedWeight(separatePathAvailable: true, out var fused, quant)) continue;
                fused.Scale = quant[0].Scale;
                _quantWeights.Add(fusedName, fused);
                for (int i = 0; i < names.Length; i++) { _quantWeights.Remove(names[i]); quant[i].Dispose(); }
                _fusedAttention[layer] = true;
            }
            else if (names.All(name => _weights.ContainsKey(name)) && AllowWeightFusionCopies)
            {
                int width = 2 * (Config.NumHeads + Config.NumKVHeads) * _headDim;
                var fused = new Tensor(_allocator, DType.Float32, width, Config.HiddenSize);
                try
                {
                    int offset = 0;
                    foreach (string name in names)
                    {
                        var source = _weights[name];
                        using var destination = fused.Narrow(0, offset, source.Sizes[0]);
                        Ops.Copy(destination, source);
                        offset += (int)source.Sizes[0];
                    }
                    _weights.Add(fusedName, fused);
                }
                catch { fused.Dispose(); throw; }
                foreach (string name in names) { var source = _weights[name]; _weights.Remove(name); source.Dispose(); }
                _fusedAttention[layer] = true;
            }
        }
        ReportDeclinedFusionCopies();
        if (FusedAttentionLayerCount != 0)
            Console.WriteLine($"Muse Spark fused Q/K/V/gate projections: {FusedAttentionLayerCount} layers.");
    }

    private (Tensor Query, Tensor Key, Tensor Value, Tensor Gate) AttentionProjections(
        Tensor input, int layer, int start, int[] positions, out bool preparedQK)
    {
        preparedQK = false;
        string p = _prefixes[layer];
        Tensor q = null, k = null, v = null, gate = null;
        try
        {
            if (!_fusedAttention[layer])
            {
                q = LinearForward(input, p + "attn_q.weight");
                k = LinearForward(input, p + "attn_k.weight");
                v = LinearForward(input, p + "attn_v.weight");
                gate = LinearForward(input, p + "attn_gate.weight");
            }
            else
            {
                using var packed = LinearForward(input, p + "attn_qkvg.weight");
                int qWidth = Config.NumHeads * _headDim, kvWidth = Config.NumKVHeads * _headDim;
                Tensor Part(int offset, int width)
                {
                    using var view = packed.Narrow(1, offset, width);
                    // A single row can retain the zero-copy view. Multiple rows
                    // have a packed stride and need contiguous head-norm input.
                    return input.Sizes[0] == 1 ? view.CopyRef() : Ops.NewContiguous(view);
                }
                if (UseFusedQKPrep)
                {
                    q = new Tensor(_allocator, DType.Float32, input.Sizes[0], qWidth);
                    k = new Tensor(_allocator, DType.Float32, input.Sizes[0], kvWidth);
                    PrepareFusedQK(q, k, packed, layer, start, positions);
                    preparedQK = true;
                }
                else { q = Part(0, qWidth); k = Part(qWidth, kvWidth); }
                v = Part(qWidth + kvWidth, kvWidth); gate = Part(qWidth + 2 * kvWidth, qWidth);
            }
            return (q, k, v, gate);
        }
        catch { q?.Dispose(); k?.Dispose(); v?.Dispose(); gate?.Dispose(); throw; }
    }
}
