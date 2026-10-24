// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using TensorSharp.GGML;

namespace TensorSharp.Models;

public partial class MuseSparkModel
{
    /// <summary>Compact only this rank's routes. Distinct zero-weight filler IDs
    /// avoid duplicate expert IDs in ggml_mul_mat_id. A rank with no routes does
    /// not launch an expert graph at all.</summary>
    internal static (int Used, int[] Selected, float[] Weights) SparkCompactRoutes(
        int[] selected, float[] weights, int rows, int used, int firstExpert, int expertCount)
    {
        if (rows <= 0 || used <= 0 || expertCount <= 0 || firstExpert < 0 ||
            selected.Length != checked(rows * used) || weights.Length != selected.Length)
            throw new ArgumentException("Invalid expert-parallel routing geometry.");
        int maximum = 0;
        for (int row = 0; row < rows; row++)
        {
            int count = 0;
            for (int slot = 0; slot < used; slot++)
                if ((uint)(selected[row * used + slot] - firstExpert) < (uint)expertCount) count++;
            maximum = Math.Max(maximum, count);
        }
        if (maximum == 0) return (0, Array.Empty<int>(), Array.Empty<float>());
        if (maximum > expertCount) throw new ArgumentException("Repeated expert IDs are not valid routes.");
        var ids = new int[checked(rows * maximum)];
        var routeWeights = new float[ids.Length];
        for (int row = 0; row < rows; row++)
        {
            int count = 0, offset = row * maximum;
            for (int slot = 0; slot < used; slot++)
            {
                int local = selected[row * used + slot] - firstExpert;
                if ((uint)local >= (uint)expertCount) continue;
                for (int prior = 0; prior < count; prior++)
                    if (ids[offset + prior] == local) throw new ArgumentException("Repeated expert IDs are not valid routes.");
                ids[offset + count] = local;
                routeWeights[offset + count++] = weights[row * used + slot];
            }
            for (int filler = 0; count < maximum; filler++)
            {
                bool present = false;
                for (int slot = 0; slot < count; slot++) present |= ids[offset + slot] == filler;
                if (!present) ids[offset + count++] = filler;
            }
        }
        return (maximum, ids, routeWeights);
    }

    private Tensor TensorParallelExperts(Tensor input, int layer, int rows, int[] selected, float[] routing)
    {
        var partials = new Tensor[TpDegree];
        try
        {
            _sparkPhaseFrame?.Next(MuseSparkPhase.ExpertFanout);
            _tpGroup.RunPerRank(rank =>
            {
                using var timing = _sparkPhaseFrame?.BeginRank(rank, MuseSparkRankPhase.ExpertRoutes);
                var routes = SparkCompactRoutes(selected, routing, rows, _numExpertsUsed,
                    _tpExpertFirst[rank], _tpExpertCount[rank]);
                var output = new Tensor(_tpGroup.GetAllocator(rank), DType.Float32, rows, _latentSize);
                partials[rank] = output;
                if (routes.Used == 0) { Ops.Fill(output, 0f); timing?.Next(MuseSparkRankPhase.Cleanup); return; }
                timing?.Next(MuseSparkRankPhase.ExpertInputCopy);
                using var local = SparkRankInput(input, rank);
                timing?.Next(MuseSparkRankPhase.ExpertCompute);
                if (IsGgmlBackend && _sparkTpGate[layer] != null)
                {
                    var gate = _sparkTpGate[layer][rank]; var up = _sparkTpUp[layer][rank]; var down = _sparkTpDown[layer][rank];
                    GgmlBasicOps.MoEFFNPrefill(local, output, rows, _latentSize, _expertFfnLength,
                        _tpExpertCount[rank], routes.Used, routes.Selected, routes.Weights,
                        gate.Data, gate.GgmlType, gate.PerExpertNe0, gate.PerExpertNe1, gate.TotalRawBytes,
                        up.Data, up.GgmlType, up.PerExpertNe0, up.PerExpertNe1, up.TotalRawBytes,
                        down.Data, down.GgmlType, down.PerExpertNe0, down.PerExpertNe1, down.TotalRawBytes,
                        null, null, null, GgmlBasicOps.MoEActivation.SwiGLUSplit, runOnCpu: false,
                        postExpertNorm: _sparkTpPostNorm[layer], postExpertNormEps: _postNormEps);
                    InvalidateTensorDeviceCache(output);
                }
                else SparkRankGroupedExperts(local, output, layer, rank, rows, routes.Used, routes.Selected, routes.Weights);
                timing?.Next(MuseSparkRankPhase.Cleanup);
            });
            _sparkPhaseFrame?.Next(MuseSparkPhase.ExpertReduce);
            return SparkReduce(partials);
        }
        catch { foreach (var partial in partials) partial?.Dispose(); throw; }
    }

    private unsafe void SparkRankGroupedExperts(Tensor input, Tensor output, int layer, int rank,
        int rows, int used, int[] selected, float[] routing)
    {
        var buckets = new List<int>[_tpExpertCount[rank]];
        for (int route = 0; route < selected.Length; route++)
            if (routing[route] != 0f) (buckets[selected[route]] ??= new List<int>()).Add(route);
        float* source = GetFloatPtr(input), result = GetFloatPtr(output);
        new Span<float>(result, rows * _latentSize).Clear();
        for (int expert = 0; expert < buckets.Length; expert++)
        {
            var routes = buckets[expert];
            if (routes == null) continue;
            using var gathered = new Tensor(_tpGroup.GetAllocator(rank), DType.Float32, routes.Count, _latentSize);
            float* gp = GetFloatPtr(gathered);
            for (int row = 0; row < routes.Count; row++)
                new ReadOnlySpan<float>(source + (long)(routes[row] / used) * _latentSize, _latentSize)
                    .CopyTo(new Span<float>(gp + (long)row * _latentSize, _latentSize));
            InvalidateTensorDeviceCache(gathered);
            int globalExpert = _tpExpertFirst[rank] + expert;
            using var gate = SparkRankExpertLinear(gathered, layer, globalExpert, "gate", _expertFfnLength, rank);
            using var up = SparkRankExpertLinear(gathered, layer, globalExpert, "up", _expertFfnLength, rank);
            Ops.SiLUMul(gate, gate, up);
            using var down = SparkRankExpertLinear(gate, layer, globalExpert, "down", _latentSize, rank);
            float* dp = GetFloatPtr(down);
            float[] scale = _sparkTpPostNorm[layer];
            for (int row = 0; row < routes.Count; row++)
            {
                float sumSquares = 0;
                float* values = dp + (long)row * _latentSize;
                for (int i = 0; i < _latentSize; i++) { values[i] *= scale[i]; sumSquares += values[i] * values[i]; }
                float inverseRms = 1f / MathF.Sqrt(sumSquares / _latentSize + _postNormEps);
                int route = routes[row];
                // Keep normalization before routing multiplication, as in the
                // independent decoder and native post-expert graph.
                for (int i = 0; i < _latentSize; i++)
                    result[(long)(route / used) * _latentSize + i] += (values[i] * inverseRms) * routing[route];
            }
        }
        InvalidateTensorDeviceCache(output);
    }

    private Tensor SparkRankExpertLinear(Tensor input, int layer, int expert, string kind, int width, int rank)
    {
        string name = _prefixes[layer] + "ffn_" + kind + "_exps";
        var output = new Tensor(_tpGroup.GetAllocator(rank), DType.Float32, input.Sizes[0], width);
        try
        {
            if (_quantWeights.TryGetValue(name + $".{expert}.weight", out var weight))
            {
                AddmmQuantManaged(output, input, weight);
                if (weight.Scale != 1f) Ops.Mul(output, output, weight.Scale);
            }
            else
            {
                using var plane = _weights[name + ".weight"].Select(0, expert);
                SparkFloatProjection(output, input, plane);
            }
            return output;
        }
        catch { output.Dispose(); throw; }
    }
}
