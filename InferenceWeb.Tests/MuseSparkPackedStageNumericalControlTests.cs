// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Reflection;
using System.Text.Json;
using System.Text.Json.Serialization;
using TensorSharp;
using TensorSharp.GGML;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;

namespace InferenceWeb.Tests;

public sealed partial class MuseSparkPackedHeadNumericalControlTests
{
    [MuseSparkTensorParallelGpuFact(5)]
    public void NarrowPackedFloatMixedSequencesRetainIndependentScalarOracle()
        => CheckNarrowPacked(BackendType.GgmlCuda);

    [Fact]
    public void ManagedCpuNarrowPackedMixedSequencesRetainIndependentScalarOracle()
        => CheckNarrowPacked(BackendType.Cpu);

    private void CheckNarrowPacked(BackendType backend)
    {
        var fixture = new MuseSparkSyntheticModelBuilder(heads: 32, kvHeads: 16,
            experts: 36, used: 2, headDim: 16, latent: 32, ffn: 32);
        MuseSparkTensorParallelGpuTests.WriteQ8Fixture(fixture, Path.Combine(_directory, "narrow-q8.gguf"));
        string path = fixture.Write(Path.Combine(_directory, "narrow-f32.gguf"));
        int[][] prompts = [[7, 21, 42, 11, 9, 86, 102, 68], [81, 53, 29, 40, 16, 12, 63, 5], [17, 2, 199, 7]];
        // Same seventeen per-sequence token positions as the original wider
        // schedule, now in <=3 total-row calls. This independently tests the
        // FP32 vector-matmul arithmetic regime without replacing the wide gate.
        (int[], int[])[] schedule =
        [
            ([0, 1], [1, 2]), ([1, 0], [1, 2]), ([0, 1], [2, 1]),
            ([2, 1], [2, 1]), ([0, 2, 1], [1, 1, 1]), ([2, 1], [1, 1]),
        ];
        var checks = new MuseSparkGpuChecks(_output);
        foreach (int degree in new[] { 1, 5 })
        {
            var rows = MuseSparkGpuSchedules.Packed(path, degree, prompts, 2, schedule, false,
                captureBatch: (_, capture) =>
                {
                    for (int sequence = 0; sequence < capture.SequenceIndices.Length; sequence++)
                    {
                        int index = capture.SequenceIndices[sequence];
                        int position = capture.Positions[capture.Starts[sequence + 1] - 1];
                        checks.Compare($"degree={degree}, batch={capture.Batch}, returned seq={index} pos={position}",
                            fixture.Reference(prompts[index][..(position + 1)]), capture.ReturnedLogits[sequence],
                            3e-5, 3e-5);
                    }
                }, requireExactHeadRows: false, backend: backend);
            Assert.Equal(17, rows.Length);
            foreach (var row in rows)
                checks.Compare($"degree={degree}, captured seq={row.Sequence} pos={row.Position}",
                    fixture.Reference(prompts[row.Sequence][..(row.Position + 1)]), row.Values, 3e-5, 3e-5);
        }
        checks.Complete("narrow-packed-f32-scalar-controls");
    }

    /// <summary>Diagnostic replay of the first actual packed batch. It does not
    /// alter a production precision setting or regrade the original failure.</summary>
    [MuseSparkTensorParallelGpuFact(5)]
    public void PackedFloatFirstBatchRetainsLayerAndSameInputProjectionControls()
        => CheckPackedStages(BackendType.GgmlCuda);

    [Fact]
    public void ManagedCpuPackedFirstBatchStageReplayMatchesForwardAndScalarProjection()
        => CheckPackedStages(BackendType.Cpu);

    private void CheckPackedStages(BackendType backend)
    {
        var fixture = new MuseSparkSyntheticModelBuilder(heads: 32, kvHeads: 16,
            experts: 36, used: 2, headDim: 16, latent: 32, ffn: 32);
        MuseSparkTensorParallelGpuTests.WriteQ8Fixture(fixture, Path.Combine(_directory, "stage-q8.gguf"));
        string path = fixture.Write(Path.Combine(_directory, "stage-f32.gguf"));
        int[][] prompts = [[7, 21, 42, 11, 9, 86, 102, 68], [81, 53, 29, 40, 16, 12, 63, 5]];
        (int[], int[])[] schedule = [([0, 1], [3, 5])];
        var controls = new List<PackedStageControl>();
        var checks = new MuseSparkGpuChecks(_output);
        foreach (int degree in new[] { 1, 5 })
            MuseSparkGpuSchedules.Packed(path, degree, prompts, 2, schedule, false,
                captureBatch: (_, capture) => checks.Compare($"degree={degree} replay final hidden",
                    capture.Hidden, controls.Last().Stages["final"], 3e-5, 3e-5),
                requireExactHeadRows: false,
                captureContext: (model, context) => controls.Add(ReplayPackedStages(model, context, degree, fixture, backend)),
                backend: backend);

        string directory = Environment.GetEnvironmentVariable("TS_MUSE_SPARK_TP_CASE_OUTPUT_DIRECTORY");
        if (!string.IsNullOrWhiteSpace(directory))
        {
            Directory.CreateDirectory(directory);
            using var stream = new FileStream(Path.Combine(directory, "packed-stage-controls-raw.json"),
                FileMode.CreateNew, FileAccess.Write, FileShare.Read);
            JsonSerializer.Serialize(stream, new
            {
                status = "diagnostic_only_original_wide_f32_failure_preserved",
                prompts, controls,
                projectionWeights = new[] { "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_gate.weight" }
                    .ToDictionary(name => "blk.0." + name, name => fixture.Weights["blk.0." + name].Data),
            }, new JsonSerializerOptions
            {
                WriteIndented = true, NumberHandling = JsonNumberHandling.AllowNamedFloatingPointLiterals,
            });
        }
        if (controls.Count == 2)
            foreach (string stage in controls[0].Stages.Keys)
                checks.Compare(stage + " TP/single", controls[0].Stages[stage], controls[1].Stages[stage],
                    3e-5, 3e-5, required: backend == BackendType.Cpu);
        foreach (var control in controls)
            foreach (var projection in control.Projections)
            {
                string label = $"degree={control.Degree}, {projection.Name}";
                checks.Compare(label + " original/scalar", projection.Scalar, projection.Original,
                    3e-5, 3e-5, required: backend == BackendType.Cpu);
                checks.Compare(label + " split/scalar", projection.Scalar, projection.Split,
                    3e-5, 3e-5, required: backend == BackendType.Cpu);
                checks.Compare(label + " split/original", projection.Original, projection.Split,
                    3e-5, 3e-5, required: backend == BackendType.Cpu);
            }
        checks.Complete("packed-stage-controls-verdicts");
    }

    private static PackedStageControl ReplayPackedStages(MuseSparkModel model, BatchedForwardContext context,
        int degree, MuseSparkSyntheticModelBuilder fixture, BackendType backend)
    {
        int rows = context.Positions.Count;
        int[] positions = context.Positions.ToArray(), starts = context.QueryStartLoc.ToArray();
        int[] slots = context.SlotMapping.ToArray();
        int[] lengths = context.Sequences.Select((sequence, s) =>
            sequence.NumComputedTokens + context.NumScheduledTokens[s]).ToArray();
        int[] offsets = new int[lengths.Length];
        for (int s = 1; s < offsets.Length; s++) offsets[s] = offsets[s - 1] + context.BlockTables[s - 1].Length;
        int[] blocks = context.BlockTables.SelectMany(value => value).ToArray();
        int[] tokens = context.Sequences.SelectMany((sequence, s) =>
            Enumerable.Range(sequence.NumComputedTokens, context.NumScheduledTokens[s]).Select(sequence.TokenAt)).ToArray();
        var stages = new Dictionary<string, float[]>();
        var projections = new List<PackedProjectionControl>();
        Tensor hidden = null;
        bool pin = backend == BackendType.GgmlCuda;
        int priorRank = pin ? GgmlBasicOps.GetActiveRank() : 0;
        bool priorPin = pin && GgmlBasicOps.PinRank(0);
        try
        {
            using (var embedding = InvokeTensor(model, "Embedding", tokens))
                hidden = InvokeTensor(model, "Norm", embedding, model.Config.Eps, null);
            stages.Add("embedding-normalized", Values(hidden));
            using var qp = InvokeTensor(model, "SparkBatchPositions", positions, model.Config.NumHeads);
            using var kp = InvokeTensor(model, "SparkBatchPositions", positions, model.Config.NumKVHeads);
            for (int layer = 0; layer < model.Config.NumLayers; layer++)
            {
                string p = $"blk.{layer}.";
                using var pre = InvokeTensor(model, "Norm", hidden, model.Config.Eps, p + "attn_norm.weight");
                stages.Add(p + "pre-attention", Values(pre));
                // For layer zero the normalized input is exactly shared between
                // executions. Capture each full vs row-sharded projection before
                // attention, then isolate input-column sharding on one fixed input.
                if (layer == 0)
                {
                    foreach (string name in new[] { "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_gate.weight" })
                        stages.Add(p + name, JoinedProjection(model, pre, p + name, degree));
                    float[] sameInput = Enumerable.Range(0, rows * model.Config.NumHeads * 16)
                        .Select(i => MathF.Sin(i * 0.113f) * 0.7f + MathF.Cos(i * 0.071f) * 0.3f).ToArray();
                    projections.Add(IsolateOutputProjection(model, sameInput, rows, p + "attn_output.weight", degree,
                        fixture.Weights[p + "attn_output.weight"].Data));
                }
                using var attention = InvokeTensor(model, "SparkBatchAttention", pre, layer, rows, lengths.Length,
                    qp, kp, positions, starts, slots, lengths, context.BlockTables, blocks, offsets);
                stages.Add(p + "attention-output", Values(attention));
                using var ffnInput = InvokeTensor(model, "PostNormResidual", hidden, attention,
                    p + "attn_resid_alpha.weight", p + "attn_resid_beta.weight", null);
                stages.Add(p + "attention-residual", Values(ffnInput));
                hidden.Dispose(); hidden = null;
                using var ffnNorm = InvokeTensor(model, "Norm", ffnInput, model.Config.Eps, p + "ffn_norm.weight");
                stages.Add(p + "pre-moe", Values(ffnNorm));
                using var moe = InvokeTensor(model, "MoE", ffnNorm, layer, rows);
                stages.Add(p + "moe-output", Values(moe));
                hidden = InvokeTensor(model, "PostNormResidual", ffnInput, moe,
                    p + "ffn_resid_alpha.weight", p + "ffn_resid_beta.weight", p + "post_ffw_norm.weight");
                stages.Add(p + "ffn-residual", Values(hidden));
            }
            using var final = InvokeTensor(model, "Norm", hidden, model.Config.Eps, "output_norm.weight");
            stages.Add("final", Values(final));
            return new PackedStageControl(degree, stages, projections);
        }
        finally { hidden?.Dispose(); if (pin) GgmlBasicOps.RestoreRankPin(priorPin, priorRank); }
    }

    private static float[] JoinedProjection(MuseSparkModel model, Tensor input, string name, int degree)
    {
        if (degree == 1)
        {
            using var output = InvokeTensor(model, "LinearForward", input, name);
            return Values(output);
        }
        var arrays = new float[degree][];
        var group = Field<ITensorParallelGroup>(model, "_tpGroup");
        group.RunPerRank(rank =>
        {
            using var local = InvokeTensor(model, "SparkRankInput", input, rank);
            using var output = InvokeTensor(model, "SparkRankLinear", local, name, rank);
            arrays[rank] = Values(output);
        });
        int rows = (int)input.Sizes[0], width = arrays.Sum(value => value.Length / rows);
        var result = new float[rows * width];
        for (int row = 0; row < rows; row++)
        {
            int offset = row * width;
            foreach (float[] array in arrays)
            {
                int localWidth = array.Length / rows;
                array.AsSpan(row * localWidth, localWidth).CopyTo(result.AsSpan(offset, localWidth));
                offset += localWidth;
            }
        }
        return result;
    }

    private static PackedProjectionControl IsolateOutputProjection(MuseSparkModel model, float[] input,
        int rows, string name, int degree, float[] weights)
    {
        int width = input.Length / rows, outputWidth = weights.Length / width;
        float[] scalar = new float[rows * outputWidth];
        for (int row = 0; row < rows; row++)
            for (int column = 0; column < outputWidth; column++)
            {
                double sum = 0;
                for (int inner = 0; inner < width; inner++)
                    sum += (double)input[row * width + inner] * weights[column * width + inner];
                scalar[row * outputWidth + column] = (float)sum;
            }
        using var tensor = InvokeTensor(model, "CreateFloatTensor", input, new long[] { rows, width });
        float[] original, split;
        if (degree == 1)
        {
            using var fullOutput = InvokeTensor(model, "LinearForward", tensor, name);
            original = Values(fullOutput);
            // Five explicit, contiguous input-column partitions using the same
            // stored weights and generic GEMM as the TP F32 implementation.
            using var weight = InvokeTensor(model, "CreateFloatTensor", weights, new long[] { outputWidth, width });
            var partials = new List<float[]>();
            for (int rank = 0; rank < 5; rank++)
            {
                var (first, count) = MuseSparkModel.SparkPartition(16, 5, rank);
                using var iv = tensor.Narrow(1, first * 32, count * 32);
                using var local = new Tensor(tensor.Allocator, DType.Float32, iv.Sizes);
                Ops.Copy(local, iv);
                using var wv = weight.Narrow(1, first * 32, count * 32);
                using var wc = new Tensor(tensor.Allocator, DType.Float32, wv.Sizes);
                Ops.Copy(wc, wv);
                using var wt = wc.Transpose();
                using var output = new Tensor(local.Allocator, DType.Float32, rows, outputWidth);
                Ops.Addmm(output, 0f, output, 1f, local, wt);
                partials.Add(Values(output));
            }
            split = Sum(partials);
        }
        else
        {
            var partials = new Tensor[degree];
            var group = Field<ITensorParallelGroup>(model, "_tpGroup");
            try
            {
                group.RunPerRank(rank =>
                {
                    var (first, count) = MuseSparkModel.SparkPartition(16, degree, rank);
                    using var view = tensor.Narrow(1, first * 32, count * 32);
                    using var local = InvokeTensor(model, "SparkRankInput", view, rank);
                    partials[rank] = InvokeTensor(model, "SparkRankLinear", local, name, rank);
                });
                split = Sum(partials.Select(Values).ToArray());
                using var reduced = InvokeTensor(model, "SparkReduce", (object)partials);
                original = Values(reduced);
            }
            finally { foreach (var partial in partials) partial?.Dispose(); }
        }
        return new PackedProjectionControl(name, input, weights, scalar, original, split);
    }

    private static float[] Sum(IReadOnlyList<float[]> arrays)
    {
        float[] sum = new float[arrays[0].Length];
        foreach (float[] values in arrays)
            for (int i = 0; i < sum.Length; i++) sum[i] += values[i];
        return sum;
    }

    private static float[] Values(Tensor tensor) => tensor.GetElementsAsFloat(checked((int)tensor.ElementCount()));

    private static Tensor InvokeTensor(MuseSparkModel model, string name, params object[] args)
    {
        for (Type type = typeof(MuseSparkModel); type != null; type = type.BaseType)
        {
            MethodInfo method = type.GetMethod(name, BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.DeclaredOnly);
            if (method != null) return (Tensor)method.Invoke(model, args)!;
        }
        throw new MissingMethodException(name);
    }

    private static T Field<T>(MuseSparkModel model, string name)
    {
        for (Type type = typeof(MuseSparkModel); type != null; type = type.BaseType)
        {
            FieldInfo field = type.GetField(name, BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.DeclaredOnly);
            if (field != null) return (T)field.GetValue(model)!;
        }
        throw new MissingFieldException(name);
    }

    private sealed record PackedStageControl(int Degree, Dictionary<string, float[]> Stages,
        List<PackedProjectionControl> Projections);
    private sealed record PackedProjectionControl(string Name, float[] Input, float[] Weight,
        float[] Scalar, float[] Original, float[] Split);
}
