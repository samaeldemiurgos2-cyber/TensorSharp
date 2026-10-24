// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Text.Json;
using System.Text.Json.Serialization;
using TensorSharp;
using TensorSharp.GGML;
using TensorSharp.Runtime.Paged;
using TensorSharp.Runtime.Scheduling;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>Capture all controls before asserting. Optional raw evidence is
/// written exclusively, so a later run cannot replace a failed verdict.</summary>
internal sealed class MuseSparkGpuChecks(ITestOutputHelper output)
{
    private readonly List<Check> _checks = [];

    internal void Compare(string label, float[] expected, float[] actual,
        double maximumBound, double rmsBound, bool required = true)
    {
        Assert.Equal(expected.Length, actual.Length);
        bool finite = expected.All(float.IsFinite) && actual.All(float.IsFinite);
        double maximum = expected.Zip(actual, (a, b) => Math.Abs((double)a - b)).Max();
        double rms = Math.Sqrt(expected.Zip(actual, (a, b) => ((double)a - b) * ((double)a - b)).Average());
        bool passed = finite && maximum <= maximumBound && rms <= rmsBound;
        output.WriteLine($"{label}: required={required}, finite={finite}, max={maximum:G12}, rms={rms:G12}, " +
            $"bounds={maximumBound:G9}/{rmsBound:G9}, passed={passed}");
        _checks.Add(new Check(label, required, finite, maximum, rms, maximumBound, rmsBound,
            passed, expected.ToArray(), actual.ToArray()));
    }

    internal void Complete(string caseName)
    {
        string directory = Environment.GetEnvironmentVariable("TS_MUSE_SPARK_TP_CASE_OUTPUT_DIRECTORY");
        if (!string.IsNullOrWhiteSpace(directory))
        {
            Directory.CreateDirectory(directory);
            using var stream = new FileStream(Path.Combine(directory, caseName + ".json"),
                FileMode.CreateNew, FileAccess.Write, FileShare.Read);
            JsonSerializer.Serialize(stream, _checks, new JsonSerializerOptions
            {
                WriteIndented = true,
                NumberHandling = JsonNumberHandling.AllowNamedFloatingPointLiterals,
            });
        }
        var failed = _checks.Where(value => value.Required && !value.Passed).ToArray();
        Assert.True(failed.Length == 0, string.Join(Environment.NewLine, failed.Select(value =>
            $"{value.Label}: finite={value.Finite}, max={value.Maximum:G12}, rms={value.Rms:G12}")));
    }

    private sealed record Check(string Label, bool Required, bool Finite, double Maximum, double Rms,
        double MaximumBound, double RmsBound, bool Passed, float[] Expected, float[] Actual);
}

internal sealed record MuseSparkGpuRow(int Sequence, int Position, float[] Values);
internal sealed record MuseSparkPackedCapture(int Batch, int[] SequenceIndices, int[] Starts,
    int[] Positions, float[] Hidden, float[] AllLogits, float[][] ReturnedLogits);

internal static class MuseSparkGpuSchedules
{
    internal static void AssertQKPreparationUsed(MuseSparkModel model, int degree)
    {
        if (degree > 1 && Environment.GetEnvironmentVariable("TS_MUSE_SPARK_FUSE_QK_PREP") == "1")
            Assert.True(model.FusedQKPrepCallCount >= degree * model.Config.NumLayers,
                "Enabled TP Q/K preparation did not execute on every rank/layer.");
    }

    internal static IEnumerable<MuseSparkGpuRow> ReturnedRows(MuseSparkPackedCapture batch)
    {
        for (int sequence = 0; sequence < batch.SequenceIndices.Length; sequence++)
            yield return new MuseSparkGpuRow(batch.SequenceIndices[sequence],
                batch.Positions[batch.Starts[sequence + 1] - 1], batch.ReturnedLogits[sequence]);
    }

    internal static MuseSparkGpuRow[] Ordinary(string path, int degree, int[][] prompts)
    {
        using var model = Assert.IsType<MuseSparkModel>(ModelBase.Create(path, BackendType.GgmlCuda, tpDegree: degree));
        var rows = new List<MuseSparkGpuRow>();
        for (int sequence = 0; sequence < prompts.Length; sequence++)
        {
            model.ResetKVCache();
            float[] capture = new float[prompts[sequence].Length * model.Config.VocabSize];
            model.SpecForward(prompts[sequence], null, capture, allLogitsRows: true);
            for (int position = 0; position < prompts[sequence].Length; position++)
                rows.Add(new MuseSparkGpuRow(sequence, position,
                    capture.AsSpan(position * model.Config.VocabSize, model.Config.VocabSize).ToArray()));
        }
        AssertQKPreparationUsed(model, degree);
        Assert.True(model.OwnedF32AttentionCalls >= (long)degree * model.Config.NumLayers * prompts.Length,
            "F32 ordinary model execution did not confirm the owned CUDA attention path.");
        Assert.Equal(0, model.OwnedF32PagedAttentionCalls);
        return rows.ToArray();
    }

    internal static MuseSparkGpuRow[] Packed(string path, int degree, int[][] prompts, int blockSize,
        (int[] Indices, int[] Counts)[] schedule, bool requireOwnedAttention,
        Action<MuseSparkModel, MuseSparkPackedCapture> captureBatch = null, bool requireExactHeadRows = true,
        Action<MuseSparkModel, BatchedForwardContext> captureContext = null,
        BackendType backend = BackendType.GgmlCuda)
    {
        using var cpuGroup = backend == BackendType.Cpu && degree > 1 ? new MuseSparkTensorParallelTests.CpuGroup(degree) : null;
        using var model = cpuGroup == null
            ? Assert.IsType<MuseSparkModel>(ModelBase.Create(path, backend, tpDegree: degree))
            : new MuseSparkModel(path, backend, degree, cpuGroup);
        Assert.True(model.BatchedForwardAvailable);
        var pool = new BlockPool(32, blockSize, model.ComputeKVBlockByteSize(blockSize));
        var sequences = prompts.Select((tokens, index) =>
            new SequenceState($"tp{index}", tokens, 8, blockSize, SamplingConfig.Default)).ToArray();
        // Round-robin allocation produces nonconsecutive physical pages.
        int blockCount = (prompts.Max(tokens => tokens.Length) + blockSize - 1) / blockSize;
        for (int block = 0; block < blockCount; block++)
            foreach (var sequence in sequences)
                sequence.BlockTable.AppendBlock(pool.AllocateNew(1)[0]);
        if (requireOwnedAttention) Assert.Equal(BackendType.GgmlCuda, backend);
        var rows = new List<MuseSparkGpuRow>();
        int batch = 0;
        foreach (var (indices, counts) in schedule)
        {
            var starts = new List<int> { 0 };
            var positions = new List<int>();
            var slots = new List<int>();
            var tables = new int[indices.Length][];
            for (int s = 0; s < indices.Length; s++)
            {
                var sequence = sequences[indices[s]];
                tables[s] = sequence.BlockTable.Blocks.Select(block => block.Id).ToArray();
                starts.Add(starts[^1] + counts[s]);
                for (int position = sequence.NumComputedTokens; position < sequence.NumComputedTokens + counts[s]; position++)
                {
                    positions.Add(position);
                    slots.Add(tables[s][position / blockSize] * blockSize + position % blockSize);
                }
            }
            var capture = new float[positions.Count * model.Config.VocabSize];
            var hidden = captureBatch == null ? null : new float[positions.Count * model.Config.HiddenSize];
            var context = new BatchedForwardContext
            {
                Sequences = indices.Select(index => sequences[index]).ToList(),
                NumScheduledTokens = counts.ToList(), QueryStartLoc = starts,
                Positions = positions, SlotMapping = slots, BlockTables = tables,
                MaxQueryLen = counts.Max(), MaxSeqLen = positions.Max() + 1,
                CaptureLogitsAll = capture,
                CaptureHiddenAll = hidden,
            };
            var actual = model.ForwardBatch(context);
            captureContext?.Invoke(model, context);
            // Diagnostic consumers receive both head results before the original
            // identity assertion. The ordinary regression keeps that assertion.
            captureBatch?.Invoke(model, new MuseSparkPackedCapture(batch++, indices.ToArray(), starts.ToArray(),
                positions.ToArray(), hidden.ToArray(), capture.ToArray(), actual.Select(row => row.ToArray()).ToArray()));
            for (int s = 0; s < indices.Length; s++)
            {
                var sequence = sequences[indices[s]];
                for (int row = 0; row < counts[s]; row++)
                    rows.Add(new MuseSparkGpuRow(indices[s], sequence.NumComputedTokens + row,
                        capture.AsSpan((starts[s] + row) * model.Config.VocabSize, model.Config.VocabSize).ToArray()));
                sequence.AdvanceComputedTokens(counts[s]);
                if (requireExactHeadRows)
                    Assert.Equal(capture.AsSpan((starts[s + 1] - 1) * model.Config.VocabSize, model.Config.VocabSize).ToArray(), actual[s]);
            }
        }
        Assert.Equal(0, model.CacheSeqLen);
        if (backend == BackendType.GgmlCuda)
        {
            AssertQKPreparationUsed(model, degree);
            Assert.True(model.OwnedF32PagedAttentionCalls >= (long)degree * model.Config.NumLayers * schedule.Length,
                "F32 packed model execution did not confirm the owned CUDA attention path.");
            Assert.Equal(0, model.OwnedF32AttentionCalls);
        }
        else
        {
            Assert.Equal(0, model.OwnedF32PagedAttentionCalls);
            Assert.Equal(0, model.OwnedF32AttentionCalls);
        }
        return rows.ToArray();
    }
}
