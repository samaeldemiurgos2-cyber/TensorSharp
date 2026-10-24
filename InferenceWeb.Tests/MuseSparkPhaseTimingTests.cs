// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Reflection;
using TensorSharp;
using TensorSharp.Models;

namespace InferenceWeb.Tests;

[Collection("Muse Spark model")]
public sealed class MuseSparkPhaseTimingTests : IDisposable
{
    private readonly string _directory = Path.Combine(Path.GetTempPath(), "ts-spark-timing-" + Guid.NewGuid().ToString("N"));
    private readonly NativeEnvScope _environment = new();
    private readonly KvCacheDtype _oldDtype = KvCacheDtypeConfig.Current;
    private readonly bool _oldExplicit = KvCacheDtypeConfig.IsExplicitlySet;

    public MuseSparkPhaseTimingTests()
    {
        Directory.CreateDirectory(_directory);
        _environment.Set("MAX_CONTEXT", "64");
        _environment.Set("TS_MUSE_SPARK_PREFILL_CHUNK", "3");
        _environment.Set("TS_MUSE_SPARK_FUSE_QKVG", "0");
        _environment.Set("TS_MUSE_SPARK_LAYER_SPLIT", null);
        MoeCpuOffloadConfig.Reset();
        KvCacheDtypeConfig.Set(KvCacheDtype.F32);
    }

    public void Dispose()
    {
        KvCacheDtypeConfig.RestoreForTests(_oldDtype, _oldExplicit);
        MoeCpuOffloadConfig.Reset();
        _environment.Dispose();
        Directory.Delete(_directory, true);
    }

    [Fact]
    public void CallerIntervalsPartitionWallTimeAndOverlappingRanksStaySeparate()
    {
        long now = 100;
        var frame = new MuseSparkPhaseRecorder(2, () => now, 1000);
        now = 105; frame.Next(MuseSparkPhase.AttentionFanout);
        using (var rank0 = frame.BeginRank(0, MuseSparkRankPhase.AttentionInputCopy))
        using (var rank1 = frame.BeginRank(1, MuseSparkRankPhase.AttentionInputCopy))
        {
            now = 110; rank0.Next(MuseSparkRankPhase.AttentionProjections);
            now = 111; rank1.Next(MuseSparkRankPhase.AttentionProjections);
            now = 125;
        }
        now = 128; frame.Next(MuseSparkPhase.AttentionReduce);
        now = 132; frame.Next(MuseSparkPhase.Cleanup);
        now = 135;
        var report = frame.Complete(1, "ordinary", 1, 0, false);
        Assert.Equal(35, report.wall_ms);
        Assert.Equal(35, report.caller_exclusive_ms.Values.Sum());
        Assert.Equal(40, report.rank_overlapping_ms.Sum(r => r.Values.Sum()));
        Assert.Equal(new long[] { 1, 1 }, report.rank_intervals);
        Assert.Equal(0, report.cache_start); // One-row prefill remains explicitly identifiable.
        Assert.Throws<InvalidOperationException>(() => frame.Complete(2, "ordinary", 1, 1, false));
    }

    [Theory]
    [InlineData(1)]
    [InlineData(3)]
    public void EnabledAndDisabledOrdinaryExecutionHasIdenticalLogitsAndSeparateCallRecords(int degree)
    {
        var fixture = new MuseSparkSyntheticModelBuilder(heads: 8, kvHeads: 4, experts: 8, used: 2);
        string path = fixture.Write(Path.Combine(_directory, "model.gguf"));
        List<float[]> Run(bool enabled, out List<MuseSparkPhaseReport> reports)
        {
            _environment.Set("TS_MUSE_SPARK_PHASE_TIMING", enabled ? "1" : "0");
            using var group = degree > 1 ? new MuseSparkTensorParallelTests.CpuGroup(degree) : null;
            using var model = new MuseSparkModel(path, BackendType.Cpu, degree, group);
            var records = new List<MuseSparkPhaseReport>();
            model.PhaseTimingSink = records.Add;
            var outputs = new List<float[]> { model.Forward([7]), model.Forward([21, 42]), model.Forward([11]) };
            if (!enabled) Assert.Null(model.LastPhaseTimingReport);
            else Assert.Same(records.Last(), model.LastPhaseTimingReport);
            reports = records;
            return outputs;
        }
        var expected = Run(false, out var disabled);
        var actual = Run(true, out var reports);
        Assert.Empty(disabled);
        for (int i = 0; i < expected.Count; i++) Assert.Equal(expected[i], actual[i]);
        Assert.Equal(new[] { 1, 2, 1 }, reports.Select(r => r.rows));
        Assert.Equal(new[] { 0, 1, 3 }, reports.Select(r => r.cache_start));
        Assert.Equal(new long[] { 1, 2, 3 }, reports.Select(r => r.sequence));
        foreach (var report in reports)
        {
            Assert.False(report.failed);
            Assert.Equal(report.wall_ms, report.caller_exclusive_ms.Values.Sum(), 8);
            Assert.All(report.rank_intervals, n => Assert.Equal(degree > 1 ? 4 : 0, n));
            if (degree > 1)
            {
                Assert.True(report.caller_exclusive_ms["AttentionFanout"] > 0);
                Assert.True(report.caller_exclusive_ms["ExpertReduce"] > 0);
                Assert.All(report.rank_overlapping_ms, r => Assert.True(r["AttentionProjections"] > 0));
            }
        }
    }

    [Fact]
    public void PackedExecutionKeepsIdenticalCapturedLogitsAndUnknownSingleCachePosition()
    {
        var fixture = new MuseSparkSyntheticModelBuilder(heads: 8, kvHeads: 4, experts: 8, used: 2);
        string path = fixture.Write(Path.Combine(_directory, "packed.gguf"));
        int[][] prompts = [[7, 21, 42], [11, 9, 86]];
        (int[], int[])[] schedule = [([0, 1], [1, 2]), ([1, 0], [1, 2])];
        _environment.Set("TS_MUSE_SPARK_PHASE_TIMING", "0");
        var expected = MuseSparkGpuSchedules.Packed(path, 3, prompts, 2, schedule, false,
            requireExactHeadRows: false, backend: BackendType.Cpu);
        _environment.Set("TS_MUSE_SPARK_PHASE_TIMING", "1");
        var reports = new List<MuseSparkPhaseReport>();
        var actual = MuseSparkGpuSchedules.Packed(path, 3, prompts, 2, schedule, false,
            captureBatch: (model, _) => reports.Add(model.LastPhaseTimingReport),
            requireExactHeadRows: false, backend: BackendType.Cpu);
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i].Values, actual[i].Values);
        Assert.Equal(2, reports.Count);
        Assert.All(reports, report =>
        {
            Assert.False(report.failed);
            Assert.Equal("packed", report.path);
            Assert.Equal(-1, report.cache_start);
            Assert.Equal(3, report.rows);
            Assert.Equal(report.wall_ms, report.caller_exclusive_ms.Values.Sum(), 8);
            Assert.Equal(new long[] { 4, 4, 4 }, report.rank_intervals);
        });
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void CleanupExceptionsMarkTheCompletedRecordFailed(bool outerFinally)
    {
        _environment.Set("TS_MUSE_SPARK_PHASE_TIMING", "1");
        string path = new MuseSparkSyntheticModelBuilder().Write(Path.Combine(_directory, "failure.gguf"));
        using var model = new MuseSparkModel(path, BackendType.Cpu);
        var reports = new List<MuseSparkPhaseReport>();
        model.PhaseTimingSink = reports.Add;
        int Body()
        {
            if (outerFinally)
            {
                try { return 1; }
                finally { throw new InvalidOperationException("outer cleanup control"); }
            }
            using var cleanup = new ThrowingCleanup();
            return 1;
        }
        var method = typeof(MuseSparkModel).GetMethod("RecordSparkForward", BindingFlags.NonPublic | BindingFlags.Instance)!
            .MakeGenericMethod(typeof(int));
        var exception = Assert.Throws<TargetInvocationException>(() =>
            method.Invoke(model, ["test", 1, 0, (Func<int>)Body]));
        Assert.IsType<InvalidOperationException>(exception.InnerException);
        Assert.Single(reports);
        Assert.True(reports[0].failed);
        Assert.Equal(reports[0].wall_ms, reports[0].caller_exclusive_ms.Values.Sum(), 8);
        Assert.Null(typeof(MuseSparkModel).GetField("_sparkPhaseFrame", BindingFlags.NonPublic | BindingFlags.Instance)!.GetValue(model));
        model.Forward([7]);
        Assert.Equal(2, reports.Count);
        Assert.False(reports[1].failed);
    }

    private sealed class ThrowingCleanup : IDisposable
    {
        public void Dispose() => throw new InvalidOperationException("using cleanup control");
    }
}
