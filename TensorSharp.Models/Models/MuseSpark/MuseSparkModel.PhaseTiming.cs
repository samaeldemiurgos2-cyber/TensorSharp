// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text.Json;
using TensorSharp.GGML;

namespace TensorSharp.Models;

public partial class MuseSparkModel
{
    private readonly bool _sparkPhaseTimingEnabled =
        Environment.GetEnvironmentVariable("TS_MUSE_SPARK_PHASE_TIMING") == "1";
    private MuseSparkPhaseRecorder _sparkPhaseFrame;
    private long _sparkPhaseSequence;
    internal MuseSparkPhaseReport LastPhaseTimingReport { get; private set; }
    internal Action<MuseSparkPhaseReport> PhaseTimingSink { get; set; }

    // Forward is already serialized by the model/scheduler contract. Rank
    // workers write separate rows and are joined before this wrapper completes.
    // The disabled call sites do not allocate a frame/delegate or read a clock.
    private T RecordSparkForward<T>(string path, int rows, int cacheStart, Func<T> body)
    {
        var poolBefore = _sparkHostPoolContext?.GetHostPoolStatistics();
        var frame = new MuseSparkPhaseRecorder(IsTensorParallel ? TpDegree : 1);
        _sparkPhaseFrame = frame;
        bool succeeded = false;
        try
        {
            T result = body();
            succeeded = true; // Body includes all using/finally cleanup.
            return result;
        }
        finally
        {
            _sparkPhaseFrame = null;
            var report = frame.Complete(++_sparkPhaseSequence, path, rows, cacheStart, !succeeded);
            report = report with { host_pool_before = poolBefore,
                host_pool_after = _sparkHostPoolContext?.GetHostPoolStatistics() };
            LastPhaseTimingReport = report;
            if (PhaseTimingSink != null) PhaseTimingSink(report);
            else Console.WriteLine("[MUSE-SPARK-TIMING] " + JsonSerializer.Serialize(report));
        }
    }
}

internal enum MuseSparkPhase
{
    Setup, Embedding, LayerNorms, AttentionSingle, AttentionFanout, AttentionReduce,
    Residual, Router, LatentPreparation, CpuExperts, ExpertsSingle, ExpertFanout,
    ExpertReduce, LatentUp, OutputNorm, OutputHead, Cleanup,
}

internal enum MuseSparkRankPhase
{
    AttentionInputCopy, AttentionProjections, AttentionQkNormRope,
    AttentionKvAndAttention, AttentionGateOutput, ExpertRoutes, ExpertInputCopy,
    ExpertCompute, Cleanup,
}

internal sealed record MuseSparkPhaseReport(
    long sequence, string path, int rows, int cache_start, bool failed, double wall_ms,
    Dictionary<string, double> caller_exclusive_ms,
    Dictionary<string, double>[] rank_overlapping_ms, long[] rank_intervals,
    string scope = "Managed synchronous forward wall intervals; caller phases partition wall_ms. " +
        "Rank intervals overlap each other and caller fan-out: do not sum them into wall_ms. " +
        "No extra device synchronization. rows/cache_start do not classify prefill versus decode. " +
        "JSON construction/output is excluded. Failed-call unwind can remain in its active phase; " +
        "use successful calls for normal phase attribution.",
    GgmlMemoryPoolStatistics? host_pool_before = null,
    GgmlMemoryPoolStatistics? host_pool_after = null);

internal sealed class MuseSparkPhaseRecorder
{
    private readonly Func<long> _clock;
    private readonly double _msPerTick;
    private readonly long _start;
    private long _last;
    private MuseSparkPhase _phase = MuseSparkPhase.Setup;
    private readonly long[] _caller = new long[Enum.GetValues<MuseSparkPhase>().Length];
    private readonly long[][] _ranks;
    private readonly long[] _rankIntervals;
    private bool _completed;

    internal MuseSparkPhaseRecorder(int ranks, Func<long> clock = null, long frequency = 0)
    {
        if (ranks <= 0) throw new ArgumentOutOfRangeException(nameof(ranks));
        _clock = clock ?? Stopwatch.GetTimestamp;
        _msPerTick = 1000d / (frequency > 0 ? frequency : Stopwatch.Frequency);
        _ranks = new long[ranks][];
        for (int r = 0; r < ranks; r++)
            _ranks[r] = new long[Enum.GetValues<MuseSparkRankPhase>().Length];
        _rankIntervals = new long[ranks];
        _start = _last = _clock();
    }

    internal void Next(MuseSparkPhase phase)
    {
        long now = _clock();
        _caller[(int)_phase] += now - _last;
        _last = now;
        _phase = phase;
    }

    internal RankInterval BeginRank(int rank, MuseSparkRankPhase phase)
    {
        if ((uint)rank >= (uint)_ranks.Length) throw new ArgumentOutOfRangeException(nameof(rank));
        _rankIntervals[rank]++;
        return new RankInterval(this, rank, phase);
    }

    internal MuseSparkPhaseReport Complete(long sequence, string path, int rows, int cacheStart, bool failed)
    {
        if (_completed) throw new InvalidOperationException("A phase record can only complete once.");
        _completed = true;
        long end = _clock();
        _caller[(int)_phase] += end - _last;
        var caller = new Dictionary<string, double>();
        foreach (var phase in Enum.GetValues<MuseSparkPhase>())
            caller.Add(phase.ToString(), _caller[(int)phase] * _msPerTick);
        var ranks = new Dictionary<string, double>[_ranks.Length];
        for (int rank = 0; rank < ranks.Length; rank++)
        {
            ranks[rank] = new Dictionary<string, double>();
            foreach (var phase in Enum.GetValues<MuseSparkRankPhase>())
                ranks[rank].Add(phase.ToString(), _ranks[rank][(int)phase] * _msPerTick);
        }
        return new MuseSparkPhaseReport(sequence, path, rows, cacheStart, failed,
            (end - _start) * _msPerTick, caller, ranks, (long[])_rankIntervals.Clone());
    }

    internal sealed class RankInterval : IDisposable
    {
        private readonly MuseSparkPhaseRecorder _owner;
        private readonly int _rank;
        private long _last;
        private MuseSparkRankPhase _phase;
        private bool _disposed;

        internal RankInterval(MuseSparkPhaseRecorder owner, int rank, MuseSparkRankPhase phase)
        {
            _owner = owner; _rank = rank; _phase = phase;
            _last = owner._clock();
        }

        internal void Next(MuseSparkRankPhase phase)
        {
            long now = _owner._clock();
            _owner._ranks[_rank][(int)_phase] += now - _last;
            _last = now; _phase = phase;
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            long now = _owner._clock();
            _owner._ranks[_rank][(int)_phase] += now - _last;
        }
    }
}
