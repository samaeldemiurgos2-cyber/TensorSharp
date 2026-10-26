// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
namespace TensorSharp.GGML;

/// <summary>
/// Immutable host-pool snapshot. Counters are cumulative for this context;
/// they count host allocation/release operations, not device allocations or
/// kernel time. Snapshots taken after workers join include completed releases.
/// </summary>
public readonly record struct GgmlMemoryPoolStatistics(
    long PoolHits, long FreshAllocations, long SystemFrees, long SystemFreeFailures,
    int RetainedBlocks, int OutstandingBlocks, long RetainedBytes, long PeakRetainedBytes,
    int MaxPooledBlocks, long MaxRetainedBlockBytes, bool IsClosed)
{
    /// <summary>Whether this pool opted into large Linux CUDA host-mapping hints.</summary>
    public bool HugePageAdviceEnabled { get; init; }
    public long HugePageAdviceRequests { get; init; }
    /// <summary>Hints accepted by madvise, not proof of physical huge-page backing.
    /// Inspect /proc/PID/smaps for actual AnonHugePages.</summary>
    public long HugePageAdviceAccepted { get; init; }
    public long HugePageAdviceFailures { get; init; }
    /// <summary>Cumulative bytes in accepted hints, not live retained bytes.</summary>
    public long HugePageAdvisedBytes { get; init; }
}
