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
    int MaxPooledBlocks, long MaxRetainedBlockBytes, bool IsClosed);
