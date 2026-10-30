// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.Threading;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda;

/// <summary>Per-call GEMM policy; TF32 is an explicit approximation diagnostic
/// retaining F32 storage/exponents, not a full-mantissa F32 guarantee.</summary>
public enum CudaEmbeddingAttentionComputePolicy
{
    PedanticF32,
    FastTf32WithF32Storage,
}

/// <summary>Encoder-owned score workspace for accurate F32 embedding attention.
/// Reuse on one serialized allocator stream; dispose before that allocator.
/// No model/device-weight cache entries or process-global state are owned.</summary>
public sealed class CudaEmbeddingAttentionPlan : IDisposable
{
    internal CudaAllocator Allocator { get; }
    private Tensor _scores;
    private bool _disposed;
    private readonly Dictionary<(int Length, int Radius), MaskEntry> _masks = new();
    private bool _pendingMaskWork;
    private long _maskClock;
    private sealed class MaskEntry
    {
        public Tensor Tensor;
        public long LastUse;
    }
    public int HeadTile { get; }
    public CudaEmbeddingAttentionComputePolicy ComputePolicy { get; }
    public int SelectedComputeType { get; }
    public long ScoreBudgetBytes { get; }
    /// <summary>Requested live tensor capacity; the allocator may round its
    /// underlying pooled allocation. Zero after this plan has been disposed.</summary>
    public long ScoreCapacityBytes => (_scores?.ElementCount() ?? 0) * sizeof(float);
    public long SuccessfulCalls { get; private set; }
    public long SuccessfulSegments { get; private set; }
    public long SuccessfulHeadTiles { get; private set; }
    public long SuccessfulQueryRows { get; private set; }
    public long SuccessfulWindowCalls { get; private set; }
    public int MaskEntries => _masks.Count;
    public long MaskCapacityBytes { get; private set; }
    public long MaskCacheHits { get; private set; }
    public long MaskUploads { get; private set; }
    public long MaskEvictions { get; private set; }
    public const int MaximumWindowSequence = 2048;
    public const int MaximumMaskEntries = 4;
    public const long MaskBudgetBytes = 64L * 1024 * 1024;
    public const int ComputeType = 69; // CUBLAS_COMPUTE_32F_PEDANTIC, CUDA 13.4 cublas_api.h.
    public const int Tf32ComputeType = 77; // CUBLAS_COMPUTE_32F_FAST_TF32, CUDA 13.4 cublas_api.h.
    internal static bool SupportsFastTf32(int computeCapabilityMajor) => computeCapabilityMajor >= 8;

    public CudaEmbeddingAttentionPlan(CudaAllocator allocator, int headTile = 2,
        long scoreBudgetBytes = 512L * 1024 * 1024,
        CudaEmbeddingAttentionComputePolicy computePolicy = CudaEmbeddingAttentionComputePolicy.PedanticF32)
    {
        Allocator = allocator ?? throw new ArgumentNullException(nameof(allocator));
        if (headTile is < 1 or > 2) throw new ArgumentOutOfRangeException(nameof(headTile));
        if (scoreBudgetBytes < sizeof(float) || scoreBudgetBytes > 512L * 1024 * 1024)
            throw new ArgumentOutOfRangeException(nameof(scoreBudgetBytes), "Attention requires at most 512 MiB of score workspace.");
        if (computePolicy is not (CudaEmbeddingAttentionComputePolicy.PedanticF32 or CudaEmbeddingAttentionComputePolicy.FastTf32WithF32Storage))
            throw new ArgumentOutOfRangeException(nameof(computePolicy));
        // A requested approximation must not silently become a pre-Ampere F32
        // fallback and then be reported as qualified TF32 policy execution.
        // cuBLAS still chooses its algorithm per eligible call; this capability
        // check does not promise that every matrix uses a Tensor Core kernel.
        if (computePolicy == CudaEmbeddingAttentionComputePolicy.FastTf32WithF32Storage &&
            !SupportsFastTf32(CudaDevice.GetDevice(allocator.DeviceId).ComputeCapabilityMajor))
            throw new NotSupportedException("Embedding TF32 attention requires SM80 or newer.");
        ComputePolicy = computePolicy;
        SelectedComputeType = computePolicy == CudaEmbeddingAttentionComputePolicy.PedanticF32 ? ComputeType : Tf32ComputeType;
        HeadTile = headTile; ScoreBudgetBytes = scoreBudgetBytes;
    }

    internal bool TryPrepare(int longestSequence, int groupHeads, out int tileHeads)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        long oneHeadBytes = (long)longestSequence * longestSequence * sizeof(float);
        tileHeads = (int)Math.Min(Math.Min(HeadTile, groupHeads), ScoreBudgetBytes / oneHeadBytes);
        if (tileHeads < 1) return false;
        long elements = (long)tileHeads * longestSequence * longestSequence;
        if (_scores == null || _scores.ElementCount() < elements)
        {
            var larger = new Tensor(Allocator, DType.Float32, elements);
            _scores?.Dispose(); _scores = larger;
        }
        return true;
    }

    internal Tensor ScoreRows(int heads, int length)
    {
        using var active = _scores.Narrow(0, 0, (long)heads * length * length);
        return active.View((long)heads * length, length);
    }

    // All required keys are reserved before attention dispatch. A different
    // request may evict only after its caller's ordinary output readback has
    // completed; this retains host upload mirrors while asynchronous copies
    // can still reference them. Repeated layers need no waits or new uploads.
    internal bool TryPrepareMasks(int[] lengths, int radius)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var required = new HashSet<(int Length, int Radius)>();
        long requiredBytes = 0;
        foreach (int length in lengths)
        {
            if (length <= radius + 1) continue; // Every key is already visible.
            var key = (length, radius);
            if (required.Add(key)) requiredBytes += (long)length * length * sizeof(float);
        }
        if (required.Count > MaximumMaskEntries || requiredBytes > MaskBudgetBytes) return false;
        int missing = 0; long missingBytes = 0;
        foreach (var key in required)
            if (!_masks.ContainsKey(key)) { missing++; missingBytes += (long)key.Length * key.Length * sizeof(float); }
        if (_pendingMaskWork && (_masks.Count + missing > MaximumMaskEntries ||
            MaskCapacityBytes + missingBytes > MaskBudgetBytes)) return false;
        while (_masks.Count + missing > MaximumMaskEntries || MaskCapacityBytes + missingBytes > MaskBudgetBytes)
        {
            (int Length, int Radius) victim = default; long oldest = long.MaxValue;
            foreach (var pair in _masks)
                if (!required.Contains(pair.Key) && pair.Value.LastUse < oldest)
                { victim = pair.Key; oldest = pair.Value.LastUse; }
            if (oldest == long.MaxValue) return false;
            _masks[victim].Tensor.Dispose(); _masks.Remove(victim);
            MaskCapacityBytes -= (long)victim.Length * victim.Length * sizeof(float); MaskEvictions++;
        }
        foreach (var key in required)
        {
            if (_masks.TryGetValue(key, out var cached))
            { cached.LastUse = ++_maskClock; MaskCacheHits++; continue; }
            var values = new float[checked(key.Length * key.Length)];
            for (int row = 0; row < key.Length; row++)
            for (int column = 0; column < key.Length; column++)
                values[row * key.Length + column] = Math.Abs(row - column) <= radius ? 0f : float.NegativeInfinity;
            var mask = Tensor.FromArray(Allocator, values);
            _masks.Add(key, new MaskEntry { Tensor = mask, LastUse = ++_maskClock });
            MaskCapacityBytes += values.LongLength * sizeof(float);
            // Set before upload so exceptional cleanup cannot free its mirror
            // while an issued asynchronous copy is still pending.
            _pendingMaskWork = true;
            ((CudaStorage)mask.Storage).EnsureDeviceCurrent(); MaskUploads++;
        }
        return true;
    }

    internal Tensor WindowMask(int length, int radius)
    {
        if (length <= radius + 1) return null;
        _pendingMaskWork = true;
        return _masks[(length, radius)].Tensor;
    }

    /// <summary>Call only after the encoder's existing final output readback or
    /// another completed synchronization on this allocator stream. This method
    /// performs no synchronization; it permits safe cache eviction on the next
    /// request. Never call immediately after enqueueing attention.</summary>
    public void NotifyReadbackCompleted()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        _pendingMaskWork = false;
    }

    /// <summary>Exceptional cleanup for an interrupted window-attention step.
    /// Drains only when mask uploads/uses remain pending, retaining their host
    /// mirrors until the stream completes. Normal readback adds no new wait.</summary>
    public void AbortPendingWork()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (!_pendingMaskWork) return;
        if (!Allocator.Context.IsDisposed) Allocator.Synchronize();
        _pendingMaskWork = false;
    }

    internal void RecordSuccess(int segments, long tiles, long queryRows, bool window)
    {
        SuccessfulCalls++; SuccessfulSegments += segments;
        SuccessfulHeadTiles += tiles; SuccessfulQueryRows += queryRows;
        if (window) SuccessfulWindowCalls++;
    }

    public void Dispose()
    {
        if (_disposed) return;
        AbortPendingWork();
        foreach (var mask in _masks.Values) mask.Tensor.Dispose();
        _masks.Clear(); MaskCapacityBytes = 0;
        _scores?.Dispose(); _scores = null; _disposed = true;
    }
}

/// <summary>Materialized embedding attention with F32 storage, exponent range,
/// and pedantic F32 cuBLAS score/value products by default. An explicitly
/// selected plan may use approximate TF32 for both products. No Half activation
/// narrowing, per-call synchronize, K/V repeat, custom PTX, or global math-mode override.</summary>
public static class CudaEmbeddingAttentionOps
{
    /// <summary>Token-major [batch,tokens,heads,dimensions] attention. Equal
    /// batches supply one length per batch. Packed batch1 supplies each complete
    /// sequence length; their sum must equal tokens. Every segment is isolated.
    /// This API has no window mask: callers may use it for local attention only
    /// when each complete sequence is within that window's inclusive radius.</summary>
    public static bool TryGlobalAttention(Tensor result, Tensor query, Tensor key, Tensor value,
        CudaEmbeddingAttentionPlan plan, int[] sequenceLengths, float scale = 1f, CancellationToken cancellationToken = default)
        => TryAttention(result, query, key, value, plan, sequenceLengths, null, scale, cancellationToken);

    /// <summary>Symmetric, bidirectional local attention: a query may see keys
    /// at inclusive distance &lt;= windowRadius within its own segment. Sequences
    /// above2048 and requests exceeding the4-entry/64MiB mask cache return false
    /// before dispatch. Storage stays F32; GEMMs use the plan's per-call policy
    /// (pedantic69 by default). Notify the plan
    /// after normal output readback before a later request may evict masks.</summary>
    public static bool TryWindowAttention(Tensor result, Tensor query, Tensor key, Tensor value,
        CudaEmbeddingAttentionPlan plan, int[] sequenceLengths, int windowRadius, float scale = 1f, CancellationToken cancellationToken = default)
        => windowRadius < 0 || windowRadius >= CudaEmbeddingAttentionPlan.MaximumWindowSequence ? false :
            TryAttention(result, query, key, value, plan, sequenceLengths, windowRadius, scale, cancellationToken);

    private static bool TryAttention(Tensor result, Tensor query, Tensor key, Tensor value,
        CudaEmbeddingAttentionPlan plan, int[] sequenceLengths, int? windowRadius, float scale, CancellationToken cancellationToken)
    {
        // A canceled start performs no allocation/upload/write. A later cancel
        // before GEMM dispatch throws; queued uploads/GEMMs are not interrupted,
        // and the encoder must drain its exceptional path before disposal.
        cancellationToken.ThrowIfCancellationRequested();
        if (plan == null || sequenceLengths == null || sequenceLengths.Length == 0 || !float.IsFinite(scale) ||
            result == null || query == null || key == null || value == null ||
            result.DimensionCount != 4 || query.DimensionCount != 4 || key.DimensionCount != 4 || value.DimensionCount != 4 ||
            !CudaKernelOps.TryGetContiguousFloat(result, out var ys, out _, out _) ||
            !CudaKernelOps.TryGetContiguousFloat(query, out var qs, out _, out _) ||
            !CudaKernelOps.TryGetContiguousFloat(key, out var ks, out _, out _) ||
            !CudaKernelOps.TryGetContiguousFloat(value, out var vs, out _, out _) ||
            !ReferenceEquals(plan.Allocator, ys.AllocatorImpl) || !ReferenceEquals(plan.Allocator, qs.AllocatorImpl) ||
            !ReferenceEquals(plan.Allocator, ks.AllocatorImpl) || !ReferenceEquals(plan.Allocator, vs.AllocatorImpl) ||
            ReferenceEquals(ys, qs) || ReferenceEquals(ys, ks) || ReferenceEquals(ys, vs)) return false;
        for (int axis = 0; axis < 4; axis++)
            if (query.Sizes[axis] <= 0 || result.Sizes[axis] != query.Sizes[axis] || key.Sizes[axis] <= 0 ||
                value.Sizes[axis] != key.Sizes[axis]) return false;
        long batchLong = query.Sizes[0], tokensLong = query.Sizes[1], headsLong = query.Sizes[2], dimensionsLong = query.Sizes[3];
        long kvHeadsLong = key.Sizes[2];
        if (batchLong > 1024 || tokensLong > int.MaxValue || headsLong > 128 || kvHeadsLong > headsLong ||
            headsLong % kvHeadsLong != 0 || dimensionsLong > 1024 ||
            key.Sizes[0] != batchLong || key.Sizes[1] != tokensLong || key.Sizes[3] != dimensionsLong) return false;
        int batch = (int)batchLong, tokens = (int)tokensLong, heads = (int)headsLong;
        int kvHeads = (int)kvHeadsLong, dimensions = (int)dimensionsLong, groupHeads = heads / kvHeads;
        int longest = 0; long total = 0;
        foreach (int length in sequenceLengths)
        {
            if (length <= 0 || length > (windowRadius.HasValue ? CudaEmbeddingAttentionPlan.MaximumWindowSequence : 8192)) return false;
            longest = Math.Max(longest, length); total += length;
        }
        if (batch > 1)
        {
            if (sequenceLengths.Length != batch) return false;
            foreach (int length in sequenceLengths) if (length != tokens) return false;
        }
        else if (total != tokens) return false;
        // Prepare the largest segment before any dispatch so ragged segments
        // never resize/free their score workspace partway through a step.
        CudaAllocator allocator = plan.Allocator;
        allocator.Context.MakeCurrent();
        if (!plan.TryPrepare(longest, groupHeads, out int maximumTile)) return false;
        if (!CudaKernelOps.TryGetKernels(allocator, out _)) return false;
        if (windowRadius.HasValue && !plan.TryPrepareMasks(sequenceLengths, windowRadius.Value)) return false;
        cancellationToken.ThrowIfCancellationRequested();
        qs.EnsureDeviceCurrent(); ks.EnsureDeviceCurrent(); vs.EnsureDeviceCurrent(); ys.EnsureDeviceCurrent();
        allocator.Blas.SetStream(allocator.Stream.Handle);
        int start = 0; long tiles = 0, queryRows = 0;
        for (int segment = 0; segment < sequenceLengths.Length; segment++)
        {
            int length = sequenceLengths[segment];
            long rowBase = batch > 1 ? (long)segment * tokens : start;
            for (int kv = 0; kv < kvHeads; kv++)
            for (int first = 0; first < groupHeads; first += maximumTile)
            {
                int tileHeads = Math.Min(maximumTile, groupHeads - first), firstHead = kv * groupHeads + first;
                using var scores = plan.ScoreRows(tileHeads, length);
                var ss = (CudaStorage)scores.Storage;
                IntPtr scorePointer = ss.DevicePtrAtElement(scores.StorageOffset);
                IntPtr q = qs.DevicePtrAtElement(query.StorageOffset + rowBase * heads * dimensions + firstHead * dimensions);
                IntPtr k = ks.DevicePtrAtElement(key.StorageOffset + rowBase * kvHeads * dimensions + kv * dimensions);
                IntPtr v = vs.DevicePtrAtElement(value.StorageOffset + rowBase * kvHeads * dimensions + kv * dimensions);
                IntPtr y = ys.DevicePtrAtElement(result.StorageOffset + rowBase * heads * dimensions + firstHead * dimensions);
                float alpha = scale, beta = 0;
                // Column-major K[D,key]^T * Q[D,query] produces scores[key,query].
                // Its memory is the desired contiguous row-major [query,key].
                // K stride0 broadcasts one KV head across this GQA query group.
                CublasApi.cublasGemmStridedBatchedEx(allocator.Blas.Handle, CublasApi.CUBLAS_OP_T, CublasApi.CUBLAS_OP_N,
                    length, length, dimensions, ref alpha,
                    k, CublasApi.CUDA_R_32F, kvHeads * dimensions, 0,
                    q, CublasApi.CUDA_R_32F, heads * dimensions, dimensions, ref beta,
                    scorePointer, CublasApi.CUDA_R_32F, length, (long)length * length, tileHeads,
                    plan.SelectedComputeType, CublasApi.CUBLAS_GEMM_DEFAULT).ThrowOnCublasError();
                ss.MarkDeviceModified();
                if (windowRadius.HasValue)
                {
                    Tensor mask = plan.WindowMask(length, windowRadius.Value);
                    if (mask != null)
                    {
                        // Flatten one NxN mask per score head: the existing
                        // row-broadcast kernel applies it to every head tile.
                        using var headsByMask = scores.View(tileHeads, (long)length * length);
                        if (!CudaKernelOps.TryBinaryRowBroadcast(headsByMask, headsByMask, mask, CudaBinaryOp.Add))
                            throw new NotSupportedException("CUDA embedding window attention requires the existing F32 row-broadcast kernel.");
                    }
                }
                if (!CudaKernelOps.TrySoftmax(scores, scores))
                    throw new NotSupportedException("CUDA embedding attention requires the existing F32 row softmax kernel.");
                alpha = 1;
                // V[D,key] * P[key,query] produces output[D,query]. Leading
                // dimension preserves token-major interleaving; head matrices
                // write disjoint head elements within every query token.
                CublasApi.cublasGemmStridedBatchedEx(allocator.Blas.Handle, CublasApi.CUBLAS_OP_N, CublasApi.CUBLAS_OP_N,
                    dimensions, length, length, ref alpha,
                    v, CublasApi.CUDA_R_32F, kvHeads * dimensions, 0,
                    scorePointer, CublasApi.CUDA_R_32F, length, (long)length * length, ref beta,
                    y, CublasApi.CUDA_R_32F, heads * dimensions, dimensions, tileHeads,
                    plan.SelectedComputeType, CublasApi.CUBLAS_GEMM_DEFAULT).ThrowOnCublasError();
                tiles++; queryRows += (long)tileHeads * length;
            }
            if (batch == 1) start += length;
        }
        ys.MarkDeviceModified();
        plan.RecordSuccess(sequenceLengths.Length, tiles, queryRows, windowRadius.HasValue);
        return true;
    }
}
