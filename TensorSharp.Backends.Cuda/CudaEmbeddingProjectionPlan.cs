// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.Threading;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda;

/// <summary>Static range proof for one exact, immutable post-RMS projection.
/// This is an exponent/partial-dot proof, not an accuracy guarantee for Half
/// operand rounding. A proof alone cannot request Half execution; the plan
/// must also create an immutable weight descriptor and own normalization.</summary>
public sealed class CudaEmbeddingProjectionProof
{
    public int Inputs { get; }
    public int Outputs { get; }
    public double MaximumGain { get; }
    public double NormalizedComponentBound { get; }
    public double MaximumPartialDotBound { get; }
    public double MaximumWeight { get; }
    public const double Limit = 16384;

    private CudaEmbeddingProjectionProof(int inputs, int outputs, double gain, double component, double dot, double weight)
    { Inputs = inputs; Outputs = outputs; MaximumGain = gain; NormalizedComponentBound = component; MaximumPartialDotBound = dot; MaximumWeight = weight; }

    /// <summary>Same complete/partial Cauchy-L2 bounds as the model's
    /// CanUseHalfProjection. Weights are original F32, row-major [out,in].
    /// The twofold margin covers narrowing/norm rounding; absolute subnormal
    /// rounding and the actual rounded Half weights are included.</summary>
    public static bool TryCreate(ReadOnlySpan<float> gains, ReadOnlySpan<float> rowMajorWeights, int inputs,
        out CudaEmbeddingProjectionProof proof)
    {
        proof = null;
        if (inputs <= 0 || gains.Length != inputs || rowMajorWeights.Length == 0 || rowMajorWeights.Length % inputs != 0) return false;
        double maxGain = 0;
        foreach (float gain in gains)
        { if (!float.IsFinite(gain)) return false; maxGain = Math.Max(maxGain, Math.Abs((double)gain)); }
        double component = 2 * Math.Sqrt(inputs) * maxGain;
        if (component > Limit) return false;
        double maxDot = 0, maxWeight = 0;
        for (int row = 0; row < rowMajorWeights.Length; row += inputs)
        {
            double squared = 0, absoluteHalfWeights = 0;
            for (int column = 0; column < inputs; column++)
            {
                float value = rowMajorWeights[row + column];
                if (!float.IsFinite(value) || Math.Abs(value) > Limit) return false;
                double half = (double)(System.Half)value;
                double weighted = gains[column] * half;
                squared += weighted * weighted; absoluteHalfWeights += Math.Abs(half);
                maxWeight = Math.Max(maxWeight, Math.Abs((double)value));
            }
            double dot = 2 * Math.Sqrt(inputs * squared) + Math.ScaleB(absoluteHalfWeights, -24);
            if (!double.IsFinite(dot) || dot > Limit) return false;
            maxDot = Math.Max(maxDot, dot);
        }
        proof = new(inputs, rowMajorWeights.Length / inputs, maxGain, component, maxDot, maxWeight);
        return true;
    }
}

/// <summary>Encoder-owned, experimental normalized Half projections. The plan
/// owns the normalization-to-Half boundary, immutable gain/weight snapshots and
/// a bounded input workspace. It has no global cache or environment default.
/// Use on the allocator's serialized stream and dispose before that allocator.</summary>
public sealed class CudaEmbeddingProjectionPlan : IDisposable
{
    /// <summary>CUDA 13.4 cublas_api.h:208-212: default F32 compute/intermediates (68), Half A/B,
    /// F32 C and F32 alpha/beta. Split-K output reductions remain F32 because
    /// output and accumulator have the same type. No handle math mode changes.
    /// https://docs.nvidia.com/cuda/cublas/index.html#cublascomputetype-t
    /// https://docs.nvidia.com/cuda/cublas/index.html#gemm-algorithms-numerical-behavior</summary>
    public const int Float32ComputeType = 68;
    public const int PedanticComputeType = 69;
    public const int MaximumPreparedWeights = 256;
    private readonly CudaAllocator _allocator;
    private readonly List<PreparedWeight> _weights = new();
    private readonly List<Tensor> _retiredInputs = new();
    private Tensor _halfInput;
    private long _inputElements, _retiredInputBytes;
    private bool _pending, _disposed;
    public int ComputeType { get; }
    public long WeightBudgetBytes { get; }
    public long WorkspaceBudgetBytes { get; }
    public long WeightCapacityBytes { get; private set; }
    public long WorkspaceCapacityBytes => (_halfInput == null ? 0 : AllocationBytes(_inputElements)) + _retiredInputBytes;
    public int PreparedWeightCount => _weights.Count;
    public long SuccessfulCalls { get; private set; }
    public long SuccessfulRows { get; private set; }
    public long WeightUses { get; private set; }
    public long WorkspaceReuses { get; private set; }
    public long ProofRejections { get; private set; }
    public long BudgetRejections { get; private set; }

    /// <summary>Descriptor cannot accept mutable replacement weights/gains or
    /// expose its resident Half tensor. Its proof and storage come from the
    /// same immutable constructor snapshot.</summary>
    public sealed class PreparedWeight
    {
        internal readonly CudaEmbeddingProjectionPlan Owner;
        internal readonly Tensor Gains, HalfWeights;
        internal readonly long Bytes;
        public CudaEmbeddingProjectionProof Proof { get; }
        internal PreparedWeight(CudaEmbeddingProjectionPlan owner, Tensor gains, Tensor weights, long bytes, CudaEmbeddingProjectionProof proof)
        { Owner = owner; Gains = gains; HalfWeights = weights; Bytes = bytes; Proof = proof; }
    }

    public CudaEmbeddingProjectionPlan(CudaAllocator allocator, bool pedantic = false,
        long weightBudgetBytes = 512L * 1024 * 1024, long workspaceBudgetBytes = 128L * 1024 * 1024)
    {
        _allocator = allocator ?? throw new ArgumentNullException(nameof(allocator));
        if (weightBudgetBytes < 256 || weightBudgetBytes > 512L * 1024 * 1024) throw new ArgumentOutOfRangeException(nameof(weightBudgetBytes));
        if (workspaceBudgetBytes < 256 || workspaceBudgetBytes > 128L * 1024 * 1024) throw new ArgumentOutOfRangeException(nameof(workspaceBudgetBytes));
        ComputeType = pedantic ? PedanticComputeType : Float32ComputeType;
        WeightBudgetBytes = weightBudgetBytes; WorkspaceBudgetBytes = workspaceBudgetBytes;
    }

    /// <summary>Prepare exact F32 gains and row-major [out,in] weights at model
    /// load time. No device/host pointer serves as a cache identity. The plan
    /// retains copied Half weights and F32 gains until disposed, including
    /// their host mirrors during asynchronous uploads.</summary>
    public bool TryPrepareWeight(ReadOnlySpan<float> gains, ReadOnlySpan<float> rowMajorWeights, int inputs,
        out PreparedWeight weight, CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this); cancellationToken.ThrowIfCancellationRequested(); weight = null;
        if (!CudaEmbeddingProjectionProof.TryCreate(gains, rowMajorWeights, inputs, out var proof))
        { ProofRejections++; return false; }
        long bytes = AllocationBytes((long)inputs * proof.Outputs) + CudaDeviceMemoryPool.RoundAllocationSize((long)inputs * sizeof(float));
        if (_weights.Count >= MaximumPreparedWeights || bytes > WeightBudgetBytes - WeightCapacityBytes) { BudgetRejections++; return false; }
        if (_allocator.Kernels == null) return false;
        Tensor norm = null, half = null;
        try
        {
            norm = new Tensor(_allocator, DType.Float32, inputs); norm.SetElementsAsFloat(gains.ToArray());
            half = new Tensor(_allocator, DType.Float16, proof.Outputs, inputs);
            // CUDA's bulk float setter accepts F32 storage only. Snapshot the
            // proven finite weights as IEEE Half bits into the owned host mirror;
            // that mirror remains alive for the asynchronous upload below.
            var roundedWeights = new ushort[rowMajorWeights.Length];
            for (int index = 0; index < roundedWeights.Length; index++)
                roundedWeights[index] = BitConverter.HalfToUInt16Bits((System.Half)rowMajorWeights[index]);
            unsafe
            {
                fixed (ushort* source = roundedWeights)
                    half.Storage.CopyToStorage(half.StorageOffset, (IntPtr)source, (long)roundedWeights.Length * sizeof(ushort));
            }
            cancellationToken.ThrowIfCancellationRequested();
            // Transfer ownership before uploads. If a CUDA operation throws,
            // the plan retains mirrors even if draining the stream also fails.
            var prepared = new PreparedWeight(this, norm, half, bytes, proof);
            _weights.Add(prepared); WeightCapacityBytes += bytes; norm = half = null;
            _pending = true; ((CudaStorage)prepared.Gains.Storage).EnsureDeviceCurrent();
            ((CudaStorage)prepared.HalfWeights.Storage).EnsureDeviceCurrent();
            weight = prepared; return true;
        }
        finally { norm?.Dispose(); half?.Dispose(); }
    }

    /// <summary>Only this atomic API executes Half projection: it computes
    /// weighted F32 RMSNorm from finite raw input using the descriptor's exact
    /// gains, then narrows its own result. An arbitrary supplied normalized
    /// tensor is never trusted; normalizedOutput is overwritten before casting.
    /// Raw 80K input before RMSNorm is supported. Nonfinite inputs propagate
    /// as in the existing F32 RMS operation; callers must supply finite model
    /// activations. Restrict integration to post-norm QKV and gate/up sites.
    /// Returns false before writes for unsupported layouts/aliases/budgets.</summary>
    public bool TryNormalizeAndProject(Tensor result, Tensor normalizedOutput, Tensor rawInput, PreparedWeight weight,
        float epsilon, CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this); cancellationToken.ThrowIfCancellationRequested();
        if (weight == null || !ReferenceEquals(weight.Owner, this) || !float.IsFinite(epsilon) || epsilon <= 0 ||
            !Valid(result) || !Valid(normalizedOutput) || !Valid(rawInput) ||
            ReferenceEquals(result.Storage, rawInput.Storage) || ReferenceEquals(result.Storage, normalizedOutput.Storage) ||
            ReferenceEquals(normalizedOutput.Storage, rawInput.Storage)) return false;
        long rows = rawInput.Sizes[0], inputs = weight.Proof.Inputs, outputs = weight.Proof.Outputs;
        if (rows <= 0 || rows > int.MaxValue || rawInput.Sizes[1] != inputs || normalizedOutput.Sizes[0] != rows ||
            normalizedOutput.Sizes[1] != inputs || result.Sizes[0] != rows || result.Sizes[1] != outputs) return false;
        long elements = checked(rows * inputs);
        if (elements > int.MaxValue || _allocator.Kernels == null) return false;
        if (!PrepareWorkspace(elements)) return false;
        cancellationToken.ThrowIfCancellationRequested();
        _allocator.Context.MakeCurrent();
        _pending = true;
        var normStorage = (CudaStorage)normalizedOutput.Storage; var resultStorage = (CudaStorage)result.Storage;
        // Partial views must retain the caller's surrounding host values on
        // device before either writer invalidates the complete host mirror.
        if (normalizedOutput.StorageOffset != 0 || normalizedOutput.ElementCount() != normStorage.ElementCount)
            normStorage.EnsureDeviceCurrent();
        if (result.StorageOffset != 0 || result.ElementCount() != resultStorage.ElementCount)
            resultStorage.EnsureDeviceCurrent();
        if (!CudaKernelOps.TryRMSNorm(normalizedOutput, rawInput, weight.Gains, null, epsilon)) return false;
        var halfStorage = (CudaStorage)_halfInput.Storage; var weightStorage = (CudaStorage)weight.HalfWeights.Storage;
        _allocator.Kernels.LaunchConvertF32F16(normStorage.DevicePtrAtElement(normalizedOutput.StorageOffset),
            halfStorage.DevicePtrAtElement(0), elements, _allocator.Stream.Handle);
        _allocator.Blas.SetStream(_allocator.Stream.Handle);
        float alpha = 1, beta = 0;
        // Row-major original W[out,in] is the column-major W[in,out] view;
        // transpose W, leave narrowed X[in,rows] unchanged, write Y[out,rows].
        CublasApi.cublasGemmEx(_allocator.Blas.Handle, CublasApi.CUBLAS_OP_T, CublasApi.CUBLAS_OP_N,
            (int)outputs, (int)rows, (int)inputs, ref alpha,
            weightStorage.DevicePtrAtElement(0), CublasApi.CUDA_R_16F, (int)inputs,
            halfStorage.DevicePtrAtElement(0), CublasApi.CUDA_R_16F, (int)inputs, ref beta,
            resultStorage.DevicePtrAtElement(result.StorageOffset), CublasApi.CUDA_R_32F, (int)outputs,
            ComputeType, CublasApi.CUBLAS_GEMM_DEFAULT).ThrowOnCublasError();
        resultStorage.MarkDeviceModified(); SuccessfulCalls++; SuccessfulRows += rows; WeightUses++; return true;
    }

    private bool Valid(Tensor value) => value != null && value.ElementType == DType.Float32 && value.DimensionCount == 2 &&
        value.IsContiguous() && value.StorageOffset >= 0 && value.StorageOffset <= value.Storage.ElementCount &&
        value.ElementCount() <= value.Storage.ElementCount - value.StorageOffset &&
        value.Storage is CudaStorage storage && ReferenceEquals(storage.AllocatorImpl, _allocator);
    private static long AllocationBytes(long halfElements) => CudaDeviceMemoryPool.RoundAllocationSize(checked(halfElements * 2));
    private bool PrepareWorkspace(long elements)
    {
        if (_halfInput != null && _inputElements >= elements) { WorkspaceReuses++; return true; }
        long bytes = AllocationBytes(elements);
        long retained = _pending ? WorkspaceCapacityBytes : 0;
        if (bytes > WorkspaceBudgetBytes - retained) { BudgetRejections++; return false; }
        var replacement = new Tensor(_allocator, DType.Float16, elements);
        if (_halfInput != null)
        {
            if (_pending) { _retiredInputs.Add(_halfInput); _retiredInputBytes += AllocationBytes(_inputElements); }
            else _halfInput.Dispose();
        }
        _halfInput = replacement; _inputElements = elements; return true;
    }

    /// <summary>Call only after the caller's ordinary output readback completed.
    /// No synchronization is issued. Earlier scratch allocations can be freed.</summary>
    public void NotifyReadbackCompleted()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        _pending = false; foreach (var tensor in _retiredInputs) tensor.Dispose(); _retiredInputs.Clear(); _retiredInputBytes = 0;
    }

    /// <summary>Drain only fault/cancellation cleanup, before releasing storage
    /// potentially referenced by queued kernels or uploads.</summary>
    public void AbortPendingWork()
    {
        if (_pending && !_allocator.Context.IsDisposed) { _allocator.Context.MakeCurrent(); _allocator.Stream.Synchronize(); }
        _pending = false; foreach (var tensor in _retiredInputs) tensor.Dispose(); _retiredInputs.Clear(); _retiredInputBytes = 0;
    }

    public void Dispose()
    {
        if (_disposed) return;
        // Do not free mirrors if the drain fails; a caller can retry cleanup.
        AbortPendingWork(); _disposed = true;
        _halfInput?.Dispose(); _halfInput = null; _inputElements = 0;
        foreach (var weight in _weights) { weight.HalfWeights.Dispose(); weight.Gains.Dispose(); }
        _weights.Clear(); WeightCapacityBytes = 0;
    }
}
