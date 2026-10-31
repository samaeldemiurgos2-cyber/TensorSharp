// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using TensorSharp.Cuda;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Models;

/// <summary>
/// The on-node group of a cross-node layer split on <c>--backend cuda</c>: only the GPU count. The direct-CUDA
/// engines create their own per-device contexts, allocators and streams, so this group initializes no ggml
/// backend, no meta-backend and no peer access, and it never reduces anything: whole-layer stages exchange their
/// boundary state through the node pipeline (<see cref="ILayerSplitGroup.Send"/> / Receive).
/// </summary>
public sealed class DirectCudaLayerSplitDevices : ITensorParallelGroup
{
    public DirectCudaLayerSplitDevices(int degree)
    {
        if (degree is < 1 or > 8)
            throw new ArgumentOutOfRangeException(nameof(degree), "--layer-split takes 1..8 GPUs per node.");
        int visible;
        try { visible = CudaDevice.GetDeviceCount(); }
        catch (Exception ex) when (ex is DllNotFoundException or EntryPointNotFoundException or CudaException or TypeInitializationException)
        {
            throw new NotSupportedException("--backend cuda needs a CUDA driver, and none could be initialized: " + ex.Message, ex);
        }
        if (visible < degree)
            throw new NotSupportedException($"--layer-split {degree} asks for {degree} CUDA device(s) on this node, but {visible} are visible.");
        Degree = degree;
    }

    public int Degree { get; }
    public int GlobalDegree => Degree;
    public int GlobalRankOffset => 0;
    public int NodeCount => 1;
    // Placement only: never activates tensor sharding.
    public bool IsActive => false;

    public IAllocator GetAllocator(int rank) => throw new NotSupportedException(
        "The direct-CUDA layer-split group carries only its GPU count; the engine owns its device allocators.");

    public void AllReduce(Tensor[] tensors) => throw new NotSupportedException("Layer splitting does not reduce tensors.");
    public void Synchronize() { }
    public void Barrier() { }
    public void BroadcastControl(int op, int[] payload) => throw new NotSupportedException();
    public (int op, int[] payload) ReceiveControl() => throw new NotSupportedException();
    public void Dispose() { }
}
