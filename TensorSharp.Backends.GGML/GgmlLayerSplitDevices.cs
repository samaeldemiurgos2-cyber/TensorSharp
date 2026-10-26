using System;

namespace TensorSharp.GGML;

/// <summary>Local devices for whole-layer placement. No sharding or collective initialization.</summary>
public sealed class GgmlLayerSplitDevices : ITensorParallelGroup
{
    private readonly GgmlAllocator[] _allocators;
    private bool _disposed;
    public GgmlContext Context { get; }

    public GgmlLayerSplitDevices(GgmlContext context)
    {
        Context = context ?? throw new ArgumentNullException(nameof(context));
        _allocators = new GgmlAllocator[context.Degree];
        for (int i = 0; i < _allocators.Length; ++i)
            _allocators[i] = new GgmlAllocator(context, i);
    }

    public int Degree => _allocators.Length;
    public int GlobalDegree => Degree;
    public int GlobalRankOffset => 0;
    public int NodeCount => 1;
    public bool IsActive => false;
    public IAllocator GetAllocator(int rank) => _allocators[rank];
    public void AllReduce(Tensor[] tensors) => throw new NotSupportedException("Layer splitting does not reduce tensors.");
    public void BroadcastControl(int op, int[] payload) => throw new NotSupportedException();
    public (int op, int[] payload) ReceiveControl() => throw new NotSupportedException();
    public void Synchronize() => GgmlBasicOps.HostReadBarrier();
    public void Barrier() => Synchronize();
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        Context.ReleasePoolForDisposal();
    }
}
