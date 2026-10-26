using System;
using System.Net;

namespace TensorSharp.Distributed;

/// <summary>Whole-layer node pipeline using the same peer mesh and driver protocol as TP.</summary>
public sealed class DistributedLayerSplitGroup : ILayerSplitGroup, INestedTensorParallelGroup, IDistributedExecutionGroup
{
    private readonly TcpCommunicator _tcp;
    private bool _disposed;
    public ITensorParallelGroup LocalGroup { get; }
    public int NodeId { get; }
    public int Degree => LocalGroup.Degree;
    public int NodeCount { get; }
    public int GlobalDegree => checked(Degree * NodeCount);
    public int GlobalRankOffset => checked(Degree * NodeId);
    // Layer placement must never activate the tensor sharding paths.
    public bool IsActive => false;

    public DistributedLayerSplitGroup(ITensorParallelGroup localGroup, int nodeId, IPEndPoint[] endpoints)
    {
        ArgumentNullException.ThrowIfNull(localGroup);
        var config = new DistributedTpConfig(nodeId, localGroup.Degree, endpoints);
        LocalGroup = localGroup;
        NodeId = nodeId;
        NodeCount = config.PeerEndpoints.Length;
        try
        {
            _tcp = new TcpCommunicator(nodeId, config.PeerEndpoints);
            // Reject different placement degrees before any model weight is loaded.
            _tcp.ValidateConfiguration($"layer-split:{Degree}");
            Console.WriteLine($"Distributed layer splitting: node {NodeId}/{NodeCount}, {Degree} local GPU(s), {GlobalDegree} total.");
        }
        catch
        {
            _tcp?.Dispose();
            LocalGroup.Dispose();
            throw;
        }
    }

    public IAllocator GetAllocator(int rank) => LocalGroup.GetAllocator(rank);
    public void Send(float[] buffer, int count, int destination)
    {
        ValidateBuffer(buffer, count);
        _tcp.SendTensor(buffer.AsSpan(0, count), destination);
    }
    public void Receive(float[] buffer, int count, int source)
    {
        ValidateBuffer(buffer, count);
        _tcp.ReceiveTensor(buffer.AsSpan(0, count), source);
    }
    private static void ValidateBuffer(float[] buffer, int count)
    {
        ArgumentNullException.ThrowIfNull(buffer);
        if ((uint)count > (uint)buffer.Length) throw new ArgumentOutOfRangeException(nameof(count));
    }
    public void AllReduce(Tensor[] tensors) => throw new NotSupportedException("Whole-layer execution does not reduce weight shards.");
    // Used only to agree on startup metadata, never during layer execution.
    public void CrossNodeAllReduce(float[] buffer, int count)
    {
        ValidateBuffer(buffer, count);
        _tcp.AllReduce(buffer, count);
    }
    public void Synchronize() => LocalGroup.Synchronize();
    public void Barrier() { Synchronize(); _tcp.Barrier(); }
    public void BroadcastControl(int op, int[] payload) => _tcp.BroadcastControl(op, payload);
    public (int op, int[] payload) ReceiveControl() => _tcp.ReceiveControl();
    public void Abort(Exception failure) => _tcp.Abort(failure);
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _tcp.Dispose();
        LocalGroup.Dispose();
    }
}
