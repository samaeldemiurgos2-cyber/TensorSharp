using System;

namespace TensorSharp;

/// <summary>
/// Ordered whole-layer pipeline. Each node owns a contiguous range of layers,
/// distributed over its local devices. Activations travel to the next node;
/// the final node returns logits to the driver. No tensor reduction is issued.
/// </summary>
public interface ILayerSplitGroup : ITensorParallelGroup
{
    int NodeId { get; }

    /// <summary>Send exactly <paramref name="count"/> lossless 32-bit words.</summary>
    void Send(float[] buffer, int count, int destination);

    /// <summary>Receive exactly <paramref name="count"/> lossless 32-bit words.</summary>
    void Receive(float[] buffer, int count, int source);
}
