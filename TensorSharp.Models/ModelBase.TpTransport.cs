using System;

namespace TensorSharp.Models
{
    public abstract partial class ModelBase
    {
        /// <summary>
        /// Combine a byte buffer across every rank of the tensor-parallel group, all nodes
        /// included: each node passes <paramref name="buffer"/> holding the bytes it owns
        /// (its local ranks' shards) and zeros elsewhere, and on return every node holds
        /// the union. Exact for any bytes (<see cref="TpByteTransport"/>). One collective,
        /// so every node must call it at the same point - on a worker, from the mirrored
        /// operation the driver broadcast.
        /// </summary>
        protected void TpAllReduceBytes(Span<byte> buffer)
        {
            if (!IsTensorParallel || buffer.Length == 0)
                return;
            float[] packed = TpByteTransport.Pack(buffer);
            int n = packed.Length;
            int tp = TpDegree;
            var parts = new Tensor[tp];
            try
            {
                // This node's bytes ride on its local rank 0; its other ranks add nothing.
                parts[0] = new Tensor(_tpGroup.GetAllocator(0), DType.Float32, n);
                parts[0].SetElementsAsFloat(packed);
                for (int r = 1; r < tp; r++)
                {
                    parts[r] = new Tensor(_tpGroup.GetAllocator(r), DType.Float32, n);
                    Ops.Fill(parts[r], 0f);
                }
                _tpGroup.AllReduce(parts);
                TpByteTransport.Unpack(parts[0].GetElementsAsFloat(n), buffer);
            }
            finally
            {
                for (int r = 0; r < tp; r++)
                    parts[r]?.Dispose();
            }
        }

        /// <summary>
        /// Send the driver's <paramref name="buffer"/> to every node: on return each node
        /// holds the driver's bytes. A collective, called by every node at the same point.
        /// A multi-node group with a byte channel (<see cref="ICrossNodeByteGroup"/>) sends
        /// the bytes once over the wire as they are; any other group falls back to
        /// <see cref="TpAllReduceBytes"/>, so a worker's buffer must then be zeros.
        /// </summary>
        protected void TpBroadcastBytesFromDriver(Span<byte> buffer)
        {
            if (TpNodeCount <= 1 || buffer.Length == 0)
                return;
            if (_tpGroup is ICrossNodeByteGroup bytes)
                bytes.CrossNodeBroadcastBytes(buffer);
            else
                TpAllReduceBytes(buffer);
        }

        /// <summary>
        /// Collect every node's owned bytes (zeros elsewhere) on the driver, which ends with
        /// their union; a worker's buffer is left as it was, unless the group has no byte
        /// channel and the <see cref="TpAllReduceBytes"/> fallback gives every node the union.
        /// </summary>
        protected void TpGatherBytesToDriver(Span<byte> buffer)
        {
            if (TpNodeCount <= 1 || buffer.Length == 0)
                return;
            if (_tpGroup is ICrossNodeByteGroup bytes)
                bytes.CrossNodeGatherBytes(buffer);
            else
                TpAllReduceBytes(buffer);
        }
    }
}
