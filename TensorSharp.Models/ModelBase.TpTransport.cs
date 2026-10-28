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
    }
}
