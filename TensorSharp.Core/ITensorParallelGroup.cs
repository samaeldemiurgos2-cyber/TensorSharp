using System;

namespace TensorSharp
{
    /// <summary>
    /// Abstraction over a tensor-parallel group that may span local GPUs
    /// (single process) or multiple nodes connected by a network.
    /// Model code uses this interface to place per-rank tensors and
    /// invoke collective operations without knowing the physical topology.
    /// </summary>
    public interface ITensorParallelGroup : IDisposable
    {
        /// <summary>Number of GPUs in this parallel group (local to this process).</summary>
        int Degree { get; }

        /// <summary>True when TP is active (degree > 1).</summary>
        bool IsActive { get; }

        /// <summary>
        /// Total number of GPUs across all nodes. Equals <see cref="Degree"/>
        /// for single-node groups; larger for multi-node distributed groups.
        /// Weight sharding uses this value to compute shard sizes.
        /// </summary>
        int GlobalDegree { get; }

        /// <summary>
        /// The global rank offset for this node. Local rank <c>r</c> maps to
        /// global rank <c>GlobalRankOffset + r</c>. Zero for single-node groups.
        /// </summary>
        int GlobalRankOffset { get; }

        /// <summary>Number of nodes in the distributed group (1 for local-only).</summary>
        int NodeCount { get; }

        /// <summary>
        /// Allocator that places tensors on <paramref name="rank"/>'s device.
        /// Typed as <see cref="IAllocator"/> rather than <see cref="CudaAllocator"/>
        /// so a group can be backed by any multi-device backend — the GGML
        /// backend hands back a per-rank <c>GgmlAllocator</c>. Call sites that
        /// need CUDA-specific behaviour pattern-match on the concrete type.
        /// </summary>
        IAllocator GetAllocator(int rank);

        /// <summary>
        /// Run <paramref name="body"/> once per rank. The default is a plain
        /// sequential loop, which is right for backends whose per-rank work is
        /// already asynchronous (CUDA streams). Backends that submit
        /// synchronously override this to dispatch the ranks concurrently,
        /// otherwise tensor parallelism would serialize the GPUs and be slower
        /// than a single device.
        /// </summary>
        void RunPerRank(Action<int> body)
        {
            for (int r = 0; r < Degree; r++)
                body(r);
        }

        /// <summary>
        /// In-place AllReduce (element-wise sum) across all GPUs.
        /// tensors[i] must be a F32 tensor on GPU i, all the same shape.
        /// After this call every tensor holds the global sum.
        /// For multi-node groups this performs a hierarchical reduce:
        /// local P2P first, then network exchange, then local broadcast.
        /// </summary>
        void AllReduce(Tensor[] tensors);

        /// <summary>Synchronize all GPU streams.</summary>
        void Synchronize();

        /// <summary>
        /// Block until every node in the group has reached this point.
        /// Single-node groups return immediately.
        /// </summary>
        void Barrier();

        /// <summary>
        /// Driver→worker control channel for multi-node lockstep execution.
        /// The driver node (global rank offset 0) calls this to broadcast an
        /// op code plus an int payload (e.g. the tokens for a forward pass) to
        /// every worker node before running the op locally. Single-node groups
        /// have no workers and do not implement this.
        /// </summary>
        void BroadcastControl(int op, int[] payload);

        /// <summary>
        /// Worker-side counterpart to <see cref="BroadcastControl"/>: block until
        /// the driver broadcasts the next control message, and return it.
        /// </summary>
        (int op, int[] payload) ReceiveControl();
    }

    /// <summary>
    /// A tensor-parallel group layered on top of another group — the multi-node
    /// group wrapping the on-node one, for instance. Lets callers reach the group
    /// that actually owns this node's devices without taking a dependency on the
    /// wrapper's assembly.
    /// </summary>
    public interface INestedTensorParallelGroup
    {
        ITensorParallelGroup LocalGroup { get; }

        /// <summary>
        /// Sum <paramref name="buffer"/>[0..count) element-wise across every NODE
        /// (this node's local ranks are assumed already reduced) and leave the
        /// result in place on all of them. Lets a fused whole-model graph do its
        /// own cross-node reduction at each segment boundary rather than
        /// dropping to a per-op forward.
        /// </summary>
        void CrossNodeAllReduce(float[] buffer, int count);
    }

    /// <summary>
    /// Raw host-byte movement between the nodes of a multi-node group, for state that
    /// is copied rather than summed (KV snapshots): no device staging, no float encoding.
    /// Every node must call each operation at the same point, like any collective.
    /// </summary>
    public interface ICrossNodeByteGroup
    {
        /// <summary>Node 0's <paramref name="buffer"/> replaces every other node's.</summary>
        void CrossNodeBroadcastBytes(Span<byte> buffer);

        /// <summary>Each node passes the bytes it owns and zeros elsewhere; node 0 ends
        /// with their union (bitwise OR). Other nodes' buffers are left unchanged.</summary>
        void CrossNodeGatherBytes(Span<byte> buffer);
    }
}
