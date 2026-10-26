using System;
using System.Net;
using TensorSharp.Cuda;

namespace TensorSharp.Distributed
{
    /// <summary>
    /// Multi-node tensor-parallel group that combines local CUDA P2P
    /// communication (within a node) with TCP communication (across nodes).
    ///
    /// AllReduce is hierarchical:
    ///   1. Local P2P AllReduce across GPUs within this node
    ///   2. Copy the rank-0 result to a host buffer
    ///   3. TCP AllReduce across node representatives
    ///   4. Broadcast the reduced result back to all local GPUs
    ///
    /// This minimises network traffic: only one buffer per AllReduce
    /// crosses the network, regardless of how many local GPUs participate.
    /// </summary>
    public sealed class DistributedTensorParallelGroup : ITensorParallelGroup, INestedTensorParallelGroup, IDistributedExecutionGroup
    {
        /// <summary>The group that owns this node's GPUs.</summary>
        public ITensorParallelGroup LocalGroup => _localGroup;

        private readonly ITensorParallelGroup _localGroup;
        private readonly TcpCommunicator _tcp;
        private readonly int _nodeId;
        private readonly int _nodeCount;
        private bool _disposed;

        /// <param name="localDegree">Number of GPUs on this node.</param>
        /// <param name="nodeId">This node's ID (0..nodeCount-1).</param>
        /// <param name="peerEndpoints">
        /// TCP endpoints for every node in the group, indexed by node ID.
        /// </param>
        public DistributedTensorParallelGroup(int localDegree, int nodeId, IPEndPoint[] peerEndpoints)
            : this(new TensorParallelGroup(localDegree), nodeId, peerEndpoints)
        {
        }

        /// <summary>
        /// Wrap an already-constructed local group. Lets the caller choose which
        /// backend owns the on-node GPUs — <see cref="TensorParallelGroup"/> for
        /// direct CUDA, a GGML group for the ggml backends — while this class
        /// supplies the cross-node layer unchanged.
        /// </summary>
        /// <param name="localGroup">Group covering this node's GPUs.</param>
        /// <param name="nodeId">This node's ID (0..nodeCount-1).</param>
        /// <param name="peerEndpoints">
        /// TCP endpoints for every node in the group, indexed by node ID.
        /// </param>
        public DistributedTensorParallelGroup(ITensorParallelGroup localGroup, int nodeId, IPEndPoint[] peerEndpoints)
        {
            ArgumentNullException.ThrowIfNull(peerEndpoints);
            _nodeId = nodeId;
            _nodeCount = peerEndpoints.Length;

            if (localGroup == null)
                throw new ArgumentNullException(nameof(localGroup));
            if (_nodeCount < 2)
                throw new ArgumentException("Distributed TP requires at least 2 nodes.", nameof(peerEndpoints));

            _localGroup = localGroup;
            int localDegree = localGroup.Degree;
            try
            {
                _tcp = new TcpCommunicator(nodeId, peerEndpoints);
                _tcp.ValidateConfiguration($"tensor-parallel:{localDegree}");
            }
            catch { _tcp?.Dispose(); localGroup.Dispose(); throw; }

            Console.WriteLine($"Distributed tensor parallelism: node {nodeId}/{_nodeCount}, " +
                $"{localDegree} local GPU(s), {_nodeCount * localDegree} total across cluster.");
        }

        /// <summary>Number of local GPUs on this node.</summary>
        public int Degree => _localGroup.Degree;

        /// <summary>True when TP is active (always true for distributed groups).</summary>
        public bool IsActive => true;

        /// <summary>Total GPUs across all nodes.</summary>
        public int GlobalDegree => Degree * _nodeCount;

        /// <summary>First global rank on this node.</summary>
        public int GlobalRankOffset => _nodeId * Degree;

        /// <summary>Number of nodes in the cluster.</summary>
        public int NodeCount => _nodeCount;

        public IAllocator GetAllocator(int rank) => _localGroup.GetAllocator(rank);

        /// <summary>Delegate rank fan-out to the local group's dispatch policy.</summary>
        public void RunPerRank(Action<int> body) => _localGroup.RunPerRank(body);

        /// <summary>
        /// Hierarchical AllReduce: local P2P reduce → TCP reduce → local broadcast.
        /// <paramref name="tensors"/> has one tensor per local GPU.
        /// After this call every tensor on every node holds the global sum.
        /// </summary>
        public void AllReduce(Tensor[] tensors)
        {
            if (tensors == null || tensors.Length != Degree)
                throw new ArgumentException($"Expected {Degree} tensors, got {tensors?.Length ?? 0}.");

            // Phase 1: Local P2P AllReduce across GPUs within this node.
            _localGroup.AllReduce(tensors);

            // Phase 2: Copy rank-0 GPU data to host buffer.
            int elementCount = checked((int)tensors[0].ElementCount());
            var hostData = tensors[0].GetElementsAsFloat(elementCount);

            // Phase 3: Reduce the actual host copy in-place. GC pinning alone
            // does not register page-locked CUDA/NIC memory; additional pinned
            // arrays only introduced copies on this portable staging path.
            _tcp.AllReduce(hostData, elementCount);

            // Phase 4: Broadcast the reduced result from host to all local GPUs.
            for (int r = 0; r < Degree; r++)
            {
                tensors[r].SetElementsAsFloat(hostData);
                tensors[r].EnsureDeviceCurrent();
            }

            // For multi-GPU nodes, sync so every GPU sees the result before
            // the next layer reads it. Single-GPU nodes (Degree==1) already
            // synchronised inside EnsureDeviceCurrent; the extra Synchronize
            // was a redundant full-device barrier on every AllReduce.
            if (Degree > 1)
                _localGroup.Synchronize();
        }

        /// <summary>Cross-node half of <see cref="AllReduce(Tensor[])"/>: the
        /// caller has already reduced this node's ranks, so only the TCP
        /// exchange between nodes remains.</summary>
        public void CrossNodeAllReduce(float[] buffer, int count)
        {
            ArgumentNullException.ThrowIfNull(buffer);
            if ((uint)count > (uint)buffer.Length)
                throw new ArgumentOutOfRangeException(nameof(count));
            _tcp.AllReduce(buffer, count);
        }

        public void Synchronize()
        {
            _localGroup.Synchronize();
        }

        /// <summary>Block until every node has reached this point.</summary>
        public void Barrier()
        {
            _localGroup.Synchronize();
            _tcp.Barrier();
        }

        /// <summary>Driver (node 0) broadcasts a control op + payload to worker nodes.</summary>
        public void BroadcastControl(int op, int[] payload) => _tcp.BroadcastControl(op, payload);

        /// <summary>Worker node blocks for the next control message from the driver.</summary>
        public (int op, int[] payload) ReceiveControl() => _tcp.ReceiveControl();

        public void Abort(Exception failure) => _tcp.Abort(failure);

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;

            _tcp?.Dispose();
            _localGroup?.Dispose();
        }
    }
}
