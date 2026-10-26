using System;

namespace TensorSharp;

/// <summary>Deterministic contiguous whole-layer ownership, independent of model family.</summary>
public sealed class LayerSplitPlan
{
    public int LayerCount { get; }
    public int LocalDegree { get; }
    public int NodeId { get; }
    public int NodeCount { get; }
    public int GlobalDegree => checked(LocalDegree * NodeCount);
    public int FirstLayer { get; }
    public int EndLayer { get; }
    private readonly int[] _localBoundaries;
    public bool IsFirst => NodeId == 0;
    public bool IsLast => NodeId == NodeCount - 1;

    public LayerSplitPlan(int layerCount, int localDegree, int nodeId = 0, int nodeCount = 1)
    {
        if (layerCount < 1) throw new ArgumentOutOfRangeException(nameof(layerCount));
        if (localDegree < 1) throw new ArgumentOutOfRangeException(nameof(localDegree));
        if (nodeCount < 1) throw new ArgumentOutOfRangeException(nameof(nodeCount));
        if ((uint)nodeId >= (uint)nodeCount) throw new ArgumentOutOfRangeException(nameof(nodeId));
        if (layerCount < checked(localDegree * nodeCount))
            throw new NotSupportedException($"Cannot distribute {layerCount} layers over {localDegree * nodeCount} GPUs: each GPU must own at least one layer.");
        LayerCount = layerCount;
        LocalDegree = localDegree;
        NodeId = nodeId;
        NodeCount = nodeCount;
        FirstLayer = (int)((long)LayerCount * NodeId / NodeCount);
        EndLayer = (int)((long)LayerCount * (NodeId + 1) / NodeCount);
        _localBoundaries = new int[localDegree + 1];
        for (int rank = 0; rank <= localDegree; rank++)
            _localBoundaries[rank] = (int)((long)layerCount * (nodeId * localDegree + rank) / GlobalDegree);
    }

    private LayerSplitPlan(int layerCount, int localDegree, int nodeId, int nodeCount,
        int[] nodeBoundaries, int[] localBoundaries) : this(layerCount, localDegree, nodeId, nodeCount)
    {
        FirstLayer = nodeBoundaries[nodeId];
        EndLayer = nodeBoundaries[nodeId + 1];
        _localBoundaries = localBoundaries;
    }

    /// <summary>Minimize the maximum estimated resident bytes per node, then per
    /// local device, including the embedding and head at their actual endpoints.</summary>
    public static LayerSplitPlan CreateWeighted(long[] layerBytes, int localDegree,
        int nodeId = 0, int nodeCount = 1, long embeddingBytes = 0, long headBytes = 0)
    {
        ArgumentNullException.ThrowIfNull(layerBytes);
        // Validate topology before indexing or allocating the dynamic program.
        _ = new LayerSplitPlan(layerBytes.Length, localDegree, nodeId, nodeCount);
        int[] nodes = Partition(layerBytes, nodeCount, localDegree, embeddingBytes, headBytes);
        int first = nodes[nodeId], end = nodes[nodeId + 1];
        long[] owned = layerBytes.AsSpan(first, end - first).ToArray();
        int[] devices = Partition(owned, localDegree, 1,
            nodeId == 0 ? embeddingBytes : 0, nodeId == nodeCount - 1 ? headBytes : 0);
        for (int rank = 0; rank < devices.Length; rank++) devices[rank] += first;
        return new LayerSplitPlan(layerBytes.Length, localDegree, nodeId, nodeCount, nodes, devices);
    }

    /// <summary>Exact minimax partition of an ordered layer sequence. Equal-cost
    /// solutions choose the smallest preceding boundary, deterministically.</summary>
    public static int[] Partition(long[] layerBytes, int partitions, int minimumLayers,
        long firstBytes = 0, long lastBytes = 0)
    {
        ArgumentNullException.ThrowIfNull(layerBytes);
        if (partitions < 1 || minimumLayers < 1 ||
            (long)partitions * minimumLayers > layerBytes.Length)
            throw new ArgumentException("Every partition must own its minimum number of layers.");
        if (firstBytes < 0 || lastBytes < 0) throw new ArgumentOutOfRangeException(nameof(firstBytes));
        int count = layerBytes.Length;
        long[] prefix = new long[count + 1];
        for (int layer = 0; layer < count; layer++)
        {
            if (layerBytes[layer] < 0) throw new ArgumentOutOfRangeException(nameof(layerBytes));
            prefix[layer + 1] = checked(prefix[layer] + layerBytes[layer]);
        }
        _ = checked(prefix[count] + firstBytes + lastBytes);
        var best = new long[partitions + 1, count + 1];
        var cuts = new int[partitions + 1, count + 1];
        var reachable = new bool[partitions + 1, count + 1];
        reachable[0, 0] = true;
        for (int stage = 1; stage <= partitions; stage++)
        for (int end = stage * minimumLayers; end <= count - (partitions - stage) * minimumLayers; end++)
        for (int begin = (stage - 1) * minimumLayers; begin <= end - minimumLayers; begin++)
        {
            if (!reachable[stage - 1, begin]) continue;
            long bytes = prefix[end] - prefix[begin] + (stage == 1 ? firstBytes : 0) +
                (stage == partitions ? lastBytes : 0);
            long maximum = Math.Max(best[stage - 1, begin], bytes);
            if (!reachable[stage, end] || maximum < best[stage, end])
            {
                reachable[stage, end] = true;
                best[stage, end] = maximum;
                cuts[stage, end] = begin;
            }
        }
        int[] boundaries = new int[partitions + 1];
        boundaries[partitions] = count;
        for (int stage = partitions; stage > 0; stage--)
            boundaries[stage - 1] = cuts[stage, boundaries[stage]];
        return boundaries;
    }

    public bool Owns(int layer) => layer >= FirstLayer && layer < EndLayer;
    public int LocalRank(int layer)
    {
        if (!Owns(layer)) throw new ArgumentOutOfRangeException(nameof(layer), "Layer belongs to a different node.");
        int rank = 0;
        while (layer >= _localBoundaries[rank + 1]) rank++;
        return rank;
    }

    public static bool TryGetLayer(string weightName, out int layer)
    {
        layer = -1;
        if (weightName == null || !weightName.StartsWith("blk.", StringComparison.Ordinal)) return false;
        int separator = weightName.IndexOf('.', 4);
        return separator > 4 && int.TryParse(weightName.AsSpan(4, separator - 4), out layer) && layer >= 0;
    }
}
