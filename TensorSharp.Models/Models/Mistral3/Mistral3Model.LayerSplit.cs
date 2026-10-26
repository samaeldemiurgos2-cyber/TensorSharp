using System;
using TensorSharp.Runtime.Scheduling;

namespace TensorSharp.Models;

public partial class Mistral3Model
{
    // Distributed requests use the existing serialized driver/worker protocol.
    // The paged batch protocol does not yet transmit per-sequence metadata.
    public bool BatchedForwardAvailable => LayerPlacement == null;

    protected override bool ShouldLoadWeight(GgufTensorInfo info)
    {
        if (LayerPlacement == null) return true;
        if (LayerSplitPlan.TryGetLayer(info.Name, out int layer)) return LayerPlacement.Owns(layer);
        if (info.Name == "token_embd.weight")
            return LayerPlacement.IsFirst || (LayerPlacement.IsLast && !_gguf.Tensors.ContainsKey("output.weight"));
        if (info.Name.StartsWith("output", StringComparison.Ordinal)) return LayerPlacement.IsLast;
        return true;
    }

    protected override int PreloadRankForWeight(string name)
    {
        if (LayerPlacement == null) return 0;
        if (LayerSplitPlan.TryGetLayer(name, out int layer)) return LayerPlacement.LocalRank(layer);
        if (name.StartsWith("output", StringComparison.Ordinal) ||
            (name == "token_embd.weight" && !LayerPlacement.IsFirst)) return LayerSplitDegree - 1;
        return 0;
    }

    protected override IAllocator AllocatorForWeight(string name)
    {
        if (LayerPlacement == null) return base.AllocatorForWeight(name);
        if (LayerSplitPlan.TryGetLayer(name, out int layer)) return LayerAllocator(layer);
        return LayerAllocator(name.StartsWith("output", StringComparison.Ordinal) ||
            (name == "token_embd.weight" && !LayerPlacement.IsFirst)
            ? LayerPlacement.EndLayer - 1 : LayerPlacement.FirstLayer);
    }

    protected override bool ShouldRetainCudaHostQuantWeight(string name)
        => (LayerPlacement != null && !LayerSplitPlan.TryGetLayer(name, out _)) || base.ShouldRetainCudaHostQuantWeight(name);

    private float[] ForwardLayerSplit(int[] tokens)
    {
        ArgumentNullException.ThrowIfNull(tokens);
        int seqLen = tokens.Length;
        int startPos = _cacheSeqLen;
        EnsureCacheCapacity(checked(startPos + seqLen));
        _forwardSw.Start();
        try
        {
            _logitsBuffer = ForwardLayerPipeline(tokens,
                embed: () =>
                {
                    Tensor hidden = Embedding(tokens);
                    foreach (var (embeddings, position) in _pendingVisionEmbeddingsList)
                    {
                        InjectVisionEmbeddings(hidden, embeddings, position, startPos);
                        embeddings.Dispose();
                    }
                    _pendingVisionEmbeddingsList.Clear();
                    return hidden;
                },
                runLayer: (hidden, layer) => TransformerBlock(hidden, layer, seqLen, startPos),
                project: hidden =>
                {
                    using Tensor normed = RMSNormOp(hidden, "output_norm.weight");
                    using Tensor lastRow = normed.Narrow(0, seqLen - 1, 1);
                    using Tensor lastHidden = Ops.NewContiguous(lastRow);
                    using Tensor logits = LinearForward(lastHidden, "output.weight") ??
                        LinearForward(lastHidden, "token_embd.weight");
                    return TensorToFloatArray(logits);
                });
            _cacheSeqLen += seqLen;
            _forwardCount++;
            return _logitsBuffer;
        }
        finally { _forwardSw.Stop(); }
    }
}
