// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// MiniMax-M3.1's multimodal contract. The managed MiniMaxM3VisionEncoder (mmproj
// clip.projector_type = minimax_m3) turns each image into [N, hidden] rows; the
// injector expands every rendered ]<]image[>[ into ]<]start of image[>[ + N x
// ]<]image[>[ + ]<]end of image[>[ and hands the span here. The text tower uses
// plain 1-D positions, so the rows replace the placeholder embeddings at ordinary
// sequential cache positions: they are queued at their ABSOLUTE positions and the
// native executor takes them from the forward call that covers them, which lets a
// large image span survive prefill chunking.
using System;
using System.Collections.Generic;
using System.IO;
using TensorSharp.Models.Architecture;
using TensorSharp.Runtime;

namespace TensorSharp.Models;

public partial class MiniMaxM3Model : IVisionCapableModel, IMultimodalPromptExpander
{
    public MiniMaxM3VisionEncoder VisionEncoder { get; private set; }

    bool IVisionCapableModel.IsVisionEncoderLoaded => VisionEncoder != null;

    /// <summary>
    /// Load the MiniMax-M3.1 tower. The projector type and output width are checked from the
    /// header before the ~860M-parameter tower is dequantized, so a projector for another
    /// model is refused without allocating it.
    /// </summary>
    public void LoadVisionEncoder(string mmProjPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(mmProjPath);
        using (var gguf = new GgufFile(mmProjPath))
            ValidateVisionProjector(gguf, Config.HiddenSize, mmProjPath);
        var encoder = new MiniMaxM3VisionEncoder(mmProjPath, _allocator);
        encoder.SetHostModel(this);
        VisionEncoder?.Dispose();
        VisionEncoder = encoder;
    }

    internal static void ValidateVisionProjector(GgufFile gguf, int hiddenSize, string mmProjPath)
    {
        string projector = gguf.GetString("clip.vision.projector_type") ?? gguf.GetString("clip.projector_type");
        if (projector != MiniMaxM3VisionEncoder.ProjectorType)
            throw new ArgumentException(
                $"'{Path.GetFileName(mmProjPath)}' is a '{projector ?? "(none)"}' projector; MiniMax-M3.1 needs its " +
                $"clip.projector_type = {MiniMaxM3VisionEncoder.ProjectorType} mmproj.");
        uint width = gguf.GetUint32("clip.vision.projection_dim", 0);
        if (width != hiddenSize)
            throw new ArgumentException(
                $"'{Path.GetFileName(mmProjPath)}' projects to {width} dims; this MiniMax-M3.1 model's hidden size is {hiddenSize}.");
    }

    /// <summary>
    /// Queue one span of image rows (<c>[tokens, hidden]</c>) that starts
    /// <paramref name="insertPosition"/> tokens into the batch about to be forwarded on the
    /// active sequence. Takes ownership of <paramref name="embeddings"/>.
    /// </summary>
    public void SetVisionEmbeddings(Tensor embeddings, int insertPosition)
    {
        ArgumentNullException.ThrowIfNull(embeddings);
        try
        {
            // The injector clamps prefix reuse to the start of an image span, so a span
            // never begins inside the cached prefix; a negative offset is a caller bug.
            if (insertPosition < 0)
                throw new ArgumentOutOfRangeException(nameof(insertPosition));
            int hidden = Config.HiddenSize;
            if (embeddings.DimensionCount != 2 || embeddings.Sizes[0] < 1 || embeddings.Sizes[1] != hidden)
                throw new ArgumentException($"Image rows must have shape [tokens, {hidden}].", nameof(embeddings));

            int rows = checked((int)embeddings.Sizes[0]);
            int count = checked(rows * hidden);
            float[] data;
            if (embeddings.IsContiguous())
                data = embeddings.GetElementsAsFloat(count);
            else
            {
                using var contiguous = Ops.NewContiguous(embeddings);
                data = contiguous.GetElementsAsFloat(count);
            }

            // The offset is relative to the sequence head the next forward starts from.
            lock (_nativeSync)
                QueueNativeVisionRows(data, rows, checked(_cacheSeqLen + insertPosition));
        }
        finally
        {
            embeddings.Dispose();
        }
    }

    List<int> IMultimodalPromptExpander.ExpandMultimodalPrompt(
        ModelMultimodalInjector injector, List<ChatMessage> history, List<int> inputTokens)
        => injector.ProcessMiniMaxM3History(VisionEncoder, history, inputTokens);

    /// <summary>Release the tower (called from <see cref="Dispose"/>).</summary>
    private void DisposeVisionEncoder()
    {
        VisionEncoder?.Dispose();
        VisionEncoder = null;
    }
}
