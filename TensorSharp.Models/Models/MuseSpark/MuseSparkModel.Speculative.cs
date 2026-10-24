// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using TensorSharp.Runtime.Speculative;

namespace TensorSharp.Models;

// Muse Spark has no recurrent state. Verification keeps the accepted KV prefix
// and rewinds the position; rejected rows are causally masked and overwritten.
// No compatible learned assistant is assumed: the generic n-gram drafter can
// use this target contract without loading any extra weights.
public partial class MuseSparkModel : ISpeculativeTarget
{
    // Under native --tp the executor keeps one graph (ggml's meta backend holds
    // per-GPU views only for the graph it ran last), so alternating draft-verify
    // and decode graphs would rebuild every step.
    public bool SpeculationProfitable => IsGgmlBackend && _nativeTpDegree == 0;
    public int SpecPrefillChunkSize => _prefillChunk;
    public int SpecMaxDraftTokens => Math.Max(0, _prefillChunk - 1);
    public bool SpecVerifyPersistsAcceptedKv => true;
    public bool SpecPlainStepUsesForward => true;
    public bool SpecSupportsMultimodalPrefill => true;

    public void SpecEnsureCapacity(int requiredSeqLen)
    {
        if (!UsesNativeExecutor) EnsureCacheCapacity(requiredSeqLen);
    }
    public void SpecSnapshotRecurrentState() { }
    public void SpecRestoreRecurrentState() { }
    public void SpecRewindCache(int length) => TruncateKVCacheCore(length);

    public void SpecForward(int[] tokens, float[] hAllOut, float[] logitsOut, bool allLogitsRows)
    {
        if (tokens == null || tokens.Length == 0)
            throw new ArgumentException("Verification requires at least one token.", nameof(tokens));
        ArgumentNullException.ThrowIfNull(logitsOut);
        int logitCount = checked((allLogitsRows ? tokens.Length : 1) * Config.VocabSize);
        if (logitsOut.Length < logitCount)
            throw new ArgumentException("The verification logits buffer is too short.", nameof(logitsOut));
        if (hAllOut != null && hAllOut.Length < checked(tokens.Length * Config.HiddenSize))
            throw new ArgumentException("The verification hidden-state buffer is too short.", nameof(hAllOut));
        if ((long)_cacheSeqLen + tokens.Length > _maxContextLength)
            throw new InvalidOperationException("Muse Spark context length exceeded.");
        if (UsesNativeExecutor)
        {
            // The native executor chunks into ubatches itself and returns every
            // requested row; it has no hidden-state output (n-gram drafting needs none).
            if (hAllOut != null)
                throw new NotSupportedException("The native Muse Spark executor does not export hidden states.");
            ForwardNative(tokens, allLogitsRows, logitsOut);
            return;
        }

        // Chunking limits prefill scratch while retaining the full row contract.
        // Caller owns GpuComputeLock for the entire verify/accept/rewind operation.
        for (int offset = 0; offset < tokens.Length; offset += _prefillChunk)
        {
            int rows = Math.Min(_prefillChunk, tokens.Length - offset);
            int[] part = offset == 0 && rows == tokens.Length ? tokens : tokens.AsSpan(offset, rows).ToArray();
            float[] features = hAllOut == null ? null : new float[checked(rows * Config.HiddenSize)];
            bool needLogits = allLogitsRows || offset + rows == tokens.Length;
            float[] result = ForwardChunk(part, needLogits || features != null, allLogitsRows, features);
            if (needLogits) result.CopyTo(logitsOut, allLogitsRows ? checked(offset * Config.VocabSize) : 0);
            if (features != null) features.CopyTo(hAllOut, checked(offset * Config.HiddenSize));
        }
    }
}
