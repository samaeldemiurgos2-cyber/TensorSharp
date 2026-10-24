// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using TensorSharp.Runtime.Speculative;

namespace TensorSharp.Models;

// The speculative TARGET half. MiniMax-M3.1 has no recurrent state: verification
// keeps the accepted K/V/index-key prefix and rewinds the slot head, and every verify
// row selects its own sparse-attention blocks at its own position, exactly as a plain
// decode of that token would. The generic n-gram drafter needs no extra weights.
//
// The DSpark drafter (--draft-model, a dflash GGUF reading the outputs of target layers
// 3/17/31/45/59) is the draft-head half, MiniMaxM3Model.Dspark.cs: the native executor
// loads it with the target and commits its context inside every trunk graph, so a
// verify also keeps the drafter in step and its rows need no hidden states from here.
// The generic managed DFlash host is refused: it assumes SwiGLU and full-width RoPE,
// where the M3.1 drafter uses SwiGLU-OAI, partial RoPE and a Markov head.
public partial class MiniMaxM3Model : ISpeculativeTarget
{
    // A layer split keeps one graph family per call shape; a future tensor-parallel
    // executor on ggml's meta backend holds views for one graph only, so alternating
    // verify and decode graphs would rebuild every step there.
    public bool SpeculationProfitable => UsesNativeExecutor && _nativeTpDegree == 0;
    public int SpecPrefillChunkSize => UsesNativeExecutor ? _nativeUbatch : 0;
    // One verify (the draft plus its anchor) fits one native ubatch.
    public int SpecMaxDraftTokens => UsesNativeExecutor ? Math.Max(0, _nativeUbatch - 1) : 0;
    public bool SpecVerifyPersistsAcceptedKv => true;
    public bool SpecPlainStepUsesForward => true;
    public bool SpecSupportsMultimodalPrefill => true;

    /// <summary>SpecForward, rewind and the slot head all act on the ACTIVE native slot - the one
    /// <see cref="BindSequenceCache"/> selected - so a request served from its own slot can speculate.</summary>
    public bool SpecTrunkFollowsBoundCache => UsesNativeExecutor;

    /// <summary>The drafter reads its features on the device, so the rows handed between the
    /// trunk and the drafter are a single unused float (DeepSeek V4's native DSpark does the same).</summary>
    public int SpecFeatureSize => HasDraftHead ? 1 : Config.HiddenSize;

    protected override string DFlashDrafterRefusal => MiniMaxM3Architecture.DraftModelRefusal;

    // The executor grows each slot's rows on demand inside its own forward.
    public void SpecEnsureCapacity(int requiredSeqLen) { }
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
        if (!UsesNativeExecutor)
            throw new InvalidOperationException("MiniMax-M3.1 runs only on its native executor, which this instance did not load.");
        if ((long)_cacheSeqLen + tokens.Length > _maxContextLength)
            throw new InvalidOperationException("MiniMax-M3.1 context length exceeded.");
        // The native executor chunks into ubatches itself and returns every requested
        // row; it exports no hidden states. n-gram drafting needs none, and the DSpark
        // drafter's context is committed by this very forward on the device.
        if (hAllOut != null && !HasDraftHead)
            throw new NotSupportedException("The native MiniMax-M3.1 executor does not export hidden states.");
        ForwardNative(tokens, allLogitsRows, logitsOut);
        if (hAllOut != null)
            Array.Clear(hAllOut);
    }
}
