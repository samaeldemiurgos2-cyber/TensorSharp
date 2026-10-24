// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// DSpark speculative decoding for MiniMax-M3.1: the draft-head half (the target half is
// MiniMaxM3Model.Speculative.cs). The drafter (MiniMax-M3.1-Flash-DSpark, GGUF arch dflash
// with a Markov head and a confidence head) reads the outputs of target layers 3/17/31/45/59
// (its dflash.target_layers 4/18/32/46/60 name the inputs of the next layer). The native
// executor loads it with the target, beside the output head it borrows, and commits the
// drafter's sliding-window context inside every trunk graph, so prefill and verification
// need nothing from the host and a draft is one native call per block
// (ggml_ops_minimax_m3_dspark.inc).
using System;
using System.IO;
using System.Linq;
using TensorSharp.GGML;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Speculative;

namespace TensorSharp.Models;

public partial class MiniMaxM3Model : ISpeculativeModel
{
    /// <summary>The validated DSpark drafter the executor loads with the target, or null.</summary>
    private string _dsparkPath;

    /// <summary>Rows one native draft proposes; 0 when no drafter is loaded.</summary>
    private int _dsparkBlock;

    /// <summary>True when the native executor loaded a DSpark drafter.</summary>
    public bool HasDraftHead => _dsparkBlock > 0 && UsesNativeExecutor;

    /// <summary>DSpark proposes a whole block per pass (<see cref="BlockDraftSpeculator"/>).</summary>
    public DraftHeadKind DraftHeadKind => HasDraftHead ? DraftHeadKind.Block : DraftHeadKind.None;

    public int DraftBlockSize => HasDraftHead ? _dsparkBlock : 0;

    /// <summary>Every trunk forward commits the drafter's context for the rows it processes, so
    /// prefill runs as one call and needs no per-row hidden states.</summary>
    public bool DraftSelfCatchUp => HasDraftHead;

    /// <summary>The drafter masks every context row by the position it holds: after an adopted
    /// prefix, a rewind or a slot copy it drafts from whatever part of its window the slot has
    /// written and never from a stale row.</summary>
    public bool DraftHeadResumesAfterGap => HasDraftHead;

    public int DraftBlock(int lastToken, float[] hPrev, int position, int[] draftOut, float[] confOut)
    {
        if (!HasDraftHead)
            throw new InvalidOperationException("No DSpark drafter is loaded for this MiniMax-M3.1 model.");
        lock (_nativeSync)
        {
            // The executor drafts at the active slot's head; a caller at any other position
            // would get drafts for the wrong context.
            int head = GgmlMiniMaxM3Native.NPast(_native);
            if (position != head)
                throw new InvalidOperationException(
                    $"MiniMax-M3.1 DSpark draft requested at position {position}; the sequence head is at {head}.");
            return GgmlMiniMaxM3DsparkNative.Draft(_native, lastToken, draftOut, confOut);
        }
    }

    /// <summary>Nothing to replay: the trunk graph that verified or decoded these tokens already
    /// wrote their drafter context.</summary>
    public void DraftCatchUp(int[] tokens, float[] hRows, int startPos) { }

    /// <summary>The drafter statistics of the native executor (see <see cref="GgmlMiniMaxM3DsparkNative.Stats"/>).</summary>
    internal long[] DsparkStatsForTests()
    {
        lock (_nativeSync) return _native == IntPtr.Zero ? null : GgmlMiniMaxM3DsparkNative.Stats(_native);
    }

    /// <summary>Validate a configured drafter against this target before the executor maps anything.</summary>
    private void PrepareDspark(string draftModelPath)
    {
        if (string.IsNullOrWhiteSpace(draftModelPath))
            return;
        string path = draftModelPath.Trim();
        MiniMaxM3Dspark.ValidateAgainstTarget(path, Config.NumLayers, Config.HiddenSize, Config.VocabSize);
        _dsparkPath = path;
    }

    /// <summary>After the executor loaded: the drafter it reports must be the one requested.</summary>
    private void InitDsparkHead()
    {
        if (_dsparkPath == null)
            return;
        _dsparkBlock = GgmlMiniMaxM3DsparkNative.BlockSize(_native);
        if (_dsparkBlock <= 0)
            throw new ModelLoadRefusedException(
                $"[minimax-m3] the native executor did not load the DSpark drafter {Path.GetFileName(_dsparkPath)}.");
        long[] stats = GgmlMiniMaxM3DsparkNative.Stats(_native);
        Console.WriteLine($"MiniMax-M3.1 DSpark drafter: {Path.GetFileName(_dsparkPath)}, block {_dsparkBlock}" +
            (stats != null ? $", {stats[GgmlMiniMaxM3DsparkNative.StatLayers]} layers over " +
                $"{stats[GgmlMiniMaxM3DsparkNative.StatTaps]} tapped target layers on GPU " +
                $"{stats[GgmlMiniMaxM3DsparkNative.StatDevice]}, {stats[GgmlMiniMaxM3DsparkNative.StatRingRows]}-row " +
                "context ring per sequence" : "") + ".");
    }
}

/// <summary>
/// What a <c>--draft-model</c> must be for MiniMax-M3.1: a DSpark drafter (GGUF arch
/// <c>dflash</c> with a Markov head) trained for this target. The checks that need only the
/// file run before the target is mapped (<see cref="ValidateFile"/>); the ones against the
/// target's geometry run once its metadata is parsed (<see cref="ValidateAgainstTarget"/>).
/// The native loader checks every tensor's shape again and refuses the load on a mismatch.
/// </summary>
internal static class MiniMaxM3Dspark
{
    private const string Arch = "dflash";

    private static NotSupportedException Refuse(string path, string why)
        => new($"--draft-model {Path.GetFileName(path)}: {why}");

    /// <summary>The drafter-only checks: the file is a DSpark drafter in the one layout the
    /// executor runs (every row predicts, swigluoai, a sliding window on every layer).</summary>
    internal static void ValidateFile(string path)
    {
        if (!File.Exists(path))
            throw Refuse(path, $"the file does not exist ({path}).");
        GgufFile gguf;
        try
        {
            gguf = GgufFile.OpenWithoutSiblingShards(path);
        }
        catch (Exception ex) when (ex is IOException or InvalidDataException or UnauthorizedAccessException)
        {
            throw Refuse(path, $"cannot be read as a GGUF file ({ex.Message}).");
        }
        using (gguf)
        {
            string arch = gguf.GetString("general.architecture");
            if (!string.Equals(arch, Arch, StringComparison.Ordinal))
                throw Refuse(path, $"general.architecture is '{arch ?? "missing"}'; MiniMax-M3.1 drafts with a DSpark " +
                    "drafter (arch dflash with a Markov head), such as MiniMax-M3.1-Flash-DSpark.");
            if (!gguf.Tensors.ContainsKey("markov_w1.weight") || !gguf.Tensors.ContainsKey("markov_w2.weight"))
                throw Refuse(path, "has no Markov head (markov_w1/markov_w2): a plain DFlash drafter, which " +
                    "MiniMax-M3.1 does not run.");
            if (!gguf.GetBool($"{Arch}.sample_from_anchor", true))
                throw Refuse(path, "uses the bonus-anchor block layout (dflash.sample_from_anchor = false); " +
                    "MiniMax-M3.1's DSpark predicts from every block row and that is the only layout implemented.");
            string act = gguf.GetString($"{Arch}.hidden_activation");
            if (act != null && !string.Equals(act, "swigluoai", StringComparison.Ordinal))
                throw Refuse(path, $"hidden_activation is '{act}'; the executor runs swigluoai drafters only.");
            if (gguf.GetUint32($"{Arch}.attention.sliding_window", 0) == 0)
                throw Refuse(path, "has no dflash.attention.sliding_window; the executor keeps a sliding-window " +
                    "draft context per sequence.");
            if (gguf.GetBoolArray($"{Arch}.attention.sliding_window_pattern") is { } swa && swa.Any(s => !s))
                throw Refuse(path, "has a draft layer that attends the whole sequence; only sliding-window drafters " +
                    "are supported.");
            if ((gguf.GetInt32Array($"{Arch}.target_layers") ?? Array.Empty<int>()).Length == 0)
                throw Refuse(path, "has no dflash.target_layers.");
        }
    }

    /// <summary>
    /// The checks against the loaded target: hidden size, the tapped layers (the input of layer
    /// k, k = 1..<paramref name="numLayers"/>, where k = numLayers is the final pre-norm stream),
    /// the Markov head's vocabulary and the mask token.
    /// </summary>
    internal static void ValidateAgainstTarget(string path, int numLayers, int hidden, int vocab)
    {
        ValidateFile(path);
        using GgufFile gguf = GgufFile.OpenWithoutSiblingShards(path);
        uint embd = gguf.GetUint32($"{Arch}.embedding_length", 0);
        if (embd != (uint)hidden)
            throw Refuse(path, $"embedding_length {embd} does not match the target's hidden size {hidden}: it was " +
                "trained for another model.");
        int[] taps = gguf.GetInt32Array($"{Arch}.target_layers");
        string list = "[" + string.Join(",", taps) + "]";
        foreach (int k in taps)
        {
            if (k < 1 || k > numLayers)
                throw Refuse(path, $"dflash.target_layers {list} names layer input {k}; this {numLayers}-layer target " +
                    $"exports the outputs of layers 0..{numLayers - 1} (inputs 1..{numLayers}): it was trained for " +
                    "another model.");
        }
        if (taps.Distinct().Count() != taps.Length)
            throw Refuse(path, $"dflash.target_layers {list} repeats a layer.");
        if (gguf.Tensors.TryGetValue("markov_w1.weight", out var w1) && (w1.Shape.Length < 2 || w1.Shape[1] != (ulong)vocab))
            throw Refuse(path, $"its Markov head covers {(w1.Shape.Length < 2 ? 0 : w1.Shape[1])} tokens; the target's " +
                $"vocabulary has {vocab}: it was trained for another model.");
        uint mask = gguf.GetUint32("tokenizer.ggml.mask_token_id", uint.MaxValue);
        if (mask >= (uint)vocab)
            throw Refuse(path, mask == uint.MaxValue
                ? "has no tokenizer.ggml.mask_token_id (the token filling the draft block)."
                : $"mask token {mask} is outside the target's {vocab}-token vocabulary.");
    }
}
