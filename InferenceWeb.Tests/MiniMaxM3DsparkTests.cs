// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// The managed half of MiniMax-M3.1's DSpark drafter: what a --draft-model must be
// (a dflash GGUF with a Markov head, every row a prediction, swigluoai, a sliding
// window on every layer, trained for this target's layers, hidden size and
// vocabulary), checked before the target is mapped, and the draft-head contract an
// instance without the native executor reports. The drafting itself is covered by
// the native fixture test (TensorSharp.GGML.Native/tests/minimax_m3_dspark_test.cpp).
using System.IO;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Models.Architecture;
using TensorSharp.Runtime.Speculative;
using static InferenceWeb.Tests.SyntheticGguf;
using GgufTensor = InferenceWeb.Tests.SyntheticGguf.Tensor;

namespace InferenceWeb.Tests;

[Collection(MiniMaxM3EnvironmentCollection.Name)]
public sealed class MiniMaxM3DsparkTests : IDisposable
{
    private const int Hidden = MiniMaxM3SyntheticModelBuilder.Hidden, Vocab = MiniMaxM3SyntheticModelBuilder.Vocab;
    private const int Layers = 4, Rank = 8, DraftFfn = 128, Heads = 4, KvHeads = 2, HeadDim = 32;

    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ts-minimax-m3-dspark-" + Guid.NewGuid().ToString("N"));
    private readonly KvCacheDtype _restoreDtype = KvCacheDtypeConfig.Current;
    private readonly bool _restoreExplicit = KvCacheDtypeConfig.IsExplicitlySet;
    private readonly string _restoreMsa = Environment.GetEnvironmentVariable(MiniMaxM3Architecture.MsaModeVariable);

    public MiniMaxM3DsparkTests()
    {
        Directory.CreateDirectory(_dir);
        Environment.SetEnvironmentVariable(MiniMaxM3Architecture.MsaModeVariable, null);
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F16, false);
    }

    public void Dispose()
    {
        Environment.SetEnvironmentVariable(MiniMaxM3Architecture.MsaModeVariable, _restoreMsa);
        KvCacheDtypeConfig.RestoreForTests(_restoreDtype, _restoreExplicit);
        Directory.Delete(_dir, true);
    }

    /// <summary>A small DSpark drafter for the synthetic target (4 layers, hidden 64,
    /// vocabulary 256), taps [2, 4]; <paramref name="editKv"/> and
    /// <paramref name="editTensors"/> break it one way at a time.</summary>
    private string WriteDrafter(Action<List<Kv>> editKv = null, Action<List<GgufTensor>> editTensors = null,
        int[] taps = null, int hidden = Hidden, int vocab = Vocab)
    {
        taps ??= new[] { 2, 4 };
        var kv = new List<Kv>
        {
            new Str { Key = "general.architecture", V = "dflash" },
            new U32 { Key = "dflash.block_count", V = 1 },
            new U32 { Key = "dflash.embedding_length", V = (uint)hidden },
            new U32 { Key = "dflash.feed_forward_length", V = DraftFfn },
            new U32 { Key = "dflash.attention.head_count", V = Heads },
            new U32 { Key = "dflash.attention.head_count_kv", V = KvHeads },
            new U32 { Key = "dflash.attention.key_length", V = HeadDim },
            new U32 { Key = "dflash.attention.value_length", V = HeadDim },
            new U32 { Key = "dflash.rope.dimension_count", V = HeadDim / 2 },
            new F32 { Key = "dflash.rope.freq_base", V = 1e7f },
            new F32 { Key = "dflash.attention.layer_norm_rms_epsilon", V = 1e-6f },
            new U32 { Key = "dflash.block_size", V = 7 },
            new I32Arr { Key = "dflash.target_layers", V = taps },
            new U32 { Key = "dflash.attention.sliding_window", V = 4096 },
            new Bool { Key = "dflash.sample_from_anchor", V = true },
            new Bool { Key = "dflash.has_confidence_head", V = true },
            new Str { Key = "dflash.hidden_activation", V = "swigluoai" },
            new U32 { Key = "tokenizer.ggml.mask_token_id", V = (uint)(vocab - 6) },
        };
        editKv?.Invoke(kv);
        var t = new List<GgufTensor>
        {
            Gen("fc.weight", 0.05f, taps.Length * hidden, hidden),
            GenAround("enc.output_norm.weight", 1f, 0.01f, hidden),
            GenAround("output_norm.weight", 1f, 0.01f, hidden),
            Gen("markov_w1.weight", 0.05f, Rank, vocab),
            Gen("markov_w2.weight", 0.05f, Rank, vocab),
            Gen("conf_proj.weight", 0.05f, hidden + Rank, 1),
            Gen("conf_proj.bias", 0.05f, 1),
            GenAround("blk.0.attn_norm.weight", 1f, 0.01f, hidden),
            Gen("blk.0.attn_q.weight", 0.05f, hidden, Heads * HeadDim),
            Gen("blk.0.attn_k.weight", 0.05f, hidden, KvHeads * HeadDim),
            Gen("blk.0.attn_v.weight", 0.05f, hidden, KvHeads * HeadDim),
            Gen("blk.0.attn_output.weight", 0.05f, Heads * HeadDim, hidden),
            GenAround("blk.0.attn_q_norm.weight", 1f, 0.01f, HeadDim),
            GenAround("blk.0.attn_k_norm.weight", 1f, 0.01f, HeadDim),
            GenAround("blk.0.ffn_norm.weight", 1f, 0.01f, hidden),
            Gen("blk.0.ffn_gate.weight", 0.05f, hidden, DraftFfn),
            Gen("blk.0.ffn_up.weight", 0.05f, hidden, DraftFfn),
            Gen("blk.0.ffn_down.weight", 0.05f, DraftFfn, hidden),
        };
        editTensors?.Invoke(t);
        string path = Path.Combine(_dir, Guid.NewGuid().ToString("N") + ".gguf");
        SyntheticGguf.Write(path, kv, t);
        return path;
    }

    private static ModelArchitectureDescriptor Arch()
    {
        Assert.True(ModelArchitectureRegistry.TryGet("minimax-m3", out var descriptor));
        return descriptor;
    }

    private static void ApplyTunables(string draft)
        => Arch().ApplyNativeTunables(new ModelCreateContext("/model-is-not-opened.gguf", BackendType.GgmlCuda, probe: null,
            tpDegree: 1, tpGroup: null, draftModelPath: draft));

    private static void Replace(List<Kv> kv, Kv value)
    {
        kv.RemoveAll(k => k.Key == value.Key);
        kv.Add(value);
    }

    [Fact]
    public void ADsparkDrafterPassesTheChecksMadeBeforeTheTargetIsMapped()
    {
        string path = WriteDrafter();
        ApplyTunables(path);
        MiniMaxM3Dspark.ValidateAgainstTarget(path, Layers, Hidden, Vocab);
    }

    public static TheoryData<string, string> FileRefusals => new()
    {
        { "arch", "general.architecture is 'qwen3'" },
        { "markov", "has no Markov head" },
        { "anchor", "bonus-anchor block layout" },
        { "activation", "hidden_activation is 'silu'" },
        { "window", "has no dflash.attention.sliding_window" },
        { "pattern", "attends the whole sequence" },
        { "taps", "has no dflash.target_layers" },
    };

    [Theory]
    [MemberData(nameof(FileRefusals))]
    public void ADrafterTheExecutorCannotRunIsRefusedWithItsReason(string broken, string reason)
    {
        string path = broken switch
        {
            "arch" => WriteDrafter(kv => Replace(kv, new Str { Key = "general.architecture", V = "qwen3" })),
            "markov" => WriteDrafter(editTensors: t => t.RemoveAll(x => x.Name.StartsWith("markov_", StringComparison.Ordinal))),
            "anchor" => WriteDrafter(kv => Replace(kv, new Bool { Key = "dflash.sample_from_anchor", V = false })),
            "activation" => WriteDrafter(kv => Replace(kv, new Str { Key = "dflash.hidden_activation", V = "silu" })),
            "window" => WriteDrafter(kv => kv.RemoveAll(k => k.Key == "dflash.attention.sliding_window")),
            "pattern" => WriteDrafter(kv => kv.Add(new BoolArrForTests("dflash.attention.sliding_window_pattern", false))),
            "taps" => WriteDrafter(kv => kv.RemoveAll(k => k.Key == "dflash.target_layers")),
            _ => throw new ArgumentOutOfRangeException(nameof(broken)),
        };
        var error = Assert.Throws<NotSupportedException>(() => ApplyTunables(path));
        Assert.StartsWith("--draft-model ", error.Message, StringComparison.Ordinal);
        Assert.Contains(reason, error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ADrafterTrainedForAnotherTargetIsRefusedWithItsReason()
    {
        // taps name the INPUT of layer k: k = Layers is the final pre-norm stream, k = 0 the embedding
        MiniMaxM3Dspark.ValidateAgainstTarget(WriteDrafter(taps: new[] { 1, Layers }), Layers, Hidden, Vocab);
        var tooDeep = Assert.Throws<NotSupportedException>(() =>
            MiniMaxM3Dspark.ValidateAgainstTarget(WriteDrafter(taps: new[] { 4, 18, 32, 46, 60 }), Layers, Hidden, Vocab));
        Assert.Contains("names layer input 18", tooDeep.Message, StringComparison.Ordinal);
        Assert.Contains("trained for another model", tooDeep.Message, StringComparison.Ordinal);
        var embedding = Assert.Throws<NotSupportedException>(() =>
            MiniMaxM3Dspark.ValidateAgainstTarget(WriteDrafter(taps: new[] { 0, 2 }), Layers, Hidden, Vocab));
        Assert.Contains("names layer input 0", embedding.Message, StringComparison.Ordinal);
        var repeated = Assert.Throws<NotSupportedException>(() =>
            MiniMaxM3Dspark.ValidateAgainstTarget(WriteDrafter(taps: new[] { 2, 2 }), Layers, Hidden, Vocab));
        Assert.Contains("repeats a layer", repeated.Message, StringComparison.Ordinal);
        var hidden = Assert.Throws<NotSupportedException>(() =>
            MiniMaxM3Dspark.ValidateAgainstTarget(WriteDrafter(hidden: 2 * Hidden), Layers, Hidden, Vocab));
        Assert.Contains($"embedding_length {2 * Hidden} does not match the target's hidden size {Hidden}", hidden.Message,
            StringComparison.Ordinal);
        var vocab = Assert.Throws<NotSupportedException>(() =>
            MiniMaxM3Dspark.ValidateAgainstTarget(WriteDrafter(vocab: 2 * Vocab), Layers, Hidden, Vocab));
        Assert.Contains($"covers {2 * Vocab} tokens", vocab.Message, StringComparison.Ordinal);
        var mask = Assert.Throws<NotSupportedException>(() => MiniMaxM3Dspark.ValidateAgainstTarget(
            WriteDrafter(kv => Replace(kv, new U32 { Key = "tokenizer.ggml.mask_token_id", V = Vocab + 3 })), Layers, Hidden, Vocab));
        Assert.Contains("outside the target's", mask.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void WithoutTheExecutorThereIsNoDraftHead()
    {
        string target = new MiniMaxM3SyntheticModelBuilder().Write(Path.Combine(_dir, "target.gguf"));
        using var model = new MiniMaxM3Model(target, BackendType.Cpu, 1, null, 1, null, loadExecutor: false);
        Assert.False(model.HasDraftHead);
        Assert.Equal(DraftHeadKind.None, model.DraftHeadKind);
        Assert.Equal(0, model.DraftBlockSize);
        Assert.False(model.DraftSelfCatchUp);
        Assert.Equal(model.Config.HiddenSize, ((ISpeculativeTarget)model).SpecFeatureSize);
        Assert.Throws<InvalidOperationException>(() => model.DraftBlock(1, null, 0, new int[7], new float[7]));
        // Speculation picks no block drafter for it.
        Assert.False(model is IDraftHead { HasDraftHead: true });
    }

    /// <summary>A GGUF bool array (type 9 of type 7), which SyntheticGguf does not carry.</summary>
    private sealed class BoolArrForTests : Kv
    {
        private readonly bool[] _v;

        public BoolArrForTests(string key, params bool[] values)
        {
            Key = key;
            _v = values;
        }

        public override void Write(BinaryWriter w)
        {
            w.Write(9u); w.Write(7u); w.Write((ulong)_v.Length);
            foreach (bool b in _v) w.Write(b);
        }
    }
}
