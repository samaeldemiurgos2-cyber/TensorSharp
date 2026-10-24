// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// Builds a tiny, deterministic minimax-m3 GGUF with the real checkpoint's tensor
// inventory and types: Q8_0 attention, dense and shared-expert matrices, F32 norms,
// indexer and router, NVFP4 routed experts with per-expert F32 .scale vectors and
// BF16 embedding/output. Every matrix's leading dimension is a multiple of 64 (the
// NVFP4 block) so the same layout quantizes the same way. The managed tests use it
// to drive the constructor's metadata parse, the tensor validation and the load
// refusals; the numeric parity fixture is eng/minimax-m3-fixture.py's.
using static InferenceWeb.Tests.SyntheticGguf;

namespace InferenceWeb.Tests;

internal sealed class MiniMaxM3SyntheticModelBuilder
{
    internal const int Hidden = 64, Heads = 4, KvHeads = 2, HeadDim = 32, RopeDim = 16;
    internal const int DenseFfn = 128, Experts = 4, Used = 2, ExpertFfn = 64;
    internal const int IndexerDim = 32, IndexerTopK = 2, IndexerBlock = 16;
    internal const int Vocab = 256, Bos = 1, Eos = 2;

    /// <summary>ggml's GGML_TYPE_NVFP4: 64 values in 4 UE4M3 sub-block scales and 32 nibble bytes.</summary>
    internal const GgmlType Nvfp4 = (GgmlType)40;

    public int Layers { get; init; } = 4;
    public int DenseLead { get; init; } = 2;
    /// <summary>MiniMax-M3.1 carries the indexer on every layer; MiniMax-M3 only on the MoE layers.</summary>
    public bool IndexerOnDenseLayers { get; init; } = true;
    public bool ExpertScales { get; init; } = true;
    public int IndexerHeads { get; init; } = KvHeads;
    public uint GatingFunc { get; init; } = 2;
    public int ContextLength { get; init; } = 4096;

    /// <summary>Applied to the tensor list after it is built (drop, reshape or retype a tensor).</summary>
    public Action<List<Tensor>> EditTensors { get; init; }

    /// <summary>Applied to the metadata after it is built.</summary>
    public Action<List<Kv>> EditMetadata { get; init; }

    public string Write(string path)
    {
        var tensors = BuildTensors();
        EditTensors?.Invoke(tensors);
        var kv = BuildMetadata();
        EditMetadata?.Invoke(kv);
        SyntheticGguf.Write(path, kv, tensors);
        return path;
    }

    internal List<Kv> BuildMetadata()
    {
        const string a = "minimax-m3";
        var kv = new List<Kv>
        {
            new Str { Key = "general.architecture", V = a },
            new Str { Key = "general.name", V = "MiniMax-M3 synthetic" },
            new U32 { Key = $"{a}.block_count", V = (uint)Layers },
            new U32 { Key = $"{a}.context_length", V = (uint)ContextLength },
            new U32 { Key = $"{a}.embedding_length", V = Hidden },
            new U32 { Key = $"{a}.feed_forward_length", V = DenseFfn },
            new U32 { Key = $"{a}.attention.head_count", V = Heads },
            new U32 { Key = $"{a}.attention.head_count_kv", V = KvHeads },
            new F32 { Key = $"{a}.rope.freq_base", V = 1e7f },
            new F32 { Key = $"{a}.attention.layer_norm_rms_epsilon", V = 1e-6f },
            new U32 { Key = $"{a}.expert_count", V = Experts },
            new U32 { Key = $"{a}.expert_used_count", V = Used },
            new U32 { Key = $"{a}.expert_gating_func", V = GatingFunc },
            new U32 { Key = $"{a}.attention.key_length", V = HeadDim },
            new U32 { Key = $"{a}.attention.value_length", V = HeadDim },
            new U32 { Key = $"{a}.expert_feed_forward_length", V = ExpertFfn },
            new U32 { Key = $"{a}.rope.dimension_count", V = RopeDim },
            new U32 { Key = $"{a}.expert_shared_count", V = 1 },
            new F32 { Key = $"{a}.expert_weights_scale", V = 2f },
            new Bool { Key = $"{a}.expert_weights_norm", V = true },
            new U32 { Key = $"{a}.attention.indexer.head_count", V = (uint)IndexerHeads },
            new U32 { Key = $"{a}.attention.indexer.key_length", V = IndexerDim },
            new U32 { Key = $"{a}.attention.indexer.top_k", V = IndexerTopK },
            new U32 { Key = $"{a}.attention.indexer.block_size", V = IndexerBlock },
            new U32 { Key = $"{a}.attention.indexer.local_blocks", V = 1 },
            new U32 { Key = $"{a}.leading_dense_block_count", V = (uint)DenseLead },
        };
        AddByteTokenizer(kv, Vocab, Bos, Eos);
        return kv;
    }

    internal List<Tensor> BuildTensors()
    {
        var tensors = new List<Tensor>
        {
            Typed(Gen("token_embd.weight", 0.5f, Hidden, Vocab), GgmlType.BF16),
            GenAround("output_norm.weight", 1f, 0.1f, Hidden),
            Typed(Gen("output.weight", 0.1f, Hidden, Vocab), GgmlType.BF16),
        };
        for (int l = 0; l < Layers; l++)
        {
            string p = $"blk.{l}.";
            bool dense = l < DenseLead;
            tensors.Add(GenAround(p + "attn_norm.weight", 1f, 0.1f, Hidden));
            tensors.Add(GenAround(p + "ffn_norm.weight", 1f, 0.1f, Hidden));
            tensors.Add(Typed(Gen(p + "attn_q.weight", 0.1f, Hidden, Heads * HeadDim), GgmlType.Q8_0));
            tensors.Add(Typed(Gen(p + "attn_k.weight", 0.1f, Hidden, KvHeads * HeadDim), GgmlType.Q8_0));
            tensors.Add(Typed(Gen(p + "attn_v.weight", 0.1f, Hidden, KvHeads * HeadDim), GgmlType.Q8_0));
            tensors.Add(Typed(Gen(p + "attn_output.weight", 0.1f, Heads * HeadDim, Hidden), GgmlType.Q8_0));
            tensors.Add(GenAround(p + "attn_q_norm.weight", 1f, 0.1f, HeadDim));
            tensors.Add(GenAround(p + "attn_k_norm.weight", 1f, 0.1f, HeadDim));
            if (!dense || IndexerOnDenseLayers)
            {
                tensors.Add(Gen(p + "indexer.q_proj.weight", 0.1f, Hidden, IndexerHeads * IndexerDim));
                tensors.Add(Gen(p + "indexer.k_proj.weight", 0.1f, Hidden, IndexerDim));
                tensors.Add(GenAround(p + "indexer.q_norm.weight", 1f, 0.1f, IndexerDim));
                tensors.Add(GenAround(p + "indexer.k_norm.weight", 1f, 0.1f, IndexerDim));
            }
            if (dense)
            {
                tensors.Add(Typed(Gen(p + "ffn_gate.weight", 0.1f, Hidden, DenseFfn), GgmlType.Q8_0));
                tensors.Add(Typed(Gen(p + "ffn_up.weight", 0.1f, Hidden, DenseFfn), GgmlType.Q8_0));
                tensors.Add(Typed(Gen(p + "ffn_down.weight", 0.1f, DenseFfn, Hidden), GgmlType.Q8_0));
                continue;
            }
            tensors.Add(Gen(p + "ffn_gate_inp.weight", 0.2f, Hidden, Experts));
            // The real routing bias sits around 16..37, well above the sigmoid scores it is added to.
            tensors.Add(GenAround(p + "exp_probs_b.bias", 16f + l, 0.05f, Experts));
            tensors.Add(Nvfp4Blocks(p + "ffn_gate_exps.weight", Hidden, ExpertFfn, Experts));
            tensors.Add(Nvfp4Blocks(p + "ffn_up_exps.weight", Hidden, ExpertFfn, Experts));
            tensors.Add(Nvfp4Blocks(p + "ffn_down_exps.weight", ExpertFfn, Hidden, Experts));
            if (ExpertScales)
                foreach (string e in new[] { "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps" })
                    tensors.Add(GenAround(p + e + ".scale", 1e-3f, 2e-4f, Experts));
            tensors.Add(Typed(Gen(p + "ffn_gate_shexp.weight", 0.1f, Hidden, ExpertFfn), GgmlType.Q8_0));
            tensors.Add(Typed(Gen(p + "ffn_up_shexp.weight", 0.1f, Hidden, ExpertFfn), GgmlType.Q8_0));
            tensors.Add(Typed(Gen(p + "ffn_down_shexp.weight", 0.1f, ExpertFfn, Hidden), GgmlType.Q8_0));
        }
        return tensors;
    }

    private static Tensor Typed(Tensor tensor, GgmlType type)
    {
        tensor.Type = type;
        return tensor;
    }

    /// <summary>
    /// NVFP4 block bytes seeded by the tensor name: per 64 values, four UE4M3 sub-block scales
    /// (kept in a moderate exponent range, never the 0x7F NaN code) and 32 bytes of E2M1 nibbles.
    /// </summary>
    internal static Tensor Nvfp4Blocks(string name, params int[] dims)
    {
        if (dims[0] % 64 != 0) throw new ArgumentException($"{name}: NVFP4 rows hold whole 64-value blocks.");
        long values = 1;
        foreach (int d in dims) values *= d;
        var raw = new byte[values / 64 * 36];
        uint state = 2166136261;
        foreach (char ch in name) state = unchecked((state ^ ch) * 16777619);
        for (long b = 0; b < raw.Length / 36; b++)
        {
            for (int i = 0; i < 36; i++)
            {
                state ^= state << 13; state ^= state >> 17; state ^= state << 5;
                raw[b * 36 + i] = i < 4 ? (byte)(0x20 + (state >> 28)) : (byte)(state >> 24);
            }
        }
        return new Tensor
        {
            Name = name,
            Dims = dims.Select(d => (ulong)d).ToArray(),
            Type = Nvfp4,
            RawBlocks = raw,
        };
    }

    /// <summary>Remove the tensor called <paramref name="name"/>.</summary>
    internal static Action<List<Tensor>> Drop(string name) => list =>
    {
        if (list.RemoveAll(t => t.Name == name) != 1)
            throw new InvalidOperationException($"fixture has no tensor {name}");
    };

    /// <summary>Replace the dimensions of the tensor called <paramref name="name"/>, keeping its byte count.</summary>
    internal static Action<List<Tensor>> Reshape(string name, params int[] dims) => list =>
    {
        var t = list.Single(x => x.Name == name);
        long count = dims.Aggregate(1L, (n, d) => n * d);
        if (t.RawBlocks == null && count != t.Data.Length)
            t.Data = new float[count];
        t.Dims = dims.Select(d => (ulong)d).ToArray();
    };

    /// <summary>Store the tensor called <paramref name="name"/> as <paramref name="type"/>.</summary>
    internal static Action<List<Tensor>> Retype(string name, GgmlType type) => list =>
        list.Single(x => x.Name == name).Type = type;

    /// <summary>Set the U32 metadata key <paramref name="key"/> (architecture-prefixed) to <paramref name="value"/>.</summary>
    internal static Action<List<Kv>> SetU32(string key, uint value) => kv =>
    {
        kv.RemoveAll(e => e.Key == key);
        kv.Add(new U32 { Key = key, V = value });
    };
}
