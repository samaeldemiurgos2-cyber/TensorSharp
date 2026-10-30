// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// A tiny DeepSeek V4.1 checkpoint with the released Flash ATTENTION geometry: a 512-wide shared K=V head
// (the owned CUDA attention's MQA shape), a 128-token sliding window, an indexer that keeps 512 compressed
// rows, and a candidate mask of 65 blocks of 8 rows owned by the first ratio-1 indexer. Everything else is
// small (5 layers, 256 hidden, 4 heads, 4 experts), so a prompt of a few thousand tokens runs in seconds.
//
// Those attention constants are what decide which owned attention kernel a prefill micro-batch gets: the
// raw ring is pad256(window + ubatch) rows (512 at ubatch 256), and the sparse kernel takes a launch of more
// than 8 queries over at least 2,048 keys (ggml_ops_precision_policy.h). With them, this fixture crosses
// the same kernel boundaries at the same prompt lengths as the 1017 checkpoints: ratio-1 layers at
// 1,025 tokens and ratio-2 layers at 2,050 (docs/validation/.../ggml-sparse-fa-chunk).
//
// Topology (eng/dsv41-fixture.py's, which the native loader's fixtures use): ratios [0,2,2,1,1], kv sources
// {1,3}, index sources {1,3,4}, candidate source 3, Engram at layer 1 in TensorSharp's per-layer layout.
// Projections are Q8_0, so ggml-cuda runs them on its q8_1-activation MMQ path as it does the 1017's
// quantized projections; compressor, router and hyper-connection weights are F32. It is an engine fixture,
// not a language model: values are seeded noise that depends only on each tensor's name.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using static InferenceWeb.Tests.SyntheticGguf;

namespace InferenceWeb.Tests;

internal static class Dsv41PrefillSplitFixture
{
    public const int Layers = 5;
    public const int Hidden = 256;
    public const int Vocab = 256;
    public const int Heads = 4;
    public const int HeadDim = 512;
    public const int RopeDims = 64;
    public const int QLoraRank = 64;
    public const int OLoraRank = 64;
    public const int OGroups = 2;
    public const int Experts = 4;
    public const int ExpertsUsed = 2;
    public const int ExpertFf = 256;
    public const int SlidingWindow = 128;
    public const int IndexerHeads = 32;
    public const int IndexerDim = 128;
    public const int IndexerTopK = 512;
    public const int CandidateSource = 3;
    public const int CandidateTopkBlocks = 65;
    public const int CandidateBlockSize = 8;
    public const int Hc = 4;
    public const int ContextLength = 4096;

    public static readonly int[] Ratios = { 0, 2, 2, 1, 1 };
    public static readonly int[] KvSources = { 1, 3 };
    public static readonly int[] IndexSources = { 1, 3, 4 };

    private const int EngramLayer = 1, EngramNgram = 3, EngramHeads = 2, EngramHeadDim = 32, EngramPadId = 2;
    private static readonly uint[] EngramPrimes = { 97, 101, 103, 107 };
    private static readonly ulong[] EngramOffsets = { 0, 97, 198, 301 };
    private static readonly ulong[] EngramMultipliers = { 29662608052400909, 1873410163432121, 13919202536988139 };

    private static readonly string Root = System.IO.Path.Combine(System.IO.Path.GetTempPath(),
        $"ts-dsv41-prefill-split-{Environment.ProcessId}-{Guid.NewGuid():N}");
    private static readonly Lazy<string> Written = new(() => Write(Root));

    static Dsv41PrefillSplitFixture()
    {
        AppDomain.CurrentDomain.ProcessExit += (_, _) =>
        {
            try { Directory.Delete(Root, recursive: true); } catch (IOException) { } catch (UnauthorizedAccessException) { }
        };
    }

    /// <summary>The fixture in this process's temp directory, written on first use.</summary>
    public static string Path => Written.Value;

    /// <summary>Writes the checkpoint into <paramref name="dir"/>; returns its path.</summary>
    public static string Write(string dir)
    {
        Directory.CreateDirectory(dir);
        string path = System.IO.Path.Combine(dir, "deepseek41-prefill-split.gguf");
        SyntheticGguf.Write(path, Metadata(), Tensors());
        return path;
    }

    private static List<Kv> Metadata()
    {
        const string a = "deepseek41.";
        var kv = new List<Kv>
        {
            new Str { Key = "general.architecture", V = "deepseek41" },
            new Str { Key = "general.name", V = "Synthetic DeepSeek V4.1 prefill-split fixture" },
            new U32 { Key = a + "block_count", V = Layers },
            new U32 { Key = a + "context_length", V = ContextLength },
            new U32 { Key = a + "embedding_length", V = Hidden },
            new U32 { Key = a + "attention.head_count", V = Heads },
            new U32 { Key = a + "attention.head_count_kv", V = 1 },
            new U32 { Key = a + "attention.key_length", V = HeadDim },
            new U32 { Key = a + "attention.value_length", V = HeadDim },
            new U32 { Key = a + "rope.dimension_count", V = RopeDims },
            new U32 { Key = a + "attention.q_lora_rank", V = QLoraRank },
            new U32 { Key = a + "attention.output_lora_rank", V = OLoraRank },
            new U32 { Key = a + "attention.output_group_count", V = OGroups },
            new U32 { Key = a + "attention.sliding_window", V = SlidingWindow },
            new U32 { Key = a + "expert_count", V = Experts },
            new U32 { Key = a + "expert_used_count", V = ExpertsUsed },
            new U32 { Key = a + "expert_shared_count", V = 1 },
            new U32 { Key = a + "expert_feed_forward_length", V = ExpertFf },
            new U32 { Key = a + "expert_gating_func", V = 4 },
            new F32 { Key = a + "expert_weights_scale", V = 1.5f },
            new Bool { Key = a + "expert_weights_norm", V = true },
            new U32 { Key = a + "hash_layer_count", V = 0 },
            new U32 { Key = a + "attention.indexer.head_count", V = IndexerHeads },
            new U32 { Key = a + "attention.indexer.key_length", V = IndexerDim },
            new U32 { Key = a + "attention.indexer.top_k", V = IndexerTopK },
            new I32 { Key = a + "attention.indexer.candidate_source_layer", V = CandidateSource },
            new U32 { Key = a + "attention.indexer.candidate_top_k", V = CandidateTopkBlocks },
            new U32 { Key = a + "attention.indexer.candidate_block_size", V = CandidateBlockSize },
            new I32Arr { Key = a + "attention.compress_ratios", V = Ratios },
            new U32 { Key = a + "hyper_connection.count", V = Hc },
            new U32 { Key = a + "hyper_connection.sinkhorn_iterations", V = 20 },
            new F32 { Key = a + "hyper_connection.epsilon", V = 1e-6f },
            new F32 { Key = a + "attention.layer_norm_rms_epsilon", V = 1e-6f },
            new F32 { Key = a + "rope.freq_base", V = 10000f },
            new F32 { Key = a + "attention.compress_rope_freq_base", V = 160000f },
            new Str { Key = a + "rope.scaling.type", V = "yarn" },
            new F32 { Key = a + "rope.scaling.factor", V = 2f },
            new U32 { Key = a + "rope.scaling.original_context_length", V = 2048 },
            new F32 { Key = a + "rope.scaling.yarn_beta_fast", V = 32f },
            new F32 { Key = a + "rope.scaling.yarn_beta_slow", V = 1f },
            new F32Arr { Key = a + "swiglu_clamp_exp", V = Enumerable.Repeat(10f, Layers).ToArray() },
            new F32Arr { Key = a + "swiglu_clamp_shexp", V = Enumerable.Repeat(10f, Layers).ToArray() },
            // Engram, TensorSharp's per-layer layout (one table per Engram layer, a compressed pad id).
            new I32Arr { Key = a + "engram.layer_ids", V = new[] { EngramLayer } },
            new U32 { Key = a + "engram.max_ngram_size", V = EngramNgram },
            new U32 { Key = a + "engram.head_count", V = EngramHeads },
            new U32 { Key = a + "engram.key_length", V = EngramHeadDim },
            new U32 { Key = a + "engram.pad_id", V = EngramPadId },
            new U64Arr { Key = a + "engram.multipliers", V = EngramMultipliers },
            new U32Arr { Key = a + "engram.primes", V = EngramPrimes },
            new U64Arr { Key = a + "engram.offsets", V = EngramOffsets },
            new I32Arr { Key = a + "engram.token_map", V = Enumerable.Range(0, Vocab).ToArray() },
            // Token ids only; a byte-level vocabulary needs a merges array to load.
            new Str { Key = "tokenizer.ggml.model", V = "gpt2" },
            new Str { Key = "tokenizer.ggml.pre", V = "joyai-llm" },
            new StrArr { Key = "tokenizer.ggml.tokens", V = Enumerable.Range(0, Vocab).Select(i => "t" + i).ToArray() },
            new I32Arr { Key = "tokenizer.ggml.token_type", V = Enumerable.Repeat(1, Vocab).ToArray() },
            new StrArr { Key = "tokenizer.ggml.merges", V = new[] { "a b" } },
            new U32 { Key = "tokenizer.ggml.bos_token_id", V = 0 },
            new U32 { Key = "tokenizer.ggml.eos_token_id", V = 1 },
            new Bool { Key = "tokenizer.ggml.add_bos_token", V = false },
            new Bool { Key = "tokenizer.ggml.add_eos_token", V = false },
        };
        return kv;
    }

    private static List<Tensor> Tensors()
    {
        var t = new List<Tensor>();
        // eng/dsv41-fixture.py's value scales: a projection's standard deviation is 1/sqrt(fan-in).
        // Normalish(scale) has a standard deviation of 0.577 * scale.
        Tensor Normal(string name, float std, params int[] dims) => Gen(name, std * 1.7320508f, dims);
        Tensor Q8(string name, float std, params int[] dims)
        {
            Tensor tensor = Normal(name, std, dims);
            tensor.Type = GgmlType.Q8_0;
            return tensor;
        }
        Tensor Proj(string name, params int[] dims) => Q8(name, 1f / MathF.Sqrt(dims[0]), dims);
        Tensor Float(string name, params int[] dims) => Normal(name, 1f / MathF.Sqrt(dims[0]), dims);
        Tensor Norm(string name, params int[] dims) => GenAround(name, 1f, 0.02f * 1.7320508f, dims);

        int head = HeadDim, mix = (2 + Hc) * Hc;
        t.Add(Q8("token_embd.weight", 0.3f, Hidden, Vocab));
        t.Add(Norm("output_norm.weight", Hidden));
        t.Add(Proj("output.weight", Hidden, Vocab));
        for (int il = 0; il < Layers; il++)
        {
            string p = $"blk.{il}.";
            foreach (string part in new[] { "attn", "ffn" })
            {
                t.Add(Norm(p + part + "_norm.weight", Hidden));
                t.Add(Float(p + $"hc_{part}_fn.weight", Hc * Hidden, mix));
                t.Add(Normal(p + $"hc_{part}_base.weight", 0.2f, mix));
                t.Add(new Tensor { Name = p + $"hc_{part}_scale.weight", Dims = new ulong[] { 3 }, Data = new[] { 0.2f, 0.3f, 0.4f } });
            }
            t.Add(Proj(p + "attn_q_a.weight", Hidden, QLoraRank));
            t.Add(Norm(p + "attn_q_a_norm.weight", QLoraRank));
            t.Add(Proj(p + "attn_q_b.weight", QLoraRank, head * Heads));
            t.Add(Proj(p + "attn_kv.weight", Hidden, head));
            t.Add(Norm(p + "attn_kv_a_norm.weight", head));
            t.Add(Proj(p + "attn_output_a.weight", head * Heads / OGroups, OLoraRank, OGroups));
            t.Add(Proj(p + "attn_output_b.weight", OLoraRank * OGroups, Hidden));
            t.Add(Normal(p + "attn_sinks.weight", 0.2f, Heads));
            if (KvSources.Contains(il))
            {
                t.Add(Float(p + "attn_compressor_kv.weight", Hidden, head));
                t.Add(Norm(p + "attn_compressor_norm.weight", head));
                if (Ratios[il] > 1)
                    t.Add(Float(p + "attn_compressor_gate.weight", Hidden, head));
                t.Add(Proj(p + "indexer.attn_k.weight", head, IndexerDim));
                t.Add(Norm(p + "indexer.k_norm.weight", IndexerDim));
            }
            if (IndexSources.Contains(il))
            {
                t.Add(Proj(p + "indexer.attn_q_b.weight", QLoraRank, IndexerHeads * IndexerDim));
                t.Add(Proj(p + "indexer.proj.weight", Hidden, IndexerHeads));
            }
            t.Add(Float(p + "ffn_gate_inp.weight", Hidden, Experts));
            t.Add(Normal(p + "exp_probs_b.bias", 0.1f, Experts));
            t.Add(Proj(p + "ffn_gate_exps.weight", Hidden, ExpertFf, Experts));
            t.Add(Proj(p + "ffn_up_exps.weight", Hidden, ExpertFf, Experts));
            t.Add(Proj(p + "ffn_down_exps.weight", ExpertFf, Hidden, Experts));
            t.Add(Proj(p + "ffn_gate_shexp.weight", Hidden, ExpertFf));
            t.Add(Proj(p + "ffn_up_shexp.weight", Hidden, ExpertFf));
            t.Add(Proj(p + "ffn_down_shexp.weight", ExpertFf, Hidden));
            if (il == EngramLayer)
            {
                int rows = (int)EngramPrimes.Sum(v => (long)v);
                int columns = (EngramNgram - 1) * EngramHeads * EngramHeadDim;
                t.Add(Q8(p + "engram_embd.weight", 0.3f, EngramHeadDim, rows));
                t.Add(Norm(p + "engram_k.weight", Hidden, Hc));
                t.Add(Norm(p + "engram_q.weight", Hidden, Hc));
                t.Add(Proj(p + "engram_wkv.weight", columns, (Hc + 1) * Hidden));
            }
        }
        return t;
    }

    private sealed class I32 : Kv { public int V; public override void Write(BinaryWriter w) { w.Write(5u); w.Write(V); } }
}
