// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// A tiny DeepSeek V4.1 checkpoint with the layout of unsloth's DeepSeek-V4.1-Flash-1017 UD-IQ2_XXS
// file: llama.cpp's COMBINED Engram layout (deepseek41.ple.* metadata, one Q5_K per_layer_token_embd
// table alone in the last shard, engram_kv and the fused engram_gate = q*k, a raw pad token that the
// token map compresses), the 1017's quant mix per role (IQ1_S / IQ2_S / IQ2_XXS gate+up over IQ2_S /
// IQ3_XXS / IQ2_XS / IQ2_XXS / Q2_K down experts, Q6_K and Q8_0 attention, Q5_K and Q8_0 shared experts,
// BF16 compressor and indexer, a Q8_0 embedding and a Q6_K head) and its cache-sharing topology in
// miniature: ratios [0,0,2,2,1,1], kv sources {2,4}, index sources {2,4,5}, the candidate mask owned by
// layer 4. The expert width is 768, three 256-value super-blocks, so a down row of IQ2_S, IQ1_S or
// IQ2_XXS is only 2-byte aligned, as the 1017's 2304-wide rows are. The indexer keeps 16 compressed rows
// and the candidate mask 4 blocks of 8, so a prompt of ~150 tokens already runs the sparse selection and
// the pruning the real model reaches past 16K.
//
// The same weights can be written in TensorSharp's per-layer Engram layout (blk.N.engram_embd /
// engram_wkv / engram_q = q*k / engram_k = ones), which is the bit-exact reference for the layout adapter.
//
// It is an engine fixture, not a language model: the weights are seeded noise (a tensor's values depend
// only on its name) quantized by ggml's own quantizers (an all-ones importance matrix for the IQ types,
// which need one), conditioned (see "conditioning" below) so engines that round differently stay close
// instead of compounding their differences through the architecture's discrete steps.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using static InferenceWeb.Tests.SyntheticGguf;

namespace InferenceWeb.Tests;

internal static class Dsv41SyntheticModelBuilder
{
    public const int Layers = 6;
    public const int Hidden = 1024;
    public const int Vocab = 256;
    public const int Heads = 4;
    public const int HeadDim = 512;
    public const int RopeDims = 64;
    public const int QLoraRank = 256;
    public const int OLoraRank = 256;
    public const int OGroups = 2;
    public const int Experts = 16;
    public const int ExpertsUsed = 4;
    public const int ExpertFf = 768;
    public const int SlidingWindow = 16;
    public const int IndexerHeads = 32;
    public const int IndexerDim = 128;
    public const int IndexerTopK = 16;
    public const int CandidateTopkBlocks = 4;
    public const int CandidateBlockSize = 8;
    public const int CandidateSource = 4;
    public const int Hc = 4;
    public const int EngramNgram = 3;
    public const int EngramHeads = 2;
    public const int EngramHeadDim = 256;
    public const int CompressedVocab = 200;
    /// <summary>The raw pad token llama.cpp's layout stores, and the id the token map compresses it to.</summary>
    public const int RawPad = 1, CompressedPad = 2;

    public static readonly int[] Ratios = Topology.Default.Ratios;
    public static readonly int[] KvSources = Topology.Default.KvSources;
    public static readonly int[] IndexSources = Topology.Default.IndexSources;
    public static readonly int[] EngramLayers = Topology.Default.EngramLayers;

    /// <summary>
    /// The fixture's cache-sharing structure. Every tensor's values depend only on its name, so two topologies
    /// share the weights of every tensor they both have.
    /// </summary>
    public sealed class Topology
    {
        public string Name { get; init; }
        public int Layers { get; init; }
        public int[] Ratios { get; init; }
        public int[] KvSources { get; init; }
        public int[] IndexSources { get; init; }
        public int CandidateSource { get; init; }
        /// <summary>Exactly two Engram layers (the hash parameters are per table).</summary>
        public int[] EngramLayers { get; init; }

        /// <summary>The 1017's in miniature: ratios [0,0,2,2,1,1], kv sources {2,4}, index sources {2,4,5},
        /// candidate source 4. Every group and the candidate mask are read by the layer right after them.</summary>
        public static readonly Topology Default = new()
        {
            Name = "default", Layers = 6, Ratios = new[] { 0, 0, 2, 2, 1, 1 }, KvSources = new[] { 2, 4 },
            IndexSources = new[] { 2, 4, 5 }, CandidateSource = 4, EngramLayers = new[] { 1, 3 },
        };

        /// <summary>
        /// The 1017's long ratio-1 group: ratios [0,0,2,2,1,1,1,1], kv sources {2,4}, index sources {2,4,6,7},
        /// candidate source 4. Group 4 is read by layers 4-7 and the candidate mask by index sources 6 and 7, so
        /// a split can put a device between the source and a reader that itself receives the state (the 1017's
        /// group 20 at its 2x2 cut, whose rows and mask node 0 receives on one GPU and hands to the next).
        /// </summary>
        public static readonly Topology Span = new()
        {
            Name = "span", Layers = 8, Ratios = new[] { 0, 0, 2, 2, 1, 1, 1, 1 }, KvSources = new[] { 2, 4 },
            IndexSources = new[] { 2, 4, 6, 7 }, CandidateSource = 4, EngramLayers = new[] { 1, 3 },
        };
    }

    // The 1017's routed-expert layouts, one per layer: every gate/up and down type the file carries.
    private const int Q2_K = 10, Q5_K = 13, Q6_K = 14, IQ2_XXS = 16, IQ2_XS = 17, IQ3_XXS = 18, IQ1_S = 19, IQ2_S = 22;
    public static readonly int[] GateUpTypes = { IQ1_S, IQ2_S, IQ2_XXS, IQ2_S, IQ1_S, IQ2_XXS };
    public static readonly int[] DownTypes = { IQ2_S, IQ3_XXS, IQ2_XS, IQ2_XXS, Q2_K, IQ3_XXS };

    /// <summary>Hash buckets per (lookback, head) column of each Engram layer's table.</summary>
    private static readonly uint[][] EngramPrimes = { new uint[] { 97, 101, 103, 107 }, new uint[] { 109, 113, 127, 131 } };
    private static readonly ulong[][] EngramMultipliers =
    {
        new ulong[] { 29662608052400909, 1873410163432121, 13919202536988139 },
        new ulong[] { 76632096046245, 4839876093313, 35959672319349 },
    };

    // ---- conditioning ----
    //
    // The architecture's discrete steps (MoE top-k routing, the indexer's top-k, the FP8 / NVFP4 / MXFP4 cache
    // rounding) turn the engines' different activation rounding (q8_1 for ggml-cuda's MMQ and the direct decode
    // kernels, F16 for the direct tensor-core experts and dequantized GEMMs, q8_K and BF16 on ggml-cpu,
    // F16-accumulated prefill attention on ggml-cuda) into large changes of whatever they feed, and an average
    // of random-direction value rows is a short vector that any such change moves a lot. With every block
    // writing a quarter-size update, ggml_cuda and ggml_cpu - two correct engines - differed by 9e-2 relative
    // L2 on this fixture's logits, so no engine could be held to 2e-2 on it. Traced per stage: an attention
    // output moved 2-40% for a 1e-3 change of its input (FP8 flips, and swapped rows where the sparse
    // selection engages); a routed-expert block 0.8% between F16 and q8_1 activations, and 35% for a token
    // whose top-4 changed. Hence:
    //  - attention and the shared expert write small updates (AttnGain, ShexpGain), and every hidden state
    //    shares one component (EmbedCommon), so attention averages are not short and a swapped row moves them
    //    little;
    //  - the routed experts write the largest update (ExpertDownGain): their share of the logits is what an
    //    expert decoder's error is seen through, against the rounding of everything else;
    //  - each token has four home experts per layer, whose router rows point along that token's own embedding
    //    (RouterHome), so its top-4 usually has a wide gap to the fifth; the selection still differs token to
    //    token, the selection bias still decides some of it (183 of the 1128 rows the parity tests trace), and
    //    every expert serves some of the prompt. Near ties remain (about 60 of those rows within 2e-2), and the
    //    two engines' rounding settles one of them differently against either ggml backend: the parity tests
    //    tell such a flip from a decoder error by the router's own margin;
    //  - the head favours one next token per token (HeadBigram), so the argmax is decided by a margin and not
    //    by near-ties among 256 random logits. That also leaves the argmax check little power of its own: the
    //    relative L2 and the parity tests' per-row stage checks carry the gate.
    // Measured (Dsv41CudaEngineParityTests): every direct route at most 8.2e-3 from ggml_cuda and 9.7e-3 from
    // ggml_cpu, which differ from each other by up to 1e-2; an IQ1_S decoder with its delta sign inverted
    // 3.9e-2 to 7.7e-2 from both, on every route (the first, 16-step form of the script). Attention and the
    // shared expert write so little that the logits see them only coarsely, hence the stage checks.
    private const float AttnGain = 0.02f;
    private const float ShexpGain = 0.02f;
    private const float ExpertDownGain = 0.2f;
    private const float EngramGain = 0.05f;
    private const float EmbedCommon = 1f;
    private const float RouterHome = 0.3f;
    private const float HeadBigram = 0.3f;

    public enum Layout
    {
        /// <summary>llama.cpp's, as the 1017 ships: two shards, the combined table alone in the second.</summary>
        Combined,
        /// <summary>TensorSharp's: one file, one table per Engram layer, the gate as engram_q with engram_k ones.</summary>
        PerLayer,
    }

    public sealed record Options
    {
        public Layout Layout { get; init; } = Layout.Combined;
        public Topology Topology { get; init; } = Topology.Default;
        /// <summary>The indexer's top-k; above the visible rows, no layer selects.</summary>
        public int IndexerTopK { get; init; } = Dsv41SyntheticModelBuilder.IndexerTopK;
        /// <summary>The candidate budget in blocks; null leaves the key out (the loaders' 2048 default).</summary>
        public int? CandidateTopkBlocks { get; init; } = Dsv41SyntheticModelBuilder.CandidateTopkBlocks;
    }

    /// <summary>The token map of the fixture's Engram hash: 256 raw tokens onto 200 compressed ids, a
    /// fixed permutation that sends the raw pad 1 to 2, with tokens 200..255 colliding with 0..55.</summary>
    public static int[] TokenMap { get; } = BuildTokenMap();

    private static int[] BuildTokenMap()
    {
        var perm = Enumerable.Range(0, CompressedVocab).ToArray();
        var rng = new Random(4109);
        for (int i = perm.Length - 1; i > 0; i--)
        {
            int j = rng.Next(i + 1);
            (perm[i], perm[j]) = (perm[j], perm[i]);
        }
        int at = Array.IndexOf(perm, CompressedPad);
        (perm[RawPad], perm[at]) = (perm[at], perm[RawPad]);
        return Enumerable.Range(0, Vocab).Select(t => perm[t % CompressedVocab]).ToArray();
    }

    // ---------------------------------------------------------------- the process-wide copies

    /// <summary>TS_TEST_DSV41_FIXTURE_OUT keeps the process-wide copies in that directory instead of a temporary
    /// one, so a tool outside the test process (eng/validation/DeepSeek41DistributedProbe) can load the same files.</summary>
    private static readonly string KeepRoot = Environment.GetEnvironmentVariable("TS_TEST_DSV41_FIXTURE_OUT");
    private static readonly string Root = !string.IsNullOrEmpty(KeepRoot) ? KeepRoot : Path.Combine(Path.GetTempPath(),
        $"ts-dsv41-synthetic-{Environment.ProcessId}-{Guid.NewGuid():N}");
    private static readonly Lazy<string> CombinedFixture = new(() => Write(Path.Combine(Root, "combined"), new Options()));
    private static readonly Lazy<string> PerLayerFixture = new(() =>
        Write(Path.Combine(Root, "per-layer"), new Options { Layout = Layout.PerLayer }));
    private static int _variant;

    static Dsv41SyntheticModelBuilder()
    {
        if (!string.IsNullOrEmpty(KeepRoot))
            return;
        AppDomain.CurrentDomain.ProcessExit += (_, _) =>
        {
            try { Directory.Delete(Root, recursive: true); } catch (IOException) { } catch (UnauthorizedAccessException) { }
        };
    }

    /// <summary>The fixture in this process's temp directory, written once per layout: its first shard.</summary>
    public static string Fixture(Layout layout = Layout.Combined)
        => (layout == Layout.Combined ? CombinedFixture : PerLayerFixture).Value;

    private static readonly System.Collections.Concurrent.ConcurrentDictionary<Topology, Lazy<string>> TopologyFixtures = new();

    /// <summary>The combined-layout fixture of another topology, written once per process: its first shard.</summary>
    public static string Fixture(Topology topology)
        => topology == Topology.Default ? Fixture() : TopologyFixtures.GetOrAdd(topology, t => new Lazy<string>(() =>
            Write(Path.Combine(Root, "topology-" + t.Name), new Options { Topology = t }))).Value;

    /// <summary>A variant of the fixture (same weights, other options) in this process's temp directory.</summary>
    public static string Variant(Options options)
        => Write(Path.Combine(Root, "variant-" + Interlocked.Increment(ref _variant)), options);

    // ---------------------------------------------------------------- writing

    /// <summary>Writes the checkpoint into <paramref name="dir"/>; returns the path to load (the first shard).</summary>
    public static string Write(string dir, Options options = null)
    {
        options ??= new Options();
        Directory.CreateDirectory(dir);
        Topology topology = options.Topology;
        Weights w = WeightsOf(topology);
        var tensors = new List<Tensor>(w.Common);
        for (int e = 0; e < topology.EngramLayers.Length; e++)
        {
            string p = $"blk.{topology.EngramLayers[e]}.";
            if (options.Layout == Layout.Combined)
            {
                tensors.Add(Raw(p + "engram_kv.weight", GgmlTypeOf(Q6_K), w.EngramKv[e], EngramColumns * EngramHeadDim, (Hc + 1) * Hidden));
                tensors.Add(Raw(p + "engram_gate.weight", GgmlType.F32, w.EngramGate[e], Hidden, Hc));
            }
            else
            {
                tensors.Add(Raw(p + "engram_embd.weight", GgmlTypeOf(Q5_K), w.EngramTable[e], EngramHeadDim, (int)EngramPrimes[e].Sum(v => (long)v)));
                tensors.Add(Raw(p + "engram_wkv.weight", GgmlTypeOf(Q6_K), w.EngramKv[e], EngramColumns * EngramHeadDim, (Hc + 1) * Hidden));
                tensors.Add(Raw(p + "engram_q.weight", GgmlType.F32, w.EngramGate[e], Hidden, Hc));
                tensors.Add(Raw(p + "engram_k.weight", GgmlType.F32, w.Ones, Hidden, Hc));
            }
        }

        if (options.Layout == Layout.PerLayer)
        {
            string path = Path.Combine(dir, "deepseek41-synthetic-1017.gguf");
            SyntheticGguf.Write(path, Metadata(options, split: null), tensors);
            return path;
        }
        string first = Path.Combine(dir, "deepseek41-synthetic-1017-00001-of-00002.gguf");
        SyntheticGguf.Write(first, Metadata(options, split: (0, 2)), tensors);
        long tableRows = EngramPrimes.Sum(p => p.Sum(v => (long)v));
        SyntheticGguf.Write(Path.Combine(dir, "deepseek41-synthetic-1017-00002-of-00002.gguf"),
            new List<Kv> { new U16 { Key = "split.no", V = 1 }, new U16 { Key = "split.count", V = 2 } },
            new List<Tensor> { Raw("per_layer_token_embd.weight", GgmlTypeOf(Q5_K), w.EngramTable.SelectMany(b => b).ToArray(), EngramHeadDim, (int)tableRows) });
        return first;
    }

    private static int EngramColumns => (EngramNgram - 1) * EngramHeads;

    private static GgmlType GgmlTypeOf(int ggmlType) => (GgmlType)ggmlType;

    private static Tensor Raw(string name, GgmlType type, byte[] bytes, params int[] dims)
        => new() { Name = name, Dims = dims.Select(d => (ulong)d).ToArray(), Type = type, RawBlocks = bytes };

    private static List<Kv> Metadata(Options options, (int No, int Count)? split)
    {
        const string a = "deepseek41.";
        Topology t = options.Topology;
        int Layers = t.Layers, CandidateSource = t.CandidateSource;
        int[] KvSources = t.KvSources, IndexSources = t.IndexSources, Ratios = t.Ratios, EngramLayers = t.EngramLayers;
        var kv = new List<Kv>
        {
            new Str { Key = "general.architecture", V = "deepseek41" },
            new Str { Key = "general.name", V = "Synthetic DeepSeek V4.1 1017-shaped engine fixture" },
            new U32 { Key = "general.quantization_version", V = 2 },
            new U32 { Key = a + "block_count", V = (uint)Layers },
            new U32 { Key = a + "context_length", V = 4096 },
            new U32 { Key = a + "embedding_length", V = Hidden },
            new U32 { Key = a + "embedding_length_out", V = Hc * Hidden },
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
            new U32 { Key = a + "attention.indexer.top_k", V = (uint)options.IndexerTopK },
            new I32 { Key = a + "attention.indexer.candidate_source_layer", V = CandidateSource },
            new U32 { Key = a + "attention.indexer.candidate_block_size", V = CandidateBlockSize },
            new I32Arr { Key = a + "attention.kv_source_layers", V = KvSources },
            new I32Arr { Key = a + "attention.indexer.source_layers", V = IndexSources },
            new I32Arr { Key = a + "attention.compress_ratios", V = Ratios },
            new U32 { Key = a + "hyper_connection.count", V = Hc },
            new U32 { Key = a + "hyper_connection.sinkhorn_iterations", V = 20 },
            new F32 { Key = a + "hyper_connection.epsilon", V = 1e-6f },
            new F32 { Key = a + "attention.layer_norm_rms_epsilon", V = 1e-20f },
            new F32 { Key = a + "rope.freq_base", V = 10000f },
            new F32 { Key = a + "attention.compress_rope_freq_base", V = 160000f },
            new Str { Key = a + "rope.scaling.type", V = "yarn" },
            new F32 { Key = a + "rope.scaling.factor", V = 2f },
            new U32 { Key = a + "rope.scaling.original_context_length", V = 2048 },
            new F32 { Key = a + "rope.scaling.yarn_beta_fast", V = 32f },
            new F32 { Key = a + "rope.scaling.yarn_beta_slow", V = 1f },
            new F32Arr { Key = a + "swiglu_clamp_exp", V = Enumerable.Repeat(10f, Layers).ToArray() },
            new F32Arr { Key = a + "swiglu_clamp_shexp", V = Enumerable.Repeat(10f, Layers).ToArray() },
            new I32Arr { Key = a + "engram.token_map", V = TokenMap },
        };
        if (options.CandidateTopkBlocks is { } blocks)
            kv.Add(new U32 { Key = a + "attention.indexer.candidate_top_k_blocks", V = (uint)blocks });

        // Each layer's buckets run on from the previous layer's in llama.cpp's layout; TensorSharp's
        // starts every layer's at its own table's first row.
        var offsets = new List<ulong>();
        ulong row = 0;
        foreach (uint[] primes in EngramPrimes)
        {
            if (options.Layout == Layout.PerLayer) row = 0;
            foreach (uint prime in primes)
            {
                offsets.Add(row);
                row += prime;
            }
        }
        if (options.Layout == Layout.Combined)
        {
            kv.Add(new U32Arr { Key = a + "ple.layers", V = EngramLayers.Select(l => (uint)l).ToArray() });
            kv.Add(new U32 { Key = a + "ple.ngram_size", V = EngramNgram });
            kv.Add(new U32 { Key = a + "ple.heads_per_ngram", V = EngramHeads });
            kv.Add(new U32 { Key = a + "embedding_length_per_layer_input", V = EngramHeadDim });
            kv.Add(new U64Arr { Key = a + "ple.layer_multipliers", V = EngramMultipliers.SelectMany(m => m).ToArray() });
            kv.Add(new U32Arr { Key = a + "ple.head_vocab_sizes", V = EngramPrimes.SelectMany(p => p).ToArray() });
            kv.Add(new U64Arr { Key = a + "ple.head_offsets", V = offsets });
            kv.Add(new U32 { Key = a + "engram.pad_token_id", V = RawPad });
            kv.Add(new U32 { Key = a + "engram.compressed_vocab_size", V = CompressedVocab });
        }
        else
        {
            kv.Add(new I32Arr { Key = a + "engram.layer_ids", V = EngramLayers });
            kv.Add(new U32 { Key = a + "engram.max_ngram_size", V = EngramNgram });
            kv.Add(new U32 { Key = a + "engram.head_count", V = EngramHeads });
            kv.Add(new U32 { Key = a + "engram.key_length", V = EngramHeadDim });
            kv.Add(new U64Arr { Key = a + "engram.multipliers", V = EngramMultipliers.SelectMany(m => m).ToArray() });
            kv.Add(new U32Arr { Key = a + "engram.primes", V = EngramPrimes.SelectMany(p => p).ToArray() });
            kv.Add(new U64Arr { Key = a + "engram.offsets", V = offsets });
            kv.Add(new U32 { Key = a + "engram.pad_id", V = CompressedPad });
        }

        // engram.py-style token ids; the fixtures feed ids, so the merges are never used, but a
        // byte-level vocabulary needs a merges array to load.
        kv.Add(new Str { Key = "tokenizer.ggml.model", V = "gpt2" });
        kv.Add(new Str { Key = "tokenizer.ggml.pre", V = "joyai-llm" });
        kv.Add(new StrArr { Key = "tokenizer.ggml.tokens", V = Enumerable.Range(0, Vocab).Select(i => "t" + i).ToArray() });
        kv.Add(new I32Arr { Key = "tokenizer.ggml.token_type", V = Enumerable.Repeat(1, Vocab).ToArray() });
        kv.Add(new StrArr { Key = "tokenizer.ggml.merges", V = new[] { "a b" } });
        kv.Add(new U32 { Key = "tokenizer.ggml.bos_token_id", V = 0 });
        kv.Add(new U32 { Key = "tokenizer.ggml.eos_token_id", V = 1 });
        kv.Add(new Bool { Key = "tokenizer.ggml.add_bos_token", V = false });
        kv.Add(new Bool { Key = "tokenizer.ggml.add_eos_token", V = false });
        if (split is { } s)
        {
            kv.Add(new U16 { Key = "split.no", V = (ushort)s.No });
            kv.Add(new U16 { Key = "split.count", V = (ushort)s.Count });
        }
        return kv;
    }

    // ---------------------------------------------------------------- the weights

    private sealed class Weights
    {
        /// <summary>Every tensor but the Engram ones, which differ by layout.</summary>
        public List<Tensor> Common = new();
        public byte[][] EngramKv, EngramGate, EngramTable;
        public byte[] Ones;
    }

    private static readonly System.Collections.Concurrent.ConcurrentDictionary<Topology, Lazy<Weights>> WeightsByTopology = new();

    private static Weights WeightsOf(Topology topology)
        => WeightsByTopology.GetOrAdd(topology, k => new Lazy<Weights>(() => BuildWeights(k))).Value;

    private static Weights BuildWeights(Topology topology)
    {
        if (topology.EngramLayers.Length != EngramPrimes.Length || topology.Ratios.Length != topology.Layers)
            throw new ArgumentException($"topology {topology.Name}: {EngramPrimes.Length} Engram layers and one ratio per layer are required");
        int Layers = topology.Layers;
        int[] KvSources = topology.KvSources, IndexSources = topology.IndexSources, Ratios = topology.Ratios,
            EngramLayers = topology.EngramLayers;
        var w = new Weights();
        var t = w.Common;

        // A projection y[out] = W[out, in] x[in] of unit-sized outputs for unit-sized inputs.
        Tensor Proj(string name, int type, float gain, params int[] dims)
        {
            float[] data = Gen(name, gain * 1.5f / MathF.Sqrt(dims[0]), dims).Data;
            return Raw(name, GgmlTypeOf(type), Encode(type, data, dims[0]), dims);
        }
        // Seeded values of roughly normal shape (Gen's spread is 0.58 of its scale).
        Tensor Float(string name, float std, params int[] dims) => Gen(name, std * 1.732f, dims);
        Tensor Norm(string name, int n) => GenAround(name, 1f, 0.035f, n);
        Tensor ExpertsProj(string name, int type, float gain, int inDim, int outDim)
        {
            float[] data = Gen(name, gain * 1.5f / MathF.Sqrt(inDim), inDim, outDim, Experts).Data;
            return Raw(name, GgmlTypeOf(type), Encode(type, data, inDim), inDim, outDim, Experts);
        }

        // The embedding: a token part plus one component every token shares.
        float[] tokenPart = Gen("token_embd.weight", 1f, Hidden, Vocab).Data;
        var tokenNorm = new float[Vocab];
        for (int tok = 0; tok < Vocab; tok++)
        {
            double norm = 0;
            for (int i = 0; i < Hidden; i++) norm += tokenPart[tok * Hidden + i] * (double)tokenPart[tok * Hidden + i];
            tokenNorm[tok] = (float)Math.Sqrt(norm);
        }
        // gain times the unit direction of a token's part, added to a row.
        void Pull(float[] rows, int row, float gain, int tok)
        {
            float k = gain / tokenNorm[tok];
            for (int i = 0; i < Hidden; i++) rows[row * Hidden + i] += k * tokenPart[tok * Hidden + i];
        }
        float[] common = Gen("token_embd.common", 1f, Hidden).Data;
        float[] embd = new float[tokenPart.Length];
        for (int i = 0; i < embd.Length; i++)
            embd[i] = tokenPart[i] + EmbedCommon * common[i % Hidden];
        t.Add(Raw("token_embd.weight", GgmlType.Q8_0, QuantizeQ8_0(embd, Hidden), Hidden, Vocab));
        t.Add(Norm("output_norm.weight", Hidden));
        // The head: unit projections plus, for each token v, a pull along the token part of the one token
        // (v * 37 + 11) % 256 it follows.
        float[] head = Gen("output.weight", 1.5f / MathF.Sqrt(Hidden), Hidden, Vocab).Data;
        for (int v = 0; v < Vocab; v++)
            Pull(head, v, HeadBigram, (v * 37 + 11) % Vocab);
        t.Add(Raw("output.weight", GgmlTypeOf(Q6_K), Encode(Q6_K, head, Hidden), Hidden, Vocab));

        for (int il = 0; il < Layers; il++)
        {
            string p = $"blk.{il}.";
            // The 1017 keeps attention in Q8_0 (27 layers) or Q6_K (13), and the shared expert in Q8_0 or
            // Q5_K alongside.
            int attn = il % 2 == 0 ? 8 : Q6_K;
            int shexp = il % 2 == 0 ? 8 : Q5_K;
            foreach (string part in new[] { "attn", "ffn" })
            {
                t.Add(Norm(p + part + "_norm.weight", Hidden));
                t.Add(Float(p + $"hc_{part}_fn.weight", 1f / MathF.Sqrt(Hc * Hidden), Hc * Hidden, (2 + Hc) * Hc));
                t.Add(Float(p + $"hc_{part}_base.weight", 0.2f, (2 + Hc) * Hc));
                t.Add(new Tensor { Name = p + $"hc_{part}_scale.weight", Dims = new ulong[] { 3 }, Data = new[] { 0.2f, 0.3f, 0.4f } });
            }
            t.Add(Proj(p + "attn_q_a.weight", attn, 1f, Hidden, QLoraRank));
            t.Add(Norm(p + "attn_q_a_norm.weight", QLoraRank));
            t.Add(Proj(p + "attn_q_b.weight", attn, 1f, QLoraRank, HeadDim * Heads));
            t.Add(Proj(p + "attn_kv.weight", attn, 1f, Hidden, HeadDim));
            t.Add(Norm(p + "attn_kv_a_norm.weight", HeadDim));
            t.Add(Proj(p + "attn_output_a.weight", attn, 1f, HeadDim * Heads / OGroups, OLoraRank, OGroups));
            t.Add(Proj(p + "attn_output_b.weight", attn, AttnGain, OLoraRank * OGroups, Hidden));
            t.Add(Float(p + "attn_sinks.weight", 0.2f, Heads));
            if (KvSources.Contains(il))
            {
                t.Add(Proj(p + "attn_compressor_kv.weight", 30, 1f, Hidden, HeadDim));
                t.Add(Norm(p + "attn_compressor_norm.weight", HeadDim));
                if (Ratios[il] > 1)
                    t.Add(Proj(p + "attn_compressor_gate.weight", 30, 1f, Hidden, HeadDim));
                t.Add(Proj(p + "indexer.attn_k.weight", 30, 1f, HeadDim, IndexerDim));
                t.Add(Norm(p + "indexer.k_norm.weight", IndexerDim));
            }
            if (IndexSources.Contains(il))
            {
                t.Add(Proj(p + "indexer.attn_q_b.weight", 30, 1f, QLoraRank, IndexerHeads * IndexerDim));
                t.Add(Proj(p + "indexer.proj.weight", 30, 1f, Hidden, IndexerHeads));
            }
            // The router: unit projections plus, for each token, a pull along its token part in the rows of
            // its four home experts of this layer.
            Tensor router = Float(p + "ffn_gate_inp.weight", 1f / MathF.Sqrt(Hidden), Hidden, Experts);
            var homes = new Random(1017 + il);
            for (int tok = 0; tok < Vocab; tok++)
                foreach (int e in Enumerable.Range(0, Experts).OrderBy(_ => homes.Next()).Take(ExpertsUsed))
                    Pull(router.Data, e, RouterHome, tok);
            t.Add(router);
            t.Add(Float(p + "exp_probs_b.bias", 0.1f, Experts));
            // Layers past the default six repeat its expert layouts.
            int gateUp = GateUpTypes[il % GateUpTypes.Length], down = DownTypes[il % DownTypes.Length];
            t.Add(ExpertsProj(p + "ffn_gate_exps.weight", gateUp, 1f, Hidden, ExpertFf));
            t.Add(ExpertsProj(p + "ffn_up_exps.weight", gateUp, 1f, Hidden, ExpertFf));
            t.Add(ExpertsProj(p + "ffn_down_exps.weight", down, ExpertDownGain, ExpertFf, Hidden));
            t.Add(Proj(p + "ffn_gate_shexp.weight", shexp, 1f, Hidden, ExpertFf));
            t.Add(Proj(p + "ffn_up_shexp.weight", shexp, 1f, Hidden, ExpertFf));
            t.Add(Proj(p + "ffn_down_shexp.weight", shexp, ShexpGain, ExpertFf, Hidden));
        }

        // Engram: per layer a Q5_K table of head_dim-wide rows, a Q6_K projection whose first hc * n_embd
        // outputs are the keys and whose last n_embd are the value (a small update, EngramGain), and
        // engram.py's q_weight * k_weight rounded once to F32.
        int columns = EngramColumns;
        w.EngramKv = new byte[EngramLayers.Length][];
        w.EngramGate = new byte[EngramLayers.Length][];
        w.EngramTable = new byte[EngramLayers.Length][];
        for (int e = 0; e < EngramLayers.Length; e++)
        {
            string p = $"blk.{EngramLayers[e]}.";
            int rows = (int)EngramPrimes[e].Sum(v => (long)v);
            w.EngramTable[e] = Encode(Q5_K, Gen(p + "engram_embd.weight", 1f, EngramHeadDim, rows).Data, EngramHeadDim);
            int inDim = columns * EngramHeadDim;
            float[] kv = Gen(p + "engram_kv.weight", 1.5f / MathF.Sqrt(inDim), inDim, (Hc + 1) * Hidden).Data;
            for (long i = (long)Hc * Hidden * inDim; i < kv.LongLength; i++)
                kv[i] *= EngramGain;
            w.EngramKv[e] = Encode(Q6_K, kv, inDim);
            float[] q = GenAround(p + "engram_q.weight", 1f, 0.035f, Hidden, Hc).Data;
            float[] k = GenAround(p + "engram_k.weight", 1f, 0.035f, Hidden, Hc).Data;
            var gate = new float[q.Length];
            for (int i = 0; i < gate.Length; i++)
                gate[i] = q[i] * k[i];
            w.EngramGate[e] = F32Bytes(gate);
        }
        var ones = new float[Hidden * Hc];
        Array.Fill(ones, 1f);
        w.Ones = F32Bytes(ones);
        return w;
    }

    private static byte[] F32Bytes(float[] values)
    {
        var bytes = new byte[values.Length * sizeof(float)];
        Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
        return bytes;
    }

    /// <summary>The bytes of <paramref name="data"/> (rows of <paramref name="rowWidth"/>) stored as
    /// <paramref name="ggmlType"/>: F32 and BF16 directly, Q8_0 exactly as ggml's quantize_row_q8_0 does,
    /// and the K and IQ formats through ggml's own quantizers.</summary>
    private static byte[] Encode(int ggmlType, float[] data, int rowWidth)
    {
        switch (ggmlType)
        {
            case 0:
                return F32Bytes(data);
            case 30:
                return new Tensor { Dims = new[] { (ulong)data.Length }, Data = data, Type = GgmlType.BF16 }.Raw();
            case 8:
                return QuantizeQ8_0(data, rowWidth);
            default:
                return QuantizeNative(ggmlType, data, rowWidth);
        }
    }

    [DllImport("GgmlOps", CallingConvention = CallingConvention.Cdecl)]
    private static extern void ggml_quantize_init(int type);

    [DllImport("GgmlOps", CallingConvention = CallingConvention.Cdecl)]
    private static extern unsafe UIntPtr ggml_quantize_chunk(int type, float* source, byte* destination,
        long start, long rows, long rowWidth, float* importance);

    private static readonly object QuantizeInitLock = new();

    /// <summary>
    /// ggml_quantize_chunk over blocks of rows in parallel (the chunks are independent once the type's
    /// tables are initialized). The IQ formats need an importance matrix; an all-ones one weighs every
    /// column alike. The K formats are quantized without one, as llama-quantize does without --imatrix.
    /// </summary>
    private static unsafe byte[] QuantizeNative(int ggmlType, float[] data, int rowWidth)
    {
        (int rowBytes256, bool needsImportance) = ggmlType switch
        {
            Q2_K => (84, false),
            Q5_K => (176, false),
            Q6_K => (210, false),
            IQ2_XXS => (66, true),
            IQ2_XS => (74, true),
            IQ3_XXS => (98, true),
            IQ1_S => (50, true),
            IQ2_S => (82, true),
            _ => throw new ArgumentOutOfRangeException(nameof(ggmlType), $"no fixture encoder for ggml type {ggmlType}"),
        };
        if (rowWidth % 256 != 0 || data.Length % rowWidth != 0)
            throw new ArgumentException($"ggml type {ggmlType} rows must hold whole 256-value blocks");
        long rows = data.Length / rowWidth;
        long rowBytes = rowWidth / 256 * rowBytes256;
        var result = new byte[rows * rowBytes];
        var importance = new float[rowWidth];
        Array.Fill(importance, 1f);
        lock (QuantizeInitLock)
            ggml_quantize_init(ggmlType);
        const int ChunkRows = 64;
        long chunks = (rows + ChunkRows - 1) / ChunkRows;
        fixed (float* src = data)
        fixed (byte* dst = result)
        fixed (float* imp = importance)
        {
            float* s = src;
            byte* d = dst;
            float* im = needsImportance ? imp : null;
            long written = 0;
            Parallel.For(0L, chunks, c =>
            {
                long first = c * ChunkRows, n = Math.Min(ChunkRows, rows - first);
                ulong bytes = ggml_quantize_chunk(ggmlType, s, d, first * rowWidth, n, rowWidth, im).ToUInt64();
                Interlocked.Add(ref written, (long)bytes);
            });
            if (written != result.LongLength)
                throw new InvalidOperationException($"ggml type {ggmlType}: the quantizer wrote {written} bytes, expected {result.LongLength}");
        }
        return result;
    }

    // ---------------------------------------------------------------- metadata kinds SyntheticGguf lacks

    private sealed class U16 : Kv { public ushort V; public override void Write(BinaryWriter w) { w.Write(2u); w.Write(V); } }

    private sealed class I32 : Kv { public int V; public override void Write(BinaryWriter w) { w.Write(5u); w.Write(V); } }
}
