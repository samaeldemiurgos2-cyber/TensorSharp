// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using TensorSharp;
using TensorSharp.Cpu;

namespace InferenceWeb.Tests;

/// <summary>
/// A whole V4.1 model in llama.cpp's combined Engram layout (unsloth's DeepSeek-V4.1-Flash-1017
/// GGUFs) on both managed executors, against the same model in TensorSharp's per-layer layout.
///
/// <para>The combined copy is written from the per-layer fixture the way llama.cpp lays the 1017
/// out: <c>deepseek41.ple.*</c> metadata and a raw pad token, the Engram tables concatenated into
/// one <c>per_layer_token_embd.weight</c> that starts a few rows in and sits alone in a second
/// shard, <c>engram_kv</c> for <c>engram_wkv</c>, and <c>engram_gate</c> holding q*k in place of
/// <c>engram_q</c> and <c>engram_k</c>. A per-layer copy that carries the same folded product
/// (<c>engram_q</c> = q*k, <c>engram_k</c> = ones) is the exact reference: every byte the two
/// executors read is the same, so the logits must be identical to the bit. Against the original
/// fixture the only difference is the rounding of rms(x)*(q*k) . rms(key) against
/// (rms(x)*q) . (rms(key)*k), which the pure C# executor's gate absorbs and the direct-CUDA
/// engine's quantized activations can amplify.</para>
///
/// <para>The fixture is the one <see cref="Dsv41CudaSlotTests"/> runs
/// (<c>python eng/dsv41-fixture.py DIR --cuda-attn --q8</c>, then
/// <c>TS_TEST_DSV41_CUDA_FIXTURE_DIR=DIR</c>); without it these tests skip.</para>
/// </summary>
public sealed class Dsv41CombinedEngramFixtureTests : IDisposable
{
    private const string FixtureVar = "TS_TEST_DSV41_CUDA_FIXTURE_DIR";
    private const string Fixture = "deepseek41-fixture";

    /// <summary>Rows of the combined table before the first Engram layer's slice: a slice that
    /// does not start at row 0 is what a missing offset rebase or a misplaced view gets wrong.</summary>
    private const int LeadingRows = 5;

    private readonly string _dir = Path.Combine(Path.GetTempPath(), "dsv41-combined-" + Guid.NewGuid().ToString("N"));

    public Dsv41CombinedEngramFixtureTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { } catch (UnauthorizedAccessException) { }
    }

    private static string PerLayerFixture() =>
        TestGates.FindGguf(Environment.GetEnvironmentVariable(FixtureVar), Fixture);

    private static int[] Tokens(string gguf)
    {
        string path = Path.Combine(Path.GetDirectoryName(gguf), "tokens.json");
        return File.Exists(path)
            ? JsonSerializer.Deserialize<int[]>(File.ReadAllText(path))
            : new[] { 0, 15, 32, 64, 128, 13, 254, 18, 22, 26, 16, 4, 11, 3, 18, 9 };
    }

    [ModelFact(FixtureVar, Fixture)]
    public void ThePureCSharpExecutorReadsTheCombinedLayoutExactly()
    {
        string original = PerLayerFixture();
        int[] tokens = Tokens(original);
        float[][] Run(string path)
        {
            using var executor = new DeepSeek4CpuExecutor(path, maxContext: 256, nUbatch: 32, nThreads: 4,
                new CpuAllocator(BlasEnum.DotNet));
            return Steps(tokens, executor.VocabSize, executor.Forward);
        }
        float[][] combined = Run(WriteLayout(original, _dir, combined: true));
        AssertIdentical(Run(WriteLayout(original, _dir, combined: false)), combined);
        // Its gate rounds the folded product away (measured: identical to the bit here).
        AssertClose(Run(original), combined, 1e-4);
    }

    [CudaFact(FixtureVar, Fixture)]
    public void TheDirectCudaEngineReadsTheCombinedLayoutExactly()
    {
        string original = PerLayerFixture();
        int[] tokens = Tokens(original);
        float[][] Run(string path)
        {
            using var executor = new DeepSeek4CudaExecutor(path, maxContext: 256, nUbatch: 32, nGpu: 1);
            return Steps(tokens, executor.VocabSize, executor.Forward);
        }
        float[][] combined = Run(WriteLayout(original, _dir, combined: true));
        AssertIdentical(Run(WriteLayout(original, _dir, combined: false)), combined);
        // The engine quantizes activations, so the folded product's last-bit differences can tip
        // a rounding step: measured 1.1e-2 of the largest logit on this fixture, against 0.1-0.3
        // (absolute) between this engine and the pure C# executor on the same file.
        AssertClose(Run(original), combined, 5e-2);
    }

    /// <summary>The candidate pruning budget is <c>candidate_top_k_blocks</c> in the 1017 and
    /// <c>candidate_top_k</c> in TensorSharp's own files. Every engine has to take it from either
    /// name: one that knew only the other would fall back to 2048 and, past the budget, prune
    /// differently from the rest on the same file. The fixture's own budget under the 1017's name
    /// must give its logits to the bit; with no budget at all (the 2048 default) they must change,
    /// or this prompt never reaches pruning and the first check proves nothing.</summary>
    [ModelFact(FixtureVar, Fixture, GgmlBackend = BackendType.GgmlCpu)]
    public void EveryEngineReadsTheCandidateBudgetUnderEitherName()
    {
        const string topK = "deepseek41.attention.indexer.candidate_top_k";
        const string blocks = "deepseek41.attention.indexer.candidate_top_k_blocks";
        string original = PerLayerFixture();
        int[] tokens = Tokens(original);
        string renamed = WriteMetadata(original, Path.Combine(_dir, "renamed.gguf"), m => { m[blocks] = m[topK]; m.Remove(topK); });
        string unbudgeted = WriteMetadata(original, Path.Combine(_dir, "unbudgeted.gguf"), m => m.Remove(topK));

        float[][] Native(string path)
        {
            using var model = new DeepSeek4Model(path, BackendType.GgmlCpu);
            return Steps(tokens, model.Config.VocabSize, (t, logits) => model.Forward(t).CopyTo(logits, 0));
        }
        float[][] Managed(string path)
        {
            using var executor = new DeepSeek4CpuExecutor(path, maxContext: 256, nUbatch: 32, nThreads: 4,
                new CpuAllocator(BlasEnum.DotNet));
            return Steps(tokens, executor.VocabSize, executor.Forward);
        }
        string[] keys = { "MAX_CONTEXT", "TS_DSV4_UBATCH", "TS_DSV41_YOCO_KEEP" };
        string[] values = { "256", "32", "0" };
        string[] old = keys.Select(Environment.GetEnvironmentVariable).ToArray();
        try
        {
            for (int i = 0; i < keys.Length; i++) Environment.SetEnvironmentVariable(keys[i], values[i]);
            foreach (var (engine, run) in new (string, Func<string, float[][]>)[] { ("ggml_cpu", Native), ("cpu", Managed) })
            {
                float[][] expected = run(original);
                Assert.True(Identical(expected, run(renamed)), $"{engine} reads a different budget from {blocks}");
                Assert.False(Identical(expected, run(unbudgeted)), $"{engine}: the prompt never reaches candidate pruning");
            }
        }
        finally
        {
            for (int i = 0; i < keys.Length; i++) Environment.SetEnvironmentVariable(keys[i], old[i]);
        }
    }

    private static bool Identical(float[][] a, float[][] b) =>
        a.Length == b.Length && a.Zip(b).All(p =>
            MemoryMarshal.AsBytes(p.First.AsSpan()).SequenceEqual(MemoryMarshal.AsBytes(p.Second.AsSpan())));

    /// <summary>A prefill of half the prompt, then the rest one token at a time; the logits after
    /// each call.</summary>
    private static float[][] Steps(int[] tokens, int vocab, Action<int[], float[]> forward)
    {
        var rows = new List<float[]>();
        int half = tokens.Length / 2;
        var logits = new float[vocab];
        forward(tokens[..half], logits);
        rows.Add((float[])logits.Clone());
        for (int i = half; i < tokens.Length; i++)
        {
            forward(new[] { tokens[i] }, logits);
            rows.Add((float[])logits.Clone());
        }
        return rows.ToArray();
    }

    private static void AssertIdentical(float[][] perLayer, float[][] combined)
    {
        Assert.Equal(perLayer.Length, combined.Length);
        for (int s = 0; s < perLayer.Length; s++)
            for (int i = 0; i < perLayer[s].Length; i++)
                Assert.True(BitConverter.SingleToInt32Bits(perLayer[s][i]) == BitConverter.SingleToInt32Bits(combined[s][i]),
                    $"step {s} logit {i}: per-layer {perLayer[s][i]}, combined {combined[s][i]}");
    }

    /// <summary>Same argmax at every step, and logits within <paramref name="tolerance"/> of the
    /// largest one. A wrong Engram row or gate moves them by far more.</summary>
    private static void AssertClose(float[][] original, float[][] combined, double tolerance)
    {
        Assert.Equal(original.Length, combined.Length);
        double worst = 0;
        for (int s = 0; s < original.Length; s++)
        {
            float scale = Math.Max(1f, original[s].Max(MathF.Abs));
            for (int i = 0; i < original[s].Length; i++)
            {
                Assert.True(float.IsFinite(combined[s][i]), $"step {s} logit {i} is {combined[s][i]}");
                worst = Math.Max(worst, Math.Abs(original[s][i] - combined[s][i]) / scale);
            }
            Assert.Equal(Array.IndexOf(original[s], original[s].Max()), Array.IndexOf(combined[s], combined[s].Max()));
        }
        Console.WriteLine($"[dsv41-combined] worst logit difference from the original layout: {worst:E3} of the largest logit");
        Assert.True(worst <= tolerance, $"the layouts differ by {worst:E3} of the largest logit");
    }

    // ---------------------------------------------------------------- the layout rewrite

    private static readonly string[] PerLayerKeys =
    {
        "deepseek41.engram.pad_id", "deepseek41.engram.max_ngram_size", "deepseek41.engram.head_count",
        "deepseek41.engram.key_length", "deepseek41.engram.layer_ids", "deepseek41.engram.multipliers",
        "deepseek41.engram.primes", "deepseek41.engram.offsets",
    };

    /// <summary>Writes <paramref name="source"/> (TensorSharp's per-layer Engram layout) into
    /// <paramref name="dir"/> in llama.cpp's combined layout, as two shards with the combined
    /// table alone in the second, and returns the first shard's path; or, without
    /// <paramref name="combined"/>, as one per-layer file whose engram_q holds q*k and whose
    /// engram_k is ones -- the same gate the combined copy carries.</summary>
    internal static string WriteLayout(string source, string dir, bool combined)
    {
        using var src = GgufFile.OpenWithoutSiblingShards(source);
        byte[] Bytes(GgufTensorInfo info)
        {
            var bytes = new byte[src.GetTensorByteCount(info)];
            using var stream = File.OpenRead(source);
            stream.Position = src.DataOffset + (long)info.Offset;
            stream.ReadExactly(bytes);
            return bytes;
        }
        float[] Floats(GgufTensorInfo info)
        {
            byte[] raw = Bytes(info);
            var values = new float[info.NumElements];
            IntPtr p = Marshal.AllocHGlobal(raw.Length);
            try
            {
                Marshal.Copy(raw, 0, p, raw.Length);
                ManagedQuantizedOps.DequantizeToFloat32((int)info.Type, p, values, 0, values.Length);
            }
            finally { Marshal.FreeHGlobal(p); }
            return values;
        }
        ulong[] U64s(string key) => ((Array)src.Metadata[key]).Cast<object>().Select(Convert.ToUInt64).ToArray();
        // engram.py's q_weight * k_weight, rounded once to F32 as llama.cpp's converter stores it.
        byte[] FoldedGate(string layer)
        {
            float[] q = Floats(src.Tensors[layer + "engram_q.weight"]), k = Floats(src.Tensors[layer + "engram_k.weight"]);
            var gate = new byte[q.Length * sizeof(float)];
            for (int i = 0; i < q.Length; i++)
                BitConverter.TryWriteBytes(gate.AsSpan(i * sizeof(float)), q[i] * k[i]);
            return gate;
        }
        string LayerOf(string name) =>
            name.StartsWith("blk.", StringComparison.Ordinal) ? name[..(name.IndexOf('.', 4) + 1)] : null;

        if (!combined)
        {
            var all = new List<SyntheticGguf.Tensor>();
            foreach (var (name, info) in src.Tensors)
            {
                if (name.EndsWith(".engram_q.weight", StringComparison.Ordinal))
                    all.Add(Raw(name, info.Shape, GgmlTensorType.F32, FoldedGate(LayerOf(name))));
                else if (name.EndsWith(".engram_k.weight", StringComparison.Ordinal))
                {
                    var ones = new byte[info.NumElements * sizeof(float)];
                    for (long i = 0; i < info.NumElements; i++)
                        BitConverter.TryWriteBytes(ones.AsSpan((int)i * sizeof(float)), 1f);
                    all.Add(Raw(name, info.Shape, GgmlTensorType.F32, ones));
                }
                else
                    all.Add(Raw(name, info.Shape, info.Type, Bytes(info)));
            }
            string foldedPath = Path.Combine(dir, "deepseek41-folded.gguf");
            SyntheticGguf.Write(foldedPath,
                src.Metadata.Where(e => e.Key != "general.alignment")
                    .Select(e => (SyntheticGguf.Kv)new AnyKv { Key = e.Key, V = e.Value }).ToList(),
                all);
            return foldedPath;
        }

        var m = src.Metadata;
        uint ngram = Convert.ToUInt32(m["deepseek41.engram.max_ngram_size"]);
        uint heads = Convert.ToUInt32(m["deepseek41.engram.head_count"]);
        uint headDim = Convert.ToUInt32(m["deepseek41.engram.key_length"]);
        uint pad = Convert.ToUInt32(m["deepseek41.engram.pad_id"]);
        ulong[] ids = U64s("deepseek41.engram.layer_ids");
        ulong[] offsets = U64s("deepseek41.engram.offsets");
        ulong[] map = U64s("deepseek41.engram.token_map");
        int columns = (int)((ngram - 1) * heads);

        // The tables, back to back after LeadingRows rows that no lookup reads; each layer's
        // offsets run on from where the previous layer's table ended, as llama.cpp writes them.
        var tables = ids.Select(id => src.Tensors[$"blk.{id}.engram_embd.weight"]).ToArray();
        long rowBytes = src.GetTensorByteCount(new GgufTensorInfo { Shape = new ulong[] { headDim }, Type = tables[0].Type });
        var table = new MemoryStream();
        table.Write(Bytes(tables[0]), 0, (int)(LeadingRows * rowBytes));
        var headOffsets = new ulong[offsets.Length];
        ulong rowsSoFar = LeadingRows;
        for (int e = 0; e < tables.Length; e++)
        {
            Assert.Equal(tables[0].Type, tables[e].Type);
            for (int c = 0; c < columns; c++)
                headOffsets[e * columns + c] = rowsSoFar + offsets[e * columns + c];
            table.Write(Bytes(tables[e]));
            rowsSoFar += tables[e].Shape[1];
        }

        var kv = new List<SyntheticGguf.Kv>();
        foreach (var (key, value) in m)
        {
            if (key != "general.alignment" && key != "split.count" && !PerLayerKeys.Contains(key))
                kv.Add(new AnyKv { Key = key, V = value });
        }
        kv.Add(new AnyKv { Key = "split.count", V = (ushort)2 });
        kv.Add(new AnyKv { Key = "deepseek41.ple.ngram_size", V = ngram });
        kv.Add(new AnyKv { Key = "deepseek41.ple.heads_per_ngram", V = heads });
        kv.Add(new AnyKv { Key = "deepseek41.embedding_length_per_layer_input", V = headDim });
        kv.Add(new AnyKv { Key = "deepseek41.ple.layers", V = ids.Select(id => (uint)id).ToArray() });
        kv.Add(new AnyKv { Key = "deepseek41.ple.layer_multipliers", V = U64s("deepseek41.engram.multipliers") });
        kv.Add(new AnyKv { Key = "deepseek41.ple.head_vocab_sizes", V = U64s("deepseek41.engram.primes").Select(p => (uint)p).ToArray() });
        kv.Add(new AnyKv { Key = "deepseek41.ple.head_offsets", V = headOffsets });
        // The raw token whose compressed id is the pad, as engram.py's config stores it.
        kv.Add(new AnyKv { Key = "deepseek41.engram.pad_token_id", V = (uint)Array.IndexOf(map, (ulong)pad) });
        kv.Add(new AnyKv { Key = "deepseek41.engram.compressed_vocab_size", V = (uint)(map.Max() + 1) });

        var first = new List<SyntheticGguf.Tensor>();
        foreach (var (name, info) in src.Tensors)
        {
            string layer = LayerOf(name);
            if (layer == null || !ids.Any(id => layer == $"blk.{id}."))
            {
                first.Add(Raw(name, info.Shape, info.Type, Bytes(info)));
                continue;
            }
            switch (name[layer.Length..])
            {
                case "engram_embd.weight":
                case "engram_k.weight":
                    break;
                case "engram_wkv.weight":
                    first.Add(Raw(layer + "engram_kv.weight", info.Shape, info.Type, Bytes(info)));
                    break;
                case "engram_q.weight":
                    first.Add(Raw(layer + "engram_gate.weight", info.Shape, GgmlTensorType.F32, FoldedGate(layer)));
                    break;
                default:
                    first.Add(Raw(name, info.Shape, info.Type, Bytes(info)));
                    break;
            }
        }

        string firstPath = Path.Combine(dir, "deepseek41-combined-00001-of-00002.gguf");
        SyntheticGguf.Write(firstPath, kv, first);
        SyntheticGguf.Write(Path.Combine(dir, "deepseek41-combined-00002-of-00002.gguf"),
            new List<SyntheticGguf.Kv> { new AnyKv { Key = "split.count", V = (ushort)2 } },
            new List<SyntheticGguf.Tensor>
            {
                Raw(Dsv41EngramData.CombinedTableName, new[] { (ulong)headDim, rowsSoFar }, tables[0].Type, table.ToArray()),
            });
        return firstPath;
    }

    /// <summary>Copies <paramref name="source"/> to <paramref name="path"/> with its metadata
    /// changed by <paramref name="edit"/>; every tensor keeps its bytes.</summary>
    private static string WriteMetadata(string source, string path, Action<Dictionary<string, object>> edit)
    {
        using var src = GgufFile.OpenWithoutSiblingShards(source);
        var metadata = src.Metadata.Where(e => e.Key != "general.alignment").ToDictionary(e => e.Key, e => e.Value);
        edit(metadata);
        var tensors = new List<SyntheticGguf.Tensor>();
        using var stream = File.OpenRead(source);
        foreach (var (name, info) in src.Tensors)
        {
            var bytes = new byte[src.GetTensorByteCount(info)];
            stream.Position = src.DataOffset + (long)info.Offset;
            stream.ReadExactly(bytes);
            tensors.Add(Raw(name, info.Shape, info.Type, bytes));
        }
        SyntheticGguf.Write(path, metadata.Select(e => (SyntheticGguf.Kv)new AnyKv { Key = e.Key, V = e.Value }).ToList(), tensors);
        return path;
    }

    private static SyntheticGguf.Tensor Raw(string name, ulong[] shape, GgmlTensorType type, byte[] bytes) =>
        new() { Name = name, Dims = (ulong[])shape.Clone(), Type = (SyntheticGguf.GgmlType)(int)type, RawBlocks = bytes };

    /// <summary>A metadata entry of any type the GGUF reader produces, written back as that type.</summary>
    private sealed class AnyKv : SyntheticGguf.Kv
    {
        public object V;

        private static uint Code(Type t) =>
            t == typeof(byte) ? 0u : t == typeof(sbyte) ? 1u : t == typeof(ushort) ? 2u : t == typeof(short) ? 3u :
            t == typeof(uint) ? 4u : t == typeof(int) ? 5u : t == typeof(float) ? 6u : t == typeof(bool) ? 7u :
            t == typeof(string) ? 8u : t == typeof(ulong) ? 10u : t == typeof(long) ? 11u : t == typeof(double) ? 12u :
            throw new NotSupportedException($"GGUF has no metadata type for {t}");

        private static void Value(BinaryWriter w, object v)
        {
            switch (v)
            {
                case byte x: w.Write(x); break;
                case sbyte x: w.Write(x); break;
                case ushort x: w.Write(x); break;
                case short x: w.Write(x); break;
                case uint x: w.Write(x); break;
                case int x: w.Write(x); break;
                case float x: w.Write(x); break;
                case bool x: w.Write((byte)(x ? 1 : 0)); break;
                case ulong x: w.Write(x); break;
                case long x: w.Write(x); break;
                case double x: w.Write(x); break;
                case string x:
                    byte[] utf8 = Encoding.UTF8.GetBytes(x);
                    w.Write((ulong)utf8.Length);
                    w.Write(utf8);
                    break;
                default: throw new NotSupportedException($"GGUF has no metadata type for {v?.GetType()}");
            }
        }

        public override void Write(BinaryWriter w)
        {
            if (V is Array array)
            {
                w.Write(9u);
                w.Write(Code(array.GetType().GetElementType()));
                w.Write((ulong)array.Length);
                foreach (object element in array)
                    Value(w, element);
                return;
            }
            w.Write(Code(V.GetType()));
            Value(w, V);
        }
    }
}
