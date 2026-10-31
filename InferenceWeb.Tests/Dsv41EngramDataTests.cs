// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Runtime.InteropServices;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

/// <summary>
/// The Engram GGUF metadata decides which of 384 million rows each token reads. A
/// single wrong multiplier, prime or offset selects a different row and changes
/// every embedding silently, so the parser and the hashing are pinned here
/// against an independently written expectation rather than against themselves.
/// </summary>
public class Dsv41EngramDataTests : IDisposable
{
    private readonly List<string> _temp = new();

    public void Dispose()
    {
        foreach (string p in _temp)
        {
            try { File.Delete(p); } catch { /* best effort */ }
        }
    }

    /// <summary>An empty GGUF that the parser has read, for fixtures that
    /// write their metadata and tensor headers straight into it.</summary>
    private GgufFile EmptyFile()
    {
        string path = Path.Combine(Path.GetTempPath(), $"dsv41-engram-{Guid.NewGuid():N}.gguf");
        using (var writer = new BinaryWriter(File.Create(path)))
        {
            writer.Write(0x46554747u); writer.Write(3u);
            writer.Write(0UL); writer.Write(0UL);
        }
        _temp.Add(path);
        return GgufFile.OpenWithoutSiblingShards(path);
    }

    private static void Header(GgufFile file, string name, params ulong[] shape) =>
        file.Tensors[name] = new GgufTensorInfo { Name = name, Shape = shape, Type = GgmlTensorType.F32 };

    /// <summary>Creates embedded metadata with two distinct layer-major hash
    /// layouts. The GGUF parser reads the file; tests may mutate metadata to
    /// check rejection without allocating embedding-table payloads.</summary>
    private GgufFile EmbeddedFixture()
    {
        var file = EmptyFile();
        var m = file.Metadata;
        m["tokenizer.ggml.tokens"] = Enumerable.Range(0, 8).Select(i => $"token{i}").ToArray();
        m["deepseek41.block_count"] = 6u;
        m["deepseek41.engram.pad_id"] = 3u;
        m["deepseek41.engram.max_ngram_size"] = 3u;
        m["deepseek41.engram.head_count"] = 2u;
        m["deepseek41.engram.key_length"] = 16u;
        m["deepseek41.engram.layer_ids"] = new uint[] { 1, 5 };
        m["deepseek41.engram.multipliers"] = new ulong[] { 7, 9, 11, 9, 11, 13 };
        m["deepseek41.engram.primes"] = new uint[] { 11, 13, 15, 17, 15, 17, 19, 21 };
        m["deepseek41.engram.offsets"] = new ulong[] { 0, 11, 24, 39, 0, 15, 32, 51 };
        // The embedded pad ID is compressed already; TokenMap[3] != 3.
        m["deepseek41.engram.token_map"] = new int[] { 0, 1, 2, 0, 3, 1, 2, 3 };
        m["deepseek41.attention.compress_ratios"] = new uint[] { 2, 2, 1, 1, 1, 1 };
        void Tensor(string name, params ulong[] shape) =>
            file.Tensors[name] = new GgufTensorInfo { Name = name, Shape = shape, Type = GgmlTensorType.F32 };
        Tensor("token_embd.weight", 16, 8);
        Tensor("blk.1.engram_embd.weight", 16, 56);
        Tensor("blk.5.engram_embd.weight", 16, 72);
        Tensor("blk.0.attn_compressor_kv.weight", 16, 16);
        Tensor("blk.2.attn_compressor_kv.weight", 16, 16);
        Tensor("blk.0.indexer.attn_q_b.weight", 16, 16);
        Tensor("blk.2.indexer.attn_q_b.weight", 16, 16);
        Tensor("blk.4.indexer.attn_q_b.weight", 16, 16);
        return file;
    }

    [Fact]
    public void Load_ReadsEveryFieldOfAWellFormedEmbeddedGguf()
    {
        using var file = EmbeddedFixture();
        var data = Dsv41EngramData.Load(file);

        Assert.Equal(8u, data.VocabSize);
        Assert.Equal(4u, data.CompressedVocabSize);
        Assert.Equal(3u, data.PadId);
        Assert.Equal(3u, data.MaxNgramSize);
        Assert.Equal(2u, data.HeadCount);
        Assert.Equal(16u, data.HeadDim);
        Assert.Equal(4u, data.HashColumns);              // (3-1) lookbacks x 2 heads
        Assert.Equal(new[] { 1, 5 }, data.Layers.Select(l => l.Id));
        // Offsets must be the running sum of the bucket sizes, and the last
        // bucket must land exactly on the row count.
        foreach (var layer in data.Layers)
        {
            long running = 0;
            for (int c = 0; c < layer.Primes.Length; c++)
            {
                Assert.Equal(running, layer.Offsets[c]);
                running += layer.Primes[c];
            }
            Assert.Equal(layer.Rows, running);
        }
    }

    [Theory]
    [InlineData("deepseek41.engram.token_map")]
    [InlineData("deepseek41.engram.multipliers")]
    [InlineData("deepseek41.engram.primes")]
    [InlineData("deepseek41.engram.offsets")]
    public void Load_RequiresEmbeddedMetadata(string key)
    {
        using var file = EmbeddedFixture();
        file.Metadata.Remove(key);
        var error = Assert.Throws<InvalidDataException>(() => Dsv41EngramData.Load(file));
        Assert.Contains(key, error.Message);
    }

    [Theory]
    [InlineData("map-length")]
    [InlineData("map-gap")]
    [InlineData("map-negative")]
    [InlineData("map-overflow")]
    [InlineData("offset-gap")]
    [InlineData("multiplier-even")]
    [InlineData("multiplier-overflow")]
    [InlineData("layer-order")]
    [InlineData("table-rows")]
    [InlineData("table-width")]
    [InlineData("unlisted-table")]
    [InlineData("vocab")]
    [InlineData("pad")]
    [InlineData("candidate")]
    [InlineData("candidate-topk")]
    [InlineData("candidate-topk-overflow")]
    [InlineData("candidate-block-overflow")]
    public void Load_RejectsInvalidEmbeddedGeometry(string fault)
    {
        using var file = EmbeddedFixture();
        var m = file.Metadata;
        switch (fault)
        {
            case "map-length": m["deepseek41.engram.token_map"] = new int[] { 0, 1, 2, 3 }; break;
            case "map-gap": m["deepseek41.engram.token_map"] = new int[] { 0, 1, 2, 0, 4, 1, 2, 4 }; break;
            case "map-negative": m["deepseek41.engram.token_map"] = new int[] { -1, 1, 2, 0, 3, 1, 2, 3 }; break;
            case "map-overflow": m["deepseek41.engram.token_map"] = new ulong[] { 0, 1, 2, 0, 3, 1, 2, ulong.MaxValue }; break;
            case "offset-gap": ((ulong[])m["deepseek41.engram.offsets"])[6]++; break;
            case "multiplier-even": ((ulong[])m["deepseek41.engram.multipliers"])[0] = 8; break;
            case "multiplier-overflow": ((ulong[])m["deepseek41.engram.multipliers"])[5] = ulong.MaxValue; break;
            case "layer-order": m["deepseek41.engram.layer_ids"] = new uint[] { 5, 1 }; break;
            case "table-rows": file.Tensors["blk.5.engram_embd.weight"].Shape[1]++; break;
            case "unlisted-table": file.Tensors["blk.3.engram_embd.weight"] = file.Tensors["blk.5.engram_embd.weight"]; break;
            case "table-width": file.Tensors["blk.5.engram_embd.weight"].Shape[0]++; break;
            case "vocab": file.Tensors["token_embd.weight"].Shape[1]++; break;
            case "pad": m["deepseek41.engram.pad_id"] = 4u; break;
            case "candidate": m["deepseek41.attention.indexer.candidate_source_layer"] = 1; break;
            case "candidate-topk-overflow": m["deepseek41.attention.indexer.candidate_top_k"] = 2147483648u; break;
            case "candidate-block-overflow": m["deepseek41.attention.indexer.candidate_block_size"] = 2147483648u; break;
            case "candidate-topk": m["deepseek41.attention.indexer.candidate_top_k"] = 0u; break;
        }
        Assert.Throws<InvalidDataException>(() => Dsv41EngramData.Load(file));
    }

    [Fact]
    public void Load_DerivesCacheOwnershipAndAppliesCandidateOverrides()
    {
        using var file = EmbeddedFixture();
        var data = Dsv41EngramData.Load(file);
        Assert.Equal(new[] { 0, 2 }, data.KvSourceLayerIds);
        Assert.Equal(new[] { 0, 2, 4 }, data.IndexSourceLayerIds);
        Assert.Equal(2, data.CandidateSourceLayerId);
        Assert.Equal(2048u, data.CandidateTopkBlocks);
        Assert.Equal(8u, data.CandidateBlockSize);
        file.Metadata["deepseek41.attention.indexer.candidate_source_layer"] = 4;
        file.Metadata["deepseek41.attention.indexer.candidate_top_k"] = 2u;
        file.Metadata["deepseek41.attention.indexer.candidate_block_size"] = 2u;
        data = Dsv41EngramData.Load(file);
        Assert.Equal(4, data.CandidateSourceLayerId);
        Assert.Equal(2u, data.CandidateTopkBlocks);
        Assert.Equal(2u, data.CandidateBlockSize);
        file.Metadata["deepseek41.attention.indexer.candidate_source_layer"] = -1;
        Assert.Equal(-1, Dsv41EngramData.Load(file).CandidateSourceLayerId);
    }

    /// <summary>
    /// The hashing recurrence, checked against the formula written out longhand:
    /// a rolling XOR of (compressed token * multiplier) over the last
    /// MaxNgramSize tokens, reduced per column by that column's prime and shifted
    /// by its offset. Every head at one lookback shares the rolling value.
    /// </summary>
    [Fact]
    public void HashTokens_MatchesTheRecurrenceWrittenOutLonghand()
    {
        using var file = EmbeddedFixture();
        var data = Dsv41EngramData.Load(file);

        int[] tokens = { 5, 2, 7, 0 };
        int[] history = null; int historyLength = 0;
        int[] actual = data.HashTokens(tokens, 0, ref history, ref historyLength);

        uint columns = data.HashColumns;
        Assert.Equal(data.Layers.Length * tokens.Length * (int)columns, actual.Length);

        int[] compressed = tokens.Select(t => data.TokenMap[t]).ToArray();
        int padCompressed = (int)data.PadId;

        for (int li = 0; li < data.Layers.Length; li++)
        {
            var layer = data.Layers[li];
            for (int i = 0; i < tokens.Length; i++)
            {
                ulong rolling = 0;
                for (uint shift = 0; shift < data.MaxNgramSize; shift++)
                {
                    // Before the start of the sequence the lookback is blocked and
                    // reads the pad token instead.
                    int token = (i < shift) ? padCompressed : compressed[i - (int)shift];
                    rolling ^= (ulong)token * layer.Multipliers[shift];
                    if (shift == 0) continue;
                    for (uint head = 0; head < data.HeadCount; head++)
                    {
                        uint column = (shift - 1) * data.HeadCount + head;
                        int expected = (int)(rolling % layer.Primes[column] + (ulong)layer.Offsets[column]);
                        Assert.Equal(expected, actual[(li * tokens.Length + i) * columns + column]);
                    }
                }
            }
        }
    }

    [Fact]
    public void HashTokens_EveryRowLandsInsideItsOwnColumnBucket()
    {
        using var file = EmbeddedFixture();
        var data = Dsv41EngramData.Load(file);
        int[] tokens = { 1, 6, 3, 4, 0, 7 };
        int[] history = null; int historyLength = 0;
        int[] rows = data.HashTokens(tokens, 0, ref history, ref historyLength);

        uint columns = data.HashColumns;
        for (int li = 0; li < data.Layers.Length; li++)
        {
            var layer = data.Layers[li];
            for (int i = 0; i < tokens.Length; i++)
            {
                for (uint c = 0; c < columns; c++)
                {
                    int row = rows[(li * tokens.Length + i) * columns + c];
                    Assert.InRange(row, (int)layer.Offsets[c], (int)(layer.Offsets[c] + layer.Primes[c] - 1));
                    Assert.InRange(row, 0, (int)layer.Rows - 1);
                }
            }
        }
    }

    /// <summary>
    /// A lookback that reaches a visual position is blocked, and so is every
    /// longer one behind it. That is what keeps an image from leaking into the
    /// n-grams of the text that follows it.
    /// </summary>
    [Fact]
    public void HashTokens_AnImagePositionBlocksItselfAndEveryLookbackThroughIt()
    {
        using var file = EmbeddedFixture();
        var data = Dsv41EngramData.Load(file);
        uint columns = data.HashColumns;

        int[] history = null; int historyLength = 0;
        int[] withImage = data.HashTokens(new[] { 1, -1, 4 }, 0, ref history, ref historyLength);

        // Position 2 looks back one to the image and two past it, so with
        // MaxNgramSize 3 every lookback it has is blocked; only the token itself
        // and pad substitutions contribute.
        int[] h2 = null; int l2 = 0;
        int[] allBlocked = data.HashTokens(new[] { -1, -1, 4 }, 0, ref h2, ref l2);
        for (int li = 0; li < data.Layers.Length; li++)
        {
            for (uint c = 0; c < columns; c++)
            {
                int a = withImage[(li * 3 + 2) * columns + c];
                int b = allBlocked[(li * 3 + 2) * columns + c];
                Assert.Equal(b, a);
            }
        }
    }

    /// <summary>History spans calls, because a lookback reaches tokens the
    /// previous ubatch supplied.</summary>
    [Fact]
    public void HashTokens_LooksBackIntoAnEarlierCall()
    {
        using var file = EmbeddedFixture();
        var data = Dsv41EngramData.Load(file);
        uint columns = data.HashColumns;

        int[] whole = null; int wholeLen = 0;
        int[] oneShot = data.HashTokens(new[] { 5, 2, 7, 0 }, 0, ref whole, ref wholeLen);

        int[] split = null; int splitLen = 0;
        data.HashTokens(new[] { 5, 2 }, 0, ref split, ref splitLen);
        int[] tail = data.HashTokens(new[] { 7, 0 }, 2, ref split, ref splitLen);

        for (int li = 0; li < data.Layers.Length; li++)
        {
            for (int i = 0; i < 2; i++)
            {
                for (uint c = 0; c < columns; c++)
                {
                    int expected = oneShot[(li * 4 + (2 + i)) * columns + c];
                    int actual = tail[(li * 2 + i) * columns + c];
                    Assert.Equal(expected, actual);
                }
            }
        }
    }

    [Fact]
    public void HashTokens_RefusesANonContiguousHistory()
    {
        using var file = EmbeddedFixture();
        var data = Dsv41EngramData.Load(file);
        int[] history = null; int historyLength = 0;
        data.HashTokens(new[] { 1, 2 }, 0, ref history, ref historyLength);
        Assert.Throws<InvalidOperationException>(() =>
            data.HashTokens(new[] { 3 }, 5, ref history, ref historyLength));
    }

    /// <summary>Validates the embedded metadata in the published checkpoint.
    /// Unavailable checkpoints are reported as skipped, never passed.</summary>
    [Dsv41RealGgufFact]
    public void Load_ReadsTheRealCheckpointEmbeddedGguf()
    {
        string path = Environment.GetEnvironmentVariable("TS_TEST_DSV41_REAL_GGUF");

        using var file = new GgufFile(path);
        var data = Dsv41EngramData.Load(file);

        Assert.Equal(99092u, data.CompressedVocabSize);
        Assert.Equal(4u, data.MaxNgramSize);
        Assert.Equal(8u, data.HeadCount);
        Assert.Equal(256u, data.HeadDim);
        Assert.Equal(24u, data.HashColumns);
        Assert.Equal(new[] { 1, 14 }, data.Layers.Select(l => l.Id));
        foreach (var layer in data.Layers)
        {
            // Both published tables are ~384 million rows.
            Assert.InRange(layer.Rows, 380_000_000L, 390_000_000L);
            Assert.Equal(24, layer.Primes.Length);
            Assert.Equal(layer.Rows, layer.Offsets[^1] + layer.Primes[^1]);
        }

        // Hashing the real map must land every row inside its own bucket.
        int[] history = null; int historyLength = 0;
        int[] rows = data.HashTokens(new[] { 100, 2000, 55, 129279, 0 }, 0, ref history, ref historyLength);
        for (int li = 0; li < data.Layers.Length; li++)
        {
            var layer = data.Layers[li];
            for (int i = 0; i < 5; i++)
            {
                for (int c = 0; c < 24; c++)
                {
                    int row = rows[(li * 5 + i) * 24 + c];
                    Assert.InRange(row, (int)layer.Offsets[c], (int)(layer.Offsets[c] + layer.Primes[c] - 1));
                }
            }
        }
    }

    /// <summary>
    /// The canonical hash oracle. These row ids were produced by a NumPy
    /// reference using the official tokenizer normalization, outside TensorSharp
    /// entirely. Matching them
    /// is what proves the managed hashing agrees with the checkpoint, including
    /// the image position at index 4 and the lookbacks it blocks.
    /// Set TS_TEST_DSV41_REAL_GGUF to the real GGUF metadata to enable it.
    /// </summary>
    [Dsv41RealGgufFact]
    public void HashTokens_MatchesTheCanonicalNumPyOracle()
    {
        string path = Environment.GetEnvironmentVariable("TS_TEST_DSV41_REAL_GGUF");

        using var file = new GgufFile(path);
        var data = Dsv41EngramData.Load(file);
        Assert.Equal(new[] { 2, 8, 14, 20 }, data.KvSourceLayerIds);
        Assert.Equal(new[] { 2, 8, 14, 20, 24, 28, 32, 36 }, data.IndexSourceLayerIds);
        Assert.Equal(20, data.CandidateSourceLayerId);
        Assert.Equal(8u, data.CandidateBlockSize);
        Assert.Equal(2048u, data.CandidateTopkBlocks);

        AssertMatchesTheNumPyOracle(data);
    }

    /// <summary>The NumPy oracle's tokens (index 4 is an image position) and its row ids,
    /// [table][token][one of <see cref="OracleColumns"/>].</summary>
    private static readonly int[] OracleTokens = { 0, 100, 101, 102, -1, 103, 104 };
    private static readonly int[] OracleColumns = { 0, 7, 8, 23 };
    private static readonly int[][][] OracleRows =
    {
        new[]
        {
            new[] { 5702652, 121476532, 131476717, 380066193 },
            new[] { 9357813, 120674457, 142523656, 371635201 },
            new[] { 14967493, 120087816, 140350287, 379577895 },
            new[] { 8085103, 120276459, 133334380, 369222544 },
            new[] { 4299726, 112312540, 129401291, 370096053 },
            new[] { 5558068, 126975926, 140551575, 383461561 },
            new[] { 13583392, 116008417, 137862774, 377607687 },
        },
        new[]
        {
            new[] { 14361964, 120604547, 132225184, 372375971 },
            new[] { 15410290, 112117398, 140943284, 383683101 },
            new[] { 3011271, 114792526, 137397375, 378010528 },
            new[] { 5098593, 122052292, 143543847, 368813437 },
            new[] { 6788701, 120694389, 131862336, 381853422 },
            new[] { 10655595, 119663028, 134680611, 369662410 },
            new[] { 4014438, 123436036, 130889979, 374480667 },
        },
    };

    private static void AssertMatchesTheNumPyOracle(Dsv41EngramData data)
    {
        int[] history = null; int historyLength = 0;
        int[] actual = data.HashTokens(OracleTokens, 0, ref history, ref historyLength);
        for (int layer = 0; layer < 2; layer++)
            for (int token = 0; token < OracleTokens.Length; token++)
                for (int c = 0; c < OracleColumns.Length; c++)
                    Assert.Equal(OracleRows[layer][token][c],
                        actual[(layer * OracleTokens.Length + token) * 24 + OracleColumns[c]]);
    }

    /// <summary>
    /// Chunking a prompt must not change a single row id: history is indexed by
    /// absolute position, so a lookback reaches back into an earlier ubatch. The
    /// C# executor splits every prompt longer than its microbatch, so this is the
    /// property that keeps long prompts correct.
    /// </summary>
    [Dsv41RealGgufFact]
    public void HashTokens_ChunkingAPromptChangesNothing()
    {
        string path = Environment.GetEnvironmentVariable("TS_TEST_DSV41_REAL_GGUF");

        using var file = new GgufFile(path);
        var data = Dsv41EngramData.Load(file);
        int[] tokens = { 0, 100, 101, 102, -1, 103, 104 };
        int columns = (int)data.HashColumns;

        int[] whole = null; int wholeLen = 0;
        int[] oneShot = data.HashTokens(tokens, 0, ref whole, ref wholeLen);

        foreach (int[] split in new[] { new[] { 1, 6 }, new[] { 3, 4 }, new[] { 1, 1, 1, 1, 1, 1, 1 } })
        {
            int[] history = null; int historyLength = 0;
            int at = 0;
            foreach (int take in split)
            {
                int[] chunk = data.HashTokens(tokens.AsSpan(at, take), at, ref history, ref historyLength);
                for (int li = 0; li < data.Layers.Length; li++)
                    for (int i = 0; i < take; i++)
                        for (int c = 0; c < columns; c++)
                            Assert.Equal(
                                oneShot[(li * tokens.Length + at + i) * columns + c],
                                chunk[(li * take + i) * columns + c]);
                at += take;
            }
        }
    }

    /// <summary>The 1017 GGUFs spell the candidate budget candidate_top_k_blocks; TensorSharp's
    /// own files spell it candidate_top_k. Either is read, and a file with both must agree.</summary>
    [Fact]
    public void Load_ReadsTheCandidateBudgetUnderEitherName()
    {
        using var file = EmbeddedFixture();
        var m = file.Metadata;
        m["deepseek41.attention.indexer.candidate_top_k_blocks"] = 5u;
        Assert.Equal(5u, Dsv41EngramData.Load(file).CandidateTopkBlocks);
        m["deepseek41.attention.indexer.candidate_top_k"] = 5u;
        Assert.Equal(5u, Dsv41EngramData.Load(file).CandidateTopkBlocks);
        m["deepseek41.attention.indexer.candidate_top_k"] = 6u;
        Assert.Contains("candidate_top_k_blocks disagrees with",
            Assert.Throws<InvalidDataException>(() => Dsv41EngramData.Load(file)).Message);
        m.Remove("deepseek41.attention.indexer.candidate_top_k_blocks");
        Assert.Equal(6u, Dsv41EngramData.Load(file).CandidateTopkBlocks);
    }

    // ---- llama.cpp's combined layout (unsloth's DeepSeek-V4.1-Flash-1017 GGUFs) ----

    /// <summary>
    /// llama.cpp's layout as the native combined-layout test writes it
    /// (dsv41_engram_gguf_test.cpp): <c>deepseek41.ple.*</c> keys whose head offsets run on
    /// across the two layers of one combined table, and a raw pad token that the token map
    /// compresses.
    /// </summary>
    private GgufFile NativeCombinedFixture()
    {
        var file = EmptyFile();
        var m = file.Metadata;
        m["tokenizer.ggml.tokens"] = new[] { "a", "b", "c", "d", "e", "f" };
        m["deepseek41.block_count"] = 4u;
        m["deepseek41.ple.ngram_size"] = 3u;
        m["deepseek41.ple.heads_per_ngram"] = 2u;
        m["deepseek41.embedding_length_per_layer_input"] = 2u;
        m["deepseek41.engram.pad_token_id"] = 1u;
        m["deepseek41.engram.compressed_vocab_size"] = 4u;
        m["deepseek41.ple.layers"] = new ulong[] { 1, 3 };
        m["deepseek41.ple.layer_multipliers"] = new ulong[] { 7, 11, 13, 17, 19, 23 };
        m["deepseek41.ple.head_vocab_sizes"] = new ulong[] { 3, 5, 7, 11, 13, 17, 19, 23 };
        m["deepseek41.ple.head_offsets"] = new ulong[] { 0, 3, 8, 15, 26, 39, 56, 75 };
        m["deepseek41.engram.token_map"] = new ulong[] { 0, 2, 1, 2, 3, 1 };
        m["deepseek41.attention.compress_ratios"] = new ulong[] { 0, 2, 1, 1 };
        Header(file, "token_embd.weight", 2, 6);
        Header(file, Dsv41EngramData.CombinedTableName, 2, 98);
        Header(file, "blk.1.attn_compressor_kv.weight", 2, 2);
        Header(file, "blk.2.attn_compressor_kv.weight", 2, 2);
        Header(file, "blk.1.indexer.attn_q_b.weight", 2, 2);
        Header(file, "blk.2.indexer.attn_q_b.weight", 2, 2);
        Header(file, "blk.3.indexer.attn_q_b.weight", 2, 2);
        return file;
    }

    /// <summary>
    /// The native combined-layout test, ported case for case. One adaptation: the native loader
    /// is handed the per-layer views its caller already made, while the managed one derives them
    /// itself, so this tensor table holds the combined table instead of blk.N.engram_embd.
    /// </summary>
    [Fact]
    public void Load_ReadsLlamaCppsCombinedLayoutAsTheNativeLoaderDoes()
    {
        using var ple = NativeCombinedFixture();
        using var tensorSharp = EmbeddedFixture();
        Assert.True(Dsv41EngramData.HasCombinedMetadata(ple));
        Assert.False(Dsv41EngramData.HasCombinedMetadata(tensorSharp));
        Assert.Equal(new[] { new Dsv41EngramData.CombinedSlice(1, 0, 26), new Dsv41EngramData.CombinedSlice(3, 26, 72) },
            Dsv41EngramData.CombinedSlices(ple, 98));

        var data = Dsv41EngramData.Load(ple);
        Assert.Equal(Dsv41EngramData.CombinedSlices(ple, 98), data.Slices);
        Assert.Equal(new[] { ("blk.1.engram_embd.weight", 0L, 26L), ("blk.3.engram_embd.weight", 26L, 72L) },
            data.TableViews());
        // pad_token_id 1 is compressed through the token map (engram.py), giving 2.
        Assert.Equal(2u, data.PadId);
        Assert.Equal(4u, data.CompressedVocabSize);
        Assert.Equal(new long[] { 0, 3, 8, 15 }, data.Layers[0].Offsets);
        Assert.Equal(new long[] { 0, 13, 30, 49 }, data.Layers[1].Offsets);
        Assert.Equal(72L, data.Layers[1].Rows);
        Assert.Equal(2u, data.HeadDim);
        Assert.Equal(2u, data.HeadCount);
        int[] history = null; int historyLength = 0;
        int[] rows = data.HashTokens(new[] { 2, -1, 4 }, 0, ref history, ref historyLength);
        Assert.Equal(((3 * 7) ^ (2 * 11)) % 3, rows[8]);
        Assert.Equal(((3 * 7) ^ (2 * 11) ^ (2 * 13)) % 7 + 8, rows[10]);
        Assert.Equal(((3 * 17) ^ (2 * 19)) % 13, rows[20]);
        Assert.Equal(((3 * 17) ^ (2 * 19) ^ (2 * 23)) % 19 + 30, rows[22]);

        // A table one row short; a gap between two buckets of one layer; a compressed vocabulary
        // the token map does not produce; a pad token past the vocabulary.
        Assert.Throws<InvalidDataException>(() => Dsv41EngramData.CombinedSlices(ple, 97));
        ple.Tensors[Dsv41EngramData.CombinedTableName].Shape[1] = 97;
        Assert.Throws<InvalidDataException>(() => Dsv41EngramData.Load(ple));
        ple.Tensors[Dsv41EngramData.CombinedTableName].Shape[1] = 98;
        var m = ple.Metadata;
        m["deepseek41.ple.head_offsets"] = new ulong[] { 0, 3, 8, 15, 26, 39, 57, 75 };
        Assert.Throws<InvalidDataException>(() => Dsv41EngramData.CombinedSlices(ple, 98));
        Assert.Throws<InvalidDataException>(() => Dsv41EngramData.Load(ple));
        m["deepseek41.ple.head_offsets"] = new ulong[] { 0, 3, 8, 15, 26, 39, 56, 75 };
        m["deepseek41.engram.compressed_vocab_size"] = 5u;
        Assert.Throws<InvalidDataException>(() => Dsv41EngramData.Load(ple));
        m["deepseek41.engram.compressed_vocab_size"] = 4u;
        m["deepseek41.engram.pad_token_id"] = 6u;
        Assert.Throws<InvalidDataException>(() => Dsv41EngramData.Load(ple));
        m["deepseek41.engram.pad_token_id"] = 1u;
        Assert.Equal(2u, Dsv41EngramData.Load(ple).PadId);
    }

    /// <summary>What only the managed loader has to refuse, because it is the one that turns the
    /// combined table into per-layer tables: a file with both kinds of table, a combined table
    /// nothing addresses or no shard holds, and one that is not two-dimensional.</summary>
    [Theory]
    [InlineData("listed-per-layer-table", "has both blk.3.engram_embd.weight and llama.cpp's combined")]
    [InlineData("unlisted-per-layer-table", "has both blk.0.engram_embd.weight and llama.cpp's combined")]
    [InlineData("tensorsharp-metadata-too", "has both TensorSharp's per-layer Engram metadata")]
    [InlineData("no-ple-layers", "but no deepseek41.ple.layers metadata")]
    [InlineData("no-table", "no shard holds it")]
    [InlineData("three-dimensional-table", "must be two-dimensional")]
    public void Load_RefusesAMixedOrIncompleteCombinedLayout(string fault, string message)
    {
        using var file = NativeCombinedFixture();
        switch (fault)
        {
            case "listed-per-layer-table": Header(file, "blk.3.engram_embd.weight", 2, 72); break;
            case "unlisted-per-layer-table": Header(file, "blk.0.engram_embd.weight", 2, 26); break;
            case "tensorsharp-metadata-too": file.Metadata["deepseek41.engram.layer_ids"] = new uint[] { 1, 3 }; break;
            case "no-ple-layers": file.Metadata.Remove("deepseek41.ple.layers"); break;
            case "no-table": file.Tensors.Remove(Dsv41EngramData.CombinedTableName); break;
            case "three-dimensional-table": file.Tensors[Dsv41EngramData.CombinedTableName].Shape = new ulong[] { 2, 49, 2 }; break;
        }
        Assert.Contains(message, Assert.Throws<InvalidDataException>(() => Dsv41EngramData.Load(file)).Message);
    }

    /// <summary>
    /// One Engram geometry written in both layouts. Two layers, so a combined loader that did not
    /// rebase the second layer's offsets fails; a combined table whose first layer starts three
    /// rows in, so one that assumed row 0 fails; and a token map that moves the pad (raw 3 is
    /// compressed 0), so one that did not compress the raw pad token fails.
    /// </summary>
    private (GgufFile TensorSharp, GgufFile Combined) OneGeometryInBothLayouts()
    {
        GgufFile Common()
        {
            var file = EmptyFile();
            var m = file.Metadata;
            m["tokenizer.ggml.tokens"] = Enumerable.Range(0, 8).Select(i => $"token{i}").ToArray();
            m["deepseek41.block_count"] = 6u;
            m["deepseek41.engram.token_map"] = new int[] { 0, 1, 2, 0, 3, 1, 2, 3 };
            m["deepseek41.attention.compress_ratios"] = new uint[] { 2, 2, 1, 1, 1, 1 };
            Header(file, "token_embd.weight", 16, 8);
            Header(file, "blk.0.attn_compressor_kv.weight", 16, 16);
            Header(file, "blk.2.attn_compressor_kv.weight", 16, 16);
            Header(file, "blk.0.indexer.attn_q_b.weight", 16, 16);
            Header(file, "blk.2.indexer.attn_q_b.weight", 16, 16);
            Header(file, "blk.4.indexer.attn_q_b.weight", 16, 16);
            return file;
        }
        var multipliers = new ulong[] { 7, 9, 11, 9, 11, 13 };
        var primes = new uint[] { 11, 13, 15, 17, 15, 17, 19, 21 };

        var tensorSharp = Common();
        var t = tensorSharp.Metadata;
        t["deepseek41.engram.pad_id"] = 0u;
        t["deepseek41.engram.max_ngram_size"] = 3u;
        t["deepseek41.engram.head_count"] = 2u;
        t["deepseek41.engram.key_length"] = 16u;
        t["deepseek41.engram.layer_ids"] = new uint[] { 1, 5 };
        t["deepseek41.engram.multipliers"] = multipliers;
        t["deepseek41.engram.primes"] = primes;
        t["deepseek41.engram.offsets"] = new ulong[] { 0, 11, 24, 39, 0, 15, 32, 51 };
        Header(tensorSharp, "blk.1.engram_embd.weight", 16, 56);
        Header(tensorSharp, "blk.5.engram_embd.weight", 16, 72);

        var combined = Common();
        var c = combined.Metadata;
        c["deepseek41.engram.pad_token_id"] = 3u;
        c["deepseek41.engram.compressed_vocab_size"] = 4u;
        c["deepseek41.ple.ngram_size"] = 3u;
        c["deepseek41.ple.heads_per_ngram"] = 2u;
        c["deepseek41.embedding_length_per_layer_input"] = 16u;
        c["deepseek41.ple.layers"] = new uint[] { 1, 5 };
        c["deepseek41.ple.layer_multipliers"] = multipliers;
        c["deepseek41.ple.head_vocab_sizes"] = primes;
        c["deepseek41.ple.head_offsets"] = new ulong[] { 3, 14, 27, 42, 59, 74, 91, 110 };
        Header(combined, Dsv41EngramData.CombinedTableName, 16, 131);
        return (tensorSharp, combined);
    }

    [Fact]
    public void BothLayoutsOfOneGeometryLoadAndHashIdentically()
    {
        var (tensorSharpFile, combinedFile) = OneGeometryInBothLayouts();
        using (tensorSharpFile)
        using (combinedFile)
        {
            var a = Dsv41EngramData.Load(tensorSharpFile);
            var b = Dsv41EngramData.Load(combinedFile);
            Assert.Null(a.Slices);
            Assert.Empty(a.TableViews());
            Assert.Equal(new[] { new Dsv41EngramData.CombinedSlice(1, 3, 56), new Dsv41EngramData.CombinedSlice(5, 59, 72) },
                b.Slices);

            Assert.Equal(a.VocabSize, b.VocabSize);
            Assert.Equal(a.CompressedVocabSize, b.CompressedVocabSize);
            Assert.Equal(0u, b.PadId);
            Assert.Equal(a.PadId, b.PadId);
            Assert.Equal(a.MaxNgramSize, b.MaxNgramSize);
            Assert.Equal(a.HeadCount, b.HeadCount);
            Assert.Equal(a.HeadDim, b.HeadDim);
            Assert.Equal(a.TokenizerHash, b.TokenizerHash);
            Assert.Equal(a.TokenMap, b.TokenMap);
            Assert.Equal(a.KvSourceLayerIds, b.KvSourceLayerIds);
            Assert.Equal(a.IndexSourceLayerIds, b.IndexSourceLayerIds);
            Assert.Equal(a.CandidateSourceLayerId, b.CandidateSourceLayerId);
            Assert.Equal(a.CandidateTopkBlocks, b.CandidateTopkBlocks);
            Assert.Equal(a.CandidateBlockSize, b.CandidateBlockSize);
            Assert.Equal(a.Layers.Length, b.Layers.Length);
            for (int i = 0; i < a.Layers.Length; i++)
            {
                Assert.Equal(a.Layers[i].Id, b.Layers[i].Id);
                Assert.Equal(a.Layers[i].Rows, b.Layers[i].Rows);
                Assert.Equal(a.Layers[i].Multipliers, b.Layers[i].Multipliers);
                Assert.Equal(a.Layers[i].Primes, b.Layers[i].Primes);
                Assert.Equal(a.Layers[i].Offsets, b.Layers[i].Offsets);
            }

            // Random sequences, image positions included, each hashed in two calls so the
            // second one's lookbacks reach into the first one's history.
            var rng = new Random(4109);
            for (int s = 0; s < 1000; s++)
            {
                int[] tokens = Enumerable.Range(0, rng.Next(1, 17))
                    .Select(_ => rng.Next(10) == 0 ? -1 : rng.Next(8)).ToArray();
                int split = rng.Next(tokens.Length + 1);
                int[] ha = null, hb = null; int la = 0, lb = 0;
                Assert.Equal(a.HashTokens(tokens.AsSpan(0, split), 0, ref ha, ref la),
                             b.HashTokens(tokens.AsSpan(0, split), 0, ref hb, ref lb));
                Assert.Equal(a.HashTokens(tokens.AsSpan(split), split, ref ha, ref la),
                             b.HashTokens(tokens.AsSpan(split), split, ref hb, ref lb));
            }
        }
    }

    /// <summary>
    /// The views the direct-CUDA executor reads are the combined table's own bytes: row r of a
    /// layer's view is row FirstRow + r of the table, in the shard that holds it, through every
    /// route the executor reads a table by (the shard's own mapping, then the weight descriptor
    /// the engine streams from and the streaming source's mapping). Two shards, as the 1017 ships
    /// them: the metadata in the first, the combined table (Q5_K, 176 bytes per 256-value row)
    /// behind another tensor in the last.
    /// </summary>
    [Fact]
    public void RowViewsAreTheCombinedTablesOwnBytesInTheShardThatHoldsIt()
    {
        const int Q5K = 13, RowBytes = 176, TableRows = 131;
        string dir = Path.Combine(Path.GetTempPath(), $"dsv41-views-{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        try
        {
            string first = Path.Combine(dir, "v41-00001-of-00002.gguf");
            string second = Path.Combine(dir, "v41-00002-of-00002.gguf");
            var kv = new List<SyntheticGguf.Kv>
            {
                new SyntheticGguf.Str { Key = "general.architecture", V = "deepseek41" },
                new SyntheticGguf.U32 { Key = "split.count", V = 2 },
                new SyntheticGguf.U32 { Key = "deepseek41.block_count", V = 6 },
                new SyntheticGguf.U32Arr { Key = "deepseek41.engram.token_map", V = new uint[] { 0, 1, 2, 0, 3, 1, 2, 3 } },
                new SyntheticGguf.U32Arr { Key = "deepseek41.attention.compress_ratios", V = new uint[] { 2, 2, 1, 1, 1, 1 } },
                new SyntheticGguf.U32 { Key = "deepseek41.engram.pad_token_id", V = 3 },
                new SyntheticGguf.U32 { Key = "deepseek41.ple.ngram_size", V = 3 },
                new SyntheticGguf.U32 { Key = "deepseek41.ple.heads_per_ngram", V = 2 },
                new SyntheticGguf.U32 { Key = "deepseek41.embedding_length_per_layer_input", V = 256 },
                new SyntheticGguf.U32Arr { Key = "deepseek41.ple.layers", V = new uint[] { 1, 5 } },
                new SyntheticGguf.U64Arr { Key = "deepseek41.ple.layer_multipliers", V = new ulong[] { 7, 9, 11, 9, 11, 13 } },
                new SyntheticGguf.U32Arr { Key = "deepseek41.ple.head_vocab_sizes", V = new uint[] { 11, 13, 15, 17, 15, 17, 19, 21 } },
                new SyntheticGguf.U64Arr { Key = "deepseek41.ple.head_offsets", V = new ulong[] { 3, 14, 27, 42, 59, 74, 91, 110 } },
            };
            SyntheticGguf.AddByteTokenizer(kv, 8, 0, 1);
            var small = new List<SyntheticGguf.Tensor> { SyntheticGguf.Gen("token_embd.weight", 0.1f, 16, 8) };
            foreach (string name in new[] { "blk.0.attn_compressor_kv.weight", "blk.2.attn_compressor_kv.weight",
                         "blk.0.indexer.attn_q_b.weight", "blk.2.indexer.attn_q_b.weight", "blk.4.indexer.attn_q_b.weight" })
                small.Add(SyntheticGguf.Gen(name, 0.1f, 16, 16));
            SyntheticGguf.Write(first, kv, small);
            // Every row's bytes differ from every other row's, so a view one row off reads wrong bytes.
            var table = new byte[TableRows * RowBytes];
            for (int i = 0; i < table.Length; i++)
                table[i] = (byte)(i / RowBytes * 37 + i % RowBytes * 11 + 5);
            SyntheticGguf.Write(second, new List<SyntheticGguf.Kv> { new SyntheticGguf.U32 { Key = "split.count", V = 2 } },
                new List<SyntheticGguf.Tensor>
                {
                    SyntheticGguf.Gen("blk.5.ffn_norm.weight", 0.1f, 40),
                    new() { Name = Dsv41EngramData.CombinedTableName, Dims = new ulong[] { 256, TableRows },
                            Type = (SyntheticGguf.GgmlType)Q5K, RawBlocks = table },
                });

            long tableStart;
            ulong tableOffset;
            using (var shard = GgufFile.OpenWithoutSiblingShards(second))
            {
                tableOffset = shard.Tensors[Dsv41EngramData.CombinedTableName].Offset;
                tableStart = shard.DataOffset + (long)tableOffset;
            }
            Assert.NotEqual(0UL, tableOffset);
            byte[] onDisk = File.ReadAllBytes(second);

            using var set = new GgufShardSet(first, "test");
            var data = Dsv41EngramData.Load(set.First, set.InfoOf);
            var views = data.TableViews();
            Assert.Equal(new[] { ("blk.1.engram_embd.weight", 3L, 56L), ("blk.5.engram_embd.weight", 59L, 72L) }, views);
            set.ReplaceWithRowViews(Dsv41EngramData.CombinedTableName, views);
            Assert.False(set.Has(Dsv41EngramData.CombinedTableName));

            var rng = new Random(176);
            void SameRows(IntPtr mapped, long firstRow, long rows)
            {
                var row = new byte[RowBytes];
                foreach (long r in new[] { 0, rows - 1 }.Concat(Enumerable.Range(0, 24).Select(_ => (long)rng.Next((int)rows))))
                {
                    Marshal.Copy(mapped + (nint)(r * RowBytes), row, 0, RowBytes);
                    Assert.True(row.AsSpan().SequenceEqual(onDisk.AsSpan((int)(tableStart + (firstRow + r) * RowBytes), RowBytes)),
                        $"row {r} of the view at {firstRow}");
                }
            }

            foreach (var (name, firstRow, rows) in views)
            {
                GgufTensorInfo info = set.InfoOf(name);
                Assert.Equal(new ulong[] { 256, (ulong)rows }, info.Shape);
                Assert.Equal((GgmlTensorType)Q5K, info.Type);
                Assert.Equal(tableOffset + (ulong)(firstRow * RowBytes), info.Offset);
                // Named after the table, so a sibling-aggregating GgufFile would still find its shard.
                Assert.Equal(Dsv41EngramData.CombinedTableName, info.Name);
                var (path, offset, bytes, mapped) = set.MapTensor(name);
                Assert.Equal(second, path);
                Assert.Equal(tableStart + firstRow * RowBytes, offset);
                Assert.Equal(rows * RowBytes, bytes);
                SameRows(mapped, firstRow, rows);
            }

            set.BeginStreaming();
            foreach (var (name, firstRow, rows) in views)
            {
                var desc = set.Weight(name);
                Assert.NotNull(desc.Source);
                Assert.Equal(tableStart + firstRow * RowBytes, desc.SourceOffset);
                Assert.Equal(RowBytes, desc.RowBytes);
                Assert.Equal(256, desc.Ne0);
                Assert.Equal((int)rows, desc.Ne1);
                var (_, offset, _, mapped) = set.MapTensor(name);
                Assert.Equal(tableStart + firstRow * RowBytes, offset);
                SameRows(mapped, firstRow, rows);
            }

            // The combined entry is gone, so it cannot be viewed (or streamed) a second time.
            Assert.Throws<InvalidOperationException>(() =>
                set.ReplaceWithRowViews(Dsv41EngramData.CombinedTableName, views));
        }
        finally
        {
            try { Directory.Delete(dir, recursive: true); } catch (IOException) { } catch (UnauthorizedAccessException) { }
        }
    }

    [Fact]
    public void RowViewsRefuseRowsOutsideTheTableAndNamesInUse()
    {
        var tensors = new Dictionary<string, (GgufFile File, GgufTensorInfo Info)>(StringComparer.Ordinal);
        using var file = EmptyFile();
        var table = new GgufTensorInfo { Name = "t", Shape = new ulong[] { 32, 10 }, Type = GgmlTensorType.F32, Offset = 64 };
        tensors["t"] = (file, table);
        tensors["used"] = (file, new GgufTensorInfo { Name = "used", Shape = new ulong[] { 32 } });
        Assert.Throws<InvalidDataException>(() => GgufShardSet.ReplaceWithRowViews(tensors, "t", new[] { ("a", 0L, 11L) }, "test"));
        Assert.Throws<InvalidDataException>(() => GgufShardSet.ReplaceWithRowViews(tensors, "t", new[] { ("a", 10L, 1L) }, "test"));
        Assert.Throws<InvalidDataException>(() => GgufShardSet.ReplaceWithRowViews(tensors, "t", new[] { ("a", -1L, 2L) }, "test"));
        Assert.Throws<InvalidDataException>(() => GgufShardSet.ReplaceWithRowViews(tensors, "t", new[] { ("used", 0L, 2L) }, "test"));
        Assert.Throws<InvalidDataException>(() => GgufShardSet.ReplaceWithRowViews(tensors, "t", new[] { ("a", 0L, 2L), ("a", 2L, 2L) }, "test"));
        Assert.Equal(2, tensors.Count);   // nothing half-applied
        GgufShardSet.ReplaceWithRowViews(tensors, "t", new[] { ("a", 0L, 4L), ("b", 4L, 6L) }, "test");
        Assert.False(tensors.ContainsKey("t"));
        Assert.Equal(64UL + 4 * 32 * 4, tensors["b"].Info.Offset);
        Assert.Equal(new ulong[] { 32, 6 }, tensors["b"].Info.Shape);
    }

    /// <summary>
    /// The real header of unsloth's DeepSeek-V4.1-Flash-1017 UD-IQ2_XXS (shard 1 of 6, metadata
    /// only, 5.78 MB) in llama.cpp's combined layout. The tensors live in shards 2-6, so the two
    /// tables the loader sizes against stand in with their published shapes, and the cache-source
    /// tensors with the layer lists the header records. Its hash geometry is the published
    /// TensorSharp checkpoint's, so the same NumPy oracle holds.
    /// Set TS_TEST_DSV41_1017_HEADER to that shard to enable it.
    /// </summary>
    [ModelFact(Header1017Var)]
    public void Load_ReadsTheReal1017HeaderInLlamaCppsCombinedLayout()
    {
        using var file = GgufFile.OpenWithoutSiblingShards(Environment.GetEnvironmentVariable(Header1017Var));
        Assert.Empty(file.Tensors);
        static int[] Ints(object raw) => ((Array)raw).Cast<object>().Select(Convert.ToInt32).ToArray();
        int[] kvSources = Ints(file.Metadata["deepseek41.attention.kv_source_layers"]);
        int[] indexSources = Ints(file.Metadata["deepseek41.attention.indexer.source_layers"]);
        GgufTensorInfo Find(string name)
        {
            ulong[] shape =
                name == "token_embd.weight" ? new ulong[] { 5120, 129280 } :
                name == Dsv41EngramData.CombinedTableName ? new ulong[] { 256, 768_022_850 } :
                kvSources.Any(il => name == $"blk.{il}.attn_compressor_kv.weight") ||
                indexSources.Any(il => name == $"blk.{il}.indexer.attn_q_b.weight") ? new ulong[] { 1, 1 } : null;
            return shape == null ? null : new GgufTensorInfo { Name = name, Shape = shape, Type = GgmlTensorType.Q5_K };
        }

        Assert.True(Dsv41EngramData.HasCombinedMetadata(file));
        var data = Dsv41EngramData.Load(file, Find);
        Assert.Equal(new[] { 1, 14 }, data.Layers.Select(l => l.Id));
        Assert.Equal(new[]
        {
            new Dsv41EngramData.CombinedSlice(1, 0, 384_006_168),
            new Dsv41EngramData.CombinedSlice(14, 384_006_168, 384_016_682),
        }, data.Slices);
        // The two slices are the whole table, and the table is the whole of shard 6.
        Assert.Equal(768_022_850L, data.Slices[1].FirstRow + data.Slices[1].Rows);
        Assert.Equal(135_172_021_600L, (data.Slices[1].FirstRow + data.Slices[1].Rows) * 176);
        // token_map[engram.pad_token_id 2]; not tokenizer.ggml.padding_token_id, which is 1.
        Assert.Equal(2u, data.PadId);
        Assert.Equal(99092u, data.CompressedVocabSize);
        Assert.Equal(129280u, data.VocabSize);
        Assert.Equal(4u, data.MaxNgramSize);
        Assert.Equal(8u, data.HeadCount);
        Assert.Equal(256u, data.HeadDim);
        Assert.Equal(24u, data.HashColumns);
        foreach (var (layer, slice) in data.Layers.Zip(data.Slices))
        {
            Assert.Equal(slice.Rows, layer.Rows);
            Assert.Equal(0L, layer.Offsets[0]);
            Assert.Equal(layer.Rows, layer.Offsets[^1] + layer.Primes[^1]);
        }
        Assert.Equal(new[] { 2, 8, 14, 20 }, data.KvSourceLayerIds);
        Assert.Equal(new[] { 2, 8, 14, 20, 24, 28, 32, 36 }, data.IndexSourceLayerIds);
        Assert.Equal(20, data.CandidateSourceLayerId);
        Assert.Equal(8u, data.CandidateBlockSize);
        // Read from candidate_top_k_blocks, the 1017's own name for it.
        Assert.False(file.Metadata.ContainsKey("deepseek41.attention.indexer.candidate_top_k"));
        Assert.Equal(2048u, data.CandidateTopkBlocks);
        AssertMatchesTheNumPyOracle(data);
    }

    internal const string Header1017Var = "TS_TEST_DSV41_1017_HEADER";

    [Fact]
    public void FingerprintToken_FoldsLengthThenBytes()
    {
        // Independently: FNV-1a over the 8 little-endian length bytes, then the
        // UTF-8 bytes, starting from the supplied seed.
        const ulong seed = 1469598103934665603UL;
        ulong expected = seed;
        byte[] utf8 = System.Text.Encoding.UTF8.GetBytes("ab");
        for (int i = 0; i < 8; i++)
            expected = (expected ^ (byte)((ulong)utf8.Length >> (8 * i))) * 1099511628211UL;
        foreach (byte b in utf8)
            expected = (expected ^ b) * 1099511628211UL;

        Assert.Equal(expected, Dsv41EngramData.FingerprintToken(seed, "ab"));
    }
}

public sealed class Dsv41RealGgufFactAttribute : FactAttribute
{
    public Dsv41RealGgufFactAttribute()
    {
        string path = Environment.GetEnvironmentVariable("TS_TEST_DSV41_REAL_GGUF");
        if (string.IsNullOrWhiteSpace(path) || !File.Exists(path))
            Skip = "Requires TS_TEST_DSV41_REAL_GGUF pointing to the published DeepSeek V4.1 Flash first GGUF shard.";
    }
}
