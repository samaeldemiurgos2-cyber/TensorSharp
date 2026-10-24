// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Runtime.InteropServices;
using System.Text;

namespace InferenceWeb.Tests;

public sealed class GgufSplitShardTests : IDisposable
{
    private readonly string _directory = Path.Combine(Path.GetTempPath(), "ts-split-" + Guid.NewGuid().ToString("N"));

    public GgufSplitShardTests() => Directory.CreateDirectory(_directory);
    public void Dispose() => Directory.Delete(_directory, true);

    [Theory]
    [InlineData(1, false)]
    [InlineData(2, false)]
    [InlineData(3, false)]
    [InlineData(1, true)]
    [InlineData(2, true)]
    [InlineData(3, true)]
    public void EveryEntryShardUsesCanonicalMetadataAndEachTensorsPhysicalBytes(int entry, bool firstHasWeights)
    {
        var fixture = WriteSplit(firstHasWeights);
        using var file = new GgufFile(fixture.Paths[entry - 1]);
        file.ThrowIfTruncated();

        Assert.True(file.IsSplit);
        Assert.Equal(fixture.Paths, file.FilePaths);
        Assert.Equal("muse-spark", file.GetString("general.architecture"));
        Assert.Equal(4u, file.GetUint32("muse-spark.embedding_length"));
        Assert.Equal("{{ messages[0]['content'] }}", file.GetString("tokenizer.chat_template"));
        Assert.Equal(0u, file.GetUint32("split.no"));
        Assert.Equal(256, file.GetStringArray("tokenizer.ggml.tokens")!.Length);
        var tokenizer = ModelBase.CreateTokenizerFromGguf(file);
        const string text = "Shard 2: caf\u00e9\n";
        var tokens = tokenizer.Encode(text, addSpecial: false);
        Assert.Equal(Encoding.UTF8.GetBytes(text).Select(b => (int)b), tokens);
        Assert.Equal(text, tokenizer.Decode(tokens));

        using var first = GgufFile.OpenWithoutSiblingShards(fixture.Paths[0]);
        Assert.Equal(first.DataOffset, file.DataOffset);
        Assert.Equal(first.GetRequiredLength(out _), file.GetRequiredLength(out _));
        if (!firstHasWeights)
        {
            // Published Muse Spark shard 1 can end before the padding assumed
            // by DataOffset; merely opening through shard 3 must not demand it.
            Assert.Empty(first.Tensors);
            Assert.True(new FileInfo(fixture.Paths[0]).Length < first.DataOffset);
        }

        Assert.Equal(fixture.Tensors.Sum(tensors => tensors.Count), file.Tensors.Count);
        var physicalOffsets = new HashSet<long>();
        for (int shard = 0; shard < fixture.Paths.Length; shard++)
        {
            using var physical = GgufFile.OpenWithoutSiblingShards(fixture.Paths[shard]);
            physicalOffsets.Add(physical.DataOffset);
            foreach (var tensor in fixture.Tensors[shard])
            {
                var info = file.Tensors[tensor.Name];
                Assert.Equal(physical.Tensors[tensor.Name].Offset, info.Offset);
                Assert.Equal(tensor.Raw(), file.ReadTensorData(info));
                var floats = new float[tensor.Data.Length];
                file.ReadTensorDataToFloat32(info, floats, floats.Length);
                Assert.Equal(tensor.Data, floats);
                Assert.True(file.TryGetTensorDataPointer(info, out IntPtr pointer));
                var mapped = new byte[tensor.Raw().Length];
                Marshal.Copy(pointer, mapped, 0, mapped.Length);
                Assert.Equal(tensor.Raw(), mapped);
            }
        }
        // Distinct headers make using the root offset for a sibling fail even
        // though each shard's first tensor has the same relative offset zero.
        Assert.Equal(3, physicalOffsets.Count);
        Assert.All(fixture.Tensors.Skip(1), tensors => Assert.Equal(2, tensors.Count));
        Assert.Equal(32ul, file.Tensors[fixture.Tensors[2][1].Name].Offset);
    }

    [Theory]
    [InlineData(2)]
    [InlineData(3)]
    public void StandaloneOpenOfLaterShardRemainsPhysicalOnly(int entry)
    {
        var fixture = WriteSplit(firstHasWeights: false);
        string path = fixture.Paths[entry - 1];
        using var file = GgufFile.OpenWithoutSiblingShards(path);
        Assert.False(file.IsSplit);
        Assert.Equal(path, Assert.Single(file.FilePaths));
        Assert.Null(file.GetString("general.architecture"));
        Assert.Null(file.GetStringArray("tokenizer.ggml.tokens"));
        Assert.Equal((uint)(entry - 1), file.GetUint32("split.no"));
        Assert.Equal(2, file.Tensors.Count);
        foreach (var tensor in fixture.Tensors[entry - 1])
            Assert.Equal(tensor.Raw(), file.ReadTensorData(file.Tensors[tensor.Name]));
    }

    [Fact]
    public void LaterShardReportsMissingMetadataShardAsIncompleteCheckpoint()
    {
        var fixture = WriteSplit(firstHasWeights: false);
        File.Delete(fixture.Paths[0]);
        var exception = Assert.Throws<FileNotFoundException>(() => new GgufFile(fixture.Paths[2]));
        Assert.Equal(fixture.Paths[0], exception.FileName);
        Assert.Contains(Path.GetFileName(fixture.Paths[0]), exception.Message);
        Assert.Contains("missing", exception.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Theory]
    [InlineData(1, 3)]
    [InlineData(3, 2)]
    public void AnyEntryShardReportsMissingWeightSibling(int entry, int missing)
    {
        var fixture = WriteSplit(firstHasWeights: true);
        File.Delete(fixture.Paths[missing - 1]);
        var exception = Assert.Throws<FileNotFoundException>(() => new GgufFile(fixture.Paths[entry - 1]));
        Assert.Equal(fixture.Paths[missing - 1], exception.FileName);
        Assert.Contains(Path.GetFileName(fixture.Paths[missing - 1]), exception.Message);
        Assert.Contains("missing", exception.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void ShardLookingStandaloneFileDoesNotRedirectOrReadNeighbors()
    {
        string selected = Path.Combine(_directory, "standalone-00002-of-00003.gguf");
        string neighbor = Path.Combine(_directory, "standalone-00001-of-00003.gguf");
        File.WriteAllText(neighbor, "This is deliberately not GGUF.");
        var tensor = Tensor("standalone", 19);
        SyntheticGguf.Write(selected,
            [new SyntheticGguf.Str { Key = "general.architecture", V = "test-standalone" }], [tensor]);
        using var file = new GgufFile(selected);
        Assert.Equal(selected, Assert.Single(file.FilePaths));
        Assert.False(file.IsSplit);
        Assert.Equal("test-standalone", file.GetString("general.architecture"));
        Assert.Equal(tensor.Raw(), file.ReadTensorData(file.Tensors[tensor.Name]));
    }

    [Theory]
    [InlineData("invalid-00001-of-00003.gguf", 2)]
    [InlineData("invalid-00002-of-00003.gguf", 2)]
    [InlineData("invalid-00000-of-00003.gguf", 3)]
    [InlineData("invalid-00004-of-00003.gguf", 3)]
    public void InconsistentSplitFilenameFailsBeforeOpeningNeighbors(string name, int splitCount)
    {
        string path = Path.Combine(_directory, name);
        SyntheticGguf.Write(path, [new U16 { Key = "split.count", V = (ushort)splitCount }], []);
        Assert.Throws<InvalidDataException>(() => new GgufFile(path));
        // The raw inspection API must still allow examining the malformed
        // physical header without opening or resolving any other file.
        using var raw = GgufFile.OpenWithoutSiblingShards(path);
        Assert.Equal((uint)splitCount, raw.GetUint32("split.count"));
        Assert.Equal(path, Assert.Single(raw.FilePaths));
    }

    private (string[] Paths, List<SyntheticGguf.Tensor>[] Tensors) WriteSplit(bool firstHasWeights)
    {
        string[] paths = Enumerable.Range(1, 3)
            .Select(i => Path.Combine(_directory, $"checkpoint-{i:D5}-of-00003.gguf")).ToArray();
        var tensors = new List<SyntheticGguf.Tensor>[3];
        for (int shard = 0; shard < paths.Length; shard++)
        {
            var metadata = new List<SyntheticGguf.Kv>
            {
                new U16 { Key = "split.count", V = 3 },
                new U16 { Key = "split.no", V = (ushort)shard },
                new SyntheticGguf.Str { Key = "fixture.header_padding", V = new string('x', 37 + shard * 193) },
            };
            if (shard == 0)
            {
                metadata.Add(new SyntheticGguf.Str { Key = "general.architecture", V = "muse-spark" });
                metadata.Add(new SyntheticGguf.U32 { Key = "muse-spark.embedding_length", V = 4 });
                metadata.Add(new SyntheticGguf.Str { Key = "tokenizer.chat_template", V = "{{ messages[0]['content'] }}" });
                SyntheticGguf.AddByteTokenizer(metadata, 256, 0, 1);
            }
            tensors[shard] = shard == 0 && !firstHasWeights
                ? [] : [Tensor($"shard{shard}.first", shard * 10 + 2), Tensor($"shard{shard}.second", shard * 10 + 5)];
            SyntheticGguf.Write(paths[shard], metadata, tensors[shard]);
            if (shard == 0 && !firstHasWeights)
            {
                long tableEnd;
                bool aligned;
                using (var physical = GgufFile.OpenWithoutSiblingShards(paths[shard]))
                {
                    tableEnd = physical.GetRequiredLength(out _);
                    aligned = tableEnd == physical.DataOffset;
                }
                if (aligned)
                {
                    ((SyntheticGguf.Str)metadata[2]).V += "x";
                    SyntheticGguf.Write(paths[shard], metadata, []);
                    using var physical = GgufFile.OpenWithoutSiblingShards(paths[shard]);
                    tableEnd = physical.GetRequiredLength(out _);
                }
                using var stream = new FileStream(paths[shard], FileMode.Open, FileAccess.Write);
                stream.SetLength(tableEnd);
            }
        }
        return (paths, tensors);
    }

    private static SyntheticGguf.Tensor Tensor(string name, float value) => new()
    {
        Name = name,
        Dims = [4],
        Data = [value, -value, value + 0.25f, value + 0.5f],
    };

    private sealed class U16 : SyntheticGguf.Kv
    {
        public ushort V;
        public override void Write(BinaryWriter writer)
        {
            writer.Write((uint)GgufValueType.Uint16);
            writer.Write(V);
        }
    }
}
