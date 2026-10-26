// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Text;
using TensorSharp.Models;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

public sealed class DeepSeek41ByteFixtureTokenizerTests
{
    [Theory]
    [InlineData("Reply with only 42.")]
    [InlineData("你好，世界 🚀")]
    [InlineData("line one\nline two\tvalue")]
    public void HttpFixtureByteVocabularyEncodesEveryInputByteWithoutLoadingWeights(string text)
    {
        string path = Path.Combine(Path.GetTempPath(), "dsv41-byte-tokenizer-" + Guid.NewGuid().ToString("N") + ".gguf");
        try
        {
            var metadata = new List<SyntheticGguf.Kv>
            {
                new SyntheticGguf.Str { Key = "general.architecture", V = "deepseek41" }
            };
            SyntheticGguf.AddByteTokenizer(metadata, 256, 0, 1);
            metadata.OfType<SyntheticGguf.Str>().Single(item => item.Key == "tokenizer.ggml.pre").V = "joyai-llm";
            SyntheticGguf.Write(path, metadata, new List<SyntheticGguf.Tensor>());
            using var gguf = GgufFile.OpenWithoutSiblingShards(path);
            var tokenizer = ModelBase.CreateTokenizerFromGguf(gguf);
            var ids = tokenizer.Encode(text, addSpecial: false);
            Assert.NotEmpty(ids);
            Assert.Equal(Encoding.UTF8.GetBytes(text).Select(value => (int)value), ids);
            Assert.Equal(text, tokenizer.Decode(ids));
        }
        finally { File.Delete(path); }
    }
}
