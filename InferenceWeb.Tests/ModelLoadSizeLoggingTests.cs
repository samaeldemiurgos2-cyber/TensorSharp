// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using Microsoft.Extensions.Logging;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Server;

namespace InferenceWeb.Tests;

public sealed class ModelLoadSizeLoggingTests : IDisposable
{
    private readonly string _directory = Path.Combine(Path.GetTempPath(), "ts-load-size-" + Guid.NewGuid().ToString("N"));

    public ModelLoadSizeLoggingTests() => Directory.CreateDirectory(_directory);
    public void Dispose() => Directory.Delete(_directory, true);

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    public void SplitCheckpointLogCountsEveryDeclaredShardOnce(int entryShard)
    {
        string first = Path.Combine(_directory, "weights-00001-of-00002.gguf");
        string second = Path.Combine(_directory, "weights-00002-of-00002.gguf");
        var metadata = new List<SyntheticGguf.Kv>
        {
            new SyntheticGguf.U32 { Key = "split.count", V = 2 },
        };
        SyntheticGguf.Write(first, metadata, []);
        metadata.Add(new SyntheticGguf.Str { Key = "test.padding", V = new string('x', 100) });
        SyntheticGguf.Write(second, metadata, []);
        var logger = new SizeLogger();
        using var lifecycle = new ModelLifecycleService(logger, (path, _, _, _) => new EmptyModel(path));
        lifecycle.LoadModel(entryShard == 1 ? first : second, null, "cpu");
        Assert.Equal(new FileInfo(first).Length + new FileInfo(second).Length, Assert.Single(logger.ModelSizes));
        Assert.Equal(2, lifecycle.Model.ModelFilePaths.Count);
    }

    [Fact]
    public void ShardLikeFilenameWithoutSplitMetadataDoesNotCountNeighbors()
    {
        string first = Path.Combine(_directory, "weights-00001-of-00002.gguf");
        SyntheticGguf.Write(first, [], []);
        File.WriteAllBytes(Path.Combine(_directory, "weights-00002-of-00002.gguf"), new byte[4096]);
        var logger = new SizeLogger();
        using var lifecycle = new ModelLifecycleService(logger, (path, _, _, _) => new EmptyModel(path));
        lifecycle.LoadModel(first, null, "cpu");
        Assert.Equal(new FileInfo(first).Length, Assert.Single(logger.ModelSizes));
    }

    private sealed class EmptyModel(string path) : ModelBase(path, BackendType.Cpu)
    {
        protected override float[] ForwardCore(int[] tokens) => [];
        protected override void ResetKVCacheCore() { }
    }

    private sealed class SizeLogger : ILogger
    {
        public List<long> ModelSizes { get; } = [];
        public IDisposable BeginScope<TState>(TState state) where TState : notnull => null;
        public bool IsEnabled(LogLevel level) => true;
        public void Log<TState>(LogLevel level, EventId eventId, TState state, Exception exception,
            Func<TState, Exception, string> formatter)
        {
            if (state is IEnumerable<KeyValuePair<string, object>> fields)
                foreach (var field in fields)
                    if (field.Key == "ModelBytes") ModelSizes.Add((long)field.Value);
        }
    }
}
