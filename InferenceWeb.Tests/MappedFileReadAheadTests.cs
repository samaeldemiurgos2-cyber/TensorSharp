using TensorSharp.Models.Video;

namespace InferenceWeb.Tests;

[Collection("EnvironmentVariables")]
public sealed class MappedFileReadAheadTests
{
    private static string WriteFile(long bytes)
    {
        string path = Path.GetTempFileName();
        using var stream = File.OpenWrite(path);
        var block = new byte[1 << 20];
        for (int i = 0; i < block.Length; ++i) block[i] = (byte)(i * 31 + 7);
        for (long written = 0; written < bytes; written += block.Length)
            stream.Write(block, 0, (int)Math.Min(block.Length, bytes - written));
        return path;
    }

    [Theory]
    [InlineData(1)] [InlineData(4)]
    public void ReadsEveryPlannedByteOnceAcrossStagesAndRaggedTails(int workers)
    {
        string path = WriteFile((20L << 20) + 333);
        try
        {
            // Out-of-order ranges with ragged lengths in three stages; the window is
            // larger than the plan, so every chunk is read without the consumer.
            var stages = new[]
            {
                new[] { (17L << 20, (3L << 20) + 333) },
                new[] { (0L, (9L << 20) + 5), (12L << 20, 1L << 20) },
                new[] { ((9L << 20) + 5, 77L) },
            };
            long planned = stages.SelectMany(stage => stage).Sum(range => range.Item2);
            using var read = MappedFileReadAhead.TryStart(path, stages, windowBytes: long.MaxValue, workers: workers, honorSwitch: false);
            Assert.NotNull(read);
            read.Wait();
            Assert.True(read.Completed);
            Assert.Null(read.Failure);
            Assert.Equal(planned, read.PlannedBytes);
            Assert.Equal(planned, read.ReadBytes + read.ResidentBytes);
            Assert.Equal(workers, read.Workers);
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void StageWindowHoldsLaterChunksUntilTheConsumerEnters()
    {
        string path = WriteFile(32L << 20);
        try
        {
            var stages = Enumerable.Range(0, 4).Select(i => new[] { ((long)i << 23, 8L << 20) }).ToArray();
            // An 8 MiB window admits stage 0 and stage 1 (which starts exactly 8 MiB
            // past stage 0), never stage 2 while the consumer is still in stage 0.
            using var read = MappedFileReadAhead.TryStart(path, stages, windowBytes: 8L << 20, workers: 2, honorSwitch: false);
            Assert.NotNull(read);
            for (int attempt = 0; attempt < 200 && read.ReadBytes + read.ResidentBytes < 16L << 20; ++attempt) Thread.Sleep(10);
            Thread.Sleep(50);
            Assert.Equal(16L << 20, read.ReadBytes + read.ResidentBytes);
            Assert.False(read.Completed);
            read.Enter(2);
            read.Wait();
            Assert.True(read.Completed);
            Assert.Equal(32L << 20, read.ReadBytes + read.ResidentBytes);
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void StopEndsAWaitingReaderAndReleaseRequiresStop()
    {
        string path = WriteFile(16L << 20);
        try
        {
            var stages = new[] { new[] { (0L, 8L << 20) }, new[] { (8L << 20, 8L << 20) } };
            var read = MappedFileReadAhead.TryStart(path, stages, windowBytes: 1, workers: 1, honorSwitch: false);
            Assert.NotNull(read);
            Assert.Throws<InvalidOperationException>(() => read.ReleaseCache());
            read.Stop();
            Assert.False(read.Completed);
            long released = read.ReleaseCache();
            if (OperatingSystem.IsMacOS() || OperatingSystem.IsMacCatalyst() || OperatingSystem.IsLinux()) Assert.Equal(16L << 20, released);
            else Assert.Equal(0, released);
            read.Dispose();
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void ResidentChunksAreSkippedOnASecondPass()
    {
        if (!(OperatingSystem.IsMacOS() || OperatingSystem.IsLinux())) return; // mincore only
        string path = WriteFile(16L << 20);
        try
        {
            using (var first = MappedFileReadAhead.TryStart(path, MappedFileReadAhead.WholeFile(path), windowBytes: long.MaxValue, workers: 2, honorSwitch: false))
                first!.Wait();
            using var second = MappedFileReadAhead.TryStart(path, MappedFileReadAhead.WholeFile(path), windowBytes: long.MaxValue, workers: 2, honorSwitch: false);
            second!.Wait();
            Assert.Equal(16L << 20, second.ReadBytes + second.ResidentBytes);
            Assert.True(second.ResidentBytes > 0, "a just-read file should be at least partly cached");
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void SwitchDisablesItAndAMissingFileIsReportedOnce()
    {
        string previous = Environment.GetEnvironmentVariable("TS_VIDEO_READAHEAD");
        string path = WriteFile(1L << 20);
        try
        {
            Environment.SetEnvironmentVariable("TS_VIDEO_READAHEAD", "0");
            Assert.Null(MappedFileReadAhead.TryStart(path, MappedFileReadAhead.WholeFile(path)));
            Environment.SetEnvironmentVariable("TS_VIDEO_READAHEAD", null);
            var warnings = new List<string>();
            Assert.Null(MappedFileReadAhead.TryStart(path + ".missing", new[] { new[] { (0L, 1L) } }, warnings.Add));
            Assert.Single(warnings);
            Assert.Contains("page faults", warnings[0]);
        }
        finally { Environment.SetEnvironmentVariable("TS_VIDEO_READAHEAD", previous); File.Delete(path); }
    }
}
