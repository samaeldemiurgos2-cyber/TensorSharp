// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp;

namespace InferenceWeb.Tests;

/// <summary>
/// RefCounted's lock-free count under concurrency. Tensor-parallel loading narrows one
/// weight per rank inside Parallel.For, so views of one storage are taken and released
/// from several threads at once; MuseSparkTensorParallelQKPrepTests intermittently failed
/// with "object has already been destroyed" when a Release derived its new count from a
/// second read of the field, and two final releases could both destroy.
/// </summary>
public sealed class RefCountedConcurrencyTests
{
    private sealed class Counted : RefCounted
    {
        public int Destroyed;
        protected override void Destroy() => Interlocked.Increment(ref Destroyed);
    }

    private static void RunOnThreads(int count, Action body)
    {
        using var start = new ManualResetEventSlim(false);
        var threads = new Thread[count];
        Exception failure = null;
        for (int i = 0; i < count; i++)
        {
            threads[i] = new Thread(() =>
            {
                start.Wait();
                try { body(); }
                catch (Exception ex) { Interlocked.CompareExchange(ref failure, ex, null); }
            });
            threads[i].Start();
        }
        start.Set();
        foreach (Thread thread in threads) thread.Join();
        if (failure != null) throw failure;
    }

    [Fact]
    public void ConcurrentAddRefReleasePairsKeepTheCount()
    {
        for (int round = 0; round < 10; round++)
        {
            var counted = new Counted();
            RunOnThreads(Math.Max(4, Environment.ProcessorCount), () =>
            {
                for (int i = 0; i < 20000; i++) { counted.AddRef(); counted.Release(); }
            });
            Assert.Equal(0, counted.Destroyed);
            counted.Release();
            Assert.Equal(1, counted.Destroyed);
            Assert.Throws<InvalidOperationException>(counted.Release);
        }
    }

    [Fact]
    public void ConcurrentFinalReleasesDestroyExactlyOnce()
    {
        const int holders = 8;
        for (int round = 0; round < 200; round++)
        {
            var counted = new Counted();
            for (int i = 1; i < holders; i++) counted.AddRef();
            RunOnThreads(holders, counted.Release);
            Assert.Equal(1, counted.Destroyed);
        }
    }
}
