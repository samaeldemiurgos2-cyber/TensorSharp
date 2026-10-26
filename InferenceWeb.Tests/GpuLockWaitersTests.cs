// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;
using TensorSharp.Runtime.Scheduling;

namespace InferenceWeb.Tests;

public sealed class GpuLockWaitersTests
{
    [Fact]
    public void CountsWaitersPerLock()
    {
        object gpuLock = new(), other = new();
        Assert.Equal(0, GpuLockWaiters.Count(gpuLock));
        GpuLockWaiters.Enter(gpuLock);
        GpuLockWaiters.Enter(gpuLock);
        Assert.Equal(2, GpuLockWaiters.Count(gpuLock));
        Assert.Equal(0, GpuLockWaiters.Count(other));
        GpuLockWaiters.Exit(gpuLock);
        GpuLockWaiters.Exit(gpuLock);
        Assert.Equal(0, GpuLockWaiters.Count(gpuLock));
    }

    [Fact]
    public void YieldReturnsAtOnceWithoutWaiters()
    {
        var watch = Stopwatch.StartNew();
        GpuLockWaiters.YieldToWaiters(new object(), TimeSpan.FromSeconds(5));
        Assert.True(watch.Elapsed < TimeSpan.FromSeconds(1));
    }

    [Fact]
    public void WaitingThreadTakesTheLockBeforeTheWorkersNextStep()
    {
        object gpuLock = new();
        var order = new System.Collections.Concurrent.ConcurrentQueue<string>();
        using var released = new ManualResetEventSlim();
        Task waiter;
        lock (gpuLock)
        {
            // A request thread starts waiting while the worker is inside a step.
            waiter = Task.Run(() =>
            {
                GpuLockWaiters.Enter(gpuLock);
                lock (gpuLock)
                {
                    GpuLockWaiters.Exit(gpuLock);
                    order.Enqueue("request");
                    Thread.Sleep(20);
                }
            });
            SpinWait.SpinUntil(() => GpuLockWaiters.Count(gpuLock) == 1, TimeSpan.FromSeconds(5));
            order.Enqueue("step 1");
        }
        // The worker's next step yields to the registered waiter first.
        GpuLockWaiters.YieldToWaiters(gpuLock, TimeSpan.FromSeconds(5));
        lock (gpuLock)
            order.Enqueue("step 2");
        Assert.True(waiter.Wait(TimeSpan.FromSeconds(5)));
        Assert.Equal(new[] { "step 1", "request", "step 2" }, order.ToArray());
    }

    [Fact]
    public void YieldIsBoundedWhenAWaiterNeverTakesTheLock()
    {
        object gpuLock = new();
        GpuLockWaiters.Enter(gpuLock);
        try
        {
            var watch = Stopwatch.StartNew();
            GpuLockWaiters.YieldToWaiters(gpuLock, TimeSpan.FromMilliseconds(50));
            Assert.True(watch.Elapsed >= TimeSpan.FromMilliseconds(45));
            Assert.True(watch.Elapsed < TimeSpan.FromSeconds(2));
        }
        finally
        {
            GpuLockWaiters.Exit(gpuLock);
        }
    }
}
