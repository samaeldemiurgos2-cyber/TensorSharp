// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// host_read_barrier's drain gate without ggml: the sync it runs is a function
// the test blocks on command, standing in for a Metal command buffer that has
// not retired. The bug this guards cleared the flag before the sync, so a
// caller arriving mid-drain returned at once: a model unload freed buffers a
// queued blit was still reading, or a thread read a result before it landed.
#include "ggml_ops_host_read_gate.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <vector>

namespace {
using clock_type = std::chrono::steady_clock;
constexpr auto kHang = std::chrono::seconds(10);   // nothing here may take this long
constexpr auto kSettle = std::chrono::milliseconds(200);

int failures = 0;
void expect(bool condition, const char* what)
{
    if (!condition) { std::fprintf(stderr, "FAIL: %s\n", what); ++failures; }
}

// A thread blocked for kHang is a deadlock: report it and exit, since it can
// never be joined.
template <typename Pred>
bool wait_for(Pred pred, clock_type::duration limit)
{
    const auto deadline = clock_type::now() + limit;
    while (!pred())
    {
        if (clock_type::now() >= deadline) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return true;
}
void must(bool condition, const char* what)
{
    if (condition) return;
    std::fprintf(stderr, "FAIL (hang): %s\n", what);
    std::fflush(stderr);
    std::_Exit(1);
}

// A sync that does not return until released: the GPU work it drains is still
// running until then.
struct BlockingSync
{
    std::mutex mutex;
    std::condition_variable cv;
    bool released = false;
    std::atomic<bool> entered{false};
    std::atomic<bool> done{false};
    std::atomic<int> calls{0};

    void operator()()
    {
        calls.fetch_add(1);
        entered.store(true);
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait_for(lock, kHang, [&] { return released; });
        done.store(true);
    }
    void release()
    {
        { std::lock_guard<std::mutex> lock(mutex); released = true; }
        cv.notify_all();
    }
};

void sequential()
{
    tsg::HostReadGate gate;
    int calls = 0;
    auto sync = [&] { ++calls; };
    expect(gate.idle(), "a new gate is idle");
    expect(!gate.drain(sync) && calls == 0, "nothing marked: no sync");
    gate.mark_pending();
    expect(!gate.idle(), "a mark leaves the gate busy");
    expect(gate.drain(sync) && calls == 1, "a mark is drained once");
    expect(gate.idle() && !gate.drain(sync) && calls == 1, "a drained mark is not drained again");
    gate.mark_pending();
    gate.reset();
    expect(gate.idle() && !gate.drain(sync) && calls == 1, "reset forgets a mark without a sync");
}

// The drainer's sync is in flight; a second caller (the owner thread reading
// its result, or a release_all freeing its buffers) must not return before it
// completes, and must not sync again for work the drainer already took.
void second_caller_waits_for_drain()
{
    tsg::HostReadGate gate;
    BlockingSync gpu;
    gate.mark_pending();
    bool drainer_synced = false;
    std::thread drainer([&] { drainer_synced = gate.drain([&] { gpu(); }); });
    must(wait_for([&] { return gpu.entered.load(); }, kHang), "drainer never reached its sync");

    std::atomic<int> second_syncs{0};
    std::atomic<bool> calling{false}, returned{false}, saw_done{false};
    bool second_synced = true;
    std::thread second([&] {
        calling.store(true);
        second_synced = gate.drain([&] { second_syncs.fetch_add(1); });
        saw_done.store(gpu.done.load());
        returned.store(true);
    });
    must(wait_for([&] { return calling.load(); }, kHang), "second caller never started");
    // A correct gate never returns here; the old exchange returned at once.
    expect(!wait_for([&] { return returned.load(); }, kSettle),
           "second caller returned while the drain was in flight");
    gpu.release();
    must(wait_for([&] { return returned.load(); }, kHang), "second caller deadlocked after the drain");
    drainer.join();
    second.join();
    expect(saw_done.load(), "second caller returned before the drained work completed");
    expect(drainer_synced && !second_synced && second_syncs.load() == 0 && gpu.calls.load() == 1,
           "the work was drained exactly once, by the first caller");
    expect(gate.idle(), "gate idle after the drain");
}

// Work marked while a drain is in flight (after its exchange) belongs to no
// sync yet: a caller arriving behind it waits for the drain, then syncs itself.
// That covers the work only if the second sync waits for everything queued
// before it, which BlockingSync models and ggml-metal does not promise when the
// submission raced the first sync: ggml_metal_synchronize nils cmd_buf_last
// after its wait while graph_compute and get/set_tensor_async set it unlocked
// (ExternalProjects/ggml/src/ggml-metal/ggml-metal-context.m:256-312 vs
// :358-366, :404-412, :577, :597), so a buffer committed during the first sync
// can be lost to the second. This checks the gate under the single-submitter
// assumption stated in ggml_ops_host_read_gate.h, not ggml-metal's sync.
void mark_during_drain_is_drained_next()
{
    tsg::HostReadGate gate;
    BlockingSync gpu;
    gate.mark_pending();
    std::thread drainer([&] { gate.drain([&] { gpu(); }); });
    must(wait_for([&] { return gpu.entered.load(); }, kHang), "drainer never reached its sync");
    gate.mark_pending();

    std::atomic<bool> returned{false}, first_done_at_sync{false};
    bool second_synced = false;
    std::thread second([&] {
        second_synced = gate.drain([&] { first_done_at_sync.store(gpu.done.load()); });
        returned.store(true);
    });
    expect(!wait_for([&] { return returned.load(); }, kSettle),
           "caller behind a drain returned while it was in flight");
    gpu.release();
    must(wait_for([&] { return returned.load(); }, kHang), "caller behind a drain deadlocked");
    drainer.join();
    second.join();
    expect(second_synced && first_done_at_sync.load(),
           "the later mark was synced by the next caller, after the first drain");
    expect(gate.idle(), "gate idle after both drains");
}

// Many threads queue, mark and drain against one gate. The sync "completes"
// everything queued when it started; when a drain returns, the caller's own
// queued work must be complete, whoever ran the sync.
void stress()
{
    constexpr int kThreads = 8;
    constexpr int kIterations = 20000;
    tsg::HostReadGate gate;
    std::atomic<long long> queued{0}, completed{0};
    std::atomic<int> violations{0};
    auto sync = [&] {
        const long long snapshot = queued.load();
        std::this_thread::yield();   // keep the drain in flight for a while
        long long seen = completed.load();
        while (seen < snapshot && !completed.compare_exchange_weak(seen, snapshot)) {}
    };
    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t)
    {
        threads.emplace_back([&, t] {
            for (int i = 0; i < kIterations; ++i)
            {
                long long mine = 0;
                if ((i + t) % 3 != 0)   // some callers only read
                {
                    mine = queued.fetch_add(1) + 1;
                    gate.mark_pending();
                }
                gate.drain(sync);
                if (completed.load() < mine) violations.fetch_add(1);
            }
        });
    }
    for (auto& thread : threads) thread.join();
    if (violations.load() != 0)
        std::fprintf(stderr, "  %d drains returned before the caller's work completed\n", violations.load());
    expect(violations.load() == 0, "every drain covered the caller's own marked work");
    expect(gate.idle() && completed.load() == queued.load(), "all queued work drained");
}
}

int main()
{
    sequential();
    second_caller_waits_for_drain();
    mark_during_drain_is_drained_next();
    stress();
    if (failures != 0)
    {
        std::fprintf(stderr, "%d host read gate check(s) failed\n", failures);
        return 1;
    }
    std::printf("host read gate: all checks passed\n");
    return 0;
}
