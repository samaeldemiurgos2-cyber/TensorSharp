// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#pragma once

// The drain behind host_read_barrier (ggml_ops_internal.h). Kept free of ggml so
// tests/host_read_gate_test.cpp can drive it with a sync it blocks on command.
#include <atomic>
#include <mutex>

namespace tsg
{
    // Records that an op left GPU work queued that host code must not read past
    // (Metal async compute: a zero-copy result or a queued blit download), and
    // drains it. One per process, like the backend queue it stands for.
    //
    // A drain used to be a single exchange of the flag followed by the sync. The
    // flag is cleared BEFORE the sync runs, so a second caller arriving mid-drain
    // saw it clear and returned at once while the work it had to wait for was
    // still on the GPU: a model unload (ThreadCacheRegistry::release_all) then
    // freed buffers a queued blit was still reading, or the thread that queued
    // the blit read its output before it landed.
    //
    // Drains now run one at a time under mutex_, with draining_ up from before a
    // drain's exchange until after its sync, and the lock-free check reads
    // pending_ and THEN draining_, every access seq_cst. Say work W was marked
    // before a call and the check still lets the call return. Its pending_ read
    // saw false, which some drainer D's exchange wrote after W's mark (a read
    // that follows the mark cannot see an older value; reset() is the other
    // writer, and runs only after a teardown synchronized the backend). D raised
    // draining_ before that exchange, which precedes the pending_ read, which
    // precedes the draining_ read, so the latter cannot see a value older than
    // D's raise. Having seen false, it saw D's lowering or a later drainer's:
    // that follows D's sync and every drain before it, including the first one
    // to clear W's mark, whose sync started after W was queued. Read the flags in
    // the other order and both can come back false while D is still syncing. A
    // caller that takes the lock waits for every earlier drain the same way.
    //
    // ASSUMPTION (Metal; not enforced here): that argument proves only that a
    // sync ran after the work was queued, so the guarantees on drain() hold only
    // if a sync waits for everything queued before it started. ggml-metal's
    // does only while no other thread submits to the same Metal backend during
    // it. ggml_metal_synchronize waits on ctx->cmd_buf_last, then sets it to nil
    // and empties cmd_bufs_ext and buf_refs
    // (ExternalProjects/ggml/src/ggml-metal/ggml-metal-context.m:256-312), and
    // graph_compute and set/get_tensor_async write those fields with no lock
    // (:358-366, :404-412, :577, :597). A command buffer committed between that
    // wait and the nil is forgotten: the next sync, the one a later drain runs
    // for the work marked with it, finds cmd_buf_last nil and returns while that
    // buffer is still on the GPU (and the first sync can find it unfinished in
    // cmd_bufs_ext and latch has_error). So "work marked before the call has
    // completed" and "work marked during a drain is waited out" assume one
    // thread at a time submits to and drains a given Metal backend. Lifting it
    // takes one TensorSharp-owned lock that every Metal submission (graph
    // compute, async upload and download) and sync_backend hold, so that a sync
    // never overlaps a submission; ggml stays unchanged. CUDA and Vulkan never
    // mark the gate, so none of this applies to them.
    //
    // mutex_ is a leaf: only sync() runs under it, and ggml_backend_synchronize
    // calls back into nothing but the log callback, so a caller may hold any
    // other lock (a ThreadCacheRegistry slot, the teardown lock). Only Metal
    // async compute and a host-pointer bind (paged attention on Metal or the CPU
    // backend) ever raise the flag. CUDA and Vulkan never do, so their callers
    // never reach the lock: no rank thread can be held behind a sync that waits
    // for that rank's half of an in-kernel collective.
    class HostReadGate
    {
    public:
        // seq_cst like the drains' exchanges, so an exchange that reads a later
        // thread's mark is still ordered after this one. Costs nothing extra on
        // ARM64 (Metal), where a release store is the same instruction.
        void mark_pending() noexcept
        {
            pending_.store(true, std::memory_order_seq_cst);
        }

        // Nothing marked and nothing being drained: the whole cost of a barrier
        // on a backend that never defers a sync. Two loads, no write.
        bool idle() const noexcept
        {
            return !pending_.load(std::memory_order_seq_cst) &&
                   !draining_.load(std::memory_order_seq_cst);
        }

        // Returns once all work marked before the call has completed, including
        // when another thread runs the sync that completes it, provided sync()
        // waits for everything queued before it started (on Metal, the
        // single-submitter assumption above). Calls sync() itself only if the
        // work is still marked once it holds the lock, and returns true iff it did.
        template <typename Sync>
        bool drain(Sync&& sync)
        {
            if (idle())
                return false;
            std::lock_guard<std::mutex> lock(mutex_);
            draining_.store(true, std::memory_order_seq_cst);
            struct Lower
            {
                std::atomic<bool>& flag;
                ~Lower() { flag.store(false, std::memory_order_seq_cst); }
            } lower{draining_};
            if (!pending_.exchange(false, std::memory_order_seq_cst))
                return false;
            sync();
            return true;
        }

        // Forgets marked work without draining it. Only for a teardown that has
        // already synchronized and freed the backend the work was queued on.
        void reset()
        {
            std::lock_guard<std::mutex> lock(mutex_);
            pending_.store(false, std::memory_order_seq_cst);
        }

    private:
        std::atomic<bool> pending_{false};
        std::atomic<bool> draining_{false};
        std::mutex mutex_;
    };
}
