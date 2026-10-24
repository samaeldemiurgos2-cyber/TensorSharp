// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#include "ggml_ops_deepseek41_tp_cuda.h"

#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "ggml-cuda/common.cuh"   // ggml_backend_cuda_context (stream access)

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <map>
#include <mutex>
#include <set>
#include <thread>

namespace tsg_dsv41_tp_cuda
{
namespace
{
cudaStream_t stream_of(ggml_backend_t cuda_backend, int & device)
{
    auto * context = static_cast<ggml_backend_cuda_context *>(cuda_backend->context);
    device = context->device;
    return context->stream(context->device, 0);
}

// Portable pinned pages, so every device's copy engine can use them.
struct pinned
{
    void * data = nullptr;
    size_t capacity = 0;
};

// Per-layer description the rank threads read while a layer is being issued.
struct layer_job
{
    const layer_work * layer = nullptr;
    const std::vector<rank_work> * ranks = nullptr;
    const std::vector<int> * device = nullptr;
    const std::vector<cudaStream_t> * stream = nullptr;
    cudaEvent_t ready = nullptr;
};
}

struct exchange
{
    int ranks = 0;
    // Recorded on rank streams: hidden rows staged, output rows staged.
    std::vector<cudaEvent_t> gate_done, output_done;
    std::vector<int> rank_device;
    // Recorded on a layer device's stream once the node's inputs are staged.
    std::map<int, cudaEvent_t> inputs_ready;
    std::set<int> devices;
    pinned inputs, hidden, outputs;

    // Rank r > 0 is issued by worker r - 1; rank 0 by the calling thread. Every
    // rank issues its gate phase, then waits at a host barrier until all ranks
    // have RECORDED their gate event (a stream wait on an event refers to its
    // most recent record, so recording must precede every wait), then issues
    // its down phase. Workers spin between layers while decode keeps them busy
    // and sleep after an idle stretch.
    std::vector<std::thread> workers;
    std::mutex mutex;
    std::condition_variable wake;
    std::atomic<uint64_t> generation{0};
    std::atomic<int> gate_issued{0}, finished{0};
    std::atomic<bool> failed{false};
    std::atomic<bool> stopping{false};
    layer_job job;
    std::vector<std::string> errors;
    // A launched layer whose output rows (launched_bytes) the next gather() collects.
    bool launched = false;
    size_t launched_bytes = 0;
};

#define TSG_TP_TRY(call)                                                                 \
    do                                                                                    \
    {                                                                                     \
        const cudaError_t tsg_tp_status = (call);                                         \
        if (tsg_tp_status != cudaSuccess)                                                 \
        {                                                                                 \
            error = std::string("V4.1 TP exchange: ") + #call + ": " +                    \
                    cudaGetErrorString(tsg_tp_status);                                     \
            return false;                                                                 \
        }                                                                                 \
    } while (0)

// Gate/up/SwiGLU on one rank. Its hidden rows land straight in their place
// within every expert/token column of the shared staging.
static bool issue_gate(exchange & ex, int r, std::string & error)
{
    const auto & layer = *ex.job.layer;
    const auto & rank = (*ex.job.ranks)[r];
    const int device = (*ex.job.device)[r];
    const cudaStream_t stream = (*ex.job.stream)[r];
    const size_t x_bytes = ggml_nbytes(layer.x), ids_bytes = ggml_nbytes(layer.ids);
    const size_t weight_bytes = ggml_nbytes(layer.weights);
    const char * inputs = static_cast<const char *>(ex.inputs.data);
    char * hidden = static_cast<char *>(ex.hidden.data);
    TSG_TP_TRY(cudaSetDevice(device));
    TSG_TP_TRY(cudaStreamWaitEvent(stream, ex.job.ready, 0));
    TSG_TP_TRY(cudaMemcpyAsync(rank.input->data, inputs, x_bytes, cudaMemcpyHostToDevice, stream));
    TSG_TP_TRY(cudaMemcpyAsync(rank.ids->data, inputs + x_bytes, ids_bytes, cudaMemcpyHostToDevice, stream));
    TSG_TP_TRY(cudaMemcpyAsync(rank.weights->data, inputs + x_bytes + ids_bytes, weight_bytes,
                               cudaMemcpyHostToDevice, stream));
    if (ggml_backend_graph_compute_async(rank.backend, rank.gate_graph) != GGML_STATUS_SUCCESS)
    {
        error = "V4.1 TP gate/up graph execution failed";
        return false;
    }
    TSG_TP_TRY(cudaSetDevice(device));
    const ggml_tensor * rows = rank.hidden;
    TSG_TP_TRY(cudaMemcpy2DAsync(hidden + size_t(rank.first_row) * sizeof(float),
                                 size_t(layer.full_rows) * sizeof(float), rows->data, rows->nb[1],
                                 size_t(rows->ne[0]) * sizeof(float), size_t(rows->ne[1] * rows->ne[2]),
                                 cudaMemcpyDeviceToHost, stream));
    TSG_TP_TRY(cudaEventRecord(ex.gate_done[r], stream));
    return true;
}

// The down projection's output rows on one rank, from the complete hidden
// activations (no partial sums cross a device).
static bool issue_down(exchange & ex, int r, std::string & error)
{
    const auto & layer = *ex.job.layer;
    const auto & rank = (*ex.job.ranks)[r];
    const int device = (*ex.job.device)[r];
    const cudaStream_t stream = (*ex.job.stream)[r];
    const size_t hidden_bytes = size_t(layer.full_rows) * layer.used * layer.tokens * sizeof(float);
    const char * hidden = static_cast<const char *>(ex.hidden.data);
    char * outputs = static_cast<char *>(ex.outputs.data);
    TSG_TP_TRY(cudaSetDevice(device));
    for (int s = 0; s < ex.ranks; ++s) TSG_TP_TRY(cudaStreamWaitEvent(stream, ex.gate_done[s], 0));
    TSG_TP_TRY(cudaMemcpyAsync(rank.hidden_input->data, hidden, hidden_bytes, cudaMemcpyHostToDevice, stream));
    if (ggml_backend_graph_compute_async(rank.backend, rank.down_graph) != GGML_STATUS_SUCCESS)
    {
        error = "V4.1 TP down graph execution failed";
        return false;
    }
    TSG_TP_TRY(cudaSetDevice(device));
    const ggml_tensor * rows = rank.output;
    TSG_TP_TRY(cudaMemcpy2DAsync(outputs + size_t(rank.first_output) * sizeof(float),
                                 size_t(layer.embedding) * sizeof(float), rows->data, rows->nb[1],
                                 size_t(rows->ne[0]) * sizeof(float), size_t(rows->ne[1]),
                                 cudaMemcpyDeviceToHost, stream));
    TSG_TP_TRY(cudaEventRecord(ex.output_done[r], stream));
    return true;
}

// A rank graph can throw: TensorSharp's CUDA matmul scratch reports a failed
// growth that way. The exception must not leave issue_rank, or a worker would
// terminate the process and rank 0 would leave the workers in this layer's
// barrier while the next launch resets its counters.
template <typename Phase>
static bool guarded(Phase phase, std::string & error)
{
    try { return phase(); }
    catch (const std::exception & e) { error = std::string("V4.1 TP exchange: ") + e.what(); }
    catch (...) { error = "V4.1 TP exchange: a rank graph threw"; }
    return false;
}

// One rank's share of a layer: gate phase, barrier, down phase. A failure
// (a false return or an exception) still arrives at the barrier and the join,
// so no rank waits forever, and it stops every rank's down phase.
static void issue_rank(exchange & ex, int r)
{
    std::string & error = ex.errors[r];
    error.clear();
    if (!guarded([&] { return issue_gate(ex, r, error); }, error)) ex.failed.store(true, std::memory_order_release);
    ex.gate_issued.fetch_add(1, std::memory_order_acq_rel);
    while (ex.gate_issued.load(std::memory_order_acquire) < ex.ranks) std::this_thread::yield();
    if (!ex.failed.load(std::memory_order_acquire) && !guarded([&] { return issue_down(ex, r, error); }, error))
        ex.failed.store(true, std::memory_order_release);
    ex.finished.fetch_add(1, std::memory_order_acq_rel);
}

static void worker(exchange * ex, int r)
{
    uint64_t seen = 0;
    for (;;)
    {
        // Spin while layers keep arriving (decode issues one every few hundred
        // microseconds), then sleep until the next forward.
        const auto idle_since = std::chrono::steady_clock::now();
        uint64_t current;
        while ((current = ex->generation.load(std::memory_order_acquire)) == seen &&
               !ex->stopping.load(std::memory_order_acquire))
        {
            if (std::chrono::steady_clock::now() - idle_since < std::chrono::milliseconds(2))
            {
                std::this_thread::yield();
                continue;
            }
            std::unique_lock<std::mutex> lock(ex->mutex);
            ex->wake.wait_for(lock, std::chrono::milliseconds(100), [&] {
                return ex->stopping.load(std::memory_order_acquire) ||
                       ex->generation.load(std::memory_order_acquire) != seen;
            });
        }
        if (ex->stopping.load(std::memory_order_acquire)) return;
        seen = current;
        issue_rank(*ex, r);
    }
}

exchange * create(int ranks)
{
    auto * ex = new exchange();
    ex->ranks = ranks;
    ex->gate_done.assign(ranks, nullptr);
    ex->output_done.assign(ranks, nullptr);
    ex->rank_device.assign(ranks, -1);
    ex->errors.assign(ranks, std::string());
    try
    {
        for (int r = 1; r < ranks; ++r) ex->workers.emplace_back(worker, ex, r);
    }
    catch (...)
    {
        destroy(ex);
        throw;
    }
    return ex;
}

static void drain(exchange & ex)
{
    for (const int device : ex.devices)
    {
        if (cudaSetDevice(device) == cudaSuccess) cudaDeviceSynchronize();
    }
}

void destroy(exchange * ex)
{
    if (!ex) return;
    {
        std::lock_guard<std::mutex> lock(ex->mutex);
        ex->stopping.store(true, std::memory_order_release);
    }
    ex->wake.notify_all();
    for (auto & thread : ex->workers) if (thread.joinable()) thread.join();
    drain(*ex);
    for (size_t r = 0; r < ex->gate_done.size(); ++r)
    {
        if (ex->rank_device[r] < 0) continue;
        cudaSetDevice(ex->rank_device[r]);
        if (ex->gate_done[r]) cudaEventDestroy(ex->gate_done[r]);
        if (ex->output_done[r]) cudaEventDestroy(ex->output_done[r]);
    }
    for (auto & entry : ex->inputs_ready)
    {
        cudaSetDevice(entry.first);
        cudaEventDestroy(entry.second);
    }
    for (pinned * buffer : {&ex->inputs, &ex->hidden, &ex->outputs})
        if (buffer->data) cudaFreeHost(buffer->data);
    delete ex;
}

int device_of(ggml_backend_t cuda_backend)
{
    return static_cast<ggml_backend_cuda_context *>(cuda_backend->context)->device;
}

void poison(ggml_tensor * dst, void * stream, int device)
{
    if (cudaSetDevice(device) != cudaSuccess) return;
    // All-ones bytes are a quiet NaN in F32.
    cudaMemsetAsync(dst->data, 0xFF, ggml_nbytes(dst), (cudaStream_t) stream);
}

static bool event_on(cudaEvent_t & event, int device, std::string & error)
{
    if (event) return true;
    TSG_TP_TRY(cudaSetDevice(device));
    TSG_TP_TRY(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    return true;
}

static bool reserve(exchange & ex, pinned & buffer, size_t bytes, std::string & error)
{
    if (buffer.capacity >= bytes) return true;
    // Every queue that could still read or write the old pages drains first.
    // This happens once per new largest batch, never in steady state.
    drain(ex);
    if (buffer.data) TSG_TP_TRY(cudaFreeHost(buffer.data));
    buffer.data = nullptr;
    buffer.capacity = 0;
    const size_t rounded = (bytes + (size_t(1) << 20) - 1) & ~((size_t(1) << 20) - 1);
    TSG_TP_TRY(cudaHostAlloc(&buffer.data, rounded, cudaHostAllocPortable));
    buffer.capacity = rounded;
    return true;
}

bool launch(exchange & ex, const layer_work & layer, const std::vector<rank_work> & ranks,
            std::string & error)
{
    const int count = (int) ranks.size();
    if (count != ex.ranks || !layer.x || !layer.ids || !layer.weights || !layer.stream)
    {
        error = "V4.1 TP exchange: invalid layer or rank description";
        return false;
    }
    const size_t x_bytes = ggml_nbytes(layer.x), ids_bytes = ggml_nbytes(layer.ids);
    const size_t weight_bytes = ggml_nbytes(layer.weights);
    const size_t hidden_bytes = size_t(layer.full_rows) * layer.used * layer.tokens * sizeof(float);
    const size_t output_bytes = size_t(layer.embedding) * layer.tokens * sizeof(float);
    if (!ggml_is_contiguous(layer.x) || !ggml_is_contiguous(layer.ids) || !ggml_is_contiguous(layer.weights))
    {
        error = "V4.1 TP exchange: node inputs must be contiguous and match the layer geometry";
        return false;
    }

    std::vector<int> device(count);
    std::vector<cudaStream_t> stream(count);
    ex.devices.insert(layer.device);
    for (int r = 0; r < count; ++r)
    {
        const auto & rank = ranks[r];
        const ggml_tensor * hidden = rank.hidden;
        const ggml_tensor * output = rank.output;
        if (ggml_nbytes(rank.input) != x_bytes || ggml_nbytes(rank.ids) != ids_bytes ||
            ggml_nbytes(rank.weights) != weight_bytes || ggml_nbytes(rank.hidden_input) != hidden_bytes ||
            !ggml_is_contiguous(hidden) || !ggml_is_contiguous(output) ||
            hidden->nb[1] != size_t(hidden->ne[0]) * sizeof(float) ||
            output->nb[1] != size_t(output->ne[0]) * sizeof(float) ||
            rank.first_row + hidden->ne[0] > layer.full_rows ||
            rank.first_output + output->ne[0] > layer.embedding ||
            hidden->ne[1] * hidden->ne[2] != layer.used * layer.tokens || output->ne[1] != layer.tokens)
        {
            error = "V4.1 TP exchange: rank graph tensors do not match the layer geometry";
            return false;
        }
        stream[r] = stream_of(rank.cuda_backend, device[r]);
        if (ex.rank_device[r] < 0) ex.rank_device[r] = device[r];
        if (ex.rank_device[r] != device[r])
        {
            error = "V4.1 TP exchange: a rank changed device";
            return false;
        }
        ex.devices.insert(device[r]);
        if (!event_on(ex.gate_done[r], device[r], error) || !event_on(ex.output_done[r], device[r], error))
            return false;
    }
    cudaEvent_t & ready = ex.inputs_ready[layer.device];
    if (!event_on(ready, layer.device, error)) return false;
    if (!reserve(ex, ex.inputs, x_bytes + ids_bytes + weight_bytes, error) ||
        !reserve(ex, ex.hidden, hidden_bytes, error) || !reserve(ex, ex.outputs, output_bytes, error))
        return false;

    // The staging pages are shared by every layer. Reuse is ordered by the
    // data itself: a layer's inputs depend on the previous layer's gathered
    // output (a forward's first TP layer, on the previous forward having
    // completed), which every rank produced only after reading this staging.
    // A failed layer has no gather, so it drains its ranks instead (below).
    char * inputs = static_cast<char *>(ex.inputs.data);
    const auto layer_stream = static_cast<cudaStream_t>(layer.stream);
    ex.launched = false;

    // 1. The node's inputs leave the layer device once.
    TSG_TP_TRY(cudaSetDevice(layer.device));
    TSG_TP_TRY(cudaMemcpyAsync(inputs, layer.x->data, x_bytes, cudaMemcpyDeviceToHost, layer_stream));
    TSG_TP_TRY(cudaMemcpyAsync(inputs + x_bytes, layer.ids->data, ids_bytes, cudaMemcpyDeviceToHost, layer_stream));
    TSG_TP_TRY(cudaMemcpyAsync(inputs + x_bytes + ids_bytes, layer.weights->data, weight_bytes,
                               cudaMemcpyDeviceToHost, layer_stream));
    TSG_TP_TRY(cudaEventRecord(ready, layer_stream));

    // 2-3. Every rank's gate and down phases, issued concurrently.
    ex.job.layer = &layer;
    ex.job.ranks = &ranks;
    ex.job.device = &device;
    ex.job.stream = &stream;
    ex.job.ready = ready;
    ex.failed.store(false, std::memory_order_relaxed);
    ex.gate_issued.store(0, std::memory_order_relaxed);
    ex.finished.store(0, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lock(ex.mutex);
        ex.generation.fetch_add(1, std::memory_order_acq_rel);
    }
    ex.wake.notify_all();
    issue_rank(ex, 0);
    while (ex.finished.load(std::memory_order_acquire) < count) std::this_thread::yield();
    if (ex.failed.load(std::memory_order_acquire))
    {
        for (const auto & message : ex.errors)
            if (!message.empty()) { error = message; break; }
        if (error.empty()) error = "V4.1 TP exchange: a rank failed";
        // No gather follows a failed layer, so nothing would order the ranks
        // that did issue (their gate rows land in the shared hidden staging)
        // before a later launch's writes there. Drain them now; every event
        // they wait on is already recorded.
        for (int r = 0; r < count; ++r)
            if (cudaSetDevice(device[r]) == cudaSuccess) cudaStreamSynchronize(stream[r]);
        cudaGetLastError();
        return false;
    }
    ex.launched = true;
    ex.launched_bytes = output_bytes;
    return true;
}

bool gather(exchange & ex, ggml_tensor * dst, void * stream, int device, std::string & error)
{
    if (!ex.launched)
    {
        error = "V4.1 TP exchange: gather without a launched layer";
        return false;
    }
    ex.launched = false;
    const size_t output_bytes = ggml_nbytes(dst);
    if (output_bytes != ex.launched_bytes || !ggml_is_contiguous(dst))
    {
        error = "V4.1 TP exchange: gather destination does not match the launched layer";
        return false;
    }
    // 4. The layer device continues once every rank's rows are staged.
    const auto layer_stream = static_cast<cudaStream_t>(stream);
    TSG_TP_TRY(cudaSetDevice(device));
    for (int r = 0; r < ex.ranks; ++r) TSG_TP_TRY(cudaStreamWaitEvent(layer_stream, ex.output_done[r], 0));
    TSG_TP_TRY(cudaMemcpyAsync(dst->data, ex.outputs.data, output_bytes, cudaMemcpyHostToDevice, layer_stream));
    return true;
}

bool enqueue(exchange & ex, const layer_work & layer, const std::vector<rank_work> & ranks,
             std::string & error)
{
    return launch(ex, layer, ranks, error) && gather(ex, layer.dst, layer.stream, layer.device, error);
}
}
