// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// Device-ordered exchange for DeepSeek V4.1 routed-MoE tensor parallelism.
//
// A V4.1 layer's routed experts are sharded over the TP ranks while attention
// stays on the layer's own GPU. Each layer therefore moves its activations
// out to every rank, gathers the ranks' hidden rows for the down projection,
// and brings the down output rows back. This module enqueues that whole
// exchange on CUDA streams, ordered only by CUDA events, so the host issues a
// layer and moves on: no stream synchronization, no CPU graph split.
//
// Every transfer is staged through portable pinned host memory with the
// device's own copy engine. That is deliberate: peer-to-peer copies are not
// usable on every PCIe host (the TP probe found an advertised but broken peer
// path on the five-A40 box), and a staged copy needs no peer mapping at all.
#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <string>
#include <vector>

namespace tsg_dsv41_tp_cuda
{
// One rank's share of a layer. `backend` submits the rank's ggml graphs (the
// fused wrapper when there is one); `cuda_backend` is the ggml-cuda backend
// whose stream and device that submission uses.
struct rank_work
{
    ggml_backend_t backend = nullptr;
    ggml_backend_t cuda_backend = nullptr;
    ggml_cgraph * gate_graph = nullptr;
    ggml_cgraph * down_graph = nullptr;
    // Graph inputs the exchange fills, and outputs it drains. All contiguous.
    ggml_tensor * input = nullptr;        // F32 [embedding, 1, tokens]
    ggml_tensor * ids = nullptr;          // I32 [used, tokens]
    ggml_tensor * weights = nullptr;      // F32 [1, used, tokens]
    ggml_tensor * hidden = nullptr;       // F32 [rows, used, tokens], this rank's rows
    ggml_tensor * hidden_input = nullptr; // F32 [full_rows, used, tokens]
    ggml_tensor * output = nullptr;       // F32 [output_rows, tokens], this rank's rows
    int64_t first_row = 0;                // of `hidden` within full_rows
    int64_t first_output = 0;             // of `output` within embedding
};

// The layer device's side: the node's inputs and output, and the stream the
// scheduler runs that device's subgraph on.
struct layer_work
{
    const ggml_tensor * x = nullptr;       // F32 [embedding, tokens]
    const ggml_tensor * ids = nullptr;     // I32 [used, tokens]
    const ggml_tensor * weights = nullptr; // F32 [1, used, tokens]
    ggml_tensor * dst = nullptr;           // F32 [embedding, tokens]
    void * stream = nullptr;               // cudaStream_t on `device`
    int device = 0;
    int64_t embedding = 0, full_rows = 0, used = 0, tokens = 0;
};

struct exchange;
exchange * create(int ranks);
// Drains every queue the exchange used before releasing its memory.
void destroy(exchange * ex);

// Enqueue one layer. Returns false with `error` set if a CUDA call fails;
// work enqueued before the failure stays queued and the caller poisons dst.
bool enqueue(exchange & ex, const layer_work & layer, const std::vector<rank_work> & ranks,
             std::string & error);

// enqueue() in two halves, so the layer device can compute something else
// (the shared expert) while the ranks run: launch() stages the inputs and
// issues every rank; gather() makes `stream` wait for the ranks and copies
// the routed rows into `dst`. One gather per launch, in order.
bool launch(exchange & ex, const layer_work & layer, const std::vector<rank_work> & ranks,
            std::string & error);
bool gather(exchange & ex, ggml_tensor * dst, void * stream, int device, std::string & error);

// Fill `dst` with NaNs on `stream` (a failed layer must not look valid).
void poison(ggml_tensor * dst, void * stream, int device);

// The CUDA device and default stream of a ggml-cuda backend.
int device_of(ggml_backend_t cuda_backend);
}
