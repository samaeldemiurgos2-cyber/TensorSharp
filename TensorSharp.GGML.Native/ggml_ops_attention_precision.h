// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#pragma once
#include "ggml.h"
#include "ggml-backend.h"

// F32 queries and F32/F16/BF16 K/V. Q: [D,N,H,B], K: [D,S,Hk,Bk],
// V: [Dv,S,Hv,Bv]. K/V heads and batches use quotient broadcasting;
// additive masks [S,Npad,Hm,Bm] use modulo broadcasting. Optional F32 sinks
// are per-query-head logits added to the denominator with a zero value.
// Returns F32 [Dv,H,N,B]. Fully masked rows return zero, with or without sinks.
extern "C" ggml_tensor * tsg_attention_f32(ggml_context *, ggml_tensor * q, ggml_tensor * k,
    ggml_tensor * v, ggml_tensor * mask, ggml_tensor * sinks, float scale);

// Ordered mask compaction, shared across heads. Capacity is a performance hint:
// overflowing or exceptional rows retain the full-mask path, never truncate.
extern "C" ggml_tensor * tsg_attention_f32_sparse(ggml_context *, ggml_tensor * q, ggml_tensor * k,
    ggml_tensor * v, ggml_tensor * mask, ggml_tensor * sinks, float scale, int capacity);

// Full bidirectional F32 vision attention for 64/72-wide heads, without masks or
// head/batch broadcasting. Q/K/V [D,N,H,B], output [D,H,N,B]. CUDA shares K/V
// tiles across queries and never allocates the quadratic attention matrix.
// The CPU callback is an independent double-precision reference.
extern "C" ggml_tensor * tsg_vision_attention_f32(ggml_context *, ggml_tensor * q,
    ggml_tensor * k, ggml_tensor * v, float scale);

// The same contract on tensor cores (CUDA sm_80+): Q/K/V operands are rounded to
// F16 (K and V by a ggml cast this adds), scores, softmax and output accumulate
// in F32. The CPU callback is the double-precision reference of
// tsg_vision_attention_f32.
extern "C" ggml_tensor * tsg_vision_attention_mma(ggml_context *, ggml_tensor * q,
    ggml_tensor * k, ggml_tensor * v, float scale);

// Place the complete owned subgraph, including the shared in-place output
// allocation, on one scheduler backend. Existing input producers keep their
// placement and the scheduler transfers their required views normally.
// The placement traversal allocates C++ containers. Explicitly permit unwind
// through this internal C-linkage call under MSVC /EHsc; its caller translates
// allocation failure to the model's failure status.
extern "C" ggml_tensor * tsg_attention_f32_on_backend(ggml_context *, ggml_backend_sched_t,
    ggml_backend_t, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
    ggml_tensor * mask, ggml_tensor * sinks, float scale, int sparse_capacity = 0) noexcept(false);

#ifdef TSG_GGML_USE_CUDA
void tsg_attention_cuda_compute(ggml_tensor *, ggml_backend_t);
// True when the backend's device has the tensor cores tsg_vision_attention_mma needs.
bool tsg_vision_mma_supported(ggml_backend_t);
#endif
