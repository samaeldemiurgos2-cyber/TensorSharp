// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#pragma once

// How many compressed rows a DeepSeek V4 / V4.1 micro-batch's graph gives its
// attention (comp_plan::n_kv). DeepSeek V4.1 attends over
// K = [raw ring (ring_raw rows) | n_kv compressed rows], and that width is not
// only a shape: the owned F32 attention picks its kernel from it (sparse at
// 2,048 keys and more, ggml_ops_precision_policy.h) and sizes its split-key
// partitions from it (ggml_ops_attention_precision.cpp). Kept free of ggml and
// of the model struct so tests/dsv41_call_invariance_test.cpp can check the
// rule without a model; ggml_ops_deepseek4.cpp's build_comp_plan,
// dsv4_acquire_graph and dsv4_acquire_batched_graph are the production callers.

#include <algorithm>
#include <cstdint>

namespace tsg_dsv4_plan {

// The call end past which plain V4's floor stops growing, so very long
// prefills do not over-pad their early attention.
constexpr int64_t COMP_FLOOR_POSITION_CAP = 8192;

// Power-of-two buckets of at least 256 rows over max(visible_rows, floor_rows),
// capped at the cache's kv_rows. The graph shape then changes only at bucket
// doublings, so the graph cache reuses an entry across long position
// stretches instead of building (and device-synchronizing on a fresh arena
// allocation) a graph every micro-batch.
inline int64_t comp_bucket_rows(int64_t visible_rows, int64_t floor_rows, int64_t kv_rows)
{
    int64_t bucket = 256;
    const int64_t needed = std::max<int64_t>(std::max(visible_rows, floor_rows), 1);
    while (bucket < needed) bucket <<= 1;
    return std::min(bucket, kv_rows);
}

// The row floor dsv4_acquire_graph gives the micro-batch [p0, p0 + nt) of a
// forward call that ends at position `call_end` (exclusive), at compression
// `ratio`.
//
// Plain V4 floors every micro-batch at the call's end (capped at
// COMP_FLOOR_POSITION_CAP positions): all chunks of one prefill then share one
// graph-shape pair, and its pipelined layer-split prefill never stalls on a
// mid-flight allocation.
//
// DeepSeek V4.1 has no floor: a micro-batch's bucket comes from its own
// visible rows. Its K width selects arithmetic, so a floor taken from the call
// end made a prompt's KV state and logits depend on how the prompt was split
// into forward calls. A whole-prefix prefill padded its early micro-batches to
// the final bucket and ran them on the sparse kernel, while a chunked prefill
// padded each call only to its own end and ran the same micro-batches tiled.
// Both kernels are F32-accurate but round differently, and the lite 1017
// checkpoint amplified that to logits 4.5e-2..1.6e-1 apart, with changed
// argmax tokens, between a 4,204-token prompt prefilled whole and in 256-token
// chunks. V4.1 never pipelines its prefill, so the floor only saved graph
// builds: at most one more per bucket doubling, each cached.
inline int64_t comp_floor_rows(bool v41, int64_t call_end, int64_t p0, int64_t nt, int64_t ratio)
{
    if (v41 || ratio <= 0) return 0;
    const int64_t pos_end = std::max(call_end, p0 + nt);
    return std::min<int64_t>(pos_end, COMP_FLOOR_POSITION_CAP) / ratio;
}

} // namespace tsg_dsv4_plan
