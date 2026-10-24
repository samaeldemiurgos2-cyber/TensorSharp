// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#include "ggml_ops_attention_precision.h"
#include "ggml_ops_dsv4_fused.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "ggml-cuda/common.cuh"
#include <atomic>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <math_constants.h>

namespace {
struct view {
    const char * data;
    int64_t ne[4];
    size_t nb[4];
    int type;
};
view tensor_view(const ggml_tensor * t) {
    view result;
    result.data = static_cast<const char *>(t->data);
    result.type = t->type;
    for (int i = 0; i < 4; ++i) {
        result.ne[i] = t->ne[i];
        result.nb[i] = t->nb[i];
    }
    return result;
}
__device__ __forceinline__ float read(view t, int64_t x, int64_t y = 0, int64_t h = 0, int64_t b = 0) {
    const char * p = t.data + x * t.nb[0] + y * t.nb[1] + h * t.nb[2] + b * t.nb[3];
    if (t.type == GGML_TYPE_F32)
        return *reinterpret_cast<const float *>(p);
    if (t.type == GGML_TYPE_F16)
        return __half2float(*reinterpret_cast<const half *>(p));
    return __uint_as_float(uint32_t(*reinterpret_cast<const uint16_t *>(p)) << 16);
}

// One thread owns one query vector, while the block shares each K/V tile.
// All operands, dot products, online-softmax state and weighted sums stay F32.
// Unlike upstream's generic 72-wide flash kernel, this never rounds Q/K/V or
// accumulators to half; unlike decomposed attention, scratch is O(N * D).
template <int WIDTH>
__global__ __launch_bounds__(128) void vision_kernel(view q, view k, view v, float scale, float * output) {
    constexpr int QUERIES = 128, KEYS = 16;
    __shared__ float keys[KEYS * WIDTH], values[KEYS * WIDTH];
    // Query-major register arrays for all scores add register pressure and make
    // the compiler emit an enormous unrolled instruction stream. Keep this
    // small score tile key-major in shared memory: adjacent query threads use
    // adjacent banks, and the two vector loops below keep their independent
    // per-component accumulators in registers.
    __shared__ float probabilities[KEYS * QUERIES];
    const int head = blockIdx.y, batch = blockIdx.z;
    const int64_t query = int64_t(blockIdx.x) * QUERIES + threadIdx.x;
    const bool active = query < q.ne[1];
    float query_values[WIDTH], accumulators[WIDTH];
#pragma unroll
    for (int x = 0; x < WIDTH; ++x) {
        query_values[x] = active ? read(q, x, query, head, batch) : 0;
        accumulators[x] = 0;
    }
    float maximum = -CUDART_INF_F, denominator = 0;
    for (int64_t first = 0; first < k.ne[1]; first += KEYS) {
        const int count = int(min(int64_t(KEYS), k.ne[1] - first));
        for (int i = threadIdx.x; i < count * WIDTH; i += QUERIES) {
            keys[i] = read(k, i % WIDTH, first + i / WIDTH, head, batch);
            values[i] = read(v, i % WIDTH, first + i / WIDTH, head, batch);
        }
        __syncthreads();
        if (active) {
            float next = maximum;
#pragma unroll 1
            for (int j = 0; j < KEYS; ++j) {
                float dot0 = 0, dot1 = 0, dot2 = 0, dot3 = 0;
                if (j < count) {
#pragma unroll
                    for (int x = 0; x < WIDTH; x += 4) {
                        dot0 = fmaf(query_values[x], keys[j * WIDTH + x], dot0);
                        dot1 = fmaf(query_values[x + 1], keys[j * WIDTH + x + 1], dot1);
                        dot2 = fmaf(query_values[x + 2], keys[j * WIDTH + x + 2], dot2);
                        dot3 = fmaf(query_values[x + 3], keys[j * WIDTH + x + 3], dot3);
                    }
                }
                const float dot = (dot0 + dot1) + (dot2 + dot3);
                const float score = j < count ? __fmul_rn(dot, scale) : -CUDART_INF_F;
                probabilities[j * QUERIES + threadIdx.x] = score;
                next = fmaxf(next, score);
            }
            const float correction = expf(maximum - next);
            denominator *= correction;
#pragma unroll 1
            for (int j = 0; j < KEYS; ++j) {
                const float probability = j < count ? expf(probabilities[j * QUERIES + threadIdx.x] - next) : 0;
                probabilities[j * QUERIES + threadIdx.x] = probability;
                denominator += probability;
            }
#pragma unroll
            for (int x = 0; x < WIDTH; ++x) accumulators[x] *= correction;
#pragma unroll 1
            for (int j = 0; j < count; ++j) {
                const float probability = probabilities[j * QUERIES + threadIdx.x];
#pragma unroll
                for (int x = 0; x < WIDTH; ++x)
                    accumulators[x] = fmaf(probability, values[j * WIDTH + x], accumulators[x]);
            }
            maximum = next;
        }
        // An inactive tail query participates in both barriers; no tile can be
        // overwritten until all active queries have consumed it.
        __syncthreads();
    }
    if (active) {
        const int64_t row = head + q.ne[2] * (query + q.ne[1] * batch);
#pragma unroll
        for (int x = 0; x < WIDTH; ++x)
            output[row * WIDTH + x] = accumulators[x] / denominator;
    }
}

// A block owns one query/head and one key partition. Sixteen scores at a
// time share the cached F32 query. Both online softmax and the V weighting
// remain F32; no score matrix or host synchronization is needed.
template <int MAX_VALUE, bool SPARSE>
__global__ void partial_kernel(view q, view k, view v, view mask, view compact, bool masked, float scale, int chunk, int splits,
                               float * output) {
    constexpr int TILE = 16, THREADS = 128;
    extern __shared__ float query_values[];
    __shared__ float scores[TILE], weights[TILE], maximum, denominator, correction;
    __shared__ int64_t selected_keys[TILE];
    const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    const int64_t row = blockIdx.x, split = row % splits, h = (row / splits) % q.ne[2];
    const int64_t query = (row / (splits * q.ne[2])) % q.ne[1], batch = row / (splits * q.ne[2] * q.ne[1]);
    const int64_t kh = h / (q.ne[2] / k.ne[2]), kb = batch / (q.ne[3] / k.ne[3]);
    const int64_t vh = h / (q.ne[2] / v.ne[2]), vb = batch / (q.ne[3] / v.ne[3]);
    const int32_t * indices = nullptr;
    int64_t count = k.ne[1];
    if constexpr (SPARSE) {
        const int64_t index_row = query + q.ne[1] *
            (h % compact.ne[2] + compact.ne[2] * (batch % compact.ne[3]));
        const auto * candidate = reinterpret_cast<const int32_t *>(compact.data) + index_row * compact.ne[0];
        if (candidate[0] >= 0 && candidate[0] < compact.ne[0]) {
            count = candidate[0];
            indices = candidate + 1;
        }
        chunk = int(max(int64_t(16), ((count + splits - 1) / splits + 15) / 16 * 16));
    }
    const int64_t begin = split * chunk, end = min(count, begin + chunk);
    float values[(MAX_VALUE + THREADS - 1) / THREADS] = {};
    for (int x = tid; x < q.ne[0]; x += THREADS)
        query_values[x] = read(q, x, query, h, batch);
    if (tid == 0) {
        maximum = -CUDART_INF_F;
        denominator = 0;
    }
    __syncthreads();
    for (int64_t first = begin; first < end; first += TILE) {
        for (int j = warp; j < TILE; j += 4) {
            const int64_t entry = first + j;
            const int64_t key = entry < end && indices ? indices[entry] : entry;
            const float bias =
                entry < end ? (masked ? read(mask, key, query, h % mask.ne[2], batch % mask.ne[3]) : 0) : -CUDART_INF_F;
            float dot = 0;
            if (bias != -CUDART_INF_F) {
                for (int x = 2 * lane; x < q.ne[0]; x += 64) {
                    dot = fmaf(query_values[x], read(k, x, key, kh, kb), dot);
                    if (x + 1 < q.ne[0])
                        dot = fmaf(query_values[x + 1], read(k, x + 1, key, kh, kb), dot);
                }
                for (int shift = 16; shift; shift /= 2)
                    dot += __shfl_down_sync(0xffffffff, dot, shift);
            }
            if (lane == 0) {
                if constexpr (SPARSE) selected_keys[j] = key;
                scores[j] = bias == -CUDART_INF_F ? -CUDART_INF_F : __fmul_rn(dot, scale) + bias;
            }
        }
        __syncthreads();
        if (tid == 0) {
            float next = maximum;
            for (int j = 0; j < TILE; ++j)
                next = fmaxf(next, scores[j]);
            correction = next == -CUDART_INF_F ? 1 : expf(maximum - next);
            float sum = denominator * correction;
            for (int j = 0; j < TILE; ++j) {
                weights[j] = scores[j] == -CUDART_INF_F ? 0 : expf(scores[j] - next);
                sum += weights[j];
            }
            maximum = next;
            denominator = sum;
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < (MAX_VALUE + THREADS - 1) / THREADS; ++i) {
            const int x = tid + i * THREADS;
            if (x < v.ne[0]) {
                float sum = values[i] * correction;
#pragma unroll
                for (int j = 0; j < TILE; ++j)
                    if (weights[j] != 0)
                        sum = fmaf(weights[j], read(v, x, SPARSE ? selected_keys[j] : first + j, vh, vb), sum);
                values[i] = sum;
            }
        }
        __syncthreads();
    }
    float * result = output + row * (v.ne[0] + 2);
#pragma unroll
    for (int i = 0; i < (MAX_VALUE + THREADS - 1) / THREADS; ++i) {
        const int x = tid + i * THREADS;
        if (x < v.ne[0])
            result[x] = values[i];
    }
    if (tid == 0) {
        result[v.ne[0]] = maximum;
        result[v.ne[0] + 1] = denominator;
    }
}

// Decode-class attention whose heads all share one F16 K=V head of width 512:
// DeepSeek V4.1's MLA decode and DSpark verify. partial_kernel gives every head
// its own block, so each one re-reads the same keys (64x the K/V traffic) and
// serializes softmax on one thread. Here a block owns one query, one key
// partition and four heads: each K/V tile is loaded into shared memory once for
// all four, and only the partition's visible keys are loaded at all.
//
// A decode step sees few visible keys, packed into few partitions, so a block
// also splits its own keys over four streams of four warps (one per head):
// stream s takes every fourth 16-key tile of the visible list, keeps its own
// online-softmax state, and the streams merge in stream order at the end. A
// key is scored by a lane pair, each half over 256 dimensions with four
// interleaved F32 accumulators combined in a fixed order; softmax and the V sum
// stay F32. Partition, tile/stream assignment and every reduction order depend
// only on the key extent and the query's own mask, never on the launch width,
// so verify rows equal decode rows bit for bit. Output rows use
// partial_kernel's layout; finish_kernel is shared.
constexpr int MQA_WIDTH = 512, MQA_HEADS = 4, MQA_STREAMS = 4, MQA_TILE = 16, MQA_WINDOW = 128;
constexpr int MQA_STREAM_THREADS = MQA_HEADS * 32, MQA_THREADS = MQA_STREAMS * MQA_STREAM_THREADS;
constexpr int MQA_ROW_BYTES = MQA_WIDTH * 2 + 16;   // 1040: 16-byte loads hit distinct banks
constexpr size_t MQA_TILE_BYTES = size_t(MQA_TILE) * MQA_ROW_BYTES;
constexpr size_t MQA_SHARED = size_t(MQA_HEADS) * MQA_WIDTH * sizeof(float) + MQA_STREAMS * MQA_TILE_BYTES;
static_assert(MQA_TILE_BYTES >= size_t(MQA_HEADS) * (MQA_WIDTH + 2) * sizeof(float), "stream state must fit its tile");

__device__ __forceinline__ void mqa_stream_barrier(int stream) {
    asm volatile("bar.sync %0, %1;" :: "r"(stream + 1), "r"(MQA_STREAM_THREADS) : "memory");
}

// SPARSE: the keys are each query's compacted visible list (TSG_ATTN_MASK_COMPACT),
// as partial_kernel's sparse prefill reads them; an overflowing or exceptional
// row takes the full key range instead, exactly like partial_kernel.
template <bool SPARSE>
__global__ __launch_bounds__(MQA_THREADS) void mqa_partial_kernel(view q, view kv, view mask, view compact, bool masked,
                                                                  float scale, int chunk, int splits, float * output) {
    extern __shared__ __align__(16) unsigned char mqa_shared[];
    float * qs = reinterpret_cast<float *>(mqa_shared);
    __shared__ int keys[MQA_WINDOW];
    __shared__ float biases[MQA_WINDOW];
    __shared__ int warp_counts[MQA_WINDOW / 32];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int head_slot = warp % MQA_HEADS, stream = warp / MQA_HEADS, stream_tid = tid % MQA_STREAM_THREADS;
    unsigned char * tile = mqa_shared + size_t(MQA_HEADS) * MQA_WIDTH * sizeof(float) + stream * MQA_TILE_BYTES;
    const int key_lane = lane & (MQA_TILE - 1), half_index = lane / MQA_TILE;
    const int64_t split = blockIdx.x, h = int64_t(blockIdx.y) * MQA_HEADS + head_slot;
    const int64_t query = blockIdx.z % q.ne[1], batch = blockIdx.z / q.ne[1];
    const int64_t kb = batch / (q.ne[3] / kv.ne[3]);
    const char * base = kv.data + kb * kv.nb[3];
    for (int i = tid; i < MQA_HEADS * MQA_WIDTH; i += MQA_THREADS)
        qs[i] = read(q, i % MQA_WIDTH, query, int64_t(blockIdx.y) * MQA_HEADS + i / MQA_WIDTH, batch);
    const int32_t * list = nullptr;
    int64_t extent = kv.ne[1];
    if constexpr (SPARSE) {
        const auto * candidate = reinterpret_cast<const int32_t *>(compact.data) +
            (query + q.ne[1] * (batch % compact.ne[3])) * compact.ne[0];
        if (candidate[0] >= 0 && candidate[0] < compact.ne[0]) {
            extent = candidate[0];
            list = candidate + 1;
            chunk = int(max(int64_t(16), ((extent + splits - 1) / splits + 15) / 16 * 16));
        }
    }
    // Dense: a key range. Sparse: a range of positions in the query's list.
    const int64_t begin = split * chunk, end = min(extent, begin + chunk);
    float maximum = -CUDART_INF_F, denominator = 0;
    float values[16] = {};
    int tile_index = 0;   // across windows, so stream assignment follows the visible-key order
    for (int64_t window = begin; window < end; window += MQA_WINDOW) {
        __syncthreads();   // the previous window's list is no longer read
        if (warp < MQA_WINDOW / 32) {
            const int64_t entry = window + tid;
            const int64_t key = entry < end && list ? int64_t(list[entry]) : entry;
            const float bias = entry < end ? (masked ? read(mask, key, query, 0, batch % mask.ne[3]) : 0) : -CUDART_INF_F;
            const bool visible = bias != -CUDART_INF_F;
            const unsigned bits = __ballot_sync(0xffffffffu, visible);
            if (lane == 0) warp_counts[warp] = __popc(bits);
            // Each warp stores its own run after the counts are visible.
            asm volatile("bar.sync %0, %1;" :: "r"(MQA_STREAMS + 1), "r"(MQA_WINDOW) : "memory");
            int offset = __popc(bits & ((1u << lane) - 1u));
            for (int i = 0; i < warp; ++i) offset += warp_counts[i];
            if (visible) { keys[offset] = int(key); biases[offset] = bias; }
        }
        __syncthreads();
        int count = 0;
        for (int i = 0; i < MQA_WINDOW / 32; ++i) count += warp_counts[i];
        const int tiles = (count + MQA_TILE - 1) / MQA_TILE;
        for (int t = 0; t < tiles; ++t, ++tile_index) {
            if (tile_index % MQA_STREAMS != stream) continue;
            const int first = t * MQA_TILE, n = min(MQA_TILE, count - first);
            // 64 uint4 per 1 KiB row; the stream's 128 threads fill two rows per pass.
            for (int i = stream_tid; i < MQA_TILE * (MQA_WIDTH / 8); i += MQA_STREAM_THREADS) {
                const int r = i / (MQA_WIDTH / 8), c = i % (MQA_WIDTH / 8);
                uint4 chunk_data = make_uint4(0, 0, 0, 0);
                if (r < n) chunk_data = *reinterpret_cast<const uint4 *>(base + int64_t(keys[first + r]) * kv.nb[1] + c * 16);
                *reinterpret_cast<uint4 *>(tile + r * MQA_ROW_BYTES + c * 16) = chunk_data;
            }
            mqa_stream_barrier(stream);
            float score = -CUDART_INF_F;
            {
                // Lane pair (j, j + 16) scores key j: halves [0, 256) and [256, 512).
                const float * qh = qs + head_slot * MQA_WIDTH + half_index * (MQA_WIDTH / 2);
                const unsigned char * row = tile + key_lane * MQA_ROW_BYTES + half_index * MQA_WIDTH;
                float acc[4] = {};
#pragma unroll 4
                for (int c = 0; c < MQA_WIDTH / 16; ++c) {
                    const uint4 packed = *reinterpret_cast<const uint4 *>(row + c * 16);
                    const half2 * k2 = reinterpret_cast<const half2 *>(&packed);
                    const float4 qa = *reinterpret_cast<const float4 *>(qh + c * 8);
                    const float4 qb = *reinterpret_cast<const float4 *>(qh + c * 8 + 4);
                    const float2 k0 = __half22float2(k2[0]), k1 = __half22float2(k2[1]);
                    const float2 k2v = __half22float2(k2[2]), k3 = __half22float2(k2[3]);
                    acc[0] = fmaf(qa.x, k0.x, acc[0]); acc[0] = fmaf(qa.y, k0.y, acc[0]);
                    acc[1] = fmaf(qa.z, k1.x, acc[1]); acc[1] = fmaf(qa.w, k1.y, acc[1]);
                    acc[2] = fmaf(qb.x, k2v.x, acc[2]); acc[2] = fmaf(qb.y, k2v.y, acc[2]);
                    acc[3] = fmaf(qb.z, k3.x, acc[3]); acc[3] = fmaf(qb.w, k3.y, acc[3]);
                }
                const float part = (acc[0] + acc[1]) + (acc[2] + acc[3]);
                const float other = __shfl_xor_sync(0xffffffffu, part, MQA_TILE);
                const float dot = half_index == 0 ? part + other : other + part;
                if (key_lane < n) score = __fmul_rn(dot, scale) + biases[first + key_lane];
            }
            float tile_max = score;
            for (int shift = MQA_TILE / 2; shift; shift /= 2) tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffffu, tile_max, shift));
            const float next = fmaxf(maximum, tile_max);
            const float correction = next == -CUDART_INF_F ? 1 : expf(maximum - next);
            const float weight = score == -CUDART_INF_F ? 0 : expf(score - next);
            float tile_sum = weight;
            for (int shift = MQA_TILE / 2; shift; shift /= 2) tile_sum += __shfl_xor_sync(0xffffffffu, tile_sum, shift);
            denominator = fmaf(denominator, correction, tile_sum);
            maximum = next;
#pragma unroll
            for (int i = 0; i < 16; ++i) values[i] *= correction;
            for (int j = 0; j < n; ++j) {
                const float w = __shfl_sync(0xffffffffu, weight, j);
                const unsigned char * row = tile + j * MQA_ROW_BYTES;
#pragma unroll
                for (int part = 0; part < 2; ++part) {
                    const uint4 packed = *reinterpret_cast<const uint4 *>(row + (lane + 32 * part) * 16);
                    const half2 * v2 = reinterpret_cast<const half2 *>(&packed);
#pragma unroll
                    for (int p = 0; p < 4; ++p) {
                        const float2 v = __half22float2(v2[p]);
                        values[part * 8 + 2 * p] = fmaf(w, v.x, values[part * 8 + 2 * p]);
                        values[part * 8 + 2 * p + 1] = fmaf(w, v.y, values[part * 8 + 2 * p + 1]);
                    }
                }
            }
            mqa_stream_barrier(stream);   // the stream's next tile overwrites its rows
        }
    }
    // Merge the streams in stream order. Each stream parks its state in its
    // own (now idle) tile buffer; stream 0 combines and writes the partial row.
    __syncthreads();
    float * state = reinterpret_cast<float *>(tile) + head_slot * (MQA_WIDTH + 2);
#pragma unroll
    for (int part = 0; part < 2; ++part)
#pragma unroll
        for (int p = 0; p < 8; ++p) state[(lane + 32 * part) * 8 + p] = values[part * 8 + p];
    if (lane == 0) { state[MQA_WIDTH] = maximum; state[MQA_WIDTH + 1] = denominator; }
    __syncthreads();
    if (stream != 0) return;
    const unsigned char * tiles_base = mqa_shared + size_t(MQA_HEADS) * MQA_WIDTH * sizeof(float);
    float merged_max = -CUDART_INF_F;
    for (int s2 = 0; s2 < MQA_STREAMS; ++s2)
        merged_max = fmaxf(merged_max, reinterpret_cast<const float *>(tiles_base + s2 * MQA_TILE_BYTES)[head_slot * (MQA_WIDTH + 2) + MQA_WIDTH]);
    float merged_den = 0, merged[16] = {};
    for (int s2 = 0; s2 < MQA_STREAMS; ++s2) {
        const float * other = reinterpret_cast<const float *>(tiles_base + s2 * MQA_TILE_BYTES) + head_slot * (MQA_WIDTH + 2);
        const float factor = other[MQA_WIDTH] == -CUDART_INF_F ? 0 : expf(other[MQA_WIDTH] - merged_max);
        merged_den = fmaf(other[MQA_WIDTH + 1], factor, merged_den);
#pragma unroll
        for (int part = 0; part < 2; ++part)
#pragma unroll
            for (int p = 0; p < 8; ++p)
                merged[part * 8 + p] = fmaf(other[(lane + 32 * part) * 8 + p], factor, merged[part * 8 + p]);
    }
    float * result = output + (split + splits * (h + q.ne[2] * (query + q.ne[1] * batch))) * (MQA_WIDTH + 2);
#pragma unroll
    for (int part = 0; part < 2; ++part)
#pragma unroll
        for (int p = 0; p < 8; ++p) result[(lane + 32 * part) * 8 + p] = merged[part * 8 + p];
    if (lane == 0) {
        result[MQA_WIDTH] = merged_max;
        result[MQA_WIDTH + 1] = merged_den;
    }
}

// DeepSeek V4.1's MLA shape: every head shares one F16 K=V head of width 512.
bool mqa_shape(const view & q, const view & k, const view & v) {
    return k.data == v.data && k.type == GGML_TYPE_F16 && k.ne[0] == MQA_WIDTH && v.ne[0] == MQA_WIDTH &&
        q.ne[0] == MQA_WIDTH && k.ne[2] == 1 && v.ne[2] == 1;
}

bool mqa_enabled() {
    // TS_DSV41_MQA_DECODE=0 keeps partial_kernel for A/B and diagnosis.
    static const bool enabled = [] { const char * e = std::getenv("TS_DSV41_MQA_DECODE"); return !e || std::atoi(e) != 0; }();
    return enabled;
}

// The MQA kernel's preconditions; anything else keeps partial_kernel.
bool mqa_eligible(const view & q, const view & k, const view & v, const view & mask, const view & compact, bool masked) {
    return mqa_enabled() && mqa_shape(q, k, v) && k.nb[0] == 2 && k.nb[1] % 16 == 0 &&
        reinterpret_cast<uintptr_t>(k.data) % 16 == 0 && q.ne[2] % MQA_HEADS == 0 && q.ne[2] / MQA_HEADS <= 65535 &&
        q.ne[3] % k.ne[3] == 0 && q.ne[1] * q.ne[3] <= 65535 && (!masked || mask.ne[2] == 1) &&
        (!compact.data || (masked && compact.ne[2] == 1 && compact.ne[1] == q.ne[1]));
}

// A stable, block-local prefix sum emits selected positions in key order.
// Count every visible entry even after capacity is exhausted; consumers then
// take the full-mask fallback for that row. No host readback is required.
__global__ void compact_kernel(view mask, int keys, int queries, int capacity, int32_t * output) {
    __shared__ int count, exceptional, warp_counts[8];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid / 32;
    const int64_t row = blockIdx.x, query = row % queries;
    const int64_t h = row / queries % mask.ne[2], batch = row / (queries * mask.ne[2]);
    int32_t * result = output + row * (int64_t(capacity) + 1);
    if (tid == 0) { count = 0; exceptional = 0; }
    __syncthreads();
    for (int64_t first = 0; first < keys; first += 256) {
        const int64_t key = first + tid;
        const float bias = key < keys ? read(mask, key, query, h, batch) : -CUDART_INF_F;
        const bool selected = bias != -CUDART_INF_F;
        if (selected && !isfinite(bias)) atomicExch(&exceptional, 1);
        const unsigned bits = __ballot_sync(0xffffffff, selected);
        const unsigned before = lane == 0 ? 0 : (0xffffffffu >> (32 - lane));
        int offset = __popc(bits & before);
        if (lane == 0) warp_counts[warp] = __popc(bits);
        __syncthreads();
        for (int i = 0; i < warp; ++i) offset += warp_counts[i];
        offset += count;
        if (selected && offset < capacity) result[offset + 1] = int(key);
        __syncthreads();
        if (tid == 0)
            for (int i = 0; i < 8; ++i) count += warp_counts[i];
        __syncthreads();
    }
    if (tid == 0) result[0] = exceptional ? -1 : count;
}

__global__ void finish_kernel(const float * partial, view sinks, bool has_sinks, int width, int heads, int splits,
                              float * output) {
    __shared__ float scales[64], reciprocal;
    const int64_t row = blockIdx.x;
    const float * p = partial + row * splits * (width + 2);
    if (threadIdx.x == 0) {
        const float sink = has_sinks ? read(sinks, row % heads) : -CUDART_INF_F;
        float maximum = sink;
        for (int s = 0; s < splits; ++s)
            maximum = fmaxf(maximum, p[s * (width + 2) + width]);
        float sum = has_sinks && isfinite(maximum) ? expf(sink - maximum) : 0;
        for (int s = 0; s < splits; ++s) {
            scales[s] = isfinite(maximum) ? expf(p[s * (width + 2) + width] - maximum) : 0;
            sum = fmaf(p[s * (width + 2) + width + 1], scales[s], sum);
        }
        reciprocal = sum > 0 ? 1.0f / sum : 0;
    }
    __syncthreads();
    for (int x = threadIdx.x; x < width; x += blockDim.x) {
        float value = 0;
        for (int s = 0; s < splits; ++s)
            value = fmaf(p[s * (width + 2) + x], scales[s], value);
        output[row * width + x] = value * reciprocal;
    }
}

__global__ void softmax_kernel(view scores, view mask, view sinks, bool masked, bool has_sinks, float scale,
                               float * output) {
    constexpr int THREADS = 256;
    __shared__ float reduction[THREADS];
    const int tid = threadIdx.x;
    const int64_t row = blockIdx.x, query = row % scores.ne[1];
    const int64_t head = (row / scores.ne[1]) % scores.ne[2];
    const int64_t batch = row / (scores.ne[1] * scores.ne[2]);
    float * result = output + row * scores.ne[0];
    float maximum = has_sinks ? read(sinks, head) : -CUDART_INF_F;
    for (int64_t key = tid; key < scores.ne[0]; key += THREADS) {
        const float bias = masked ? read(mask, key, query, head % mask.ne[2], batch % mask.ne[3]) : 0;
        const float value = __fmul_rn(read(scores, key, query, head, batch), scale) + bias;
        result[key] = value;
        maximum = fmaxf(maximum, value);
    }
    reduction[tid] = maximum;
    __syncthreads();
    for (int stride = THREADS / 2; stride; stride /= 2) {
        if (tid < stride)
            reduction[tid] = fmaxf(reduction[tid], reduction[tid + stride]);
        __syncthreads();
    }
    maximum = reduction[0];
    float sum = tid == 0 && has_sinks && isfinite(maximum) ? expf(read(sinks, head) - maximum) : 0;
    for (int64_t key = tid; key < scores.ne[0]; key += THREADS) {
        const float value = isfinite(maximum) ? expf(result[key] - maximum) : 0;
        result[key] = value;
        sum += value;
    }
    __syncthreads();
    reduction[tid] = sum;
    __syncthreads();
    for (int stride = THREADS / 2; stride; stride /= 2) {
        if (tid < stride)
            reduction[tid] += reduction[tid + stride];
        __syncthreads();
    }
    const float reciprocal = reduction[0] > 0 ? 1.0f / reduction[0] : 0;
    for (int64_t key = tid; key < scores.ne[0]; key += THREADS)
        result[key] *= reciprocal;
}
} // namespace

void tsg_attention_cuda_compute(ggml_tensor * dst, ggml_backend_t backend) {
    auto * context = static_cast<ggml_backend_cuda_context *>(backend->context);
    const cudaStream_t stream = context->stream(context->device, 0);
    ggml_custom_op_params params;
    std::memcpy(&params, dst->op_params, sizeof(params));
    const auto & desc = *static_cast<const tsg_dsv4_fused_desc *>(params.userdata);
    CUDA_CHECK(cudaSetDevice(context->device));
    if (desc.kind == TSG_ATTN_VISION_F32) {
        const auto q = tensor_view(dst->src[0]), k = tensor_view(dst->src[1]), v = tensor_view(dst->src[2]);
        GGML_ASSERT((q.ne[0] == 64 || q.ne[0] == 72) && (q.ne[1] + 127) / 128 <= INT_MAX && q.ne[2] <= 65535 && q.ne[3] <= 65535);
        const dim3 blocks(unsigned((q.ne[1] + 127) / 128), unsigned(q.ne[2]), unsigned(q.ne[3]));
        if (q.ne[0] == 64)
            vision_kernel<64><<<blocks, 128, 0, stream>>>(q, k, v, desc.f0, static_cast<float *>(dst->data));
        else
            vision_kernel<72><<<blocks, 128, 0, stream>>>(q, k, v, desc.f0, static_cast<float *>(dst->data));
    } else if (desc.kind == TSG_ATTN_MASK_COMPACT) {
        const int64_t rows = ggml_nelements(dst) / dst->ne[0];
        GGML_ASSERT(rows <= INT_MAX);
        compact_kernel<<<unsigned(rows), 256, 0, stream>>>(tensor_view(dst->src[0]), desc.i0,
            int(dst->ne[1]), int(dst->ne[0] - 1), static_cast<int32_t *>(dst->data));
    } else if (desc.kind == TSG_ATTN_F32_SOFTMAX) {
        const int64_t rows = ggml_nelements(dst) / dst->ne[0];
        GGML_ASSERT(rows <= INT_MAX);
        softmax_kernel<<<unsigned(rows), 256, 0, stream>>>(tensor_view(dst->src[0]), tensor_view(dst->src[1]),
                                                           tensor_view(dst->src[2]), desc.i0 != 0, desc.i1 != 0,
                                                           desc.f0, static_cast<float *>(dst->data));
    } else if (desc.kind == TSG_ATTN_F32_PARTIAL) {
        const auto q = tensor_view(dst->src[0]), k = tensor_view(dst->src[1]), v = tensor_view(dst->src[2]);
        const auto mask = tensor_view(dst->src[3]);
        const auto compact = dst->src[4] ? tensor_view(dst->src[4]) : view{};
        const int64_t rows = dst->ne[1] * dst->ne[2] * dst->ne[3];
        GGML_ASSERT(rows <= INT_MAX);
        const unsigned blocks = unsigned(rows);
        const size_t shared = q.ne[0] * sizeof(float);
        auto * out = static_cast<float *>(dst->data);
        const bool mqa = mqa_eligible(q, k, v, mask, compact, desc.i0 != 0);
        if (!mqa && mqa_enabled() && mqa_shape(q, k, v)) {
            static std::atomic<bool> warned{false};
            if (!warned.exchange(true))
                std::fprintf(stderr, "[dsv41] an MLA attention launch cannot use the MQA kernel (layout: alignment, "
                    "head count or per-head mask); it runs the per-head kernel, several times slower\n");
        }
        if (mqa) {
            // Over 48 KiB of dynamic shared memory needs a per-device opt-in.
            static std::atomic<uint64_t> configured{0};
            const uint64_t bit = uint64_t(1) << (context->device & 63);
            if (!(configured.load() & bit)) {
                CUDA_CHECK(cudaFuncSetAttribute(mqa_partial_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                    int(MQA_SHARED)));
                CUDA_CHECK(cudaFuncSetAttribute(mqa_partial_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                    int(MQA_SHARED)));
                configured.fetch_or(bit);
            }
            const dim3 grid(unsigned(dst->ne[1]), unsigned(q.ne[2] / MQA_HEADS), unsigned(q.ne[1] * q.ne[3]));
            if (compact.data)
                mqa_partial_kernel<true><<<grid, MQA_THREADS, MQA_SHARED, stream>>>(q, k, mask, compact, desc.i0 != 0,
                    desc.f0, desc.i2, int(dst->ne[1]), out);
            else
                mqa_partial_kernel<false><<<grid, MQA_THREADS, MQA_SHARED, stream>>>(q, k, mask, compact, desc.i0 != 0,
                    desc.f0, desc.i2, int(dst->ne[1]), out);
            CUDA_CHECK(cudaGetLastError());
            return;
        }
#define TSG_LAUNCH_ATTN(W)                                                                                             \
    if (compact.data)                                                                                                  \
        partial_kernel<W, true><<<blocks, 128, shared, stream>>>(q, k, v, mask, compact, desc.i0 != 0, desc.f0, desc.i2, int(dst->ne[1]), out); \
    else                                                                                                              \
        partial_kernel<W, false><<<blocks, 128, shared, stream>>>(q, k, v, mask, compact, desc.i0 != 0, desc.f0, desc.i2, int(dst->ne[1]), out)
        if (v.ne[0] <= 128) {
            TSG_LAUNCH_ATTN(128);
        } else if (v.ne[0] <= 256) {
            TSG_LAUNCH_ATTN(256);
        } else if (v.ne[0] <= 512) {
            TSG_LAUNCH_ATTN(512);
        } else if (v.ne[0] <= 1024) {
            TSG_LAUNCH_ATTN(1024);
        } else {
            TSG_LAUNCH_ATTN(2048);
        }
#undef TSG_LAUNCH_ATTN
    } else {
        GGML_ASSERT(desc.kind == TSG_ATTN_F32_FINISH);
        const int64_t rows = ggml_nelements(dst) / dst->ne[0];
        GGML_ASSERT(rows <= INT_MAX && dst->src[0]->ne[1] <= 64);
        finish_kernel<<<unsigned(rows), 128, 0, stream>>>(
            static_cast<const float *>(dst->src[0]->data), tensor_view(dst->src[1]), desc.i1 != 0, int(dst->ne[0]),
            int(dst->ne[1]), int(dst->src[0]->ne[1]), static_cast<float *>(dst->data));
    }
    const cudaError_t launch_status = cudaGetLastError();
    if (launch_status != cudaSuccess) {
        // Preserve the failing owned operation's geometry even after CUDA has
        // entered an error state and pointer-attribute queries are unavailable.
        std::fprintf(stderr, "TensorSharp F32 attention failure: kind=%d device=%d flags=%d,%d,%d scale=%g\n",
                     int(desc.kind), context->device, desc.i0, desc.i1, desc.i2, double(desc.f0));
        for (int i = -1; i < GGML_MAX_SRC; ++i) {
            const ggml_tensor * tensor = i < 0 ? dst : dst->src[i];
            if (!tensor) continue;
            std::fprintf(stderr,
                "  %s%d name=%s type=%d data=%p ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu] view=%p offset=%zu buffer=%s\n",
                i < 0 ? "dst" : "src", i < 0 ? 0 : i, tensor->name, int(tensor->type), tensor->data,
                (long long)tensor->ne[0], (long long)tensor->ne[1], (long long)tensor->ne[2], (long long)tensor->ne[3],
                tensor->nb[0], tensor->nb[1], tensor->nb[2], tensor->nb[3],
                tensor->view_src ? tensor->view_src->data : nullptr, tensor->view_offs,
                tensor->buffer ? ggml_backend_buffer_name(tensor->buffer) : "none");
        }
        std::fflush(stderr);
    }
    CUDA_CHECK(launch_status);
}
