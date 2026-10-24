// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// ---------------------------------------------------------------------------
// MiniMax Sparse Attention CUDA kernels (sm_80+, written for A40 / sm_86).
// Semantics, layouts and the launch plan are in ggml_ops_minimax_msa.h.
//
// One call is up to four launches on the caller's stream:
//   1. score: block maxima of the index scores, F16 mma.sync with F32
//      accumulation. A CTA owns 16/n_head_kv tokens x n_head_kv index heads
//      per warp (64 rows) and a share of the tile's candidate blocks; it never
//      writes per-position scores, only the candidate block maxima F32
//      [t][h][block].
//      Tiles whose tokens are all in the dense regime (pos < topk*block) exit
//      at once: their selection does not need scores.
//   2. top-k: one CTA per (token, index head) selects the forced local blocks
//      plus the best candidates by radix select, ties to the lower block id,
//      and stores the ids ascending with -1 padding (512/1024-thread CTAs
//      when the launch has few rows with many candidates, as in decode).
//   3. attention: one CTA per (4-token tile, KV group, split). Each warp owns
//      one token and packs the group's query heads into the MMA M dimension.
//      The CTA streams the union of its tokens' selected blocks through
//      shared memory (cp.async, two stages) and every warp skips the tiles
//      its token did not select, so a K/V tile is loaded once for all tokens
//      that use it (a one-token tile, as in decode, takes its selection as
//      the union without sorting). Online softmax in F32 (base 2), P*V with
//      F16 operands.
//   4. merge (split-K launches only): log-sum-exp combine of the splits.
// The dense regime runs the same attention kernel with every block selected,
// which is exact dense causal attention.
// ---------------------------------------------------------------------------

#include "ggml_ops_minimax_msa.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <climits>
#include <cstdio>
#include <cstring>

#ifdef TSG_GGML_USE_CUDA
#include "ggml_ops_dsv4_fused.h"

#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "ggml-cuda.h"
#include "ggml-cuda/common.cuh"   // ggml_backend_cuda_context (device, stream)
#endif

#ifdef TSG_MSA_TEST_HOOKS
// Phase boundaries of the last launch: start, scores, selection, attention, merge.
cudaEvent_t tsg_msa_test_phase_events[5] = {};
#endif

namespace {

constexpr int kD = 128;                  // head_dim (main and index)
constexpr int kRow = kD + 8;             // shared tile row pitch in halves: 272 B, conflict-free ldmatrix
constexpr int kScoreWarps = tsg_msa_score_warps;
constexpr int kAttnWarps = tsg_msa_attn_tokens;   // tokens per attention CTA
constexpr int kTopkThreads = 128;
constexpr int kTopkWideRows = 64;        // launches of at most this many (token, index head) rows (decode)
                                         // size the CTA to the candidates (512 or 1024 threads)
constexpr int kTopkSmemFloats = 8192;    // candidates ranked in shared memory (larger rows rank in place)
constexpr int kScoreCtas = 512;          // score grid target
constexpr unsigned kFull = 0xffffffffu;

__device__ __forceinline__ uint32_t smem_addr(const void * p)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// cp.async and the m16n8k16 F16 MMA are sm_80 instructions. A build whose
// architecture list also names an older target (the release archives start at
// sm_75) compiles these helpers to a trap there; the launcher never starts a
// kernel on such a device (msa_device_supported, status -4), and the executor
// does not route sparse attention to it (tsg_msa_cuda_capability).
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
#define TSG_MSA_SM80_DEVICE_CODE 0
#else
#define TSG_MSA_SM80_DEVICE_CODE 1
#endif

__device__ __forceinline__ void cp_async16(void * smem, const void * gmem)
{
#if TSG_MSA_SM80_DEVICE_CODE
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_addr(smem)), "l"(gmem) : "memory");
#else
    (void) smem; (void) gmem;
    __trap();
#endif
}

__device__ __forceinline__ void cp_async_commit()
{
#if TSG_MSA_SM80_DEVICE_CODE
    asm volatile("cp.async.commit_group;\n" ::: "memory");
#else
    __trap();
#endif
}

template <int N>
__device__ __forceinline__ void cp_async_wait()
{
#if TSG_MSA_SM80_DEVICE_CODE
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
#else
    __trap();
#endif
}

__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], const half * p)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_addr(p)));
}

__device__ __forceinline__ void ldsm_x4_t(uint32_t (&r)[4], const half * p)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_addr(p)));
}

// D = A*B + D, m16n8k16, F16 operands, F32 accumulation.
__device__ __forceinline__ void mma_f16(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1)
{
#if TSG_MSA_SM80_DEVICE_CODE
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
    (void) c; (void) a; (void) b0; (void) b1;
    __trap();
#endif
}

__device__ __forceinline__ uint32_t pack_h2(const __half2 h)
{
    return *reinterpret_cast<const uint32_t *>(&h);
}

// Rows [key0, key0+KT) x columns [col0, col0+128) of an F16 [rows][pitch]
// matrix into a [KT][kRow] shared tile. Rows past `limit` are zero filled:
// they are masked, but a stale NaN there would still poison P*V (0 * NaN).
template <int KT, int NT>
__device__ __forceinline__ void load_tile(half * dst, const half * __restrict__ src, const int64_t pitch,
                                          const int col0, const int key0, const int limit, const int tid)
{
#pragma unroll
    for (int c = tid; c < KT * 16; c += NT)
    {
        const int r = c >> 4, part = (c & 15) * 8;
        half * d = dst + r * kRow + part;
        const int key = key0 + r;
        if (key <= limit) cp_async16(d, src + (int64_t) key * pitch + col0 + part);
        else *reinterpret_cast<uint4 *>(d) = make_uint4(0, 0, 0, 0);
    }
}

// Candidate blocks of a token that needs scores: [0, ce). Dense-regime tokens
// select every visible block and need none (ce = 0).
__device__ __forceinline__ int candidate_end(const int p, const int block, const int topk, const int local)
{
    const int own = p / block;
    return own + 1 <= topk ? 0 : own + 1 - min(local, own + 1);
}

// ---------------------------------------------------------------------------
// 1. Block scores
// ---------------------------------------------------------------------------

template <int KT, bool PRECISE>
__global__ void __launch_bounds__(kScoreWarps * 32)
msa_score_kernel(const float * __restrict__ iq, const half * __restrict__ idx, const int64_t idx_pitch,
                 const int32_t * __restrict__ pos, float * __restrict__ scores,
                 const int nt, const int H, const int block, const int topk, const int local,
                 const int rows, const int nblk)
{
    constexpr int NT = kScoreWarps * 32;
    __shared__ __align__(16) half s_k[2][KT * kRow];
    __shared__ int s_hi, s_maxpos;

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, c = lane & 3;
    const int tpw = 16 / H, tpc = kScoreWarps * tpw;
    const int t_base = blockIdx.x * tpc;

    if (tid == 0) { s_hi = 0; s_maxpos = -1; }
    __syncthreads();
    if (tid < tpc && t_base + tid < nt)
    {
        const int p = pos[t_base + tid];
        atomicMax(&s_hi, candidate_end(p, block, topk, local));
        atomicMax(&s_maxpos, p);
    }
    __syncthreads();
    // The grid's y CTAs split this tile's candidate blocks, so the work tracks
    // the live positions, not rows_stride.
    const int span = (s_hi + (int) gridDim.y - 1) / (int) gridDim.y;
    const int b0 = blockIdx.y * span;
    const int b1 = min(b0 + span, s_hi);
    if (b0 >= b1) return;
    const int limit = min(s_maxpos, rows - 1);

    // This thread's MMA rows g and g+8: (token, index head) pairs.
    int tok[2], head[2], prow[2], cend[2];
#pragma unroll
    for (int r = 0; r < 2; ++r)
    {
        const int ri = g + 8 * r, tl = ri / H;
        tok[r] = t_base + warp * tpw + tl;
        head[r] = ri % H;
        const bool valid = ri < tpw * H && tok[r] < nt;
        prow[r] = valid ? pos[tok[r]] : -1;
        cend[r] = valid ? candidate_end(prow[r], block, topk, local) : 0;
    }
    const bool active = __any_sync(kFull, cend[0] > b0 || cend[1] > b0);

    uint32_t ahi[8][4];
    uint32_t alo[PRECISE ? 8 : 1][4];
#pragma unroll
    for (int kk = 0; kk < 8; ++kk)
    {
#pragma unroll
        for (int e = 0; e < 4; ++e)
        {
            const int r = e & 1;
            const int col = kk * 16 + 2 * c + (e >> 1) * 8;
            float2 x = make_float2(0.0f, 0.0f);
            if (prow[r] >= 0)
                x = *reinterpret_cast<const float2 *>(iq + ((int64_t) tok[r] * H + head[r]) * kD + col);
            const __half2 hi = __floats2half2_rn(x.x, x.y);
            ahi[kk][e] = pack_h2(hi);
            if constexpr (PRECISE)
                alo[kk][e] = pack_h2(__floats2half2_rn(x.x - __low2float(hi), x.y - __high2float(hi)));
        }
    }

    const int spb = block / KT;
    const int nsub = (b1 - b0) * spb;
    load_tile<KT, NT>(s_k[0], idx, idx_pitch, 0, b0 * block, limit, tid);
    cp_async_commit();

    float run0 = -INFINITY, run1 = -INFINITY;
    for (int i = 0; i < nsub; ++i)
    {
        if (i + 1 < nsub)
        {
            const int nb = b0 + (i + 1) / spb, ns = (i + 1) % spb;
            load_tile<KT, NT>(s_k[(i + 1) & 1], idx, idx_pitch, 0, nb * block + ns * KT, limit, tid);
        }
        cp_async_commit();
        cp_async_wait<1>();
        __syncthreads();

        if (active)
        {
            const int blk = b0 + i / spb, sub = i % spb;
            const half * ks = s_k[i & 1];
            float acc[KT / 8][4];
#pragma unroll
            for (int j = 0; j < KT / 8; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.0f;
            const int mi = lane >> 3, rr = lane & 7;
#pragma unroll
            for (int kk = 0; kk < 8; ++kk)
            {
#pragma unroll
                for (int np = 0; np < KT / 16; ++np)
                {
                    uint32_t b[4];
                    ldsm_x4(b, ks + (np * 16 + (mi >> 1) * 8 + rr) * kRow + kk * 16 + (mi & 1) * 8);
                    mma_f16(acc[2 * np], ahi[kk], b[0], b[1]);
                    mma_f16(acc[2 * np + 1], ahi[kk], b[2], b[3]);
                    if constexpr (PRECISE)
                    {
                        mma_f16(acc[2 * np], alo[kk], b[0], b[1]);
                        mma_f16(acc[2 * np + 1], alo[kk], b[2], b[3]);
                    }
                }
            }
            const int key0 = blk * block + sub * KT;
#pragma unroll
            for (int j = 0; j < KT / 8; ++j)
            {
                const int key = key0 + j * 8 + 2 * c;
                run0 = fmaxf(run0, key     <= prow[0] ? acc[j][0] : -INFINITY);
                run0 = fmaxf(run0, key + 1 <= prow[0] ? acc[j][1] : -INFINITY);
                run1 = fmaxf(run1, key     <= prow[1] ? acc[j][2] : -INFINITY);
                run1 = fmaxf(run1, key + 1 <= prow[1] ? acc[j][3] : -INFINITY);
            }
            if (sub == spb - 1)
            {
                run0 = fmaxf(run0, __shfl_xor_sync(kFull, run0, 1));
                run0 = fmaxf(run0, __shfl_xor_sync(kFull, run0, 2));
                run1 = fmaxf(run1, __shfl_xor_sync(kFull, run1, 1));
                run1 = fmaxf(run1, __shfl_xor_sync(kFull, run1, 2));
                if (c == 0)
                {
                    if (blk < cend[0]) scores[((int64_t) tok[0] * H + head[0]) * nblk + blk] = run0;
                    if (blk < cend[1]) scores[((int64_t) tok[1] * H + head[1]) * nblk + blk] = run1;
                }
                run0 = run1 = -INFINITY;
            }
        }
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// 2. Selection
// ---------------------------------------------------------------------------

// Exclusive scan of two values per thread over the whole CTA; returns this
// thread's base and writes the total to *total (all threads).
template <int NT>
__device__ __forceinline__ int cta_scan2(const int v0, const int v1, int * s_warp, int * total)
{
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int x = v0 + v1;
#pragma unroll
    for (int off = 1; off < 32; off <<= 1)
    {
        const int y = __shfl_up_sync(kFull, x, off);
        if (lane >= off) x += y;
    }
    if (lane == 31) s_warp[warp] = x;
    __syncthreads();
    int base = 0, sum = 0;
#pragma unroll
    for (int w = 0; w < NT / 32; ++w)
    {
        base += w < warp ? s_warp[w] : 0;
        sum += s_warp[w];
    }
    __syncthreads();
    *total = sum;
    return base + x - (v0 + v1);
}

// Block score -> radix key: a larger score is a larger key, -0 and +0 are one
// key (they compare equal), and NaN is 0, below every other key, so it is
// never selected.
__device__ __forceinline__ uint32_t score_key(const float v)
{
    if (v != v) return 0u;
    const uint32_t b = v == 0.0f ? 0u : __float_as_uint(v);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

// One CTA per (token, index head). The forced local blocks are taken as they
// are; the best r = topk - forced candidates come from a radix select over
// the candidates' keys (8 bits a pass, most significant first, stopping once
// the boundary digit's bucket is taken whole), and the candidates that share
// the boundary prefix are ranked in block order, so ties go to the lower
// block id. Ids are stored ascending with -1 padding (the attention kernel's
// one-token path relies on that order). NT is kTopkThreads, or 512 / 1024
// when the launch has too few rows to fill the GPU (decode) and the rows have
// enough candidates to repay the wider barriers: a pass is then about one
// candidate per thread instead of a loop.
template <int NT>
__global__ void __launch_bounds__(NT)
msa_topk_kernel(const float * __restrict__ scores, const int nblk, const int32_t * __restrict__ pos,
                int32_t * __restrict__ sel, const int H, const int block, const int topk, const int local,
                const int cap)
{
    static_assert(NT >= 64 && NT % 32 == 0 && NT <= 1024, "a thread per chosen id, whole warps");
    extern __shared__ float s_dyn[];
    __shared__ int s_hist[256];
    __shared__ int s_chosen[64];
    __shared__ int s_warp[NT / 32];
    __shared__ int s_count, s_digit, s_need, s_done;

    const int row = blockIdx.x, t = row / H, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int p = pos[t], own = p / block, nv = own + 1;
    int32_t * o = sel + (int64_t) row * topk;
    if (nv <= topk)
    {
        for (int j = tid; j < topk; j += NT) o[j] = j < nv ? j : -1;
        return;
    }

    const int nf = min(local, nv), ce = nv - nf, r = topk - nf;
    const float * buf = scores + (int64_t) row * nblk;
    if (ce <= cap)
    {
        for (int b = tid; b < ce; b += NT) s_dyn[b] = buf[b];
        buf = s_dyn;
    }
    if (tid == 0) { s_count = 0; s_done = 0; }

    // Invariant: the selection is every valid key whose masked bits exceed
    // `prefix`, plus the first `need` (in block order) whose masked bits equal it.
    uint32_t prefix = 0, mask = 0;
    int need = r;
    for (int shift = 24; shift >= 0 && r > 0; shift -= 8)
    {
        for (int i = tid; i < 256; i += NT) s_hist[i] = 0;
        __syncthreads();
        // Warp-uniform trip count: __match_any_sync needs every lane.
        for (int b0 = 0; b0 < ce; b0 += NT)
        {
            const int b = b0 + tid;
            const uint32_t k = b < ce ? score_key(buf[b]) : 0u;
            const bool in = k != 0u && (k & mask) == prefix;
            const int digit = in ? (int) ((k >> shift) & 255u) : -1;
            // Similar scores share their top digits: one atomic per distinct
            // digit of the warp instead of one per lane.
            const unsigned same = __match_any_sync(kFull, digit);
            if (in && lane == __ffs(same) - 1) atomicAdd(&s_hist[digit], __popc(same));
        }
        __syncthreads();
        if (warp == 0)
        {
            // Lane l owns digits 255-8l down to 248-8l; `above` counts the keys
            // in higher digits.
            int c[8], sum = 0;
#pragma unroll
            for (int j = 0; j < 8; ++j) { c[j] = s_hist[255 - 8 * lane - j]; sum += c[j]; }
            int incl = sum;
#pragma unroll
            for (int off = 1; off < 32; off <<= 1)
            {
                const int y = __shfl_up_sync(kFull, incl, off);
                if (lane >= off) incl += y;
            }
            const int total = __shfl_sync(kFull, incl, 31);
            int above = incl - sum;
            if (total <= need)
            {
                // Every remaining key is taken (fewer valid candidates than
                // slots happens only with NaN scores).
                if (lane == 0) { s_digit = -1; s_need = total; s_done = 1; }
            }
            else if (above < need && need <= above + sum)
            {
#pragma unroll
                for (int j = 0; j < 8; ++j)
                {
                    if (above + c[j] >= need)
                    {
                        s_digit = 255 - 8 * lane - j;
                        s_need = need - above;
                        s_done = above + c[j] == need;
                        break;
                    }
                    above += c[j];
                }
            }
        }
        __syncthreads();
        if (s_digit >= 0)
        {
            prefix |= (uint32_t) s_digit << shift;
            mask |= 255u << shift;
        }
        need = s_need;
        if (s_done) break;
    }

    if (r > 0)
    {
        // Rank the boundary keys in block order: each thread owns a contiguous run.
        const int per = (ce + NT - 1) / NT, lo = min(ce, tid * per), hi = min(ce, lo + per);
        int n_eq = 0;
        for (int b = lo; b < hi; ++b)
        {
            const uint32_t k = score_key(buf[b]);
            n_eq += k != 0u && (k & mask) == prefix;
        }
        int n_eq_total;
        int rank = cta_scan2<NT>(n_eq, 0, s_warp, &n_eq_total);
        for (int b = lo; b < hi; ++b)
        {
            const uint32_t k = score_key(buf[b]);
            if (k == 0u) continue;
            const uint32_t km = k & mask;
            bool take = km > prefix;
            if (km == prefix) take = rank++ < need;
            if (take) s_chosen[atomicAdd(&s_count, 1)] = b;
        }
    }
    __syncthreads();

    // Candidates all precede the forced blocks, so the ascending list is the
    // sorted candidates followed by the forced run.
    const int cnt = s_count;
    if (tid < cnt)
    {
        const int id = s_chosen[tid];
        int rk = 0;
        for (int j = 0; j < cnt; ++j) rk += s_chosen[j] < id;
        o[rk] = id;
    }
    for (int j = cnt + tid; j < topk; j += NT)
        o[j] = j - cnt < nf ? own - nf + 1 + (j - cnt) : -1;
}

// ---------------------------------------------------------------------------
// 3. Block-sparse attention
// ---------------------------------------------------------------------------

template <int KT>
__global__ void __launch_bounds__(kAttnWarps * 32)
msa_attn_kernel(const float * __restrict__ q, const half * __restrict__ kc, const half * __restrict__ vc,
                const int64_t k_pitch, const int64_t v_pitch, const int32_t * __restrict__ pos,
                const int32_t * __restrict__ sel, float * __restrict__ out, float * __restrict__ part,
                const int nt, const int n_head, const int H, const int block, const int topk, const int rows,
                const float scale_log2, const int nsplit)
{
    constexpr int NW = kAttnWarps, NT = NW * 32, NE = NW * 64;
    extern __shared__ __align__(16) unsigned char s_raw[];
    half * s_kv = reinterpret_cast<half *>(s_raw);   // [stage][K | V][KT][kRow]
    __shared__ int s_key[NE];
    __shared__ int s_ublk[NE];
    __shared__ int s_umask[NE];
    __shared__ int s_uoff[NE + 1];
    __shared__ int s_pos[NW];
    __shared__ int s_warp[NW];

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, c = lane & 3;
    const int t0 = blockIdx.x * NW, grp = blockIdx.y, split = blockIdx.z;
    const int G = n_head / H;
    const int spb = block / KT;

    if (tid < NW) s_pos[tid] = t0 + tid < nt ? pos[t0 + tid] : -1;

    int n_union;
    if (t0 + 1 >= nt)
    {
        // One token (decode, a ragged last tile): its selection already is
        // the union, ascending and distinct with -1 padding (msa_topk_kernel).
        static_assert(NT >= 64, "a thread per selected block");
        const int b = tid < topk ? sel[((int64_t) t0 * H + grp) * topk + tid] : -1;
        if (b >= 0) { s_ublk[tid] = b; s_umask[tid] = 1; }
        n_union = __syncthreads_count(b >= 0);
    }
    else
    {
        // Union of the tile's selections: sort (block << 4 | warp), then one entry
        // per distinct block with a bit per token that selected it.
        const int ne = NW * topk;
        int np2 = 1;
        while (np2 < ne) np2 <<= 1;
        for (int e = tid; e < np2; e += NT)
        {
            int key = INT_MAX;
            if (e < ne)
            {
                const int w = e / topk, t = t0 + w;
                if (t < nt)
                {
                    const int b = sel[((int64_t) t * H + grp) * topk + e % topk];
                    if (b >= 0) key = (b << 4) | w;
                }
            }
            s_key[e] = key;
        }
        __syncthreads();
        for (int size = 2; size <= np2; size <<= 1)
        {
            for (int stride = size >> 1; stride > 0; stride >>= 1)
            {
                for (int i = tid; i < np2 / 2; i += NT)
                {
                    const int lo = 2 * i - (i & (stride - 1)), hi = lo + stride;
                    const bool asc = (lo & size) == 0;
                    const int a = s_key[lo], b = s_key[hi];
                    if ((a > b) == asc) { s_key[lo] = b; s_key[hi] = a; }
                }
                __syncthreads();
            }
        }
        int head[2];
#pragma unroll
        for (int r = 0; r < 2; ++r)
        {
            const int i = 2 * tid + r;
            head[r] = i < np2 && s_key[i] != INT_MAX && (i == 0 || (s_key[i] >> 4) != (s_key[i - 1] >> 4));
        }
        int u = cta_scan2<NT>(head[0], head[1], s_warp, &n_union);
#pragma unroll
        for (int r = 0; r < 2; ++r)
        {
            if (!head[r]) continue;
            const int i = 2 * tid + r, blk = s_key[i] >> 4;
            int mask = 0;
            for (int k = i; k < np2 && s_key[k] != INT_MAX && (s_key[k] >> 4) == blk; ++k) mask |= 1 << (s_key[k] & 15);
            s_ublk[u] = blk;
            s_umask[u] = mask;
            ++u;
        }
    }
    int maxpos = -1;
#pragma unroll
    for (int w = 0; w < NW; ++w) maxpos = max(maxpos, s_pos[w]);
    const int limit = min(maxpos, rows - 1);
    __syncthreads();

    // Shared-memory tiles per union block: only those that start at or before
    // the tile's last position (the own block's future tail is skipped).
    int cnt[2];
#pragma unroll
    for (int r = 0; r < 2; ++r)
    {
        const int i = 2 * tid + r;
        cnt[r] = 0;
        if (i < n_union)
        {
            const int first = s_ublk[i] * block;
            cnt[r] = first <= limit ? min(spb, (limit - first) / KT + 1) : 0;
        }
    }
    int n_sub;
    const int off = cta_scan2<NT>(cnt[0], cnt[1], s_warp, &n_sub);
    if (2 * tid < n_union) s_uoff[2 * tid] = off;
    if (2 * tid + 1 < n_union) s_uoff[2 * tid + 1] = off + cnt[0];
    __syncthreads();

    const int per = (n_sub + nsplit - 1) / nsplit;
    const int i0 = min(n_sub, split * per), i1 = min(n_sub, i0 + per);

    auto locate = [&](const int i, int & key0, int & mask)
    {
        int lo = 0, hi = n_union - 1;
        while (lo < hi)
        {
            const int mid = (lo + hi + 1) >> 1;
            if (s_uoff[mid] <= i) lo = mid; else hi = mid - 1;
        }
        key0 = s_ublk[lo] * block + (i - s_uoff[lo]) * KT;
        mask = s_umask[lo];
    };
    auto stage = [&](const int s) { return s_kv + s * 2 * KT * kRow; };

    const int t = t0 + warp;
    const bool tvalid = t < nt;
    const int p = tvalid ? s_pos[warp] : -1;

    // Query fragments: rows g, g+8 = the group's heads (rows >= G are zero).
    uint32_t qa[8][4];
#pragma unroll
    for (int kk = 0; kk < 8; ++kk)
    {
#pragma unroll
        for (int e = 0; e < 4; ++e)
        {
            const int row = g + 8 * (e & 1);
            const int col = kk * 16 + 2 * c + (e >> 1) * 8;
            float2 x = make_float2(0.0f, 0.0f);
            if (tvalid && row < G)
                x = *reinterpret_cast<const float2 *>(q + ((int64_t) t * n_head + grp * G + row) * kD + col);
            qa[kk][e] = pack_h2(__floats2half2_rn(x.x, x.y));
        }
    }

    float o[16][4];
#pragma unroll
    for (int n = 0; n < 16; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.0f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;

    if (i0 < i1)
    {
        int key0, mask;
        locate(i0, key0, mask);
        load_tile<KT, NT>(stage(0), kc, k_pitch, grp * kD, key0, limit, tid);
        load_tile<KT, NT>(stage(0) + KT * kRow, vc, v_pitch, grp * kD, key0, limit, tid);
    }
    cp_async_commit();

    const int mi = lane >> 3, rr = lane & 7;
    for (int i = i0; i < i1; ++i)
    {
        if (i + 1 < i1)
        {
            int key0n, maskn;
            locate(i + 1, key0n, maskn);
            half * st = stage((i + 1 - i0) & 1);
            load_tile<KT, NT>(st, kc, k_pitch, grp * kD, key0n, limit, tid);
            load_tile<KT, NT>(st + KT * kRow, vc, v_pitch, grp * kD, key0n, limit, tid);
        }
        cp_async_commit();
        cp_async_wait<1>();
        __syncthreads();

        int key0, mask;
        locate(i, key0, mask);
        if (tvalid && ((mask >> warp) & 1) && key0 <= p)
        {
            const half * ks = stage((i - i0) & 1);
            const half * vs = ks + KT * kRow;

            float s[KT / 8][4];
#pragma unroll
            for (int j = 0; j < KT / 8; ++j) s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.0f;
#pragma unroll
            for (int kk = 0; kk < 8; ++kk)
            {
#pragma unroll
                for (int np = 0; np < KT / 16; ++np)
                {
                    uint32_t b[4];
                    ldsm_x4(b, ks + (np * 16 + (mi >> 1) * 8 + rr) * kRow + kk * 16 + (mi & 1) * 8);
                    mma_f16(s[2 * np], qa[kk], b[0], b[1]);
                    mma_f16(s[2 * np + 1], qa[kk], b[2], b[3]);
                }
            }

            float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
            for (int j = 0; j < KT / 8; ++j)
            {
                const int key = key0 + j * 8 + 2 * c;
                s[j][0] = key     <= p ? s[j][0] * scale_log2 : -INFINITY;
                s[j][1] = key + 1 <= p ? s[j][1] * scale_log2 : -INFINITY;
                s[j][2] = key     <= p ? s[j][2] * scale_log2 : -INFINITY;
                s[j][3] = key + 1 <= p ? s[j][3] * scale_log2 : -INFINITY;
                mx0 = fmaxf(mx0, fmaxf(s[j][0], s[j][1]));
                mx1 = fmaxf(mx1, fmaxf(s[j][2], s[j][3]));
            }
            mx0 = fmaxf(mx0, __shfl_xor_sync(kFull, mx0, 1));
            mx0 = fmaxf(mx0, __shfl_xor_sync(kFull, mx0, 2));
            mx1 = fmaxf(mx1, __shfl_xor_sync(kFull, mx1, 1));
            mx1 = fmaxf(mx1, __shfl_xor_sync(kFull, mx1, 2));
            // key0 <= p, so every row has a visible key here and mn is finite.
            const float mn0 = fmaxf(m0, mx0), mn1 = fmaxf(m1, mx1);
            const float a0 = exp2f(m0 - mn0), a1 = exp2f(m1 - mn1);
            m0 = mn0;
            m1 = mn1;

            uint32_t pa[KT / 16][4];
            float rs0 = 0.0f, rs1 = 0.0f;
#pragma unroll
            for (int j = 0; j < KT / 8; ++j)
            {
                const float p00 = exp2f(s[j][0] - mn0), p01 = exp2f(s[j][1] - mn0);
                const float p10 = exp2f(s[j][2] - mn1), p11 = exp2f(s[j][3] - mn1);
                rs0 += p00 + p01;
                rs1 += p10 + p11;
                pa[j >> 1][(j & 1) * 2 + 0] = pack_h2(__floats2half2_rn(p00, p01));
                pa[j >> 1][(j & 1) * 2 + 1] = pack_h2(__floats2half2_rn(p10, p11));
            }
            l0 = l0 * a0 + rs0;
            l1 = l1 * a1 + rs1;
#pragma unroll
            for (int n = 0; n < 16; ++n)
            {
                o[n][0] *= a0; o[n][1] *= a0;
                o[n][2] *= a1; o[n][3] *= a1;
            }
#pragma unroll
            for (int kk = 0; kk < KT / 16; ++kk)
            {
#pragma unroll
                for (int np = 0; np < 8; ++np)
                {
                    uint32_t b[4];
                    ldsm_x4_t(b, vs + (kk * 16 + (mi & 1) * 8 + rr) * kRow + np * 16 + (mi >> 1) * 8);
                    mma_f16(o[2 * np], pa[kk], b[0], b[1]);
                    mma_f16(o[2 * np + 1], pa[kk], b[2], b[3]);
                }
            }
        }
        __syncthreads();
    }

    l0 += __shfl_xor_sync(kFull, l0, 1);
    l0 += __shfl_xor_sync(kFull, l0, 2);
    l1 += __shfl_xor_sync(kFull, l1, 1);
    l1 += __shfl_xor_sync(kFull, l1, 2);
    if (!tvalid) return;

#pragma unroll
    for (int r = 0; r < 2; ++r)
    {
        const int row = g + 8 * r;
        if (row >= G) continue;
        const int64_t qrow = (int64_t) t * n_head + grp * G + row;
        const float m = r ? m1 : m0, l = r ? l1 : l0;
        if (nsplit == 1)
        {
            const float inv = l > 0.0f ? 1.0f / l : 0.0f;
            float * dst = out + qrow * kD;
#pragma unroll
            for (int n = 0; n < 16; ++n)
                *reinterpret_cast<float2 *>(dst + n * 8 + 2 * c) = make_float2(o[n][2 * r] * inv, o[n][2 * r + 1] * inv);
        }
        else
        {
            float * dst = part + (qrow * nsplit + split) * (kD + 2);
#pragma unroll
            for (int n = 0; n < 16; ++n)
                *reinterpret_cast<float2 *>(dst + n * 8 + 2 * c) = make_float2(o[n][2 * r], o[n][2 * r + 1]);
            if (c == 0) { dst[kD] = m; dst[kD + 1] = l; }
        }
    }
}

// ---------------------------------------------------------------------------
// 4. Split-K merge
// ---------------------------------------------------------------------------

// One CTA per query row, one thread per dim. The split weights come from one
// parallel pass over the (m, l) pairs, and the partial outputs are read with
// independent loads: a loop of dependent reads over 32 splits is 32 L2 trips.
__global__ void __launch_bounds__(kD)
msa_merge_kernel(const float * __restrict__ part, float * __restrict__ out, const int nsplit)
{
    static_assert(tsg_msa_max_split <= 2 * kD, "two splits per thread");
    __shared__ float s_w[tsg_msa_max_split];
    __shared__ float s_red[2][kD / 32];
    const int64_t row = blockIdx.x;
    const int d = threadIdx.x, lane = d & 31, warp = d >> 5;
    const float * pp = part + row * nsplit * (kD + 2);

    float m_own[2], mx = -INFINITY;
#pragma unroll
    for (int i = 0; i < 2; ++i)
    {
        const int s = d + i * kD;
        m_own[i] = s < nsplit ? pp[(int64_t) s * (kD + 2) + kD] : -INFINITY;
        mx = fmaxf(mx, m_own[i]);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(kFull, mx, off));
    if (lane == 0) s_red[0][warp] = mx;
    __syncthreads();
#pragma unroll
    for (int w = 0; w < kD / 32; ++w) mx = fmaxf(mx, s_red[0][w]);

    // Empty splits (m = -inf, l = 0, zero output) weigh 0, also when every
    // split is empty.
    float l = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i)
    {
        const int s = d + i * kD;
        if (s >= nsplit) continue;
        const float w = m_own[i] == -INFINITY ? 0.0f : exp2f(m_own[i] - mx);
        s_w[s] = w;
        l += w * pp[(int64_t) s * (kD + 2) + kD + 1];
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) l += __shfl_xor_sync(kFull, l, off);
    if (lane == 0) s_red[1][warp] = l;
    __syncthreads();
    l = 0.0f;
#pragma unroll
    for (int w = 0; w < kD / 32; ++w) l += s_red[1][w];

    float acc = 0.0f;
#pragma unroll 8
    for (int s = 0; s < nsplit; ++s) acc += s_w[s] * pp[(int64_t) s * (kD + 2) + d];
    out[row * kD + d] = l > 0.0f ? acc / l : 0.0f;
}

// ---------------------------------------------------------------------------
// Launcher
// ---------------------------------------------------------------------------

template <int KT>
size_t attn_smem_bytes()
{
    return size_t(2) * 2 * KT * kRow * sizeof(half);
}

// Dynamic shared memory above 48 KiB needs a per-device opt-in.
template <int KT>
bool attn_prepare()
{
    constexpr size_t bytes = size_t(2) * 2 * KT * kRow * sizeof(half);
    if (bytes <= 48 * 1024) return true;
    static std::atomic<uint64_t> configured{0};
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) return false;
    const uint64_t bit = uint64_t(1) << (device & 63);
    if (configured.load() & bit) return true;
    if (cudaFuncSetAttribute(msa_attn_kernel<KT>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(bytes)) != cudaSuccess)
        return false;
    configured.fetch_or(bit);
    return true;
}

// Score CTAs of one variant that the current device holds at once (one wave),
// cached per device; 0 when the occupancy query fails.
template <int KT, bool PRECISE>
int score_resident_ctas()
{
    static std::atomic<int> cache[64] = {};
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) return 0;
    std::atomic<int> & slot = cache[device & 63];
    int n = slot.load(std::memory_order_relaxed);
    if (n > 0) return n;
    int sms = 0, per_sm = 0;
    if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device) != cudaSuccess ||
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, msa_score_kernel<KT, PRECISE>, kScoreWarps * 32, 0)
            != cudaSuccess)
    {
        (void) cudaGetLastError();
        return 0;
    }
    n = sms * per_sm;
    slot.store(n, std::memory_order_relaxed);
    return n;
}

template <int KT, bool PRECISE>
void launch_score(cudaStream_t stream, const float * iq, const half * idx, int64_t idx_pitch,
                  const int32_t * pos, float * scores, int n, const tsg_msa_params & p, int nblk)
{
    // Many tiles (prefill): ~kScoreCtas CTAs, a few waves of compute-bound
    // MMA work. Few tiles (decode, short chunks): the index-K stream is
    // bandwidth-bound, so one full wave that splits each tile's blocks; more
    // CTAs only add a partial last wave and per-CTA pipeline warm-up.
    const int tpc = kScoreWarps * (16 / p.n_head_kv);
    const int tiles = (n + tpc - 1) / tpc;
    const int resident = score_resident_ctas<KT, PRECISE>();
    int chunks = resident > 0 && tiles * 4 <= resident ? resident / tiles : std::max(1, kScoreCtas / tiles);
    chunks = std::max(1, std::min(chunks, nblk));
    msa_score_kernel<KT, PRECISE><<<dim3((unsigned) tiles, (unsigned) chunks), kScoreWarps * 32, 0, stream>>>(
        iq, idx, idx_pitch, pos, scores, n, p.n_head_kv, p.block, p.topk, p.local_blocks, p.rows_stride, nblk);
}

template <int KT>
bool launch_attn(dim3 grid, cudaStream_t stream, const float * q, const half * k, const half * v,
                 int64_t k_pitch, int64_t v_pitch, const int32_t * pos, const int32_t * sel, float * out,
                 float * part, const tsg_msa_params & p, int nsplit)
{
    if (!attn_prepare<KT>()) return false;
    msa_attn_kernel<KT><<<grid, kAttnWarps * 32, attn_smem_bytes<KT>(), stream>>>(q, k, v, k_pitch, v_pitch, pos,
        sel, out, part, p.n_tokens, p.n_head, p.n_head_kv, p.block, p.topk, p.rows_stride,
        p.kq_scale * 1.4426950408889634f, nsplit);
    return true;
}

// Whether the current device runs the sm_80 kernels (cached per device): on an
// older one they would trap (see TSG_MSA_SM80_DEVICE_CODE).
bool msa_device_supported()
{
    static std::atomic<int> cache[64] = {};   // 0 unknown, 1 yes, 2 no
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) return false;
    std::atomic<int> & slot = cache[device & 63];
    int v = slot.load(std::memory_order_relaxed);
    if (v == 0)
    {
        int major = 0;
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess)
        {
            (void) cudaGetLastError();
            return false;
        }
        v = major >= tsg_msa_min_cuda_major ? 1 : 2;
        slot.store(v, std::memory_order_relaxed);
    }
    return v == 1;
}

int msa_launch(const tsg_msa_params & p, const float * q, const float * iq, const half * k, const half * v,
               const half * idx, const int64_t k_pitch, const int64_t v_pitch, const int64_t idx_pitch,
               const int32_t * pos, float * out, void * workspace, const size_t workspace_bytes,
               cudaStream_t stream)
{
    tsg_msa_plan plan;
    const int status = tsg_msa_make_plan(&p, &plan);
    if (status != 0) return status;
    if (!msa_device_supported()) return -4;
    if (!workspace || workspace_bytes < plan.bytes) return -2;
    // cp.async moves 16-byte pieces.
    if (k_pitch % 8 || v_pitch % 8 || idx_pitch % 8 ||
        (reinterpret_cast<uintptr_t>(k) | reinterpret_cast<uintptr_t>(v) | reinterpret_cast<uintptr_t>(idx)) % 16)
        return -1;
    (void) cudaGetLastError();

    auto * base = static_cast<unsigned char *>(workspace);
    auto * sel = reinterpret_cast<int32_t *>(base);
    auto * scores = reinterpret_cast<float *>(base + plan.off_scores);
    auto * part = plan.nsplit > 1 ? reinterpret_cast<float *>(base + plan.off_partials) : nullptr;
    const int H = p.n_head_kv, nt = p.n_tokens;

#ifdef TSG_MSA_TEST_HOOKS
    if (tsg_msa_test_phase_events[0]) cudaEventRecord(tsg_msa_test_phase_events[0], stream);
#endif
    for (int tc = 0; tc < nt; tc += plan.score_chunk)
    {
        const int n = std::min(plan.score_chunk, nt - tc);
        const float * iq_c = iq + (int64_t) tc * H * kD;
        const int32_t * pos_c = pos + tc;
        const bool precise = p.index_precision != 0;
        switch (plan.score_kt * 2 + (precise ? 1 : 0))
        {
            case 128: launch_score<64, false>(stream, iq_c, idx, idx_pitch, pos_c, scores, n, p, plan.nblk); break;
            case 129: launch_score<64, true >(stream, iq_c, idx, idx_pitch, pos_c, scores, n, p, plan.nblk); break;
            case 64:  launch_score<32, false>(stream, iq_c, idx, idx_pitch, pos_c, scores, n, p, plan.nblk); break;
            case 65:  launch_score<32, true >(stream, iq_c, idx, idx_pitch, pos_c, scores, n, p, plan.nblk); break;
            case 32:  launch_score<16, false>(stream, iq_c, idx, idx_pitch, pos_c, scores, n, p, plan.nblk); break;
            default:  launch_score<16, true >(stream, iq_c, idx, idx_pitch, pos_c, scores, n, p, plan.nblk); break;
        }
#ifdef TSG_MSA_TEST_HOOKS
        if (tsg_msa_test_phase_events[1] && tc + n >= nt) cudaEventRecord(tsg_msa_test_phase_events[1], stream);
#endif
        const int cap = std::min(plan.nblk, kTopkSmemFloats);
        int32_t * sel_c = sel + (int64_t) tc * H * p.topk;
        // The score row pitch bounds the candidates (the live ones are only
        // known on the device).
        const int width = n * H > kTopkWideRows || plan.nblk <= 512 ? kTopkThreads : plan.nblk <= 2048 ? 512 : 1024;
        const dim3 grid((unsigned) (n * H));
        const size_t smem = cap * sizeof(float);
        if (width == 1024)
            msa_topk_kernel<1024><<<grid, 1024, smem, stream>>>(scores, plan.nblk, pos_c, sel_c, H, p.block, p.topk,
                                                                p.local_blocks, cap);
        else if (width == 512)
            msa_topk_kernel<512><<<grid, 512, smem, stream>>>(scores, plan.nblk, pos_c, sel_c, H, p.block, p.topk,
                                                              p.local_blocks, cap);
        else
            msa_topk_kernel<kTopkThreads><<<grid, kTopkThreads, smem, stream>>>(scores, plan.nblk, pos_c, sel_c, H,
                                                                                p.block, p.topk, p.local_blocks, cap);
    }
#ifdef TSG_MSA_TEST_HOOKS
    if (tsg_msa_test_phase_events[2]) cudaEventRecord(tsg_msa_test_phase_events[2], stream);
#endif

    const dim3 grid((unsigned) plan.n_tiles, (unsigned) H, (unsigned) plan.nsplit);
    bool launched = false;
    switch (plan.attn_kt)
    {
        case 64: launched = launch_attn<64>(grid, stream, q, k, v, k_pitch, v_pitch, pos, sel, out, part, p, plan.nsplit); break;
        case 32: launched = launch_attn<32>(grid, stream, q, k, v, k_pitch, v_pitch, pos, sel, out, part, p, plan.nsplit); break;
        default: launched = launch_attn<16>(grid, stream, q, k, v, k_pitch, v_pitch, pos, sel, out, part, p, plan.nsplit); break;
    }
    if (!launched) return -3;
#ifdef TSG_MSA_TEST_HOOKS
    if (tsg_msa_test_phase_events[3]) cudaEventRecord(tsg_msa_test_phase_events[3], stream);
#endif
    if (plan.nsplit > 1)
        msa_merge_kernel<<<(unsigned) ((int64_t) nt * p.n_head), kD, 0, stream>>>(part, out, plan.nsplit);
#ifdef TSG_MSA_TEST_HOOKS
    if (tsg_msa_test_phase_events[4]) cudaEventRecord(tsg_msa_test_phase_events[4], stream);
#endif
    return cudaGetLastError() == cudaSuccess ? 0 : -3;
}

} // namespace

int tsg_msa_attention_cuda(const tsg_msa_params * p, const float * q, const float * iq, const void * k,
                           const void * v, const void * idx, const int32_t * pos, float * out, void * workspace,
                           size_t workspace_bytes, void * cuda_stream)
{
    if (!p) return -1;
    const int64_t kv_pitch = (int64_t) p->n_head_kv * p->head_dim;
    return msa_launch(*p, q, iq, static_cast<const half *>(k), static_cast<const half *>(v),
                      static_cast<const half *>(idx), kv_pitch, kv_pitch, p->head_dim, pos, out, workspace,
                      workspace_bytes, static_cast<cudaStream_t>(cuda_stream));
}

#ifdef TSG_GGML_USE_CUDA
bool tsg_msa_cuda_capability(ggml_backend_t backend, int * major, int * minor)
{
    *major = *minor = 0;
    if (!backend || !backend->context) return false;
    // The context's leading member is its device (see tsg_msa_cuda_compute).
    const int device = static_cast<ggml_backend_cuda_context *>(backend->context)->device;
    if (cudaDeviceGetAttribute(major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess ||
        cudaDeviceGetAttribute(minor, cudaDevAttrComputeCapabilityMinor, device) != cudaSuccess)
    {
        (void) cudaGetLastError();
        *major = *minor = 0;
        return false;
    }
    return true;
}

void tsg_msa_cuda_compute(ggml_tensor * dst, ggml_backend_t backend)
{
    // Only the context's leading members (device, streams) are read here.
    // Everything after its USE_CUDA_GRAPH section, the memory pool included,
    // sits at other offsets outside ggml-cuda's own translation units
    // (GGML_CUDA_USE_GRAPHS is defined for that directory only), so the
    // workspace is the tail of the node's own output (tsg_msa_build).
    auto * context = static_cast<ggml_backend_cuda_context *>(backend->context);
    CUDA_CHECK(cudaSetDevice(context->device));
    const cudaStream_t stream = context->stream(context->device, 0);
    tsg_msa_params p;
    tsg_msa_params_from_node(dst, &p);
    const ggml_tensor * q = dst->src[0], * iq = dst->src[1], * k = dst->src[2], * v = dst->src[3];
    const ggml_tensor * idx = dst->src[4], * pos = dst->src[5];
    const size_t out_bytes = (size_t) p.n_tokens * p.n_head * p.head_dim * sizeof(float);
    const size_t bytes = tsg_msa_workspace_bytes(&p);
    GGML_ASSERT(ggml_nbytes(dst) >= out_bytes + bytes && "minimax-msa: node was not built by tsg_msa_build");
    void * workspace = static_cast<unsigned char *>(dst->data) + out_bytes;
    const int status = msa_launch(p, static_cast<const float *>(q->data), static_cast<const float *>(iq->data),
        static_cast<const half *>(k->data), static_cast<const half *>(v->data), static_cast<const half *>(idx->data),
        int64_t(k->nb[1] / 2), int64_t(v->nb[1] / 2), int64_t(idx->nb[1] / 2),
        static_cast<const int32_t *>(pos->data), static_cast<float *>(dst->data), workspace, bytes, stream);
    if (status != 0)
    {
        std::fprintf(stderr, "[minimax-msa] CUDA%d launch failed: %s (nt=%d heads=%d/%d rows=%d block=%d topk=%d)\n",
                     context->device, tsg_msa_status_string(status), p.n_tokens, p.n_head, p.n_head_kv,
                     p.rows_stride, p.block, p.topk);
        CUDA_CHECK(cudaGetLastError());
        GGML_ABORT("minimax-msa: kernel launch failed");
    }
}
#endif
