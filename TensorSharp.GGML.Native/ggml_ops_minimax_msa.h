// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// ---------------------------------------------------------------------------
// MiniMax Sparse Attention (MSA, MiniMax-M3 / M3.1): lightning-indexer block
// scoring, per-index-head block selection and block-sparse attention for one
// sequence and one layer, as ONE operation.
//
// Semantics (HF modeling_minimax_m3_vl.py, see the M3.1 design notes):
//   s[h,t,j]  = <iq[t,h], idx[j]>                      unscaled, F32 accumulate
//   B[h,t,b]  = max_{j in block b, j <= pos[t]} s[h,t,j]
//   own block = pos[t] / block, and the local_blocks-1 blocks before it, are
//               forced; the rest of the top min(topk, own+1) blocks are the
//               highest B among the blocks strictly before them. Ties go to
//               the LOWER block id (torch.topk leaves ties unspecified).
//   index head h selects for KV head h, i.e. query heads h*G .. h*G+G-1,
//   G = n_head / n_head_kv; attention inside the selected blocks stays token
//   causal (j <= pos[t]); scale kq_scale, softmax F32, no sinks, no gate.
// A token with pos < topk*block sees at most topk blocks, all selected: its
// result is exactly dense causal attention and its scores are never computed.
//
// Layouts (C2/C3; ggml ne order in brackets):
//   q   F32 [nt][n_head][head_dim]     [head_dim, n_head, nt]
//   iq  F32 [nt][n_head_kv][head_dim]  [head_dim, n_head_kv, nt]
//   k,v F16 [rows][n_head_kv*head_dim] [n_head_kv*head_dim, rows]
//   idx F16 [rows][head_dim]           [head_dim, rows]
//   pos I32 [nt] absolute positions == cache rows; token t reads rows
//       0..pos[t], which the caller must already have written
//   out F32 [nt][n_head][head_dim]     [head_dim, n_head, nt]
// Rows are absolute slots (no context shift, no left padding).
//
// The CUDA path (sm_80+, built for the CUDA backend only) runs tensor-core
// kernels with F16 operands and F32 accumulation (cp.async and the m16n8k16
// F16 MMA need compute capability 8.0: a build that also targets an older
// architecture compiles kernels that trap there, and the launcher refuses such
// a device with status -4 before launching anything): the index query is rounded
// to F16 (index_precision 0) or split into two F16 halves whose products sum
// to ~F32 precision (index_precision 1, parity runs against an F32 indexer);
// Q is rounded to F16 and P to F16 before P*V, as in flash attention. The CPU
// path is the double-accumulating reference with identical selection rules.
// ---------------------------------------------------------------------------

#pragma once

#include <cstddef>
#include <cstdint>

struct ggml_context;
struct ggml_tensor;

struct tsg_msa_params
{
    int n_tokens;        // nt query tokens (one sequence)
    int n_head;          // 64
    int n_head_kv;       // 4 (== number of index heads); <= 16
    int head_dim;        // 128 (main and index); only 128 is implemented
    int block;           // MSA block size in rows (128 real; fixtures 16); multiple of 16
    int topk;            // 16 real (fixtures 4); 1..64
    int local_blocks;    // 1; 0..topk
    int rows_stride;     // rows the call may address: every pos must be < rows_stride.
                         // Kernel grids scale with it, so keep it near max(pos)+1.
    float kq_scale;      // 1/sqrt(128)
    int index_precision; // 0 = F16 index query (default), 1 = split-F16 (~F32) index query.
                         // CUDA only; the CPU reference always scores in double.
};

// Workspace layout (device bytes, each region 256-byte aligned), documented
// because the kernel test reads the selection back:
//   [0]               I32 sel[nt][n_head_kv][topk]  ascending block ids, -1 padded
//   [scores]          F32 block scores for a chunk of tokens (internal)
//   [partials]        F32 split-K partial outputs (internal)
size_t tsg_msa_workspace_bytes(const tsg_msa_params * p);

// Validates p; returns 0 on success, a negative code otherwise:
//   -1 invalid parameters, -2 workspace too small, -3 CUDA launch failure,
//   -4 the current CUDA device is older than compute capability 8.0.
const char * tsg_msa_status_string(int status);
int tsg_msa_validate(const tsg_msa_params * p);

// Device pointers; k/v rows are n_head_kv*head_dim halves apart and idx rows
// head_dim halves apart (C2). Launched asynchronously on `cuda_stream`.
int tsg_msa_attention_cuda(const tsg_msa_params * p, const float * q, const float * iq,
                           const void * k, const void * v, const void * idx, const int32_t * pos,
                           float * out, void * workspace, size_t workspace_bytes, void * cuda_stream);

// Identical semantics on host memory (reference for tests and the CPU backend).
int tsg_msa_attention_cpu(const tsg_msa_params * p, const float * q, const float * iq,
                          const uint16_t * k, const uint16_t * v, const uint16_t * idx, const int32_t * pos,
                          float * out);

// Reference pieces, exported for the kernel test:
// the selection alone (sel I32 [nt][n_head_kv][topk]) and, optionally, the
// candidate block scores it ranked (scores F32 [nt][n_head_kv][ceil(rows/block)],
// -inf where not a candidate);
int tsg_msa_select_cpu(const tsg_msa_params * p, const float * iq, const uint16_t * idx,
                       const int32_t * pos, int32_t * sel, float * scores, int t0, int t1);
// and attention over a given selection. Both run tokens [t0, t1) only.
int tsg_msa_attend_cpu(const tsg_msa_params * p, const float * q, const uint16_t * k,
                       const uint16_t * v, const int32_t * pos, const int32_t * sel,
                       float * out, int t0, int t1);

// Internal: the launch plan shared by tsg_msa_workspace_bytes and the CUDA
// launcher. A score CTA is tsg_msa_score_warps warps of 16 (token, index
// head) rows; an attention CTA is tsg_msa_attn_tokens warps, one token each.
constexpr int tsg_msa_score_warps = 4;
constexpr int tsg_msa_attn_tokens = 4;
constexpr int tsg_msa_max_split = 256;   // split-K factor bound (the merge kernel's shared weights)
struct tsg_msa_plan
{
    int score_kt;        // keys per shared tile in the score kernel (64/32/16)
    int attn_kt;         // keys per shared tile in the attention kernel (32/16)
    int nblk;            // ceil(rows_stride / block): score row pitch
    int score_chunk;     // tokens per score + top-k pass (bounds the score buffer)
    int n_tiles;         // attention tiles: ceil(nt / tsg_msa_attn_tokens)
    int nsplit;          // split-K factor of the attention launch (1 = none)
    size_t off_scores, off_partials, bytes;
};
int tsg_msa_make_plan(const tsg_msa_params * p, tsg_msa_plan * plan);

// Graph builder. Emits ONE GGML_OP_CUSTOM node that the TensorSharp fused
// CUDA backend (tsg_dsv4_fused_backend_init) runs with the kernels above and
// the CPU backend runs with the reference (all CPU threads), and returns a
// contiguous F32 [head_dim, n_head, nt] view of its result. The node itself is
// F32 1-D: the result, then tsg_msa_workspace_bytes of CUDA workspace (sel,
// scores, partials), so the graph allocator owns the scratch and reuses it
// after the result's last reader (the CPU backend never touches the tail).
// Shapes come from the tensors: q [head_dim, n_head, nt] F32 contiguous; iq
// [head_dim, n_head_kv, nt] F32 contiguous; k_cache, v_cache
// [n_head_kv*head_dim, rows] F16 and idx_cache [head_dim, rows] F16 with
// contiguous rows (views with any row pitch are fine; rows = idx_cache->ne[1]
// replaces p.rows_stride, and the workspace grows with it: whole slot caches
// are fine, the kernels' work follows the live positions); pos [nt] I32. From
// `p` only block, topk, local_blocks, kq_scale and index_precision are used.
// The node's descriptor is allocated from `ctx` (ggml_new_buffer, under 128
// bytes), so it lives as long as the graph; budget that in the context size.
// The caller orders the cache writes of these rows before the node (graph
// order: the node reads the caches, not the writes' results).
ggml_tensor * tsg_msa_build(ggml_context * ctx, const tsg_msa_params & p, ggml_tensor * q, ggml_tensor * iq,
                            ggml_tensor * k_cache, ggml_tensor * v_cache, ggml_tensor * idx_cache,
                            ggml_tensor * pos);

// The parameters of a node emitted by tsg_msa_build (its shapes + descriptor);
// accepts the custom node or the view tsg_msa_build returned.
void tsg_msa_params_from_node(const ggml_tensor * node, tsg_msa_params * p);

// The oldest compute capability (major version) the CUDA kernels run on.
constexpr int tsg_msa_min_cuda_major = 8;

#ifdef TSG_GGML_USE_CUDA
typedef struct ggml_backend * ggml_backend_t;
// Runs a tsg_msa_build node on the paired ggml-cuda backend's stream, with the
// workspace in the node's tail. Reached through the node descriptor's
// cuda_compute hook, so the fused backend needs no link dependency on it.
void tsg_msa_cuda_compute(ggml_tensor * node, ggml_backend_t cuda_backend);
// The compute capability of a ggml-cuda backend's device; false when it cannot
// be read. The kernels run only where *major >= tsg_msa_min_cuda_major.
bool tsg_msa_cuda_capability(ggml_backend_t cuda_backend, int * major, int * minor);
#endif
