// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// ---------------------------------------------------------------------------
// MiniMax-M3 (MiniMax-M3.1-Flash, GGUF arch "minimax-m3") whole-model executor
// for the GGML backends.
//
// The model is ~232 GiB at UD-Q4_K_XL, so it is spread across every GPU with,
// when it must, some layers' routed experts left in system RAM. The executor
// follows the Muse Spark executor (ggml_ops_muse_spark_exec.cpp) and
// llama.cpp's graph (src/models/minimax-m3.cpp):
//
//   * it loads the split GGUF itself and places whole layers on the GPUs,
//     sized against each device's free memory (or TS_MINIMAX_M3_LAYER_SPLIT);
//   * `--n-cpu-moe N` serves the first N layers' routed experts zero-copy from
//     the GGUF mapping, so the scheduler runs exactly those mul_mat_id nodes
//     on the CPU backend and everything else stays on the GPUs;
//   * every sequence slot owns three F16 caches per layer on the device that
//     reads them (K [512, rows], V [512, rows], index keys [128, rows]); a
//     row IS the position, so MSA blocks are views of the cache;
//   * ONE ggml graph per ubatch goes through ggml_backend_sched, and an LRU of
//     built+allocated graphs keyed by shape lets decode replay the same graph
//     (and, on CUDA, the same captured CUDA graphs) token after token;
//   * a batched decode step evaluates one token for each of several slots in
//     a single graph; each slot's sparse attention uses its own positions.
//
// Numerical order (llama.cpp's graph, unsloth minimax-m31 @ ef8f94cc):
//   embd (BF16 rows, no norm)
//   per layer: rms_norm * attn_norm (the GGUF norms hold Gemma's 1+w already)
//              -> Q/K/V (Q8_0) -> per-head rms_norm * q_norm / k_norm
//              -> NEOX RoPE over dims [0, 64) of each 128-dim head
//              indexer: iq = Wiq x [128, 4], ik = Wik x [128, 1] (F32),
//              per-head rms_norm * own norms, the same RoPE; ik is cached
//              -> attention (scale 1/sqrt(128)): MiniMax Sparse Attention
//                 (block scores max over 128 rows of <iq_h, ik>, own block
//                 forced, top-16 blocks per index head h, used by KV head h)
//              -> wo -> + residual
//              rms_norm * ffn_norm ->
//                leading dense layers: swiglu_oai(gate, up) -> down
//                MoE: router F32 -> sigmoid -> top-4 of (sigmoid + bias) ->
//                     unbiased weights / sum * 2.0; NVFP4 experts with the
//                     per-expert .scale applied to gate/up BEFORE
//                     swiglu_oai(1.702, 7.0) and to down before the weight;
//                     + shared expert (swiglu_oai)
//              -> + residual
//   output_norm -> output (BF16)
//
// Attention (TS_MINIMAX_M3_MSA):
//   auto   - a query at position p < top_k * block (2048) can see at most
//            top_k blocks, all of them selected, so dense causal attention IS
//            its sparse attention: a ubatch (segment) whose last position is
//            below that runs plain flash attention over the slot's rows; any
//            other runs sparse attention, through TensorSharp's kernels
//            (`kernel` below) on a CUDA GPU and through the ggml ops (`ggml`)
//            on any other device (the index keys are cached either way).
//   dense  - dense attention everywhere (debugging; changes results past 2048).
//   ggml   - sparse attention for every ubatch, through upstream ggml ops:
//            llama.cpp's algorithm with an identity position map. Decode
//            (one token of a slot) scores the index-key view, max-pools 128
//            rows, forces the own block, takes the top-k blocks and gathers
//            their K/V rows for one grouped flash attention. A prompt chunk
//            builds a per-(token, KV group) block mask over every row and runs
//            masked flash attention, in token chunks sized by a byte budget
//            (TS_MINIMAX_M3_MSA_CHUNK_MB) rather than by 2^31 elements.
//            The gathered blocks reach flash attention in ascending order,
//            as the masked rows of a prompt chunk do, so a one-token decode
//            and the verification row of the same position sum identically.
//   kernel - TensorSharp's MSA kernels (ggml_ops_minimax_msa, contract C3):
//            scoring, selection and block-sparse attention in one custom node
//            per segment, run by the fused-op wrapper of each CUDA backend
//            (compute capability 8.0+) or, on a CPU-only load, by the CPU
//            reference. A GPU without the wrapper (Metal, Vulkan, an older
//            CUDA GPU) and builds without those files refuse it; auto keeps
//            the ggml ops on such a GPU and says so.
//
// Tensor parallelism (tp > 1, ggml_ops_minimax_m3_tp): the same graph on
// upstream ggml's meta backend over N CUDA GPUs, attention mirrored and the
// FFN split along n_ff (one all-reduce per layer); see that header for why.
// ---------------------------------------------------------------------------

#include "ggml_ops_internal.h"
#include "ggml_ops_minimax_m3_tp.h"
#include "dsv4_file_warm.h"
#if defined(TSG_GGML_USE_CUDA)
#include "ggml_ops_dsv4_fused.h"
#endif
// TensorSharp's MSA kernels (contract C3, ggml_ops_minimax_msa.*): building
// with TSG_MINIMAX_MSA_KERNELS=1 enables TS_MINIMAX_M3_MSA=kernel.
#if defined(TSG_MINIMAX_MSA_KERNELS)
#include "ggml_ops_minimax_msa.h"
#endif

#include "gguf.h"
#include "ggml_ops_minimax_m3_dspark.h"

#include <cerrno>
#include <cinttypes>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cctype>
#include <cstring>
#include <chrono>
#include <algorithm>
#include <array>
#include <functional>
#include <atomic>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <set>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

#if !defined(_WIN32)
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <pthread.h>
#include <sched.h>
#endif

namespace tsg_mm3
{

/// GPUs one model can span. ggml-cuda addresses at most GGML_CUDA_MAX_DEVICES
/// (16) and ggml-vulkan GGML_VK_MAX_DEVICES (16); TensorSharp builds ggml-base
/// with GGML_SCHED_MAX_BACKENDS=32, so these plus the CPU fit one scheduler.
static constexpr int MAX_GPUS = 16;
/// Sequences one batched decode step serves at most.
static constexpr int MAX_BATCHED_DECODE = 16;
// Graphs of at most this many rows read the F16 indexer copies (project_index):
// ggml-cuda runs their products as matrix-vector kernels, which stream the
// weights. A prompt's products are GEMMs, and on F16 weights an 8192-token
// prefill on 8x A40 ran at 1576 tok/s against 1691 on F32 (32768: 2156 / 2426).
static constexpr int64_t IDX_F16_MAX_ROWS = 8;
/// Default granularity of the attention windows. Decode reuses one graph shape
/// for a whole stretch of this many positions instead of rebuilding every token.
static constexpr int64_t KV_PAD_DEFAULT = 256;
/// (token, expert) pairs host_moe_matmul handles in one call.
static constexpr int64_t HOST_MOE_MAX_PICKS = 512;
/// swiglu_oai constants of every MiniMax-M3 FFN (dense, shared, routed).
static constexpr float SWIGLU_ALPHA = 1.702f;
static constexpr float SWIGLU_LIMIT = 7.0f;

static int64_t pad_to(int64_t v, int64_t p) { return ((v + p - 1) / p) * p; }

// Options set through TSGgml_MiniMaxM3SetOption take precedence over the
// process environment: .NET's Environment.SetEnvironmentVariable does not
// reach getenv on Linux, so the managed side forwards its TS_MINIMAX_M3_*
// settings here before loading.
static std::mutex g_opt_mu;
static std::map<std::string, std::string> g_opts;

static std::string opt_get(const char * name)
{
    {
        std::lock_guard<std::mutex> lk(g_opt_mu);
        auto it = g_opts.find(name);
        if (it != g_opts.end()) return it->second;
    }
    const char * e = getenv(name);
    return e ? std::string(e) : std::string();
}

static bool env_flag(const char * name, bool fallback)
{
    const std::string e = opt_get(name);
    if (e.empty()) return fallback;
    return atoi(e.c_str()) != 0;
}

static long env_long(const char * name, long fallback)
{
    const std::string e = opt_get(name);
    if (e.empty()) return fallback;
    char * end = nullptr;
    const long v = strtol(e.c_str(), &end, 10);
    return end != e.c_str() ? v : fallback;
}

static bool debug_on()
{
    static const bool on = env_flag("TS_MINIMAX_M3_EXEC_DEBUG", false);
    return on;
}

/// How attention runs (TS_MINIMAX_M3_MSA; see the file comment).
enum msa_mode { MSA_AUTO = 0, MSA_DENSE = 1, MSA_GGML = 2, MSA_KERNEL = 3 };

static const char * msa_mode_name(int mode)
{
    switch (mode)
    {
        case MSA_DENSE: return "dense";
        case MSA_GGML: return "ggml";
        case MSA_KERNEL: return "kernel";
        default: return "auto";
    }
}

#if defined(TSG_GGML_TEST_HOOKS)
// Fault injection for the tests (TSGgml_MiniMaxM3TestInjectFault): from the
// injection on, thread start number g_test_fail_thread (0: the next) throws as
// an exhausted thread quota does, and upload job number g_test_fail_job throws
// std::bad_alloc inside its worker; -1 disables either. g_test_msa_old_gpus
// makes every CUDA GPU report compute capability 7.5 to the MSA kernel gate.
static std::atomic<int> g_test_fail_thread{ -1 };
static std::atomic<int> g_test_fail_job{ -1 };
static std::atomic<bool> g_test_msa_old_gpus{ false };
// DSpark pipeline ordering (TSGgml_MiniMaxM3TestDsparkPipeline): large F32
// products added to every DSpark commit on the drafter's GPU (a slow drafter
// stage that earlier stages run ahead of), and dropping forward_pipelined's
// waits for tap edges that skip stages (to show the race they close).
static std::atomic<int> g_test_ds_slow_products{ 0 };
static std::atomic<bool> g_test_ds_no_tap_wait{ false };

/// True once for the counter's n-th call from its injection.
static bool test_fault_due(std::atomic<int> & counter)
{
    int v = counter.load();
    while (v >= 0 && !counter.compare_exchange_weak(v, v - 1)) {}
    return v == 0;
}
#endif

// ---------------------------------------------------------------------------
// hyper-parameters / weights / slots
// ---------------------------------------------------------------------------
struct hparams
{
    int32_t n_layer = 0;
    int32_t n_embd = 0;
    int32_t n_head = 0;
    int32_t n_head_kv = 0;
    int32_t head_dim = 0;
    int32_t n_rot = 0;
    int32_t n_ff = 0;              // leading dense layers
    int32_t n_ff_exp = 0;          // routed and shared experts
    int32_t n_expert = 0;
    int32_t n_expert_used = 0;
    int32_t n_expert_shared = 0;
    int32_t n_lead = 0;            // leading dense layers
    int32_t n_vocab = 0;
    int32_t n_ctx_train = 0;
    float rms_eps = 1e-6f;
    float rope_base = 10000000.0f;
    float w_scale = 1.0f;
    bool w_norm = true;
    int32_t gating = 2;

    // indexer (MiniMax Sparse Attention)
    int32_t idx_n_head = 0;        // one index head per KV head
    int32_t idx_dim = 0;
    int32_t idx_topk = 0;          // blocks selected per query and index head
    int32_t idx_blk = 0;           // rows per block
    int32_t idx_local = 0;         // trailing blocks always selected (the query's own block first)

    int64_t n_q() const { return (int64_t) n_head * head_dim; }
    int64_t n_kv() const { return (int64_t) n_head_kv * head_dim; }
    float kq_scale() const { return 1.0f / sqrtf((float) head_dim); }
    /// Queries at positions below this see at most idx_topk blocks, all of
    /// them selected: their sparse attention is dense causal attention.
    int64_t dense_span() const { return (int64_t) idx_topk * idx_blk; }
};

struct layer
{
    ggml_tensor * attn_norm = nullptr;
    ggml_tensor * wq = nullptr;
    ggml_tensor * wk = nullptr;
    ggml_tensor * wv = nullptr;
    ggml_tensor * wqkv = nullptr;     // [n_embd, nq + nkv + nkv] when the three share a type
    ggml_tensor * wo = nullptr;
    ggml_tensor * q_norm = nullptr;
    ggml_tensor * k_norm = nullptr;

    ggml_tensor * idx_wq = nullptr;   // [n_embd, idx_n_head * idx_dim]
    ggml_tensor * idx_wk = nullptr;   // [n_embd, idx_dim]
    ggml_tensor * idx_wqk = nullptr;  // both, when they share a type
    // F16 copies of F32 indexer projections for decode-shaped graphs (TS_MINIMAX_M3_INDEXER_F16)
    ggml_tensor * idx_wq_dec = nullptr;
    ggml_tensor * idx_wk_dec = nullptr;
    ggml_tensor * idx_wqk_dec = nullptr;
    ggml_tensor * idx_q_norm = nullptr;
    ggml_tensor * idx_k_norm = nullptr;

    ggml_tensor * ffn_norm = nullptr;
    // leading dense layers
    ggml_tensor * ffn_gate = nullptr;
    ggml_tensor * ffn_up = nullptr;
    ggml_tensor * ffn_down = nullptr;
    // MoE layers
    ggml_tensor * gate_inp = nullptr;
    ggml_tensor * exp_probs_b = nullptr;
    ggml_tensor * gate_exps = nullptr;
    ggml_tensor * up_exps = nullptr;
    ggml_tensor * down_exps = nullptr;
    ggml_tensor * gate_s = nullptr;   // per-expert output scale [n_expert] (NVFP4: 1/global scale)
    ggml_tensor * up_s = nullptr;
    ggml_tensor * down_s = nullptr;
    ggml_tensor * gate_shexp = nullptr;
    ggml_tensor * up_shexp = nullptr;
    ggml_tensor * down_shexp = nullptr;

    int device = 0;
    bool dense = false;            // leading dense FFN (no experts)
    bool msa = false;              // carries indexer tensors: sparse attention
    bool cpu_moe = false;          // any routed-expert stack on the host
    bool cpu_stack[3] = {};        // gate, up, down stack on the host
};

struct slot
{
    int id = 0;
    int64_t n_past = 0;
    std::vector<ggml_tensor *> k;   // [layer] F16 [n_kv, cap]
    std::vector<ggml_tensor *> v;   // [layer] F16 [n_kv, cap]
    std::vector<ggml_tensor *> ik;  // [layer] F16 [idx_dim, cap] index keys (sparse layers)
    // Rows allocated per layer. They grow (doubling, copied on the device) as
    // the sequence does, so an idle or short slot holds little.
    int64_t cap = 0;
};

struct tensor_source
{
    int shard = -1;
    size_t offset = 0;
    size_t size = 0;
    ggml_type type = GGML_TYPE_F32;
    int64_t ne[4] = {1, 1, 1, 1};
};

struct pending_weight
{
    ggml_tensor * t = nullptr;
    int shard = -1;
    size_t file_off = 0;
    size_t tensor_off = 0;   // destination offset inside t (fused tensors)
    size_t bytes = 0;        // bytes read from the file
    bool to_f16 = false;     // F32 in the file, F16 on the device (bytes / 2 written)
};

/// Per-device inputs of one graph. Only devices that host a layer get them.
struct graph_inputs
{
    ggml_tensor * tokens = nullptr;                 // I32 [nt] (embedding device)
    ggml_tensor * embd_rows = nullptr;              // F32 [n_embd, n_ovr]
    ggml_tensor * embd_idx = nullptr;               // I64 [n_ovr]
    ggml_tensor * pos[MAX_GPUS + 1] = {};           // I32 [nt]
    ggml_tensor * kv_idx[MAX_GPUS + 1] = {};        // I64 [nt] cache rows (= positions)
    ggml_tensor * mask[MAX_GPUS + 1] = {};          // F16 causal rows, flat: segment i is [n_kv_i, len_i]
    ggml_tensor * bias[MAX_GPUS + 1] = {};          // F32 local-block force, flat: sparse segment i is [nblk_i, len_i]
    ggml_tensor * out_ids = nullptr;                // I32 [n_out] (last device)
};

/// One segment of a graph: `len` consecutive tokens of one slot (a decode
/// step is a segment of length 1).
struct bd_token
{
    int slot_id = 0;
    int64_t p = 0;          // position of the segment's first token (inputs only, never matched)
    int64_t len = 1;
    int64_t n_kv = 0;       // visible rows (padded)
    bool sparse = false;    // MSA layers run sparse attention (otherwise dense causal)
    int64_t chunk = 0;      // sparse prompt segments: tokens per block-mask chunk
};

struct graph_entry
{
    // ---- key ----
    int kind = 0;               // 0 = one slot, nt tokens; 1 = segments of several slots
    int slot_id = -1;
    int64_t nt = 0;
    int64_t p0 = 0;             // only for the key's window math (not part of the match)
    int64_t n_out = 0;
    int64_t n_ovr = 0;
    bool want_logits = true;
    std::vector<bd_token> bd;   // kind 0: one segment

    // ---- graph ----
    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_backend_sched_t sched = nullptr;
    graph_inputs inp;
    ggml_tensor * logits = nullptr;
    bool owns_sched = true;             // false: the pipeline's persistent scheduler
    size_t sched_nodes = 0;             // the graph size `sched` was created for

    ~graph_entry()
    {
        if (sched && owns_sched) ggml_backend_sched_free(sched);
        if (ctx) ggml_free(ctx);
    }
};

struct shard_files
{
    std::vector<std::string> paths;
};

/// Pipelined prefill across the layer-split GPUs (forward_pipelined). One
/// persistent scheduler with ggml's rotating input copies runs consecutive
/// micro-batches of a prompt back to back, so GPU k works on micro-batch j
/// while GPU k+1 works on j-1 (llama.cpp's pipeline parallelism).
struct pipeline_state
{
    bool enabled = false;
    bool forced = false;                     // TS_MINIMAX_M3_PIPELINE=1
    bool ubatch_fixed = false;               // TS_MINIMAX_M3_PIPELINE_UBATCH set: every call uses `ubatch` rows
    int64_t ubatch = 512;                    // the fixed micro-batch, or the adaptive one's upper bound
    ggml_backend_sched_t sched = nullptr;
    int n_copies = 1;
    size_t n_nodes = 0;
    std::vector<int> stages;                 // devices holding layers, in layer order
    std::vector<ggml_backend_event_t> done;  // [stage * n_copies + copy]: stage finished the micro-batch in that copy slot
    std::vector<ggml_backend_event_t> staged;// [stage * n_staging + slot]: uploads from that staging slot finished
    std::vector<uint8_t> busy;               // staging slot has uploads in flight
    int n_staging = 0;
    ggml_backend_buffer_t staging = nullptr; // pinned host memory, n_staging slots
    size_t slot_bytes = 0;
    std::vector<uint8_t> meta;               // graph metadata, rebuilt per micro-batch
    std::vector<ggml_backend_buffer_type_t> bufts;
    std::set<std::string> warned;
    uint64_t n_calls = 0, n_ubatches = 0, n_declines = 0;
    int64_t res_u = 0, res_kv = 0;           // shapes the scheduler is reserved for
    int res_nodes = 0;                       // node count of the reserved graph (its topology)
    // The smallest shape whose reservation failed: larger calls decline without
    // retrying the allocation until a slot gives memory back (slot_free).
    int64_t fail_u = 0, fail_kv = 0;

    void release()
    {
        res_u = res_kv = 0;
        res_nodes = 0;
        if (sched) { ggml_backend_sched_free(sched); sched = nullptr; }
        for (auto e : done) if (e) ggml_backend_event_free(e);
        for (auto e : staged) if (e) ggml_backend_event_free(e);
        done.clear(); staged.clear(); busy.clear();
        if (staging) { ggml_backend_buffer_free(staging); staging = nullptr; }
        slot_bytes = 0;
        n_staging = 0;
    }
};

struct model
{
    hparams hp;

    // backends[d]: device d's own backend (weights, caches, events, uploads).
    // sched_backends[d]: what the scheduler runs device d's subgraph on, which
    // is a TensorSharp fused-op wrapper of backends[d] when the sparse
    // attention kernels need one (TS_MINIMAX_M3_MSA=kernel), else backends[d].
    ggml_backend_t backends[MAX_GPUS + 1] = {};
    ggml_backend_t wrappers[MAX_GPUS] = {};
    int n_gpu = 0;
    int n_backends = 0;
    ggml_threadpool_t cpu_threadpool = nullptr;
    int cpu_threads = 0;

    ggml_backend_t sched_backends[MAX_GPUS + 1] = {};
    ggml_backend_buffer_type_t sched_bufts[MAX_GPUS + 1] = {};
    int n_sched_backends = 0;

    ggml_context * w_ctx[MAX_GPUS + 1] = {};
    ggml_backend_buffer_t w_buf[MAX_GPUS + 1] = {};
    ggml_context * repack_ctx = nullptr;          // host expert stacks in the CPU_REPACK layout
    ggml_backend_buffer_t repack_buf = nullptr;
    size_t repack_bytes = 0;
    ggml_context * numa_ctx = nullptr;            // host expert stacks placed by NUMA first touch
    ggml_backend_buffer_t numa_buf = nullptr;
    size_t numa_bytes = 0;
    std::vector<int> worker_cpu;                  // CPU of each CPU-backend worker (by ith) when pinned

    shard_files shards;
    std::vector<void *> mmap_addrs;
    std::vector<size_t> mmap_sizes;
    std::vector<ggml_backend_buffer_t> mmap_bufs;
    size_t mmap_weight_bytes = 0;

    ggml_tensor * tok_embd = nullptr;
    ggml_tensor * output_norm = nullptr;
    ggml_tensor * output = nullptr;
    int embd_device = 0;     // MAX: n_gpu means host
    int out_device = 0;

    std::vector<layer> layers;
    int n_cpu_moe = 0;
    int n_msa_layers = 0;

    int32_t n_ctx = 0;
    int32_t n_ubatch = 0;
    int64_t kv_pad = KV_PAD_DEFAULT;
    bool op_offload = false;
    int64_t host_moe_max_tokens = 0;   // batches of 2..this many tokens use host_moe_matmul (0: never)
    bool fused_qkv = false;
    bool fused_idx = false;
    bool idx_f16 = false;              // indexer projections converted to F16 at load
    int msa = MSA_AUTO;
    // Devices whose sparse segments run TensorSharp's MSA kernels (every device
    // under MSA_KERNEL; under MSA_AUTO the CUDA GPUs with a fused-op wrapper).
    bool msa_kernel_dev[MAX_GPUS + 1] = {};
    // Every layer is sparse and runs the kernels: a sparse segment then needs no
    // causal mask and no block-force bias (the kernels read the positions).
    bool sparse_maskless = false;
    size_t msa_chunk_bytes = 0;        // byte budget of one sparse prompt chunk's masks and scores
    bool alloc_failed = false;   // the last acquire_graph ran out of device memory
    std::vector<size_t> max_graph_bytes;   // per backend: the largest compute buffer any graph has needed

    struct slot_ctx { int slot_id; ggml_context * ctx; ggml_backend_buffer_t buf; };
    std::vector<slot_ctx> slot_ctxs;
    // Caches of freed slots that never grew, kept for the next slot_alloc
    // (slot_pool_take): a chat turn allocates a slot for its cloned prefix and
    // one for its prompt-end checkpoint, and allocating one cost a K/V buffer
    // per GPU plus the room check. Released whenever memory is short.
    struct pooled_slot { std::vector<slot_ctx> ctxs; std::vector<ggml_tensor *> k, v, ik; int64_t cap = 0; };
    std::vector<pooled_slot> slot_pool;
    std::map<int, std::unique_ptr<slot>> slots;
    slot * active_slot = nullptr;
    int next_slot_id = 0;

    std::list<std::unique_ptr<graph_entry>> graph_cache;
    int graph_cache_cap = 12;

    std::vector<int> reclaimable;
    std::vector<int> reclaimed;

    // vision rows queued for the next forward call: index = absolute position of the first row
    struct embd_override { int64_t index; std::vector<float> rows; };
    std::vector<embd_override> embd_ovr;

    // host scratch
    std::vector<int32_t> h_tokens;
    std::vector<int32_t> h_pos;
    std::vector<int64_t> h_kv_idx;
    std::vector<uint16_t> h_mask;
    std::vector<float> h_bias;
    std::vector<int32_t> h_out_ids;

    // stats
    uint64_t n_graph_builds = 0;
    uint64_t n_graph_reuses = 0;
    uint64_t n_sparse_segments = 0;    // segments computed with sparse attention
    uint64_t n_dense_segments = 0;     // segments computed with dense attention

    pipeline_state pipe;

    // Tensor parallelism (load with tp > 1): backends[0] is one ggml meta
    // backend over tp.dev.devs (n_gpu == 1), and tp.ud's plan splits the FFN
    // and the head among them; attention and the caches are mirrored.
    struct tp_state
    {
        int n = 0;
        tsg_mm3_tp::devices dev;
        std::unique_ptr<tsg_mm3_tp::userdata> ud;
        // F32 0..idx_blk-1 and 0..n_head_kv-1 for the sparse decode: ggml_arange
        // has no source the meta backend could take a split state from.
        ggml_tensor * arange_rows = nullptr;
        ggml_tensor * arange_heads = nullptr;
        // Rows of the largest graph computed so far: the meta backend's
        // reduction scratch is sized for it (tsg_mm3_tp::reduction_scratch_bytes).
        int64_t max_rows = 0;
    } tp;
    dspark_state ds;                   // DSpark drafter (ggml_ops_minimax_m3_dspark.inc); unloaded without one

    std::mutex mu;

    ~model()
    {
        ds.release();     // its graphs and context rings reference the backends
        pipe.release();   // its scheduler and events reference the backends
        graph_cache.clear();
        slots.clear();
        for (auto & ps : slot_pool)
            for (auto & sc : ps.ctxs) slot_ctxs.push_back(sc);
        slot_pool.clear();
        for (auto & sc : slot_ctxs)
        {
            if (sc.buf) ggml_backend_buffer_free(sc.buf);
            if (sc.ctx) ggml_free(sc.ctx);
        }
        slot_ctxs.clear();
        for (int i = 0; i <= MAX_GPUS; i++)
        {
            if (w_buf[i]) ggml_backend_buffer_free(w_buf[i]);
            if (w_ctx[i]) ggml_free(w_ctx[i]);
        }
        if (repack_buf) ggml_backend_buffer_free(repack_buf);
        if (repack_ctx) ggml_free(repack_ctx);
        if (numa_buf) ggml_backend_buffer_free(numa_buf);
        if (numa_ctx) ggml_free(numa_ctx);
        for (ggml_backend_buffer_t b : mmap_bufs)
            if (b) ggml_backend_buffer_free(b);
#if !defined(_WIN32)
        for (size_t i = 0; i < mmap_addrs.size(); i++)
            if (mmap_addrs[i]) munmap(mmap_addrs[i], mmap_sizes[i]);
#endif
        // A wrapper uses its device's backend: free it first.
        for (int i = 0; i < MAX_GPUS; i++)
            if (wrappers[i]) ggml_backend_free(wrappers[i]);
        for (int i = 0; i < n_backends; i++)
            if (backends[i]) ggml_backend_free(backends[i]);
        if (cpu_threadpool) ggml_threadpool_free(cpu_threadpool);
    }

    int host_dev() const { return n_gpu; }
};

// ---------------------------------------------------------------------------
// gguf helpers
// ---------------------------------------------------------------------------
static bool kv_u32(gguf_context * g, const char * key, int32_t * out)
{
    const int64_t id = gguf_find_key(g, key);
    if (id < 0) return false;
    switch (gguf_get_kv_type(g, id))
    {
        case GGUF_TYPE_UINT32: *out = (int32_t) gguf_get_val_u32(g, id); return true;
        case GGUF_TYPE_INT32:  *out = gguf_get_val_i32(g, id); return true;
        case GGUF_TYPE_UINT64: *out = (int32_t) gguf_get_val_u64(g, id); return true;
        case GGUF_TYPE_INT64:  *out = (int32_t) gguf_get_val_i64(g, id); return true;
        case GGUF_TYPE_UINT16: *out = (int32_t) gguf_get_val_u16(g, id); return true;
        default: return false;
    }
}

/// A per-layer array key (llama.cpp's get_key_or_arr) whose entries all agree,
/// or a scalar. False when absent, not integral, or not uniform.
static bool kv_u32_uniform(gguf_context * g, const char * key, int32_t * out)
{
    if (kv_u32(g, key, out)) return true;
    const int64_t id = gguf_find_key(g, key);
    if (id < 0 || gguf_get_kv_type(g, id) != GGUF_TYPE_ARRAY) return false;
    const enum gguf_type at = gguf_get_arr_type(g, id);
    const size_t n = gguf_get_arr_n(g, id);
    if (n == 0 || (at != GGUF_TYPE_INT32 && at != GGUF_TYPE_UINT32)) return false;
    const int32_t * data = (const int32_t *) gguf_get_arr_data(g, id);
    for (size_t i = 1; i < n; i++) if (data[i] != data[0]) return false;
    *out = data[0];
    return true;
}

static bool kv_f32(gguf_context * g, const char * key, float * out)
{
    const int64_t id = gguf_find_key(g, key);
    if (id < 0) return false;
    if (gguf_get_kv_type(g, id) == GGUF_TYPE_FLOAT32) { *out = gguf_get_val_f32(g, id); return true; }
    if (gguf_get_kv_type(g, id) == GGUF_TYPE_FLOAT64) { *out = (float) gguf_get_val_f64(g, id); return true; }
    return false;
}

static bool kv_bool(gguf_context * g, const char * key, bool * out)
{
    const int64_t id = gguf_find_key(g, key);
    if (id < 0 || gguf_get_kv_type(g, id) != GGUF_TYPE_BOOL) return false;
    *out = gguf_get_val_bool(g, id);
    return true;
}

static std::string akey(const char * suffix)
{
    return std::string("minimax-m3.") + suffix;
}

/// Any shard of a split set resolves to the full list (shard 1 carries the
/// metadata; the others only carry tensors).
static shard_files resolve_shards(const std::string & any_path, int * split_count_out)
{
    shard_files res;
    std::string first = any_path;
    // "-0000N-of-0000M" -> "-00001-of-0000M"
    const size_t of = first.rfind("-of-");
    if (of != std::string::npos && of >= 6 && first[of - 6] == '-')
    {
        bool digits = true;
        for (size_t i = of - 5; i < of; i++) digits = digits && isdigit((unsigned char) first[i]);
        if (digits) first.replace(of - 5, 5, "00001");
    }

    int32_t split_count = 1;
    {
        ggml_context * meta = nullptr;
        gguf_init_params ip = { true, &meta };
        gguf_context * g = gguf_init_from_file(first.c_str(), ip);
        if (g)
        {
            kv_u32(g, "split.count", &split_count);
            gguf_free(g);
        }
        if (meta) ggml_free(meta);
    }
    if (split_count_out) *split_count_out = split_count;
    if (split_count <= 1) { res.paths.push_back(first); return res; }

    const std::string marker = "-00001-of-";
    const size_t pos = first.find(marker);
    if (pos == std::string::npos) { res.paths.push_back(first); return res; }
    for (int i = 1; i <= split_count; i++)
    {
        char buf[64];
        snprintf(buf, sizeof(buf), "-%05d-of-", i);
        std::string p = first;
        p.replace(pos, marker.size(), buf);
        res.paths.push_back(p);
    }
    return res;
}

static size_t file_size_of(const char * path)
{
#if defined(_WIN32)
    FILE * f = fopen(path, "rb");
    if (!f) return (size_t) -1;
    _fseeki64(f, 0, SEEK_END);
    long long sz = _ftelli64(f);
    fclose(f);
    return sz < 0 ? (size_t) -1 : (size_t) sz;
#else
    struct stat st;
    if (stat(path, &st) != 0) return (size_t) -1;
    return (size_t) st.st_size;
#endif
}

static ggml_backend_buffer_t mmap_shard(model & m, int si)
{
#if defined(_WIN32)
    (void) m; (void) si;
    return nullptr;
#else
    if (si < 0 || si >= (int) m.shards.paths.size()) return nullptr;
    if (m.mmap_bufs.size() < m.shards.paths.size())
    {
        m.mmap_addrs.resize(m.shards.paths.size(), nullptr);
        m.mmap_sizes.resize(m.shards.paths.size(), 0);
        m.mmap_bufs.resize(m.shards.paths.size(), nullptr);
    }
    if (m.mmap_bufs[si]) return m.mmap_bufs[si];

    const int fd = open(m.shards.paths[si].c_str(), O_RDONLY);
    if (fd < 0) return nullptr;
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size <= 0) { close(fd); return nullptr; }
    void * addr = mmap(nullptr, (size_t) st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (addr == MAP_FAILED) return nullptr;

    ggml_backend_buffer_t buf = ggml_backend_cpu_buffer_from_ptr(addr, (size_t) st.st_size);
    if (!buf) { munmap(addr, (size_t) st.st_size); return nullptr; }
    // WEIGHTS: ggml_backend_sched pins a node to the backend owning a WEIGHTS
    // input. Left at ANY, the offloaded mul_mat_id would be scheduled on a GPU
    // and the whole expert stack copied over the bus every token.
    ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    m.mmap_addrs[si] = addr;
    m.mmap_sizes[si] = (size_t) st.st_size;
    m.mmap_bufs[si] = buf;
    return buf;
#endif
}

struct load_job
{
    ggml_tensor * t;
    int shard;
    size_t file_off;
    size_t tensor_off;
    size_t len;          // bytes read from the file
    bool to_f16;         // convert F32 -> F16 (len / 2 bytes written)
};

/// The outcome of a team of loader threads: the first failure's reason. The
/// workers poll `failed` to stop early.
struct team_status
{
    std::atomic<bool> failed{ false };
    std::mutex mu;
    std::string error;

    void fail(const std::string & why)
    {
        std::lock_guard<std::mutex> lk(mu);
        if (error.empty()) error = why;
        failed.store(true);
    }
};

/// Starts a thread; as a thread quota does, the test hooks can make it throw.
template <typename... A>
static void start_thread(std::vector<std::thread> & threads, A &&... args)
{
#if defined(TSG_GGML_TEST_HOOKS)
    if (test_fault_due(g_test_fail_thread))
        throw std::system_error(std::make_error_code(std::errc::resource_unavailable_try_again), "injected thread start failure");
#endif
    threads.emplace_back(std::forward<A>(args)...);
}

/// Runs work(i) for i in [0, n) on n threads and waits for all of them. A
/// thread that cannot start (std::system_error under a pids or ulimit quota)
/// or a worker that throws marks `st` failed with the reason, and every thread
/// that did start is joined on every path: a joinable std::thread destroyed by
/// an unwinding exception, or an exception leaving a thread function,
/// terminates the process instead of refusing the load. False on failure.
template <typename F>
static bool run_team(int n, const char * what, team_status & st, F && work)
{
    auto body = [&](int i)
    {
        try { work(i); }
        catch (const std::exception & e) { st.fail(std::string(what) + " " + std::to_string(i) + " failed: " + e.what()); }
        catch (...) { st.fail(std::string(what) + " " + std::to_string(i) + " failed"); }
    };
    std::vector<std::thread> threads;
    try
    {
        threads.reserve((size_t) std::max(n, 0));
        for (int i = 0; i < n; i++) start_thread(threads, body, i);
    }
    catch (const std::exception & e)
    {
        st.fail("cannot start " + std::string(what) + " " + std::to_string(threads.size() + 1) + " of " + std::to_string(n) +
                ": " + e.what());
    }
    for (auto & t : threads) t.join();
    return !st.failed.load();
}

/// `set_mu`, when given, serializes the device writes (a meta buffer's
/// split-state cache is not thread-safe); the file reads stay parallel. False
/// with the reason in *why on failure.
static bool upload_parallel(const shard_files & shards, std::vector<load_job> & jobs, int n_threads,
                            std::atomic<size_t> & done_bytes, std::string * why, std::mutex * set_mu = nullptr)
{
    std::sort(jobs.begin(), jobs.end(), [](const load_job & a, const load_job & b)
    {
        if (a.shard != b.shard) return a.shard < b.shard;
        return a.file_off < b.file_off;
    });
    if (n_threads > (int) jobs.size()) n_threads = (int) jobs.size();
    if (n_threads < 1) n_threads = 1;

    std::atomic<size_t> cursor(0);
    team_status st;
    const bool drop_cache = env_flag("TS_MINIMAX_M3_DROP_UPLOAD_CACHE", true);

    // The descriptors close on every path: a worker may throw (bad_alloc).
    struct file_set
    {
        std::vector<FILE *> f;
        explicit file_set(size_t n) : f(n, nullptr) {}
        ~file_set() { for (FILE * x : f) if (x) fclose(x); }
    };
    auto worker = [&](int)
    {
        file_set files(shards.paths.size());
        std::vector<uint8_t> staging;
        std::vector<ggml_fp16_t> half;
        while (!st.failed.load(std::memory_order_relaxed))
        {
            const size_t i = cursor.fetch_add(1, std::memory_order_relaxed);
            if (i >= jobs.size()) break;
            const load_job & j = jobs[i];
#if defined(TSG_GGML_TEST_HOOKS)
            if (test_fault_due(g_test_fail_job)) throw std::bad_alloc();
#endif
            FILE *& f = files.f[(size_t) j.shard];
            if (!f)
            {
                f = fopen(shards.paths[(size_t) j.shard].c_str(), "rb");
                if (!f)
                {
                    st.fail("cannot open " + shards.paths[(size_t) j.shard]);
                    break;
                }
            }
            if (staging.size() < j.len) staging.resize(j.len);
#if defined(_WIN32)
            _fseeki64(f, (long long) j.file_off, SEEK_SET);
#else
            fseeko(f, (off_t) j.file_off, SEEK_SET);
#endif
            if (fread(staging.data(), 1, j.len, f) != j.len)
            {
                st.fail(std::string("short read for ") + j.t->name + " at offset " + std::to_string(j.file_off));
                break;
            }
            if (j.to_f16)
            {
                const size_t n = j.len / sizeof(float);
                half.resize(n);
                ggml_fp32_to_fp16_row((const float *) staging.data(), half.data(), (int64_t) n);
                std::unique_lock<std::mutex> lk;
                if (set_mu) lk = std::unique_lock<std::mutex>(*set_mu);
                ggml_backend_tensor_set(j.t, half.data(), j.tensor_off, n * sizeof(ggml_fp16_t));
            }
            else
            {
                std::unique_lock<std::mutex> lk;
                if (set_mu) lk = std::unique_lock<std::mutex>(*set_mu);
                ggml_backend_tensor_set(j.t, staging.data(), j.tensor_off, j.len);
            }
            // The device copy is the only one this process needs: drop the file
            // pages so a GPU-bound read does not evict the mapped CPU experts
            // from a container's page-cache budget. Only Linux can drop one
            // range: macOS declares neither posix_fadvise nor POSIX_FADV_DONTNEED
            // and offers whole-descriptor F_NOCACHE instead (as the DSV4 loader
            // does). Anywhere else the request has no effect and says so.
            if (drop_cache)
            {
#if defined(__linux__)
                posix_fadvise(fileno(f), (off_t) j.file_off, (off_t) j.len, POSIX_FADV_DONTNEED);
#elif defined(__APPLE__)
                // Idempotent and per-descriptor: from here this handle's reads
                // bypass the page cache entirely.
                fcntl(fileno(f), F_NOCACHE, 1);
#else
                static std::once_flag warned;
                std::call_once(warned, []() {
                    fprintf(stderr, "[minimax-m3] TS_MINIMAX_M3_DROP_UPLOAD_CACHE has no effect on this platform; "
                                    "the checkpoint's pages stay in the page cache\n");
                });
#endif
            }
            done_bytes.fetch_add(j.len, std::memory_order_relaxed);
        }
    };

    if (run_team(n_threads, "loader thread", st, worker)) return true;
    if (why) *why = st.error;
    return false;
}

static bool ieq(const char * a, const char * b)
{
    if (!a || !b) return false;
    for (; *a && *b; ++a, ++b)
        if (tolower((unsigned char) *a) != tolower((unsigned char) *b)) return false;
    return *a == *b;
}

static bool backend_matches(const char * reg_name, const char * want)
{
    if (!want || !*want) return true;
    if (!reg_name) return false;
    if (ieq(want, "CUDA")) return ieq(reg_name, "CUDA") || ieq(reg_name, "ROCm") || ieq(reg_name, "MUSA");
    if (ieq(want, "Metal")) return ieq(reg_name, "Metal") || ieq(reg_name, "MTL");
    return ieq(reg_name, want);
}

static size_t host_mem_allowance()
{
    size_t limit = 0;
#ifdef __linux__
    if (FILE * f = fopen("/sys/fs/cgroup/memory.max", "r"))
    {
        char v[64] = { 0 };
        if (fscanf(f, "%63s", v) == 1 && strcmp(v, "max") != 0) limit = (size_t) atoll(v);
        fclose(f);
    }
    if (limit == 0)
    {
        if (FILE * f = fopen("/sys/fs/cgroup/memory/memory.limit_in_bytes", "r"))
        {
            long long v = 0;
            if (fscanf(f, "%lld", &v) == 1 && v > 0 && v < (1ll << 62)) limit = (size_t) v;
            fclose(f);
        }
    }
    if (limit == 0)
    {
        if (FILE * f = fopen("/proc/meminfo", "r"))
        {
            char line[256];
            while (fgets(line, sizeof(line), f))
            {
                long long kb = 0;
                if (sscanf(line, "MemTotal: %lld kB", &kb) == 1) { limit = (size_t) kb * 1024; break; }
            }
            fclose(f);
        }
    }
#endif
    return limit;
}

// ---------------------------------------------------------------------------
// NUMA placement of host experts
//
// Decoding reads each selected expert once per token, so the host experts'
// throughput is memory bandwidth. On a two-socket host, pages placed wherever
// the loader happened to touch them make about half of every read cross the
// socket link: the measured rate was ~100 GB/s against ~170 GB/s with every
// worker reading local memory. ggml's CPU mul_mat_id statically gives worker
// `ith` rows [ith*dr0, (ith+1)*dr0) of each routed expert when one token
// routes to it (decode), so pinning the workers and placing each worker's rows
// on its own node by first touch makes those reads local. No memory-policy
// syscalls are needed (containers commonly forbid them).
// ---------------------------------------------------------------------------
#if defined(__linux__)
static std::vector<int> parse_cpulist(const char * path)
{
    std::vector<int> out;
    FILE * f = fopen(path, "r");
    if (!f) return out;
    char buf[4096] = { 0 };
    if (fgets(buf, sizeof(buf), f))
        for (char * q = buf; *q && *q != '\n';)
        {
            int a = (int) strtol(q, &q, 10), b = a;
            if (*q == '-') b = (int) strtol(q + 1, &q, 10);
            for (int c = a; c <= b; c++) out.push_back(c);
            if (*q == ',') q++;
            else if (*q && *q != '\n') break;
        }
    fclose(f);
    return out;
}

/// Allowed CPUs per NUMA node, one hyper-thread per core first.
static std::vector<std::vector<int>> numa_node_cpus()
{
    std::vector<std::vector<int>> nodes;
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) return nodes;
    for (int n = 0; n < 64; n++)
    {
        char path[128];
        snprintf(path, sizeof(path), "/sys/devices/system/node/node%d/cpulist", n);
        std::vector<int> cpus = parse_cpulist(path);
        if (cpus.empty()) { if (n > 0) break; else continue; }
        std::vector<int> first, rest;
        std::vector<uint8_t> seen;
        for (int c : cpus)
        {
            if (c < 0 || c >= CPU_SETSIZE || !CPU_ISSET(c, &allowed)) continue;
            snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", c);
            std::vector<int> sib = parse_cpulist(path);
            const int core = sib.empty() ? c : *std::min_element(sib.begin(), sib.end());
            if ((size_t) core >= seen.size()) seen.resize((size_t) core + 1, 0);
            (seen[(size_t) core] ? rest : first).push_back(c);
            seen[(size_t) core] = 1;
        }
        first.insert(first.end(), rest.begin(), rest.end());
        if (!first.empty()) nodes.push_back(first);
    }
    return nodes;
}

/// `nth` worker CPUs spread evenly across the nodes, ascending (the order in
/// which ggml's strict cpumask hands them out).
static std::vector<int> numa_worker_cpus(const std::vector<std::vector<int>> & nodes, int nth)
{
    std::vector<int> out;
    const int nn = (int) nodes.size();
    for (int k = 0; k < nn; k++)
    {
        const int take = nth / nn + (k < nth % nn ? 1 : 0);
        for (int i = 0; i < take; i++) out.push_back(nodes[(size_t) k][(size_t) i % nodes[(size_t) k].size()]);
    }
    std::sort(out.begin(), out.end());
    return out;
}

static void probe_cpu_cb(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void * user)
{
    (void) dst; (void) a; (void) nth;
    auto * v = (std::vector<int> *) user;
    if (ith >= 0 && ith < (int) v->size()) (*v)[(size_t) ith] = sched_getcpu();
}

/// The CPU each worker of `cpu`'s threadpool actually runs on, by ith.
static std::vector<int> probe_worker_cpus(ggml_backend_t cpu, int nth)
{
    std::vector<int> v((size_t) nth, -1);
    ggml_init_params p = { 8 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * c = ggml_init(p);
    ggml_tensor * a = ggml_new_tensor_1d(c, GGML_TYPE_F32, 1);
    ggml_tensor * out = ggml_map_custom1(c, a, probe_cpu_cb, nth, &v);
    ggml_cgraph * gf = ggml_new_graph(c);
    ggml_build_forward_expand(gf, out);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(c, cpu);
    if (buf)
    {
        // Several runs: a worker that has not started yet in one run still
        // reports from its pinned CPU in the next.
        for (int r = 0; r < 3; r++) ggml_backend_graph_compute(cpu, gf);
        ggml_backend_buffer_free(buf);
    }
    ggml_free(c);
    return v;
}
#endif

// ---------------------------------------------------------------------------
// Host experts for a few tokens (batched decode)
//
// ggml's CPU mul_mat_id keeps worker `ith` on rows [ith*dr0, (ith+1)*dr0) only
// while each expert serves one token. When two tokens of a batched decode step
// route to the same expert, that expert switches to dynamically claimed 16-row
// chunks, so workers read rows another socket placed (eng/validation/
// muse-spark-cpu-moe-bench.cpp: 188 -> 143 GB/s with half the experts shared).
// This kernel keeps the static row ownership for any batch and reads each
// weight row once for every token that selected its expert. It uses ggml's own
// from_float and vec_dot for the weight type, so results are bit-identical.
// ---------------------------------------------------------------------------

/// host_moe_quantize's userdata: the vec_dot type the rows are quantized to.
/// A custom op's userdata is a pointer that other code may dereference (the
/// fused-op wrappers check every custom node for their descriptor), so it
/// points at one of these process-lifetime objects, never at a cast integer.
/// The tag keeps it from reading as any descriptor.
struct alignas(16) host_moe_quant_args
{
    uint64_t tag = 0x4D4D33514E545950ull;   // "MM3QNTYP"
    ggml_type type = GGML_TYPE_COUNT;
};

static void * host_moe_quant_userdata(ggml_type t)
{
    static const std::array<host_moe_quant_args, GGML_TYPE_COUNT> args = []
    {
        std::array<host_moe_quant_args, GGML_TYPE_COUNT> a;
        for (int i = 0; i < GGML_TYPE_COUNT; i++) a[(size_t) i].type = (ggml_type) i;
        return a;
    }();
    return (void *) &args[(size_t) t];
}

/// args: a [K, R, T] F32 -> dst I8 holding R*T rows of the type in `userdata` (host_moe_quant_args).
static void host_moe_quantize(ggml_tensor * dst, int ith, int nth, void * userdata)
{
    const ggml_tensor * a = dst->src[0];
    const ggml_type vdt = ((const host_moe_quant_args *) userdata)->type;
    const auto * tr = ggml_get_type_traits_cpu(vdt);
    const size_t row = ggml_row_size(vdt, a->ne[0]);
    const int64_t rows = a->ne[1] * a->ne[2];
    for (int64_t i = ith; i < rows; i += nth)
    {
        const int64_t r = i % a->ne[1], t = i / a->ne[1];
        tr->from_float((const float *) ((const char *) a->data + r * a->nb[1] + t * a->nb[2]),
                       (char *) dst->data + (size_t) i * row, a->ne[0]);
    }
}

/// args: w [K, N, E], aq (host_moe_quantize of [K, R, T], R = 1 or n_used), ids [n_used, T]
/// -> dst [N, n_used, T] F32, the same result as ggml_mul_mat_id(w, a, ids).
static void host_moe_matmul(ggml_tensor * dst, int ith, int nth, void * userdata)
{
    (void) userdata;
    const ggml_tensor * w = dst->src[0];
    const ggml_tensor * aq = dst->src[1];
    const ggml_tensor * ids = dst->src[2];
    const auto * wt = ggml_get_type_traits_cpu(w->type);
    const int64_t K = w->ne[0], N = w->ne[1], n_used = ids->ne[0], T = ids->ne[1];
    const size_t arow = ggml_row_size(wt->vec_dot_type, K);
    const int64_t rows_per_tok = aq->ne[1];
    const int64_t dr0 = (N + nth - 1) / nth;
    const int64_t r0 = ith * dr0, r1 = std::min<int64_t>(N, r0 + dr0);
    if (r0 >= r1) return;

    struct pick { int32_t e, s, t; };
    pick picks[HOST_MOE_MAX_PICKS];
    int64_t np = 0;
    for (int64_t t = 0; t < T; t++)
        for (int64_t s = 0; s < n_used; s++)
            picks[np++] = { *(const int32_t *) ((const char *) ids->data + s * ids->nb[0] + t * ids->nb[1]),
                            (int32_t) s, (int32_t) t };
    std::sort(picks, picks + np, [](const pick & a, const pick & b) { return a.e < b.e; });
    for (int64_t i = 0; i < np;)
    {
        int64_t j = i;
        while (j < np && picks[j].e == picks[i].e) j++;
        const char * we = (const char *) w->data + (size_t) picks[i].e * w->nb[2];
        for (int64_t r = r0; r < r1; r++)
        {
            const char * wr = we + (size_t) r * w->nb[1];
            for (int64_t p = i; p < j; p++)
            {
                const pick & k = picks[p];
                const char * ar = (const char *) aq->data + (size_t) (k.t * rows_per_tok + (rows_per_tok == 1 ? 0 : k.s)) * arow;
                float * out = (float *) ((char *) dst->data + k.s * dst->nb[1] + k.t * dst->nb[2]);
                wt->vec_dot((int) K, out + r, 0, wr, 0, ar, 0, 1);
            }
        }
        i = j;
    }
}


// ---------------------------------------------------------------------------
// slots
// ---------------------------------------------------------------------------
/// Cache bytes one token takes on a layer: K and V, plus the index key on a
/// sparse layer.
static size_t kv_bytes_per_token_layer(const hparams & hp, bool msa)
{
    return ((size_t) hp.n_kv() * 2 + (msa ? (size_t) hp.idx_dim : 0)) * sizeof(uint16_t);
}

/// The first `rows` cache rows of `src`, copied into `dst`.
struct row_copy { ggml_tensor * src; ggml_tensor * dst; int64_t rows; };

/// Copy cache rows device to device. The shapes match because every slot is
/// allocated with the same geometry; rows grow per slot, so a source and its
/// destination may differ in length. A slot's layer caches sit on the layer's
/// GPU, so the copies are enqueued on that GPU's stream and each GPU is waited
/// for once (ggml_backend_tensor_copy waits after every copy: 180 waits for a
/// 60-layer slot, on the path of every turn that clones a cached prefix).
static bool copy_cache_rows(model & m, const std::vector<row_copy> & copies)
{
    if (m.tp.n > 1)
    {
        std::vector<tsg_mm3_tp::row_copy> tc;
        for (const row_copy & rc : copies) tc.push_back({ rc.src, rc.dst, rc.rows });
        const bool ok = tsg_mm3_tp::copy_rows(m.backends[0], tc);
        m.graph_cache.clear();   // the meta backend kept only the copy graph's views
        return ok;
    }
    auto backend_of = [&](const ggml_tensor * t) -> ggml_backend_t
    {
        ggml_backend_buffer_t buf = t->view_src ? t->view_src->buffer : t->buffer;
        if (!buf) return nullptr;
        ggml_backend_dev_t dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(buf));
        for (int d = 0; d < m.n_gpu; d++)
            if (ggml_backend_get_device(m.backends[d]) == dev) return m.backends[d];
        return nullptr;
    };
    std::vector<ggml_backend_t> pending;
    for (const row_copy & rc : copies)
    {
        if (!rc.src || !rc.dst) continue;
        const int64_t rows = std::min<int64_t>(rc.rows, std::min<int64_t>(rc.src->ne[1], rc.dst->ne[1]));
        if (rows <= 0) continue;
        ggml_backend_t be = backend_of(rc.src);
        if (be && be != backend_of(rc.dst)) be = nullptr;
        if (be && std::find(pending.begin(), pending.end(), be) == pending.end()) pending.push_back(be);
        if (rows == rc.src->ne[1] && rows == rc.dst->ne[1])
        {
            if (be) ggml_backend_tensor_copy_async(be, be, rc.src, rc.dst);
            else ggml_backend_tensor_copy(rc.src, rc.dst);
            continue;
        }
        ggml_init_params p = { 4 * ggml_tensor_overhead(), nullptr, true };
        ggml_context * c = ggml_init(p);
        ggml_tensor * vs = ggml_view_2d(c, rc.src, rc.src->ne[0], rows, rc.src->nb[1], 0);
        ggml_tensor * vd = ggml_view_2d(c, rc.dst, rc.dst->ne[0], rows, rc.dst->nb[1], 0);
        ggml_backend_view_init(vs);
        ggml_backend_view_init(vd);
        // An enqueued copy holds the addresses, not the views: they can go now.
        if (be) ggml_backend_tensor_copy_async(be, be, vs, vd);
        else ggml_backend_tensor_copy(vs, vd);
        ggml_free(c);
    }
    for (ggml_backend_t be : pending) ggml_backend_synchronize(be);
    return true;
}

static void dev_memory(const model & m, int d, size_t * free_b, size_t * total_b)
{
    *free_b = 0;
    *total_b = 0;
    if (m.tp.n > 1 && d == 0) return tsg_mm3_tp::memory(m.tp.dev, free_b, total_b);   // the tightest GPU, not the sum
    ggml_backend_dev_memory(ggml_backend_get_device(m.backends[d]), free_b, total_b);
}

/// Allocate a slot's caches with `rows` rows in fresh per-device buffers and
/// point the slot's tensors at them. `admit` checks that each device keeps room
/// for the largest graph it has run. False, with nothing allocated or changed,
/// on failure.
/// Free the pooled caches of freed slots (memory is short). True if any were.
static bool slot_pool_release(model & m)
{
    if (m.slot_pool.empty()) return false;
    for (auto & ps : m.slot_pool)
        for (auto & sc : ps.ctxs)
        {
            if (sc.buf) ggml_backend_buffer_free(sc.buf);
            if (sc.ctx) ggml_free(sc.ctx);
        }
    m.slot_pool.clear();
    m.pipe.fail_u = m.pipe.fail_kv = 0;   // memory came back: retry
    return true;
}

static bool slot_alloc_rows(model & m, slot & s, int64_t rows, bool admit)
{
    const int nl = m.hp.n_layer;
    std::vector<ggml_context *> ctxs((size_t) m.n_gpu + 1, nullptr);
    for (int d = 0; d <= m.n_gpu; d++)
    {
        ggml_init_params p = { (size_t) (3 * nl + 8) * ggml_tensor_overhead(), nullptr, true };
        ctxs[(size_t) d] = ggml_init(p);
    }
    std::vector<ggml_tensor *> k((size_t) nl, nullptr), v((size_t) nl, nullptr), ik((size_t) nl, nullptr);
    for (int il = 0; il < nl; il++)
    {
        const layer & L = m.layers[(size_t) il];
        ggml_context * c = ctxs[(size_t) L.device];
        k[(size_t) il] = ggml_new_tensor_2d(c, GGML_TYPE_F16, m.hp.n_kv(), rows);
        v[(size_t) il] = ggml_new_tensor_2d(c, GGML_TYPE_F16, m.hp.n_kv(), rows);
        ggml_format_name(k[(size_t) il], "slot%d_k.%d", s.id, il);
        ggml_format_name(v[(size_t) il], "slot%d_v.%d", s.id, il);
        if (L.msa)
        {
            ik[(size_t) il] = ggml_new_tensor_2d(c, GGML_TYPE_F16, m.hp.idx_dim, rows);
            ggml_format_name(ik[(size_t) il], "slot%d_ik.%d", s.id, il);
        }
    }

    if (admit)
    {
        // Each device must keep room to run the largest graph it has needed so
        // far, plus a reserve; otherwise a running request's next graph fails.
        // Cached graphs only cache: their compute buffers count as reclaimable,
        // and the cache is dropped when that is what it takes to fit.
        // The graph room is capped (TS_MINIMAX_M3_ADMIT_GRAPH_MB): one long
        // prefill's graph would otherwise make every later slot evict retained
        // prefix state (shared-prompt checkpoints included) or be refused, while
        // a graph that later does not fit only frees retained slots and runs
        // that call in smaller ubatches.
        const size_t reserve = (size_t) env_long("TS_MINIMAX_M3_SLOT_HEADROOM_MB", 256) * 1024 * 1024;
        const size_t graph_cap = (size_t) std::max(0L, env_long("TS_MINIMAX_M3_ADMIT_GRAPH_MB", 512)) * 1024 * 1024;
        bool drop_graphs = false;
        for (int d = 0; d < m.n_gpu; d++)
        {
            if (!ggml_get_first_tensor(ctxs[(size_t) d])) continue;
            size_t free_b = 0, total_b = 0;
            dev_memory(m, d, &free_b, &total_b);
            if (!total_b) continue;
            size_t need = 0;
            for (ggml_tensor * t = ggml_get_first_tensor(ctxs[(size_t) d]); t; t = ggml_get_next_tensor(ctxs[(size_t) d], t))
                need += ggml_nbytes(t) + 512;
            size_t cached = 0, largest = d < (int) m.max_graph_bytes.size() ? m.max_graph_bytes[(size_t) d] : 0;
            for (const auto & e : m.graph_cache)
                if (e->sched)
                {
                    const size_t b = ggml_backend_sched_get_buffer_size(e->sched, m.sched_backends[d]);
                    cached += b;
                    largest = std::max(largest, b);
                }
            // The pipelined prefill's buffers are rebuilt on its next call.
            if (m.pipe.sched) cached += ggml_backend_sched_get_buffer_size(m.pipe.sched, m.sched_backends[d]);
            // Under --tp the room stays whole: a graph that does not fit is measured
            // and declined before ggml's meta backend could abort on it, but a call
            // retries in smaller ubatches only a few times there (forward).
            if (m.tp.n <= 1) largest = std::min(largest, graph_cap);
            if (free_b + cached < need + largest + reserve && slot_pool_release(m))
            {
                for (auto c : ctxs) if (c) ggml_free(c);
                return slot_alloc_rows(m, s, rows, admit);   // once more without the pooled caches
            }
            if (free_b + cached < need + largest + reserve)
            {
                fprintf(stderr, "[minimax-m3] no room for %" PRId64 " K/V rows of slot %d on device %d (needs %.0f MiB + %.0f MiB for "
                        "graphs, %.0f MiB free + %.0f MiB of cached graphs)\n", rows, s.id, d, need / 1048576.0,
                        (largest + reserve) / 1048576.0, free_b / 1048576.0, cached / 1048576.0);
                for (auto c : ctxs) if (c) ggml_free(c);
                return false;
            }
            if (free_b < need + reserve) drop_graphs = true;
        }
        if (drop_graphs) { m.graph_cache.clear(); m.pipe.release(); }
    }

    std::vector<ggml_backend_buffer_t> bufs((size_t) m.n_gpu + 1, nullptr);
    bool ok = true;
    for (int d = 0; d <= m.n_gpu && ok; d++)
    {
        if (!ggml_get_first_tensor(ctxs[(size_t) d])) continue;
        bufs[(size_t) d] = ggml_backend_alloc_ctx_tensors(ctxs[(size_t) d], m.backends[d]);
        if (!bufs[(size_t) d]) ok = false;
        // Zeroed: rows past a sequence's head are read under a -inf mask, and
        // 0 * -inf would be NaN where 0 is not.
        else ggml_backend_buffer_clear(bufs[(size_t) d], 0);
    }
    if (!ok)
    {
        for (auto b : bufs) if (b) ggml_backend_buffer_free(b);
        for (auto c : ctxs) if (c) ggml_free(c);
        if (slot_pool_release(m)) return slot_alloc_rows(m, s, rows, admit);
        fprintf(stderr, "[minimax-m3] K/V allocation failed for slot %d\n", s.id);
        return false;
    }
    for (int d = 0; d <= m.n_gpu; d++)
    {
        if (!bufs[(size_t) d]) { if (ctxs[(size_t) d]) ggml_free(ctxs[(size_t) d]); continue; }
        m.slot_ctxs.push_back({ s.id, ctxs[(size_t) d], bufs[(size_t) d] });
    }
    s.k = k;
    s.v = v;
    s.ik = ik;
    s.cap = rows;
    return true;
}

static bool reclaim_for_graph(model & m, const int32_t * reads, int n_reads);

static void drop_slot_graphs(model & m, int slot_id)
{
    for (auto it = m.graph_cache.begin(); it != m.graph_cache.end(); )
    {
        bool refs = (*it)->slot_id == slot_id;
        for (const bd_token & b : (*it)->bd) if (b.slot_id == slot_id) refs = true;
        it = refs ? m.graph_cache.erase(it) : std::next(it);
    }
}

static int64_t slot_init_rows(const model & m)
{
    const int64_t want = std::max<int64_t>(m.kv_pad, env_long("TS_MINIMAX_M3_SLOT_INIT_ROWS", 4096));
    return std::min<int64_t>(m.n_ctx, pad_to(want, m.kv_pad));
}

/// Grow slot `s`'s rows to at least `rows` (doubling), copying the rows it
/// holds. Graphs bound to the old rows are dropped. When memory is short,
/// retained slots are freed for the growth, never `s` itself nor the `n_protect`
/// slots in `protect` (the slots the caller reads next: a slot copy's source,
/// the other segments of a batched graph).
static bool slot_reserve_rows(model & m, slot & s, int64_t rows, const int32_t * protect = nullptr, int n_protect = 0)
{
    if (rows <= s.cap) return true;
    if (rows > m.n_ctx) return false;
    int64_t cap = std::max<int64_t>(rows, std::min<int64_t>(m.n_ctx, 2 * s.cap));
    cap = std::min<int64_t>(m.n_ctx, pad_to(cap, m.kv_pad));
    const std::vector<ggml_tensor *> old_k = s.k, old_v = s.v, old_ik = s.ik;
    std::vector<model::slot_ctx> old;
    for (auto it = m.slot_ctxs.begin(); it != m.slot_ctxs.end(); )
    {
        if (it->slot_id == s.id) { old.push_back(*it); it = m.slot_ctxs.erase(it); }
        else ++it;
    }
    const int64_t old_cap = s.cap;
    bool ok = slot_alloc_rows(m, s, cap, true);
    if (!ok)
    {
        // Cached graphs (the pipeline's included) and retained slots give their memory back first.
        m.graph_cache.clear();
        m.pipe.release();
        std::vector<int32_t> reads(protect, protect + std::max(0, n_protect));
        reads.push_back(s.id);
        while (!ok && reclaim_for_graph(m, reads.data(), (int) reads.size())) ok = slot_alloc_rows(m, s, cap, true);
        // Unchecked, an allocation that does not fit aborts ggml's meta backend (--tp).
        if (!ok && m.tp.n <= 1) ok = slot_alloc_rows(m, s, cap, false);
    }
    if (!ok)
    {
        for (auto & o : old) m.slot_ctxs.push_back(o);
        return false;
    }
    const int64_t keep = std::min(old_cap, s.n_past);
    std::vector<row_copy> copies;
    for (int il = 0; il < m.hp.n_layer; il++)
    {
        copies.push_back({ old_k[(size_t) il], s.k[(size_t) il], keep });
        copies.push_back({ old_v[(size_t) il], s.v[(size_t) il], keep });
        copies.push_back({ old_ik[(size_t) il], s.ik[(size_t) il], keep });
    }
    const bool copied = copy_cache_rows(m, copies);
    if (!copied) fprintf(stderr, "[minimax-m3] copying slot %d's cache rows into its grown cache failed\n", s.id);
    for (auto & o : old)
    {
        if (o.buf) ggml_backend_buffer_free(o.buf);
        if (o.ctx) ggml_free(o.ctx);
    }
    drop_slot_graphs(m, s.id);
    if (debug_on()) fprintf(stderr, "[minimax-m3] slot %d rows %" PRId64 " -> %" PRId64 "\n", s.id, old_cap, cap);
    return copied;
}

/// Give `s` the caches of a pooled freed slot, cleared as a fresh allocation is.
static bool slot_pool_take(model & m, slot & s)
{
    if (m.slot_pool.empty()) return false;
    model::pooled_slot ps = std::move(m.slot_pool.back());
    m.slot_pool.pop_back();
    for (auto & sc : ps.ctxs)
    {
        if (sc.buf) ggml_backend_buffer_clear(sc.buf, 0);
        sc.slot_id = s.id;
        m.slot_ctxs.push_back(sc);
    }
    s.k = std::move(ps.k);
    s.v = std::move(ps.v);
    s.ik = std::move(ps.ik);
    s.cap = ps.cap;
    return true;
}

static slot * slot_alloc(model & m, bool admit)
{
    auto s = std::unique_ptr<slot>(new slot());
    s->id = m.next_slot_id++;
    if (!slot_pool_take(m, *s) && !slot_alloc_rows(m, *s, slot_init_rows(m), admit)) return nullptr;
    dspark_slot_added(m, *s);
    slot * raw = s.get();
    m.slots[s->id] = std::move(s);
    return raw;
}

// Freed slots whose caches stay pooled for reuse (slot_pool_take).
static constexpr size_t SLOT_POOL_MAX = 2;

/// Free a slot. Its caches go to the pool when they never grew past a new
/// slot's rows (pool_ok and room in the pool), and are freed otherwise.
static void slot_free(model & m, int slot_id, bool pool_ok = true)
{
    drop_slot_graphs(m, slot_id);
    dspark_slot_removed(m, slot_id);
    auto si = m.slots.find(slot_id);
    const bool pool = pool_ok && si != m.slots.end() && si->second->cap == slot_init_rows(m) &&
                      m.slot_pool.size() < SLOT_POOL_MAX;
    model::pooled_slot ps;
    if (pool)
    {
        ps.k = si->second->k;
        ps.v = si->second->v;
        ps.ik = si->second->ik;
        ps.cap = si->second->cap;
    }
    else
        m.pipe.fail_u = m.pipe.fail_kv = 0;   // memory came back: retry
    m.slots.erase(slot_id);
    m.reclaimable.erase(std::remove(m.reclaimable.begin(), m.reclaimable.end(), slot_id), m.reclaimable.end());
    for (auto it = m.slot_ctxs.begin(); it != m.slot_ctxs.end(); )
    {
        if (it->slot_id != slot_id) { ++it; continue; }
        if (pool) ps.ctxs.push_back(*it);
        else
        {
            if (it->buf) ggml_backend_buffer_free(it->buf);
            if (it->ctx) ggml_free(it->ctx);
        }
        it = m.slot_ctxs.erase(it);
    }
    if (pool) m.slot_pool.push_back(std::move(ps));
}

static bool reclaim_for_graph(model & m, const int32_t * reads, int n_reads)
{
    if (slot_pool_release(m)) return true;   // pooled caches of freed slots go first
    for (size_t i = 0; i < m.reclaimable.size(); i++)
    {
        const int id = m.reclaimable[i];
        if (std::find(reads, reads + n_reads, id) != reads + n_reads) continue;
        if (m.active_slot && m.active_slot->id == id) continue;
        if (!m.slots.count(id))
        {
            m.reclaimable.erase(m.reclaimable.begin() + (std::ptrdiff_t) i--);
            continue;
        }
        slot_free(m, id, false);   // the memory is what the graph needs
        m.reclaimed.push_back(id);
        fprintf(stderr, "[minimax-m3] freed retained sequence slot %d for a graph that did not fit\n", id);
        return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// windows
// ---------------------------------------------------------------------------
/// Attention reads cache rows [0, n_kv). `pad` is a multiple of the MSA block,
/// so the rows always split into whole blocks.
static int64_t plan_n_kv(const model & m, int64_t p_end, int64_t pad)
{
    return std::min<int64_t>(pad_to(std::max<int64_t>(p_end, 1), pad), m.n_ctx);
}

/// Whether a segment whose last position is p_end - 1 runs sparse attention
/// on the MSA layers (TS_MINIMAX_M3_MSA). Below dense_span every block a query
/// can see is selected, so dense attention is exact there.
static bool segment_sparse(const model & m, int64_t p_end)
{
    if (m.n_msa_layers == 0) return false;
    switch (m.msa)
    {
        case MSA_DENSE: return false;
        case MSA_AUTO: return p_end > m.hp.dense_span();
        default: return true;
    }
}

/// Tokens per block-mask chunk of a sparse segment of `len` tokens over
/// `n_kv` rows: one chunk's F32 causal rows, scores [n_kv, heads, tokens], F32
/// block mask and F16 flash-attention mask stay within the byte budget
/// (TS_MINIMAX_M3_MSA_CHUNK_MB), and every [n_kv, heads, tokens] tensor below
/// 2^31 elements (CUDA's pool2d indexes with int). The chunks are balanced.
static int64_t msa_chunk_tokens(const model & m, int64_t n_kv, int64_t len)
{
    if (len <= 1) return 0;
    const int64_t heads = m.hp.idx_n_head;
    const size_t per_token = (size_t) n_kv * (sizeof(float) + (size_t) heads * (2 * sizeof(float) + sizeof(uint16_t) + sizeof(float)));
    int64_t c = (int64_t) std::max<size_t>(1, m.msa_chunk_bytes / std::max<size_t>(1, per_token));
    const int64_t elem_cap = (int64_t) (INT32_MAX / std::max<int64_t>(1, n_kv * heads));
    c = std::max<int64_t>(1, std::min<int64_t>({ c, elem_cap, len }));
    const int64_t n_chunks = (len + c - 1) / c;
    return (len + n_chunks - 1) / n_chunks;
}

static constexpr uint16_t F16_NEG_INF = 0xFC00;

/// Causal mask rows for queries at positions p0..p0+nt-1 over cache rows
/// [0, n_kv): row j is visible to position q when j <= q.
static void fill_mask(uint16_t * out, int64_t n_kv, int64_t p0, int64_t nt)
{
    for (int64_t t = 0; t < nt; t++)
    {
        uint16_t * row = out + (size_t) (t * n_kv);
        const int64_t visible = std::min<int64_t>(n_kv, p0 + t + 1);
        std::fill_n(row, visible, (uint16_t) 0);
        std::fill_n(row + visible, n_kv - visible, F16_NEG_INF);
    }
}

/// Local-block force for queries at p0..p0+nt-1 over nblk blocks: +1e30 on
/// the query's own block and the idx_local - 1 blocks before it (llama.cpp's
/// llm_graph_input_msa bias), 0 elsewhere.
static void fill_bias(const hparams & hp, float * out, int64_t nblk, int64_t p0, int64_t nt)
{
    for (int64_t t = 0; t < nt; t++)
    {
        float * row = out + (size_t) (t * nblk);
        std::fill(row, row + nblk, 0.0f);
        const int64_t own = (p0 + t) / hp.idx_blk;
        for (int64_t l = 0; l < hp.idx_local && own - l >= 0; l++)
            if (own - l < nblk) row[own - l] = 1e30f;
    }
}

/// Host inputs a segment's attention reads: the causal mask (dense attention
/// and the ggml sparse path) and the local-block force (the ggml sparse path).
/// The MSA kernels take positions only.
static bool seg_needs_mask(const model & m, const bd_token & b) { return !(b.sparse && m.sparse_maskless); }
static bool seg_needs_bias(const model & m, const bd_token & b) { return b.sparse && !m.sparse_maskless; }

/// Flat mask and bias element counts of a graph's segments.
static void seg_sizes(const model & m, const std::vector<bd_token> & bd, int64_t * total_mask, int64_t * total_bias)
{
    *total_mask = 0;
    *total_bias = 0;
    for (const bd_token & b : bd)
    {
        if (seg_needs_mask(m, b)) *total_mask += b.n_kv * b.len;
        if (seg_needs_bias(m, b)) *total_bias += (b.n_kv / m.hp.idx_blk) * b.len;
    }
}

/// The host inputs of a graph's segments, in segment order: positions, cache
/// rows, causal masks, and the local-block bias of the sparse segments. `p0s`
/// gives each segment's first position (the cached graph's own bd only gives
/// shapes).
static void fill_segment_inputs(model & m, const std::vector<bd_token> & bd, const int64_t * p0s)
{
    int64_t T = 0, total_mask = 0, total_bias = 0;
    for (const bd_token & b : bd) T += b.len;
    seg_sizes(m, bd, &total_mask, &total_bias);
    m.h_pos.resize((size_t) T);
    m.h_kv_idx.resize((size_t) T);
    m.h_mask.resize((size_t) total_mask);
    m.h_bias.resize((size_t) std::max<int64_t>(total_bias, 1));
    int64_t row = 0, off_m = 0, off_b = 0;
    for (size_t i = 0; i < bd.size(); i++)
    {
        const bd_token & b = bd[i];
        const int64_t p0 = p0s[i];
        for (int64_t t = 0; t < b.len; t++)
        {
            m.h_pos[(size_t) (row + t)] = (int32_t) (p0 + t);
            m.h_kv_idx[(size_t) (row + t)] = p0 + t;
        }
        if (seg_needs_mask(m, b))
        {
            fill_mask(m.h_mask.data() + off_m, b.n_kv, p0, b.len);
            off_m += b.n_kv * b.len;
        }
        if (seg_needs_bias(m, b))
        {
            const int64_t nblk = b.n_kv / m.hp.idx_blk;
            fill_bias(m.hp, m.h_bias.data() + off_b, nblk, p0, b.len);
            off_b += nblk * b.len;
        }
        row += b.len;
    }
}

// ---------------------------------------------------------------------------
// graph construction
// ---------------------------------------------------------------------------
struct builder
{
    model & m;
    const hparams & hp;
    graph_entry & g;
    ggml_context * ctx;
    ggml_cgraph * gf;
    std::vector<ggml_tensor *> ds_taps;   // the DSpark drafter's tapped layer outputs, in feature order

    builder(model & mm, graph_entry & ge) : m(mm), hp(mm.hp), g(ge), ctx(ge.ctx), gf(ge.gf) {}

    void pin(ggml_tensor * t, int dev)
    {
        if (dev < 0 || dev > m.n_gpu) return;
        if (t->op != GGML_OP_NONE && !ggml_backend_supports_op(m.sched_backends[dev], t)) return;
        ggml_backend_sched_set_tensor_backend(g.sched, t, m.sched_backends[dev]);
    }

    ggml_tensor * new_input(ggml_type type, int64_t ne0, int64_t ne1, const char * name, int dev)
    {
        ggml_tensor * t = ne1 > 0 ? ggml_new_tensor_2d(ctx, type, ne0, ne1) : ggml_new_tensor_1d(ctx, type, ne0);
        ggml_set_input(t);
        ggml_set_name(t, name);
        pin(t, dev);
        return t;
    }

    /// rms_norm(x) * w; the GGUF norm weights already hold Gemma's 1 + w.
    ggml_tensor * rms(ggml_tensor * x, ggml_tensor * w)
    {
        x = ggml_rms_norm(ctx, x, hp.rms_eps);
        if (w) x = ggml_mul(ctx, x, w);
        return x;
    }

    /// NEOX RoPE over the first n_rot dims of each head; the rest pass through.
    ggml_tensor * rope(ggml_tensor * x, ggml_tensor * pos)
    {
        return ggml_rope_ext(ctx, x, pos, nullptr, hp.n_rot, GGML_ROPE_TYPE_NEOX, hp.n_ctx_train,
                             hp.rope_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    }

    /// Q [hd, n_head, n], K and V [hd, n_head_kv, n] for `cur` [n_embd, n].
    void project_qkv(const layer & L, ggml_tensor * cur, int64_t n, ggml_tensor ** q, ggml_tensor ** k, ggml_tensor ** v)
    {
        const int64_t nq = hp.n_q(), nkv = hp.n_kv();
        if (L.wqkv)
        {
            ggml_tensor * all = ggml_mul_mat(ctx, L.wqkv, cur);       // [nq + 2 nkv, n]
            const size_t es = ggml_element_size(all);
            *q = ggml_view_3d(ctx, all, hp.head_dim, hp.n_head, n, hp.head_dim * es, all->nb[1], 0);
            *k = ggml_view_3d(ctx, all, hp.head_dim, hp.n_head_kv, n, hp.head_dim * es, all->nb[1], (size_t) nq * es);
            *v = ggml_view_3d(ctx, all, hp.head_dim, hp.n_head_kv, n, hp.head_dim * es, all->nb[1], (size_t) (nq + nkv) * es);
        }
        else
        {
            *q = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, L.wq, cur), hp.head_dim, hp.n_head, n);
            *k = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, L.wk, cur), hp.head_dim, hp.n_head_kv, n);
            *v = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, L.wv, cur), hp.head_dim, hp.n_head_kv, n);
        }
    }

    /// Index queries [idx_dim, idx_n_head, n] and keys [idx_dim, 1, n], normed
    /// per head and roped like Q and K.
    void project_index(const layer & L, ggml_tensor * cur, int64_t n, ggml_tensor * pos, ggml_tensor ** iq, ggml_tensor ** ik)
    {
        const int64_t d = hp.idx_dim, nh = hp.idx_n_head;
        // A decode-shaped graph reads the F16 copies (half the bytes of a
        // matrix-vector product); a prompt's matrix product runs on the F32
        // weights, which measured faster there.
        const bool dec = n <= IDX_F16_MAX_ROWS;
        ggml_tensor * wqk = dec && L.idx_wqk_dec ? L.idx_wqk_dec : L.idx_wqk;
        ggml_tensor * wq = dec && L.idx_wq_dec ? L.idx_wq_dec : L.idx_wq;
        ggml_tensor * wk = dec && L.idx_wk_dec ? L.idx_wk_dec : L.idx_wk;
        ggml_tensor *q, *k;
        if (wqk)
        {
            ggml_tensor * all = ggml_mul_mat(ctx, wqk, cur);    // [d * nh + d, n]
            ggml_prec_set_acc(all, GGML_PREC_F32);
            const size_t es = ggml_element_size(all);
            q = ggml_view_3d(ctx, all, d, nh, n, d * es, all->nb[1], 0);
            k = ggml_view_3d(ctx, all, d, 1, n, d * es, all->nb[1], (size_t) (d * nh) * es);
        }
        else
        {
            ggml_tensor * qm = ggml_mul_mat(ctx, wq, cur);
            ggml_tensor * km = ggml_mul_mat(ctx, wk, cur);
            ggml_prec_set_acc(qm, GGML_PREC_F32);
            ggml_prec_set_acc(km, GGML_PREC_F32);
            q = ggml_reshape_3d(ctx, qm, d, nh, n);
            k = ggml_reshape_3d(ctx, km, d, 1, n);
        }
        *iq = rope(rms(q, L.idx_q_norm), pos);
        *ik = rope(rms(k, L.idx_k_norm), pos);
    }

    /// Dense causal attention over cache rows [0, n_kv) of slot `s`.
    ggml_tensor * attend_dense(slot & s, int il, int dev, ggml_tensor * q /*[hd, n_head, n]*/, int64_t n_kv, ggml_tensor * mask)
    {
        ggml_tensor * kc = s.k[(size_t) il];
        ggml_tensor * vc = s.v[(size_t) il];
        const size_t row = kc->nb[1];
        if (q->ne[2] == 1 && dev == m.host_dev())
        {
            // One query on the CPU. ggml-cpu splits a lone [hd, 1, n_head] query's
            // rows across threads from 512 rows on and merges the partial sums,
            // which rounds differently from the row-by-row kernel of a verification
            // or batched graph. With the KV groups on ne[3] (the sparse path's
            // layout) the query takes that row-by-row kernel, so a one-token decode
            // stays bit-identical to the verification row of its position at any
            // context length. The causal row broadcasts over the groups.
            ggml_tensor * kg = ggml_view_4d(ctx, kc, hp.head_dim, n_kv, 1, hp.n_head_kv, row, row * n_kv,
                                            ggml_row_size(kc->type, hp.head_dim), 0);
            ggml_tensor * vg = ggml_view_4d(ctx, vc, hp.head_dim, n_kv, 1, hp.n_head_kv, vc->nb[1], vc->nb[1] * n_kv,
                                            ggml_row_size(vc->type, hp.head_dim), 0);
            return grouped_fa(q, kg, vg, ggml_reshape_4d(ctx, mask, n_kv, 1, 1, 1));
        }
        ggml_tensor * k = ggml_view_3d(ctx, kc, hp.head_dim, hp.n_head_kv, n_kv, ggml_row_size(kc->type, hp.head_dim), row, 0);
        ggml_tensor * v = ggml_view_3d(ctx, vc, hp.head_dim, hp.n_head_kv, n_kv, ggml_row_size(vc->type, hp.head_dim), row, 0);
        ggml_tensor * qp = ggml_permute(ctx, q, 0, 2, 1, 3);           // [hd, n, n_head]
        ggml_tensor * kp = ggml_permute(ctx, k, 0, 2, 1, 3);           // [hd, n_kv, n_head_kv]
        ggml_tensor * vp = ggml_permute(ctx, v, 0, 2, 1, 3);
        ggml_tensor * fa = ggml_flash_attn_ext(ctx, qp, kp, vp, mask, hp.kq_scale(), 0.0f, 0.0f);
        ggml_prec_set_acc(fa, GGML_PREC_F32);
        return fa;                                                     // [hd, n_head, n]
    }

    /// One flash attention for every KV group with the groups on ne[3]
    /// (llama.cpp's build_attn_msa_fa): q [hd, n_head, T], k/v [hd, n_keys, 1, C],
    /// mask F16 [n_keys, R, 1, C]. C = groups x (tokens when R == 1).
    /// Returns [hd, n_head, T].
    ggml_tensor * grouped_fa(ggml_tensor * q, ggml_tensor * k, ggml_tensor * v, ggml_tensor * mask)
    {
        const int64_t D = q->ne[0], HQ = q->ne[1], T = q->ne[2];
        const int64_t C = k->ne[3];
        const int64_t Gp = hp.n_head / hp.n_head_kv;
        const int64_t R = HQ * T / (Gp * C);
        GGML_ASSERT(Gp * C * R == HQ * T);
        // [D, HQ, T] -> [D, Gp, C, R] -> [D, R, Gp, C]
        ggml_tensor * q4 = ggml_permute(ctx, ggml_reshape_4d(ctx, q, D, Gp, C, R), 0, 2, 3, 1);
        ggml_tensor * o = ggml_flash_attn_ext(ctx, q4, k, v, mask, hp.kq_scale(), 0.0f, 0.0f);
        ggml_prec_set_acc(o, GGML_PREC_F32);
        // [D, Gp, R, C] -> [D, Gp, C, R] -> [D, HQ, T]
        o = ggml_permute(ctx, o, 0, 1, 3, 2);
        if (!ggml_is_contiguous(o)) o = ggml_cont(ctx, o);
        return ggml_reshape_3d(ctx, o, D, HQ, T);
    }

    /// The index keys of rows [0, n_kv) for scoring. The scores must stay F32
    /// (block selection is decided by them): ggml-cuda multiplies an F16 view by
    /// the F32 queries in F32 under GGML_PREC_F32, but the CPU backend rounds
    /// the queries to F16 for an F16 matrix, so there the keys are widened to
    /// F32 first (llama.cpp's gathered keys are F32 on every backend).
    ggml_tensor * index_keys(slot & s, int il, int dev, int64_t n_kv)
    {
        ggml_tensor * ikc = s.ik[(size_t) il];
        ggml_tensor * ikp = ggml_view_2d(ctx, ikc, hp.idx_dim, n_kv, ikc->nb[1], 0);
        return dev == m.host_dev() ? ggml_cast(ctx, ikp, GGML_TYPE_F32) : ikp;
    }

    /// Sparse attention of ONE token of slot `s` (llama.cpp's MSA decode with
    /// rows = positions): score the index-key view, max-pool each block, force
    /// the own block, take the top-k blocks per index head and attend over
    /// their gathered K/V rows. q [hd, n_head, 1], iq [idx_dim, n_idx, 1],
    /// mask F16 [n_kv, 1] causal, bias F32 [nblk, 1].
    ggml_tensor * msa_decode(slot & s, int il, int dev, ggml_tensor * q, ggml_tensor * iq, int64_t n_kv,
                             ggml_tensor * mask, ggml_tensor * bias)
    {
        const int64_t D = hp.head_dim, Hkv = hp.n_head_kv, Hd = hp.idx_n_head;
        const int64_t blk = hp.idx_blk, nblk = n_kv / blk;
        const int64_t K = std::min<int64_t>(hp.idx_topk, nblk);
        ggml_tensor * ikp = index_keys(s, il, dev, n_kv);                                   // [idx_dim, n_kv]
        ggml_tensor * pm = ggml_cast(ctx, mask, GGML_TYPE_F32);                             // [n_kv, 1]: 0 / -inf
        ggml_tensor * sc = ggml_mul_mat(ctx, ikp, ggml_reshape_2d(ctx, iq, hp.idx_dim, Hd)); // [n_kv, Hd]
        ggml_prec_set_acc(sc, GGML_PREC_F32);
        // future rows come out -inf, so they never rank into a block's max
        sc = ggml_add_inplace(ctx, sc, pm);
        ggml_tensor * bs = ggml_pool_2d(ctx, sc, GGML_OP_POOL_MAX, (int) blk, 1, (int) blk, 1, 0, 0);   // [nblk, Hd]
        bs = ggml_add(ctx, bs, bias);
        ggml_tensor * idx = ggml_top_k(ctx, bs, (int) K);                                   // I32 [K, Hd]
        if (!ggml_is_contiguous(idx)) idx = ggml_cont(ctx, idx);
        // The selected blocks in ascending order, so the gathered rows reach
        // flash attention in the order the prompt path's masked rows do (top_k
        // returns them in no particular order). The CPU accumulates P*V in F16
        // row by row, so only the same order makes a one-token decode
        // bit-identical to the verification row of the same position.
        ggml_tensor * idf = ggml_cast(ctx, idx, GGML_TYPE_F32);                            // [K, Hd], exact
        ggml_tensor * ord = ggml_argsort(ctx, idf, GGML_SORT_ORDER_ASC);                    // I32 [K, Hd]
        idf = ggml_get_rows(ctx, ggml_reshape_3d(ctx, idf, 1, K, Hd), ord);                // [1, K, Hd] ascending

        // rows tj[t, k, h] = blk * idx[k, h] + t; K/V rows tr = tj * Hkv + h of the
        // [D, Hkv * n_kv] view (index head h selects for KV head h).
        ggml_tensor * a = ggml_scale(ctx, idf, (float) blk);
        ggml_tensor * ar = m.tp.arange_rows;   // tensor parallel: a constant (the meta backend cannot run ggml_arange)
        if (!ar) { ar = ggml_arange(ctx, 0.0f, (float) blk, 1.0f); pin(ar, dev); }
        ggml_tensor * tj = ggml_add(ctx, ggml_repeat_4d(ctx, a, blk, K, Hd, 1), ggml_reshape_3d(ctx, ar, blk, 1, 1));
        ggml_tensor * tokj = ggml_cast(ctx, ggml_reshape_1d(ctx, tj, blk * K * Hd), GGML_TYPE_I32);
        ggml_tensor * ah = m.tp.arange_heads;
        if (!ah) { ah = ggml_arange(ctx, 0.0f, (float) Hkv, 1.0f); pin(ah, dev); }
        ggml_tensor * tr = ggml_add(ctx, ggml_scale(ctx, tj, (float) Hkv), ggml_reshape_3d(ctx, ah, 1, 1, Hd));
        ggml_tensor * tokr = ggml_cast(ctx, ggml_reshape_1d(ctx, tr, blk * K * Hd), GGML_TYPE_I32);

        ggml_tensor * kc = s.k[(size_t) il];
        ggml_tensor * vc = s.v[(size_t) il];
        ggml_tensor * k3 = ggml_view_2d(ctx, kc, D, Hkv * n_kv, ggml_row_size(kc->type, D), 0);
        ggml_tensor * v3 = ggml_view_2d(ctx, vc, D, Hkv * n_kv, ggml_row_size(vc->type, D), 0);
        ggml_tensor * kg = ggml_get_rows(ctx, k3, tokr);                                    // F32 [D, blk * K * Hd]
        ggml_tensor * vg = ggml_get_rows(ctx, v3, tokr);
        ggml_tensor * mg = ggml_get_rows(ctx, ggml_reshape_2d(ctx, pm, 1, n_kv), tokj);     // F32 [1, blk * K * Hd]
        ggml_tensor * kfa = ggml_cast(ctx, ggml_reshape_4d(ctx, kg, D, blk * K, 1, Hd), GGML_TYPE_F16);
        ggml_tensor * vfa = ggml_cast(ctx, ggml_reshape_4d(ctx, vg, D, blk * K, 1, Hd), GGML_TYPE_F16);
        ggml_tensor * mfa = ggml_cast(ctx, ggml_reshape_4d(ctx, mg, blk * K, 1, 1, Hd), GGML_TYPE_F16);
        return grouped_fa(q, kfa, vfa, mfa);
    }

    /// Sparse attention of `len` consecutive tokens of slot `s` (llama.cpp's MSA
    /// batch path with rows = positions): per chunk of tokens, score every row,
    /// select blocks, expand the per-(token, group) block mask to rows, add the
    /// causal mask and run one masked flash attention with the groups on ne[3].
    /// q [hd, n_head, len], iq [idx_dim, n_idx, len], mask F16 [n_kv, len],
    /// bias F32 [nblk, len].
    ggml_tensor * msa_prompt(slot & s, int il, int dev, ggml_tensor * q, ggml_tensor * iq, int64_t len, int64_t n_kv,
                             ggml_tensor * mask, ggml_tensor * bias, int64_t chunk)
    {
        const int64_t D = hp.head_dim, Hkv = hp.n_head_kv, Hd = hp.idx_n_head;
        const int64_t blk = hp.idx_blk, nblk = n_kv / blk;
        const int64_t K = std::min<int64_t>(hp.idx_topk, nblk);
        if (chunk <= 0) chunk = len;
        ggml_tensor * ikp = index_keys(s, il, dev, n_kv);                                   // [idx_dim, n_kv]
        ggml_tensor * kc = s.k[(size_t) il];
        ggml_tensor * vc = s.v[(size_t) il];
        // cache views with the KV groups on ne[3]: [D, n_kv, 1, Hkv]
        ggml_tensor * kfa = ggml_view_4d(ctx, kc, D, n_kv, 1, Hkv, kc->nb[1], kc->nb[1] * n_kv, ggml_row_size(kc->type, D), 0);
        ggml_tensor * vfa = ggml_view_4d(ctx, vc, D, n_kv, 1, Hkv, vc->nb[1], vc->nb[1] * n_kv, ggml_row_size(vc->type, D), 0);

        ggml_tensor * out = nullptr;
        for (int64_t c0 = 0; c0 < len; c0 += chunk)
        {
            const int64_t nt = std::min(chunk, len - c0);
            ggml_tensor * q_c = ggml_view_3d(ctx, q, D, hp.n_head, nt, q->nb[1], q->nb[2], (size_t) c0 * q->nb[2]);
            ggml_tensor * iq_c = ggml_view_2d(ctx, iq, hp.idx_dim, Hd * nt, iq->nb[1], (size_t) c0 * iq->nb[2]);
            // F32 copy of this chunk's causal rows: 0 / -inf by position
            ggml_tensor * pm_c = ggml_cast(ctx, ggml_view_2d(ctx, mask, n_kv, nt, mask->nb[1], (size_t) c0 * mask->nb[1]),
                                           GGML_TYPE_F32);
            ggml_tensor * bias_c = ggml_view_3d(ctx, bias, nblk, 1, nt, bias->nb[1], bias->nb[1], (size_t) c0 * bias->nb[1]);

            // block scores, unscaled: only the top-k ordering matters
            ggml_tensor * sc = ggml_mul_mat(ctx, ikp, iq_c);                                // [n_kv, Hd * nt]
            ggml_prec_set_acc(sc, GGML_PREC_F32);
            sc = ggml_reshape_3d(ctx, sc, n_kv, Hd, nt);
            sc = ggml_add_inplace(ctx, sc, ggml_reshape_3d(ctx, pm_c, n_kv, 1, nt));
            ggml_tensor * bs = ggml_pool_2d(ctx, sc, GGML_OP_POOL_MAX, (int) blk, 1, (int) blk, 1, 0, 0);   // [nblk, Hd, nt]
            bs = ggml_add(ctx, bs, bias_c);                                                 // own block forced
            ggml_tensor * idx = ggml_top_k(ctx, bs, (int) K);                               // I32 [K, Hd, nt]
            if (!ggml_is_contiguous(idx)) idx = ggml_cont(ctx, idx);

            // block mask: -inf everywhere, 0 at the selected blocks. set_rows
            // writes through a view of the canvas, so the chain stays on the
            // layer's device.
            ggml_tensor * canvas = ggml_repeat_4d(ctx, ggml_scale_bias(ctx, bias_c, 0.0f, -INFINITY), nblk, Hd, nt, 1);
            ggml_tensor * zero = ggml_scale(ctx, ggml_cast(ctx, idx, GGML_TYPE_F32), 0.0f);
            ggml_tensor * bm = ggml_set_rows(ctx, ggml_reshape_3d(ctx, canvas, 1, nblk, Hd * nt),
                                             ggml_reshape_3d(ctx, zero, 1, K, Hd * nt),
                                             ggml_reshape_2d(ctx, idx, K, Hd * nt));
            pin(canvas, dev);
            pin(bm, dev);
            bm = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, bm, nblk, Hd, nt), 0, 2, 1, 3));   // [nblk, nt, Hd]
            // block -> rows: row j belongs to block j / blk (rows are positions)
            ggml_tensor * bmr = ggml_repeat_4d(ctx, ggml_reshape_4d(ctx, bm, 1, nblk, nt, Hd), blk, nblk, nt, Hd);
            ggml_tensor * m4 = ggml_reshape_3d(ctx, bmr, n_kv, nt, Hd);
            m4 = ggml_add_inplace(ctx, m4, ggml_reshape_3d(ctx, pm_c, n_kv, nt, 1));
            m4 = ggml_cast(ctx, ggml_reshape_4d(ctx, m4, n_kv, nt, 1, Hd), GGML_TYPE_F16);

            ggml_tensor * o = grouped_fa(q_c, kfa, vfa, m4);                                // [D, n_head, nt]
            out = out ? ggml_concat(ctx, out, o, 2) : o;
        }
        return out;
    }

    /// One attention segment of layer `il`: `len` tokens of slot `s` from row
    /// `row` of the graph's q / iq. Returns [hd, n_head, len].
    ggml_tensor * attend_segment(int il, int dev, const bd_token & B, int64_t row, ggml_tensor * q, ggml_tensor * iq,
                                 ggml_tensor * mask_flat, int64_t off_mask, ggml_tensor * bias_flat, int64_t off_bias)
    {
        const layer & L = m.layers[(size_t) il];
        slot & sl = *m.slots.at(B.slot_id);
        ggml_tensor * qi = ggml_view_3d(ctx, q, hp.head_dim, hp.n_head, B.len, q->nb[1], q->nb[2], (size_t) row * q->nb[2]);
        const bool kernel = L.msa && B.sparse && m.msa_kernel_dev[dev];
        ggml_tensor * mask = kernel ? nullptr
                                    : ggml_view_2d(ctx, mask_flat, B.n_kv, B.len, (size_t) B.n_kv * sizeof(uint16_t),
                                                   (size_t) off_mask * sizeof(uint16_t));
        if (!L.msa || !B.sparse) return attend_dense(sl, il, dev, qi, B.n_kv, mask);

        const int64_t nblk = B.n_kv / hp.idx_blk;
        ggml_tensor * iqi = ggml_view_3d(ctx, iq, hp.idx_dim, hp.idx_n_head, B.len, iq->nb[1], iq->nb[2], (size_t) row * iq->nb[2]);
#if defined(TSG_MINIMAX_MSA_KERNELS)
        // The MSA kernels (TS_MINIMAX_M3_MSA=kernel, or auto on a CUDA GPU):
        // scoring, selection and block-sparse attention in ONE TensorSharp node
        // (contract C3), run by the fused-op wrapper of the layer's GPU
        // (m.wrappers) or its CPU reference.
        if (kernel)
        {
            tsg_msa_params p = {};
            p.n_tokens = (int) B.len;
            p.n_head = hp.n_head;
            p.n_head_kv = hp.n_head_kv;
            p.head_dim = hp.head_dim;
            p.block = hp.idx_blk;
            p.topk = hp.idx_topk;
            p.local_blocks = hp.idx_local;
            p.rows_stride = (int) sl.k[(size_t) il]->ne[1];
            p.kq_scale = hp.kq_scale();
            // Block choice from ~F32 scores (split-F16 index query), as the
            // ggml path and llama.cpp score: an F16 index query could reorder
            // blocks whose scores the reference keeps apart.
            p.index_precision = 1;
            ggml_tensor * pos = ggml_view_1d(ctx, g.inp.pos[dev], B.len, (size_t) row * sizeof(int32_t));
            return tsg_msa_build(ctx, p, qi, iqi, sl.k[(size_t) il], sl.v[(size_t) il], sl.ik[(size_t) il], pos);
        }
#endif
        ggml_tensor * bias = ggml_view_2d(ctx, bias_flat, nblk, B.len, (size_t) nblk * sizeof(float),
                                          (size_t) off_bias * sizeof(float));
        if (B.len == 1) return msa_decode(sl, il, dev, qi, iqi, B.n_kv, mask, bias);
        return msa_prompt(sl, il, dev, qi, iqi, B.len, B.n_kv, mask, bias, B.chunk);
    }

    /// Activations quantized for host_moe_matmul, shared by the stacks that read them.
    struct quantized { ggml_type type = GGML_TYPE_COUNT; ggml_tensor * t = nullptr; };

    /// One expert stack applied to `a`: ggml_mul_mat_id, or host_moe_matmul for a
    /// small batch on a host stack in ggml's plain layout (not CPU_REPACK).
    ggml_tensor * expert_matmul(ggml_tensor * w, bool on_host, ggml_tensor * a, ggml_tensor * ids, int64_t n, quantized & aq)
    {
        if (!on_host || n < 2 || n > m.host_moe_max_tokens || ids->ne[0] * n > HOST_MOE_MAX_PICKS
            || !w->buffer || !ggml_backend_buffer_is_host(w->buffer) || w->buffer == m.repack_buf)
        {
            ggml_tensor * r = ggml_mul_mat_id(ctx, w, a, ids);
            // llama.cpp's build_lora_mm_id; only the cuBLAS fallback reads it.
            if (w->type == GGML_TYPE_NVFP4) ggml_prec_set_acc(r, GGML_PREC_BF16);
            return r;
        }
        const ggml_type vdt = ggml_get_type_traits_cpu(w->type)->vec_dot_type;
        if (aq.type != vdt)
        {
            ggml_tensor * args[1] = { a };
            aq.t = ggml_custom_4d(ctx, GGML_TYPE_I8, (int64_t) ggml_row_size(vdt, a->ne[0]), a->ne[1], a->ne[2], 1,
                                  args, 1, host_moe_quantize, GGML_N_TASKS_MAX, host_moe_quant_userdata(vdt));
            aq.type = vdt;
        }
        ggml_tensor * args[3] = { w, aq.t, ids };
        return ggml_custom_4d(ctx, GGML_TYPE_F32, w->ne[1], ids->ne[0], ids->ne[1], 1, args, 3,
                              host_moe_matmul, GGML_N_TASKS_MAX, nullptr);
    }

    /// Per-expert output scale (llama.cpp's build_lora_mm_id): res [rows,
    /// n_used, n] * s[selected]. This node order is what ggml-cuda fuses into
    /// the MMVQ kernel at decode (MUL_MAT_ID, RESHAPE, REPEAT, GET_ROWS, MUL).
    ggml_tensor * expert_scale(ggml_tensor * res, ggml_tensor * w_s, ggml_tensor * ids, int64_t n)
    {
        if (!w_s) return res;
        ggml_tensor * s = ggml_reshape_3d(ctx, w_s, 1, hp.n_expert, 1);
        s = ggml_repeat_4d(ctx, s, 1, hp.n_expert, n, 1);
        s = ggml_get_rows(ctx, s, ids);                                       // [1, n_used, n]
        return ggml_mul(ctx, res, s);
    }

    /// llama.cpp's build_ffn(PAR, SWIGLU_OAI_MOE): down(swiglu_oai(gate(x), up(x))).
    ggml_tensor * ffn_swiglu_oai(ggml_tensor * cur, ggml_tensor * w_gate, ggml_tensor * w_up, ggml_tensor * w_down)
    {
        ggml_tensor * up = ggml_mul_mat(ctx, w_up, cur);
        ggml_tensor * gate = ggml_mul_mat(ctx, w_gate, cur);
        ggml_tensor * h = ggml_swiglu_oai(ctx, gate, up, SWIGLU_ALPHA, SWIGLU_LIMIT);
        return ggml_mul_mat(ctx, w_down, h);
    }

    /// Routed experts + shared expert for the ffn-normed `cur` [n_embd, n], in
    /// llama.cpp's build_moe_ffn node order (the CUDA topk-moe and MMVQ
    /// scale+GLU fusions match on it).
    ggml_tensor * build_moe(const layer & L, ggml_tensor * cur, int64_t n)
    {
        const int64_t n_expert = hp.n_expert;
        const int64_t n_used = hp.n_expert_used;

        // Router in F32 end to end: exp_probs_b is ~16..37 while the sigmoid is
        // in (0, 1), so F16 or BF16 would scramble the selection.
        ggml_tensor * logits = ggml_mul_mat(ctx, L.gate_inp, cur);                 // [n_expert, n]
        ggml_tensor * probs = ggml_sigmoid(ctx, logits);
        ggml_tensor * selection = L.exp_probs_b ? ggml_add(ctx, probs, L.exp_probs_b) : probs;
        ggml_tensor * selected = ggml_argsort_top_k(ctx, selection, (int) n_used);  // I32 [n_used, n]

        ggml_tensor * weights = ggml_get_rows(ctx, ggml_reshape_3d(ctx, probs, 1, n_expert, n), selected); // [1, n_used, n]
        if (hp.w_norm)
        {
            weights = ggml_reshape_2d(ctx, weights, n_used, n);
            ggml_tensor * sum = ggml_sum_rows(ctx, weights);
            sum = ggml_clamp(ctx, sum, 6.103515625e-5f, INFINITY);
            weights = ggml_div(ctx, weights, sum);
            weights = ggml_reshape_3d(ctx, weights, 1, n_used, n);
        }
        if (hp.w_scale != 0.0f && hp.w_scale != 1.0f)
            weights = ggml_scale(ctx, weights, hp.w_scale);
        ggml_build_forward_expand(gf, weights);

        ggml_tensor * x = ggml_reshape_3d(ctx, cur, hp.n_embd, 1, n);
        quantized xq;
        ggml_tensor * up = expert_matmul(L.up_exps, L.cpu_stack[1], x, selected, n, xq);    // [ff, n_used, n]
        up = expert_scale(up, L.up_s, selected, n);
        ggml_tensor * gate = expert_matmul(L.gate_exps, L.cpu_stack[0], x, selected, n, xq);
        gate = expert_scale(gate, L.gate_s, selected, n);
        // The scales act on the true gate/up values, before swiglu_oai's clamps.
        ggml_tensor * h = ggml_swiglu_oai(ctx, gate, up, SWIGLU_ALPHA, SWIGLU_LIMIT);
        quantized hq;
        ggml_tensor * experts = expert_matmul(L.down_exps, L.cpu_stack[2], h, selected, n, hq);  // [n_embd, n_used, n]
        experts = expert_scale(experts, L.down_s, selected, n);
        experts = ggml_mul(ctx, experts, weights);
        ggml_build_forward_expand(gf, experts);

        ggml_tensor * out = nullptr;
        for (int64_t e = 0; e < n_used; e++)
        {
            ggml_tensor * view = ggml_view_2d(ctx, experts, hp.n_embd, n, experts->nb[2], (size_t) e * experts->nb[1]);
            ggml_build_forward_expand(gf, view);
            out = out ? ggml_add(ctx, out, view) : view;
        }
        if (n_used == 1) out = ggml_cont(ctx, out);

        ggml_tensor * shexp = ffn_swiglu_oai(cur, L.gate_shexp, L.up_shexp, L.down_shexp);
        return ggml_add(ctx, out, shexp);
    }

    std::vector<uint8_t> used_devices() const
    {
        std::vector<uint8_t> used((size_t) m.n_gpu + 1, 0);
        for (int il = 0; il < hp.n_layer; il++) used[(size_t) m.layers[(size_t) il].device] = 1;
        return used;
    }

    ggml_tensor * embed(int64_t n)
    {
        g.inp.tokens = new_input(GGML_TYPE_I32, n, 0, "inp_tokens", m.embd_device);
        ggml_tensor * emb = ggml_get_rows(ctx, m.tok_embd, g.inp.tokens);           // [n_embd, n] F32
        if (g.n_ovr > 0)
        {
            g.inp.embd_rows = new_input(GGML_TYPE_F32, hp.n_embd, g.n_ovr, "inp_embd_rows", m.embd_device);
            g.inp.embd_idx = new_input(GGML_TYPE_I64, g.n_ovr, 0, "inp_embd_idx", m.embd_device);
            // set_rows writes THROUGH a view of the gathered rows, so it must run
            // where those rows live: left to the scheduler, a GPU could be handed
            // a write into host memory (an illegal access on CUDA).
            pin(emb, m.embd_device);
            emb = ggml_set_rows(ctx, emb, g.inp.embd_rows, g.inp.embd_idx);
            pin(emb, m.embd_device);
        }
        return emb;
    }

    ggml_tensor * head(ggml_tensor * x /*[n_embd, rows]*/)
    {
        x = rms(x, m.output_norm);
        return ggml_mul_mat(ctx, m.output, x);
    }

    // ---- segments of one or several slots ----------------------------------
    // Every projection, the router and the experts run once over all T rows;
    // only the cache writes and the attention are built per segment, because
    // each segment reads its own slot's history at its own positions. Each
    // segment's mask and bias are a contiguous block of one flat input, so
    // the views stay contiguous. A kind-0 graph is one segment.
    void build()
    {
        const int64_t T = g.nt;
        const int64_t S = (int64_t) g.bd.size();
        auto used = used_devices();
        int64_t total_mask = 0, total_bias = 0;
        seg_sizes(m, g.bd, &total_mask, &total_bias);
        for (int d = 0; d <= m.n_gpu; d++)
        {
            if (!used[(size_t) d]) continue;
            char nb[64];
            snprintf(nb, sizeof(nb), "inp_pos.%d", d);
            g.inp.pos[d] = new_input(GGML_TYPE_I32, T, 0, nb, d);
            snprintf(nb, sizeof(nb), "inp_kv_idx.%d", d);
            g.inp.kv_idx[d] = new_input(GGML_TYPE_I64, T, 0, nb, d);
            snprintf(nb, sizeof(nb), "inp_mask.%d", d);
            if (total_mask > 0) g.inp.mask[d] = new_input(GGML_TYPE_F16, total_mask, 0, nb, d);
            bool sparse_here = false;
            for (int il = 0; il < hp.n_layer; il++)
                if (m.layers[(size_t) il].device == d && m.layers[(size_t) il].msa) sparse_here = total_bias > 0;
            if (sparse_here)
            {
                snprintf(nb, sizeof(nb), "inp_msa_bias.%d", d);
                g.inp.bias[d] = new_input(GGML_TYPE_F32, total_bias, 0, nb, d);
            }
        }
        const int last = hp.n_layer - 1;
        // Rows the head reads: kind 0 keeps n_out (the last token, or all of
        // them); segments keep each segment's last token.
        const int64_t n_rows_out = g.kind == 0 ? g.n_out : S;
        const bool prune = g.want_logits && n_rows_out < T;
        if (prune) g.inp.out_ids = new_input(GGML_TYPE_I32, n_rows_out, 0, "inp_out_ids", m.layers[(size_t) last].device);

        ggml_tensor * x = embed(T);
        for (int il = 0; il < hp.n_layer; il++)
        {
            const layer & L = m.layers[(size_t) il];
            const int d = L.device;
            ggml_tensor * cur = rms(x, L.attn_norm);
            ggml_tensor *q, *k, *v;
            project_qkv(L, cur, T, &q, &k, &v);
            q = rope(rms(q, L.q_norm), g.inp.pos[d]);
            k = rope(rms(k, L.k_norm), g.inp.pos[d]);
            ggml_tensor *iq = nullptr, *ik = nullptr;
            if (L.msa) project_index(L, cur, T, g.inp.pos[d], &iq, &ik);

            ggml_tensor * k2 = ggml_reshape_2d(ctx, ggml_is_contiguous(k) ? k : ggml_cont(ctx, k), hp.n_kv(), T);
            ggml_tensor * v2 = ggml_reshape_2d(ctx, ggml_is_contiguous(v) ? v : ggml_cont(ctx, v), hp.n_kv(), T);
            ggml_tensor * ik2 = L.msa ? ggml_reshape_2d(ctx, ggml_is_contiguous(ik) ? ik : ggml_cont(ctx, ik), hp.idx_dim, T) : nullptr;
            if (!ggml_is_contiguous(q)) q = ggml_cont(ctx, q);
            if (iq && !ggml_is_contiguous(iq)) iq = ggml_cont(ctx, iq);

            ggml_tensor * cat = nullptr;
            int64_t row = 0, off_m = 0, off_b = 0;
            for (int64_t i = 0; i < S; i++)
            {
                const bd_token & B = g.bd[(size_t) i];
                slot & sl = *m.slots.at(B.slot_id);
                ggml_tensor * idx = ggml_view_1d(ctx, g.inp.kv_idx[d], B.len, (size_t) row * sizeof(int64_t));
                // cache writes precede every read of this layer's caches
                if (L.msa)
                {
                    ggml_tensor * iki = ggml_view_2d(ctx, ik2, hp.idx_dim, B.len, ik2->nb[1], (size_t) row * ik2->nb[1]);
                    ggml_build_forward_expand(gf, ggml_set_rows(ctx, sl.ik[(size_t) il], iki, idx));
                }
                ggml_tensor * ki = ggml_view_2d(ctx, k2, hp.n_kv(), B.len, k2->nb[1], (size_t) row * k2->nb[1]);
                ggml_tensor * vi = ggml_view_2d(ctx, v2, hp.n_kv(), B.len, v2->nb[1], (size_t) row * v2->nb[1]);
                ggml_build_forward_expand(gf, ggml_set_rows(ctx, sl.k[(size_t) il], ki, idx));
                ggml_build_forward_expand(gf, ggml_set_rows(ctx, sl.v[(size_t) il], vi, idx));

                ggml_tensor * out = attend_segment(il, d, B, row, q, iq, g.inp.mask[d], off_m, g.inp.bias[d], off_b);
                cat = cat ? ggml_concat(ctx, cat, out, 2) : out;               // [hd, n_head, rows so far]
                row += B.len;
                if (seg_needs_mask(m, B)) off_m += B.n_kv * B.len;
                if (seg_needs_bias(m, B)) off_b += (B.n_kv / hp.idx_blk) * B.len;
            }
            ggml_tensor * attn = ggml_mul_mat(ctx, L.wo, ggml_reshape_2d(ctx, cat, hp.n_q(), T));
            ggml_tensor * res = x;
            int64_t rows = T;
            // A DSpark drafter that taps the final stream needs the last layer on every row.
            if (il == last && !dspark_taps_final(m))
            {
                if (!g.want_logits)
                {
                    // A non-final prefill chunk only leaves its caches behind; the
                    // last layer's FFN and the head would be computed for nobody.
                    ggml_build_forward_expand(gf, attn);
                    return;
                }
                if (prune)
                {
                    attn = ggml_get_rows(ctx, attn, g.inp.out_ids);
                    res = ggml_get_rows(ctx, res, g.inp.out_ids);
                    rows = n_rows_out;
                }
            }
            ggml_tensor * ffn_inp = ggml_add(ctx, attn, res);
            ggml_tensor * y = rms(ffn_inp, L.ffn_norm);
            ggml_tensor * ffn = L.dense ? ffn_swiglu_oai(y, L.ffn_gate, L.ffn_up, L.ffn_down) : build_moe(L, y, rows);
            x = ggml_add(ctx, ffn, ffn_inp);
            dspark_tap(*this, il, x);
        }
        if (dspark_taps_final(m))
        {
            if (!g.want_logits) return;
            if (prune) x = ggml_get_rows(ctx, x, g.inp.out_ids);
        }
        ggml_tensor * logits = head(x);
        ggml_set_output(logits);
        ggml_set_name(logits, "logits");
        g.logits = logits;
        ggml_build_forward_expand(gf, logits);
    }
};

static bool key_matches(const graph_entry & e, const graph_entry & k)
{
    if (e.kind != k.kind || e.nt != k.nt || e.want_logits != k.want_logits || e.n_out != k.n_out || e.n_ovr != k.n_ovr) return false;
    if (e.bd.size() != k.bd.size()) return false;
    for (size_t i = 0; i < e.bd.size(); i++)
    {
        const bd_token & a = e.bd[i];
        const bd_token & b = k.bd[i];
        if (a.slot_id != b.slot_id || a.len != b.len || a.n_kv != b.n_kv || a.sparse != b.sparse || a.chunk != b.chunk) return false;
    }
    return true;
}

/// Graph size: a dense layer is ~100 nodes; a sparse decode segment adds ~40
/// per layer and a sparse prompt segment ~30 per chunk.
static size_t graph_nodes(const model & m, const graph_entry & k)
{
    size_t per_layer = 128;
    for (const bd_token & b : k.bd)
    {
        if (!b.sparse || m.sparse_maskless) per_layer += 24;   // dense, or one MSA kernel node
        else if (b.len == 1) per_layer += 64;
        else per_layer += 24 + 40 * (size_t) ((b.len + std::max<int64_t>(1, b.chunk) - 1) / std::max<int64_t>(1, b.chunk));
    }
    if (const long v = env_long("TS_MINIMAX_M3_NODES_PER_LAYER", 0); v > 0) per_layer = (size_t) v;
    return (size_t) m.hp.n_layer * per_layer + 1024 + dspark_graph_nodes(m, k.bd.size());
}

/// Room kept free on the tightest GPU next to a measured --tp graph: the
/// measurement runs without the builder's backend pins (below), and buffers are
/// rounded per allocation.
static constexpr size_t TP_GRAPH_MARGIN = (size_t) 128 << 20;

/// Under --tp, whether the new graph in `e` fits the tightest GPU, measured
/// before anything is allocated: ggml's meta backend aborts the process when
/// one GPU cannot allocate its share of a buffer, so the scheduler's own
/// failure path never runs there. The meta compute buffer holds whole tensors
/// on every GPU (ggml_backend_sched_reserve_size reports it without
/// allocating), and a graph with more rows than any before also grows the
/// reduction scratch. Retained slots (never `reads`) are freed while it does
/// not fit. `e` is a throwaway: reserve_size resets its scheduler first (so it
/// measures without the builder's pins, which TP_GRAPH_MARGIN covers) and
/// leaves it without buffers.
static bool tp_graph_fits(model & m, graph_entry & e, const std::vector<int32_t> & reads)
{
    std::vector<size_t> sizes((size_t) m.n_sched_backends, 0);
    ggml_backend_sched_reserve_size(e.sched, e.gf, sizes.data());
    const size_t compute = sizes[0];
    const size_t scratch = e.nt > m.tp.max_rows
        ? tsg_mm3_tp::reduction_scratch_bytes(m.tp.n, m.hp.n_embd, m.hp.n_expert_used, e.nt) -
          tsg_mm3_tp::reduction_scratch_bytes(m.tp.n, m.hp.n_embd, m.hp.n_expert_used, m.tp.max_rows)
        : 0;
    for (;;)
    {
        size_t free_b = 0, total_b = 0;
        tsg_mm3_tp::memory(m.tp.dev, &free_b, &total_b);
        if (debug_on())
            fprintf(stderr, "[minimax-m3] --tp: graph nt=%" PRId64 " needs %.0f MiB compute + %.0f MiB reduction scratch per GPU, "
                    "%.0f MiB free on the tightest\n", e.nt, compute / 1048576.0, scratch / 1048576.0, free_b / 1048576.0);
        if (free_b >= compute + scratch + TP_GRAPH_MARGIN)
        {
            // Slot admission keeps room for the largest graph by this same measure
            // (above the allocated size: the measurement runs without the pins), so
            // a graph it kept room for passes this check again.
            m.max_graph_bytes.resize((size_t) m.n_gpu + 1, 0);
            m.max_graph_bytes[0] = std::max(m.max_graph_bytes[0], compute);
            return true;
        }
        if (!reclaim_for_graph(m, reads.data(), (int) reads.size()))
        {
            fprintf(stderr, "[minimax-m3] --tp: a %" PRId64 "-token graph needs %.0f MiB of compute buffers (+%.0f MiB of "
                    "reduction scratch) on every GPU and the tightest has %.0f MiB free; it is not allocated\n", e.nt,
                    compute / 1048576.0, scratch / 1048576.0, free_b / 1048576.0);
            return false;
        }
    }
}

static graph_entry * acquire_graph(model & m, std::unique_ptr<graph_entry> key, bool * reused)
{
    for (auto it = m.graph_cache.begin(); it != m.graph_cache.end(); ++it)
    {
        if (!key_matches(**it, *key)) continue;
        auto entry = std::move(*it);
        m.graph_cache.erase(it);
        graph_entry * raw = entry.get();
        m.graph_cache.push_front(std::move(entry));
        if (reused) *reused = true;
        m.n_graph_reuses++;
        return raw;
    }

    // Under --tp two whole-graph compute buffers would sit on every GPU (and
    // the meta backend keeps views only for its last graph): the one graph
    // kept goes before the next is built.
    if (m.tp.n > 1) m.graph_cache.clear();

    // Reads: the slots this graph touches (never reclaimed for it).
    std::vector<int32_t> reads;
    for (const bd_token & b : key->bd) reads.push_back(b.slot_id);

    // A prefill graph holds compute buffers sized for its ubatch on every
    // device; keep only one of them cached next to the small decode graphs.
    // The evicted one hands its scheduler to the new graph: a prompt of a new
    // length then reuses those buffers instead of freeing and allocating
    // every device's compute memory again (part of each request's TTFT).
    const int64_t large = 64;
    const size_t n_nodes = graph_nodes(m, *key);
    ggml_backend_sched_t spare = nullptr;
    size_t spare_nodes = 0;
    if (key->nt > large)
    {
        for (auto it = m.graph_cache.begin(); it != m.graph_cache.end(); )
        {
            graph_entry & old = **it;
            if (old.nt <= large) { ++it; continue; }
            if (!spare && old.owns_sched && old.sched && old.sched_nodes >= n_nodes)
            {
                spare = old.sched;
                spare_nodes = old.sched_nodes;
                old.sched = nullptr;
            }
            it = m.graph_cache.erase(it);
        }
    }

    // A fresh context and graph in `e`, built (its pins land on e.sched).
    auto build = [&](graph_entry & e) -> bool
    {
        if (e.ctx) { ggml_free(e.ctx); e.ctx = nullptr; }
        ggml_init_params gp = { ggml_tensor_overhead() * n_nodes + ggml_graph_overhead_custom(n_nodes, false), nullptr, true };
        e.ctx = ggml_init(gp);
        if (!e.ctx) return false;
        e.gf = ggml_new_graph_custom(e.ctx, n_nodes, false);
        builder b(m, e);
        b.build();
        return true;
    };

    auto new_entry = [&]()
    {
        std::unique_ptr<graph_entry> e(new graph_entry());
        e->kind = key->kind; e->slot_id = key->slot_id; e->nt = key->nt; e->p0 = key->p0;
        e->n_out = key->n_out; e->n_ovr = key->n_ovr; e->want_logits = key->want_logits; e->bd = key->bd;
        return e;
    };

    for (;;)
    {
        if (m.tp.n > 1)
        {
            // Checked and measured before anything is allocated, on a throwaway
            // scheduler (a measurement leaves its scheduler without buffers): an op
            // the GPUs cannot run, or compute buffers that do not fit, would abort
            // ggml's meta backend. A graph that does not fit is declined like a
            // failed allocation on the layer split, and the call retries in halves.
            std::unique_ptr<graph_entry> probe = new_entry();
            probe->sched = ggml_backend_sched_new(m.sched_backends, m.sched_bufts, m.n_sched_backends, n_nodes, false, m.op_offload);
            if (!probe->sched || !build(*probe)) return nullptr;
            if (std::string why; !tsg_mm3_tp::check_graph(m.backends[0], probe->gf, m.tok_embd, &why))
            {
                tsg::set_last_error("[minimax-m3] --tp: " + why);
                fprintf(stderr, "[minimax-m3] --tp: %s\n", why.c_str());
                return nullptr;
            }
            if (!tp_graph_fits(m, *probe, reads))
            {
                m.alloc_failed = true;
                return nullptr;
            }
        }

        std::unique_ptr<graph_entry> e = new_entry();
        if (spare)
        {
            ggml_backend_sched_reset(spare);
            e->sched = spare;
            e->sched_nodes = spare_nodes;
            spare = nullptr;
        }
        else
        {
            e->sched = ggml_backend_sched_new(m.sched_backends, m.sched_bufts, m.n_sched_backends, n_nodes, false, m.op_offload);
            e->sched_nodes = n_nodes;
        }
        if (!e->sched) return nullptr;
        if (!build(*e)) return nullptr;

        const auto t0 = std::chrono::steady_clock::now();
        if (ggml_backend_sched_alloc_graph(e->sched, e->gf))
        {
            m.n_graph_builds++;
            m.max_graph_bytes.resize((size_t) m.n_gpu + 1, 0);
            for (int d = 0; d <= m.n_gpu; d++)
                m.max_graph_bytes[(size_t) d] = std::max(m.max_graph_bytes[(size_t) d],
                                                         ggml_backend_sched_get_buffer_size(e->sched, m.sched_backends[d]));
            if (debug_on())
            {
                std::string bufs;
                for (int d = 0; d <= m.n_gpu; d++)
                {
                    char b[48];
                    snprintf(b, sizeof(b), "%s%.0f", d ? "/" : "", ggml_backend_sched_get_buffer_size(e->sched, m.sched_backends[d]) / 1048576.0);
                    bufs += b;
                }
                int n_sparse = 0;
                for (const bd_token & s : e->bd) n_sparse += s.sparse ? 1 : 0;
                fprintf(stderr, "[minimax-m3] built graph kind=%d nt=%" PRId64 " segments=%zu sparse=%d n_kv0=%" PRId64
                        " chunk0=%" PRId64 " nodes=%d splits=%d alloc=%.1f ms compute MiB=%s\n", e->kind, e->nt, e->bd.size(),
                        n_sparse, e->bd.empty() ? 0 : e->bd[0].n_kv, e->bd.empty() ? 0 : e->bd[0].chunk,
                        ggml_graph_n_nodes(e->gf), ggml_backend_sched_get_n_splits(e->sched),
                        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(), bufs.c_str());
            }
            graph_entry * raw = e.get();
            m.graph_cache.push_front(std::move(e));
            while ((int) m.graph_cache.size() > m.graph_cache_cap) m.graph_cache.pop_back();
            if (reused) *reused = false;
            return raw;
        }
        e.reset();
        // Out of memory: drop the cached graphs first (each holds compute
        // buffers, the pipeline's too), then retained slots, before giving up.
        if (!m.graph_cache.empty() || m.pipe.sched)
        {
            m.graph_cache.clear();
            m.pipe.release();
            continue;
        }
        if (!reclaim_for_graph(m, reads.data(), (int) reads.size()))
        {
            fprintf(stderr, "[minimax-m3] failed to allocate a graph (kind=%d nt=%" PRId64 ")\n", key->kind, key->nt);
            m.alloc_failed = true;
            return nullptr;
        }
    }
}

static inline bool live(const ggml_tensor * t) { return t && t->buffer; }

static void set_i32(ggml_tensor * t, const int32_t * v, size_t n)
{
    if (live(t)) ggml_backend_tensor_set(t, v, 0, n * sizeof(int32_t));
}

/// Upload the host inputs fill_segment_inputs left in m.h_* to every device copy.
static void upload_segment_inputs(model & m, graph_entry & g)
{
    const size_t T = m.h_pos.size();
    for (int d = 0; d <= m.n_gpu; d++)
    {
        set_i32(g.inp.pos[d], m.h_pos.data(), T);
        if (live(g.inp.kv_idx[d])) ggml_backend_tensor_set(g.inp.kv_idx[d], m.h_kv_idx.data(), 0, T * sizeof(int64_t));
        if (live(g.inp.mask[d])) ggml_backend_tensor_set(g.inp.mask[d], m.h_mask.data(), 0, m.h_mask.size() * sizeof(uint16_t));
        if (live(g.inp.bias[d])) ggml_backend_tensor_set(g.inp.bias[d], m.h_bias.data(), 0, (size_t) ggml_nelements(g.inp.bias[d]) * sizeof(float));
    }
}

static void count_segments(model & m, const std::vector<bd_token> & bd)
{
    for (const bd_token & b : bd) (b.sparse ? m.n_sparse_segments : m.n_dense_segments)++;
}

// TS_MINIMAX_M3_PROFILE=N: time every node (synchronously, so absolute totals
// include per-node sync overhead) and print a per-backend / per-op summary every
// N graph computations. A diagnostic only; it defeats CUDA graphs.
struct profiler
{
    ggml_backend_sched_t sched = nullptr;
    std::chrono::steady_clock::time_point last;
    std::map<std::string, double> ms;     // "backend op" -> milliseconds
    std::map<std::string, int64_t> count;
    int64_t calls = 0;
    double wall_ms = 0;
};
static profiler g_prof;

static bool prof_eval(ggml_tensor * t, bool ask, void * user)
{
    (void) user;
    if (ask) return true;
    const auto now = std::chrono::steady_clock::now();
    ggml_backend_t be = ggml_backend_sched_get_tensor_backend(g_prof.sched, t);
    std::string key = std::string(be ? ggml_backend_name(be) : "?") + " " + ggml_op_desc(t);
    g_prof.ms[key] += std::chrono::duration<double, std::milli>(now - g_prof.last).count();
    g_prof.count[key]++;
    g_prof.last = std::chrono::steady_clock::now();
    return true;
}

static bool compute(model & m, graph_entry & g)
{
    static const long prof_every = env_long("TS_MINIMAX_M3_PROFILE", 0);
    const auto t0 = std::chrono::steady_clock::now();
    if (prof_every > 0)
    {
        g_prof.sched = g.sched;
        g_prof.last = t0;
        ggml_backend_sched_set_eval_callback(g.sched, prof_eval, nullptr);
    }
    if (ggml_backend_sched_graph_compute_async(g.sched, g.gf) != GGML_STATUS_SUCCESS)
    {
        fprintf(stderr, "[minimax-m3] graph compute failed\n");
        return false;
    }
    ggml_backend_sched_synchronize(g.sched);
    if (m.tp.n > 1) m.tp.max_rows = std::max(m.tp.max_rows, g.nt);   // the reduction scratch now fits this many rows
    if (prof_every > 0)
    {
        ggml_backend_sched_set_eval_callback(g.sched, nullptr, nullptr);
        g_prof.wall_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (++g_prof.calls % prof_every == 0)
        {
            std::vector<std::pair<double, std::string>> rows;
            for (auto & kv : g_prof.ms) rows.push_back({ kv.second, kv.first });
            std::sort(rows.rbegin(), rows.rend());
            fprintf(stderr, "[minimax-m3 profile] %" PRId64 " graphs, %.1f ms/graph wall (nt=%" PRId64 ", kind=%d)\n",
                    g_prof.calls, g_prof.wall_ms / g_prof.calls, g.nt, g.kind);
            for (size_t i = 0; i < rows.size() && i < 24; i++)
                fprintf(stderr, "  %8.2f ms/graph  %6" PRId64 " nodes  %s\n", rows[i].first / g_prof.calls,
                        g_prof.count[rows[i].second] / g_prof.calls, rows[i].second.c_str());
            g_prof = profiler();
        }
    }
    return true;
}

/// The single segment of a kind-0 graph over the active slot at p0..p0+nt-1.
static bd_token single_segment(const model & m, const slot & s, int64_t p0, int64_t nt)
{
    bd_token b;
    b.slot_id = s.id;
    b.p = p0;
    b.len = nt;
    b.n_kv = plan_n_kv(m, p0 + nt, m.kv_pad);
    b.sparse = segment_sparse(m, p0 + nt);
    b.chunk = b.sparse ? msa_chunk_tokens(m, b.n_kv, nt) : 0;
    return b;
}

/// One ubatch of the active slot. `ovr` rows (vision) replace token
/// embeddings at rows ovr_idx of this ubatch.
static bool forward_ubatch(model & m, const int32_t * tokens, int64_t nt, bool want_logits, bool all_rows,
                           float * logits_out, const float * ovr_rows, const int64_t * ovr_idx, int64_t n_ovr)
{
    slot & s = *m.active_slot;
    const int64_t p0 = s.n_past;
    if (p0 + nt > m.n_ctx)
    {
        fprintf(stderr, "[minimax-m3] context overflow: %" PRId64 " + %" PRId64 " > %d\n", p0, nt, m.n_ctx);
        return false;
    }

    const bd_token seg = single_segment(m, s, p0, nt);
    if (!slot_reserve_rows(m, s, seg.n_kv))
    {
        fprintf(stderr, "[minimax-m3] no device memory for %" PRId64 " K/V rows of slot %d\n", p0 + nt, s.id);
        return false;
    }
    std::unique_ptr<graph_entry> key(new graph_entry());
    key->kind = 0;
    key->slot_id = s.id;
    key->nt = nt;
    key->p0 = p0;
    key->want_logits = want_logits;
    key->n_out = want_logits ? (all_rows ? nt : 1) : 0;
    key->n_ovr = n_ovr;
    key->bd.push_back(seg);

    bool reused = false;
    graph_entry * g = acquire_graph(m, std::move(key), &reused);
    if (!g) return false;

    m.h_tokens.assign(tokens, tokens + nt);
    fill_segment_inputs(m, g->bd, &p0);
    set_i32(g->inp.tokens, m.h_tokens.data(), (size_t) nt);
    upload_segment_inputs(m, *g);
    if (want_logits && live(g->inp.out_ids))
    {
        m.h_out_ids.resize((size_t) g->n_out);
        for (int64_t i = 0; i < g->n_out; i++) m.h_out_ids[(size_t) i] = (int32_t) (g->n_out == 1 ? nt - 1 : i);
        set_i32(g->inp.out_ids, m.h_out_ids.data(), (size_t) g->n_out);
    }
    if (n_ovr > 0)
    {
        if (live(g->inp.embd_rows)) ggml_backend_tensor_set(g->inp.embd_rows, ovr_rows, 0, (size_t) n_ovr * m.hp.n_embd * sizeof(float));
        if (live(g->inp.embd_idx)) ggml_backend_tensor_set(g->inp.embd_idx, ovr_idx, 0, (size_t) n_ovr * sizeof(int64_t));
    }

    dspark_rows_written(m, s, p0, nt);
    if (!compute(m, *g)) return false;
    s.n_past += nt;
    count_segments(m, g->bd);
    if (want_logits && logits_out && g->logits)
        ggml_backend_tensor_get(g->logits, logits_out, 0, (size_t) g->n_out * m.hp.n_vocab * sizeof(float));
    return true;
}


// ---------------------------------------------------------------------------
// pipelined prefill
// ---------------------------------------------------------------------------
// forward_ubatch runs one micro-batch through every GPU and waits, so with
// layers split over S GPUs each GPU idles (S-1)/S of a prefill. Here the
// micro-batches of one call go to ONE persistent scheduler created with
// parallel=true: ggml then keeps n_copies rotating copies of every graph input
// and of every tensor handed from one GPU to the next, and each micro-batch is
// only enqueued (graph_compute_async); one synchronize ends the call. Each GPU
// runs its splits in order on its own stream, so micro-batch j+1 reads the
// cache rows j wrote without a host wait. llama.cpp's layer split does the same.
//
// What keeps the in-flight micro-batches apart:
//  * the scheduler is reserved for the call's largest micro-batch (its last
//    window) before anything is in flight. Every micro-batch builds the same
//    graph topology (one logits row each; the same attention path and the
//    same number of sparse chunks, both decided by the call's last window),
//    only with a window that grows, so ggml's allocator keeps every node and
//    copy slot at its reserved address and never reallocates mid-call (a
//    reallocation would drain the pipeline);
//  * inputs pinned to a GPU are uploaded with tensor_set_async on that GPU's
//    stream, from a pinned staging ring the host refills only after the
//    uploads that read a slot have run;
//  * ggml copies a GPU's output into the next GPU's copy slot on the PRODUCER's
//    stream, while its own guard waits on the consumer's. Before micro-batch j
//    every producer therefore waits for its consumer to finish micro-batch
//    j - n_copies, the last reader of that slot: the next stage for the
//    residual stream, and the drafter's (last) stage for a DSpark tap made on a
//    stage further back.
// A call of n tokens is cut into k = ceil(n / TS_MINIMAX_M3_PIPELINE_UBATCH)
// equal micro-batches; the last is padded with fewer than k copies of the final
// token, so no partial micro-batch has to cross every GPU alone. The padding
// writes cache rows past the call's end, which are rewritten before any query
// reads them. Near the context limit, where it cannot pad, the call keeps
// floor(n / k)-row micro-batches and runs the few left-over tokens on the
// sequential path.
// A sparse call's micro-batches run sparse attention from the first one on:
// below the dense span that selects every visible block, so it is exact.
// A micro-batch's masks (F16 causal + F32 block force) stage at most this much:
// a context long enough to need more prefills one micro-batch at a time.
static constexpr size_t PIPE_MASK_BUDGET = (size_t) 96 << 20;
// Adaptive micro-batches (TS_MINIMAX_M3_PIPELINE_UBATCH unset) have at least
// PIPE_MIN_UBATCH rows, and shrink to PIPE_MASK_MIN_UBATCH only where a long
// context's masks need it. A 512-token prompt as two 256-row micro-batches ran
// at 446 tok/s on 8x A40, one 512-row pass at 394.
static constexpr int64_t PIPE_MIN_UBATCH = 256;
static constexpr int64_t PIPE_MASK_MIN_UBATCH = 128;

static bool pipe_warn(model & m, const std::string & reason)
{
    if (m.pipe.warned.insert(reason).second)
    {
        fprintf(stderr, "[minimax-m3] pipelined prefill declined (%s); this call runs one micro-batch at a time\n", reason.c_str());
        fflush(stderr);
    }
    m.pipe.n_declines++;
    return false;
}

/// How a call of n tokens at position p0 runs pipelined: n_full micro-batches
/// of u rows, the last padded with `pad` copies of the final token. The
/// micro-batch is TS_MINIMAX_M3_PIPELINE_UBATCH when set. Otherwise the call
/// takes the count pipe_adaptive_count picks, with micro-batches between
/// PIPE_MIN_UBATCH and the prefill ubatch, smaller where a long context's masks
/// need it.
enum pipe_verdict { PIPE_OK, PIPE_TOO_SHORT, PIPE_MASKS, PIPE_CTX };
struct pipe_shape { int64_t u = 0, n_full = 0, pad = 0, p_last = 0, n_kv = 0, chunk = 0; bool sparse = false; };

static size_t pipe_mask_bytes(const model & m, int64_t u, int64_t n_kv, bool sparse)
{
    bd_token b;
    b.sparse = sparse;
    return (seg_needs_mask(m, b) ? (size_t) n_kv * (size_t) u * sizeof(uint16_t) : 0) +
           (seg_needs_bias(m, b) ? (size_t) (n_kv / m.hp.idx_blk) * (size_t) u * sizeof(float) : 0);
}

static std::vector<int> pipe_stage_list(const model & m);

/// Adaptive micro-batch count of an n-token call. A stage spends about
/// t0 * (1 + u / R) on a u-row micro-batch: any micro-batch past a few hundred
/// rows streams every routed expert, and the expert products only grow with the
/// rows once each expert's share fills a 128-column matrix-multiply tile, at
/// R = 128 * n_expert / n_expert_used rows (4096 for MiniMax-M3.1). k
/// micro-batches over S stages take k + S - 1 stage times, so the call takes
/// the k in [2, n / u_min] (and at least n / u_max) that minimizes
/// (k + S - 1) * (1 + n / (k R)). On 8x A40 (8 stages, prefill tok/s) that is
/// 2 for 2048 rows (1025; a quarter of the call as 4 x 512 ran at 734, one
/// 2048-row pass at 759), 3 for 4096 (1445; 4 x 1024 1144, 2 x 2048 1315) and
/// 4 for 8192 (1631; 8 x 1024 1454). An extra micro-batch measured dearer than
/// the one stage time the estimate charges, which only favors these counts.
static int64_t pipe_adaptive_count(const model & m, int64_t n, int64_t u_min, int64_t u_max)
{
    const size_t n_stages = m.pipe.stages.empty() ? pipe_stage_list(m).size() : m.pipe.stages.size();
    const double S = (double) std::max<size_t>(1, n_stages);
    const double R = m.hp.n_expert > 0 && m.hp.n_expert_used > 0 ? 128.0 * m.hp.n_expert / m.hp.n_expert_used
                                                                  : (double) PIPE_MIN_UBATCH;
    const int64_t k_lo = std::max<int64_t>(2, (n + u_max - 1) / u_max);
    const int64_t k_hi = std::max<int64_t>(k_lo, n / std::max<int64_t>(1, u_min));
    int64_t best = k_lo;
    double best_cost = INFINITY;
    for (int64_t k = k_lo; k <= k_hi; k++)
    {
        const double cost = ((double) k + S - 1.0) * (1.0 + (double) n / ((double) k * R));
        if (cost < best_cost * (1.0 - 1e-9)) { best_cost = cost; best = k; }
    }
    return best;
}

static pipe_verdict pipe_plan(const model & m, int64_t p0, int64_t n, pipe_shape * out)
{
    const pipeline_state & P = m.pipe;
    if (n < 2 || p0 + n > m.n_ctx) return n < 2 ? PIPE_TOO_SHORT : PIPE_CTX;
    const int64_t u_max = P.ubatch_fixed ? P.ubatch : std::max<int64_t>(1, P.ubatch);
    const int64_t u_min = P.ubatch_fixed ? P.ubatch : std::min<int64_t>(PIPE_MIN_UBATCH, u_max);
    int64_t target = P.ubatch;
    if (!P.ubatch_fixed)
    {
        const int64_t k = pipe_adaptive_count(m, n, u_min, u_max);
        target = std::clamp<int64_t>((n + k - 1) / k, u_min, u_max);
    }
    for (;;)
    {
        pipe_shape sh;
        sh.n_full = (n + target - 1) / target;
        if (sh.n_full < 2) return PIPE_TOO_SHORT;
        sh.u = (n + sh.n_full - 1) / sh.n_full;
        sh.pad = sh.n_full * sh.u - n;                       // < n_full
        if (p0 + sh.n_full * sh.u > m.n_ctx) { sh.u = n / sh.n_full; sh.pad = 0; }
        sh.p_last = p0 + sh.n_full * sh.u;
        // The last micro-batch has the largest window; it decides the path.
        sh.n_kv = plan_n_kv(m, sh.p_last, m.kv_pad);
        sh.sparse = segment_sparse(m, sh.p_last);
        sh.chunk = sh.sparse ? msa_chunk_tokens(m, sh.n_kv, sh.u) : 0;
        if (pipe_mask_bytes(m, sh.u, sh.n_kv, sh.sparse) <= PIPE_MASK_BUDGET) { *out = sh; return PIPE_OK; }
        if (P.ubatch_fixed || sh.u <= PIPE_MASK_MIN_UBATCH) return PIPE_MASKS;
        target = std::max<int64_t>(PIPE_MASK_MIN_UBATCH, sh.u / 2);
    }
}

static size_t pipe_slot_layout(const model & m, int64_t u, int64_t n_kv, bool sparse, size_t * off_kv, size_t * off_out,
                               size_t * off_mask, size_t * off_bias)
{
    auto al = [](size_t x) { return (x + 255) & ~(size_t) 255; };
    size_t o = al((size_t) u * sizeof(int32_t));                       // pos
    *off_kv = o;   o = al(o + (size_t) u * sizeof(int64_t));
    *off_out = o;  o = al(o + sizeof(int32_t));
    bd_token b;
    b.sparse = sparse;
    *off_mask = o; o = al(o + (seg_needs_mask(m, b) ? (size_t) (n_kv * u) * sizeof(uint16_t) : 0));
    *off_bias = o; o = al(o + (seg_needs_bias(m, b) ? (size_t) (n_kv / m.hp.idx_blk * u) * sizeof(float) : 0));
    return o;
}

/// The GPUs holding layers, in layer order: the pipeline's stages.
static std::vector<int> pipe_stage_list(const model & m)
{
    std::vector<int> stages;
    for (int il = 0; il < m.hp.n_layer; il++)
    {
        const int d = m.layers[(size_t) il].device;
        if (d < m.n_gpu && (stages.empty() || stages.back() != d)) stages.push_back(d);
    }
    return stages;
}

/// Scheduler, events and staging for micro-batches of `u` rows over windows
/// up to n_kv. Runs only while nothing is in flight.
static bool pipe_ensure(model & m, int64_t u, int64_t n_kv, bool sparse, size_t n_nodes)
{
    pipeline_state & P = m.pipe;
    if (P.sched && n_nodes > P.n_nodes) P.release();
    if (!P.sched)
    {
        P.n_nodes = n_nodes;
        P.bufts.assign(m.sched_bufts, m.sched_bufts + m.n_sched_backends);
        // Host-side tensors (gathered token rows) in pinned memory, as llama.cpp does.
        if (m.n_gpu > 0)
            if (ggml_backend_buffer_type_t host = ggml_backend_dev_host_buffer_type(ggml_backend_get_device(m.backends[0])))
                P.bufts[(size_t) m.n_gpu] = host;
        P.sched = ggml_backend_sched_new(m.sched_backends, P.bufts.data(), m.n_sched_backends, P.n_nodes, true, m.op_offload);
        if (!P.sched) return false;
        P.n_copies = std::max(1, ggml_backend_sched_get_n_copies(P.sched));
        P.meta.resize(ggml_tensor_overhead() * P.n_nodes + ggml_graph_overhead_custom(P.n_nodes, false));
        P.stages = pipe_stage_list(m);
        P.n_staging = (int) P.stages.size() + 2;
        P.done.assign(P.stages.size() * (size_t) P.n_copies, nullptr);
        P.staged.assign(P.stages.size() * (size_t) P.n_staging, nullptr);
        for (size_t i = 0; i < P.stages.size(); i++)
        {
            ggml_backend_dev_t dev = ggml_backend_get_device(m.backends[P.stages[i]]);
            for (int c = 0; c < P.n_copies; c++) P.done[i * (size_t) P.n_copies + (size_t) c] = ggml_backend_event_new(dev);
            for (int h = 0; h < P.n_staging; h++) P.staged[i * (size_t) P.n_staging + (size_t) h] = ggml_backend_event_new(dev);
        }
        P.busy.assign((size_t) P.n_staging, 0);
    }
    size_t a, b, c, d;
    const size_t need = pipe_slot_layout(m, u, n_kv, sparse, &a, &b, &c, &d);
    if (need > P.slot_bytes)
    {
        if (P.staging) { ggml_backend_buffer_free(P.staging); P.staging = nullptr; }
        ggml_backend_buffer_type_t host = m.n_gpu > 0 ? ggml_backend_dev_host_buffer_type(ggml_backend_get_device(m.backends[0])) : nullptr;
        ggml_backend_buffer_type_t bt = host ? host : ggml_backend_cpu_buffer_type();
        // Doubling amortizes growth, but never past what the mask budget can need.
        const size_t grown = std::max(need, std::min(P.slot_bytes * 2, PIPE_MASK_BUDGET + ((size_t) 1 << 20)));
        P.staging = ggml_backend_buft_alloc_buffer(bt, grown * (size_t) P.n_staging);
        // ggml-cuda falls back to pageable memory when pinning fails, and an
        // async upload from pageable memory may wait for the stream.
        if (!P.staging || (host && ggml_backend_buffer_get_type(P.staging) != host))
        {
            if (P.staging) { ggml_backend_buffer_free(P.staging); P.staging = nullptr; }
            P.slot_bytes = 0;
            return false;
        }
        P.slot_bytes = grown;
        std::fill(P.busy.begin(), P.busy.end(), 0);
    }
    return true;
}

static void pipe_upload(ggml_backend_t be, ggml_tensor * t, const void * src, size_t bytes)
{
    if (live(t)) ggml_backend_tensor_set_async(be, t, src, 0, bytes);
}

/// Returns 1 when [0, *done) of the call ran pipelined and every requested
/// output is in place (*done == n), -1 when it declined after *done tokens
/// (the caller continues sequentially from there), 0 on failure.
static int forward_pipelined(model & m, const int32_t * tokens, int64_t n, float * logits_out, int64_t * done)
{
    pipeline_state & P = m.pipe;
    *done = 0;
    slot & s = *m.active_slot;
    const int64_t p0 = s.n_past;
    pipe_shape sh;
    switch (pipe_plan(m, p0, n, &sh))
    {
        case PIPE_OK: break;
        case PIPE_MASKS:
            pipe_warn(m, "masks of a context this long exceed the staging budget even in " +
                      std::to_string(m.pipe.ubatch_fixed ? m.pipe.ubatch : PIPE_MASK_MIN_UBATCH) + "-row micro-batches");
            return -1;
        default: return -1;                       // too short to overlap, or past the context (the sequential path reports it)
    }
    const int64_t n_full = sh.n_full, u = sh.u, pad = sh.pad, p_last = sh.p_last;
    std::vector<int32_t> padded;
    if (pad > 0)
    {
        padded.assign(tokens, tokens + n);
        padded.resize((size_t) (n + pad), tokens[n - 1]);
        tokens = padded.data();
    }

    // The largest micro-batch is the last: its window sizes the reservation and the staging.
    const int64_t n_kv = sh.n_kv;
    if (P.fail_u && u >= P.fail_u && n_kv >= P.fail_kv)
        return pipe_warn(m, "its compute buffers do not fit"), -1;
    // Cache rows are reserved up front: growing them moves tensors that in-flight
    // micro-batches still read.
    if (!slot_reserve_rows(m, s, plan_n_kv(m, std::max(p0 + n, p_last), m.kv_pad)))
        return pipe_warn(m, "no device memory to reserve this call's K/V rows up front"), -1;

    auto segment_at = [&](int64_t pj) -> bd_token
    {
        bd_token b;
        b.slot_id = s.id; b.p = pj; b.len = u;
        b.n_kv = plan_n_kv(m, pj + u, m.kv_pad);
        b.sparse = sh.sparse;
        b.chunk = sh.chunk;
        return b;
    };
    std::unique_ptr<graph_entry> probe_key(new graph_entry());
    probe_key->bd.push_back(segment_at(p_last - u));
    const size_t n_nodes = graph_nodes(m, *probe_key);
    if (!pipe_ensure(m, u, n_kv, sh.sparse, n_nodes)) { pipe_warn(m, "no pinned host memory for its input staging"); return -1; }

    size_t off_kv, off_out, off_mask, off_bias;
    pipe_slot_layout(m, u, n_kv, sh.sparse, &off_kv, &off_out, &off_mask, &off_bias);
    uint8_t * stage_base = (uint8_t *) ggml_backend_buffer_get_base(P.staging);
    const int C = P.n_copies;
    const size_t S = P.stages.size();
    const auto t_start = std::chrono::steady_clock::now();
    double enqueue_ms = 0.0;

    std::unique_ptr<graph_entry> g;
    auto new_graph = [&](int64_t pj) -> bool
    {
        if (g) { g->ctx = (ggml_free(g->ctx), nullptr); }
        g.reset(new graph_entry());
        g->owns_sched = false;
        g->kind = 0; g->slot_id = s.id; g->nt = u; g->p0 = pj;
        g->want_logits = true; g->n_out = 1; g->n_ovr = 0;
        g->bd.push_back(segment_at(pj));
        ggml_init_params gp = { P.meta.size(), P.meta.data(), true };
        g->ctx = ggml_init(gp);
        if (!g->ctx) return false;
        g->gf = ggml_new_graph_custom(g->ctx, P.n_nodes, false);
        g->sched = P.sched;
        ggml_backend_sched_reset(P.sched);
        builder b(m, *g);
        b.build();
        return true;
    };
    // Smaller windows later in the call (and in later calls) fit the reserved
    // layout as long as the topology is the same: ggml reallocates (draining
    // every GPU and moving the input copies) when the node count differs.
    bool reserve = u > P.res_u || n_kv > P.res_kv;
    if (!reserve)
    {
        if (!new_graph(p0)) return 0;
        reserve = ggml_graph_n_nodes(g->gf) != P.res_nodes;
    }
    if (reserve)
    {
        if (!new_graph(p_last - u) || !ggml_backend_sched_reserve(P.sched, g->gf))
        {
            // A failed reserve may have regrown some GPUs' buffers already: give
            // them back to the sequential path, and do not retry this shape.
            P.release();
            P.fail_u = u; P.fail_kv = n_kv;
            pipe_warn(m, "its compute buffers do not fit");
            return -1;
        }
        P.res_u = u; P.res_kv = n_kv;
        P.res_nodes = ggml_graph_n_nodes(g->gf);
    }

    // Copy slots rotate from 0 after a synchronize of an unallocated scheduler.
    ggml_backend_sched_reset(P.sched);
    ggml_backend_sched_synchronize(P.sched);

    // The DSpark commit concatenates the tapped layers' outputs on the drafter's
    // GPU (the last stage), so a tap made on an earlier stage crosses straight to
    // it, past the stages between. ggml copies it into the drafter stage's copy
    // slot on the PRODUCER's stream, and the adjacent-stage guard below orders a
    // producer only after the NEXT stage finished micro-batch j - C: with three
    // or more stages, a producer two or more stages back could overwrite the slot
    // while the drafter's stage still reads it for j - C. Those producers also
    // wait for the drafter's stage itself.
    std::vector<size_t> tap_stages;   // stages (indexes into P.stages) whose taps skip to the drafter's stage
    size_t ds_stage = S;
    if (m.ds.loaded)
    {
        for (size_t i = 0; i < S; i++) if (P.stages[i] == m.ds.dev) ds_stage = i;
        for (const int32_t k : m.ds.target_layers)
        {
            if (k < 1 || k > m.hp.n_layer) continue;
            const int dev = m.layers[(size_t) k - 1].device;
            for (size_t i = 0; i < S; i++)
                if (P.stages[i] == dev && ds_stage < S && i + 1 < ds_stage &&
                    std::find(tap_stages.begin(), tap_stages.end(), i) == tap_stages.end())
                    tap_stages.push_back(i);
        }
    }
#if defined(TSG_GGML_TEST_HOOKS)
    if (g_test_ds_no_tap_wait.load()) tap_stages.clear();
#endif

    std::vector<const void *> first_pos_addr((size_t) C, nullptr);
    bool in_flight = false;
    // Host time per phase (TS_MINIMAX_M3_EXEC_DEBUG): where an enqueue that is
    // not much shorter than the call itself is spent.
    double ms_wait = 0, ms_build = 0, ms_inputs = 0, ms_submit = 0;
    auto lap = [](std::chrono::steady_clock::time_point & t) {
        const auto now = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(now - t).count();
        t = now;
        return ms;
    };
    for (int64_t j = 0; j < n_full; j++)
    {
        const auto t_j = std::chrono::steady_clock::now();
        const int64_t pj = p0 + j * u;
        const int c = (int) (j % C);
        const int h = (int) (j % P.n_staging);
        uint8_t * slot_mem = stage_base + (size_t) h * P.slot_bytes;
        auto t_phase = t_j;
        // The uploads that last read this staging slot must have run.
        if (P.busy[(size_t) h])
            for (size_t i = 0; i < S; i++)
                if (ggml_backend_event_t ev = P.staged[i * (size_t) P.n_staging + (size_t) h]) ggml_backend_event_synchronize(ev);
        ms_wait += lap(t_phase);

        // ---- graph -----------------------------------------------------------
        if (!new_graph(pj)) { if (in_flight) ggml_backend_sched_synchronize(P.sched); return 0; }
        const bd_token B = g->bd[0];
        if (!ggml_backend_sched_alloc_graph(P.sched, g->gf))
        {
            // ggml may have freed buffers it then failed to regrow: rebuild on the next call.
            if (in_flight) { ggml_backend_sched_synchronize(P.sched); }
            P.release();
            if (j == 0) { pipe_warn(m, "its compute buffers do not fit"); return -1; }
            fprintf(stderr, "[minimax-m3] pipelined prefill: graph allocation failed after %" PRId64 " micro-batches\n", j);
            return 0;
        }
        // ggml must hand out the copy slots in order; anything else would let
        // a micro-batch overwrite inputs another one still reads. The probe is
        // the first stage-0 input the graph reads (an input no node reads has
        // no copies and no memory).
        if (!P.stages.empty() && C > 1)
        {
            const int d0 = P.stages[0];
            const ggml_tensor * probe = nullptr;
            for (const ggml_tensor * t : { g->inp.pos[d0], g->inp.kv_idx[d0], g->inp.mask[d0], g->inp.bias[d0] })
                if (live(t)) { probe = t; break; }
            const void * a = probe ? probe->data : nullptr;
            const bool repeats = a && j >= C && a != first_pos_addr[(size_t) c];
            const bool collides = a && j >= 1 && a == first_pos_addr[(size_t) ((c + C - 1) % C)];
            if (j < C) first_pos_addr[(size_t) c] = a;
            if (repeats || collides)
            {
                if (in_flight) ggml_backend_sched_synchronize(P.sched);
                P.enabled = false;
                P.release();
                fprintf(stderr, "[minimax-m3] pipelined prefill disabled: ggml did not rotate its input copies as expected; "
                        "prompts now prefill one %d-row micro-batch at a time\n", m.n_ubatch);
                *done = j * u;
                s.n_past = pj;
                return -1;
            }
        }

        ms_build += lap(t_phase);
        // ---- inputs ------------------------------------------------------------
        int32_t * pos = (int32_t *) slot_mem;
        int64_t * kv = (int64_t *) (slot_mem + off_kv);
        int32_t * out_id = (int32_t *) (slot_mem + off_out);
        uint16_t * mk = (uint16_t *) (slot_mem + off_mask);
        float * bias = (float *) (slot_mem + off_bias);
        for (int64_t i = 0; i < u; i++)
        {
            pos[i] = (int32_t) (pj + i);
            kv[i] = pj + i;
        }
        *out_id = (int32_t) (u - 1 - (j + 1 == n_full ? pad : 0));   // the last real token
        const int64_t nblk = B.n_kv / m.hp.idx_blk;
        if (seg_needs_mask(m, B)) fill_mask(mk, B.n_kv, pj, u);
        if (seg_needs_bias(m, B)) fill_bias(m.hp, bias, nblk, pj, u);
        set_i32(g->inp.tokens, tokens + j * u, (size_t) u);   // host buffer: a plain copy
        for (int d = 0; d <= m.n_gpu; d++)
        {
            ggml_backend_t be = m.backends[d];
            pipe_upload(be, g->inp.pos[d], pos, (size_t) u * sizeof(int32_t));
            pipe_upload(be, g->inp.kv_idx[d], kv, (size_t) u * sizeof(int64_t));
            if (seg_needs_mask(m, B)) pipe_upload(be, g->inp.mask[d], mk, (size_t) (B.n_kv * u) * sizeof(uint16_t));
            if (seg_needs_bias(m, B)) pipe_upload(be, g->inp.bias[d], bias, (size_t) (nblk * u) * sizeof(float));
        }
        if (live(g->inp.out_ids))
        {
            ggml_backend_t be = ggml_backend_sched_get_tensor_backend(P.sched, g->inp.out_ids);
            if (be) pipe_upload(be, g->inp.out_ids, out_id, sizeof(int32_t));
            else set_i32(g->inp.out_ids, out_id, 1);
        }
        for (size_t i = 0; i < S; i++)
            if (ggml_backend_event_t ev = P.staged[i * (size_t) P.n_staging + (size_t) h])
                ggml_backend_event_record(ev, m.backends[P.stages[i]]);
        P.busy[(size_t) h] = 1;

        ms_inputs += lap(t_phase);
        // ---- ordering guards and submission ----------------------------------
        if (j >= C)
        {
            for (size_t i = 0; i + 1 < S; i++)
                if (ggml_backend_event_t ev = P.done[(i + 1) * (size_t) C + (size_t) c])
                    ggml_backend_event_wait(m.backends[P.stages[i]], ev);
            for (const size_t i : tap_stages)
                if (ggml_backend_event_t ev = P.done[ds_stage * (size_t) C + (size_t) c])
                    ggml_backend_event_wait(m.backends[P.stages[i]], ev);
        }
        if (ggml_backend_sched_graph_compute_async(P.sched, g->gf) != GGML_STATUS_SUCCESS)
        {
            ggml_backend_sched_synchronize(P.sched);
            fprintf(stderr, "[minimax-m3] pipelined prefill: graph compute failed\n");
            return 0;
        }
        in_flight = true;
        for (size_t i = 0; i < S; i++)
            if (ggml_backend_event_t ev = P.done[i * (size_t) C + (size_t) c])
                ggml_backend_event_record(ev, m.backends[P.stages[i]]);
        ms_submit += lap(t_phase);
        dspark_rows_written(m, s, pj, u);
        s.n_past = pj + u;
        P.n_ubatches++;
        (B.sparse ? m.n_sparse_segments : m.n_dense_segments)++;
        enqueue_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_j).count();
    }

    ggml_backend_sched_synchronize(P.sched);
    std::fill(P.busy.begin(), P.busy.end(), 0);
    if (pad > 0) s.n_past = p0 + n;   // the padding is not part of the sequence
    *done = std::min<int64_t>(n, n_full * u);
    if (*done == n && logits_out && g->logits)
        ggml_backend_tensor_get(g->logits, logits_out, 0, (size_t) m.hp.n_vocab * sizeof(float));
    P.n_calls++;
    if (debug_on())
        fprintf(stderr, "[minimax-m3] pipelined prefill: %" PRId64 " x %" PRId64 " rows (+%" PRId64 " padding, %s) over %zu stages, "
                "enqueue %.0f ms (staging wait %.0f, graph %.0f, inputs %.0f, submit %.0f), total %.0f ms (%.0f tok/s)\n",
                n_full, u, pad, sh.sparse ? "sparse" : "dense", S, enqueue_ms, ms_wait, ms_build, ms_inputs, ms_submit,
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_start).count(),
                (double) *done / std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count());
    return *done == n ? 1 : -1;
}

/// Evaluate `n` tokens on the active slot in ubatches. Logits of the last
/// token (or every row when all_rows) land in logits_out. Queued vision rows
/// replace the embeddings at their absolute positions within this call.
static bool forward(model & m, const int32_t * tokens, int64_t n, float * logits_out, bool all_rows)
{
    if (!m.active_slot || n <= 0) return false;
    const int64_t base = m.active_slot->n_past;   // absolute position of tokens[0]
    bool has_ovr = false;
    for (const auto & o : m.embd_ovr)
    {
        const int64_t nrows = (int64_t) (o.rows.size() / (size_t) m.hp.n_embd);
        if (o.index < base + n && o.index + nrows > base) has_ovr = true;
        if (o.index + nrows > base + n)
        {
            static std::once_flag warned;
            std::call_once(warned, [&]() {
                fprintf(stderr, "[minimax-m3] vision rows queued for positions %" PRId64 "..%" PRId64 " reach past this call's "
                        "last position %" PRId64 "; the rows past it are dropped (queue each call's rows with that call)\n",
                        o.index, o.index + nrows - 1, base + n - 1);
            });
        }
    }
    int64_t off = 0;
    if (m.pipe.enabled && !all_rows && n >= 2 && has_ovr)
        pipe_warn(m, "image rows replace token embeddings inside pipelined micro-batches");
    if (m.pipe.enabled && !all_rows && !has_ovr && n >= 2)
    {
        const int rc = forward_pipelined(m, tokens, n, logits_out, &off);
        if (rc == 0) { m.embd_ovr.clear(); return false; }
        if (rc > 0) { m.embd_ovr.clear(); return true; }
        // declined after `off` tokens: the rest runs one micro-batch at a time
    }
    // vision overrides relative to this ubatch's token array
    std::vector<float> rows;
    std::vector<int64_t> idx;
    int64_t ubatch = m.n_ubatch;
    int halvings = 0;
    for (; off < n;)
    {
        const int64_t cnt = std::min<int64_t>(ubatch, n - off);
        rows.clear(); idx.clear();
        for (const auto & o : m.embd_ovr)
        {
            const int64_t nrows = (int64_t) (o.rows.size() / (size_t) m.hp.n_embd);
            for (int64_t r = 0; r < nrows; r++)
            {
                const int64_t at = o.index + r - base;
                if (at < off || at >= off + cnt) continue;
                idx.push_back(at - off);
                rows.insert(rows.end(), o.rows.begin() + (size_t) r * m.hp.n_embd, o.rows.begin() + (size_t) (r + 1) * m.hp.n_embd);
            }
        }
        const bool last = off + cnt == n;
        const bool want = last || all_rows;
        float * out = nullptr;
        if (want && logits_out) out = all_rows ? logits_out + (size_t) off * m.hp.n_vocab : logits_out;
        m.alloc_failed = false;
        if (!forward_ubatch(m, tokens + off, cnt, want, all_rows, out,
                            rows.empty() ? nullptr : rows.data(), idx.empty() ? nullptr : idx.data(), (int64_t) idx.size()))
        {
            // A ubatch whose graph did not fit leaves nothing computed: retry
            // the same tokens in halves for the rest of this call. The next
            // call starts at the full ubatch again, since memory is shared with
            // slots that come and go. Under --tp every failed build leaves views
            // of the static tensors behind that the meta backend holds only so
            // many of.
            if (m.alloc_failed && cnt > 64 && (m.tp.n <= 1 || ++halvings <= 3))
            {
                ubatch = std::max<int64_t>(64, cnt / 2);
                fprintf(stderr, "[minimax-m3] a %" PRId64 "-token graph does not fit; continuing with %" PRId64 "-token ubatches\n",
                        cnt, ubatch);
                continue;
            }
            m.embd_ovr.clear();
            return false;
        }
        off += cnt;
    }
    m.embd_ovr.clear();
    return true;
}

/// One graph over several slots: segment i is lens[i] tokens of slot
/// slot_ids[i] starting at positions[i] (which must be that slot's head).
/// logits_out receives one row per segment: its last token's logits. Each
/// segment's sparse attention runs at that segment's own positions.
static bool forward_segments(model & m, int n, const int32_t * slot_ids, const int32_t * lens, const int32_t * tokens,
                             const int32_t * positions, float * logits_out)
{
    if (n < 1 || n > MAX_BATCHED_DECODE) return false;
    std::unique_ptr<graph_entry> key(new graph_entry());
    key->kind = 1;
    key->want_logits = true;
    key->n_out = n;
    int64_t T = 0;
    bool decode_only = true;
    for (int i = 0; i < n; i++) { if (lens[i] < 1) return false; T += lens[i]; decode_only &= lens[i] == 1; }
    if (T > std::max<int64_t>(m.n_ubatch, n)) return false;
    key->nt = T;
    // A coarser pad for decode: N sequences cross window boundaries N times as
    // often, and each crossing rebuilds the graph. It stays a multiple of
    // kv_pad, so the rows still split into whole MSA blocks.
    const int64_t pad = decode_only ? pad_to(std::max<int64_t>(m.kv_pad, env_long("TS_MINIMAX_M3_BD_KV_PAD", 2 * m.kv_pad)), m.kv_pad)
                                    : m.kv_pad;
    std::vector<int64_t> p0s((size_t) n);
    for (int i = 0; i < n; i++)
    {
        auto it = m.slots.find(slot_ids[i]);
        if (it == m.slots.end()) return false;
        for (int j = 0; j < i; j++) if (slot_ids[j] == slot_ids[i]) return false;
        slot & sl = *it->second;
        if (positions[i] != sl.n_past || sl.n_past + lens[i] > m.n_ctx) return false;
        bd_token b;
        b.slot_id = sl.id;
        b.p = sl.n_past;
        b.len = lens[i];
        b.n_kv = plan_n_kv(m, sl.n_past + lens[i], pad);
        b.sparse = segment_sparse(m, sl.n_past + lens[i]);
        b.chunk = b.sparse ? msa_chunk_tokens(m, b.n_kv, b.len) : 0;
        // The graph reads every segment's slot: none of them is reclaimed for another's growth.
        if (!slot_reserve_rows(m, sl, b.n_kv, slot_ids, n)) return false;
        key->bd.push_back(b);
        p0s[(size_t) i] = sl.n_past;
    }

    bool reused = false;
    graph_entry * g = acquire_graph(m, std::move(key), &reused);
    if (!g) return false;

    // The cached graph matched on window shapes only: positions come from
    // this call, never from the entry (which keeps the first caller's).
    m.h_tokens.assign(tokens, tokens + T);
    fill_segment_inputs(m, g->bd, p0s.data());
    m.h_out_ids.resize((size_t) n);
    int64_t row = 0;
    for (int i = 0; i < n; i++) { row += g->bd[(size_t) i].len; m.h_out_ids[(size_t) i] = (int32_t) (row - 1); }
    set_i32(g->inp.tokens, m.h_tokens.data(), (size_t) T);
    set_i32(g->inp.out_ids, m.h_out_ids.data(), (size_t) n);
    upload_segment_inputs(m, *g);
    for (int i = 0; i < n; i++) dspark_rows_written(m, *m.slots.at(slot_ids[i]), p0s[(size_t) i], lens[i]);
    if (!compute(m, *g)) return false;
    for (int i = 0; i < n; i++) m.slots.at(slot_ids[i])->n_past += lens[i];
    count_segments(m, g->bd);
    ggml_backend_tensor_get(g->logits, logits_out, 0, (size_t) n * m.hp.n_vocab * sizeof(float));
    return true;
}

static bool forward_batched_decode(model & m, int n, const int32_t * slot_ids, const int32_t * tokens,
                                   const int32_t * positions, float * logits_out)
{
    std::vector<int32_t> ones((size_t) std::max(n, 0), 1);
    return forward_segments(m, n, slot_ids, ones.data(), tokens, positions, logits_out);
}


// ---------------------------------------------------------------------------
// load
// ---------------------------------------------------------------------------
/// Routed-expert weight stacks: what the placement may leave on the host. The
/// per-expert `.scale` vectors are dense (they stay with the layer).
static bool is_expert_weight(const char * name)
{
    return strstr(name, "_exps.weight") != nullptr;
}

/// Tensor parallelism over `tp` CUDA GPUs (ggml_ops_minimax_m3_tp): what it
/// cannot run is refused before any GPU is touched. Returns the meta backend,
/// or null after reporting the refusal.
static ggml_backend_t tp_open(model & m, int tp, const char * backend_name, const char * layer_split, int n_cpu_moe_req)
{
    const char * why = nullptr;
    if (!backend_name || strcmp(backend_name, "CUDA") != 0)
        why = "tensor parallelism (--tp) runs on CUDA GPUs only; use --layer-split N on this backend";
    else if (layer_split)
        why = "--tp cannot be combined with TS_MINIMAX_M3_LAYER_SPLIT (a layer split); use one or the other";
    else if (n_cpu_moe_req > 0)
        why = "--tp keeps every expert in GPU memory; --n-cpu-moe needs --layer-split";
    else if (m.msa == MSA_KERNEL)
        why = "TS_MINIMAX_M3_MSA=kernel cannot run under --tp: TensorSharp's sparse-attention node runs on the fused-op "
              "wrapper of a CUDA backend, which ggml's meta backend cannot hold; use auto or ggml (upstream ggml's sparse ops)";
    else if (opt_get("TS_MINIMAX_M3_PIPELINE") == "1")
        why = "TS_MINIMAX_M3_PIPELINE=1 cannot run under --tp (it pipelines a layer split's GPUs)";
    else if (env_long("TS_MINIMAX_M3_PROFILE", 0) > 0)
        why = "TS_MINIMAX_M3_PROFILE reads graph slices between --tp's reductions, which would profile wrong values";
    else if (env_long("TS_MINIMAX_M3_GRAPH_CACHE", 0) > 1)
        why = "TS_MINIMAX_M3_GRAPH_CACHE above 1 cannot run under --tp (its backend keeps one graph's GPU views)";
    else if (env_flag("TS_MINIMAX_M3_CPU_REPACK", false))
        why = "TS_MINIMAX_M3_CPU_REPACK=1 repacks host experts, which --tp does not use";
    if (why) { tsg::report_load_refusal("[minimax-m3] %s\n", why); return nullptr; }
    m.tp.ud.reset(new tsg_mm3_tp::userdata());
    std::string err;
    ggml_backend_t be = tsg_mm3_tp::open(tp, m.tp.ud.get(), &m.tp.dev, &err);
    if (!be) { tsg::report_load_refusal("[minimax-m3] %s\n", err.c_str()); return nullptr; }
    m.tp.n = m.tp.dev.n;
    return be;
}

static model * load(const char * gguf_path, int n_gpu_req, int n_ctx, int n_ubatch, int n_threads,
                    int n_cpu_moe_req, const char * backend_name, bool ctx_is_hard_limit, const char * layer_split_arg, int tp,
                    const char * draft_path = nullptr)
{
    const auto t_start = std::chrono::steady_clock::now();
    std::unique_ptr<model> m(new model());

    // --- attention mode ---------------------------------------------------
    {
        const std::string mode = opt_get("TS_MINIMAX_M3_MSA");
        if (mode.empty() || mode == "auto") m->msa = MSA_AUTO;
        else if (mode == "dense") m->msa = MSA_DENSE;
        else if (mode == "ggml") m->msa = MSA_GGML;
        else if (mode == "kernel") m->msa = MSA_KERNEL;
        else
        {
            tsg::report_load_refusal("[minimax-m3] TS_MINIMAX_M3_MSA must be auto, dense, ggml or kernel, not '%s'\n", mode.c_str());
            return nullptr;
        }
#if !defined(TSG_MINIMAX_MSA_KERNELS)
        if (m->msa == MSA_KERNEL)
        {
            tsg::report_load_refusal("[minimax-m3] TS_MINIMAX_M3_MSA=kernel runs TensorSharp's sparse-attention kernels "
                                     "(ggml_ops_minimax_msa), which this build does not include yet; use auto or ggml\n");
            return nullptr;
        }
#endif
        const long mb = env_long("TS_MINIMAX_M3_MSA_CHUNK_MB", 256);
        if (mb < 1)
        {
            tsg::report_load_refusal("[minimax-m3] TS_MINIMAX_M3_MSA_CHUNK_MB must be a positive number of MiB, not '%s'\n",
                                     opt_get("TS_MINIMAX_M3_MSA_CHUNK_MB").c_str());
            return nullptr;
        }
        m->msa_chunk_bytes = (size_t) mb << 20;
    }
    const std::string layer_split_str = layer_split_arg && *layer_split_arg ? std::string(layer_split_arg)
                                                                            : opt_get("TS_MINIMAX_M3_LAYER_SPLIT");
    const char * layer_split = layer_split_str.empty() ? nullptr : layer_split_str.c_str();

    // --- backends ---------------------------------------------------------
    const bool cpu_only = backend_name && strcmp(backend_name, "CPU") == 0;
    if (!cpu_only && n_gpu_req > MAX_GPUS && tp <= 1)
    {
        tsg::report_load_refusal("[minimax-m3] %d GPUs requested; the native executor spans at most %d\n", n_gpu_req, MAX_GPUS);
        return nullptr;
    }
    int n_gpu = 0;
    if (tp > 1)
    {
        ggml_backend_t be = tp_open(*m, tp, backend_name, layer_split, n_cpu_moe_req);
        if (!be) return nullptr;
        m->backends[n_gpu++] = be;
        n_gpu_req = 1;
        if (n_cpu_moe_req < 0) n_cpu_moe_req = 0;   // automatic: every expert in GPU memory
    }
    for (int pass = 0; pass < 2 && n_gpu == 0 && !cpu_only; pass++)
    {
        const char * want = pass == 0 ? backend_name : nullptr;
        for (size_t i = 0; i < ggml_backend_dev_count() && n_gpu < MAX_GPUS; i++)
        {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
            const char * reg_name = reg ? ggml_backend_reg_name(reg) : nullptr;
            // ggml-cuda IGPUs count too, as in tsg::enumerate_gpu_devices: some
            // virtualized hosts report a discrete card as integrated. Other
            // registries keep integrated GPUs out of a model this size.
            const enum ggml_backend_dev_type dt = ggml_backend_dev_type(dev);
            const bool cuda_reg = reg_name && strcmp(reg_name, "CUDA") == 0;
            if (dt != GGML_BACKEND_DEVICE_TYPE_GPU && !(cuda_reg && dt == GGML_BACKEND_DEVICE_TYPE_IGPU)) continue;
            if (!backend_matches(reg_name, want)) continue;
            if (n_gpu_req > 0 && n_gpu >= n_gpu_req) break;
            ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
            if (!be) continue;
            m->backends[n_gpu++] = be;
        }
    }
    m->n_gpu = n_gpu;

    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!cpu) { tsg::report_load_refusal("[minimax-m3] CPU backend unavailable\n"); return nullptr; }
    {
        int cpu_threads = n_threads > 0 ? n_threads : tsg::available_cpu_parallelism();
        if (n_cpu_moe_req != 0 || n_gpu == 0) cpu_threads = tsg::available_cpu_parallelism();
        // Routed experts on the host beside GPUs: their decode is memory-bandwidth bound (the same token rate
        // from 32 to 54 workers on a 2x24-core host), and a layer's experts meet at a barrier, so a worker that
        // shares its core with the serving process's own threads (.NET runtime and server GC, HTTP, the
        // scheduler) stalls the whole layer. Half the CPU quota brought the server's decode up to the
        // executor's own rate (7.7-8.4 -> 14-17 tok/s on 8x A40 with the experts of 24 layers on the host) at
        // a quarter less host throughput for prompts. TS_CPU_MOE_THREADS / --cpu-moe-threads override it.
        if (n_cpu_moe_req != 0 && n_gpu > 0) cpu_threads = std::max(2, cpu_threads / 2);
        const int explicit_threads = tsg::host_moe_explicit_thread_count();
        if (explicit_threads > 0) cpu_threads = explicit_threads;
        if (const long v = env_long("TS_CPU_MOE_THREADS", 0); v > 0) cpu_threads = (int) v;
        ggml_backend_cpu_set_n_threads(cpu, cpu_threads);
        ggml_threadpool_params tpp = ggml_threadpool_params_default(cpu_threads);
        m->cpu_threadpool = ggml_threadpool_new(&tpp);
        if (m->cpu_threadpool) ggml_backend_cpu_set_threadpool(cpu, m->cpu_threadpool);
        m->cpu_threads = cpu_threads;
    }
    m->backends[n_gpu] = cpu;
    m->n_backends = n_gpu + 1;
#if defined(TSG_MINIMAX_MSA_KERNELS) && defined(TSG_GGML_USE_CUDA)
    // The sparse-attention node runs on a TensorSharp wrapper that REPLACES
    // its GPU's CUDA backend in the scheduler (one split per device); weights,
    // caches, uploads and events stay on the CUDA backend itself. auto runs
    // the kernels on every CUDA GPU that gets one.
    if (m->msa == MSA_KERNEL || m->msa == MSA_AUTO)
        for (int d = 0; d < n_gpu; d++)
        {
            ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(m->backends[d]));
            if (!reg || strcmp(ggml_backend_reg_name(reg), "CUDA") != 0) continue;
            // The kernels are sm_80 code (cp.async, m16n8k16 MMA); on an older GPU
            // they would trap, so it keeps upstream ggml's sparse ops.
            int cc_major = 0, cc_minor = 0;
            bool cc_ok = tsg_msa_cuda_capability(m->backends[d], &cc_major, &cc_minor);
#if defined(TSG_GGML_TEST_HOOKS)
            if (g_test_msa_old_gpus.load()) { cc_major = 7; cc_minor = 5; }
#endif
            cc_ok = cc_ok && cc_major >= tsg_msa_min_cuda_major;
            if (!cc_ok && m->msa == MSA_KERNEL)
            {
                tsg::report_load_refusal("[minimax-m3] TS_MINIMAX_M3_MSA=kernel: TensorSharp's MSA kernels need compute "
                                         "capability %d.0 or newer and GPU %d is %d.%d; use auto or ggml (upstream ggml's "
                                         "sparse ops)\n", tsg_msa_min_cuda_major, d, cc_major, cc_minor);
                return nullptr;
            }
            if (!cc_ok)
            {
                fprintf(stderr, "[minimax-m3] GPU %d has compute capability %d.%d; TensorSharp's MSA kernels need %d.0 or "
                        "newer, so its layers run sparse attention through the ggml ops instead (slower on long contexts)\n",
                        d, cc_major, cc_minor, tsg_msa_min_cuda_major);
                continue;
            }
            m->wrappers[d] = tsg_dsv4_fused_backend_init(m->backends[d]);
            if (!m->wrappers[d] && m->msa == MSA_KERNEL)
            {
                tsg::report_load_refusal("[minimax-m3] TS_MINIMAX_M3_MSA=kernel: creating the fused-op backend for GPU %d failed\n", d);
                return nullptr;
            }
            if (!m->wrappers[d])
                fprintf(stderr, "[minimax-m3] creating the fused-op backend for GPU %d failed: its layers run sparse attention "
                        "through the ggml ops instead of TensorSharp's MSA kernels (slower on long contexts)\n", d);
            m->msa_kernel_dev[d] = m->wrappers[d] != nullptr;
        }
#endif
#if defined(TSG_MINIMAX_MSA_KERNELS)
    // kernel: every device, the host through the CPU reference. A GPU runs the
    // kernels only through its fused-op wrapper (CUDA): on any other GPU the
    // scheduler would hand each sparse layer's node to the CPU reference, with
    // the slot's whole caches copied to the host on every call. That is refused
    // rather than run silently; a CPU-only load keeps the reference.
    if (m->msa == MSA_KERNEL)
    {
        for (int d = 0; d < n_gpu; d++)
            if (!m->wrappers[d])
            {
                ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(m->backends[d]));
                tsg::report_load_refusal("[minimax-m3] TS_MINIMAX_M3_MSA=kernel runs TensorSharp's sparse-attention kernels on "
                                         "CUDA GPUs (or on the CPU of a CPU-only load); GPU %d (%s) has none, so each of its "
                                         "sparse layers would run the CPU reference on host copies of the caches. Use auto "
                                         "or ggml (upstream ggml's sparse ops) on this backend\n",
                                         d, reg ? ggml_backend_reg_name(reg) : "unknown");
                return nullptr;
            }
        for (int d = 0; d <= n_gpu; d++) m->msa_kernel_dev[d] = true;
    }
#endif
    for (int i = 0; i <= n_gpu; i++)
    {
        m->sched_backends[i] = i < n_gpu && m->wrappers[i] ? m->wrappers[i] : m->backends[i];
        m->sched_bufts[i] = ggml_backend_get_default_buffer_type(m->backends[i]);
    }
    m->n_sched_backends = n_gpu + 1;
    // Host tensors that cross into the meta backend (the gathered token rows)
    // are copied to every GPU synchronously: from pinned memory, as llama.cpp.
    if (m->tp.n > 1)
        if (ggml_backend_buffer_type_t host = ggml_backend_dev_host_buffer_type(m->tp.dev.meta))
            m->sched_bufts[n_gpu] = host;
    if (!cpu_only && n_gpu_req > 0 && n_gpu < n_gpu_req)
    {
        // The layers would silently crowd onto fewer GPUs than asked for.
        tsg::report_load_refusal("[minimax-m3] %d GPU(s) requested but only %d usable %s device(s) found\n",
                                 n_gpu_req, n_gpu, backend_name ? backend_name : "GPU");
        return nullptr;
    }
    if (const long v = env_long("TS_MINIMAX_M3_GRAPH_CACHE", 0); v > 0) m->graph_cache_cap = (int) v;
    // The meta backend keeps per-GPU views only for the graph it ran last.
    if (m->tp.n > 1) m->graph_cache_cap = 1;

    // --- metadata ---------------------------------------------------------
    int32_t split_count = 1;
    m->shards = resolve_shards(gguf_path, &split_count);
    hparams & hp = m->hp;
    {
        ggml_context * meta0 = nullptr;
        gguf_init_params ip = { true, &meta0 };
        gguf_context * g0 = gguf_init_from_file(m->shards.paths[0].c_str(), ip);
        if (!g0) { tsg::report_load_refusal("[minimax-m3] failed to open %s\n", m->shards.paths[0].c_str()); return nullptr; }
        bool ok = true;
        const int64_t ai = gguf_find_key(g0, "general.architecture");
        if (ai < 0 || gguf_get_kv_type(g0, ai) != GGUF_TYPE_STRING || strcmp(gguf_get_val_str(g0, ai), "minimax-m3") != 0)
            ok = false;
        ok = ok && kv_u32(g0, akey("block_count").c_str(), &hp.n_layer);
        ok = ok && kv_u32(g0, akey("embedding_length").c_str(), &hp.n_embd);
        ok = ok && kv_u32(g0, akey("attention.head_count").c_str(), &hp.n_head);
        ok = ok && kv_u32(g0, akey("attention.head_count_kv").c_str(), &hp.n_head_kv);
        ok = ok && kv_f32(g0, akey("attention.layer_norm_rms_epsilon").c_str(), &hp.rms_eps);
        ok = ok && kv_u32(g0, akey("expert_count").c_str(), &hp.n_expert);
        ok = ok && kv_u32(g0, akey("expert_used_count").c_str(), &hp.n_expert_used);
        ok = ok && kv_u32(g0, akey("expert_shared_count").c_str(), &hp.n_expert_shared);
        ok = ok && kv_u32_uniform(g0, akey("expert_feed_forward_length").c_str(), &hp.n_ff_exp);
        ok = ok && kv_u32(g0, akey("expert_gating_func").c_str(), &hp.gating);
        ok = ok && kv_u32(g0, akey("attention.indexer.head_count").c_str(), &hp.idx_n_head);
        ok = ok && kv_u32(g0, akey("attention.indexer.key_length").c_str(), &hp.idx_dim);
        ok = ok && kv_u32(g0, akey("attention.indexer.top_k").c_str(), &hp.idx_topk);
        ok = ok && kv_u32(g0, akey("attention.indexer.block_size").c_str(), &hp.idx_blk);
        ok = ok && kv_u32(g0, akey("attention.indexer.local_blocks").c_str(), &hp.idx_local);
        kv_u32(g0, akey("leading_dense_block_count").c_str(), &hp.n_lead);
        if (hp.n_lead > 0) ok = ok && kv_u32_uniform(g0, akey("feed_forward_length").c_str(), &hp.n_ff);
        int32_t key_len = 0;
        kv_u32(g0, akey("attention.key_length").c_str(), &key_len);
        hp.head_dim = key_len > 0 ? key_len : (hp.n_head > 0 ? hp.n_embd / hp.n_head : 0);
        hp.n_rot = hp.head_dim;
        kv_u32(g0, akey("rope.dimension_count").c_str(), &hp.n_rot);
        kv_f32(g0, akey("rope.freq_base").c_str(), &hp.rope_base);
        kv_f32(g0, akey("expert_weights_scale").c_str(), &hp.w_scale);
        hp.w_norm = false;   // llama.cpp's default when the key is absent
        kv_bool(g0, akey("expert_weights_norm").c_str(), &hp.w_norm);
        kv_u32(g0, akey("context_length").c_str(), &hp.n_ctx_train);
        gguf_free(g0);
        if (meta0) ggml_free(meta0);
        const char * why = nullptr;
        if (!ok) why = "missing minimax-m3 metadata";
        else if (hp.n_layer <= 0 || hp.n_head <= 0 || hp.n_head_kv <= 0 || hp.n_head % hp.n_head_kv != 0 || hp.head_dim <= 0)
            why = "invalid attention geometry";
        else if (hp.n_rot <= 0 || hp.n_rot % 2 != 0 || hp.n_rot > hp.head_dim) why = "invalid rope.dimension_count";
        else if (hp.gating != 2) why = "expert_gating_func is not sigmoid (2)";
        else if (hp.n_expert <= 0 || hp.n_expert_used <= 0 || hp.n_expert_used > hp.n_expert || hp.n_expert_shared < 1)
            why = "invalid expert counts (a shared expert is required)";
        else if (hp.n_lead < 0 || hp.n_lead > hp.n_layer) why = "invalid leading_dense_block_count";
        else if (hp.idx_n_head != hp.n_head_kv) why = "the indexer needs one head per KV head";
        else if (hp.idx_dim <= 0 || hp.idx_blk <= 0 || hp.idx_topk <= 0 || hp.idx_local < 0) why = "invalid indexer geometry";
        if (why)
        {
            tsg::report_load_refusal("[minimax-m3] %s in %s\n", why, m->shards.paths[0].c_str());
            return nullptr;
        }
    }

    // --- tensor sources ---------------------------------------------------
    std::map<std::string, tensor_source> sources;
    const bool idx_f16_req = env_flag("TS_MINIMAX_M3_INDEXER_F16", n_gpu > 0);
    std::vector<size_t> layer_dense((size_t) hp.n_layer, 0);
    // [layer][0 gate, 1 up, 2 down] routed-expert stack bytes
    std::vector<std::array<size_t, 3>> exp_bytes((size_t) hp.n_layer, std::array<size_t, 3>{ 0, 0, 0 });
    // Multiply-adds per token of each layer's weights (a routed stack counts
    // n_expert_used / n_expert of its elements): what a prefill stage spends.
    std::vector<double> layer_work((size_t) hp.n_layer, 0.0);
    for (size_t si = 0; si < m->shards.paths.size(); si++)
    {
        ggml_context * meta = nullptr;
        gguf_init_params sp = { true, &meta };
        gguf_context * g = gguf_init_from_file(m->shards.paths[si].c_str(), sp);
        if (!g) { tsg::report_load_refusal("[minimax-m3] failed to open shard %s\n", m->shards.paths[si].c_str()); return nullptr; }
        const size_t data_off = gguf_get_data_offset(g);
        const size_t fsz = file_size_of(m->shards.paths[si].c_str());
        for (int64_t ti = 0; ti < gguf_get_n_tensors(g); ti++)
        {
            const char * name = gguf_get_tensor_name(g, ti);
            ggml_tensor * mt = ggml_get_tensor(meta, name);
            if (!mt) continue;
            tensor_source src;
            src.shard = (int) si;
            src.offset = data_off + gguf_get_tensor_offset(g, ti);
            src.size = ggml_nbytes(mt);
            src.type = mt->type;
            for (int d = 0; d < 4; d++) src.ne[d] = mt->ne[d];
            if (fsz != (size_t) -1 && src.offset + src.size > fsz)
            {
                tsg::report_load_refusal("[minimax-m3] %s is incomplete (tensor %s ends past EOF); re-download it\n",
                                         m->shards.paths[si].c_str(), name);
                gguf_free(g); ggml_free(meta);
                return nullptr;
            }
            sources[name] = src;
            int bid = -1;
            if (sscanf(name, "blk.%d.", &bid) == 1 && bid >= 0 && bid < hp.n_layer)
            {
                const int k = strstr(name, "ffn_gate_exps.weight") ? 0 : strstr(name, "ffn_up_exps.weight") ? 1
                            : strstr(name, "ffn_down_exps.weight") ? 2 : -1;
                if (k >= 0 && is_expert_weight(name)) exp_bytes[(size_t) bid][(size_t) k] = src.size;
                else layer_dense[(size_t) bid] += src.size;
                // the F16 decode copy of an F32 indexer projection (project_index)
                if (idx_f16_req && src.type == GGML_TYPE_F32 && strstr(name, ".indexer.") && strstr(name, "_proj.weight"))
                    layer_dense[(size_t) bid] += src.size / 2;
                const double elems = (double) mt->ne[0] * (double) mt->ne[1] * (double) mt->ne[2] * (double) mt->ne[3];
                layer_work[(size_t) bid] += k >= 0 && hp.n_expert > 0 ? elems * hp.n_expert_used / hp.n_expert : elems;
            }
        }
        gguf_free(g);
        ggml_free(meta);
    }
    auto src_of = [&](const char * name) -> const tensor_source *
    {
        auto it = sources.find(name);
        return it == sources.end() ? nullptr : &it->second;
    };
    {
        const tensor_source * te = src_of("token_embd.weight");
        if (!te) { tsg::report_load_refusal("[minimax-m3] token_embd.weight missing\n"); return nullptr; }
        hp.n_vocab = (int32_t) te->ne[1];
    }
    const tensor_source * out_src = src_of("output.weight");
    const tensor_source * out_norm_src = src_of("output_norm.weight");
    if (!out_src || !out_norm_src) { tsg::report_load_refusal("[minimax-m3] output weights missing\n"); return nullptr; }
    size_t head_bytes = out_src->size + out_norm_src->size;

    m->layers.resize((size_t) hp.n_layer);
    for (int il = 0; il < hp.n_layer; il++)
    {
        layer & L = m->layers[(size_t) il];
        L.dense = il < hp.n_lead;
        // A layer is sparse iff it carries the indexer (llama.cpp's is_msa_layer);
        // MiniMax-M3.1 has it on every layer, the dense leading ones included.
        int n_idx = 0;
        for (const char * sfx : { "indexer.q_proj.weight", "indexer.k_proj.weight", "indexer.q_norm.weight", "indexer.k_norm.weight" })
        {
            char nm[96];
            snprintf(nm, sizeof(nm), "blk.%d.%s", il, sfx);
            n_idx += src_of(nm) ? 1 : 0;
        }
        if (n_idx != 0 && n_idx != 4)
        {
            tsg::report_load_refusal("[minimax-m3] layer %d has %d of the 4 indexer tensors\n", il, n_idx);
            return nullptr;
        }
        if (n_idx == 0 && !L.dense)
        {
            tsg::report_load_refusal("[minimax-m3] MoE layer %d has no indexer tensors\n", il);
            return nullptr;
        }
        L.msa = n_idx == 4;
        m->n_msa_layers += L.msa ? 1 : 0;
    }
    if (m->msa == MSA_DENSE && m->n_msa_layers > 0)
        fprintf(stderr, "[minimax-m3] TS_MINIMAX_M3_MSA=dense: every position attends to every earlier one instead of its "
                "selected blocks, which changes results for positions from %" PRId64 " on (debugging only)\n", hp.dense_span());

    m->n_ctx = n_ctx > 0 ? n_ctx : 8192;
    if (hp.n_ctx_train > 0 && m->n_ctx > hp.n_ctx_train) m->n_ctx = hp.n_ctx_train;
    {
        // Windows split into whole MSA blocks: the pad is a multiple of the block.
        const long want = std::max(1L, env_long("TS_MINIMAX_M3_KV_PAD", (long) KV_PAD_DEFAULT));
        m->kv_pad = pad_to(want, hp.idx_blk);
        if (m->kv_pad != want)
            fprintf(stderr, "[minimax-m3] TS_MINIMAX_M3_KV_PAD=%ld is not a multiple of the %d-row MSA block; windows are "
                    "padded to %" PRId64 " rows instead\n", want, hp.idx_blk, m->kv_pad);
    }
    m->n_ctx = (int32_t) pad_to(m->n_ctx, m->kv_pad);
    m->n_ubatch = n_ubatch > 0 ? n_ubatch : 512;
    {
        size_t dspark_bytes = 0;   // a DSpark drafter reads the output head, so it is placed beside it
        if (!dspark_scan(*m, draft_path, sources, &dspark_bytes)) return nullptr;
        head_bytes += dspark_bytes;
    }

    // --- placement --------------------------------------------------------
    // Rows grow with each sequence, so the plan reserves
    // TS_MINIMAX_M3_PLAN_CTX_TOKENS of them in total (one full context by
    // default, like llama.cpp's unified KV).
    const long plan_ctx_env = env_long("TS_MINIMAX_M3_PLAN_CTX_TOKENS", 0);
    const bool plan_ctx_explicit = plan_ctx_env > 0;
    if (!plan_ctx_explicit && !opt_get("TS_MINIMAX_M3_PLAN_CTX_TOKENS").empty())
        fprintf(stderr, "[minimax-m3] TS_MINIMAX_M3_PLAN_CTX_TOKENS=%s is not a positive token count; the plan "
                "reserves K/V rows for the whole context instead\n", opt_get("TS_MINIMAX_M3_PLAN_CTX_TOKENS").c_str());
    size_t plan_ctx_tokens = plan_ctx_explicit ? (size_t) plan_ctx_env : (size_t) m->n_ctx;
    // Re-size everything the context sizes (the placement plan reads these).
    auto set_ctx = [&](int32_t ctx)
    {
        m->n_ctx = ctx;
        if (!plan_ctx_explicit) plan_ctx_tokens = (size_t) ctx;
    };
    auto kv_layer_bytes_of = [&](int il) -> size_t
    {
        return kv_bytes_per_token_layer(hp, m->layers[(size_t) il].msa) * plan_ctx_tokens;
    };
    // --- tensor parallelism: the split plan, and what each GPU must hold ---
    if (m->tp.n > 1)
    {
        tsg_mm3_tp::shape ts;
        ts.n_layer = hp.n_layer; ts.n_lead = hp.n_lead; ts.n_ff = hp.n_ff; ts.n_ff_exp = hp.n_ff_exp; ts.n_vocab = hp.n_vocab;
        // Quantizations may differ per layer: the largest block bounds every share.
        auto widest = [&](const char * suffix, ggml_type * type, int64_t * n_ff)
        {
            for (int il = 0; il < hp.n_layer; il++)
            {
                char nm[96];
                snprintf(nm, sizeof(nm), "blk.%d.%s", il, suffix);
                const tensor_source * t = src_of(nm);
                if (!t) continue;
                if (ggml_blck_size(t->type) > ggml_blck_size(*type)) *type = t->type;
                if (n_ff) *n_ff = t->ne[0];   // a layer that differs is caught as unplanned at allocation
            }
        };
        widest("ffn_down.weight", &ts.ffn_down_type, nullptr);
        widest("ffn_down_exps.weight", &ts.exp_down_type, nullptr);
        widest("ffn_down_shexp.weight", &ts.shexp_down_type, &ts.n_ff_shexp);
        std::string err;
        if (!tsg_mm3_tp::make_plan(ts, m->tp.n, &m->tp.ud->p, &err))
        {
            tsg::report_load_refusal("[minimax-m3] %s\n", err.c_str());
            return nullptr;
        }
        m->tp.ud->s = ts;
        // Per GPU: its slices of the FFN and the head plus a whole copy of the
        // rest (the token table stays in the host mapping), the F16 decode
        // copies of the F32 indexer projections (mirrored, like the originals);
        // every GPU holds the whole graph's compute buffers, the sparse chunks'
        // budget and the meta backend's reduction scratch (ceil(log2 n) buffers
        // of the largest partial sum of a ubatch), and every slot's caches whole.
        tsg_mm3_tp::budget b;
        b.n = m->tp.n;
        for (const auto & kv : sources)
        {
            if (kv.first == "token_embd.weight") continue;
            const bool f16_copy = idx_f16_req && kv.second.type == GGML_TYPE_F32 && kv.first.find(".indexer.") != std::string::npos &&
                                  kv.first.find("_proj.weight") != std::string::npos;
            for (int j = 0; j < b.n; j++)
            {
                b.weights[j] += tsg_mm3_tp::device_bytes(*m->tp.ud, kv.first.c_str(), kv.second.type, kv.second.ne, j) + 4096;
                if (f16_copy)
                    b.weights[j] += tsg_mm3_tp::device_bytes(*m->tp.ud, (kv.first + ".f16").c_str(), GGML_TYPE_F16, kv.second.ne, j) + 4096;
            }
        }
        for (int j = 0; j < b.n; j++)
        {
            size_t f = 0, t = 0;
            ggml_backend_dev_memory(m->tp.dev.devs[j], &f, &t);
            b.free[j] = f;
        }
        b.reserve = (size_t) env_long("TS_MINIMAX_M3_VRAM_RESERVE_MB", 2048 + m->n_ubatch / 4) * 1024 * 1024 +
                    (m->n_msa_layers > 0 && m->msa != MSA_DENSE ? m->msa_chunk_bytes : 0) +
                    tsg_mm3_tp::reduction_scratch_bytes(b.n, hp.n_embd, hp.n_expert_used, m->n_ubatch);
        b.dev0_extra = (size_t) env_long("TS_MINIMAX_M3_DEV0_RESERVE_MB", 0) * 1024 * 1024;
        for (int il = 0; il < hp.n_layer; il++) b.kv_per_token += kv_bytes_per_token_layer(hp, m->layers[(size_t) il].msa);
        const int64_t min_ctx = pad_to(4096, m->kv_pad);
        const int32_t advertised = m->n_ctx;
        const int64_t largest = tsg_mm3_tp::max_kv_tokens(b, m->kv_pad, (int64_t) hp.n_ctx_train > 0 ? hp.n_ctx_train : INT32_MAX);
        if (tsg_mm3_tp::misfit(b, (int64_t) plan_ctx_tokens) >= 0 && !ctx_is_hard_limit && !plan_ctx_explicit &&
            m->n_ctx > min_ctx && largest >= min_ctx)
        {
            set_ctx((int32_t) std::min<int64_t>(largest, m->n_ctx));
            fprintf(stderr, "[minimax-m3] context capped to %d tokens (asked for %d) so each GPU's share of the FFN, the "
                    "mirrored attention and the mirrored K/V fit; set MAX_CONTEXT to choose a size\n", m->n_ctx, advertised);
        }
        if (const int j = tsg_mm3_tp::misfit(b, (int64_t) plan_ctx_tokens); j >= 0)
        {
            const double gib = 1073741824.0;
            tsg::report_load_refusal("[minimax-m3] tensor parallel over %d GPUs: GPU %d needs %.1f GiB (weights %.1f, K/V %.1f for "
                                     "%zu tokens, compute and reductions %.1f) but has %.1f GiB free; the largest context that "
                                     "fits is %" PRId64 " tokens: set a smaller MAX_CONTEXT or TS_MINIMAX_M3_PLAN_CTX_TOKENS, a "
                                     "smaller TS_MINIMAX_M3_UBATCH, or use more GPUs\n", b.n, j,
                                     tsg_mm3_tp::need_bytes(b, j, (int64_t) plan_ctx_tokens) / gib, b.weights[j] / gib,
                                     plan_ctx_tokens * b.kv_per_token / gib, plan_ctx_tokens,
                                     (b.reserve + (j == 0 ? b.dev0_extra : 0)) / gib, b.free[j] / gib, largest);
            return nullptr;
        }
        for (int j = 0; j < b.n; j++)
            fprintf(stderr, "[minimax-m3] tensor parallel GPU %d: %.1f GiB weights + %.1f GiB K/V (%zu tokens) + %.1f GiB "
                    "reserve of %.1f GiB free\n", j, b.weights[j] / 1073741824.0, plan_ctx_tokens * b.kv_per_token / 1073741824.0,
                    plan_ctx_tokens, (b.reserve + (j == 0 ? b.dev0_extra : 0)) / 1073741824.0, b.free[j] / 1073741824.0);
    }

    std::vector<size_t> dev_budget((size_t) std::max(n_gpu, 1), 0);
    {
        // Compute buffers grow with the ubatch, and ggml-cuda keeps its scratch
        // pool at the largest size any kernel needed; sparse prompt chunks add
        // their byte budget on top.
        const size_t reserve = (size_t) env_long("TS_MINIMAX_M3_VRAM_RESERVE_MB", 2048 + m->n_ubatch / 4) * 1024 * 1024 +
                               (m->n_msa_layers > 0 && m->msa != MSA_DENSE ? m->msa_chunk_bytes : 0);
        const size_t dev0_extra = (size_t) env_long("TS_MINIMAX_M3_DEV0_RESERVE_MB", 0) * 1024 * 1024;
        for (int d = 0; d < n_gpu; d++)
        {
            size_t free_b = 0, total_b = 0;
            ggml_backend_dev_memory(ggml_backend_get_device(m->backends[d]), &free_b, &total_b);
            size_t r = reserve + (d == 0 ? dev0_extra : 0);
            dev_budget[(size_t) d] = free_b > r ? free_b - r : 0;
        }
    }
    if (m->tp.n > 1) dev_budget[0] = SIZE_MAX / 4;   // checked per GPU above: the placement sees one unbounded device

    // Which routed-expert stacks stay on the host, and which GPU holds each
    // layer. An explicit --n-cpu-moe N pins the leading N layers' experts to
    // the host (llama.cpp's meaning; the dense leading layers have none);
    // automatic placement may leave ANY stack on the host. A DP over
    // contiguous layer ranges per GPU minimizes the host expert bytes (the
    // decode cost): each GPU holds its layers' dense parts, then as many of
    // their gate/up/down stacks as fit.
    const int L = hp.n_layer;
    const bool auto_cpu = n_cpu_moe_req < 0;
    const int forced_prefix = n_gpu == 0 ? L : auto_cpu ? 0 : std::max(0, std::min(n_cpu_moe_req, L));
    std::vector<int> assign((size_t) L, 0);
    std::vector<uint8_t> cpu_stack((size_t) L * 3, 0);   // [layer * 3 + k]
    auto dense_of = [&](int il) -> size_t
    {
        return layer_dense[(size_t) il] + kv_layer_bytes_of(il) + (il == L - 1 ? head_bytes : 0);
    };
    // Host expert bytes when device d holds layers [a, b); -1 if even the
    // dense parts do not fit (or, without `extra`, if the unforced stacks do not).
    // `gpu_used` receives the bytes the range keeps on the device.
    auto range_cost = [&](int d, int a, int b, bool extra, std::vector<uint8_t> * mark,
                          size_t * gpu_used = nullptr) -> int64_t
    {
        const size_t cap = dev_budget[(size_t) d];
        size_t used = 0;
        for (int il = a; il < b; il++) used += dense_of(il);
        if (used > cap) return -1;
        const size_t left = cap - used;
        int64_t host = 0;
        // Free stacks grouped by size; GGUF quantizations give a few distinct sizes.
        std::map<size_t, std::vector<int>, std::greater<size_t>> buckets;
        size_t free_total = 0;
        for (int il = a; il < b; il++)
            for (int k = 0; k < 3; k++)
            {
                const size_t bytes = exp_bytes[(size_t) il][(size_t) k];
                if (bytes == 0) continue;
                if (il < forced_prefix) { host += (int64_t) bytes; if (mark) (*mark)[(size_t) il * 3 + k] = 1; continue; }
                buckets[bytes].push_back(il * 3 + k);
                free_total += bytes;
            }
        if (free_total <= left)
        {
            if (gpu_used) *gpu_used = used + free_total;
            return host;
        }
        if (!extra) return -1;
        // Best fill of `left`: enumerate how many of the largest stacks go to
        // the GPU, then fill greedily with the smaller sizes (exact for two sizes).
        std::vector<std::pair<size_t, std::vector<int>>> bs(buckets.begin(), buckets.end());
        size_t best_gpu = 0;
        int best_first = 0;
        const int n0 = (int) bs[0].second.size();
        for (int c0 = 0; c0 <= n0; c0++)
        {
            size_t g = (size_t) c0 * bs[0].first;
            if (g > left) break;
            for (size_t bi = 1; bi < bs.size(); bi++)
            {
                const size_t fit = std::min<size_t>((left - g) / bs[bi].first, bs[bi].second.size());
                g += fit * bs[bi].first;
            }
            if (g > best_gpu) { best_gpu = g; best_first = c0; }
        }
        host += (int64_t) (free_total - best_gpu);
        if (gpu_used) *gpu_used = used + best_gpu;
        if (mark)
        {
            size_t g = (size_t) best_first * bs[0].first;
            for (size_t bi = 0; bi < bs.size(); bi++)
            {
                const size_t take = bi == 0 ? (size_t) best_first
                                            : std::min<size_t>((left - g) / bs[bi].first, bs[bi].second.size());
                if (bi > 0) g += take * bs[bi].first;
                // the last stacks of each size stay on the host (the range's leading layers keep theirs on GPU)
                const auto & items = bs[bi].second;
                for (size_t i = take; i < items.size(); i++) (*mark)[(size_t) items[i]] = 1;
            }
        }
        return host;
    };
    // Per-token work of layers [a, b): their weights' multiply-adds plus the
    // attention every layer spends alike (at most top_k blocks of keys).
    auto range_work = [&](int a, int b) -> double
    {
        const double attn = (double) hp.n_head * hp.head_dim * 2.0 *
                            (double) std::min<int64_t>(m->n_ctx, std::max<int64_t>(1, hp.dense_span()));
        double w = 0.0;
        for (int il = a; il < b; il++) w += layer_work[(size_t) il] + attn;
        return w;
    };
    // Among plans with the fewest host bytes, take those whose busiest GPU
    // does the least per-token work (a pipelined prefill runs at the pace of
    // its slowest stage; decode visits every GPU in turn and does not care),
    // then the one whose GPUs are filled most evenly (smallest sum of squared
    // fill fractions), so every GPU keeps spare memory for K/V growth. The
    // first pass finds that work bound (fewest host bytes, then the smallest
    // maximum: exact, as both only grow along a prefix), the second minimizes
    // host bytes and the spread (both sums over GPUs) among ranges within it.
    auto place = [&](bool extra) -> int64_t
    {
        const int64_t INF = INT64_MAX / 4;
        double work_cap = INFINITY;
        {
            std::vector<std::vector<int64_t>> bh1((size_t) n_gpu + 1, std::vector<int64_t>((size_t) L + 1, INF));
            std::vector<std::vector<double>> bw1((size_t) n_gpu + 1, std::vector<double>((size_t) L + 1, INFINITY));
            bh1[0][0] = 0;
            bw1[0][0] = 0.0;
            for (int d = 0; d < n_gpu; d++)
                for (int a = 0; a <= L; a++)
                {
                    if (bh1[(size_t) d][(size_t) a] >= INF) continue;
                    for (int b = a; b <= L; b++)
                    {
                        const int64_t c = b == a ? 0 : range_cost(d, a, b, extra, nullptr);
                        if (c < 0) break;
                        const int64_t host = bh1[(size_t) d][(size_t) a] + c;
                        const double w = std::max(bw1[(size_t) d][(size_t) a], range_work(a, b));
                        int64_t & h = bh1[(size_t) d + 1][(size_t) b];
                        double & bw = bw1[(size_t) d + 1][(size_t) b];
                        if (host < h || (host == h && w < bw)) { h = host; bw = w; }
                    }
                }
            if (bh1[(size_t) n_gpu][(size_t) L] >= INF) return -1;
            work_cap = bw1[(size_t) n_gpu][(size_t) L] * (1.0 + 1e-9);
        }
        std::vector<std::vector<int64_t>> best((size_t) n_gpu + 1, std::vector<int64_t>((size_t) L + 1, INF));
        std::vector<std::vector<double>> spread((size_t) n_gpu + 1, std::vector<double>((size_t) L + 1, 0.0));
        std::vector<std::vector<int>> from((size_t) n_gpu + 1, std::vector<int>((size_t) L + 1, -1));
        best[0][0] = 0;
        for (int d = 0; d < n_gpu; d++)
            for (int a = 0; a <= L; a++)
            {
                if (best[(size_t) d][(size_t) a] >= INF) continue;
                for (int b = a; b <= L; b++)
                {
                    size_t used = 0;
                    if (b > a && range_work(a, b) > work_cap) break;   // longer ranges only do more work
                    const int64_t c = b == a ? 0 : range_cost(d, a, b, extra, nullptr, &used);
                    if (c < 0) break;   // longer ranges only need more room
                    const double fill = dev_budget[(size_t) d] > 0 ? (double) used / (double) dev_budget[(size_t) d] : 0.0;
                    const int64_t host = best[(size_t) d][(size_t) a] + c;
                    const double sq = spread[(size_t) d][(size_t) a] + fill * fill;
                    int64_t & bh = best[(size_t) d + 1][(size_t) b];
                    double & bsq = spread[(size_t) d + 1][(size_t) b];
                    if (host < bh || (host == bh && sq < bsq - 1e-12))
                    {
                        bh = host;
                        bsq = sq;
                        from[(size_t) d + 1][(size_t) b] = a;
                    }
                }
            }
        if (best[(size_t) n_gpu][(size_t) L] >= INF) return -1;
        std::fill(cpu_stack.begin(), cpu_stack.end(), 0);
        for (int d = n_gpu, b = L; d > 0; d--)
        {
            const int a = from[(size_t) d][(size_t) b];
            for (int il = a; il < b; il++) assign[(size_t) il] = d - 1;
            if (b > a) range_cost(d - 1, a, b, extra, &cpu_stack);
            b = a;
        }
        return best[(size_t) n_gpu][(size_t) L];
    };

    int n_cpu_moe = L;
    if (n_gpu == 0)
    {
        for (int il = 0; il < L; il++)
            for (int k = 0; k < 3; k++) cpu_stack[(size_t) il * 3 + k] = exp_bytes[(size_t) il][(size_t) k] > 0 ? 1 : 0;
    }
    else
    {
        // Explicit layers per GPU (TS_MINIMAX_M3_LAYER_SPLIT); empty = automatic.
        std::vector<int> counts;
        if (layer_split)
        {
            int sum = 0;
            for (const char * p = layer_split; *p;)
            {
                char * end = nullptr;
                const long v = strtol(p, &end, 10);
                if (end == p || (*end != ',' && *end != '\0') || v < 0 || v > (long) (L - sum))
                {
                    tsg::report_load_refusal("[minimax-m3] layer split '%s' must list non-negative layer counts "
                                             "summing to %d\n", layer_split, L);
                    return nullptr;
                }
                counts.push_back((int) v);
                sum += (int) v;
                p = *end == ',' ? end + 1 : end;
            }
            if ((int) counts.size() > n_gpu || sum != L)
            {
                tsg::report_load_refusal("[minimax-m3] layer split '%s' must list at most %d counts summing to %d\n",
                                         layer_split, n_gpu, L);
                return nullptr;
            }
        }
        // The first GPU whose explicit range does not fit, or -1.
        auto split_misfit = [&](std::vector<uint8_t> * mark) -> int
        {
            for (int d = 0, a = 0; d < (int) counts.size(); a += counts[(size_t) d], d++)
                if (counts[(size_t) d] > 0 && range_cost(d, a, a + counts[(size_t) d], auto_cpu, mark) < 0) return d;
            return -1;
        };
        auto fits = [&]() { return counts.empty() ? place(auto_cpu) >= 0 : split_misfit(nullptr) < 0; };

        // An advertised context that does not fit is capped, not refused (GLM's
        // rule, and the per-op path's); only MAX_CONTEXT makes it a requirement.
        // The plan reserves every row of a full context, so a smaller one leaves
        // room for the expert stacks the placement asked for.
        const int64_t min_ctx = pad_to(4096, m->kv_pad);
        const int32_t advertised = m->n_ctx;
        bool floor_failed = false;   // not even min_ctx fits; the refusal speaks about that size
        if (!ctx_is_hard_limit && !plan_ctx_explicit && m->n_ctx > min_ctx && !fits())
        {
            set_ctx((int32_t) min_ctx);
            if (fits())
            {
                int64_t lo = min_ctx / m->kv_pad, hi = advertised / m->kv_pad;   // lo fits, hi does not
                while (hi - lo > 1)
                {
                    const int64_t mid = lo + (hi - lo) / 2;
                    set_ctx((int32_t) (mid * m->kv_pad));
                    if (fits()) lo = mid; else hi = mid;
                }
                set_ctx((int32_t) (lo * m->kv_pad));
                fprintf(stderr, "[minimax-m3] context capped to %d tokens (asked for %d) so the weights and the K/V "
                        "plan fit the %d GPU(s) with the requested expert placement; set MAX_CONTEXT to choose a size\n",
                        m->n_ctx, advertised, n_gpu);
            }
            else
            {
                floor_failed = true;
            }
        }
        // A smaller MAX_CONTEXT helps only when this one was not already the floor.
        const char * smaller_ctx_hint = floor_failed ? "" : ", or set a smaller MAX_CONTEXT / TS_MINIMAX_M3_PLAN_CTX_TOKENS";

        if (!counts.empty())
        {
            int il = 0;
            for (size_t d = 0; d < counts.size(); d++)
                for (int c = 0; c < counts[d]; c++) assign[(size_t) il++] = (int) d;
            std::fill(cpu_stack.begin(), cpu_stack.end(), 0);
            if (const int d = split_misfit(&cpu_stack); d >= 0)
            {
                tsg::report_load_refusal("[minimax-m3] layer split '%s' does not fit GPU %d at a %d-token context%s\n",
                                         layer_split, d, m->n_ctx, smaller_ctx_hint);
                return nullptr;
            }
        }
        else if (place(auto_cpu) < 0)
        {
            if (place(true) >= 0)
                tsg::report_load_refusal("[minimax-m3] the model does not fit %d GPU(s) at a %d-token context%s with the "
                                         "experts of %d leading layer(s) on the host (--n-cpu-moe); "
                                         "TS_MINIMAX_M3_CPU_MOE=auto places only the expert stacks that must stay on the "
                                         "host%s\n",
                                         n_gpu, m->n_ctx, floor_failed ? " (the smallest it tries)" : "",
                                         forced_prefix, smaller_ctx_hint);
            else
                tsg::report_load_refusal("[minimax-m3] the model does not fit %d GPU(s) at a %d-token context even with "
                                         "every expert on the host%s\n", n_gpu, m->n_ctx, smaller_ctx_hint);
            return nullptr;
        }
    }
    n_cpu_moe = 0;
    for (int il = 0; il < L; il++)
    {
        layer & ly = m->layers[(size_t) il];
        ly.device = n_gpu > 0 ? assign[(size_t) il] : 0;
        for (int k = 0; k < 3; k++) ly.cpu_stack[k] = cpu_stack[(size_t) il * 3 + k] != 0;
        ly.cpu_moe = ly.cpu_stack[0] || ly.cpu_stack[1] || ly.cpu_stack[2];
        n_cpu_moe += ly.cpu_moe ? 1 : 0;
    }
    m->n_cpu_moe = n_cpu_moe;
    auto layer_gpu_bytes = [&](int il) -> size_t
    {
        size_t b = layer_dense[(size_t) il] + kv_layer_bytes_of(il);
        for (int k = 0; k < 3; k++)
            if (!m->layers[(size_t) il].cpu_stack[k]) b += exp_bytes[(size_t) il][(size_t) k];
        return b;
    };
    m->embd_device = m->host_dev();   // token rows are gathered on the host, as llama.cpp does
    m->out_device = n_gpu > 0 ? m->layers[(size_t) hp.n_layer - 1].device : 0;

    {
        std::vector<size_t> used((size_t) std::max(n_gpu, 1), 0);
        std::vector<int> nl((size_t) std::max(n_gpu, 1), 0);
        for (int il = 0; il < hp.n_layer && n_gpu > 0; il++)
        {
            used[(size_t) m->layers[(size_t) il].device] += layer_gpu_bytes(il);
            nl[(size_t) m->layers[(size_t) il].device]++;
        }
        if (n_gpu > 0) used[(size_t) m->out_device] += head_bytes;
        for (int d = 0; d < n_gpu && m->tp.n <= 1; d++)
            fprintf(stderr, "[minimax-m3] GPU %d: %d layers, %.1f GiB weights+KV of %.1f GiB budget\n",
                    d, nl[(size_t) d], used[(size_t) d] / 1073741824.0, dev_budget[(size_t) d] / 1073741824.0);
        if (n_cpu_moe > 0)
        {
            size_t hb = 0;
            std::string which;
            for (int il = 0; il < L; il++)
            {
                const layer & ly = m->layers[(size_t) il];
                if (!ly.cpu_moe) continue;
                std::string part;
                for (int k = 0; k < 3; k++)
                    if (ly.cpu_stack[k]) { hb += exp_bytes[(size_t) il][(size_t) k]; part += "gud"[k]; }
                which += (which.empty() ? "" : ",") + std::to_string(il) + (part == "gud" ? "" : ":" + part);
            }
            fprintf(stderr, "[minimax-m3] routed experts on the host (%.1f GiB, %d layers, %d threads): %s\n",
                    hb / 1073741824.0, n_cpu_moe, m->cpu_threads, which.c_str());
        }
    }

    // --- weight tensors ---------------------------------------------------
    for (int d = 0; d <= n_gpu; d++)
    {
        ggml_init_params p = { (size_t) (hp.n_layer * 40 + 64) * ggml_tensor_overhead(), nullptr, true };
        m->w_ctx[d] = ggml_init(p);
    }
    std::vector<pending_weight> pending;
    bool missing = false;
    auto make_in = [&](ggml_context * c, const char * name, bool required) -> ggml_tensor *
    {
        const tensor_source * s = src_of(name);
        if (!s) { if (required) { fprintf(stderr, "[minimax-m3] missing tensor %s\n", name); missing = true; } return nullptr; }
        ggml_tensor * t = ggml_new_tensor_4d(c, s->type, s->ne[0], s->ne[1], s->ne[2], s->ne[3]);
        ggml_set_name(t, name);
        pending.push_back({ t, s->shard, s->offset, 0, s->size, false });
        return t;
    };
    auto make = [&](int dev, const char * name, bool required) -> ggml_tensor *
    {
        return make_in(m->w_ctx[dev], name, required);
    };

    // Host expert stacks whose type the CPU backend can repack into its
    // interleaved GEMV/GEMM layout are copied into a CPU_REPACK buffer instead
    // of being served from the mapping (TS_MINIMAX_M3_CPU_REPACK=1).
    // NUMA placement (default on a multi-node host) copies every host expert
    // stack into memory placed beside the worker that reads it, and pins the
    // CPU workers. It supersedes the mapping and the repacked layout.
    bool host_numa = false;
#if defined(__linux__)
    if (n_gpu > 0 && n_cpu_moe > 0 && env_flag("TS_MINIMAX_M3_HOST_NUMA", true))
    {
        const auto nodes = numa_node_cpus();
        if (nodes.size() >= 2)
        {
            // m->cpu_threads already leaves the serving process half the quota (see the CPU backend above).
            const int nth = m->cpu_threads;
            const std::vector<int> cpus = numa_worker_cpus(nodes, nth);
            ggml_threadpool_params tpp = ggml_threadpool_params_default(nth);
            for (int c : cpus) if (c >= 0 && c < GGML_MAX_N_THREADS) tpp.cpumask[c] = true;
            tpp.strict_cpu = true;
            ggml_threadpool_t pool = ggml_threadpool_new(&tpp);
            if (pool)
            {
                ggml_backend_t cpu_be = m->backends[n_gpu];
                ggml_backend_cpu_set_n_threads(cpu_be, nth);
                ggml_backend_cpu_set_threadpool(cpu_be, pool);
                if (m->cpu_threadpool) ggml_threadpool_free(m->cpu_threadpool);
                m->cpu_threadpool = pool;
                m->cpu_threads = nth;
                m->worker_cpu = probe_worker_cpus(cpu_be, nth);
                host_numa = std::find(m->worker_cpu.begin(), m->worker_cpu.end(), -1) == m->worker_cpu.end();
                fprintf(stderr, "[minimax-m3] host experts: %d pinned workers over %zu NUMA nodes%s\n", nth, nodes.size(),
                        host_numa ? "" : " (worker CPUs unknown; placement off)");
            }
        }
    }
#endif
    if (host_numa)
    {
        ggml_init_params p = { (size_t) (hp.n_layer * 4 + 8) * ggml_tensor_overhead(), nullptr, true };
        m->numa_ctx = ggml_init(p);
    }

    ggml_backend_buffer_type_t repack_buft = nullptr;
    ggml_context * repack_ctx = nullptr;
    if (n_gpu > 0 && n_cpu_moe > 0 && !host_numa && env_flag("TS_MINIMAX_M3_CPU_REPACK", false))
    {
        ggml_backend_dev_t cpu_dev = ggml_backend_get_device(m->backends[n_gpu]);
        ggml_backend_reg_t cpu_reg = cpu_dev ? ggml_backend_dev_backend_reg(cpu_dev) : nullptr;
        auto get_extra = cpu_reg ? (ggml_backend_dev_get_extra_bufts_t)
            ggml_backend_reg_get_proc_address(cpu_reg, "ggml_backend_dev_get_extra_bufts") : nullptr;
        if (get_extra)
            for (ggml_backend_buffer_type_t * b = get_extra(cpu_dev); b && *b; ++b)
                if (strcmp(ggml_backend_buft_name(*b), "CPU_REPACK") == 0) repack_buft = *b;
        if (repack_buft)
        {
            ggml_init_params p = { (size_t) (hp.n_layer * 4 + 8) * ggml_tensor_overhead(), nullptr, true };
            repack_ctx = ggml_init(p);
        }
    }
    auto repackable = [&](const tensor_source * s) -> bool
    {
        if (!repack_buft || !s || s->ne[2] <= 1) return false;
        ggml_init_params p = { 8 * ggml_tensor_overhead(), nullptr, true };
        ggml_context * c = ggml_init(p);
        ggml_tensor * w = ggml_new_tensor_3d(c, s->type, s->ne[0], s->ne[1], s->ne[2]);
        ggml_tensor * x = ggml_new_tensor_3d(c, GGML_TYPE_F32, s->ne[0], 1, 1);
        ggml_tensor * ids = ggml_new_tensor_2d(c, GGML_TYPE_I32, 1, 1);
        ggml_tensor * op = ggml_mul_mat_id(c, w, x, ids);
        ggml_backend_buffer_t dummy = ggml_backend_buft_alloc_buffer(repack_buft, 0);
        w->buffer = dummy;
        const bool ok = dummy && ggml_backend_dev_supports_op(ggml_backend_get_device(m->backends[n_gpu]), op);
        w->buffer = nullptr;
        if (dummy) ggml_backend_buffer_free(dummy);
        ggml_free(c);
        return ok;
    };
    std::map<ggml_type, bool> repack_by_type;
    auto make_expert = [&](int il, int de, const char * suffix) -> ggml_tensor *
    {
        char name[256];
        snprintf(name, sizeof(name), "blk.%d.%s", il, suffix);
        const tensor_source * s = src_of(name);
        if (de == m->host_dev() && m->numa_ctx && s && s->ne[2] > 1) return make_in(m->numa_ctx, name, true);
        if (de == m->host_dev() && repack_ctx && s)
        {
            auto it = repack_by_type.find(s->type);
            if (it == repack_by_type.end()) it = repack_by_type.emplace(s->type, repackable(s)).first;
            if (it->second) return make_in(repack_ctx, name, true);
        }
        return make(de, name, true);
    };
    char nb[256];
    auto lname = [&](int il, const char * suffix) -> const char * { snprintf(nb, sizeof(nb), "blk.%d.%s", il, suffix); return nb; };

    // Concatenate same-typed projections of one input into one matrix (rows
    // along ne1): one weight read and one launch. Returns null when they do
    // not share a type and shape, leaving the caller to make them separately.
    // `f16` converts F32 sources to F16 on the way (TS_MINIMAX_M3_INDEXER_F16).
    auto make_fused = [&](int d, const std::vector<const tensor_source *> & parts, const char * name, bool f16) -> ggml_tensor *
    {
        for (const tensor_source * s : parts)
            if (!s || s->type != parts[0]->type || s->ne[0] != hp.n_embd || s->ne[2] != 1 || s->ne[3] != 1) return nullptr;
        int64_t rows = 0;
        for (const tensor_source * s : parts) rows += s->ne[1];
        const bool conv = f16 && parts[0]->type == GGML_TYPE_F32;
        ggml_tensor * t = ggml_new_tensor_2d(m->w_ctx[d], conv ? GGML_TYPE_F16 : parts[0]->type, hp.n_embd, rows);
        ggml_set_name(t, name);
        size_t off = 0;
        for (const tensor_source * s : parts)
        {
            pending.push_back({ t, s->shard, s->offset, off, s->size, conv });
            off += conv ? s->size / 2 : s->size;
        }
        return t;
    };
    // One indexer projection, or (`f16`) its F16 copy for decode-shaped graphs.
    auto make_index_proj = [&](int d, const char * name, bool f16) -> ggml_tensor *
    {
        const tensor_source * s = src_of(name);
        if (!s) { fprintf(stderr, "[minimax-m3] missing tensor %s\n", name); missing = true; return nullptr; }
        if (!f16 || s->type != GGML_TYPE_F32) return make(d, name, true);
        ggml_tensor * t = ggml_new_tensor_4d(m->w_ctx[d], GGML_TYPE_F16, s->ne[0], s->ne[1], s->ne[2], s->ne[3]);
        ggml_format_name(t, "%s.f16", name);
        pending.push_back({ t, s->shard, s->offset, 0, s->size, true });
        return t;
    };

    const bool want_fuse = env_flag("TS_MINIMAX_M3_FUSE_QKV", true);
    // The indexer projections ship as F32 but hold BF16 values (all but 7 of
    // MiniMax-M3.1-Flash's 236M are exact in F16), so an F16 copy halves the
    // 0.94 GB a decode step reads for them at no loss in the weights; decode-
    // shaped graphs read it (IDX_F16_MAX_ROWS), prompts the F32 original, and
    // the products accumulate in F32 either way (project_index). The CPU
    // backend would round the activations to F16 for an F16 matrix, so a
    // CPU-only run keeps F32 alone unless asked; TS_MINIMAX_M3_INDEXER_F16=0
    // keeps F32 alone on the GPUs too.
    m->idx_f16 = idx_f16_req;
    m->tok_embd = make(m->embd_device, "token_embd.weight", true);
    m->output_norm = make(m->out_device, "output_norm.weight", true);
    m->output = make(m->out_device, "output.weight", true);
    dspark_make_weights(*m, make);
    int n_fused = 0, n_fused_idx = 0, n_idx_f32 = 0;
    for (int il = 0; il < hp.n_layer; il++)
    {
        layer & Ly = m->layers[(size_t) il];
        const int d = Ly.device;
        Ly.attn_norm = make(d, lname(il, "attn_norm.weight"), true);
        Ly.q_norm = make(d, lname(il, "attn_q_norm.weight"), true);
        Ly.k_norm = make(d, lname(il, "attn_k_norm.weight"), true);
        Ly.wo = make(d, lname(il, "attn_output.weight"), true);
        Ly.ffn_norm = make(d, lname(il, "ffn_norm.weight"), true);

        // Q/K/V as one matrix: one weight read and one launch per layer.
        {
            std::string nq = lname(il, "attn_q.weight"), nk = lname(il, "attn_k.weight"), nv = lname(il, "attn_v.weight");
            const tensor_source *sq = src_of(nq.c_str()), *sk = src_of(nk.c_str()), *sv = src_of(nv.c_str());
            if (!sq || !sk || !sv) { fprintf(stderr, "[minimax-m3] layer %d attention projections missing\n", il); missing = true; continue; }
            char fname[96];
            snprintf(fname, sizeof(fname), "blk.%d.attn_qkv.weight", il);
            Ly.wqkv = want_fuse ? make_fused(d, { sq, sk, sv }, fname, false) : nullptr;
            if (Ly.wqkv) n_fused++;
            else
            {
                Ly.wq = make(d, nq.c_str(), true);
                Ly.wk = make(d, nk.c_str(), true);
                Ly.wv = make(d, nv.c_str(), true);
            }
        }
        if (Ly.msa)
        {
            Ly.idx_q_norm = make(d, lname(il, "indexer.q_norm.weight"), true);
            Ly.idx_k_norm = make(d, lname(il, "indexer.k_norm.weight"), true);
            std::string nq = lname(il, "indexer.q_proj.weight"), nk = lname(il, "indexer.k_proj.weight");
            const tensor_source *sq = src_of(nq.c_str()), *sk = src_of(nk.c_str());
            if (sq && sq->type == GGML_TYPE_F32) n_idx_f32++;
            char fname[96];
            snprintf(fname, sizeof(fname), "blk.%d.indexer.qk_proj.weight", il);
            const bool dec16 = m->idx_f16 && sq && sq->type == GGML_TYPE_F32;
            Ly.idx_wqk = want_fuse ? make_fused(d, { sq, sk }, fname, false) : nullptr;
            if (Ly.idx_wqk)
            {
                n_fused_idx++;
                snprintf(fname, sizeof(fname), "blk.%d.indexer.qk_proj.f16", il);
                if (dec16) Ly.idx_wqk_dec = make_fused(d, { sq, sk }, fname, true);
            }
            else
            {
                Ly.idx_wq = make_index_proj(d, nq.c_str(), false);
                Ly.idx_wk = make_index_proj(d, nk.c_str(), false);
                if (dec16) Ly.idx_wq_dec = make_index_proj(d, nq.c_str(), true);
                if (dec16 && sk && sk->type == GGML_TYPE_F32) Ly.idx_wk_dec = make_index_proj(d, nk.c_str(), true);
            }
        }
        if (Ly.dense)
        {
            Ly.ffn_gate = make(d, lname(il, "ffn_gate.weight"), true);
            Ly.ffn_up = make(d, lname(il, "ffn_up.weight"), true);
            Ly.ffn_down = make(d, lname(il, "ffn_down.weight"), true);
        }
        else
        {
            Ly.gate_inp = make(d, lname(il, "ffn_gate_inp.weight"), true);
            Ly.exp_probs_b = make(d, lname(il, "exp_probs_b.bias"), false);
            Ly.gate_exps = make_expert(il, Ly.cpu_stack[0] ? m->host_dev() : d, "ffn_gate_exps.weight");
            Ly.up_exps = make_expert(il, Ly.cpu_stack[1] ? m->host_dev() : d, "ffn_up_exps.weight");
            Ly.down_exps = make_expert(il, Ly.cpu_stack[2] ? m->host_dev() : d, "ffn_down_exps.weight");
            Ly.gate_s = make(d, lname(il, "ffn_gate_exps.scale"), false);
            Ly.up_s = make(d, lname(il, "ffn_up_exps.scale"), false);
            Ly.down_s = make(d, lname(il, "ffn_down_exps.scale"), false);
            Ly.gate_shexp = make(d, lname(il, "ffn_gate_shexp.weight"), true);
            Ly.up_shexp = make(d, lname(il, "ffn_up_shexp.weight"), true);
            Ly.down_shexp = make(d, lname(il, "ffn_down_shexp.weight"), true);
            for (ggml_tensor * s : { Ly.gate_s, Ly.up_s, Ly.down_s })
                if (s && (s->type != GGML_TYPE_F32 || ggml_nelements(s) != hp.n_expert))
                {
                    tsg::report_load_refusal("[minimax-m3] %s must be F32 [%d] (one output scale per expert)\n", s->name, hp.n_expert);
                    return nullptr;
                }
        }
    }
    if (missing) { tsg::report_load_refusal("[minimax-m3] required tensors are missing\n"); return nullptr; }
    if (m->tp.n > 1 && m->n_msa_layers > 0)
    {
        m->tp.arange_rows = ggml_new_tensor_1d(m->w_ctx[0], GGML_TYPE_F32, hp.idx_blk);
        m->tp.arange_heads = ggml_new_tensor_1d(m->w_ctx[0], GGML_TYPE_F32, hp.n_head_kv);
        ggml_set_name(m->tp.arange_rows, tsg_mm3_tp::ARANGE_ROWS);
        ggml_set_name(m->tp.arange_heads, tsg_mm3_tp::ARANGE_HEADS);
    }
    m->fused_qkv = n_fused == hp.n_layer;
    m->fused_idx = n_fused_idx == m->n_msa_layers && m->n_msa_layers > 0;
    if (m->idx_f16 && n_idx_f32 < m->n_msa_layers)
        fprintf(stderr, "[minimax-m3] the F16 indexer (TS_MINIMAX_M3_INDEXER_F16) converts F32 indexer projections only; %d of %d layers "
                "store another type and keep it\n", m->n_msa_layers - n_idx_f32, m->n_msa_layers);

    // --- host tensors straight from the GGUF mapping ----------------------
    std::map<const ggml_tensor *, std::vector<const pending_weight *>> by_tensor;
    for (const pending_weight & pw : pending) by_tensor[pw.t].push_back(&pw);
    std::vector<tsg_dsv4::file_warm_range> warm;
    {
        ggml_context * hctx = m->w_ctx[m->host_dev()];
        for (ggml_tensor * t = ggml_get_first_tensor(hctx); t; t = ggml_get_next_tensor(hctx, t))
        {
            const auto & pws = by_tensor[t];
            if (pws.size() != 1 || pws[0]->to_f16) continue;
            const pending_weight & pw = *pws[0];
            ggml_backend_buffer_t buf = mmap_shard(*m, pw.shard);
            if (!buf) continue;
            char * base = (char *) m->mmap_addrs[(size_t) pw.shard];
            if (ggml_backend_tensor_alloc(buf, t, base + pw.file_off) != GGML_STATUS_SUCCESS) continue;
            warm.push_back({ pw.shard, pw.file_off, ggml_nbytes(t), t->data });
        }
        for (const auto & r : tsg_dsv4::merge_file_warm_ranges(warm)) m->mmap_weight_bytes += r.bytes;
    }

    for (int d = 0; d <= n_gpu; d++)
    {
        if (!ggml_get_first_tensor(m->w_ctx[d])) continue;
        bool unallocated = false;
        for (ggml_tensor * t = ggml_get_first_tensor(m->w_ctx[d]); t; t = ggml_get_next_tensor(m->w_ctx[d], t))
            if (!t->data) { unallocated = true; break; }
        if (!unallocated) continue;
        m->w_buf[d] = ggml_backend_alloc_ctx_tensors(m->w_ctx[d], m->backends[d]);
        if (!m->w_buf[d]) { tsg::report_load_refusal("[minimax-m3] weight allocation failed on device %d\n", d); return nullptr; }
        ggml_backend_buffer_set_usage(m->w_buf[d], GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }
    if (m->tp.n > 1 && m->tp.ud->unplanned.load())
    {
        tsg::report_load_refusal("[minimax-m3] internal: tensor %s has no tensor-parallel split\n", m->tp.ud->unplanned_name.c_str());
        return nullptr;
    }
    for (ggml_tensor * t : { m->tp.arange_rows, m->tp.arange_heads })
        if (t)
        {
            std::vector<float> v((size_t) t->ne[0]);
            std::iota(v.begin(), v.end(), 0.0f);
            ggml_backend_tensor_set(t, v.data(), 0, ggml_nbytes(t));
        }
    std::vector<const pending_weight *> numa_jobs;
    if (m->numa_ctx && ggml_get_first_tensor(m->numa_ctx))
    {
        // Untouched pages: each lands on the node of the helper that writes it first.
        m->numa_buf = ggml_backend_alloc_ctx_tensors(m->numa_ctx, m->backends[n_gpu]);
        if (!m->numa_buf)
        {
            tsg::report_load_refusal("[minimax-m3] host memory for the expert stacks is unavailable; "
                                     "TS_MINIMAX_M3_HOST_NUMA=0 serves them from the GGUF mapping\n");
            return nullptr;
        }
        ggml_backend_buffer_set_usage(m->numa_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        for (const pending_weight & pw : pending)
            if (pw.t->buffer == m->numa_buf) { numa_jobs.push_back(&pw); m->numa_bytes += pw.bytes; }
    }
    std::vector<const pending_weight *> repack_jobs;
    if (repack_ctx && ggml_get_first_tensor(repack_ctx))
    {
        m->repack_ctx = repack_ctx;
        m->repack_buf = ggml_backend_alloc_ctx_tensors_from_buft(repack_ctx, repack_buft);
        if (!m->repack_buf)
        {
            tsg::report_load_refusal("[minimax-m3] host memory for the repacked expert stacks is unavailable; "
                                     "set TS_MINIMAX_M3_CPU_REPACK=0 to serve them from the mapping\n");
            return nullptr;
        }
        ggml_backend_buffer_set_usage(m->repack_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        for (const pending_weight & pw : pending)
            if (pw.t->buffer == m->repack_buf) { repack_jobs.push_back(&pw); m->repack_bytes += pw.bytes; }
    }
    else if (repack_ctx)
    {
        ggml_free(repack_ctx);
    }

    // --- upload -----------------------------------------------------------
    {
        const size_t chunk = (size_t) std::max(1L, env_long("TS_MINIMAX_M3_LOAD_CHUNK_MB", 64)) * 1024 * 1024;
        int load_threads = (int) std::max(1L, env_long("TS_MINIMAX_M3_LOAD_THREADS", 16));
        std::vector<load_job> jobs;
        size_t total = 0;
        for (const pending_weight & pw : pending)
        {
            if (pw.t->buffer && std::find(m->mmap_bufs.begin(), m->mmap_bufs.end(), pw.t->buffer) != m->mmap_bufs.end())
                continue;
            if (m->repack_buf && pw.t->buffer == m->repack_buf) continue;   // repacked below
            if (m->numa_buf && pw.t->buffer == m->numa_buf) continue;       // placed below
            // A converted chunk is whole F32 elements; it lands at half the offset.
            // A split tensor (--tp) is written in whole slabs along its split axis.
            const size_t step = m->tp.n > 1 && pw.t->buffer == m->w_buf[0] ? tsg_mm3_tp::upload_step(pw.t, *m->tp.ud, chunk) : chunk;
            for (size_t off = 0; off < pw.bytes; off += step)
                jobs.push_back({ pw.t, pw.shard, pw.file_off + off, pw.tensor_off + (pw.to_f16 ? off / 2 : off),
                                 std::min(step, pw.bytes - off), pw.to_f16 });
            total += pw.bytes;
        }
        std::atomic<size_t> done(0);
        std::atomic<bool> finished(false);
        // The progress line's thread is stopped and joined on every path out of
        // this block (a joinable std::thread destroyed by an exception terminates).
        std::vector<std::thread> progress;
        struct stop_progress
        {
            std::atomic<bool> & finished;
            std::vector<std::thread> & threads;
            ~stop_progress() { finished.store(true); for (auto & t : threads) if (t.joinable()) t.join(); }
        } stop_progress_guard{ finished, progress };
        try
        {
            start_thread(progress, [&]()
            {
                auto t0 = std::chrono::steady_clock::now();
                while (!finished.load())
                {
                    for (int i = 0; i < 100 && !finished.load(); i++) std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    if (finished.load()) break;
                    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                    fprintf(stderr, "[minimax-m3] uploading weights: %.1f / %.1f GiB (%.2f GiB/s)\n",
                            done.load() / 1073741824.0, total / 1073741824.0, done.load() / 1073741824.0 / std::max(s, 1e-3));
                }
            });
        }
        catch (const std::exception & e)
        {
            fprintf(stderr, "[minimax-m3] cannot start the upload progress thread (%s); the weights upload without "
                    "progress lines\n", e.what());
        }
        std::mutex meta_set_mu;
        std::string upload_error;
        const bool ok = upload_parallel(m->shards, jobs, load_threads, done, &upload_error, m->tp.n > 1 ? &meta_set_mu : nullptr);
        finished.store(true);
        for (auto & t : progress) if (t.joinable()) t.join();
        if (!ok)
        {
            tsg::report_load_refusal("[minimax-m3] weight upload failed: %s%s\n", upload_error.c_str(),
                                     upload_error.rfind("cannot start", 0) == 0 ? " (set TS_MINIMAX_M3_LOAD_THREADS lower)" : "");
            return nullptr;
        }

#if defined(__linux__)
        if (!numa_jobs.empty())
        {
            // In batches: bring a batch's file ranges into the page cache with
            // parallel sequential reads, let one helper per worker (pinned to
            // that worker's CPU) copy exactly the rows the worker reads, then
            // drop the batch's file pages.
            const auto t_numa = std::chrono::steady_clock::now();
            const int nth = (int) m->worker_cpu.size();
            const size_t batch_cap = (size_t) std::max(1L, env_long("TS_MINIMAX_M3_NUMA_BATCH_GB", 16)) << 30;
            size_t first = 0;
            while (first < numa_jobs.size())
            {
                size_t last = first, bytes = 0;
                std::vector<tsg_dsv4::file_warm_range> ranges;
                while (last < numa_jobs.size() && (last == first || bytes + numa_jobs[last]->bytes <= batch_cap))
                {
                    const pending_weight * pw = numa_jobs[last];
                    if (!mmap_shard(*m, pw->shard)) { tsg::report_load_refusal("[minimax-m3] cannot map shard %d\n", pw->shard); return nullptr; }
                    ranges.push_back({ pw->shard, pw->file_off, pw->bytes, (char *) m->mmap_addrs[(size_t) pw->shard] + pw->file_off });
                    bytes += pw->bytes;
                    last++;
                }
                tsg_dsv4::file_warm_options options;
                options.threads = load_threads;
                const auto warmed = tsg_dsv4::warm_file_ranges(m->shards.paths, ranges, options);
                if (!warmed.ok) { tsg::report_load_refusal("[minimax-m3] reading expert stacks failed: %s\n", warmed.error.c_str()); return nullptr; }
                // Every helper's rows must land: a helper that cannot start is a refusal.
                team_status st;
                run_team(nth, "NUMA placement helper", st, [&](int c)
                {
                    cpu_set_t set;
                    CPU_ZERO(&set);
                    CPU_SET(m->worker_cpu[(size_t) c], &set);
                    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
                    for (size_t j = first; j < last; j++)
                    {
                        const pending_weight * pw = numa_jobs[j];
                        ggml_tensor * t = pw->t;
                        const char * src = (const char *) m->mmap_addrs[(size_t) pw->shard] + pw->file_off;
                        const int64_t nr0 = t->ne[1];
                        const int64_t dr0 = (nr0 + nth - 1) / nth;
                        const int64_t r0 = c * dr0, r1 = std::min<int64_t>(nr0, r0 + dr0);
                        if (r0 >= r1) continue;
                        for (int64_t e = 0; e < t->ne[2] * t->ne[3]; e++)
                        {
                            const size_t o = (size_t) e * t->nb[2] + (size_t) r0 * t->nb[1];
                            memcpy((char *) t->data + o, src + o, (size_t) (r1 - r0) * t->nb[1]);
                        }
                    }
                });
                if (st.failed.load())
                {
                    tsg::report_load_refusal("[minimax-m3] placing host expert stacks failed: %s; TS_MINIMAX_M3_HOST_NUMA=0 "
                                             "serves them from the GGUF mapping\n", st.error.c_str());
                    return nullptr;
                }
                for (size_t j = first; j < last; j++)
                {
                    const pending_weight * pw = numa_jobs[j];
                    const char * src = (const char *) m->mmap_addrs[(size_t) pw->shard] + pw->file_off;
                    posix_madvise((void *) ((uintptr_t) src & ~(uintptr_t) 4095), pw->bytes + ((uintptr_t) src & 4095), POSIX_MADV_DONTNEED);
                    if (int fd = open(m->shards.paths[(size_t) pw->shard].c_str(), O_RDONLY); fd >= 0)
                    {
                        posix_fadvise(fd, (off_t) pw->file_off, (off_t) pw->bytes, POSIX_FADV_DONTNEED);
                        close(fd);
                    }
                }
                first = last;
            }
            fprintf(stderr, "[minimax-m3] placed %zu host expert stacks (%.1f GiB) beside their %d pinned workers in %.1fs\n",
                    numa_jobs.size(), m->numa_bytes / 1073741824.0, nth,
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - t_numa).count());
        }
#endif

        if (!repack_jobs.empty())
        {
            // Bring the sources into the page cache with parallel sequential
            // reads (faulting a network mapping page by page is far slower),
            // repack each stack from the mapping, then drop the source pages:
            // the repacked copy is the only one the executor reads.
            const auto t_repack = std::chrono::steady_clock::now();
            std::vector<tsg_dsv4::file_warm_range> src_ranges;
            for (const pending_weight * pw : repack_jobs)
            {
                if (!mmap_shard(*m, pw->shard)) { tsg::report_load_refusal("[minimax-m3] cannot map shard %d\n", pw->shard); return nullptr; }
                src_ranges.push_back({ pw->shard, pw->file_off, pw->bytes, (char *) m->mmap_addrs[(size_t) pw->shard] + pw->file_off });
            }
            tsg_dsv4::file_warm_options options;
            options.threads = load_threads;
            const auto warmed = tsg_dsv4::warm_file_ranges(m->shards.paths, src_ranges, options);
            if (!warmed.ok) { tsg::report_load_refusal("[minimax-m3] reading expert stacks failed: %s\n", warmed.error.c_str()); return nullptr; }
            std::atomic<size_t> next(0);
            team_status st;
            auto worker = [&](int)
            {
                for (size_t i; !st.failed.load() && (i = next.fetch_add(1)) < repack_jobs.size(); )
                {
                    const pending_weight * pw = repack_jobs[i];
                    const char * src = (const char *) m->mmap_addrs[(size_t) pw->shard] + pw->file_off;
                    ggml_backend_tensor_set(pw->t, src, 0, pw->bytes);
#if !defined(_WIN32)
                    posix_madvise((void *) ((uintptr_t) src & ~(uintptr_t) 4095), pw->bytes + ((uintptr_t) src & 4095), POSIX_MADV_DONTNEED);
#endif
#if defined(__linux__)
                    // macOS has no ranged page-cache drop (no posix_fadvise); the
                    // madvise above only deactivates the mapped pages there, so
                    // they are the first reclaimed rather than dropped now.
                    if (int fd = open(m->shards.paths[(size_t) pw->shard].c_str(), O_RDONLY); fd >= 0)
                    {
                        posix_fadvise(fd, (off_t) pw->file_off, (off_t) pw->bytes, POSIX_FADV_DONTNEED);
                        close(fd);
                    }
#endif
                }
            };
            const int nt = std::min<int>(load_threads, (int) repack_jobs.size());
            if (!run_team(nt, "repack thread", st, worker))
            {
                tsg::report_load_refusal("[minimax-m3] repacking host expert stacks failed: %s; set TS_MINIMAX_M3_CPU_REPACK=0 "
                                         "to serve them from the mapping\n", st.error.c_str());
                return nullptr;
            }
            fprintf(stderr, "[minimax-m3] repacked %zu host expert stacks (%.1f GiB) for the CPU kernels in %.1fs\n",
                    repack_jobs.size(), m->repack_bytes / 1073741824.0,
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - t_repack).count());
        }

        if (m->mmap_weight_bytes > 0 && env_flag("TS_MINIMAX_M3_WARM_HOST", true))
        {
            const size_t allow = host_mem_allowance();
            const size_t headroom = (size_t) 8 * 1024 * 1024 * 1024;
            if (allow == 0 || m->mmap_weight_bytes + headroom <= allow)
            {
                const auto t_warm = std::chrono::steady_clock::now();
                tsg_dsv4::file_warm_options options;
                options.threads = load_threads;
                options.populate = true;
                const auto result = tsg_dsv4::warm_file_ranges(m->shards.paths, warm, options);
                if (!result.ok)
                {
                    tsg::report_load_refusal("[minimax-m3] prefaulting host experts failed: %s\n", result.error.c_str());
                    return nullptr;
                }
                fprintf(stderr, "[minimax-m3] prefaulted %.1f GiB of host weights in %.1fs (%.1f GiB read)\n",
                        m->mmap_weight_bytes / 1073741824.0,
                        std::chrono::duration<double>(std::chrono::steady_clock::now() - t_warm).count(),
                        result.bytes_read / 1073741824.0);
            }
            else
            {
                fprintf(stderr, "[minimax-m3] skipping host prefault: %.1f GiB mapped exceeds the %.1f GiB allowance\n",
                        m->mmap_weight_bytes / 1073741824.0, allow / 1073741824.0);
            }
        }
    }

    // Offloading a host expert stack to the GPU for a prefill-sized batch copies
    // every expert of the layer over the bus and needs a device buffer for them.
    m->op_offload = env_flag("TS_MINIMAX_M3_OP_OFFLOAD", n_cpu_moe == 0);
    m->host_moe_max_tokens = std::max(0L, env_long("TS_MINIMAX_M3_HOST_MOE_TOKENS", MAX_BATCHED_DECODE));

    // Sparse segments skip the causal mask and the block force when no layer
    // reads them: every layer is sparse and runs the kernels.
    m->sparse_maskless = m->n_msa_layers == hp.n_layer && hp.n_layer > 0;
    for (const layer & L : m->layers) m->sparse_maskless = m->sparse_maskless && m->msa_kernel_dev[L.device];
    if (!dspark_finish_load(*m)) return nullptr;

    // --- the first sequence slot -----------------------------------------
    slot * s0 = slot_alloc(*m, false);
    if (!s0) { tsg::report_load_refusal("[minimax-m3] K/V cache for %d tokens does not fit\n", m->n_ctx); return nullptr; }
    m->active_slot = s0;

    // --- grow the GPU kernel-scratch pools to their prefill peak ----------
    // ggml-cuda keeps a per-device pool for kernel scratch (quantized
    // activations, expert-sorted rows) outside every graph's compute buffer.
    // It grows on first use and never shrinks. Grown later, after sequence
    // slots and retained prefix state took the memory, the growth fails
    // inside a kernel and ggml aborts the process. One full-size prefill now
    // reserves the peak while memory is free, so admission sees what is
    // really left.
    if (n_gpu > 0 && env_flag("TS_MINIMAX_M3_WARM_POOLS", true))
    {
        const int64_t nt = std::min<int64_t>(m->n_ubatch, m->n_ctx);
        std::vector<int32_t> toks((size_t) nt);
        for (int64_t i = 0; i < nt; i++) toks[(size_t) i] = (int32_t) (1 + i % std::max<int32_t>(1, hp.n_vocab - 1));
        const auto t_warm = std::chrono::steady_clock::now();
        const bool warmed = forward(*m, toks.data(), nt, nullptr, false);
        s0->n_past = 0;
        m->embd_ovr.clear();
        m->graph_cache.clear();
        m->n_sparse_segments = m->n_dense_segments = 0;
        fprintf(stderr, "[minimax-m3] reserved GPU scratch with a %" PRId64 "-token prefill in %.1fs%s\n", nt,
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t_warm).count(), warmed ? "" : " (failed)");
    }

    // --- pipelined prefill ----------------------------------------------------
    {
        pipeline_state & P = m->pipe;
        const std::string mode = opt_get("TS_MINIMAX_M3_PIPELINE");
        if (!mode.empty() && mode != "0" && mode != "1")
        {
            tsg::report_load_refusal("[minimax-m3] TS_MINIMAX_M3_PIPELINE must be 0 or 1 (unset: automatic), not '%s'\n", mode.c_str());
            return nullptr;
        }
        long ub = m->n_ubatch;                  // adaptive: the largest micro-batch
        P.ubatch_fixed = false;
        if (const std::string ubs = opt_get("TS_MINIMAX_M3_PIPELINE_UBATCH"); !ubs.empty())
        {
            char * end = nullptr;
            errno = 0;
            const long v = strtol(ubs.c_str(), &end, 10);
            if (end == ubs.c_str() || *end != '\0' || errno == ERANGE || v < 1 || v > m->n_ubatch)
            {
                tsg::report_load_refusal("[minimax-m3] TS_MINIMAX_M3_PIPELINE_UBATCH must be an integer in [1, %d] (the prefill ubatch), not '%s'\n",
                                         m->n_ubatch, ubs.c_str());
                return nullptr;
            }
            ub = v;
            P.ubatch_fixed = true;
        }
        P.ubatch = ub;
        P.forced = mode == "1";
        // Overlap needs asynchronous copies between GPUs and streams that wait
        // on another GPU's events (CUDA, ROCm, MUSA). Host experts stall the
        // host thread inside every micro-batch, and the profiler synchronizes
        // every node, so neither gains anything.
        bool all_cuda = n_gpu > 0;
        const char * gpu_reg = nullptr;
        for (int d = 0; d < n_gpu; d++)
        {
            ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(m->backends[d]));
            gpu_reg = reg ? ggml_backend_reg_name(reg) : "unknown";
            all_cuda = all_cuda && reg && backend_matches(ggml_backend_reg_name(reg), "CUDA");
        }
        if (P.forced && n_gpu >= 2 && !all_cuda)
        {
            tsg::report_load_refusal("[minimax-m3] TS_MINIMAX_M3_PIPELINE=1 needs GPUs whose streams can wait on another GPU's events "
                                     "(CUDA, ROCm, MUSA); the %s backend's %d GPUs cannot run a pipelined prefill. "
                                     "Unset the option (automatic) or set it to 0\n", gpu_reg ? gpu_reg : "unknown", n_gpu);
            return nullptr;
        }
        const char * why = nullptr;
        if (m->tp.n > 1) why = "tensor parallel: every GPU works on every layer";
        else if (mode == "0") why = "TS_MINIMAX_M3_PIPELINE=0";
        else if (env_long("TS_MINIMAX_M3_PROFILE", 0) > 0) why = "TS_MINIMAX_M3_PROFILE synchronizes every node";
        else if (!P.forced && pipe_stage_list(*m).size() < 2) why = "the layers sit on fewer than two GPUs";
        else if (!P.forced && !all_cuda) why = "needs CUDA's asynchronous cross-GPU copies";
        else if (!P.forced && n_cpu_moe > 0) why = "host experts stall each micro-batch";
        P.enabled = why == nullptr;
        if (P.enabled && n_gpu > 0 && env_flag("TS_MINIMAX_M3_WARM_POOLS", true))
        {
            // Build the pipeline's scheduler now rather than in the first request,
            // and check that it engages: two fixed micro-batches, or a prompt of
            // up to 2048 rows (two adaptive micro-batches of half that).
            const int64_t nt = std::min<int64_t>(P.ubatch_fixed ? 2 * P.ubatch : std::min<int64_t>(2048, 2 * P.ubatch), m->n_ctx);
            std::vector<int32_t> toks((size_t) nt);
            for (int64_t i = 0; i < nt; i++) toks[(size_t) i] = (int32_t) (1 + i % std::max<int32_t>(1, hp.n_vocab - 1));
            const uint64_t before = P.n_calls;
            forward(*m, toks.data(), nt, nullptr, false);
            s0->n_past = 0;
            m->graph_cache.clear();
            m->n_sparse_segments = m->n_dense_segments = 0;
            if (P.n_calls == before)
            {
                P.enabled = false;
                P.release();
                why = "its warm-up did not run pipelined";
            }
        }
        if (P.enabled)
        {
            if (P.stages.empty()) P.stages = pipe_stage_list(*m);
            char rows[96];
            if (P.ubatch_fixed) snprintf(rows, sizeof(rows), "%" PRId64 "-row micro-batches", P.ubatch);
            else snprintf(rows, sizeof(rows), "micro-batches sized per prompt, %" PRId64 "-%" PRId64 " rows",
                          std::min<int64_t>(PIPE_MIN_UBATCH, P.ubatch), P.ubatch);
            if (P.sched)
                fprintf(stderr, "[minimax-m3] pipelined prefill: on (%s over %zu GPU stage(s), %d input copies)\n",
                        rows, P.stages.size(), P.n_copies);
            else
                fprintf(stderr, "[minimax-m3] pipelined prefill: on (%s over %zu GPU stage(s); its scheduler is built on first use)\n",
                        rows, P.stages.size());
        }
        else
            fprintf(stderr, "[minimax-m3] pipelined prefill: off (%s); prompts run one %d-row micro-batch at a time\n", why, m->n_ubatch);
    }

    int n_kernel_layers = 0;
    for (const layer & L : m->layers) n_kernel_layers += L.msa && m->msa_kernel_dev[L.device] ? 1 : 0;
    char placement[64];
    if (m->tp.n > 1) snprintf(placement, sizeof(placement), "tensor-parallel over %d GPUs", m->tp.n);
    else snprintf(placement, sizeof(placement), "on %d GPU(s)", n_gpu);
    fprintf(stderr, "[minimax-m3] loaded %d layers %s + host in %.1fs: ctx %d, ubatch %d, attention %s "
            "(%d sparse layers, %d of them on TensorSharp's MSA kernels, dense below position %" PRId64 ", %zu MiB per sparse "
            "chunk), fused qkv %s, indexer %s%s, op offload %s, host weights %.1f GiB\n",
            hp.n_layer, placement, std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count(),
            m->n_ctx, m->n_ubatch, msa_mode_name(m->msa), m->n_msa_layers, n_kernel_layers, hp.dense_span(), m->msa_chunk_bytes >> 20,
            m->fused_qkv ? "yes" : "no", m->idx_f16 ? "F32 (decode F16)" : "F32", m->fused_idx ? " fused" : "",
            m->op_offload ? "on" : "off", m->mmap_weight_bytes / 1073741824.0);
    return m.release();
}

} // namespace tsg_mm3

#include "ggml_ops_minimax_m3_dspark.inc"

using namespace tsg_mm3;

namespace
{
/// TS_MINIMAX_M3_EXEC_DEBUG: one stderr line per state-changing ABI call (what
/// the managed scheduler asked for, the active slot's head, the wall time), so
/// a request's host-side latency can be split into its executor calls.
struct call_trace
{
    const model * m;
    const char * what;
    int64_t a, b;
    int rc = 0;
    std::chrono::steady_clock::time_point t0;
    call_trace(const model * m_, const char * what_, int64_t a_ = 0, int64_t b_ = 0)
        : m(m_), what(what_), a(a_), b(b_), t0(std::chrono::steady_clock::now()) {}
    int ret(int r) { rc = r; return r; }
    ~call_trace()
    {
        if (!debug_on() || !m) return;
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        fprintf(stderr, "[minimax-m3] call %s(%" PRId64 ", %" PRId64 ") slot %d head %" PRId64 " -> %d in %.1f ms\n", what, a, b,
                m->active_slot ? m->active_slot->id : -1, m->active_slot ? m->active_slot->n_past : (int64_t) -1, rc, ms);
    }
};
} // namespace

// ---------------------------------------------------------------------------
// C ABI (contract C1: Muse Spark's ABI with the TSGgml_MiniMaxM3 prefix)
// ---------------------------------------------------------------------------
/// tp > 1 splits every layer over the first tp CUDA GPUs (tensor parallelism on
/// ggml's meta backend; n_gpu and layer_split then do not apply); tp <= 1 is
/// the layer split.
/// layer_split null or empty falls back to TS_MINIMAX_M3_LAYER_SPLIT, then to
/// the automatic placement.
TSG_EXPORT void * TSGgml_MiniMaxM3LoadModelTp(const char * gguf_path, int n_gpu, int n_ctx, int n_ubatch, int n_threads,
                                              int n_cpu_moe, const char * backend_name, int ctx_is_hard_limit,
                                              const char * layer_split, int tp)
{
    try
    {
        return load(gguf_path, n_gpu, n_ctx, n_ubatch, n_threads, n_cpu_moe, backend_name, ctx_is_hard_limit != 0,
                    layer_split, tp);
    }
    catch (const std::exception & e)
    {
        tsg::report_load_refusal("[minimax-m3] load failed: %s\n", e.what());
        return nullptr;
    }
}

/// Set (value non-null) or clear (null) an option that overrides the process
/// environment for subsequent loads and calls.
TSG_EXPORT void TSGgml_MiniMaxM3SetOption(const char * name, const char * value)
{
    if (!name || !*name) return;
    std::lock_guard<std::mutex> lk(g_opt_mu);
    if (value) g_opts[name] = value;
    else g_opts.erase(name);
}

TSG_EXPORT void TSGgml_MiniMaxM3Free(void * handle)
{
    delete (model *) handle;
}

TSG_EXPORT int TSGgml_MiniMaxM3VocabSize(void * handle) { return handle ? ((model *) handle)->hp.n_vocab : 0; }
TSG_EXPORT int TSGgml_MiniMaxM3CtxSize(void * handle) { return handle ? ((model *) handle)->n_ctx : 0; }
TSG_EXPORT int TSGgml_MiniMaxM3HiddenSize(void * handle) { return handle ? ((model *) handle)->hp.n_embd : 0; }

TSG_EXPORT int TSGgml_MiniMaxM3NPast(void * handle)
{
    model * m = (model *) handle;
    if (!m) return -1;
    std::lock_guard<std::mutex> lk(m->mu);
    return m->active_slot ? (int) m->active_slot->n_past : -1;
}

TSG_EXPORT int TSGgml_MiniMaxM3Forward(void * handle, const int32_t * tokens, int n_tokens, float * logits_out, int all_rows)
{
    model * m = (model *) handle;
    if (!m || !tokens || n_tokens <= 0) return 0;
    std::lock_guard<std::mutex> lk(m->mu);
    call_trace tr(m, "Forward", n_tokens, all_rows);
    try { return tr.ret(forward(*m, tokens, n_tokens, logits_out, all_rows != 0) ? 1 : 0); }
    catch (const std::exception & e) { fprintf(stderr, "[minimax-m3] forward failed: %s\n", e.what()); return 0; }
}

TSG_EXPORT int TSGgml_MiniMaxM3ForwardBatchedDecode(void * handle, int n, const int32_t * slot_ids, const int32_t * tokens,
                                                    const int32_t * positions, float * logits_out)
{
    model * m = (model *) handle;
    if (!m || n < 1 || !slot_ids || !tokens || !positions || !logits_out) return 0;
    std::lock_guard<std::mutex> lk(m->mu);
    call_trace tr(m, "ForwardBatchedDecode", n, positions[0]);
    try { return tr.ret(forward_batched_decode(*m, n, slot_ids, tokens, positions, logits_out) ? 1 : 0); }
    catch (const std::exception & e) { fprintf(stderr, "[minimax-m3] batched decode failed: %s\n", e.what()); return 0; }
}

/// Several slots' token segments in one graph; one logits row per segment.
TSG_EXPORT int TSGgml_MiniMaxM3ForwardSegments(void * handle, int n, const int32_t * slot_ids, const int32_t * lens,
                                               const int32_t * tokens, const int32_t * positions, float * logits_out)
{
    model * m = (model *) handle;
    if (!m || n < 1 || !slot_ids || !lens || !tokens || !positions || !logits_out) return 0;
    std::lock_guard<std::mutex> lk(m->mu);
    call_trace tr(m, "ForwardSegments", n, lens[0]);
    try { return tr.ret(forward_segments(*m, n, slot_ids, lens, tokens, positions, logits_out) ? 1 : 0); }
    catch (const std::exception & e) { fprintf(stderr, "[minimax-m3] segment forward failed: %s\n", e.what()); return 0; }
}

TSG_EXPORT int TSGgml_MiniMaxM3Ubatch(void * handle) { return handle ? ((model *) handle)->n_ubatch : 0; }

/// Queue `n_rows` rows [n_rows, n_embd] that replace the token embeddings at
/// absolute positions index .. index + n_rows - 1 of the active slot's
/// sequence in the next Forward call (rows outside that call are dropped;
/// the queue empties after every Forward).
TSG_EXPORT int TSGgml_MiniMaxM3QueueVisionRows(void * handle, const float * rows, int n_rows, int index)
{
    model * m = (model *) handle;
    if (!m || !rows || n_rows <= 0 || index < 0) return 0;
    std::lock_guard<std::mutex> lk(m->mu);
    model::embd_override o;
    o.index = index;
    o.rows.assign(rows, rows + (size_t) n_rows * m->hp.n_embd);
    m->embd_ovr.push_back(std::move(o));
    return 1;
}

TSG_EXPORT void TSGgml_MiniMaxM3ClearVisionRows(void * handle)
{
    model * m = (model *) handle;
    if (!m) return;
    std::lock_guard<std::mutex> lk(m->mu);
    m->embd_ovr.clear();
}

TSG_EXPORT int TSGgml_MiniMaxM3ResetChecked(void * handle)
{
    model * m = (model *) handle;
    if (!m) return 0;
    std::lock_guard<std::mutex> lk(m->mu);
    if (!m->active_slot) return 0;
    m->active_slot->n_past = 0;
    m->embd_ovr.clear();
    return 1;
}

/// Every layer keeps every row (no sliding window), so any rewind is exact:
/// rows past the new head are rewritten before a later query reads them.
TSG_EXPORT int TSGgml_MiniMaxM3Rewind(void * handle, int n_past)
{
    model * m = (model *) handle;
    if (!m) return 0;
    std::lock_guard<std::mutex> lk(m->mu);
    call_trace tr(m, "Rewind", n_past);
    slot * s = m->active_slot;
    if (!s || n_past < 0 || n_past > s->n_past) return 0;
    s->n_past = n_past;
    return tr.ret(1);
}

TSG_EXPORT int TSGgml_MiniMaxM3SlotAlloc(void * handle)
{
    model * m = (model *) handle;
    if (!m) return -1;
    std::lock_guard<std::mutex> lk(m->mu);
    slot * s = slot_alloc(*m, true);
    return s ? s->id : -1;
}

TSG_EXPORT int TSGgml_MiniMaxM3SetActiveSlot(void * handle, int slot_id)
{
    model * m = (model *) handle;
    if (!m) return 0;
    std::lock_guard<std::mutex> lk(m->mu);
    call_trace tr(m, "SetActiveSlot", slot_id);
    auto it = m->slots.find(slot_id);
    if (it == m->slots.end()) return 0;
    m->active_slot = it->second.get();
    return tr.ret(1);
}

TSG_EXPORT int TSGgml_MiniMaxM3SlotFree(void * handle, int slot_id)
{
    model * m = (model *) handle;
    if (!m) return 0;
    std::lock_guard<std::mutex> lk(m->mu);
    if (m->active_slot && m->active_slot->id == slot_id) return 0;
    if (!m->slots.count(slot_id)) return 0;
    slot_free(*m, slot_id);
    return 1;
}

/// Make slot `dst` an independent copy of slot `src` (its K, V and index-key
/// rows and its head). Continuing either afterwards never disturbs the other.
TSG_EXPORT int TSGgml_MiniMaxM3SlotCopy(void * handle, int src_id, int dst_id)
{
    model * m = (model *) handle;
    if (!m || src_id == dst_id) return 0;
    std::lock_guard<std::mutex> lk(m->mu);
    call_trace tr(m, "SlotCopy", src_id, dst_id);
    auto si = m->slots.find(src_id), di = m->slots.find(dst_id);
    if (si == m->slots.end() || di == m->slots.end()) return 0;
    slot & d = *di->second;
    try
    {
        const int64_t rows = std::min<int64_t>(si->second->cap, pad_to(std::max<int64_t>(si->second->n_past, 1), m->kv_pad));
        // Growing dst may free retained slots for memory; the source is often one
        // of them (a prefix-cache clone copies a retained checkpoint), so it is
        // kept out of that, and looked up again after.
        const int32_t keep[1] = { src_id };
        if (!slot_reserve_rows(*m, d, rows, keep, 1)) return 0;
        si = m->slots.find(src_id);
        if (si == m->slots.end()) return 0;
        slot & s = *si->second;
        std::vector<row_copy> copies;
        for (int il = 0; il < m->hp.n_layer; il++)
        {
            copies.push_back({ s.k[(size_t) il], d.k[(size_t) il], rows });
            copies.push_back({ s.v[(size_t) il], d.v[(size_t) il], rows });
            copies.push_back({ s.ik[(size_t) il], d.ik[(size_t) il], rows });
        }
        if (!copy_cache_rows(*m, copies) || !dspark_slot_copy(*m, s, d)) return 0;
        d.n_past = s.n_past;
        return tr.ret(1);
    }
    catch (const std::exception & e)
    {
        fprintf(stderr, "[minimax-m3] slot copy failed: %s\n", e.what());
        return 0;
    }
}

TSG_EXPORT int TSGgml_MiniMaxM3SlotNPast(void * handle, int slot_id)
{
    model * m = (model *) handle;
    if (!m) return -1;
    std::lock_guard<std::mutex> lk(m->mu);
    auto it = m->slots.find(slot_id);
    return it == m->slots.end() ? -1 : (int) it->second->n_past;
}

TSG_EXPORT int TSGgml_MiniMaxM3SetReclaimableSlots(void * handle, const int32_t * slot_ids, int n)
{
    model * m = (model *) handle;
    if (!m || n < 0 || (n > 0 && !slot_ids)) return 0;
    std::lock_guard<std::mutex> lk(m->mu);
    m->reclaimable.assign(slot_ids, slot_ids + n);
    return 1;
}

TSG_EXPORT int TSGgml_MiniMaxM3TakeReclaimedSlots(void * handle, int32_t * out, int cap)
{
    model * m = (model *) handle;
    if (!m || cap < 0 || (cap > 0 && !out)) return 0;
    std::lock_guard<std::mutex> lk(m->mu);
    const int n = std::min(cap, (int) m->reclaimed.size());
    for (int i = 0; i < n; i++) out[i] = m->reclaimed[(size_t) i];
    m->reclaimed.erase(m->reclaimed.begin(), m->reclaimed.begin() + n);
    return n;
}

/// The micro-batch rows a prompt of n tokens starting at position p0 would
/// prefill pipelined with (0: it would run one ubatch at a time), so the
/// scheduler can size a lone prompt's steps to calls that pipeline.
TSG_EXPORT int TSGgml_MiniMaxM3PipelinePlan(void * handle, int64_t p0, int64_t n)
{
    model * m = (model *) handle;
    if (!m || p0 < 0 || n <= 0) return 0;
    std::lock_guard<std::mutex> lk(m->mu);
    const pipeline_state & P = m->pipe;
    if (!P.enabled) return 0;
    pipe_shape sh;
    if (pipe_plan(*m, p0, n, &sh) != PIPE_OK || sh.p_last > m->n_ctx) return 0;
    // A shape whose compute buffers did not fit is not retried (forward_pipelined).
    if (P.fail_u && sh.u >= P.fail_u && sh.n_kv >= P.fail_kv) return 0;
    return (int) sh.u;
}

/// Diagnostics. [0..11] are Muse Spark's: graphs built / reused, host weight
/// bytes, layers with host experts, GPUs, slots, pipelined calls, micro-batches,
/// declines, pipeline micro-batch, pipeline stages, tensor-parallel degree
/// (0: layer split). MiniMax-M3 appends: [12] attention mode (0 auto, 1 dense,
/// 2 ggml, 3 kernel), [13] segments computed with sparse attention, [14]
/// segments computed with dense attention, [15] sparse (MSA) layers, [16] the
/// sparse layers that run TensorSharp's MSA kernels.
TSG_EXPORT int TSGgml_MiniMaxM3GetStats(void * handle, int64_t * out, int cap)
{
    model * m = (model *) handle;
    if (!m || !out || cap < 6) return 0;
    std::lock_guard<std::mutex> lk(m->mu);
    out[0] = (int64_t) m->n_graph_builds;
    out[1] = (int64_t) m->n_graph_reuses;
    out[2] = (int64_t) m->mmap_weight_bytes;
    out[3] = m->n_cpu_moe;
    out[4] = m->n_gpu;
    out[5] = (int64_t) m->slots.size();
    if (cap < 10) return 6;
    out[6] = (int64_t) m->pipe.n_calls;       // prefill calls that ran pipelined
    out[7] = (int64_t) m->pipe.n_ubatches;    // micro-batches they enqueued
    out[8] = (int64_t) m->pipe.n_declines;    // calls that wanted to and could not
    out[9] = m->pipe.enabled ? m->pipe.ubatch : 0;   // the fixed micro-batch, or the adaptive one's upper bound
    if (cap < 11) return 10;
    out[10] = m->pipe.enabled ? (int64_t) pipe_stage_list(*m).size() : 0;   // GPU stages a pipelined call spans
    if (cap < 12) return 11;
    out[11] = m->tp.n > 1 ? m->tp.n : 0;      // tensor-parallel degree (0: layer split)
    if (cap < 16) return 12;
    out[12] = m->msa;
    out[13] = (int64_t) m->n_sparse_segments;
    out[14] = (int64_t) m->n_dense_segments;
    out[15] = m->n_msa_layers;
    if (cap < 17) return 16;
    int64_t n_kernel = 0;
    for (const layer & L : m->layers) n_kernel += L.msa && m->msa_kernel_dev[L.device] ? 1 : 0;
    out[16] = n_kernel;
    return 17;
}

#if defined(TSG_GGML_TEST_HOOKS)
#ifndef TSG_TEST_EXPORT
#define TSG_TEST_EXPORT TSG_EXPORT
#endif
namespace tsg_mm3
{
/// TSGgml_MiniMaxM3TestAttention's body: one synthetic single-layer slot whose
/// caches hold the caller's rows, and the builder's own attention per graph.
static bool test_attention(const char * backend_name, int path, const int32_t * geom, const float * q, const float * iq,
                           const uint16_t * k, const uint16_t * v, const uint16_t * idx, int64_t rows, int64_t p0,
                           int64_t nt, int64_t chunk, float * out)
{
    std::unique_ptr<model> mp(new model());
    model & m = *mp;
    hparams & hp = m.hp;
    hp.n_layer = 1;
    hp.n_head = geom[0];
    hp.n_head_kv = geom[1];
    hp.head_dim = geom[2];
    hp.idx_n_head = geom[3];
    hp.idx_dim = geom[4];
    hp.idx_blk = geom[5];
    hp.idx_topk = geom[6];
    hp.idx_local = geom[7];
    m.kv_pad = geom[8];
    m.n_ctx = (int32_t) rows;
    m.msa = MSA_GGML;
    m.msa_chunk_bytes = (size_t) env_long("TS_MINIMAX_M3_MSA_CHUNK_MB", 256) << 20;
    if (path < 0 || path > 2 || nt < 1 || p0 < 0 || p0 + nt > rows || hp.idx_blk <= 0 || m.kv_pad % hp.idx_blk != 0
        || rows % hp.idx_blk != 0 || hp.idx_n_head != hp.n_head_kv || hp.n_head % hp.n_head_kv != 0)
    {
        fprintf(stderr, "[minimax-m3 test] invalid attention test geometry\n");
        return false;
    }

    if (backend_name && strcmp(backend_name, "CPU") != 0)
    {
        for (size_t i = 0; i < ggml_backend_dev_count() && m.n_gpu == 0; i++)
        {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
            if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) continue;
            if (!reg || !backend_matches(ggml_backend_reg_name(reg), backend_name)) continue;
            if ((m.backends[0] = ggml_backend_dev_init(dev, nullptr))) m.n_gpu = 1;
        }
        if (m.n_gpu == 0)
        {
            fprintf(stderr, "[minimax-m3 test] no %s device\n", backend_name);
            return false;
        }
    }
    m.backends[m.n_gpu] = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!m.backends[m.n_gpu]) return false;
    m.n_backends = m.n_sched_backends = m.n_gpu + 1;
    for (int i = 0; i < m.n_backends; i++)
    {
        m.sched_backends[i] = m.backends[i];
        m.sched_bufts[i] = ggml_backend_get_default_buffer_type(m.backends[i]);
    }
    const int dev = 0;   // the layer's device: the GPU, or the host without one

    slot s;
    {
        ggml_init_params cp = { 3 * ggml_tensor_overhead(), nullptr, true };
        ggml_context * cctx = ggml_init(cp);
        if (!cctx) return false;
        s.k.push_back(ggml_new_tensor_2d(cctx, GGML_TYPE_F16, hp.n_kv(), rows));
        s.v.push_back(ggml_new_tensor_2d(cctx, GGML_TYPE_F16, hp.n_kv(), rows));
        s.ik.push_back(ggml_new_tensor_2d(cctx, GGML_TYPE_F16, hp.idx_dim, rows));
        ggml_backend_buffer_t cbuf = ggml_backend_alloc_ctx_tensors(cctx, m.backends[dev]);
        m.slot_ctxs.push_back({ 0, cctx, cbuf });
        if (!cbuf) return false;
        ggml_backend_tensor_set(s.k[0], k, 0, ggml_nbytes(s.k[0]));
        ggml_backend_tensor_set(s.v[0], v, 0, ggml_nbytes(s.v[0]));
        ggml_backend_tensor_set(s.ik[0], idx, 0, ggml_nbytes(s.ik[0]));
        s.cap = rows;
    }

    // One graph: tokens t0..t0+n-1 (all of them, or one per graph on path 1).
    auto run = [&](int64_t t0, int64_t n) -> bool
    {
        const int64_t pq = p0 + t0;
        const int64_t n_kv = plan_n_kv(m, pq + n, m.kv_pad);
        const int64_t nblk = n_kv / hp.idx_blk;
        const int64_t ch = path == 2 ? (chunk > 0 ? std::min(chunk, n) : msa_chunk_tokens(m, n_kv, n)) : 0;
        const size_t n_nodes = 512 + (size_t) 64 * (size_t) (ch > 0 ? (n + ch - 1) / ch : 1);
        graph_entry g;
        g.nt = n;
        ggml_init_params gp = { ggml_tensor_overhead() * n_nodes + ggml_graph_overhead_custom(n_nodes, false), nullptr, true };
        g.ctx = ggml_init(gp);
        if (!g.ctx) return false;
        g.gf = ggml_new_graph_custom(g.ctx, n_nodes, false);
        g.sched = ggml_backend_sched_new(m.sched_backends, m.sched_bufts, m.n_sched_backends, n_nodes, false, false);
        if (!g.sched) return false;
        builder b(m, g);
        ggml_tensor * qt = b.new_input(GGML_TYPE_F32, hp.n_q(), n, "test_q", dev);
        ggml_tensor * iqt = b.new_input(GGML_TYPE_F32, (int64_t) hp.idx_dim * hp.idx_n_head, n, "test_iq", dev);
        ggml_tensor * mask = b.new_input(GGML_TYPE_F16, n_kv, n, "test_mask", dev);
        ggml_tensor * bias = b.new_input(GGML_TYPE_F32, nblk, n, "test_bias", dev);
        ggml_tensor * q3 = ggml_reshape_3d(g.ctx, qt, hp.head_dim, hp.n_head, n);
        ggml_tensor * iq3 = ggml_reshape_3d(g.ctx, iqt, hp.idx_dim, hp.idx_n_head, n);
        ggml_tensor * o = path == 0 ? b.attend_dense(s, 0, dev, q3, n_kv, mask)
                        : n == 1    ? b.msa_decode(s, 0, dev, q3, iq3, n_kv, mask, bias)
                                    : b.msa_prompt(s, 0, dev, q3, iq3, n, n_kv, mask, bias, ch);
        if (!ggml_is_contiguous(o)) o = ggml_cont(g.ctx, o);
        ggml_set_output(o);
        ggml_build_forward_expand(g.gf, o);
        if (!ggml_backend_sched_alloc_graph(g.sched, g.gf))
        {
            fprintf(stderr, "[minimax-m3 test] graph allocation failed\n");
            return false;
        }
        std::vector<uint16_t> hm((size_t) (n_kv * n));
        std::vector<float> hb((size_t) (nblk * n));
        fill_mask(hm.data(), n_kv, pq, n);
        fill_bias(hp, hb.data(), nblk, pq, n);
        if (live(qt)) ggml_backend_tensor_set(qt, q + (size_t) t0 * hp.n_q(), 0, ggml_nbytes(qt));
        if (live(iqt)) ggml_backend_tensor_set(iqt, iq + (size_t) (t0 * hp.idx_dim * hp.idx_n_head), 0, ggml_nbytes(iqt));
        if (live(mask)) ggml_backend_tensor_set(mask, hm.data(), 0, ggml_nbytes(mask));
        if (live(bias)) ggml_backend_tensor_set(bias, hb.data(), 0, ggml_nbytes(bias));
        if (!compute(m, g)) return false;
        ggml_backend_tensor_get(o, out + (size_t) t0 * hp.n_q(), 0, (size_t) n * hp.n_q() * sizeof(float));
        return true;
    };
    if (path == 1)
    {
        for (int64_t t = 0; t < nt; t++)
            if (!run(t, 1)) return false;
        return true;
    }
    return run(0, nt);
}
} // namespace tsg_mm3

/// Test hook (native test builds only): the executor's own attention graphs
/// over one synthetic sequence whose caches already hold `rows` rows, for the
/// nt queries at positions p0..p0+nt-1 -- attend_dense / msa_decode /
/// msa_prompt exactly as a forward builds them, without the projections or
/// the cache writes. path 0: dense causal attention, all nt tokens in one
/// graph; 1: sparse, one graph per token (decode); 2: sparse, all nt tokens in
/// one graph (a prompt chunk or a verification), `chunk` tokens per block-mask
/// chunk (0: the executor's byte budget). geom = { n_head, n_head_kv, head_dim,
/// idx_n_head, idx_dim, block, topk, local_blocks, kv_pad }. q F32
/// [nt][n_head][head_dim], iq F32 [nt][idx_n_head][idx_dim], k / v F16
/// [rows][n_head_kv * head_dim], idx F16 [rows][idx_dim] (contract C2); out F32
/// [nt][n_head][head_dim]. backend "CPU" or a GPU registry ("CUDA", "Metal").
/// Returns 1, or 0 with the reason on stderr.
TSG_TEST_EXPORT int TSGgml_MiniMaxM3TestAttention(const char * backend_name, int path, const int32_t * geom, const float * q,
                                                  const float * iq, const uint16_t * k, const uint16_t * v,
                                                  const uint16_t * idx, int64_t rows, int64_t p0, int64_t nt, int64_t chunk,
                                                  float * out)
{
    if (!geom || !q || !iq || !k || !v || !idx || !out) return 0;
    try
    {
        return test_attention(backend_name, path, geom, q, iq, k, v, idx, rows, p0, nt, chunk, out) ? 1 : 0;
    }
    catch (const std::exception & e)
    {
        fprintf(stderr, "[minimax-m3 test] attention test failed: %s\n", e.what());
        return 0;
    }
}

/// Fault injection for the load's thread teams: from now on, thread start
/// number `thread_start` (0: the next one) throws std::system_error as an
/// exhausted thread quota does, and weight-upload job number `upload_job`
/// throws std::bad_alloc inside its worker; -1 disables either.
TSG_TEST_EXPORT void TSGgml_MiniMaxM3TestInjectFault(int thread_start, int upload_job)
{
    tsg_mm3::g_test_fail_thread.store(thread_start < 0 ? -1 : thread_start);
    tsg_mm3::g_test_fail_job.store(upload_job < 0 ? -1 : upload_job);
}

/// Non-zero: every CUDA GPU reports compute capability 7.5 to the MSA kernel
/// gate of the next loads (TensorSharp's MSA kernels need 8.0).
TSG_TEST_EXPORT void TSGgml_MiniMaxM3TestMsaOldGpus(int on)
{
    tsg_mm3::g_test_msa_old_gpus.store(on != 0);
}

/// DSpark pipeline ordering, for graphs built from now on: `slow_products`
/// large F32 products (~0.55 TFLOP each) in every DSpark commit on the
/// drafter's GPU (a drafter stage the earlier pipeline stages run ahead of; 0:
/// none), and
/// `no_tap_wait` non-zero drops the pipelined prefill's waits for DSpark taps
/// that skip stages (the race they close becomes observable).
TSG_TEST_EXPORT void TSGgml_MiniMaxM3TestDsparkPipeline(int slow_products, int no_tap_wait)
{
    tsg_mm3::g_test_ds_slow_products.store(std::max(0, slow_products));
    tsg_mm3::g_test_ds_no_tap_wait.store(no_tap_wait != 0);
}
#endif
