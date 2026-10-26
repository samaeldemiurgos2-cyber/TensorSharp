// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// ---------------------------------------------------------------------------
// Muse Spark whole-model executor for the GGML backends.
//
// Muse Spark (816B-A42B) does not fit a host the way a per-op managed forward
// wants it to: every layer is ~6 GiB at Q3, so the model is spread across every
// GPU with the leading layers' routed experts left in system RAM. Driving that
// one operation at a time spends more time in host round trips and per-op graph
// setup than in arithmetic, so this executor follows llama.cpp's structure
// (src/models/muse-spark.cpp) and TensorSharp's GLM executor:
//
//   * it loads the split GGUF itself and places whole layers on the GPUs,
//     sized against each device's free memory;
//   * `--n-cpu-moe N` serves the first N layers' routed experts zero-copy from
//     the GGUF mapping, so the scheduler runs exactly those mul_mat_id nodes on
//     the CPU backend and everything else stays on the GPUs;
//   * every sequence slot owns F16 K/V rows on the device that reads them;
//   * ONE ggml graph per ubatch goes through ggml_backend_sched, and an LRU of
//     built+allocated graphs keyed by shape lets decode replay the same graph
//     (and, on CUDA, the same captured CUDA graphs) token after token;
//   * a batched decode step evaluates one token for each of several slots in
//     a single graph, so concurrent requests share every weight read.
//
// Numerical order (llama.cpp's graph, verified against it):
//   embd -> rms_norm
//   per layer: attn_norm -> Q/K/V/gate -> per-head Q/K rms_norm (weighted)
//              -> RoPE on sliding layers only -> causal GQA attention (2048
//              window on sliding layers) -> per-head rms_norm -> * sigmoid(gate)
//              -> wo -> rms_norm(post eps) -> x*alpha + attn*beta
//              ffn_norm -> latent_down -> rms_norm(pre_exp_norm)
//              router(ffn_norm out): sigmoid, +bias for selection only, top-8,
//              normalized unbiased weights
//              experts SwiGLU -> *post_exp_norm -> rms_norm(post eps) -> *weight
//              -> sum -> latent_up -> *post_ffw_norm -> rms_norm(post eps)
//              -> x*alpha + ffn*beta
//   output_norm -> output -> *logit_scale -> softcap(20)
// ---------------------------------------------------------------------------

#include "ggml_ops_cuda_launch_policy.h"
#include "ggml_ops_internal.h"
#include "ggml_ops_muse_spark_tp.h"
#include "dsv4_file_warm.h"

#include "gguf.h"

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

namespace tsg_mspark
{

/// GPUs one model can span. ggml-cuda addresses at most GGML_CUDA_MAX_DEVICES
/// (16) and ggml-vulkan GGML_VK_MAX_DEVICES (16); TensorSharp builds ggml-base
/// with GGML_SCHED_MAX_BACKENDS=32, so these plus the CPU fit one scheduler.
static constexpr int MAX_GPUS = 16;
/// Sequences one batched decode step serves at most.
static constexpr int MAX_BATCHED_DECODE = 16;
/// Default granularity of the attention windows. Decode reuses one graph shape
/// for a whole stretch of this many positions instead of rebuilding every token.
static constexpr int64_t KV_PAD_DEFAULT = 256;
/// (token, expert) pairs host_moe_matmul handles in one call.
static constexpr int64_t HOST_MOE_MAX_PICKS = 512;

static int64_t pad_to(int64_t v, int64_t p) { return ((v + p - 1) / p) * p; }

// Options set through TSGgml_MuseSparkSetOption take precedence over the
// process environment: .NET's Environment.SetEnvironmentVariable does not
// reach getenv on Linux, so the managed side forwards its TS_MUSE_SPARK_*
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
    static const bool on = env_flag("TS_MUSE_SPARK_EXEC_DEBUG", false);
    return on;
}

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
    int32_t n_swa = 0;
    int32_t latent = 0;
    int32_t n_ff_exp = 0;
    int32_t n_expert = 0;
    int32_t n_expert_used = 0;
    int32_t n_vocab = 0;
    int32_t n_ctx_train = 0;
    float rms_eps = 1e-5f;
    float post_eps = 1e-8f;
    float rope_base = 500000.0f;
    float rope_base_swa = 500000.0f;
    float logit_scale = 1.0f;
    float softcap = 0.0f;
    float w_scale = 1.0f;
    bool w_norm = true;
    int32_t gating = 2;
    std::vector<uint8_t> is_swa;

    int64_t n_q() const { return (int64_t) n_head * head_dim; }
    int64_t n_kv() const { return (int64_t) n_head_kv * head_dim; }
    float kq_scale() const { return 1.0f / sqrtf((float) head_dim); }
};

struct layer
{
    ggml_tensor * attn_norm = nullptr;
    ggml_tensor * wq = nullptr;
    ggml_tensor * wk = nullptr;
    ggml_tensor * wv = nullptr;
    ggml_tensor * wg = nullptr;       // attention output gate
    ggml_tensor * wqkvg = nullptr;    // [n_embd, nq + nkv + nkv + nq] when the four share a type
    ggml_tensor * wo = nullptr;
    ggml_tensor * q_norm = nullptr;
    ggml_tensor * k_norm = nullptr;
    ggml_tensor * attn_alpha = nullptr;
    ggml_tensor * attn_beta = nullptr;

    ggml_tensor * ffn_norm = nullptr;
    ggml_tensor * post_ffw_norm = nullptr;
    ggml_tensor * ffn_alpha = nullptr;
    ggml_tensor * ffn_beta = nullptr;
    ggml_tensor * gate_inp = nullptr;
    ggml_tensor * exp_probs_b = nullptr;
    ggml_tensor * lat_down = nullptr;
    ggml_tensor * lat_up = nullptr;
    ggml_tensor * pre_exp_norm = nullptr;
    ggml_tensor * post_exp_norm = nullptr;
    ggml_tensor * gate_exps = nullptr;
    ggml_tensor * up_exps = nullptr;
    ggml_tensor * down_exps = nullptr;

    int device = 0;
    bool cpu_moe = false;          // any routed-expert stack on the host
    bool cpu_stack[3] = {};        // gate, up, down stack on the host
    bool swa = false;
};

struct slot
{
    int id = 0;
    int64_t n_past = 0;
    std::vector<ggml_tensor *> k;   // [layer] F16 [n_kv, rows]: n_ctx rows (global) or the ring (sliding)
    std::vector<ggml_tensor *> v;
    // Sliding layers keep a ring of rows: row r holds position swa_lab[r], or
    // -1 when it holds nothing a later query may read.
    std::vector<int64_t> swa_lab;
    // Rows allocated for the global layers. They grow (doubling, copied on the
    // device) as the sequence does, so an idle or short slot holds little.
    int64_t cap_g = 0;
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
    size_t bytes = 0;
};

/// Per-device inputs of one graph. Only devices that host a layer get them.
struct graph_inputs
{
    ggml_tensor * tokens = nullptr;                 // I32 [nt] (embedding device)
    ggml_tensor * embd_rows = nullptr;              // F32 [n_embd, n_ovr]
    ggml_tensor * embd_idx = nullptr;               // I64 [n_ovr]
    ggml_tensor * pos[MAX_GPUS + 1] = {};           // I32 [nt]
    ggml_tensor * kv_idx[MAX_GPUS + 1] = {};        // I64 [nt] global-layer rows (= positions)
    ggml_tensor * kv_idx_s[MAX_GPUS + 1] = {};      // I64 [nt] sliding-layer ring rows
    ggml_tensor * mask_g[MAX_GPUS + 1] = {};        // F16 [n_kv_g, nt]   (batched: [n_kv_g_max, nt])
    ggml_tensor * mask_s[MAX_GPUS + 1] = {};        // F16 [n_kv_s, nt]   (segments: flat, each segment's rows contiguous)
    ggml_tensor * out_ids = nullptr;                // I32 [n_out] (last device)
};

/// One segment of a multi-sequence graph: `len` consecutive tokens of one
/// slot (a decode step is a segment of length 1).
struct bd_token
{
    int slot_id = 0;
    int64_t p = 0;          // position of the segment's first token (inputs only, never matched)
    int64_t len = 1;
    int64_t n_kv_g = 0;     // visible global window (padded)
    int64_t s0 = 0;         // first cache row of the sliding window
    int64_t n_kv_s = 0;     // rows of the sliding window (padded)
};

struct graph_entry
{
    // ---- key ----
    int kind = 0;               // 0 = one slot, nt tokens; 1 = segments of several slots
    int slot_id = -1;
    int64_t nt = 0;
    int64_t p0 = 0;             // only for the key's window math (not part of the match)
    int64_t n_kv_g = 0;
    int64_t s0 = 0;
    int64_t n_kv_s = 0;
    int64_t n_out = 0;
    int64_t n_ovr = 0;
    bool want_logits = true;
    std::vector<bd_token> bd;

    // ---- graph ----
    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_backend_sched_t sched = nullptr;
    graph_inputs inp;
    ggml_tensor * logits = nullptr;
    ggml_tensor * hidden = nullptr;     // post-norm rows (when requested)
    bool owns_sched = true;             // false: the pipeline's persistent scheduler

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
    bool forced = false;                     // TS_MUSE_SPARK_PIPELINE=1
    bool ubatch_fixed = false;               // TS_MUSE_SPARK_PIPELINE_UBATCH set: every call uses `ubatch` rows
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
    int64_t res_u = 0, res_kv_g = 0, res_kv_s = 0;   // shapes the scheduler is reserved for
    int res_nodes = 0;                               // node count of the reserved graph (its topology)
    // The smallest shape whose reservation failed: larger calls decline without
    // retrying the allocation until a slot gives memory back (slot_free).
    int64_t fail_u = 0, fail_kv_g = 0, fail_kv_s = 0;

    void release()
    {
        res_u = res_kv_g = res_kv_s = 0;
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

    ggml_backend_t backends[MAX_GPUS + 1] = {};
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

    int32_t n_ctx = 0;
    int32_t n_ubatch = 0;
    int32_t swa_ring = 0;    // rows a sliding layer keeps per slot
    int64_t kv_pad = KV_PAD_DEFAULT;
    bool op_offload = false;
    int64_t host_moe_max_tokens = 0;   // batches of 2..this many tokens use host_moe_matmul (0: never)
    bool fused_qkvg = false;
    bool alloc_failed = false;   // the last acquire_graph ran out of device memory
    std::vector<size_t> max_graph_bytes;   // per backend: the largest compute buffer any graph has needed

    struct slot_ctx { int slot_id; int part; ggml_context * ctx; ggml_backend_buffer_t buf; };   // part 0 ring, 1 global
    std::vector<slot_ctx> slot_ctxs;
    std::map<int, std::unique_ptr<slot>> slots;
    slot * active_slot = nullptr;
    int next_slot_id = 0;

    std::list<std::unique_ptr<graph_entry>> graph_cache;
    int graph_cache_cap = 12;

    std::vector<int> reclaimable;
    std::vector<int> reclaimed;

    // vision rows queued for the next forward call: index = row in that call's token array
    struct embd_override { int64_t index; std::vector<float> rows; };
    std::vector<embd_override> embd_ovr;

    // host scratch
    std::vector<int32_t> h_tokens;
    std::vector<int32_t> h_pos;
    std::vector<int64_t> h_kv_idx;
    std::vector<int64_t> h_kv_idx_s;
    std::vector<uint16_t> h_mask_g;
    std::vector<uint16_t> h_mask_s;
    std::vector<int32_t> h_out_ids;

    // stats
    uint64_t n_graph_builds = 0;
    uint64_t n_graph_reuses = 0;

    pipeline_state pipe;

    // Tensor parallelism (load with tp > 1): backends[0] is one ggml meta
    // backend over tp.devs (n_gpu == 1), and tp.ud's plan splits every weight
    // and K/V tensor among them.
    struct tp_state
    {
        int n = 0;
        ggml_backend_dev_t devs[MAX_GPUS] = {};
        ggml_backend_dev_t meta_dev = nullptr;
        std::unique_ptr<tsg_mspark_tp::userdata> ud;
    } tp;

    std::mutex mu;

    ~model()
    {
        pipe.release();   // its scheduler and events reference the backends
        graph_cache.clear();
        slots.clear();
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

static bool kv_bool_arr(gguf_context * g, const char * key, std::vector<uint8_t> & out)
{
    const int64_t id = gguf_find_key(g, key);
    if (id < 0 || gguf_get_kv_type(g, id) != GGUF_TYPE_ARRAY) return false;
    const enum gguf_type at = gguf_get_arr_type(g, id);
    const size_t n = gguf_get_arr_n(g, id);
    out.assign(n, 0);
    const void * data = gguf_get_arr_data(g, id);
    for (size_t i = 0; i < n; i++)
    {
        switch (at)
        {
            case GGUF_TYPE_BOOL:
            case GGUF_TYPE_INT8:
            case GGUF_TYPE_UINT8:  out[i] = ((const uint8_t *) data)[i] ? 1 : 0; break;
            case GGUF_TYPE_INT32:
            case GGUF_TYPE_UINT32: out[i] = ((const uint32_t *) data)[i] ? 1 : 0; break;
            default: return false;
        }
    }
    return true;
}

static std::string akey(const char * suffix)
{
    return std::string("muse-spark.") + suffix;
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
    size_t len;
};

/// `set_mu`, when given, serializes the device writes (a meta buffer's split
/// bookkeeping is not thread-safe); the file reads still run in parallel.
static bool upload_parallel(const shard_files & shards, std::vector<load_job> & jobs, int n_threads,
                            std::atomic<size_t> & done_bytes, std::mutex * set_mu = nullptr)
{
    std::sort(jobs.begin(), jobs.end(), [](const load_job & a, const load_job & b)
    {
        if (a.shard != b.shard) return a.shard < b.shard;
        return a.file_off < b.file_off;
    });
    if (n_threads > (int) jobs.size()) n_threads = (int) jobs.size();
    if (n_threads < 1) n_threads = 1;

    std::atomic<size_t> cursor(0);
    std::atomic<bool> failed(false);
    const bool drop_cache = env_flag("TS_MUSE_SPARK_DROP_UPLOAD_CACHE", true);

    auto worker = [&]()
    {
        std::vector<FILE *> files(shards.paths.size(), nullptr);
        std::vector<uint8_t> staging;
        while (!failed.load(std::memory_order_relaxed))
        {
            const size_t i = cursor.fetch_add(1, std::memory_order_relaxed);
            if (i >= jobs.size()) break;
            const load_job & j = jobs[i];
            FILE *& f = files[(size_t) j.shard];
            if (!f)
            {
                f = fopen(shards.paths[(size_t) j.shard].c_str(), "rb");
                if (!f)
                {
                    fprintf(stderr, "[muse-spark] cannot open %s\n", shards.paths[(size_t) j.shard].c_str());
                    failed.store(true);
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
                fprintf(stderr, "[muse-spark] short read for %s at %zu\n", j.t->name, j.file_off);
                failed.store(true);
                break;
            }
            if (set_mu)
            {
                std::lock_guard<std::mutex> lk(*set_mu);
                ggml_backend_tensor_set(j.t, staging.data(), j.tensor_off, j.len);
            }
            else
            {
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
                    fprintf(stderr, "[muse-spark] TS_MUSE_SPARK_DROP_UPLOAD_CACHE has no effect on this platform; "
                                    "the checkpoint's pages stay in the page cache\n");
                });
#endif
            }
            done_bytes.fetch_add(j.len, std::memory_order_relaxed);
        }
        for (FILE * f : files) if (f) fclose(f);
    };

    std::vector<std::thread> pool;
    pool.reserve((size_t) n_threads);
    for (int i = 0; i < n_threads; i++) pool.emplace_back(worker);
    for (auto & th : pool) th.join();
    return !failed.load();
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

/// args: a [K, R, T] F32 -> dst I8 holding R*T rows of `userdata` (the weight's vec_dot type).
static void host_moe_quantize(ggml_tensor * dst, int ith, int nth, void * userdata)
{
    const ggml_tensor * a = dst->src[0];
    const ggml_type vdt = (ggml_type) (intptr_t) userdata;
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
static size_t kv_bytes_per_token_layer(const hparams & hp)
{
    return (size_t) hp.n_kv() * 2 /* K+V */ * sizeof(uint16_t);
}

/// The first `rows` cache rows of `src`, copied into `dst`.
struct row_copy { ggml_tensor * src; ggml_tensor * dst; int64_t rows; };

/// Copy cache rows device to device. The shapes match because every slot is
/// allocated with the same geometry; global rows grow per slot, so a source
/// and its destination may differ in length. Under tensor parallelism the
/// slices of a tensor live on every GPU and the meta buffer cannot copy, so
/// one graph of copies runs on the meta backend: each GPU copies its own
/// slices, nothing crosses the host.
static bool copy_cache_rows(model & m, const std::vector<row_copy> & copies)
{
    if (m.tp.n > 1)
    {
        const size_t n_nodes = copies.size() * 4 + 8;
        ggml_init_params p = { n_nodes * ggml_tensor_overhead() + ggml_graph_overhead_custom(n_nodes, false), nullptr, true };
        ggml_context * c = ggml_init(p);
        if (!c) return false;
        ggml_cgraph * gf = ggml_new_graph_custom(c, n_nodes, false);
        for (const row_copy & rc : copies)
        {
            const int64_t rows = std::min<int64_t>(rc.rows, std::min<int64_t>(rc.src->ne[1], rc.dst->ne[1]));
            if (rows <= 0) continue;
            ggml_tensor * vs = ggml_view_2d(c, rc.src, rc.src->ne[0], rows, rc.src->nb[1], 0);
            ggml_tensor * vd = ggml_view_2d(c, rc.dst, rc.dst->ne[0], rows, rc.dst->nb[1], 0);
            ggml_build_forward_expand(gf, ggml_cpy(c, vs, vd));
        }
        bool ok = true;
        if (ggml_graph_n_nodes(gf) > 0)
        {
            for (int i = 0; i < ggml_graph_n_nodes(gf); i++)
            {
                ggml_tensor * node = ggml_graph_node(gf, i);
                for (int k = 0; k < GGML_MAX_SRC; k++)
                    if (ggml_tensor * src = node->src[k]; src && src->view_src && !src->buffer) ggml_backend_view_init(src);
                if (node->view_src && !node->buffer) ggml_backend_view_init(node);
            }
            ok = ggml_backend_graph_compute(m.backends[0], gf) == GGML_STATUS_SUCCESS;
        }
        ggml_free(c);
        // The meta backend keeps per-GPU views only for the graph it computed
        // last; the cached graph is rebuilt on its next use.
        m.graph_cache.clear();
        return ok;
    }
    for (const row_copy & rc : copies)
    {
        const int64_t rows = std::min<int64_t>(rc.rows, std::min<int64_t>(rc.src->ne[1], rc.dst->ne[1]));
        if (rows <= 0) continue;
        if (rows == rc.src->ne[1] && rows == rc.dst->ne[1])
        {
            ggml_backend_tensor_copy(rc.src, rc.dst);
            continue;
        }
        ggml_init_params p = { 4 * ggml_tensor_overhead(), nullptr, true };
        ggml_context * c = ggml_init(p);
        ggml_tensor * vs = ggml_view_2d(c, rc.src, rc.src->ne[0], rows, rc.src->nb[1], 0);
        ggml_tensor * vd = ggml_view_2d(c, rc.dst, rc.dst->ne[0], rows, rc.dst->nb[1], 0);
        ggml_backend_view_init(vs);
        ggml_backend_view_init(vd);
        ggml_backend_tensor_copy(vs, vd);
        ggml_free(c);
    }
    return true;
}

/// Free and total memory of sched backend d: under tensor parallelism the
/// meta backend's device reports sums, but each GPU holds its own slices, so
/// the smallest GPU is what bounds an allocation.
static void dev_memory(const model & m, int d, size_t * free_b, size_t * total_b)
{
    *free_b = 0;
    *total_b = 0;
    if (m.tp.n > 1 && d == 0)
    {
        for (int j = 0; j < m.tp.n; j++)
        {
            size_t f = 0, t = 0;
            ggml_backend_dev_memory(m.tp.devs[j], &f, &t);
            if (j == 0 || f < *free_b) *free_b = f;
            if (j == 0 || t < *total_b) *total_b = t;
        }
        return;
    }
    ggml_backend_dev_memory(ggml_backend_get_device(m.backends[d]), free_b, total_b);
}

/// Bytes the largest GPU slice of `t` takes (all of it without tensor parallelism).
static size_t dev_nbytes(const model & m, const ggml_tensor * t)
{
    if (m.tp.n <= 1) return ggml_nbytes(t);
    size_t most = 0;
    for (int j = 0; j < m.tp.n; j++) most = std::max(most, tsg_mspark_tp::device_bytes(*m.tp.ud, t->name, t->type, t->ne, j));
    return most;
}

/// Allocate one part of a slot (0: the sliding-layer rings, 1: the global
/// layers with `rows` rows) in fresh per-device buffers and point the slot's
/// tensors at it. `admit` checks that each device keeps room for the largest
/// graph it has run. False, with nothing allocated or changed, on failure.
static bool slot_alloc_part(model & m, slot & s, int part, int64_t rows, bool admit)
{
    const int nl = m.hp.n_layer;
    std::vector<ggml_context *> ctxs((size_t) m.n_gpu + 1, nullptr);
    for (int d = 0; d <= m.n_gpu; d++)
    {
        ggml_init_params p = { (size_t) (2 * nl + 8) * ggml_tensor_overhead(), nullptr, true };
        ctxs[(size_t) d] = ggml_init(p);
    }
    std::vector<ggml_tensor *> k((size_t) nl, nullptr), v((size_t) nl, nullptr);
    for (int il = 0; il < nl; il++)
    {
        const bool swa = m.layers[(size_t) il].swa;
        if (swa != (part == 0)) continue;
        const int d = m.layers[(size_t) il].device;
        const int64_t r = swa ? m.swa_ring : rows;
        k[(size_t) il] = ggml_new_tensor_2d(ctxs[(size_t) d], GGML_TYPE_F16, m.hp.n_kv(), r);
        v[(size_t) il] = ggml_new_tensor_2d(ctxs[(size_t) d], GGML_TYPE_F16, m.hp.n_kv(), r);
        ggml_format_name(k[(size_t) il], "slot%d_k.%d", s.id, il);
        ggml_format_name(v[(size_t) il], "slot%d_v.%d", s.id, il);
    }

    if (admit)
    {
        // Each device must keep room to run the largest graph it has needed so
        // far, plus a reserve; otherwise a running request's next graph fails.
        // Cached graphs only cache: their compute buffers count as reclaimable,
        // and the cache is dropped when that is what it takes to fit.
        // The graph room is capped (TS_MUSE_SPARK_ADMIT_GRAPH_MB): one long
        // prefill's 2048-row graph would otherwise make every later slot evict
        // retained prefix state (shared-prompt checkpoints included) or be
        // refused, while a graph that later does not fit only frees retained
        // slots and runs that call in smaller ubatches.
        const size_t reserve = (size_t) env_long("TS_MUSE_SPARK_SLOT_HEADROOM_MB", 256) * 1024 * 1024;
        const size_t graph_cap = (size_t) std::max(0L, env_long("TS_MUSE_SPARK_ADMIT_GRAPH_MB", 512)) * 1024 * 1024;
        bool drop_graphs = false;
        for (int d = 0; d < m.n_gpu; d++)
        {
            if (!ggml_get_first_tensor(ctxs[(size_t) d])) continue;
            size_t free_b = 0, total_b = 0;
            dev_memory(m, d, &free_b, &total_b);
            if (!total_b) continue;
            size_t need = 0;
            for (ggml_tensor * t = ggml_get_first_tensor(ctxs[(size_t) d]); t; t = ggml_get_next_tensor(ctxs[(size_t) d], t))
                need += dev_nbytes(m, t) + 512;
            size_t cached = 0, largest = d < (int) m.max_graph_bytes.size() ? m.max_graph_bytes[(size_t) d] : 0;
            for (const auto & e : m.graph_cache)
                if (e->sched)
                {
                    const size_t b = ggml_backend_sched_get_buffer_size(e->sched, m.backends[d]);
                    cached += b;
                    largest = std::max(largest, b);
                }
            // The pipelined prefill's buffers are rebuilt on its next call.
            if (m.pipe.sched) cached += ggml_backend_sched_get_buffer_size(m.pipe.sched, m.backends[d]);
            largest = std::min(largest, graph_cap);
            if (free_b + cached < need + largest + reserve)
            {
                fprintf(stderr, "[muse-spark] no room for %s on device %d (needs %.0f MiB + %.0f MiB for graphs, "
                        "%.0f MiB free + %.0f MiB of cached graphs)\n", part == 0 ? "another sequence slot" : "longer K/V rows",
                        d, need / 1048576.0, (largest + reserve) / 1048576.0, free_b / 1048576.0, cached / 1048576.0);
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
        else ggml_backend_buffer_clear(bufs[(size_t) d], 0);
    }
    if (!ok)
    {
        for (auto b : bufs) if (b) ggml_backend_buffer_free(b);
        for (auto c : ctxs) if (c) ggml_free(c);
        fprintf(stderr, "[muse-spark] K/V allocation failed for slot %d\n", s.id);
        return false;
    }
    for (int d = 0; d <= m.n_gpu; d++)
    {
        if (!bufs[(size_t) d]) { if (ctxs[(size_t) d]) ggml_free(ctxs[(size_t) d]); continue; }
        m.slot_ctxs.push_back({ s.id, part, ctxs[(size_t) d], bufs[(size_t) d] });
    }
    for (int il = 0; il < nl; il++)
        if (k[(size_t) il]) { s.k[(size_t) il] = k[(size_t) il]; s.v[(size_t) il] = v[(size_t) il]; }
    if (part == 1) s.cap_g = rows;
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
    const int64_t want = std::max<int64_t>(m.kv_pad, env_long("TS_MUSE_SPARK_SLOT_INIT_ROWS", 4096));
    return std::min<int64_t>(m.n_ctx, pad_to(want, m.kv_pad));
}

/// Grow slot `s`'s global rows to at least `rows` (doubling), copying the
/// rows it holds. Graphs bound to the old rows are dropped.
static bool slot_reserve_rows(model & m, slot & s, int64_t rows)
{
    if (rows <= s.cap_g) return true;
    if (rows > m.n_ctx) return false;
    int64_t cap = std::max<int64_t>(rows, std::min<int64_t>(m.n_ctx, 2 * s.cap_g));
    cap = std::min<int64_t>(m.n_ctx, pad_to(cap, m.kv_pad));
    std::vector<ggml_tensor *> old_k = s.k, old_v = s.v;
    std::vector<model::slot_ctx> old;
    for (auto it = m.slot_ctxs.begin(); it != m.slot_ctxs.end(); )
    {
        if (it->slot_id == s.id && it->part == 1) { old.push_back(*it); it = m.slot_ctxs.erase(it); }
        else ++it;
    }
    const int64_t old_cap = s.cap_g;
    bool ok = slot_alloc_part(m, s, 1, cap, true);
    if (!ok)
    {
        // Cached graphs (the pipeline's included) and retained slots give their memory back first.
        m.graph_cache.clear();
        m.pipe.release();
        const int32_t self = s.id;
        while (!ok && reclaim_for_graph(m, &self, 1)) ok = slot_alloc_part(m, s, 1, cap, true);
        if (!ok) ok = slot_alloc_part(m, s, 1, cap, false);
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
        if (m.layers[(size_t) il].swa) continue;
        copies.push_back({ old_k[(size_t) il], s.k[(size_t) il], keep });
        copies.push_back({ old_v[(size_t) il], s.v[(size_t) il], keep });
    }
    const bool copied = copy_cache_rows(m, copies);
    if (!copied) fprintf(stderr, "[muse-spark] copying slot %d's K/V rows into its grown cache failed\n", s.id);
    for (auto & o : old)
    {
        if (o.buf) ggml_backend_buffer_free(o.buf);
        if (o.ctx) ggml_free(o.ctx);
    }
    drop_slot_graphs(m, s.id);
    if (debug_on()) fprintf(stderr, "[muse-spark] slot %d global rows %" PRId64 " -> %" PRId64 "\n", s.id, old_cap, cap);
    return copied;
}

static slot * slot_alloc(model & m, bool admit)
{
    auto s = std::unique_ptr<slot>(new slot());
    s->id = m.next_slot_id++;
    const int nl = m.hp.n_layer;
    s->k.assign((size_t) nl, nullptr);
    s->v.assign((size_t) nl, nullptr);
    s->swa_lab.assign((size_t) m.swa_ring, -1);
    if (!slot_alloc_part(m, *s, 0, 0, admit)) return nullptr;
    if (!slot_alloc_part(m, *s, 1, slot_init_rows(m), admit))
    {
        for (auto it = m.slot_ctxs.begin(); it != m.slot_ctxs.end(); )
        {
            if (it->slot_id != s->id) { ++it; continue; }
            if (it->buf) ggml_backend_buffer_free(it->buf);
            if (it->ctx) ggml_free(it->ctx);
            it = m.slot_ctxs.erase(it);
        }
        return nullptr;
    }
    slot * raw = s.get();
    m.slots[s->id] = std::move(s);
    return raw;
}

static void slot_free(model & m, int slot_id)
{
    drop_slot_graphs(m, slot_id);
    m.pipe.fail_u = m.pipe.fail_kv_g = m.pipe.fail_kv_s = 0;   // memory came back: retry
    m.slots.erase(slot_id);
    m.reclaimable.erase(std::remove(m.reclaimable.begin(), m.reclaimable.end(), slot_id), m.reclaimable.end());
    for (auto it = m.slot_ctxs.begin(); it != m.slot_ctxs.end(); )
    {
        if (it->slot_id != slot_id) { ++it; continue; }
        if (it->buf) ggml_backend_buffer_free(it->buf);
        if (it->ctx) ggml_free(it->ctx);
        it = m.slot_ctxs.erase(it);
    }
}

static bool reclaim_for_graph(model & m, const int32_t * reads, int n_reads)
{
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
        slot_free(m, id);
        m.reclaimed.push_back(id);
        fprintf(stderr, "[muse-spark] freed retained sequence slot %d for a graph that did not fit\n", id);
        return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// windows
// ---------------------------------------------------------------------------
/// Global layers read cache rows [0, n_kv_g).
static int64_t plan_n_kv_g(const model & m, int64_t p_end, int64_t pad)
{
    return std::min<int64_t>(pad_to(std::max<int64_t>(p_end, 1), pad), m.n_ctx);
}

/// Sliding layers read ring rows [0, n_kv_s). Until positions reach the
/// ring's length the rows ARE the positions and only the written prefix is
/// read; after that the whole ring is, and the mask picks the live window.
static void plan_swa(const model & m, int64_t p0, int64_t p_end, int64_t pad, int64_t * s0, int64_t * n_kv_s)
{
    (void) p0;
    *s0 = 0;
    *n_kv_s = p_end <= m.swa_ring ? std::min<int64_t>(pad_to(std::max<int64_t>(p_end, 1), pad), m.swa_ring)
                                  : m.swa_ring;
}

/// Record that positions [p0, p0 + n) are written to the slot's ring.
static void swa_write(const model & m, slot & s, int64_t p0, int64_t n)
{
    for (int64_t p = p0; p < p0 + n; p++) s.swa_lab[(size_t) (p % m.swa_ring)] = p;
}

/// Mask rows for queries at positions p0..p0+nt-1 over ring rows [0, n_rows).
static void fill_swa_mask(const model & m, const slot & s, std::vector<uint16_t> & out, int64_t n_rows, int64_t p0, int64_t nt)
{
    out.assign((size_t) (n_rows * nt), 0xFC00);
    for (int64_t t = 0; t < nt; t++)
    {
        const int64_t q = p0 + t;
        for (int64_t r = 0; r < n_rows; r++)
        {
            const int64_t lab = s.swa_lab[(size_t) r];
            if (lab >= 0 && lab <= q && q - lab < m.hp.n_swa) out[(size_t) (t * n_rows + r)] = 0;
        }
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

    builder(model & mm, graph_entry & ge) : m(mm), hp(mm.hp), g(ge), ctx(ge.ctx), gf(ge.gf) {}

    void pin(ggml_tensor * t, int dev)
    {
        if (dev < 0 || dev > m.n_gpu) return;
        if (t->op != GGML_OP_NONE && !ggml_backend_supports_op(m.backends[dev], t)) return;
        ggml_backend_sched_set_tensor_backend(g.sched, t, m.backends[dev]);
    }

    ggml_tensor * new_input(ggml_type type, int64_t ne0, int64_t ne1, const char * name, int dev)
    {
        ggml_tensor * t = ne1 > 0 ? ggml_new_tensor_2d(ctx, type, ne0, ne1) : ggml_new_tensor_1d(ctx, type, ne0);
        ggml_set_input(t);
        ggml_set_name(t, name);
        pin(t, dev);
        return t;
    }

    ggml_tensor * rms(ggml_tensor * x, ggml_tensor * w, float eps)
    {
        x = ggml_rms_norm(ctx, x, eps);
        if (w) x = ggml_mul(ctx, x, w);
        return x;
    }

    ggml_tensor * rope(ggml_tensor * x, ggml_tensor * pos)
    {
        return ggml_rope_ext(ctx, x, pos, nullptr, hp.n_rot, GGML_ROPE_TYPE_NORMAL, hp.n_ctx_train,
                             hp.rope_base_swa, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    }

    /// Q, K, V and the attention gate for `cur` [n_embd, n].
    void project_qkvg(const layer & L, ggml_tensor * cur, int64_t n,
                      ggml_tensor ** q, ggml_tensor ** k, ggml_tensor ** v, ggml_tensor ** gate)
    {
        const int64_t nq = hp.n_q(), nkv = hp.n_kv();
        if (L.wqkvg)
        {
            ggml_tensor * all = ggml_mul_mat(ctx, L.wqkvg, cur);      // [nq + 2 nkv + nq, n]
            const size_t es = ggml_element_size(all);
            *q = ggml_view_3d(ctx, all, hp.head_dim, hp.n_head, n, hp.head_dim * es, all->nb[1], 0);
            *k = ggml_view_3d(ctx, all, hp.head_dim, hp.n_head_kv, n, hp.head_dim * es, all->nb[1], (size_t) nq * es);
            *v = ggml_view_3d(ctx, all, hp.head_dim, hp.n_head_kv, n, hp.head_dim * es, all->nb[1], (size_t) (nq + nkv) * es);
            *gate = ggml_view_2d(ctx, all, nq, n, all->nb[1], (size_t) (nq + 2 * nkv) * es);
        }
        else
        {
            *q = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, L.wq, cur), hp.head_dim, hp.n_head, n);
            *k = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, L.wk, cur), hp.head_dim, hp.n_head_kv, n);
            *v = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, L.wv, cur), hp.head_dim, hp.n_head_kv, n);
            *gate = ggml_mul_mat(ctx, L.wg, cur);
        }
    }

    /// One attention over cache rows [first, first + n_kv) of slot `s`.
    ggml_tensor * attend(slot & s, int il, ggml_tensor * q /*[hd, n_head, n]*/, int64_t first, int64_t n_kv,
                         ggml_tensor * mask)
    {
        ggml_tensor * kc = s.k[(size_t) il];
        ggml_tensor * vc = s.v[(size_t) il];
        const size_t row = kc->nb[1];
        ggml_tensor * k = ggml_view_3d(ctx, kc, hp.head_dim, hp.n_head_kv, n_kv,
                                       ggml_row_size(kc->type, hp.head_dim), row, (size_t) first * row);
        ggml_tensor * v = ggml_view_3d(ctx, vc, hp.head_dim, hp.n_head_kv, n_kv,
                                       ggml_row_size(vc->type, hp.head_dim), row, (size_t) first * row);
        ggml_tensor * qp = ggml_permute(ctx, q, 0, 2, 1, 3);           // [hd, n, n_head]
        ggml_tensor * kp = ggml_permute(ctx, k, 0, 2, 1, 3);           // [hd, n_kv, n_head_kv]
        ggml_tensor * vp = ggml_permute(ctx, v, 0, 2, 1, 3);
        ggml_tensor * fa = ggml_flash_attn_ext(ctx, qp, kp, vp, mask, hp.kq_scale(), 0.0f, 0.0f);
        ggml_prec_set_acc(fa, GGML_PREC_F32);
        return fa;                                                     // [hd, n_head, n]
    }

    /// Attention block for one slot and `n` tokens; returns the projected output [n_embd, n].
    ggml_tensor * build_attention(int il, ggml_tensor * cur, ggml_tensor * pos, ggml_tensor * kv_idx,
                                  ggml_tensor * kv_idx_s, slot & s, ggml_tensor * mask_g, ggml_tensor * mask_s, int64_t n)
    {
        const layer & L = m.layers[(size_t) il];
        ggml_tensor *q, *k, *v, *gate;
        project_qkvg(L, cur, n, &q, &k, &v, &gate);

        q = rms(q, L.q_norm, hp.rms_eps);
        k = rms(k, L.k_norm, hp.rms_eps);
        if (L.swa)
        {
            q = rope(q, pos);
            k = rope(k, pos);
        }

        ggml_tensor * kc = s.k[(size_t) il];
        ggml_tensor * vc = s.v[(size_t) il];
        if (!ggml_is_contiguous(k)) k = ggml_cont(ctx, k);
        if (!ggml_is_contiguous(v)) v = ggml_cont(ctx, v);
        ggml_tensor * rows = L.swa ? kv_idx_s : kv_idx;
        ggml_build_forward_expand(gf, ggml_set_rows(ctx, kc, ggml_reshape_2d(ctx, k, hp.n_kv(), n), rows));
        ggml_build_forward_expand(gf, ggml_set_rows(ctx, vc, ggml_reshape_2d(ctx, v, hp.n_kv(), n), rows));

        ggml_tensor * out = L.swa ? attend(s, il, q, 0, g.n_kv_s, mask_s)
                                  : attend(s, il, q, 0, g.n_kv_g, mask_g);
        return finish_attention(L, out, gate, n);
    }

    /// per-head norm, sigmoid gate, output projection, post norm.
    ggml_tensor * finish_attention(const layer & L, ggml_tensor * out /*[hd, n_head, n]*/, ggml_tensor * gate, int64_t n)
    {
        out = ggml_rms_norm(ctx, out, hp.rms_eps);                      // per head, no weight
        out = ggml_reshape_2d(ctx, out, hp.n_q(), n);
        if (!ggml_is_contiguous(gate)) gate = ggml_cont(ctx, gate);
        out = ggml_mul(ctx, out, ggml_sigmoid(ctx, gate));
        out = ggml_mul_mat(ctx, L.wo, out);
        return ggml_rms_norm(ctx, out, hp.post_eps);
    }

    /// Activations quantized for host_moe_matmul, shared by the stacks that read them.
    struct quantized { ggml_type type = GGML_TYPE_COUNT; ggml_tensor * t = nullptr; };

    /// One expert stack applied to `a`: ggml_mul_mat_id, or host_moe_matmul for a
    /// small batch on a host stack in ggml's plain layout (not CPU_REPACK).
    ggml_tensor * expert_matmul(ggml_tensor * w, bool on_host, ggml_tensor * a, ggml_tensor * ids, int64_t n, quantized & aq)
    {
        if (!on_host || n < 2 || n > m.host_moe_max_tokens || ids->ne[0] * n > HOST_MOE_MAX_PICKS
            || !w->buffer || !ggml_backend_buffer_is_host(w->buffer) || w->buffer == m.repack_buf)
            return ggml_mul_mat_id(ctx, w, a, ids);
        const ggml_type vdt = ggml_get_type_traits_cpu(w->type)->vec_dot_type;
        if (aq.type != vdt)
        {
            ggml_tensor * args[1] = { a };
            aq.t = ggml_custom_4d(ctx, GGML_TYPE_I8, (int64_t) ggml_row_size(vdt, a->ne[0]), a->ne[1], a->ne[2], 1,
                                  args, 1, host_moe_quantize, GGML_N_TASKS_MAX, (void *) (intptr_t) vdt);
            aq.type = vdt;
        }
        ggml_tensor * args[3] = { w, aq.t, ids };
        return ggml_custom_4d(ctx, GGML_TYPE_F32, w->ne[1], ids->ne[0], ids->ne[1], 1, args, 3,
                              host_moe_matmul, GGML_N_TASKS_MAX, nullptr);
    }

    ggml_tensor * build_moe(int il, ggml_tensor * cur /*[n_embd, n] ffn-normed*/, int64_t n)
    {
        const layer & L = m.layers[(size_t) il];
        const int64_t n_expert = hp.n_expert;
        const int64_t n_used = hp.n_expert_used;

        ggml_tensor * moe_in = ggml_mul_mat(ctx, L.lat_down, cur);                 // [latent, n]
        moe_in = rms(moe_in, L.pre_exp_norm, hp.rms_eps);

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

        ggml_tensor * x = ggml_reshape_3d(ctx, moe_in, hp.latent, 1, n);
        quantized xq;
        ggml_tensor * up = expert_matmul(L.up_exps, L.cpu_stack[1], x, selected, n, xq);    // [ff, n_used, n]
        ggml_tensor * gt = expert_matmul(L.gate_exps, L.cpu_stack[0], x, selected, n, xq);
        ggml_tensor * h = ggml_swiglu_split(ctx, gt, up);
        quantized hq;
        ggml_tensor * experts = expert_matmul(L.down_exps, L.cpu_stack[2], h, selected, n, hq);  // [latent, n_used, n]

        // Normalize every expert's output BEFORE its routing weight applies.
        experts = ggml_mul(ctx, experts, L.post_exp_norm);
        experts = ggml_rms_norm(ctx, experts, hp.post_eps);
        experts = ggml_mul(ctx, experts, weights);
        ggml_build_forward_expand(gf, experts);

        ggml_tensor * out = nullptr;
        for (int64_t e = 0; e < n_used; e++)
        {
            ggml_tensor * view = ggml_view_2d(ctx, experts, hp.latent, n, experts->nb[2], (size_t) e * experts->nb[1]);
            ggml_build_forward_expand(gf, view);
            out = out ? ggml_add(ctx, out, view) : view;
        }
        if (n_used == 1) out = ggml_cont(ctx, out);

        out = ggml_mul_mat(ctx, L.lat_up, out);                                    // [n_embd, n]
        out = ggml_mul(ctx, out, L.post_ffw_norm);
        return ggml_rms_norm(ctx, out, hp.post_eps);
    }

    ggml_tensor * residual(ggml_tensor * x, ggml_tensor * alpha, ggml_tensor * branch, ggml_tensor * beta)
    {
        return ggml_add(ctx, ggml_mul(ctx, x, alpha), ggml_mul(ctx, branch, beta));
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
        return rms(emb, nullptr, hp.rms_eps);
    }

    ggml_tensor * head(ggml_tensor * x /*[n_embd, rows]*/)
    {
        x = rms(x, m.output_norm, hp.rms_eps);
        ggml_tensor * logits = ggml_mul_mat(ctx, m.output, x);
        logits = ggml_scale(ctx, logits, hp.logit_scale);
        if (hp.softcap > 0.0f)
        {
            logits = ggml_scale(ctx, logits, 1.0f / hp.softcap);
            logits = ggml_tanh(ctx, logits);
            logits = ggml_scale(ctx, logits, hp.softcap);
        }
        return logits;
    }

    // ---- one slot, nt tokens ---------------------------------------------
    void build_single(slot & s)
    {
        const int64_t nt = g.nt;
        auto used = used_devices();
        for (int d = 0; d <= m.n_gpu; d++)
        {
            if (!used[(size_t) d]) continue;
            char nb[64];
            snprintf(nb, sizeof(nb), "inp_pos.%d", d);
            g.inp.pos[d] = new_input(GGML_TYPE_I32, nt, 0, nb, d);
            snprintf(nb, sizeof(nb), "inp_kv_idx.%d", d);
            g.inp.kv_idx[d] = new_input(GGML_TYPE_I64, nt, 0, nb, d);
            bool has_g = false, has_s = false;
            for (int il = 0; il < hp.n_layer; il++)
                if (m.layers[(size_t) il].device == d) (m.layers[(size_t) il].swa ? has_s : has_g) = true;
            if (has_g)
            {
                snprintf(nb, sizeof(nb), "mask_g.%d", d);
                g.inp.mask_g[d] = new_input(GGML_TYPE_F16, g.n_kv_g, nt, nb, d);
            }
            if (has_s)
            {
                snprintf(nb, sizeof(nb), "mask_s.%d", d);
                g.inp.mask_s[d] = new_input(GGML_TYPE_F16, g.n_kv_s, nt, nb, d);
                snprintf(nb, sizeof(nb), "inp_kv_idx_s.%d", d);
                g.inp.kv_idx_s[d] = new_input(GGML_TYPE_I64, nt, 0, nb, d);
            }
        }

        ggml_tensor * x = embed(nt);
        const int last = hp.n_layer - 1;
        if (g.want_logits)
            g.inp.out_ids = new_input(GGML_TYPE_I32, g.n_out, 0, "inp_out_ids", m.layers[(size_t) last].device);

        for (int il = 0; il < hp.n_layer; il++)
        {
            const layer & L = m.layers[(size_t) il];
            const int d = L.device;
            ggml_tensor * attn = build_attention(il, rms(x, L.attn_norm, hp.rms_eps), g.inp.pos[d],
                                                 g.inp.kv_idx[d], g.inp.kv_idx_s[d], s, g.inp.mask_g[d], g.inp.mask_s[d], nt);
            ggml_tensor * res = x;
            int64_t rows = nt;
            if (il == last)
            {
                if (!g.want_logits)
                {
                    // A non-final prefill chunk only leaves its K/V behind; the last
                    // layer's FFN and the head would be computed for nobody.
                    ggml_build_forward_expand(gf, attn);
                    return;
                }
                if (g.n_out < nt)
                {
                    attn = ggml_get_rows(ctx, attn, g.inp.out_ids);
                    res = ggml_get_rows(ctx, res, g.inp.out_ids);
                    rows = g.n_out;
                }
            }
            ggml_tensor * ffn_inp = residual(res, L.attn_alpha, attn, L.attn_beta);
            ggml_tensor * ffn = build_moe(il, rms(ffn_inp, L.ffn_norm, hp.rms_eps), rows);
            x = residual(ffn_inp, L.ffn_alpha, ffn, L.ffn_beta);
        }

        ggml_tensor * logits = head(x);
        ggml_set_output(logits);
        ggml_set_name(logits, "logits");
        g.logits = logits;
        ggml_build_forward_expand(gf, logits);
    }

    // ---- several slots, one segment of consecutive tokens each ------------
    // Every projection, the router and the experts run once over all T rows;
    // only the cache writes and the attention are built per segment, because
    // each segment reads its own slot's history. Each segment's masks are a
    // contiguous block of one flat input, so the views stay contiguous.
    static void seg_mask_sizes(const graph_entry & g, int64_t * total_g, int64_t * total_s)
    {
        *total_g = 0; *total_s = 0;
        for (const bd_token & b : g.bd) { *total_g += b.n_kv_g * b.len; *total_s += b.n_kv_s * b.len; }
    }

    void build_segments()
    {
        const int64_t T = g.nt;
        const int64_t S = (int64_t) g.bd.size();
        auto used = used_devices();
        int64_t total_g = 0, total_s = 0;
        seg_mask_sizes(g, &total_g, &total_s);
        for (int d = 0; d <= m.n_gpu; d++)
        {
            if (!used[(size_t) d]) continue;
            char nb[64];
            snprintf(nb, sizeof(nb), "seg_pos.%d", d);
            g.inp.pos[d] = new_input(GGML_TYPE_I32, T, 0, nb, d);
            snprintf(nb, sizeof(nb), "seg_kv_idx.%d", d);
            g.inp.kv_idx[d] = new_input(GGML_TYPE_I64, T, 0, nb, d);
            bool has_g = false, has_s = false;
            for (int il = 0; il < hp.n_layer; il++)
                if (m.layers[(size_t) il].device == d) (m.layers[(size_t) il].swa ? has_s : has_g) = true;
            if (has_g)
            {
                snprintf(nb, sizeof(nb), "seg_mask_g.%d", d);
                g.inp.mask_g[d] = new_input(GGML_TYPE_F16, total_g, 0, nb, d);
            }
            if (has_s)
            {
                snprintf(nb, sizeof(nb), "seg_mask_s.%d", d);
                g.inp.mask_s[d] = new_input(GGML_TYPE_F16, total_s, 0, nb, d);
                snprintf(nb, sizeof(nb), "seg_kv_idx_s.%d", d);
                g.inp.kv_idx_s[d] = new_input(GGML_TYPE_I64, T, 0, nb, d);
            }
        }
        const int last = hp.n_layer - 1;
        const bool prune = S < T;   // decode-only graphs already have one row per segment
        if (prune) g.inp.out_ids = new_input(GGML_TYPE_I32, S, 0, "seg_out_ids", m.layers[(size_t) last].device);

        ggml_tensor * x = embed(T);
        for (int il = 0; il < hp.n_layer; il++)
        {
            const layer & L = m.layers[(size_t) il];
            const int d = L.device;
            ggml_tensor * cur = rms(x, L.attn_norm, hp.rms_eps);
            ggml_tensor *q, *k, *v, *gate;
            project_qkvg(L, cur, T, &q, &k, &v, &gate);
            q = rms(q, L.q_norm, hp.rms_eps);
            k = rms(k, L.k_norm, hp.rms_eps);
            if (L.swa)
            {
                q = rope(q, g.inp.pos[d]);
                k = rope(k, g.inp.pos[d]);
            }
            ggml_tensor * k2 = ggml_reshape_2d(ctx, ggml_cont(ctx, k), hp.n_kv(), T);
            ggml_tensor * v2 = ggml_reshape_2d(ctx, ggml_cont(ctx, v), hp.n_kv(), T);
            if (!ggml_is_contiguous(q)) q = ggml_cont(ctx, q);

            ggml_tensor * cat = nullptr;
            int64_t row = 0, off_g = 0, off_s = 0;
            for (int64_t i = 0; i < S; i++)
            {
                const bd_token & B = g.bd[(size_t) i];
                slot & sl = *m.slots.at(B.slot_id);
                ggml_tensor * idx_src = L.swa ? g.inp.kv_idx_s[d] : g.inp.kv_idx[d];
                ggml_tensor * idx = ggml_view_1d(ctx, idx_src, B.len, (size_t) row * sizeof(int64_t));
                ggml_tensor * ki = ggml_view_2d(ctx, k2, hp.n_kv(), B.len, k2->nb[1], (size_t) row * k2->nb[1]);
                ggml_tensor * vi = ggml_view_2d(ctx, v2, hp.n_kv(), B.len, v2->nb[1], (size_t) row * v2->nb[1]);
                ggml_build_forward_expand(gf, ggml_set_rows(ctx, sl.k[(size_t) il], ki, idx));
                ggml_build_forward_expand(gf, ggml_set_rows(ctx, sl.v[(size_t) il], vi, idx));

                ggml_tensor * qi = ggml_view_3d(ctx, q, hp.head_dim, hp.n_head, B.len, q->nb[1], q->nb[2], (size_t) row * q->nb[2]);
                ggml_tensor * out;
                if (L.swa)
                {
                    ggml_tensor * ms = g.inp.mask_s[d];
                    ggml_tensor * mask = ggml_view_2d(ctx, ms, B.n_kv_s, B.len, (size_t) B.n_kv_s * sizeof(uint16_t),
                                                      (size_t) off_s * sizeof(uint16_t));
                    out = attend(sl, il, qi, 0, B.n_kv_s, mask);
                }
                else
                {
                    ggml_tensor * mg = g.inp.mask_g[d];
                    ggml_tensor * mask = ggml_view_2d(ctx, mg, B.n_kv_g, B.len, (size_t) B.n_kv_g * sizeof(uint16_t),
                                                      (size_t) off_g * sizeof(uint16_t));
                    out = attend(sl, il, qi, 0, B.n_kv_g, mask);
                }
                cat = cat ? ggml_concat(ctx, cat, out, 2) : out;               // [hd, n_head, rows so far]
                row += B.len;
                off_g += B.n_kv_g * B.len;
                off_s += B.n_kv_s * B.len;
            }
            ggml_tensor * attn = finish_attention(L, cat, gate, T);
            ggml_tensor * res = x;
            int64_t rows = T;
            if (il == last && prune)
            {
                attn = ggml_get_rows(ctx, attn, g.inp.out_ids);
                res = ggml_get_rows(ctx, res, g.inp.out_ids);
                rows = S;
            }
            ggml_tensor * ffn_inp = residual(res, L.attn_alpha, attn, L.attn_beta);
            ggml_tensor * ffn = build_moe(il, rms(ffn_inp, L.ffn_norm, hp.rms_eps), rows);
            x = residual(ffn_inp, L.ffn_alpha, ffn, L.ffn_beta);
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
    if (e.kind != k.kind || e.nt != k.nt || e.want_logits != k.want_logits) return false;
    if (e.kind == 0)
        return e.slot_id == k.slot_id && e.n_kv_g == k.n_kv_g && e.s0 == k.s0 && e.n_kv_s == k.n_kv_s &&
               e.n_out == k.n_out && e.n_ovr == k.n_ovr;
    if (e.bd.size() != k.bd.size()) return false;
    for (size_t i = 0; i < e.bd.size(); i++)
    {
        const bd_token & a = e.bd[i];
        const bd_token & b = k.bd[i];
        if (a.slot_id != b.slot_id || a.len != b.len || a.n_kv_g != b.n_kv_g || a.s0 != b.s0 || a.n_kv_s != b.n_kv_s) return false;
    }
    return true;
}

/// Under --tp every layer op runs on ggml's meta backend: an op the GPUs do not
/// support would be scheduled on the CPU and come back as a whole (mirrored)
/// tensor, which the meta backend cannot combine with its GPU slices (it
/// aborts). Checked before allocation so the call fails with a reason instead.
/// Only the token-row gather (from the host table) belongs on the CPU.
static bool tp_check_graph(const model & m, ggml_cgraph * gf)
{
    if (m.tp.n <= 1) return true;
    for (int i = 0; i < ggml_graph_n_nodes(gf); i++)
    {
        ggml_tensor * node = ggml_graph_node(gf, i);
        if (node->op == GGML_OP_NONE || node->op == GGML_OP_VIEW || node->op == GGML_OP_RESHAPE ||
            node->op == GGML_OP_PERMUTE || node->op == GGML_OP_TRANSPOSE) continue;
        bool host_side = false;
        for (int k = 0; k < GGML_MAX_SRC && !host_side; k++)
            if (const ggml_tensor * src = node->src[k]; src && src == m.tok_embd) host_side = true;
        if (host_side) continue;
        if (ggml_nelements(node) == 0 || !ggml_backend_supports_op(m.backends[0], node))
        {
            tsg::set_last_error(std::string("[muse-spark] --tp: ") + ggml_op_desc(node) + " (" + node->name +
                                ") cannot run on the GPUs at this shape, and a tensor-parallel layer cannot fall back to the CPU");
            fprintf(stderr, "[muse-spark] --tp: %s (%s, [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]) cannot run on the "
                    "GPUs at this shape, and a tensor-parallel layer cannot fall back to the CPU\n", ggml_op_desc(node), node->name,
                    node->ne[0], node->ne[1], node->ne[2], node->ne[3]);
            return false;
        }
    }
    return true;
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

    // Under --tp two whole-graph compute buffers would sit on every GPU: the
    // one graph kept goes before the next is built.
    if (m.tp.n > 1) m.graph_cache.clear();

    // Reads: the slots this graph touches (never reclaimed for it).
    std::vector<int32_t> reads;
    if (key->kind == 0) reads.push_back(key->slot_id);
    for (const bd_token & b : key->bd) reads.push_back(b.slot_id);

    // A prefill graph holds compute buffers sized for its ubatch on every
    // device; keep only one of them cached next to the small decode graphs.
    const int64_t large = 64;
    if (key->nt > large)
    {
        for (auto it = m.graph_cache.begin(); it != m.graph_cache.end(); )
            it = (*it)->nt > large ? m.graph_cache.erase(it) : std::next(it);
    }

    size_t nodes_per_layer = 160;
    if (key->kind == 1) nodes_per_layer += key->bd.size() * 24;
    if (const long v = env_long("TS_MUSE_SPARK_NODES_PER_LAYER", 0); v > 0) nodes_per_layer = (size_t) v;
    const size_t n_nodes = (size_t) m.hp.n_layer * nodes_per_layer + 1024;

    for (;;)
    {
        std::unique_ptr<graph_entry> e(new graph_entry());
        e->kind = key->kind; e->slot_id = key->slot_id; e->nt = key->nt; e->p0 = key->p0;
        e->n_kv_g = key->n_kv_g; e->s0 = key->s0; e->n_kv_s = key->n_kv_s; e->n_out = key->n_out;
        e->n_ovr = key->n_ovr; e->want_logits = key->want_logits; e->bd = key->bd;

        ggml_init_params gp = { ggml_tensor_overhead() * n_nodes + ggml_graph_overhead_custom(n_nodes, false), nullptr, true };
        e->ctx = ggml_init(gp);
        if (!e->ctx) return nullptr;
        e->gf = ggml_new_graph_custom(e->ctx, n_nodes, false);
        e->sched = ggml_backend_sched_new(m.sched_backends, m.sched_bufts, m.n_sched_backends, n_nodes, false, m.op_offload);
        if (!e->sched) return nullptr;

        builder b(m, *e);
        if (e->kind == 0) b.build_single(*m.slots.at(e->slot_id));
        else              b.build_segments();
        if (!tp_check_graph(m, e->gf)) return nullptr;

        const auto t0 = std::chrono::steady_clock::now();
        if (ggml_backend_sched_alloc_graph(e->sched, e->gf))
        {
            m.n_graph_builds++;
            m.max_graph_bytes.resize((size_t) m.n_gpu + 1, 0);
            for (int d = 0; d <= m.n_gpu; d++)
                m.max_graph_bytes[(size_t) d] = std::max(m.max_graph_bytes[(size_t) d],
                                                         ggml_backend_sched_get_buffer_size(e->sched, m.backends[d]));
            if (debug_on())
            {
                std::string bufs;
                for (int d = 0; d <= m.n_gpu; d++)
                {
                    char b[48];
                    snprintf(b, sizeof(b), "%s%.0f", d ? "/" : "", ggml_backend_sched_get_buffer_size(e->sched, m.backends[d]) / 1048576.0);
                    bufs += b;
                }
                fprintf(stderr, "[muse-spark] built graph kind=%d nt=%" PRId64 " n_kv_g=%" PRId64 " s0=%" PRId64 " n_kv_s=%" PRId64
                        " nodes=%d splits=%d alloc=%.1f ms compute MiB=%s\n", e->kind, e->nt, e->n_kv_g, e->s0, e->n_kv_s,
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
            fprintf(stderr, "[muse-spark] failed to allocate a graph (kind=%d nt=%" PRId64 ")\n", key->kind, key->nt);
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

static constexpr uint16_t F16_NEG_INF = 0xFC00;

/// Causal (+ sliding) mask rows for queries at positions p0..p0+nt-1 over
/// cache rows [first, first + n_kv).
static void fill_mask(std::vector<uint16_t> & out, int64_t first, int64_t n_kv, int64_t p0, int64_t nt, int64_t window)
{
    out.assign((size_t) (n_kv * nt), F16_NEG_INF);
    for (int64_t t = 0; t < nt; t++)
    {
        const int64_t q = p0 + t;
        const int64_t lo = window > 0 ? std::max<int64_t>(first, q - window + 1) : first;
        const int64_t hi = std::min<int64_t>(first + n_kv - 1, q);
        for (int64_t j = lo; j <= hi; j++) out[(size_t) (t * n_kv + (j - first))] = 0;
    }
}

// TS_MUSE_SPARK_PROFILE=N: time every node (synchronously, so absolute totals
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
    static const long prof_every = env_long("TS_MUSE_SPARK_PROFILE", 0);
    const auto t0 = std::chrono::steady_clock::now();
    if (prof_every > 0)
    {
        g_prof.sched = g.sched;
        g_prof.last = t0;
        ggml_backend_sched_set_eval_callback(g.sched, prof_eval, nullptr);
    }
    if (ggml_backend_sched_graph_compute_async(g.sched, g.gf) != GGML_STATUS_SUCCESS)
    {
        fprintf(stderr, "[muse-spark] graph compute failed\n");
        return false;
    }
    ggml_backend_sched_synchronize(g.sched);
    if (prof_every > 0)
    {
        ggml_backend_sched_set_eval_callback(g.sched, nullptr, nullptr);
        g_prof.wall_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (++g_prof.calls % prof_every == 0)
        {
            std::vector<std::pair<double, std::string>> rows;
            for (auto & kv : g_prof.ms) rows.push_back({ kv.second, kv.first });
            std::sort(rows.rbegin(), rows.rend());
            fprintf(stderr, "[muse-spark profile] %" PRId64 " graphs, %.1f ms/graph wall (nt=%" PRId64 ", kind=%d)\n",
                    g_prof.calls, g_prof.wall_ms / g_prof.calls, g.nt, g.kind);
            for (size_t i = 0; i < rows.size() && i < 24; i++)
                fprintf(stderr, "  %8.2f ms/graph  %6" PRId64 " nodes  %s\n", rows[i].first / g_prof.calls,
                        g_prof.count[rows[i].second] / g_prof.calls, rows[i].second.c_str());
            g_prof = profiler();
        }
    }
    (void) m;
    return true;
}

/// One ubatch of the active slot. `ovr` rows (vision) replace token embeddings.
static bool forward_ubatch(model & m, const int32_t * tokens, int64_t nt, bool want_logits, bool all_rows,
                           float * logits_out, const float * ovr_rows, const int64_t * ovr_idx, int64_t n_ovr)
{
    slot & s = *m.active_slot;
    const int64_t p0 = s.n_past;
    if (p0 + nt > m.n_ctx)
    {
        fprintf(stderr, "[muse-spark] context overflow: %" PRId64 " + %" PRId64 " > %d\n", p0, nt, m.n_ctx);
        return false;
    }

    if (!slot_reserve_rows(m, s, plan_n_kv_g(m, p0 + nt, m.kv_pad)))
    {
        fprintf(stderr, "[muse-spark] no device memory for %" PRId64 " K/V rows of slot %d\n", p0 + nt, s.id);
        return false;
    }
    std::unique_ptr<graph_entry> key(new graph_entry());
    key->kind = 0;
    key->slot_id = s.id;
    key->nt = nt;
    key->p0 = p0;
    key->n_kv_g = plan_n_kv_g(m, p0 + nt, m.kv_pad);
    plan_swa(m, p0, p0 + nt, m.kv_pad, &key->s0, &key->n_kv_s);
    key->want_logits = want_logits;
    key->n_out = want_logits ? (all_rows ? nt : 1) : 0;
    key->n_ovr = n_ovr;

    bool reused = false;
    graph_entry * g = acquire_graph(m, std::move(key), &reused);
    if (!g) return false;

    m.h_tokens.assign(tokens, tokens + nt);
    m.h_pos.resize((size_t) nt);
    m.h_kv_idx.resize((size_t) nt);
    m.h_kv_idx_s.resize((size_t) nt);
    for (int64_t i = 0; i < nt; i++)
    {
        m.h_pos[(size_t) i] = (int32_t) (p0 + i);
        m.h_kv_idx[(size_t) i] = p0 + i;
        m.h_kv_idx_s[(size_t) i] = (p0 + i) % m.swa_ring;
    }
    // Rows are labeled before the mask is built: this ubatch writes them
    // before any of its attention reads them.
    swa_write(m, s, p0, nt);

    set_i32(g->inp.tokens, m.h_tokens.data(), (size_t) nt);
    bool need_g = false, need_s = false;
    for (int d = 0; d <= m.n_gpu; d++) { need_g |= live(g->inp.mask_g[d]); need_s |= live(g->inp.mask_s[d]); }
    if (need_g) fill_mask(m.h_mask_g, 0, g->n_kv_g, p0, nt, 0);
    if (need_s) fill_swa_mask(m, s, m.h_mask_s, g->n_kv_s, p0, nt);
    for (int d = 0; d <= m.n_gpu; d++)
    {
        set_i32(g->inp.pos[d], m.h_pos.data(), (size_t) nt);
        if (live(g->inp.kv_idx[d])) ggml_backend_tensor_set(g->inp.kv_idx[d], m.h_kv_idx.data(), 0, (size_t) nt * sizeof(int64_t));
        if (live(g->inp.kv_idx_s[d])) ggml_backend_tensor_set(g->inp.kv_idx_s[d], m.h_kv_idx_s.data(), 0, (size_t) nt * sizeof(int64_t));
        if (live(g->inp.mask_g[d])) ggml_backend_tensor_set(g->inp.mask_g[d], m.h_mask_g.data(), 0, m.h_mask_g.size() * 2);
        if (live(g->inp.mask_s[d])) ggml_backend_tensor_set(g->inp.mask_s[d], m.h_mask_s.data(), 0, m.h_mask_s.size() * 2);
    }
    if (want_logits)
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

    if (!compute(m, *g)) return false;
    s.n_past += nt;
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
// runs its splits in order on its own stream, so micro-batch j+1 reads the K/V
// rows j wrote without a host wait. llama.cpp's layer split does the same.
//
// What keeps the in-flight micro-batches apart:
//  * the scheduler is reserved for the call's largest micro-batch (its last
//    K/V window) before anything is in flight. Every micro-batch builds the same
//    graph topology (one logits row each), only with a window that grows, so
//    ggml's allocator keeps every node and copy slot at its reserved address
//    and never reallocates mid-call (a reallocation would drain the pipeline);
//  * inputs pinned to a GPU are uploaded with tensor_set_async on that GPU's
//    stream, from a pinned staging ring the host refills only after the
//    uploads that read a slot have run;
//  * ggml copies a GPU's output into the next GPU's copy slot on the PRODUCER's
//    stream, while its own guard waits on the consumer's. Before micro-batch j
//    every producer therefore waits for its consumer to finish micro-batch
//    j - n_copies, the last reader of that slot.
// A call of n tokens is cut into k = ceil(n / TS_MUSE_SPARK_PIPELINE_UBATCH)
// equal micro-batches; the last is padded with fewer than k copies of the final
// token, so no partial micro-batch has to cross every GPU alone. The padding
// writes K/V rows past the call's end: global rows there are rewritten before
// any query reads them, and the ring rows it overwrites held positions outside
// every later window; their labels are cleared, so a rewind that needs them is
// refused instead of reading them. Near the context limit, where it cannot
// pad, the call keeps floor(n / k)-row micro-batches and runs the few left-over
// tokens on the sequential path.
// A micro-batch's masks (global + sliding window, F16) stage at most this much:
// a context long enough to need more prefills one micro-batch at a time.
static constexpr size_t PIPE_MASK_BUDGET = (size_t) 96 << 20;
// Adaptive micro-batches (TS_MUSE_SPARK_PIPELINE_UBATCH unset) have at least
// PIPE_MIN_UBATCH rows, and shrink to PIPE_MASK_MIN_UBATCH only where a long
// context's masks need it.
static constexpr int64_t PIPE_MIN_UBATCH = 512;
static constexpr int64_t PIPE_MASK_MIN_UBATCH = 128;

static bool pipe_warn(model & m, const std::string & reason)
{
    if (m.pipe.warned.insert(reason).second)
    {
        fprintf(stderr, "[muse-spark] pipelined prefill declined (%s); this call runs one micro-batch at a time\n", reason.c_str());
        fflush(stderr);
    }
    m.pipe.n_declines++;
    return false;
}

/// How a call of n tokens at position p0 runs pipelined: n_full micro-batches
/// of u rows, the last padded with `pad` copies of the final token. The
/// micro-batch is TS_MUSE_SPARK_PIPELINE_UBATCH when set. Otherwise it is about
/// a quarter of the call, between PIPE_MIN_UBATCH and the prefill ubatch, and
/// smaller where a long context's masks need it: on 10x A40 four micro-batches
/// per call beat both llama.cpp's fixed 512 rows (one 4K call 3.7 s against
/// 4.5 s, one 16K call 10.2 s against 15.3 s at 2048 rows) and two large ones.
enum pipe_verdict { PIPE_OK, PIPE_TOO_SHORT, PIPE_MASKS, PIPE_CTX };
struct pipe_shape { int64_t u = 0, n_full = 0, pad = 0, p_last = 0, n_kv_g = 0, s0 = 0, n_kv_s = 0; };

static pipe_verdict pipe_plan(const model & m, int64_t p0, int64_t n, pipe_shape * out)
{
    const pipeline_state & P = m.pipe;
    if (n < 2 || p0 + n > m.n_ctx) return n < 2 ? PIPE_TOO_SHORT : PIPE_CTX;
    const int64_t u_max = P.ubatch_fixed ? P.ubatch : std::max<int64_t>(1, P.ubatch);
    const int64_t u_min = P.ubatch_fixed ? P.ubatch : std::min<int64_t>(PIPE_MIN_UBATCH, u_max);
    int64_t target = P.ubatch_fixed ? P.ubatch : std::clamp<int64_t>((n + 3) / 4, u_min, u_max);
    for (;;)
    {
        pipe_shape sh;
        sh.n_full = (n + target - 1) / target;
        if (sh.n_full < 2) return PIPE_TOO_SHORT;
        sh.u = (n + sh.n_full - 1) / sh.n_full;
        sh.pad = sh.n_full * sh.u - n;                       // < n_full
        if (p0 + sh.n_full * sh.u > m.n_ctx) { sh.u = n / sh.n_full; sh.pad = 0; }
        sh.p_last = p0 + sh.n_full * sh.u;
        // The last micro-batch has the largest windows.
        sh.n_kv_g = plan_n_kv_g(m, sh.p_last, m.kv_pad);
        plan_swa(m, sh.p_last - sh.u, sh.p_last, m.kv_pad, &sh.s0, &sh.n_kv_s);
        const size_t mask_bytes = (size_t) (sh.n_kv_g + sh.n_kv_s) * (size_t) sh.u * sizeof(uint16_t);
        if (mask_bytes <= PIPE_MASK_BUDGET) { *out = sh; return PIPE_OK; }
        if (P.ubatch_fixed || sh.u <= PIPE_MASK_MIN_UBATCH) return PIPE_MASKS;
        target = std::max<int64_t>(PIPE_MASK_MIN_UBATCH, sh.u / 2);
    }
}

static size_t pipe_slot_layout(int64_t u, int64_t n_kv_g, int64_t n_kv_s, size_t * off_kv, size_t * off_kv_s,
                               size_t * off_out, size_t * off_mg, size_t * off_ms)
{
    auto al = [](size_t x) { return (x + 255) & ~(size_t) 255; };
    size_t o = al((size_t) u * sizeof(int32_t));                       // pos
    *off_kv = o;   o = al(o + (size_t) u * sizeof(int64_t));
    *off_kv_s = o; o = al(o + (size_t) u * sizeof(int64_t));
    *off_out = o;  o = al(o + sizeof(int32_t));
    *off_mg = o;   o = al(o + (size_t) (n_kv_g * u) * sizeof(uint16_t));
    *off_ms = o;   o = al(o + (size_t) (n_kv_s * u) * sizeof(uint16_t));
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
/// up to (n_kv_g, n_kv_s). Runs only while nothing is in flight.
static bool pipe_ensure(model & m, int64_t u, int64_t n_kv_g, int64_t n_kv_s)
{
    pipeline_state & P = m.pipe;
    if (!P.sched)
    {
        P.n_nodes = (size_t) m.hp.n_layer * 160 + 1024;
        if (const long v = env_long("TS_MUSE_SPARK_NODES_PER_LAYER", 0); v > 0) P.n_nodes = (size_t) m.hp.n_layer * (size_t) v + 1024;
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
    size_t a, b, c, d, e;
    const size_t need = pipe_slot_layout(u, n_kv_g, n_kv_s, &a, &b, &c, &d, &e);
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

    // The largest micro-batch is the last: its windows size the reservation and the staging.
    const int64_t n_kv_g = sh.n_kv_g, s0 = sh.s0, n_kv_s = sh.n_kv_s;
    if (P.fail_u && u >= P.fail_u && n_kv_g >= P.fail_kv_g && n_kv_s >= P.fail_kv_s)
        return pipe_warn(m, "its compute buffers do not fit"), -1;
    // K/V rows are reserved up front: growing them moves tensors that in-flight
    // micro-batches still read.
    if (!slot_reserve_rows(m, s, plan_n_kv_g(m, std::max(p0 + n, p_last), m.kv_pad)))
        return pipe_warn(m, "no device memory to reserve this call's K/V rows up front"), -1;
    if (!pipe_ensure(m, u, n_kv_g, n_kv_s)) { pipe_warn(m, "no pinned host memory for its input staging"); return -1; }

    size_t off_kv, off_kv_s, off_out, off_mg, off_ms;
    pipe_slot_layout(u, n_kv_g, n_kv_s, &off_kv, &off_kv_s, &off_out, &off_mg, &off_ms);
    uint8_t * stage_base = (uint8_t *) ggml_backend_buffer_get_base(P.staging);
    const int C = P.n_copies;
    const size_t S = P.stages.size();
    const auto t_start = std::chrono::steady_clock::now();
    double enqueue_ms = 0.0;

    std::unique_ptr<graph_entry> g;
    auto new_graph = [&](int64_t pj, int64_t kv_g, int64_t ss0, int64_t kv_s) -> bool
    {
        if (g) { g->ctx = (ggml_free(g->ctx), nullptr); }
        g.reset(new graph_entry());
        g->owns_sched = false;
        g->kind = 0; g->slot_id = s.id; g->nt = u; g->p0 = pj; g->n_kv_g = kv_g; g->s0 = ss0; g->n_kv_s = kv_s;
        g->want_logits = true; g->n_out = 1; g->n_ovr = 0;
        ggml_init_params gp = { P.meta.size(), P.meta.data(), true };
        g->ctx = ggml_init(gp);
        if (!g->ctx) return false;
        g->gf = ggml_new_graph_custom(g->ctx, P.n_nodes, false);
        g->sched = P.sched;
        ggml_backend_sched_reset(P.sched);
        builder b(m, *g);
        b.build_single(s);
        return true;
    };
    // Smaller windows later in the call (and in later calls) fit the reserved
    // layout as long as the topology is the same: ggml reallocates (draining
    // every GPU and moving the input copies) when the node count differs.
    bool reserve = u > P.res_u || n_kv_g > P.res_kv_g || n_kv_s > P.res_kv_s;
    if (!reserve)
    {
        int64_t s0_first = 0, kv_s_first = 0;
        plan_swa(m, p0, p0 + u, m.kv_pad, &s0_first, &kv_s_first);
        if (!new_graph(p0, plan_n_kv_g(m, p0 + u, m.kv_pad), s0_first, kv_s_first)) return 0;
        reserve = ggml_graph_n_nodes(g->gf) != P.res_nodes;
    }
    if (reserve)
    {
        if (!new_graph(p_last - u, n_kv_g, s0, n_kv_s) || !ggml_backend_sched_reserve(P.sched, g->gf))
        {
            // A failed reserve may have regrown some GPUs' buffers already: give
            // them back to the sequential path, and do not retry this shape.
            P.release();
            P.fail_u = u; P.fail_kv_g = n_kv_g; P.fail_kv_s = n_kv_s;
            pipe_warn(m, "its compute buffers do not fit");
            return -1;
        }
        P.res_u = u; P.res_kv_g = n_kv_g; P.res_kv_s = n_kv_s;
        P.res_nodes = ggml_graph_n_nodes(g->gf);
    }

    // Copy slots rotate from 0 after a synchronize of an unallocated scheduler.
    ggml_backend_sched_reset(P.sched);
    ggml_backend_sched_synchronize(P.sched);

    std::vector<const void *> first_pos_addr((size_t) C, nullptr);
    std::vector<uint16_t> mask;
    bool in_flight = false;
    // Host time per phase (TS_MUSE_SPARK_EXEC_DEBUG): where an enqueue that is
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
        const int64_t kv_g_j = plan_n_kv_g(m, pj + u, m.kv_pad);
        int64_t s0_j = 0, kv_s_j = 0;
        plan_swa(m, pj, pj + u, m.kv_pad, &s0_j, &kv_s_j);
        if (!new_graph(pj, kv_g_j, s0_j, kv_s_j)) { if (in_flight) ggml_backend_sched_synchronize(P.sched); return 0; }
        if (!ggml_backend_sched_alloc_graph(P.sched, g->gf))
        {
            // ggml may have freed buffers it then failed to regrow: rebuild on the next call.
            if (in_flight) { ggml_backend_sched_synchronize(P.sched); }
            P.release();
            if (j == 0) { pipe_warn(m, "its compute buffers do not fit"); return -1; }
            fprintf(stderr, "[muse-spark] pipelined prefill: graph allocation failed after %" PRId64 " micro-batches\n", j);
            return 0;
        }
        // ggml must hand out the copy slots in order; anything else would let
        // a micro-batch overwrite inputs another one still reads. The probe is
        // the first stage-0 input the graph reads (a stage of only global layers
        // never reads positions, one of only sliding layers never reads kv_idx;
        // an input no node reads has no copies and no memory).
        if (!P.stages.empty() && C > 1)
        {
            const int d0 = P.stages[0];
            const ggml_tensor * probe = nullptr;
            for (const ggml_tensor * t : { g->inp.pos[d0], g->inp.kv_idx[d0], g->inp.kv_idx_s[d0], g->inp.mask_g[d0], g->inp.mask_s[d0] })
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
                fprintf(stderr, "[muse-spark] pipelined prefill disabled: ggml did not rotate its input copies as expected; "
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
        int64_t * kv_s = (int64_t *) (slot_mem + off_kv_s);
        int32_t * out_id = (int32_t *) (slot_mem + off_out);
        uint16_t * mg = (uint16_t *) (slot_mem + off_mg);
        uint16_t * ms = (uint16_t *) (slot_mem + off_ms);
        for (int64_t i = 0; i < u; i++)
        {
            pos[i] = (int32_t) (pj + i);
            kv[i] = pj + i;
            kv_s[i] = (pj + i) % m.swa_ring;
        }
        *out_id = (int32_t) (u - 1 - (j + 1 == n_full ? pad : 0));   // the last real token
        // Rows are labeled before the mask is built: this micro-batch writes
        // them before any of its attention reads them.
        swa_write(m, s, pj, u);
        bool need_g = false, need_s = false;
        for (int d = 0; d <= m.n_gpu; d++) { need_g |= live(g->inp.mask_g[d]); need_s |= live(g->inp.mask_s[d]); }
        if (need_g)
        {
            fill_mask(mask, 0, kv_g_j, pj, u, 0);
            memcpy(mg, mask.data(), mask.size() * sizeof(uint16_t));
        }
        if (need_s)
        {
            fill_swa_mask(m, s, mask, kv_s_j, pj, u);
            memcpy(ms, mask.data(), mask.size() * sizeof(uint16_t));
        }
        set_i32(g->inp.tokens, tokens + j * u, (size_t) u);   // host buffer: a plain copy
        for (int d = 0; d <= m.n_gpu; d++)
        {
            ggml_backend_t be = m.backends[d];
            pipe_upload(be, g->inp.pos[d], pos, (size_t) u * sizeof(int32_t));
            pipe_upload(be, g->inp.kv_idx[d], kv, (size_t) u * sizeof(int64_t));
            pipe_upload(be, g->inp.kv_idx_s[d], kv_s, (size_t) u * sizeof(int64_t));
            pipe_upload(be, g->inp.mask_g[d], mg, (size_t) (kv_g_j * u) * sizeof(uint16_t));
            pipe_upload(be, g->inp.mask_s[d], ms, (size_t) (kv_s_j * u) * sizeof(uint16_t));
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
            for (size_t i = 0; i + 1 < S; i++)
                if (ggml_backend_event_t ev = P.done[(i + 1) * (size_t) C + (size_t) c])
                    ggml_backend_event_wait(m.backends[P.stages[i]], ev);
        if (ggml_backend_sched_graph_compute_async(P.sched, g->gf) != GGML_STATUS_SUCCESS)
        {
            ggml_backend_sched_synchronize(P.sched);
            fprintf(stderr, "[muse-spark] pipelined prefill: graph compute failed\n");
            return 0;
        }
        in_flight = true;
        for (size_t i = 0; i < S; i++)
            if (ggml_backend_event_t ev = P.done[i * (size_t) C + (size_t) c])
                ggml_backend_event_record(ev, m.backends[P.stages[i]]);
        ms_submit += lap(t_phase);
        s.n_past = pj + u;
        P.n_ubatches++;
        enqueue_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_j).count();
    }

    ggml_backend_sched_synchronize(P.sched);
    std::fill(P.busy.begin(), P.busy.end(), 0);
    if (pad > 0)
    {
        // The padding is not part of the sequence.
        for (int64_t p = p0 + n; p < p_last; p++)
            if (s.swa_lab[(size_t) (p % m.swa_ring)] == p) s.swa_lab[(size_t) (p % m.swa_ring)] = -1;
        s.n_past = p0 + n;
    }
    *done = std::min<int64_t>(n, n_full * u);
    if (*done == n && logits_out && g->logits)
        ggml_backend_tensor_get(g->logits, logits_out, 0, (size_t) m.hp.n_vocab * sizeof(float));
    P.n_calls++;
    if (debug_on())
        fprintf(stderr, "[muse-spark] pipelined prefill: %" PRId64 " x %" PRId64 " rows (+%" PRId64 " padding) over %zu stages, "
                "enqueue %.0f ms (staging wait %.0f, graph %.0f, inputs %.0f, submit %.0f), total %.0f ms (%.0f tok/s)\n",
                n_full, u, pad, S, enqueue_ms, ms_wait, ms_build, ms_inputs, ms_submit,
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_start).count(),
                (double) *done / std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count());
    return *done == n ? 1 : -1;
}

/// Evaluate `n` tokens on the active slot in ubatches. Logits of the last
/// token (or every row when all_rows) land in logits_out.
static bool forward(model & m, const int32_t * tokens, int64_t n, float * logits_out, bool all_rows)
{
    if (!m.active_slot || n <= 0) return false;
    int64_t off = 0;
    if (m.pipe.enabled && !all_rows && n >= 2 && !m.embd_ovr.empty())
        pipe_warn(m, "image rows replace token embeddings inside pipelined micro-batches");
    if (m.pipe.enabled && !all_rows && m.embd_ovr.empty() && n >= 2)
    {
        const int rc = forward_pipelined(m, tokens, n, logits_out, &off);
        if (rc == 0) { m.embd_ovr.clear(); return false; }
        if (rc > 0) { m.embd_ovr.clear(); return true; }
        // declined after `off` tokens: the rest runs one micro-batch at a time
    }
    // vision overrides relative to this call's token array
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
                const int64_t at = o.index + r;
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
            // slots that come and go.
            // Under --tp every failed build leaves views of the static tensors
            // behind that ggml's meta backend can only hold so many of.
            if (m.alloc_failed && cnt > 64 && (m.tp.n <= 1 || ++halvings <= 3))
            {
                ubatch = std::max<int64_t>(64, cnt / 2);
                fprintf(stderr, "[muse-spark] a %" PRId64 "-token graph does not fit; continuing with %" PRId64 "-token ubatches\n",
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
/// logits_out receives one row per segment: its last token's logits.
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
    // often, and each crossing rebuilds the graph.
    const int64_t pad = decode_only ? std::max<int64_t>(m.kv_pad, env_long("TS_MUSE_SPARK_BD_KV_PAD", 2 * m.kv_pad)) : m.kv_pad;
    for (int i = 0; i < n; i++)
    {
        auto it = m.slots.find(slot_ids[i]);
        if (it == m.slots.end()) return false;
        for (int j = 0; j < i; j++) if (slot_ids[j] == slot_ids[i]) return false;
        slot & sl = *it->second;
        if (positions[i] != sl.n_past || sl.n_past + lens[i] > m.n_ctx) return false;
        if (!slot_reserve_rows(m, sl, plan_n_kv_g(m, sl.n_past + lens[i], pad))) return false;
        bd_token b;
        b.slot_id = sl.id;
        b.p = sl.n_past;
        b.len = lens[i];
        b.n_kv_g = plan_n_kv_g(m, sl.n_past + lens[i], pad);
        plan_swa(m, sl.n_past, sl.n_past + lens[i], pad, &b.s0, &b.n_kv_s);
        key->bd.push_back(b);
    }

    bool reused = false;
    graph_entry * g = acquire_graph(m, std::move(key), &reused);
    if (!g) return false;

    int64_t total_g = 0, total_s = 0;
    builder::seg_mask_sizes(*g, &total_g, &total_s);
    m.h_tokens.assign(tokens, tokens + T);
    m.h_pos.resize((size_t) T);
    m.h_kv_idx.resize((size_t) T);
    m.h_kv_idx_s.resize((size_t) T);
    m.h_mask_g.resize((size_t) total_g);
    m.h_mask_s.resize((size_t) total_s);
    m.h_out_ids.resize((size_t) n);
    std::vector<uint16_t> rows;
    int64_t row = 0, off_g = 0, off_s = 0;
    for (int i = 0; i < n; i++)
    {
        // The cached graph matched on window shapes only: positions come from
        // this call, never from the entry (which keeps the first caller's).
        const bd_token & b = g->bd[(size_t) i];
        slot & sl = *m.slots.at(b.slot_id);
        const int64_t p0 = positions[i];
        for (int64_t t = 0; t < b.len; t++)
        {
            m.h_pos[(size_t) (row + t)] = (int32_t) (p0 + t);
            m.h_kv_idx[(size_t) (row + t)] = p0 + t;
            m.h_kv_idx_s[(size_t) (row + t)] = (p0 + t) % m.swa_ring;
        }
        swa_write(m, sl, p0, b.len);
        fill_mask(rows, 0, b.n_kv_g, p0, b.len, 0);
        std::copy(rows.begin(), rows.end(), m.h_mask_g.begin() + off_g);
        fill_swa_mask(m, sl, rows, b.n_kv_s, p0, b.len);
        std::copy(rows.begin(), rows.end(), m.h_mask_s.begin() + off_s);
        m.h_out_ids[(size_t) i] = (int32_t) (row + b.len - 1);
        row += b.len;
        off_g += b.n_kv_g * b.len;
        off_s += b.n_kv_s * b.len;
    }
    set_i32(g->inp.tokens, m.h_tokens.data(), (size_t) T);
    set_i32(g->inp.out_ids, m.h_out_ids.data(), (size_t) n);
    for (int d = 0; d <= m.n_gpu; d++)
    {
        set_i32(g->inp.pos[d], m.h_pos.data(), (size_t) T);
        if (live(g->inp.kv_idx[d])) ggml_backend_tensor_set(g->inp.kv_idx[d], m.h_kv_idx.data(), 0, (size_t) T * sizeof(int64_t));
        if (live(g->inp.kv_idx_s[d])) ggml_backend_tensor_set(g->inp.kv_idx_s[d], m.h_kv_idx_s.data(), 0, (size_t) T * sizeof(int64_t));
        if (live(g->inp.mask_g[d])) ggml_backend_tensor_set(g->inp.mask_g[d], m.h_mask_g.data(), 0, m.h_mask_g.size() * 2);
        if (live(g->inp.mask_s[d])) ggml_backend_tensor_set(g->inp.mask_s[d], m.h_mask_s.data(), 0, m.h_mask_s.size() * 2);
    }
    if (!compute(m, *g)) return false;
    for (int i = 0; i < n; i++) m.slots.at(slot_ids[i])->n_past += lens[i];
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
static bool is_expert_name(const char * name)
{
    return strstr(name, "_exps.") != nullptr;
}

/// Tensor parallelism over `tp` CUDA GPUs: one ggml meta backend (the
/// mechanism behind llama.cpp's -sm tensor) whose split-state callback is
/// tsg_mspark_tp's plan, filled in once the metadata is read. Returns the
/// backend, or null after reporting the refusal.
static ggml_backend_t tp_open(model & m, int tp, const char * backend_name, const char * layer_split, int n_cpu_moe_req)
{
    // What tensor parallelism cannot run, refused before any GPU is touched.
    const char * why = nullptr;
    char buf[256];
    if (!backend_name || strcmp(backend_name, "CUDA") != 0)
        why = "tensor parallelism (--tp) runs on CUDA GPUs only; use --layer-split N on this backend";
    else if (tp > MAX_GPUS)
        why = (snprintf(buf, sizeof(buf), "%d GPUs requested; tensor parallelism spans at most %d", tp, MAX_GPUS), buf);
    else if (layer_split && *layer_split)
        why = "--tp cannot be combined with TS_MUSE_SPARK_LAYER_SPLIT (a layer split); use one or the other";
    else if (n_cpu_moe_req > 0)
        why = "--tp keeps every expert in GPU memory; --n-cpu-moe needs --layer-split";
    else if (opt_get("TS_MUSE_SPARK_PIPELINE") == "1")
        why = "TS_MUSE_SPARK_PIPELINE=1 cannot run under --tp (it pipelines a layer split's GPUs)";
    else if (env_long("TS_MUSE_SPARK_PROFILE", 0) > 0)
        why = "TS_MUSE_SPARK_PROFILE reads graph slices between --tp's reductions, which would profile wrong values";
    else if (env_long("TS_MUSE_SPARK_GRAPH_CACHE", 0) > 1)
        why = "TS_MUSE_SPARK_GRAPH_CACHE above 1 cannot run under --tp (its backend keeps one graph's GPU views)";
    else if (opt_get("TS_MUSE_SPARK_FUSE_QKVG") == "1")
        why = "TS_MUSE_SPARK_FUSE_QKVG=1 cannot run under --tp (each GPU holds a slice of q, k, v and gate separately)";
    else if (opt_get("TS_MUSE_SPARK_CPU_REPACK") == "1")
        why = "TS_MUSE_SPARK_CPU_REPACK=1 repacks host experts, which --tp does not use";
    if (why) { tsg::report_load_refusal("[muse-spark] %s\n", why); return nullptr; }

    // The first `tp` CUDA GPUs in index order (CUDA_VISIBLE_DEVICES selects
    // them), opened only through the meta device so each gets one stream.
    int n = 0;
    std::vector<int> cuda_index;
    int cuda_seen = 0;
    for (size_t i = 0; i < ggml_backend_dev_count() && n < tp; i++)
    {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        const char * reg_name = reg ? ggml_backend_reg_name(reg) : nullptr;
        const enum ggml_backend_dev_type dt = ggml_backend_dev_type(dev);
        if (!reg_name || strcmp(reg_name, "CUDA") != 0) continue;
        if (dt != GGML_BACKEND_DEVICE_TYPE_GPU && dt != GGML_BACKEND_DEVICE_TYPE_IGPU) continue;
        m.tp.devs[n++] = dev;
        cuda_index.push_back(cuda_seen++);
    }
    if (n < tp)
    {
        tsg::report_load_refusal("[muse-spark] --tp %d needs %d CUDA GPUs; %d are visible\n", tp, tp, n);
        return nullptr;
    }
    // Every GPU routes the experts itself from mirrored weights: unlike GPUs
    // could pick differently and sum mismatched partials without an error.
    for (int j = 1; j < n; j++)
        if (strcmp(ggml_backend_dev_description(m.tp.devs[j]), ggml_backend_dev_description(m.tp.devs[0])) != 0)
        {
            tsg::report_load_refusal("[muse-spark] --tp needs identical GPUs; GPU %d is %s and GPU 0 is %s\n", j,
                                     ggml_backend_dev_description(m.tp.devs[j]), ggml_backend_dev_description(m.tp.devs[0]));
            return nullptr;
        }
#if defined(TSG_GGML_USE_CUDA)
    // Before the meta backend creates its communicator.
    tsg::tp_cuda_prepare_collective(cuda_index.data(), n);
#endif
    m.tp.n = n;
    m.tp.ud.reset(new tsg_mspark_tp::userdata());
    static std::mutex meta_mu;   // ggml's meta device cache is not thread-safe
    std::lock_guard<std::mutex> lk(meta_mu);
    m.tp.meta_dev = ggml_backend_meta_device(m.tp.devs, (size_t) n, tsg_mspark_tp::split_state_cb, m.tp.ud.get());
    ggml_backend_t be = m.tp.meta_dev ? ggml_backend_dev_init(m.tp.meta_dev, nullptr) : nullptr;
    if (!be) tsg::report_load_refusal("[muse-spark] creating the tensor-parallel backend over %d GPUs failed\n", n);
    return be;
}

static model * load(const char * gguf_path, int n_gpu_req, int n_ctx, int n_ubatch, int n_threads,
                    int n_cpu_moe_req, const char * backend_name, bool ctx_is_hard_limit, const char * layer_split, int tp)
{
    const auto t_start = std::chrono::steady_clock::now();
    std::unique_ptr<model> m(new model());

    // --- backends ---------------------------------------------------------
    const bool cpu_only = backend_name && strcmp(backend_name, "CPU") == 0;
    if (!cpu_only && !tsg::configure_cuda_launch_policy())
        throw std::runtime_error("Failed to configure the CUDA launch policy");
    if (!cpu_only && n_gpu_req > MAX_GPUS && tp <= 1)
    {
        tsg::report_load_refusal("[muse-spark] %d GPUs requested; the native executor spans at most %d\n", n_gpu_req, MAX_GPUS);
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
    if (!cpu) { tsg::report_load_refusal("[muse-spark] CPU backend unavailable\n"); return nullptr; }
    {
        int cpu_threads = n_threads > 0 ? n_threads : tsg::available_cpu_parallelism();
        if (n_cpu_moe_req != 0 || n_gpu == 0) cpu_threads = tsg::available_cpu_parallelism();
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
    for (int i = 0; i <= n_gpu; i++)
    {
        m->sched_backends[i] = m->backends[i];
        m->sched_bufts[i] = ggml_backend_get_default_buffer_type(m->backends[i]);
    }
    m->n_sched_backends = n_gpu + 1;
    // Host tensors that cross into the meta backend (the gathered token rows)
    // are copied to every GPU synchronously: from pinned memory, as llama.cpp.
    if (m->tp.n > 1)
        if (ggml_backend_buffer_type_t host = ggml_backend_dev_host_buffer_type(m->tp.meta_dev))
            m->sched_bufts[n_gpu] = host;
    if (!cpu_only && n_gpu_req > 0 && n_gpu < n_gpu_req)
    {
        // The layers would silently crowd onto fewer GPUs than asked for.
        tsg::report_load_refusal("[muse-spark] %d GPU(s) requested but only %d usable %s device(s) found\n",
                                 n_gpu_req, n_gpu, backend_name ? backend_name : "GPU");
        return nullptr;
    }
    if (const long v = env_long("TS_MUSE_SPARK_GRAPH_CACHE", 0); v > 0) m->graph_cache_cap = (int) v;
    // The meta backend keeps per-GPU views only for the graph it ran last, so
    // under --tp an older cached graph could not run again.
    if (m->tp.n > 1) m->graph_cache_cap = 1;

    // --- metadata ---------------------------------------------------------
    int32_t split_count = 1;
    m->shards = resolve_shards(gguf_path, &split_count);
    hparams & hp = m->hp;
    {
        ggml_context * meta0 = nullptr;
        gguf_init_params ip = { true, &meta0 };
        gguf_context * g0 = gguf_init_from_file(m->shards.paths[0].c_str(), ip);
        if (!g0) { tsg::report_load_refusal("[muse-spark] failed to open %s\n", m->shards.paths[0].c_str()); return nullptr; }
        bool ok = true;
        const int64_t ai = gguf_find_key(g0, "general.architecture");
        if (ai < 0 || gguf_get_kv_type(g0, ai) != GGUF_TYPE_STRING ||
            (strcmp(gguf_get_val_str(g0, ai), "muse-spark") != 0 && strcmp(gguf_get_val_str(g0, ai), "muse_spark") != 0))
            ok = false;
        ok = ok && kv_u32(g0, akey("block_count").c_str(), &hp.n_layer);
        ok = ok && kv_u32(g0, akey("embedding_length").c_str(), &hp.n_embd);
        ok = ok && kv_u32(g0, akey("attention.head_count").c_str(), &hp.n_head);
        ok = ok && kv_u32(g0, akey("attention.head_count_kv").c_str(), &hp.n_head_kv);
        ok = ok && kv_u32(g0, akey("attention.sliding_window").c_str(), &hp.n_swa);
        ok = ok && kv_u32(g0, akey("moe_latent_size").c_str(), &hp.latent);
        ok = ok && kv_u32(g0, akey("expert_feed_forward_length").c_str(), &hp.n_ff_exp);
        ok = ok && kv_u32(g0, akey("expert_count").c_str(), &hp.n_expert);
        ok = ok && kv_u32(g0, akey("expert_used_count").c_str(), &hp.n_expert_used);
        ok = ok && kv_f32(g0, akey("attention.layer_norm_rms_epsilon").c_str(), &hp.rms_eps);
        ok = ok && kv_f32(g0, akey("post_norm_epsilon").c_str(), &hp.post_eps);
        ok = ok && kv_bool_arr(g0, akey("attention.sliding_window_pattern").c_str(), hp.is_swa);
        int32_t key_len = 0;
        kv_u32(g0, akey("attention.key_length").c_str(), &key_len);
        hp.head_dim = key_len > 0 ? key_len : (hp.n_head > 0 ? hp.n_embd / hp.n_head : 0);
        hp.n_rot = hp.head_dim;
        kv_u32(g0, akey("rope.dimension_count").c_str(), &hp.n_rot);
        kv_f32(g0, akey("rope.freq_base").c_str(), &hp.rope_base);
        hp.rope_base_swa = hp.rope_base;
        kv_f32(g0, akey("rope.freq_base_swa").c_str(), &hp.rope_base_swa);
        kv_f32(g0, akey("logit_scale").c_str(), &hp.logit_scale);
        kv_f32(g0, akey("final_logit_softcapping").c_str(), &hp.softcap);
        kv_f32(g0, akey("expert_weights_scale").c_str(), &hp.w_scale);
        kv_bool(g0, akey("expert_weights_norm").c_str(), &hp.w_norm);
        kv_u32(g0, akey("expert_gating_func").c_str(), &hp.gating);
        kv_u32(g0, akey("context_length").c_str(), &hp.n_ctx_train);
        gguf_free(g0);
        if (meta0) ggml_free(meta0);
        if (!ok || hp.n_layer <= 0 || hp.n_head <= 0 || hp.n_head_kv <= 0 || hp.n_head % hp.n_head_kv != 0 ||
            hp.head_dim <= 0 || (int) hp.is_swa.size() < hp.n_layer || hp.gating != 2 || hp.n_rot % 2 != 0)
        {
            tsg::report_load_refusal("[muse-spark] missing or invalid muse-spark metadata in %s\n", m->shards.paths[0].c_str());
            return nullptr;
        }
    }

    // --- tensor sources ---------------------------------------------------
    std::map<std::string, tensor_source> sources;
    std::vector<size_t> layer_dense((size_t) hp.n_layer, 0), layer_exps((size_t) hp.n_layer, 0);
    // [layer][0 gate, 1 up, 2 down] routed-expert stack bytes
    std::vector<std::array<size_t, 3>> exp_bytes((size_t) hp.n_layer, std::array<size_t, 3>{ 0, 0, 0 });
    for (size_t si = 0; si < m->shards.paths.size(); si++)
    {
        ggml_context * meta = nullptr;
        gguf_init_params sp = { true, &meta };
        gguf_context * g = gguf_init_from_file(m->shards.paths[si].c_str(), sp);
        if (!g) { tsg::report_load_refusal("[muse-spark] failed to open shard %s\n", m->shards.paths[si].c_str()); return nullptr; }
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
                tsg::report_load_refusal("[muse-spark] %s is incomplete (tensor %s ends past EOF); re-download it\n",
                                         m->shards.paths[si].c_str(), name);
                gguf_free(g); ggml_free(meta);
                return nullptr;
            }
            sources[name] = src;
            int bid = -1;
            if (sscanf(name, "blk.%d.", &bid) == 1 && bid >= 0 && bid < hp.n_layer)
            {
                (is_expert_name(name) ? layer_exps : layer_dense)[(size_t) bid] += src.size;
                const int k = strstr(name, "ffn_gate_exps") ? 0 : strstr(name, "ffn_up_exps") ? 1 : strstr(name, "ffn_down_exps") ? 2 : -1;
                if (k >= 0) exp_bytes[(size_t) bid][(size_t) k] = src.size;
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
        if (!te) { tsg::report_load_refusal("[muse-spark] token_embd.weight missing\n"); return nullptr; }
        hp.n_vocab = (int32_t) te->ne[1];
    }
    const tensor_source * out_src = src_of("output.weight");
    const tensor_source * out_norm_src = src_of("output_norm.weight");
    if (!out_src || !out_norm_src) { tsg::report_load_refusal("[muse-spark] output weights missing\n"); return nullptr; }
    const size_t head_bytes = out_src->size + out_norm_src->size;

    m->n_ctx = n_ctx > 0 ? n_ctx : 8192;
    if (hp.n_ctx_train > 0 && m->n_ctx > hp.n_ctx_train) m->n_ctx = hp.n_ctx_train;
    m->kv_pad = std::max(1L, env_long("TS_MUSE_SPARK_KV_PAD", (long) KV_PAD_DEFAULT));
    m->n_ctx = (int32_t) pad_to(m->n_ctx, m->kv_pad);
    m->n_ubatch = n_ubatch > 0 ? n_ubatch : 512;
    // Like llama.cpp's iSWA cache, a sliding layer keeps its window plus a
    // ubatch; the extra margin bounds how far a slot can be rewound exactly
    // (prefix reuse, speculative rollback). TS_MUSE_SPARK_SWA_FULL keeps
    // every position instead.
    const long swa_margin = std::max(0L, env_long("TS_MUSE_SPARK_SWA_REWIND", 512));
    const bool swa_full = env_flag("TS_MUSE_SPARK_SWA_FULL", false);
    auto size_swa_ring = [&]
    {
        int64_t ring = pad_to((int64_t) hp.n_swa + m->n_ubatch + MAX_BATCHED_DECODE + swa_margin, m->kv_pad);
        if (swa_full || ring > m->n_ctx) ring = m->n_ctx;
        m->swa_ring = (int32_t) ring;
    };
    size_swa_ring();
    m->layers.resize((size_t) hp.n_layer);
    for (int il = 0; il < hp.n_layer; il++) m->layers[(size_t) il].swa = hp.is_swa[(size_t) il] != 0;

    // --- placement --------------------------------------------------------
    const size_t plan_slots = (size_t) std::max(1L, env_long("TS_MUSE_SPARK_PLAN_SLOTS", 1));
    // Every slot keeps a full ring on each sliding layer; global rows grow with
    // each sequence, so the plan reserves TS_MUSE_SPARK_PLAN_CTX_TOKENS of them
    // in total (one full context by default, like llama.cpp's unified KV).
    const long plan_ctx_env = env_long("TS_MUSE_SPARK_PLAN_CTX_TOKENS", 0);
    const bool plan_ctx_explicit = plan_ctx_env > 0;
    if (!plan_ctx_explicit && !opt_get("TS_MUSE_SPARK_PLAN_CTX_TOKENS").empty())
        fprintf(stderr, "[muse-spark] TS_MUSE_SPARK_PLAN_CTX_TOKENS=%s is not a positive token count; the plan "
                "reserves global K/V rows for the whole context instead\n", opt_get("TS_MUSE_SPARK_PLAN_CTX_TOKENS").c_str());
    size_t plan_ctx_tokens = plan_ctx_explicit ? (size_t) plan_ctx_env : (size_t) m->n_ctx;
    // Re-size everything the context sizes (the placement plan reads these).
    auto set_ctx = [&](int32_t ctx)
    {
        m->n_ctx = ctx;
        size_swa_ring();
        if (!plan_ctx_explicit) plan_ctx_tokens = (size_t) ctx;
    };
    auto kv_layer_bytes_of = [&](int il) -> size_t
    {
        return hp.is_swa[(size_t) il] ? kv_bytes_per_token_layer(hp) * (size_t) m->swa_ring * plan_slots
                                      : kv_bytes_per_token_layer(hp) * plan_ctx_tokens;
    };
    // --- tensor parallelism: the split plan, and what each GPU must hold ---
    if (m->tp.n > 1)
    {
        tsg_mspark_tp::shape ts;
        ts.n_layer = hp.n_layer; ts.n_head = hp.n_head; ts.n_head_kv = hp.n_head_kv; ts.head_dim = hp.head_dim;
        ts.n_ff_exp = hp.n_ff_exp; ts.n_vocab = hp.n_vocab;
        ts.is_swa.assign(hp.is_swa.begin(), hp.is_swa.begin() + hp.n_layer);
        // Quantizations may differ per layer: the largest block bounds every share.
        for (int il = 0; il < hp.n_layer; il++)
        {
            char nm[96];
            snprintf(nm, sizeof(nm), "blk.%d.attn_output.weight", il);
            if (const tensor_source * t = src_of(nm); t && ggml_blck_size(t->type) > ggml_blck_size(ts.wo_type)) ts.wo_type = t->type;
            snprintf(nm, sizeof(nm), "blk.%d.ffn_down_exps.weight", il);
            if (const tensor_source * t = src_of(nm); t && ggml_blck_size(t->type) > ggml_blck_size(ts.down_type)) ts.down_type = t->type;
        }
        std::string err;
        if (!tsg_mspark_tp::make_plan(ts, m->tp.n, &m->tp.ud->p, &err))
        {
            tsg::report_load_refusal("[muse-spark] %s\n", err.c_str());
            return nullptr;
        }
        m->tp.ud->s = ts;
        const tsg_mspark_tp::userdata & ud = *m->tp.ud;
        const int n = m->tp.n;
        // Weights per GPU: its slices of the split tensors and a whole copy of
        // the rest. The token table stays in the host mapping.
        std::vector<size_t> w_dev((size_t) n, 0), free_dev((size_t) n, 0);
        for (const auto & kv : sources)
        {
            if (kv.first == "token_embd.weight") continue;
            for (int j = 0; j < n; j++)
                w_dev[(size_t) j] += tsg_mspark_tp::device_bytes(ud, kv.first.c_str(), kv.second.type, kv.second.ne, j) + 4096;
        }
        for (int j = 0; j < n; j++)
        {
            size_t f = 0, t = 0;
            ggml_backend_dev_memory(m->tp.devs[j], &f, &t);
            free_dev[(size_t) j] = f;
        }
        // Compute buffers are the whole graph's on every GPU, plus the reduction
        // scratch ggml's meta backend keeps (ceil(log2 n) MoE partials of a ubatch).
        int steps = 0;
        while ((1 << steps) < n) steps++;
        const size_t reduce = (size_t) steps * (size_t) hp.latent * (size_t) hp.n_expert_used * (size_t) m->n_ubatch * sizeof(float);
        const size_t reserve = (size_t) env_long("TS_MUSE_SPARK_VRAM_RESERVE_MB", 2048 + m->n_ubatch / 4) * 1024 * 1024 + reduce;
        // GPU 0 also holds the managed side's vision encoder.
        const size_t dev0_extra = (size_t) env_long("TS_MUSE_SPARK_DEV0_RESERVE_MB", 0) * 1024 * 1024;
        auto kv_dev = [&](int j) -> size_t
        {
            size_t b = 0;
            for (int il = 0; il < hp.n_layer; il++)
            {
                const size_t rows = hp.is_swa[(size_t) il] ? (size_t) m->swa_ring * plan_slots : plan_ctx_tokens;
                b += rows * 2 * (size_t) (ud.p.groups[(size_t) il][(size_t) j] * hp.head_dim) * sizeof(uint16_t);
            }
            return b;
        };
        auto misfit = [&]() -> int
        {
            for (int j = 0; j < n; j++)
                if (w_dev[(size_t) j] + kv_dev(j) + reserve + (j == 0 ? dev0_extra : 0) > free_dev[(size_t) j]) return j;
            return -1;
        };
        const int64_t min_ctx = pad_to(4096, m->kv_pad);
        const int32_t advertised = m->n_ctx;
        if (misfit() >= 0 && !ctx_is_hard_limit && !plan_ctx_explicit && m->n_ctx > min_ctx)
        {
            set_ctx((int32_t) min_ctx);
            if (misfit() < 0)
            {
                int64_t lo = min_ctx / m->kv_pad, hi = advertised / m->kv_pad;   // lo fits, hi does not
                while (hi - lo > 1)
                {
                    const int64_t mid = lo + (hi - lo) / 2;
                    set_ctx((int32_t) (mid * m->kv_pad));
                    if (misfit() < 0) lo = mid; else hi = mid;
                }
                set_ctx((int32_t) (lo * m->kv_pad));
                fprintf(stderr, "[muse-spark] context capped to %d tokens (the GGUF advertises %d) so each GPU's share of the "
                        "weights and the KV plan fit; set MAX_CONTEXT to choose a size\n", m->n_ctx, advertised);
            }
        }
        if (const int j = misfit(); j >= 0)
        {
            tsg::report_load_refusal("[muse-spark] tensor parallel: GPU %d needs %.1f GiB (weights %.1f, K/V %.1f, compute and "
                                     "reductions %.1f) at a %d-token context but has %.1f GiB free; set a smaller MAX_CONTEXT, "
                                     "TS_MUSE_SPARK_PLAN_SLOTS or TS_MUSE_SPARK_UBATCH, or use more GPUs\n", j,
                                     (w_dev[(size_t) j] + kv_dev(j) + reserve + (j == 0 ? dev0_extra : 0)) / 1073741824.0,
                                     w_dev[(size_t) j] / 1073741824.0, kv_dev(j) / 1073741824.0,
                                     (reserve + (j == 0 ? dev0_extra : 0)) / 1073741824.0, m->n_ctx, free_dev[(size_t) j] / 1073741824.0);
            return nullptr;
        }
        for (int j = 0; j < n; j++)
            fprintf(stderr, "[muse-spark] tensor parallel GPU %d: %.1f GiB weights + %.1f GiB K/V of %.1f GiB free\n", j,
                    w_dev[(size_t) j] / 1073741824.0, kv_dev(j) / 1073741824.0, free_dev[(size_t) j] / 1073741824.0);
    }

    std::vector<size_t> dev_budget((size_t) std::max(n_gpu, 1), 0);
    if (m->tp.n > 1)
    {
        // Checked per GPU above: the layer placement sees one unbounded device.
        dev_budget[0] = SIZE_MAX / 4;
    }
    else
    {
        // Compute buffers grow with the ubatch (~0.95 GiB per GPU at 2048 rows), and
        // ggml-cuda keeps its scratch pool at the largest size any kernel needed.
        const size_t reserve = (size_t) env_long("TS_MUSE_SPARK_VRAM_RESERVE_MB", 2048 + m->n_ubatch / 4) * 1024 * 1024;
        const size_t dev0_extra = (size_t) env_long("TS_MUSE_SPARK_DEV0_RESERVE_MB", 0) * 1024 * 1024;
        for (int d = 0; d < n_gpu; d++)
        {
            size_t free_b = 0, total_b = 0;
            ggml_backend_dev_memory(ggml_backend_get_device(m->backends[d]), &free_b, &total_b);
            size_t r = reserve + (d == 0 ? dev0_extra : 0);
            dev_budget[(size_t) d] = free_b > r ? free_b - r : 0;
        }
    }

    // Which routed-expert stacks stay on the host, and which GPU holds each
    // layer. An explicit --n-cpu-moe N pins the leading N layers' experts to
    // the host (llama.cpp's meaning); automatic placement may leave ANY stack on
    // the host. A DP over contiguous layer ranges per GPU minimizes the host
    // expert bytes (the decode cost): each GPU holds its layers' dense parts,
    // then as many of their gate/up/down stacks as fit. Placing stacks rather
    // than whole layers fills the room a whole ~6 GiB layer would not fit.
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
    // Among plans with the fewest host bytes, prefer the one whose GPUs are
    // filled most evenly (smallest sum of squared fill fractions). Without it a
    // plan that fits on fewer GPUs left the leading ones empty and packed the
    // rest to their budgets, so their spare memory for K/V growth went unused.
    // Both terms are sums over GPUs, so the lexicographic DP stays exact.
    auto place = [&](bool extra) -> int64_t
    {
        const int64_t INF = INT64_MAX / 4;
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
                    const int64_t c = b == a ? 0 : range_cost(d, a, b, extra, nullptr, &used);
                    if (c < 0) break;   // longer ranges only need more room
                    const double fill = dev_budget[(size_t) d] > 0 ? (double) used / (double) dev_budget[(size_t) d] : 0.0;
                    const int64_t host = best[(size_t) d][(size_t) a] + c;
                    const double sq = spread[(size_t) d][(size_t) a] + fill * fill;
                    int64_t & bh = best[(size_t) d + 1][(size_t) b];
                    double & bs = spread[(size_t) d + 1][(size_t) b];
                    if (host < bh || (host == bh && sq < bs - 1e-12))
                    {
                        bh = host;
                        bs = sq;
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
        std::fill(cpu_stack.begin(), cpu_stack.end(), 1);
    }
    else
    {
        // Explicit layers per GPU (TS_MUSE_SPARK_LAYER_SPLIT); empty = automatic.
        std::vector<int> counts;
        if (layer_split && *layer_split)
        {
            int sum = 0;
            for (const char * p = layer_split; *p;)
            {
                char * end = nullptr;
                const long v = strtol(p, &end, 10);
                if (end == p || (*end != ',' && *end != '\0') || v < 0 || v > (long) (L - sum))
                {
                    tsg::report_load_refusal("[muse-spark] layer split '%s' must list non-negative layer counts "
                                             "summing to %d\n", layer_split, L);
                    return nullptr;
                }
                counts.push_back((int) v);
                sum += (int) v;
                p = *end == ',' ? end + 1 : end;
            }
            if ((int) counts.size() > n_gpu || sum != L)
            {
                tsg::report_load_refusal("[muse-spark] layer split '%s' must list at most %d counts summing to %d\n",
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
        // The plan reserves every global-layer row of a full context, so a
        // smaller one leaves room for the expert stacks the placement asked for.
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
                fprintf(stderr, "[muse-spark] context capped to %d tokens (the GGUF advertises %d) so the weights and "
                        "the KV plan fit the %d GPU(s) with the requested expert placement; set MAX_CONTEXT to "
                        "choose a size\n", m->n_ctx, advertised, n_gpu);
            }
            else
            {
                floor_failed = true;
            }
        }
        // A smaller MAX_CONTEXT helps only when this one was not already the floor.
        const char * smaller_ctx_hint = floor_failed ? "" : ", or set a smaller MAX_CONTEXT / TS_MUSE_SPARK_PLAN_SLOTS";

        if (!counts.empty())
        {
            int il = 0;
            for (size_t d = 0; d < counts.size(); d++)
                for (int c = 0; c < counts[d]; c++) assign[(size_t) il++] = (int) d;
            std::fill(cpu_stack.begin(), cpu_stack.end(), 0);
            if (const int d = split_misfit(&cpu_stack); d >= 0)
            {
                tsg::report_load_refusal("[muse-spark] layer split '%s' does not fit GPU %d at a %d-token context%s\n",
                                         layer_split, d, m->n_ctx, smaller_ctx_hint);
                return nullptr;
            }
        }
        else if (place(auto_cpu) < 0)
        {
            if (place(true) >= 0)
                tsg::report_load_refusal("[muse-spark] the model does not fit %d GPU(s) at a %d-token context%s with the "
                                         "experts of %d leading layer(s) on the host (--n-cpu-moe); "
                                         "TS_MUSE_SPARK_CPU_MOE=auto places only the expert stacks that must stay on the "
                                         "host%s\n",
                                         n_gpu, m->n_ctx, floor_failed ? " (the smallest it tries)" : "",
                                         forced_prefix, smaller_ctx_hint);
            else
                tsg::report_load_refusal("[muse-spark] the model does not fit %d GPU(s) at a %d-token context even with "
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
            fprintf(stderr, "[muse-spark] GPU %d: %d layers, %.1f GiB weights+KV of %.1f GiB budget\n",
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
            fprintf(stderr, "[muse-spark] routed experts on the host (%.1f GiB, %d layers, %d threads): %s\n",
                    hb / 1073741824.0, n_cpu_moe, m->cpu_threads, which.c_str());
        }
    }

    // --- weight tensors ---------------------------------------------------
    for (int d = 0; d <= n_gpu; d++)
    {
        ggml_init_params p = { (size_t) (hp.n_layer * 32 + 64) * ggml_tensor_overhead(), nullptr, true };
        m->w_ctx[d] = ggml_init(p);
    }
    std::vector<pending_weight> pending;
    bool missing = false;
    auto make_in = [&](ggml_context * c, const char * name, bool required) -> ggml_tensor *
    {
        const tensor_source * s = src_of(name);
        if (!s) { if (required) { fprintf(stderr, "[muse-spark] missing tensor %s\n", name); missing = true; } return nullptr; }
        ggml_tensor * t = ggml_new_tensor_4d(c, s->type, s->ne[0], s->ne[1], s->ne[2], s->ne[3]);
        ggml_set_name(t, name);
        pending.push_back({ t, s->shard, s->offset, 0, s->size });
        return t;
    };
    auto make = [&](int dev, const char * name, bool required) -> ggml_tensor *
    {
        return make_in(m->w_ctx[dev], name, required);
    };

    // Host expert stacks whose type the CPU backend can repack into its
    // interleaved GEMV/GEMM layout (Q4_K on AVX2+: the down projections here)
    // are copied into a CPU_REPACK buffer instead of being served from the
    // mapping. The repacked kernels read the same bytes with fewer, wider
    // loads. llama.cpp gets the same layout only with mmap disabled.
    // NUMA placement (default on a multi-node host) copies every host expert
    // stack into memory placed beside the worker that reads it, and pins the
    // CPU workers. It supersedes the mapping and the repacked layout, which
    // measured slower for both decode and prefill on a two-socket host.
    bool host_numa = false;
#if defined(__linux__)
    if (n_gpu > 0 && n_cpu_moe > 0 && env_flag("TS_MUSE_SPARK_HOST_NUMA", true))
    {
        const auto nodes = numa_node_cpus();
        if (nodes.size() >= 2)
        {
            int nth = m->cpu_threads;
            if (!env_long("TS_CPU_MOE_THREADS", 0) && tsg::host_moe_explicit_thread_count() <= 0)
                nth = std::max(2, nth - std::max(2, nth / 6));   // leave the host threads room within the quota
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
                fprintf(stderr, "[muse-spark] host experts: %d pinned workers over %zu NUMA nodes%s\n", nth, nodes.size(),
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
    if (n_gpu > 0 && n_cpu_moe > 0 && !host_numa && env_flag("TS_MUSE_SPARK_CPU_REPACK", false))
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

    // Under --tp each GPU holds its own slices of q, k, v and gate.
    const bool want_fuse = m->tp.n <= 1 && env_flag("TS_MUSE_SPARK_FUSE_QKVG", true);
    m->tok_embd = make(m->embd_device, "token_embd.weight", true);
    m->output_norm = make(m->out_device, "output_norm.weight", true);
    m->output = make(m->out_device, "output.weight", true);
    int n_fused = 0;
    for (int il = 0; il < hp.n_layer; il++)
    {
        layer & L = m->layers[(size_t) il];
        const int d = L.device;
        L.attn_norm = make(d, lname(il, "attn_norm.weight"), true);
        L.q_norm = make(d, lname(il, "attn_q_norm.weight"), true);
        L.k_norm = make(d, lname(il, "attn_k_norm.weight"), true);
        L.wo = make(d, lname(il, "attn_output.weight"), true);
        L.attn_alpha = make(d, lname(il, "attn_resid_alpha.weight"), true);
        L.attn_beta = make(d, lname(il, "attn_resid_beta.weight"), true);
        L.ffn_norm = make(d, lname(il, "ffn_norm.weight"), true);
        L.post_ffw_norm = make(d, lname(il, "post_ffw_norm.weight"), true);
        L.ffn_alpha = make(d, lname(il, "ffn_resid_alpha.weight"), true);
        L.ffn_beta = make(d, lname(il, "ffn_resid_beta.weight"), true);
        L.gate_inp = make(d, lname(il, "ffn_gate_inp.weight"), true);
        L.exp_probs_b = make(d, lname(il, "exp_probs_b.bias"), false);
        L.lat_down = make(d, lname(il, "ffn_latent_down.weight"), true);
        L.lat_up = make(d, lname(il, "ffn_latent_up.weight"), true);
        L.pre_exp_norm = make(d, lname(il, "ffn_pre_exp_norm.weight"), true);
        L.post_exp_norm = make(d, lname(il, "ffn_post_exp_norm.weight"), true);
        L.gate_exps = make_expert(il, L.cpu_stack[0] ? m->host_dev() : d, "ffn_gate_exps.weight");
        L.up_exps = make_expert(il, L.cpu_stack[1] ? m->host_dev() : d, "ffn_up_exps.weight");
        L.down_exps = make_expert(il, L.cpu_stack[2] ? m->host_dev() : d, "ffn_down_exps.weight");

        // Q/K/V/gate as one matrix: one weight read and one launch per layer.
        std::string nq = lname(il, "attn_q.weight"), nk = lname(il, "attn_k.weight");
        std::string nv = lname(il, "attn_v.weight"), ng = lname(il, "attn_gate.weight");
        const tensor_source *sq = src_of(nq.c_str()), *sk = src_of(nk.c_str()), *sv = src_of(nv.c_str()), *sg = src_of(ng.c_str());
        if (!sq || !sk || !sv || !sg) { fprintf(stderr, "[muse-spark] layer %d attention projections missing\n", il); missing = true; continue; }
        const bool fusable = want_fuse && sq->type == sk->type && sk->type == sv->type && sv->type == sg->type &&
                             sq->ne[0] == hp.n_embd && sk->ne[0] == hp.n_embd && sv->ne[0] == hp.n_embd && sg->ne[0] == hp.n_embd &&
                             sq->ne[2] == 1 && sk->ne[2] == 1 && sv->ne[2] == 1 && sg->ne[2] == 1;
        if (fusable)
        {
            const int64_t rows = sq->ne[1] + sk->ne[1] + sv->ne[1] + sg->ne[1];
            L.wqkvg = ggml_new_tensor_2d(m->w_ctx[d], sq->type, hp.n_embd, rows);
            ggml_format_name(L.wqkvg, "blk.%d.attn_qkvg.weight", il);
            size_t off = 0;
            for (const tensor_source * s : { sq, sk, sv, sg })
            {
                pending.push_back({ L.wqkvg, s->shard, s->offset, off, s->size });
                off += s->size;
            }
            n_fused++;
        }
        else
        {
            L.wq = make(d, nq.c_str(), true);
            L.wk = make(d, nk.c_str(), true);
            L.wv = make(d, nv.c_str(), true);
            L.wg = make(d, ng.c_str(), true);
        }
    }
    if (missing) { tsg::report_load_refusal("[muse-spark] required tensors are missing\n"); return nullptr; }
    m->fused_qkvg = n_fused == hp.n_layer;

    // --- host tensors straight from the GGUF mapping ----------------------
    std::map<const ggml_tensor *, std::vector<const pending_weight *>> by_tensor;
    for (const pending_weight & pw : pending) by_tensor[pw.t].push_back(&pw);
    std::vector<tsg_dsv4::file_warm_range> warm;
    {
        ggml_context * hctx = m->w_ctx[m->host_dev()];
        for (ggml_tensor * t = ggml_get_first_tensor(hctx); t; t = ggml_get_next_tensor(hctx, t))
        {
            const auto & pws = by_tensor[t];
            if (pws.size() != 1) continue;
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
        if (!m->w_buf[d]) { tsg::report_load_refusal("[muse-spark] weight allocation failed on device %d\n", d); return nullptr; }
        ggml_backend_buffer_set_usage(m->w_buf[d], GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }
    if (m->tp.n > 1 && m->tp.ud->unplanned.load())
    {
        tsg::report_load_refusal("[muse-spark] internal: tensor %s has no tensor-parallel split\n", m->tp.ud->unplanned_name.c_str());
        return nullptr;
    }
    std::vector<const pending_weight *> numa_jobs;
    if (m->numa_ctx && ggml_get_first_tensor(m->numa_ctx))
    {
        // Untouched pages: each lands on the node of the helper that writes it first.
        m->numa_buf = ggml_backend_alloc_ctx_tensors(m->numa_ctx, m->backends[n_gpu]);
        if (!m->numa_buf)
        {
            tsg::report_load_refusal("[muse-spark] host memory for the expert stacks is unavailable; "
                                     "TS_MUSE_SPARK_HOST_NUMA=0 serves them from the GGUF mapping\n");
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
            tsg::report_load_refusal("[muse-spark] host memory for the repacked expert stacks is unavailable; "
                                     "set TS_MUSE_SPARK_CPU_REPACK=0 to serve them from the mapping\n");
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
        const size_t chunk = (size_t) std::max(1L, env_long("TS_MUSE_SPARK_LOAD_CHUNK_MB", 64)) * 1024 * 1024;
        int load_threads = (int) std::max(1L, env_long("TS_MUSE_SPARK_LOAD_THREADS", 16));
        std::vector<load_job> jobs;
        size_t total = 0;
        for (const pending_weight & pw : pending)
        {
            if (pw.t->buffer && std::find(m->mmap_bufs.begin(), m->mmap_bufs.end(), pw.t->buffer) != m->mmap_bufs.end())
                continue;
            if (m->repack_buf && pw.t->buffer == m->repack_buf) continue;   // repacked below
            if (m->numa_buf && pw.t->buffer == m->numa_buf) continue;       // placed below
            // A split tensor is written in whole slabs along its split axis (rows
            // for axis 0, whole matrices for axis 1): each write is scattered.
            size_t step = chunk;
            if (m->tp.n > 1 && pw.t->buffer == m->w_buf[0])
            {
                const ggml_backend_meta_split_state st = tsg_mspark_tp::split_state(pw.t, *m->tp.ud);
                if ((int) st.axis >= 0 && (int) st.axis < GGML_MAX_DIMS)
                {
                    const size_t slab = pw.t->nb[(int) st.axis + 1];
                    step = std::max<size_t>(1, chunk / slab) * slab;
                }
            }
            for (size_t off = 0; off < pw.bytes; off += step)
                jobs.push_back({ pw.t, pw.shard, pw.file_off + off, pw.tensor_off + off, std::min(step, pw.bytes - off) });
            total += pw.bytes;
        }
        std::atomic<size_t> done(0);
        std::atomic<bool> finished(false);
        std::thread progress([&]()
        {
            auto t0 = std::chrono::steady_clock::now();
            while (!finished.load())
            {
                for (int i = 0; i < 100 && !finished.load(); i++) std::this_thread::sleep_for(std::chrono::milliseconds(100));
                if (finished.load()) break;
                const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                fprintf(stderr, "[muse-spark] uploading weights: %.1f / %.1f GiB (%.2f GiB/s)\n",
                        done.load() / 1073741824.0, total / 1073741824.0, done.load() / 1073741824.0 / std::max(s, 1e-3));
            }
        });
        std::mutex meta_set_mu;
        const bool ok = upload_parallel(m->shards, jobs, load_threads, done, m->tp.n > 1 ? &meta_set_mu : nullptr);
        finished.store(true);
        progress.join();
        if (!ok) { tsg::report_load_refusal("[muse-spark] weight upload failed\n"); return nullptr; }

#if defined(__linux__)
        if (!numa_jobs.empty())
        {
            // In batches: bring a batch's file ranges into the page cache with
            // parallel sequential reads, let one helper per worker (pinned to
            // that worker's CPU) copy exactly the rows the worker reads, then
            // drop the batch's file pages.
            const auto t_numa = std::chrono::steady_clock::now();
            const int nth = (int) m->worker_cpu.size();
            const size_t batch_cap = (size_t) std::max(1L, env_long("TS_MUSE_SPARK_NUMA_BATCH_GB", 16)) << 30;
            size_t first = 0;
            while (first < numa_jobs.size())
            {
                size_t last = first, bytes = 0;
                std::vector<tsg_dsv4::file_warm_range> ranges;
                while (last < numa_jobs.size() && (last == first || bytes + numa_jobs[last]->bytes <= batch_cap))
                {
                    const pending_weight * pw = numa_jobs[last];
                    if (!mmap_shard(*m, pw->shard)) { tsg::report_load_refusal("[muse-spark] cannot map shard %d\n", pw->shard); return nullptr; }
                    ranges.push_back({ pw->shard, pw->file_off, pw->bytes, (char *) m->mmap_addrs[(size_t) pw->shard] + pw->file_off });
                    bytes += pw->bytes;
                    last++;
                }
                tsg_dsv4::file_warm_options options;
                options.threads = load_threads;
                const auto warmed = tsg_dsv4::warm_file_ranges(m->shards.paths, ranges, options);
                if (!warmed.ok) { tsg::report_load_refusal("[muse-spark] reading expert stacks failed: %s\n", warmed.error.c_str()); return nullptr; }
                std::vector<std::thread> helpers;
                for (int c = 0; c < nth; c++)
                    helpers.emplace_back([&, c]()
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
                for (auto & h : helpers) h.join();
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
            fprintf(stderr, "[muse-spark] placed %zu host expert stacks (%.1f GiB) beside their %d pinned workers in %.1fs\n",
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
                if (!mmap_shard(*m, pw->shard)) { tsg::report_load_refusal("[muse-spark] cannot map shard %d\n", pw->shard); return nullptr; }
                src_ranges.push_back({ pw->shard, pw->file_off, pw->bytes, (char *) m->mmap_addrs[(size_t) pw->shard] + pw->file_off });
            }
            tsg_dsv4::file_warm_options options;
            options.threads = load_threads;
            const auto warmed = tsg_dsv4::warm_file_ranges(m->shards.paths, src_ranges, options);
            if (!warmed.ok) { tsg::report_load_refusal("[muse-spark] reading expert stacks failed: %s\n", warmed.error.c_str()); return nullptr; }
            std::atomic<size_t> next(0);
            auto worker = [&]()
            {
                for (size_t i; (i = next.fetch_add(1)) < repack_jobs.size(); )
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
            std::vector<std::thread> pool;
            const int nt = std::min<int>(load_threads, (int) repack_jobs.size());
            for (int i = 0; i < nt; i++) pool.emplace_back(worker);
            for (auto & th : pool) th.join();
            fprintf(stderr, "[muse-spark] repacked %zu host expert stacks (%.1f GiB) for the CPU kernels in %.1fs\n",
                    repack_jobs.size(), m->repack_bytes / 1073741824.0,
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - t_repack).count());
        }

        if (m->mmap_weight_bytes > 0 && env_flag("TS_MUSE_SPARK_WARM_HOST", true))
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
                    tsg::report_load_refusal("[muse-spark] prefaulting host experts failed: %s\n", result.error.c_str());
                    return nullptr;
                }
                fprintf(stderr, "[muse-spark] prefaulted %.1f GiB of host weights in %.1fs (%.1f GiB read)\n",
                        m->mmap_weight_bytes / 1073741824.0,
                        std::chrono::duration<double>(std::chrono::steady_clock::now() - t_warm).count(),
                        result.bytes_read / 1073741824.0);
            }
            else
            {
                fprintf(stderr, "[muse-spark] skipping host prefault: %.1f GiB mapped exceeds the %.1f GiB allowance\n",
                        m->mmap_weight_bytes / 1073741824.0, allow / 1073741824.0);
            }
        }
    }

    // Offloading a host expert stack to the GPU for a prefill-sized batch copies
    // every expert of the layer over the bus and needs a device buffer for them.
    m->op_offload = env_flag("TS_MUSE_SPARK_OP_OFFLOAD", n_cpu_moe == 0);
    m->host_moe_max_tokens = std::max(0L, env_long("TS_MUSE_SPARK_HOST_MOE_TOKENS", MAX_BATCHED_DECODE));

    // --- the first sequence slot -----------------------------------------
    slot * s0 = slot_alloc(*m, false);
    if (!s0) { tsg::report_load_refusal("[muse-spark] KV cache for %d tokens does not fit\n", m->n_ctx); return nullptr; }
    m->active_slot = s0;

    // --- grow the GPU kernel-scratch pools to their prefill peak ----------
    // ggml-cuda keeps a per-device pool for kernel scratch (quantized
    // activations, expert-sorted rows) outside every graph's compute buffer.
    // It grows on first use and never shrinks. Grown later, after sequence
    // slots and retained prefix state took the memory, the growth fails
    // inside a kernel and ggml aborts the process (seen: the first 4K prompt
    // after a run of concurrent requests). One full-size prefill now reserves
    // the peak while memory is free, so admission sees what is really left.
    if (n_gpu > 0 && env_flag("TS_MUSE_SPARK_WARM_POOLS", true))
    {
        const int64_t nt = std::min<int64_t>(m->n_ubatch, m->n_ctx);
        std::vector<int32_t> toks((size_t) nt);
        for (int64_t i = 0; i < nt; i++) toks[(size_t) i] = (int32_t) (1 + i % std::max<int32_t>(1, hp.n_vocab - 1));
        const auto t_warm = std::chrono::steady_clock::now();
        const bool warmed = forward(*m, toks.data(), nt, nullptr, false);
        s0->n_past = 0;
        std::fill(s0->swa_lab.begin(), s0->swa_lab.end(), -1);
        m->embd_ovr.clear();
        m->graph_cache.clear();
        fprintf(stderr, "[muse-spark] reserved GPU scratch with a %" PRId64 "-token prefill in %.1fs%s\n", nt,
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t_warm).count(), warmed ? "" : " (failed)");
    }

    // --- pipelined prefill ----------------------------------------------------
    {
        pipeline_state & P = m->pipe;
        const std::string mode = opt_get("TS_MUSE_SPARK_PIPELINE");
        if (!mode.empty() && mode != "0" && mode != "1")
        {
            tsg::report_load_refusal("[muse-spark] TS_MUSE_SPARK_PIPELINE must be 0 or 1 (unset: automatic), not '%s'\n", mode.c_str());
            return nullptr;
        }
        long ub = m->n_ubatch;                  // adaptive: the largest micro-batch
        P.ubatch_fixed = false;
        if (const std::string ubs = opt_get("TS_MUSE_SPARK_PIPELINE_UBATCH"); !ubs.empty())
        {
            char * end = nullptr;
            errno = 0;
            const long v = strtol(ubs.c_str(), &end, 10);
            if (end == ubs.c_str() || *end != '\0' || errno == ERANGE || v < 1 || v > m->n_ubatch)
            {
                tsg::report_load_refusal("[muse-spark] TS_MUSE_SPARK_PIPELINE_UBATCH must be an integer in [1, %d] (the prefill ubatch), not '%s'\n",
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
            tsg::report_load_refusal("[muse-spark] TS_MUSE_SPARK_PIPELINE=1 needs GPUs whose streams can wait on another GPU's events "
                                     "(CUDA, ROCm, MUSA); the %s backend's %d GPUs cannot run a pipelined prefill. "
                                     "Unset the option (automatic) or set it to 0\n", gpu_reg ? gpu_reg : "unknown", n_gpu);
            return nullptr;
        }
        const char * why = nullptr;
        if (m->tp.n > 1) why = "tensor parallel: every GPU works on every layer";
        else if (mode == "0") why = "TS_MUSE_SPARK_PIPELINE=0";
        else if (env_long("TS_MUSE_SPARK_PROFILE", 0) > 0) why = "TS_MUSE_SPARK_PROFILE synchronizes every node";
        else if (!P.forced && n_gpu < 2) why = "fewer than two GPUs";
        else if (!P.forced && !all_cuda) why = "needs CUDA's asynchronous cross-GPU copies";
        else if (!P.forced && n_cpu_moe > 0) why = "host experts stall each micro-batch";
        P.enabled = why == nullptr;
        if (P.enabled && n_gpu > 0 && env_flag("TS_MUSE_SPARK_WARM_POOLS", true))
        {
            // Build the pipeline's scheduler now rather than in the first request,
            // and check that it engages: two fixed micro-batches, or four of the
            // smallest adaptive ones.
            const int64_t nt = std::min<int64_t>(P.ubatch_fixed ? 2 * P.ubatch : 4 * std::min<int64_t>(PIPE_MIN_UBATCH, P.ubatch),
                                                 m->n_ctx);
            std::vector<int32_t> toks((size_t) nt);
            for (int64_t i = 0; i < nt; i++) toks[(size_t) i] = (int32_t) (1 + i % std::max<int32_t>(1, hp.n_vocab - 1));
            const uint64_t before = P.n_calls;
            forward(*m, toks.data(), nt, nullptr, false);
            s0->n_past = 0;
            std::fill(s0->swa_lab.begin(), s0->swa_lab.end(), -1);
            m->graph_cache.clear();
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
            else snprintf(rows, sizeof(rows), "micro-batches of a quarter of each prompt, %" PRId64 "-%" PRId64 " rows",
                          std::min<int64_t>(PIPE_MIN_UBATCH, P.ubatch), P.ubatch);
            if (P.sched)
                fprintf(stderr, "[muse-spark] pipelined prefill: on (%s over %zu GPU stage(s), %d input copies)\n",
                        rows, P.stages.size(), P.n_copies);
            else
                fprintf(stderr, "[muse-spark] pipelined prefill: on (%s over %zu GPU stage(s); its scheduler is built on first use)\n",
                        rows, P.stages.size());
        }
        else
            fprintf(stderr, "[muse-spark] pipelined prefill: off (%s); prompts run one %d-row micro-batch at a time\n", why, m->n_ubatch);
    }

    char placement[64];
    if (m->tp.n > 1) snprintf(placement, sizeof(placement), "tensor-parallel over %d GPUs", m->tp.n);
    else snprintf(placement, sizeof(placement), "on %d GPU(s)", n_gpu);
    fprintf(stderr, "[muse-spark] loaded %d layers %s + host in %.1fs: ctx %d, sliding ring %d, ubatch %d, "
            "fused qkvg %s, op offload %s, host weights %.1f GiB\n",
            hp.n_layer, placement, std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count(),
            m->n_ctx, m->swa_ring, m->n_ubatch, m->fused_qkvg ? "yes" : "no", m->op_offload ? "on" : "off",
            m->mmap_weight_bytes / 1073741824.0);
    return m.release();
}

} // namespace tsg_mspark

using namespace tsg_mspark;

// ---------------------------------------------------------------------------
// C ABI
// ---------------------------------------------------------------------------
TSG_EXPORT void * TSGgml_MuseSparkLoadModel(const char * gguf_path, int n_gpu, int n_ctx, int n_ubatch, int n_threads,
                                            int n_cpu_moe, const char * backend_name, int ctx_is_hard_limit,
                                            const char * layer_split)
{
    try
    {
        return load(gguf_path, n_gpu, n_ctx, n_ubatch, n_threads, n_cpu_moe, backend_name, ctx_is_hard_limit != 0, layer_split, 0);
    }
    catch (const std::exception & e)
    {
        tsg::report_load_refusal("[muse-spark] load failed: %s\n", e.what());
        return nullptr;
    }
}

/// TSGgml_MuseSparkLoadModel with tensor parallelism: tp > 1 splits every
/// layer over the first tp CUDA GPUs (n_gpu and layer_split then do not apply);
/// tp <= 1 is the layer split.
TSG_EXPORT void * TSGgml_MuseSparkLoadModelTp(const char * gguf_path, int n_gpu, int n_ctx, int n_ubatch, int n_threads,
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
        tsg::report_load_refusal("[muse-spark] load failed: %s\n", e.what());
        return nullptr;
    }
}

/// Set (value non-null) or clear (null) an option that overrides the process
/// environment for subsequent loads and calls.
TSG_EXPORT void TSGgml_MuseSparkSetOption(const char * name, const char * value)
{
    if (!name || !*name) return;
    std::lock_guard<std::mutex> lk(g_opt_mu);
    if (value) g_opts[name] = value;
    else g_opts.erase(name);
}

TSG_EXPORT void TSGgml_MuseSparkFree(void * handle)
{
    delete (model *) handle;
}

TSG_EXPORT int TSGgml_MuseSparkVocabSize(void * handle) { return handle ? ((model *) handle)->hp.n_vocab : 0; }
TSG_EXPORT int TSGgml_MuseSparkCtxSize(void * handle) { return handle ? ((model *) handle)->n_ctx : 0; }
TSG_EXPORT int TSGgml_MuseSparkHiddenSize(void * handle) { return handle ? ((model *) handle)->hp.n_embd : 0; }

TSG_EXPORT int TSGgml_MuseSparkNPast(void * handle)
{
    model * m = (model *) handle;
    if (!m) return -1;
    std::lock_guard<std::mutex> lk(m->mu);
    return m->active_slot ? (int) m->active_slot->n_past : -1;
}

TSG_EXPORT int TSGgml_MuseSparkForward(void * handle, const int32_t * tokens, int n_tokens, float * logits_out, int all_rows)
{
    model * m = (model *) handle;
    if (!m || !tokens || n_tokens <= 0) return 0;
    std::lock_guard<std::mutex> lk(m->mu);
    try { return forward(*m, tokens, n_tokens, logits_out, all_rows != 0) ? 1 : 0; }
    catch (const std::exception & e) { fprintf(stderr, "[muse-spark] forward failed: %s\n", e.what()); return 0; }
}

TSG_EXPORT int TSGgml_MuseSparkForwardBatchedDecode(void * handle, int n, const int32_t * slot_ids, const int32_t * tokens,
                                                    const int32_t * positions, float * logits_out)
{
    model * m = (model *) handle;
    if (!m || n < 1 || !slot_ids || !tokens || !positions || !logits_out) return 0;
    std::lock_guard<std::mutex> lk(m->mu);
    try { return forward_batched_decode(*m, n, slot_ids, tokens, positions, logits_out) ? 1 : 0; }
    catch (const std::exception & e) { fprintf(stderr, "[muse-spark] batched decode failed: %s\n", e.what()); return 0; }
}

/// Several slots' token segments in one graph; one logits row per segment.
TSG_EXPORT int TSGgml_MuseSparkForwardSegments(void * handle, int n, const int32_t * slot_ids, const int32_t * lens,
                                               const int32_t * tokens, const int32_t * positions, float * logits_out)
{
    model * m = (model *) handle;
    if (!m || n < 1 || !slot_ids || !lens || !tokens || !positions || !logits_out) return 0;
    std::lock_guard<std::mutex> lk(m->mu);
    try { return forward_segments(*m, n, slot_ids, lens, tokens, positions, logits_out) ? 1 : 0; }
    catch (const std::exception & e) { fprintf(stderr, "[muse-spark] segment forward failed: %s\n", e.what()); return 0; }
}

TSG_EXPORT int TSGgml_MuseSparkUbatch(void * handle) { return handle ? ((model *) handle)->n_ubatch : 0; }

TSG_EXPORT int TSGgml_MuseSparkQueueVisionRows(void * handle, const float * rows, int n_rows, int index)
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

TSG_EXPORT void TSGgml_MuseSparkClearVisionRows(void * handle)
{
    model * m = (model *) handle;
    if (!m) return;
    std::lock_guard<std::mutex> lk(m->mu);
    m->embd_ovr.clear();
}

TSG_EXPORT int TSGgml_MuseSparkResetChecked(void * handle)
{
    model * m = (model *) handle;
    if (!m) return 0;
    std::lock_guard<std::mutex> lk(m->mu);
    if (!m->active_slot) return 0;
    m->active_slot->n_past = 0;
    std::fill(m->active_slot->swa_lab.begin(), m->active_slot->swa_lab.end(), -1);
    m->embd_ovr.clear();
    return 1;
}

TSG_EXPORT int TSGgml_MuseSparkRewind(void * handle, int n_past)
{
    model * m = (model *) handle;
    if (!m) return 0;
    std::lock_guard<std::mutex> lk(m->mu);
    slot * s = m->active_slot;
    if (!s || n_past < 0 || n_past > s->n_past) return 0;
    if (n_past == s->n_past) return 1;
    // Global layers keep every row, so their part of any rewind is exact. A
    // sliding layer's ring must still hold the window the next query reads.
    for (int64_t p = std::max<int64_t>(0, (int64_t) n_past - (m->hp.n_swa - 1)); p < n_past; p++)
        if (s->swa_lab[(size_t) (p % m->swa_ring)] != p) return 0;
    for (auto & lab : s->swa_lab) if (lab >= n_past) lab = -1;
    s->n_past = n_past;
    return 1;
}

TSG_EXPORT int TSGgml_MuseSparkSlotAlloc(void * handle)
{
    model * m = (model *) handle;
    if (!m) return -1;
    std::lock_guard<std::mutex> lk(m->mu);
    slot * s = slot_alloc(*m, true);
    return s ? s->id : -1;
}

TSG_EXPORT int TSGgml_MuseSparkSetActiveSlot(void * handle, int slot_id)
{
    model * m = (model *) handle;
    if (!m) return 0;
    std::lock_guard<std::mutex> lk(m->mu);
    auto it = m->slots.find(slot_id);
    if (it == m->slots.end()) return 0;
    m->active_slot = it->second.get();
    return 1;
}

TSG_EXPORT int TSGgml_MuseSparkSlotFree(void * handle, int slot_id)
{
    model * m = (model *) handle;
    if (!m) return 0;
    std::lock_guard<std::mutex> lk(m->mu);
    if (m->active_slot && m->active_slot->id == slot_id) return 0;
    if (!m->slots.count(slot_id)) return 0;
    slot_free(*m, slot_id);
    return 1;
}

/// Make slot `dst` an independent copy of slot `src` (its rows, ring labels
/// and head). Continuing either afterwards never disturbs the other.
TSG_EXPORT int TSGgml_MuseSparkSlotCopy(void * handle, int src_id, int dst_id)
{
    model * m = (model *) handle;
    if (!m || src_id == dst_id) return 0;
    std::lock_guard<std::mutex> lk(m->mu);
    auto si = m->slots.find(src_id), di = m->slots.find(dst_id);
    if (si == m->slots.end() || di == m->slots.end()) return 0;
    slot & s = *si->second;
    slot & d = *di->second;
    try
    {
        const int64_t global_rows = std::min<int64_t>(s.cap_g, pad_to(std::max<int64_t>(s.n_past, 1), m->kv_pad));
        if (!slot_reserve_rows(*m, d, global_rows)) return 0;
        std::vector<row_copy> copies;
        for (int il = 0; il < m->hp.n_layer; il++)
        {
            const int64_t rows = m->layers[(size_t) il].swa ? m->swa_ring : global_rows;
            copies.push_back({ s.k[(size_t) il], d.k[(size_t) il], rows });
            copies.push_back({ s.v[(size_t) il], d.v[(size_t) il], rows });
        }
        if (!copy_cache_rows(*m, copies)) return 0;
        d.swa_lab = s.swa_lab;
        d.n_past = s.n_past;
        return 1;
    }
    catch (const std::exception & e)
    {
        fprintf(stderr, "[muse-spark] slot copy failed: %s\n", e.what());
        return 0;
    }
}

TSG_EXPORT int TSGgml_MuseSparkSlotNPast(void * handle, int slot_id)
{
    model * m = (model *) handle;
    if (!m) return -1;
    std::lock_guard<std::mutex> lk(m->mu);
    auto it = m->slots.find(slot_id);
    return it == m->slots.end() ? -1 : (int) it->second->n_past;
}

TSG_EXPORT int TSGgml_MuseSparkSetReclaimableSlots(void * handle, const int32_t * slot_ids, int n)
{
    model * m = (model *) handle;
    if (!m || n < 0 || (n > 0 && !slot_ids)) return 0;
    std::lock_guard<std::mutex> lk(m->mu);
    m->reclaimable.assign(slot_ids, slot_ids + n);
    return 1;
}

TSG_EXPORT int TSGgml_MuseSparkTakeReclaimedSlots(void * handle, int32_t * out, int cap)
{
    model * m = (model *) handle;
    if (!m || cap < 0 || (cap > 0 && !out)) return 0;
    std::lock_guard<std::mutex> lk(m->mu);
    const int n = std::min(cap, (int) m->reclaimed.size());
    for (int i = 0; i < n; i++) out[i] = m->reclaimed[(size_t) i];
    m->reclaimed.erase(m->reclaimed.begin(), m->reclaimed.begin() + n);
    return n;
}

/// Diagnostics: graphs built / reused, host expert bytes, CPU MoE layers.
/// The micro-batch rows a prompt of n tokens starting at position p0 would
/// prefill pipelined with (0: it would run one ubatch at a time), so the
/// scheduler can size a lone prompt's steps to calls that pipeline.
TSG_EXPORT int TSGgml_MuseSparkPipelinePlan(void * handle, int64_t p0, int64_t n)
{
    model * m = (model *) handle;
    if (!m || p0 < 0 || n <= 0) return 0;
    std::lock_guard<std::mutex> lk(m->mu);
    const pipeline_state & P = m->pipe;
    if (!P.enabled) return 0;
    pipe_shape sh;
    if (pipe_plan(*m, p0, n, &sh) != PIPE_OK || sh.p_last > m->n_ctx) return 0;
    // A shape whose compute buffers did not fit is not retried (forward_pipelined).
    if (P.fail_u && sh.u >= P.fail_u && sh.n_kv_g >= P.fail_kv_g && sh.n_kv_s >= P.fail_kv_s) return 0;
    return (int) sh.u;
}

TSG_EXPORT int TSGgml_MuseSparkGetStats(void * handle, int64_t * out, int cap)
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
    return 12;
}
