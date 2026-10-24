// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
#include "ggml_ops_internal.h"

using namespace tsg;

// MiniMax-M3.1 vision head as ONE device graph: the per-patch projector
// (mm.1 -> erf-GELU -> mm.2), the concatenation of each 2x2 merge block's four
// patch rows, and the patch-merge MLP (mm.merger.fc1 -> erf-GELU -> fc2).
//
// The 32 ViT blocks before it run through the shared Qwen-VL whole-encoder graph
// (TSGgml_Qwen21VisionEncoderF32). Run as public per-op tensor calls, this head
// moved every [patches, 6144] intermediate across PCIe once per matmul, bias and
// activation (~600 MB per full-budget image, ~150 ms on an A40); here only the
// tower output goes up and the [tokens, 6144] embeddings come back. Weights are
// F32 host pointers, bound device-resident through the shared weight cache (the
// caller invalidates them when it frees the weights).
//
// Patches arrive in merge-block order, so the 2x2 concatenation of a contiguous
// [patches, text_hidden] tensor is a free reshape to [patches/cells, cells*text_hidden].

namespace
{
    // Mirrors MiniMaxM3VisionHeadDesc in GgmlMiniMaxM3VisionNative.cs; struct_bytes
    // guards the two layouts against drifting apart.
    struct MiniMaxM3VisionHeadDesc
    {
        std::int32_t struct_bytes;
        std::int32_t num_patches;    // rows of hidden; a multiple of cells
        std::int32_t cells;          // spatial_merge_size^2
        std::int32_t vision_hidden;  // tower width (1280)
        std::int32_t proj_hidden;    // mm.1 output width
        std::int32_t text_hidden;    // mm.2 and mm.merger.fc2 output width
        std::int32_t merge_hidden;   // mm.merger.fc1 output width
        std::int32_t reserved;
        const float* hidden;         // [num_patches, vision_hidden]
        float* output;               // [num_patches / cells, text_hidden]
        const float* w1;  const float* b1;    // [proj_hidden, vision_hidden], [proj_hidden]
        const float* w2;  const float* b2;    // [text_hidden, proj_hidden], [text_hidden]
        const float* wf1; const float* bf1;   // [merge_hidden, cells * text_hidden], [merge_hidden]
        const float* wf2; const float* bf2;   // [text_hidden, merge_hidden], [text_hidden]
    };

    struct Upload
    {
        ggml_tensor* tensor;
        const void* data;
        std::size_t bytes;
    };

    int minimax_m3_vision_head_impl(const MiniMaxM3VisionHeadDesc& d)
    {
        if (!ensure_backend())
            return 0;
        if (d.struct_bytes != static_cast<std::int32_t>(sizeof(MiniMaxM3VisionHeadDesc)))
        {
            set_last_error("MiniMax-M3 vision head: descriptor layout mismatch.");
            return 0;
        }
        if (d.num_patches <= 0 || d.cells <= 0 || d.num_patches % d.cells != 0 ||
            d.vision_hidden <= 0 || d.proj_hidden <= 0 || d.text_hidden <= 0 || d.merge_hidden <= 0)
        {
            set_last_error("MiniMax-M3 vision head: bad geometry.");
            return 0;
        }
        if (d.hidden == nullptr || d.output == nullptr || d.w1 == nullptr || d.b1 == nullptr ||
            d.w2 == nullptr || d.b2 == nullptr || d.wf1 == nullptr || d.bf1 == nullptr ||
            d.wf2 == nullptr || d.bf2 == nullptr)
        {
            set_last_error("MiniMax-M3 vision head: a required pointer is null.");
            return 0;
        }

        const std::int64_t patches = d.num_patches;
        const std::int64_t tokens = patches / d.cells;
        const std::int64_t merged = static_cast<std::int64_t>(d.cells) * d.text_hidden;

        PooledContextHandle context;
        if (!context.init(static_cast<std::size_t>(1) * 1024 * 1024))
        {
            set_last_error("MiniMax-M3 vision head: ctx init failed.");
            return 0;
        }
        ggml_context* ctx = context.value;

        std::vector<Upload> uploads;
        ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);
        // Device-resident weight keyed by its host pointer (persists across encodes);
        // on a cache miss the tensor becomes a graph input uploaded after allocation.
        // A NEW resident copy is filled before anything can abandon the graph: the
        // cache publishes the entry first and later hits assume it holds the bytes.
        auto bind = [&](ggml_tensor* t, const void* data, std::size_t bytes) {
            if (t == nullptr)
                return false;
            if (bytes >= 4096 && dev != nullptr)
            {
                ggml_backend_buffer_t buf = nullptr;
                void* addr = nullptr;
                bool need = false;
                if (try_get_cacheable_tensor_buffer(g_backend, dev, t, const_cast<void*>(data), bytes, buf, addr, need))
                {
                    if (ggml_backend_tensor_alloc(buf, t, addr) == GGML_STATUS_SUCCESS)
                    {
                        if (need)
                        {
                            host_read_barrier();
                            ggml_backend_tensor_set(t, resolve_upload_source(data), 0, bytes);
                        }
                        return true;
                    }
                    invalidate_cached_buffer(const_cast<void*>(data));
                }
            }
            ggml_set_input(t);
            uploads.push_back({ t, data, bytes });
            return true;
        };
        auto weight = [&](std::int64_t in, std::int64_t out, const float* data) {
            ggml_tensor* t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, in, out);
            return bind(t, data, static_cast<std::size_t>(in * out) * sizeof(float)) ? t : nullptr;
        };
        auto bias = [&](std::int64_t n, const float* data) {
            ggml_tensor* t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
            return bind(t, data, static_cast<std::size_t>(n) * sizeof(float)) ? t : nullptr;
        };

        ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d.vision_hidden, patches);
        ggml_tensor* w1 = weight(d.vision_hidden, d.proj_hidden, d.w1);
        ggml_tensor* b1 = bias(d.proj_hidden, d.b1);
        ggml_tensor* w2 = weight(d.proj_hidden, d.text_hidden, d.w2);
        ggml_tensor* b2 = bias(d.text_hidden, d.b2);
        ggml_tensor* wf1 = weight(merged, d.merge_hidden, d.wf1);
        ggml_tensor* bf1 = bias(d.merge_hidden, d.bf1);
        ggml_tensor* wf2 = weight(d.merge_hidden, d.text_hidden, d.wf2);
        ggml_tensor* bf2 = bias(d.text_hidden, d.bf2);
        if (x == nullptr || w1 == nullptr || b1 == nullptr || w2 == nullptr || b2 == nullptr ||
            wf1 == nullptr || bf1 == nullptr || wf2 == nullptr || bf2 == nullptr)
        {
            set_last_error("MiniMax-M3 vision head: tensor creation failed.");
            return 0;
        }
        ggml_set_input(x);

        // Bias adds broadcast the [n] vector over every row.
        ggml_tensor* z = ggml_add(ctx, ggml_mul_mat(ctx, w1, x), b1);          // [proj_hidden, patches]
        z = ggml_gelu_erf_inplace(ctx, z);
        z = ggml_add(ctx, ggml_mul_mat(ctx, w2, z), b2);                       // [text_hidden, patches]
        z = ggml_reshape_2d(ctx, z, merged, tokens);                           // 2x2 block concat
        z = ggml_add(ctx, ggml_mul_mat(ctx, wf1, z), bf1);                     // [merge_hidden, tokens]
        z = ggml_gelu_erf_inplace(ctx, z);
        ggml_tensor* out = ggml_add(ctx, ggml_mul_mat(ctx, wf2, z), bf2);      // [text_hidden, tokens]
        if (out == nullptr)
        {
            set_last_error("MiniMax-M3 vision head: graph build failed.");
            return 0;
        }
        ggml_set_output(out);

        ggml_cgraph* graph = ggml_new_graph_custom(ctx, 64, false);
        if (graph == nullptr)
        {
            set_last_error("MiniMax-M3 vision head: graph creation failed.");
            return 0;
        }
        ggml_build_forward_expand(graph, out);

        // Local allocator: the [patches, 6144] activations are returned before the
        // language model runs again.
        ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(g_backend));
        if (galloc == nullptr || !ggml_gallocr_alloc_graph(galloc, graph))
        {
            if (galloc != nullptr)
                ggml_gallocr_free(galloc);
            set_last_error("MiniMax-M3 vision head: gallocr allocation failed.");
            return 0;
        }

        host_read_barrier();
        ggml_backend_tensor_set(x, d.hidden, 0, static_cast<std::size_t>(patches * d.vision_hidden) * sizeof(float));
        for (const Upload& u : uploads)
            ggml_backend_tensor_set(u.tensor, resolve_upload_source(u.data), 0, u.bytes);

        const ggml_status status = tsg::compute_graph(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS)
        {
            ggml_gallocr_free(galloc);
            set_last_error("MiniMax-M3 vision head: graph compute failed.");
            return 0;
        }
        finalize_compute_with_download(out, d.output, static_cast<std::size_t>(tokens * d.text_hidden) * sizeof(float));
        host_read_barrier();
        ggml_gallocr_free(galloc);

        clear_last_error();
        return 1;
    }
}

TSG_EXPORT int TSGgml_MiniMaxM3VisionHeadF32(const MiniMaxM3VisionHeadDesc* desc)
{
    try
    {
        if (desc == nullptr)
        {
            set_last_error("MiniMax-M3 vision head: descriptor is null.");
            return 0;
        }
        return minimax_m3_vision_head_impl(*desc);
    }
    catch (const std::exception& ex)
    {
        set_last_error(ex.what());
        return 0;
    }
    catch (...)
    {
        set_last_error("MiniMax-M3 vision head: unknown error.");
        return 0;
    }
}
