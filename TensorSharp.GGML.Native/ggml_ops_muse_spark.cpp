// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#include "ggml_ops_internal.h"

using namespace tsg;

// A bounded preprocessing graph: no KV ownership, persistent tensors or model
// state. Packed row padding is readable but never written; Q/K outputs are exact
// contiguous spans. Mark every uploaded leaf and both outputs before gallocr
// planning so late uploads cannot alias a temporary or the other output.
TSG_EXPORT int TSGgml_MuseSparkPrepareQKF32(
    TensorView2DDesc query, TensorView2DDesc key, TensorView2DDesc packed,
    const float* query_norm, const float* key_norm, const int32_t* positions,
    int position_count, int query_heads, int kv_heads, int head_dim,
    float eps, int apply_rope, int rope_dim, float freq_base, float freq_scale)
{
    try
    {
        if (!ensure_backend() || !validate_desc(query, "query") ||
            !validate_desc(key, "key") || !validate_desc(packed, "packed")) return 0;
        const int64_t q_width = static_cast<int64_t>(query_heads) * head_dim;
        const int64_t k_width = static_cast<int64_t>(kv_heads) * head_dim;
        if (q_width <= 0 || k_width <= 0 || q_width > INT32_MAX || k_width > INT32_MAX ||
            (packed.dim0 > 1 && packed.stride0 > (INT64_MAX / 4 - packed.dim1) / (packed.dim0 - 1)))
        {
            set_last_error("muse_spark_qk_prep: dimensions exceed descriptor bounds.");
            return 0;
        }
        const int64_t packed_width = 2 * (q_width + k_width);
        auto exact_output = [](const TensorView2DDesc& d) {
            return d.stride1 == 1 && (d.dim0 == 1 || d.stride0 == d.dim1) &&
                d.raw_bytes == static_cast<int64_t>(d.dim0) * d.dim1 * sizeof(float);
        };
        const int64_t packed_span = (static_cast<int64_t>(packed.dim0 - 1) * packed.stride0 + packed.dim1) * sizeof(float);
        if (query_heads <= 0 || kv_heads <= 0 || head_dim <= 0 || query_heads % kv_heads != 0 ||
            q_width != query.dim1 || k_width != key.dim1 || packed_width != packed.dim1 ||
            query.dim0 != packed.dim0 || key.dim0 != packed.dim0 ||
            !exact_output(query) || !exact_output(key) || packed.stride1 != 1 ||
            packed.stride0 < packed.dim1 || packed.raw_bytes != packed_span ||
            query_norm == nullptr || key_norm == nullptr || !std::isfinite(eps) || eps <= 0 ||
            (apply_rope != 0 && apply_rope != 1) ||
            (apply_rope && (positions == nullptr || position_count != packed.dim0 ||
                rope_dim <= 0 || rope_dim > head_dim || rope_dim % 2 != 0 ||
                !std::isfinite(freq_base) || freq_base <= 0 || !std::isfinite(freq_scale) || freq_scale <= 0)))
        {
            set_last_error("muse_spark_qk_prep: invalid geometry, spans, weights, positions or RoPE parameters.");
            return 0;
        }
        auto overlaps = [](const void* a, size_t an, const void* b, size_t bn) {
            const uintptr_t x = reinterpret_cast<uintptr_t>(a), y = reinterpret_cast<uintptr_t>(b);
            return x <= y ? y - x < an : x - y < bn;
        };
        const size_t norm_bytes = static_cast<size_t>(head_dim) * sizeof(float);
        for (const auto& output : { query, key })
        {
            if (overlaps(output.data, output.raw_bytes, packed.data, packed.raw_bytes) ||
                overlaps(output.data, output.raw_bytes, query_norm, norm_bytes) ||
                overlaps(output.data, output.raw_bytes, key_norm, norm_bytes) ||
                (apply_rope && overlaps(output.data, output.raw_bytes, positions, static_cast<size_t>(position_count) * sizeof(int32_t))))
            {
                set_last_error("muse_spark_qk_prep: outputs must not overlap inputs.");
                return 0;
            }
        }
        if (overlaps(query.data, query.raw_bytes, key.data, key.raw_bytes))
        {
            set_last_error("muse_spark_qk_prep: query and key outputs must not overlap.");
            return 0;
        }
        PooledContextHandle context;
        if (!context.init(1024 * 1024))
        {
            set_last_error("muse_spark_qk_prep: failed to allocate context.");
            return 0;
        }
        auto* ctx = context.value;
        auto input = create_standard_binding(ctx, packed);
        auto q_output = create_standard_binding(ctx, query);
        auto k_output = create_standard_binding(ctx, key);
        auto* qw = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_dim);
        auto* kw = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_dim);
        auto* pos = apply_rope ? ggml_new_tensor_1d(ctx, GGML_TYPE_I32, packed.dim0) : nullptr;
        for (auto* leaf : { input.storage, qw, kw, pos }) if (leaf) ggml_set_input(leaf);
        ggml_set_output(q_output.storage);
        ggml_set_output(k_output.storage);
        auto prepare = [&](int heads, size_t offset, ggml_tensor* weight, ggml_tensor* output) {
            // GGML RoPE indexes positions by ne2 (token), not by flattened head.
            auto* slice = ggml_view_3d(ctx, input.storage, head_dim, heads, packed.dim0,
                static_cast<size_t>(head_dim) * sizeof(float),
                static_cast<size_t>(packed.stride0) * sizeof(float), offset);
            auto* normalized = ggml_mul(ctx, ggml_rms_norm(ctx, ggml_cont(ctx, slice), eps), weight);
            auto* rotated = apply_rope ? ggml_rope_ext(ctx, normalized, pos, nullptr,
                rope_dim, 0, 0, freq_base, freq_scale, 0.f, 1.f, 0.f, 0.f) : normalized;
            auto* flat = ggml_reshape_2d(ctx, rotated, static_cast<int64_t>(heads) * head_dim, packed.dim0);
            auto* copy = ggml_cpy(ctx, flat, output);
            ggml_set_output(copy);
            return copy;
        };
        auto* q_copy = prepare(query_heads, 0, qw, q_output.tensor);
        auto* k_copy = prepare(kv_heads, static_cast<size_t>(q_width) * sizeof(float), kw, k_output.tensor);
        auto* graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, q_copy);
        ggml_build_forward_expand(graph, k_copy);
        BufferHandle fallback(nullptr);
        if (!alloc_graph_reuse_gallocr(graph))
        {
            fallback.value = ggml_backend_alloc_ctx_tensors(ctx, g_backend);
            if (!fallback.value)
            {
                set_last_error("muse_spark_qk_prep: failed to allocate graph buffer.");
                return 0;
            }
        }
        upload_binding(input, packed.data, packed.raw_bytes);
        ggml_backend_tensor_set(qw, query_norm, 0, norm_bytes);
        ggml_backend_tensor_set(kw, key_norm, 0, norm_bytes);
        if (pos) ggml_backend_tensor_set(pos, positions, 0, static_cast<size_t>(position_count) * sizeof(int32_t));
        if (compute_graph(g_backend, graph) != GGML_STATUS_SUCCESS)
        {
            set_last_error("muse_spark_qk_prep: graph execution failed.");
            return 0;
        }
        // The reusable allocator may be repacked by the next op: complete BOTH
        // downloads before returning, independently of asynchronous Metal mode.
        sync_backend(g_backend);
        ggml_backend_tensor_get(q_output.storage, query.data, 0, query.raw_bytes);
        ggml_backend_tensor_get(k_output.storage, key.data, 0, key.raw_bytes);
        clear_last_error();
        return 1;
    }
    catch (const std::exception& ex) { set_last_error(ex.what()); return 0; }
    catch (...) { set_last_error("muse_spark_qk_prep: unknown failure."); return 0; }
}

// Spark's FFN scale precedes RMS normalization. Neither that scale nor the
// two residual gates may be folded through the normalization operation.
TSG_EXPORT int TSGgml_MuseSparkPostNormResidualF32(
    TensorView2DDesc result, TensorView2DDesc stream, TensorView2DDesc branch,
    void* alpha_data, void* beta_data, void* pre_scale_data, int channels, float eps)
{
    try
    {
        if (!ensure_backend() || !validate_desc(result, "result") ||
            !validate_desc(stream, "stream") || !validate_desc(branch, "branch")) return 0;
        auto contiguous = [](const TensorView2DDesc& d) {
            return d.stride1 == 1 && d.stride0 == d.dim1 &&
                d.raw_bytes == static_cast<int64_t>(d.dim0) * d.dim1 * sizeof(float);
        };
        if (channels <= 0 || result.dim1 != channels ||
            result.dim0 != stream.dim0 || result.dim1 != stream.dim1 ||
            result.dim0 != branch.dim0 || result.dim1 != branch.dim1 ||
            alpha_data == nullptr || beta_data == nullptr || !std::isfinite(eps) || eps <= 0 ||
            !contiguous(result) || !contiguous(stream) || !contiguous(branch))
        {
            set_last_error("muse_spark_residual: invalid shapes, weights, epsilon or layout.");
            return 0;
        }
        PooledContextHandle context;
        if (!context.init(1024 * 1024))
        {
            set_last_error("muse_spark_residual: failed to allocate context.");
            return 0;
        }
        auto output = create_standard_binding(context.value, result);
        auto residual = create_standard_binding(context.value, stream);
        auto input = create_standard_binding(context.value, branch);
        auto* alpha = ggml_new_tensor_1d(context.value, GGML_TYPE_F32, channels);
        auto* beta = ggml_new_tensor_1d(context.value, GGML_TYPE_F32, channels);
        auto* scale = pre_scale_data ? ggml_new_tensor_1d(context.value, GGML_TYPE_F32, channels) : nullptr;
        // All leaves are uploaded before execution, including weights that are
        // first consumed late in the graph. Keep them distinct from temporaries
        // throughout a shared-gallocr plan (and retain output until download).
        for (auto* leaf : { residual.storage, input.storage, alpha, beta, scale })
            if (leaf) ggml_set_input(leaf);
        ggml_set_output(output.storage);
        auto* scaled_input = scale ? ggml_mul(context.value, input.tensor, scale) : input.tensor;
        auto* normed = ggml_rms_norm(context.value, scaled_input, eps);
        auto* gated_stream = ggml_mul(context.value, residual.tensor, alpha);
        auto* gated_branch = ggml_mul(context.value, normed, beta);
        // Match the unfused decoder's stored-F32 boundaries. CUDA may otherwise
        // fuse RMSNorm, beta multiplication and the residual add into one
        // expression; its different rounding can change the next quantized
        // projection's activation bins. Keep one graph and reusable allocation.
        ggml_set_output(normed);
        ggml_set_output(gated_branch);
        auto* sum = ggml_add(context.value, gated_stream, gated_branch);
        auto* copy = ggml_cpy(context.value, sum, output.tensor);
        ggml_set_output(copy);
        auto* graph = ggml_new_graph(context.value);
        ggml_build_forward_expand(graph, copy);
        BufferHandle buffer(nullptr);
        if (!alloc_graph_reuse_gallocr(graph))
        {
            buffer.value = ggml_backend_alloc_ctx_tensors(context.value, g_backend);
            if (!buffer.value)
            {
                set_last_error("muse_spark_residual: failed to allocate backend buffer.");
                return 0;
            }
        }
        upload_binding(residual, stream.data, residual.raw_bytes);
        upload_binding(input, branch.data, input.raw_bytes);
        const std::size_t bytes = static_cast<std::size_t>(channels) * sizeof(float);
        upload_binding({ alpha, alpha, bytes }, alpha_data, bytes);
        upload_binding({ beta, beta, bytes }, beta_data, bytes);
        if (scale) upload_binding({ scale, scale, bytes }, pre_scale_data, bytes);
        if (compute_graph(g_backend, graph) != GGML_STATUS_SUCCESS)
        {
            set_last_error("muse_spark_residual: graph execution failed.");
            return 0;
        }
        finalize_compute(false, output.storage, result.data, output.raw_bytes);
        clear_last_error();
        return 1;
    }
    catch (const std::exception& ex) { set_last_error(ex.what()); return 0; }
    catch (...) { set_last_error("muse_spark_residual: unknown failure."); return 0; }
}

// Spark-only scratch reuse. Callers follow the same serialized compute contract
// as Spark Q/K preparation and residuals: the shared rank/backend gallocr is not
// a concurrent execution lease. Complete the download before any other graph
// can repack it. No caller pointer or context survives this invocation.
// Returns 1 for successful reused-scratch execution, 2 for successful fresh
// allocation (requested or fallback), and 0 for failure. Allocation success
// alone is never reported as an executed optimization.
TSG_EXPORT int TSGgml_MuseSparkNormGateF32(
    TensorView2DDesc result, TensorView2DDesc input, const float* gamma,
    TensorView2DDesc gate, int head_dim, float eps, int with_gate, int reuse_scratch)
{
    try
    {
        if (!ensure_backend() || !validate_desc(result, "result") ||
            !validate_desc(input, "input") || (with_gate && !validate_desc(gate, "gate"))) return 0;
        auto exact_span = [](const TensorView2DDesc& d) {
            return d.stride1 == 1 && d.stride0 >= d.dim1 &&
                (d.dim0 == 1 || d.stride0 <= (INT64_MAX / 4 - d.dim1) / (d.dim0 - 1)) &&
                d.raw_bytes == (static_cast<int64_t>(d.dim0 - 1) * d.stride0 + d.dim1) * sizeof(float);
        };
        if (head_dim <= 0 || input.dim1 % head_dim != 0 || gamma == nullptr ||
            !std::isfinite(eps) || eps <= 0 || (with_gate != 0 && with_gate != 1) ||
            (reuse_scratch != 0 && reuse_scratch != 1) ||
            result.dim0 != input.dim0 || result.dim1 != input.dim1 ||
            !exact_span(result) || (result.dim0 > 1 && result.stride0 != result.dim1) || !exact_span(input) ||
            (with_gate && (gate.dim0 != input.dim0 || gate.dim1 != input.dim1 || !exact_span(gate))))
        {
            set_last_error("muse_spark_norm_gate: invalid dimensions, exact spans, epsilon or mode.");
            return 0;
        }
        auto overlaps = [](const void* a, size_t an, const void* b, size_t bn) {
            const uintptr_t x = reinterpret_cast<uintptr_t>(a), y = reinterpret_cast<uintptr_t>(b);
            return x <= y ? y - x < an : x - y < bn;
        };
        const size_t weight_bytes = static_cast<size_t>(head_dim) * sizeof(float);
        if (overlaps(result.data, result.raw_bytes, input.data, input.raw_bytes) ||
            overlaps(result.data, result.raw_bytes, gamma, weight_bytes) ||
            (with_gate && overlaps(result.data, result.raw_bytes, gate.data, gate.raw_bytes)))
        {
            set_last_error("muse_spark_norm_gate: output must not overlap input storage.");
            return 0;
        }
        PooledContextHandle context;
        if (!context.init(1024 * 1024))
        {
            set_last_error("muse_spark_norm_gate: failed to allocate context.");
            return 0;
        }
        auto* ctx = context.value;
        auto source = create_standard_binding(ctx, input);
        auto output = create_standard_binding(ctx, result);
        TensorBinding gate_input{};
        if (with_gate) gate_input = create_standard_binding(ctx, gate);
        auto* weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_dim);
        for (auto* leaf : { source.storage, weight, gate_input.storage }) if (leaf) ggml_set_input(leaf);
        ggml_set_output(output.storage);
        // Preserve the original per-head RMS -> learned/ones multiplication ->
        // sigmoid gate multiplication. Never move scales through RMSNorm.
        auto* heads = ggml_reshape_2d(ctx, ggml_cont(ctx, source.tensor), head_dim,
            static_cast<int64_t>(input.dim0) * (input.dim1 / head_dim));
        auto* norm = ggml_mul(ctx, ggml_rms_norm(ctx, heads, eps), ggml_cont(ctx, weight));
        auto* flat = ggml_reshape_2d(ctx, norm, input.dim1, input.dim0);
        auto* value = with_gate ? make_fused_act_mul_tensor(ctx, FusedActMulOpCode::SigmoidMul,
            flat, gate_input.tensor) : flat;
        auto* copy = ggml_cpy(ctx, value, output.tensor);
        ggml_set_output(copy);
        auto* graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, copy);
        const bool reused = reuse_scratch && alloc_graph_reuse_gallocr(graph);
        BufferHandle fresh(nullptr);
        if (!reused)
        {
            fresh.value = ggml_backend_alloc_ctx_tensors(ctx, g_backend);
            if (!fresh.value)
            {
                set_last_error("muse_spark_norm_gate: failed to allocate graph buffer.");
                return 0;
            }
        }
        upload_binding(source, input.data, input.raw_bytes);
        ggml_backend_tensor_set(weight, gamma, 0, weight_bytes);
        if (with_gate) upload_binding(gate_input, gate.data, gate.raw_bytes);
        if (compute_graph(g_backend, graph) != GGML_STATUS_SUCCESS)
        {
            set_last_error("muse_spark_norm_gate: graph execution failed.");
            return 0;
        }
        sync_backend(g_backend);
        ggml_backend_tensor_get(output.storage, result.data, 0, result.raw_bytes);
        clear_last_error();
        return reused ? 1 : 2;
    }
    catch (const std::exception& ex) { set_last_error(ex.what()); return 0; }
    catch (...) { set_last_error("muse_spark_norm_gate: unknown failure."); return 0; }
}
