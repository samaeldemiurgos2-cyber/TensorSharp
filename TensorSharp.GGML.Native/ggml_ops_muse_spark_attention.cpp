// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#include "ggml_ops_internal.h"
#include "ggml_ops_attention_precision.h"
#if defined(TSG_GGML_USE_CUDA)
#include "ggml_ops_dsv4_fused.h"
#endif

using namespace tsg;

namespace {
    // Ordinary ggml CUDA FlashAttention converts F32 K/V operands to F16.
    // Spark's explicit F32 route instead uses the existing TensorSharp-owned
    // online/tiled attention implementation. Successful return 2 identifies
    // actual owned CUDA execution; it is not an upstream FlashAttention count.
    constexpr int owned_cuda_success = 2;

    bool exact_matrix(const TensorView2DDesc& d, int64_t rows, int64_t columns)
    {
        constexpr int64_t element_bytes = sizeof(float);
        return d.data != nullptr && rows > 0 && columns > 0 &&
            rows <= INT32_MAX && columns <= INT32_MAX &&
            rows <= INT64_MAX / element_bytes / columns &&
            d.dim0 == rows && d.dim1 == columns && d.stride1 == 1 &&
            d.stride0 == columns && d.raw_bytes == rows * columns * element_bytes;
    }

    bool overlaps(const void* a, int64_t an, const void* b, int64_t bn)
    {
        if (a == nullptr || b == nullptr || an <= 0 || bn <= 0) return false;
        const auto x = reinterpret_cast<uintptr_t>(a), y = reinterpret_cast<uintptr_t>(b);
        return x <= y ? y - x < static_cast<uint64_t>(an)
                      : x - y < static_cast<uint64_t>(bn);
    }

    bool valid_geometry(int heads, int kv_heads, int width, int window, float scale)
    {
        return heads > 0 && kv_heads > 0 && heads % kv_heads == 0 &&
            width > 0 && width <= 2048 && window >= 0 && std::isfinite(scale) && scale > 0 &&
            static_cast<int64_t>(heads) * width <= INT32_MAX &&
            static_cast<int64_t>(kv_heads) * width <= INT32_MAX;
    }

    ggml_backend_t owned_backend_locked()
    {
#if defined(TSG_GGML_USE_CUDA)
        // Caller holds the same per-rank mutex as explicit-F32 projections.
        // Shutdown and physical-device reassignment retire this wrapper before
        // its raw backend. No additional persistent graph/session owns it.
        auto& state = dev();
        if (state.spark_float32_backend == nullptr)
            state.spark_float32_backend = tsg_dsv4_fused_backend_init(g_backend);
        return state.spark_float32_backend;
#else
        return nullptr;
#endif
    }

    bool require_cuda()
    {
        if (!ensure_backend()) return false;
#if defined(TSG_GGML_USE_CUDA)
        if (g_backend_type == BACKEND_TYPE_CUDA) return true;
#endif
        set_last_error("Muse Spark explicit-F32 attention requires CUDA.");
        return false;
    }

    // All inputs are exact contiguous spans. Format 0 is head-first; format 1
    // is token-first (including gathered paged KV). Output is always token-first.
    bool attention_chunk(const float* query, const float* key, const float* value, float* output,
        int heads, int kv_heads, int width, int rows, int keys, int first_position,
        int window, float scale, int format, int query_head_stride, ggml_backend_t executor)
    {
        PooledContextHandle context;
        if (!context.init(4 * 1024 * 1024))
        {
            set_last_error("Muse Spark F32 attention failed to create graph context.");
            return false;
        }
        auto* ctx = context.value;
        auto* q_storage = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, width,
            format == 0 ? rows : heads, format == 0 ? heads : rows);
        auto* k_storage = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, width,
            format == 0 ? keys : kv_heads, format == 0 ? kv_heads : keys);
        auto* v_storage = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, width,
            format == 0 ? keys : kv_heads, format == 0 ? kv_heads : keys);
        auto* mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, keys, rows);
        auto* destination = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
            static_cast<int64_t>(heads) * width, rows);
        for (auto* input : {q_storage, k_storage, v_storage, mask}) ggml_set_input(input);
        ggml_set_output(destination);
        auto* q = format == 0 ? q_storage : ggml_permute(ctx, q_storage, 0, 2, 1, 3);
        auto* k = format == 0 ? k_storage : ggml_permute(ctx, k_storage, 0, 2, 1, 3);
        auto* v = format == 0 ? v_storage : ggml_permute(ctx, v_storage, 0, 2, 1, 3);
        auto* attended = tsg_attention_f32(ctx, q, k, v, mask, nullptr, scale);
        auto* flat = ggml_reshape_2d(ctx, attended, static_cast<int64_t>(heads) * width, rows);
        auto* copied = ggml_cpy(ctx, flat, destination);
        ggml_set_output(copied);
        auto* graph = ggml_new_graph_custom(ctx, 8192, false);
        ggml_build_forward_expand(graph, copied);

        BufferHandle fallback(nullptr);
        if (!alloc_graph_reuse_gallocr(graph))
        {
            fallback.value = ggml_backend_alloc_ctx_tensors(ctx, g_backend);
            if (fallback.value == nullptr)
            {
                set_last_error("Muse Spark F32 attention failed to allocate graph tensors.");
                return false;
            }
        }
        // Every uploaded leaf is pinned before allocation. Finish all downloads
        // before the next operation can repack this rank's reusable allocator.
        host_read_barrier();
        if (format == 0 && query_head_stride != rows)
        {
            const size_t head_bytes = static_cast<size_t>(rows) * width * sizeof(float);
            for (int head = 0; head < heads; ++head)
                ggml_backend_tensor_set(q_storage,
                    query + static_cast<size_t>(head) * query_head_stride * width,
                    static_cast<size_t>(head) * head_bytes, head_bytes);
        }
        else ggml_backend_tensor_set(q_storage, query, 0, ggml_nbytes(q_storage));
        ggml_backend_tensor_set(k_storage, key, 0, ggml_nbytes(k_storage));
        ggml_backend_tensor_set(v_storage, value, 0, ggml_nbytes(v_storage));
        thread_local std::vector<float> mask_values;
        mask_values.resize(static_cast<size_t>(keys) * rows);
        for (int row = 0; row < rows; ++row)
        {
            const int64_t position = static_cast<int64_t>(first_position) + row;
            const int64_t first = window > 0 ? std::max<int64_t>(0, position - window + 1) : 0;
            float* mask_row = mask_values.data() + static_cast<size_t>(row) * keys;
            for (int key_index = 0; key_index < keys; ++key_index)
                mask_row[key_index] = key_index < first || key_index > position ? -INFINITY : 0.0f;
        }
        ggml_backend_tensor_set(mask, mask_values.data(), 0, ggml_nbytes(mask));
        if (compute_graph(executor, graph) != GGML_STATUS_SUCCESS)
        {
            set_last_error("Muse Spark owned F32 attention graph execution failed.");
            return false;
        }
        sync_backend(g_backend);
        ggml_backend_tensor_get(destination, output, 0, ggml_nbytes(destination));
        return true;
    }

    bool attention(const float* query, const float* key, const float* value, float* output,
        int heads, int kv_heads, int width, int rows, int keys, int first_position,
        int window, float scale, int format, ggml_backend_t executor)
    {
        // Bound graph metadata and the additive mask even when the configured
        // prefill chunk is larger than the usual 256 rows. One mask row is the
        // minimum; allocation failure remains a reported backend failure.
        const int chunk_rows = static_cast<int>(std::max<int64_t>(1,
            std::min<int64_t>(256, (32LL * 1024 * 1024) / (static_cast<int64_t>(keys) * sizeof(float)))));
        for (int first = 0; first < rows;)
        {
            const int count = std::min(chunk_rows, rows - first);
            const size_t query_offset = static_cast<size_t>(first) * width * (format == 0 ? 1 : heads);
            const size_t output_offset = static_cast<size_t>(first) * width * heads;
            if (!attention_chunk(query + query_offset, key, value, output + output_offset,
                    heads, kv_heads, width, count, keys, first_position + first,
                    window, scale, format, rows, executor)) return false;
            first += count;
        }
        return true;
    }
}

TSG_EXPORT int TSGgml_MuseSparkPrefillAttentionF32(
    TensorView2DDesc query, TensorView2DDesc key, TensorView2DDesc value, TensorView2DDesc output,
    int num_heads, int num_kv_heads, int head_dim, int seq_len, int kv_len,
    int mask_start_pos, int sliding_window, float scale, int input_format)
{
    try
    {
        if (!valid_geometry(num_heads, num_kv_heads, head_dim, sliding_window, scale) ||
            seq_len <= 0 || kv_len <= 0 || mask_start_pos < 0 ||
            static_cast<int64_t>(mask_start_pos) + seq_len > kv_len ||
            (input_format != 0 && input_format != 1))
        {
            set_last_error("Invalid Muse Spark F32 attention geometry or positions.");
            return 0;
        }
        const int64_t q_rows = input_format == 0 ? num_heads : seq_len;
        const int64_t q_columns = static_cast<int64_t>(input_format == 0 ? seq_len : num_heads) * head_dim;
        const int64_t kv_rows = input_format == 0 ? num_kv_heads : kv_len;
        const int64_t kv_columns = static_cast<int64_t>(input_format == 0 ? kv_len : num_kv_heads) * head_dim;
        if (!exact_matrix(query, q_rows, q_columns) ||
            !exact_matrix(key, kv_rows, kv_columns) || !exact_matrix(value, kv_rows, kv_columns) ||
            !exact_matrix(output, seq_len, static_cast<int64_t>(num_heads) * head_dim) ||
            overlaps(output.data, output.raw_bytes, query.data, query.raw_bytes) ||
            overlaps(output.data, output.raw_bytes, key.data, key.raw_bytes) ||
            overlaps(output.data, output.raw_bytes, value.data, value.raw_bytes))
        {
            set_last_error("Muse Spark F32 attention requires exact disjoint output and contiguous input spans.");
            return 0;
        }
        if (!require_cuda()) return 0;
        std::lock_guard<std::mutex> lock(dev().spark_float32_mutex);
        auto* executor = owned_backend_locked();
        if (executor == nullptr)
        {
            set_last_error("Muse Spark owned F32 CUDA executor is unavailable.");
            return 0;
        }
        if (!attention(static_cast<const float*>(query.data), static_cast<const float*>(key.data),
                static_cast<const float*>(value.data), static_cast<float*>(output.data),
                num_heads, num_kv_heads, head_dim, seq_len, kv_len, mask_start_pos,
                sliding_window, scale, input_format, executor)) return 0;
        clear_last_error();
        return owned_cuda_success;
    }
    catch (const std::exception& ex) { set_last_error(ex.what()); return 0; }
    catch (...) { set_last_error("Unknown Muse Spark F32 attention failure."); return 0; }
}

TSG_EXPORT int TSGgml_MuseSparkPagedAttentionF32(
    TensorView2DDesc query, TensorView2DDesc key_pages, TensorView2DDesc value_pages, TensorView2DDesc output,
    const int* query_start_loc, const int* seq_lens, const int* positions,
    const int* block_table_flat, const int* block_table_offsets,
    int num_seqs, int block_table_entries, int num_heads, int num_kv_heads, int head_dim,
    int block_size, int sliding_window, float scale)
{
    try
    {
        if (!valid_geometry(num_heads, num_kv_heads, head_dim, sliding_window, scale) ||
            num_seqs <= 0 || block_table_entries <= 0 || block_size <= 0 ||
            query_start_loc == nullptr || seq_lens == nullptr || positions == nullptr ||
            block_table_flat == nullptr || block_table_offsets == nullptr)
        {
            set_last_error("Invalid Muse Spark F32 paged attention geometry or metadata.");
            return 0;
        }
        const int64_t q_width = static_cast<int64_t>(num_heads) * head_dim;
        const int64_t kv_width = static_cast<int64_t>(num_kv_heads) * head_dim;
        if (!exact_matrix(query, query.dim0, q_width) || !exact_matrix(output, query.dim0, q_width) ||
            !exact_matrix(key_pages, key_pages.dim0, kv_width) ||
            !exact_matrix(value_pages, key_pages.dim0, kv_width) || key_pages.dim0 % block_size != 0 ||
            overlaps(output.data, output.raw_bytes, query.data, query.raw_bytes) ||
            overlaps(output.data, output.raw_bytes, key_pages.data, key_pages.raw_bytes) ||
            overlaps(output.data, output.raw_bytes, value_pages.data, value_pages.raw_bytes) ||
            overlaps(output.data, output.raw_bytes, query_start_loc, (static_cast<int64_t>(num_seqs) + 1) * sizeof(int)) ||
            overlaps(output.data, output.raw_bytes, seq_lens, static_cast<int64_t>(num_seqs) * sizeof(int)) ||
            overlaps(output.data, output.raw_bytes, positions, static_cast<int64_t>(query.dim0) * sizeof(int)) ||
            overlaps(output.data, output.raw_bytes, block_table_flat, static_cast<int64_t>(block_table_entries) * sizeof(int)) ||
            overlaps(output.data, output.raw_bytes, block_table_offsets, static_cast<int64_t>(num_seqs) * sizeof(int)))
        {
            set_last_error("Invalid Muse Spark F32 paged attention spans or output alias.");
            return 0;
        }
        if (query_start_loc[0] != 0 || query_start_loc[num_seqs] != query.dim0)
        {
            set_last_error("Muse Spark F32 paged query ranges must cover the supplied rows.");
            return 0;
        }
        struct Sequence { int start, rows, first_position, first_key, live_keys, table_offset; };
        std::vector<Sequence> sequences;
        int max_live = 0;
        const int physical_blocks = key_pages.dim0 / block_size;
        // Validate ALL sequence/page metadata before writing any output.
        for (int s = 0; s < num_seqs; ++s)
        {
            const int begin = query_start_loc[s], end = query_start_loc[s + 1];
            if (begin < 0 || begin > end || end > query.dim0 || seq_lens[s] < 0)
            {
                set_last_error("Invalid Muse Spark F32 paged query range.");
                return 0;
            }
            if (begin == end) continue;
            const int first_position = positions[begin], length = seq_lens[s];
            const int table_offset = block_table_offsets[s];
            const int64_t needed_blocks = (static_cast<int64_t>(length) + block_size - 1) / block_size;
            if (first_position < 0 || length <= 0 ||
                static_cast<int64_t>(first_position) + end - begin > length ||
                table_offset < 0 || table_offset > block_table_entries ||
                needed_blocks > block_table_entries - table_offset)
            {
                set_last_error("Invalid Muse Spark F32 paged positions or block-table range.");
                return 0;
            }
            for (int row = begin; row < end; ++row)
                if (positions[row] != static_cast<int64_t>(first_position) + row - begin)
                {
                    set_last_error("Muse Spark F32 paged queries must have consecutive positions within each sequence.");
                    return 0;
                }
            const int first_key = sliding_window > 0 ? std::max(0, first_position - sliding_window + 1) : 0;
            // SWA never dereferences older pages. They may be evicted/sentinel
            // entries; validate every page in the live union, not unused history.
            for (int64_t block = first_key / block_size; block < needed_blocks; ++block)
                if (block_table_flat[table_offset + block] < 0 ||
                    block_table_flat[table_offset + block] >= physical_blocks)
                {
                    set_last_error("Muse Spark F32 paged block ID exceeds the supplied page storage.");
                    return 0;
                }
            const int live = length - first_key;
            max_live = std::max(max_live, live);
            sequences.push_back({begin, end - begin, first_position, first_key, live, table_offset});
        }
        if (!require_cuda()) return 0;
        std::lock_guard<std::mutex> lock(dev().spark_float32_mutex);
        auto* executor = owned_backend_locked();
        if (executor == nullptr)
        {
            set_last_error("Muse Spark owned F32 CUDA executor is unavailable.");
            return 0;
        }
        thread_local std::vector<float> gathered_keys, gathered_values;
        gathered_keys.resize(static_cast<size_t>(max_live) * kv_width);
        gathered_values.resize(static_cast<size_t>(max_live) * kv_width);
        const auto* k = static_cast<const float*>(key_pages.data);
        const auto* v = static_cast<const float*>(value_pages.data);
        for (const auto& sequence : sequences)
        {
            const int end = sequence.first_key + sequence.live_keys;
            for (int key_index = sequence.first_key; key_index < end;)
            {
                const int block_id = block_table_flat[sequence.table_offset + key_index / block_size];
                const int in_block = key_index % block_size;
                const int count = std::min(block_size - in_block, end - key_index);
                const size_t source = (static_cast<size_t>(block_id) * block_size + in_block) * kv_width;
                const size_t destination = static_cast<size_t>(key_index - sequence.first_key) * kv_width;
                const size_t bytes = static_cast<size_t>(count) * kv_width * sizeof(float);
                std::memcpy(gathered_keys.data() + destination, k + source, bytes);
                std::memcpy(gathered_values.data() + destination, v + source, bytes);
                key_index += count;
            }
            if (!attention(static_cast<const float*>(query.data) + static_cast<size_t>(sequence.start) * q_width,
                    gathered_keys.data(), gathered_values.data(),
                    static_cast<float*>(output.data) + static_cast<size_t>(sequence.start) * q_width,
                    num_heads, num_kv_heads, head_dim, sequence.rows, sequence.live_keys,
                    sequence.first_position - sequence.first_key, sliding_window, scale, 1, executor)) return 0;
        }
        clear_last_error();
        return owned_cuda_success;
    }
    catch (const std::exception& ex) { set_last_error(ex.what()); return 0; }
    catch (...) { set_last_error("Unknown Muse Spark F32 paged attention failure."); return 0; }
}
