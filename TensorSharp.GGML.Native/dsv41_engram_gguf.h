// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#pragma once

#include "dsv41_engram.h"
#include "gguf.h"

namespace tsg_dsv41 {

// Name of the single tensor that holds every Engram table in the llama.cpp
// layout (Unsloth's DeepSeek-V4.1-Flash-1017 GGUFs). TensorSharp's own layout
// keeps one `blk.N.engram_embd.weight` per Engram layer instead.
inline constexpr const char * combined_engram_table = "per_layer_token_embd.weight";

// One Engram layer's slice of the combined table: rows [first_row, first_row + rows).
struct combined_engram_slice {
    int32_t id = 0;
    uint64_t first_row = 0, rows = 0;
};

namespace detail {

[[noreturn]] inline void fail_metadata(const std::string & key) {
    throw std::runtime_error("Missing or invalid DeepSeek V4.1 GGUF metadata: " + key);
}

inline int64_t gguf_integer(gguf_context * file, const char * key, int64_t fallback = -2) {
    const int64_t id = gguf_find_key(file, key);
    if (id < 0) { if (fallback != -2) return fallback; fail_metadata(key); }
    switch (gguf_get_kv_type(file, id)) {
        case GGUF_TYPE_UINT32: return gguf_get_val_u32(file, id);
        case GGUF_TYPE_INT32: return gguf_get_val_i32(file, id);
        case GGUF_TYPE_UINT64: {
            const uint64_t value = gguf_get_val_u64(file, id);
            if (value > uint64_t(INT64_MAX)) fail_metadata(key);
            return int64_t(value);
        }
        case GGUF_TYPE_INT64: return gguf_get_val_i64(file, id);
        default: fail_metadata(key);
    }
}

inline uint32_t gguf_u32(gguf_context * file, const char * key, int64_t fallback = -2) {
    const int64_t value = gguf_integer(file, key, fallback);
    if (value < 0 || uint64_t(value) > UINT32_MAX) fail_metadata(key);
    return uint32_t(value);
}

inline std::vector<uint64_t> gguf_array(gguf_context * file, const char * key, size_t max_count) {
    const int64_t id = gguf_find_key(file, key);
    if (id < 0 || gguf_get_kv_type(file, id) != GGUF_TYPE_ARRAY) fail_metadata(key);
    const size_t count = gguf_get_arr_n(file, id);
    if (!count || count > max_count) fail_metadata(key);
    const void * raw = gguf_get_arr_data(file, id);
    std::vector<uint64_t> values(count);
    for (size_t i = 0; i < count; ++i) {
        switch (gguf_get_arr_type(file, id)) {
            case GGUF_TYPE_UINT32: values[i] = static_cast<const uint32_t *>(raw)[i]; break;
            case GGUF_TYPE_INT32: {
                const auto v = static_cast<const int32_t *>(raw)[i];
                if (v < 0) fail_metadata(key);
                values[i] = uint64_t(v); break;
            }
            case GGUF_TYPE_UINT64: values[i] = static_cast<const uint64_t *>(raw)[i]; break;
            case GGUF_TYPE_INT64: {
                const auto v = static_cast<const int64_t *>(raw)[i];
                if (v < 0) fail_metadata(key);
                values[i] = uint64_t(v); break;
            }
            default: fail_metadata(key);
        }
    }
    return values;
}

} // namespace detail

// True when the file describes Engram with llama.cpp's `deepseek41.ple.*` keys.
inline bool has_combined_engram_metadata(gguf_context * file) {
    return gguf_find_key(file, "deepseek41.engram.layer_ids") < 0 &&
           gguf_find_key(file, "deepseek41.ple.layers") >= 0;
}

// The llama.cpp layout addresses one combined table with offsets that run on
// from one Engram layer to the next. Each layer's buckets must still be one
// contiguous row range, which is what lets the loader treat the slice as that
// layer's own `blk.N.engram_embd.weight` (a byte range of the combined tensor).
inline std::vector<combined_engram_slice> combined_engram_slices(gguf_context * file, uint64_t table_rows) {
    using namespace detail;
    const uint32_t ngram = gguf_u32(file, "deepseek41.ple.ngram_size");
    const uint32_t heads = gguf_u32(file, "deepseek41.ple.heads_per_ngram");
    if (ngram < 2 || ngram > 16 || !heads || heads > 128) fail_metadata("deepseek41.ple dimensions");
    const size_t columns = size_t(ngram - 1) * heads;
    const auto ids = gguf_array(file, "deepseek41.ple.layers", 128);
    const auto primes = gguf_array(file, "deepseek41.ple.head_vocab_sizes", ids.size() * columns);
    const auto offsets = gguf_array(file, "deepseek41.ple.head_offsets", ids.size() * columns);
    if (primes.size() != ids.size() * columns || offsets.size() != ids.size() * columns)
        fail_metadata("deepseek41.ple flattened array lengths");
    std::vector<combined_engram_slice> slices;
    for (size_t i = 0; i < ids.size(); ++i) {
        combined_engram_slice slice;
        if (ids[i] > uint64_t(INT32_MAX)) fail_metadata("deepseek41.ple.layers");
        slice.id = int32_t(ids[i]);
        slice.first_row = offsets[i * columns];
        for (size_t j = 0; j < columns; ++j) {
            const uint64_t prime = primes[i * columns + j];
            if (prime < 2 || offsets[i * columns + j] != slice.first_row + slice.rows)
                fail_metadata("deepseek41.ple bucket layout is not contiguous per layer");
            slice.rows += prime;
        }
        if (slice.rows > uint64_t(INT32_MAX) || slice.first_row > table_rows || slice.rows > table_rows - slice.first_row)
            fail_metadata(std::string("deepseek41.ple bucket rows exceed ") + combined_engram_table);
        slices.push_back(slice);
    }
    return slices;
}

// Engram's compact lookup data is part of the GGUF metadata. TensorShape returns
// the GGML-order dimensions of a tensor across all shards, or an empty vector.
// No Engram table payload is read while validating this metadata.
//
// Two layouts are accepted. TensorSharp's (`deepseek41.engram.*`, one
// `blk.N.engram_embd.weight` per layer, a pad id that is already compressed)
// and llama.cpp's (`deepseek41.ple.*` plus `deepseek41.engram.token_map`/
// `pad_token_id`, one combined table). For the latter the caller must expose
// each layer's slice as `blk.N.engram_embd.weight` (see combined_engram_slices);
// the per-layer offsets are rebased to that slice and the raw pad token is
// mapped through the token map, exactly as engram.py's NgramHashState does.
template<typename TensorShape>
engram_data load_engram_gguf(gguf_context * file, TensorShape tensor_shape) {
    auto fail = [](const std::string & key) -> void { detail::fail_metadata(key); };
    auto integer = [&](const char * key, int64_t fallback = -2) { return detail::gguf_integer(file, key, fallback); };
    auto u32 = [&](const char * key, int64_t fallback = -2) { return detail::gguf_u32(file, key, fallback); };
    auto array = [&](const char * key, size_t max_count) { return detail::gguf_array(file, key, max_count); };
    engram_data result;
    const int64_t tokens = gguf_find_key(file, "tokenizer.ggml.tokens");
    if (tokens < 0 || gguf_get_kv_type(file, tokens) != GGUF_TYPE_ARRAY ||
        gguf_get_arr_type(file, tokens) != GGUF_TYPE_STRING ||
        !gguf_get_arr_n(file, tokens) || gguf_get_arr_n(file, tokens) > 1048576)
        fail("tokenizer.ggml.tokens");
    result.vocab_size = uint32_t(gguf_get_arr_n(file, tokens));
    result.tokenizer_hash = UINT64_C(14695981039346656037);
    for (uint32_t i = 0; i < result.vocab_size; ++i)
        result.tokenizer_hash = engram_data::fingerprint_token(result.tokenizer_hash, gguf_get_arr_str(file, tokens, i));
    const auto embedding = tensor_shape("token_embd.weight");
    if (embedding.size() < 2 || embedding[1] != result.vocab_size)
        fail("token embedding vocabulary does not match tokenizer");
    const bool combined = has_combined_engram_metadata(file);
    std::vector<uint64_t> ids, multipliers, primes, offsets;
    if (combined) {
        result.max_ngram_size = u32("deepseek41.ple.ngram_size");
        result.n_heads = u32("deepseek41.ple.heads_per_ngram");
        result.head_dim = u32("deepseek41.embedding_length_per_layer_input");
    } else {
        result.pad_id = u32("deepseek41.engram.pad_id");
        result.max_ngram_size = u32("deepseek41.engram.max_ngram_size");
        result.n_heads = u32("deepseek41.engram.head_count");
        result.head_dim = u32("deepseek41.engram.key_length");
    }
    if (result.max_ngram_size < 2 || result.max_ngram_size > 16 ||
        !result.n_heads || result.n_heads > 128 || !result.head_dim || result.head_dim > 65536)
        fail("deepseek41.engram dimensions");
    const size_t columns = result.hash_columns();
    if (combined) {
        ids = array("deepseek41.ple.layers", 128);
        multipliers = array("deepseek41.ple.layer_multipliers", ids.size() * result.max_ngram_size);
        primes = array("deepseek41.ple.head_vocab_sizes", ids.size() * columns);
        offsets = array("deepseek41.ple.head_offsets", ids.size() * columns);
    } else {
        ids = array("deepseek41.engram.layer_ids", 128);
        multipliers = array("deepseek41.engram.multipliers", ids.size() * result.max_ngram_size);
        primes = array("deepseek41.engram.primes", ids.size() * columns);
        offsets = array("deepseek41.engram.offsets", ids.size() * columns);
    }
    if (multipliers.size() != ids.size() * result.max_ngram_size ||
        primes.size() != ids.size() * columns || offsets.size() != ids.size() * columns)
        fail("deepseek41.engram flattened array lengths");
    if (combined) {
        // Rebase each layer's offsets onto its own slice of the combined table.
        for (size_t i = 0; i < ids.size(); ++i) {
            const uint64_t first = offsets[i * columns];
            for (size_t j = 0; j < columns; ++j) offsets[i * columns + j] -= first;
        }
    }
    const auto map = array("deepseek41.engram.token_map", result.vocab_size);
    for (uint64_t value : map) {
        if (value >= result.vocab_size) fail("deepseek41.engram.token_map");
        result.token_map.push_back(int32_t(value));
        result.compressed_vocab_size = std::max(result.compressed_vocab_size, uint32_t(value + 1));
    }
    if (combined) {
        // engram.py: `self.pad_id = token_map[args.engram_pad_id]`. The
        // llama.cpp layout stores the raw token, so compress it here.
        const uint32_t pad_token = u32("deepseek41.engram.pad_token_id");
        if (pad_token >= result.token_map.size()) fail("deepseek41.engram.pad_token_id");
        result.pad_id = uint32_t(result.token_map[pad_token]);
        if (gguf_find_key(file, "deepseek41.engram.compressed_vocab_size") >= 0 &&
            u32("deepseek41.engram.compressed_vocab_size") != result.compressed_vocab_size)
            fail("deepseek41.engram.compressed_vocab_size does not match the token map");
    }
    const uint32_t n_layer = u32("deepseek41.block_count");
    if (!n_layer || n_layer > 128) fail("deepseek41.block_count");
    for (size_t i = 0; i < ids.size(); ++i) {
        if (ids[i] >= n_layer) fail("deepseek41.engram.layer_ids");
        engram_data::layer_layout layer;
        layer.id = int32_t(ids[i]);
        const auto shape = tensor_shape("blk." + std::to_string(layer.id) + ".engram_embd.weight");
        if (shape.size() < 2 || shape[0] != result.head_dim || shape[1] <= 0 ||
            (shape.size() > 2 && std::any_of(shape.begin() + 2, shape.end(), [](int64_t d) { return d != 1; })))
            fail("Engram embedding tensor dimensions at layer " + std::to_string(layer.id));
        layer.rows = uint64_t(shape[1]);
        for (size_t j = 0; j < result.max_ngram_size; ++j)
            layer.multipliers.push_back(multipliers[i * result.max_ngram_size + j]);
        for (size_t j = 0; j < columns; ++j) {
            if (primes[i * columns + j] > UINT32_MAX) fail("deepseek41.engram.primes");
            layer.primes.push_back(uint32_t(primes[i * columns + j]));
            layer.offsets.push_back(offsets[i * columns + j]);
        }
        result.layers.push_back(std::move(layer));
    }
    const auto ratios = array("deepseek41.attention.compress_ratios", 256);
    if (ratios.size() < n_layer) fail("deepseek41.attention.compress_ratios");
    int32_t candidate = -1;
    for (uint32_t i = 0; i < n_layer; ++i) {
        const std::string prefix = "blk." + std::to_string(i) + ".";
        if (!tensor_shape(prefix + "engram_embd.weight").empty() &&
            std::find(ids.begin(), ids.end(), i) == ids.end())
            fail("Engram embedding tensor is absent from deepseek41.engram.layer_ids");
        if (!tensor_shape(prefix + "attn_compressor_kv.weight").empty())
            result.kv_source_layer_ids.push_back(int32_t(i));
        if (!tensor_shape(prefix + "indexer.attn_q_b.weight").empty()) {
            result.index_source_layer_ids.push_back(int32_t(i));
            if (candidate < 0 && ratios[i] == 1) candidate = int32_t(i);
        }
    }
    // Published V4.1 Flash omits these config values. Its first ratio-1
    // indexer owns the candidate mask; optional explicit metadata supports
    // checkpoints and small fixtures with a different pruning geometry.
    const int64_t source = integer("deepseek41.attention.indexer.candidate_source_layer", candidate);
    if (source < -1 || source >= n_layer) fail("deepseek41.attention.indexer.candidate_source_layer");
    result.candidate_source_layer_id = int32_t(source);
    if (source >= 0) {
        result.candidate_topk_blocks = u32("deepseek41.attention.indexer.candidate_top_k", 2048);
        result.candidate_block_size = u32("deepseek41.attention.indexer.candidate_block_size", 8);
        if (std::find(result.index_source_layer_ids.begin(), result.index_source_layer_ids.end(), source) ==
            result.index_source_layer_ids.end()) fail("candidate source must own an indexer");
    }
    result.validate(result.vocab_size);
    return result;
}

} // namespace tsg_dsv41
