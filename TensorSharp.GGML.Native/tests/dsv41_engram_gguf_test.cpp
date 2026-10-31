// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#ifdef NDEBUG
#undef NDEBUG
#endif
#include "../dsv41_engram_gguf.h"
#include <cassert>
#include <iostream>
#include <map>
#include <memory>

int main() {
    auto file = std::unique_ptr<gguf_context, decltype(&gguf_free)>(gguf_init_empty(), gguf_free);
    const char * tokens[] = {"a", "b", "c", "d", "e", "f"};
    gguf_set_arr_str(file.get(), "tokenizer.ggml.tokens", tokens, 6);
    auto set = [&](const char * key, std::initializer_list<uint64_t> values) {
        gguf_set_arr_data(file.get(), key, GGUF_TYPE_UINT64, values.begin(), values.size());
    };
    gguf_set_val_u32(file.get(), "deepseek41.block_count", 4);
    gguf_set_val_u32(file.get(), "deepseek41.engram.pad_id", 1);
    gguf_set_val_u32(file.get(), "deepseek41.engram.max_ngram_size", 3);
    gguf_set_val_u32(file.get(), "deepseek41.engram.head_count", 2);
    gguf_set_val_u32(file.get(), "deepseek41.engram.key_length", 2);
    set("deepseek41.engram.layer_ids", {1, 3});
    set("deepseek41.engram.multipliers", {7, 11, 13, 17, 19, 23});
    set("deepseek41.engram.primes", {3, 5, 7, 11, 13, 17, 19, 23});
    set("deepseek41.engram.offsets", {0, 3, 8, 15, 0, 13, 30, 49});
    set("deepseek41.engram.token_map", {0, 2, 1, 2, 3, 1});
    set("deepseek41.attention.compress_ratios", {0, 2, 1, 1});
    std::map<std::string, std::vector<int64_t>> shapes = {
        {"token_embd.weight", {2, 6}}, {"blk.1.engram_embd.weight", {2, 26}}, {"blk.3.engram_embd.weight", {2, 72}},
        {"blk.1.attn_compressor_kv.weight", {2, 2}}, {"blk.2.attn_compressor_kv.weight", {2, 2}},
        {"blk.1.indexer.attn_q_b.weight", {2, 2}}, {"blk.2.indexer.attn_q_b.weight", {2, 2}},
        {"blk.3.indexer.attn_q_b.weight", {2, 2}},
    };
    auto shape = [&](const std::string & name) {
        const auto found = shapes.find(name);
        return found == shapes.end() ? std::vector<int64_t>{} : found->second;
    };
    auto load = [&] { return tsg_dsv41::load_engram_gguf(file.get(), shape); };
    auto bad = [&] {
        bool caught = false;
        try { load(); } catch (const std::runtime_error &) { caught = true; }
        assert(caught);
    };
    const auto data = load();
    assert(data.vocab_size == 6 && data.compressed_vocab_size == 4);
    assert(data.kv_source_layer_ids == std::vector<int32_t>({1, 2}));
    assert(data.index_source_layer_ids == std::vector<int32_t>({1, 2, 3}));
    assert(data.candidate_source_layer_id == 2 && data.candidate_topk_blocks == 2048 && data.candidate_block_size == 8);
    assert(data.layers[0].multipliers == std::vector<uint64_t>({7, 11, 13}));
    assert(data.layers[1].multipliers == std::vector<uint64_t>({17, 19, 23}));
    std::vector<int32_t> history;
    const int32_t input[] = {2, -1, 4};
    const auto rows = data.hash_tokens(input, 3, 0, history);
    // pad_id is already compressed: token_map[1] is 2, but padding is 1.
    // Position 2 uses compressed token 3 then two pad substitutions (1).
    assert(rows[8] == ((3 * 7) ^ 11) % 3);
    assert(rows[10] == ((3 * 7) ^ 11 ^ 13) % 7 + 8);
    assert(rows[20] == ((3 * 17) ^ 19) % 13);
    assert(rows[22] == ((3 * 17) ^ 19 ^ 23) % 19 + 30);
    gguf_set_val_i32(file.get(), "deepseek41.attention.indexer.candidate_source_layer", -1);
    assert(load().candidate_source_layer_id == -1);
    gguf_set_val_i32(file.get(), "deepseek41.attention.indexer.candidate_source_layer", 0); bad();
    gguf_set_val_i32(file.get(), "deepseek41.attention.indexer.candidate_source_layer", 2);
    gguf_set_val_u32(file.get(), "deepseek41.attention.indexer.candidate_top_k", 0); bad();
    gguf_set_val_u32(file.get(), "deepseek41.attention.indexer.candidate_top_k", UINT32_MAX); bad();
    gguf_set_val_u32(file.get(), "deepseek41.attention.indexer.candidate_top_k", 2);
    gguf_set_val_u32(file.get(), "deepseek41.attention.indexer.candidate_block_size", UINT32_MAX); bad();
    gguf_set_val_u32(file.get(), "deepseek41.attention.indexer.candidate_block_size", 2);
    assert(load().candidate_topk_blocks == 2 && load().candidate_block_size == 2);
    // Unsloth's 1017 GGUFs name the budget candidate_top_k_blocks. Either name is
    // read, as the managed loader reads them, and a file with both must agree.
    gguf_set_val_u32(file.get(), "deepseek41.attention.indexer.candidate_top_k_blocks", 2);
    assert(load().candidate_topk_blocks == 2);
    gguf_set_val_u32(file.get(), "deepseek41.attention.indexer.candidate_top_k_blocks", 3); bad();
    gguf_remove_key(file.get(), "deepseek41.attention.indexer.candidate_top_k");
    assert(load().candidate_topk_blocks == 3);
    gguf_set_val_u32(file.get(), "deepseek41.attention.indexer.candidate_top_k_blocks", 0); bad();
    gguf_remove_key(file.get(), "deepseek41.attention.indexer.candidate_top_k_blocks");
    gguf_set_val_u32(file.get(), "deepseek41.attention.indexer.candidate_top_k", 2);
    set("deepseek41.engram.offsets", {0, 3, 8, 15, 0, 13, 31, 49}); bad();
    set("deepseek41.engram.offsets", {0, 3, 8, 15, 0, 13, 30, 49});
    set("deepseek41.engram.multipliers", {7, 11, 13, 17, 19}); bad();
    set("deepseek41.engram.multipliers", {7, 11, 13, 17, 19, UINT64_MAX}); bad();
    set("deepseek41.engram.multipliers", {7, 11, 13, 17, 19, 23});
    set("deepseek41.engram.token_map", {0, 1, 2, 2, 4, 1}); bad();
    set("deepseek41.engram.token_map", {0, 1, 2, 2, 3}); bad();
    set("deepseek41.engram.token_map", {0, 2, 1, 2, 3, 1});
    shapes["blk.0.engram_embd.weight"] = {2, 26}; bad();
    shapes.erase("blk.0.engram_embd.weight");
    shapes["token_embd.weight"][1] = 7; bad();
    shapes["token_embd.weight"][1] = 6;
    gguf_set_val_u32(file.get(), "deepseek41.engram.pad_id", 4); bad();
    gguf_set_val_u32(file.get(), "deepseek41.engram.pad_id", 1);
    shapes["blk.3.engram_embd.weight"][1] = 73; bad();
    shapes["blk.3.engram_embd.weight"][1] = 72;
    gguf_remove_key(file.get(), "deepseek41.engram.token_map"); bad();

    // llama.cpp layout: deepseek41.ple.* keys, offsets running on across the
    // layers of one combined table, and a raw (uncompressed) pad token.
    auto ple = std::unique_ptr<gguf_context, decltype(&gguf_free)>(gguf_init_empty(), gguf_free);
    gguf_set_arr_str(ple.get(), "tokenizer.ggml.tokens", tokens, 6);
    auto pset = [&](const char * key, std::initializer_list<uint64_t> values) {
        gguf_set_arr_data(ple.get(), key, GGUF_TYPE_UINT64, values.begin(), values.size());
    };
    gguf_set_val_u32(ple.get(), "deepseek41.block_count", 4);
    gguf_set_val_u32(ple.get(), "deepseek41.ple.ngram_size", 3);
    gguf_set_val_u32(ple.get(), "deepseek41.ple.heads_per_ngram", 2);
    gguf_set_val_u32(ple.get(), "deepseek41.embedding_length_per_layer_input", 2);
    gguf_set_val_u32(ple.get(), "deepseek41.engram.pad_token_id", 1);
    gguf_set_val_u32(ple.get(), "deepseek41.engram.compressed_vocab_size", 4);
    pset("deepseek41.ple.layers", {1, 3});
    pset("deepseek41.ple.layer_multipliers", {7, 11, 13, 17, 19, 23});
    pset("deepseek41.ple.head_vocab_sizes", {3, 5, 7, 11, 13, 17, 19, 23});
    pset("deepseek41.ple.head_offsets", {0, 3, 8, 15, 26, 39, 56, 75});
    pset("deepseek41.engram.token_map", {0, 2, 1, 2, 3, 1});
    pset("deepseek41.attention.compress_ratios", {0, 2, 1, 1});
    assert(tsg_dsv41::has_combined_engram_metadata(ple.get()));
    assert(!tsg_dsv41::has_combined_engram_metadata(file.get()));
    const auto slices = tsg_dsv41::combined_engram_slices(ple.get(), 98);
    assert(slices.size() == 2 && slices[0].id == 1 && slices[0].first_row == 0 && slices[0].rows == 26);
    assert(slices[1].id == 3 && slices[1].first_row == 26 && slices[1].rows == 72);
    auto pload = [&] { return tsg_dsv41::load_engram_gguf(ple.get(), shape); };
    auto pbad = [&](auto && call) {
        bool caught = false;
        try { call(); } catch (const std::runtime_error &) { caught = true; }
        assert(caught);
    };
    const auto pdata = pload();
    // pad_token_id 1 is compressed through the token map (engram.py), giving 2.
    assert(pdata.pad_id == 2 && pdata.compressed_vocab_size == 4);
    assert(pdata.layers[0].offsets == std::vector<uint64_t>({0, 3, 8, 15}));
    assert(pdata.layers[1].offsets == std::vector<uint64_t>({0, 13, 30, 49}));
    assert(pdata.layers[1].rows == 72 && pdata.head_dim == 2 && pdata.n_heads == 2);
    std::vector<int32_t> phistory;
    const auto prows = pdata.hash_tokens(input, 3, 0, phistory);
    assert(prows[8] == ((3 * 7) ^ (2 * 11)) % 3);
    assert(prows[10] == ((3 * 7) ^ (2 * 11) ^ (2 * 13)) % 7 + 8);
    assert(prows[20] == ((3 * 17) ^ (2 * 19)) % 13);
    assert(prows[22] == ((3 * 17) ^ (2 * 19) ^ (2 * 23)) % 19 + 30);
    pbad([&] { tsg_dsv41::combined_engram_slices(ple.get(), 97); });
    pset("deepseek41.ple.head_offsets", {0, 3, 8, 15, 26, 39, 57, 75});
    pbad([&] { tsg_dsv41::combined_engram_slices(ple.get(), 98); });
    pbad(pload);
    pset("deepseek41.ple.head_offsets", {0, 3, 8, 15, 26, 39, 56, 75});
    gguf_set_val_u32(ple.get(), "deepseek41.engram.compressed_vocab_size", 5); pbad(pload);
    gguf_set_val_u32(ple.get(), "deepseek41.engram.compressed_vocab_size", 4);
    gguf_set_val_u32(ple.get(), "deepseek41.engram.pad_token_id", 6); pbad(pload);
    gguf_set_val_u32(ple.get(), "deepseek41.engram.pad_token_id", 1);
    assert(pload().pad_id == 2);
    std::cout << "DeepSeek V4.1 embedded GGUF Engram validation passed\n";
}
