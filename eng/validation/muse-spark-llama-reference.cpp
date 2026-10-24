// TensorSharp-owned reference driver. Link against an unchanged llama.cpp build.
// g++ -std=c++17 -O2 muse-spark-llama-reference.cpp -I/path/llama.cpp/include \
//     -I/path/llama.cpp/ggml/include -L/path/llama.cpp/build/bin \
//     -Wl,-rpath,/path/llama.cpp/build/bin -lllama -o muse-spark-llama-reference
// Usage: driver model.gguf 1,2,3,4 logits.f32 [chunk_tokens=1] [gpu_layers=0]
// Output is contiguous native little-endian F32 [token_count, vocab_size].
// Exact input token IDs bypass tokenizer/chat templates; this is numerical
// integration coverage, not semantic quality or large-model performance proof.
#include "llama.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    try {
        if (argc < 4 || argc > 6) {
            std::cerr << "Usage: " << argv[0] << " model.gguf token,csv logits.f32 [chunk=1] [gpu_layers=0]\n";
            return 2;
        }
        std::vector<llama_token> tokens;
        std::stringstream input(argv[2]);
        std::string part;
        while (std::getline(input, part, ',')) {
            size_t parsed = 0;
            int token = std::stoi(part, &parsed);
            if (parsed != part.size() || token < 0) throw std::runtime_error("Invalid token ID");
            tokens.push_back(token);
        }
        int chunk = argc >= 5 ? std::stoi(argv[4]) : 1;
        int gpu_layers = argc >= 6 ? std::stoi(argv[5]) : 0;
        if (tokens.empty() || chunk < 1 || tokens.size() > 1024 * 1024) throw std::runtime_error("Invalid token count/chunk");
        llama_backend_init();
        auto params = llama_model_default_params();
        params.n_gpu_layers = gpu_layers;
        params.check_tensors = true;
        auto started = std::chrono::steady_clock::now();
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(argv[1], params), llama_model_free);
        if (!model) throw std::runtime_error("Model load failed");
        int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
        for (auto token : tokens) if (token >= vocab) throw std::runtime_error("Input token exceeds vocabulary");
        auto context_params = llama_context_default_params();
        context_params.n_ctx = std::max<size_t>(512, tokens.size() + 1);
        context_params.n_batch = chunk;
        context_params.n_ubatch = chunk;
        context_params.n_threads = 1;
        context_params.n_threads_batch = 1;
        context_params.type_k = GGML_TYPE_F32;
        context_params.type_v = GGML_TYPE_F32;
        context_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        std::unique_ptr<llama_context, decltype(&llama_free)> context(
            llama_init_from_model(model.get(), context_params), llama_free);
        if (!context) throw std::runtime_error("Context initialization failed");
        auto loaded = std::chrono::steady_clock::now();
        std::ofstream output(argv[3], std::ios::binary | std::ios::trunc);
        if (!output) throw std::runtime_error("Cannot open output");
        struct batch_owner {
            llama_batch value;
            explicit batch_owner(int capacity) : value(llama_batch_init(capacity, 0, 1)) {}
            ~batch_owner() { llama_batch_free(value); }
        } batch(chunk);
        for (size_t offset = 0; offset < tokens.size(); offset += chunk) {
            int count = std::min<size_t>(chunk, tokens.size() - offset);
            batch.value.n_tokens = count;
            for (int i = 0; i < count; ++i) {
                batch.value.token[i] = tokens[offset + i];
                batch.value.pos[i] = offset + i;
                batch.value.n_seq_id[i] = 1;
                batch.value.seq_id[i][0] = 0;
                batch.value.logits[i] = true;
            }
            int status = llama_decode(context.get(), batch.value);
            if (status != 0) throw std::runtime_error("llama_decode failed: " + std::to_string(status));
            for (int i = 0; i < count; ++i) {
                const float * row = llama_get_logits_ith(context.get(), i);
                if (!row) throw std::runtime_error("Missing output logits");
                for (int j = 0; j < vocab; ++j) if (!std::isfinite(row[j])) throw std::runtime_error("Nonfinite logits");
                output.write(reinterpret_cast<const char *>(row), vocab * sizeof(float));
            }
        }
        output.close();
        if (!output) throw std::runtime_error("Output write failed");
        auto ended = std::chrono::steady_clock::now();
        std::cout << "{\"token_ids\":[";
        for (size_t i = 0; i < tokens.size(); ++i) {
            if (i) std::cout << ',';
            std::cout << tokens[i];
        }
        std::cout << "],\"tokens\":" << tokens.size() << ",\"vocab\":" << vocab
                  << ",\"chunk\":" << chunk << ",\"gpu_layers\":" << gpu_layers
                  << ",\"kv_type\":\"f32\",\"flash_attention\":false,\"load_ms\":"
                  << std::chrono::duration<double, std::milli>(loaded - started).count()
                  << ",\"forward_ms\":" << std::chrono::duration<double, std::milli>(ended - loaded).count() << "}\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
