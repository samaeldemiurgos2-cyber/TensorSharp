// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace tsg_dsv41_vision {

struct metadata {
    int layers = 0, dim = 0, heads = 0, intermediate = 0;
    int patch_size = 0, downsample_ratio = 0, max_image_tokens = 0;
    int min_pixels = 0, max_wh_ratio = 0;
    float rope_theta = 0;
    int text_dim = 0, text_layers = 0, image_token_id = 0, n_experts = 0;
    uint64_t tokenizer_hash = 0;
    bool bf16_activations = false;
    // A llama.cpp mmproj (clip, projector deepseek41v) has no router biases,
    // text layer count or tokenizer fingerprint: the biases come from the text
    // GGUF's blk.N.exp_probs_b_vl.bias, and its general.basename/version
    // identify the checkpoint instead (empty when the file does not say).
    bool router_bias_in_text = false;
    std::string source_basename, source_version;
};

class encoder {
public:
    static std::shared_ptr<encoder> load(const std::string & path, const std::string & backend_name,
                                         int device, int n_threads);
    ~encoder();
    encoder(const encoder &) = delete;
    encoder & operator=(const encoder &) = delete;
    const metadata & params() const;
    // Decoded F32 host vectors. The text graph uploads each bias to its layer
    // device; this encoder does not own or mutate language-model state.
    // Empty (any layer is out of range) when params().router_bias_in_text.
    const std::vector<float> & router_bias(int layer) const;
    // Input: row-major patches [n_h*n_w,3,patch_size,patch_size], normalized
    // and BF16-rounded for the original checkpoint. Output: complete span
    // START + (aligned features + NEWLINE) per row + END, each text_dim wide.
    int span_rows(int n_h, int n_w) const;
    int encode(const float * patches, int n_h, int n_w, float * output, int output_float_capacity);
private:
    struct implementation;
    std::unique_ptr<implementation> impl;
    encoder();
};

// The C handle owns a shared_ptr. AttachVision retains this returned owner so
// freeing the managed encoder handle cannot invalidate an attached text model.
std::shared_ptr<encoder> from_handle(void * handle);

} // namespace tsg_dsv41_vision
