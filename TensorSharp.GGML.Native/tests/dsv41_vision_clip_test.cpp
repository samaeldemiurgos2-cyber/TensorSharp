// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// llama.cpp's DeepSeek V4.1 mmproj (clip names, F16 or BF16 matrices with an
// F32 patch convolution, SwiGLU split into gate and up) must load into the same
// encoder as TensorSharp's own companion holding the same BF16 weights: the
// image spans have to match bit for bit. Malformed clip files are refused.
#include "ggml.h"
#include "gguf.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <functional>
#include <map>
#include <random>
#include <string>
#include <vector>

extern "C" {
void * TSGgml_Dsv41VisionLoad(const char * path, const char * backend, int device, int threads);
void TSGgml_Dsv41VisionFree(void * handle);
int TSGgml_Dsv41VisionInfo(void * handle, int32_t * info, int count);
int TSGgml_Dsv41VisionEncode(void * handle, const float * patches, int height, int width, float * output, int capacity);
}

namespace {
constexpr int LAYERS = 2, DIM = 8, HEADS = 2, INTER = 12, PATCH = 2, RATIO = 2, TEXT = 16;
constexpr int TEXT_LAYERS = 3, EXPERTS = 4, MIN_PIXELS = 3136;
constexpr uint32_t IMAGE_TOKEN = 129264;

int failures = 0;
void check(bool ok, const std::string & what) {
    if (!ok) { std::fprintf(stderr, "FAIL %s\n", what.c_str()); ++failures; }
}

// One official tensor: ggml dimensions and F32 values (BF16-representable).
struct tensor_values { int64_t ne0, ne1; std::vector<float> data; };
using official = std::map<std::string, tensor_values>;

official make_weights() {
    std::mt19937 rng(41);
    official w;
    auto add = [&](const std::string & name, int64_t ne0, int64_t ne1, float centre, float spread) {
        std::uniform_real_distribution<float> d(centre - spread, centre + spread);
        tensor_values t{ne0, ne1, std::vector<float>(size_t(ne0 * ne1))};
        for (auto & v : t.data) v = ggml_bf16_to_fp32(ggml_fp32_to_bf16(d(rng)));
        w[name] = std::move(t);
    };
    add("vision.patch_embed.proj.weight", 3 * PATCH * PATCH, DIM, 0, 0.5f);
    add("vision.patch_embed.proj.bias", DIM, 1, 0, 0.1f);
    for (int layer = 0; layer < LAYERS; ++layer) {
        const std::string p = "vision.blocks." + std::to_string(layer) + ".";
        add(p + "norm1.weight", DIM, 1, 1, 0.2f);
        add(p + "norm2.weight", DIM, 1, 1, 0.2f);
        add(p + "attn.wqkv.weight", DIM, 3 * DIM, 0, 0.5f);
        add(p + "attn.wqkv.bias", 3 * DIM, 1, 0, 0.1f);
        add(p + "attn.wo.weight", DIM, DIM, 0, 0.5f);
        add(p + "attn.wo.bias", DIM, 1, 0, 0.1f);
        add(p + "mlp.w1.weight", DIM, 2 * INTER, 0, 0.5f);
        add(p + "mlp.w2.weight", INTER, DIM, 0, 0.5f);
    }
    add("vision.norm.weight", DIM, 1, 1, 0.2f);
    add("aligner.w1.weight", DIM * RATIO * RATIO, TEXT, 0, 0.3f);
    add("aligner.w1.bias", TEXT, 1, 0, 0.1f);
    add("aligner.w2.weight", TEXT, TEXT, 0, 0.3f);
    add("aligner.w2.bias", TEXT, 1, 0, 0.1f);
    add("image_start", TEXT, 1, 0, 1);
    add("image_end", TEXT, 1, 0, 1);
    add("image_newline", TEXT, 1, 0, 1);
    for (int layer = 0; layer < TEXT_LAYERS; ++layer)
        add("layers." + std::to_string(layer) + ".ffn.gate.bias_vl", EXPERTS, 1, 0, 1);
    return w;
}

// Writes tensors (name, type, ne[4], values) into a GGUF with `metadata`.
struct file_tensor { std::string name; ggml_type type; int64_t ne[4]; std::vector<float> data; };
void write_gguf(const std::string & path, const std::function<void(gguf_context *)> & metadata,
                const std::vector<file_tensor> & tensors) {
    size_t bytes = 0;
    for (const auto & t : tensors) bytes += t.data.size() * sizeof(float) + ggml_tensor_overhead() + 64;
    ggml_context * ctx = ggml_init({bytes + 4096, nullptr, false});
    gguf_context * file = gguf_init_empty();
    metadata(file);
    for (const auto & t : tensors) {
        ggml_tensor * tensor = ggml_new_tensor(ctx, t.type, 4, t.ne);
        ggml_set_name(tensor, t.name.c_str());
        for (size_t i = 0; i < t.data.size(); ++i) {
            if (t.type == GGML_TYPE_F32) static_cast<float *>(tensor->data)[i] = t.data[i];
            else if (t.type == GGML_TYPE_F16) static_cast<ggml_fp16_t *>(tensor->data)[i] = ggml_fp32_to_fp16(t.data[i]);
            else static_cast<ggml_bf16_t *>(tensor->data)[i] = ggml_fp32_to_bf16(t.data[i]);
        }
        gguf_add_tensor(file, tensor);
    }
    gguf_write_to_file(file, path.c_str(), false);
    gguf_free(file);
    ggml_free(ctx);
}

file_tensor plain(const std::string & name, ggml_type type, const tensor_values & v) {
    return {name, v.ne1 == 1 ? GGML_TYPE_F32 : type, {v.ne0, v.ne1, 1, 1}, v.data};
}

void write_companion(const std::string & path, const official & w) {
    std::vector<file_tensor> tensors;
    for (const auto & [name, v] : w) tensors.push_back(plain(name, GGML_TYPE_BF16, v));
    write_gguf(path, [](gguf_context * g) {
        gguf_set_val_str(g, "general.architecture", "deepseek41_vision");
        gguf_set_val_u64(g, "deepseek41.tokenizer_hash", 1234);
        gguf_set_val_u32(g, "deepseek41.hidden_size", TEXT);
        gguf_set_val_u32(g, "deepseek41.num_hidden_layers", TEXT_LAYERS);
        gguf_set_val_u32(g, "deepseek41.image_token_id", IMAGE_TOKEN);
        const std::string v = "deepseek41.vision.";
        gguf_set_val_u32(g, (v + "num_hidden_layers").c_str(), LAYERS);
        gguf_set_val_u32(g, (v + "hidden_size").c_str(), DIM);
        gguf_set_val_u32(g, (v + "num_attention_heads").c_str(), HEADS);
        gguf_set_val_u32(g, (v + "intermediate_size").c_str(), INTER);
        gguf_set_val_u32(g, (v + "patch_size").c_str(), PATCH);
        gguf_set_val_f32(g, (v + "rope_theta").c_str(), 10000.0f);
        gguf_set_val_u32(g, (v + "downsample_ratio").c_str(), RATIO);
        gguf_set_val_u32(g, (v + "max_image_tokens").c_str(), 1024);
        gguf_set_val_u32(g, (v + "min_pixels").c_str(), MIN_PIXELS);
        gguf_set_val_u32(g, (v + "max_wh_ratio").c_str(), 0);
    }, tensors);
}

// Mutations a malformed clip file applies before it is written.
struct clip_options {
    ggml_type matrix_type = GGML_TYPE_F16;
    std::string projector = "deepseek41v";
    float mean = 0.5f;
    bool drop_up = false, extra_tensor = false;
};

void write_clip(const std::string & path, const official & w, const clip_options & o) {
    std::vector<file_tensor> tensors;
    auto copy = [&](const std::string & to, const std::string & from) { tensors.push_back(plain(to, o.matrix_type, w.at(from))); };
    {
        // llama.cpp keeps the patch convolution F32 in a BF16 mmproj (unsloth's mmproj-BF16 does).
        const auto & patch = w.at("vision.patch_embed.proj.weight");
        const ggml_type type = o.matrix_type == GGML_TYPE_BF16 ? GGML_TYPE_F32 : o.matrix_type;
        tensors.push_back({"v.patch_embd.weight", type, {PATCH, PATCH, 3, DIM}, patch.data});
    }
    copy("v.patch_embd.bias", "vision.patch_embed.proj.bias");
    for (int layer = 0; layer < LAYERS; ++layer) {
        const std::string to = "v.blk." + std::to_string(layer) + ".", from = "vision.blocks." + std::to_string(layer) + ".";
        copy(to + "ln1.weight", from + "norm1.weight");
        copy(to + "ln2.weight", from + "norm2.weight");
        copy(to + "attn_qkv.weight", from + "attn.wqkv.weight");
        copy(to + "attn_qkv.bias", from + "attn.wqkv.bias");
        copy(to + "attn_out.weight", from + "attn.wo.weight");
        copy(to + "attn_out.bias", from + "attn.wo.bias");
        copy(to + "ffn_down.weight", from + "mlp.w2.weight");
        // The fused w1 is [gate; up] along its output rows.
        const auto & w1 = w.at(from + "mlp.w1.weight");
        const auto half = w1.data.begin() + DIM * INTER;
        tensors.push_back({to + "ffn_gate.weight", o.matrix_type, {DIM, INTER, 1, 1}, std::vector<float>(w1.data.begin(), half)});
        if (!o.drop_up)
            tensors.push_back({to + "ffn_up.weight", o.matrix_type, {DIM, INTER, 1, 1}, std::vector<float>(half, w1.data.end())});
    }
    copy("v.post_ln.weight", "vision.norm.weight");
    copy("mm.1.weight", "aligner.w1.weight");
    copy("mm.1.bias", "aligner.w1.bias");
    copy("mm.2.weight", "aligner.w2.weight");
    copy("mm.2.bias", "aligner.w2.bias");
    copy("v.token_embd.img_start", "image_start");
    copy("v.token_embd.img_end", "image_end");
    copy("v.image_newline", "image_newline");
    if (o.extra_tensor) copy("v.blk.0.attn_extra.weight", "vision.blocks.0.attn.wo.weight");
    write_gguf(path, [&](gguf_context * g) {
        gguf_set_val_str(g, "general.architecture", "clip");
        gguf_set_val_str(g, "general.basename", "DeepSeek-V4.1-Flash");
        gguf_set_val_str(g, "general.version", "1017");
        gguf_set_val_bool(g, "clip.has_vision_encoder", true);
        gguf_set_val_str(g, "clip.projector_type", o.projector.c_str());
        gguf_set_val_bool(g, "clip.use_silu", true);
        gguf_set_val_u32(g, "clip.vision.projection_dim", TEXT);
        gguf_set_val_u32(g, "clip.vision.image_size", PATCH * RATIO * 16);
        gguf_set_val_u32(g, "clip.vision.patch_size", PATCH);
        gguf_set_val_u32(g, "clip.vision.embedding_length", DIM);
        gguf_set_val_u32(g, "clip.vision.feed_forward_length", INTER);
        gguf_set_val_u32(g, "clip.vision.block_count", LAYERS);
        gguf_set_val_u32(g, "clip.vision.attention.head_count", HEADS);
        gguf_set_val_f32(g, "clip.vision.attention.layer_norm_epsilon", 1e-6f);
        gguf_set_val_u32(g, "clip.vision.projector.scale_factor", RATIO);
        gguf_set_val_u32(g, "clip.vision.image_min_pixels", MIN_PIXELS);
        const float mean[3] = {o.mean, 0.5f, 0.5f}, std[3] = {0.5f, 0.5f, 0.5f};
        gguf_set_arr_data(g, "clip.vision.image_mean", GGUF_TYPE_FLOAT32, mean, 3);
        gguf_set_arr_data(g, "clip.vision.image_std", GGUF_TYPE_FLOAT32, std, 3);
    }, tensors);
}

struct encoded { std::vector<int32_t> info; std::vector<float> span; };

bool encode(const std::string & path, encoded & out) {
    void * handle = TSGgml_Dsv41VisionLoad(path.c_str(), "CPU", 0, 2);
    if (!handle) return false;
    out.info.assign(8, 0);
    bool ok = TSGgml_Dsv41VisionInfo(handle, out.info.data(), 8) == 0;
    // A 3x3 patch grid merges to 2x2 (padded): START + 2 x (2 features + NEWLINE) + END.
    constexpr int HEIGHT = 3, WIDTH = 3, ROWS = 2 * (2 + 1) + 2;
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> pixel(-1, 1);
    std::vector<float> patches(size_t(HEIGHT * WIDTH * 3 * PATCH * PATCH));
    for (auto & v : patches) v = ggml_bf16_to_fp32(ggml_fp32_to_bf16(pixel(rng)));
    out.span.assign(size_t(ROWS * TEXT), 0.0f);
    ok = ok && TSGgml_Dsv41VisionEncode(handle, patches.data(), HEIGHT, WIDTH, out.span.data(), int(out.span.size())) == ROWS;
    TSGgml_Dsv41VisionFree(handle);
    return ok;
}
} // namespace

int main() {
    const auto dir = std::filesystem::temp_directory_path() / ("tsg-dsv41-vision-clip-" + std::to_string(std::random_device{}()));
    std::filesystem::create_directories(dir);
    const official w = make_weights();
    const std::string companion = (dir / "deepseek41.vision.gguf").string();
    write_companion(companion, w);
    encoded expected;
    check(encode(companion, expected), "TensorSharp companion loads and encodes");

    for (const ggml_type type : {GGML_TYPE_F16, GGML_TYPE_BF16}) {
        const std::string path = (dir / (std::string("mmproj-") + ggml_type_name(type) + ".gguf")).string();
        clip_options o;
        o.matrix_type = type;
        write_clip(path, w, o);
        encoded actual;
        const std::string what = std::string("clip mmproj ") + ggml_type_name(type);
        check(encode(path, actual), what + " loads and encodes");
        check(actual.info == expected.info, what + " reports the companion's geometry");
        check(actual.span.size() == expected.span.size() &&
              std::memcmp(actual.span.data(), expected.span.data(), actual.span.size() * sizeof(float)) == 0,
              what + " encodes the companion's image span bit for bit");
    }

    struct refusal { const char * what; std::function<void(clip_options &)> mutate; };
    const refusal refusals[] = {
        {"a V4 (deepseek4v) projector", [](clip_options & o) { o.projector = "deepseek4v"; }},
        {"a missing SwiGLU up projection", [](clip_options & o) { o.drop_up = true; }},
        {"a tensor the mapping does not know", [](clip_options & o) { o.extra_tensor = true; }},
        {"pixel normalization other than 0.5", [](clip_options & o) { o.mean = 0.48145466f; }},
    };
    for (const auto & r : refusals) {
        clip_options o;
        r.mutate(o);
        const std::string path = (dir / "mmproj-bad.gguf").string();
        write_clip(path, w, o);
        void * handle = TSGgml_Dsv41VisionLoad(path.c_str(), "CPU", 0, 2);
        check(handle == nullptr, std::string("refuses ") + r.what);
        if (handle) TSGgml_Dsv41VisionFree(handle);
    }
    std::filesystem::remove_all(dir);
    if (failures) return 1;
    std::puts("deepseek41v mmproj: F16 and BF16 load bit-identical to the companion; 4 malformed files refused");
    return 0;
}
