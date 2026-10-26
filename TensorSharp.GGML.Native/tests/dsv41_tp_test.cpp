// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#include "ggml_ops_deepseek41_tp.h"
#include "ggml_ops_dsv4_fused.h"
#include "ggml_ops_matmul_precision.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "gguf.h"
#if defined(TSG_GGML_USE_CUDA)
#include "ggml-cuda.h"
#include "ggml_ops_dsv4_fused.h"
#include "ggml_ops_matmul_precision.h"
#endif

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <stdexcept>
#include <thread>

#if defined(TSG_GGML_USE_CUDA)
// Standalone collective code reports details through its returned error text;
// this fixture does not link the process-wide managed ABI error storage.
namespace tsg { void set_last_error(const std::string &) {} }
#endif

namespace
{
void require(bool value, const char * message)
{
    if (!value) throw std::runtime_error(message);
}
struct fixture
{
    ggml_type type;
    int embedding, hidden, experts = 8, used = 6;
    float clamp = .1f;
    std::filesystem::path path;
    tsg_dsv41_tp::source gate, up, down;
    std::vector<char> gate_data, up_data, down_data;
    fixture(const std::string & directory, int layer)
    {
        const std::string prefix = "blk." + std::to_string(layer) + ".";
        std::vector<std::filesystem::path> files;
        for (const auto & entry : std::filesystem::directory_iterator(directory))
            if (entry.path().extension() == ".gguf") files.push_back(entry.path());
        std::sort(files.begin(), files.end());
        clamp = 10.0f;
        int loaded = 0;
        for (const auto & file : files)
        {
            ggml_context * metadata = nullptr;
            auto * gguf = gguf_init_from_file(file.string().c_str(), {true, &metadata});
            require(gguf != nullptr && metadata != nullptr, "Cannot read checkpoint metadata");
            const auto used_key = gguf_find_key(gguf, "deepseek41.expert_used_count");
            if (used_key >= 0) used = gguf_get_val_u32(gguf, used_key);
            const auto clamp_key = gguf_find_key(gguf, "deepseek41.swiglu_clamp_exp");
            if (clamp_key >= 0 && gguf_get_arr_n(gguf, clamp_key) > size_t(layer))
                clamp = ((const float *) gguf_get_arr_data(gguf, clamp_key))[layer];
            auto load = [&](const char * suffix, tsg_dsv41_tp::source & src, std::vector<char> & bytes) {
                const auto name = prefix + suffix;
                const auto index = gguf_find_tensor(gguf, name.c_str());
                if (index < 0) return;
                require(src.path.empty(), "Duplicate checkpoint expert tensor");
                auto * tensor = ggml_get_tensor(metadata, name.c_str());
                require(tensor != nullptr, "Missing checkpoint tensor metadata");
                src.path = file.string(); src.type = tensor->type;
                src.offset = gguf_get_data_offset(gguf) + gguf_get_tensor_offset(gguf, index);
                std::copy_n(tensor->ne, 4, src.ne.begin());
                bytes.resize(ggml_nbytes(tensor));
                std::ifstream input(file, std::ios::binary);
                input.seekg(src.offset);
                input.read(bytes.data(), bytes.size());
                require(input.gcount() == (std::streamsize) bytes.size(), "Truncated checkpoint expert tensor");
                ++loaded;
            };
            load("ffn_gate_exps.weight", gate, gate_data);
            load("ffn_up_exps.weight", up, up_data);
            load("ffn_down_exps.weight", down, down_data);
            ggml_free(metadata); gguf_free(gguf);
        }
        require(loaded == 3, "Checkpoint does not contain all three requested expert tensors");
        type = gate.type; embedding = gate.ne[0]; hidden = gate.ne[1]; experts = gate.ne[2];
    }
    fixture(ggml_type format, ggml_type down_format = GGML_TYPE_COUNT, int full_embedding = 0)
        : type(format), embedding(full_embedding > 0 ? full_embedding : format == GGML_TYPE_F32 ? 16 : 256),
          hidden(format == GGML_TYPE_F32 ? 18 : 2304)
    {
        std::mt19937 random(410912);
        path = std::filesystem::temp_directory_path() /
            ("dsv41-tp-" + std::to_string(std::random_device{}()) + ".weights");
        std::ofstream file(path, std::ios::binary);
        require((bool) file, "Cannot create TP fixture file");
        file << "unaligned-offset-fixture";
        auto write = [&](tsg_dsv41_tp::source & src, std::vector<char> & bytes, int inner, int rows, ggml_type storage) {
            src.path = path.string();
            src.offset = (size_t) file.tellp();
            src.type = storage;
            src.ne = {inner, rows, experts, 1};
            std::uniform_real_distribution<float> uniform(-.2f, .2f);
            std::vector<float> values((size_t) inner * rows * experts);
            for (float & x : values) x = uniform(random) / std::sqrt((float) inner);
            bytes.resize(ggml_row_size(storage, inner) * rows * experts);
            require(ggml_quantize_chunk(storage, values.data(), bytes.data(), 0, rows * experts, inner, nullptr) == bytes.size(),
                    "Cannot quantize TP test weights");
            file.write(bytes.data(), bytes.size());
        };
        write(gate, gate_data, embedding, hidden, format);
        write(up, up_data, embedding, hidden, format);
        write(down, down_data, hidden, embedding, down_format == GGML_TYPE_COUNT ? format : down_format);
    }
    // Independently materialize a column/row partition for diagnostic graphs,
    // without using executor upload_strip or writing another weight file.
    fixture(const fixture & full, tsg_dsv41_tp::strip part)
        : type(full.type), embedding(full.embedding), hidden((int) part.count), experts(full.experts), used(full.used), clamp(full.clamp)
    {
        auto copy = [&](const tsg_dsv41_tp::source & src, const std::vector<char> & source,
                        tsg_dsv41_tp::source & dst, std::vector<char> & bytes, bool inner) {
            dst = src;
            dst.ne[inner ? 0 : 1] = part.count;
            const size_t full_row = ggml_row_size(src.type, src.ne[0]);
            const size_t count = inner ? src.ne[1] * src.ne[2] : src.ne[2];
            const size_t width = inner ? ggml_row_size(src.type, part.count) : full_row * part.count;
            const size_t offset = inner ? ggml_row_size(src.type, part.first) : full_row * part.first;
            const size_t stride = inner ? full_row : full_row * src.ne[1];
            bytes.resize(count * width);
            for (size_t i = 0; i < count; ++i)
                std::memcpy(bytes.data() + i * width, source.data() + offset + i * stride, width);
        };
        copy(full.gate, full.gate_data, gate, gate_data, false);
        copy(full.up, full.up_data, up, up_data, false);
        copy(full.down, full.down_data, down, down_data, true);
    }
    ~fixture() { if (!path.empty()) { std::error_code error; std::filesystem::remove(path, error); } }
};

// Every evaluation, oracle and diagnostic selects the same experts with the
// same routing weights so their outputs are directly comparable.
void fill_routing(const fixture & data, int tokens, std::vector<int> & selected, std::vector<float> & routing)
{
    selected.resize((size_t) data.used * tokens);
    routing.resize(selected.size());
    for (int token = 0; token < tokens; ++token) for (int e = 0; e < data.used; ++e)
    {
        selected[token * data.used + e] = (token + e) % data.experts;
        routing[token * data.used + e] = float(e + 1) / (data.used * (data.used + 1) / 2);
    }
}

// A single-device reference backend. ggml CUDA evaluates F32 weights on its
// TF32 tensor-core path whatever precision the node requests (a 2^-11 input
// rounding, which is not an F32 oracle), so on CUDA the reference routes F32
// weights through TensorSharp's explicit-F32 matmul exactly like the
// executor's ranks do. Quantized, BF16 and F16 weights keep ggml's own paths.
struct reference_backend
{
    ggml_backend_t backend = nullptr, inner = nullptr;
    explicit reference_backend(ggml_backend_dev_t device)
    {
        backend = device ? ggml_backend_dev_init(device, nullptr) : ggml_backend_cpu_init();
        if (ggml_backend_is_cpu(backend)) ggml_backend_cpu_set_n_threads(backend, 1);
#if defined(TSG_GGML_USE_CUDA)
        if (auto * wrapped = tsg_dsv4_fused_backend_init(backend)) { inner = backend; backend = wrapped; }
#endif
    }
    ~reference_backend()
    {
        if (backend) ggml_backend_free(backend);
        if (inner) ggml_backend_free(inner);
    }
    ggml_tensor * mul_mat_id(ggml_context * ctx, ggml_tensor * w, ggml_tensor * x, ggml_tensor * ids) const
    {
#if defined(TSG_GGML_USE_CUDA)
        if (w->type == GGML_TYPE_F32 && inner) return tsg_matmul_id_f32(ctx, w, x, ids);
#endif
        return ggml_mul_mat_id(ctx, w, x, ids);
    }
};

struct evaluation
{
    reference_backend device;
    ggml_backend_t backend = nullptr;
    ggml_backend_t cuda_backend = nullptr;  // non-owning: device.inner
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * x = nullptr, * ids = nullptr, * weights = nullptr, * out = nullptr;
    ggml_tensor * gate_out = nullptr, * up_out = nullptr, * hidden_out = nullptr;
    ggml_tensor * raw_gate = nullptr, * raw_up = nullptr;
    tsg_dsv4_fused_desc strip_desc, swiglu_desc, reduce_desc;
    // With tp, the TP node runs on the CPU backend (host-driven ranks) unless
    // tp_layer_device names a CUDA device: the node then runs on that device's
    // fused backend and enqueues every rank there (device-resident path).
    evaluation(const fixture & data, int tokens, tsg_dsv41_tp::executor * tp, int layer,
               ggml_backend_dev_t reference_device = nullptr, bool preserve_taps = false, bool shared_once = false,
               int full_rows = 0, int first_row = 0, bool fused = false, ggml_backend_dev_t tp_layer_device = nullptr,
               bool split_launch = false)
        : device(tp ? tp_layer_device : reference_device), backend(device.backend)
    {
#if defined(TSG_GGML_USE_CUDA)
        // The owned strip kernel runs inside TensorSharp's fused CUDA backend,
        // which reference_backend already wraps around a CUDA reference device.
        if (full_rows) {
            cuda_backend = device.inner;
            require(cuda_backend != nullptr, "Cannot create CUDA strip test backend");
        }
#else
        require(!full_rows, "CUDA strip test unavailable in a CPU build");
#endif
        ctx = ggml_init({1024 * 1024, nullptr, true});
        x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, data.embedding, tokens);
        if (tp && layer == 1)
        {
            // Match argsort_top_k: selected IDs retain the full router's
            // original column stride and must be materialized by the TP hook.
            auto * storage = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, data.used + 3, tokens);
            ids = ggml_view_2d(ctx, storage, data.used, tokens, storage->nb[1], 0);
        }
        else ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, data.used, tokens);
        weights = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, data.used, tokens);
        ggml_tensor * gate = nullptr, * up = nullptr, * down = nullptr;
        ggml_tensor * zero_shared = nullptr;
        // The production caller adds the unsharded shared-expert output after
        // the routed TP reduction. Exercise that boundary with a known tensor;
        // the independent scalar check below rejects rank-count multiplication.
        ggml_tensor * shared = shared_once ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, data.embedding, tokens) : nullptr;
        if (tp && split_launch)
        {
            // The production split: launch, the shared expert, then gather.
            require(shared != nullptr && tp_layer_device != nullptr, "Split TP evaluation needs a shared tensor and a layer device");
            out = tp->build(ctx, layer, x, weights, ids, shared);
        }
        else if (tp) out = tp->build(ctx, layer, x, weights, ids);
        else
        {
            gate = ggml_new_tensor_3d(ctx, data.gate.type, data.embedding, data.hidden, data.experts);
            up = ggml_new_tensor_3d(ctx, data.up.type, data.embedding, data.hidden, data.experts);
            down = ggml_new_tensor_3d(ctx, data.down.type, data.hidden, data.embedding, data.experts);
            auto * input = ggml_reshape_3d(ctx, x, data.embedding, 1, tokens);
            auto projection = [&](ggml_tensor * weight) {
                if (!full_rows) return device.mul_mat_id(ctx, weight, input, ids);
#if defined(TSG_GGML_USE_CUDA)
                require(tsg_matmul_id_quant_strip_supported(cuda_backend, weight, tokens, full_rows, first_row),
                    "Test shape does not engage the quantized CUDA strip operation");
#endif
                strip_desc.kind = TSG_MATMUL_ID_QUANT_STRIP;
                strip_desc.i0 = full_rows; strip_desc.i1 = first_row;
                return tsg_matmul_id_quant_strip(ctx, weight, input, ids, &strip_desc);
            };
            if (full_rows) {
                strip_desc.kind=TSG_MATMUL_ID_QUANT_PAIR;strip_desc.i0=full_rows;strip_desc.i1=first_row;
                auto * pair=tsg_matmul_id_quant_pair(ctx,gate,up,input,ids,&strip_desc);
                raw_gate=ggml_view_3d(ctx,pair,pair->ne[0],pair->ne[1],pair->ne[2],pair->nb[1],pair->nb[2],0);
                raw_up=ggml_view_3d(ctx,pair,pair->ne[0],pair->ne[1],pair->ne[2],pair->nb[1],pair->nb[2],pair->nb[3]);
            } else { raw_gate = projection(gate); raw_up = projection(up); }
            auto * g = ggml_clamp(ctx, raw_gate, -INFINITY, data.clamp);
            auto * u = ggml_clamp(ctx, raw_up, -data.clamp, data.clamp);
            auto * h = ggml_swiglu_split(ctx, g, u);
            if (fused)
            {
                swiglu_desc.kind = TSG_DSV4_FUSED_SWIGLU_CLAMP;
                swiglu_desc.f0 = data.clamp;
                ggml_tensor * inputs[] = {raw_gate, raw_up};
                h = ggml_custom_4d(ctx, GGML_TYPE_F32, data.hidden, data.used, tokens, 1,
                    inputs, 2, tsg_dsv4_fused_cpu, 1, &swiglu_desc);
            }
            gate_out = g; up_out = u; hidden_out = h;
            if (preserve_taps) {
                ggml_set_output(raw_gate); ggml_set_output(raw_up);
                ggml_set_output(g); ggml_set_output(u); ggml_set_output(h);
            }
            auto * down_values = device.mul_mat_id(ctx, down, h, ids);
            if (fused)
            {
                zero_shared = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, data.embedding, tokens);
                reduce_desc.kind = TSG_DSV4_FUSED_EXPERT_REDUCE;
                ggml_tensor * inputs[] = {down_values, weights, zero_shared};
                out = ggml_custom_4d(ctx, GGML_TYPE_F32, data.embedding, tokens, 1, 1,
                    inputs, 3, tsg_dsv4_fused_cpu, 1, &reduce_desc);
            }
            else
            {
                auto * e = ggml_mul(ctx, down_values, weights);
                for (int expert = 0; expert < data.used; ++expert)
                {
                    auto * view = ggml_view_2d(ctx, e, data.embedding, tokens, e->nb[2], expert * e->nb[1]);
                    out = out ? ggml_add(ctx, out, view) : view;
                }
            }
        }
        if (shared) out = ggml_add(ctx, out, shared);
        graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, out);
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i)
        {
            auto * node = ggml_graph_node(graph, i);
            if (node->op == GGML_OP_MUL_MAT_ID && node->src[0]->type == GGML_TYPE_F32)
            {
                ggml_prec_set_acc(node, GGML_PREC_F32);
                ggml_prec_set_src(node, GGML_PREC_F32, 1);
            }
        }
        buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        require(buffer != nullptr, "Cannot allocate TP test graph");
        if (!tp)
        {
            ggml_backend_tensor_set(gate, data.gate_data.data(), 0, data.gate_data.size());
            ggml_backend_tensor_set(up, data.up_data.data(), 0, data.up_data.size());
            ggml_backend_tensor_set(down, data.down_data.data(), 0, data.down_data.size());
        }
        if (zero_shared) ggml_backend_tensor_memset(zero_shared, 0, 0, ggml_nbytes(zero_shared));
        std::vector<float> input(data.embedding * tokens), routing;
        std::vector<int> selected;
        for (size_t i = 0; i < input.size(); ++i) input[i] = std::sin((float) (i + 1) * .13f);
        fill_routing(data, tokens, selected, routing);
        ggml_backend_tensor_set(x, input.data(), 0, input.size() * sizeof(float));
        for (int token = 0; token < tokens; ++token)
            ggml_backend_tensor_set(ids, selected.data() + token * data.used,
                                    token * ids->nb[1], data.used * sizeof(int));
        ggml_backend_tensor_set(weights, routing.data(), 0, routing.size() * sizeof(float));
        if (shared)
        {
            std::vector<float> values(data.embedding * tokens);
            for (size_t i = 0; i < values.size(); ++i) values[i] = (i % 2 ? -.125f : .25f);
            ggml_backend_tensor_set(shared, values.data(), 0, values.size() * sizeof(float));
        }
    }
    ~evaluation()
    {
        if (buffer) ggml_backend_buffer_free(buffer);
        if (ctx) ggml_free(ctx);
    }
    std::vector<float> run()
    {
        require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "TP test forward failed");
        std::vector<float> result(ggml_nelements(out));
        ggml_backend_tensor_get(out, result.data(), 0, result.size() * sizeof(float));
        return result;
    }
};

// Small F32 oracle: no ggml graph, TP split/upload helper or sliced reference
// participates. Direct logical tensor indexing and double dot products make
// this independent of both the rank executor and the full-weight graph oracle.
std::vector<float> scalar_f32_reference(const fixture & data, int tokens, bool shared_once)
{
    require(data.gate.type == GGML_TYPE_F32 && data.down.type == GGML_TYPE_F32,
            "Scalar TP oracle requires F32 weights");
    auto weight = [](const std::vector<char> & bytes, size_t index) {
        float value;
        std::memcpy(&value, bytes.data() + index * sizeof(float), sizeof(float));
        return value;
    };
    std::vector<float> result(data.embedding * tokens);
    for (int token = 0; token < tokens; ++token)
    {
        for (int selected = 0; selected < data.used; ++selected)
        {
            const int expert = (token + selected) % data.experts;
            const float routing = float(selected + 1) / (data.used * (data.used + 1) / 2);
            std::vector<float> hidden(data.hidden);
            for (int row = 0; row < data.hidden; ++row)
            {
                double gate = 0, up = 0;
                for (int k = 0; k < data.embedding; ++k)
                {
                    const float x = std::sin(float(token * data.embedding + k + 1) * .13f);
                    const size_t index = ((size_t) expert * data.hidden + row) * data.embedding + k;
                    gate += double(weight(data.gate_data, index)) * x;
                    up += double(weight(data.up_data, index)) * x;
                }
                const float g = std::min(.1f, (float) gate), u = std::clamp((float) up, -.1f, .1f);
                hidden[row] = g / (1.0f + std::exp(-g)) * u;
            }
            for (int row = 0; row < data.embedding; ++row)
            {
                double dot = 0;
                for (int k = 0; k < data.hidden; ++k)
                    dot += double(weight(data.down_data, ((size_t) expert * data.embedding + row) * data.hidden + k)) * hidden[k];
                result[token * data.embedding + row] += (float) dot * routing;
            }
        }
    }
    if (shared_once)
        for (size_t i = 0; i < result.size(); ++i) result[i] += (i % 2 ? -.125f : .25f);
    return result;
}

void check_partitions()
{
    // Down output rows: complete tilings, and multiples of 32 (ggml-cuda's MMF
    // row block) whenever the width allows, preferring 128 (the owned quantized
    // strip kernel) and then 64. A 2-row fallback at 3/5 ranks on width 256 sent
    // F16 down projections through an F16-accumulating fallback.
    for (int64_t width : {int64_t(256), int64_t(512), int64_t(4096), int64_t(5120), int64_t(130)})
    for (int ranks : {2, 3, 4, 5, 6, 7, 8})
    for (int layer = 0; layer < ranks; ++layer)
    {
        const auto parts = tsg_dsv41_tp::split_outputs(width, ranks, layer);
        int64_t end = 0, alignment = 2;
        for (const int64_t candidate : {int64_t(128), int64_t(64), int64_t(32)})
            if (width % candidate == 0 && width / candidate >= ranks) { alignment = candidate; break; }
        for (const auto & part : parts)
        {
            require(part.first == end && part.count > 0 && part.count % alignment == 0,
                    "TP output strips are incomplete, overlapping, or not aligned to the widest possible block");
            end += part.count;
        }
        require((int) parts.size() == ranks && end == width, "TP output strips lost rows");
    }
    for (int ranks : {2, 3, 4, 5, 6, 7, 8})
    {
        std::vector<int64_t> aggregate(ranks);
        for (int layer = 0; layer < ranks; ++layer)
        {
            const auto strips = tsg_dsv41_tp::split(2304, 256, ranks, layer);
            int64_t end = 0;
            int wide = 0;
            for (int rank = 0; rank < ranks; ++rank)
            {
                const auto & part = strips[rank];
                require(part.first == end && part.first % 256 == 0 && part.count % 256 == 0 && part.count > 0,
                        "TP split is incomplete, overlapping, or unaligned");
                require(part.count == (9 / ranks) * 256 || part.count == (9 / ranks + 1) * 256,
                        "TP strip is not balanced by quantization blocks");
                wide += part.count > (9 / ranks) * 256;
                aggregate[rank] += part.count;
                end += part.count;
            }
            require(end == 2304 && wide == 9 % ranks, "TP split lost rows or assigned the wrong number of wider strips");
            if (ranks == 7 && layer == 6)
                require(strips[0].count == 512 && strips[6].count == 512 &&
                        std::all_of(strips.begin() + 1, strips.begin() + 6, [](auto part) { return part.count == 256; }),
                        "Seven-rank wider strips did not rotate across the last/first rank boundary");
        }
        require(std::all_of(aggregate.begin(), aggregate.end(), [](int64_t count) { return count == 2304; }),
                "A complete layer rotation leaves imbalanced expert strips");
    }
    for (auto type : {GGML_TYPE_BF16, GGML_TYPE_F16})
    for (int64_t width : {int64_t(2304), int64_t(2368)})
    for (int ranks : {2, 3, 4, 5, 6, 7, 8})
    {
        std::vector<int64_t> aggregate(ranks);
        for (int layer = 0; layer < ranks; ++layer)
        {
            const auto parts = tsg_dsv41_tp::split_weights(width, type, ranks, layer);
            int64_t end = 0;
            int wide = 0;
            for (int rank = 0; rank < ranks; ++rank)
            {
                const auto part = parts[rank];
                require(part.first == end && part.first % 64 == 0 && part.count > 0 && part.count % 64 == 0,
                        "Floating TP weights lost full coverage or matrix alignment");
                require(part.count == (width / 64 / ranks) * 64 || part.count == (width / 64 / ranks + 1) * 64,
                        "Floating TP strips are not balanced by 64-channel blocks");
                wide += part.count > (width / 64 / ranks) * 64;
                end += part.count;
                aggregate[rank] += part.count;
            }
            require(end == width && wide == width / 64 % ranks, "Floating TP split lost blocks");
            if (width == 2368 && ranks == 7 && layer == 6)
                require(parts[6].count == 384 && parts[0].count == 384 && parts[1].count == 320,
                        "Floating TP wider strips did not wrap across the last/first rank boundary");
        }
        require(std::all_of(aggregate.begin(), aggregate.end(), [=](int64_t count) { return count == width; }),
                "Floating TP complete layer rotation is unbalanced");
    }
    for (auto type : {GGML_TYPE_F32, GGML_TYPE_BF16, GGML_TYPE_F16, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K})
    for (int ranks : {2, 4, 7, 8})
    {
        // F32/quantized layouts must not change. Floating small/non-64 widths
        // must still form valid nonempty strips through the original fallback.
        const int64_t width = type == GGML_TYPE_BF16 ? 128 : type == GGML_TYPE_F16 ? 130 : 2304;
        if ((type == GGML_TYPE_BF16 || type == GGML_TYPE_F16) && width % 64 == 0 && width / 64 >= ranks) continue;
        const auto before = tsg_dsv41_tp::split(width, ggml_blck_size(type), ranks, 1);
        const auto after = tsg_dsv41_tp::split_weights(width, type, ranks, 1);
        for (int rank = 0; rank < ranks; ++rank)
            require(before[rank].first == after[rank].first && before[rank].count == after[rank].count,
                    "Typed split changed F32/quantized layout or small floating fallback");
    }
}

std::vector<float> values(ggml_tensor * tensor)
{
    std::vector<float> output(ggml_nelements(tensor));
    ggml_backend_tensor_get(tensor, output.data(), 0, output.size() * sizeof(float));
    return output;
}

struct difference
{
    double max_abs = 0, rel_l2 = 0;
};

difference measure(const std::vector<float> & reference, const std::vector<float> & actual)
{
    require(reference.size() == actual.size(), "Compared TP outputs have different sizes");
    double maximum = 0, error = 0, scale = 0;
    for (size_t i = 0; i < reference.size(); ++i)
    {
        require(std::isfinite(reference[i]) && std::isfinite(actual[i]), "Compared TP outputs contain nonfinite values");
        const double d = double(actual[i]) - reference[i];
        maximum = std::max(maximum, std::abs(d)); error += d * d; scale += double(reference[i]) * reference[i];
    }
    return {maximum, std::sqrt(error / std::max(1e-30, scale))};
}

void report_difference(const char * label, const std::vector<float> & reference, const std::vector<float> & actual)
{
    const auto d = measure(reference, actual);
    std::cout << "DIAGNOSTIC " << label << " max_abs=" << d.max_abs << " rel_l2=" << d.rel_l2 << std::endl;
}

// The executor's partition evaluated without the executor: the test slices
// the fixture bytes itself (no upload_strip) and runs ordinary ggml graphs on
// one device. Gate/up/SwiGLU run per hidden-row strip and their rows are
// concatenated; the down projection then runs per output-row strip over the
// complete hidden activations, and those rows are concatenated too. Nothing
// is summed across strips, exactly like the executor.
struct partitioned_output
{
    std::vector<float> output, gate, up, hidden;
};

std::vector<float> down_output_rows(const fixture & data, int tokens, const std::vector<tsg_dsv41_tp::strip> & strips,
                                    ggml_backend_dev_t device, const std::vector<float> & hidden);

partitioned_output partitioned_reference(const fixture & data, int tokens, int ranks, int layer,
                                         ggml_backend_dev_t device, bool taps)
{
    partitioned_output result;
    result.gate.resize((size_t) data.hidden * data.used * tokens);
    result.up.resize(result.gate.size());
    result.hidden.resize(result.gate.size());
    for (auto part : tsg_dsv41_tp::split_weights(data.hidden, data.down.type, ranks, layer))
    {
        fixture subset(data, part);
        evaluation sliced(subset, tokens, nullptr, layer, device, true);
        sliced.run();
        const auto g = values(sliced.gate_out), u = values(sliced.up_out), h = values(sliced.hidden_out);
        for (int row = 0; row < tokens * data.used; ++row)
        {
            std::copy_n(g.data() + row * part.count, part.count, result.gate.data() + row * data.hidden + part.first);
            std::copy_n(u.data() + row * part.count, part.count, result.up.data() + row * data.hidden + part.first);
            std::copy_n(h.data() + row * part.count, part.count, result.hidden.data() + row * data.hidden + part.first);
        }
    }
    result.output = down_output_rows(data, tokens, tsg_dsv41_tp::split_outputs(data.embedding, ranks, layer),
                                     device, result.hidden);
    if (!taps)
    {
        result.gate.clear();
        result.up.clear();
        result.hidden.clear();
    }
    return result;
}

// Only the down projection, over the given row strips of the down weights
// (one strip covering every row for the unsplit launch), fed with externally
// supplied hidden activations. This separates the down projection from the
// column-parallel gate/up evaluation that precedes it in the graphs.
std::vector<float> down_projection(const fixture & data, int tokens, const std::vector<tsg_dsv41_tp::strip> & strips,
                                   ggml_backend_dev_t device, const std::vector<float> & hidden)
{
    std::vector<float> sum((size_t) data.embedding * tokens);
    std::vector<int> selected;
    std::vector<float> routing;
    fill_routing(data, tokens, selected, routing);
    for (auto part : strips)
    {
        fixture subset(data, part);
        reference_backend reference(device);
        ggml_backend_t backend = reference.backend;
        ggml_context * ctx = ggml_init({1024 * 1024, nullptr, true});
        auto * h = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, part.count, data.used, tokens);
        auto * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, data.used, tokens);
        auto * weights = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, data.used, tokens);
        auto * down = ggml_new_tensor_3d(ctx, subset.down.type, part.count, data.embedding, data.experts);
        auto * e = ggml_mul(ctx, reference.mul_mat_id(ctx, down, h, ids), weights);
        ggml_tensor * out = nullptr;
        for (int expert = 0; expert < data.used; ++expert)
        {
            auto * view = ggml_view_2d(ctx, e, data.embedding, tokens, e->nb[2], expert * e->nb[1]);
            out = out ? ggml_add(ctx, out, view) : view;
        }
        auto * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, out);
        auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        require(buffer != nullptr, "Cannot allocate down-only strip graph");
        ggml_backend_tensor_set(down, subset.down_data.data(), 0, subset.down_data.size());
        std::vector<float> slice((size_t) part.count * data.used * tokens);
        for (int row = 0; row < tokens * data.used; ++row)
            std::copy_n(hidden.data() + (size_t) row * data.hidden + part.first, part.count, slice.data() + (size_t) row * part.count);
        ggml_backend_tensor_set(h, slice.data(), 0, slice.size() * sizeof(float));
        ggml_backend_tensor_set(ids, selected.data(), 0, selected.size() * sizeof(int));
        ggml_backend_tensor_set(weights, routing.data(), 0, routing.size() * sizeof(float));
        require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "Down-only strip forward failed");
        const auto output = values(out);
        for (size_t i = 0; i < sum.size(); ++i) sum[i] += output[i];
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
    }
    return sum;
}

// The down projection over output-row strips (the executor's split), each
// strip fed the complete hidden activations; strips are concatenated by row.
std::vector<float> down_output_rows(const fixture & data, int tokens, const std::vector<tsg_dsv41_tp::strip> & strips,
                                    ggml_backend_dev_t device, const std::vector<float> & hidden)
{
    std::vector<float> rows_out((size_t) data.embedding * tokens, 0.0f);
    std::vector<int> selected;
    std::vector<float> routing;
    fill_routing(data, tokens, selected, routing);
    const size_t row_bytes = ggml_row_size(data.down.type, data.hidden);
    for (auto part : strips)
    {
        std::vector<char> slice(row_bytes * part.count * data.experts);
        for (int expert = 0; expert < data.experts; ++expert)
            std::memcpy(slice.data() + row_bytes * part.count * expert,
                        data.down_data.data() + row_bytes * ((size_t) expert * data.embedding + part.first),
                        row_bytes * part.count);
        reference_backend reference(device);
        ggml_backend_t backend = reference.backend;
        ggml_context * ctx = ggml_init({1024 * 1024, nullptr, true});
        auto * h = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, data.hidden, data.used, tokens);
        auto * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, data.used, tokens);
        auto * weights = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, data.used, tokens);
        auto * down = ggml_new_tensor_3d(ctx, data.down.type, data.hidden, part.count, data.experts);
        auto * e = ggml_mul(ctx, reference.mul_mat_id(ctx, down, h, ids), weights);
        ggml_tensor * out = nullptr;
        for (int expert = 0; expert < data.used; ++expert)
        {
            auto * view = ggml_view_2d(ctx, e, part.count, tokens, e->nb[2], expert * e->nb[1]);
            out = out ? ggml_add(ctx, out, view) : view;
        }
        if (data.used == 1) out = ggml_cont(ctx, out);
        auto * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, out);
        auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        require(buffer != nullptr, "Cannot allocate down output-row graph");
        ggml_backend_tensor_set(down, slice.data(), 0, slice.size());
        ggml_backend_tensor_set(h, hidden.data(), 0, hidden.size() * sizeof(float));
        ggml_backend_tensor_set(ids, selected.data(), 0, selected.size() * sizeof(int));
        ggml_backend_tensor_set(weights, routing.data(), 0, routing.size() * sizeof(float));
        require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "Down output-row forward failed");
        const auto output = values(out);
        for (int token = 0; token < tokens; ++token)
            std::copy_n(output.data() + (size_t) token * part.count, part.count,
                        rows_out.data() + (size_t) token * data.embedding + part.first);
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
    }
    return rows_out;
}

std::vector<float> dequantize(const tsg_dsv41_tp::source & src, const std::vector<char> & bytes)
{
    const int64_t rows = src.ne[1] * src.ne[2];
    const size_t row_bytes = ggml_row_size(src.type, src.ne[0]);
    std::vector<float> out((size_t) src.ne[0] * rows);
    const auto * traits = ggml_get_type_traits(src.type);
    for (int64_t row = 0; row < rows; ++row)
    {
        if (src.type == GGML_TYPE_F32) std::memcpy(out.data() + row * src.ne[0], bytes.data() + row * row_bytes, row_bytes);
        else traits->to_float(bytes.data() + row * row_bytes, out.data() + row * src.ne[0], src.ne[0]);
    }
    return out;
}

// Exact down projection of externally supplied hidden activations: weights
// dequantized once, every dot product in double, no activation quantization.
std::vector<float> exact_down(const fixture & data, int tokens, const std::vector<float> & hidden)
{
    const auto down = dequantize(data.down, data.down_data);
    std::vector<int> selected;
    std::vector<float> routing;
    fill_routing(data, tokens, selected, routing);
    std::vector<double> result((size_t) data.embedding * tokens);
    for (int token = 0; token < tokens; ++token) for (int slot = 0; slot < data.used; ++slot)
    {
        const int expert = selected[token * data.used + slot];
        const float * h = hidden.data() + ((size_t) token * data.used + slot) * data.hidden;
        for (int row = 0; row < data.embedding; ++row)
        {
            const size_t base = ((size_t) expert * data.embedding + row) * data.hidden;
            double dot = 0;
            for (int k = 0; k < data.hidden; ++k) dot += double(down[base + k]) * h[k];
            result[(size_t) token * data.embedding + row] += dot * routing[token * data.used + slot];
        }
    }
    return std::vector<float>(result.begin(), result.end());
}

// Host model of ggml CUDA's batched (MMQ) down projection input: activations
// requantized to Q8 per 32 values as q = roundf(x*127/amax), scale 1/(127/amax)
// (quantize_mmq_q8_1, D4 layout), then dequantized and projected exactly.
// The int8 dot products themselves are exact, so this isolates the only
// lossy step of that path.
std::vector<float> requantized_down(const fixture & data, int tokens, const std::vector<float> & hidden, size_t * changed = nullptr,
                                    const std::vector<float> * other = nullptr)
{
    auto quantize = [](const std::vector<float> & values, std::vector<float> & codes) {
        std::vector<float> q(values.size());
        codes.assign(values.size(), 0.0f);
        for (size_t first = 0; first < values.size(); first += 32)
        {
            float amax = 0;
            for (size_t i = first; i < std::min(first + 32, values.size()); ++i) amax = std::max(amax, std::abs(values[i]));
            if (amax == 0) continue;
            const float d_inv = 127.0f / amax, d = 1.0f / d_inv;
            for (size_t i = first; i < std::min(first + 32, values.size()); ++i)
            {
                codes[i] = std::round(values[i] * d_inv);
                q[i] = codes[i] * d;
            }
        }
        return q;
    };
    std::vector<float> codes, other_codes;
    const auto q = quantize(hidden, codes);
    if (changed && other)
    {
        // The host reproduces the device's rounding of 127/amax and x*d_inv
        // only approximately, so a flip the device makes can be missed here;
        // the device-side probes above are the authoritative measurement.
        quantize(*other, other_codes);
        *changed = 0;
        for (size_t i = 0; i < codes.size(); ++i) *changed += codes[i] != other_codes[i];
    }
    return exact_down(data, tokens, q);
}

// Exact routed-MoE output for any weight format: every weight dequantized
// once, every dot product and the activation in double, and no activation
// quantization anywhere. Neither ggml graph nor the executor participates.
std::vector<float> exact_reference(const fixture & data, int tokens)
{
    const auto gate = dequantize(data.gate, data.gate_data), up = dequantize(data.up, data.up_data),
               down = dequantize(data.down, data.down_data);
    std::vector<int> selected;
    std::vector<float> routing;
    fill_routing(data, tokens, selected, routing);
    std::vector<double> result((size_t) data.embedding * tokens);
    std::vector<double> x(data.embedding), hidden(data.hidden);
    for (int token = 0; token < tokens; ++token)
    {
        for (int k = 0; k < data.embedding; ++k) x[k] = std::sin(float(token * data.embedding + k + 1) * .13f);
        for (int slot = 0; slot < data.used; ++slot)
        {
            const int expert = selected[token * data.used + slot];
            for (int row = 0; row < data.hidden; ++row)
            {
                const size_t base = ((size_t) expert * data.hidden + row) * data.embedding;
                double g = 0, u = 0;
                for (int k = 0; k < data.embedding; ++k) { g += double(gate[base + k]) * x[k]; u += double(up[base + k]) * x[k]; }
                g = std::min(0.1, g); u = std::clamp(u, -0.1, 0.1);
                hidden[row] = g / (1.0 + std::exp(-g)) * u;
            }
            for (int row = 0; row < data.embedding; ++row)
            {
                const size_t base = ((size_t) expert * data.embedding + row) * data.hidden;
                double dot = 0;
                for (int k = 0; k < data.hidden; ++k) dot += double(down[base + k]) * hidden[k];
                result[(size_t) token * data.embedding + row] += dot * routing[token * data.used + slot];
            }
        }
    }
    return std::vector<float>(result.begin(), result.end());
}

void quantized_scratch_lifecycle(ggml_backend_dev_t device)
{
    fixture data(GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, 5120);
    tsg_dsv41_tp::executor tp(std::vector<ggml_backend_dev_t>(2, device), data.used);
    tp.add_layer(0,data.gate,data.up,data.down,.1f);
    const char * fault_name = std::getenv("TS_TP_TEST_REAL_OOM") ?
        "TS_TP_TEST_EXHAUST_SCRATCH" : "TS_TP_TEST_FAIL_SCRATCH_GROWTH";
    auto fault=[&](bool enabled) {
#if defined(_WIN32)
        _putenv_s(fault_name,enabled?"1":"");
#else
        if(enabled)setenv(fault_name,"1",1);
        else unsetenv(fault_name);
#endif
    };
    try {
        for (int shape : {16, 16, 16, -129, 16, 129, 129, 129, 17, 65, 257, 16, 129}) {
            const bool fail=shape<0;const int tokens=std::abs(shape);
            fault(fail);tp.begin_forward();
            const auto actual=evaluation(data,tokens,&tp,0).run();
            fault(false);
            if(fail) {
                require(tp.error().find("scratch")!=std::string::npos,"Scratch allocation failure did not reach TP error latch");
                require(std::all_of(actual.begin(),actual.end(),[](float v){return std::isnan(v);}),"Scratch failure did not invalidate TP output");
            } else {
                require(tp.error().empty(),tp.error().c_str());
                const auto expected=evaluation(data,tokens,nullptr,0,device).run();
                double error=0,norm=0,maximum=0;
                for(size_t i=0;i<actual.size();++i) {
                    require(std::isfinite(actual[i]),"Recovered scratch produced nonfinite output");
                    const double delta=double(actual[i])-expected[i];error+=delta*delta;norm+=double(expected[i])*expected[i];maximum=std::max(maximum,std::abs(delta));
                }
                require(maximum<2e-6 && std::sqrt(error/std::max(1e-30,norm))<1e-5,"Recovered scratch differs from original unsplit oracle");
            }
            std::cout<<"SCRATCH_LIFECYCLE tokens="<<tokens<<" injected="<<fail<<" passed"<<std::endl;
        }
    } catch(...) {fault(false);throw;}
}

// The device-resident exchange issues rank 0 on the calling thread and every
// other rank on a worker. A rank graph that throws (a failed scratch
// reservation) must reach the TP error latch and NaN output on either kind of
// rank, without terminating the process or leaving a rank in the layer
// barrier, and the next forward must recover bit-identically to the
// host-driven node.
void resident_scratch_failure(const std::vector<ggml_backend_dev_t> & devices)
{
    fixture data(GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, 5120);
    tsg_dsv41_tp::executor tp(std::vector<ggml_backend_dev_t>{devices[0], devices[1]}, data.used);
    tp.add_layer(0, data.gate, data.up, data.down, .1f);
    require(tp.device_resident(), "Two CUDA ranks did not select the device-resident exchange");
    struct step { int tokens; const char * fault; bool split; };
    // The fault fires on every scratch reservation of the named device, which
    // only the owned quantized strip kernels make: "0" fails rank 0 (the
    // calling thread), "1" the worker rank, "all" both.
    const step steps[] = {
        {16, nullptr, false}, {129, "0", false}, {129, nullptr, false},
        {257, "1", true}, {257, nullptr, true}, {513, "all", true}, {16, nullptr, true},
        {513, nullptr, false}, {16, "1", false}, {1025, nullptr, true},
    };
#if defined(TSG_GGML_USE_CUDA)
    {
        // Without the strip kernel nothing reserves scratch and no fault fires.
        reference_backend probe(devices[0]);
        ggml_context * shapes = ggml_init({ggml_tensor_overhead(), nullptr, true});
        require(shapes != nullptr, "Cannot create the strip precondition context");
        auto * gate = ggml_new_tensor_3d(shapes, data.gate.type, data.embedding, data.hidden, data.experts);
        bool supported = probe.inner != nullptr;
        for (const step & s : steps)
            if (s.fault && supported)
                supported = tsg_matmul_id_quant_strip_supported(probe.inner, gate, s.tokens, data.hidden, 0);
        ggml_free(shapes);
        if (!supported)
        {
            std::cout << "The quantized strip kernel does not run these shapes on this GPU; "
                         "the injected fault cannot fire (not run)" << std::endl;
            std::exit(77);
        }
    }
#endif
    const char * fault_name = "TS_TP_TEST_FAIL_SCRATCH_DEVICE";
    auto fault = [&](const char * value) {
#if defined(_WIN32)
        _putenv_s(fault_name, value ? value : "");
#else
        if (value) setenv(fault_name, value, 1);
        else unsetenv(fault_name);
#endif
    };
    try {
        for (const step & s : steps) {
            fault(s.fault);
            tp.begin_forward();
            const auto actual = evaluation(data, s.tokens, &tp, 0, nullptr, false, s.split,
                                           0, 0, false, devices[0], s.split).run();
            fault(nullptr);
            if (s.fault) {
                // launch() reports the lowest failing rank, so "all" names rank 0's device.
                const std::string expected = std::string("Injected TensorSharp CUDA matmul scratch failure on device ") +
                    (std::strcmp(s.fault, "1") == 0 ? "1" : "0");
                require(tp.error().find(expected) != std::string::npos,
                        ("The targeted rank's failure did not reach the TP error latch: " + tp.error()).c_str());
                require(std::all_of(actual.begin(), actual.end(), [](float v) { return std::isnan(v); }),
                        "A rank's scratch failure did not invalidate the device-resident TP output");
            } else {
                require(tp.error().empty(), tp.error().c_str());
                const auto host = evaluation(data, s.tokens, &tp, 0, nullptr, false, s.split).run();
                require(tp.error().empty(), tp.error().c_str());
                require(actual.size() == host.size() &&
                        std::memcmp(actual.data(), host.data(), actual.size() * sizeof(float)) == 0,
                        "Device-resident TP after a rank failure differs from host-driven TP");
            }
            std::cout << "RESIDENT_SCRATCH_FAILURE tokens=" << s.tokens << " fault=" << (s.fault ? s.fault : "none")
                      << " split=" << s.split << " passed" << std::endl;
        }
    } catch (...) { fault(nullptr); throw; }
}

void quantized_strip_projections(ggml_backend_dev_t device)
{
    int checks = 0;
    for (auto format : {GGML_TYPE_Q2_K, GGML_TYPE_Q4_K})
    {
        fixture full(format, format == GGML_TYPE_Q2_K ? GGML_TYPE_Q3_K : GGML_TYPE_Q6_K, 5120);
        for (int used : {1, 6}) for (int tokens : {9, 16, 17, 33, 65, 129})
        {
            full.used = used;
            evaluation unsplit(full, tokens, nullptr, 0, device, true);
            const auto expected = unsplit.run();
            const auto gate = values(unsplit.raw_gate), up = values(unsplit.raw_up);
            // Rotate unequal seven-rank strips, and cover all retained rank
            // counts at the original failing 16-token checkpoint shape.
            for (int ranks : (tokens == 16 ? std::vector<int>{2, 4, 7, 8} : std::vector<int>{7}))
            {
                const int layer = ranks - 1;
                std::vector<float> sum(expected.size());
                for (auto part : tsg_dsv41_tp::split_weights(full.hidden, full.down.type, ranks, layer))
                {
                    fixture subset(full, part);
                    evaluation sliced(subset, tokens, nullptr, layer, device, true, false, full.hidden, int(part.first));
                    const auto output = sliced.run();
                    const auto actual_gate = values(sliced.raw_gate), actual_up = values(sliced.raw_up);
                    for (int row = 0; row < tokens * used; ++row)
                    {
                        require(std::memcmp(actual_gate.data() + row * part.count,
                            gate.data() + row * full.hidden + part.first, part.count * sizeof(float)) == 0,
                            "Quantized gate strip differs from untouched unsplit CUDA projection");
                        require(std::memcmp(actual_up.data() + row * part.count,
                            up.data() + row * full.hidden + part.first, part.count * sizeof(float)) == 0,
                            "Quantized up strip differs from untouched unsplit CUDA projection");
                    }
                    for (size_t i = 0; i < output.size(); ++i) sum[i] += output[i];
                }
                double error = 0, norm = 0, maximum = 0;
                for (size_t i = 0; i < sum.size(); ++i)
                {
                    require(std::isfinite(sum[i]), "Quantized strip result is nonfinite");
                    const double delta = double(sum[i]) - expected[i];
                    error += delta * delta; norm += double(expected[i]) * expected[i];
                    maximum = std::max(maximum, std::abs(delta));
                }
                const double relative = std::sqrt(error / std::max(1e-30, norm));
                std::cout << "QUANT_STRIP " << ggml_type_name(format) << " tokens=" << tokens << " used=" << used
                    << " ranks=" << ranks << " bitwise_gate_up=true max_abs=" << maximum << " rel_l2=" << relative << std::endl;
                require(maximum < 2e-6 && relative < 1e-5,
                    "Quantized strip MoE exceeds original unsplit reference tolerance");
                ++checks;
            }
        }
    }
    std::cout << "Passed " << checks << " quantized strip projection and MoE comparisons\n";
}

void diagnose(const fixture & data, int tokens, int ranks, int layer, ggml_backend_dev_t device,
              const std::vector<float> & tp_output)
{
    evaluation full(data, tokens, nullptr, layer, device, true);
    const auto full_output = full.run();
    const auto full_gate = values(full.gate_out), full_up = values(full.up_out), full_hidden = values(full.hidden_out);
    const auto parts = partitioned_reference(data, tokens, ranks, layer, device, true);
    report_difference("gate-column-strips", full_gate, parts.gate);
    report_difference("up-column-strips", full_up, parts.up);
    report_difference("swiglu-column-strips", full_hidden, parts.hidden);
    report_difference("same-device-manual-strips", full_output, parts.output);
    report_difference("manual-strips-versus-tp", parts.output, tp_output);
    // The row-parallel down split given identical inputs, and the unsplit
    // down launch given the partition's inputs: which side amplifies.
    const auto strips = tsg_dsv41_tp::split_weights(data.hidden, data.down.type, ranks, layer);
    const std::vector<tsg_dsv41_tp::strip> unsplit = {{0, data.hidden}};
    report_difference("device-down-strips-with-unsplit-hidden", full_output,
                      down_projection(data, tokens, strips, device, full_hidden));
    report_difference("device-unsplit-down-with-partitioned-hidden", full_output,
                      down_projection(data, tokens, unsplit, device, parts.hidden));
    // The same input perturbation through an exact projection is continuous;
    // through ggml CUDA's Q8 activation requantization it is not.
    report_difference("exact-down-partitioned-versus-unsplit-hidden",
                      exact_down(data, tokens, full_hidden), exact_down(data, tokens, parts.hidden));
    size_t changed = 0;
    const auto modelled_full = requantized_down(data, tokens, full_hidden);
    const auto modelled_parts = requantized_down(data, tokens, parts.hidden, &changed, &full_hidden);
    report_difference("requantized-model-down-partitioned-versus-unsplit-hidden", modelled_full, modelled_parts);
    std::cout << "DIAGNOSTIC host-modelled-Q8-code-flips=" << changed << " of=" << parts.hidden.size() << std::endl;
    // Which side is closer to the truth: both device evaluations carry the
    // same activation-quantization error, far above their mutual difference.
    const auto exact = exact_reference(data, tokens);
    report_difference("unsplit-versus-exact", exact, full_output);
    report_difference("manual-strips-versus-exact", exact, parts.output);
    report_difference("tp-versus-exact", exact, tp_output);
}

void set_fault(const char * stage)
{
#if defined(_WIN32)
    _putenv_s("TS_DSV41_TEST_FAIL_STAGE", stage ? stage : "");
#else
    if (stage) setenv("TS_DSV41_TEST_FAIL_STAGE", stage, 1);
    else unsetenv("TS_DSV41_TEST_FAIL_STAGE");
#endif
}

void failure_recovery(const std::vector<ggml_backend_dev_t> & devices, bool gate_fence = true)
{
    fixture data(GGML_TYPE_F32);
    const auto expected = evaluation(data, 3, nullptr, 0).run();
    int checks = 0;
    auto check = [&](bool condition, const char * message) { require(condition, message); ++checks; };
    auto compare = [&](const std::vector<float> & output) {
        double error = 0, norm = 0;
        for (size_t i = 0; i < output.size(); ++i)
        {
            require(std::isfinite(output[i]), "Recovered TP graph produced nonfinite output");
            error += std::pow(double(output[i]) - expected[i], 2);
            norm += double(expected[i]) * expected[i];
        }
        check(std::sqrt(error / norm) < 1e-5, "Recovered TP graph differs from independent full reference");
    };
    for (bool single : {false, true})
    for (const char * stage : {"tp-graph", "tp-rank", "tp-down-rank", "tp-sync", "tp-rank-and-sync"})
    {
        // The candidate additionally preserves a submission exception when
        // draining that same rank also reports a failure.
        if (!single && std::strcmp(stage, "tp-rank-and-sync") == 0) continue;
        tsg_dsv41_tp::executor tp(devices, data.used);
        tp.test_single_fanout(single);
        tp.test_gate_fence(gate_fence);
        if (!gate_fence) tp.test_device_gather(true);
        for (int layer : {0, 1}) tp.add_layer(layer, data.gate, data.up, data.down, .1f);
        tp.begin_forward();
        tp.test_set_position(0);
        set_fault(stage);
        const auto failed = evaluation(data, 3, &tp, 0).run();
        set_fault(nullptr);
        const char * expected_error = std::strcmp(stage, "tp-rank-and-sync") == 0 ? "tp-rank" : stage;
        check(tp.error().find(expected_error) != std::string::npos, "Rank fault did not reach the TP error latch");
        check(std::all_of(failed.begin(), failed.end(), [](float x) { return std::isnan(x); }),
              "Failed TP callback did not mark its output invalid");
        const auto first_error = tp.error();
        // A healthy later layer must preserve the earlier request error.
        compare(evaluation(data, 3, &tp, 1).run());
        check(tp.error() == first_error, "A later healthy layer concealed a prior rank fault");
        tp.begin_forward();
        check(tp.error().empty(), "A new request retained a stale TP error");
        // Reuse the failed shape, including the genuinely incomplete graph
        // produced by tp-graph. A prematurely published context would crash.
        compare(evaluation(data, 3, &tp, 0).run());
        check(tp.error().empty(), "Same-shape recovery retained an error");
        compare(evaluation(data, 3, &tp, 1).run());
        check(tp.error().empty(), "Recovered TP executor damaged another layer");
    }
    std::cout << "Passed " << checks << " TP failure lifecycle checks\n";
}

// An independent two-node sum transport. It deliberately shares no executor
// buffers: each participant contributes a copy and receives the complete sum.
// The departure barrier prevents the next gather from overwriting a result a
// slower node has not consumed yet. A bounded wait turns protocol drift into a
// failed comparison instead of hanging the native test suite indefinitely.
struct two_node_sum
{
    std::mutex mutex;
    std::condition_variable changed;
    std::vector<float> sum;
    int arrived = 0, departed = 0, generation = 0, fail_generation = -1;
    bool mismatch = false;

    static bool reduce(void * user, float * data, int count)
    {
        // Production uses the sentinel to terminate the existing transport.
        // This symmetric in-process fixture has no socket to close; its caller
        // controls request boundaries explicitly when checking native recovery.
        if (count < 0) return false;
        auto & state = *static_cast<two_node_sum *>(user);
        std::unique_lock<std::mutex> lock(state.mutex);
        const int round = state.generation;
        if (!state.arrived) { state.sum.assign(count, 0.0f); state.mismatch = false; }
        if ((int) state.sum.size() != count) state.mismatch = true;
        else for (int i = 0; i < count; ++i) state.sum[i] += data[i];
        ++state.arrived;
        state.changed.notify_all();
        if (!state.changed.wait_for(lock, std::chrono::seconds(15), [&] { return state.arrived == 2; }))
            return false;
        const bool success = !state.mismatch && round != state.fail_generation;
        if (success) std::copy(state.sum.begin(), state.sum.end(), data);
        if (++state.departed == 2)
        {
            state.arrived = state.departed = 0;
            ++state.generation;
            state.changed.notify_all();
        }
        else if (!state.changed.wait_for(lock, std::chrono::seconds(15), [&] { return state.generation != round; }))
            return false;
        return success;
    }
};

void distributed_rank_abort(ggml_backend_dev_t cpu)
{
    struct observer
    {
        int aborted = 0, normal = 0;
        static bool reduce(void * user, float * data, int count)
        {
            auto & value = *static_cast<observer *>(user);
            if (count == -1 && data == nullptr) ++value.aborted;
            else ++value.normal;
            return false;
        }
    } callback;
    fixture data(GGML_TYPE_F32);
    tsg_dsv41_tp::executor tp(std::vector<ggml_backend_dev_t>(2, cpu), data.used, 2,
                             4, 0, observer::reduce, &callback);
    for (int layer : {0, 1}) tp.add_layer(layer, data.gate, data.up, data.down, data.clamp);
    tp.begin_forward();
    // The rank-1 fault occurs before this node reaches its first collective.
    // Without a terminal notification its peer can wait forever in AllReduce.
    set_fault("tp-rank");
    const auto failed = evaluation(data, 3, &tp, 0).run();
    set_fault(nullptr);
    require(tp.error().find("tp-rank") != std::string::npos,
            "Distributed rank failure lost its original error");
    require(callback.aborted == 1 && callback.normal == 0,
            "Local distributed rank failure did not issue exactly one terminal transport abort before gather");
    require(std::all_of(failed.begin(), failed.end(), [](float x) { return std::isnan(x); }),
            "Local distributed rank failure returned apparently valid output");
    const auto original = tp.error();
    const auto later = evaluation(data, 3, &tp, 1).run();
    require(tp.error() == original && callback.aborted == 1 && callback.normal == 0,
            "A later layer replaced the original failure or re-entered an aborted transport");
    require(std::all_of(later.begin(), later.end(), [](float x) { return std::isnan(x); }),
            "A later layer concealed an aborted distributed request");
    std::cout << "Passed distributed local-rank terminal abort regression\n";
}

void distributed_numerical(ggml_backend_dev_t cpu)
{
    distributed_rank_abort(cpu);
    int checks = 0;
    for (auto formats : {std::make_pair(GGML_TYPE_F32, GGML_TYPE_F32),
                         std::make_pair(GGML_TYPE_BF16, GGML_TYPE_BF16),
                         std::make_pair(GGML_TYPE_Q4_K, GGML_TYPE_Q6_K)})
    for (int local_ranks : {1, 2})
    {
        fixture data(formats.first, formats.second);
        two_node_sum transport;
        std::unique_ptr<tsg_dsv41_tp::executor> nodes[2];
        for (int node = 0; node < 2; ++node)
        {
            nodes[node] = std::make_unique<tsg_dsv41_tp::executor>(
                std::vector<ggml_backend_dev_t>(local_ranks, cpu), data.used, 2,
                local_ranks * 2, node * local_ranks, two_node_sum::reduce, &transport);
            for (int layer : {0, 1}) nodes[node]->add_layer(layer, data.gate, data.up, data.down, data.clamp);
        }
        size_t total = 0;
        for (auto & node : nodes) for (int rank = 0; rank < local_ranks; ++rank) total += node->rank_weight_bytes(rank);
        require(total == 2 * (data.gate_data.size() + data.up_data.size() + data.down_data.size()),
                "Distributed TP duplicated or dropped global weight strips");
        auto pair = [&](int tokens, int layer, bool shared, bool reset_error = true) {
            std::vector<float> output[2];
            std::exception_ptr errors[2];
            std::thread threads[2];
            for (int node = 0; node < 2; ++node)
            {
                if (reset_error) nodes[node]->begin_forward();
                threads[node] = std::thread([&, node] {
                    try { output[node] = evaluation(data, tokens, nodes[node].get(), layer, nullptr, false, shared).run(); }
                    catch (...) { errors[node] = std::current_exception(); }
                });
            }
            for (auto & thread : threads) thread.join();
            for (const auto & error : errors) if (error) std::rethrow_exception(error);
            return std::make_pair(std::move(output[0]), std::move(output[1]));
        };
        for (bool shared : {false, true}) for (int tokens : {1, 3, 1}) for (int layer : {0, 1})
        {
            const auto expected = data.type == GGML_TYPE_F32 ? scalar_f32_reference(data, tokens, shared)
                : evaluation(data, tokens, nullptr, layer, nullptr, false, shared).run();
            const int before = transport.generation;
            auto output = pair(tokens, layer, shared);
            require(transport.generation == before + 2, "Distributed MoE must execute both activation and output collectives");
            for (auto * actual : {&output.first, &output.second})
            {
                const auto diff = measure(expected, *actual);
                require(diff.max_abs < (data.type == GGML_TYPE_F32 ? 1e-7 : 2e-6) && diff.rel_l2 < 1e-5,
                        "Distributed TP differs from the independent full-weight oracle");
                ++checks;
            }
            for (const auto & node : nodes) require(node->error().empty(), node->error().c_str());
        }
        if (data.type == GGML_TYPE_F32)
        {
            // Fail each collective separately. Both nodes must return poisoned
            // vectors and preserve the failure until the next request starts.
            for (int collective : {0, 1})
            {
                transport.fail_generation = transport.generation + collective;
                auto failed = pair(3, 0, false);
                for (auto * output : {&failed.first, &failed.second})
                    require(std::all_of(output->begin(), output->end(), [](float x) { return std::isnan(x); }),
                            "Failed distributed collective returned apparently valid output");
                for (const auto & node : nodes)
                    require(node->error().find(collective ? "output gather failed" : "activation gather failed") != std::string::npos,
                            "Distributed collective failure was not latched");
                transport.fail_generation = -1;
                const auto first_error = nodes[0]->error(), second_error = nodes[1]->error();
                pair(3, 1, false, false);
                require(nodes[0]->error() == first_error && nodes[1]->error() == second_error,
                        "A healthy later layer concealed a distributed failure");
                const auto recovered = pair(3, 0, false);
                const auto expected = scalar_f32_reference(data, 3, false);
                for (auto * output : {&recovered.first, &recovered.second})
                    require(measure(expected, *output).max_abs < 1e-7, "Distributed TP did not recover for the next request");
                for (const auto & node : nodes) require(node->error().empty(), "Recovered distributed request retained an error");
                ++checks;
            }
        }
        std::cout << "Distributed TP nodes=2 local_ranks=" << local_ranks
                  << " gate=" << ggml_type_name(data.gate.type) << " down=" << ggml_type_name(data.down.type)
                  << " numerical, shared-output and shape-reuse checks passed\n";
    }
    std::cout << "Passed " << checks << " distributed TP numerical/failure checks (CPU-backed simulated transport)\n";
}

void paired_fanout(const std::vector<ggml_backend_dev_t> & devices, int pairs,
                   bool compare_staging = false)
{
    fixture data(GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, 5120);
    tsg_dsv41_tp::executor tp(devices, data.used);
    tp.add_layer(0, data.gate, data.up, data.down, .1f);
    for (int tokens : (compare_staging ? std::vector<int>{1, 16, 256} : std::vector<int>{1, 16}))
    {
        evaluation model(data, tokens, &tp, 0);
        auto run = [&](bool single) {
            if (compare_staging) tp.test_pinned_staging(single);
            else tp.test_single_fanout(single);
            tp.begin_forward();
            const auto start = std::chrono::steady_clock::now();
            auto output = model.run();
            const double milliseconds = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count();
            require(tp.error().empty(), tp.error().c_str());
            return std::make_pair(std::move(output), milliseconds);
        };
        for (int warm = 0; warm < 10; ++warm) { run(false); run(true); }
        std::vector<double> separate, combined;
        for (int pair = 0; pair < pairs; ++pair)
        {
            // Alternate order to keep consistent order bias out of the pair.
            auto first = run(pair % 2 != 0), second = run(pair % 2 == 0);
            require(first.first.size() == second.first.size() &&
                std::memcmp(first.first.data(), second.first.data(), first.first.size() * sizeof(float)) == 0,
                "Paired TP execution modes are not bitwise identical");
            require(std::all_of(first.first.begin(), first.first.end(), [](float x) { return std::isfinite(x); }),
                    "Paired TP output contains nonfinite values");
            separate.push_back(pair % 2 ? second.second : first.second);
            combined.push_back(pair % 2 ? first.second : second.second);
        }
        auto array = [](const std::vector<double> & values) {
            std::cout << '[';
            for (size_t i = 0; i < values.size(); ++i) { if (i) std::cout << ','; std::cout << values[i]; }
            std::cout << ']';
        };
        std::cout << std::setprecision(17) << (compare_staging ? "STAGING_BENCH {\"ranks\":" : "FANOUT_BENCH {\"ranks\":") << devices.size()
                  << ",\"tokens\":" << tokens << ",\"embedding\":5120,\"hidden\":2304,\"experts\":8,\"top_k\":6"
                  << ",\"gate_up\":\"q2_k\",\"down\":\"q3_k\",\"pairs\":" << pairs
                  << ",\"bitwise_equal\":true,"
                  << (compare_staging ? "\"pageable_ms\":" : "\"two_fanout_ms\":");
        array(separate);
        std::cout << (compare_staging ? ",\"pinned_ms\":" : ",\"single_fanout_ms\":"); array(combined);
        std::cout << "}" << std::endl;
    }
}

void paired_upload(const std::vector<ggml_backend_dev_t> & devices, int pairs)
{
    fixture data(GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, 5120);
    // Include the first pinned allocation, then reuse it across a model-sized
    // layer stack. Short fresh-executor trials mostly measure cudaMallocHost.
    constexpr int layers = 40;
    auto run = [&](bool pipeline) {
        tsg_dsv41_tp::executor tp(devices, data.used);
        tp.test_pipelined_upload(pipeline);
        const auto start = std::chrono::steady_clock::now();
        for (int layer = 0; layer < layers; ++layer)
            tp.add_layer(layer, data.gate, data.up, data.down, .1f);
        const double milliseconds = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
        tp.begin_forward();
        std::vector<float> output;
        for (int layer = 0; layer < layers; ++layer)
        {
            const auto current = evaluation(data, 1, &tp, layer).run();
            output.insert(output.end(), current.begin(), current.end());
        }
        require(tp.error().empty(), tp.error().c_str());
        return std::make_pair(std::move(output), milliseconds);
    };
    run(false); run(true);
    std::vector<double> original, pipeline;
    for (int pair = 0; pair < pairs; ++pair)
    {
        auto first = run(pair % 2 != 0), second = run(pair % 2 == 0);
        require(first.first.size() == second.first.size() &&
            std::memcmp(first.first.data(), second.first.data(), first.first.size() * sizeof(float)) == 0,
            "Original and pipelined TP uploads produced different outputs");
        require(std::all_of(first.first.begin(), first.first.end(), [](float x) { return std::isfinite(x); }),
                "Upload comparison output contains nonfinite values");
        original.push_back(pair % 2 ? second.second : first.second);
        pipeline.push_back(pair % 2 ? first.second : second.second);
    }
    auto array = [](const std::vector<double> & values) {
        std::cout << '[';
        for (size_t i = 0; i < values.size(); ++i) { if (i) std::cout << ','; std::cout << values[i]; }
        std::cout << ']';
    };
    std::cout << std::setprecision(17) << "UPLOAD_BENCH {\"ranks\":" << devices.size()
              << ",\"layers\":" << layers << ",\"embedding\":5120,\"hidden\":2304,\"experts\":8,\"top_k\":6"
              << ",\"gate_up\":\"q2_k\",\"down\":\"q3_k\",\"pairs\":" << pairs
              << ",\"weight_bytes\":" << layers * (data.gate_data.size() + data.up_data.size() + data.down_data.size())
              << ",\"cache\":\"warm synthetic fixture\",\"bitwise_equal\":true,\"pageable_ms\":";
    array(original);
    std::cout << ",\"pipelined_ms\":"; array(pipeline);
    std::cout << "}" << std::endl;
}

void checkpoint_diagnostic(const std::vector<ggml_backend_dev_t> & devices,
                           const std::string & directory, int layer, int tokens, bool require_exact)
{
    fixture data(directory, layer);
    std::cout << "REAL_CHECKPOINT_DIAGNOSTIC layer=" << layer << " tokens=" << tokens
              << " embedding=" << data.embedding << " hidden=" << data.hidden
              << " experts=" << data.experts << " used=" << data.used << " clamp=" << data.clamp
              << " gate=" << ggml_type_name(data.gate.type) << " down=" << ggml_type_name(data.down.type) << std::endl;
    tsg_dsv41_tp::executor tp(devices, data.used);
    tp.add_layer(layer, data.gate, data.up, data.down, data.clamp);
    tp.begin_forward();
    const auto actual = evaluation(data, tokens, &tp, layer).run();
    require(tp.error().empty(), tp.error().c_str());
    require(std::all_of(actual.begin(), actual.end(), [](float value) { return std::isfinite(value); }),
            "Real checkpoint TP output contains nonfinite values");
    // The production path: the node on the layer device's fused backend.
    // Only the transport differs from the host-driven run above.
    std::vector<float> resident;
    if (tp.device_resident())
    {
        resident = evaluation(data, tokens, &tp, layer, nullptr, false, false, 0, 0, false, devices[0]).run();
        require(tp.error().empty(), tp.error().c_str());
        const bool same = resident.size() == actual.size() &&
            std::memcmp(resident.data(), actual.data(), actual.size() * sizeof(float)) == 0;
        std::cout << "REAL device-resident TP " << (same ? "bit-identical to" : "DIFFERS from")
                  << " host-driven TP" << std::endl;
        require(same, "Real checkpoint device-resident TP differs from host-driven TP");
    }
    evaluation fused(data, tokens, nullptr, layer, devices[0], true, false, 0, 0, true);
    const auto expected = fused.run();
    evaluation plain(data, tokens, nullptr, layer, devices[0], true);
    const auto unfused = plain.run();
    report_difference("real-tp-versus-production-fused", expected, actual);
    report_difference("real-plain-versus-production-fused", expected, unfused);
    report_difference("real-swiglu-plain-versus-production-fused", values(fused.hidden_out), values(plain.hidden_out));
    const auto partitioned = partitioned_reference(data, tokens, devices.size(), layer, devices[0], true);
    report_difference("real-partitioned-versus-production-fused", expected, partitioned.output);
    report_difference("real-tp-versus-partitioned", partitioned.output, actual);
    report_difference("real-gate-partitioned-versus-full", values(plain.gate_out), partitioned.gate);
    report_difference("real-up-partitioned-versus-full", values(plain.up_out), partitioned.up);
    report_difference("real-swiglu-partitioned-versus-full", values(plain.hidden_out), partitioned.hidden);
    if (require_exact)
    {
        require(actual.size() == expected.size() &&
                std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)) == 0,
                "Real checkpoint TP output differs from the production fused full-weight oracle");
        std::cout << "Passed real checkpoint bitwise production-oracle comparison\n";
    }
    else std::cout << "Real checkpoint diagnostic completed; differences above are measurements, not a passing quality gate.\n";
}

void paired_gather(const std::vector<ggml_backend_dev_t> & devices, const fixture & data, int layer, int pairs,
                   bool compare_fence = false)
{
    // Reuse identical weights, graphs, pinned storage and an already-created
    // communicator; this isolates transport from loading/communicator setup.
    tsg_dsv41_tp::executor tp(devices, data.used);
    tp.add_layer(layer, data.gate, data.up, data.down, data.clamp);
    for (int tokens : {1, 16, 128})
    {
        evaluation model(data, tokens, &tp, layer);
        auto run = [&](bool device) {
            tp.test_device_gather(compare_fence || device);
            tp.test_gate_fence(compare_fence && !device);
            tp.begin_forward();
            const auto start = std::chrono::steady_clock::now();
            auto output = model.run();
            const double ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count();
            require(tp.error().empty(), tp.error().c_str());
            return std::make_pair(std::move(output), ms);
        };
        for (int warm = 0; warm < 5; ++warm) { run(false); run(true); }
        std::vector<double> host, device;
        for (int pair = 0; pair < pairs; ++pair)
        {
            auto first = run(pair % 2 != 0), second = run(pair % 2 == 0);
            require(first.first.size() == second.first.size() &&
                    std::memcmp(first.first.data(), second.first.data(), first.first.size() * sizeof(float)) == 0,
                    "Host and device TP gathers are not bitwise identical");
            require(std::all_of(first.first.begin(), first.first.end(), [](float value) { return std::isfinite(value); }),
                    "Gather comparison produced nonfinite output");
            host.push_back(pair % 2 ? second.second : first.second);
            device.push_back(pair % 2 ? first.second : second.second);
        }
        auto array = [](const std::vector<double> & values) {
            std::cout << '[';
            for (size_t i = 0; i < values.size(); ++i) { if (i) std::cout << ','; std::cout << values[i]; }
            std::cout << ']';
        };
        std::cout << std::setprecision(17) << (compare_fence ? "GATE_FENCE_BENCH {\"ranks\":" : "GATHER_BENCH {\"ranks\":") << devices.size()
                  << ",\"tokens\":" << tokens << ",\"layer\":" << layer
                  << ",\"embedding\":" << data.embedding << ",\"hidden\":" << data.hidden
                  << ",\"experts\":" << data.experts << ",\"top_k\":" << data.used
                  << ",\"gate_up\":\"" << ggml_type_name(data.gate.type)
                  << "\",\"down\":\"" << ggml_type_name(data.down.type) << "\",\"pairs\":" << pairs
                  << ",\"bitwise_equal\":true," << (compare_fence ? "\"fenced_ms\":" : "\"host_ms\":");
        array(host);
        std::cout << (compare_fence ? ",\"enqueued_ms\":" : ",\"device_ms\":"); array(device);
        std::cout << "}" << std::endl;
    }
}
}

int main(int argc, char ** argv)
{
    try
    {
        bool cuda = false, checkpoint_shape = false, diagnostic = false, down_f32 = false, failure_only = false, strip_only = false;
        bool resident_failure_only = false, distributed_only = false;
        int cuda_ranks = 0, fanout_pairs = 0, staging_pairs = 0, upload_pairs = 0, gather_pairs = 0, gate_fence_pairs = 0;
        bool without_gate_fence = false;
        std::string model_directory;
        int model_layer = 0, model_tokens = 16;
        bool model_exact = false;
        for (int arg = 1; arg < argc; ++arg)
        {
            const std::string option = argv[arg];
            if (option == "--cuda" && arg + 1 < argc) { cuda = true; cuda_ranks = std::stoi(argv[++arg]); }
            else if (option == "--checkpoint-shape") checkpoint_shape = true;
            else if (option == "--diagnose") diagnostic = true;
            else if (option == "--diagnostic-down-f32") down_f32 = true;
            else if (option == "--failure-only") failure_only = true;
            else if (option == "--distributed-only") distributed_only = true;
            else if (option == "--quant-strip-only") strip_only = true;
            else if (option == "--resident-failure-only") resident_failure_only = true;
            else if (option == "--fanout-pairs" && arg + 1 < argc) fanout_pairs = std::stoi(argv[++arg]);
            else if (option == "--staging-pairs" && arg + 1 < argc) staging_pairs = std::stoi(argv[++arg]);
            else if (option == "--upload-pairs" && arg + 1 < argc) upload_pairs = std::stoi(argv[++arg]);
            else if (option == "--gather-pairs" && arg + 1 < argc) gather_pairs = std::stoi(argv[++arg]);
            else if (option == "--gate-fence-pairs" && arg + 1 < argc) gate_fence_pairs = std::stoi(argv[++arg]);
            else if (option == "--without-gate-fence") without_gate_fence = true;
            else if (option == "--model-dir" && arg + 1 < argc) model_directory = argv[++arg];
            else if (option == "--model-layer" && arg + 1 < argc) model_layer = std::stoi(argv[++arg]);
            else if (option == "--model-tokens" && arg + 1 < argc) model_tokens = std::stoi(argv[++arg]);
            else if (option == "--model-exact") model_exact = true;
            else throw std::runtime_error("Usage: GgmlOpsDsv41TpTest [--cuda 2..8] [--checkpoint-shape] [--diagnose] [--diagnostic-down-f32] [--failure-only [--without-gate-fence]|--resident-failure-only|--fanout-pairs N|--staging-pairs N|--upload-pairs N|--gather-pairs N|--gate-fence-pairs N] | --cuda 1 --quant-strip-only | --cuda N --model-dir DIR [--model-layer L --model-tokens T|--gather-pairs N|--gate-fence-pairs N]");
        }
        if (cuda && !(cuda_ranks >= 2 && cuda_ranks <= 8) && !(strip_only && cuda_ranks == 1)) return 1;
        require(!strip_only || (cuda && !failure_only && !fanout_pairs && !staging_pairs && !upload_pairs), "Quantized strip test requires CUDA and an exclusive test mode");
        require(!down_f32 || checkpoint_shape, "The F32-down control requires --checkpoint-shape");
        require(fanout_pairs >= 0 && fanout_pairs <= 10000 && !(fanout_pairs && failure_only), "Invalid fanout comparison count");
        require(staging_pairs >= 0 && staging_pairs <= 10000 &&
                !(staging_pairs && (!cuda || failure_only || fanout_pairs)), "Staging comparison requires CUDA and an exclusive test mode");
        require(upload_pairs >= 0 && upload_pairs <= 1000 &&
                !(upload_pairs && (!cuda || failure_only || fanout_pairs || staging_pairs)), "Upload comparison requires CUDA and an exclusive test mode");
        require(gather_pairs >= 0 && gather_pairs <= 10000 &&
                !(gather_pairs && (!cuda || failure_only || fanout_pairs || staging_pairs || upload_pairs || strip_only)),
                "Gather comparison requires CUDA and an exclusive test mode");
        require(gate_fence_pairs >= 0 && gate_fence_pairs <= 10000 &&
                !(gate_fence_pairs && (!cuda || failure_only || fanout_pairs || staging_pairs || upload_pairs || strip_only || gather_pairs)),
                "Gate fence comparison requires CUDA and an exclusive test mode");
        require(!without_gate_fence || (cuda && failure_only), "Gate fence override requires CUDA failure checks");
        std::vector<ggml_backend_dev_t> devices;
#if defined(TSG_GGML_USE_CUDA)
        if (cuda)
        {
            if (ggml_backend_cuda_get_device_count() < cuda_ranks) return 77;
            auto reg = ggml_backend_cuda_reg();
            for (int i = 0; i < cuda_ranks; ++i) devices.push_back(ggml_backend_reg_dev_get(reg, i));
        }
#else
        if (cuda) return 77;
#endif
        const auto cpu = ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0);
        if (distributed_only)
        {
            require(!cuda && !failure_only && !strip_only && !checkpoint_shape && !diagnostic &&
                    !fanout_pairs && !staging_pairs && !upload_pairs && !gather_pairs && !gate_fence_pairs &&
                    model_directory.empty(), "Distributed CPU numerical checks require an exclusive test mode");
            distributed_numerical(cpu);
            return 0;
        }
        if (resident_failure_only)
        {
            require(cuda && devices.size() >= 2 && !failure_only && !strip_only && !checkpoint_shape && !diagnostic &&
                    !fanout_pairs && !staging_pairs && !upload_pairs && !gather_pairs && !gate_fence_pairs &&
                    model_directory.empty(), "The device-resident failure test requires --cuda 2..8 and no other mode");
            resident_scratch_failure(devices);
            return 0;
        }
        if (gather_pairs || gate_fence_pairs)
        {
            require(model_layer >= 0, "Gather comparison requires a nonnegative layer");
            const int pairs = gather_pairs ? gather_pairs : gate_fence_pairs;
            if (!model_directory.empty()) paired_gather(devices, fixture(model_directory, model_layer), model_layer, pairs, gate_fence_pairs != 0);
            else paired_gather(devices, fixture(GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, 5120), 0, pairs, gate_fence_pairs != 0);
            return 0;
        }
        if (!model_directory.empty())
        {
            require(cuda && model_layer >= 0 && model_tokens > 0, "Real checkpoint diagnostics require CUDA and nonnegative layer/positive tokens");
            checkpoint_diagnostic(devices, model_directory, model_layer, model_tokens, model_exact);
            return 0;
        }
        if (strip_only) { quantized_scratch_lifecycle(devices[0]); quantized_strip_projections(devices[0]); return 0; }
        if (failure_only)
        {
            failure_recovery(cuda ? devices : std::vector<ggml_backend_dev_t>(2, cpu), !without_gate_fence);
            return 0;
        }
        if (fanout_pairs)
        {
            if (cuda) paired_fanout(devices, fanout_pairs);
            else for (int ranks : {2, 4, 8}) paired_fanout(std::vector<ggml_backend_dev_t>(ranks, cpu), fanout_pairs);
            return 0;
        }
        if (staging_pairs)
        {
            paired_fanout(devices, staging_pairs, true);
            return 0;
        }
        if (upload_pairs)
        {
            paired_upload(devices, upload_pairs);
            return 0;
        }
        // Quantized down rows have nine blocks, including seven unequal ranks.
        check_partitions();
        bool rejected = false;
        try { tsg_dsv41_tp::split(256, 256, 2, 0); } catch (const std::exception &) { rejected = true; }
        require(rejected, "TP accepted empty rank strips");
        int checks = 0;
        std::vector<std::string> failures;
        // The published Q2_K checkpoint stores routed gate/up in Q2_K and
        // down in Q3_K. Its nine down blocks must remain independently typed
        // and aligned when every rank receives a different hidden-width strip.
        const std::vector<std::pair<ggml_type, ggml_type>> formats = checkpoint_shape
            ? (down_f32
                ? std::vector<std::pair<ggml_type, ggml_type>>{{GGML_TYPE_Q2_K, GGML_TYPE_F32}}
                : std::vector<std::pair<ggml_type, ggml_type>>{{GGML_TYPE_Q2_K, GGML_TYPE_Q3_K}, {GGML_TYPE_Q4_K, GGML_TYPE_Q6_K}})
            : std::vector<std::pair<ggml_type, ggml_type>>{{GGML_TYPE_F32, GGML_TYPE_F32},
                {GGML_TYPE_BF16, GGML_TYPE_BF16}, {GGML_TYPE_F16, GGML_TYPE_F16}, {GGML_TYPE_Q2_K, GGML_TYPE_Q2_K},
                {GGML_TYPE_Q2_K, GGML_TYPE_Q3_K}, {GGML_TYPE_Q4_K, GGML_TYPE_Q4_K},
                {GGML_TYPE_Q6_K, GGML_TYPE_Q6_K}, {GGML_TYPE_Q4_K, GGML_TYPE_Q6_K}};
        for (auto [type, down_type] : formats)
        {
            fixture data(type, down_type, checkpoint_shape ? 5120 : 0);
            std::cout << "Fixture gate=" << ggml_type_name(type) << " up=" << ggml_type_name(type)
                      << " down=" << ggml_type_name(down_type) << " embedding=" << data.embedding
                      << " hidden=" << data.hidden << " experts=" << data.experts << " top_k=" << data.used << std::endl;
            std::map<int, std::vector<float>> expected;
            for (int tokens : {1, 5, 16})
                expected[tokens] = evaluation(data, tokens, nullptr, 0, cuda ? devices[0] : nullptr).run();
            for (int ranks : {2, 3, 4, 5, 6, 7, 8})
            {
                if (cuda && ranks != cuda_ranks) continue;
                auto rank_devices = cuda ? devices : std::vector<ggml_backend_dev_t>(ranks, cpu);
                tsg_dsv41_tp::executor tp(rank_devices, data.used);
                // Reuse scratch across layers and shapes, including seven-rank
                // wider strips wrapping from rank six back to rank zero.
                const std::vector<int> layers = ranks == 7 ? std::vector<int>{0, 1, 6} : std::vector<int>{0, 1};
                for (int layer : layers) tp.add_layer(layer, data.gate, data.up, data.down, .1f);
                size_t total = 0;
                for (int rank = 0; rank < ranks; ++rank) total += tp.rank_weight_bytes(rank);
                require(total == layers.size() * (data.gate_data.size() + data.up_data.size() + data.down_data.size()),
                        "TP duplicated or dropped weight bytes");
                if (type == GGML_TYPE_F32)
                {
                    for (bool shared_once : {false, true})
                    {
                        const auto independent = scalar_f32_reference(data, 3, shared_once);
                        const auto actual = evaluation(data, 3, &tp, layers.back(), nullptr, false, shared_once).run();
                        require(tp.error().empty(), tp.error().c_str());
                        for (size_t i = 0; i < actual.size(); ++i)
                            require(std::isfinite(actual[i]) && std::abs(actual[i] - independent[i]) < 1e-7,
                                    "TP differs from scalar F32 oracle or counts shared output more than once");
                        std::cout << "Scalar F32 oracle ranks=" << ranks << " shared_output=" << shared_once << " passed\n";
                        ++checks;
                    }
                }
                const auto within = [&](difference d) {
                    return d.max_abs < (type == GGML_TYPE_F32 ? 1e-7 : 2e-6) && d.rel_l2 < 1e-5;
                };
                for (int tokens : {1, 5, 16, 1}) for (int layer : layers)
                {
                    auto actual = evaluation(data, tokens, &tp, layer).run();
                    require(tp.error().empty(), tp.error().c_str());
                    for (float value : actual) require(std::isfinite(value), "TP produced nonfinite output");
                    if (cuda && tp.device_resident())
                    {
                        // The production path: the node on a layer device's fused
                        // backend, every rank enqueued behind its stream. Only the
                        // transport differs from the host-driven run, so the result
                        // must be bit-identical. Take the layer device from both
                        // ends of the rank list to cross device orderings.
                        for (auto * layer_device : {devices.front(), devices.back()})
                        {
                            const auto resident = evaluation(data, tokens, &tp, layer, nullptr, false, false,
                                                             0, 0, false, layer_device).run();
                            require(tp.error().empty(), tp.error().c_str());
                            if (resident.size() != actual.size() ||
                                std::memcmp(resident.data(), actual.data(), actual.size() * sizeof(float)) != 0)
                                failures.push_back(std::string(ggml_type_name(type)) + "/" + ggml_type_name(down_type) +
                                    " ranks=" + std::to_string(ranks) + " layer=" + std::to_string(layer) +
                                    " tokens=" + std::to_string(tokens) + ": device-resident TP differs from host-driven TP");
                            ++checks;
                        }
                        // The launch/shared-expert/gather split, against the
                        // host-driven node plus the same shared tensor.
                        const auto host_shared = evaluation(data, tokens, &tp, layer, nullptr, false, true).run();
                        const auto split = evaluation(data, tokens, &tp, layer, nullptr, false, true,
                                                      0, 0, false, devices.back(), true).run();
                        require(tp.error().empty(), tp.error().c_str());
                        if (split.size() != host_shared.size() ||
                            std::memcmp(split.data(), host_shared.data(), split.size() * sizeof(float)) != 0)
                            failures.push_back(std::string(ggml_type_name(type)) + "/" + ggml_type_name(down_type) +
                                " ranks=" + std::to_string(ranks) + " layer=" + std::to_string(layer) +
                                " tokens=" + std::to_string(tokens) + ": split launch/gather TP differs from host-driven TP");
                        ++checks;
                    }
                    const auto against_full = measure(expected[tokens], actual);
                    std::cout << ggml_type_name(type) << "/" << ggml_type_name(down_type) << " ranks=" << ranks << " layer=" << layer
                              << " tokens=" << tokens << " max_abs=" << against_full.max_abs << " rel_l2=" << against_full.rel_l2 << std::endl;
                    const auto device = cuda ? devices[0] : cpu;
                    if (diagnostic && !within(against_full)) diagnose(data, tokens, ranks, layer, device, actual);
                    // Additional record, not a pass criterion: the same weights
                    // evaluated with the executor's row partition as ordinary
                    // ggml graphs on the reference device. ggml's batched MMQ
                    // path (stream-k over the launch's tile count, then Q8
                    // requantization of the hidden activations) makes that
                    // partitioned evaluation move away from the unsplit launch
                    // on checkpoint shapes from nine tokens; the executor's owned
                    // quantized strip kernel keeps the unsplit launch's
                    // partitions, so it is expected to match the full-weight
                    // reference and may therefore sit outside this one.
                    // --diagnose prints the mechanism and each side's distance
                    // from an exact F64 reference.
                    const auto partitioned = partitioned_reference(data, tokens, ranks, layer, device, false).output;
                    const auto oracle_shift = measure(expected[tokens], partitioned);
                    const auto against_partitioned = measure(partitioned, actual);
                    std::cout << "  ORACLE full-weight=" << (within(against_full) ? "pass" : "FAIL")
                              << " partitioned-reference=" << (within(against_partitioned) ? "within" : "outside")
                              << " (TP versus partitioned: max_abs=" << against_partitioned.max_abs
                              << " rel_l2=" << against_partitioned.rel_l2
                              << "; partitioned versus full-weight: max_abs=" << oracle_shift.max_abs
                              << " rel_l2=" << oracle_shift.rel_l2 << ")" << std::endl;
                    // The strict full-weight comparison and its original
                    // tolerances remain the pass criterion. Record and carry on,
                    // so one run reports every failing fixture and rank count.
                    if (!within(against_full))
                        failures.push_back(std::string(ggml_type_name(type)) + "/" + ggml_type_name(down_type) +
                            " ranks=" + std::to_string(ranks) + " layer=" + std::to_string(layer) +
                            " tokens=" + std::to_string(tokens) + ": TP differs from the full-weight reference (max_abs=" +
                            std::to_string(against_full.max_abs) + " rel_l2=" + std::to_string(against_full.rel_l2) + ")");
                    ++checks;
                }
            }
        }
        for (const auto & failure : failures) std::cerr << "FAIL " << failure << '\n';
        require(failures.empty(), "TP comparisons failed (listed above)");
        require(checks > 0, "No TP numerical comparisons executed");
        std::cout << "Passed " << checks << " true MoE tensor-parallel comparisons\n";
        return 0;
    }
    catch (const std::exception & error)
    {
        set_fault(nullptr);
        std::cerr << error.what() << '\n';
        return 1;
    }
}
