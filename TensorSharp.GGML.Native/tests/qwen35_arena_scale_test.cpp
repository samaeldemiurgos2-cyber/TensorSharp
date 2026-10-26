// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
// Exercise the actual persistent arena graph, comparing sidecar scales with
// the same scales folded exactly into F32 weights. No model download is needed.
#include "ggml_ops_transformer_common.h"

extern "C" {
int TSGgml_IsBackendAvailable(int);
const char* TSGgml_GetLastError();
void TSGgml_Qwen35ArenaResetBatchedDecodeCache();
void TSGgml_ClearHostBufferCache();
void TSGgml_Shutdown();
int TSGgml_Qwen35ArenaDecodeBatchedHidden(
    const TSGgmlQwen35LayerDesc*, int, int, const int32_t*, const int32_t*, const int32_t*,
    void**, void**, void**, void**, const int32_t*, const int32_t*,
    int,int,int,int,int,int,int,int,int,int,int,float,float,float,
    int,int,int,int,int,float,void*,int,const void*,int,int64_t,int64_t,int64_t,
    const void*,const void*,int,int64_t,int64_t,int64_t,int32_t*,int,const float*);
}

namespace {
constexpr int H = 128, HD = 64, NH = 2, KH = 1, GH = 128, GV = 2;
constexpr int FF = 192, VOCAB = 96, CONV = 4, CAP = 256, LAYERS = 4, STEPS = 5;
constexpr int CD = GH * (2 + GV), VD = GH * GV;

struct Weights {
    TSGgmlQwen35LayerDesc desc[LAYERS]{};
    std::array<float, TSQ35_SC_COUNT> scales[LAYERS];
    std::vector<std::vector<float>> storage;
    std::vector<float> head, norm;
    uint32_t rng = 12345;

    float* tensor(int n, float magnitude = .06f, float constant = 0) {
        storage.emplace_back(n);
        for (float& v : storage.back()) {
            rng = rng * 1664525u + 1013904223u;
            v = constant != 0 ? constant : (float(int(rng >> 16) - 32768) / 32768.f) * magnitude;
        }
        return storage.back().data();
    }

    // Powers of two make folded weights exact, including the negative control.
    static float scale(int slot) { return slot % 3 == 0 ? .5f : slot % 3 == 1 ? 2.f : .25f; }

    explicit Weights(int mode, int kv_heads = KH) : head(H * VOCAB), norm(H, 1.f) {
        for (int l = 0; l < LAYERS; ++l) {
            auto& d = desc[l];
            d.struct_bytes = sizeof(d);
            d.is_recurrent = l % 2;
            d.separate_qkv = l == 2;
            d.ff_dense = FF;
            scales[l].fill(1.f);
            d.proj_scales = mode == 1 ? scales[l].data() : nullptr;
            d.attn_norm_w = tensor(H, 0, 1.f);
            d.post_attn_norm_w = tensor(H, 0, 1.f);
            auto mat = [&](void*& p, int64_t& ne0, int64_t& ne1, int64_t& bytes,
                           int in, int out, int slot) {
                ne0 = in; ne1 = out; bytes = int64_t(in) * out * sizeof(float);
                p = tensor(in * out);
                scales[l][slot] = scale(slot);
                if (mode == 2)
                    for (int i = 0; i < in * out; ++i) static_cast<float*>(p)[i] *= scale(slot);
            };
#define MATRIX(field, in, out, slot) mat(d.field##_w, d.field##_ne0, d.field##_ne1, d.field##_bytes, in, out, slot)
            if (!d.is_recurrent) {
                MATRIX(qkv, H, 2 * NH * HD + (d.separate_qkv ? 0 : 2 * kv_heads * HD), TSQ35_SC_QKV);
                if (d.separate_qkv) {
                    MATRIX(k, H, kv_heads * HD, TSQ35_SC_K);
                    MATRIX(v, H, kv_heads * HD, TSQ35_SC_V);
                }
                MATRIX(o, NH * HD, H, TSQ35_SC_O);
                d.q_norm_w = tensor(HD, 0, 1.f);
                d.k_norm_w = tensor(HD, 0, 1.f);
            } else {
                const bool packed = l == 1;
                MATRIX(gdn_qkv, H, CD + (packed ? VD + 2 * GV : 0), TSQ35_SC_GDN_QKV);
                if (!packed) {
                    MATRIX(gdn_gate, H, VD, TSQ35_SC_GDN_GATE);
                    MATRIX(ssm_beta, H, GV, TSQ35_SC_BETA);
                    MATRIX(ssm_alpha, H, GV, TSQ35_SC_ALPHA);
                }
                MATRIX(ssm_out, VD, H, TSQ35_SC_SSM_OUT);
                d.conv1d_w = tensor(CONV * CD, .4f);
                d.ssm_dt_w = tensor(GV, 0, .1f);
                d.ssm_a_w = tensor(GV, 0, -.5f);
                d.ssm_norm_w = tensor(GH, 0, 1.f);
            }
            if (l < 2) {
                MATRIX(gu, H, FF * 2, TSQ35_SC_GU);
            } else {
                MATRIX(ffn_gate, H, FF, TSQ35_SC_FFN_GATE);
                MATRIX(ffn_up, H, FF, TSQ35_SC_FFN_UP);
            }
            MATRIX(down, FF, H, TSQ35_SC_DOWN);
#undef MATRIX
        }
        float* data = tensor(int(head.size()));
        std::copy(data, data + head.size(), head.begin());
    }
};

struct Result {
    std::vector<float> logits, states;
    std::vector<int> sampled;
};

bool run(int mode, int n, ggml_type kv_type, Result& result) {
    Weights weights(mode);
    const int kvbytes = int(ggml_row_size(kv_type, HD)) * CAP * KH;
    std::vector<std::vector<uint8_t>> k(2 * n, std::vector<uint8_t>(kvbytes)), v = k;
    std::vector<std::vector<float>> conv(2 * n, std::vector<float>((CONV - 1) * CD));
    std::vector<std::vector<float>> delta(2 * n, std::vector<float>(GH * GH * GV));
    std::vector<void*> kp(2 * n), vp(2 * n), cp(2 * n), dp(2 * n);
    for (int i = 0; i < 2 * n; ++i) {
        kp[i] = k[i].data(); vp[i] = v[i].data(); cp[i] = conv[i].data(); dp[i] = delta[i].data();
    }
    std::vector<int> tokens(n), positions(n), host_auth(n, 1), capacities(n, CAP), sampled(n);
    std::vector<float> input(H * n), logits(VOCAB * n);
    bool ok = true;
    for (int step = 0; step < STEPS; ++step) {
        for (int lane = 0; lane < n; ++lane) {
            positions[lane] = step;
            for (int d = 0; d < H; ++d)
                input[lane * H + d] = std::sin(float(d + lane * 37 + step * 13) * .17f);
        }
        int status = TSGgml_Qwen35ArenaDecodeBatchedHidden(weights.desc, LAYERS, n,
            tokens.data(), positions.data(), nullptr, kp.data(), vp.data(), cp.data(), dp.data(),
            host_auth.data(), capacities.data(), NH, KH, HD, HD, 2, kv_type,
            CONV, GH, GH, 1, GV, 1e-6f, 10000.f, 1.f, 0,0,0,0,0,1.f,
            logits.data(), VOCAB, weights.head.data(), GGML_TYPE_F32, H, VOCAB,
            int64_t(weights.head.size() * sizeof(float)), weights.norm.data(), nullptr,
            GGML_TYPE_F32, H, VOCAB, 0, sampled.data(), 1, input.data());
        if (status != 1) {
            std::fprintf(stderr, "FAIL mode%d n%d kv%d step%d status%d: %s\n",
                mode, n, int(kv_type), step, status, TSGgml_GetLastError());
            ok = false;
            break;
        }
        std::fill(host_auth.begin(), host_auth.end(), 0);
        result.logits.insert(result.logits.end(), logits.begin(), logits.end());
        result.sampled.insert(result.sampled.end(), sampled.begin(), sampled.end());
    }
    // Reset flushes recurrent states back to their host arrays before freeing
    // the graph. Compare those as well as logits: a stale state can hide until
    // the next request moves from the arena back to the solo kernel.
    TSGgml_Qwen35ArenaResetBatchedDecodeCache();
    for (const auto& a : conv) result.states.insert(result.states.end(), a.begin(), a.end());
    for (const auto& a : delta) result.states.insert(result.states.end(), a.begin(), a.end());
    TSGgml_ClearHostBufferCache();
    return ok;
}

// The two extra holders join at their own position zero, leave with live
// recurrent state, then return. The reference flushes after every step, so no
// graph or arena slot can silently serve as the source of its next state.
bool run_width_transitions(ggml_type kv_type, bool flush_each_step, Result& result, bool padded = false) {
    const int kv_heads = padded ? 2 : KH;
    Weights weights(0, kv_heads);
    constexpr int holders = 4;
    const int kvbytes = int(ggml_row_size(kv_type, HD)) * CAP * kv_heads;
    std::vector<std::vector<uint8_t>> k(2 * holders, std::vector<uint8_t>(kvbytes)), v = k;
    std::vector<std::vector<float>> conv(2 * holders, std::vector<float>((CONV - 1) * CD));
    std::vector<std::vector<float>> delta(2 * holders, std::vector<float>(GH * GH * GV));
    std::array<int, holders> next_position{}, host_auth{1, 1, 1, 1};
    bool ok = true;
    const std::array<int, 8> widths = padded
        ? std::array<int, 8>{3, 3, 4, 4, 3, 3, 4, 4}
        : std::array<int, 8>{2, 2, 4, 4, 2, 2, 4, 4};
    for (int n : widths) {
        std::vector<void*> kp(2 * n), vp(2 * n), cp(2 * n), dp(2 * n);
        for (int l = 0; l < 2; ++l) for (int lane = 0; lane < n; ++lane) {
            kp[l*n+lane] = k[l*holders+lane].data(); vp[l*n+lane] = v[l*holders+lane].data();
            cp[l*n+lane] = conv[l*holders+lane].data(); dp[l*n+lane] = delta[l*holders+lane].data();
        }
        std::vector<int> tokens(n), capacities(n, CAP), sampled(n);
        std::vector<float> input(H * n), logits(VOCAB * n);
        for (int lane = 0; lane < n; ++lane)
            for (int d = 0; d < H; ++d)
                input[lane*H+d] = std::sin(float(d + lane*37 + next_position[lane]*13) * .17f);
        int status = TSGgml_Qwen35ArenaDecodeBatchedHidden(weights.desc, LAYERS, n,
            tokens.data(), next_position.data(), nullptr, kp.data(), vp.data(), cp.data(), dp.data(),
            host_auth.data(), capacities.data(), NH, kv_heads, HD, HD, 2, kv_type,
            CONV, GH, GH, 1, GV, 1e-6f, 10000.f, 1.f, 0,0,0,0,0,1.f,
            logits.data(), VOCAB, weights.head.data(), GGML_TYPE_F32, H, VOCAB,
            int64_t(weights.head.size() * sizeof(float)), weights.norm.data(), nullptr,
            GGML_TYPE_F32, H, VOCAB, 0, sampled.data(), 1, input.data());
        if (status != 1) {
            std::fprintf(stderr, "FAIL width transition n%d kv%d pos%d status%d: %s\n",
                n, int(kv_type), next_position[0], status, TSGgml_GetLastError());
            ok = false;
            break;
        }
        result.logits.insert(result.logits.end(), logits.begin(), logits.end());
        result.sampled.insert(result.sampled.end(), sampled.begin(), sampled.end());
        for (int lane = 0; lane < n; ++lane) { ++next_position[lane]; host_auth[lane] = 0; }
        if (flush_each_step) TSGgml_Qwen35ArenaResetBatchedDecodeCache();
    }
    TSGgml_Qwen35ArenaResetBatchedDecodeCache();
    for (const auto& a : conv) result.states.insert(result.states.end(), a.begin(), a.end());
    for (const auto& a : delta) result.states.insert(result.states.end(), a.begin(), a.end());
    TSGgml_ClearHostBufferCache();
    return ok;
}

float max_diff(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return INFINITY;
    float worst = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) return INFINITY;
        worst = std::max(worst, std::fabs(a[i] - b[i]));
    }
    return worst;
}
}

int main(int argc, char** argv) {
    if ((argc != 2 && argc != 3) || (std::strcmp(argv[1], "cuda") && std::strcmp(argv[1], "metal"))) return 2;
    const char* mode = argc == 3 ? argv[2] : "";
    const bool transitions_only = !std::strcmp(mode, "transitions");
    const bool default_policy = !std::strcmp(mode, "default");
    const bool override_policy = !std::strcmp(mode, "pdl0") || !std::strcmp(mode, "pdl1");
    if (argc == 3 && !transitions_only && !default_policy && !override_policy) return 2;
    if (default_policy || override_policy) {
#ifdef _WIN32
        if (_putenv_s("GGML_CUDA_PDL", override_policy ? mode + 3 : "")) return 1;
#else
        if (override_policy ? setenv("GGML_CUDA_PDL", mode + 3, 1) : unsetenv("GGML_CUDA_PDL")) return 1;
#endif
    }
    if (!TSGgml_IsBackendAvailable(!std::strcmp(argv[1], "cuda") ? 3 : 1)) return 77;
    if (override_policy) {
        const char* actual = std::getenv("GGML_CUDA_PDL");
        if (!actual || std::strcmp(actual, mode + 3)) return 1;
        TSGgml_Shutdown();
        std::printf("PASS actual CUDA backend preserves explicit PDL=%s\n", actual);
        return 0;
    }
    int checks = 0;
    if (!transitions_only) for (int n : {2, 4}) for (ggml_type kv : {GGML_TYPE_F16, GGML_TYPE_Q8_0}) {
        // Preserve this three-build order across the whole 2/4-lane matrix.
        // Extra graphs between cases can change allocator reuse and mask a
        // CUDA capture ordering error exposed by an earlier case.
        Result unscaled, sidecar, folded;
        if (!run(0, n, kv, unscaled) || !run(1, n, kv, sidecar) || !run(2, n, kv, folded)) return 1;
        const float logits_error = max_diff(sidecar.logits, folded.logits);
        const float state_error = max_diff(sidecar.states, folded.states);
        const float omitted_scale_error = max_diff(unscaled.logits, folded.logits);
        std::printf("n%d kv%d: max logits error %.8g, state error %.8g, omitted scales error %.8g\n",
            n, int(kv), logits_error, state_error, omitted_scale_error);
        if (!(logits_error < .002f && state_error < .002f && omitted_scale_error > .05f)
            || sidecar.sampled != folded.sampled) {
            for (int step = 0; step < STEPS; ++step) {
                float error = 0;
                for (int i = step*n*VOCAB; i < (step+1)*n*VOCAB; ++i)
                    error = std::max(error, std::fabs(sidecar.logits[i]-folded.logits[i]));
                std::fprintf(stderr, "step %d sidecar/folded logits error %.8g\n",step,error);
            }
            return 1;
        }
        checks += STEPS;
    }
    // Separate lifecycle controls: recreating an identical scaled or unscaled
    // graph must also preserve logits, recurrent state and sampled IDs.
    if (!transitions_only) for (int n : {2, 4}) for (ggml_type kv : {GGML_TYPE_F16, GGML_TYPE_Q8_0}) {
        for (int variant : {0, 1}) {
            Result first, repeated;
            if (!run(variant, n, kv, first) || !run(variant, n, kv, repeated)) return 1;
            const float logits_error = max_diff(first.logits, repeated.logits);
            const float state_error = max_diff(first.states, repeated.states);
            std::printf("repeat n%d kv%d variant%d: logits/state error %.8g/%.8g\n",
                n, int(kv), variant, logits_error, state_error);
            if (!(logits_error < .002f && state_error < .002f) || first.sampled != repeated.sampled) return 1;
            checks += STEPS;
        }
    }
    for (bool padded : {false, true}) for (ggml_type kv : {GGML_TYPE_F16, GGML_TYPE_Q8_0}) {
        Result reference, transitions;
        if (!run_width_transitions(kv, true, reference, padded) || !run_width_transitions(kv, false, transitions, padded)) return 1;
        const float logits_error = max_diff(reference.logits, transitions.logits);
        const float state_error = max_diff(reference.states, transitions.states);
        std::printf("%s kv%d: logits/state error %.8g/%.8g\n",
            padded ? "3->4->3->4 (2 KV heads)" : "2->4->2->4", int(kv), logits_error, state_error);
        if (!(logits_error < .002f && state_error < .002f) || reference.sampled != transitions.sampled) return 1;
        checks += 8;
    }
    TSGgml_Shutdown();
    std::printf("PASS %d arena steps: all 13 projection scales, fused/split attention/GDN/FFN, "
        "2/4 lanes and width transitions, F16/Q8 KV, replay, sampled tokens and flushed state\n", checks);
    return 0;
}
