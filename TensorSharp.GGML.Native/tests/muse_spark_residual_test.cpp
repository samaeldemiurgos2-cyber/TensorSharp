// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

struct View { void* data; int rows, channels, row_stride, channel_stride; int64_t bytes; };
struct View4 { void* data; int ne0, ne1, ne2, ne3; int64_t nb1, nb2, nb3, bytes; };
extern "C" {
int TSGgml_IsBackendAvailable(int);
int TSGgml_MuseSparkPostNormResidualF32(View, View, View, void*, void*, void*, int, float);
int TSGgml_NormF32(int, View4, View4, View4, View4, int, float);
int TSGgml_BinaryTensorF32(int, View4, View4, View4);
const char* TSGgml_GetLastError();
void TSGgml_Shutdown();
}

static View view(std::vector<float>& values, int rows, int width) {
    return {values.data(), rows, width, width, 1, int64_t(rows) * width * int64_t(sizeof(float))};
}

static View4 view4(std::vector<float>& values, int rows, int width) {
    const int64_t row_bytes = int64_t(width) * sizeof(float), bytes = rows * row_bytes;
    return {values.data(), width, rows, 1, 1, row_bytes, bytes, bytes, bytes};
}

// Compare the actual exported fused operation with the original separate
// native operations on identical inputs. A scalar tolerance cannot expose a
// changed stored-F32 boundary before a later quantized activation projection.
static bool check_stored_boundaries() {
    const int shapes[][2] = {{1,32},{3,32},{9,64},{128,64},{1,8192},{3,8192},{9,32},{1,32}};
    int cases = 0;
    for (int seed : {17, 173}) for (const auto& shape : shapes) for (bool pre_scale : {false, true}) {
        const int rows = shape[0], width = shape[1], count = rows * width;
        std::vector<float> stream(count), branch(count), scaled(count), normed(count), stream_part(count),
            branch_part(count), separate(count), fused(count), alpha(width), beta(width), scale(width), ones(width, 1.f);
        uint32_t state = seed;
        auto random = [&]() {
            state ^= state << 13; state ^= state >> 17; state ^= state << 5;
            return float(int32_t(state % 200001) - 100000) / 32767.f;
        };
        for (int c = 0; c < width; ++c) {
            alpha[c] = random(); beta[c] = random(); scale[c] = random();
        }
        for (int i = 0; i < count; ++i) { stream[i] = random(); branch[i] = random(); }
        auto checked = [&](int status) {
            if (status == 1) return true;
            std::fprintf(stderr, "stored-boundary operation failed: %s\n", TSGgml_GetLastError());
            return false;
        };
        if (pre_scale && !checked(TSGgml_BinaryTensorF32(3, view4(scaled, rows, width),
                view4(branch, rows, width), view4(scale, 1, width)))) return false;
        auto& norm_input = pre_scale ? scaled : branch;
        if (!checked(TSGgml_NormF32(2, view4(normed, rows, width), view4(norm_input, rows, width),
                view4(ones, 1, width), {}, 0, 1e-8f)) ||
            !checked(TSGgml_BinaryTensorF32(3, view4(stream_part, rows, width), view4(stream, rows, width), view4(alpha, 1, width))) ||
            !checked(TSGgml_BinaryTensorF32(3, view4(branch_part, rows, width), view4(normed, rows, width), view4(beta, 1, width))) ||
            !checked(TSGgml_BinaryTensorF32(1, view4(separate, rows, width), view4(stream_part, rows, width), view4(branch_part, rows, width))) ||
            !checked(TSGgml_MuseSparkPostNormResidualF32(view(fused, rows, width), view(stream, rows, width),
                view(branch, rows, width), alpha.data(), beta.data(), pre_scale ? scale.data() : nullptr, width, 1e-8f))) return false;
        for (int i = 0; i < count; ++i) {
            if (!std::isfinite(fused[i]) || !std::isfinite(separate[i]) || fused[i] != separate[i]) {
                std::fprintf(stderr, "stored-F32 boundary mismatch: rows=%d width=%d seed=%d pre_scale=%d index=%d fused=%.9g separate=%.9g delta=%.9g\n",
                    rows, width, seed, int(pre_scale), i, fused[i], separate[i], fused[i] - separate[i]);
                return false;
            }
        }
        ++cases;
    }
    std::printf("PASS: %d shared-input stored-F32 boundary cases against separate native operations\n", cases);
    return true;
}

int main(int argc, char** argv) {
    const bool cuda = argc == 2 && std::strcmp(argv[1], "cuda") == 0;
    if (argc > 2 || (argc == 2 && !cuda && std::strcmp(argv[1], "cpu") != 0)) return 2;
    if (!TSGgml_IsBackendAvailable(cuda ? 3 : 2)) return 77;
    constexpr int rows = 3, width = 32;
    std::vector<float> out(rows * (width + 1) + 8, 71.f), stream(rows * width), branch(stream.size());
    std::vector<float> alpha(width), beta(width), scale(width);
    for (size_t i = 0; i < stream.size(); ++i) { stream[i] = std::sin(float(i) * .2f); branch[i] = std::cos(float(i) * .3f); }
    for (int i = 0; i < width; ++i) { alpha[i] = (i % 5 - 2) * .4f; beta[i] = (i % 7 - 3) * .3f; scale[i] = (i % 9 - 4) * .6f; }
    const int64_t bytes = rows * width * sizeof(float);
    View result{out.data(), rows, width, width, 1, bytes};
    View s{stream.data(), rows, width, width, 1, bytes}, b{branch.data(), rows, width, width, 1, bytes};
    auto call = [&](View r, View sv, View bv, void* a, void* be, int channels, float eps) {
        return TSGgml_MuseSparkPostNormResidualF32(r, sv, bv, a, be, scale.data(), channels, eps);
    };
    auto refuse = [&](View r, View sv, View bv, void* a, void* be, int channels, float eps) {
        return call(r, sv, bv, a, be, channels, eps) == 0 &&
            std::all_of(out.begin(), out.end(), [](float x) { return x == 71.f; });
    };
    for (int target = 0; target < 3; ++target) {
        for (int kind = 0; kind < 3; ++kind) {
            View r = result, sv = s, bv = b;
            View& invalid = target == 0 ? r : target == 1 ? sv : bv;
            if (kind == 0) { invalid.row_stride++; invalid.bytes += rows * sizeof(float); }
            if (kind == 1) invalid.bytes += sizeof(float);
            if (kind == 2) invalid.channels--;
            if (!refuse(r, sv, bv, alpha.data(), beta.data(), width, 1e-5f)) {
                std::fprintf(stderr, "FAIL: invalid descriptor %d/%d mutated output\n", target, kind); return 1;
            }
        }
    }
    for (float eps : {0.f, -1.f, std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity()})
        if (!refuse(result, s, b, alpha.data(), beta.data(), width, eps)) return 1;
    if (!refuse(result, s, b, nullptr, beta.data(), width, 1e-5f) ||
        !refuse(result, s, b, alpha.data(), nullptr, width, 1e-5f) ||
        !refuse(result, s, b, alpha.data(), beta.data(), width - 1, 1e-5f)) return 1;
    if (call(result, s, b, alpha.data(), beta.data(), width, 1e-5f) != 1) return 1;
    for (int r = 0; r < rows; ++r) {
        double squares = 0;
        for (int c = 0; c < width; ++c) { float v = branch[r * width + c] * scale[c]; squares += double(v) * v; }
        const double inv = 1 / std::sqrt(squares / width + 1e-5f);
        for (int c = 0; c < width; ++c) {
            int i = r * width + c;
            float v = branch[i] * scale[c];
            float expected = stream[i] * alpha[c] + float(v * inv) * beta[c];
            if (!std::isfinite(out[i]) || std::abs(out[i] - expected) > 3e-5f) return 1;
        }
    }
    if (!std::all_of(out.begin() + rows * width, out.end(), [](float x) { return x == 71.f; })) return 1;
    // Aliasing the branch also retains the pre-call input during normalization.
    if (call(b, s, b, alpha.data(), beta.data(), width, 1e-5f) != 1) return 1;
    for (size_t i = 0; i < branch.size(); ++i) if (std::abs(branch[i] - out[i]) > 3e-5f) return 1;
    if (!check_stored_boundaries()) return 1;
    TSGgml_Shutdown();
    std::puts("PASS: Spark residual arithmetic, aliasing, exact spans and non-mutating rejection");
    return 0;
}
