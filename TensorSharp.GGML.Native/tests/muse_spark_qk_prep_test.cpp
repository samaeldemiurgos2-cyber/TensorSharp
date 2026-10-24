// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

struct View { void* data; int rows, columns, row_stride, column_stride; int64_t bytes; };
extern "C" {
int TSGgml_IsBackendAvailable(int);
int TSGgml_MuseSparkPrepareQKF32(View, View, View, const float*, const float*, const int32_t*,
    int, int, int, int, float, int, int, float, float);
const char* TSGgml_GetLastError();
void TSGgml_Shutdown();
}

int main(int argc, char** argv) {
    const bool cuda = argc == 2 && std::strcmp(argv[1], "cuda") == 0;
    if (argc > 2 || (argc == 2 && !cuda && std::strcmp(argv[1], "cpu") != 0)) return 2;
    if (!TSGgml_IsBackendAvailable(cuda ? 3 : 2)) return 77;
    constexpr int hd = 64, qh = 4, kh = 2, qw = qh * hd, kw = kh * hd, width = 2 * (qw + kw);
    int cases = 0;
    // Large-small-large repeats also change weights/positions: retained scratch
    // must not preserve previous-call inputs or alias late-uploaded leaves.
    for (int rows : {1, 2, 9, 128, 256, 1, 256})
    for (int padding : {0, 7})
    for (bool rope : {false, true}) {
        const int stride = width + padding;
        const int elements = (rows - 1) * stride + width;
        std::vector<float> source(elements + 8, 71.f), q(rows * qw + 8, 71.f), k(rows * kw + 8, 71.f);
        std::vector<float> qn(hd), kn(hd);
        std::vector<int32_t> pos(rows);
        for (int r = 0; r < rows; ++r) {
            pos[r] = r % 3 == 0 ? 65537 + cases : r % 3 == 1 ? 2048 : r * 7;
            for (int c = 0; c < width; ++c)
                source[r * stride + c] = std::sin(float(r * width + c) * .021f + cases) * .7f;
        }
        for (int c = 0; c < hd; ++c) { qn[c] = (c % 11 - 5) * .13f + cases * .001f; kn[c] = (c % 7 - 3) * .21f; }
        const auto original = source;
        View sv{source.data(), rows, width, stride, 1, int64_t(elements) * 4};
        View qv{q.data(), rows, qw, qw, 1, int64_t(rows) * qw * 4};
        View kv{k.data(), rows, kw, kw, 1, int64_t(rows) * kw * 4};
        constexpr float eps = 1e-5f, base = 500000.f, scale = .75f;
        const int result = TSGgml_MuseSparkPrepareQKF32(qv, kv, sv, qn.data(), kn.data(),
            rope ? pos.data() : nullptr, rope ? rows : 0, qh, kh, hd, eps, rope, hd, base, scale);
        if (result != 1) { std::fprintf(stderr, "FAIL: %s\n", TSGgml_GetLastError()); return 1; }
        for (int part = 0; part < 2; ++part) {
            int heads = part ? kh : qh, offset = part ? qw : 0;
            auto& out = part ? k : q; auto& norm = part ? kn : qn;
            for (int r = 0; r < rows; ++r)
            for (int h = 0; h < heads; ++h) {
                const float* x = source.data() + r * stride + offset + h * hd;
                double sum = 0; for (int c = 0; c < hd; ++c) sum += double(x[c]) * x[c];
                const float inv = float(1. / std::sqrt(sum / hd + eps));
                for (int c = 0; c < hd; c += 2) {
                    const float a = x[c] * inv * norm[c], b = x[c + 1] * inv * norm[c + 1];
                    // The upstream kernel generates frequencies in F32 by a
                    // recurrence. Reference is independent F64; large positions
                    // amplify that rounding, hence the explicit bounded tolerance.
                    const double theta = double(pos[r]) * scale * std::pow(double(base), -double(c) / hd);
                    const float ea = rope ? float(a * std::cos(theta) - b * std::sin(theta)) : a;
                    const float eb = rope ? float(a * std::sin(theta) + b * std::cos(theta)) : b;
                    const int i = (r * heads + h) * hd + c;
                    if (!std::isfinite(out[i]) || !std::isfinite(out[i + 1]) ||
                        std::abs(out[i] - ea) > (rope ? .008f : 4e-6f) ||
                        std::abs(out[i + 1] - eb) > (rope ? .008f : 4e-6f)) {
                        std::fprintf(stderr, "FAIL: scalar rows=%d rope=%d channel=%d expected=%g got=%g\n", rows, rope, i, ea, out[i]); return 1;
                    }
                }
            }
            if (!std::all_of(out.end() - 8, out.end(), [](float x) { return x == 71.f; })) return 1;
        }
        if (source != original) return 1;
        if (cases == 0) {
            auto refuses = [&](View oq, View ok, View in, const int32_t* p, int count, int rd, float e) {
                std::fill(q.begin(), q.end(), 71.f); std::fill(k.begin(), k.end(), 71.f);
                return TSGgml_MuseSparkPrepareQKF32(oq, ok, in, qn.data(), kn.data(), p, count,
                    qh, kh, hd, e, 1, rd, base, scale) == 0 &&
                    std::all_of(q.begin(), q.end(), [](float x) { return x == 71.f; }) &&
                    std::all_of(k.begin(), k.end(), [](float x) { return x == 71.f; }) && source == original;
            };
            View bad = sv; bad.bytes += 4;
            if (!refuses(qv, kv, bad, pos.data(), rows, hd, eps)) return 1;
            bad = qv; bad.bytes += 4;
            if (!refuses(bad, kv, sv, pos.data(), rows, hd, eps)) return 1;
            bad = kv; bad.data = q.data();
            if (!refuses(qv, bad, sv, pos.data(), rows, hd, eps)) return 1;
            if (!refuses(qv, kv, sv, nullptr, rows, hd, eps) ||
                !refuses(qv, kv, sv, pos.data(), 0, hd, eps) ||
                !refuses(qv, kv, sv, pos.data(), rows, hd - 1, eps) ||
                !refuses(qv, kv, sv, pos.data(), rows, hd, std::numeric_limits<float>::quiet_NaN())) return 1;
        }
        ++cases;
    }
    TSGgml_Shutdown();
    std::printf("PASS: %d packed Q/K scalar cases, read-only source/padding, output sentinels, changed leaves and invalid spans\n", cases);
    return 0;
}
