// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

struct View { void* data; int rows, cols, row_stride, col_stride; int64_t bytes; };
extern "C" {
int TSGgml_IsBackendAvailable(int);
int TSGgml_GetGpuDeviceCount(int);
int TSGgml_MultiDeviceInit(int, const int*, int);
int TSGgml_TensorParallelInitLoopback(int, int);
int TSGgml_SetActiveDevice(int);
int TSGgml_MuseSparkProjectionFloat32(View, View, View);
int TSGgml_RecreateBackend();
const char* TSGgml_GetLastError();
void TSGgml_InvalidateHostBuffer(void*);
void TSGgml_ClearHostBufferCache();
void TSGgml_ReleaseReuseComputeBuffers();
void TSGgml_MatmulTestAbortAfterWeightBind(int);
int TSGgml_HasResidentDeviceCopy(void*, int64_t);
void TSGgml_Shutdown();
}

namespace {
constexpr int inner = 512, guard = 8;
constexpr float sentinel = -731.25f;
double largest_error = 0;
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
struct Fixture {
    int rows, cols, input_stride;
    std::vector<float> x, w, y;
    View input, weight, output;
    Fixture(int rows, int cols, bool padded_input = false)
        : rows(rows), cols(cols), input_stride(inner + (padded_input ? 3 : 0)),
          x(guard * 2 + (rows - 1) * input_stride + inner, sentinel),
          w(guard * 2 + cols * inner, sentinel), y(guard * 2 + rows * cols, sentinel) {
        for (int r = 0; r < rows; ++r) for (int k = 0; k < inner; ++k)
            x[guard + r * input_stride + k] = .12f * std::sin(float(r * 13 + k) * .029f) + .000113f;
        for (int c = 0; c < cols; ++c) for (int k = 0; k < inner; ++k)
            w[guard + c * inner + k] = .13f * std::cos(float(c * 7 + k) * .031f) + .000217f;
        input = { x.data() + guard, rows, inner, input_stride, 1,
            int64_t((rows - 1) * input_stride + inner) * int64_t(sizeof(float)) };
        weight = { w.data() + guard, cols, inner, inner, 1, int64_t(cols) * inner * int64_t(sizeof(float)) };
        output = { y.data() + guard, rows, cols, cols, 1, int64_t(rows) * cols * int64_t(sizeof(float)) };
    }
    ~Fixture() { TSGgml_InvalidateHostBuffer(weight.data); }
    void run() {
        std::fill(y.begin(), y.end(), sentinel);
        const auto before_x = x, before_w = w;
        require(TSGgml_MuseSparkProjectionFloat32(output, input, weight) == 1,
            "Explicit-F32 CUDA projection failed");
        require(x == before_x && w == before_w, "Projection modified an input");
        require(std::all_of(y.begin(), y.begin() + guard, [](float v) { return v == sentinel; }) &&
            std::all_of(y.end() - guard, y.end(), [](float v) { return v == sentinel; }),
            "Projection overwrote output sentinels");
        for (int r = 0; r < rows; ++r) for (int c = 0; c < cols; ++c) {
            double expected = 0;
            for (int k = 0; k < inner; ++k)
                expected += double(x[guard + r * input_stride + k]) * w[guard + c * inner + k];
            const double error = std::abs(double(y[guard + r * cols + c]) - expected);
            largest_error = std::max(largest_error, error);
            require(std::isfinite(y[guard + r * cols + c]) && error <= 3e-5,
                "Explicit-F32 projection exceeded the independent double oracle");
        }
    }
};
void reject_layouts(Fixture& f) {
    std::fill(f.y.begin(), f.y.end(), sentinel);
    for (int which = 0; which < 3; ++which) {
        auto y = f.output, x = f.input, w = f.weight;
        View& v = which == 0 ? y : which == 1 ? x : w;
        ++v.bytes;
        require(TSGgml_MuseSparkProjectionFloat32(y, x, w) == 0, "Oversized descriptor accepted");
        require(std::all_of(f.y.begin(), f.y.end(), [](float n) { return n == sentinel; }),
            "Rejected descriptor changed output");
    }
    auto transposed = f.weight;
    transposed.row_stride = 1; transposed.col_stride = f.cols;
    require(TSGgml_MuseSparkProjectionFloat32(f.output, f.input, transposed) == 0,
        "Non-row-major weight accepted");
    auto overlap = f.output; overlap.data = f.input.data;
    const auto before = f.x;
    require(TSGgml_MuseSparkProjectionFloat32(overlap, f.input, f.weight) == 0 && f.x == before,
        "Overlapping output accepted or modified input");
}
void run_shapes() {
    for (int cols : { 48, 64, 256, 257 }) for (int rows : { 1, 3, 8, 9, 32 }) {
        Fixture f(rows, cols, rows == 9);
        f.run(); f.run(); // Private executor and resident weight reuse.
        TSGgml_InvalidateHostBuffer(f.weight.data);
        for (int k = 0; k < inner; ++k) f.w[guard + k] *= .75f;
        f.run(); // Explicit invalidation reloads changed bytes.
    }
    // Exercise the configured prefill batch size, including the two shard
    // widths that selected different upstream precision paths.
    for (int cols : { 48, 64 }) {
        Fixture f(256, cols); f.run();
    }
}
void clear_buffers() {
    TSGgml_ClearHostBufferCache();
    TSGgml_ReleaseReuseComputeBuffers();
}
}

int main(int argc, char** argv) {
    const bool cpu = argc == 2 && std::strcmp(argv[1], "cpu") == 0;
    const bool multi = argc == 2 && std::strcmp(argv[1], "cuda-multi") == 0;
    if (argc != 2 || (!cpu && !multi && std::strcmp(argv[1], "cuda") != 0)) return 2;
    if (!TSGgml_IsBackendAvailable(cpu ? 2 : 3)) return 77;
    if (multi && TSGgml_GetGpuDeviceCount(3) < 2) { TSGgml_Shutdown(); return 77; }
    try {
        if (cpu) {
            Fixture f(9, 48);
            require(TSGgml_MuseSparkProjectionFloat32(f.output, f.input, f.weight) == 0,
                "CUDA precision entry silently executed another backend");
            require(std::all_of(f.y.begin(), f.y.end(), [](float n) { return n == sentinel; }),
                "Unsupported-backend refusal changed output");
        } else {
            {
                Fixture f(9, 48);
                reject_layouts(f);
                for (int mode : { 1, 2 }) {
                    TSGgml_InvalidateHostBuffer(f.weight.data);
                    TSGgml_MatmulTestAbortAfterWeightBind(mode);
                    require(TSGgml_MuseSparkProjectionFloat32(f.output, f.input, f.weight) == 0,
                        "Injected cold failure was not returned");
                    require(!TSGgml_HasResidentDeviceCopy(f.weight.data, f.weight.bytes),
                        "Failed precise projection retained a cold resident entry");
                    f.run();
                }
            }
            run_shapes();
            clear_buffers();
            if (multi) {
                int devices[] = { 1, 0 };
                require(TSGgml_MultiDeviceInit(3, devices, 2) == 1, "Reordered multi-device init failed");
                for (int rank = 0; rank < 2; ++rank) {
                    require(TSGgml_SetActiveDevice(rank) == 1, "Rank selection failed");
                    Fixture f(9, 48); f.run();
                }
                clear_buffers();
                require(TSGgml_MultiDeviceInit(3, devices, 2) == 1, "Rank backend replacement failed");
                require(TSGgml_SetActiveDevice(1) == 1, "Replaced rank selection failed");
                { Fixture f(3, 64); f.run(); }
            } else {
                require(TSGgml_TensorParallelInitLoopback(3, 2) == 1, "Loopback init failed");
                require(TSGgml_SetActiveDevice(1) == 1, "Loopback rank selection failed");
                { Fixture f(9, 48); f.run(); }
                clear_buffers();
                require(TSGgml_TensorParallelInitLoopback(3, 2) == 1, "Loopback backend replacement failed");
                require(TSGgml_SetActiveDevice(1) == 1, "Replaced loopback rank selection failed");
                { Fixture f(3, 64); f.run(); }
            }
            require(TSGgml_RecreateBackend() == 1, "Backend recreation failed");
            { Fixture f(32, 257); f.run(); }
        }
        TSGgml_Shutdown();
        std::printf("PASS: Spark explicit-F32 projection %s, maximum double-oracle error %.9g\n", argv[1], largest_error);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s; native: %s\n", e.what(), TSGgml_GetLastError());
        TSGgml_Shutdown();
        return 1;
    }
}
