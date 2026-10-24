// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

extern "C" {
int TSGgml_IsBackendAvailable(int);
int TSGgml_HasResidentDeviceCopy(void*, int64_t);
int64_t TSGgml_DeviceCopyCacheResidentBytes();
int TSGgml_PreloadQuantizedWeight(void*, void*, int, int64_t, int64_t, int64_t);
void TSGgml_ClearHostBufferCache();
void TSGgml_InvalidateHostBuffer(void*);
void TSGgml_Shutdown();
int TSGgml_MoEFFNPrefillSwiGLUQuantF32(
    float*, float*, int, int, int, int, int, const int32_t*, const float*,
    void*, int, int64_t, int64_t, int64_t,
    void*, int, int64_t, int64_t, int64_t,
    void*, int, int64_t, int64_t, int64_t,
    const float*, const float*, const float*, int, float, float, int);
}

int main(int argc, char** argv) {
    const bool cuda = argc == 2 && std::strcmp(argv[1], "cuda") == 0;
    if (argc > 2 || (argc == 2 && !cuda && std::strcmp(argv[1], "cpu") != 0)) return 2;
    if (!TSGgml_IsBackendAvailable(cuda ? 3 : 2)) return 77;
    auto require = [](bool ok, const char* label) {
        if (!ok) std::fprintf(stderr, "FAIL: %s\n", label);
        return ok;
    };
    std::vector<float> preload(64 * 64, .001f), gate(64 * 64 * 2, .01f), up(gate.size(), .02f), down(gate.size(), .03f);
    const int64_t bytes = int64_t(gate.size() * sizeof(float)), preload_bytes = int64_t(preload.size() * sizeof(float));
    int key = 0;
    const auto before = TSGgml_DeviceCopyCacheResidentBytes();
    for (int i = 0; i < 3; ++i) {
        if (!require(TSGgml_HasResidentDeviceCopy(gate.data(), bytes) == 0 &&
            TSGgml_HasResidentDeviceCopy(nullptr, bytes) == 0 &&
            TSGgml_HasResidentDeviceCopy(gate.data(), 0) == 0 &&
            TSGgml_HasResidentDeviceCopy(gate.data(), -1) == 0, "absent/invalid query")) return 1;
    }
    if (!require(TSGgml_DeviceCopyCacheResidentBytes() == before, "probe allocated a device copy")) return 1;
    if (!require(TSGgml_PreloadQuantizedWeight(&key, preload.data(), 0, 64, 64, preload_bytes) == 1, "preload")) return 1;
    if (!require(TSGgml_HasResidentDeviceCopy(&key, preload_bytes) == int(cuda) &&
        TSGgml_HasResidentDeviceCopy(&key, preload_bytes - 1) == 0 &&
        TSGgml_HasResidentDeviceCopy(preload.data(), preload_bytes) == 0, "preloaded key and exact size")) return 1;

    std::vector<float> input(2 * 64, .1f), output(input.size()), first;
    const int32_t ids[] = {0, 1}; const float weights[] = {1, 1};
    auto run = [&]() {
        return TSGgml_MoEFFNPrefillSwiGLUQuantF32(input.data(), output.data(), 2, 64, 64, 2, 1, ids, weights,
            gate.data(), 0, 64, 64, bytes, up.data(), 0, 64, 64, bytes, down.data(), 0, 64, 64, bytes,
            nullptr, nullptr, nullptr, 0, 1.702f, 7.f, 0) == 1;
    };
    if (!require(run(), "lazy MoE upload")) return 1;
    first = output;
    const auto resident = TSGgml_DeviceCopyCacheResidentBytes();
    for (void* pointer : {static_cast<void*>(gate.data()), static_cast<void*>(up.data()), static_cast<void*>(down.data())})
        if (!require(TSGgml_HasResidentDeviceCopy(pointer, bytes) == int(cuda) &&
            TSGgml_HasResidentDeviceCopy(pointer, bytes + 1) == 0, "lazy copy query")) return 1;
    if (!require(TSGgml_DeviceCopyCacheResidentBytes() == resident && run(), "probe mutated cache")) return 1;
    for (size_t i = 0; i < output.size(); ++i)
        if (!require(std::isfinite(output[i]) && std::fabs(output[i] - first[i]) < 1e-6f, "repeat output")) return 1;
    TSGgml_InvalidateHostBuffer(gate.data());
    if (!require(TSGgml_HasResidentDeviceCopy(gate.data(), bytes) == 0 && run(), "invalidate and reload")) return 1;
    for (size_t i = 0; i < output.size(); ++i)
        if (!require(std::fabs(output[i] - first[i]) < 1e-6f, "reload output")) return 1;
    TSGgml_ClearHostBufferCache();
    if (!require(TSGgml_HasResidentDeviceCopy(&key, preload_bytes) == 0 &&
        TSGgml_HasResidentDeviceCopy(gate.data(), bytes) == 0, "cleared query")) return 1;
    TSGgml_Shutdown();
    std::puts("PASS: read-only exact device-copy queries, preload/lazy cache, invalidation and reload");
    return 0;
}
