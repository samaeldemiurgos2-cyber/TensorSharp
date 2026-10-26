// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
// Device-to-device copies between host-keyed device copies: KV-style row copies
// across different capacities, zeroed padding, in-place reuse of an existing
// destination, refusal paths, and host bytes left untouched until a sync.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

extern "C" {
int TSGgml_IsBackendAvailable(int);
const char* TSGgml_GetLastError();
int TSGgml_SyncHostBuffer(void*, size_t);
void TSGgml_InvalidateHostBuffer(void*);
int TSGgml_HasResidentDeviceCopy(void*, std::int64_t);
std::int64_t TSGgml_CachedBufferBytes(void*);
int TSGgml_EnsureDeviceCopyFromHost(void*, std::int64_t);
int TSGgml_CopyDeviceCopyRanges(void*, std::int64_t, void*, std::int64_t,
    const std::int64_t*, const std::int64_t*, const std::int64_t*, int, int);
void TSGgml_Shutdown();
}

namespace
{
    constexpr int kHeads = 3;
    constexpr int kRowBytes = 64 + 8;   // a block-quantized row size, not a power of two
    constexpr int kSrcCap = 8;
    constexpr int kDstCap = 12;

    std::uint8_t pattern(int head, int row, int byte) { return static_cast<std::uint8_t>(17 * head + 5 * row + byte + 1); }

    bool copy_rows(std::vector<std::uint8_t>& src, std::vector<std::uint8_t>& dst, int rows, std::int64_t dst_bytes)
    {
        std::int64_t so[kHeads], d[kHeads], len[kHeads];
        for (int h = 0; h < kHeads; h++)
        {
            so[h] = static_cast<std::int64_t>(h) * kSrcCap * kRowBytes;
            d[h] = static_cast<std::int64_t>(h) * kDstCap * kRowBytes;
            len[h] = static_cast<std::int64_t>(rows) * kRowBytes;
        }
        return TSGgml_CopyDeviceCopyRanges(src.data(), static_cast<std::int64_t>(src.size()),
            dst.data(), dst_bytes, so, d, len, kHeads, 1) == 1;
    }

    bool check_rows(const std::vector<std::uint8_t>& dst, int rows, const char* what)
    {
        for (int h = 0; h < kHeads; h++)
            for (int r = 0; r < kDstCap; r++)
                for (int b = 0; b < kRowBytes; b++)
                {
                    std::uint8_t expected = r < rows ? pattern(h, r, b) : 0;
                    std::uint8_t actual = dst[(static_cast<std::size_t>(h) * kDstCap + r) * kRowBytes + b];
                    if (actual != expected)
                    {
                        std::fprintf(stderr, "FAIL %s: head %d row %d byte %d = %d, expected %d\n",
                            what, h, r, b, actual, expected);
                        return false;
                    }
                }
        return true;
    }
}

int main(int argc, char** argv)
{
    if (argc != 2) return 2;
    const int backend = !std::strcmp(argv[1], "cuda") ? 3 : !std::strcmp(argv[1], "metal") ? 1 : 2;
    if (!TSGgml_IsBackendAvailable(backend)) return 77;

    std::vector<std::uint8_t> src(static_cast<std::size_t>(kHeads) * kSrcCap * kRowBytes);
    std::vector<std::uint8_t> dst(static_cast<std::size_t>(kHeads) * kDstCap * kRowBytes, 0xAB);
    for (int h = 0; h < kHeads; h++)
        for (int r = 0; r < kSrcCap; r++)
            for (int b = 0; b < kRowBytes; b++)
                src[(static_cast<std::size_t>(h) * kSrcCap + r) * kRowBytes + b] = pattern(h, r, b);
    const auto dst_bytes = static_cast<std::int64_t>(dst.size());

    // No device copy yet: refused, destination untouched.
    if (copy_rows(src, dst, 5, dst_bytes) || TSGgml_CachedBufferBytes(dst.data()) != 0)
    {
        std::fprintf(stderr, "FAIL copy without a resident source was accepted\n");
        return 1;
    }
    if (!TSGgml_EnsureDeviceCopyFromHost(src.data(), static_cast<std::int64_t>(src.size())))
    {
        if (backend == 2) { std::puts("SKIP cpu backend keeps no device copies"); return 77; }
        std::fprintf(stderr, "FAIL upload: %s\n", TSGgml_GetLastError());
        return 1;
    }

    if (!copy_rows(src, dst, 5, dst_bytes))
    {
        std::fprintf(stderr, "FAIL first copy: %s\n", TSGgml_GetLastError());
        return 1;
    }
    for (std::uint8_t byte : dst)
        if (byte != 0xAB) { std::fprintf(stderr, "FAIL host bytes changed before a sync\n"); return 1; }
    if (TSGgml_HasResidentDeviceCopy(dst.data(), dst_bytes) != 1 || TSGgml_CachedBufferBytes(dst.data()) != dst_bytes)
    {
        std::fprintf(stderr, "FAIL destination copy is not resident at its host size\n");
        return 1;
    }
    if (!TSGgml_SyncHostBuffer(dst.data(), dst.size()) || !check_rows(dst, 5, "new destination"))
        return 1;

    // Reuse in place: more rows, stale host bytes ignored, padding zeroed again.
    std::memset(dst.data(), 0xCD, dst.size());
    if (!copy_rows(src, dst, 8, dst_bytes) || !TSGgml_SyncHostBuffer(dst.data(), dst.size()) || !check_rows(dst, 8, "reused destination"))
        return 1;

    // A differently sized destination entry is never replaced underneath a graph.
    if (copy_rows(src, dst, 2, dst_bytes - kRowBytes))
    {
        std::fprintf(stderr, "FAIL resized destination was accepted\n");
        return 1;
    }
    if (!TSGgml_SyncHostBuffer(dst.data(), dst.size()) || !check_rows(dst, 8, "after refused resize"))
        return 1;

    TSGgml_InvalidateHostBuffer(dst.data());
    TSGgml_InvalidateHostBuffer(src.data());
    if (TSGgml_CachedBufferBytes(dst.data()) != 0 || TSGgml_CachedBufferBytes(src.data()) != 0)
    {
        std::fprintf(stderr, "FAIL invalidation left a cached copy\n");
        return 1;
    }
    TSGgml_Shutdown();
    std::puts("PASS device copy ranges: refused without source, new and reused destinations, zeroed padding, host bytes untouched, resize refused");
    return 0;
}
