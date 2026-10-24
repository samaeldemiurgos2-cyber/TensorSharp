// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

struct View { void* data; int rows, cols, row_stride, col_stride; int64_t bytes; };
extern "C" {
int TSGgml_IsBackendAvailable(int);
int TSGgml_GetGpuDeviceCount(int);
int TSGgml_MultiDeviceInit(int, const int*, int);
int TSGgml_SetActiveDevice(int);
int TSGgml_GetActiveDevice();
int TSGgml_GetTensorParallelDegree();
int TSGgml_RecreateBackend();
void TSGgml_ReleaseReuseComputeBuffers();
void TSGgml_Shutdown();
const char* TSGgml_GetLastError();
int TSGgml_MuseSparkPrefillAttentionF32(View, View, View, View,
    int, int, int, int, int, int, int, float, int);
int TSGgml_MuseSparkPagedAttentionF32(View, View, View, View,
    const int*, const int*, const int*, const int*, const int*,
    int, int, int, int, int, int, int, float);
}

namespace {
constexpr int dimension = 64, group = 8, block = 16, guard = 8;
constexpr float scale = .125f, sentinel = -913.75f;
constexpr double absolute = 3e-6, relative = 3e-6; // Existing owned F32 attention oracle.
int graphs = 0, checks = 0;
double largest_error = 0;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

View view(float* p, int rows, int cols) {
    return {p, rows, cols, cols, 1, int64_t(rows) * cols * int64_t(sizeof(float))};
}

struct Sequence {
    int rows, first, keys, query_offset, table_offset;
    std::vector<float> key, value; // Token first, independently retained for the oracle.
};

struct Fixture {
    int kv_heads, heads, total_rows = 0, physical_tokens = 0;
    std::vector<Sequence> sequences;
    std::vector<int> starts, lengths, positions, tables, offsets;
    std::vector<float> query, key_pages, value_pages, output;

    Fixture(int rows, int kv, bool mixed = false, int first_position = 17) : kv_heads(kv), heads(kv * group) {
        int logical_pages = 0;
        for (int s = 0; s < (mixed ? 2 : 1); ++s) {
            const int n = s ? 3 : rows, first = s ? 7 : first_position, keys = first + n;
            sequences.push_back({n, first, keys, total_rows, logical_pages, {}, {}});
            starts.push_back(total_rows); lengths.push_back(keys); offsets.push_back(logical_pages);
            total_rows += n;
            logical_pages += (keys + block - 1) / block;
            for (int row = 0; row < n; ++row) positions.push_back(first + row);
        }
        starts.push_back(total_rows);
        // Reverse odd-numbered physical pages; unused even pages contain poison.
        for (int p = 0; p < logical_pages; ++p) tables.push_back(2 * (logical_pages - p) - 1);
        physical_tokens = (2 * logical_pages + 1) * block;
        query.resize(size_t(total_rows) * heads * dimension);
        key_pages.resize(size_t(physical_tokens) * kv_heads * dimension);
        value_pages.resize(key_pages.size());
        output.resize(guard * 2 + query.size(), sentinel);
        fill(1);
    }

    void fill(int seed, int pattern = 0) {
        std::fill(key_pages.begin(), key_pages.end(), sentinel);
        std::fill(value_pages.begin(), value_pages.end(), sentinel);
        for (size_t si = 0; si < sequences.size(); ++si) {
            auto& seq = sequences[si];
            seq.key.resize(size_t(seq.keys) * kv_heads * dimension);
            seq.value.resize(seq.key.size());
            for (int row = 0; row < seq.rows; ++row)
                for (int head = 0; head < heads; ++head)
                    for (int d = 0; d < dimension; ++d) {
                        const size_t index = (size_t(seq.query_offset + row) * heads + head) * dimension + d;
                        query[index] = pattern ? 0.f :
                            .4f * std::sin(float(row * 17 + head * 13 + d) * .031f + seed * .3f) + .00012207f;
                    }
            for (int token = 0; token < seq.keys; ++token)
                for (int head = 0; head < kv_heads; ++head)
                    for (int d = 0; d < dimension; ++d) {
                        const size_t logical = (size_t(token) * kv_heads + head) * dimension + d;
                        const float k = pattern ? 0.f :
                            .3f * std::cos(float(token * 11 + head * 23 + d) * .027f + seed * .2f) - .000213f;
                        const float v = pattern == 1 ? .25006103515625f : pattern == 2 ? 70000.125f :
                            .5f * std::sin(float(token * 7 + head * 3 + d) * .041f + seed * .5f + float(si)) + .000314f;
                        seq.key[logical] = k; seq.value[logical] = v;
                        const int physical = tables[seq.table_offset + token / block] * block + token % block;
                        const size_t index = (size_t(physical) * kv_heads + head) * dimension + d;
                        key_pages[index] = k; value_pages[index] = v;
                    }
        }
    }

    std::vector<double> oracle(int window) const {
        std::vector<double> expected(query.size()), scores, probabilities;
        for (const auto& seq : sequences) {
            scores.resize(seq.keys); probabilities.resize(seq.keys);
            for (int row = 0; row < seq.rows; ++row)
                for (int head = 0; head < heads; ++head) {
                    const int end = seq.first + row, first = window ? std::max(0, end - window + 1) : 0;
                    const int kv_head = head / group;
                    const size_t q = (size_t(seq.query_offset + row) * heads + head) * dimension;
                    double maximum = -std::numeric_limits<double>::infinity();
                    for (int token = first; token <= end; ++token) {
                        const size_t k = (size_t(token) * kv_heads + kv_head) * dimension;
                        double dot = 0;
                        for (int d = 0; d < dimension; ++d) dot += double(query[q + d]) * seq.key[k + d];
                        scores[token] = dot * double(scale); maximum = std::max(maximum, scores[token]);
                    }
                    double sum = 0;
                    for (int token = first; token <= end; ++token)
                        sum += probabilities[token] = std::exp(scores[token] - maximum);
                    // Token-major accumulation keeps all 64 oracle channels in
                    // a small contiguous working set; every output is checked.
                    double values[dimension] = {};
                    for (int token = first; token <= end; ++token) {
                        const size_t v = (size_t(token) * kv_heads + kv_head) * dimension;
                        for (int d = 0; d < dimension; ++d) values[d] += probabilities[token] * seq.value[v + d];
                    }
                    for (int d = 0; d < dimension; ++d) expected[q + d] = values[d] / sum;
                }
        }
        return expected;
    }

    bool close(const std::vector<float>& actual, const std::vector<double>& expected, bool diagnose) const {
        require(actual.size() == expected.size(), "Oracle size mismatch");
        bool okay = true;
        double maximum = -1;
        size_t worst = 0;
        for (size_t i = 0; i < actual.size(); ++i) {
            double error = std::abs(double(actual[i]) - expected[i]);
            if (!std::isfinite(actual[i]) || !std::isfinite(expected[i]))
                error = std::numeric_limits<double>::infinity();
            if (error > maximum) { maximum = error; worst = i; }
            if (!std::isfinite(error) || error > absolute + relative * std::abs(expected[i])) okay = false;
        }
        if (diagnose) {
            largest_error = std::max(largest_error, maximum);
            if (!okay) std::fprintf(stderr,
                "attention mismatch rank=%d rows=%d heads=%d/%d row=%zu head=%zu channel=%zu actual=%.12g expected=%.12g max=%.9g bound=%.9g\n",
                TSGgml_GetActiveDevice(), total_rows, heads, kv_heads, worst / (heads * dimension),
                (worst / dimension) % heads, worst % dimension, double(actual[worst]), expected[worst],
                maximum, absolute + relative * std::abs(expected[worst]));
        }
        return okay;
    }

    View qview() { return view(query.data(), total_rows, heads * dimension); }
    View kview() { return view(key_pages.data(), physical_tokens, kv_heads * dimension); }
    View vview() { return view(value_pages.data(), physical_tokens, kv_heads * dimension); }
    View oview() { return view(output.data() + guard, total_rows, heads * dimension); }

    int paged(int window) {
        return TSGgml_MuseSparkPagedAttentionF32(qview(), kview(), vview(), oview(),
            starts.data(), lengths.data(), positions.data(), tables.data(), offsets.data(),
            int(sequences.size()), int(tables.size()), heads, kv_heads, dimension, block, window, scale);
    }

    void sentinels() const {
        require(std::all_of(output.begin(), output.begin() + guard, [](float f) { return f == sentinel; }) &&
                std::all_of(output.end() - guard, output.end(), [](float f) { return f == sentinel; }),
                "Attention changed output sentinels");
    }

    void run(int window, const std::vector<double>& expected) {
        const auto before_q = query, before_k = key_pages, before_v = value_pages;
        std::fill(output.begin(), output.end(), sentinel);
        require(paged(window) == 2, "Paged attention did not execute the owned CUDA path");
        ++graphs; sentinels();
        const std::vector<float> paged_result(output.begin() + guard, output.end() - guard);
        require(close(paged_result, expected, true), "Paged attention exceeded the double oracle");
        for (int format : {0, 1}) {
            std::fill(output.begin(), output.end(), sentinel);
            for (const auto& seq : sequences) {
                std::vector<float> q(size_t(seq.rows) * heads * dimension), k(seq.key.size()), v(seq.value.size());
                const float* original_q = query.data() + size_t(seq.query_offset) * heads * dimension;
                for (int row = 0; row < seq.rows; ++row)
                    for (int h = 0; h < heads; ++h)
                        for (int d = 0; d < dimension; ++d)
                            q[(format ? size_t(row) * heads + h : size_t(h) * seq.rows + row) * dimension + d] =
                                original_q[(size_t(row) * heads + h) * dimension + d];
                for (int token = 0; token < seq.keys; ++token)
                    for (int h = 0; h < kv_heads; ++h)
                        for (int d = 0; d < dimension; ++d) {
                            const size_t src = (size_t(token) * kv_heads + h) * dimension + d;
                            const size_t dst = (format ? size_t(token) * kv_heads + h : size_t(h) * seq.keys + token) * dimension + d;
                            k[dst] = seq.key[src]; v[dst] = seq.value[src];
                        }
                const auto saved_q = q, saved_k = k, saved_v = v;
                auto qd = view(q.data(), format ? seq.rows : heads, (format ? heads : seq.rows) * dimension);
                auto kd = view(k.data(), format ? seq.keys : kv_heads, (format ? kv_heads : seq.keys) * dimension);
                auto vd = kd; vd.data = v.data();
                auto od = view(output.data() + guard + size_t(seq.query_offset) * heads * dimension,
                    seq.rows, heads * dimension);
                require(TSGgml_MuseSparkPrefillAttentionF32(qd, kd, vd, od, heads, kv_heads, dimension,
                    seq.rows, seq.keys, seq.first, window, scale, format) == 2,
                    "Ordinary attention did not execute the owned CUDA path");
                ++graphs;
                require(q == saved_q && k == saved_k && v == saved_v, "Ordinary attention modified input");
            }
            sentinels();
            const std::vector<float> ordinary(output.begin() + guard, output.end() - guard);
            require(close(ordinary, expected, true), "Ordinary attention exceeded the double oracle");
            const std::vector<double> paged_double(paged_result.begin(), paged_result.end());
            require(close(ordinary, paged_double, true), "Ordinary and paged attention diverged");
        }
        require(query == before_q && key_pages == before_k && value_pages == before_v,
                "Paged attention modified input or unused physical pages");
        // These controls exercise the same predicate against actual outputs.
        auto damaged = paged_result;
        damaged[damaged.size() / 2] += 1.f;
        require(!close(damaged, expected, false), "Single-cell corruption was not detected");
        damaged = paged_result;
        damaged[0] = std::numeric_limits<float>::quiet_NaN();
        require(!close(damaged, expected, false), "Non-finite output was not detected");
        ++checks;
    }
};

void reject_inputs(Fixture& f) {
    auto q = f.qview(), k = f.kview(), v = f.vview(), o = f.oview();
    auto call = [&](View a, View b, View c, View d, int entries) {
        return TSGgml_MuseSparkPagedAttentionF32(a, b, c, d, f.starts.data(), f.lengths.data(),
            f.positions.data(), f.tables.data(), f.offsets.data(), int(f.sequences.size()), entries,
            f.heads, f.kv_heads, dimension, block, 0, scale);
    };
    auto unchanged = [&]() {
        require(std::all_of(f.output.begin(), f.output.end(), [](float n) { return n == sentinel; }),
                "Rejected attention request changed output");
    };
    std::fill(f.output.begin(), f.output.end(), sentinel);
    for (int which = 0; which < 4; ++which) {
        auto a = q, b = k, c = v, d = o;
        auto& bad = which == 0 ? a : which == 1 ? b : which == 2 ? c : d;
        ++bad.bytes;
        require(call(a, b, c, d, int(f.tables.size())) == 0, "Oversized paged span accepted");
        unchanged();
    }
    auto bad = q; bad.col_stride = 2;
    require(call(bad, k, v, o, int(f.tables.size())) == 0, "Strided paged query accepted"); unchanged();
    require(call(q, k, v, o, int(f.tables.size()) - 1) == 0, "Truncated page table accepted"); unchanged();
    const int page = f.tables[0];
    for (int invalid : {-1, f.physical_tokens / block}) {
        f.tables[0] = invalid;
        require(call(q, k, v, o, int(f.tables.size())) == 0, "Invalid physical page accepted"); unchanged();
    }
    f.tables[0] = page;
    const int position = f.positions[0]; f.positions[0] = f.lengths[0];
    require(call(q, k, v, o, int(f.tables.size())) == 0, "Out-of-range query position accepted"); unchanged();
    f.positions[0] = position;
    // Flat single-sequence ordinary spans use exactly the same logical values.
    auto& seq = f.sequences[0];
    auto kd = view(seq.key.data(), seq.keys, f.kv_heads * dimension), vd = kd;
    vd.data = seq.value.data();
    for (int which = 0; which < 4; ++which) {
        auto a = q, b = kd, c = vd, d = o;
        auto& malformed = which == 0 ? a : which == 1 ? b : which == 2 ? c : d;
        ++malformed.bytes;
        require(TSGgml_MuseSparkPrefillAttentionF32(a, b, c, d, f.heads, f.kv_heads, dimension,
            seq.rows, seq.keys, seq.first, 0, scale, 1) == 0, "Oversized ordinary span accepted");
        unchanged();
    }
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
            Fixture f(3, 1);
            require(f.paged(0) == 0, "CUDA-only paged entry silently executed CPU");
            auto& seq = f.sequences[0];
            require(TSGgml_MuseSparkPrefillAttentionF32(f.qview(),
                view(seq.key.data(), seq.keys, dimension), view(seq.value.data(), seq.keys, dimension),
                f.oview(), f.heads, f.kv_heads, dimension, seq.rows, seq.keys, seq.first, 0, scale, 1) == 0,
                "CUDA-only ordinary entry silently executed CPU");
            require(std::all_of(f.output.begin(), f.output.end(), [](float f) { return f == sentinel; }),
                    "Unsupported backend changed output");
        } else {
            if (multi) {
                const int devices[] = {1, 0};
                require(TSGgml_MultiDeviceInit(3, devices, 2) == 1, "Reordered rank initialization failed");
            }
            { Fixture f(9, 2); reject_inputs(f); }
            int index = 0;
            for (int rows : {1, 3, 9, 32, 256})
                for (int kv : {1, 2, 16})
                    for (int window : {0, 31}) {
                        const int rank = multi ? index % 2 : 0;
                        require(TSGgml_SetActiveDevice(rank) == 1, "Rank selection failed");
                        Fixture f(rows, kv); f.fill(++index);
                        auto expected = f.oracle(window);
                        f.run(window, expected);
                        if (rows == 9 && kv == 2 && window == 0) {
                            f.run(window, expected); // Same-pointer warm reuse.
                            f.fill(index + 113); f.run(window, f.oracle(window)); // New values, same geometry.
                        }
                    }
            // Both low bits below F16 resolution and values outside F16 range
            // must survive: substituting generic flash attention cannot pass.
            for (int pattern : {1, 2}) {
                Fixture f(3, 2); f.fill(3, pattern); f.run(0, f.oracle(0));
            }
            { Fixture mixed(9, 16, true); mixed.fill(53); mixed.run(31, mixed.oracle(31)); }
            for (int rows : {256, 1, 256}) {
                Fixture f(rows, 2); f.fill(rows + 29); f.run(31, f.oracle(31));
            }
            // Cross the native 256-query chunk boundary. Format zero must
            // retain the original full head stride for the final one-row chunk.
            for (int window : {0, 31}) {
                Fixture f(257, 2); f.fill(701); f.run(window, f.oracle(window));
            }
            {
                Fixture f(9, 2, false, 65); f.fill(811);
                constexpr int window = 31;
                const int first_live_block = (f.sequences[0].first - window + 1) / block;
                // A sliding-window cache may have evicted entire older pages.
                // These table sentinels are outside the live union and unread.
                for (int p = 0; p < first_live_block; ++p) f.tables[p] = -1;
                f.run(window, f.oracle(window));
                // The same sentinel in the first LIVE page must fail before
                // writing output, even though earlier evicted pages are valid.
                f.tables[first_live_block] = -1;
                std::fill(f.output.begin(), f.output.end(), sentinel);
                require(f.paged(window) == 0, "Evicted live SWA page accepted");
                require(std::all_of(f.output.begin(), f.output.end(), [](float n) { return n == sentinel; }),
                        "Rejected live SWA page changed output");
            }
            TSGgml_ReleaseReuseComputeBuffers();
            if (multi) require(TSGgml_SetActiveDevice(1) == 1, "Final rank-one selection failed");
            require(TSGgml_RecreateBackend() == 1, "Backend recreation failed");
            require(TSGgml_GetActiveDevice() == 0 && TSGgml_GetTensorParallelDegree() == 1,
                    "Recreation did not reset the active rank/degree");
            { Fixture f(32, 2); f.fill(901); f.run(0, f.oracle(0)); }
        }
        TSGgml_Shutdown();
        std::printf("PASS: Spark F32 attention %s; %d full-array oracle cases, %d owned CUDA calls; max error %.12g; bound 3e-6+3e-6*abs(reference)\n",
            argv[1], checks, graphs, largest_error);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s; native: %s\n", error.what(), TSGgml_GetLastError());
        TSGgml_Shutdown();
        return 1;
    }
}
