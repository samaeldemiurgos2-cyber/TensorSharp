// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#pragma once

// Model-independent contiguous pipeline partitioning. Costs are resident bytes,
// not checkpoint bytes: callers exclude host weights and include device caches.
// Every node derives the same plan from metadata before allocating weights.
#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace tsg_layer_partition {

inline size_t checked_add(size_t a, size_t b)
{
    if (b > std::numeric_limits<size_t>::max() - a)
        throw std::overflow_error("Distributed layer resident byte count overflow");
    return a + b;
}

struct plan
{
    std::vector<int> boundaries;      // node k owns [boundaries[k], boundaries[k+1])
    std::vector<size_t> resident;     // includes fixed and boundary-dependent bytes
    size_t peak = 0;
};

// `fixed` prices node-specific residents such as embeddings and output heads.
// `at_start[b]` optionally prices state replicas a stage starting at b owns,
// independent of its end. For end-dependent costs use a separate planner.
// Among equal optima choose the smallest preceding boundary at each DP step.
inline plan balance(const std::vector<size_t> & layers, const std::vector<size_t> & fixed,
                    int min_layers = 1, const std::vector<size_t> & at_start = {})
{
    if (layers.size() > (size_t) std::numeric_limits<int>::max() || fixed.empty() ||
        min_layers < 1 || fixed.size() > layers.size() / (size_t) min_layers ||
        (!at_start.empty() && at_start.size() != layers.size()))
        throw std::invalid_argument("Distributed layer split requires at least one layer per selected device and valid costs");
    const int n = (int) layers.size(), nodes = (int) fixed.size();
    std::vector<size_t> prefix((size_t) n + 1, 0);
    for (int i = 0; i < n; ++i) prefix[i + 1] = checked_add(prefix[i], layers[i]);
    auto cost = [&](int node, int begin, int end) {
        return checked_add(checked_add(prefix[end] - prefix[begin], fixed[node]),
                           at_start.empty() ? 0 : at_start[begin]);
    };
    // Reachability is separate from cost, so SIZE_MAX remains a valid total.
    std::vector<std::vector<size_t>> peak((size_t) nodes + 1, std::vector<size_t>((size_t) n + 1));
    std::vector<std::vector<int>> previous((size_t) nodes + 1, std::vector<int>((size_t) n + 1, -1));
    previous[0][0] = 0;
    for (int k = 1; k <= nodes; ++k)
        for (int end = k * min_layers; end <= n - (nodes - k) * min_layers; ++end)
            for (int begin = (k - 1) * min_layers; begin <= end - min_layers; ++begin)
            {
                if (previous[k - 1][begin] < 0) continue;
                const size_t candidate = std::max(peak[k - 1][begin], cost(k - 1, begin, end));
                if (previous[k][end] < 0 || candidate < peak[k][end])
                {
                    peak[k][end] = candidate;
                    previous[k][end] = begin;
                }
            }
    plan out;
    out.boundaries.resize((size_t) nodes + 1);
    out.resident.resize(nodes);
    out.boundaries[nodes] = n;
    out.peak = peak[nodes][n];
    for (int k = nodes; k > 0; --k)
    {
        const int end = out.boundaries[k], begin = previous[k][end];
        if (begin < 0) throw std::logic_error("Distributed layer partition has no feasible topology");
        out.boundaries[k - 1] = begin;
        out.resident[k - 1] = cost(k - 1, begin, end);
    }
    return out;
}

} // namespace tsg_layer_partition
