// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#include "layer_partition.h"
#include <cstdio>
#include <functional>
#include <random>
#include <string>

static int failures = 0;
static void expect(bool condition, const char * what)
{
    if (!condition) { std::fprintf(stderr, "FAIL: %s\n", what); ++failures; }
}
static void rejects(const std::function<void()> & fn, const char * what)
{
    try { fn(); expect(false, what); } catch (const std::exception &) {}
}

int main()
{
    using tsg_layer_partition::balance;
    // With a 26-byte budget per node, a 4/4 split needs 20/32 and fails.
    // Moving a layer to node zero needs 25/27; moving two needs 30/22.
    // A 27-byte budget admits the optimal 5/3 split, still using two GPUs/node.
    auto p = balance({5,5,5,5,5,5,5,5}, {0,12}, 2);
    expect(p.boundaries == std::vector<int>({0,5,8}) && p.peak == 27,
           "endpoint head changes the node cut and fits a budget equal-count exceeds");
    expect(p.resident[0] <= 27 && p.resident[1] <= 27 && 4 * 5 + 12 > 27,
           "balanced capacity fits both nodes");
    p = balance({1,1,1,1,1}, {0,0});
    expect(p.boundaries == std::vector<int>({0,2,5}), "equal peaks choose earliest predecessor");
    p = balance({0,0,0,0,0,0,0}, {0,0,0}, 2);
    expect(p.boundaries == std::vector<int>({0,2,4,7}), "minimum local device count and zero-cost ties");
    p = balance({3,3,3,3}, {0,0}, 1, {0,0,20,0});
    expect(p.boundaries == std::vector<int>({0,1,4}) && p.peak == 9,
           "shared cache replica costs alter the optimum");
    p = balance({2,7}, {11});
    expect(p.boundaries == std::vector<int>({0,2}) && p.peak == 20, "single-node endpoint cost");
    p = balance({std::numeric_limits<size_t>::max()}, {0});
    expect(p.peak == std::numeric_limits<size_t>::max(), "maximum size_t is reachable");
    rejects([] { balance({1,1,1}, {0,0}, 2); }, "insufficient layers per GPU");
    rejects([] { balance({1}, {}); }, "empty topology");
    rejects([] { balance({1}, {0}, 0); }, "invalid minimum");
    rejects([] { balance({1,1}, {0}, 1, {0}); }, "wrong replica vector length");
    rejects([] { balance({std::numeric_limits<size_t>::max(),1}, {0}); }, "prefix overflow");
    rejects([] { balance({std::numeric_limits<size_t>::max()}, {1}); }, "fixed cost overflow");
    rejects([] { balance({std::numeric_limits<size_t>::max()}, {0}, 1, {1}); }, "replica overflow");

    // Exhaustive independent enumeration catches DP recurrence/pruning errors
    // over uneven weights, endpoint costs, replica costs and device counts.
    std::mt19937 random(1017);
    for (int trial = 0; trial < 500; ++trial)
    {
        const int nodes = 1 + random() % 4, minimum = 1 + random() % 2;
        const int n = nodes * minimum + random() % 5;
        std::vector<size_t> layers(n), fixed(nodes), replicas(n);
        for (auto & x : layers) x = random() % 20;
        for (auto & x : fixed) x = random() % 25;
        for (auto & x : replicas) x = random() % 10;
        size_t best = std::numeric_limits<size_t>::max();
        std::function<void(int,int,size_t)> visit = [&](int node, int begin, size_t peak) {
            size_t bytes = fixed[node] + replicas[begin];
            for (int end = begin + 1; end <= n - (nodes - node - 1) * minimum; ++end)
            {
                bytes += layers[end - 1];
                if (end - begin < minimum) continue;
                const size_t next = std::max(peak, bytes);
                if (node + 1 == nodes) { if (end == n) best = std::min(best, next); }
                else visit(node + 1, end, next);
            }
        };
        visit(0, 0, 0);
        p = balance(layers, fixed, minimum, replicas);
        expect(p.peak == best, "DP matches exhaustive minimum peak");
    }
    std::printf("layer partition: %s (500 exhaustive comparisons)\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
