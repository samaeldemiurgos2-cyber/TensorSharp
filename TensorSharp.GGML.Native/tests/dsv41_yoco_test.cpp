// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#include "dsv41_yoco.h"

#include <cstdio>
#include <cstdlib>
#include <string>

static int checks = 0;
static void require(bool result, const char * name)
{
    if (!result) { std::fprintf(stderr, "FAIL %s\n", name); std::exit(1); }
    ++checks;
}

// The released V4.1 layout (40 layers): ratio 0 for layers 0-1, ratio 2 for
// 2-19 sharing kv sources 2/8/14, ratio 1 from layer 20 on reading layer 20's
// cache, Engram on layers 1 and 14, indexer candidate source 20.
static tsg_dsv41_yoco::layout released()
{
    tsg_dsv41_yoco::layout l;
    l.n_layer = 40;
    for (int il = 0; il < 40; ++il)
    {
        const int ratio = il < 2 ? 0 : il < 20 ? 2 : 1;
        l.compress_ratios.push_back(ratio);
        l.kv_source.push_back(ratio == 0 ? il : ratio == 2 ? (il < 8 ? 2 : il < 14 ? 8 : 14) : 20);
    }
    l.engram_layers = {1, 14};
    l.candidate_source = 20;
    return l;
}

int main()
{
    namespace y = tsg_dsv41_yoco;
    // Parsing: unset is the default; 0 is off; values below n_swa + 8 refuse.
    {
        const auto unset = y::parse(nullptr, 128);
        require(!unset.explicit_value && unset.keep == 140 && unset.error.empty(), "unset keeps DeepInfer's 140");
        const auto off = y::parse("0", 128);
        require(off.explicit_value && off.keep == 0 && off.error.empty(), "0 turns replay off");
        require(y::parse("140", 128).keep == 140, "explicit 140");
        require(y::parse("136", 128).error.empty(), "the minimum n_swa + 8 is accepted");
        for (const char * bad : {"135", "1", "-1", "", "abc", "140x", "1073741825", "99999999999999999999"})
            require(!y::parse(bad, 128).error.empty(), "malformed or too-short values refuse");
        require(y::minimum_keep(128) == 136, "minimum is n_swa + 8");
        const auto wide = y::parse(nullptr, 256);
        require(!wide.explicit_value && wide.keep == 264 && wide.error.empty(),
                "unset keeps at least the window minimum for a wider window");
    }
    // Layout eligibility.
    {
        std::string reason;
        auto l = released();
        require(y::replay_layer(l, reason) == 20 && reason.empty(), "released layout replays from layer 20");
        l.drafter_targets = {37, 38, 39};
        require(y::replay_layer(l, reason) == 20, "a drafter reading layers 37-39 is compatible");
        for (const auto & bad : {std::vector<int>{20, 38, 39}, std::vector<int>{3}})
        {
            auto d = released();
            d.drafter_targets = bad;
            require(y::replay_layer(d, reason) < 0 && !reason.empty(), "a drafter target at or before layer 20 refuses");
        }
        auto engram = released();
        engram.engram_layers = {1, 25};
        require(y::replay_layer(engram, reason) < 0, "an Engram layer after layer 20 refuses");
        auto candidate = released();
        candidate.candidate_source = 14;
        require(y::replay_layer(candidate, reason) < 0, "a candidate source before layer 20 refuses");
        auto owner = released();
        owner.kv_source[30] = 30;
        require(y::replay_layer(owner, reason) < 0, "a deeper layer owning a cache refuses");
        auto none = released();
        for (auto & ratio : none.compress_ratios) if (ratio == 1) ratio = 2;
        require(y::replay_layer(none, reason) < 0, "no ratio-1 layer refuses");
    }
    // Replay start, kept tails.
    {
        require(y::start(140, 0, 2048) == 1908, "replay starts keep before the call's end");
        require(y::start(140, 0, 100) < 0, "a call shorter than keep runs every layer");
        require(y::start(140, 1950, 2048) < 0, "a call whose replay would start before it runs every layer");
        require(y::start(140, 1900, 2048) == 1908, "a chunked call keeps its own tail");
        require(y::start(0, 0, 4096) < 0, "keep 0 is off");
        require(y::start(140, 2048, 2048) < 0, "a call start at its end (visual, verify) is off");
        require(y::kept(1908, 512, 1536, false, 0) == 140, "the last micro-batch keeps the tail");
        require(y::kept(1908, 512, 1024, false, 0) == 0, "an earlier micro-batch keeps nothing");
        require(y::kept(1908, 512, 1536, true, 0) < 0, "verify rows run every layer");
        require(y::kept(1908, 512, 1536, false, 3) < 0, "image micro-batches run every layer");
        require(y::kept(1908, 1, 2047, false, 0) < 0, "single-token steps run every layer");
        require(y::kept(-1, 512, 1536, false, 0) < 0, "no replay start, no narrowing");
    }
    // Rewinds: a head whose window reaches a skipped span cannot reuse.
    {
        const std::vector<std::pair<int64_t, int64_t>> skipped = {{0, 1908}};
        require(!y::blocks(skipped, 2047, 128), "the call's own end reuses");
        require(!y::blocks(skipped, 2035, 128), "a 13-token rewind reuses");
        require(y::blocks(skipped, 2034, 128), "a 14-token rewind re-prefills");
        require(y::blocks(skipped, 100, 128), "a rewind into the skipped span re-prefills");
        require(!y::blocks({}, 100, 128), "nothing skipped, nothing blocked");
        // The head's own row is written by the next forward before it is read.
        require(!y::blocks({{2048, 3000}}, 2048, 128), "a rewind to the start of a truncated call reuses");
        require(y::blocks({{2048, 3000}}, 2049, 128), "a rewind one past a truncated call's start re-prefills");
        require(!y::blocks({{1000, 2000}}, 2127, 128), "a window that starts at a span's end reuses");
        require(y::blocks({{1000, 2000}}, 2126, 128), "a window that reaches into a span re-prefills");
    }
    std::printf("%d YOCO bounded-replay policy checks passed\n", checks);
    return 0;
}
