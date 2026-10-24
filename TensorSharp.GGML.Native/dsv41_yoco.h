// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#pragma once

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

// YOCO bounded replay for DeepSeek V4.1 (DeepInfer's YOCO prefill truncation,
// DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE=compact with _KEEP_TOKENS=140).
//
// From the first ratio-1 layer on (layer 20 of 40 in the released checkpoint),
// every layer reads the compressed cache that layer itself writes from its
// attention INPUT, so the deeper layers' only per-position state of their own
// is the sliding window. A multi-token call therefore runs those layers over
// its last `keep` positions only, each replayed position attending the window
// inside the replay. That is approximate, which is why the replay window must
// stay a little longer than the sliding window (n_swa + 8 minimum).
//
// On by default with keep = 140. TS_DSV41_YOCO_KEEP=0 runs every layer over
// every position; any other value must be at least n_swa + 8. A checkpoint
// whose layout cannot run it falls back to exact prefill with one note under
// the default and is refused when a value was given explicitly.
//
// This header is the arithmetic only (no model types), so the policy is
// testable without a checkpoint: tests/dsv41_yoco_test.cpp.
namespace tsg_dsv41_yoco
{
constexpr int64_t default_keep = 140;

inline int64_t minimum_keep(int64_t n_swa) { return n_swa + 8; }

struct setting
{
    bool explicit_value = false; // TS_DSV41_YOCO_KEEP was set
    int64_t keep = default_keep; // 0 = off
    std::string error;           // non-empty: malformed or out-of-range value
};

// `value` is the environment variable's text, or nullptr when it is unset.
inline setting parse(const char * value, int64_t n_swa)
{
    setting s;
    if (!value)
    {
        // DeepInfer's 140 suits V4.1's 128-token window; a checkpoint with a
        // wider window still gets the minimum replay it needs.
        s.keep = std::max(default_keep, minimum_keep(n_swa));
        return s;
    }
    s.explicit_value = true;
    char * end = nullptr;
    errno = 0;
    const long long keep = std::strtoll(value, &end, 10);
    const int64_t minimum = minimum_keep(n_swa);
    if (end == value || *end || errno == ERANGE || keep < 0 || keep > (1ll << 30) || (keep > 0 && keep < minimum))
    {
        s.error = "TS_DSV41_YOCO_KEEP must be 0 (off) or a token count of at least " + std::to_string(minimum) +
                  " (sliding window + 8); unset uses " + std::to_string(default_keep);
        s.keep = 0;
        return s;
    }
    s.keep = keep;
    return s;
}

struct layout
{
    int n_layer = 0;
    std::vector<int> compress_ratios;   // per layer
    std::vector<int> kv_source;         // per layer: the layer whose compressed cache it reads
    std::vector<int> engram_layers;
    int candidate_source = -1;          // indexer candidate source, -1 when absent
    std::vector<int> drafter_targets;   // DSpark target layers (V4.1: their attention inputs); empty without a drafter
};

// The first layer bounded replay narrows (the first ratio-1 layer), or -1 with
// `reason` saying why this checkpoint cannot run it.
inline int replay_layer(const layout & l, std::string & reason)
{
    reason.clear();
    int first = -1;
    for (int il = 0; il < l.n_layer && il < (int) l.compress_ratios.size() && first < 0; ++il)
        if (l.compress_ratios[il] == 1) first = il;
    if (first < 0)
    {
        reason = "it has no ratio-1 layer";
        return -1;
    }
    if (first >= (int) l.kv_source.size() || l.kv_source[first] != first)
    {
        reason = "its first ratio-1 layer does not own its compressed cache";
        return -1;
    }
    for (int il = first + 1; il < l.n_layer; ++il)
        if (l.compress_ratios[il] != 0 && l.kv_source[il] != first)
        {
            reason = "layer " + std::to_string(il) + " owns a compressed cache after layer " + std::to_string(first);
            return -1;
        }
    for (const int il : l.engram_layers)
        if (il >= first)
        {
            reason = "Engram layer " + std::to_string(il) + " is not before layer " + std::to_string(first);
            return -1;
        }
    // The candidate mask is built over the whole micro-batch at its source
    // layer and added to the replay layers' index masks.
    if (l.candidate_source >= 0 && l.candidate_source < first)
    {
        reason = "its indexer candidate source (layer " + std::to_string(l.candidate_source) +
                 ") is before layer " + std::to_string(first);
        return -1;
    }
    // A drafter reads its target layers' inputs for every position it keeps.
    // A target at or before the narrowing layer sees the whole micro-batch
    // while the replayed ones see only the tail.
    for (const int target : l.drafter_targets)
        if (target <= first)
        {
            reason = "the drafter reads layer " + std::to_string(target) + ", not after layer " + std::to_string(first);
            return -1;
        }
    return first;
}

// First position of a call [call_start, call_end) that the replay layers
// process, or -1 when the call runs every layer over every position.
inline int64_t start(int64_t keep, int64_t call_start, int64_t call_end)
{
    if (keep <= 0) return -1;
    const int64_t first = call_end - keep;
    return first > call_start ? first : -1;
}

// How many tokens of the micro-batch [p0, p0 + nt) - its tail - the replay
// layers process, or -1 for a full graph. Speculative verify (per-row logits),
// image micro-batches and single-token steps always run every layer.
inline int64_t kept(int64_t replay_start, int64_t nt, int64_t p0, bool all_logits, int image_tokens)
{
    if (replay_start < 0 || all_logits || image_tokens || nt <= 1) return -1;
    return std::clamp<int64_t>(p0 + nt - std::max(p0, replay_start), 0, nt);
}

// True when a head at `target` would read replay-layer window rows for
// positions a call skipped ([first, second) spans). The next forward's first
// query, at `target`, reads (target - n_swa, target]; it writes row `target`
// itself before reading it, so only (target - n_swa, target) must be intact.
inline bool blocks(const std::vector<std::pair<int64_t, int64_t>> & skipped, int64_t target, int64_t n_swa)
{
    for (const auto & span : skipped)
        if (std::max(span.first, target - n_swa + 1) < std::min(span.second, target)) return true;
    return false;
}
}
