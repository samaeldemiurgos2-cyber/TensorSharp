// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// DeepSeek V4.1 prefill must not depend on how a prompt is split into forward
// calls. Every micro-batch's attention width (raw ring + compressed-row
// bucket) chooses the owned F32 attention kernel (sparse at 2,048 keys and
// more, tiled below) and, for decode-class launches, the split-key partition
// count. Those choices change the rounding, and V4.1 quantizes what attention
// produces into its caches, so a width that followed the caller's call end made
// a prompt prefilled whole and the same prompt prefilled in 256-token chunks
// build different caches: on the lite 1017 checkpoint, logits 4.5e-2..1.6e-1
// apart with changed argmax tokens at 4,204 tokens, and the exact boundaries
// 1,024/1,025, 1,659/1,660 and 2,049/2,050 that the bisect measured.
//
// This checks the rule production uses (dsv4_comp_bucket.h, the same gate
// function attn_mha calls) on the released checkpoint's geometry, without a
// model: for every micro-batch, the whole-prompt call and the chunked calls
// must give it the same width and the same kernel. It also pins that the
// former call-end floor -- still plain V4's rule -- fails this check at exactly
// the measured boundaries, so the check can see the defect it guards.
#include "dsv4_comp_bucket.h"
#include "dsv4_ubatch_plan.h"
#include "ggml_ops_precision_policy.h"

#include <cstdio>
#include <string>
#include <vector>

using namespace tsg_dsv4_plan;

static int failures = 0;

static void expect(bool condition, const std::string & what)
{
    if (!condition)
    {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

// The released DeepSeek V4.1 Flash geometry (sliding window 128, indexer top-k
// 512) at a 64K context; the compression ratios V4.1 uses are 2 and 1.
struct geometry
{
    int64_t n_swa = 128, top_k = 512, n_ctx = 65536, ubatch = 256;
};

struct launch
{
    int64_t p0 = 0, nt = 0;
    int64_t keys[2] = {};      // attention width at ratio 1, ratio 2
    int capacity[2] = {};      // the owned attention's sparse capacity (0 = dense kernels)
};

static int64_t kv_rows(const geometry & g, int64_t ratio) { return pad(g.n_ctx / ratio + 1, 256); }

// One micro-batch [p0, p0 + nt) of a forward call ending at `call_end`, as
// dsv4_acquire_graph -> build_comp_plan -> build_attention_v41 -> attn_mha see
// it. `v41` false applies plain V4's floor, which V4.1 used before the fix.
static launch micro_batch(const geometry & g, bool v41, int64_t call_end, int64_t p0, int64_t nt)
{
    launch l;
    l.p0 = p0;
    l.nt = nt;
    const int64_t ring = ring_rows(g.n_swa, g.ubatch);
    for (int i = 0; i < 2; ++i)
    {
        const int64_t ratio = i + 1;
        // build_comp_plan: the last row's visible compressed rows, (pos + 1) / ratio.
        const int64_t visible = (p0 + nt) / ratio;
        const int64_t bucket = comp_bucket_rows(visible, comp_floor_rows(v41, call_end, p0, nt, ratio), kv_rows(g, ratio));
        l.keys[i] = ring + bucket;
        l.capacity[i] = tsg_dsv41_owned_sparse_capacity(nt, l.keys[i], g.n_swa, g.top_k, nullptr);
    }
    return l;
}

// The micro-batches of a prompt of `prompt` tokens sent in calls of `chunk`
// tokens (the whole prompt in one call when chunk >= prompt), followed by
// `decode` single-token calls.
static std::vector<launch> schedule(const geometry & g, bool v41, int64_t prompt, int64_t chunk, int64_t decode)
{
    std::vector<launch> out;
    for (int64_t start = 0; start < prompt; start += chunk)
    {
        const int64_t end = std::min(prompt, start + chunk);
        for (int64_t p0 = start; p0 < end; p0 += g.ubatch)
            out.push_back(micro_batch(g, v41, end, p0, std::min(g.ubatch, end - p0)));
    }
    for (int64_t step = 0; step < decode; ++step)
        out.push_back(micro_batch(g, v41, prompt + step + 1, prompt + step, 1));
    return out;
}

// Micro-batches whose width or kernel differ between the whole-prompt call and
// the chunked calls, at ratio index `ratio_index` (0 = ratio 1, 1 = ratio 2).
static int differing(const geometry & g, bool v41, int64_t prompt, int64_t chunk, int ratio_index, bool kernel_only)
{
    const auto whole = schedule(g, v41, prompt, prompt, 4);
    const auto chunked = schedule(g, v41, prompt, chunk, 4);
    if (whole.size() != chunked.size()) return -1;
    int count = 0;
    for (size_t i = 0; i < whole.size(); ++i)
    {
        if (whole[i].p0 != chunked[i].p0 || whole[i].nt != chunked[i].nt) return -1;
        const bool kernel = (whole[i].capacity[ratio_index] > 0) != (chunked[i].capacity[ratio_index] > 0);
        const bool width = whole[i].keys[ratio_index] != chunked[i].keys[ratio_index];
        count += kernel_only ? kernel : (kernel || width);
    }
    return count;
}

static std::string label(const geometry & g, int64_t prompt, int64_t chunk)
{
    return "ubatch " + std::to_string(g.ubatch) + ", prompt " + std::to_string(prompt) + ", chunk " + std::to_string(chunk);
}

int main()
{
    // 1. The bucket rule itself.
    expect(comp_bucket_rows(1, 0, 1 << 20) == 256, "bucket floor is 256 rows");
    expect(comp_bucket_rows(257, 0, 1 << 20) == 512, "bucket rounds 257 up to 512");
    expect(comp_bucket_rows(1025, 0, 1 << 20) == 2048, "bucket rounds 1,025 up to 2,048");
    expect(comp_bucket_rows(9000, 0, 8448) == 8448, "bucket is capped at the cache rows");
    expect(comp_bucket_rows(10, 3000, 1 << 20) == 4096, "a floor raises the bucket");

    // 2. Plain V4 keeps its call-end floor (capped at 8,192 positions): every
    //    chunk of a pipelined prefill shares one graph shape.
    expect(comp_floor_rows(false, 4204, 0, 256, 4) == 4204 / 4, "V4 floors at the call end");
    expect(comp_floor_rows(false, 20000, 0, 256, 128) == 8192 / 128, "V4's floor stops at 8,192 positions");
    expect(comp_floor_rows(false, 100, 256, 256, 4) == 512 / 4, "V4's floor covers the micro-batch itself");

    // 3. V4.1 has none: a micro-batch's bucket is its own.
    for (int64_t call_end : {256, 1024, 4204, 65536})
        expect(comp_floor_rows(true, call_end, 0, 256, 1) == 0, "V4.1 has no call-end floor (call end " + std::to_string(call_end) + ")");

    // 4. Whole and chunked prefill give every micro-batch the same width and
    //    kernel, at every measured boundary, ubatch and chunking of the bisect.
    const int64_t prompts[] = {1024, 1025, 1536, 1659, 1660, 2049, 2050, 4200, 4204, 8534, 20000};
    const int64_t ubatches[] = {128, 256, 512, 1024};
    int checked = 0;
    for (int64_t ubatch : ubatches)
    {
        geometry g;
        g.ubatch = ubatch;
        for (int64_t prompt : prompts)
            for (int64_t chunk : {ubatch, 2 * ubatch, 4 * ubatch, (2304 / ubatch) * ubatch, 9 * ubatch})
            {
                if (chunk <= 0 || chunk % ubatch) continue;
                for (int r = 0; r < 2; ++r)
                {
                    const int d = differing(g, true, prompt, chunk, r, false);
                    expect(d == 0, "V4.1 ratio " + std::to_string(r + 1) + ", " + label(g, prompt, chunk) + ": " +
                        std::to_string(d) + " micro-batches change attention width or kernel with the call split");
                    ++checked;
                }
            }
    }
    expect(checked > 300, "checked " + std::to_string(checked) + " configurations");

    // 5. The former rule (the call-end floor) breaks exactly where the bisect
    //    measured it on the lite 1017 (chunk 256, ubatch 256): ratio-1 layers
    //    switch kernel from 1,025 tokens, ratio-2 layers from 2,050.
    {
        geometry g;
        auto old_kernel_switches = [&](int64_t prompt, int r) { return differing(g, false, prompt, 256, r, true); };
        expect(old_kernel_switches(1024, 0) == 0, "old rule: 1,024 tokens keep the ratio-1 kernel");
        expect(old_kernel_switches(1025, 0) > 0, "old rule: 1,025 tokens switch the ratio-1 kernel");
        expect(old_kernel_switches(2049, 1) == 0, "old rule: 2,049 tokens keep the ratio-2 kernel");
        expect(old_kernel_switches(2050, 1) > 0, "old rule: 2,050 tokens switch the ratio-2 kernel");
        expect(old_kernel_switches(4204, 1) > 0, "old rule: S8c's 4,204 tokens switch the ratio-2 kernel");
        // A chunk that ends past the 2,048-key gate keeps every micro-batch's
        // kernel (the bisect's chunk-2304 control at 4,200 tokens).
        expect(differing(g, false, 4200, 2304, 0, true) == 0 && differing(g, false, 4200, 2304, 1, true) == 0,
            "old rule: chunk 2,304 at 4,200 tokens keeps every kernel");
        // ubatch 1,024 widens the raw ring to 1,280 rows and moves the
        // ratio-2 threshold to 1,025/1,026 tokens (measured).
        geometry wide;
        wide.ubatch = 1024;
        expect(differing(wide, false, 1025, 1024, 1, true) == 0, "old rule, ubatch 1,024: 1,025 tokens keep the ratio-2 kernel");
        expect(differing(wide, false, 1026, 1024, 1, true) > 0, "old rule, ubatch 1,024: 1,026 tokens switch the ratio-2 kernel");
    }

    if (failures)
    {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("deepseek41 call-invariant attention width: %d configurations OK\n", checked);
    return 0;
}
