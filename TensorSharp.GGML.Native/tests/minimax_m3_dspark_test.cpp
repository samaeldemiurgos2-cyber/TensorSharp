// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// MiniMax-M3 executor + DSpark drafter (ggml_ops_minimax_m3_dspark.inc) on
// eng/minimax-m3-fixture.py's small fixture (model.gguf + dspark.gguf, v3).
//
// The fixture's target follows a successor chain: after token t in
// 100000..104095 it predicts t + 1 by several logits (every 13th chain token
// skips to t + 2), and the drafter's Markov code drafts t + 1. So on a prompt
// that ends in 100000 the drafts are mostly accepted and greedy tokens are far
// from any near-tie. Checks:
//   1. speculative greedy decoding (draft, verify [anchor, drafts] with every
//      row's logits, accept the longest matching prefix, rewind the rest) emits
//      exactly the plain greedy tokens, with a high acceptance rate;
//   2. after that run (rejected verify rows, rewinds and draft blocks left in
//      the drafter's context ring), the drafter context of every position in
//      the window equals the one a slot that only ever decoded plainly holds,
//      and the next draft from both slots is the same;
//   3. a rewind to an earlier position, then a draft, matches a slot that
//      prefilled exactly that prefix (rows past the rewind are never context);
//   4. a slot copy drafts like its source; a batched decode step commits the
//      drafter context of every slot it advances;
//   5. a sequence longer than the context ring (cuda only): the window
//      survives the wrap, speculative == plain.
// cuda-multi (two GPUs, layer split 3+3, so the taps cross GPUs and prompts
// prefill pipelined): a pipelined prefill commits the same drafter context as
// a prefill fed in chunks too short to pipeline, both slots draft alike, and
// speculative greedy == plain greedy.
// cuda-multi4 (four GPUs, layer split 2+2+1+1, 64-row micro-batches): the
// same over four pipeline stages and 19 micro-batches, so the taps of layers
// 1 and 3 (stages 0 and 1) cross straight to the drafter's stage 3, past the
// stages between. A test hook makes that stage slow (large products in every
// commit), so the earlier stages run ahead of it: a tap's copy must wait for
// the drafter stage's read of its copy slot (n_copies micro-batches back), or
// the commit of an earlier micro-batch reads a later one's tapped rows. The
// same prefill with those waits dropped (the hook's other half) is reported as
// the control.
//
// Usage: GgmlOpsMiniMaxM3DsparkTest cpu|cuda|cuda-multi|cuda-multi4 [FIXTURE_DIR]
//   FIXTURE_DIR holds model.gguf and dspark.gguf [$TS_MM3_FIXTURE_DIR or
//   /workspace/mm3/fixtures/small]. Exit 77 when the backend or the fixture is
//   unavailable (the fixture is generated, not checked in: see its README).
//
// Real-model validation (not a ctest; the real checkpoint and its drafter):
//   GgmlOpsMiniMaxM3DsparkTest real MODEL DRAFT TOKENS_FILE [N] [N_GPU] [CUDA|CPU]
//   TOKENS_FILE: one prompt per line, space-separated token ids. Per prompt:
//   plain greedy decoding of N tokens [512] with each step's top-2 logits;
//   the same tokens verified in teacher-forced 8-row blocks (every verify row
//   against the one-token decode of that position: top-1 agreement, |dlogit|);
//   speculative greedy decoding with the drafter, its first difference from
//   plain greedy and the plain top-2 margin there (a divergence the verify
//   rows' rounding cannot explain is a failure); acceptance and ms per token.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <cstring>
#include <string>
#include <vector>

extern "C" {
int TSGgml_IsBackendAvailable(int);
int TSGgml_GetGpuDeviceCount(int);
const char * TSGgml_GetLastError();
void * TSGgml_MiniMaxM3LoadModelDraft(const char *, int, int, int, int, int, const char *, int, const char *, int, const char *);
void TSGgml_MiniMaxM3SetOption(const char *, const char *);
void TSGgml_MiniMaxM3Free(void *);
int TSGgml_MiniMaxM3VocabSize(void *);
int TSGgml_MiniMaxM3NPast(void *);
int TSGgml_MiniMaxM3Forward(void *, const int32_t *, int, float *, int);
int TSGgml_MiniMaxM3ForwardBatchedDecode(void *, int, const int32_t *, const int32_t *, const int32_t *, float *);
int TSGgml_MiniMaxM3Rewind(void *, int);
int TSGgml_MiniMaxM3SlotAlloc(void *);
int TSGgml_MiniMaxM3SetActiveSlot(void *, int);
int TSGgml_MiniMaxM3SlotFree(void *, int);
int TSGgml_MiniMaxM3SlotCopy(void *, int, int);
int TSGgml_MiniMaxM3DsparkBlockSize(void *);
int TSGgml_MiniMaxM3DsparkDraft(void *, int, int32_t *, float *);
int TSGgml_MiniMaxM3DsparkStats(void *, int64_t *, int);
int TSGgml_MiniMaxM3TestDsparkContext(void *, int, int, int64_t, float *, float *);
int TSGgml_MiniMaxM3GetStats(void *, int64_t *, int);
void TSGgml_MiniMaxM3TestDsparkPipeline(int, int);
}

namespace
{
constexpr int CHAIN_FIRST = 100000, CHAIN_LEN = 4096;
constexpr int BACKEND_CPU = 2, BACKEND_CUDA = 3;   // TSGgml backend ids
// Relative bound on the drafter context two feedings of the same tokens leave:
// rows written by different batch shapes round differently (amplified through
// the target layers); a row read at the wrong position or left stale differs by
// O(1).
constexpr double CONTEXT_BOUND = 0.1;

int g_fail = 0;

void check(bool ok, const char * what, const std::string & detail = "")
{
    printf("[%s] %s%s%s\n", ok ? "ok" : "FAIL", what, detail.empty() ? "" : ": ", detail.c_str());
    if (!ok) g_fail++;
}

struct rng
{
    uint64_t s;
    explicit rng(uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ull + 1) {}
    uint32_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return (uint32_t) (s >> 32); }
};

/// Random text tokens (none from the chain) followed by the chain's first id.
std::vector<int32_t> chain_prompt(uint64_t seed, int n_random)
{
    rng r(seed);
    std::vector<int32_t> t;
    while ((int) t.size() < n_random)
    {
        const int32_t v = (int32_t) (1000 + r.next() % 190000);
        if (v >= CHAIN_FIRST && v < CHAIN_FIRST + CHAIN_LEN) continue;
        t.push_back(v);
    }
    t.push_back(CHAIN_FIRST);
    return t;
}

int argmax(const float * row, int n)
{
    return (int) (std::max_element(row, row + n) - row);
}

struct ctx
{
    void * h = nullptr;
    int vocab = 0;
    int block = 0;
    std::vector<float> logits;     // one row
    std::vector<float> rows;       // verify rows
};

/// Prefill `toks` on the active slot; returns the greedy next token.
int prefill(ctx & c, const std::vector<int32_t> & toks)
{
    if (!TSGgml_MiniMaxM3Forward(c.h, toks.data(), (int) toks.size(), c.logits.data(), 0)) return -1;
    return argmax(c.logits.data(), c.vocab);
}

/// Plain greedy: `n` tokens after `first` (the token sampled from the prefill).
std::vector<int32_t> plain_greedy(ctx & c, int32_t first, int n)
{
    std::vector<int32_t> out;
    int32_t tok = first;
    for (int i = 0; i < n; i++)
    {
        out.push_back(tok);
        if ((int) out.size() == n) break;
        if (!TSGgml_MiniMaxM3Forward(c.h, &tok, 1, c.logits.data(), 0)) { out.clear(); return out; }
        tok = argmax(c.logits.data(), c.vocab);
    }
    return out;
}

struct spec_result
{
    std::vector<int32_t> tokens;
    int drafted = 0, accepted = 0, steps = 0;
    double ms = 0;
    // per draft position i: verifies whose drafts 0..i-1 were all accepted (position
    // i was judged), how many of those accepted draft i, and the confidence sum there
    std::vector<int> judged, taken;
    std::vector<double> conf_sum;
};

/// Speculative greedy decoding, exactly SpeculativeExecution's loop with a
/// block drafter and no confidence gate: `n` tokens after `first`.
spec_result spec_greedy(ctx & c, int32_t first, int n)
{
    spec_result r;
    const auto t0 = std::chrono::steady_clock::now();
    int32_t anchor = first;
    std::vector<int32_t> draft((size_t) c.block);
    std::vector<float> conf((size_t) c.block);
    while ((int) r.tokens.size() < n)
    {
        r.tokens.push_back(anchor);
        if ((int) r.tokens.size() == n) break;
        const int n_past = TSGgml_MiniMaxM3NPast(c.h);
        const int k = std::min(TSGgml_MiniMaxM3DsparkDraft(c.h, anchor, draft.data(), conf.data()), n - (int) r.tokens.size());
        r.steps++;
        if (k <= 0)
        {
            if (!TSGgml_MiniMaxM3Forward(c.h, &anchor, 1, c.logits.data(), 0)) { r.tokens.clear(); return r; }
            anchor = argmax(c.logits.data(), c.vocab);
            continue;
        }
        std::vector<int32_t> batch(1, anchor);
        batch.insert(batch.end(), draft.begin(), draft.begin() + k);
        c.rows.resize((size_t) (k + 1) * c.vocab);
        if (!TSGgml_MiniMaxM3Forward(c.h, batch.data(), k + 1, c.rows.data(), 1)) { r.tokens.clear(); return r; }
        int m = 0;
        int next = argmax(c.rows.data(), c.vocab);
        while (m < k && next == draft[(size_t) m])
        {
            r.tokens.push_back(next);
            m++;
            next = argmax(c.rows.data() + (size_t) m * c.vocab, c.vocab);
            if ((int) r.tokens.size() == n) break;
        }
        r.drafted += k;
        r.accepted += m;
        r.judged.resize((size_t) c.block); r.taken.resize((size_t) c.block); r.conf_sum.resize((size_t) c.block);
        for (int i = 0; i < k && i <= m; i++)
        {
            r.judged[(size_t) i]++;
            r.taken[(size_t) i] += i < m;
            r.conf_sum[(size_t) i] += conf[(size_t) i];
        }
        if (m < k && !TSGgml_MiniMaxM3Rewind(c.h, n_past + m + 1)) { r.tokens.clear(); return r; }
        anchor = next;
    }
    r.tokens.resize((size_t) std::min<int>((int) r.tokens.size(), n));
    r.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return r;
}

int first_difference(const std::vector<int32_t> & a, const std::vector<int32_t> & b)
{
    for (size_t i = 0; i < std::min(a.size(), b.size()); i++)
        if (a[i] != b[i]) return (int) i;
    return a.size() == b.size() ? -1 : (int) std::min(a.size(), b.size());
}

/// Largest |difference| between the drafter context of positions [p0, p1) in
/// two slots, relative to the largest |value|; -1 when a slot misses a row.
double context_diff(ctx & c, int slot_a, int slot_b, int64_t p0, int64_t p1, int n_layers, int kv)
{
    std::vector<float> ka((size_t) kv), va((size_t) kv), kb((size_t) kv), vb((size_t) kv);
    double dmax = 0, vmax = 1e-12;
    for (int il = 0; il < n_layers; il++)
        for (int64_t p = p0; p < p1; p++)
        {
            if (!TSGgml_MiniMaxM3TestDsparkContext(c.h, slot_a, il, p, ka.data(), va.data()) ||
                !TSGgml_MiniMaxM3TestDsparkContext(c.h, slot_b, il, p, kb.data(), vb.data()))
                return -1;
            for (int i = 0; i < kv; i++)
            {
                dmax = std::max({ dmax, (double) fabsf(ka[(size_t) i] - kb[(size_t) i]), (double) fabsf(va[(size_t) i] - vb[(size_t) i]) });
                vmax = std::max({ vmax, (double) fabsf(ka[(size_t) i]), (double) fabsf(va[(size_t) i]) });
            }
        }
    return dmax / vmax;
}

std::string fmt(const char * f, double a, double b = 0, double c = 0)
{
    char buf[256];
    snprintf(buf, sizeof(buf), f, a, b, c);
    return buf;
}

double ms_since(std::chrono::steady_clock::time_point t0)
{
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

struct top2
{
    int32_t id1 = -1, id2 = -1;
    float v1 = -INFINITY, v2 = -INFINITY;
};

top2 top_two(const float * row, int n)
{
    top2 t;
    for (int i = 0; i < n; i++)
    {
        if (row[i] > t.v1) { t.id2 = t.id1; t.v2 = t.v1; t.id1 = i; t.v1 = row[i]; }
        else if (row[i] > t.v2) { t.id2 = i; t.v2 = row[i]; }
    }
    return t;
}

/// The real checkpoint and its drafter (see the usage at the top). Returns 1
/// when a speculative run leaves plain greedy where the top-2 margin is wider
/// than twice the largest |dlogit| the teacher-forced verify rows showed.
int run_real(int argc, char ** argv)
{
    if (argc < 5)
    {
        printf("usage: %s real MODEL DRAFT TOKENS_FILE [N] [N_GPU] [CUDA|CPU]\n", argv[0]);
        return 2;
    }
    const char * model_path = argv[2];
    const char * draft_path = argv[3];
    const int N = argc > 5 ? atoi(argv[5]) : 512;
    const std::string backend = argc > 7 ? argv[7] : "CUDA";
    const int n_gpu = backend == "CPU" ? 0 : argc > 6 ? atoi(argv[6]) : std::max(1, TSGgml_GetGpuDeviceCount(BACKEND_CUDA));
    std::vector<std::vector<int32_t>> prompts;
    {
        std::ifstream in(argv[4]);
        std::string line;
        while (std::getline(in, line))
        {
            std::istringstream ss(line);
            std::vector<int32_t> p;
            for (long v; ss >> v; ) p.push_back((int32_t) v);
            if (!p.empty()) prompts.push_back(p);
        }
    }
    if (prompts.empty()) { printf("FAIL: no prompts in %s\n", argv[4]); return 1; }
    size_t longest = 0;
    for (const auto & p : prompts) longest = std::max(longest, p.size());
    const int n_ctx = (int) (((longest + (size_t) N + 64 + 1023) / 1024) * 1024);
    const auto t_load = std::chrono::steady_clock::now();
    ctx c;
    c.h = TSGgml_MiniMaxM3LoadModelDraft(model_path, n_gpu, n_ctx, 2048, 0, 0, backend.c_str(), 1, nullptr, 0, draft_path);
    if (!c.h) { printf("FAIL load: %s\n", TSGgml_GetLastError()); return 1; }
    c.vocab = TSGgml_MiniMaxM3VocabSize(c.h);
    c.block = TSGgml_MiniMaxM3DsparkBlockSize(c.h);
    c.logits.resize((size_t) c.vocab);
    printf("loaded %s + %s on %d %s device(s) in %.1f s: ctx %d, block %d, %zu prompt(s), %d new tokens each\n", model_path,
           draft_path, std::max(1, n_gpu), backend.c_str(), ms_since(t_load) / 1000.0, n_ctx, c.block, prompts.size(), N);
    const int spec_slot = TSGgml_MiniMaxM3SlotAlloc(c.h);
    if (spec_slot < 0 || c.block < 1) { printf("FAIL: no second slot or no drafter\n"); return 1; }

    int failures = 0;
    double sum_plain_ms = 0, sum_spec_ms = 0;
    int sum_plain_n = 0, sum_spec_n = 0, sum_drafted = 0, sum_accepted = 0;
    std::vector<int32_t> last_plain;
    for (size_t pi = 0; pi < prompts.size(); pi++)
    {
        const std::vector<int32_t> & prompt = prompts[pi];
        // ---- plain greedy, with each step's top two --------------------------
        TSGgml_MiniMaxM3SetActiveSlot(c.h, 0);
        TSGgml_MiniMaxM3Rewind(c.h, 0);
        if (prefill(c, prompt) < 0) { printf("FAIL: prefill\n"); return 1; }
        std::vector<top2> rec;
        std::vector<int32_t> plain;
        const auto t_plain = std::chrono::steady_clock::now();
        for (int i = 0; i < N; i++)
        {
            rec.push_back(top_two(c.logits.data(), c.vocab));
            plain.push_back(rec.back().id1);
            if (i + 1 < N && !TSGgml_MiniMaxM3Forward(c.h, &plain.back(), 1, c.logits.data(), 0)) { printf("FAIL: decode\n"); return 1; }
        }
        const double plain_ms = ms_since(t_plain);

        // ---- the same tokens verified in teacher-forced blocks ---------------
        TSGgml_MiniMaxM3SetActiveSlot(c.h, spec_slot);
        TSGgml_MiniMaxM3Rewind(c.h, 0);
        const int first_tf = prefill(c, prompt);
        int tf_rows = 0, tf_agree = 0, tf_max_at = -1;
        double tf_max = 0, tf_sum = 0, tf_flip_margin = 0;
        std::vector<double> tf_all;
        for (int i = 0; i + 1 < N; )
        {
            const int k = std::min(c.block, N - 2 - i);   // rows i..i+k predict i+1..i+k+1
            c.rows.resize((size_t) (k + 1) * c.vocab);
            if (!TSGgml_MiniMaxM3Forward(c.h, plain.data() + i, k + 1, c.rows.data(), 1)) { printf("FAIL: verify\n"); return 1; }
            for (int j = 0; j <= k; j++)
            {
                const top2 & want = rec[(size_t) (i + j + 1)];
                const float * row = c.rows.data() + (size_t) j * c.vocab;
                const double d = fabs((double) row[want.id1] - want.v1);
                if (d > tf_max) { tf_max = d; tf_max_at = (int) prompt.size() + i + j; }
                tf_sum += d;
                tf_all.push_back(d);
                tf_rows++;
                if (argmax(row, c.vocab) == want.id1) tf_agree++;
                else tf_flip_margin = std::max(tf_flip_margin, (double) (want.v1 - want.v2));
            }
            i += k + 1;
        }

        // ---- speculative greedy with the drafter -----------------------------
        TSGgml_MiniMaxM3Rewind(c.h, 0);
        const int first_s = prefill(c, prompt);
        const spec_result sp = spec_greedy(c, first_s, N);
        const int d = first_difference(plain, sp.tokens);
        std::string where = "none";
        if (d >= 0 && d < (int) rec.size())
        {
            const top2 & r = rec[(size_t) d];
            const double margin = r.v1 - r.v2;
            const bool second = sp.tokens.size() > (size_t) d && sp.tokens[(size_t) d] == r.id2;
            where = fmt("token %.0f, plain top-2 margin %.4f", d, margin) + (second ? ", speculative took the plain runner-up" : ", speculative took another token");
            if (margin > 2.0 * tf_max) failures++;
        }
        std::sort(tf_all.begin(), tf_all.end());
        const double tf_p99 = tf_all.empty() ? 0.0 : tf_all[(size_t) ((tf_all.size() - 1) * 0.99)];
        printf("prompt %zu (%zu tokens): first token %s; teacher-forced verify rows: top-1 %d/%d, |dlogit| mean %.4f p99 %.4f "
               "max %.4f (query position %d)%s\n",
               pi, prompt.size(), first_tf == plain[0] && first_s == plain[0] ? "identical" : "DIFFERS", tf_agree, tf_rows,
               tf_rows ? tf_sum / tf_rows : 0.0, tf_p99, tf_max, tf_max_at,
               tf_agree < tf_rows ? fmt(", widest flipped plain margin %.4f", tf_flip_margin).c_str() : "");
        printf("  speculative vs plain: first difference at %s\n", where.c_str());
        std::string cal;
        for (size_t i = 0; i < sp.judged.size(); i++)
            if (sp.judged[i])
                cal += fmt(" %.0f:%.2f/%.2f", (double) i, (double) sp.taken[i] / sp.judged[i], sp.conf_sum[i] / sp.judged[i]);
        printf("  draft position: accepted given the prefix / mean confidence:%s\n", cal.c_str());
        std::string ids;
        for (int32_t t : sp.tokens) ids += " " + std::to_string(t);
        printf("  speculative tokens:%s\n", ids.c_str());
        printf("  plain %.2f ms/token (%.2f tok/s); speculative %.2f ms/token (%.2f tok/s), %.2fx; drafted %d accepted %d (%.1f %%), "
               "%d steps, %.2f tokens/step\n", plain_ms / std::max(1, N - 1), 1000.0 * (N - 1) / plain_ms,
               sp.ms / std::max<size_t>(1, sp.tokens.size() - 1), 1000.0 * (double) (sp.tokens.size() - 1) / sp.ms,
               (plain_ms / std::max(1, N - 1)) / (sp.ms / std::max<size_t>(1, sp.tokens.size() - 1)), sp.drafted, sp.accepted,
               sp.drafted ? 100.0 * sp.accepted / sp.drafted : 0.0, sp.steps, sp.steps ? (double) sp.tokens.size() / sp.steps : 0.0);
        fflush(stdout);
        sum_plain_ms += plain_ms; sum_plain_n += N - 1;
        sum_spec_ms += sp.ms; sum_spec_n += (int) sp.tokens.size() - 1;
        sum_drafted += sp.drafted; sum_accepted += sp.accepted;
        last_plain = plain;
    }
    // ---- what a verify of r rows costs (the active slot holds the last sequence) ----
    {
        const int base = TSGgml_MiniMaxM3NPast(c.h);
        std::string curve;
        for (int r = 1; r <= c.block + 1 && r <= (int) last_plain.size(); r++)
        {
            c.rows.resize((size_t) r * c.vocab);
            double ms = 0;
            int timed = 0;
            for (int rep = 0; rep < 8; rep++)
            {
                const auto t0 = std::chrono::steady_clock::now();
                if (!TSGgml_MiniMaxM3Forward(c.h, last_plain.data(), r, c.rows.data(), 1)) { printf("FAIL: timed verify\n"); return 1; }
                const double t = ms_since(t0);
                TSGgml_MiniMaxM3Rewind(c.h, base);
                if (rep >= 2) { ms += t; timed++; }   // the first calls of a shape build its graph
            }
            curve += fmt(" %.0f:%.1f", r, ms / timed);
        }
        printf("verify ms by rows (at position %d):%s\n", base, curve.c_str());
    }
    int64_t st[9] = {};
    TSGgml_MiniMaxM3DsparkStats(c.h, st, 9);
    printf("summary: plain %.2f tok/s, speculative %.2f tok/s (%.2fx), drafted %d accepted %d (%.1f %%), %.2f ms per draft, "
           "%d unexplained divergence(s)\n", 1000.0 * sum_plain_n / sum_plain_ms, 1000.0 * sum_spec_n / sum_spec_ms,
           (sum_plain_ms / sum_plain_n) / (sum_spec_ms / sum_spec_n), sum_drafted, sum_accepted,
           sum_drafted ? 100.0 * sum_accepted / sum_drafted : 0.0, st[2] ? st[3] / 1000.0 / st[2] : 0.0, failures);
    TSGgml_MiniMaxM3Free(c.h);
    return failures ? 1 : 0;
}

/// cuda-multi / cuda-multi4: the fixture over two or four GPUs (see the header).
int run_multi(const std::string & model_path, const std::string & draft_path, int n_gpu)
{
    if (!TSGgml_IsBackendAvailable(BACKEND_CUDA) || TSGgml_GetGpuDeviceCount(BACKEND_CUDA) < n_gpu)
    {
        printf("skip: needs %d CUDA GPUs\n", n_gpu);
        return 77;
    }
    // Two GPUs: adaptive micro-batches (256..512 rows), checked against 200-token calls, each at or
    // below the 256-row pipelining floor. Four: fixed 64-row micro-batches (19 of them in one call, far
    // more than the 4 input copies), checked against 64-token calls (one micro-batch: unpipelined).
    const bool four = n_gpu == 4;
    if (four) TSGgml_MiniMaxM3SetOption("TS_MINIMAX_M3_PIPELINE_UBATCH", "64");
    ctx c;
    c.h = TSGgml_MiniMaxM3LoadModelDraft(model_path.c_str(), n_gpu, 4096, 512, 0, 0, "CUDA", 1, four ? "2,2,1,1" : "3,3", 0,
                                         draft_path.c_str());
    TSGgml_MiniMaxM3SetOption("TS_MINIMAX_M3_PIPELINE_UBATCH", nullptr);
    if (!c.h) { printf("FAIL load: %s\n", TSGgml_GetLastError()); return 1; }
    c.vocab = TSGgml_MiniMaxM3VocabSize(c.h);
    c.block = TSGgml_MiniMaxM3DsparkBlockSize(c.h);
    c.logits.resize((size_t) c.vocab);
    int64_t es[16] = {}, ds[9] = {};
    TSGgml_MiniMaxM3GetStats(c.h, es, 16);
    TSGgml_MiniMaxM3DsparkStats(c.h, ds, 9);
    check(c.block == 7 && es[4] == n_gpu && es[10] == n_gpu, "GPUs, pipelined prefill on every one, drafter loaded",
          fmt("%.0f GPUs, %.0f pipeline stages, drafter on device %.0f", (double) es[4], (double) es[10], (double) ds[5]));
    const int n_layers = (int) ds[7], kv = 4 * 128, N = 160;
    const std::vector<int32_t> prompt = chain_prompt(21, 1200);
    const size_t piece = four ? 64 : 200;

    // slot 0: one call (pipelined); slot 1: calls too short to pipeline
    TSGgml_MiniMaxM3SetActiveSlot(c.h, 0);
    TSGgml_MiniMaxM3Rewind(c.h, 0);
    const int64_t calls0 = es[6], ubatches0 = es[7];
    if (four) TSGgml_MiniMaxM3TestDsparkPipeline(2, 0);   // a slow drafter stage for the pipelined call
    const int first = prefill(c, prompt);
    if (four) TSGgml_MiniMaxM3TestDsparkPipeline(0, 0);
    TSGgml_MiniMaxM3GetStats(c.h, es, 16);
    const int64_t calls1 = es[6];
    if (four)
        check(es[7] - ubatches0 == (int64_t) ((prompt.size() + 63) / 64), "the one call ran as 64-row micro-batches",
              fmt("%.0f micro-batches", (double) (es[7] - ubatches0)));
    const int chunked = TSGgml_MiniMaxM3SlotAlloc(c.h);
    TSGgml_MiniMaxM3SetActiveSlot(c.h, chunked);
    int first_c = -1;
    for (size_t at = 0; at < prompt.size(); at += piece)
    {
        const std::vector<int32_t> part(prompt.begin() + (long) at, prompt.begin() + (long) std::min(prompt.size(), at + piece));
        first_c = prefill(c, part);
    }
    TSGgml_MiniMaxM3GetStats(c.h, es, 16);
    check(calls1 == calls0 + 1 && es[6] == calls1, "the one-call prefill ran pipelined, the chunked one did not",
          fmt("pipelined calls %.0f -> %.0f -> %.0f", (double) calls0, (double) calls1, (double) es[6]));
    const double d = context_diff(c, 0, chunked, 0, (int64_t) prompt.size(), n_layers, kv);
    check(first == first_c && d >= 0 && d < CONTEXT_BOUND, "a pipelined prefill commits the drafter context of a chunked one",
          fmt("max relative |difference| %.2e", d));
    std::vector<int32_t> da((size_t) c.block), db((size_t) c.block);
    std::vector<float> ca((size_t) c.block), cb((size_t) c.block);
    TSGgml_MiniMaxM3SetActiveSlot(c.h, 0);
    const int na = TSGgml_MiniMaxM3DsparkDraft(c.h, first, da.data(), ca.data());
    TSGgml_MiniMaxM3SetActiveSlot(c.h, chunked);
    const int nb = TSGgml_MiniMaxM3DsparkDraft(c.h, first, db.data(), cb.data());
    check(na == c.block && nb == c.block && da == db, "both slots draft the same block");

    // plain greedy on slot 0, speculative greedy on the chunked slot
    TSGgml_MiniMaxM3SetActiveSlot(c.h, 0);
    const std::vector<int32_t> plain = plain_greedy(c, first, N);
    TSGgml_MiniMaxM3SetActiveSlot(c.h, chunked);
    const spec_result sp = spec_greedy(c, first_c, N);
    const int diff = first_difference(plain, sp.tokens);
    const double acc = sp.drafted ? (double) sp.accepted / sp.drafted : 0.0;
    check(diff < 0 && (int) plain.size() == N, "speculative greedy == plain greedy over the GPUs",
          fmt("first difference %.0f, %.1f %% of %.0f drafts accepted", diff, 100.0 * acc, sp.drafted));
    check(acc > 0.5, "chain drafts are mostly accepted", fmt("%.1f %% accepted", 100.0 * acc));
    if (four)
    {
        // Control: the same one-call prefill without the tap waits. Not a check: a race needs the timing
        // to cooperate (on 4x A40 it never differed, though a peer copy there does not wait for the
        // destination GPU's work), but on a host where it differs it shows what the waits prevent.
        TSGgml_MiniMaxM3TestDsparkPipeline(2, 1);
        const int unguarded = TSGgml_MiniMaxM3SlotAlloc(c.h);
        TSGgml_MiniMaxM3SetActiveSlot(c.h, unguarded);
        prefill(c, prompt);
        const double du = context_diff(c, unguarded, chunked, 0, (int64_t) prompt.size(), n_layers, kv);
        TSGgml_MiniMaxM3TestDsparkPipeline(0, 0);
        printf("[note] control without the tap waits: max relative |difference| %.2e (%s)\n", du,
               du > CONTEXT_BOUND ? "the race the waits close" : "the race did not show this time");
        TSGgml_MiniMaxM3SetActiveSlot(c.h, 0);
        TSGgml_MiniMaxM3SlotFree(c.h, unguarded);
    }
    printf("summary: %s, drafted %d accepted %d (%.1f %%), %d failures\n", four ? "cuda-multi4" : "cuda-multi", sp.drafted,
           sp.accepted, 100.0 * acc, g_fail);
    TSGgml_MiniMaxM3Free(c.h);
    return g_fail ? 1 : 0;
}

} // namespace

int main(int argc, char ** argv)
{
    const std::string mode = argc > 1 ? argv[1] : "cpu";
    if (mode == "real") return run_real(argc, argv);
    const char * env_dir = getenv("TS_MM3_FIXTURE_DIR");
    const std::string dir = argc > 2 ? argv[2] : env_dir && *env_dir ? env_dir : "/workspace/mm3/fixtures/small";
    const std::string model_path = dir + "/model.gguf", draft_path = dir + "/dspark.gguf";
    for (const std::string & p : { model_path, draft_path })
        if (FILE * f = fopen(p.c_str(), "rb")) fclose(f);
        else { printf("skip: fixture %s not found (eng/minimax-m3-fixture.py writes it)\n", p.c_str()); return 77; }
    if (mode == "cuda-multi" || mode == "cuda-multi4") return run_multi(model_path, draft_path, mode == "cuda-multi" ? 2 : 4);
    const bool cuda = mode == "cuda";
    if (cuda ? !TSGgml_IsBackendAvailable(BACKEND_CUDA) || TSGgml_GetGpuDeviceCount(BACKEND_CUDA) < 1
             : !TSGgml_IsBackendAvailable(BACKEND_CPU))
    {
        printf("skip: %s backend unavailable\n", mode.c_str());
        return 77;
    }
    const int n_ctx = cuda ? 12288 : 2048;
    ctx c;
    c.h = TSGgml_MiniMaxM3LoadModelDraft(model_path.c_str(), cuda ? 1 : 0, n_ctx, 512, 0, 0, cuda ? "CUDA" : "CPU", 1,
                                         nullptr, 0, draft_path.c_str());
    if (!c.h) { printf("FAIL load: %s\n", TSGgml_GetLastError()); return 1; }
    c.vocab = TSGgml_MiniMaxM3VocabSize(c.h);
    c.block = TSGgml_MiniMaxM3DsparkBlockSize(c.h);
    c.logits.resize((size_t) c.vocab);
    int64_t st[9] = {};
    TSGgml_MiniMaxM3DsparkStats(c.h, st, 9);
    check(c.block == 7, "drafter loaded with block 7", fmt("block %.0f, ring %.0f rows, %.0f draft layers", c.block, (double) st[1], (double) st[7]));
    const int n_layers = (int) st[7], kv = 4 * 128;
    const int N = cuda ? 240 : 72;

    // ---- 1. speculative greedy == plain greedy on chain prompts --------------
    int total_drafted = 0, total_accepted = 0;
    int slot_spec = -1;
    std::vector<int32_t> spec_tokens, prompt_used;
    for (int pi = 0; pi < 2; pi++)
    {
        const std::vector<int32_t> prompt = chain_prompt(11 + (uint64_t) pi, pi == 0 ? 120 : 600);
        TSGgml_MiniMaxM3SetActiveSlot(c.h, 0);
        TSGgml_MiniMaxM3Rewind(c.h, 0);
        const int first = prefill(c, prompt);
        const std::vector<int32_t> plain = plain_greedy(c, first, N);
        if (slot_spec < 0) slot_spec = TSGgml_MiniMaxM3SlotAlloc(c.h);
        TSGgml_MiniMaxM3SetActiveSlot(c.h, slot_spec);
        TSGgml_MiniMaxM3Rewind(c.h, 0);
        const int first_s = prefill(c, prompt);
        const spec_result sp = spec_greedy(c, first_s, N);
        const int diff = first_difference(plain, sp.tokens);
        int chain_tokens = 0;
        for (int32_t t : plain) chain_tokens += t >= CHAIN_FIRST && t < CHAIN_FIRST + CHAIN_LEN;
        check(first == first_s && diff < 0 && (int) plain.size() == N,
              pi == 0 ? "speculative greedy == plain greedy (120-token prompt)" : "speculative greedy == plain greedy (600-token prompt)",
              fmt("first difference %.0f, %.0f chain tokens of the plain run", diff, chain_tokens));
        const double acc = sp.drafted ? (double) sp.accepted / sp.drafted : 0.0;
        printf("     drafted %d accepted %d (%.1f %%), %d steps, %.2f tokens/step, %.1f ms\n", sp.drafted, sp.accepted,
               100.0 * acc, sp.steps, sp.steps ? (double) sp.tokens.size() / sp.steps : 0.0, sp.ms);
        check(acc > 0.5, "chain drafts are mostly accepted", fmt("%.1f %% accepted", 100.0 * acc));
        total_drafted += sp.drafted;
        total_accepted += sp.accepted;
        spec_tokens = sp.tokens;
        prompt_used = prompt;
    }

    // ---- 2. the drafter context after a speculative run ---------------------
    {
        // Slot 0 decoded plainly and slot_spec speculatively over the same tokens:
        // slot 0 holds prompt + plain[0..N-2]; slot_spec holds that too, or also
        // the last token when the run ended inside an accepted draft. Both go to
        // the plain slot's head, whose next token is the last one emitted.
        const int P = (int) prompt_used.size() + N - 1;
        TSGgml_MiniMaxM3SetActiveSlot(c.h, 0);
        const int np0 = TSGgml_MiniMaxM3NPast(c.h);
        TSGgml_MiniMaxM3SetActiveSlot(c.h, slot_spec);
        const int np1 = TSGgml_MiniMaxM3NPast(c.h);
        const bool rw = TSGgml_MiniMaxM3Rewind(c.h, P) != 0;
        check(np0 == P && (np1 == P || np1 == P + 1) && rw, "speculative and plain slots end at the same position",
              fmt("%.0f vs %.0f", np0, np1));
        const double d = context_diff(c, 0, slot_spec, 0, P, n_layers, kv);
        check(d >= 0 && d < CONTEXT_BOUND, "drafter context after speculation == after plain decoding (every position)",
              fmt("max relative |difference| %.2e", d));
        std::vector<int32_t> da((size_t) c.block), db((size_t) c.block);
        std::vector<float> ca((size_t) c.block), cb((size_t) c.block);
        const int32_t anchor = spec_tokens.back();
        TSGgml_MiniMaxM3SetActiveSlot(c.h, 0);
        const int na = TSGgml_MiniMaxM3DsparkDraft(c.h, anchor, da.data(), ca.data());
        TSGgml_MiniMaxM3SetActiveSlot(c.h, slot_spec);
        const int nb = TSGgml_MiniMaxM3DsparkDraft(c.h, anchor, db.data(), cb.data());
        double cd = 0;
        for (int i = 0; i < c.block; i++) cd = std::max(cd, (double) fabsf(ca[(size_t) i] - cb[(size_t) i]));
        check(na == c.block && nb == c.block && da == db, "next draft from both slots is the same",
              fmt("confidence max |difference| %.2e", cd));
    }

    // ---- 3. rewind, then draft == a fresh prefill of the prefix --------------
    {
        std::vector<int32_t> seq = prompt_used;
        seq.insert(seq.end(), spec_tokens.begin(), spec_tokens.end() - 1);
        const int keep = (int) prompt_used.size() + N / 3;
        TSGgml_MiniMaxM3SetActiveSlot(c.h, slot_spec);
        const bool rw = TSGgml_MiniMaxM3Rewind(c.h, keep) != 0;
        std::vector<int32_t> da((size_t) c.block), db((size_t) c.block);
        std::vector<float> ca((size_t) c.block), cb((size_t) c.block);
        const int na = TSGgml_MiniMaxM3DsparkDraft(c.h, seq[(size_t) keep], da.data(), ca.data());
        const int fresh = TSGgml_MiniMaxM3SlotAlloc(c.h);
        TSGgml_MiniMaxM3SetActiveSlot(c.h, fresh);
        std::vector<int32_t> prefix(seq.begin(), seq.begin() + keep);
        prefill(c, prefix);
        const int nb = TSGgml_MiniMaxM3DsparkDraft(c.h, seq[(size_t) keep], db.data(), cb.data());
        const double d = context_diff(c, slot_spec, fresh, std::max(0, keep - 4096), keep, n_layers, kv);
        check(rw && na == c.block && nb == c.block && da == db, "draft after a rewind == draft after a fresh prefill",
              fmt("context max relative |difference| %.2e", d));
        // ---- 4. slot copy, batched decode ------------------------------------
        const int copy = TSGgml_MiniMaxM3SlotAlloc(c.h);
        const bool copied = TSGgml_MiniMaxM3SlotCopy(c.h, fresh, copy) != 0;
        std::vector<int32_t> dc((size_t) c.block);
        std::vector<float> cc((size_t) c.block);
        TSGgml_MiniMaxM3SetActiveSlot(c.h, copy);
        const int nc = TSGgml_MiniMaxM3DsparkDraft(c.h, seq[(size_t) keep], dc.data(), cc.data());
        check(copied && nc == c.block && dc == db, "a slot copy drafts like its source");
        // one batched decode step over `fresh` and `copy` with the same token, then
        // both draft from the next anchor like a slot advanced by a single forward
        const int32_t slots[2] = { fresh, copy };
        const int32_t toks[2] = { seq[(size_t) keep], seq[(size_t) keep] };
        const int32_t poss[2] = { keep, keep };
        std::vector<float> bl((size_t) 2 * c.vocab);
        const bool bd = TSGgml_MiniMaxM3ForwardBatchedDecode(c.h, 2, slots, toks, poss, bl.data()) != 0;
        TSGgml_MiniMaxM3SetActiveSlot(c.h, slot_spec);
        TSGgml_MiniMaxM3Rewind(c.h, keep);
        TSGgml_MiniMaxM3Forward(c.h, &seq[(size_t) keep], 1, c.logits.data(), 0);
        const double d1 = context_diff(c, fresh, slot_spec, std::max(0, keep + 1 - 4096), keep + 1, n_layers, kv);
        const double d2 = context_diff(c, copy, slot_spec, std::max(0, keep + 1 - 4096), keep + 1, n_layers, kv);
        check(bd && d1 >= 0 && d1 < CONTEXT_BOUND && d2 >= 0 && d2 < CONTEXT_BOUND, "a batched decode step commits every slot's drafter context",
              fmt("max relative |difference| %.2e / %.2e", d1, d2));
        TSGgml_MiniMaxM3SetActiveSlot(c.h, 0);
        TSGgml_MiniMaxM3SlotFree(c.h, copy);
        TSGgml_MiniMaxM3SlotFree(c.h, fresh);
    }

    // ---- 5. past the context ring ------------------------------------------
    if (cuda)
    {
        const int64_t rows = st[1];
        const std::vector<int32_t> prompt = chain_prompt(99, (int) rows + 700);
        TSGgml_MiniMaxM3SetActiveSlot(c.h, 0);
        TSGgml_MiniMaxM3Rewind(c.h, 0);
        const int first = prefill(c, prompt);
        const std::vector<int32_t> plain = plain_greedy(c, first, 120);
        TSGgml_MiniMaxM3SetActiveSlot(c.h, slot_spec);
        TSGgml_MiniMaxM3Rewind(c.h, 0);
        const int first_s = prefill(c, prompt);
        const spec_result sp = spec_greedy(c, first_s, 120);
        const int diff = first_difference(plain, sp.tokens);
        const double acc = sp.drafted ? (double) sp.accepted / sp.drafted : 0.0;
        check(first == first_s && diff < 0, "speculative greedy == plain greedy past the context ring",
              fmt("%.0f-token prompt, first difference %.0f, %.1f %% accepted", (double) prompt.size(), diff, 100.0 * acc));
        TSGgml_MiniMaxM3SetActiveSlot(c.h, 0);
        const int np = TSGgml_MiniMaxM3NPast(c.h);
        const double d = context_diff(c, 0, slot_spec, np - 4096, np, n_layers, kv);
        check(d >= 0 && d < CONTEXT_BOUND, "the window survives the ring wrap", fmt("max relative |difference| %.2e", d));
        total_drafted += sp.drafted;
        total_accepted += sp.accepted;
    }

    TSGgml_MiniMaxM3DsparkStats(c.h, st, 9);
    printf("summary: %s, drafted %d accepted %d (%.1f %%), %lld drafts in %.1f ms (%.2f ms each), %d failures\n",
           mode.c_str(), total_drafted, total_accepted, total_drafted ? 100.0 * total_accepted / total_drafted : 0.0,
           (long long) st[2], st[3] / 1000.0, st[2] ? st[3] / 1000.0 / st[2] : 0.0, g_fail);
    TSGgml_MiniMaxM3Free(c.h);
    return g_fail ? 1 : 0;
}
