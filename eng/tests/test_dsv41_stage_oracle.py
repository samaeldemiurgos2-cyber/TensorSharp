#!/usr/bin/env python3
"""Unit tests for eng/validation/dsv41-stage-oracle.py on synthetic inputs (no GGUF, no traces, no GPU; numpy only).

    python -m unittest eng/tests/test_dsv41_stage_oracle.py      (from the repository root)
    python -I eng/tests/test_dsv41_stage_oracle.py -v

Each gate v2 rule has a test that a mutation of the input must FAIL (or be flagged) and the clean input must pass:
duplicate / invisible / pruned ids in a top-k list, the attention recompute from the exact selection, the two-sided
precision-model rule, the semantic discrimination self-test, derived routing tie bounds (a swap at a non-tie fails,
at an exact tie passes), per-segment gating with the FP8 bin-flip explanation, the Engram hash, the ratio-1 pin
assertion, and the layer-window state round trip.
"""
import importlib.util
import math
import os
from pathlib import Path
import tempfile
import types
import unittest

import numpy as np


def load_tool():
    path = Path(__file__).resolve().parents[1] / "validation" / "dsv41-stage-oracle.py"
    spec = importlib.util.spec_from_file_location("dsv41_stage_oracle", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


O = load_tool()


def fake_model(**kw):
    m = types.SimpleNamespace(nswa=8, hd=16, nh=2, nrot=4, used=6, nex=40, cand_topk=3, cand_block=4)
    m.__dict__.update(kw)
    return m


class Selections(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.scores = rng.normal(size=40)
        self.k = 8
        self.top = np.argsort(-self.scores, kind="stable")[:self.k]

    def test_clean_list_passes(self):
        prob, vis, uniq = O.selection_integrity(self.top, 40, self.k, self.scores, "cuda")
        self.assertEqual(prob, [])
        self.assertLessEqual(O.consistent(self.scores, set(uniq.tolist())), 0)

    def test_duplicate_in_place_of_boundary_row_fails(self):
        bad = self.top.copy()
        bad[-1] = bad[0]                         # the boundary row replaced by a duplicate
        # the S8b blind spot: as a set the list is still a top-|set|, and the count is still k
        self.assertLessEqual(O.consistent(self.scores, set(bad.tolist())), 0)
        self.assertEqual(len(bad), self.k)
        prob, _, _ = O.selection_integrity(bad, 40, self.k, self.scores, "cuda")
        self.assertTrue(any("duplicate" in p for p in prob), prob)
        # ggml-style list (visible ids among padding) with a duplicate fails too
        prob, _, _ = O.selection_integrity(bad, 40, self.k, self.scores, "ggml")
        self.assertTrue(prob)

    def test_invisible_and_count(self):
        bad = self.top.copy()
        bad[0] = 45
        prob, _, _ = O.selection_integrity(bad, 40, self.k, self.scores, "cuda")
        self.assertTrue(any("invisible" in p for p in prob), prob)
        prob, _, _ = O.selection_integrity(self.top[:-1], 40, self.k, self.scores, "cuda")
        self.assertTrue(any("count" in p for p in prob), prob)

    def test_ggml_padding_when_few_visible_is_fine(self):
        sc = self.scores[:5]
        ids = np.array([3, 1, 0, 4, 2, 17, 30, 22])      # 5 visible + 3 masked pad ids, top-8 over a bucket
        prob, vis, uniq = O.selection_integrity(ids, 5, self.k, sc, "ggml")
        self.assertEqual(prob, [])
        self.assertEqual(sorted(uniq.tolist()), [0, 1, 2, 3, 4])

    def test_pruned_row_selected_fails(self):
        sc = self.scores.copy()
        sc[self.top[0]] = -np.inf                 # pruned by the candidate mask, but still listed
        prob, _, _ = O.selection_integrity(self.top, 40, self.k, sc, "cuda")
        self.assertTrue(any("-inf" in p for p in prob), prob)

    def test_exact_selection_ties(self):
        sc = np.array([5.0, 4.0, 3.0, 2.0, 2.0, 1.0])
        self.assertEqual(O.exact_selection(sc, 4, {0, 1, 2, 4}).tolist(), [0, 1, 2, 4])   # tie: engine's choice kept
        self.assertEqual(O.exact_selection(sc, 4, {0, 1, 2, 5}).tolist(), [0, 1, 2, 3])   # non-tie: the exact pick

    def test_attention_from_exact_selection_sees_duplicate(self):
        rng = np.random.default_rng(2)
        m = fake_model()
        T, R = 30, 20
        q = rng.normal(size=(T, m.nh, m.hd))
        raw = rng.normal(size=(T, m.hd)).astype(np.float16).astype(np.float64)
        comp = rng.normal(size=(R, m.hd))
        sinks = rng.normal(size=m.nh)
        sc = {p: rng.normal(size=p + 1) for p in range(T) if p < R}
        exact = {p: np.sort(np.argsort(-s, kind="stable")[:4]) for p, s in sc.items()}
        lst = {p: v.copy() for p, v in exact.items()}
        p_bad = 15
        lst[p_bad] = np.array([exact[p_bad][0], exact[p_bad][0], exact[p_bad][1], exact[p_bad][2]])  # dup for row 3
        pos = np.arange(R)
        eng = O.attention_exact(m, 0, q[:R], raw, pos, comp, lambda p: lst.get(p), 1, sinks=sinks)
        own = O.attention_exact(m, 0, q[:R], raw, pos, comp, lambda p: lst.get(p), 1, sinks=sinks)
        ex = O.attention_exact(m, 0, q[:R], raw, pos, comp, lambda p: exact.get(p), 1, sinks=sinks)
        self.assertEqual(O.rel(eng, own), 0.0)                  # the S8b check (own list) passes
        self.assertGreater(O.rel(eng[[p_bad]], ex[[p_bad]]), 1e-3)  # the exact-selection check fails
        others = [p for p in pos if p != p_bad]
        self.assertEqual(O.rel(eng[others], ex[others]), 0.0)


class PinAndBlocks(unittest.TestCase):
    def test_ratio1_pin(self):
        bs, _ = O.block_scores(np.arange(20.0), 13, 4, 1)
        self.assertEqual(bs[3], np.inf)

    def test_ratio2_pin_refused(self):
        with self.assertRaises(SystemExit):
            O.block_scores(np.arange(20.0), 13, 4, 2)


class Routing(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.m = fake_model()
        self.T, self.K = 50, 256
        self.x = rng.normal(size=(self.T, self.K))
        self.W = rng.normal(size=(self.m.nex, self.K)) * 0.05
        self.bias = rng.normal(size=self.m.nex) * 0.1
        self.l = self.x @ self.W.T
        self.s = np.sqrt(O.softplus(self.l)) + self.bias
        self.sel = np.argsort(-self.s, axis=1, kind="stable")[:, :self.m.used]

    def test_clean(self):
        ck = O.routing_checks(self.m, 0, self.sel, self.l.astype(np.float32).astype(np.float64), self.x, self.W, self.bias)
        self.assertEqual((ck["own_violations"], ck["exact_violations"], ck["integrity_violations"]), ([], [], []))

    def test_swap_at_non_tie_fails(self):
        sel = self.sel.copy()
        order = np.argsort(-self.s, axis=1, kind="stable")
        sel[7, -1] = order[7, 20]                  # the 6th pick replaced by the 21st
        ck = O.routing_checks(self.m, 0, sel, self.l, self.x, self.W, self.bias)
        self.assertIn(7, [t for t, _ in ck["own_violations"]])
        self.assertIn(7, [t for t, _ in ck["exact_violations"]])

    def test_swap_at_exact_tie_passes(self):
        l = self.l.copy()
        order = np.argsort(-self.s, axis=1, kind="stable")
        a, b = order[9, 5], order[9, 6]
        bias = self.bias.copy()
        bias[b] = bias[a]
        l[9, b] = l[9, a]                          # experts a and b tie exactly at token 9
        sel = self.sel.copy()
        sel[9, list(sel[9]).index(a)] = b
        ck = O.routing_checks(self.m, 0, sel, l, self.x, self.W, bias)
        self.assertNotIn(9, [t for t, _ in ck["own_violations"]])

    def test_duplicate_pick_fails(self):
        sel = self.sel.copy()
        sel[4, 1] = sel[4, 0]
        ck = O.routing_checks(self.m, 0, sel, self.l, self.x, self.W, self.bias)
        self.assertIn(4, ck["integrity_violations"])

    def test_score_eval_bound_covers_f32_evaluation(self):
        rng = np.random.default_rng(4)
        l = np.concatenate([rng.normal(size=20000) * 8, np.linspace(-30, 30, 2001)]).astype(np.float32)
        b = rng.normal(size=len(l)).astype(np.float32) * 0.2
        e = np.exp(l)
        sp = np.where(l > 20, l, np.log(np.float32(1) + e)).astype(np.float32)
        s32 = (np.sqrt(sp) + b).astype(np.float32).astype(np.float64)
        s64 = np.sqrt(O.softplus(l.astype(np.float64))) + b.astype(np.float64)
        bound = O.score_eval_bound(l.astype(np.float64), b.astype(np.float64))
        self.assertTrue(np.all(np.abs(s32 - s64) <= bound), float(np.max(np.abs(s32 - s64) / bound)))


class Fp8Noise(unittest.TestCase):
    def test_flip_and_scale_change_are_noise_large_change_is_not(self):
        rng = np.random.default_rng(9)
        pre = rng.normal(size=(2, 64)).astype(np.float32).astype(np.float64)
        k_exact = O.dsv41_quant(pre, 0).astype(np.float64)
        # a bin flip: nudge one value across its rounding boundary
        eng = k_exact.copy()
        scale = O.quant_scale(pre, 0).reshape(2, -1)[0, 0]
        eng[0, 3] += O.e4m3_step(abs(k_exact[0, 3]) / scale) * scale * np.sign(k_exact[0, 3] or 1.0)
        # a group-scale doubling: the second row's first group re-quantized on a grid twice as coarse
        g = pre[1, :32].astype(np.float32)
        s2 = (O.quant_scale(g, 0) * 2).reshape(-1)[0]
        eng[1, :32] = (O.e4m3((g / s2).astype(np.float32)) * s2).astype(np.float64)
        pre_m = pre + rng.normal(size=pre.shape) * 1e-4
        n, beyond = O.fp8_flip_stats(eng, k_exact, pre, pre_m)
        self.assertGreater(int(n.sum()), 1)
        self.assertEqual(int(beyond.sum()), 0)
        eng[0, 10] = k_exact[0, 10] * 3 + scale * 8          # far beyond format noise
        _, beyond = O.fp8_flip_stats(eng, k_exact, pre, pre_m)
        self.assertEqual(int(beyond.sum()), 1)


class PrecisionModel(unittest.TestCase):
    def test_q8_1_grid(self):
        rng = np.random.default_rng(5)
        x = rng.normal(size=(6, 64)).astype(np.float32)
        for mmq in (False, True):
            y = O.q8_1(x, mmq, False).reshape(-1, 32)
            v = x.reshape(-1, 32).astype(np.float64)
            d = np.abs(v).max(axis=1, keepdims=True) / 127
            self.assertTrue(np.all(np.abs(y - v) <= d * 0.51 + 1e-6))

    def test_act_model_by_type_and_call(self):
        x = np.random.default_rng(6).normal(size=(4, 64))
        calls = np.array([1, 1, 256, 256])
        y = O.act_model(x, "BF16", calls)
        self.assertTrue(np.array_equal(y[:2], x[:2]))
        self.assertFalse(np.array_equal(y[2:], x[2:]))
        self.assertTrue(np.array_equal(O.act_model(x, "F32", calls), x))
        self.assertFalse(np.array_equal(O.act_model(x, "Q8_0", calls)[:2], x[:2]))


def report_of(stage, cuda, ggml, model=None, segs=None, extra_c=None, extra_g=None):
    rc = {"engine": "cuda", "layer": 0, "stage": stage, "rel_l2": cuda, "segments": dict(segs[0]) if segs else {}}
    rg = {"engine": "ggml", "layer": 0, "stage": stage, "rel_l2": ggml, "segments": dict(segs[1]) if segs else {}}
    if model is not None:
        rg["model_rel_l2"] = model
    rc.update(extra_c or {})
    rg.update(extra_g or {})
    return {"stages": [rc, rg]}


class Gate(unittest.TestCase):
    def gate(self, rep):
        return O.oracle_gate(rep, 1.25, 1e-6, 1e-6, 1.25)

    def test_two_sided_model(self):
        f, _, _ = self.gate(report_of("qr", 2.4e-3, 2.4e-3, 2.3e-3))
        self.assertEqual(f, [])
        f, _, _ = self.gate(report_of("qr", 2.4e-3, 2.4e-3 * 3, 2.3e-3))   # oracle drift: ggml far above its model
        self.assertTrue(any("two-sided" in x for x in f), f)

    def test_replay_rule(self):
        ok = report_of("raw_k", 1e-2, 1e-2, 1e-2, extra_g={"replay_rel_l2": 1e-4})
        self.assertEqual(self.gate(ok)[0], [])
        # a subtle oracle drift leaves ggml/model near 1 but breaks the replay
        drift = report_of("attn_out", 1e-2, 1.08e-2, 1e-2, extra_g={"replay_rel_l2": 5e-3})   # 0.46 x ggml > 0.4
        self.assertIn("L0 attn_out ggml replay", self.gate(drift)[0])

    def test_unreplayed_moe_types_defer_to_discrimination(self):
        types = ["IQ2_S", "IQ2_S", "IQ2_S", "Q5_K", "Q5_K", "Q5_K"]
        r = report_of("ffn_out (own sel)", 3e-3, 1.5e-2, 1.1e-2, extra_g={"moe_types": types, "replay_rel_l2": 1.2e-2})
        self.assertEqual(self.gate(r)[0], [])
        r = report_of("ffn_out (own sel)", 3e-3, 1.5e-2, 1.1e-2,
                      extra_g={"moe_types": ["Q8_0", "IQ2_XXS", "IQ3_XXS", "Q8_0", "Q8_0", "Q8_0"], "replay_rel_l2": 1e-2})
        self.assertTrue(self.gate(r)[0])

    def test_two_sided_f32(self):
        f, _, _ = self.gate(report_of("hidden", 1e-7, 3e-6))
        self.assertTrue(any("two-sided" in x for x in f), f)

    def test_per_segment(self):
        segs = ({"0:256": 2.4e-3, "256:257": 4.0e-3}, {"0:256": 2.4e-3, "256:257": 2.5e-3})
        f, _, _ = self.gate(report_of("qr", 2.4e-3, 2.4e-3, 2.4e-3, segs))
        self.assertIn("L0 qr [256:257]", f)

    def test_raw_k_segment_flag_and_fail(self):
        segs = ({"0:256": 1e-2, "256:257": 1.5e-2}, {"0:256": 1e-2, "256:257": 1.0e-2})
        flips = lambda nc, ng, multi: (
            {"flips_by_segment": {"0:256": 900, "256:257": nc}, "beyond_noise_by_segment": {"0:256": 0, "256:257": multi}},
            {"flips_by_segment": {"0:256": 900, "256:257": ng}})
        c, g_ = flips(3, 2, 0)
        f, fl, _ = self.gate(report_of("raw_k", 1e-2, 1e-2, 1e-2, segs, c, g_))
        self.assertEqual(f, [])
        self.assertEqual(fl, ["L0 raw_k [256:257]"])
        c, g_ = flips(40, 5, 0)                     # significant excess of flips
        f, _, _ = self.gate(report_of("raw_k", 1e-2, 1e-2, 1e-2, segs, c, g_))
        self.assertIn("L0 raw_k [256:257]", f)
        c, g_ = flips(3, 2, 1)                      # a difference beyond 2 grid steps is not FP8 noise
        f, _, _ = self.gate(report_of("raw_k", 1e-2, 1e-2, 1e-2, segs, c, g_))
        self.assertIn("L0 raw_k [256:257]", f)

    def test_binomial(self):
        self.assertAlmostEqual(O.binom_sf(0, 10, 0.5), 1.0)
        self.assertAlmostEqual(O.binom_sf(10, 10, 0.5), 0.5 ** 10)
        self.assertAlmostEqual(O.binom_sf(6, 10, 0.5), sum(math.comb(10, i) for i in range(6, 11)) / 1024)


class Discrimination(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.m = fake_model(nswa=8, hd=16, nh=2)
        T = 200
        self.T = T
        self.q = rng.normal(size=(T, 2, 16)) * 0.7
        self.raw = rng.normal(size=(T, 16))
        self.sinks = np.array([0.3, -0.2])
        self.pos = np.arange(T)
        self.proj = lambda core, p: core.reshape(len(core), -1)
        truth = O.attention_exact(self.m, 0, self.q, self.raw, self.pos, sinks=self.sinks)
        self.engine = truth + rng.normal(size=truth.shape) * 2e-3 * np.linalg.norm(truth) / math.sqrt(truth.size)

    def run_disc(self, sem):
        return O.discriminate(self.m, 0, 0, self.q, self.pos, self.raw, None, None, self.engine.reshape(self.T, -1),
                              self.proj, sem, sinks=self.sinks)

    def test_true_semantics_pass(self):
        res = self.run_disc(dict(O.BASE_SEM))
        self.assertTrue(all(v["ok"] for v in res), res)
        self.assertTrue(all(v["informative"] for v in res))

    def test_rescaling_variant_judged_on_distances(self):
        rng = np.random.default_rng(10)
        base = rng.normal(size=(50, 8))
        engine = base + rng.normal(size=base.shape) * 1e-2
        v = O.discrimination_verdict("weights unnormalized", engine, base, 2.5 * base)
        self.assertTrue(v["informative"] and v["ok"], v)
        self.assertAlmostEqual(v["measured_ratio"] / v["expected_ratio"], 1.0, delta=0.02)
        # the engine computes the variant: the baseline must lose
        v = O.discrimination_verdict("weights unnormalized", 2.5 * base + rng.normal(size=base.shape) * 1e-2, base, 2.5 * base)
        self.assertFalse(v["ok"])

    def test_window_drift_fails(self):
        for mut in ("window-1", "window+1", "sinks-off"):
            sem = dict(O.BASE_SEM, **O.MUTATIONS[mut])
            res = self.run_disc(sem)
            self.assertFalse(all(v["ok"] for v in res), (mut, res))


class Engram(unittest.TestCase):
    def test_hash_matches_reference_loop(self):
        en = O.Engram.__new__(O.Engram)
        en.ngram, en.nheads, en.pad = 3, 2, 5
        en.cols = (en.ngram - 1) * en.nheads
        en.token_map = np.array([7, 3, 9, 1, 0, 2, 8, 4, 6, 5])
        primes = np.array([101, 103, 107, 109], np.uint64)
        offs = np.concatenate([[0], np.cumsum(primes)[:-1]]).astype(np.uint64)
        mult = np.array([0x9E3779B97F4A7C15 | 1, 6364136223846793005, 1442695040888963407], np.uint64)
        en.layers = {1: {"mult": mult, "primes": primes, "offsets": offs, "first": 0, "table": "x"}}
        toks = [4, 1, 1, 9, 0, 3, 3, 7]
        got = en.hash_tokens(1, toks)
        hist = [int(en.token_map[t]) for t in toks]
        M = (1 << 64) - 1
        for p in range(len(toks)):
            rolling, blocked = 0, False
            for shift in range(en.ngram):
                blocked = blocked or p < shift
                tok = en.pad if blocked else hist[p - shift]
                rolling ^= (tok * int(mult[shift])) & M
                if shift:
                    for h in range(en.nheads):
                        col = (shift - 1) * en.nheads + h
                        self.assertEqual(int(got[p, col]), rolling % int(primes[col]) + int(offs[col]))


class State(unittest.TestCase):
    def test_round_trip(self):
        m = fake_model()
        rng = np.random.default_rng(8)
        T = 12
        xs = {e: rng.normal(size=(T, 4, 8)).astype(np.float32).astype(np.float64) for e in ("cuda", "ggml")}
        pre = {e: rng.normal(size=(T, 4)) for e in ("cuda", "ggml")}
        G = {e: {2: {"comp": rng.normal(size=(6, 8)), "lid": rng.normal(size=(6, 4))}} for e in ("cuda", "ggml")}
        SEL = {e: {2: {p: np.arange(p % 3 + 1) for p in range(T)}} for e in ("cuda", "ggml")}
        SELL = {e: {2: {p: np.arange(p % 3 + 1)[::-1] for p in range(T)}} for e in ("cuda", "ggml")}
        mk = {p: (rng.random(p + 1) > 0.5) for p in range(T)}
        CAND = {"cuda": {p: (set(), v) for p, v in mk.items()}, "ggml": {p: {0, 2} for p in range(T)}}
        CX = {e: {p: (rng.normal(size=4), None) for p in range(T)} for e in ("cuda", "ggml")}
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "s.npz")
            O.save_state(path, T, 5, xs, pre, G, SEL, SELL, CAND, CX)
            meta, xs2, pre2, G2, SEL2, SELL2, CAND2, CX2 = O.load_state(path, m)
        self.assertEqual(meta, {"T": T, "next_layer": 5})
        self.assertTrue(np.array_equal(xs2["cuda"], xs["cuda"]))
        self.assertTrue(np.array_equal(G2["ggml"][2]["lid"], G["ggml"][2]["lid"]))
        self.assertTrue(all(np.array_equal(SELL2["cuda"][2][p], SELL["cuda"][2][p]) for p in range(T)))
        self.assertTrue(all(np.array_equal(CAND2["cuda"][p][1], mk[p]) for p in range(T)))
        self.assertEqual(CAND2["ggml"][3], {0, 2})
        self.assertTrue(np.array_equal(CX2["ggml"][4][0], CX["ggml"][4][0]))


if __name__ == "__main__":
    unittest.main()
