#!/usr/bin/env python3
"""Teacher-forced perplexity and in-run consistency of DeepSeek41DistributedProbe output directories.

usage: python3 eng/validation/dsv41-probe-ppl.py PROBE_DIR [PROBE_DIR ...]

PROBE_DIR is a probe's output directory (probe.json plus its logit files), from a run with --text and a
large --decode-count. Prints one JSON line per directory. Needs numpy; reads artifacts only, loads no model.

Targets: decode step j (input token P+j, where P = tokens - decode steps) predicts token P+1+j, so
--decode-count 128 scores 127 targets. The refill phase is scored the same way. The prefill vector's
prediction of token P is reported separately (prefill_nll_first_target, reset-prefill_nll_first_target).
Consistency: prefill vs reset-prefill, prefill vs the last chunk, and decode j vs refill j (max |dlogit|
and argmax agreement). A perplexity below 1, above the vocabulary size or not finite is flagged as absurd.

For a paired, gated comparison of two runs (KL, top-1, PPL ratio) use compare-distributed-quality.py.
This is a single-run check, written for the DeepSeek V4.1 1017 direct-CUDA S8 stage (per-text PPL of the
truncated "1017-lite" runs and of the full-model references).
"""
import json, os, sys
import numpy as np


def lsm(x):
    x = x.astype(np.float64)
    m = x.max()
    return x - m - np.log(np.exp(x - m).sum())


def load(d, s):
    return np.fromfile(os.path.join(d, os.path.basename(s["logits"])), np.float32)


for d in sys.argv[1:]:
    p = json.load(open(os.path.join(d, "probe.json")))
    toks = p["tokens"]
    steps = [s for s in p["steps"] if s.get("logits")]
    by = {}
    for s in steps:
        by.setdefault(s["operation"], []).append(s)
    dec, ref = by.get("decode", []), by.get("refill", [])
    P = len(toks) - len(dec)
    vocab = p.get("vocab_size")
    out = {"dir": d, "status": p.get("status"), "tokens": len(toks), "prefix": P, "vectors": len(steps),
           "weights_id": p.get("weights_id"), "load_seconds": round(p.get("load_seconds") or 0, 1)}
    for name, seq in (("decode", dec), ("refill", ref)):
        nll = []
        for j, s in enumerate(seq):
            if P + 1 + j < len(toks):
                nll.append(-lsm(load(d, s))[toks[P + 1 + j]])
        if nll:
            out[name + "_ppl"] = float(np.exp(np.mean(nll)))
            out[name + "_targets"] = len(nll)
    for name in ("prefill", "reset-prefill"):
        if by.get(name):
            out[name + "_nll_first_target"] = float(-lsm(load(d, by[name][0]))[toks[P]])
    def diff(a, b):
        va, vb = load(d, a), load(d, b)
        return float(np.abs(va.astype(np.float64) - vb).max()), int(va.argmax() == vb.argmax())
    if by.get("prefill") and by.get("reset-prefill"):
        out["prefill_vs_reset_prefill_maxabs"], _ = diff(by["prefill"][0], by["reset-prefill"][0])
    if by.get("prefill") and by.get("chunk"):
        out["prefill_vs_last_chunk_maxabs"], _ = diff(by["prefill"][0], by["chunk"][-1])
    if dec and ref:
        worst, agree = 0.0, 0
        for a, b in zip(dec, ref):
            m, t = diff(a, b)
            worst, agree = max(worst, m), agree + t
        out["decode_vs_refill_maxabs"] = worst
        out["decode_vs_refill_argmax_agree"] = f"{agree}/{min(len(dec), len(ref))}"
    flags = []
    for k in ("decode_ppl", "refill_ppl"):
        v = out.get(k)
        if v is None or not np.isfinite(v) or v < 1.0 or (vocab and v > vocab):
            flags.append(f"{k} absurd ({v}; uniform over the vocabulary is {vocab})")
    out["flags"] = flags or "none"
    print(json.dumps(out))
