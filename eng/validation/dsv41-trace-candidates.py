#!/usr/bin/env python3
"""Show whether DeepSeek V4.1 indexer candidate pruning engaged, from a ggml_cuda TS_DSV41_TRACE_DIR dump.

usage: python3 eng/validation/dsv41-trace-candidates.py TRACE_DIR PROBE_DIR UBATCH SOURCE_LAYER BUDGET_BLOCKS BLOCK_ROWS CONSUMER_LAYER...
  e.g. dsv41-trace-candidates.py trace/ run/probe 256 4 65 8 8 9

TRACE_DIR is the TS_DSV41_TRACE_DIR of a traced DeepSeek41DistributedProbe run on ggml_cuda, PROBE_DIR that
run's probe output (probe.json), UBATCH its TS_DSV4_UBATCH, SOURCE_LAYER the candidate-source layer,
BUDGET_BLOCKS x BLOCK_ROWS the candidate budget, CONSUMER_LAYER the later index-source layers to check.
Needs numpy; reads artifacts only.

The trace holds v41.LL.index_scores (F32, [n_kv, nt], one row per query) for every index-source layer,
named p<p0>_..., where p0 is the graph's first position (later passes with the same p0 overwrite earlier
ones; reset-prefill and refill write last). The candidate-source layer scores against the causal mask only;
a later index-source layer adds the candidate mask (-inf outside the selected blocks). Pruning is engaged
for a query when a consumer layer leaves fewer rows finite than the source layer does, and the consumer's
finite rows fall in at most BUDGET_BLOCKS blocks of BLOCK_ROWS rows. Prints one line per (pass, consumer)
and a final ENGAGED / NOT ENGAGED verdict.
"""
import glob, json, os, re, sys
import numpy as np

trace, probe = sys.argv[1], sys.argv[2]
ubatch, source, budget, block = (int(x) for x in sys.argv[3:7])
consumers = [int(x) for x in sys.argv[7:]]
p = json.load(open(os.path.join(probe, "probe.json")))
toks = p["tokens"]
steps = [s for s in p["steps"] if s.get("logits")]
P = len(toks) - sum(1 for s in steps if s["operation"] == "decode")
files = {}
for f in glob.glob(os.path.join(trace, "p*_v41.*.index_scores.f32")):
    m = re.match(r"p(\d+)_v41\.(\d+)\.index_scores\.f32$", os.path.basename(f))
    if m:
        files[(int(m.group(1)), int(m.group(2)))] = f
print(f"tokens {len(toks)}, prefix {P}, ubatch {ubatch}, source layer {source}, budget {budget} x {block} rows, "
      f"consumers {consumers}; {len(files)} index_scores files")


def rows(p0, layer):
    f = files.get((p0, layer))
    if f is None:
        return None
    nt = min(ubatch, P - p0) if p0 < P else 1
    v = np.fromfile(f, np.float32)
    if v.size % nt:
        raise SystemExit(f"{f}: {v.size} floats is not a multiple of nt={nt}")
    return v.reshape(nt, v.size // nt)


engaged = []
for p0 in sorted({k[0] for k in files}):
    src = rows(p0, source)
    if src is None:
        continue
    src_finite = (np.isfinite(src) & (src > -1e30)).sum(1)
    for layer in consumers:
        c = rows(p0, layer)
        if c is None:
            print(f"p0={p0}: layer {layer} has no index_scores")
            continue
        if c.shape != src.shape:
            print(f"p0={p0}: layer {layer} shape {c.shape} != source {src.shape}")
            continue
        ok = np.isfinite(c) & (c > -1e30)
        fin = ok.sum(1)
        padded = np.pad(ok, ((0, 0), (0, (-ok.shape[1]) % block)))
        blocks = padded.reshape(ok.shape[0], -1, block).any(2).sum(1)
        pruned = fin < src_finite
        for q in np.nonzero(pruned)[0]:
            engaged.append((p0 + int(q), layer, int(src_finite[q]), int(fin[q]), int(blocks[q])))
        print(f"p0={p0:6d} nt={src.shape[0]:4d} n_kv={src.shape[1]:6d} layer {layer}: source finite "
              f"{int(src_finite.min())}..{int(src_finite.max())}, consumer finite {int(fin.min())}..{int(fin.max())}, "
              f"consumer blocks max {int(blocks.max())}, pruned queries {int(pruned.sum())}/{src.shape[0]}"
              + (f" (first at position {p0 + int(np.nonzero(pruned)[0][0])})" if pruned.any() else ""))
if engaged:
    first = min(e[0] for e in engaged)
    over = [e for e in engaged if e[4] > budget]
    print(f"ENGAGED: {len(engaged)} (query, layer) pairs pruned; first pruned position {first}; "
          f"max consumer blocks {max(e[4] for e in engaged)} (budget {budget}); over budget {len(over)}")
    for e in sorted(engaged)[:3] + sorted(engaged)[-3:]:
        print(f"  position {e[0]} layer {e[1]}: visible rows {e[2]} -> kept rows {e[3]} in {e[4]} blocks")
else:
    print("NOT ENGAGED: no consumer layer left fewer rows finite than the candidate source")
