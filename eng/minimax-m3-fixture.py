#!/usr/bin/env python3
"""Write tiny synthetic MiniMax-M3.1 GGUF fixtures that llama.cpp and TensorSharp both load.

Variants (one per call; every variant derives its weights from --seed alone):
  small         arch minimax-m3: 6 layers (2 dense + 4 MoE), hidden 512, MSA block 16 and
                top-k 4, so a query at position >= 64 runs sparse.      -> OUT_DIR/model.gguf
  real-block    the same weights as `small` except the indexer, with the released MSA
                geometry, block 128 and top-k 16 (sparse from 2048).    -> OUT_DIR/model.gguf
  dspark-draft  arch dflash: a 2-layer DSpark drafter for the small target (taps [2, 4, 6],
                block 7, Markov rank 32, confidence head).              -> OUT_DIR/dspark.gguf

Which fixture for which test: `small` wherever a short sequence must run the sparse path
(prefill against decode, chunked prefill, rewind, slot copy, batched decode) and, with its
dspark.gguf, speculative decoding; `real-block` for the released MSA geometry (the dense span
below 2048, the switch to sparse attention at 2048, kernels at block 128 / top-k 16). Compare
logits on random tokens (eng/validation/minimax-m3-parity.cpp): their top two tokens are often
closer than two engines' rounding, so greedy text drifts there. Compare greedy TOKENS only on
prompts that end in the successor chain (below), which continue it by several logits.
Fixture v3 came out byte-identical from macOS (Accelerate) and Linux (OpenBLAS) numpy, but the
routing basis is a LAPACK QR and its codes a BLAS product, so another numpy build may round
them differently: compare a regenerated file's sha256 with the published manifest's.

Every tokenizer.* key (vocab 200064, merges, pre-tokenizer minimax-m2, NFC, chat template)
and the general.sampling.* defaults are copied from the real MiniMax-M3.1-Flash first shard
(--tokenizer-from), so prompts tokenize and render exactly as on the real model.

Storage follows the unsloth UD-Q4_K_XL file: attention, shared-expert and dense-FFN matrices
Q8_0; routed experts NVFP4 with a per-expert F32 `.scale` = 1/global (the dequantized true
weight is nvfp4 x scale, applied to each expert matmul's output); indexer, router and every
norm F32; token_embd and output BF16 and untied. Indexer tensors sit on every layer, as in
M3.1. Norms are stored as fp32(1 + bf16(w)), as the converter writes them, so a consumer
applies a plain RMSNorm. The NVFP4 packing reproduces ggml's quantize_row_nvfp4_ref and is
checked against gguf-py's dequantizer before the file is written.

The weights are random, but the model's two discrete decisions, the routed-expert choice and
the MSA block choice, carry margins far above engine rounding (~0.5 % activation noise from
8-bit activation quantization), so two correct engines pick the same experts and blocks and
their logits differ only by rounding. A random router and a token-dependent indexer do not
(fixture v1 measured top-1 agreement of 38 % between two llama.cpp micro-batch sizes in the
sparse range: one flipped choice moves a token's hidden state, which flips later choices).
Three hidden-dimension ranges make the decisions deterministic; no layer writes into the
first two, because every output projection has zero rows there:
  [0, 32)   one constant vector shared by all tokens. The indexer projections read only these
            dimensions, so after the per-head RMSNorm the index query and key are constant
            vectors and the index score depends on the distance p - t alone (partial RoPE).
            Each index head prefers either the most recent blocks or the oldest ones, by a
            monotone score, so the ranking of blocks never crosses; the smallest gap between
            the last selected and the first dropped block is checked for every position below
            1024 (small) / 8192 (real-block) and recorded in the manifest (~1 % of the score
            scale; F16 index rounding is ~0.1 %). Heads alternate between the two preferences.
  [32, 96)  a per-token routing code. The routers read only these dimensions, so a token's
            router logits are fixed targets divided by the residual RMS. MoE layers 2 and 4
            switch 6 experts on (p ~ 1) and pick 4 of them by exp_probs_b, spaced 0.02 apart
            on bases of 16 and 37 (the real model's 16..37), which an F16 or BF16 selection
            scrambles and F32 does not; layers 3 and 5 switch 4 experts on with distinct
            p in 0.54..0.95, so the renormalized routing weights differ. (The two lowest-bias
            experts of a bias-choice layer always rank below 4 others and are never picked.)
  [96, 512) token content, the only dimensions attention and FFN outputs write.
Scales keep the residual stream O(1) through the 6 layers: attention is moderately sharp
(per-head QK norms ~1.4), gate/up pre-activations (~N(0, 2.5^2)) hit swiglu_oai's +-7 clamps
on a fraction of a percent of values, and logits have a std of ~3.5. MoE layers 2, 3 and 5
share one global scale between gate and up per expert (as the real file appears to); layer 4
does not, so a consumer that fuses gate+up on per-element equal scales runs both ways.

Random logits leave the top two tokens of a position close more often than rounding allows,
so greedy text from two correct engines (or with and without speculative decoding) drifts
apart at such near-ties. Token ids 100000..104095 form a successor chain instead: their
embeddings are twice as large, and the output row of each chain token's successor adds that
token's embedding, so after a chain token the target predicts its successor by ~10 logits
over the runner-up. The successor is t + 1, except that every 13th chain token skips to t + 2
(the skipped token itself has no successor).
The DSpark drafter's Markov head encodes t -> t + 1 for the whole chain, so a prompt ending in
token 100000 drafts 7 tokens that the target accepts up to the next skip and rejects there:
acceptance and rollback both run, and greedy output must not change with the drafter.

A JSON manifest (hyperparameters, seed, tokenizer source, routing and MSA layout, the GGUF's
sha256) is written beside each GGUF. The GGUF is written to a hidden partial file and renamed
into place, so a consumer polling for it never reads a half-written file.

Requires the gguf-py of the llama.cpp MiniMax-M3.1 branch (NVFP4, indexer and DSpark keys):
  PYTHONPATH=<llama.cpp-m31>/gguf-py python3 eng/minimax-m3-fixture.py OUT_DIR \
      --tokenizer-from MiniMax-M3.1-Flash-...-00001-of-0000N.gguf [--seed N] [--variant small]
"""
import argparse
import hashlib
import json
import os
import sys
import zlib
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

try:
    import gguf
    from gguf import GGMLQuantizationType as QT, GGUFValueType
    from gguf.quants import NVFP4
except ImportError as e:
    sys.exit(f"error: gguf-py is not importable ({e}); set PYTHONPATH=<llama.cpp-m31>/gguf-py")

if not hasattr(QT, "NVFP4") or not hasattr(gguf.GGUFWriter, "add_indexer_block_size") \
        or not hasattr(gguf.GGUFWriter, "add_sample_from_anchor"):
    sys.exit(f"error: {gguf.__file__} lacks NVFP4 / MSA indexer / DSpark support; use the llama.cpp MiniMax-M3.1 branch's gguf-py")

FIXTURE_VERSION = 3

TARGET = dict(
    n_embd=512, n_layer=6, n_dense=2, n_head=64, n_head_kv=4, head_dim=128, n_rot=64,
    rope_base=1.0e7, eps=1.0e-6, n_ff=1024, n_expert=16, n_expert_used=4, n_ff_exp=256,
    n_shared=1, weights_scale=2.0, idx_heads=4, idx_dim=128, local_blocks=1,
    n_ctx_train=131072, n_vocab=200064,
)
# variant -> (MSA block size, MSA top-k blocks, positions whose block choice is margin-checked)
MSA = {"small": (16, 4, 1024), "real-block": (128, 16, 8192)}
DRAFT = dict(n_layer=2, n_ff=1024, block_size=7, target_layers=[2, 4, 6], markov_rank=32,
             sliding_window=4096, mask_token_id=200058, conf_bias=0.0152588)

# hidden-dimension ranges (see the module docstring); output projections never write [0, 96)
IDX_DIMS = slice(0, 32)
CODE_DIMS = slice(32, 96)
RESERVED = 96
# MoE layers whose gate and up experts get distinct global scales (all others share one)
GATE_UP_DISTINCT_LAYERS = (4,)
# per MoE layer (2, 3, 4, 5): routing kind and exp_probs_b base (real layer 3 ~16.4, 59 ~37.0)
ROUTING = (("bias-choice", 16.0), ("weighted", 23.0), ("bias-choice", 37.0), ("weighted", 30.0))
ROUTER_GAIN = 24.0               # router logit of an "on" code entry times the residual RMS
BIAS_CHOICE_ON = 6               # experts switched on in a bias-choice layer
BIAS_STEP = 0.02                 # exp_probs_b spacing in a bias-choice layer
WEIGHTED_ON = (0.027, 0.06, 0.1, 0.2)   # code entries of the 4 experts on in a weighted layer
MSA_MIN_GAP = 0.008              # refuse to write a fixture whose block choice is closer than this
# successor chain: first id, length, skip period (target t -> t + 2 at every 13th chain token)
CHAIN = (100000, 4096, 13)
CHAIN_EMBD = 2.0                 # content-dimension scale of a chain token's embedding
CHAIN_OUT = 0.065                # weight of the predecessor's embedding in a successor's output row
CHAIN_MARKOV = 8.0               # DSpark Markov code length of a chain token (logit 64 for t + 1)


def chain_successors():
    """(predecessor ids, target successor ids) of the chain.

    A skipped token (t + 1 of a skip t -> t + 2) gets no successor, so every successor's output
    row carries exactly one predecessor term.
    """
    first, n, period = CHAIN
    t = np.arange(first, first + n - 2)
    skip = (t - first) % period == period - 1
    keep = ~np.isin(t, t[skip] + 1)
    return t[keep], (t + 1 + skip)[keep]

# ggml's doubled E2M1 code book (kvalues_mxfp4); the scale decode carries the matching 0.5
KVALUES_FP4 = np.array(NVFP4.kvalues, dtype=np.float32)
# largest |value| one NVFP4 sub-block can hold: 12 (doubled E2M1 max) x 0.5 x 448 (E4M3 max)
NVFP4_MAX = 6.0 * 448.0


def rng_for(seed, name):
    """One independent stream per tensor, so a tensor's values never depend on write order."""
    return np.random.default_rng([seed, zlib.crc32(name.encode("utf-8"))])


def bf16_round(x):
    """Round-to-nearest-even to bf16, returned as float32 (ggml_compute_fp32_to_bf16)."""
    b = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    b = ((b + 0x7FFF + ((b >> 16) & 1)) >> 16).astype(np.uint32) << np.uint32(16)
    return b.view(np.float32)


def nvfp4_quantize(w):
    """float32 [..., K] (K % 64 == 0) -> ggml block_nvfp4 bytes [..., K/64*36].

    Mirrors quantize_row_nvfp4_ref: per 16 values the UE4M3 scale encodes amax/6, each value
    takes the nearest doubled-E2M1 code (first index on ties, like best_index_mxfp4). Also
    returns the exact dequantized values, which the caller checks against gguf-py.
    """
    lead, k = w.shape[:-1], w.shape[-1]
    if k % 64:
        raise ValueError(f"NVFP4 row of {k} values is not a multiple of 64")
    x = np.ascontiguousarray(w, dtype=np.float32).reshape(-1, k // 64, 4, 16)
    ue = NVFP4.fp32_to_ue4m3((np.abs(x).max(axis=-1) / np.float32(6.0)).astype(np.float32)).astype(np.uint8)
    d = NVFP4.ue4m3_to_fp32(ue).astype(np.float32)                      # includes the 0.5
    idx = np.empty(x.shape, dtype=np.uint8)
    for r0 in range(0, x.shape[0], 256):                                 # bound the 16-way error tensor
        xs, ds = x[r0:r0 + 256], d[r0:r0 + 256]
        idx[r0:r0 + 256] = np.abs(ds[..., None, None] * KVALUES_FP4 - xs[..., None]).argmin(axis=-1)
    deq = (d[..., None] * KVALUES_FP4[idx]).reshape(*lead, k)
    qs = idx[..., :8] | (idx[..., 8:] << np.uint8(4))                   # low = j, high = j + 8
    raw = np.concatenate([ue, qs.reshape(*qs.shape[:2], 32)], axis=-1)  # {d[4], qs[32]} per 64
    return raw.reshape(*lead, k // 64 * 36), deq


def gaussian(seed, name, shape, std, mean=0.0):
    return rng_for(seed, name).normal(mean, std, size=shape).astype(np.float32)


def norm_weight(seed, name, n, std, mean=0.0):
    """Gemma (1 + w) RMSNorm weight as the converter stores it: fp32(1 + bf16(w))."""
    return bf16_round(gaussian(seed, name, (n,), std, mean)) + np.float32(1.0)


def output_projection(seed, name, shape, std):
    """An output projection [..., n_out, n_in] that writes token content only (reserved rows 0)."""
    w = gaussian(seed, name, shape, std)
    w[..., :RESERVED, :] = 0.0
    return w


class Fixture:
    def __init__(self, path, arch):
        self.path = path
        self.w = gguf.GGUFWriter(str(path), arch)
        self.summary = {}

    def f32(self, name, data):
        self.w.add_tensor(name, np.ascontiguousarray(data, dtype=np.float32))

    def quant(self, name, data, qtype):
        self.w.add_tensor(name, gguf.quants.quantize(np.ascontiguousarray(data, dtype=np.float32), qtype), raw_dtype=qtype)

    def nvfp4_experts(self, name, weights, scale):
        raw, deq = nvfp4_quantize(weights)
        ref = gguf.quants.dequantize(raw, QT.NVFP4).reshape(deq.shape)
        if not np.array_equal(ref, deq):
            raise AssertionError(f"{name}: NVFP4 packing disagrees with gguf-py's dequantizer")
        nz = weights != 0
        err = float(np.sqrt(np.mean((deq[nz] - weights[nz]) ** 2) / np.mean(weights[nz] ** 2)))
        self.summary[name] = {"nvfp4_rel_rms_error": round(err, 5)}
        self.w.add_tensor(name, raw, raw_dtype=QT.NVFP4)
        self.f32(name.replace(".weight", ".scale"), scale)

    def write(self):
        self.w.write_header_to_file()
        self.w.write_kv_data_to_file()
        self.w.write_tensors_to_file()
        self.w.close()


def copy_tokenizer(fx, source):
    """Every tokenizer.* key and the general.sampling.* defaults of the real first shard."""
    reader = gguf.GGUFReader(str(source))
    copied = []
    for key, field in reader.fields.items():
        if not (key.startswith("tokenizer.") or key.startswith("general.sampling.")):
            continue
        main = field.types[0]
        if main == GGUFValueType.ARRAY:
            fx.w.add_key_value(key, field.contents(), main, sub_type=field.types[-1])
        else:
            fx.w.add_key_value(key, field.contents(), main)
        copied.append(key)
    for need in ("tokenizer.ggml.tokens", "tokenizer.ggml.merges", "tokenizer.chat_template"):
        if need not in copied:
            raise ValueError(f"{source} has no {need}; pass the real model's first shard")
    n_vocab = len(reader.fields["tokenizer.ggml.tokens"].data)
    if n_vocab != TARGET["n_vocab"]:
        raise ValueError(f"{source} has {n_vocab} tokens, expected {TARGET['n_vocab']}")
    return copied


def common_hparams(fx, n_layer, n_ff):
    t = TARGET
    w = fx.w
    w.add_type("model")
    w.add_block_count(n_layer)
    w.add_context_length(t["n_ctx_train"])
    w.add_embedding_length(t["n_embd"])
    w.add_feed_forward_length(n_ff)
    w.add_head_count(t["n_head"])
    w.add_head_count_kv(t["n_head_kv"])
    w.add_rope_freq_base(t["rope_base"])
    w.add_layer_norm_rms_eps(t["eps"])
    w.add_key_length(t["head_dim"])
    w.add_value_length(t["head_dim"])
    w.add_rope_dimension_count(t["n_rot"])


def attention_tensors(fx, seed, p, matrix_type):
    """Attention block shared by the target and the drafter (Q/K/V/O, per-head QK norms)."""
    t = TARGET
    d, hd = t["n_embd"], t["head_dim"]
    nq, nkv = t["n_head"] * hd, t["n_head_kv"] * hd
    fx.f32(f"{p}.attn_norm.weight", norm_weight(seed, f"{p}.attn_norm", d, 0.10))
    fx.quant(f"{p}.attn_q.weight", gaussian(seed, f"{p}.attn_q", (nq, d), d ** -0.5), matrix_type)
    fx.quant(f"{p}.attn_k.weight", gaussian(seed, f"{p}.attn_k", (nkv, d), d ** -0.5), matrix_type)
    fx.quant(f"{p}.attn_v.weight", gaussian(seed, f"{p}.attn_v", (nkv, d), d ** -0.5), matrix_type)
    # per-head QK norms around 1.4: score std ~2, a moderately sharp attention whose output
    # changes visibly when MSA drops blocks
    fx.f32(f"{p}.attn_q_norm.weight", norm_weight(seed, f"{p}.attn_q_norm", hd, 0.25, 0.4))
    fx.f32(f"{p}.attn_k_norm.weight", norm_weight(seed, f"{p}.attn_k_norm", hd, 0.25, 0.4))
    fx.quant(f"{p}.attn_output.weight", output_projection(seed, f"{p}.attn_output", (d, nq), 1.5 * nq ** -0.5), matrix_type)
    fx.f32(f"{p}.ffn_norm.weight", norm_weight(seed, f"{p}.ffn_norm", d, 0.10))


def dense_ffn(fx, seed, p, n_ff, matrix_type, suffix=""):
    d = TARGET["n_embd"]
    fx.quant(f"{p}.ffn_gate{suffix}.weight", gaussian(seed, f"{p}.ffn_gate{suffix}", (n_ff, d), 2.5 * d ** -0.5), matrix_type)
    fx.quant(f"{p}.ffn_up{suffix}.weight", gaussian(seed, f"{p}.ffn_up{suffix}", (n_ff, d), 2.5 * d ** -0.5), matrix_type)
    fx.quant(f"{p}.ffn_down{suffix}.weight", output_projection(seed, f"{p}.ffn_down{suffix}", (d, n_ff), 0.15 * n_ff ** -0.5), matrix_type)


def routing_codes(seed):
    """Per-token router targets L [vocab, 4 x 16] and the orthogonal map Q with code = L @ Q.

    Router m reads code dims through Q's rows 16m..16m+15, so its logit for expert e is
    ROUTER_GAIN * L[t, 16m + e] / rms(h) exactly (up to the code's bf16 rounding).
    """
    n_vocab, n_exp = TARGET["n_vocab"], TARGET["n_expert"]
    n_code = CODE_DIMS.stop - CODE_DIMS.start
    q, _ = np.linalg.qr(rng_for(seed, "routing_basis").normal(size=(n_code, n_code)))
    targets = np.full((n_vocab, len(ROUTING) * n_exp), -1.0, dtype=np.float32)
    for m, (kind, _) in enumerate(ROUTING):
        rng = rng_for(seed, f"routing_codes.{m}")
        order = np.argsort(rng.random((n_vocab, n_exp)), axis=1)
        cols = slice(m * n_exp, (m + 1) * n_exp)
        block = targets[:, cols]
        rows = np.arange(n_vocab)[:, None]
        if kind == "bias-choice":
            block[rows, order[:, :BIAS_CHOICE_ON]] = 1.0
        else:
            vals = np.array(WEIGHTED_ON, dtype=np.float32)[np.argsort(rng.random((n_vocab, len(WEIGHTED_ON))), axis=1)]
            block[rows, order[:, :len(WEIGHTED_ON)]] = vals
        targets[:, cols] = block
    return targets, q.astype(np.float64)


def block_choice_gap(score, block, top_k, p_lo, p_hi):
    """Smallest gap between the last selected and the first dropped past block over p_lo..p_hi.

    score[d] is the index score of a key d positions before the query; a full past block's
    score is its maximum (MSA "max" pooling), the query's own block is forced.
    """
    win = np.full(len(score), -np.inf)
    win[block - 1:] = sliding_window_view(score, block).max(axis=1)   # block whose farthest key is d back
    worst = np.inf
    for p in range(max(p_lo, top_k * block), p_hi):
        own = p // block
        s = np.sort(win[p - np.arange(own) * block])[::-1]
        worst = min(worst, s[top_k - 2] - s[top_k - 1])
    return worst


def index_tensors(fx, seed, p, il, variant, attn_norm, carrier):
    """Constant index query/key vectors whose RoPE'd score is a monotone function of distance."""
    t = TARGET
    d, hd, n_rot = t["n_embd"], t["idx_dim"], t["n_rot"]
    half = n_rot // 2
    block, top_k, n_check = MSA[variant]
    theta = t["rope_base"] ** (-np.arange(half) / half)
    # slowest NEOX pairs that stay monotone over the checked range: sin up to pi/2, cos up to pi
    i_sin = int(np.argmax(theta * n_check <= np.pi / 2))
    i_cos = int(np.argmax(theta * n_check <= np.pi))
    rng = rng_for(seed, f"{p}.indexer.pattern")
    ik = rng.normal(size=hd)
    g_q = norm_weight(seed, f"{p}.indexer.q_norm", hd, 0.25)
    g_k = norm_weight(seed, f"{p}.indexer.k_norm", hd, 0.25)
    dist = np.arange(n_check + 1, dtype=np.float64)
    iq_all, kinds, gaps = [], [], []
    for h in range(t["idx_heads"]):
        recent = (il + h) % 2 == 0
        a = rng.uniform(0.6, 0.9) if recent else rng.uniform(0.45, 0.6)
        # oldest-first: a sin(th_s d) - b cos(th_c d), increasing in d; most-recent-first: its negation
        comps = ((i_sin, a, np.pi / 2 if recent else -np.pi / 2), (i_cos, 1.0 - a, 0.0 if recent else np.pi))
        iq = np.zeros(hd)
        iq[n_rot:] = rng.normal(size=hd - n_rot) * 0.2 / np.abs(ik[n_rot:]).sum()   # constant offset
        for i, amp, phase in comps:
            zk = complex(ik[i], ik[i + half])
            zq = amp / abs(zk) * np.exp(1j * (phase + np.angle(zk)))
            iq[i], iq[i + half] = zq.real, zq.imag
        # the score of the vectors as built, pair by pair: Re(zq conj(zk) e^{j d theta})
        zq = iq[:half] + 1j * iq[half:n_rot]
        zk = ik[:half] + 1j * ik[half:n_rot]
        score = (np.exp(1j * np.outer(dist, theta)) * (zq * np.conj(zk))).real.sum(axis=1)
        gap = block_choice_gap(score, block, top_k, 0, n_check) / sum(c[1] for c in comps)
        if not gap >= MSA_MIN_GAP:
            raise AssertionError(f"{p} index head {h}: block-choice gap {gap:.4g} < {MSA_MIN_GAP}")
        iq_all.append(iq / g_q)
        kinds.append("recent" if recent else "oldest")
        gaps.append(round(float(gap), 5))
    # the projections read the constant carrier only: W x = v * (u . x) with u . x = 1 / rms(h)
    u = np.zeros(d)
    cw = carrier.astype(np.float64) * attn_norm[IDX_DIMS]
    u[IDX_DIMS] = cw / np.dot(cw, cw)
    fx.f32(f"{p}.indexer.q_proj.weight", np.outer(np.concatenate(iq_all), u))
    fx.f32(f"{p}.indexer.k_proj.weight", np.outer(ik / g_k, u))
    fx.f32(f"{p}.indexer.q_norm.weight", g_q)
    fx.f32(f"{p}.indexer.k_norm.weight", g_k)
    return {"heads": kinds, "min_gap_over_scale": gaps, "rope_pairs": [i_sin, i_cos]}


def write_target(fx, seed, variant):
    t = TARGET
    d, n_exp, n_ff_exp = t["n_embd"], t["n_expert"], t["n_ff_exp"]
    block, top_k, n_check = MSA[variant]
    w = fx.w
    w.add_name(f"MiniMax-M3.1 synthetic fixture ({variant})")
    w.add_description(f"TensorSharp eng/minimax-m3-fixture.py v{FIXTURE_VERSION}, variant {variant}, seed {seed}: "
                      "random weights in the MiniMax-M3.1-Flash layout, not a language model")
    common_hparams(fx, t["n_layer"], t["n_ff"])
    w.add_file_type(gguf.LlamaFileType.MOSTLY_Q8_0)
    w.add_expert_count(n_exp)
    w.add_expert_used_count(t["n_expert_used"])
    w.add_expert_gating_func(gguf.ExpertGatingFuncType.SIGMOID)
    w.add_expert_feed_forward_length(n_ff_exp)
    w.add_expert_shared_count(t["n_shared"])
    w.add_expert_weights_scale(t["weights_scale"])
    w.add_expert_weights_norm(True)
    w.add_indexer_head_count(t["idx_heads"])
    w.add_indexer_key_length(t["idx_dim"])
    w.add_indexer_top_k(top_k)
    w.add_indexer_block_size(block)
    w.add_indexer_local_blocks(t["local_blocks"])
    w.add_leading_dense_block_count(t["n_dense"])
    w.add_quantization_version(gguf.GGML_QUANT_VERSION)

    carrier = gaussian(seed, "index_carrier", (IDX_DIMS.stop,), 1.0)
    targets, basis = routing_codes(seed)
    embd = gaussian(seed, "token_embd", (t["n_vocab"], d), 1.0)
    first, n_chain, _ = CHAIN
    embd[first:first + n_chain, RESERVED:] *= CHAIN_EMBD
    embd[:, IDX_DIMS] = carrier
    embd[:, CODE_DIMS] = (targets.astype(np.float64) @ basis).astype(np.float32)
    fx.quant("token_embd.weight", embd, QT.BF16)
    carrier = bf16_round(carrier)
    fx.f32("output_norm.weight", norm_weight(seed, "output_norm", d, 0.10))
    # a successor's output row = CHAIN_OUT x its predecessor's content embedding + its random row,
    # shrunk so that the row's logit std after other tokens stays ~3.5 (the predecessor term alone
    # gives ~CHAIN_OUT x CHAIN_EMBD x sqrt(416) x 1.15 there: the normalized hidden carries ~1.15 per
    # content element, the content dims holding all of the residual stream's growth)
    out = gaussian(seed, "output", (t["n_vocab"], d), 3.5 * d ** -0.5)
    pred, succ = chain_successors()
    cross = CHAIN_OUT * CHAIN_EMBD * np.sqrt(d - RESERVED) * 1.15 / 3.5
    out[np.unique(succ)] *= np.sqrt(max(0.05, 1.0 - cross * cross))
    np.add.at(out, (succ[:, None], np.arange(RESERVED, d)[None, :]), CHAIN_OUT * embd[pred, RESERVED:])
    del embd
    fx.quant("output.weight", out, QT.BF16)
    del out

    scales, msa, routing = {}, {}, {}
    for il in range(t["n_layer"]):
        p = f"blk.{il}"
        attention_tensors(fx, seed, p, QT.Q8_0)
        msa[il] = index_tensors(fx, seed, p, il, variant, norm_weight(seed, f"{p}.attn_norm", d, 0.10), carrier)
        if il < t["n_dense"]:
            dense_ffn(fx, seed, p, t["n_ff"], QT.Q8_0)
            continue
        moe = il - t["n_dense"]
        kind, base = ROUTING[moe]
        # router logit e = ROUTER_GAIN * targets[t, 16 moe + e] / rms(h): it reads the code dims only
        ffn_norm = norm_weight(seed, f"{p}.ffn_norm", d, 0.10)
        router = np.zeros((n_exp, d))
        router[:, CODE_DIMS] = ROUTER_GAIN * basis[moe * n_exp:(moe + 1) * n_exp] / ffn_norm[CODE_DIMS]
        fx.f32(f"{p}.ffn_gate_inp.weight", router)
        if kind == "bias-choice":
            bias = base + BIAS_STEP * rng_for(seed, f"{p}.exp_probs_b").permutation(n_exp)
        else:
            bias = base + np.clip(gaussian(seed, f"{p}.exp_probs_b", (n_exp,), 0.03), -0.08, 0.08)
        fx.f32(f"{p}.exp_probs_b.bias", bias)
        routing[il] = {"kind": kind, "bias_base": base}
        # per-expert magnitudes differ, so the per-expert scales matter
        gain = rng_for(seed, f"{p}.expert_gain").uniform(0.7, 1.4, size=(n_exp, 1, 1)).astype(np.float32)
        gate = gain * gaussian(seed, f"{p}.ffn_gate_exps", (n_exp, n_ff_exp, d), 2.5 * d ** -0.5)
        up = gain * gaussian(seed, f"{p}.ffn_up_exps", (n_exp, n_ff_exp, d), 2.5 * d ** -0.5)
        down = gain * output_projection(seed, f"{p}.ffn_down_exps", (n_exp, d, n_ff_exp), 0.12 * n_ff_exp ** -0.5)
        jitter = lambda tag: rng_for(seed, f"{p}.{tag}").uniform(0.55, 0.95, size=n_exp)   # noqa: E731
        amax = lambda x: np.abs(x).reshape(n_exp, -1).max(axis=1)                         # noqa: E731
        if il in GATE_UP_DISTINCT_LAYERS:
            g_gate = NVFP4_MAX / amax(gate) * jitter("gate_global")
            g_up = NVFP4_MAX / amax(up) * jitter("up_global")
        else:
            g_gate = g_up = NVFP4_MAX / np.maximum(amax(gate), amax(up)) * jitter("gate_up_global")
        g_down = NVFP4_MAX / amax(down) * jitter("down_global")
        for name, wts, g in (("ffn_gate_exps", gate, g_gate), ("ffn_up_exps", up, g_up), ("ffn_down_exps", down, g_down)):
            # stored weight = true weight x global; .scale = 1/global restores it at the matmul output
            fx.nvfp4_experts(f"{p}.{name}.weight", wts * g.astype(np.float32)[:, None, None],
                             (1.0 / g).astype(np.float32))
        scales[il] = "distinct" if il in GATE_UP_DISTINCT_LAYERS else "shared"
        dense_ffn(fx, seed, p, t["n_ff_exp"] * t["n_shared"], QT.Q8_0, suffix="_shexp")
    fx.summary["gate_up_scales"] = scales
    return {"msa_block": block, "msa_top_k": top_k, "sparse_from_position": block * top_k,
            "msa_margin_checked_below": n_check, "msa": msa, "routing": routing,
            "successor_chain": {"first": CHAIN[0], "length": CHAIN[1], "skip_period": CHAIN[2]},
            "reserved_dims": {"index_carrier": [IDX_DIMS.start, IDX_DIMS.stop],
                              "routing_code": [CODE_DIMS.start, CODE_DIMS.stop]}}


def write_draft(fx, seed):
    t, dr = TARGET, DRAFT
    d, rank = t["n_embd"], dr["markov_rank"]
    w = fx.w
    w.add_name("MiniMax-M3.1 synthetic DSpark fixture")
    w.add_description(f"TensorSharp eng/minimax-m3-fixture.py v{FIXTURE_VERSION}, variant dspark-draft, seed {seed}: "
                      "untrained DSpark drafter for the small minimax-m3 fixture")
    common_hparams(fx, dr["n_layer"], dr["n_ff"])
    w.add_file_type(gguf.LlamaFileType.MOSTLY_BF16)
    w.add_block_size(dr["block_size"])
    w.add_target_layers(dr["target_layers"])
    w.add_sliding_window(dr["sliding_window"])
    w.add_sliding_window_pattern([True] * dr["n_layer"])
    w.add_sample_from_anchor(True)
    w.add_has_confidence_head(True)
    w.add_hidden_act("swigluoai")
    w.add_quantization_version(gguf.GGML_QUANT_VERSION)
    w.add_mask_token_id(dr["mask_token_id"])

    n_in = len(dr["target_layers"]) * d
    fx.f32("conf_proj.bias", np.array([dr["conf_bias"]], dtype=np.float32))
    fx.f32("conf_proj.weight", gaussian(seed, "conf_proj", (1, d + rank), (d + rank) ** -0.5))
    fx.quant("fc.weight", gaussian(seed, "fc", (d, n_in), n_in ** -0.5), QT.BF16)
    fx.f32("output_norm.weight", norm_weight(seed, "output_norm", d, 0.10))
    fx.f32("enc.output_norm.weight", norm_weight(seed, "enc.output_norm", d, 0.10))
    for il in range(dr["n_layer"]):
        p = f"blk.{il}"
        attention_tensors(fx, seed, p, QT.BF16)
        dense_ffn(fx, seed, p, dr["n_ff"], QT.BF16)
    # Markov bias <w2[v], w1[prev]> with a std of ~1.5 logits, beside base logits of ~3.5
    m = (1.5 / np.sqrt(rank)) ** 0.5
    w1 = gaussian(seed, "markov_w1", (t["n_vocab"], rank), m)
    w2 = gaussian(seed, "markov_w2", (t["n_vocab"], rank), m)
    # chain: w1[t] = w2[t + 1] = CHAIN_MARKOV x a unit code, so prev t drafts t + 1 by a wide margin
    first, n_chain, _ = CHAIN
    code = rng_for(seed, "markov_chain").normal(size=(n_chain, rank))
    code *= CHAIN_MARKOV / np.linalg.norm(code, axis=1, keepdims=True)
    w2[first:first + n_chain] = code
    w1[first:first + n_chain - 1] = code[1:]
    fx.quant("markov_w1.weight", w1, QT.BF16)
    fx.quant("markov_w2.weight", w2, QT.BF16)
    return {"target_layers": dr["target_layers"], "block_size": dr["block_size"],
            "markov_rank": rank, "mask_token_id": dr["mask_token_id"], "sliding_window": dr["sliding_window"],
            "successor_chain": {"first": CHAIN[0], "length": CHAIN[1], "drafts": "t + 1"}}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--variant", choices=("small", "real-block", "dspark-draft"), default="small")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tokenizer-from", type=Path, required=True,
                    help="the real MiniMax-M3.1-Flash first shard (metadata is read, tensors are not)")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = "dspark" if args.variant == "dspark-draft" else "model"
    final = args.out_dir / f"{stem}.gguf"
    partial = args.out_dir / f".{stem}.gguf.partial"
    arch = "dflash" if args.variant == "dspark-draft" else "minimax-m3"
    if gguf.MODEL_ARCH_NAMES.get(gguf.MODEL_ARCH.MINIMAXM3) != "minimax-m3":
        sys.exit("error: this gguf-py does not know arch minimax-m3")

    fx = Fixture(partial, arch)
    copied = copy_tokenizer(fx, args.tokenizer_from)
    shape = write_draft(fx, args.seed) if args.variant == "dspark-draft" else write_target(fx, args.seed, args.variant)
    fx.write()
    digest = sha256(partial)
    os.replace(partial, final)

    manifest = {
        "generator": "eng/minimax-m3-fixture.py", "fixture_version": FIXTURE_VERSION,
        "variant": args.variant, "seed": args.seed, "arch": arch, "gguf": final.name, "sha256": digest,
        "bytes": final.stat().st_size, "tokenizer_from": str(args.tokenizer_from),
        "tokenizer_from_sha256": sha256(args.tokenizer_from), "copied_keys": len(copied),
        "hparams": TARGET if args.variant != "dspark-draft" else {**TARGET, **DRAFT},
        **shape, "tensors": fx.summary,
    }
    (args.out_dir / f"{stem}.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("variant", "seed", "gguf", "bytes", "sha256")}))


if __name__ == "__main__":
    main()
