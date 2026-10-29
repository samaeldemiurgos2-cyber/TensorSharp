# GLM quantized distributed fixtures

These tools diagnose numerical differences between an unsplit GLM executor and
distributed execution. They do not measure trained-model quality or performance.
Four loopback processes can share one GPU; that tests global rank arithmetic,
not four physical GPUs.

Build `GlmDistributedFixture/GlmDistributedFixture.csproj` in Release. The
original no-profile fixture does not require native quantization. K/IQ profiles
require the built `GgmlOps` library in the generator's library search path. They
quantize deterministic F32 tensors through unchanged upstream ggml, using a
uniform importance matrix. They are synthetic weights, not copied model weights.

```text
GlmDistributedFixture FRESH_DIRECTORY [PROFILE] [LAYERS] [PREFILL_TOKENS] [EXPERT_FFN]
```

Omit `PROFILE` to preserve the original six-layer Q8 fixture. Opt-in profiles use
256 hidden channels, eight 128-wide heads, four layers by default, eight routed
experts, top-2 routing, shared experts and 2,048 expert channels. Most remaining
matrices use Q8_0; routers and scalar/vector parameters remain F32.

| Profile | KDA q/k/v and attention output | Routed gate/up | Routed down |
|---|---|---|---|
| `all` | Q6_K | IQ2_S | IQ3_S |
| `attention` | Q6_K | Q8_0 | Q8_0 |
| `experts` | Q8_0 | IQ2_S | IQ3_S |
| `q8-control` | Q8_0 | Q8_0 | Q8_0 |
| `iq3-iq4` | Q6_K | IQ3_S | IQ4_XS |
| `mixed` | Q6_K | IQ3_S then IQ2_S | IQ4_XS |
| `mixed-checkpoint` | Q8_0 at role11; Q6_K elsewhere | IQ3_S then IQ2_S | IQ4_XS |

`mixed` maps its first four layers to production roles 0, 11, 12 and 44:
KDA/dense, MLA/IQ3_S experts, KDA/IQ2_S experts, KDA/IQ2_S experts. Later
synthetic layers repeat the three routed roles. This covers the exceptional
tensor types of the pinned IQ3_XXS checkpoint; the `all` profile covers its
usual routed types. Append `-dense`, `-kda`, or `-kda-dense` to isolate sublayers.

`mixed` preserves the earlier diagnostic fixture bytes: its shared projections
are Q8_0 and all its attention outputs are Q6_K. `mixed-checkpoint` additionally
matches the pinned checkpoint's verified attention-output, shared-projection,
and first dense-FFN formats: role11 uses Q8_0 attention/shared projections;
roles12/44 use Q6_K shared projections; role0 uses a Q6_K dense FFN. Both remain
small synthetic geometry, not full reproductions of checkpoint tensors.

The generator records all parameters and the complete GGUF hash in
`fixture.json`. Default opt-in token files contain 129 prefill tokens plus eight
forced decode tokens. Set `EXPERT_FFN=1280` to exercise a tile-overlap partition
with 320 logical rows per rank at global degree four. Hidden size 256 also gives 64 output rows per rank at global degree
four, below a 128-row MMQ tile.

Use an isolated, already built `DeepSeek41DistributedProbe` runtime. Place its
explicit native library beside the probe so the runner can hash it before and
after execution. The runner does not build, replace or copy runtime files.

```text
python eng/validation/validate-glm-quantized-distributed.py \
  --probe PATH/DeepSeek41DistributedProbe.dll \
  --fixture FRESH_DIRECTORY --output FRESH_RESULT_DIRECTORY
```

The default comparison uses one unsplit process and four loopback TCP nodes,
each with one local GPU rank. `--mode layer` instead exercises distributed
layer placement. Logs, commands, runtime hashes, full F32 vectors, worker
lifecycle reports and every-vector comparisons are retained. An error or failed
gate returns a nonzero exit code. The runner kills only its own remaining
children after a failed or timed-out run.

The fixed gates are `2e-5` absolute error, RMSE, relative L2 and normalized maximum
error, cosine at least `0.999999`, and matching argmax for every vector. No failed
case is accepted solely because generated tokens match.

For narrow-kernel and MMQ boundary checks, run prefixes 6, 7, 8, 9, 127, 128 and
129 with `--prefix N --chunk N --ubatch 512`. This keeps the 129-token prefill in
one native graph. Main lifecycle runs use `--chunk 64 --ubatch 128`; both paths
include single-token decode, reset, refill and repeated prefill. Keep these
synthetic results separate from full-checkpoint qualification and benchmarks.

For the wide routed profiles, add `--require-no-expert-fallback` to require each
TP node's startup diagnostics to confirm zero full-weight projection fallbacks.
This prevents a small synthetic fixture from passing only through a replication
path that a larger checkpoint cannot afford. Keep original tiny-fixture fallback
coverage separate.

For a runtime experiment, add `--byte-reference SAVED_RESULT/node0/probe.json`
to require both the fresh unsplit and distributed vectors to have exactly the
same bytes as a previously qualified run. The saved operation/token schedule
must also match. This requirement is stronger than the numeric tolerance gates;
all original reference files remain unchanged.

The native `GgmlOpsCudaMatmulPrecisionTest --quant-strip-glm-only` mode tests
full GLM projection geometry against unchanged upstream `MUL_MAT_ID`: 288
experts, top8, 128 tokens, four output-row shards, gate/up4096→2048 and
down2048→4096. It covers IQ2_S/IQ3_S and exceptional IQ3_S/IQ4_XS formats,
balanced routing, a hot32-expert subset and graph replay with new inputs/IDs.
Use `--type iq2_s --role gate`, `--type iq3_s --role gate|down`, or
`--type iq4_xs --role down` to isolate a projection. The mode requires owned
CUDA strips and byte equality; code77 means unavailable, not passed. A
deterministic quantized row pool reduces setup cost while preserving full
weight sizes, expert strides and MMQ launch geometry.
