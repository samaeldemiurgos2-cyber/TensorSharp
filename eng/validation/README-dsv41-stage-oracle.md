# DeepSeek V4.1 per-stage oracle: direct CUDA against `ggml_cuda`

`dsv41-stage-oracle.py` checks the direct-CUDA DeepSeek V4.1 engine (`--backend cuda`)
on real weights one stage at a time. It reads two traces of the same tokens:

- one from the direct engine (`TS_DSV4_CUDA_TRACE_DIR`);
- one from `ggml_cuda` (`TS_DSV41_TRACE_DIR`), run with exact prefill (`TS_DSV41_YOCO_KEEP=0`).

It covers every layer kind: raw-cache layers (compress ratio 0) and the compressed and
sparse layers (ratio 1 or 2), meaning compressor, indexer, candidate mask, sparse
selection and attention.

For every stage it recomputes the stage in float64 from the GGUF weights. It uses gguf-py's
numpy decoders, which are independent of both engines, and each engine's own traced inputs
to that stage:

    local_err(engine, stage) = || stage_engine - exact(stage; inputs_engine) || / || exact ||

So each engine is judged on the rounding it adds at that stage, not on what it inherited.

Some roundings are part of the design: the trained cache formats (FP8 E4M3 raw K, MXFP4
indexer q/k, NVFP4 compressed KV) and the F16 caches. The oracle applies the same rounding,
ported bit for bit in float32, to its own recompute. RoPE uses an exact float64 table.
`--rope-table engine` replays the direct engine's float32 table instead.

`compare-dsv4-traces.py` diffs the two traces directly, so its distances include everything
inherited from earlier stages. Use it to find where the engines first diverge. Use this
oracle to decide whether a stage's divergence is acceptable.

The tool's docstring refers to S8b and S8c. Those are the validation stages that produced
it. Its numbered list is the set of fixes made after the S8b review.

## Files

| File | Purpose |
|---|---|
| `dsv41-stage-oracle.py` | The oracle and its gate (gate v2). Its docstring is the reference for every rule. |
| `dsv41-trace-sink.py` | Linux only. Captures long prompts within a disk budget: it keeps only one layer window's trace files and hashes every file the engine writes. |
| `dsv41-trace-digest.py` | Digests a trace directory, or compares two digests, so two runs can be checked for bit-identity without keeping both. |
| `../tests/test_dsv41_stage_oracle.py` | 29 unit tests on synthetic inputs (no GGUF, traces or GPU). |
| `../tests/test_dsv41_trace_sink.py` | Sink tests (Linux only, skipped elsewhere) and digest tests (every platform). |

Requirements:

- Python 3.10 or later, with numpy.
- gguf-py, for the recompute mode: `pip install gguf`, or `GGUF_PY=<directory holding the gguf package>`. `--gate-from`, `--merge` and the unit tests do not need it.
- For long prompts, the sink also needs Linux and a tmpfs staging directory.

## 1. Capture a trace pair

Run `DeepSeek41DistributedProbe` (see its [README](DeepSeek41DistributedProbe/README.md))
once per engine, with the same tokens, flags and environment. The recorded runs used this
setup, plus host-specific device selection and thread counts:

```sh
probe=eng/validation/DeepSeek41DistributedProbe/bin/Release/net10.0/DeepSeek41DistributedProbe.dll
run() {   # ENGINE_ENV OUTPUT BACKEND
  env MAX_CONTEXT=65536 KV_CACHE_DTYPE=f16 TS_PREFILL_WARMUP=0 TS_DSV4_UBATCH=256 \
      TS_DSV41_ENGRAM_DEVICE=0 TS_DSV41_YOCO_KEEP=0 TS_N_CPU_MOE=0 "$1" \
    dotnet "$probe" --model "$MODEL" --tokens "$TOKENS" --weights-id "$WEIGHTS_ID" \
      --output "$2" --layer-split 2 --chunk 256 --decode-count 4 --backend "$3"
}
run TS_DSV4_CUDA_TRACE_DIR="$T/cuda-trace" "$T/cuda-probe" cuda
run TS_DSV41_TRACE_DIR="$T/ggml-trace"     "$T/ggml-probe" ggml_cuda
```

- **Set `TS_DSV4_UBATCH` explicitly, and set `--chunk` to the same value.** `ggml_cuda`
  otherwise picks its own ubatch. The probe runs the prompt three times: as one call, as
  `--chunk`-sized calls, then as one call again. Each pass overwrites the trace files of the
  calls it shares with the earlier passes. If the call boundaries differ between passes, stale
  files remain, and the oracle refuses the trace with `chunks do not tile the positions`.
- **`--cuda-probe` and `--ggml-probe` are required** whenever an Engram layer is in the
  judged window. These are the probes' output directories: the oracle recomputes each
  engine's Engram lookup from the token ids in that engine's `probe.json`.
- **Expect large traces.** Neither engine's trace can be filtered by layer or stage. At 773
  tokens the direct engine writes about 455 KB per token per layer, and `ggml_cuda` writes
  282-384 KB. A 4K-token trace of ten layers is about 25 GB. Above a few hundred tokens,
  capture in layer windows (section 3).

## 2. Judge it

```sh
GGUF_PY=/path/to/gguf-py python -I eng/validation/dsv41-stage-oracle.py \
  --ggml "$T/ggml-trace" --cuda "$T/cuda-trace" \
  --cuda-probe "$T/cuda-probe" --ggml-probe "$T/ggml-probe" \
  --model "$MODEL" --json "$T/oracle.json" > "$T/oracle.txt"
python -I eng/validation/dsv41-stage-oracle.py --gate-from "$T/oracle.json"   # re-gate a report
```

What it prints:

- one line per stage and engine;
- the per-stage table: `cuda local`, `ggml local`, `cuda/ggml`, `ggml model`, `ggml/model`;
- one `PASS` / `FAIL` / `FLAG` / `info` line per check;
- a `gate: PASS` or `gate: FAIL at ...` line.

Exit status:

- The recompute run exits 0 whenever it completes. Its verdict is the `gate:` line, and
  `gate.pass` in the JSON report.
- `--gate-from` and `--merge` exit 1 when the gate fails.

Options:

- `--layers` limits the report. Earlier layers are still recomputed unless `--state-in` is given.
- `--moe-tokens a:b,...` restricts the routed-MoE oracle, which is the slowest part, to some
  token ranges. The recorded runs picked ranges that cover positions past the pruning budget,
  the multi-row prefill tail and every decode step.
- `--segments` gates each call separately (default `calls`), or explicit `a:b` ranges.
- `--discrim-stride` subsamples the attention discrimination self-test (default: automatic).
- `--factor`, `--slack`, `--exact-bound`, `--model-factor` and `--replay-factor` set the gate
  constants below.
- `--mutate` is for testing only: it perturbs the oracle's own semantics on purpose, to show
  that the two-sided rules catch an oracle error.

Measured run times on the A40 host used for the recorded runs (`OPENBLAS_NUM_THREADS=14`):

- layers 0-9 at 773 tokens: about 20 minutes;
- a two-layer window at 4,204 tokens: 6-7 minutes.

## 3. Long prompts: layer windows through the trace sink

A long prompt is captured and judged a few layers at a time. For each window `A..B`:

1. Capture both engines through the sink, keeping only the window's files.
2. Judge the window. Pass `--layers A,...,B` and `--state-in` (the previous window's
   `--state-out`). The state file carries everything that crosses layers: the hidden streams,
   the delayed hyper-connection collapse, the compressed and indexer caches, the selections
   (as sets and as the verbatim lists), and the candidate sets.
3. Delete the window's traces.

Then join the windows and gate the whole prompt:

```sh
python -I eng/validation/dsv41-stage-oracle.py --merge w0-1.json w2-3.json ... --json oracle.json
```

The probe runs with its trace directory set to a tmpfs stage. The sink takes each file there
as soon as the writer closes it (inotify, plus a rescan for files quiet for 3 s):

- It hashes the file.
- If the name matches a `--keep` pattern, it copies the file to `--dest`. The copy replaces
  any earlier version, so the last pass wins.
- It deletes the file from the stage.

```sh
python3 -I eng/validation/dsv41-trace-sink.py --stage /dev/shm/dsv41-stage --dest "$T/cuda-trace" \
  --keep '^p[0-9]+_blk(04|05)_' \
  --digest "$T/cuda.sink.digest" --log "$T/cuda.sink.log" --stop-file "$T/cuda.sink.stop" \
  --pid-file "$T/cuda.probe.pid" --max-stage-bytes 8e9 --max-dest-bytes 6.5e9 \
  --index-ratio 2:2 --index-ratio 4:1 --prefix "$PREFIX" --chunk 256 --topk 512 &
# ... start the probe with TS_DSV4_CUDA_TRACE_DIR=/dev/shm/dsv41-stage, write its pid to the pid file,
# wait for it to exit, then:
touch "$T/cuda.sink.stop"; wait
```

Keep patterns for window `A..B`:

- **Direct engine:** `^p[0-9]+_blk(AA|...|BB)_`. In the window that starts at layer 0, also
  keep `^p[0-9]+_embedding\.f32$`.
- **`ggml_cuda`:**
  `^p[0-9]+_v41\.(AA|...|BB)\.(attn_input|qr|query|raw_kv|latent|index_scores|index_top|attn_output|ffn_output|engram_output|engram_kv|output)\.(f32|i32)$`.
  Also keep the next layer's `engram_output`, which is layer B's hidden state, unless B is the
  last layer.

Flow control and the guard:

- **Flow control.** When the stage holds more than `--pause-bytes` (default 4e9), the process
  named in `--pid-file` is stopped with SIGSTOP. It resumes with SIGCONT once the stage drains
  below `--resume-bytes` (default 1.5e9). Pausing does not change the engines' numbers.
- **Guard.** If the stage exceeds `--max-stage-bytes`, or the kept files exceed
  `--max-dest-bytes`, the guard kills that process and the sink exits with status 3.

A run's digest has two parts:

- the digest tool's output over the kept files:
  `dsv41-trace-digest.py digest DEST --probe PROBE_DIR --model MODEL.gguf`;
- the sink's `--digest`, which lists every file it dropped.

Together they cover every file the run wrote. To check that a window's capture repeats an
earlier pass bit for bit, run `dsv41-trace-digest.py compare A.digest B.digest`.

The direct engine's sparse-selection dumps hold memory it never writes:

- `index_scores` holds only each query's visible rows;
- `index_top` holds only `index_count` ids per row.

Both tools therefore hash those dumps over the written part only. On the sink, this needs:

- `--index-ratio LAYER:RATIO` for each index layer in the trace (the ratio is from
  `deepseek41.attention.compress_ratios`);
- `--prefix`: the prompt length minus the decode count;
- `--chunk`: the ubatch;
- `--topk`: the indexer's top-k.

## Gate v2

All constants below are defaults that can be overridden. "Paired" stages are those both
engines trace.

- **Paired stages.** The direct engine's local error must be at most `1.25 x ggml's + 1e-6`.
  This holds in aggregate, and per segment (by default each call is a segment: each ubatch,
  the multi-row prefill tail and each decode step).
  - One exception: a `raw_k` segment over the factor is FLAGGED rather than failed when all
    three hold:
    - every element that differs from the replay is within the FP8 noise envelope
      (`2 x step + 4 x sigma`: a bin flip, or a group-scale change by one power of two, plus
      that row's q8_1 pre-quantization noise);
    - the same holds for ggml;
    - a one-sided binomial test does not reject "the direct engine flips at most 1.25x as
      often as ggml" (`p >= 0.01`).
- **F32 stages.** These are stages the direct engine computes in F32 from end to end, such as
  `attn_input`, `attn_core`, `ffn_input`, `router`, `selw`, the Engram stages and `hidden`.
  Each must be at most 1e-6, or within the paired factor where ggml traces the same stage.
  The bound applies per segment as well.
- **Committed cache formats.** For `comp_k` and `lid_k`, every element that differs from the
  format replay of the engine's own input must be a tie within the stated bound.
- **Selections.** Each selection is checked as a list.
  - The list's length must be `min(k, visible)`. For ggml, whose list is padded, this applies
    to its visible ids.
  - No duplicates, no invisible ids, and no pruned (-inf) rows while enough finite rows remain.
  - The list must be an exact top-k of the engine's own traced scores, within the derived F32
    bound.
  - The attention core is recomputed twice. `attn_core` uses the exact selection.
    `attn_core (own list)` uses the engine's verbatim list. A duplicate standing in for the
    boundary row therefore fails.
- **Candidate pruning.** Pruning must engage at the expected position, at every consumer
  layer, within the block budget. The block pin `pos // block` is asserted to be used only for
  ratio-1 sources.
- **Routing.** The picks must be unique, in range, and a top-6 of `sqrt(softplus(l)) + b`.
  This is checked two ways:
  - on the engine's own traced router logits, up to the derived F32 evaluation bound;
  - on the exact logits from its own `ffn_input`, up to that bound plus the F32 GEMM bound.

  There is no constant tie threshold.
- **Two-sided: ggml against a precision model of its own arithmetic.** The model is the
  oracle's exact stage with ggml-cuda's activation rounding:
  - q8_1 for quantized weights (MMQ above 8 rows, MMVQ at 8 or fewer; Q2_K's MMQ operand per
    64 values);
  - BF16/F16 for BF16/F16 weights above 3 rows;
  - F32 otherwise.

  The model must predict ggml's error: `ggml_local <= 1.25 x model_local + 1e-6`. It must also
  replay ggml elementwise: `rel(ggml, model) <= 0.4 x ggml_local`. F32 stages that ggml traces
  must be at most 1e-6 for ggml too. `ffn_out` at layers with a MoE weight type the model does
  not replay (anything but Q8_0, IQ2_XXS and IQ3_XXS) is reported, not bounded.
- **Semantic discrimination.** These checks apply to both engines.
  - Each engine's `attn_out` must fit the oracle's attention semantics better than every
    informative perturbation: sinks toggled, window -1 or +1, selection toggled, and
    compressed rows rotated at the other end of the block.
  - Each engine's `ffn_out` must likewise fit its MoE semantics better than: routing-weight
    normalization toggled, routed scale 1, the weakest pick dropped, the shared expert
    omitted, and the routed or shared SwiGLU clamp off.
  - A variant is informative when its effect is at least 0.1 times the baseline distance. It
    must then cost at least `d_variant / d_base >= 1 + (sqrt(1 + (effect / d_base)^2) - 1) / 2`.
- **Engram (layer 1).** The oracle recomputes each engine's lookup from its own token ids,
  using the `dsv41_engram.h` hash and the host lookup's dequantization, on both GGUF layouts.
  The direct engine's traced rows must match the recompute.

## Known limits

- **Untraced ggml stages.** ggml's intermediate compressed-path stages are not traced:
  compressor projections, indexer q/w/k, candidate mask and attention core. ggml is judged on
  its composite stages, which the replay rule ties to its own arithmetic.
- **Unreplayed ggml MoE types.** The precision model does not replay ggml's IQ1_S, IQ2_S,
  Q2_K or Q5_K routed-expert GEMMs. At those layers, `ffn_out`'s two-sided check is the MoE
  discrimination alone.
- **Aggregate-only checks on ggml.** ggml's two-sided checks and its F32 bound apply in
  aggregate only, so a drift confined to one of ggml's segments is not caught.
  - A paired stage or segment that ggml did not trace is printed as `info` and is not gated.
  - A direct-engine F32 segment over 1e-6 passes when it is within the paired factor of ggml's
    error on that segment.
- **Compressed-row replay past about 2,000 positions.** ggml rotates the compressed rows it
  commits with float32 RoPE angles before NVFP4 quantization. The oracle rebuilds ggml's rows
  from its traced latent with exact angles. Past about 2,000 positions, one float32 ulp of
  angle can decide an NVFP4 bin. So, at layers that read those rows, the replay share grows
  with position.
  - It reached 0.405 at layer 4 of the 4,204-token run, against the 0.4 limit.
  - The fix would be either to trace ggml's committed rows, or to replicate CUDA's float32
    `powf`/`cosf`/`sinf` bit for bit.
- **Stated limits, not derived bounds.** The factors 1.25 (error) and 0.4 (replay) are
  calibrated against a 64-token measurement (largest replay share 0.23). The discrimination
  rule is a stated statistical rule. The format-tie, selection and routing bounds, in
  contrast, are worst-case analytic bounds.

## Recorded results

Recorded October 30, 2026, on 2x A40 (driver 595.91.07, CUDA 13.2), with ggml `facdc44`
unmodified.

- **Model file:** a 10-layer truncation of unsloth's DeepSeek-V4.1-Flash-1017
  `UD-IQ2_XXS`, made with `dsv41-truncate-gguf.py`.
  - It keeps blocks 0-3, 20-24 and 36.
  - Its candidate budget is cut from 2,048 to 65 blocks, so pruning engages from token 521.
  - sha256 `fbff6ca5...`.
- **Engine build:** the judged build of the direct engine and the one this tool was
  committed with give bit-identical logits on the 2,003-token prompt with tracing off.
- **Configuration:** single process, `--layer-split 2`.

| Prompt (prefill + decode) | Layer windows | Checks PASS / FAIL / FLAG | Gate |
|---|---|---|---|
| 64 + 4 | 0-9 | 380 / 0 / 5 | PASS |
| 769 + 4 | 0-9 | 441 / 0 / 11 | PASS |
| 1,999 + 4 | 0-4, 5-9 | 446 / 0 / 4 | PASS |
| 4,200 + 4 | five windows of two layers | 447 / 1 / 6 | FAIL: `L4 attn_out ggml replay` (0.405 > 0.40) |

- **Direct engine.** Every check on the direct engine passed at every length.
- **Flags.** Every flag is an explained `raw_k` decode segment.
- **The single failure** is the two-sided replay check on ggml described under the
  compressed-row replay limit above. ggml/model is 1.00 at every layer, and every informative
  semantic variant fits both engines worse than the oracle.
- **Determinism.** Every window pass of the same prompt reproduced the first pass's files bit
  for bit (by digest).
- **Not covered:**
  - prompts longer than 4,204 tokens;
  - the full 1017's 2,048-block candidate budget;
  - the direct engine's batched multi-sequence decode;
  - distributed runs, YOCO and DSpark.

## Tests

```sh
python -m unittest eng/tests/test_dsv41_stage_oracle.py eng/tests/test_dsv41_trace_sink.py
python -I -B eng/tests/test_dsv41_trace_sink.py -v
```

On platforms other than Linux, the five `Sink` tests are skipped. They are not counted as
passes. `SinkElsewhere` checks that the sink refuses to run there. Run the `Sink` tests on
Linux, where they also need a `sleep` binary and `/proc`. The tests that write synthetic
index dumps (the two `Digest` tests and two `Sink` tests) need numpy and are skipped without
it; the oracle's tests need numpy throughout.

Both files run in CI on Linux, with numpy and gguf-py installed, beside
`test_dsv41_gguf.py` and `test_dsv41_truncate.py` (the "DeepSeek V4.1 GGUF, truncation and
per-stage oracle tools" step of `.github/workflows/pr-unit-tests.yml`).
