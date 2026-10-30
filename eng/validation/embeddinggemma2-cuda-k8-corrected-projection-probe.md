# K8 corrected FP16 projection screen

This is an isolated validation tool, with no production, CMake, ggml or global
CUDA policy changes. The initial implementation is an untuned CTA32×32 with
four warps and MMA16×8×8. It tests a distinct accumulation method, rather than
another set of whole-K cuBLAS residual products. It makes no speed prediction.

The method follows [Ootomo and Yokota's paper](https://arxiv.org/html/2203.03341v3)
and the author's pinned CUTLASS fork
[268162f4bf652fdfebd3d132ebcf55a3986fe570](https://github.com/enp1s0/cutlass/tree/268162f4bf652fdfebd3d132ebcf55a3986fe570).
The used [FP16 K8 overload](https://github.com/enp1s0/cutlass/blob/268162f4bf652fdfebd3d132ebcf55a3986fe570/include/cutlass/arch/mma_sm75.h#L280-L325)
resets the dominant accumulator on every K8 instruction, adds that partial with
FP32 arithmetic outside MMA, and retains the two scaled cross corrections until
the final `/2048` merge. The [multistage caller](https://github.com/enp1s0/cutlass/blob/268162f4bf652fdfebd3d132ebcf55a3986fe570/include/cutlass/gemm/threadblock/mma_multistage.h#L362-L371)
initializes correction once; its final merge is at lines532–535.
The standalone source retains the full NVIDIA BSD3 notice. This is not the
author's tuned CTA128×64×32, three-stage CUTLASS implementation. A K16 WMMA
rewrite would change the accumulation granularity and is not used here.

Each original F32 operand is split into RN-even half high plus an RN-even half
scaled residual. The host oracle searches adjacent binary16 values with Double
distances; it does not reuse the device converter. Tests include all finite half
values, midpoint ties and adjacent F32 values, signed zero, subnormal transitions,
near-overflow, infinity and NaN classification. Two half components do not
guarantee exact F32 reconstruction; each matrix reports reconstruction error and
full host/device pack-bit agreement. The [PTX specification](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma)
does not promise MMA accumulation order, rounding or subnormal handling, so
published accuracy and API success cannot replace the numerical gates.

The input/output layout is `W[M,K]`, `X[N,K]`, `Y[N,M]`, all row-major F32,
batch1. First scope is SM86 only. Actual projection families are `(M,K)` equal
to `(768,768)`, `(3072,768)` and `(768,3072)`, each at N36/108/1089/2304/2340.
All six original prior residual-probe fixtures and seven direct tail fixtures
are retained. The 80,000 fixture is explicitly FP16-unavailable; its existing
pedantic Double failure remains reported. Expected refusal is a control pass,
never a numerical FP16 pass. Nonfinite/overflow operands and invalid layout
leave published output unchanged; the forced internal range-control launch may
write only private scratch, whose guards and source integrity are checked.

Every eligible case retains the original sampled Double bound
`2e-6 + 2e-6*abs(reference)` **and** the complete pedantic F32 baseline bound.
Original4096 evenly spaced samples are preserved, with tile boundary additions;
small tail fixtures use complete Double dots. Initial, every warmup and final
checks are ANDed with maxima/counts retained. All output entries, source bytes,
prefix/suffix guards and status canaries are checked. The same nonblocking
private stream carries uploads, reset, kernel and status readback; pageable host
buffers stay alive through completion. A dedicated pedantic cuBLAS handle is
destroyed after complete work. Checked cleanup failures make the report fail.

Quality is the default mode. Timing requires all eligible cases to pass first,
then W3/R20/minimum0.1seconds ABBA (the bounded minimum is explicitly approved
for this screen). Every timed call includes on-the-fly W/X packing, range/status,
K8 reset and FP32 correction/merge, submission and stream completion. Allocation,
input upload and quality readbacks are outside timing. Only complete caller wall
latency is currently measured; no kernel-only duration or service gain is inferred.
The repeated quality/cold setup cost is not claimed to represent model loading.

Build and run local host controls without any CUDA qualification:

```sh
clang++ -x c++ -std=c++17 -O2 -Wall -Wextra -Werror \
  -DTSG_K8_HOST_CONTROLS \
  eng/validation/embeddinggemma2-cuda-k8-corrected-projection-probe.cu \
  -o docs/validation/k8-host-controls
docs/validation/k8-host-controls
```

The Linux coordinator must serialize the guarded runner after existing immutable
benchmarks finish. Use an unchanged upstream checkout and frozen native hash:

```sh
python3 eng/validation/embeddinggemma2-cuda-k8-projection-run.py \
  --mode quality --output docs/validation/k8-quality-new \
  --nvcc /usr/local/cuda/bin/nvcc \
  --host-compiler /usr/bin/g++ --cuda-root /usr/local/cuda \
  --native /absolute/frozen/libGgmlOps.so \
  --expected-native-sha256 ACTUAL_FROZEN_SHA256 \
  --ggml /absolute/unchanged/ggml
```

The runner guards source/README/runner, upstream tracked files/revision, frozen
native, compiler/CUDA tools, preprocessor closure, explicit CUDA libraries and
ldd-resolved ELF libraries before/after. Extra known dynamically loaded inputs
can be bound with repeatable `--additional-guard`. It does not claim arbitrary
unenumerated implicit inputs are covered. All generated outputs must use a fresh
ignored directory. Setup failures and unavailable CUDA stay unqualified.

For a copied TensorSharp source root with no top-level Git metadata, the normal
Git ignore check intentionally refuses to run. An explicit alternative accepts
an unchanged, SHA256-bound policy produced in the actual local Git repository:
`--copied-source-ignore-policy /absolute/policy.json` together with
`--expected-copied-source-ignore-policy-sha256 ACTUAL_POLICY_SHA256`.
Schema1 requires `checked_in_git_repository=true`, the actual `gitignore_sha256`,
and `ignored_prefixes={"docs/validation":true,"artifacts":true}`. The complete
proof, including its actual `git check-ignore` observations, is retained unchanged.
The copied root's `.gitignore` must have those exact bytes; descendant `.gitignore`
rules through the output parent are rejected on both logical and resolved paths.
Traversal and aliases outside the allowed evidence prefixes are rejected.
Policy and root `.gitignore` are guarded through compilation/execution, and the
expected proof hash is rechecked before work. This creates no fake Git metadata
and does not weaken the separate unchanged upstream ggml Git check. Partial flag
pairs reject. Real Git source roots keep the original default Git route.

Quality captures PTX, SASS and ptxas resources. Presence of an MMA or FADD string
is only metadata. Before timing, independently inspect the K8 MMA, zero HH per
K8, external RN FADD placement, persistent cross order and final merge in the
compiled artifact. A JSON instruction review must contain `reviewed=true`, a
nonempty `reviewer`, `k8_mma_and_external_rn_order=true`, and exact
`source_sha256`, `executable_sha256`, `ptx_sha256`, `sass_sha256` fields. Then
reuse that exact quality executable without rebuilding:

```sh
python3 eng/validation/embeddinggemma2-cuda-k8-projection-run.py \
  --mode benchmark --output docs/validation/k8-benchmark-new \
  --qualified-build docs/validation/k8-quality-new \
  --instruction-review /absolute/instruction-review.json \
  --expected-instruction-review-sha256 ACTUAL_REVIEW_SHA256
```

If quality used the copied-source policy, benchmark must explicitly supply the
same policy flags/bytes as well; switching ignore-policy routes is rejected.

All actual model, raw-row, complete pooled HF/controlled and both-order tower/HTTP
gates remain separate prerequisites for any future integration. No current tool
result promotes a production path, changes quality tolerances, or qualifies
unavailable/failed fixtures. QK/PV and alternative algorithms are out of scope.
