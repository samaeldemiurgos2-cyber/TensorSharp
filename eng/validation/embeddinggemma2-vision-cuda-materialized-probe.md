# Full-score F32 vision attention screen

This standalone validation tool changes no production/native build or upstream
file. Its one candidate computes exact F32 QK through a private pedantic cuBLAS
handle, stable F32 row max/exp/sum, F32 V times **unnormalized** exponentials,
then F32 division and an ordinary downstream `0.5` scale. No reduced precision,
TF32, bounded exponential approximation, or fixture-specific arithmetic is used.
The baseline block is copied byte-for-byte from the owned production source
SHA `1a610f3f4b264c8fc7e53b1c01f0eba04899f8b7e5457a19f0d4b89d16454b71`;
the runner checks both that entire file and the copied block before compilation.

Materialized vision attention was already used by the preserved v4/v5 runtime,
with <=512-query tiles, V packing and output SET/cont nodes. The distinct screen
here is complete-query score storage and direct strided QK/PV products without
V packing for the actual token-major tower layout. There is no predicted speed
or numerical pass. Existing materialized arithmetic and previous failed kernels
are separate evidence and cannot qualify this candidate.

All eight original head64 fixtures retain their exact values, scales, full
Double two-pass reference, `8e-5 + 2e-5 * abs(reference)` component bound,
source-bit `2e-6` bound and exact uniform `40000.0625` downstream result. All
candidate outputs are also compared completely with the copied baseline at the
same component bounds. Three actual N1089/2304/2340 H12/B1 token-major shapes use
Double checks on complete selected query rows across all heads, plus complete
baseline comparisons. Three additional canonical tail/padded/interleaved batch
cases keep complete Double checks. Saved baseline and candidate `.f32` files
contain every downstream output, with runner byte hashes. They use deterministic
synthetic inputs, not a learned model or semantic corpus.

The fifteenth fixture is an explicit range refusal, not candidate arithmetic
passing. Unnormalized PV needs a conservative `maxAbs(V) * keys <= FLT_MAX`
bound; normalized-first attention has a broader representable range. The tool
checks logical device inputs before dispatch, including finite Q/K/V, scans
score/sum/output status, and preserves published output on refusal/failure.
Original 80000.125 fixtures remain eligible and their gates are unchanged.
Admission checks cover dtype, shapes, non-overlapping output, positive/nonaliasing
view strides, pointer/size overflow, finite scale, and a 512-MiB score budget.
Singleton rows with an unused too-small leading dimension use the exact F32 pack;
unused huge singleton-head strides are canonicalized, and every cuBLAS stride
has a checked signed-long-long cast. Directed controls cover those boundaries.
Constructor budget refusal occurs before score allocation. No device allocator
failure/OOM is injected or exercised by that budget control; actual allocator
failure coverage remains unexecuted. Other allocation/API
failures remain unavailable/failing setups, never fallback success.

N2304 and N2340 need 243 and 250.65 MiB of score storage respectively. This is
not a total device-memory forecast: denominator, guarded input/output, optional
exact F32 packs and cuBLAS allocations are additional. One private stream/handle
and each fixture's device buffers live through completion. Every upload, pack,
GEMM, kernel and status transfer uses that stream. Nothing is cached by model or
raw-pointer identity. Every initial, warm and final numerical/integrity gate
contributes to the aggregate; failures and earlier setup negatives are retained.

Host controls on Mac build only layout/checked-size logic; CUDA code is excluded:

```sh
clang++ -x c++ -std=c++17 -O2 -Wall -Wextra -Werror \
  -DTSG_ATTN_HOST_CONTROLS_ONLY \
  eng/validation/embeddinggemma2-vision-cuda-materialized-probe.cu \
  -o /absolute/ignored/host-controls
/absolute/ignored/host-controls --host-controls
```

Actual CUDA qualification requires unchanged upstream revision
`3612c0af93f2cfc8f048295300a61187723b12ae`, SM86, explicit compiler/CUDA/native
identities, before/after guards, and independent source review. The quality-only
runner invocation is:

```sh
python3 eng/validation/embeddinggemma2-vision-cuda-materialized-probe-run.py \
  --mode quality --output docs/validation/materialized-attention-quality-new \
  --nvcc /usr/local/cuda/bin/nvcc \
  --host-compiler /usr/bin/x86_64-linux-gnu-g++-13 \
  --cuda-root /usr/local/cuda \
  --native /absolute/frozen/libGgmlOps.so --expected-native-sha256 FROZEN_SHA \
  --ggml /absolute/unchanged/ggml \
  --baseline-source /absolute/TensorSharp.GGML.Native/ggml_ops_attention_precision.cu \
  --expected-baseline-source-sha256 1a610f3f4b264c8fc7e53b1c01f0eba04899f8b7e5457a19f0d4b89d16454b71
```

Generated output must be fresh and ignored. Default uses actual Git ignore
checks. A copied source root may explicitly use the reviewed paired
`--copied-source-ignore-policy` and
`--expected-copied-source-ignore-policy-sha256` flags, with genuine unchanged local
Git proof and matching root `.gitignore`; partial flags/nested overrides reject.
A root containing any `.git` path is refused by that copied route. If the VM's
canonical tree has inert Git metadata, use a fresh source-only isolated root
inside actual ignored evidence, copy exact source/runner/README/.gitignore/proof
bytes, and record canonical-to-private hashes. Never delete or fake metadata.

The runner binds source/README/self, original owned baseline, unchanged ggml,
selected compiler/CUDA tools, dependency closure, frozen native, explicit CUDA
libraries and actual ldd library paths. The coordinator should add actual driver
libraries with `--additional-guard`. It captures PTX/SASS as presence metadata,
not proof of Tensor Core use. It preserves a produced negative quality report
and its hash even when the child exits nonzero. Unavailable/failed scenarios are
never counted as numerical/device passes.

Only after every quality gate passes may a separately hash-bound independent
compiled layout/F32/API review authorize `--mode benchmark --qualified-build DIR
--instruction-review JSON --expected-instruction-review-sha256 SHA`. The review
must name a reviewer, set `reviewed=true` and
`materialized_f32_layout_and_pedantic_calls=true`, and bind exact source,
executable, PTX and SASS hashes. The exact qualified executable is reused;
new compile inputs are prohibited. Benchmark also performs a complete untimed
quality preflight before any timings. ABBA uses W3/R20/minimum0.1 seconds, saving
raw samples for the three actual shapes. Resident inputs/scratch/handle exclude
cold allocation/upload and quality readbacks, but include range/status checks,
packing, QK, exp/sum, PV, division, downstream scale and completion. This is a
component experiment; no raw tower, pooled HF/media, HTTP, llama parity or
production promotion follows from it.

The API/layout design follows the existing owned
`CudaEmbeddingAttentionPlan.cs` and
[NVIDIA cuBLAS 12.8.1 documentation](https://docs.nvidia.com/cuda/archive/12.8.1/cublas/index.html):
pedantic F32, element-count batch strides, independent output matrices, and
stream/workspace ownership. These contracts promise neither bit identity nor
our performance. Any later native candidate needs the original independent raw
vision and full pooled/protocol/lifecycle gates plus matched complete service
measurements before production selection.
