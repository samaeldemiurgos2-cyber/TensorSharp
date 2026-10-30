# EmbeddingGemma2 CUDA weight-layout probe

This standalone .NET10 tool compares cached decoded GGUF weights stored as
`[in,out]` (the encoder's current cuBLAS `N,N` call) and `[out,in]` (`T,N`). Both
use F32 input, weight, result, `alpha=1`, `beta=0`, compute77
(`CUBLAS_COMPUTE_32F_FAST_TF32`), and the default cuBLAS algorithm. It imports the
CUDA driver and cuBLAS ABI directly and has no TensorSharp project references,
model routing changes, native rebuild, or PTX compilation.

It reads the exact pinned model's Q8_0/BF16/F32 weight matrices for local/global
QKV and attention, joined gate/up, down, PLE input/projection, output, and the
BF16 initial PLE projection. Each matrix is measured with13,29,256,512,8192 rows.
Decode/transposition, allocation, and upload are outside the warm GEMM timings
and reported separately. Layout order alternates between measurements.

GPU timing uses CUDA events on one stream. The event interval may contain idle
time from host submissions. Host dispatch and completion wall time are reported
separately; these are microprobe timings, not model/HTTP latency. Actual loaded
driver/cuBLAS library hashes, runtime, binary/source hashes and device are saved.

Independent double and ordered F32 dot products check every small quality case
and sampled outputs of every real shape. Small cases include nonzero tensor
offsets/canaries, nonaligned dimensions, activations above the F16 finite range,
and an exactly representable near-cancellation residual. TF32 error is bounded
against the sum of absolute products; it is explicitly approximate, especially
under cancellation. Compute69 (`CUBLAS_COMPUTE_32F_PEDANTIC`) is run separately
for the small quality cases and must retain the binary residual exactly.

Build locally without touching shared model/backend binaries:

```sh
dotnet build eng/validation/EmbeddingGemma2CudaLayoutProbe -c Release -o artifacts/embeddinggemma2/cuda-layout-probe
```

Execute only in a coordinated quiet Linux CUDA 13 window, preserving reports:

```sh
dotnet artifacts/embeddinggemma2/cuda-layout-probe/EmbeddingGemma2CudaLayoutProbe.dll \
  --model /workspace/models/embeddinggemma-2-Q8_0.gguf \
  --source-dir eng/validation/EmbeddingGemma2CudaLayoutProbe \
  --output docs/validation/embeddinggemma2-reference/cuda-layout-probe/report.json \
  --warmup 3 --repeats 9 --calls 4
```

The tool refuses to replace an existing report and exits nonzero on failed ABI,
shape, arithmetic or output-canary checks. `completed=true` is written only
after device allocations/handles/context are disposed. `--quality-only` limits
execution to16 primitive layout/compute cases. `--weights-only` makes no CUDA
calls and emits hashes of decoded/transpose matrices for an independent reader
cross-check; the report records `cuda_executed=false` and its narrower scope.
`validate_weights.py --candidate REPORT --output NEW_REPORT` performs that
cross-check using unchanged gguf-py and NumPy (`PYTHONPATH=llama.cpp/gguf-py`).
Production use still requires
the unchanged eight-case/8192 oracle, batch32 consistency, full16 HTTP and model
lifetime gates; a microprobe win does not establish an encoder speedup.
