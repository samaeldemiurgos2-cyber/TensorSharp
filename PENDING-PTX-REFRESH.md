# Pending: CUDA 13 committed-PTX refresh (commit B)

Commit A (this change) retires CUDA 12: every build entry point rejects an nvcc older
than 13, the runtime loads only CUDA 13 libraries, and the refresh recipe now demands
`compute_75` built by nvcc 13.0 on Linux. The committed PTX itself is still the old
`release 12.8, V12.8.93` / `.version 8.7` / `.target sm_61` baseline. Everything below
waits for the refresh, which itself waits for the concurrent DeepSeek-kernel stage
(`tensorsharp_dsv4_kernels.cu`) to merge: its source already defines one entry more
than the committed `tensorsharp_dsv4_kernels.ptx`.

Delete this file in commit B once every item is done.

## 1. Source edit that must ride with the refresh

`TensorSharp.Backends.Cuda/native/kernels/tensorsharp_dsv4_kernels.cu`, function
`ts_dsv4_dp4a` (cite the symbol; line numbers move): replace the
`#if __CUDA_ARCH__ >= 610 ... #else <scalar int8 emulation> #endif` body with

```cpp
// dp4a is native on every target CUDA 13 compiles (compute_75 and newer).
return __dp4a(a, b, c);
```

Keep the `__CUDA_ARCH__ >= 800` gate in `ts_dsv4_moe_mma_f32`. The same change was
already made in commit A for `dp4a_i8` (tensorsharp_kernels.cu) and `ts_q35_dp4a`
(tensorsharp_q35_kernels.cu).

## 2. Regenerate the five committed PTX modules (VM1, Linux, once)

```bash
python3 eng/install-cuda-redist.py --version 13.0.3 --dest /opt/cuda-13.0.3   # cuda_cccl alias fixed in A
/opt/cuda-13.0.3/bin/nvcc --version                                             # release 13.0, V13.0.88
dotnet build TensorSharp.Backends.Cuda/TensorSharp.Backends.Cuda.csproj -c Release \
  -p:TensorSharpUpdateCommittedPtx=true -p:CudaNvcc=/opt/cuda-13.0.3/bin/nvcc
head -12 TensorSharp.Backends.Cuda/native/ptx/*.ptx | grep -E 'release|\.version|\.target'
```

- Expected for all five (`tensorsharp_kernels`, `tensorsharp_dsv4_kernels`,
  `tensorsharp_glm_kernels`, `tensorsharp_q4e_kernels`, `tensorsharp_ltx25`):
  `// Cuda compilation tools, release 13.0, V13.0.88`, `.version 9.0`, `.target sm_75`,
  `.address_size 64`. Entry counts 172 / 56+ / 14 / 21 / 6.
- `native/ptx/tensorsharp_q35_kernels.ptx` must not appear (the csproj now marks q35
  `CommittedBaseline=false` and never copies it).
- Record `sha256sum` of `native/kernels/*.cu *.cuh` on the dev box and VM1 first; they
  must match (include the new `tensorsharp_iq1s_tables.cuh` if present).
- Negative check: the same command with `-p:CudaNvcc=/usr/local/cuda/bin/nvcc` (13.2)
  must stop before compiling with "must be built by nvcc 13.0 (found 13.2)".
- Commit A's `long` -> `long long` change in `tensorsharp_kernels.cu`
  (`ts_quantize_q8_1_rows_f32`, `ts_quantize_q8_1_split_rows_f32`, `ts_silu_mul_f32`)
  keeps the Linux PTX parameters at `.u64` (Windows nvcc now also emits `.u64` for
  `ts_silu_mul_f32_param_3`; verified with nvcc 13.4.92 on the dev box).

## 3. InferenceWeb.Tests/CudaKernelAvailabilityTests.cs

(The refresh instructions in the doc comment and in the
`CommittedPtx_HasAnEntryForEveryKernelInTheSource` failure message were already
updated to nvcc 13.0 / `-p:CudaNvcc` in commit A.)

- File header comment (line ~10): `it is a compute_61 baseline` -> `it is a compute_75 baseline`.
- Rename `CommittedBaseline_IsCompute61_AndNeverEnablesArchGuardedKernels` ->
  `CommittedBaseline_IsCompute75_AndNeverEnablesArchGuardedKernels`.
  - `Assert.Equal(61, CudaModule.ParsePtxTargetArch(...CommittedPtx...))` -> `Assert.Equal(75, ...)`.
  - `foreach (int ccMajor in new[] { 6, 7, 8, 9, 10, 12 }) Assert.False(CudaKernels.ArchGuardedKernelsUsable(61, ccMajor));`
    -> `foreach (int ccMajor in new[] { 7, 8, 9, 10, 12 }) Assert.False(CudaKernels.ArchGuardedKernelsUsable(75, ccMajor));`
  - Drop the now-duplicate `Assert.False(CudaKernels.ArchGuardedKernelsUsable(75, 8));`.
- `[CudaFact] CommittedBaselineModule_DisablesTensorCoreKernels_LocalBuildKeepsThem`:
  comment `the driver JITs its compute_61 PTX` -> `compute_75 PTX`, `(61)` -> `(75)`;
  `Assert.Equal(61, baseline.CompiledArch)` -> `Assert.Equal(75, baseline.CompiledArch)`.
- New `[Theory]` over the five modules: the PTX head contains
  `Cuda compilation tools, release 13.0,`, matches `(?m)^\.version 9\.0\r?$`, and
  `CudaModule.ParsePtxTargetArch(bytes) == 75`.
- New `[Fact]`: `TensorSharp.Backends.Cuda/native/ptx/tensorsharp_q35_kernels.ptx` does not exist.
- Run the CudaFact on the VMs' 595.91 (CUDA 13.2) driver: it is the first real proof that
  every CUDA 13 driver JITs the ISA 9.0 baseline (verified so far only on a 13.5 driver).

## 4. Docs that describe the refreshed baseline

- `DEVELOPMENT.md` (Build the native GGML library section, after the refresh command):
  replace the opening `A refresh must target compute_75 ... can JIT; a driver rejects PTX
  newer than itself.` with `The committed baseline targets compute_75, the oldest
  architecture CUDA 13 compiles, and is built by nvcc 13.0 on Linux, whose PTX (ISA 9.0)
  every CUDA 13 driver (R580 or newer) can JIT; a driver rejects PTX newer than itself.`
  Mirror in `DEVELOPMENT_zh-cn.md` (`刷新必须以 compute_75 ... 为目标，并且必须由 nvcc 13.0 构建`
  -> `提交的基线以 compute_75 为目标，并在 Linux 上由 nvcc 13.0 构建`).
- `docs/design/nemotron-super-vl-bolt-support.md`
  - Lines ~353-356: replace "The committed PTX baseline targets `compute_61`, and **nvcc 13
    dropped `compute_61`** ... cannot be refreshed on a Spark. Refresh it on a host with a
    CUDA 12.x nvcc (`-p:TensorSharpUpdateCommittedPtx=true -p:CudaArch=compute_61`):" with
    "The committed PTX baseline targets `compute_75` and is refreshed with nvcc 13.0 on Linux
    (`-p:TensorSharpUpdateCommittedPtx=true -p:CudaNvcc=<13.0>/bin/nvcc`); any host with a
    13.0 nvcc can refresh it, a Spark included once one is installed side by side:"
    (keep the `CommittedPtx_HasAnEntryForEveryKernelInTheSource` sentence that follows).
  - Lines ~407-409: "the committed compute_61 baseline PTX used without nvcc: it was refreshed
    with CUDA 12.8," -> "the committed compute_75 baseline PTX used without nvcc: it was
    refreshed with CUDA 13.0,".
- `docs/models/nemotron.md` ~631-632: "the committed compute_61 baseline PTX used when nvcc
  is absent (refreshed with CUDA 12.8)" -> "the committed compute_75 baseline PTX used when
  nvcc is absent (refreshed with CUDA 13.0)". `docs/models/nemotron_zh-cn.md` ~436:
  "仓库内 compute_61 基线 PTX（已用 CUDA 12.8 刷新）" -> "仓库内 compute_75 基线 PTX（已用 CUDA 13.0 刷新）".
  Keep the "review's measurements (2x A40, CUDA 12.8)" line: it is a measurement record.
- `docs/embeddings.md` ~585-590 (dated "Validation follow-up: 2026-10-14"): keep the record,
  change "Committed-baseline refreshes force that target" -> "Committed-baseline refreshes
  force the portable target", and append "(That compute_61 / CUDA 12.8 baseline was later
  retired: the committed modules now target compute_75 and are refreshed with CUDA 13.0;
  see Development.)"
- `eng/validation/README-ltx25.md` ~1307-1312: keep the recorded `...:sm_61` module hash as
  dated evidence and add one line: "Pre-retirement sm_61 / CUDA 12.8 baseline; a rerun needs
  the refreshed module's hash and `:sm_75`."
- `eng/validation/DirectCudaModelProbe/README.md` ~46-49: keep as dated evidence and add
  "(The baseline has since moved to compute_75 built by CUDA 13.0.)"

Code comments in commit A (`CudaModule.CompiledArchOf`, `CudaKernels`, `Dsv4Kernels.HasMoeMma`,
`CudaMoe`, `CudaDriverApi.CU_FUNC_ATTRIBUTE_PTX_VERSION`) already say compute_75; they become
exact once the refresh lands and need no further edit.

## 5. Not in A or B: optional follow-ups

- [D] `.github/workflows/release-binaries.yml`: add `80-real` (A100/A30 get no ggml-cuda
  image today) and `100-real` to `TENSORSHARP_GGML_NATIVE_CUDA_ARCHITECTURES` (both x64
  jobs), and `100` to `CudaFatbinArchitectures`, so no supported GPU depends on JIT of
  13.4 PTX on an R580-R595 driver. Commit A's q35 cp.async/ldmatrix gate is what makes the
  existing `75,...` fatbin list buildable.
- ggml_cuda startup diagnostics in `ggml_ops_core.cpp` (driver/runtime versions when 0
  devices are found; a per-device `cudaFuncGetAttributes` probe), deferred because that file
  carries another stage's edits.
- `CUBLAS_TENSOR_OP_MATH` (CudaCublasHandle.cs) is deprecated in cuBLAS 13; unchanged
  because switching the math mode changes FP32 GEMM numerics.
