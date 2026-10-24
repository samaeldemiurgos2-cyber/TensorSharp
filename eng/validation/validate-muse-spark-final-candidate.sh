#!/usr/bin/env bash
# PLANNED PROCEDURE. Run manually only after campaigns stop; never launches/kills a server.
# Builds into NEW_PROOF_DIR, leaving baseline output directories untouched.
set -euo pipefail
if (( $# < 2 || $# > 3 )); then
  printf 'Usage: %s /workspace/evidence/NEW_PROOF_DIR APPLIED_WORKING_TREE_MANIFEST [Q3_FIRST_SHARD]\n' "$0" >&2
  exit 2
fi
root=/workspace/TensorSharp
python=/workspace/venv/bin/python
[[ -x "$python" ]] || { echo 'Missing validation Python virtual environment.' >&2; exit 2; }
guard="$root/eng/validation/muse-spark-candidate-validation.py"
micro="$root/eng/validation/benchmark-muse-spark-cpu-experts.py"
warm="$root/eng/validation/warm-muse-spark-cpu-experts.py"
parent_manifest=$2
model=${3:-/workspace/models/muse-spark/UD-Q3_K_XL/Muse-Spark-Open-816B-A42B-UD-Q3_K_XL-00001-of-00010.gguf}
[[ -f "$parent_manifest" && -f "$model" ]] || { echo 'Missing final source manifest or Q3 model.' >&2; exit 2; }
[[ "$model" == *'/UD-Q3_K_XL/'*'-00001-of-00010.gguf' ]] || { echo 'This bounded plan requires published UD-Q3_K_XL.' >&2; exit 2; }
export PATH=/workspace/dotnet:/usr/local/cuda/bin:$PATH DOTNET_ROOT=/workspace/dotnet
export PYTHONPATH=/workspace/llama.cpp/gguf-py${PYTHONPATH:+:$PYTHONPATH}
export TS_GGML_CPU_THREADS=4 TS_CPU_MOE_THREADS=4 OMP_NUM_THREADS=4
export TS_MUSE_SPARK_STREAM_EXPERTS=0 TS_MUSE_SPARK_REUSE_CPU_MOE_SCRATCH=0
unset TS_TEACHER_TOKEN_EXPORT TS_MUSE_SPARK_FUSE_QK_PREP TS_MUSE_SPARK_FUSE_RESIDUALS
unset TS_MUSE_SPARK_RELEASE_GPU_SOURCE_PAGES TS_MUSE_SPARK_KV_UPLOAD_2D TS_MUSE_SPARK_REUSE_NORM_SCRATCH
"$python" "$guard" idle
mkdir -p /workspace/evidence
exec 9>/workspace/evidence/.muse-spark-final-candidate.lock
flock -n 9 || { echo 'Another candidate validation owns the lock.' >&2; exit 2; }
proof=$("$python" "$guard" fresh --proof "$1")
# The remote TensorSharp checkout is an applied archive without .git. Verify
# every supplied manifest file/hash before building; only ggml requires Git.
"$python" "$guard" freeze --root "$root" --proof "$proof" "$parent_manifest"
export LD_LIBRARY_PATH="$proof/native-build":/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
idle() {
  "$python" "$guard" idle
  local busy
  busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)
  [[ -z "$busy" ]] || { printf 'GPU compute processes remain: %s\n' "$busy" >&2; exit 2; }
  "$python" "$guard" verify --root "$root" --proof "$proof"
}
run() {
  local log=$1; shift
  idle
  # Only this owner shell holds the lock; build daemons must not inherit it.
  "$@" 9>&- > "$proof/$log" 2>&1
  idle
}
cd "$root"
run native-configure.log cmake -S TensorSharp.GGML.Native -B "$proof/native-build" \
  -DCMAKE_BUILD_TYPE=Release -DTENSORSHARP_GGML_NATIVE_ENABLE_CUDA=ON \
  -DTENSORSHARP_GGML_NATIVE_ENABLE_VULKAN=OFF -DTENSORSHARP_GGML_NATIVE_BUILD_TESTS=ON \
  -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
run native-build.log cmake --build "$proof/native-build" -j 12 --target \
  GgmlOps GgmlOpsMuseSparkMoeTest GgmlOpsMuseSparkCpuMoeScratchTest GgmlOpsMoeFusedBiasAliasTest \
  GgmlOpsAttentionDeviceCacheTest GgmlOpsResidentDeviceCopyTest GgmlOpsMuseSparkResidualTest GgmlOpsMuseSparkQKPrepTest \
  GgmlOpsMuseSparkKv2DUploadTest GgmlOpsMuseSparkNormGateTest GgmlOpsPagedAttentionLifecycleTest
run managed-rebuild.log dotnet build InferenceWeb.Tests/InferenceWeb.Tests.csproj -c Release -t:Rebuild --no-restore \
  -p:TensorSharpSkipGgmlNative=true -p:TensorSharpSkipMlxNative=true \
  -p:OutputPath="$proof/managed/" -p:AppendTargetFrameworkToOutputPath=false -p:AppendRuntimeIdentifierToOutputPath=false \
  -p:GgmlNativeBuildDir="$proof/native-build" -p:GgmlNativeBuildDirForTests="$proof/native-build"
cmp "$proof/native-build/libGgmlOps.so" "$proof/managed/libGgmlOps.so"
"$python" "$guard" pin-binaries --proof "$proof"
run native-regressions.log ctest --test-dir "$proof/native-build" --output-on-failure \
  -R '^(resident-device-copy|muse-spark-moe|muse-spark-cpu-moe-scratch|muse-spark-residual|muse-spark-qk-prep|muse-spark-kv-2d|muse-spark-norm-gate|moe-fused-bias-alias|attention-device-cache|paged-attention-lifecycle)' \
  --output-junit "$proof/native-regressions.xml"
# Both arena choices are separate native-library loads. QK/residual/2D/norm tests toggle
# their own managed opt-ins and assert actual execution, default-off and fallback.
for flag in 0 1; do
  for lane in cpu cuda; do
    filter='FullyQualifiedName~MuseSpark&FullyQualifiedName!~NativeCuda'
    if [[ "$lane" == cuda ]]; then filter='FullyQualifiedName~MuseSpark&FullyQualifiedName~NativeCuda'; fi
    run "$lane-arena$flag.log" env TS_MUSE_SPARK_REUSE_CPU_MOE_SCRATCH="$flag" TS_TEST_GGML_BACKEND="$lane" \
      dotnet vstest "$proof/managed/InferenceWeb.Tests.dll" "/TestCaseFilter:$filter" \
      "/Logger:trx;LogFileName=$lane-arena$flag.trx" "/ResultsDirectory:$proof"
  done
done
"$python" "$guard" tests --proof "$proof"
# Only the first layer is read. File-cache preparation and per-process warmup are
# outside native timings. Four processes counterbalance off/on order; no server.
run expert-prewarm.log "$python" "$warm" --model "$model" --cpu-moe 1 --workers 2 \
  --read-mode pread-mmap-touch --output "$proof/expert-prewarm.json"
reports=()
for entry in 0:0 1:1 2:1 3:0; do
  index=${entry%:*}; flag=${entry#*:}
  result="$proof/cpu-arena-$index-flag$flag.json"
  run "cpu-arena-$index-flag$flag.log" env TS_MUSE_SPARK_REUSE_CPU_MOE_SCRATCH="$flag" \
    TS_GGML_CPU_THREADS=32 TS_CPU_MOE_THREADS=32 OMP_NUM_THREADS=32 \
    "$python" "$micro" --backend cpu --model "$model" --library "$proof/native-build/libGgmlOps.so" \
      --layers 0 --rows 1,128,256 --threads 32 --warmup 1 --repeats 3 --seed 17817 --atol 0 --rtol 0 --output "$result"
  reports+=("$result")
done
"$python" "$guard" ab --proof "$proof" "${reports[@]}"
idle
"$python" "$guard" receipt --root "$root" --proof "$proof"
printf 'Candidate component tests and exact CPU arena A/B accepted in %s. No server was launched or candidate deployed.\n' "$proof"
printf 'Full-checkpoint quality, runtime QK/source-page/arena activation and paired llama performance still require a separate fresh campaign.\n'
