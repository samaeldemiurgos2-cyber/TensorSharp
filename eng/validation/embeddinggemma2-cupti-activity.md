This Linux diagnostic records CUDA kernel activity timestamps using the CUDA 13
CUPTI Activity API (`CUpti_ActivityKernel13` records in CUDA 13.4; build it against
the installed CUDA 13 toolkit), without hardware performance counters or
production instrumentation. It refuses a CUPTI runtime whose API version differs
from the headers it was compiled against.
It does not establish service latency or model quality. Kernel intervals can
overlap; summed kernel durations exclude host code and memory copies.

```sh
g++ -O2 -std=c++17 -fPIC -shared -pthread \
  -I /usr/local/cuda/include \
  eng/validation/embeddinggemma2-cupti-activity.cpp \
  -L /usr/local/cuda/lib64 -Wl,-rpath,/usr/local/cuda/lib64 -lcupti \
  -o artifacts/embeddinggemma2-cupti-activity.so

TSG_CUPTI_ACTIVITY_OUTPUT=/absolute/ignored/kernel-activity.json \
LD_PRELOAD=/absolute/ignored/embeddinggemma2-cupti-activity.so \
  dotnet /absolute/frozen/EmbeddingGemma2VisionProbe.dll ...
```

Run only in a coordinated quiet window against an immutable runtime. Use the
actual captured input pixels; keep warmup and measured encode counts explicit.
The trace includes warmup. A maximum of 65,536 kernel events bounds memory.
`completed` requires successful registration, enabled tracing, at least one
kernel, and zero errors, dropped/incomplete/truncated records, or callback
exceptions. Missing/failed captures are unavailable, never passing validation.
The inference process exit code, SDK library/header hashes, tool source/binary,
input/model/native runtime hashes and post-run immutability must be recorded by
the caller. Instrumented probe timings must not replace paired HTTP timings.

Validate aggregation against every recorded timestamp before using the diagnostic:

```sh
python3 eng/validation/embeddinggemma2-cupti-summary.py \
  --input artifacts/kernel-activity.json --output artifacts/kernel-summary.json
python3 -m unittest discover -s eng/validation -p test_embeddinggemma2_cupti_summary.py
```

The summary rejects loss, malformed intervals, incomplete runs, duplicate or
inconsistent aggregate names/counts/times, and mismatched totals. Its GEMM,
owned vision attention and remaining categories are descriptive kernel groups,
not an attribution of complete HTTP time.
