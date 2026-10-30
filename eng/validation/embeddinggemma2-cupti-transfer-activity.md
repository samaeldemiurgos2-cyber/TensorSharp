# CUDA kernel and transfer activity diagnostic

`embeddinggemma2-cupti-transfer-activity.cpp` is a separate validation-only
Linux `LD_PRELOAD` hook. It derives the concurrent-kernel collection from
`embeddinggemma2-cupti-activity.cpp` and additionally enables
`CUPTI_ACTIVITY_KIND_MEMCPY`. The original tool and its reports are preserved.
This tool requires CUDA 13 headers and, at runtime, the CUPTI API version it was
compiled against (`CUPTI_API_VERSION`, 130401 for CUDA 13.4); another API version
produces an incomplete report without enabling activity.

The installed CUDA 13 `cupti_activity.h` and NVIDIA's
[Memcpy7 reference](https://docs.nvidia.com/cupti/api/structCUpti__ActivityMemcpy7.html)
identify `CUpti_ActivityMemcpy7` as the `MEMCPY` record (and `CUpti_ActivityKernel13`
as the kernel record; both verified in the CUDA 13.4 header, so confirm them in the
installed one before building). The
[activity API reference](https://docs.nvidia.com/cupti/api/group__CUPTI__ACTIVITY__API.html)
defines direction kinds, dropped-record queries and buffer callbacks. Peer
copies use a separate `MEMCPY2` record, which this tool does not enable.

Build against the installed CUDA 13 toolkit, using a fresh output directory:

```sh
g++ -O2 -std=c++17 -fPIC -shared -pthread \
  -I/usr/local/cuda/include \
  eng/validation/embeddinggemma2-cupti-transfer-activity.cpp \
  -L/usr/local/cuda/lib64 -Wl,-rpath,/usr/local/cuda/lib64 \
  -lcupti -o NEW_IGNORED_DIRECTORY/cupti-transfer-activity.so
```

Select this new library explicitly in `LD_PRELOAD`, and set the existing
`TSG_CUPTI_ACTIVITY_OUTPUT` to a **new** JSON path. The hook reserves that file
atomically with `O_EXCL` before enabling activity and retains its descriptor;
it leaves existing files untouched and logs creation/write failures to stderr.
The wrapper must also reject a pre-existing output before launch and require a
fresh complete JSON report afterward. No new environment selector,
production native library, global math mode or upstream source change is
required. Use the immutable vision probe with `--benchmark-repeats 0` and
`--require-embedding-default false` for one known forward per process. Supply
the same projector, pixels, dimensions and backend as its uninstrumented
reference. Keep the hook, compiler, installed headers/library, runtime snapshot,
projector and pixels hash-bound before and after each diagnostic.

Schema version 2 keeps the original kernel event fields under `kernel_events`
and adds independent `memcpy_events`. Each copy records the direction and its
numeric enum, source/destination memory kinds, flags, bytes, GPU timestamps,
device/context/stream, driver/runtime correlation IDs, graph IDs and copy count.
Per-direction summaries retain both record counts and the raw `copy_count_sum`
because CUDA batch-copy records can represent multiple operations. The actual
installed CUDA 13.4 header documents `copyCount` as the batch operation count,
and as 1 for non-batch copies; individual raw `copy_count` values remain visible.
The combined event
limit is 65,536, and each requested buffer is 8 MiB. Invalid timestamp records,
invalid callback buffer/record bounds, dropped/truncated records and callback
exceptions prevent `completed: true`. Both activity kinds must be enabled and
observed for a complete report.

This hook records **all process activity**, including loading and any warmup.
A cold first forward can upload the cached vision weights. Match event sizes,
order and adjacent kernels to the known native call sequence before attributing
a transfer to a block. A matching size alone is insufficient: the patch
projection can use the same 768-wide activation shape as the block input.

For the validated 16-block tower, each fused block currently uploads and reads
back one contiguous F32 `[patches, 768]` hidden tensor. The source-derived
inter-block path therefore transfers the following bytes, before counting
patch embedding, pooling, projector outputs, weights, positions or other leaves:

| Patches | Bytes per block per direction | Bytes per 16-block forward per direction | Both directions |
| ---: | ---: | ---: | ---: |
| 36 | 110,592 | 1,769,472 | 3,538,944 |
| 2,304 | 7,077,888 | 113,246,208 | 226,492,416 |
| 2,340 | 7,188,480 | 115,015,680 | 230,031,360 |

These are inferred logical-copy volumes until actual activity matches are
observed. `TS_VISION_PROF` includes the hidden upload in its upload **time**, but
its `uploadMB` counter sums only cached/streamed weight and position leaves. Its
reported compute/total interval ends before the final hidden readback.

Copy timestamps measure recorded GPU activity intervals, not CPU staging,
driver API time, allocation time, service latency or independent PCIe hardware
traffic counters. Kernel and copy intervals can overlap; their sums are not
wall time. Normalize only by known complete forwards, and never subtract
process-wide activity sums from one timed sample. This instrumented diagnostic
does not establish a paired performance improvement.

A scoped whole-tower native graph could keep the hidden tensor on device across
all 16 blocks, preserving the existing owned F32 projection and attention nodes.
With one initial upload and one final readback it would remove 15 block-boundary
round trips: 202.5 MiB at 2,304 patches, by source inference. It would need a
graph-building helper separated from per-call host binding, a call-scoped
precise backend and allocator lifetime, cache-budget/failure handling and strict
raw/pooled quality proofs. It must also preserve or explicitly scope managed
per-block tracing and cooperative chat yielding. No such production change is
implemented by this tool.
