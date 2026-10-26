# Linux CUDA host mappings and transparent huge pages

`TS_GGML_HOST_HUGEPAGES=1` opts a new CUDA context's host-memory pool into Linux
`madvise(MADV_HUGEPAGE)` for owned anonymous mappings whose page-aligned extent is
at least 32 MiB. It is **off by default**; only the exact value `1` enables it.
Other operating systems, backends, smaller mappings, heap fallbacks and borrowed
buffers receive no hint. Set it before creating the context. It does not change
ggml, global kernel settings, CUDA allocations, pool retention limits, or the
mapping's original release extent.

The hint can reduce page faults when first filling large host-side cache clones.
Reusing an already faulted buffer avoids that initial work altogether, so this
option complements the bounded clone pool. It is not a substitute for reuse.

The kernel may reject the hint or accept it without using physical huge pages.
Either result preserves normal allocation and release behavior. Failures do not
log per allocation. `GgmlMemoryPoolStatistics` exposes cumulative
`HugePageAdviceRequests`, `HugePageAdviceAccepted`, `HugePageAdviceFailures`, and
`HugePageAdvisedBytes`, plus the effective `HugePageAdviceEnabled` policy. Accepted
bytes are cumulative hinted extents, not retained memory or actual huge-page
bytes. Inspect `/proc/PID/smaps` `AnonHugePages` for physical backing.
The snapshot is available through `GgmlContext.GetHostPoolStatistics()`.

With a kernel defrag policy of `madvise`, an advised allocation can incur direct
reclaim and compaction. Low memory or fragmented memory can therefore worsen
tail latency. These conditions, CUDA host registration/transfers, and inference
performance are not established by the CPU probes below. Consult the
[Linux transparent huge-page documentation](https://docs.kernel.org/6.8/admin-guide/mm/transhuge.html)
and measure the target workload before enabling this option in a service.

## Correctness probe: real managed allocation

Build with unchanged upstream ggml sources; no native library or GPU is required
by this probe:

```sh
dotnet build eng/validation/HostHugePageProbe/HostHugePageProbe.csproj -c Release \
  -p:TensorSharpSkipGgmlNative=true -p:TensorSharpSkipMlxNative=true
env -u TS_GGML_HOST_HUGEPAGES dotnet eng/validation/HostHugePageProbe/bin/Release/net10.0/HostHugePageProbe.dll \
  --out artifacts/host-huge-pages/default.json
TS_GGML_HOST_HUGEPAGES=1 dotnet eng/validation/HostHugePageProbe/bin/Release/net10.0/HostHugePageProbe.dll \
  --require-advice --exercise-failure --out artifacts/host-huge-pages/enabled.json
```

Each run allocates 40 MiB through the production managed CUDA host pool, writes
and verifies every byte, checks advice counters, releases the full extent, and
checks zero outstanding/retained blocks and zero failed releases. The enabled
run also makes a deliberately unaligned advice request to verify rejected-hint
fallback without damaging the buffer. `--require-advice` fails if a real hint is
not accepted. Linux is required; unsupported platforms exit unsuccessfully and
record that limitation. No GPU context is created. A successful probe is not
evidence of inference correctness or performance.

The default and enabled 40 MiB managed probes both passed on Ubuntu 24.04 with
.NET 10.0.10. The enabled mapping reported 40,960 KiB of `AnonHugePages`; the
deliberately invalid hint incremented only the rejection counter and preserved
every byte. Both runs released their allocation with no outstanding blocks or
release failures. These were functional checks during an unrelated native build,
so no timing comparison is qualified. The local macOS test run skips the Linux
syscall test; that skip is not counted as a Linux test pass.

## CPU copy microbenchmark

```sh
python3 eng/validation/host-mapping-probe.py \
  --output artifacts/host-huge-pages/copy.json
```

This compares fresh ordinary/advised mappings and then copies again into the
same already faulted mappings. It validates every copied byte and records
resource counters, actual `AnonHugePages`, kernel policy and host load. It does
not alter any system-wide settings. Run without concurrent benchmarks.

An exploratory Linux 6.8 VM run used 36 distinct 40 MiB buffers (1.406 GiB) and
three interleaved samples per mode. Fresh-copy p50 was 931.8 ms ordinary versus
409.6 ms advised; minor faults fell from 368,640 to 720, with the advised
destination physically backed by huge pages. Reused-buffer p50 was 222.0 versus
206.9 ms. All full-buffer comparisons passed. This small, unpressured CPU-only
sample explains a possible allocation cost; it does not establish an end-to-end
speedup or a tail-latency guarantee. Generated evidence remains in ignored
`artifacts/` or `docs/validation/`.
