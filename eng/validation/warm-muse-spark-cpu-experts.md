# Preparing CPU expert pages for inference measurements

Run this tool after an engine has uploaded its GPU weights, using the same
checkpoint paths and CPU expert count for both TensorSharp and llama.cpp:

```sh
PYTHONPATH=/workspace/llama.cpp/gguf-py python3 eng/validation/warm-muse-spark-cpu-experts.py \
  --model /workspace/models/muse-spark/UD-Q3_K_XL/Muse-Spark-Open-816B-A42B-UD-Q3_K_XL-00001-of-00010.gguf \
  --cpu-moe 32 --workers 4 --read-mode pread-mmap-touch \
  --output /workspace/evidence/cpu-expert-prewarm.json
```

The default `pread-mmap-touch` method reads each bounded chunk, discards the
temporary byte buffer, then references one byte on every covered OS page through
a temporary read-only mapping. This gives bulk reads for sparse file-cache holes
and records mapped-page references before closing the mapping. It changes no
model data, runtime settings, system cache policy, or upstream code. The default
chunk size is 8 MiB and the default worker count is one; at most eight workers are
allowed. Working buffers/mappings are bounded by workers times chunk size (plus
mapping alignment), but GGUF metadata parsing has its own memory overhead.

`--read-mode pread` provides a bulk-read baseline. `--read-mode mmap-touch`
provides page references without the bulk read, which can be slow when sparse
missing pages fault over a network filesystem. `--check-only` reads headers and
queries Linux `mincore`; it does not fault the tensor pages into memory.

The tool selects only the first N layers' gate/up/down expert tensors, validates
complete split files and exact byte ranges against GGUF metadata, and checks file
size and timestamp before and after reading. It excludes dense weights and GPU
expert layers. Re-run preparation after either engine's GPU uploads because the
uploaded source pages may displace CPU experts from a constrained file cache.

Report preparation separately from inference timing. A completed pass is not
proof of full residency: inspect `after.resident_fraction`, measured after all
temporary mappings have closed. `bytes_covered` describes selected tensor bytes;
`bytes_read` counts bytes returned by `pread`, and `bytes_sampled` counts scalar
mapped samples (a page crossing a chunk boundary can be sampled twice).
`process_io_delta.read_bytes` records Linux-attributed storage reads, including
readahead, and is distinct from both logical counters. Residency is a snapshot
and includes padding on boundary pages.

For comparable warm-cache measurements, verify residency immediately before the
measurement and record the inference server's storage-read and major-fault
deltas during requests. If cache residency remains incomplete or faults continue,
report that limitation rather than describing the result as fully warm. Check
the effective container/cgroup memory limit when planning CPU expert placement.
