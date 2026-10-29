# Qwen distributed native fixture

This tool writes a deterministic eight-layer Qwen4Exp engine fixture and drives
its native execution across separate processes. It exercises HC residuals,
GDN convolution/delta state, PLE, dense attention, QSA sparse selection, cache
growth, decode, reset and refill. Seeded weights are not a language model.

Build against the normal TensorSharp native library (upstream ggml unchanged):

```sh
dotnet build eng/validation/QwenDistributedFixture -c Release -p:TensorSharpSkipGgmlNative=true -p:TensorSharpSkipMlxNative=true
```

Make a fresh fixture directory. The ordinary fixture is 256-wide with mixed IQ
experts. `write-q4xl` is 512-wide with Q4_K gate/up and Q5_1 down experts, allowing
four global TP ranks while preserving complete 128-row MMQ output tiles.

```sh
dotnet QwenDistributedFixture.dll write artifacts/qwen-fixture
dotnet QwenDistributedFixture.dll write-q4xl artifacts/qwen-q4xl-fixture
```

Use an absolute path to the built DLL. Put the matching GgmlOps shared library
beside it or in the normal native loader path. Output directories must be fresh.
For a single-process CPU control:

```sh
dotnet QwenDistributedFixture.dll cpu-node artifacts/qwen-fixture artifacts/qwen-local -1 -
```

Start one command in each process, using the same ordered endpoint list:

```sh
dotnet QwenDistributedFixture.dll cpu-node artifacts/qwen-fixture artifacts/qwen-node0 0 127.0.0.1:35431,127.0.0.1:35432
dotnet QwenDistributedFixture.dll cpu-node artifacts/qwen-fixture artifacts/qwen-node1 1 127.0.0.1:35431,127.0.0.1:35432
```

The `cuda-layer-node` and `cuda-tp-node` modes select CUDA and use one local rank
per process. They deliberately select GPU 0, so loopback runs share a physical
GPU and do not establish multi-device capacity or throughput. Use the generic
`DeepSeek41DistributedProbe` with this fixture for actual two-node/two-GPU
qualification. Direct CPU construction tests the native adapter; production
factory capability gates currently admit distributed Qwen only on ggml_cuda.

Set `TS_KV_INITIAL_TOKENS=16` on every process to grow a live cache between the
initial 13-token chunk and the remaining 67 tokens. The driver report contains
23 finite vocabulary vectors, including four resets. Compare corresponding
vectors against a matching unsplit control with a strict absolute tolerance
of 2e-5. Keep every report, exit status and native binary identity under ignored
`artifacts/` or `docs/validation/`; distinguish physical topology and fixture
coverage from real checkpoint benchmarks.
