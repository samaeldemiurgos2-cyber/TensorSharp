# Qwen35 recurrent-state hand-off validation

Loads one real Qwen3.5-family GGUF on CUDA and drives several sequences through
the per-request holders the way the scheduler's `PerSequenceFused` path does:

1. prompt chunks that alternate between three holders, so every chunk after a
   sequence's first resumes a state another sequence displaced;
2. a capture, clone and decode before any solo decode has run, while a holder's
   state lives only in its own device copies;
3. solo fused decode steps, round robin;
4. batched arena decode steps over every holder, then - with a DFlash drafter
   (`TS_TEST_QWEN35_DRAFT`) - batched speculative verifies whose commits keep a
   different prefix per sequence each round, covering every acceptance count;
5. more solo decode, a second-turn prompt chunk per holder, more decode;
6. a release of a bound holder, then a new sequence started at position 0;
7. a checkpoint chain: capture a holder, export the checkpoint to a stream,
   clone it, extend and capture the clone, clone that and decode.

Every step's logits are hashed (SHA-256 of the raw floats); a speculative verify
step hashes its rows' argmax and drafter features. The routes the
recurrent state and K/V take between steps - device hand-off or host round trip
(`TS_QWEN35_DEVICE_STATE_HANDOFF`), and the capture/clone routes - only move
bytes, so every configuration must reproduce the all-host reference bit for bit.
A declined batched step, capture, export or clone fails the run.

```sh
export TS_TEST_QWEN35_MODEL=/models/Qwen3.8-27B-NVFP4-MTP-N4_0.gguf
export TS_TEST_QWEN35_DRAFT=/models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf   # optional: phase 4b
export MAX_CONTEXT=32768 TS_KV_INITIAL_TOKENS=128 TS_PREFIX_RELEASE_DEBUG=1
P=eng/validation/Qwen35StateHandoffProbe
dotnet build $P -c Release -p:TensorSharpSkipGgmlNative=true
cp TensorSharp.GGML.Native/build/libGgmlOps.so $P/bin/Release/net10.0/
OUT=artifacts/handoff-probe; mkdir -p $OUT
# Reference: every hand-off, capture and clone through the host.
env TS_QWEN35_DEVICE_STATE_HANDOFF=0 TS_QWEN35_BACKGROUND_CAPTURE=0 TS_QWEN35_DEVICE_HOLDER_COPY=0 \
  dotnet $P/bin/Release/net10.0/Qwen35StateHandoffProbe.dll --out $OUT/ref.jsonl
# Defaults (device captures and clones while VRAM allows).
dotnet $P/bin/Release/net10.0/Qwen35StateHandoffProbe.dll --out $OUT/default.jsonl
# Host-resident checkpoints: background captures, clones uploaded into device copies.
TS_QWEN35_DEVICE_RETAIN_RESERVE_MB=31000 \
  dotnet $P/bin/Release/net10.0/Qwen35StateHandoffProbe.dll --out $OUT/upload.jsonl
# No device holder copies: background captures, host clones.
TS_QWEN35_DEVICE_HOLDER_COPY=0 \
  dotnet $P/bin/Release/net10.0/Qwen35StateHandoffProbe.dll --out $OUT/background.jsonl
for f in default upload background; do
  dotnet $P/bin/Release/net10.0/Qwen35StateHandoffProbe.dll --compare $OUT/ref.jsonl $OUT/$f.jsonl
done
```

With `TS_PREFIX_RELEASE_DEBUG=1` the log names each capture and clone route
(`[prefix-capture] ... via device+live|device|background|host`, `[prefix-clone]
... via device|upload|host`); check that each configuration took the route it is
meant to exercise. `device+live` is a capture right after a prompt chunk that
reads the verifier's live recurrent state into the checkpoint and leaves it live
(`TS_QWEN35_CAPTURE_LIVE_STATE=0` drains it first instead). `TS_Q35_VERIFY_LOG=1` prints `devCur=True` for every verify that
started from a device-seeded state.

Missing models or devices fail the run; they never count as passing validation.
Keep reports in the ignored `artifacts/` directory.
