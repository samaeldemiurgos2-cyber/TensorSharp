# CUDA vision stack activation transfers

Use this parser only for one raw `EmbeddingGemma2VisionProbe` forward with
`--require-ggml-cuda-stack true`, the owned CUDA 13 CUPTI transfer hook, and
no warmup/repeated benchmark or per-block diagnostic overrides. Preserve the
raw probe/activity reports and capture source, library, compiler, model and
runtime hashes before and after execution. The raw vision oracle must pass its
original numerical limits separately.

```sh
python3 eng/validation/embeddinggemma2-cupti-stack-transfers.py \
  --input /absolute/capture-activity.json \
  --probe /absolute/capture-probe.json \
  --patches 2304 \
  --source /absolute/executed-source/ggml_ops_fused.cpp \
  --expected-source-sha256 ACTUAL_EXECUTED_SOURCE_SHA256 \
  --output /absolute/new-stack-transfers.json
```

Run on the capture host so the probe's absolute activity path resolves to the
same file. The parser requires actual stack observation (one stack encode,
zero retained block-loop encodes), sixteen sequential attention markers and
one upload/readback pair enclosing every marker on the same CUDA context.
It excludes the separate initial patch-projection pair and internal device
copies. Additional hidden-sized host copies cause rejection rather than being
silently discarded. The patch grid follows the production floor division by16.

The retained per-block parser, `embeddinggemma2-cupti-hidden-transfers.py`,
still requires sixteen separate pairs. Neither parser accepts the other route
as equivalent evidence. Summed GPU activity intervals measure logical copies;
they do not establish wall latency, physical PCIe traffic or end-to-end gains.
Generated evidence belongs in ignored `docs/validation/` or `artifacts/`.
