# Actual-weight GLM prefix diagnostics

`create-glm53-prefix.py` creates a numerical diagnostic from the pinned
`unsloth/GLM-5.3-Flash-GGUF` checkpoint at
`a38483c8cd5df544f53d70fb281afe97369d5ab6`. It requires the successful JSON report
from `verify-pinned-model-manifest.py` and all four original shards.

Use Python 3.11 or later and the `gguf==0.19.0` package in an isolated environment.
Keep generated GGUFs, manifests, logs and comparison results outside tracked
source directories, such as ignored `artifacts/` or a task-owned scratch volume.

```bash
python -m pip install gguf==0.19.0
python eng/validation/create-glm53-prefix.py \
  --verified-model artifacts/glm53-verified.json \
  --layers 1 \
  --output artifacts/glm53-prefixes/prefix1.gguf
```

Repeat with `--layers 4` and a fresh output to include the first three dense KDA
layers and the first MLA/MoE layer. The one-layer diagnostic contains the first
KDA/dense layer. The original embedding, output head, output norm and tokenizer
are retained in both cases. The outputs need approximately 1.29 GB and 4.49 GB
for the `UD-IQ3_XXS` checkpoint, respectively; the writer requires another 2 GiB
of free volume space before each write.

The writer truncates architecture metadata arrays, adjusts block count, removes
NextN and split-file metadata, and copies quantized tensors without dequantizing
or requantizing. After reopening the output, it verifies each retained tensor's
name, shape, quantization type and raw SHA-256. The adjacent manifest records
those hashes, the original checkpoint's verification report and shard hashes,
metadata changes, generator hash and final output hash.

These truncated checkpoints are for fixed-token numerical comparisons. They do
not measure the original model's answer quality or inference performance. Use
matching input tokens and numerical settings with `DeepSeek41DistributedProbe`,
then compare every saved vocabulary vector with
`compare-deepseek41-distributed.py`. A single-layer prefix can use one local GPU
as reference and four distributed TP ranks. Whole-layer placement across two
nodes requires enough layers for both stages; use the four-layer prefix for
that comparison.

The tool is deliberately pinned to the checkpoint used in the investigation.
It does not update or patch upstream ggml, and it never overwrites an existing
output or manifest. The original full-shard verification is linked rather than
repeated; retained tensor bytes are freshly hashed during generation.

## Prefix views without copying tensor payloads

For the pinned checkpoint, `create-glm53-prefix-view.py` can create a fresh
diagnostic directory containing a rewritten metadata-only first shard and
symlinks to the original three weight shards. It changes the same layer metadata
as the materialized prefix generator, retains the split tensor count, and hashes
all selected tensor bytes. Original files are opened only for reading.

```text
python eng/validation/create-glm53-prefix-view.py \
  --verified-model PATH/verified-model.json --layers 4 \
  --output-directory FRESH_PREFIX_VIEW_DIRECTORY
```

This view depends on its original shards remaining present and unchanged.
TensorSharp ignores the additional layer tensors after the declared prefix.
Before using larger views for localization, require the four-layer view to match
the materialized four-layer prefix on the same numerical probe, including full
microbatches and reset/refill operations. Use `TS_GGUF_PREFAULT=0` so a diagnostic
does not unnecessarily page in unused weights. These are numerical diagnostics;
they do not establish trained-model quality or benchmark throughput.

