# Prompt preparation CPU probe

This probe reads a BPE GGUF's vocabulary and chat template without loading its
weights or initializing a GPU. It measures rendering, tokenization, the combined
path with 1/2/4 simultaneous requests, and four growing conversation histories.
The fixture uses a 65,000-character system instruction and 34 tool declarations,
with thinking disabled. It records allocation counts and token hashes, and checks
that repeated/concurrent requests and continuations match a fresh tokenizer.

```sh
dotnet run --project eng/validation/PromptPreparationProbe -c Release -- \
  --model /workspace/models/Qwen3.8-27B-NVFP4-MTP-N4_0.gguf \
  --repeats 20 --out artifacts/prompt-preparation/candidate.json
```

Run the identical probe source against the baseline Runtime assembly and the
candidate Runtime assembly on the same otherwise idle host. Keep both reports in
ignored `artifacts/` or `docs/validation/`. Compare their vocabulary/template
hashes, rendered character counts, prompt token counts, and token hashes before
comparing timings. Compare medians per operation/concurrency; the cold-tokenize
row includes first-use costs and is separate from repeated requests. The probe
warms the combined path for one second before repeated measurements. Allocation
measurements include every managed thread participating in the operation.

These are synthetic text prompts, including repeated instruction paragraphs.
The timings do not include HTTP handling, GPU inference, image preparation,
request logging, or the screenshot report's original captured conversations.
They establish a CPU-path improvement only. An end-to-end performance claim
requires the separate HTTP replay matrix with matching requests/model settings.

Use `--system-text <UTF-8 file>` to substitute a code/document corpus for the
repeated instruction paragraphs. The content is repeated if shorter than
`--characters` and truncated to that character count. The report records its
path and the final system-text hash; keep that input identical between revisions.
