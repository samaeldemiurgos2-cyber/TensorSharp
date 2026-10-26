# Qwen chat TTFT and KV reuse validation

`qwen-chat-cache-benchmark.py` measures streaming HTTP time to first nonempty
model delta, first answer latency, prompt tokens, cached tokens, and generated
tokens against an already running TensorSharp server. It uses Python's standard
library. Web UI mode reproduces `/api/upload`, a named `/api/sessions` session,
and `/api/chat`; OpenAI mode uses `/v1/chat/completions`.

Use the same weights, unchanged upstream ggml revision, backend, server settings,
request arguments, and device for baseline and candidate. Run one GPU workload
at a time. Capture the actual server command, dependency revision/clean status,
native and managed binary identities, device, context/cache settings, and model
identity in a JSON file, then attach it using `--provenance`. The script discovers
the served model ID from `/v1/models`; it does not verify operator provenance.
All generated reports must stay in ignored `docs/validation/` or `artifacts/`.

For TensorAgent catalog validation, install the catalog's complete file set in
`<root>/cache/models/<id>/`, then run `eng/validation/TensorAgentHost` with
`--catalog-model <id>` instead of `--weights`. This calls the app's `UseModel`
path and applies its catalog memory policy. The launcher uses a 32 GB desktop
tier and records its eligible entries and cache budget in `connection.json`.
Pass that file to this benchmark with `--connection-file <root>/connection.json`,
`--model-discovery webui`, and the matching `--url`; requests include the loopback cookie, which is omitted
from benchmark reports. Keep the connection file private.

Start the baseline host with its baseline binaries. For the supplied Flash Next
checkpoint and photo, the following PowerShell commands reproduce the image
description followed by two `请继续` turns:

```powershell
TensorSharp.Server.Host.exe `
  --model "C:\Works\models\Qwen3.8-Flash-Next-GGUF\Qwen3.8-Flash-Next-UD-IQ1_M-00001-of-00003.gguf" `
  --backend ggml_cuda `
  --mmproj "C:\Works\models\Qwen3.8-Flash-Next-GGUF\mmproj-BF16.gguf" `
  --port 5088 --no-multi-agent --no-skills

python eng/validation/qwen-chat-cache-benchmark.py `
  --url http://127.0.0.1:5088 --protocol webui --cases image `
  --image "C:\Users\monke\OneDrive\Desktop\20241021_022843061_iOS.jpg" `
  --max-tokens 128 --turns 3 --repeats 3 --label baseline `
  --provenance docs/validation/qwen-ttft/baseline-provenance.json `
  --output docs/validation/qwen-ttft/baseline-image.json
```

Stop that host, start the candidate host with matching flags, and run:

```powershell
python eng/validation/qwen-chat-cache-benchmark.py `
  --url http://127.0.0.1:5088 --protocol webui --cases image `
  --image "C:\Users\monke\OneDrive\Desktop\20241021_022843061_iOS.jpg" `
  --max-tokens 128 --turns 3 --repeats 3 --label candidate `
  --require-reuse --require-full-reuse `
  --provenance docs/validation/qwen-ttft/candidate-provenance.json `
  --compare-with docs/validation/qwen-ttft/baseline-image.json `
  --output docs/validation/qwen-ttft/candidate-image.json
```

`--require-reuse` rejects zero reuse on a sequential continuation.
`--require-full-reuse` requires reuse to cover at least the previous turn's entire
prompt, including its image expansion; a small public-prefix hit cannot satisfy
that check. Branch retention depends on available cache memory and is measured
with `--branches`. `--invalidation` replaces the history in the same session with
an independent arithmetic question and requires the correct answer, `579`.

If answer drift would otherwise change the third request, add
`--replay-history-from docs/validation/qwen-ttft/baseline-image.json` to the
candidate command. Later requests use the completed baseline assistant answers
with the current prompts, current session, and newly uploaded image; normalized
request hashes must match the saved baseline before the harness sends them.
This keeps requested histories comparable. A divergent candidate answer can
still prevent reuse of its live cache, so replay timings measure the prescribed
history rather than a natural continuation of that candidate answer. Without
this option, each workflow continues its own generated answers.

For Qwen3.8-27B, start a host with the supplied
`C:\Works\models\Qwen\Qwen3.8-27B-UD-IQ4_XS.gguf` and `--backend ggml_cuda`,
then select `--cases text`. To exercise natural EOS within the output cap:

```powershell
python eng/validation/qwen-chat-cache-benchmark.py `
  --url http://127.0.0.1:5088 --protocol openai --cases text `
  --text-prompt "请用两句话说明红茶和绿茶的差别，不超过80个汉字。" `
  --max-tokens 256 --turns 3 --repeats 3 --branches --invalidation `
  --require-reuse --label candidate-eos `
  --output docs/validation/qwen-ttft/candidate-text-eos.json
```

Keep Web UI and OpenAI comparisons separate. OpenAI mode resends original image
bytes as a data URI, while Web UI mode uploads once and reuses its file reference.
The benchmark records upload latency separately; request TTFT includes request
transmission, server preprocessing, inference, queueing, and response transport.
Model loading and startup prefix preparation are excluded. The first benchmark
request does not establish that the server was unused or cold. Later image runs
can reuse vision encodings; inspect repetition 1 separately from later rows.

Reports retain complete SSE events, UTF-8 answers, timings, and normalized
request identities. Timing ratios require matching textual/image-byte histories
and complete streams. Aggregate performance qualification additionally requires
matching answer text, token counts, finish reasons, model ID, and protocol.
HTTP does not expose raw token IDs or logits. A capped image response and equal
text provide a bounded parity check; they do not establish unrestricted visual
accuracy. Review saved descriptions against the original photo and record any
truncation or output drift. Missing models/devices and skipped tests are not
passing coverage. Record `--no-prefix-cache` separately when using it as a cold
control, since that flag also disables startup prefix preparation.

Run the local harness regression tests without a model or GPU:

```powershell
python -m unittest discover -s eng/validation/tests -p test_qwen_chat_cache_benchmark.py -v
```

## Stateless replay and changing screenshots

Stateless APIs default to `TS_PREFIX_CACHE_ISOLATION=lineage`: private continuation
state is used only when the client history matches a previously emitted turn.
Named Web UI sessions always keep their own conversation scope. A replay from
another engine can have different assistant tokens, and a computer-use client
can replace an earlier screenshot. Neither necessarily proves the original
lineage, so default reuse may stop at the public system/tools prefix.

For a **trusted single-tenant server**, `TS_PREFIX_CACHE_ISOLATION=shared`
explicitly allows requests using the same stateless API session to share matching
private token prefixes. This removes the requirement to reproduce TensorSharp's
own prior assistant response; it does not bypass token or image-content matching.
It does not merge named Web UI sessions. Do not enable it for mutually untrusted
users: private-prefix hit timing and cache retention are shared. Record this
setting for both sides of any engine comparison. `strict` instead assigns every
stateless request a fresh private scope even if its lineage matches.

For non-rewindable recurrent holders such as Qwen3.8-27B, automatic private
checkpoints leave the model's required logits-producing prompt tail uncomputed.
An identical text retry can therefore reuse the checkpoint, whereas a snapshot
at the complete prompt end could not be rewound to produce those logits. With
images/audio, the automatic checkpoint is placed immediately before the newest
media span, so replacing that span can reuse the preceding conversation. Older
media still must match by content. Explicit cache breakpoints keep their existing
policy and override these automatic boundaries. The checkpoint is a full model
state copy, so measure its cost on the actual GPU and record how many states fit
under the configured memory budget; fake-model tests do not establish GPU speed.

Qwen active clones reserve the incoming request's fitted prompt/generation
capacity during the clone itself, using the same request reservation policy as
prefill. The estimate and allocation share that fixed capacity; a refused budget
creates no holder. Retained snapshots remain compact, and GPU mirrors are still
created lazily. An active CUDA clone can take an existing idle holder of exactly
that capacity; its old arena and device bindings are invalidated before copying.
This avoids faulting fresh anonymous host pages on every warmed request without
increasing global pool limits or changing checkpoint allocation. This avoids
copying a long prefix into a small clone only to grow
and copy it again on the first prefill. Record actual capacity and memory pressure:
`TS_KV_INITIAL_TOKENS` does not force every retained snapshot to that size.
See [Nsight request attribution](README-qwen-nsys-profile.md) to distinguish GPU
work, transfers and host/idle gaps in a short diagnostic capture.
