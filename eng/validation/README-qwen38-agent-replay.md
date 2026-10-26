# Qwen3.8 long-context and screenshot replay

`qwen38-agent-replay.py` prepares one immutable JSONL fixture and replays it over
streaming `/v1/chat/completions`. Reuse the exact fixture for every engine and
configuration. Each lane runs its requests sequentially; 1, 2 and 4 lanes are
measured in separate waves. No conversation/session identifier is sent.
See [reproducible configuration inputs](README-qwen38-perf-configurations.md)
for model identities, explicit cache/THP controls and cross-engine limitations.

```sh
python eng/validation/qwen38-agent-replay.py prepare \
  --output artifacts/qwen38-replay/fixture.jsonl --steps 30 --lanes 4

python eng/validation/qwen38-agent-replay.py run \
  --fixture artifacts/qwen38-replay/fixture.jsonl \
  --output artifacts/qwen38-replay/baseline.json \
  --url http://127.0.0.1:5088 --model <served-model-id> \
  --label baseline --provenance artifacts/qwen38-replay/baseline-provenance.json
```

Preparation requires Pillow. Replay and the harness regression tests use only
Python's standard library. Store outputs under ignored `artifacts/` or
`docs/validation/`. The provenance JSON should record the source revision, clean
upstream ggml revision, binary identities, device, model/draft/projector hashes,
exact server command, context/cache/prefill settings, and speculation window.
Record different weight quantizations or forced-tool settings as unmatched
conditions; identical request JSON does not make those model runs equivalent.

| Scenario | Request pattern | Validation |
| --- | --- | --- |
| `short1k` | Three identical requests with about 1K padding tokens | Stream integrity, usage and nonempty prose |
| `prefill8k` | Three identical requests with about 8K padding tokens | Same as above |
| `repeat19k` | Three identical requests with about 19K padding tokens | Same as above; inspect cached-token usage |
| `incremental` | About 19K initial tokens, then three prescribed continuations adding about 2.5K each | Same as above; inspect prefix reuse |
| `agent34` | Thirty screenshot observations with 34 tools; retains six prior image/assistant/tool turns | Exactly one call with the current image's code and color, no preceding prose |
| `image_replace` | Three requests differing only in their final screenshot | Same exact visual/tool check; exercises a checkpoint before changing media |
| `arithmetic` | Independent `123 + 456` request | Exactly `579` with normal completion |

Token counts above describe padding, not complete rendered prompts. Actual
counts include templates, tools and image expansion and are recorded from usage.
Generated images default to 1344×768, targeting the screenshot report's 1008
visual tokens when patch size is 16 and spatial merge is 2, without resizing.
Use `--image-width` and `--image-height` for a different target. Preparation writes
a companion `.manifest.json` containing geometry, unique PNG hashes and the
fixture hash. Confirm each engine's actual image expansion; the manifest's token
target is not a measured count.
Confirm the complete prompt remains within the configured context; do not count
server-side history truncation as coverage of the requested long-context case.

The `agent34` history through step 6 extends the preceding request's history.
At step 7, rolling retention removes the oldest turn, so its reusable token
prefix can fall back to the common system/tool prefix. Assistant/tool histories
are prescribed rather than copied from the measured engine. Differences in a
model's generated answer, tool-call formatting or IDs can consequently prevent
live continuation reuse. These timings measure replayed inputs, not a natural
agent conversation. The original screenshot report's private captured sessions
are not included, so this fixture cannot establish exact reproduction of them.

The server is not reset between waves. A wave's `first_step` is step 0 in that
wave and is **not guaranteed cold**: earlier scenarios or concurrency degrees
can warm it. Reports retain the combined wave statistics and separate
`first_step`, `later_steps`, and `by_step` groups. Start a fresh server for a
particular scenario/concurrency if a cold measurement is required. Run one GPU
workload at a time and replay configurations in the same order.

TTFT means the first nonempty SSE model delta, including transport and queueing.
A tool parser may buffer a whole call before emitting that delta. Tool-call
responses and single-chunk prose therefore have no post-first-delta rate. The
report records the last delta timestamp and delta counts to expose buffering.
Multi-chunk prose rates are still estimates because a chunk may contain multiple
tokens; use engine-side token timestamps for actual decode throughput.

Tool-history rendering includes a correctness fix at the Jinja input boundary:
OpenAI argument values deserialize as `JsonElement`, so they must be converted
to template strings/objects/arrays before the template's `is string` branch.
Previously, ordinary strings acquired literal JSON quotes in Qwen's historical
XML parameters, while some numeric/object values became quoted text. Rendering
now preserves argument types and deliberate literal quotes/backslashes, without
mutating request history or unquoting generated output. Large integers remain
JSON numbers. This intentionally changes rendered history and prompt-token hashes
for affected requests. The original wire fixture and request hashes remain the
same; token-hash changes from this fix are expected and should be attributed to
the rendering correction rather than presented as identical internal prompts.

The report excludes failed requests from successful-output throughput and latency
summaries and reports their count. A faster median with failures is not a passing
performance result. Prose checks do not assess semantic accuracy; capped outputs
are marked `truncated` even when transport checks pass. Exact screenshot/tool and
arithmetic checks do assess their narrow expected outputs. Before comparing
performance, compare request hashes, output lengths/finish reasons, failure counts,
model settings, and complete streams. Preserve all skipped or unavailable
scenarios as coverage gaps.

```sh
python -m unittest discover -s eng/validation/tests -p test_qwen38_agent_replay.py -v
```

Compare completed reports using the same fixture and requested scenario/step/
concurrency set:

```sh
python eng/validation/compare-qwen38-agent-replay.py \
  --baseline artifacts/qwen38-replay/baseline.json \
  --candidate artifacts/qwen38-replay/candidate.json \
  --candidate artifacts/qwen38-replay/velox.json \
  --output artifacts/qwen38-replay/comparison.json
```

The comparator rejects missing/different fixture or request hashes, duplicate or
different request-key sets, and unfinished reports. It revalidates stored output
checks instead of trusting the report's `passed` flag. Each case/concurrency has
separate `step0` and `later_steps` statistics, explicit maximum/valid sample counts,
and image/tool, arithmetic and prose failures. Ratios are candidate/baseline, so
less than 1 means faster. The ratio distribution is calculated from individually
paired requests, not a ratio of independently filtered medians.

If either side fails a request, that key is excluded from **both** timing samples.
Any remaining paired subset is explicitly exploratory and cannot qualify an
all-request result. Different output tokens/answers are recorded independently,
and truncated prose still does not establish semantic correctness. A comparator
exit code of zero means the requested observed checks passed; it does not mean
the candidate meets a speed target or that different quantizations are equivalent.

```sh
python -m unittest discover -s eng/validation/tests -p 'test*qwen38_agent_replay.py' -v
```

`run-qwen38-perf.py --config <config.json> --output-dir artifacts/<new-run>` owns
one local server configuration. It refuses an occupied loopback port or a busy
GPU, freezes the fixture into the run directory, checks unchanged upstream ggml,
hashes the specified binaries, records the effective cache/log directories, and
stops its own server process group on success or failure. A zero replay process
exit alone is insufficient: the runner checks complete output, the exact planned
request-key set, original and wire-request hashes, and observed validation results.
Preflight/startup failures still write failed provenance. The directory must be
new so stale reports cannot satisfy validation.

`benchmark_launch_environment` records an exact allowlist of relevant inherited,
overridden and unset settings without collecting unrelated environment secrets.
Unset means absent at launch, not disabled: the native SM12 PDL default is
applied inside the server and is reported in its startup log. Explicitly set
`TS_PREFIX_CACHE_ISOLATION=lineage` and `TS_GGML_HOST_HUGEPAGES=0` for default-policy
controls. Older reports without the launch snapshot cannot rule out inherited
overrides from their configuration file alone.

Configuration supports `startup_timeout` (default 600 seconds), `replay_timeout`
(default 7200 seconds), and `request_timeout` (passed through to replay). GPU
memory/utilization is sampled once per second during replay; those samples are
not exact instantaneous peaks. The final lifecycle result is in `provenance.json`;
the replay's embedded provenance is the snapshot recorded before its requests.

```sh
python -m unittest discover -s eng/validation/tests -p test_run_qwen38_perf.py -v
```
