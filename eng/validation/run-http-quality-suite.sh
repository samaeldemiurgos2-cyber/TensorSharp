#!/usr/bin/env bash
# Run the HTTP quality checks used for multi-node qualification against one
# OpenAI-compatible TensorSharp server. Suitable as run-multinode-case.py's
# --server-command: run-http-quality-suite.sh {url} {output} MODEL WEIGHTS_ID [PROFILE]
#   - parallel-server-eval.py: capital / arithmetic / extraction answers
#   - validate_inference.py: short, short_zh, json, json_schema, json_unicode,
#     multi_turn and tool_round_trip scenarios, one request at a time (a
#     distributed deployment serves serialized requests); PROFILE labels the
#     placement/context settings in its report
#   - validate-qwen38-tool-calls.py: streaming and non-streaming weather tool calls
# Exits non-zero if any suite fails; each suite still runs and writes its JSON.
set -u
url=$1 out=$2 model=$3 weights=${4:-unspecified} profile=${5:-multinode}
# pwd -W gives Windows-form paths under Git Bash, which Windows Python can open
here=$(cd "$(dirname "$0")" && (pwd -W 2>/dev/null || pwd)); repo=$(cd "$here/../.." && (pwd -W 2>/dev/null || pwd))
py=${PYTHON:-python3}
# Reports contain non-ASCII model output; do not use the Windows ANSI code page.
export PYTHONUTF8=1
mkdir -p "$out"
status=0
"$py" "$here/parallel-server-eval.py" --url "$url" --model "$model" --output "$out/eval.json" \
    --concurrency 1 --repeats 1 --max-tokens 48 --cases capital,arithmetic,extraction > "$out/eval.log" 2>&1 || status=1
"$py" "$repo/benchmarks/engine_comparison/validate_inference.py" --url "$url" --engine tensorsharp --model "$model" \
    --weights-id "$weights" --profile "$profile" --output "$out/quality.json" --max-tokens 128 \
    --concurrency 1 --repeats 1 \
    --scenarios short,short_zh,json,json_schema,json_unicode,multi_turn,tool_round_trip > "$out/quality.log" 2>&1 || status=1
"$py" "$here/validate-qwen38-tool-calls.py" --url "$url" --model "$model" --output "$out/tools.json" \
    --scenarios weather --thinking off --max-tokens 256 > "$out/tools.log" 2>&1 || status=1
echo "http-quality status=$status (eval/quality/tools logs in $out)"
exit $status
