#!/usr/bin/env bash
# Run eng/minimax-m3-bench.py against llama.cpp (A) and TensorSharp (B) in an ABBA rotation, one
# engine on the machine at a time: for every arm, evict the model's page cache, launch that engine's
# server in the background with the SAME profile options, run the bench, stop the server. Results
# go to OUT/llama.jsonl and OUT/ts.jsonl (appended per arm), then eng/minimax-m3-report.py.
#
#   eng/minimax-m3-abba.sh --out DIR [--sequence ABBA] [--label NAME] \
#       [--profile "--np 8 --ctx 8192"] [--bench "--scenarios parallel_2,parallel_4,parallel_8"] \
#       [--llama "extra llama helper options"] [--ts "extra ts helper options"] \
#       [--bench-llama "llama-only bench options"] [--bench-ts "ts-only bench options"] \
#       [--llama-port 8082] [--ts-port 8083]
#
# --profile options go to BOTH launch helpers (eng/minimax-m3-{llama,ts}-server.sh; e.g. --np,
# --ctx, --layer-split, --n-cpu-moe, --dspark, --mmproj, --gpus); --llama/--ts go to one only.
# --bench options go to every bench run, --bench-llama/--bench-ts to one engine's (e.g. an engine's own
# --extra-body; option strings are split on spaces, so write JSON without spaces). Servers are
# stopped by PID, never by name.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUT=""; SEQ=ABBA; LABEL=""; PROFILE=""; BENCH=""; LLAMA_OPTS=""; TS_OPTS=""; BENCH_LLAMA=""; BENCH_TS=""
LLAMA_PORT=8082; TS_PORT=8083
die() { echo "minimax-m3-abba: $*" >&2; exit 2; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) OUT=$2; shift 2 ;;
    --sequence) SEQ=$2; shift 2 ;;
    --label) LABEL=$2; shift 2 ;;
    --profile) PROFILE=$2; shift 2 ;;
    --bench) BENCH=$2; shift 2 ;;
    --llama) LLAMA_OPTS=$2; shift 2 ;;
    --ts) TS_OPTS=$2; shift 2 ;;
    --bench-llama) BENCH_LLAMA=$2; shift 2 ;;
    --bench-ts) BENCH_TS=$2; shift 2 ;;
    --llama-port) LLAMA_PORT=$2; shift 2 ;;
    --ts-port) TS_PORT=$2; shift 2 ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) die "unknown option $1" ;;
  esac
done
[[ -n "$OUT" ]] || die "--out DIR is required"
[[ "$SEQ" =~ ^[AB]+$ ]] || die "--sequence takes A (llama.cpp) and B (TensorSharp) letters"
mkdir -p "$OUT"

PID_FILE=""
stop_server() {
  [[ -n "$PID_FILE" && -f "$PID_FILE" ]] || return 0
  local pid; pid=$(cat "$PID_FILE")
  if kill -0 "$pid" 2>/dev/null; then
    kill "$pid"
    for _ in $(seq 1 60); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -0 "$pid" 2>/dev/null && kill -9 "$pid"
  fi
  rm -f "$PID_FILE"; PID_FILE=""
}
trap stop_server EXIT

arm=0
for letter in $(echo "$SEQ" | grep -o .); do
  arm=$((arm + 1))
  if [[ $letter == A ]]; then
    engine=llamacpp; port=$LLAMA_PORT; helper=$ROOT/eng/minimax-m3-llama-server.sh; opts=$LLAMA_OPTS
    bench_opts=$BENCH_LLAMA; out=$OUT/llama.jsonl
  else
    engine=tensorsharp; port=$TS_PORT; helper=$ROOT/eng/minimax-m3-ts-server.sh; opts=$TS_OPTS
    bench_opts=$BENCH_TS; out=$OUT/ts.jsonl
  fi
  log=$OUT/arm$arm-$engine.log
  echo "=== arm $arm/${#SEQ}: $engine ($(date -u +%FT%TZ)) load $(cut -d' ' -f1-3 /proc/loadavg 2>/dev/null || uptime)"
  # shellcheck disable=SC2086  # option strings are word-split on purpose
  bash "$helper" --port "$port" --log "$log" --evict --background $PROFILE $opts
  PID_FILE=$log.pid
  # shellcheck disable=SC2086
  python3 "$ROOT/eng/minimax-m3-bench.py" --engine "$engine" --url "http://127.0.0.1:$port" --out "$out" \
      --label "${LABEL:-abba}" --server-log "$log" $BENCH $bench_opts || echo "bench exited with $? (arm $arm)" >&2
  stop_server
done

if [[ -f "$OUT/ts.jsonl" && -f "$OUT/llama.jsonl" ]]; then
  python3 "$ROOT/eng/minimax-m3-report.py" --ts "$OUT/ts.jsonl" --llama "$OUT/llama.jsonl" \
      --title "${LABEL:-abba}" --out "$OUT/report.md" --html "$OUT/report.html" > /dev/null
  echo "report: $OUT/report.md"
fi
