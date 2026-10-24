#!/usr/bin/env bash
# Launch llama-server (unslothai minimax-m31 build) for MiniMax-M3.1-Flash with the settings the
# TensorSharp comparison (eng/minimax-m3-bench.py) needs. The TensorSharp twin is
# eng/minimax-m3-ts-server.sh; both take the same options, so one profile maps to both engines.
#
#   eng/minimax-m3-llama-server.sh [options] [-- extra llama-server arguments]
#
# Options (defaults in brackets; $M, $MMPROJ, $DSPARK and $LL come from the VM's /root/env.sh):
#   --model PATH        first GGUF shard [$M]
#   --bin DIR           directory holding llama-server [$LL]
#   --host ADDR         [127.0.0.1]           --port N   [8082]
#   --ctx N             context per sequence [73728: a 64K prompt plus its answer]
#   --np N              server slots [1]; -c becomes ctx x N, KV stays per sequence (--no-kv-unified)
#   --n-cpu-moe N       keep the routed experts of the first N layers on the CPU (-ncmoe)
#   --dspark PATH       DSpark draft: -md PATH --spec-type draft-dspark --spec-draft-n-max 7 --spec-draft-p-min 0
#   --mmproj PATH       vision projector (image scenario)
#   --layer-split N     whole layers over N GPUs (-sm layer, the default mode); picks N GPUs from --gpus
#   --gpus LIST         CUDA_VISIBLE_DEVICES [inherited]
#   --ubatch N          -ub N -b max(N,2048) [llama.cpp default]
#   --log FILE          server output [artifacts/minimax-m3-bench/llama-<port>.log]
#   --evict             drop the model files from the page cache first (posix_fadvise), as every
#                       timed load arm should
#   --background        start detached, write FILE.pid, return once /health answers (or fail)
#   --print             print the environment and command, start nothing
#
# Why these flags: without an explicit -np, llama-server picks 4 slots with a UNIFIED KV cache and
# MiniMax-M3 then silently runs dense attention ("MSA requires ..." warning); -fa off also replaces
# MSA with dense attention. -sm tensor is not supported for minimax-m3 and -sm row needs working P2P.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MODEL=${M:-}; BIN=${LL:-}; HOST=127.0.0.1; PORT=8082; CTX=73728; NP=1
NCPUMOE=""; DSPARK_PATH=""; MMPROJ_PATH=""; LAYER_SPLIT=""; GPUS=${CUDA_VISIBLE_DEVICES:-}; UBATCH=""
LOG=""; EVICT=0; BACKGROUND=0; PRINT=0; EXTRA=()

die() { echo "minimax-m3-llama-server: $*" >&2; exit 2; }
need() { [[ $# -ge 2 && -n "$2" ]] || die "$1 needs a value"; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) need "$@"; MODEL=$2; shift 2 ;;
    --bin) need "$@"; BIN=$2; shift 2 ;;
    --host) need "$@"; HOST=$2; shift 2 ;;
    --port) need "$@"; PORT=$2; shift 2 ;;
    --ctx) need "$@"; CTX=$2; shift 2 ;;
    --np) need "$@"; NP=$2; shift 2 ;;
    --n-cpu-moe) need "$@"; NCPUMOE=$2; shift 2 ;;
    --dspark) need "$@"; DSPARK_PATH=$2; shift 2 ;;
    --mmproj) need "$@"; MMPROJ_PATH=$2; shift 2 ;;
    --layer-split) need "$@"; LAYER_SPLIT=$2; shift 2 ;;
    --tp) die "--tp: llama.cpp has no tensor split for minimax-m3 (-sm tensor throws); compare TensorSharp --tp against --layer-split" ;;
    --gpus) need "$@"; GPUS=$2; shift 2 ;;
    --ubatch) need "$@"; UBATCH=$2; shift 2 ;;
    --log) need "$@"; LOG=$2; shift 2 ;;
    --evict) EVICT=1; shift ;;
    --background) BACKGROUND=1; shift ;;
    --print) PRINT=1; shift ;;
    --) shift; EXTRA=("$@"); break ;;
    -h|--help) sed -n '2,32p' "$0"; exit 0 ;;
    *) die "unknown option $1 (pass llama-server arguments after --)" ;;
  esac
done
[[ -n "$MODEL" ]] || die "no model: pass --model or source /root/env.sh (\$M)"
[[ -n "$BIN" ]] || die "no llama.cpp bin dir: pass --bin or source /root/env.sh (\$LL)"
[[ "$NP" =~ ^[1-9][0-9]*$ ]] || die "--np must be a positive integer (an implicit -np gives a unified KV cache and dense attention)"

if [[ -n "$LAYER_SPLIT" ]]; then
  [[ "$LAYER_SPLIT" =~ ^[1-9][0-9]*$ ]] || die "--layer-split must be a positive integer"
  if [[ -n "$GPUS" ]]; then
    IFS=, read -ra all <<< "$GPUS"
    (( ${#all[@]} >= LAYER_SPLIT )) || die "--layer-split $LAYER_SPLIT needs $LAYER_SPLIT GPUs but --gpus lists ${#all[@]}"
    GPUS=$(IFS=,; echo "${all[*]:0:LAYER_SPLIT}")
  else
    GPUS=$(seq 0 $((LAYER_SPLIT - 1)) | paste -sd, -)
  fi
fi

CMD=("$BIN/llama-server" -m "$MODEL" --jinja -fa on -ngl 999 -c $((CTX * NP)) -np "$NP" --no-kv-unified
     --host "$HOST" --port "$PORT" --no-webui)
[[ -n "$LAYER_SPLIT" ]] && CMD+=(-sm layer)
[[ -n "$NCPUMOE" ]] && CMD+=(--n-cpu-moe "$NCPUMOE")
[[ -n "$DSPARK_PATH" ]] && CMD+=(-md "$DSPARK_PATH" --spec-type draft-dspark --spec-draft-n-max 7 --spec-draft-p-min 0)
[[ -n "$MMPROJ_PATH" ]] && CMD+=(--mmproj "$MMPROJ_PATH")
[[ -n "$UBATCH" ]] && CMD+=(-ub "$UBATCH" -b $(( UBATCH > 2048 ? UBATCH : 2048 )))
CMD+=("${EXTRA[@]+"${EXTRA[@]}"}")

LOG=${LOG:-$ROOT/artifacts/minimax-m3-bench/llama-$PORT.log}
ENVS=(); [[ -n "$GPUS" ]] && ENVS+=("CUDA_VISIBLE_DEVICES=$GPUS")
# The bin dir holds libllama/libggml next to the binary.
ENVS+=("LD_LIBRARY_PATH=$BIN${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}")

if (( PRINT )); then
  printf '%s ' "${ENVS[@]}" "${CMD[@]}"; echo
  exit 0
fi

if (( EVICT )); then
  files=("$MODEL")
  shopt -s nullglob
  if [[ "$MODEL" =~ ^(.*)-[0-9]{5}-of-([0-9]{5})\.gguf$ ]]; then files=("${BASH_REMATCH[1]}"-?????-of-"${BASH_REMATCH[2]}".gguf); fi
  shopt -u nullglob
  [[ -n "$DSPARK_PATH" ]] && files+=("$DSPARK_PATH")
  [[ -n "$MMPROJ_PATH" ]] && files+=("$MMPROJ_PATH")
  python3 - "${files[@]}" <<'EOF'
import os, sys
for path in sys.argv[1:]:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
    print(f"evicted {path}")
EOF
fi

mkdir -p "$(dirname "$LOG")"
echo "$(date -u +%FT%TZ) ${ENVS[*]} ${CMD[*]}" > "$LOG"
if (( ! BACKGROUND )); then
  exec env "${ENVS[@]}" "${CMD[@]}" >> "$LOG" 2>&1
fi
env "${ENVS[@]}" "${CMD[@]}" >> "$LOG" 2>&1 &
PID=$!
echo "$PID" > "$LOG.pid"
start=$(date +%s)
while true; do
  if curl -sf "http://$HOST:$PORT/health" 2>/dev/null | grep -q '"ok"'; then
    echo "llama-server ready after $(( $(date +%s) - start ))s: pid $PID, http://$HOST:$PORT, log $LOG"
    if grep -q -E 'MSA (requires|needs)' "$LOG"; then
      echo "WARNING: llama.cpp reports MSA disabled (dense attention); see $LOG" >&2
    fi
    exit 0
  fi
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "llama-server exited during startup; last lines of $LOG:" >&2
    tail -20 "$LOG" >&2
    exit 1
  fi
  sleep 2
done
