#!/usr/bin/env bash
# Launch TensorSharp.Server.Host for MiniMax-M3.1-Flash with the settings the llama.cpp comparison
# (eng/minimax-m3-bench.py) needs. Twin of eng/minimax-m3-llama-server.sh: same options, so one
# profile maps to both engines.
#
#   eng/minimax-m3-ts-server.sh [options] [-- extra TensorSharp.Server.Host arguments]
#
# Options (defaults in brackets; $M, $MMPROJ and $DSPARK come from the VM's /root/env.sh):
#   --model PATH        first GGUF shard [$M]
#   --server-dll PATH   [<repo>/TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll]
#   --host ADDR         [127.0.0.1]           --port N   [8083]
#   --ctx N             context per sequence (MAX_CONTEXT) [73728: a 64K prompt plus its answer]
#   --np N              concurrent sequences [1]: TS_SCHED_MAX_RUNNING_SEQS=N and
#                       TS_MINIMAX_M3_PLAN_CTX_TOKENS=ctx x (N+1) (K/V rows the placement plan
#                       reserves: N requests plus the shared-prefix checkpoint)
#   --n-cpu-moe N       keep the routed experts of the first N layers on the CPU
#   --dspark PATH       DSpark draft: --draft-model PATH --spec-draft 7 --spec-pmin 0 (llama.cpp's
#                       --spec-draft-n-max 7 --spec-draft-p-min 0)
#   --mmproj PATH       vision projector (image scenario)
#   --layer-split N     whole layers over N GPUs; picks N GPUs from --gpus
#   --tp N              tensor parallelism over N GPUs (no llama.cpp counterpart for minimax-m3)
#   --gpus LIST         CUDA_VISIBLE_DEVICES [inherited]
#   --ubatch N          TS_MINIMAX_M3_UBATCH=N [executor default]
#   --no-prefix-cache   pass --no-prefix-cache (strict cold-prefill runs)
#   --log FILE          server output [artifacts/minimax-m3-bench/ts-<port>.log]; also the file
#                       eng/minimax-m3-bench.py --server-log reads speculative stats from
#   --evict             drop the model files from the page cache first (posix_fadvise)
#   --background        start detached, write FILE.pid, return once /v1/models answers (or fail)
#   --print             print the environment and command, start nothing
#
# Skills, sub-agents and the web UI are off so a request sees only its own tools; code execution
# is off by default. Sampling is not pinned: the bench sends greedy settings per request.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MODEL=${M:-}; DLL=$ROOT/TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll; HOST=127.0.0.1; PORT=8083
CTX=73728; NP=1; NCPUMOE=""; DSPARK_PATH=""; MMPROJ_PATH=""; LAYER_SPLIT=""; TP=""
GPUS=${CUDA_VISIBLE_DEVICES:-}; UBATCH=""; NO_PREFIX=0; LOG=""; EVICT=0; BACKGROUND=0; PRINT=0; EXTRA=()

die() { echo "minimax-m3-ts-server: $*" >&2; exit 2; }
need() { [[ $# -ge 2 && -n "$2" ]] || die "$1 needs a value"; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) need "$@"; MODEL=$2; shift 2 ;;
    --server-dll) need "$@"; DLL=$2; shift 2 ;;
    --host) need "$@"; HOST=$2; shift 2 ;;
    --port) need "$@"; PORT=$2; shift 2 ;;
    --ctx) need "$@"; CTX=$2; shift 2 ;;
    --np) need "$@"; NP=$2; shift 2 ;;
    --n-cpu-moe) need "$@"; NCPUMOE=$2; shift 2 ;;
    --dspark) need "$@"; DSPARK_PATH=$2; shift 2 ;;
    --mmproj) need "$@"; MMPROJ_PATH=$2; shift 2 ;;
    --layer-split) need "$@"; LAYER_SPLIT=$2; shift 2 ;;
    --tp) need "$@"; TP=$2; shift 2 ;;
    --gpus) need "$@"; GPUS=$2; shift 2 ;;
    --ubatch) need "$@"; UBATCH=$2; shift 2 ;;
    --no-prefix-cache) NO_PREFIX=1; shift ;;
    --log) need "$@"; LOG=$2; shift 2 ;;
    --evict) EVICT=1; shift ;;
    --background) BACKGROUND=1; shift ;;
    --print) PRINT=1; shift ;;
    --) shift; EXTRA=("$@"); break ;;
    -h|--help) sed -n '2,32p' "$0"; exit 0 ;;
    *) die "unknown option $1 (pass TensorSharp.Server.Host arguments after --)" ;;
  esac
done
[[ -n "$MODEL" ]] || die "no model: pass --model or source /root/env.sh (\$M)"
[[ -f "$DLL" ]] || (( PRINT )) || die "server not built: $DLL (dotnet build TensorSharp.Server.Host -c Release, or --server-dll)"
[[ "$NP" =~ ^[1-9][0-9]*$ ]] || die "--np must be a positive integer"
[[ -n "$LAYER_SPLIT" && -n "$TP" ]] && die "--layer-split and --tp cannot be combined"

SPLIT=${LAYER_SPLIT:-$TP}
if [[ -n "$SPLIT" ]]; then
  [[ "$SPLIT" =~ ^[1-9][0-9]*$ ]] || die "--layer-split/--tp must be a positive integer"
  if [[ -n "$GPUS" ]]; then
    IFS=, read -ra all <<< "$GPUS"
    (( ${#all[@]} >= SPLIT )) || die "$SPLIT GPUs requested but --gpus lists ${#all[@]}"
    GPUS=$(IFS=,; echo "${all[*]:0:SPLIT}")
  else
    GPUS=$(seq 0 $((SPLIT - 1)) | paste -sd, -)
  fi
fi

CMD=("${DOTNET:-dotnet}" "$DLL" --model "$MODEL" --backend ggml_cuda --host "$HOST" --port "$PORT"
     --no-webui --no-skills --no-multi-agent)
[[ -n "$LAYER_SPLIT" ]] && CMD+=(--layer-split "$LAYER_SPLIT")
[[ -n "$TP" ]] && CMD+=(--tp "$TP")
[[ -n "$NCPUMOE" ]] && CMD+=(--n-cpu-moe "$NCPUMOE")
[[ -n "$DSPARK_PATH" ]] && CMD+=(--draft-model "$DSPARK_PATH" --spec-draft 7 --spec-pmin 0)
[[ -n "$MMPROJ_PATH" ]] && CMD+=(--mmproj "$MMPROJ_PATH")
(( NO_PREFIX )) && CMD+=(--no-prefix-cache)
CMD+=("${EXTRA[@]+"${EXTRA[@]}"}")

LOG=${LOG:-$ROOT/artifacts/minimax-m3-bench/ts-$PORT.log}
ENVS=("MAX_CONTEXT=$CTX" "TS_SCHED_MAX_RUNNING_SEQS=$NP" "TS_MINIMAX_M3_PLAN_CTX_TOKENS=$((CTX * (NP + 1)))"
      "TENSORSHARP_LOG_DIR=$(dirname "$LOG")/ts-logs-$PORT")
[[ -n "$GPUS" ]] && ENVS+=("CUDA_VISIBLE_DEVICES=$GPUS")
[[ -n "$UBATCH" ]] && ENVS+=("TS_MINIMAX_M3_UBATCH=$UBATCH")

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
  # The host binds its port only after the model has loaded and the shared prompt is warmed.
  if curl -sf "http://$HOST:$PORT/v1/models" 2>/dev/null | grep -q '"id"'; then
    echo "TensorSharp server ready after $(( $(date +%s) - start ))s: pid $PID, http://$HOST:$PORT, log $LOG"
    exit 0
  fi
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "TensorSharp server exited during startup; last lines of $LOG:" >&2
    tail -20 "$LOG" >&2
    exit 1
  fi
  sleep 2
done
