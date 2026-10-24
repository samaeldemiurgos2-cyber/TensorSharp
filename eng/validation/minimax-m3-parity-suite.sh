#!/usr/bin/env bash
# Copyright (c) Zhongkai Fu. All rights reserved.
# https://github.com/zhongkaifu/TensorSharp
#
# This file is part of TensorSharp.
#
# TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
#
# MiniMax-M3 fixture validation and TensorSharp-vs-llama.cpp logit parity in one command, on
# the synthetic fixtures of eng/minimax-m3-fixture.py, driving eng/validation/minimax-m3-parity.cpp.
# Every result is one JSON line appended to $LOG, tagged with the fixture, the token count and
# what was measured.
#
#   minimax-m3-parity-suite.sh fixture small|real-block
#       llama.cpp alone: the noise floor (one token stream at micro-batch 512, a smaller one and
#       1, which runs every position through the decode path), the MSA-vs-dense sensitivity
#       (how far dropping blocks moves the logits; it must dwarf the floor), activation and
#       routing statistics, the MSA block-choice margins of the prefill and decode paths and
#       whether their selections agree, and the successor chain's statistics.
#   minimax-m3-parity-suite.sh ts small|real-block
#       TensorSharp (TSGgml_MiniMaxM3*): prefill in one call, chunked prefill, prefill + decode,
#       forced sparse attention (TS_MINIMAX_M3_MSA=ggml), host-resident experts of two layers,
#       each compared with llama.cpp's prefill and decode dumps of the `fixture` step (run it
#       first, with the same N_GPU). TensorSharp's logits match when they sit inside the floor.
#   minimax-m3-parity-suite.sh dspark
#       llama-server greedy output on the small fixture with and without small/dspark.gguf, on
#       three random prompts and two that end in the successor chain's first token, and every
#       generated token checked against teacher-forced logits (parity tool, greedy mode).
#
# Environment (defaults in brackets):
#   PARITY    the parity binary [minimax-m3-parity on PATH]
#   LIBLLAMA  directory of the llama.cpp MiniMax-M3.1 build: libllama, libggml-base, llama-server
#   FIXTURES  root holding small/ and real-block/ [/workspace/mm3/fixtures]
#   TSLIB     libGgmlOps exporting TSGgml_MiniMaxM3* (ts step)
#   OUT       scratch directory; a 4096-token dump is 3.3 GB [./minimax-m3-parity-out]
#   LOG       JSON-lines result file [$OUT/results.jsonl]
#   N_GPU     1 runs both engines on the GPUs CUDA_VISIBLE_DEVICES shows, 0 on the CPU [1]
#   PY        a Python 3 with numpy, for the chain statistics and the DSpark prompts [python3]
#   PORT      llama-server port of the dspark step [8195]
set -euo pipefail

STEP=${1:-}
V=${2:-small}
PARITY=${PARITY:-minimax-m3-parity}
FIXTURES=${FIXTURES:-/workspace/mm3/fixtures}
OUT=${OUT:-$PWD/minimax-m3-parity-out}
LOG=${LOG:-$OUT/results.jsonl}
N_GPU=${N_GPU:-1}
PY=${PY:-python3}
PORT=${PORT:-8195}
dev=$([ "$N_GPU" = 0 ] && echo cpu || echo gpu)

need() { [ -n "${!1:-}" ] || { echo "error: set $1 (see the header of $0)" >&2; exit 2; }; }
case $V in
    small) N=1024; EDGE=64; SMALL_UB=64; HALF=512 ;;
    real-block) N=4096; EDGE=2048; SMALL_UB=128; HALF=2048 ;;
    *) echo "error: variant must be small or real-block" >&2; exit 2 ;;
esac
F=$FIXTURES/$V/model.gguf
O=$OUT/$V-$N-$dev
mkdir -p "$O" "$(dirname "$LOG")"
tag() { sed "s/^{/{\"fixture\":\"$1\",\"n\":$2,\"device\":\"$dev\",\"what\":\"$3\",/" >> "$LOG"; tail -n 1 "$LOG" | cut -c1-400; }

fixture() {
    need LIBLLAMA
    "$PARITY" tokens random "$N" 7 "$O/tokens.bin" > /dev/null
    for ub in 512 "$SMALL_UB" 1; do
        "$PARITY" llama "$LIBLLAMA" "$F" "$O/tokens.bin" "$O/llama-ub$ub.f32" "$N_GPU" "$ub" 2> "$O/llama-ub$ub.log" | tag "$V" "$N" "llama ub$ub"
    done
    for ub in "$SMALL_UB" 1; do
        "$PARITY" compare "$O/llama-ub512.f32" "$O/llama-ub$ub.f32" "$EDGE" | tag "$V" "$N" "floor: llama ub512 vs llama ub$ub"
    done
    rm -f "$O/llama-ub$SMALL_UB.f32"   # ub512 and ub1 stay for the ts step
    MM3_LLAMA_DENSE=1 "$PARITY" llama "$LIBLLAMA" "$F" "$O/tokens.bin" "$O/llama-dense.f32" "$N_GPU" 512 2> "$O/llama-dense.log" > /dev/null
    "$PARITY" compare "$O/llama-ub512.f32" "$O/llama-dense.f32" "$EDGE" | tag "$V" "$N" "sensitivity: llama MSA vs llama dense"
    rm -f "$O/llama-dense.f32"
    "$PARITY" stats "$LIBLLAMA" "$F" "$O/tokens.bin" "$N_GPU" 512 2> /dev/null | tag "$V" "$N" "stats ub512"
    for ub in 512 1; do
        "$PARITY" msa "$LIBLLAMA" "$F" "$O/tokens.bin" "$N_GPU" "$ub" "$O/sel-ub$ub.tsv" 2> "$O/msa-ub$ub.log" | tag "$V" "$N" "msa margins ub$ub"
    done
    local decisions differ
    decisions=$(($(wc -l < "$O/sel-ub512.tsv") - 1))
    differ=$(diff <(tail -n +2 "$O/sel-ub512.tsv" | cut -f1,2,3,6 | sort) <(tail -n +2 "$O/sel-ub1.tsv" | cut -f1,2,3,6 | sort) | grep -c '^[<>]' || true)
    echo "{\"mode\":\"msa-selection-diff\",\"decisions\":$decisions,\"differing_lines\":$differ}" | tag "$V" "$N" "msa selections ub512 vs ub1"
    "$PY" - "$O/llama-ub512.f32" "$O/tokens.bin" <<'PY' | tag "$V" "$N" "successor chain"
import json, sys
import numpy as np
V, FIRST, LEN = 200064, 100000, 4096
raw = np.memmap(sys.argv[1], dtype="<f4", mode="r", offset=32).reshape(-1, V)
toks = np.fromfile(sys.argv[2], dtype="<i4")
am = np.asarray(raw.argmax(axis=1))
chain_in = (toks >= FIRST) & (toks < FIRST + LEN)
chain_top = (am >= FIRST) & (am < FIRST + LEN)
other = ~chain_in
# after a non-chain token the chain should win about LEN / V of the time, as any other id
print(json.dumps({"mode": "chain-stats", "positions": len(am), "after_non_chain": int(other.sum()),
                  "chain_argmax_after_non_chain": int((chain_top & other).sum()),
                  "chain_rows_logit_std": round(float(raw[other][:, FIRST + 1:FIRST + LEN].std()), 4),
                  "other_rows_logit_std": round(float(raw[other][:, :FIRST].std()), 4)}))
PY
}

ts_one() {   # NAME REFS ENV... -- PARITY-ARGS...: one TensorSharp dump compared with REFS, then deleted
    local name=$1 refs=$2
    shift 2
    local envs=()
    while [ "$1" != "--" ]; do envs+=("$1"); shift; done
    shift
    env ${envs[@]+"${envs[@]}"} "$PARITY" "$@" 2> "$O/$name.log" | tag "$V" "$N" "$name"
    local dump=$5 r
    for r in $refs; do
        "$PARITY" compare "$O/$r.f32" "$dump" "$EDGE" | tag "$V" "$N" "$name vs $r"
    done
    rm -f "$dump"
}

ts() {
    need TSLIB
    for r in llama-ub512 llama-ub1; do
        [ -f "$O/$r.f32" ] || { echo "error: $O/$r.f32 is missing; run the fixture step first" >&2; exit 2; }
    done
    local d=$O/ts.f32 refs="llama-ub512 llama-ub1"
    ts_one "ts prefill" "$refs" -- ts "$TSLIB" "$F" "$O/tokens.bin" "$d" "$N_GPU" 512
    ts_one "ts prefill in 64-token chunks" llama-ub512 -- ts "$TSLIB" "$F" "$O/tokens.bin" "$d" "$N_GPU" 512 64
    ts_one "ts prefill $HALF + decode" "$refs" -- ts-decode "$TSLIB" "$F" "$O/tokens.bin" "$d" "$N_GPU" 512 "$HALF"
    ts_one "ts prefill, MSA=ggml" llama-ub512 TS_MINIMAX_M3_MSA=ggml -- ts "$TSLIB" "$F" "$O/tokens.bin" "$d" "$N_GPU" 512
    ts_one "ts prefill $HALF + decode, MSA=ggml" "$refs" TS_MINIMAX_M3_MSA=ggml -- ts-decode "$TSLIB" "$F" "$O/tokens.bin" "$d" "$N_GPU" 512 "$HALF"
    ts_one "ts prefill, F32 indexer" llama-ub512 TS_MINIMAX_M3_INDEXER_F16=0 -- ts "$TSLIB" "$F" "$O/tokens.bin" "$d" "$N_GPU" 512
    ts_one "ts prefill, F16 indexer" llama-ub512 TS_MINIMAX_M3_INDEXER_F16=1 -- ts "$TSLIB" "$F" "$O/tokens.bin" "$d" "$N_GPU" 512
    ts_one "ts prefill $HALF + decode, F16 indexer" "$refs" TS_MINIMAX_M3_INDEXER_F16=1 -- ts-decode "$TSLIB" "$F" "$O/tokens.bin" "$d" "$N_GPU" 512 "$HALF"
    if [ "$N_GPU" != 0 ]; then
        ts_one "ts prefill, routed experts of layers 0-3 on the host" llama-ub512 MM3_CPU_MOE=4 -- ts "$TSLIB" "$F" "$O/tokens.bin" "$d" "$N_GPU" 512
    fi
}

dspark() {
    need LIBLLAMA
    local S=$FIXTURES/small W=$OUT/dspark
    mkdir -p "$W"
    printf ']~!b[]~b]user\nWrite a haiku about the sea.[e~[\n]~b]ai\n' > "$W/p0.txt"
    "$PARITY" tokens text "$LIBLLAMA" "$S/model.gguf" "$W/p0.txt" "$W/p0.bin" > /dev/null
    echo "200034 200019 9690 10 2000 3000 4000 5000 6000" > "$W/p1.txt"
    "$PARITY" tokens ids "$W/p1.txt" "$W/p1.bin" > /dev/null
    "$PARITY" tokens random 300 11 "$W/p2.bin" > /dev/null
    # p3, p4: random text that ends in the chain's first id, so the drafter drafts t + 1 and the
    # target accepts up to the chain's next skip (positions up to ~320 and ~800: sparse MSA)
    local p len
    for p in 3:120:12 4:600:13; do
        IFS=: read -r p len seed <<< "$p"
        "$PARITY" tokens random "$len" "$seed" "$W/p${p}r.bin" > /dev/null
        "$PY" -c "import numpy as np, sys; np.concatenate([np.fromfile(sys.argv[1], dtype='<i4'), np.array([100000], dtype='<i4')]).tofile(sys.argv[2])" "$W/p${p}r.bin" "$W/p$p.bin"
    done
    serve() {
        local run=$1
        shift
        "$LIBLLAMA/llama-server" -m "$S/model.gguf" -fa on -ngl 99 -c 4096 -np 1 --host 127.0.0.1 --port "$PORT" --no-webui "$@" > "$W/server-$run.log" 2>&1 &
        local pid=$!
        for _ in $(seq 1 120); do
            curl -s "localhost:$PORT/health" | grep -q '"ok"' && break
            kill -0 "$pid" 2> /dev/null || { echo "error: llama-server ($run) died, see $W/server-$run.log" >&2; exit 1; }
            sleep 1
        done
        for p in 0 1 2 3 4; do
            local ids
            ids=$("$PY" -c "import numpy as np, sys; print(','.join(map(str, np.fromfile(sys.argv[1], dtype='<i4').tolist())))" "$W/p$p.bin")
            curl -s "localhost:$PORT/completion" -H "Content-Type: application/json" \
                -d "{\"prompt\": [$ids], \"n_predict\": 200, \"temperature\": 0, \"top_k\": 1, \"ignore_eos\": true, \"cache_prompt\": false, \"return_tokens\": true}" \
                > "$W/$run-p$p.json"
        done
        kill "$pid"
        wait "$pid" 2> /dev/null || true
    }
    serve base
    serve dspark -md "$S/dspark.gguf" --spec-type draft-dspark --spec-draft-n-max 7 -ngld 99
    for run in base dspark; do
        for p in 0 1 2 3 4; do
            "$PY" -c "import json, sys, numpy as np; np.concatenate([np.fromfile(sys.argv[1], dtype='<i4'), np.array(json.load(open(sys.argv[2]))['tokens'], dtype='<i4')]).tofile(sys.argv[3])" \
                "$W/p$p.bin" "$W/$run-p$p.json" "$W/$run-p$p.bin"
            local n=$(($(wc -c < "$W/p$p.bin") / 4))
            "$PARITY" llama "$LIBLLAMA" "$S/model.gguf" "$W/$run-p$p.bin" "$W/$run-p$p.f32" "$N_GPU" 512 2> /dev/null > /dev/null
            "$PARITY" greedy "$W/$run-p$p.f32" "$W/$run-p$p.bin" "$n" | tag small "$n" "dspark $run p$p: generated tokens vs teacher-forced argmax"
            rm -f "$W/$run-p$p.f32"
        done
    done
    for p in 0 1 2 3 4; do
        "$PY" - "$W/base-p$p.json" "$W/dspark-p$p.json" <<'PY' | tag small 200 "dspark p$p: greedy tokens with vs without the drafter"
import json, sys
a, b = (json.load(open(x)) for x in sys.argv[1:3])
ta, tb, t = a["tokens"], b["tokens"], b.get("timings", {})
first = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), -1)
print(json.dumps({"mode": "dspark", "identical_tokens": ta == tb, "first_difference": first,
                  "draft_n": t.get("draft_n"), "draft_n_accepted": t.get("draft_n_accepted"),
                  "tok_s_base": round(a["timings"]["predicted_per_second"], 1),
                  "tok_s_dspark": round(t.get("predicted_per_second", 0.0), 1)}))
PY
    done
}

case $STEP in
    fixture) fixture ;;
    ts) ts ;;
    dspark) dspark ;;
    *) sed -n '9,40p' "$0"; exit 2 ;;
esac
