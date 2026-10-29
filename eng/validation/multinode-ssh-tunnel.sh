#!/usr/bin/env bash
# Open an authenticated SSH forward for TensorSharp's multi-node peer traffic.
#
# Run on node 0. It listens on 127.0.0.1:LOCAL_PORT and forwards to
# 127.0.0.1:REMOTE_PORT on the peer, where that node's TensorSharp process
# listens. Use the forwarded loopback endpoints in --tp-peers, for example
# --tp-peers 127.0.0.1:19500,127.0.0.1:19501 on both nodes.
#
# Latency: by default the forward rides an interactive session ('-tt' plus a
# remote 'sleep infinity'). OpenSSH sets TCP_NODELAY on its transport socket
# only for interactive sessions, at both ends; a forward-only '-N' session can
# hold small frames behind Nagle's algorithm. OpenSSH 9.5+ also adds 20 ms
# keystroke-timing obfuscation to interactive sessions, which this script turns
# off with ObscureKeystrokeTiming=no. AES-GCM moved large frames about twice as
# fast as the chacha20 default on the A40 VMs this was measured on. Measure your own route with
# eng/validation/DistributedTransportProbe; decode collectives are small and
# latency-bound, prefill transfers are large and bandwidth-bound.
#
# usage: multinode-ssh-tunnel.sh [options] user@host LOCAL_PORT REMOTE_PORT
#   -p PORT          peer SSH port (default 22)
#   -i KEY           identity file
#   -c CIPHER        SSH cipher (default aes128-gcm@openssh.com, AES-NI accelerated)
#   --forward-only   use 'ssh -N' (the previous behaviour; for A/B comparison)
#   -o OPTION        extra ssh -o option (repeatable)
# The ssh process stays in the background; stop it with kill on the printed pid.
set -euo pipefail

port=22
identity=()
cipher=(-c aes128-gcm@openssh.com)
extra=()
forward_only=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        -p) port="$2"; shift 2 ;;
        -i) identity=(-i "$2"); shift 2 ;;
        -c) cipher=(-c "$2"); shift 2 ;;
        -o) extra+=(-o "$2"); shift 2 ;;
        --forward-only) forward_only=1; shift ;;
        -h|--help) sed -n '2,27p' "$0"; exit 0 ;;
        -*) echo "unknown option: $1" >&2; exit 2 ;;
        *) break ;;
    esac
done
if [[ $# -ne 3 ]]; then
    echo "usage: $0 [options] user@host LOCAL_PORT REMOTE_PORT" >&2
    exit 2
fi
target="$1"; local_port="$2"; remote_port="$3"

common=(-o BatchMode=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=15
        -o ServerAliveCountMax=8 -o Compression=no -p "$port"
        "${identity[@]}" "${cipher[@]}" "${extra[@]}"
        -L "127.0.0.1:${local_port}:127.0.0.1:${remote_port}")

if [[ $forward_only -eq 1 ]]; then
    ssh -f -N "${common[@]}" "$target"
else
    # Older clients reject the option; they also lack the obfuscation it disables.
    if ssh -G -o ObscureKeystrokeTiming=no localhost >/dev/null 2>&1; then
        common=(-o ObscureKeystrokeTiming=no "${common[@]}")
    fi
    # Discard only the remote pty output; keep ssh's own errors visible.
    ssh -f -tt "${common[@]}" "$target" "sleep infinity" >/dev/null
fi

pid=$(pgrep -n -f -- "127.0.0.1:${local_port}:127.0.0.1:${remote_port}" || true)
echo "tunnel 127.0.0.1:${local_port} -> ${target}:127.0.0.1:${remote_port} pid=${pid:-unknown} mode=$([[ $forward_only -eq 1 ]] && echo forward-only || echo interactive-nodelay)"
