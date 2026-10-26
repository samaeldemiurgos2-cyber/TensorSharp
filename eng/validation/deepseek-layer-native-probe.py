#!/usr/bin/env python3
"""Compare native layer stages against an unsplit model on a deterministic fixture.

Separate processes exchange exact native boundary bytes over loopback TCP. This
checks the native stage adapter, not the managed communicator or serving stack.
Use the managed distributed probe on actual nodes for those additional lanes.
"""
import argparse
import ctypes as C
import json
import multiprocessing as mp
import os
from pathlib import Path
import socket
import time
import traceback


def run_node(options, node, endpoints, result):
    peers = {}
    handle = None
    try:
        os.environ["TS_DSV41_YOCO_KEEP"] = "0"
        os.environ.setdefault("TS_DSV41_ENGRAM_WARM", "0")
        # Native diagnostics are evidence, including a child's load failure.
        log = open(Path(options["output"]) / f"node-{node}.log", "w")
        os.dup2(log.fileno(), 2)
        if node >= 0:
            listener = socket.socket()
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", endpoints[node]))
            listener.listen(len(endpoints))
            listener.settimeout(60)
            for other in range(node):
                connection = socket.socket()
                connection.settimeout(60)
                until = time.monotonic() + 60
                while True:
                    try:
                        connection.connect(("127.0.0.1", endpoints[other]))
                        break
                    except ConnectionRefusedError:
                        if time.monotonic() > until:
                            raise
                        time.sleep(0.05)
                connection.sendall(node.to_bytes(4, "little"))
                peers[other] = connection
            for _ in range(node + 1, len(endpoints)):
                connection, _ = listener.accept()
                connection.settimeout(60)
                identity = bytearray()
                while len(identity) < 4:
                    fragment = connection.recv(4 - len(identity))
                    if not fragment:
                        raise EOFError("stage peer disconnected during handshake")
                    identity.extend(fragment)
                peers[int.from_bytes(identity, "little")] = connection
            listener.close()
            for connection in peers.values():
                connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        callback_errors = []
        sent = received = 0

        @C.CFUNCTYPE(C.c_bool, C.c_void_p, C.c_int, C.c_int, C.c_void_p, C.c_int)
        def transfer(user, peer, send, pointer, size):
            nonlocal sent, received
            try:
                connection = peers[peer]
                if send:
                    connection.sendall(C.string_at(pointer, size))
                    sent += size
                else:
                    view = memoryview((C.c_ubyte * size).from_address(pointer)).cast("B")
                    offset = 0
                    while offset < size:
                        count = connection.recv_into(view[offset:])
                        if not count:
                            raise EOFError("stage peer disconnected")
                        offset += count
                    received += size
                return True
            except Exception:
                callback_errors.append(traceback.format_exc())
                return False

        dll_directories = []
        if os.name == "nt" and os.environ.get("CUDA_PATH"):
            for directory in (Path(os.environ["CUDA_PATH"]) / "bin", Path(os.environ["CUDA_PATH"]) / "bin/x64"):
                if directory.is_dir():
                    dll_directories.append(os.add_dll_directory(str(directory)))
        lib = C.CDLL(options["library"])
        load = lib.TSGgml_Dsv4LoadModel
        load.argtypes = [C.c_char_p] + [C.c_int] * 4 + [C.c_char_p, C.c_int, C.c_char_p, C.c_int]
        load.restype = C.c_void_p
        distributed = lib.TSGgml_Dsv4LoadModelLayerDistributed
        distributed.argtypes = [C.c_char_p] + [C.c_int] * 5 + [C.c_char_p, C.c_int, C.c_int, C.c_void_p, C.c_void_p]
        distributed.restype = C.c_void_p
        forward = lib.TSGgml_Dsv4Forward
        forward.argtypes = [C.c_void_p, C.c_void_p, C.c_int, C.c_void_p]
        forward.restype = C.c_int
        reset = lib.TSGgml_Dsv4ResetChecked
        reset.argtypes, reset.restype = [C.c_void_p], C.c_int
        free = lib.TSGgml_Dsv4Free
        free.argtypes, free.restype = [C.c_void_p], None
        common = [options["model"].encode(), options["gpus"], 256, options["ubatch"], 2]
        handle = (load(*common, None, 0, options["backend"].encode(), 0) if node < 0 else
                  distributed(*common, 0, options["backend"].encode(), node, len(endpoints), transfer, None))
        if not handle:
            raise RuntimeError("native load failed; see node log")
        tokens = options["tokens"]
        operations = [tokens, [3], [5], None, tokens[:3], tokens[3:8], tokens[8:], [3], [5],
                      None, tokens, [7], None, tokens[:1], tokens[1:]]
        vectors = []
        timings = []
        for ids in operations:
            start = time.perf_counter()
            if ids is None:
                if reset(handle) != 1:
                    raise RuntimeError("native reset failed")
            elif ids:
                native_ids = (C.c_int * len(ids))(*ids)
                output = (C.c_float * options["vocab"])()
                if forward(handle, native_ids, len(ids), output) != 0:
                    raise RuntimeError("native forward failed; " + "".join(callback_errors))
                vectors.append(list(output))
            timings.append(time.perf_counter() - start)
        free(handle)
        handle = None
        result.put({"node": node, "ok": True, "logits": vectors, "seconds": timings,
                    "sent_bytes": sent, "received_bytes": received, "callback_errors": callback_errors})
    except BaseException:
        result.put({"node": node, "ok": False, "error": traceback.format_exc()})
    finally:
        for connection in peers.values():
            connection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nodes", type=int, default=2)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--backend", default="CPU")
    parser.add_argument("--ubatch", type=int, default=8)
    parser.add_argument("--atol", type=float, default=2e-5)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads((args.fixture / "deepseek41.config.json").read_text())
    if not config.get("fixture"):
        raise ValueError("Use the synthetic fixture, not a production checkpoint")
    if not 2 <= args.nodes <= config["config"]["text_config"]["num_hidden_layers"]:
        raise ValueError("Each node must own at least one layer")
    options = dict(library=str(args.library.resolve()), output=str(args.output.resolve()),
                   model=str((args.fixture / "deepseek41-fixture.gguf").resolve()),
                   tokens=json.loads((args.fixture / "tokens.json").read_text()),
                   vocab=config["config"]["text_config"]["vocab_size"], gpus=args.gpus,
                   backend=args.backend, ubatch=args.ubatch)
    context = mp.get_context("spawn")
    result = context.Queue()
    baseline = context.Process(target=run_node, args=(options, -1, [], result))
    baseline.start()
    control = result.get(timeout=180)
    baseline.join(timeout=10)
    if not control["ok"]:
        (args.output / "report.json").write_text(json.dumps(dict(passed=False, control=control), indent=2))
        raise RuntimeError(control)
    reservations = [socket.socket() for _ in range(args.nodes)]
    for connection in reservations:
        connection.bind(("127.0.0.1", 0))
    ports = [connection.getsockname()[1] for connection in reservations]
    for connection in reservations:
        connection.close()
    children = [context.Process(target=run_node, args=(options, node, ports, result))
                for node in range(args.nodes)]
    reports = []
    try:
        for child in children:
            child.start()
        for _ in children:
            reports.append(result.get(timeout=180))
    finally:
        for child in children:
            child.join(timeout=10)
            if child.is_alive():
                child.terminate()
                child.join()
    driver = next((report for report in reports if report["node"] == 0), None)
    import math
    max_error = None
    if driver and all(report["ok"] for report in reports):
        if len(control["logits"]) != len(driver["logits"]):
            raise RuntimeError("Forward counts differ")
        errors = [abs(a-b) for left, right in zip(control["logits"], driver["logits"])
                  for a, b in zip(left, right)]
        if not all(math.isfinite(error) for error in errors):
            raise RuntimeError("Non-finite logit error")
        max_error = max(errors)
    report = dict(control=control, nodes=reports, max_absolute_error=max_error, tolerance=args.atol,
                  passed=max_error is not None and max_error <= args.atol,
                  coverage="synthetic native adapter over loopback TCP; not managed protocol, model quality or throughput")
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({key: report[key] for key in ("passed", "max_absolute_error", "coverage")}))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
