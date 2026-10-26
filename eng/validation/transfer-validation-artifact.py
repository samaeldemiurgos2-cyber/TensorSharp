#!/usr/bin/env python3
"""Transfer one validation artifact over an existing authenticated SSH forward.

Receiver binds loopback only; sender connects loopback only. The destination
must be fresh. A SHA256 verifies bytes before the partial file is promoted.
"""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import struct


def receive(connection, count):
    result = bytearray()
    while len(result) < count:
        block = connection.recv(min(count - len(result), 1048576))
        if not block:
            raise EOFError("Transfer closed early")
        result.extend(block)
    return bytes(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--send", type=Path)
    action.add_argument("--receive", type=Path)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    if args.send:
        with args.send.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").digest()
            source.seek(0, 2)
            size = source.tell()
            source.seek(0)
            with socket.create_connection(("127.0.0.1", args.port), timeout=120) as connection:
                connection.sendall(struct.pack("<Q", size) + digest)
                while block := source.read(1048576):
                    connection.sendall(block)
                if receive(connection, 32) != digest:
                    raise ValueError("Receiver did not acknowledge SHA256")
        print(json.dumps({"path": str(args.send), "bytes": size, "sha256": digest.hex()}))
    else:
        destination = args.receive.resolve()
        partial = destination.with_suffix(destination.suffix + ".partial")
        if destination.exists() or partial.exists():
            parser.error("Output and partial output must be fresh")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.settimeout(120)
            listener.bind(("127.0.0.1", args.port))
            listener.listen(1)
            connection, _ = listener.accept()
            with connection:
                connection.settimeout(120)
                size = struct.unpack("<Q", receive(connection, 8))[0]
                if size > 16 * 1024 ** 3:
                    raise ValueError("Validation artifact exceeds 16 GiB")
                expected, digest, remaining = receive(connection, 32), hashlib.sha256(), size
                with partial.open("xb") as output:
                    while remaining:
                        block = receive(connection, min(remaining, 1048576))
                        output.write(block)
                        digest.update(block)
                        remaining -= len(block)
                if digest.digest() != expected:
                    raise ValueError("SHA256 mismatch; retained partial file")
                partial.rename(destination)
                connection.sendall(expected)
        print(json.dumps({"path": str(destination), "bytes": size, "sha256": expected.hex()}))


if __name__ == "__main__":
    main()
