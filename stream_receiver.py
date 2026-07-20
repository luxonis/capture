#!/usr/bin/env python3
"""
Host-side receiver for capture_stream_tof_dai.py.

Listens on a TCP port, accepts connections from the device sender, and writes
the received frames as the standard .dai capture layout:
    <output>/<capture_folder>/<stream>/<stream>_<timestamp>.dai
plus sidecar files (calib.json, metadata.json, info.txt, eeprom_vd55h1.bin).

Exits when at least one connection was made and all connections have closed.
See capture_stream_tof_dai.py for the wire protocol.
"""

import argparse
import os
import socket
import struct
import threading
import time

import zstandard


def recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            if buf:
                raise ConnectionError("connection closed mid-record")
            return None
        buf.extend(chunk)
    return bytes(buf)


def handle_connection(sock, output_root, stats, lock):
    dctx = zstandard.ZstdDecompressor()
    hdr = recv_exact(sock, 2)
    if hdr is None:
        return
    (folder_len,) = struct.unpack("<H", hdr)
    folder = recv_exact(sock, folder_len).decode()
    out_dir = os.path.join(output_root, os.path.basename(folder))
    os.makedirs(out_dir, exist_ok=True)
    with lock:
        if stats.get("folder") is None:
            stats["folder"] = out_dir
            print(f"[Receiver] Writing capture to {out_dir}")

    while True:
        hdr = recv_exact(sock, 2)
        if hdr is None:
            break
        (name_len,) = struct.unpack("<H", hdr)
        name = recv_exact(sock, name_len).decode()
        ts, raw_len, comp_len = struct.unpack("<QII", recv_exact(sock, 16))
        payload = recv_exact(sock, comp_len)
        data = dctx.decompress(payload, max_output_size=raw_len)

        if name.startswith("file:"):
            relpath = os.path.basename(name[len("file:"):])
            path = os.path.join(out_dir, relpath)
        else:
            stream = os.path.basename(name)
            stream_dir = os.path.join(out_dir, stream)
            os.makedirs(stream_dir, exist_ok=True)
            path = os.path.join(stream_dir, f"{stream}_{ts}.dai")
        with open(path, "wb") as f:
            f.write(data)
        with lock:
            stats["files"] += 1
            stats["bytes"] += len(data)
            stats["comp_bytes"] += len(payload)
            stats["per_stream"][name] = stats["per_stream"].get(name, 0) + 1


def main():
    parser = argparse.ArgumentParser(description="Receive a streamed ToF capture")
    parser.add_argument("--port", type=int, default=45678)
    parser.add_argument("--output", default="output", help="Output root folder (default: output)")
    parser.add_argument("--idle-timeout", type=float, default=60.0, dest="idle_timeout",
                        help="Exit if no connection arrives within this many seconds (default: 60)")
    args = parser.parse_args()

    server = socket.socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", args.port))
    server.listen(8)
    server.settimeout(1.0)
    print(f"[Receiver] Listening on port {args.port}")

    stats = {"files": 0, "bytes": 0, "comp_bytes": 0, "per_stream": {}, "folder": None}
    lock = threading.Lock()
    threads = []
    t_start = time.monotonic()
    had_connection = False

    while True:
        try:
            sock, addr = server.accept()
            had_connection = True
            t = threading.Thread(target=handle_connection, args=(sock, args.output, stats, lock))
            t.start()
            threads.append(t)
        except socket.timeout:
            threads = [t for t in threads if t.is_alive()]
            if had_connection and not threads:
                break
            if not had_connection and time.monotonic() - t_start > args.idle_timeout:
                print("[Receiver] No connection received, exiting.")
                return 1

    elapsed = time.monotonic() - t_start
    per_stream = ", ".join(f"{k}={v}" for k, v in sorted(stats["per_stream"].items())
                           if not k.startswith("file:"))
    print(f"\n[Receiver] Done: {stats['files']} files, {stats['bytes'] / 1e9:.2f} GB "
          f"({stats['comp_bytes'] / 1e9:.2f} GB on the wire) in {elapsed:.1f}s")
    print(f"[Receiver] Frames per stream: {per_stream}")
    print(f"[Receiver] Capture folder: {stats['folder']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
