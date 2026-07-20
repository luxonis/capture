#!/usr/bin/env python3
"""
Streaming ToF DAI capture — runs ON the device, sends frames to a host receiver.

Same pipeline as capture_data_tof_dai.py (raw ToF superframe + left/right,
optional RGB), but instead of writing .dai files to the device eMMC, every
frame is serialized (ImgFrame.save() to tmpfs), zstd-compressed and pushed
over TCP to stream_receiver.py running on the host. Capture length is no
longer limited by device RAM or eMMC speed:

  - up to ~150-180 frames (no RGB) the on-device buffers absorb everything
    and the saved sequence is contiguous 30 fps;
  - beyond that the sensor keeps 30 fps but the effective saved rate drops
    to what the network sustains (~21 fps on 2.5 GbE), with dropped frames
    showing up as gaps in the timestamps.

Wire protocol (little-endian), per connection:
  handshake:  u16 folder_name_len, folder_name
  record:     u16 name_len, name, u64 timestamp_ms, u32 raw_len, u32 comp_len,
              comp_len bytes of zstd data
  name == "file:<relpath>" carries a sidecar file, otherwise it is a stream
  name and the payload is one serialized .dai message.
  Connection close = end of records.

Usage (on device):
    PYTHONPATH=/data/pydeps python3 capture_stream_tof_dai.py --host <host-ip> --num-frames 300 --no-rgb
"""

import argparse
import datetime
import json
import os
import queue
import socket
import struct
import threading
import time

os.environ["DEPTHAI_AUTOCALIBRATION"] = "OFF"

import depthai as dai
import zstandard

print(f"[System] DepthAI version: {dai.__version__}")

EEPROM_DEVICE_PATH = "/data/vendor/camera/eeprom_vd55h1.bin"
SHM_DIR = "/dev/shm/tof_stream"


def parse_args():
    parser = argparse.ArgumentParser(description="Stream raw ToF superframes to a host receiver")
    parser.add_argument("--host", required=True, help="Receiver IP address (the host)")
    parser.add_argument("--port", type=int, default=45678, help="Receiver TCP port (default: 45678)")
    parser.add_argument("--socket", default="CAM_D", help="ToF camera board socket (default: CAM_D)")
    parser.add_argument("--num-frames", type=int, default=90, dest="num_frames",
                        help="Number of frame-sets to capture (default: 90)")
    parser.add_argument("--capture-name", default=None, dest="capture_name",
                        help="Optional name for the capture folder")
    parser.add_argument("--no-rgb", action="store_true", dest="no_rgb",
                        help="Disable the RGB stream")
    parser.add_argument("--rgb-resolution", default=None, dest="rgb_resolution",
                        help="RGB output resolution as WIDTHxHEIGHT (default: full sensor)")
    parser.add_argument("--warmup-frames", type=int, default=10, dest="warmup_frames",
                        help="Raw frames to discard before capture starts (default: 10)")
    parser.add_argument("--zstd-level", type=int, default=1, dest="zstd_level",
                        help="zstd compression level (default: 1)")
    parser.add_argument("--senders", type=int, default=3,
                        help="Parallel compress+send threads (default: 3)")
    parser.add_argument("--queue-items", type=int, default=100, dest="queue_items",
                        help="Max frames buffered in RAM awaiting send (default: 100)")
    return parser.parse_args()


def parse_resolution(value):
    try:
        w, h = value.lower().split("x")
        return int(w), int(h)
    except (ValueError, AttributeError):
        raise argparse.ArgumentTypeError(
            f"invalid resolution '{value}', expected WIDTHxHEIGHT (e.g. 1920x1080)")


def connect(host, port, folder):
    sock = socket.create_connection((host, port), timeout=15)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    nb = folder.encode()
    sock.sendall(struct.pack("<H", len(nb)) + nb)
    return sock


def send_record(sock, name, ts, raw_len, payload):
    nb = name.encode()
    sock.sendall(struct.pack("<H", len(nb)) + nb + struct.pack("<QII", ts, raw_len, len(payload)))
    sock.sendall(payload)


def send_sidecars(sock, cctx, device, args, folder, date):
    """Send calib/metadata/info/eeprom as file records over the control connection."""
    calib_tmp = f"{SHM_DIR}/calib.json"
    device.readCalibration().eepromToJsonFile(calib_tmp)
    with open(calib_tmp, "rb") as f:
        calib = f.read()
    os.unlink(calib_tmp)

    try:
        os_version = device.getOSVersion()
    except Exception:
        os_version = None
    metadata = json.dumps({
        "model_name": device.getDeviceName(),
        "mxId": device.getDeviceId(),
        "dai_version": dai.__version__,
        "platform": device.getPlatform().name,
        "os_version": os_version,
        "capture_type": "tof_dai_stream",
        "capture_name": args.capture_name,
        "date": date,
        "zstd_level": args.zstd_level,
    }, indent=4).encode()

    info = (b"This capture was streamed off the device (capture_stream_tof_dai.py).\n"
            b"It is expected to be used on static scenes only.\n"
            b"Frames are stored as native dai messages (.dai), one subfolder per stream.\n")

    files = {"calib.json": calib, "metadata.json": metadata, "info.txt": info}
    if os.path.exists(EEPROM_DEVICE_PATH):
        with open(EEPROM_DEVICE_PATH, "rb") as f:
            files["eeprom_vd55h1.bin"] = f.read()

    for relpath, data in files.items():
        send_record(sock, f"file:{relpath}", 0, len(data), cctx.compress(data))


def sender_worker(args, folder, work_q, stats, lock):
    cctx = zstandard.ZstdCompressor(level=args.zstd_level)
    sock = connect(args.host, args.port, folder)
    tid = threading.get_ident()
    try:
        while True:
            item = work_q.get()
            if item is None:
                work_q.task_done()
                break
            name, ts, msg = item
            tmp = f"{SHM_DIR}/{tid}_{name}_{ts}"
            try:
                msg.save(tmp)  # writes tmp + '.dai'
                del msg
                with open(tmp + ".dai", "rb") as f:
                    data = f.read()
                os.unlink(tmp + ".dai")
                comp = cctx.compress(data)
                send_record(sock, name, ts, len(data), comp)
                with lock:
                    stats["sent"] += 1
                    stats["raw_bytes"] += len(data)
                    stats["comp_bytes"] += len(comp)
            finally:
                work_q.task_done()
    finally:
        sock.close()


def main():
    args = parse_args()
    os.makedirs(SHM_DIR, exist_ok=True)
    tof_socket = getattr(dai.CameraBoardSocket, args.socket)

    device = dai.Device()
    print(f"[Device] Connected: {device.getDeviceName()} ({device.getDeviceId()})")

    date = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    name_part = f"_{args.capture_name.replace('_', '-')}" if args.capture_name else ""
    folder = f"{device.getDeviceName()}_{device.getDeviceId()}{name_part}_{date}"
    print(f"[Stream] Capture folder: {folder}")
    print(f"[Stream] Receiver: {args.host}:{args.port}")

    # Control connection: sidecar files first so the receiver creates the folder
    ctrl = connect(args.host, args.port, folder)
    send_sidecars(ctrl, zstandard.ZstdCompressor(level=args.zstd_level), device, args, folder, date)
    ctrl.close()

    work_q = queue.Queue(maxsize=args.queue_items)
    stats = {"sent": 0, "raw_bytes": 0, "comp_bytes": 0}
    lock = threading.Lock()
    senders = [threading.Thread(target=sender_worker, args=(args, folder, work_q, stats, lock))
               for _ in range(args.senders)]
    for t in senders:
        t.start()

    with dai.Pipeline(device) as pipeline:
        cam = pipeline.create(dai.node.Camera)
        cam.setSensorType(dai.CameraSensorType.TOF)
        cam.build(boardSocket=tof_socket)
        cam_left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
        cam_right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

        # Buffer budget lives in the dai queues + work_q; keep depths bounded so
        # long captures degrade to the sustained network rate instead of OOM.
        raw_q = cam.raw.createOutputQueue(maxSize=32, blocking=False)
        left_q = cam_left.requestFullResolutionOutput().createOutputQueue(maxSize=60, blocking=False)
        right_q = cam_right.requestFullResolutionOutput().createOutputQueue(maxSize=60, blocking=False)
        rgb_q = None
        if not args.no_rgb:
            cam_rgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
            if args.rgb_resolution:
                rgb_out = cam_rgb.requestOutput(parse_resolution(args.rgb_resolution))
            else:
                rgb_out = cam_rgb.requestFullResolutionOutput()
            rgb_q = rgb_out.createOutputQueue(maxSize=20, blocking=False)

        print("\n[Pipeline] Starting...")
        pipeline.start()

        num_captures = 0
        raw_count = 0
        t_start = time.monotonic()
        t_first_saved = None

        try:
            while pipeline.isRunning() and num_captures < args.num_frames:
                raw_frame = raw_q.get()
                raw_count += 1
                left_frame = left_q.tryGet()
                right_frame = right_q.tryGet()
                rgb_frame = rgb_q.tryGet() if rgb_q is not None else None

                if raw_count <= args.warmup_frames:
                    continue
                if t_first_saved is None:
                    t_first_saved = time.monotonic()

                raw_ts = int(raw_frame.getTimestamp().total_seconds() * 1000)
                work_q.put(("tof_raw", raw_ts, raw_frame), block=True)
                for name, frame in (("left", left_frame), ("right", right_frame), ("rgb", rgb_frame)):
                    if frame is not None:
                        ts = int(frame.getTimestamp().total_seconds() * 1000)
                        work_q.put((name, ts, frame), block=True)
                num_captures += 1

                if num_captures % 30 == 1:
                    el = time.monotonic() - t_first_saved
                    fps = num_captures / el if el > 0 else 0
                    print(f"[{num_captures:5d}/{args.num_frames}] fps={fps:.1f} "
                          f"queued={work_q.qsize()} sent={stats['sent']}")
        except KeyboardInterrupt:
            print("\nInterrupted.")
        capture_elapsed = time.monotonic() - (t_first_saved or t_start)
        pipeline.stop()

    print(f"[Capture] {num_captures} frame-sets in {capture_elapsed:.1f}s "
          f"({num_captures / capture_elapsed:.1f} fps), draining send queue...")
    work_q.join()
    for _ in senders:
        work_q.put(None)
    for t in senders:
        t.join(timeout=120)

    total = time.monotonic() - t_start
    ratio = stats["raw_bytes"] / stats["comp_bytes"] if stats["comp_bytes"] else 0
    print(f"\n=== Summary ===")
    print(f"  Frame-sets   : {num_captures}")
    print(f"  Messages sent: {stats['sent']}")
    print(f"  Raw data     : {stats['raw_bytes'] / 1e9:.2f} GB")
    print(f"  Sent (zstd)  : {stats['comp_bytes'] / 1e9:.2f} GB (x{ratio:.2f})")
    print(f"  Capture rate : {num_captures / capture_elapsed:.1f} fps")
    print(f"  Total time   : {total:.1f}s")


if __name__ == "__main__":
    main()
