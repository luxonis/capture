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


class StatusLed:
    """Device RGB status LED: red=capturing, green=draining, blue=idle (OS default)."""

    LEDS = ("red", "green", "blue")

    def __init__(self, enabled=True):
        self.available = enabled and all(
            os.path.isdir(f"/sys/class/leds/{c}") for c in self.LEDS)

    def _write(self, color, attr, value):
        try:
            with open(f"/sys/class/leds/{color}/{attr}", "w") as f:
                f.write(str(value))
        except OSError:
            pass

    def _solid(self, red=0, green=0, blue=0):
        if not self.available:
            return
        for color, value in zip(self.LEDS, (red, green, blue)):
            self._write(color, "trigger", "none")
            self._write(color, "brightness", value)

    def capturing(self):
        self._solid(red=255)

    def flushing(self):
        self._solid(green=255)

    def restore(self):
        if not self.available:
            return
        self._solid()
        self._write("blue", "trigger", "timer")
        self._write("blue", "brightness", 255)


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
                        help="zstd compression level; 0 = send uncompressed (default: 1)")
    parser.add_argument("--senders", type=int, default=3,
                        help="Parallel compress+send threads (default: 3)")
    parser.add_argument("--queue-items", type=int, default=100, dest="queue_items",
                        help="Max frames buffered in RAM awaiting send (default: 100)")
    parser.add_argument("--ram-buffer", action="store_true", dest="ram_buffer",
                        help="Hold the whole capture in RAM (guaranteed contiguous 30 fps), "
                             "then send it over the network after the pipeline stops; "
                             "num-frames is clamped to available RAM")
    parser.add_argument("--no-led", action="store_true", dest="no_led",
                        help="Do not drive the device status LED")
    return parser.parse_args()


def available_ram_bytes():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def parse_resolution(value):
    try:
        w, h = value.lower().split("x")
        return int(w), int(h)
    except (ValueError, AttributeError):
        raise argparse.ArgumentTypeError(
            f"invalid resolution '{value}', expected WIDTHxHEIGHT (e.g. 1920x1080)")


def latest(q):
    """Drain an output queue and return only the newest message (or None), so
    side streams pair with the current raw frame instead of a stale backlog."""
    msg = None
    while True:
        m = q.tryGet()
        if m is None:
            return msg
        msg = m


def connect(host, port, folder, compressed):
    sock = socket.create_connection((host, port), timeout=15)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    nb = folder.encode()
    sock.sendall(struct.pack("<H", len(nb)) + nb + struct.pack("<B", 1 if compressed else 0))
    return sock


def make_compressor(level):
    """Return a bytes->bytes payload function; identity when level <= 0."""
    if level <= 0:
        return lambda data: data
    return zstandard.ZstdCompressor(level=level).compress


def send_record(sock, name, ts, raw_len, payload):
    nb = name.encode()
    sock.sendall(struct.pack("<H", len(nb)) + nb + struct.pack("<QII", ts, raw_len, len(payload)))
    sock.sendall(payload)


def send_sidecars(sock, compress, device, args, folder, date):
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
        send_record(sock, f"file:{relpath}", 0, len(data), compress(data))


def sender_worker(args, folder, work_q, stats, lock):
    compress = make_compressor(args.zstd_level)
    sock = connect(args.host, args.port, folder, args.zstd_level > 0)
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
                comp = compress(data)
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

    led = StatusLed(enabled=not args.no_led)
    if led.available:
        print("[LED] Status LED: red=capturing, green=sending, blue=idle")

    date = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    name_part = f"_{args.capture_name.replace('_', '-')}" if args.capture_name else ""
    folder = f"{device.getDeviceName()}_{device.getDeviceId()}{name_part}_{date}"
    print(f"[Stream] Capture folder: {folder}")
    print(f"[Stream] Receiver: {args.host}:{args.port}")

    # Control connection: sidecar files first so the receiver creates the folder
    ctrl = connect(args.host, args.port, folder, args.zstd_level > 0)
    send_sidecars(ctrl, make_compressor(args.zstd_level), device, args, folder, date)
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
        # In --ram-buffer mode the python list is the buffer and the loop drains
        # fast, but deeper queues cheaply absorb any transient stall.
        raw_depth = 64 if args.ram_buffer else 32
        raw_q = cam.raw.createOutputQueue(maxSize=raw_depth, blocking=False)
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

        # In --ram-buffer mode the dai queues must hold the entire capture window
        # at sensor rate; frames are collected here and sent after the pipeline stops.
        ram_frames = []

        def enqueue(item):
            if args.ram_buffer:
                ram_frames.append(item)
            else:
                work_q.put(item, block=True)

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
                left_frame = latest(left_q)
                right_frame = latest(right_q)
                rgb_frame = latest(rgb_q) if rgb_q is not None else None

                if raw_count <= args.warmup_frames:
                    continue
                if t_first_saved is None:
                    t_first_saved = time.monotonic()
                    led.capturing()

                raw_ts = int(raw_frame.getTimestamp().total_seconds() * 1000)
                enqueue(("tof_raw", raw_ts, raw_frame))
                for name, frame in (("left", left_frame), ("right", right_frame), ("rgb", rgb_frame)):
                    if frame is not None:
                        ts = int(frame.getTimestamp().total_seconds() * 1000)
                        enqueue((name, ts, frame))
                num_captures += 1

                # Clamp num_frames to available RAM after the first buffered set
                if num_captures == 1 and args.ram_buffer:
                    set_bytes = sum(len(m.getData()) for (_, _, m) in ram_frames)
                    avail = available_ram_bytes()
                    if avail and set_bytes:
                        safe_max = int(avail * 0.75 / (set_bytes * 1.4))
                        if args.num_frames > safe_max:
                            print(f"[RAM] Clamping {args.num_frames} -> {safe_max} frames "
                                  f"({set_bytes / 1e6:.0f} MB/set, {avail / 1e9:.1f} GB available)")
                            args.num_frames = safe_max

                if num_captures % 30 == 1:
                    el = time.monotonic() - t_first_saved
                    fps = num_captures / el if el > 0 else 0
                    print(f"[{num_captures:5d}/{args.num_frames}] fps={fps:.1f} "
                          f"queued={work_q.qsize()} sent={stats['sent']}")
        except KeyboardInterrupt:
            print("\nInterrupted.")
        capture_elapsed = time.monotonic() - (t_first_saved or t_start)
        pipeline.stop()

    led.flushing()
    if ram_frames:
        print(f"[Send] Streaming {len(ram_frames)} RAM-buffered frames to host...")
        t_send = time.monotonic()
        for item in ram_frames:
            work_q.put(item, block=True)
        ram_frames.clear()
        work_q.join()
        print(f"[Send] Done in {time.monotonic() - t_send:.1f}s")

    print(f"[Capture] {num_captures} frame-sets in {capture_elapsed:.1f}s "
          f"({num_captures / capture_elapsed:.1f} fps), draining send queue...")
    work_q.join()
    for _ in senders:
        work_q.put(None)
    for t in senders:
        t.join(timeout=120)
    led.restore()

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
