#!/usr/bin/env python3
"""
ToF DAI Capture Script

Based on capture_data_tof_raw.py, but only captures the raw ToF superframe
(depth/amplitude/intensity/confidence decoding is disabled, no ToFBase node)
plus left/right/RGB, saving every frame with the native dai.ImgFrame.save()
serialization (ProtoSerializable, depthai-core PR #1893) instead of npy/png.
Requires a depthai build that provides the dai.ImgFrame.save() API.

Usage:
    python3 capture_data_tof_dai.py --ip <device-ip>
    python3 capture_data_tof_dai.py --ip <device-ip> --num-frames 50
    python3 capture_data_tof_dai.py --ip <device-ip> --capture-name my-scene

Automatically saves immediately. Press Ctrl+C to stop early.
"""

import argparse
import datetime
import json
import os
import queue
import subprocess
import threading
import time

os.environ["DEPTHAI_AUTOCALIBRATION"] = "OFF"

import cv2

import depthai as dai

print(f"[System] DepthAI version: {dai.__version__}")

script_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.join(script_dir, 'output')

SAVE_QUEUE_MAXSIZE = 200  # max frames buffered for saving; when full, capture blocks until the saver catches up
NUM_SAVER_THREADS = 4  # parallel workers draining the save queue (msg.save() releases the GIL)
STREAM_NAMES = ("tof_raw", "left", "right", "rgb")


def parse_args():
    parser = argparse.ArgumentParser(description="Capture raw ToF superframes as .dai files")
    parser.add_argument("--ip", default=None, help="Device IP address")
    parser.add_argument("--socket", default="CAM_D", help="Camera board socket (default: CAM_D)")
    parser.add_argument("--num-frames", type=int, default=32, dest="num_frames",
                        help="Number of frames to capture (default: 32)")
    parser.add_argument("--capture-name", default=None, dest="capture_name",
                        help="Optional name for the capture folder")
    parser.add_argument("--output", default=root_path, help="Output root folder")
    parser.add_argument("--show-streams", action="store_true", dest="show_streams",
                        help="Show live preview of streams in OpenCV windows")
    parser.add_argument("--no-rgb", action="store_true", dest="no_rgb",
                        help="Disable the RGB stream")
    parser.add_argument("--rgb-resolution", default=None, dest="rgb_resolution",
                        help="RGB output resolution as WIDTHxHEIGHT (e.g. 1920x1080); "
                             "default is the sensor's full resolution")
    return parser.parse_args()


def parse_resolution(value):
    """Parse a 'WIDTHxHEIGHT' string into an (width, height) int tuple."""
    try:
        w, h = value.lower().split("x")
        return int(w), int(h)
    except (ValueError, AttributeError):
        raise argparse.ArgumentTypeError(
            f"invalid resolution '{value}', expected format WIDTHxHEIGHT (e.g. 1920x1080)"
        )


def initialize_capture_folder(output_root, device, capture_name):
    """Create output folder and save calibration + metadata."""
    date = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    device_name = device.getDeviceName()
    device_id = device.getDeviceId()

    if capture_name:
        name = capture_name.replace('_', '-')
        base_name = f"{device_name}_{device_id}_{name}_{date}"
    else:
        base_name = f"{device_name}_{device_id}_{date}"

    out_dir = os.path.join(output_root, base_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[Capture] Folder '{out_dir}' created.")

    calib = device.readCalibration()
    calib.eepromToJsonFile(os.path.join(out_dir, 'calib.json'))

    try:
        os_version = device.getOSVersion()
    except Exception:
        os_version = None

    metadata = {
        "model_name": device_name,
        "mxId": device_id,
        "dai_version": dai.__version__,
        "platform": device.getPlatform().name,
        "os_version": os_version,
        "capture_type": "tof_dai",
        "capture_name": capture_name,
        "date": date,
    }
    with open(os.path.join(out_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)

    with open(os.path.join(out_dir, 'info.txt'), 'w') as f:
        f.write("This capture is done without the sync node.\n")
        f.write("It is expected to be used on static scenes only.\n")
        f.write("Frames are stored as native dai messages (.dai), one subfolder per stream.\n")

    return out_dir


def resolve_device_ip(args):
    """Determine the device IP for the SSH/SCP eeprom fetch.

    The device can be selected either with --ip or via the DEPTHAI_DEVICE_NAME_LIST
    env var (e.g. DEPTHAI_DEVICE_NAME_LIST=<device-ip> python capture_data_tof_dai.py).
    fetch_eeprom needs a reachable IP regardless of which one was used.
    """
    if args.ip:
        return args.ip
    env_list = os.environ.get("DEPTHAI_DEVICE_NAME_LIST", "").strip()
    if env_list:
        first = env_list.split(",")[0].strip()
        if first:
            return first
    return None


def fetch_eeprom(ip, output_folder):
    """SSH into device and copy eeprom_vd55h1.bin to the output folder."""
    remote_path = f"root@{ip}:/data/vendor/camera/eeprom_vd55h1.bin"
    local_path = os.path.join(output_folder, "eeprom_vd55h1.bin")
    print(f"[SCP] Fetching eeprom_vd55h1.bin from {ip}...")
    try:
        subprocess.run(
            ["scp", "-o", "StrictHostKeyChecking=no", remote_path, local_path],
            check=True, timeout=30,
        )
        print(f"[SCP] Saved to {local_path}")
    except subprocess.CalledProcessError as e:
        print(f"[SCP] WARNING: Failed to fetch eeprom file: {e}")
    except subprocess.TimeoutExpired:
        print(f"[SCP] WARNING: SCP timed out after 30s")


def start_capture(args, device, eeprom_ip):
    """Create the capture folder and fetch the eeprom. Returns the output folder."""
    output_folder = initialize_capture_folder(args.output, device, args.capture_name)
    _create_stream_dirs(output_folder)
    if eeprom_ip:
        fetch_eeprom(eeprom_ip, output_folder)
    else:
        print("[SCP] WARNING: no device IP resolved, skipping eeprom fetch")
    return output_folder


def _create_stream_dirs(output_folder):
    """Pre-create one subfolder per stream so the saver hot path does no dir checks."""
    for name in STREAM_NAMES:
        os.makedirs(f'{output_folder}/{name}', exist_ok=True)


def _saver_worker(save_queue):
    while True:
        try:
            item = save_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        if item is None:
            save_queue.task_done()
            break
        output_folder, name, timestamp, msg = item
        try:
            msg.save(f'{output_folder}/{name}/{name}_{timestamp}')
        finally:
            del msg
        save_queue.task_done()


def main():
    args = parse_args()

    if args.ip:
        os.environ["DEPTHAI_DEVICE_NAME_LIST"] = args.ip

    # IP used for the eeprom SCP fetch — works whether the device was selected
    # via --ip or the DEPTHAI_DEVICE_NAME_LIST env var.
    eeprom_ip = resolve_device_ip(args)

    socket = getattr(dai.CameraBoardSocket, args.socket)

    print(f"[ToF] DAI raw streaming: socket={args.socket}")

    if args.ip:
        device = dai.Device(args.ip)
    else:
        device = dai.Device()

    mxid = device.getDeviceId()
    device_name = device.getDeviceName()
    print(f"[Device] Connected: {device_name} ({mxid})")

    save_queue = queue.Queue(maxsize=SAVE_QUEUE_MAXSIZE)
    saver_threads = [
        threading.Thread(target=_saver_worker, args=(save_queue,), daemon=False)
        for _ in range(NUM_SAVER_THREADS)
    ]
    for t in saver_threads:
        t.start()

    with dai.Pipeline(device) as pipeline:
        # ToF camera node
        cam = pipeline.create(dai.node.Camera)
        cam.setSensorType(dai.CameraSensorType.TOF)
        cam.build(boardSocket=socket)

        # Left camera (CAM_B)
        cam_left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
        left_out = cam_left.requestFullResolutionOutput()

        # Right camera (CAM_C)
        cam_right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
        right_out = cam_right.requestFullResolutionOutput()

        # RGB camera (CAM_A) — optional
        rgb_q = None
        if not args.no_rgb:
            cam_rgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
            if args.rgb_resolution:
                rgb_out = cam_rgb.requestOutput(parse_resolution(args.rgb_resolution))
            else:
                rgb_out = cam_rgb.requestFullResolutionOutput()
            rgb_q = rgb_out.createOutputQueue()

        # Output queues
        raw_q = cam.raw.createOutputQueue()
        left_q = left_out.createOutputQueue()
        right_q = right_out.createOutputQueue()

        print("\n[Pipeline] Starting...")
        pipeline.start()

        output_folder = None
        saving = False
        num_captures = 0
        frame_count = 0
        raw_count = 0
        t_start = time.monotonic()

        if args.show_streams:
            print(f"\n[CONTROLS] Press 'S' to START capture, 'Q' to QUIT")
        else:
            # Auto-capture when not showing streams — start saving right away.
            print(f"\n[Capture] Saving {args.num_frames} frames immediately.")
            output_folder = start_capture(args, device, eeprom_ip)
            saving = True
            start_time = time.time()

        try:
            while pipeline.isRunning():
                # Drive from raw (always produced — must be consumed)
                raw_frame = raw_q.get()
                frame_count += 1

                elapsed = time.monotonic() - t_start
                fps = frame_count / elapsed if elapsed > 0 else 0

                left_frame = left_q.tryGet()
                right_frame = right_q.tryGet()
                rgb_frame = rgb_q.tryGet() if rgb_q is not None else None

                if raw_frame is not None:
                    raw_count += 1
                    raw_ts = int(raw_frame.getTimestamp().total_seconds() * 1000)

                    if saving and num_captures < args.num_frames:
                        save_queue.put((output_folder, "tof_raw", raw_ts, raw_frame), block=True)

                        if left_frame is not None:
                            left_ts = int(left_frame.getTimestamp().total_seconds() * 1000)
                            save_queue.put((output_folder, "left", left_ts, left_frame), block=True)

                        if right_frame is not None:
                            right_ts = int(right_frame.getTimestamp().total_seconds() * 1000)
                            save_queue.put((output_folder, "right", right_ts, right_frame), block=True)

                        if rgb_frame is not None:
                            rgb_ts = int(rgb_frame.getTimestamp().total_seconds() * 1000)
                            save_queue.put((output_folder, "rgb", rgb_ts, rgb_frame), block=True)

                        num_captures += 1

                        if num_captures >= args.num_frames:
                            end_time = time.time()
                            elapsed_cap = end_time - start_time
                            print(f"\n[Capture] Done! {num_captures} frames saved in {elapsed_cap:.1f}s")
                            pipeline.stop()
                            break

                # Status every 10 frames
                if frame_count % 10 == 1:
                    status = "CAPTURING" if saving else "IDLE"
                    print(
                        f"[{frame_count:5d}] {status} | raw={raw_frame.getFrame().shape if raw_frame else None} "
                        f"| raw_total={raw_count} saved={num_captures} FPS={fps:.1f}"
                    )

                # Show streams if requested
                if args.show_streams:
                    if left_frame is not None:
                        cv2.imshow("Left", left_frame.getCvFrame())
                    if right_frame is not None:
                        cv2.imshow("Right", right_frame.getCvFrame())
                    if rgb_frame is not None:
                        cv2.imshow("RGB", rgb_frame.getCvFrame())
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        pipeline.stop()
                        break
                    elif key == ord('s'):
                        if not saving:
                            output_folder = start_capture(args, device, eeprom_ip)
                            saving = True
                            start_time = time.time()
                            num_captures = 0
                            print(f"\n[STATUS] >>> CAPTURING {args.num_frames} frames... <<<")

        except KeyboardInterrupt:
            print("\nInterrupted.")

    for _ in saver_threads:
        save_queue.put(None)
    for t in saver_threads:
        t.join(timeout=60)
    if any(t.is_alive() for t in saver_threads):
        print("[Capture] Warning: saver thread(s) did not finish in time")

    total = time.monotonic() - t_start
    print(f"\n=== Summary ===")
    print(f"  Frames       : {frame_count}")
    print(f"  Raw frames   : {raw_count}")
    print(f"  Saved        : {num_captures}")
    print(f"  Duration     : {total:.1f}s")
    if total > 0:
        print(f"  Avg FPS      : {frame_count / total:.2f}")

    if args.show_streams:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
