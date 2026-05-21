#!/usr/bin/env python3
"""
ToF Raw Data Capture Script

Based on tof_raw_stream.py. Captures raw ToF superframes, depth maps, and
amplitude maps and saves them as .npy files.

Usage:
    python3 capture_data_tof_raw.py --ip 10.11.102.225
    python3 capture_data_tof_raw.py --ip 10.11.102.225 --num-frames 50
    python3 capture_data_tof_raw.py --ip 10.11.102.225 --capture-name my-scene
    python3 capture_data_tof_raw.py --ip 10.11.102.225 --fwp /path/to/firmware.tar.xz

Automatically saves after warmup. Press Ctrl+C to stop early.
"""

import argparse
import datetime
import json
import os
import time

import cv2
import numpy as np

import depthai as dai

print(f"[System] DepthAI version: {dai.__version__}")

script_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.join(script_dir, 'output')


def parse_args():
    parser = argparse.ArgumentParser(description="Capture raw ToF data (raw, depth, amplitude)")
    parser.add_argument("--ip", default=None, help="Device IP address")
    parser.add_argument("--socket", default="CAM_D", help="Camera board socket (default: CAM_D)")
    parser.add_argument("--preset", choices=["low", "mid", "high"], default="high",
                        help="ToF preset mode (default: high)")
    parser.add_argument("--num-frames", type=int, default=16, dest="num_frames",
                        help="Number of frames to capture (default: 16)")
    parser.add_argument("--capture-name", default=None, dest="capture_name",
                        help="Optional name for the capture folder")
    parser.add_argument("--output", default=root_path, help="Output root folder")
    parser.add_argument("--fwp", required=True, help="Path to custom RVC4 firmware package (.tar.xz)")
    parser.add_argument("--show-streams", action="store_true", dest="show_streams",
                        help="Show live preview of streams in OpenCV windows")
    parser.add_argument("--skip-warmup", action="store_true", dest="skip_warmup",
                        help="Skip warmup frames")
    parser.add_argument("--warmup-frames", type=int, default=30, dest="warmup_frames",
                        help="Number of warmup frames to skip (default: 30)")
    return parser.parse_args()


def initialize_capture_folder(output_root, device, capture_name):
    """Create output folder and save calibration + metadata."""
    date = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    device_name = device.getDeviceName()
    device_id = device.getMxId()

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

    metadata = {
        "model_name": device_name,
        "mxId": device_id,
        "dai_version": dai.__version__,
        "platform": device.getPlatform().name,
        "capture_type": "tof_raw",
        "capture_name": capture_name,
        "date": date,
    }
    with open(os.path.join(out_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)

    with open(os.path.join(out_dir, 'info.txt'), 'w') as f:
        f.write("This capture is done without the sync node.\n")
        f.write("It is expected to be used on static scenes only.\n")

    return out_dir


def main():
    args = parse_args()

    if args.ip:
        os.environ["DEPTHAI_DEVICE_NAME_LIST"] = args.ip
    os.environ["DEPTHAI_DEVICE_RVC4_FWP"] = args.fwp

    socket = getattr(dai.CameraBoardSocket, args.socket)
    preset_map = {
        "low": dai.ImageFiltersPresetMode.TOF_LOW_RANGE,
        "mid": dai.ImageFiltersPresetMode.TOF_MID_RANGE,
        "high": dai.ImageFiltersPresetMode.TOF_HIGH_RANGE,
    }
    preset_mode = preset_map[args.preset]

    print(f"[ToF] Raw streaming: socket={args.socket}, preset={args.preset}")

    if args.ip:
        device = dai.Device(args.ip)
    else:
        device = dai.Device()

    mxid = device.getDeviceId()
    device_name = device.getDeviceName()
    print(f"[Device] Connected: {device_name} ({mxid})")

    with dai.Pipeline(device) as pipeline:
        # ToF camera node
        cam = pipeline.create(dai.node.Camera)
        cam.setSensorType(dai.CameraSensorType.TOF)
        cam.build(boardSocket=socket)

        # ToFBase node (decoder)
        tof_base = pipeline.create(dai.node.ToFBase)
        tof_base.build(boardSocket=socket, presetMode=preset_mode)

        # Link camera raw → ToF decoder
        cam.raw.link(tof_base.rawInput)

        # Left camera (CAM_B)
        cam_left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
        left_out = cam_left.requestFullResolutionOutput()

        # Right camera (CAM_C)
        cam_right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
        right_out = cam_right.requestFullResolutionOutput()

        # RGB camera (CAM_A)
        cam_rgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        rgb_out = cam_rgb.requestFullResolutionOutput()

        # Output queues
        raw_q = tof_base.raw.createOutputQueue()
        depth_q = tof_base.depth.createOutputQueue()
        amp_q = tof_base.amplitude.createOutputQueue()
        left_q = left_out.createOutputQueue()
        right_q = right_out.createOutputQueue()
        rgb_q = rgb_out.createOutputQueue()

        print("\n[Pipeline] Starting...")
        pipeline.start()

        output_folder = None
        saving = False
        num_captures = 0
        frame_count = 0
        raw_count = 0
        warmup_done = args.skip_warmup
        capture_requested = not args.show_streams  # auto-capture when not showing streams
        t_start = time.monotonic()

        if args.show_streams:
            print(f"\n[Warmup] Skipping first {args.warmup_frames} frames...")
            print(f"[CONTROLS] Press 'S' to START capture after warmup, 'Q' to QUIT")
        else:
            # Waiting for warmup frames, after which it will automatically save num_frames frames
            print(f"\n[Warmup] Waiting for {args.warmup_frames} warmup frames...")
            print(f"[Capture] After warmup, will automatically save {args.num_frames} frames")

        try:
            while pipeline.isRunning():
                # Drive from depth (always produced — must be consumed)
                depth_frame = depth_q.get()
                frame_count += 1

                # Get amplitude
                amp_frame = amp_q.tryGet()

                # Get raw (passthrough)
                raw_frame = raw_q.tryGet()

                elapsed = time.monotonic() - t_start
                fps = frame_count / elapsed if elapsed > 0 else 0

                # Handle warmup
                if not warmup_done:
                    if frame_count >= args.warmup_frames:
                        warmup_done = True
                        if capture_requested:
                            # No --show-streams: start saving immediately after warmup
                            output_folder = initialize_capture_folder(
                                args.output, device, args.capture_name)
                            saving = True
                            start_time = time.time()
                            print(f"\n[Capture] Warmup done. Saving {args.num_frames} frames...")
                        else:
                            print(f"\n[Warmup] Done! Press 'S' to start capture.")
                    # During warmup, still show streams if requested
                    if args.show_streams:
                        left_q.tryGet()
                        right_q.tryGet()
                        rgb_q.tryGet()
                        if depth_frame is not None:
                            d = depth_frame.getFrame()
                            dv = (d.astype(np.float32) / d.max() * 255).astype(np.uint8) if d.max() > 0 else np.zeros_like(d, dtype=np.uint8)
                            cv2.imshow("Depth", cv2.applyColorMap(dv, cv2.COLORMAP_JET))
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            pipeline.stop()
                            break
                    continue

                # Get left/right/rgb frames
                left_frame = left_q.tryGet()
                right_frame = right_q.tryGet()
                rgb_frame = rgb_q.tryGet()

                if raw_frame is not None:
                    raw_count += 1
                    raw_data = raw_frame.getFrame()
                    raw_timestamp = int(raw_frame.getTimestamp().total_seconds() * 1000)

                    if saving and num_captures < args.num_frames:
                        if num_captures == 0:
                            print("[Processing] Applying horizontal flip + 90° rotation to depth and amplitude")

                        # Save raw superframe
                        np.save(f'{output_folder}/raw_{raw_timestamp}.npy', raw_data)

                        # Save depth (npy + colorized png) — mirrored vertically then rotated 90°
                        depth_data = depth_frame.getFrame()
                        depth_data = np.flip(depth_data, axis=1)  # mirror along vertical axis
                        depth_data = np.ascontiguousarray(np.rot90(depth_data))  # rotate 90°
                        depth_ts = int(depth_frame.getTimestamp().total_seconds() * 1000)
                        np.save(f'{output_folder}/depth_{depth_ts}.npy', depth_data)
                        depth_vis = (depth_data.astype(np.float32) / depth_data.max() * 255).astype(np.uint8) if depth_data.max() > 0 else np.zeros_like(depth_data, dtype=np.uint8)
                        depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
                        cv2.imwrite(f'{output_folder}/depth_color_{depth_ts}.png', depth_color)

                        # Save amplitude (npy + greyscale png) — mirrored horizontally then rotated 90°
                        if amp_frame is not None:
                            amp_data = amp_frame.getFrame()
                            amp_data = np.flip(amp_data, axis=1)  # mirror along vertical axis
                            amp_data = np.ascontiguousarray(np.rot90(amp_data))  # rotate 90°
                            amp_ts = int(amp_frame.getTimestamp().total_seconds() * 1000)
                            np.save(f'{output_folder}/amplitude_{amp_ts}.npy', amp_data)
                            amp_vis = (amp_data.astype(np.float32) / amp_data.max() * 255).astype(np.uint8) if amp_data.max() > 0 else np.zeros_like(amp_data, dtype=np.uint8)
                            cv2.imwrite(f'{output_folder}/amplitude_vis_{amp_ts}.png', amp_vis)

                        # Save left as PNG
                        if left_frame is not None:
                            left_data = left_frame.getCvFrame()
                            left_ts = int(left_frame.getTimestamp().total_seconds() * 1000)
                            cv2.imwrite(f'{output_folder}/left_{left_ts}.png', left_data)

                        # Save right as PNG
                        if right_frame is not None:
                            right_data = right_frame.getCvFrame()
                            right_ts = int(right_frame.getTimestamp().total_seconds() * 1000)
                            cv2.imwrite(f'{output_folder}/right_{right_ts}.png', right_data)

                        # Save RGB as PNG
                        if rgb_frame is not None:
                            rgb_data = rgb_frame.getCvFrame()
                            rgb_ts = int(rgb_frame.getTimestamp().total_seconds() * 1000)
                            cv2.imwrite(f'{output_folder}/rgb_{rgb_ts}.png', rgb_data)

                        num_captures += 1

                        if num_captures >= args.num_frames:
                            end_time = time.time()
                            elapsed_cap = end_time - start_time
                            print(f"\n[Capture] Done! {num_captures} frames saved in {elapsed_cap:.1f}s")
                            pipeline.stop()
                            break

                # Status every 10 frames
                if frame_count % 10 == 1:
                    depth_data = depth_frame.getFrame()
                    raw_info = f"raw={raw_data.shape}" if raw_frame else "raw=None"
                    status = "CAPTURING" if saving else "IDLE"
                    print(
                        f"[{frame_count:5d}] {status} | depth={depth_data.shape} {raw_info} "
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
                    if depth_frame is not None:
                        d = depth_frame.getFrame()
                        dv = (d.astype(np.float32) / d.max() * 255).astype(np.uint8) if d.max() > 0 else np.zeros_like(d, dtype=np.uint8)
                        cv2.imshow("Depth", cv2.applyColorMap(dv, cv2.COLORMAP_JET))
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        pipeline.stop()
                        break
                    elif key == ord('s'):
                        if not saving and warmup_done:
                            output_folder = initialize_capture_folder(
                                args.output, device, args.capture_name)
                            saving = True
                            start_time = time.time()
                            num_captures = 0
                            print(f"\n[STATUS] >>> CAPTURING {args.num_frames} frames... <<<")

        except KeyboardInterrupt:
            print("\nInterrupted.")

    total = time.monotonic() - t_start
    print(f"\n=== Summary ===")
    print(f"  Depth frames : {frame_count}")
    print(f"  Raw frames   : {raw_count}")
    print(f"  Saved        : {num_captures}")
    print(f"  Duration     : {total:.1f}s")
    if total > 0:
        print(f"  Avg FPS      : {frame_count / total:.2f}")

    if args.show_streams:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
