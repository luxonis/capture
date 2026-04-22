#!/usr/bin/env python3
"""
Stereo capture using holistic recording (lossless FFV1/AVI).

Records left (CAM_B), right (CAM_C), optionally RGB (CAM_A), and IMU streams.
Calibration is saved automatically as camera_info.json inside the recording.
Recording starts when the pipeline starts; press Q to stop.

Usage:
    python capture_data_stereo.py
    python capture_data_stereo.py --settings my_settings.json --output /path/to/output
    python capture_data_stereo.py --ip 10.11.0.42 --capture-name test-scene
    python capture_data_stereo.py --no-streams --num-frames 100
"""

import depthai as dai
import time
import json
import cv2
import os
import argparse
import datetime

from utils import (
    CONTROL_WINDOW_NAME,
    update_control_window,
    initialize_mono_control,
    controlQueueSend,
)

print(f"[System] DepthAI version: {dai.__version__}")

script_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.join(script_dir, 'output')


def parse_arguments():
    parser = argparse.ArgumentParser(description="Record stereo streams using holistic recording.")
    parser.add_argument("--settings", default="capture_settings.json",
                       help="Path to settings JSON file (default: capture_settings.json)")
    parser.add_argument("--output", default=root_path,
                        help="Custom output root folder")
    parser.add_argument("--ip", default=None,
                        help="IP to connect to")
    parser.add_argument("--capture-name", default=None, dest="capture_name",
                       help="Optional name for the capture (will be included in folder name)")
    parser.add_argument("--no-streams", action="store_true",
                       help="Do not show stream windows; use control window for Q")
    parser.add_argument("--num-frames", default=None, type=int,
                       help="Stop after capturing this many frames (default: unlimited)")
    return parser.parse_args()


def build_output_dir(output_root, device, capture_name=None):
    date = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    device_name = device.getDeviceName()
    device_id = device.getDeviceId()
    if capture_name:
        base_name = f"{device_name}_{device_id}_{capture_name}_{date}"
    else:
        base_name = f"{device_name}_{device_id}_{date}"
    return os.path.join(output_root, base_name)


def main(args):
    settings_path = args.settings
    if not os.path.exists(settings_path):
        raise FileNotFoundError(f"Settings file '{settings_path}' does not exist.")

    with open(settings_path) as f:
        settings = json.load(f)

    capture_name = args.capture_name
    if capture_name and '_' in capture_name:
        capture_name = capture_name.replace('_', '-')
        print(f"[Capture] Warning: Underscores replaced with hyphens: {capture_name}")

    ip = args.ip
    print(f"[Device] Connecting to device... IP: {ip}")
    device = dai.Device(ip) if ip else dai.Device()
    mxid = device.getDeviceId()
    device_name = device.getDeviceName()
    print(f"[Device] Device connected!")
    print(f"[Device] Device Name: {device_name}")
    print(f"[Device] Device ID: {mxid}")

    output_dir = build_output_dir(args.output, device, capture_name)
    os.makedirs(output_dir, exist_ok=True)

    # Save capture metadata alongside the holistic recording
    metadata = {
        "model_name": device_name,
        "mxId": mxid,
        "dai_version": dai.__version__,
        "platform": device.getPlatform().name,
        "capture_name": capture_name,
        "date": datetime.datetime.now().strftime("%Y%m%d%H%M%S"),
        "settings_name": settings_path,
        "settings": settings,
    }
    with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"[Capture] Metadata saved to {output_dir}/metadata.json")

    no_streams = args.no_streams
    num_frames_limit = args.num_frames

    if no_streams:
        cv2.namedWindow(CONTROL_WINDOW_NAME)

    with dai.Pipeline(device) as pipeline:
        # Enable holistic recording (lossless FFV1/AVI)
        record_config = dai.RecordConfig()
        record_config.outputDir = output_dir
        record_config.videoEncoding.enabled = False  # lossless FFV1
        record_config.syncCameraOutputs = False
        pipeline.enableHolisticRecord(record_config)

        # Set up cameras
        stereo_w = settings["stereoResolution"]["x"]
        stereo_h = settings["stereoResolution"]["y"]
        fps = settings["FPS"]

        left_cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
        left_out = left_cam.requestOutput((stereo_w, stereo_h), fps=fps)
        left_queue = left_out.createOutputQueue(maxSize=4, blocking=False)

        right_cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
        right_out = right_cam.requestOutput((stereo_w, stereo_h), fps=fps)
        right_queue = right_out.createOutputQueue(maxSize=4, blocking=False)

        rgb_queue = None
        output_settings = settings.get("output_settings", {})
        if output_settings.get("rgb", True):
            rgb_cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
            rgb_w = settings["rgbResolution"]["x"]
            rgb_h = settings["rgbResolution"]["y"]
            rgb_out = rgb_cam.requestOutput((rgb_w, rgb_h), fps=fps)
            rgb_queue = rgb_out.createOutputQueue(maxSize=4, blocking=False)

        # IMU — record accelerometer + gyroscope for replay pipelines
        imu = pipeline.create(dai.node.IMU)
        imu.enableIMUSensor(dai.IMUSensor.ACCELEROMETER_RAW, 100)
        imu.enableIMUSensor(dai.IMUSensor.GYROSCOPE_RAW, 100)
        imu.setBatchReportThreshold(10)
        imu.setMaxBatchReports(10)
        print("[IMU] Recording ACCELEROMETER_RAW @ 100 Hz, GYROSCOPE_RAW @ 100 Hz")

        # Camera control queues
        input_queues = {
            "left_input_control": left_cam.inputControl.createInputQueue(),
            "right_input_control": right_cam.inputControl.createInputQueue(),
        }

        pipeline.start()

        platform = pipeline.getDefaultDevice().getPlatform()
        print(f"[Device] Platform: {platform}")

        if platform == dai.Platform.RVC4:
            control = initialize_mono_control(settings)
            controlQueueSend(input_queues, control)

        if settings.get('ir', False):
            pipeline.getDefaultDevice().setIrLaserDotProjectorIntensity(settings['ir_value'])
        if settings.get('flood_light', False):
            pipeline.getDefaultDevice().setIrFloodLightIntensity(settings['flood_light_intensity'])

        print(f"\n[Capture] Recording to: {output_dir}")
        print("[Capture] Calibration saved automatically (camera_info.json).")
        print(f"[Capture] Streams: left, right"
              f"{', rgb' if rgb_queue else ''}"
              f", IMU")
        print("\n" + "="*60)
        print("[STATUS] >>> RECORDING... <<<")
        print("[CONTROLS] Press 'Q' to STOP and QUIT")
        print("="*60 + "\n")

        num_frames = 0
        start_time = time.time()
        display_failed = False

        while pipeline.isRunning():
            left_frame = left_queue.get()
            right_frame = right_queue.get()

            if left_frame is not None:
                num_frames += 1
                if not no_streams and not display_failed:
                    try:
                        cv2.imshow(f"{mxid} left", left_frame.getCvFrame())
                    except cv2.error:
                        display_failed = True
                        print("[Capture] Warning: Display not available, falling back to --no-streams mode")
                        cv2.namedWindow(CONTROL_WINDOW_NAME)

            if right_frame is not None:
                if not no_streams and not display_failed:
                    cv2.imshow(f"{mxid} right", right_frame.getCvFrame())

            if rgb_queue is not None:
                rgb_frame = rgb_queue.get()
                if rgb_frame is not None and not no_streams and not display_failed:
                    cv2.imshow(f"{mxid} rgb", rgb_frame.getCvFrame())

            if no_streams or display_failed:
                update_control_window(True, num_frames)

            if num_frames_limit and num_frames >= num_frames_limit:
                print(f"[Capture] Reached frame limit ({num_frames_limit})")
                break

            if cv2.waitKey(1) == ord('q'):
                break

        # Keep consuming frames so the pipeline continues processing
        # and the recording thread can flush remaining buffered frames.
        print("[Capture] Flushing recording...")
        flush_start = time.time()
        while pipeline.isRunning() and (time.time() - flush_start) < 10:
            try:
                left_queue.get(timeout=datetime.timedelta(seconds=1))
                right_queue.get(timeout=datetime.timedelta(seconds=1))
                if rgb_queue is not None:
                    rgb_queue.get(timeout=datetime.timedelta(seconds=1))
            except Exception:
                break
            cv2.waitKey(1)

        pipeline.stop()
        pipeline.wait()
        cv2.destroyAllWindows()

    elapsed = time.time() - start_time
    print(f"\n[Capture] Recording finished. {num_frames} frames in {elapsed:.1f}s "
          f"({num_frames / max(elapsed, 0.001):.1f} FPS)")
    print(f"[Capture] Output: {output_dir}")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
