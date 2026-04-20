#!/usr/bin/env python3

import depthai as dai
import numpy as np
import time
import json
import csv
import cv2
import os
import argparse
import queue
import threading

from utils import *
from pipeline import initialize_pipeline

SAVE_QUEUE_MAXSIZE = 200  # max frames buffered for saving; when full, capture blocks until the saver catches up


def _saver_worker(save_queue):
    while True:
        try:
            item = save_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        if item is None:
            save_queue.task_done()
            break
        output_folder, name, timestamp, frame, do_npy, do_png = item
        try:
            if do_npy: np.save(f'{output_folder}/{name}_{timestamp}.npy', frame)
            if do_png: cv2.imwrite(f'{output_folder}/{name}_{timestamp}.png', frame)
        finally:
            del frame
        save_queue.task_done()


def _imu_collector_thread(imu_queue, imu_buffer, imu_lock, stop_event):
    """Drain the IMU output queue and accumulate packets into imu_buffer.

    Runs as a daemon thread for the entire script lifetime so the hardware
    queue never overflows.  Call flush_imu_to_csv() to persist the data.
    """
    while not stop_event.is_set():
        try:
            imu_data = imu_queue.get(timeout=0.5)
        except Exception:
            continue
        for packet in imu_data.packets:
            accel = packet.acceleroMeter
            gyro  = packet.gyroscope
            ts_ns = int(accel.getTimestamp().total_seconds() * 1e9)
            row = (ts_ns, accel.x, accel.y, accel.z, gyro.x, gyro.y, gyro.z)
            with imu_lock:
                imu_buffer.append(row)


def flush_imu_to_csv(output_folder, imu_buffer, imu_lock):
    """Write buffered IMU rows to <output_folder>/imu.csv and clear the buffer."""
    with imu_lock:
        rows = list(imu_buffer)
        imu_buffer.clear()

    if not rows:
        print("[IMU] No IMU data to save.")
        return

    rows.sort(key=lambda r: r[0])
    csv_path = os.path.join(output_folder, 'imu.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp_ns', 'accel_x', 'accel_y', 'accel_z',
                         'gyro_x', 'gyro_y', 'gyro_z'])
        writer.writerows(rows)
    print(f"[IMU] Saved {len(rows)} IMU measurements to {csv_path}")


def _save_imu_sensor_to_metadata(output_folder, imu_sensor_name):
    """Append the detected IMU sensor name to the capture's metadata.json."""
    metadata_path = os.path.join(output_folder, 'metadata.json')
    try:
        with open(metadata_path) as f:
            metadata = json.load(f)
        metadata['imu_sensor'] = imu_sensor_name
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
    except Exception as e:
        print(f"[IMU] Warning: could not update metadata with IMU sensor name: {e}")


print(f"[System] DepthAI version: {dai.__version__}")

script_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.join(script_dir, 'output')


def parse_arguments(root_path):
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", default="capture_settings.json",
                       help="Path to settings JSON file (default: capture_settings.json)")
    parser.add_argument("--output", default=root_path,
                        help="Custom output folder")
    parser.add_argument("--autostart", default=-1, type=int,
                       help='Automatically start capturing after given number of seconds (-1 to disable)')
    parser.add_argument("--ip", default=None, dest="ip",
                        help="IP to connect to")
    parser.add_argument("--autostart_time", default=0,
                       help="Select a fixed time when the script is supposed to start")
    parser.add_argument("--autostart_end", default=0,
                       help="Select a fixed time for capture to end")
    parser.add_argument("--capture-name", default=None, dest="capture_name",
                       help="Optional name for the capture (will be included in folder name)")
    parser.add_argument("--no-streams", action="store_true",
                       help="Do not show stream windows (faster capture); use control window for S/Q")
    parser.add_argument("--png", action="store_true",
                       help="Save left, right, rgb as PNG (disables npy unless --npy is also set)")
    parser.add_argument("--npy", action="store_true",
                       help="Save frames as numpy (default when no format option is set)")
    return parser.parse_args()


def main(args):
    output_root = args.output
    settings_path, ip, autostart, autostart_time, wait_end, capture_name = process_argument_logic(args)
    print(f"[Device] Connecting to device... IP: {ip}")

    if ip is not None:
        device = dai.Device(ip)
    else:
        device = dai.Device()
    mxid = device.getDeviceId()

    device_name = device.getDeviceName()
    print("[Device] Device connected!")
    print(f"[Device] Device Name: {device_name}")
    print(f"[Device] Device ID: {mxid}")

    with open(settings_path) as settings_file:
        settings = json.load(settings_file)

    output_folder = None
    num_captures = 0

    save = False

    streams = count_output_streams(settings['output_settings'])
    if settings['num_captures'] == 'inf' or settings['num_captures'] == 'INF':
        settings['num_captures'] = float('inf')
    final_num_captures = settings['num_captures'] * len(streams)
    capture_limit_str = "until stopped" if settings['num_captures'] == float('inf') else f"{int(settings['num_captures'])} frames per stream"
    print(f"[Streams] Active streams: {streams}")
    print(f"[Streams] Number of streams: {len(streams)}")
    print(f"[Capture] Will capture max frames ({settings['num_captures']}) * number of streams ({len(streams)}) = {final_num_captures}")

    initial_time = time.time()
    if autostart_time:
        print(f"[Capture] Waiting till: {autostart_time}")
    elif autostart >= 0:
        print(f"[Capture] Capture will start automatically after {autostart} seconds")
    else:
        print("\n" + "="*60)
        print("[CONTROLS] Press 'S' to START capturing")
        print("[CONTROLS] Press 'Q' to QUIT")
        print("="*60 + "\n")

    no_streams = getattr(args, 'no_streams', False)
    save_npy = args.npy or not args.png
    save_png = args.png
    png_streams = ('left', 'right', 'rgb')
    save_queue = queue.Queue(maxsize=SAVE_QUEUE_MAXSIZE)
    saver_thread = threading.Thread(target=_saver_worker, args=(save_queue,), daemon=False)
    saver_thread.start()
    if no_streams:
        cv2.namedWindow(CONTROL_WINDOW_NAME)

    # IMU shared state
    imu_buffer = []
    imu_lock = threading.Lock()
    imu_stop_event = threading.Event()

    with dai.Pipeline(device) as pipeline:
        pipeline, q, input_queues, stereo_settings = initialize_pipeline(pipeline, settings)

        # --- IMU node (added after initialize_pipeline, before pipeline.start()) ---
        imu_cfg = settings.get('imu_settings', {})
        imu_frequency = int(imu_cfg.get('frequency', 100))
        imu_node = pipeline.create(dai.node.IMU)
        imu_node.enableIMUSensor(dai.IMUSensor.ACCELEROMETER_RAW, imu_frequency)
        imu_node.enableIMUSensor(dai.IMUSensor.GYROSCOPE_RAW, imu_frequency)
        imu_node.setBatchReportThreshold(int(imu_cfg.get('batch_report_threshold', 1)))
        imu_node.setMaxBatchReports(int(imu_cfg.get('max_batch_reports', 10)))
        # blocking=True with a large queue so no IMU packets are dropped
        imu_queue = imu_node.out.createOutputQueue(maxSize=500, blocking=True)

        pipeline.start()

        imu_sensor_name = pipeline.getDefaultDevice().getConnectedIMU()
        print(f"[IMU] Sensor detected: {imu_sensor_name}")

        platform = pipeline.getDefaultDevice().getPlatform()
        print(f"[Device] Platform: {platform}")
        if platform == dai.Platform.RVC4:
            control = initialize_mono_control(settings)
            controlQueueSend(input_queues, control)

        if settings['ir']: pipeline.getDefaultDevice().setIrLaserDotProjectorIntensity(settings['ir_value'])
        if settings['flood_light']: pipeline.getDefaultDevice().setIrFloodLightIntensity(settings['flood_light_intensity'])

        # Start IMU collector thread
        imu_thread = threading.Thread(
            target=_imu_collector_thread,
            args=(imu_queue, imu_buffer, imu_lock, imu_stop_event),
            daemon=True,
        )
        imu_thread.start()
        print(f"[IMU] Collector started at {imu_frequency} Hz")

        print("\n[Capture] Starting...")
        while pipeline.isRunning():
            current_time = time.time()
            if not save and check_autostart_condition(autostart, autostart_time, initial_time, current_time):
                output_folder, start_time = start_capture(
                    output_root, device, settings_path, capture_name, stereo_settings
                )
                _save_imu_sensor_to_metadata(output_folder, imu_sensor_name)
                save = True
                # Clear any pre-capture IMU data accumulated during warmup
                with imu_lock:
                    imu_buffer.clear()
                print("[Capture] Starting capture via autostart")
                print("\n" + "="*60)
                print("[STATUS] >>> CAPTURING... <<<")
                print("[CONTROLS] Press 'S' to STOP, 'Q' to QUIT")
                print("="*60 + "\n")

            if autostart >= 0 and not save:
                if autostart_time:
                    countdown_seconds = max(0, int(autostart_time.timestamp() - current_time))
                else:
                    countdown_seconds = max(0, int((initial_time + autostart) - current_time))
            else:
                countdown_seconds = None

            if settings["output_settings"]["sync"]:
                if not q['sync'].has():
                    continue
                msgGrp = q['sync'].get()
                for name, msg in msgGrp:
                    timestamp = int(msg.getTimestamp().total_seconds() * 1000)

                    if 'raw' in name:
                        dataRaw = msg.getData()
                        cvFrame = unpackRaw10(dataRaw, msg.getWidth(), msg.getHeight(), msg.getStride())
                    else:
                        cvFrame = msg.getCvFrame()

                    if save:
                        if name in ['left', 'right']:
                            if len(cvFrame.shape) == 3:
                                cvFrame = cv2.cvtColor(cvFrame, cv2.COLOR_BGR2GRAY)
                        do_png = save_png and name in png_streams
                        if save_npy or do_png:
                            save_queue.put(
                                (output_folder, name, timestamp, cvFrame.copy(), save_npy, do_png),
                                block=True
                            )
                        num_captures += 1

                    if not no_streams:
                        show_stream(name, cvFrame, timestamp, mxid, save, num_captures, capture_limit_str, countdown_seconds)
                if no_streams:
                    update_control_window(save, num_captures, capture_limit_str, countdown_seconds)
            else:
                for name in q.keys():
                    if not q[name].has():
                        continue
                    frame = q[name].get()
                    if 'raw' in name:
                        dataRaw = frame.getData()
                        cvFrame = unpackRaw10(dataRaw, frame.getWidth(), frame.getHeight(), frame.getStride())
                    else:
                        cvFrame = frame.getCvFrame()
                    timestamp = int(frame.getTimestamp().total_seconds() * 1000)
                    if save:
                        if name in ['left', 'right']:
                            if len(cvFrame.shape) == 3:
                                cvFrame = cv2.cvtColor(cvFrame, cv2.COLOR_BGR2GRAY)
                        do_png = save_png and name in png_streams
                        if save_npy or do_png:
                            save_queue.put(
                                (output_folder, name, timestamp, cvFrame.copy(), save_npy, do_png),
                                block=True
                            )
                        num_captures += 1

                    if not no_streams:
                        show_stream(name, cvFrame, timestamp, mxid, save, num_captures, capture_limit_str, countdown_seconds)
                if no_streams:
                    update_control_window(save, num_captures, capture_limit_str, countdown_seconds)

            key = cv2.waitKey(1)
            if key == ord('q'):
                if save and output_folder:
                    flush_imu_to_csv(output_folder, imu_buffer, imu_lock)
                pipeline.stop()
                break
            elif key == ord("s"):
                save = not save
                if save:
                    output_folder, start_time = start_capture(
                        output_root, device, settings_path, capture_name, stereo_settings
                    )
                    _save_imu_sensor_to_metadata(output_folder, imu_sensor_name)
                    # Clear any pre-capture IMU data accumulated during warmup
                    with imu_lock:
                        imu_buffer.clear()
                    print("[STATUS] CAPTURING...")
                else:
                    flush_imu_to_csv(output_folder, imu_buffer, imu_lock)
                    print("[STATUS] STOPPING CAPTURE")
                    stop_capture(start_time, num_captures, streams, pipeline)
                    break

            if save and check_stop_condition(wait_end, num_captures, final_num_captures, time.time()):
                print("[STATUS] STOP CONDITION MET - STOPPING CAPTURE")
                flush_imu_to_csv(output_folder, imu_buffer, imu_lock)
                stop_capture(start_time, num_captures, streams, pipeline)
                break

        imu_stop_event.set()
        imu_thread.join(timeout=2.0)

    save_queue.put(None)
    saver_thread.join(timeout=60)
    if saver_thread.is_alive():
        print("[Capture] Warning: saver thread did not finish in time")


if __name__ == "__main__":
    args = parse_arguments(root_path)
    main(args)
