import depthai as dai
import numpy as np
import cv2
import os
import json
import time
import datetime
import argparse
import screeninfo


def unpackRaw10(rawData, width, height, stride=None):
    """
    Unpacks RAW10 data from DepthAI pipeline into a 16-bit grayscale array.
    :param rawData: List of raw bytes from DepthAI (1D numpy array)
    :param width: Image width
    :param height: Image height
    :param stride: Row stride in bytes (if None, calculated as width*10/8)
    :return: Unpacked 16-bit grayscale image with dimensions width×height
    """
    if stride is None:
        stride = width * 10 // 8
    expectedSize = stride * height

    if len(rawData) < expectedSize:
        raise ValueError(f"Data too small: {len(rawData)} bytes, expected {expectedSize}")

    packedData = np.frombuffer(rawData, dtype=np.uint8)

    result = np.zeros((height, width), dtype=np.uint16)

    for row in range(height):
        rowStart = row * stride
        rowData = packedData[rowStart:rowStart + stride]
        numGroups = (width + 3) // 4
        rowBytes = numGroups * 5
        if len(rowData) < rowBytes:
            break

        rowPacked = rowData[:rowBytes].reshape(-1, 5)
        rowUnpacked = np.zeros((rowPacked.shape[0], 4), dtype=np.uint16)

        rowUnpacked[:, 0] = rowPacked[:, 0].astype(np.uint16) << 2
        rowUnpacked[:, 1] = rowPacked[:, 1].astype(np.uint16) << 2
        rowUnpacked[:, 2] = rowPacked[:, 2].astype(np.uint16) << 2
        rowUnpacked[:, 3] = rowPacked[:, 3].astype(np.uint16) << 2

        rowUnpacked[:, 0] |= (rowPacked[:, 4] & 0b00000011)
        rowUnpacked[:, 1] |= (rowPacked[:, 4] & 0b00001100) >> 2
        rowUnpacked[:, 2] |= (rowPacked[:, 4] & 0b00110000) >> 4
        rowUnpacked[:, 3] |= (rowPacked[:, 4] & 0b11000000) >> 6

        rowFlat = rowUnpacked.flatten()
        result[row, :width] = rowFlat[:width]

    result16bit = (result * 64).astype(np.uint16)
    return result16bit


def colorize_depth(frame, min_depth=20, max_depth=5000):
    depth_colorized = np.interp(frame, (min_depth, max_depth), (0, 255)).astype(np.uint8)
    return cv2.applyColorMap(depth_colorized, cv2.COLORMAP_JET)


def downscale_to_fit(frame, max_width, max_height):
    h, w = frame.shape[:2]
    scale = min(max_width / w, max_height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def show_stream(name, frame, timestamp, mxid, is_capturing=False, num_captures=0):
    """Display a single stream frame with timestamp and capture status."""
    def add_status_text(img, is_capturing, num_captures):
        h, w = img.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        
        if is_capturing:
            status_text = "CAPTURING..."
            status_color = (0, 255, 0)
            count_text = f"Frames: {num_captures}"
            cv2.putText(img, status_text, (10, 60), font, font_scale, status_color, thickness)
            cv2.putText(img, count_text, (10, 90), font, font_scale, (255, 255, 255), thickness)
        else:
            instruction_text = "Press 'S' to start capture"
            cv2.putText(img, instruction_text, (10, 60), font, font_scale, (0, 255, 255), thickness)
        return img
    
    if name in ["left", "right", "rgb", "left_raw", "right_raw", "rgb_raw"]:
        frame_timestamp = frame.copy()
        frame_timestamp = cv2.putText(frame_timestamp, f"{timestamp} ms", (10, 30), 
                                     cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        frame_timestamp = add_status_text(frame_timestamp, is_capturing, num_captures)

        screen = screeninfo.get_monitors()[0]
        screen_width, screen_height = screen.width, screen.height
        h, w = frame_timestamp.shape[:2]
        if h > screen_height or w > screen_width:
            frame_timestamp = downscale_to_fit(frame_timestamp, screen_width, screen_height)
        cv2.imshow(f"{mxid} {name}", frame_timestamp)
    elif name == "depth":
        depth_vis = colorize_depth(frame)
        depth_vis = cv2.putText(depth_vis, f"{timestamp} ms", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        depth_vis = add_status_text(depth_vis, is_capturing, num_captures)
        cv2.imshow(f"{mxid} {name}", depth_vis)
    elif name == "disparity":
        depth_vis = colorize_depth(frame, min_depth=0, max_depth=frame.max())
        depth_vis = cv2.putText(depth_vis, f"{timestamp} ms", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        depth_vis = add_status_text(depth_vis, is_capturing, num_captures)
        cv2.imshow(f"{mxid} {name}", depth_vis)


def create_and_save_metadata(device, settings_path, output_dir, capture_name, date, 
                            capture_type=None, author=None, notes=None):
    model_name = device.getDeviceName()
    mxId = device.getMxId()
    metadata = {
        "model_name": model_name,
        "mxId": mxId,
        "capture_type": capture_type,
        "capture_name": capture_name,
        "date": date,
        "notes": notes,
        "author": author,
        'settings_name': settings_path,
        "settings": json.load(open(settings_path)),
        "dai_version": dai.__version__,
    }

    os.makedirs(output_dir, exist_ok=True)

    filename = f"metadata.json"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, 'w') as json_file:
        json.dump(metadata, json_file, indent=4)

    print(f"[Capture] Metadata saved to {filepath}")


def initialize_capture(root_path, device, settings_path, capture_name=None, projector=None):
    date = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    device_name = device.getDeviceName()
    device_id = device.getMxId()
    
    if capture_name:
        base_name = f"{device_name}_{device_id}_{capture_name}_{date}"
    else:
        base_name = f"{device_name}_{device_id}_{date}"
    
    if projector is None:
        out_dir = f"{root_path}/{base_name}"
    else:
        out_dir = f"{root_path}/{base_name}_{projector}"

    if not os.path.exists(root_path):
        os.makedirs(root_path)

    if not os.path.exists(os.path.join(root_path, out_dir)):
        os.makedirs(out_dir)
        print(f"[Capture] Folder '{out_dir}' created.")
    else:
        print(f"[Capture] Folder '{out_dir}' already exists.")

    calib = device.readCalibration()
    calib.eepromToJsonFile(f'{out_dir}/calib.json')
    create_and_save_metadata(device, settings_path, out_dir, capture_name, date)

    return out_dir


def finalise_capture(start_time, end_time, num_captures, streams):
    print(f"[Capture] Capture took {end_time - start_time:.2f} seconds.")
    print(f"[Capture] Capture has {num_captures} frames combined from all streams")
    print(f"[Capture] Capture was {round((num_captures/len(streams)) / (end_time - start_time), 2)} FPS")


def count_output_streams(output_streams):
    stream_names = []
    for item in output_streams.keys():
        if item in ["tof", "sync", "rgb_png"]: 
            continue
        if output_streams[item]:
            stream_names.append(item)
    return stream_names


def controlQueueSend(input_queues, ctrl):
    for queue in input_queues.values():
        queue.send(ctrl)


def initialize_mono_control(settings):
    ctrl = dai.CameraControl()

    mono_settings = settings["monoSettings"]
    ctrl.setLumaDenoise(mono_settings["luma_denoise"])
    ctrl.setChromaDenoise(mono_settings["chroma_denoise"])
    ctrl.setSharpness(mono_settings["sharpness"])
    ctrl.setContrast(mono_settings["contrast"])

    exposure_settings = settings["exposureSettings"]
    if not exposure_settings["autoexposure"]:
        ctrl.setManualExposure(exposure_settings["expTime"], exposure_settings["sensIso"])

    return ctrl


def check_autostart_condition(autostart, autostart_time, initial_time, current_time):
    """
    Check if autostart condition is met.

    :param autostart: Number of seconds to wait before autostart (-1 to disable)
    :param autostart_time: Specific datetime to start at (0 if not set)
    :param initial_time: Time when the script started
    :param current_time: Current time
    :return: True if autostart condition is met, False otherwise
    """
    if autostart < 0:
        return False

    if autostart_time:
        return current_time >= autostart_time.timestamp()
    else:
        return current_time >= (initial_time + autostart)


def start_capture(root_path, device, settings_path, capture_name=None):
    """
    Start a new capture session.

    :param root_path: Root path for output
    :param device: DepthAI device
    :param settings_path: Path to settings file
    :param capture_name: Optional name for the capture (will be included in folder name and metadata)
    :return: Tuple of (output_folder, start_time)
    """
    output_folder = initialize_capture(root_path, device, settings_path, capture_name)
    start_time = time.time()
    print("[Capture] Starting capture")
    return output_folder, start_time


def check_stop_condition(wait_end, num_captures, final_num_captures, current_time):
    """
    Check if stop condition is met.

    :param wait_end: Datetime to stop at (0 if not set)
    :param num_captures: Current number of captures
    :param final_num_captures: Target number of captures
    :param current_time: Current time
    :return: True if stop condition is met, False otherwise
    """
    if wait_end:
        return current_time >= wait_end.timestamp()
    else:
        return num_captures >= final_num_captures


def stop_capture(start_time, num_captures, streams, pipeline):
    """
    Stop the current capture session.

    :param mxid: Device ID
    :param start_time: Capture start time
    :param num_captures: Number of captures taken
    :param streams: List of active streams
    :param pipeline: DepthAI pipeline
    """
    end_time = time.time()
    finalise_capture(start_time, end_time, num_captures, streams)
    pipeline.getDefaultDevice().setIrLaserDotProjectorIntensity(0)
    pipeline.stop()
    print()


def process_argument_logic(args):
    settings_path = args.settings
    ip = args.ip

    if not os.path.exists(settings_path):
        raise FileNotFoundError(f"Settings file '{settings_path}' does not exist.")

    today = datetime.date.today()

    if args.autostart_time:
        wait = datetime.datetime.combine(today, datetime.time.fromisoformat(args.autostart_time))
    else:
        wait = 0

    if args.autostart_end:
        wait_end = datetime.datetime.combine(today, datetime.time.fromisoformat(args.autostart_end))
    else:
        wait_end = 0

    if args.autostart_time: 
        args.autostart = 0

    capture_name = getattr(args, 'capture_name', None)
    if capture_name:
        if '_' in capture_name:
            capture_name = capture_name.replace('_', '-')
            print(f"[Capture] Warning: Underscores in capture name replaced with hyphens: {capture_name}")

    return settings_path, ip, args.autostart, wait, wait_end, capture_name


