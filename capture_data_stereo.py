#!/usr/bin/env python3

import depthai as dai
import numpy as np
import time
import json
import cv2
import os
import argparse

from utils import *
from pipeline import initialize_pipeline

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
    return parser.parse_args()

def main(args):
    settings_path, ip, autostart, autostart_time, wait_end, capture_name = process_argument_logic(args)
    print(f"[Device] Connecting to device... IP: {ip}")

    if ip is not None: 
        device = dai.Device(ip)
    else: 
        device = dai.Device()
    mxid = device.getDeviceId()

    device_name = device.getDeviceName()
    print("[Device] Device connected! ")
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
    print(f"[Streams] Active streams: {streams}")
    print(f"[Streams] Number of streams: {len(streams)}")
    print(f"[Capture] Will capture max frames ({settings['num_captures']}) * number of streams ({len(streams)}) = {final_num_captures}")

    initial_time = time.time()
    if autostart_time:
        print(f"[Capture] Waiting till: {autostart_time}")
    elif autostart >= 0:
        print(f"[Capture] Capture will start automatically after {autostart} seconds")

    with dai.Pipeline(device) as pipeline:
        pipeline, q, input_queues = initialize_pipeline(pipeline, settings)
        pipeline.start()

        platform = pipeline.getDefaultDevice().getPlatform()
        print(f"[Device] Platform: {platform}")
        if platform == dai.Platform.RVC4:
            control = initialize_mono_control(settings)
            controlQueueSend(input_queues, control)

        if settings['ir']: pipeline.getDefaultDevice().setIrLaserDotProjectorIntensity(settings['ir_value'])
        if settings['flood_light']: pipeline.getDefaultDevice().setIrFloodLightIntensity(settings['flood_light_intensity'])

        print("\n[Capture] Starting...")
        while pipeline.isRunning():
            if not save and check_autostart_condition(autostart, autostart_time, initial_time, time.time()):
                output_folder, start_time = start_capture(
                    root_path, device, settings_path, capture_name
                )
                save = True
                print("[Capture] Starting capture via autostart")

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
                        np.save(f'{output_folder}/{name}_{timestamp}.npy', cvFrame)
                        num_captures += 1
                    
                    show_stream(name, cvFrame, timestamp, mxid)
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
                        np.save(f'{output_folder}/{name}_{timestamp}.npy', cvFrame)
                        num_captures += 1
                    
                    show_stream(name, cvFrame, timestamp, mxid)

            key = cv2.waitKey(1)
            if key == ord('q'):
                pipeline.stop()
                break
            elif key == ord("s"):
                save = not save
                if save:
                    output_folder, start_time = start_capture(
                        root_path, device, settings_path, capture_name
                    )
                else:
                    stop_capture(start_time, num_captures, streams, pipeline)
                    break

            if save and check_stop_condition(wait_end, num_captures, final_num_captures, time.time()):
                stop_capture(start_time, num_captures, streams, pipeline)
                break


if __name__ == "__main__":
    args = parse_arguments(root_path)
    main(args)
