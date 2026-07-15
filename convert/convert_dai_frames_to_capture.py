#!/usr/bin/env python3
"""
Convert a .dai-only capture (per-frame ImgFrame.save() files) into the
standard flat npy capture format, optionally also saving PNGs.

capture_data_stereo.py always saves one .dai file per frame
(<stream>/<stream>_<timestamp>.dai). When it is run without --npy/--png,
that .dai file is the only thing on disk. This script loads those .dai
files back into dai.ImgFrame objects and reconstructs the standard flat
capture layout used by older captures:
    {stream}_{timestamp}.npy   (always)
    {stream}_{timestamp}.png   (only for left/right/rgb, with --png)

Accepts either layout as input:
    <capture_dir>/<stream>/<stream>_<timestamp>.dai   (per-stream subfolders)
    <capture_dir>/<stream>_<timestamp>.dai             (flat)

Usage:
    python3 convert/convert_dai_frames_to_capture.py output/OAK-4-PRO_.../
    python3 convert/convert_dai_frames_to_capture.py output/OAK-4-PRO_.../ --png
    python3 convert/convert_dai_frames_to_capture.py output/OAK-4-PRO_.../ -o output/converted --png
"""

import argparse
import os
import re
import shutil

import cv2
import numpy as np

import depthai as dai

DAI_NAME_RE = re.compile(r"^(?P<name>.+)_(?P<ts>\d+)\.dai$")

# Only these streams get PNGs written, matching capture_data_stereo.py's do_png logic
PNG_STREAMS = ("left", "right", "rgb")

SIDECAR_FILES = ("calib.json", "metadata.json", "info.txt", "eeprom_vd55h1.bin")


def unpack_raw10(raw_data, width, height, stride=None):
    """Unpack RAW10 data into a 16-bit grayscale array (mirrors utils.unpackRaw10)."""
    if stride is None:
        stride = width * 10 // 8
    expected_size = stride * height

    if len(raw_data) < expected_size:
        raise ValueError(f"Data too small: {len(raw_data)} bytes, expected {expected_size}")

    packed_data = np.frombuffer(raw_data, dtype=np.uint8)
    result = np.zeros((height, width), dtype=np.uint16)

    for row in range(height):
        row_start = row * stride
        row_data = packed_data[row_start:row_start + stride]
        num_groups = (width + 3) // 4
        row_bytes = num_groups * 5
        if len(row_data) < row_bytes:
            break

        row_packed = row_data[:row_bytes].reshape(-1, 5)
        row_unpacked = np.zeros((row_packed.shape[0], 4), dtype=np.uint16)

        row_unpacked[:, 0] = row_packed[:, 0].astype(np.uint16) << 2
        row_unpacked[:, 1] = row_packed[:, 1].astype(np.uint16) << 2
        row_unpacked[:, 2] = row_packed[:, 2].astype(np.uint16) << 2
        row_unpacked[:, 3] = row_packed[:, 3].astype(np.uint16) << 2

        row_unpacked[:, 0] |= (row_packed[:, 4] & 0b00000011)
        row_unpacked[:, 1] |= (row_packed[:, 4] & 0b00001100) >> 2
        row_unpacked[:, 2] |= (row_packed[:, 4] & 0b00110000) >> 4
        row_unpacked[:, 3] |= (row_packed[:, 4] & 0b11000000) >> 6

        result[row, :width] = row_unpacked.flatten()[:width]

    return (result * 64).astype(np.uint16)


def find_dai_files(capture_dir):
    """Yield (stream_name, timestamp, dai_path) for every .dai frame found.

    Handles both the per-stream-subfolder layout and a flat layout.
    """
    for entry in sorted(os.listdir(capture_dir)):
        entry_path = os.path.join(capture_dir, entry)
        if os.path.isdir(entry_path):
            stream_name = entry
            for fname in sorted(os.listdir(entry_path)):
                if not fname.endswith(".dai"):
                    continue
                m = DAI_NAME_RE.match(fname)
                ts = m.group("ts") if m else None
                yield stream_name, ts, os.path.join(entry_path, fname)
        elif entry.endswith(".dai"):
            m = DAI_NAME_RE.match(entry)
            if not m:
                print(f"[Convert] Skipping unrecognized file: {entry}")
                continue
            yield m.group("name"), m.group("ts"), entry_path


def load_cv_frame(dai_path, stream_name):
    """Load a .dai ImgFrame and reconstruct the same array capture_data_stereo.py saves."""
    frame = dai.ImgFrame()
    frame.load(dai_path)

    if "raw" in stream_name:
        data_raw = frame.getData()
        cv_frame = unpack_raw10(data_raw, frame.getWidth(), frame.getHeight(), frame.getStride())
    else:
        cv_frame = frame.getCvFrame()

    if stream_name in ("left", "right") and len(cv_frame.shape) == 3:
        cv_frame = cv2.cvtColor(cv_frame, cv2.COLOR_BGR2GRAY)

    return cv_frame


def convert(capture_dir, output_dir, save_png):
    os.makedirs(output_dir, exist_ok=True)

    for sidecar in SIDECAR_FILES:
        src = os.path.join(capture_dir, sidecar)
        dst = os.path.join(output_dir, sidecar)
        if os.path.isfile(src) and os.path.abspath(src) != os.path.abspath(dst):
            shutil.copy2(src, dst)
            print(f"[Convert] Copied {sidecar}")

    counts = {}
    for stream_name, ts, dai_path in find_dai_files(capture_dir):
        if ts is None:
            print(f"[Convert] Skipping {dai_path}: could not parse timestamp")
            continue

        cv_frame = load_cv_frame(dai_path, stream_name)

        np.save(os.path.join(output_dir, f"{stream_name}_{ts}.npy"), cv_frame)
        if save_png and stream_name in PNG_STREAMS:
            cv2.imwrite(os.path.join(output_dir, f"{stream_name}_{ts}.png"), cv_frame)

        counts[stream_name] = counts.get(stream_name, 0) + 1

    for stream_name, count in sorted(counts.items()):
        print(f"[Convert] {stream_name}: {count} frames")
    print(f"[Convert] Done. Output: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a .dai-only capture to the standard npy capture format."
    )
    parser.add_argument("capture_dir", help="Path to the capture folder containing .dai frames")
    parser.add_argument("--output", "-o", default=None,
                         help="Output folder (default: convert in place, next to the .dai files)")
    parser.add_argument("--png", action="store_true",
                         help="Also save left/right/rgb frames as PNG")
    return parser.parse_args()


def main():
    args = parse_args()
    capture_dir = os.path.abspath(args.capture_dir)
    output_dir = os.path.abspath(args.output) if args.output else capture_dir
    convert(capture_dir, output_dir, args.png)


if __name__ == "__main__":
    main()
