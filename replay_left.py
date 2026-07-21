#!/usr/bin/env python3
"""
Simple replay viewer for the left stream of a .dai capture.

Loads left/*.dai frames from a capture folder and plays them back at the
recorded frame rate (from the timestamps in the filenames).

Needs a depthai build with ImgFrame.load() (PR #1893), e.g.:
    PYTHONPATH=~/depthai-core/build/bindings/python \
        ~/miniconda3/envs/depthai-build/bin/python replay_left.py output/<capture>/

Controls: SPACE pause/resume, LEFT/RIGHT step one frame while paused, q/ESC quit.

Usage:
    python3 replay_left.py <capture_dir> [--stream left] [--fps N] [--loop]
"""

import argparse
import glob
import os
import re
import sys

import cv2

import depthai as dai

DAI_NAME_RE = re.compile(r"^(?P<name>.+)_(?P<ts>\d+)\.dai$")


def find_frames(capture_dir, stream):
    """Return [(timestamp_ms, path)] sorted by timestamp; accepts both layouts."""
    candidates = glob.glob(os.path.join(capture_dir, stream, "*.dai")) or \
        glob.glob(os.path.join(capture_dir, f"{stream}_*.dai"))
    frames = []
    for path in candidates:
        m = DAI_NAME_RE.match(os.path.basename(path))
        if m and m.group("name") == stream:
            frames.append((int(m.group("ts")), path))
    frames.sort()
    return frames


def main():
    parser = argparse.ArgumentParser(description="Replay one stream of a .dai capture")
    parser.add_argument("capture_dir", help="Capture folder (contains left/ etc.)")
    parser.add_argument("--stream", default="left", help="Stream to replay (default: left)")
    parser.add_argument("--fps", type=float, default=None,
                        help="Force playback FPS (default: real rate from timestamps)")
    parser.add_argument("--loop", action="store_true", help="Loop the playback")
    args = parser.parse_args()

    if not hasattr(dai.ImgFrame, "load"):
        sys.exit("This depthai build has no ImgFrame.load() — run with the "
                 "img_frame_save_load build (see README)")

    frames = find_frames(args.capture_dir, args.stream)
    if not frames:
        sys.exit(f"No {args.stream}/*.dai frames found in {args.capture_dir}")
    ts0, ts1 = frames[0][0], frames[-1][0]
    real_fps = (len(frames) - 1) / ((ts1 - ts0) / 1000) if ts1 > ts0 else 30.0
    fps = args.fps or real_fps
    print(f"{len(frames)} {args.stream} frames, recorded at {real_fps:.1f} fps, "
          f"playing at {fps:.1f} fps")

    window = f"{args.stream} replay - {os.path.basename(os.path.normpath(args.capture_dir))}"
    delay_ms = max(1, int(1000 / fps))
    i = 0
    paused = False
    msg = dai.ImgFrame()
    while True:
        ts, path = frames[i]
        msg.load(path)
        img = msg.getCvFrame()
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        label = f"{i + 1}/{len(frames)}  t={(ts - ts0) / 1000:.3f}s" + ("  [PAUSED]" if paused else "")
        cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow(window, img)

        key = cv2.waitKey(0 if paused else delay_ms) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord(" "):
            paused = not paused
            continue
        elif paused and key == 81:  # left arrow
            i = max(0, i - 1)
            continue
        elif paused and key == 83:  # right arrow
            i = min(len(frames) - 1, i + 1)
            continue
        if not paused:
            i += 1
            if i >= len(frames):
                if args.loop:
                    i = 0
                else:
                    break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
