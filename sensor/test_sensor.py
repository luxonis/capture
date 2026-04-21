#!/usr/bin/env python3
"""
Test sensor metadata capture without running the main capture pipeline.
Takes device IP and SSH password, fetches sensor_metadata.jpg via SSH + GStreamer + SCP.
"""
import argparse
import os
import sys

from . import sensor_metadata as sm


def main():
    parser = argparse.ArgumentParser(
        description="Test sensor metadata capture (SSH + GStreamer + SCP). No capture pipeline."
    )
    parser.add_argument("ip", help="Device IP or hostname")
    parser.add_argument("--password", "-p", required=True, help="SSH password for device")
    parser.add_argument("--output", "-o", default="/tmp/sensor_test", help="Output folder (default: /tmp/sensor_test)")
    parser.add_argument("--short-timeout", action="store_true", help="Use 15s menu timeout to quickly see device output on timeout")
    parser.add_argument("--camera", type=int, default=None, help="QMMF camera index (default 1). Use 0 if pipeline crashes with qmmfsrc assertion/segfault.")
    args = parser.parse_args()

    if args.short_timeout:
        sm.MENU_TIMEOUT = 15
        print("[SensorMetadata] Using 15s menu timeout for testing.", file=sys.stderr)

    gst_cmd = None
    if args.camera is not None:
        gst_cmd = sm.DEFAULT_GST_CMD.replace("camera=1", f"camera={args.camera}")

    sm.save_sensor_metadata(
        host=args.ip,
        output_folder=args.output,
        password=args.password,
        gst_cmd=gst_cmd,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
