#!/usr/bin/env python3
"""
Print metadata embedded in a sensor metadata JPEG (e.g. sensor_metadata.jpg).
Shows EXIF and, if present, maker notes / other segments.
"""
import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Print metadata embedded in a sensor metadata JPEG (sensor_metadata.jpg)."
    )
    parser.add_argument("jpeg_path", type=Path, help="Path to sensor_metadata.jpg (or any JPEG)")
    parser.add_argument("--raw", action="store_true", help="Show raw tag numbers and values")
    args = parser.parse_args()

    path = args.jpeg_path.resolve()
    if not path.is_file():
        print(f"Not a file: {path}", file=sys.stderr)
        return 1

    try:
        from PIL import Image
        from PIL.ExifTags import TAGS
    except ImportError:
        print("Install Pillow to read EXIF: pip install Pillow", file=sys.stderr)
        return 1

    img = Image.open(path)
    print(f"Image: {path.name}  size={img.size}  mode={img.mode}\n")

    exif = img.getexif()
    if exif is None:
        print("No EXIF data in this JPEG.")
        return 0

    if args.raw:
        for tag_id, value in exif.items():
            tag_name = TAGS.get(tag_id, tag_id)
            print(f"  {tag_name} ({tag_id}): {value}")
        return 0

    for tag_id, value in exif.items():
        tag_name = TAGS.get(tag_id, tag_id)
        if isinstance(value, bytes) and len(value) > 100:
            print(f"  {tag_name}: <binary, {len(value)} bytes>")
        else:
            print(f"  {tag_name}: {value}")

    ifd = getattr(exif, "get_ifd", None)
    if ifd is not None:
        try:
            exif_ifd = exif.get_ifd(0x8769)
            if exif_ifd:
                print("\n--- EXIF IFD ---")
                for k, v in exif_ifd.items():
                    name = TAGS.get(k, k)
                    if isinstance(v, bytes) and len(v) > 80:
                        print(f"  {name}: <binary, {len(v)} bytes>")
                    else:
                        print(f"  {name}: {v}")
        except Exception:
            pass

    print("\nFor vendor-specific / tuning metadata, try: exiftool", path.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
