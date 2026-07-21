# Stereo Capture Tool

DepthAI capture script for stereo (and optionally RGB/depth) streams from OAK devices. Saves each frame as a `.dai` file (native `ImgFrame` serialization) with timestamps and optional calibration/metadata.

## Requirements

- DepthAI-compatible device (OAK camera) connected via USB or network
- Python 3.9+

## Setup

### Create Virtual Environment

**Option 1: Using Conda**

```bash
conda create -n capture python=3.11
conda activate capture
```

**Option 2: Using Python venv**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install Requirements

```bash
pip install -r requirements.txt
```

## Usage

```bash
python capture_data_stereo.py [OPTIONS]
```

**Options**

| Option | Description |
|--------|-------------|
| `--output` | Custom output root folder (default: `output` next to the script) |
| `--capture-name` | Name for the capture (included in folder name and metadata) |
| `--ip` | Device IP for network connection (omit for USB) |
| `--autostart` | Start capturing after N seconds (`-1` = disabled, default) |
| `--autostart_time` | Start at a fixed datetime (e.g. from cron) |
| `--autostart_end` | Stop at a fixed datetime |
| `--no-streams` | Do not show stream windows; use control window for S/Q (faster capture) |

Captures are saved under the output folder in subfolders named by device, optional capture name, and timestamp (e.g. `output/OAK-D_abc123_myrun_20250211120000/`). Each stream gets its own subfolder with one `{stream}_{timestamp_ms}.dai` file per frame. Calibration and metadata are written in the capture folder.

To get plain `.npy`/`.png` files back out of a `.dai` capture, run `convert/convert_dai_frames_to_capture.py <capture_dir> [--png]`.

To visually inspect a capture, `replay_left.py` plays back one stream at the recorded frame rate (needs a depthai build with `ImgFrame.load()`):

```bash
python3 replay_left.py output/<capture_dir>/ [--stream left|right|rgb] [--fps N] [--loop]
# SPACE = pause, arrows = step while paused, q = quit
```

## Settings

Capture settings are embedded directly in `capture_data_stereo.py` as module-level globals (`IR`, `IR_VALUE`, `FLOOD_LIGHT`, `FLOOD_LIGHT_INTENSITY`, `STEREO_RESOLUTION`, `RGB_RESOLUTION`, `FPS`, `NUM_CAPTURES`); edit the script directly to change them. They're recorded in each capture's `metadata.json`.

- **NUM_CAPTURES**: Max frames per stream (`20`, or `float('inf')` for unlimited).
- **STEREO_RESOLUTION** / **RGB_RESOLUTION**: `{"x": width, "y": height}`.
- **IR** / **IR_VALUE**: IR laser dot projector (0–1).
- **FLOOD_LIGHT** / **FLOOD_LIGHT_INTENSITY**: IR flood light.
- **FPS**: Target FPS.

Which streams are captured (`OUTPUT_SETTINGS` in `pipeline.py`), mono camera tuning (`MONO_SETTINGS`/`EXPOSURE_SETTINGS` in `utils.py`), and stereo depth tuning (`EXTENDED_DISPARITY` in `stereo.py`) are hardcoded pipeline configuration rather than user-facing settings — edit those files directly if you need different streams or stereo tuning. `capture_settings.json` is no longer read by the script.

## Higher FPS (disable stream display)

To reduce overhead and achieve higher capture FPS, disable the live stream windows and use only the small control window:

```bash
python capture_data_stereo.py --no-streams
```


## Controls

- **s**: Start or stop capture
- **q**: Quit

---

## ToF Raw Capture

`capture_data_tof_raw.py` captures raw ToF superframes, depth, amplitude, and left/right/RGB streams from RVC4 devices with a ToF sensor.

> **Note:** This script requires a compatible version of the `depthai` and a matching RVC4 firmware package (`.tar.xz`).

### Usage

```bash
python capture_data_tof_raw.py --ip <DEVICE_IP> --fwp <PATH_TO_FWP>
```

**arguments:**

| Option | Description |
|--------|-------------|
| `--ip` | Device IP address |
| `--fwp` | Path to the RVC4 firmware package (e.g. `depthai-device-rvc4-fwp.tar.xz`) |
| `--num-frames` | Number of frames to save (default: 16) |
| `--capture-name` | Name for the capture folder |
| `--preset` | ToF preset: `low`, `mid`, `high` (default: `high`) |
| `--show-streams` | Show live preview; press `S` to start capture, `Q` to quit |
| `--skip-warmup` | Skip warmup frames |
| `--warmup-frames` | Number of warmup frames (default: 30) |

---

## On-device ToF DAI capture

When streaming to the host is too slow / the data too large, `capture_data_tof_dai.py` can run **directly on the RVC4 device**, saving `.dai` frames to the device's `/data` partition, and the capture is drained to the host afterwards over `scp` (the device has no `rsync`).

The wrapper does upload → capture → drain in one go:

```bash
./capture_on_device.sh <device-ip> --num-frames 100 --capture-name my-scene
# add DELETE_ON_DEVICE=1 to remove the remote copy after a successful pull
```

Captures land on the device under `/data/captures/` and are pulled into `./output/`.

### One-time device setup

`ImgFrame.save()` comes from depthai-core PR #1893, which is not in any release — use a CI wheel built from a commit of that PR (`feature/img_frame_save_load`). On the device (root `/` is nearly full, so install into `/data`; the device's own NumPy 1.22 works, the NumPy 2.x that pip pulls in triggers ABI warnings — hence `rm` below):

```bash
ssh root@<device-ip>
python3 -m ensurepip
mkdir -p /data/tmp
TMPDIR=/data/tmp python3 -m pip install --no-cache-dir --target /data/pydeps \
    --extra-index-url https://artifacts.luxonis.com/artifactory/luxonis-python-snapshot-local/ \
    "depthai==3.7.1.dev0+75ecac12a70954eb066cb2add25b59fae4ad42a6"
rm -rf /data/pydeps/numpy /data/pydeps/numpy-*.dist-info /data/pydeps/numpy.libs
```

To run the capture manually on the device instead of via the wrapper:

```bash
ssh root@<device-ip>
cd /data/capture
PYTHONPATH=/data/pydeps python3 capture_data_tof_dai.py --num-frames 100 --output /data/captures
```

### Capture speed: `--ram-buffer` for full 30 fps

Writing to the device eMMC sustains only ~100 MB/s while the four streams produce ~1.2 GB/s at 30 fps, so plain capture runs at ~3.6 fps. With `--ram-buffer` frames are held in RAM during capture and written out afterwards, giving a **gap-free 30 fps sequence** (verified: all inter-frame deltas 33–34 ms). `--warmup-frames` (default 10) discards startup frames so the saved window starts clean.

RAM (3.4 GB total) caps the burst length; the script measures the first frame-set and clamps `--num-frames` to what fits instead of getting OOM-killed:

| Streams | per frame-set | max burst @ 4 GB | max burst @ 8 GB (dev unit, see below) |
|---|---|---|---|
| all four (12MP RGB) | ~40 MB | ~32 frames (~1 s) | ~80 frames (~2.7 s) |
| `--no-rgb` | ~22.5 MB | ~50–70 frames (~2 s) | ~140 frames (~4.7 s) |
| `--rgb-resolution 1920x1080` | ~26 MB | ~45–60 frames | ~120 frames |

**8 GB dev units:** some dev units have 8 GB physically but boot with half of it parked offline (`mem=4G` + `memhp_default_state=offline` in the dtbo bootargs, emulating the 4 GB production SKU). Run `/data/online_ram.sh` on the device (installed there; onlines the offline memory blocks) to unlock the full 8 GB — **required again after every reboot**, and remember production units really have 4 GB.

For longer captures, use **streaming mode** (below) instead.

To take several bursts back-to-back (e.g. multiple static scenes), there are two loop wrappers; both wait between bursts until device RAM has recovered (`RAM_THRESHOLD_MB`, default: 75% of device MemTotal) and number the folders `<name>-001`, `-002`, ...

**Preferred — `capture_stream_burst_loop.sh`:** RAM burst, then the frames are zstd-streamed straight to the host over 2.5 GbE (~6 s per max burst instead of ~64 s of eMMC flush, and no scp pull step). Verified: 139-frame contiguous 30 fps bursts, ~18 s full cycle:

```bash
CAPTURE_NAME=my-scene ./capture_stream_burst_loop.sh <device-ip> 15 --no-rgb --num-frames 200
# num-frames above the RAM limit = auto-clamp to the max burst (~139 no-RGB @ 8 GB)
```

**Fallback — `capture_burst_loop.sh`:** same bursts but flushed to device eMMC and pulled with scp at the end. Slower cycle (~80 s per max burst), but works without a reachable receiver port on the host:

```bash
CAPTURE_NAME=my-scene ./capture_burst_loop.sh <device-ip> 5 --no-rgb --num-frames 70
```

### Streaming mode: unlimited length, frames land on the host

`capture_stream.sh` runs `capture_stream_tof_dai.py` on the device, which zstd-compresses every serialized frame and pushes it over TCP (2.5 GbE) to `stream_receiver.py` on the host — no device storage, no scp drain step. Output is the identical `.dai` capture layout under `./output/`.

```bash
./capture_stream.sh <device-ip> --num-frames 300 --no-rgb
```

Measured behaviour with `--no-rgb` (2.5 GbE, ~290 MB/s TCP, zstd-1 ≈ ×1.66):
- first **~140–160 frames are a contiguous 30 fps burst** (device buffers absorb the overflow),
- after that it settles at **~21 fps sustained indefinitely**, dropped frames appearing as 66 ms gaps,
- 300 frame-sets (6.8 GB raw) arrive on the host in ~20 s total.

Full-res RGB works too but the sustained rate is scene-dependent (RGB compressibility varies a lot); expect it to be lower. Requires `pip install zstandard` on the host; the device side is covered by the one-time setup below (zstandard is installed in `/data/pydeps`).

Notes:
- On-device the script drives the device's RGB status LED so you can see the state without a terminal: **red = capturing, green = flushing to disk, blinking blue = idle/done** (the OS default). Disable with `--no-led`.
- On-device the script needs no `--ip` (it connects to the local device through `depthai_gate`) and copies the eeprom file locally instead of via scp. `--show-streams` is unavailable there (no OpenCV).
- The on-disk `.dai` format written by the `75ecac12` wheel is byte-identical (for non-empty frames) to the current `feature/img_frame_save_load-rebased` build in `~/depthai-core`, so files captured on-device load fine with the local build and `convert/convert_dai_frames_to_capture.py`.
- `/data` has ~60 GB free; a 8-frame 4-stream capture is ~315 MB, so roughly 1500 frames fit per free 60 GB.


