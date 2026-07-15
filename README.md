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
| `--settings` | Path to a settings JSON file to override the embedded defaults (see Settings below) |
| `--output` | Custom output root folder (default: `output` next to the script) |
| `--capture-name` | Name for the capture (included in folder name and metadata) |
| `--ip` | Device IP for network connection (omit for USB) |
| `--autostart` | Start capturing after N seconds (`-1` = disabled, default) |
| `--autostart_time` | Start at a fixed datetime (e.g. from cron) |
| `--autostart_end` | Stop at a fixed datetime |
| `--no-streams` | Do not show stream windows; use control window for S/Q (faster capture) |

Captures are saved under the output folder in subfolders named by device, optional capture name, and timestamp (e.g. `output/OAK-D_abc123_myrun_20250211120000/`). Each stream gets its own subfolder with one `{stream}_{timestamp_ms}.dai` file per frame. Calibration and metadata are written in the capture folder.

To get plain `.npy`/`.png` files back out of a `.dai` capture, run `convert/convert_dai_frames_to_capture.py <capture_dir> [--png]`.

## Settings

The pipeline/capture settings are embedded directly in `capture_data_stereo.py` as `DEFAULT_SETTINGS`, so the script runs standalone without any settings file. Pass `--settings path/to/file.json` to override them with a custom JSON file (same shape as `DEFAULT_SETTINGS`); `capture_settings.json` in this repo is kept as an example of that shape.

- **num_captures**: Max frames per stream (`20`, or `"inf"` for unlimited).
- **output_settings**: Enable/disable streams: `left`, `right`, `left_raw`, `right_raw`, `rgb`, `depth`, `disparity`; select `sync` for synchronized capture.
- **stereoResolution** / **rgbResolution**: `{"x": width, "y": height}`.
- **ir** / **ir_value**: IR laser dot projector (0–1).
- **flood_light** / **flood_light_intensity**: IR flood light.
- **FPS**: Target FPS.

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


