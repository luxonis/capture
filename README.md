# Stereo Capture Tool

DepthAI capture script for stereo (and optionally RGB/depth) streams from OAK devices. Saves frames as `.npy` files with timestamps and optional calibration/metadata.

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
| `--settings` | Path to settings JSON (default: `capture_settings.json`) |
| `--output` | Custom output root folder (default: `output` next to the script) |
| `--capture-name` | Name for the capture (included in folder name and metadata) |
| `--ip` | Device IP for network connection (omit for USB) |
| `--autostart` | Start capturing after N seconds (`-1` = disabled, default) |
| `--autostart_time` | Start at a fixed datetime (e.g. from cron) |
| `--autostart_end` | Stop at a fixed datetime |
| `--no-streams` | Do not show stream windows; use control window for S/Q (faster capture) |
| `--png` | Save left, right, rgb as PNG (disables npy unless `--npy` is also set) |
| `--npy` | Save frames as numpy (default when no format option is set). Use with `--png` to save both. |

Captures are saved under the output folder in subfolders named by device, optional capture name, and timestamp (e.g. `output/OAK-D_abc123_myrun_20250211120000/`). By default each stream is saved as `{stream}_{timestamp_ms}.npy`. With `--png`, left/right/rgb are saved as `.png`; with both `--png` and `--npy`, those streams are saved in both formats. Calibration and metadata are written in the same folder.

## Settings

Edit `capture_settings.json` to configure the pipeline and capture.

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

