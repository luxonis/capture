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

Captures are saved under the output folder in subfolders named by device, optional capture name, and timestamp (e.g. `output/OAK-D_abc123_myrun_20250211120000/`). Each stream gets its own subfolder with one `{stream}_{timestamp_ms}.dai` file per frame. Calibration and metadata are written in the capture folder, along with `preview_left.png` — the first left frame of the capture, saved as a quick visual check.

## Settings

Capture settings are embedded directly in `capture_data_stereo.py` as module-level globals (`IR`, `IR_VALUE`, `FLOOD_LIGHT`, `FLOOD_LIGHT_INTENSITY`, `STEREO_RESOLUTION`, `RGB_RESOLUTION`, `FPS`, `NUM_CAPTURES`); edit the script directly to change them. They're recorded in each capture's `metadata.json`.

- **NUM_CAPTURES**: Max frames per stream (`20`, or `float('inf')` for unlimited).
- **STEREO_RESOLUTION** / **RGB_RESOLUTION**: `{"x": width, "y": height}`.
- **IR** / **IR_VALUE**: IR laser dot projector (0–1).
- **FLOOD_LIGHT** / **FLOOD_LIGHT_INTENSITY**: IR flood light.
- **FPS**: Target FPS.

Which streams are captured (`OUTPUT_SETTINGS` in `pipeline.py`), mono camera tuning (`MONO_SETTINGS`/`EXPOSURE_SETTINGS` in `utils.py`), and stereo depth tuning (`EXTENDED_DISPARITY` in `stereo.py`) are hardcoded pipeline configuration rather than user-facing settings — edit those files directly if you need different streams or stereo tuning. The old `capture_settings.json` file has been removed.

## Higher FPS (disable stream display)

To reduce overhead and achieve higher capture FPS, disable the live stream windows and use only the small control window:

```bash
python capture_data_stereo.py --no-streams
```


## Controls

- **s**: Start or stop capture
- **q**: Quit
