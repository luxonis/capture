# Stereo Capture Tool

DepthAI capture script for stereo (and optionally RGB) streams from OAK devices using **holistic recording** (lossless FFV1/AVI). Calibration is saved automatically. Depth is computed at replay time, not during capture.

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

### Capture

```bash
python capture_data_stereo.py [OPTIONS]
```

Recording starts immediately when the pipeline starts. Press **Q** to stop.

**Options**

| Option | Description |
|--------|-------------|
| `--settings` | Path to settings JSON (default: `capture_settings.json`) |
| `--output` | Custom output root folder (default: `output` next to the script) |
| `--capture-name` | Name for the capture (included in folder name and metadata) |
| `--ip` | Device IP for network connection (omit for USB) |
| `--no-streams` | Do not show stream windows; use control window for Q (faster capture) |
| `--num-frames` | Stop after capturing this many frames (default: unlimited) |

Captures are saved under the output folder in subfolders named by device, optional capture name, and timestamp (e.g. `output/OAK-D_abc123_myrun_20250211120000/`). Each folder contains a holistic recording (`.tar` with AVI video, MCAP timestamps, and `camera_info.json` calibration) plus `metadata.json`.

## Settings

Edit `capture_settings.json` to configure the capture.

- **stereoResolution** / **rgbResolution**: `{"x": width, "y": height}`.
- **output_settings.rgb**: Set to `false` to skip RGB recording.
- **ir** / **ir_value**: IR laser dot projector (0–1).
- **flood_light** / **flood_light_intensity**: IR flood light.
- **FPS**: Target FPS.
- **monoSettings**: Camera image processing (denoise, sharpness, contrast).
- **exposureSettings**: Auto or manual exposure.

## Higher FPS (disable stream display)

To reduce overhead and achieve higher capture FPS, disable the live stream windows:

```bash
python capture_data_stereo.py --no-streams
```

## Controls

- **q**: Stop recording and quit

