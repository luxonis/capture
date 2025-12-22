# Stereo Capture Tool

DepthAI capture script.

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
python capture_data_stereo.py [--ip IP] [--settings /path/to/settings.json] [--capture-name NAME]
```

- `--settings`: Path to settings JSON file (default: `capture_settings.json`)
- `--capture-name`: Optional name for the capture (will be included in folder name and metadata)
- `--ip`: Device IP address (optional, uses USB if not specified)
- `--autostart`: Automatically start capturing after N seconds (-1 to disable)

## Settings

Edit `capture_settings.json` to configure capture parameters. To capture more frames, change:

```json
"num_captures": 20
```

to a higher number (or use `"inf"` for unlimited capture).

## Controls

- `s`: Start/stop capture
- `q`: Quit

