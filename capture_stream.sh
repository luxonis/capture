#!/usr/bin/env bash
# Stream a ToF capture off the device directly onto this machine.
#
# Starts stream_receiver.py locally, then runs capture_stream_tof_dai.py on the
# device pointed back at this host. Frames arrive zstd-compressed over TCP and
# are written straight into ./output/<capture_folder>/ — no device storage, no
# scp drain step.
#
# Usage:
#   ./capture_stream.sh <device-ip> [capture_stream_tof_dai.py args...]
#   ./capture_stream.sh 10.11.102.225 --num-frames 300 --no-rgb
#
# Sustained rate is network-bound (~21 fps no-RGB on 2.5 GbE); the first
# ~150-180 frames (no RGB) are a contiguous 30 fps burst absorbed by device
# buffers. Requires the one-time device setup from the README (depthai +
# zstandard in /data/pydeps) and `pip install zstandard` locally.

set -euo pipefail

IP="${1:?usage: $0 <device-ip> [capture args...]}"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOTE="root@${IP}"
PORT="${PORT:-45678}"

HOSTIP=$(ip route get "$IP" | grep -oP 'src \K\S+')
[ -n "$HOSTIP" ] || { echo "Could not determine local IP toward $IP" >&2; exit 1; }

python3 -c "import zstandard" 2>/dev/null || {
    echo "Local python needs zstandard: pip install zstandard" >&2; exit 1; }

echo "[1/3] Uploading device script..."
ssh "$REMOTE" "mkdir -p /data/capture"
scp -q "$SCRIPT_DIR/capture_stream_tof_dai.py" "$REMOTE:/data/capture/"

OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/output}"
echo "[2/3] Starting local receiver on $HOSTIP:$PORT (output: $OUTPUT_DIR) ..."
mkdir -p "$OUTPUT_DIR"
python3 "$SCRIPT_DIR/stream_receiver.py" --port "$PORT" --output "$OUTPUT_DIR" &
RECV_PID=$!
trap 'kill $RECV_PID 2>/dev/null || true' EXIT
sleep 1

echo "[3/3] Running streaming capture on device..."
ssh "$REMOTE" "cd /data/capture && PYTHONPATH=/data/pydeps python3 capture_stream_tof_dai.py --host $HOSTIP --port $PORT $*"

wait "$RECV_PID"
trap - EXIT
echo "Done."
