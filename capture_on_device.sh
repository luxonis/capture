#!/usr/bin/env bash
# Run capture_data_tof_dai.py ON the device (data saved to device /data),
# then drain the capture folder back to this machine over scp.
#
# One-time device setup (already done for 10.11.102.225):
#   1. Download an aarch64 depthai wheel that has ImgFrame.save() (PR #1893),
#      e.g. from https://artifacts.luxonis.com/artifactory/luxonis-python-snapshot-local/depthai/
#      (depthai-3.7.1.dev0+75ecac12a709...aarch64.whl)
#   2. scp <wheel> root@<ip>:/data/ && ssh root@<ip> 'mkdir -p /data/pydeps && cd /data/pydeps && unzip -oq /data/<wheel>'
#
# Usage:
#   ./capture_on_device.sh <device-ip> [extra capture_data_tof_dai.py args...]
#   ./capture_on_device.sh 10.11.102.225 --num-frames 100 --capture-name my-scene
#
# Frames stay on the device under /data/captures and are copied to ./output
# afterwards. Pass KEEP_ON_DEVICE=1 to skip deleting the remote copy (default: kept).
# Pass DELETE_ON_DEVICE=1 to remove the remote copy after a successful pull.

set -euo pipefail

IP="${1:?usage: $0 <device-ip> [capture args...]}"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOTE="root@${IP}"
REMOTE_OUT="/data/captures"

echo "[1/3] Uploading capture script..."
ssh "$REMOTE" "mkdir -p /data/capture $REMOTE_OUT"
scp -q "$SCRIPT_DIR/capture_data_tof_dai.py" "$REMOTE:/data/capture/"

echo "[2/3] Running capture on device..."
# Mark pre-existing captures so we only pull the new one(s).
BEFORE=$(ssh "$REMOTE" "ls -1 $REMOTE_OUT 2>/dev/null" || true)
ssh "$REMOTE" "cd /data/capture && PYTHONPATH=/data/pydeps python3 capture_data_tof_dai.py --output $REMOTE_OUT $*"
AFTER=$(ssh "$REMOTE" "ls -1 $REMOTE_OUT")
NEW=$(comm -13 <(echo "$BEFORE" | sort) <(echo "$AFTER" | sort))

if [ -z "$NEW" ]; then
    echo "No new capture folder found on device." >&2
    exit 1
fi

echo "[3/3] Pulling capture(s) to ./output ..."
mkdir -p "$SCRIPT_DIR/output"
for d in $NEW; do
    scp -q -r "$REMOTE:$REMOTE_OUT/$d" "$SCRIPT_DIR/output/"
    echo "  pulled output/$d ($(du -sh "$SCRIPT_DIR/output/$d" | cut -f1))"
    if [ "${DELETE_ON_DEVICE:-0}" = "1" ]; then
        ssh "$REMOTE" "rm -rf $REMOTE_OUT/$d"
        echo "  deleted $REMOTE_OUT/$d on device"
    fi
done

echo "Done."
