#!/usr/bin/env bash
# Run consecutive --ram-buffer bursts on the device, each into its own capture
# folder, waiting between bursts until device RAM has recovered. All new
# capture folders are pulled to ./output at the end.
#
# Usage:
#   ./capture_burst_loop.sh <device-ip> <num-bursts> [capture_data_tof_dai.py args...]
#   ./capture_burst_loop.sh 10.11.102.225 5 --no-rgb --num-frames 70
#
# Folder names get a burst index suffix: <device>_<id>_<name>-001_<date>, ...
# Set CAPTURE_NAME to change the base name (default: burst); don't pass
# --capture-name in the extra args.
# Set RAM_THRESHOLD_MB (default 2400) to tune how much MemAvailable is
# required before the next burst starts.
# Set DELETE_ON_DEVICE=1 to remove each folder from the device after pulling.

set -euo pipefail

IP="${1:?usage: $0 <device-ip> <num-bursts> [capture args...]}"
COUNT="${2:?usage: $0 <device-ip> <num-bursts> [capture args...]}"
shift 2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOTE="root@${IP}"
REMOTE_OUT="/data/captures"
NAME="${CAPTURE_NAME:-burst}"
RAM_THRESHOLD_MB="${RAM_THRESHOLD_MB:-2400}"

echo "[Setup] Uploading capture script..."
ssh "$REMOTE" "mkdir -p /data/capture $REMOTE_OUT"
scp -q "$SCRIPT_DIR/capture_data_tof_dai.py" "$REMOTE:/data/capture/"

BEFORE=$(ssh "$REMOTE" "ls -1 $REMOTE_OUT 2>/dev/null" || true)

for i in $(seq 1 "$COUNT"); do
    if [ "$i" -gt 1 ]; then
        echo "[Wait] Waiting for device RAM >= ${RAM_THRESHOLD_MB} MB available..."
        ssh "$REMOTE" "sync; n=0; while :; do
            A=\$(free -m | awk 'NR==2{print \$7}')
            [ \"\$A\" -ge $RAM_THRESHOLD_MB ] && { echo \"  RAM ok: \${A} MB\"; break; }
            n=\$((n+1))
            [ \$n -ge 60 ] && { echo \"  WARNING: still \${A} MB after 120s, continuing anyway\"; break; }
            [ \$((n % 5)) -eq 1 ] && echo \"  waiting: \${A} MB available\"
            sleep 2
        done"
    fi
    echo "[Burst $i/$COUNT] Capturing..."
    ssh "$REMOTE" "cd /data/capture && PYTHONPATH=/data/pydeps python3 capture_data_tof_dai.py \
        --ram-buffer --output $REMOTE_OUT --capture-name ${NAME}-$(printf '%03d' "$i") $*" \
        | grep -E "RAM|Done!|Flush|Folder" || true
done

AFTER=$(ssh "$REMOTE" "ls -1 $REMOTE_OUT")
NEW=$(comm -13 <(echo "$BEFORE" | sort) <(echo "$AFTER" | sort))
[ -n "$NEW" ] || { echo "No new capture folders found on device." >&2; exit 1; }

echo "[Pull] Copying $(echo "$NEW" | wc -l) capture(s) to ./output ..."
mkdir -p "$SCRIPT_DIR/output"
for d in $NEW; do
    scp -q -r "$REMOTE:$REMOTE_OUT/$d" "$SCRIPT_DIR/output/"
    echo "  pulled output/$d ($(du -sh "$SCRIPT_DIR/output/$d" | cut -f1))"
    if [ "${DELETE_ON_DEVICE:-0}" = "1" ]; then
        ssh "$REMOTE" "rm -rf $REMOTE_OUT/$d"
    fi
done
echo "Done."
