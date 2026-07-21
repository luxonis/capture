#!/usr/bin/env bash
# Run consecutive RAM-buffered bursts on the device, each streamed straight to
# this machine over the network after capture (no device eMMC involved, no
# scp pull). Each burst is a guaranteed-contiguous 30 fps sequence in its own
# indexed folder under ./output.
#
# Usage:
#   ./capture_stream_burst_loop.sh <device-ip> <num-bursts> [capture args...]
#   ./capture_stream_burst_loop.sh 10.11.102.225 15 --no-rgb --num-frames 200
#
# Set CAPTURE_NAME for the folder base name (default: burst); don't pass
# --capture-name in the extra args.
# Set RAM_THRESHOLD_MB to override the between-burst RAM wait
# (default: auto = 75% of device MemTotal).

set -euo pipefail

usage() {
    sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}
[ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ] && usage 0
[ -z "${1:-}" ] && usage 1

IP="$1"
COUNT="${2:?usage: $0 <device-ip> <num-bursts> [capture args...]}"
case "$COUNT" in
    ''|*[!0-9]*) echo "error: <num-bursts> must be a number, got '$COUNT'" >&2; echo >&2; usage 1;;
esac
shift 2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOTE="root@${IP}"
NAME="${CAPTURE_NAME:-burst}"

if [ -z "${RAM_THRESHOLD_MB:-}" ]; then
    RAM_THRESHOLD_MB=$(ssh "$REMOTE" "awk '/MemTotal/{printf \"%d\", \$2/1024*0.75}' /proc/meminfo")
    echo "[Setup] RAM threshold: ${RAM_THRESHOLD_MB} MB (75% of device MemTotal)"
fi

for i in $(seq 1 "$COUNT"); do
    if [ "$i" -gt 1 ]; then
        ssh "$REMOTE" "sync; n=0; while :; do
            A=\$(free -m | awk 'NR==2{print \$7}')
            [ \"\$A\" -ge $RAM_THRESHOLD_MB ] && break
            n=\$((n+1)); [ \$n -ge 60 ] && { echo \"  WARNING: RAM only \${A} MB after 120s, continuing\"; break; }
            sleep 2
        done"
    fi
    echo "[Burst $i/$COUNT]"
    CAPTURE_ARGS=(--ram-buffer --capture-name "${NAME}-$(printf '%03d' "$i")" "$@")
    "$SCRIPT_DIR/capture_stream.sh" "$IP" "${CAPTURE_ARGS[@]}"
done
echo "All $COUNT bursts done."
