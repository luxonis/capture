import os
import shlex
import sys
import time
from pathlib import Path

import pexpect

DEBUG = os.environ.get("SENSOR_METADATA_DEBUG", "").strip().lower() in ("1", "true", "yes")

DEFAULT_GST_CMD = (
    'gst-pipeline-app -e qtiqmmfsrc camera=1 name=camsrc '
    '! video/x-raw,format=NV12,width=1280,height=800,framerate=30/1 '
    '! jpegenc '
    '! multipartmux boundary="frame" '
    '! tcpserversink host=0.0.0.0 port=5022 '
    'camsrc.image_1 '
    '! "image/jpeg,width=1280,height=800,framerate=30/1" '
    '! multifilesink location=/data/capture/frame%d.jpg sync=true async=false '
    'camsrc.image_2 '
    '! "video/x-bayer,format=mono,bpp=(string)10,width=1280,height=800" '
    '! multifilesink location=/data/capture/frame%d.raw sync=true async=false'
)

SSH_TIMEOUT = 180
MENU_TIMEOUT = 180
DELAY_SHORT = 0.2
DELAY_LONG_CAPTURE = 2.0


def _spawn_ssh(host: str, user: str, port: int) -> pexpect.spawn:
    print(f"[SensorMetadata] SSH connecting to {user}@{host}:{port} ...")
    cmd = (
        f"ssh -tt -p {shlex.quote(str(port))} "
        f"-o StrictHostKeyChecking=accept-new "
        f"-o UserKnownHostsFile=/dev/null "
        f"{shlex.quote(user)}@{shlex.quote(host)}"
    )
    child = pexpect.spawn(cmd, encoding="utf-8", timeout=SSH_TIMEOUT)
    child.maxread = 200000
    if DEBUG:
        child.logfile = sys.stderr
        print("[SensorMetadata] Debug: logging device output to stderr", file=sys.stderr)
    return child


def _ssh_login(child: pexpect.spawn, password: str) -> None:
    print("[SensorMetadata] Waiting for SSH login prompt ...")
    prompt_re = r"[$#] "
    while True:
        try:
            idx = child.expect(
                [
                    r"Are you sure you want to continue connecting.*\?",
                    r"(?i)password:",
                    r"Permission denied",
                    prompt_re,
                ],
                timeout=SSH_TIMEOUT,
            )
        except pexpect.EOF:
            msg = (
                "SSH connection closed (EOF). Is SSH enabled on the device? "
                "Check that the device is reachable and allows SSH on port 22."
            )
            raise RuntimeError(msg) from None
        except pexpect.TIMEOUT:
            raise RuntimeError("SSH login timed out. Is the device reachable?") from None
        if idx == 0:
            child.sendline("yes")
        elif idx == 1:
            if not password:
                raise RuntimeError("Password prompt received but password is empty.")
            child.sendline(password)
        elif idx == 2:
            raise RuntimeError("Permission denied.")
        else:
            print("[SensorMetadata] SSH logged in.")
            return


def _send_cmd(child: pexpect.spawn, cmd: str, settle: float = 0.0) -> None:
    child.sendline(cmd)
    if settle > 0:
        time.sleep(settle)


def _drive_menu(child: pexpect.spawn) -> None:
    print("[SensorMetadata] Driving GStreamer menu (wait for PLAYING, then plugin capture) ...")
    playing = False
    sent_p = False
    tries = 0
    max_tries = 40

    while tries < max_tries and not sent_p:
        try:
            idx = child.expect(
                [
                    r"Segmentation fault",
                    r"core dumped",
                    r"assertion\s+.*failed",
                    r"Already in PLAYING state",
                    r"Pipeline state changed.*to PLAYING",
                    r"pending:\s*VOID_PENDING",
                    r"Choose an option:\s*$",
                ],
                timeout=MENU_TIMEOUT,
            )
        except pexpect.TIMEOUT:
            tail = (child.before or "")[-500:] if child.before else "(empty)"
            print(f"[SensorMetadata] Timeout waiting for GStreamer menu. Last output from device:\n---\n{tail}\n---", file=sys.stderr)
            raise
        if idx in (0, 1, 2):
            tail = (child.before or "") + (child.after or "")
            hint = (
                "QMMF/camera context not available on device (e.g. when run over SSH before main app). "
                "Sensor metadata capture skipped."
                if "camera=0" in tail
                else "Try with camera index 0 (--camera 0) or ensure no other app is using the camera. "
            )
            raise RuntimeError(f"GStreamer pipeline crashed on device (qmmfsrc init failure). {hint} Device output:\n" + tail[-800:])
        if idx in (3, 4, 5):
            playing = True
            continue
        if not playing:
            time.sleep(DELAY_SHORT)
            child.sendline("3")
            tries += 1
            time.sleep(DELAY_SHORT)
        else:
            time.sleep(DELAY_SHORT)
            child.sendline("p")
            sent_p = True

    if not sent_p:
        raise RuntimeError("Never managed to enter Plugin Mode after reaching PLAYING.")

    child.expect(r"Enter plugin name or its index.*:", timeout=MENU_TIMEOUT)
    time.sleep(DELAY_SHORT)
    child.sendline("9")
    child.expect(r"Choose an option:\s*$", timeout=MENU_TIMEOUT)
    time.sleep(DELAY_SHORT)
    child.sendline("36")
    child.expect(r"Enter 'GstImageCaptureMode' value for arg0:", timeout=MENU_TIMEOUT)
    time.sleep(DELAY_SHORT)
    child.sendline("0")
    child.expect(r"Enter 'guint' value for arg1:", timeout=MENU_TIMEOUT)
    time.sleep(DELAY_SHORT)
    child.sendline("20")
    child.expect(r"Choose an option:\s*$", timeout=MENU_TIMEOUT)
    time.sleep(DELAY_LONG_CAPTURE)
    child.sendline("b")
    child.expect(r"Choose an option:\s*$", timeout=MENU_TIMEOUT)
    time.sleep(DELAY_SHORT)
    child.sendline("q")
    print("[SensorMetadata] GStreamer capture done, exiting menu.")


def _scp_pull(host: str, user: str, password: str, port: int, remote_file: str, local_file: str) -> None:
    print(f"[SensorMetadata] SCP pulling {remote_file} -> {local_file} ...")
    cmd = (
        f"scp -P {shlex.quote(str(port))} "
        f"-o StrictHostKeyChecking=accept-new "
        f"-o UserKnownHostsFile=/dev/null "
        f"{shlex.quote(user)}@{shlex.quote(host)}:{shlex.quote(remote_file)} "
        f"{shlex.quote(local_file)}"
    )
    child = pexpect.spawn(cmd, encoding="utf-8", timeout=SSH_TIMEOUT)
    while True:
        idx = child.expect(
            [
                r"Are you sure you want to continue connecting.*\?",
                r"(?i)password:",
                r"No such file or directory",
                pexpect.EOF,
            ],
            timeout=SSH_TIMEOUT,
        )
        if idx == 0:
            child.sendline("yes")
        elif idx == 1:
            if not password:
                raise RuntimeError("SCP password prompt received but password is empty.")
            child.sendline(password)
        elif idx == 2:
            raise RuntimeError(f"Remote file not found: {remote_file}")
        else:
            return


def save_sensor_metadata(
    host,
    output_folder,
    user="root",
    password="",
    port=22,
    remote_file="/data/capture/frame19.jpg",
    local_filename="sensor_metadata.jpg",
    gst_cmd=None,
):
    """
    Capture sensor metadata (JPEG with embedded metadata) on the device via SSH/GStreamer
    and save it into output_folder. RVC4 only. Call only when capture saves raw frames
    (left_raw or right_raw).
    """
    out_path = Path(output_folder).expanduser().resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    local_file = str(out_path / local_filename)

    if gst_cmd is None:
        gst_cmd = DEFAULT_GST_CMD

    child = _spawn_ssh(host, user, port)
    try:
        _ssh_login(child, password)
        print("[SensorMetadata] Sending setup commands (mount, camera metadata flags, mkdir, rm) ...")
        _send_cmd(child, "mount -o remount,rw /")
        _send_cmd(child, "echo enable3ADebugData=TRUE >> /vendor/etc/camera/camxoverridesettings.txt")
        _send_cmd(child, "echo enable3ADebugData=TRUE >> /vendor/etc/camera/camxoverridesettings.txt")
        _send_cmd(child, "echo enableTuningMetadata=TRUE >> /vendor/etc/camera/camxoverridesettings.txt")
        _send_cmd(child, "mkdir -p /data/capture/")
        _send_cmd(child, "rm -fr /data/capture/*")
        print("[SensorMetadata] Starting GStreamer pipeline on device ...")
        _send_cmd(child, gst_cmd)
        _drive_menu(child)
        print("[SensorMetadata] Exiting SSH session ...")
        _send_cmd(child, "exit")
        try:
            child.expect(pexpect.EOF, timeout=15)
        except pexpect.TIMEOUT:
            print("[SensorMetadata] SSH did not close in time, closing locally.", file=sys.stderr)
    finally:
        try:
            child.close(force=True)
        except Exception:
            pass

    _scp_pull(host, user, password, port, remote_file, local_file)
    print(f"[Capture] Sensor metadata saved: {local_file}")
