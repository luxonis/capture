#!/usr/bin/env bash
#
# Capture an image with metadata using given camera.
# 
# Requires 'expect' to be installed.
#
# Usage:
#     HOST=10.11.1.2 SSH_USER=root PASS=secretpassword ./get_metadata.sh
# The frame.jpg with metadata will be in the current folder.
set -euo pipefail

HOST="${HOST:-10.11.1.176}"
SSH_USER="${SSH_USER:-root}"
PASS="${PASS:-luxonis}"     # empty => SSH key auth
PORT="${PORT:-22}"

REMOTE_FILE="${REMOTE_FILE:-/data/capture/frame19.jpg}"
LOCAL_FILE="${LOCAL_FILE:-./frame.jpg}"

GST_CMD='gst-pipeline-app -e qtiqmmfsrc camera=1 name=camsrc ! video/x-raw,format=NV12,width=1280,height=800,framerate=30/1 ! jpegenc ! multipartmux boundary="frame" ! tcpserversink host=0.0.0.0 port=5022 camsrc.image_1 ! "image/jpeg,width=1280,height=800,framerate=30/1" ! multifilesink location=/data/capture/frame%d.jpg sync=true async=false camsrc.image_2 ! "video/x-bayer,format=mono,bpp=(string)10,width=1280,height=800" ! multifilesink location=/data/capture/frame%d.raw sync=true async=false'

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

RUN_EXP="$tmpdir/run.exp"
SCP_EXP="$tmpdir/scp.exp"

# ---------------- expect script: SSH + interactive app ----------------
cat > "$RUN_EXP" <<'EXPECT_EOF'
#!/usr/bin/expect -f
set timeout 180

# argv: host user pass port gstcmd
set host [lindex $argv 0]
set user [lindex $argv 1]
set pass [lindex $argv 2]
set port [lindex $argv 3]
set gst  [lindex $argv 4]

proc cmd {s} { send -- "$s\r" }

spawn ssh -p $port -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/dev/null $user@$host

expect {
  -re "Are you sure you want to continue connecting.*\\?" {
    send -- "yes\r"
    exp_continue
  }
  -re "(P|p)assword:" {
    if {$pass eq ""} {
      send_user "\nERROR: Password prompt received but PASS is empty.\n"
      exit 2
    }
    send -- "$pass\r"
    exp_continue
  }
  -re "Permission denied" {
    send_user "\nERROR: Permission denied.\n"
    exit 3
  }
  -re {[$#] $} {
    # shell prompt
  }
  timeout {
    send_user "\nERROR: SSH connect timed out.\n"
    exit 4
  }
}

# Prep
cmd "mount -o remount,rw /"
cmd "echo enable3ADebugData=TRUE >> /vendor/etc/camera/camxoverridesettings.txt"
cmd "echo enable3ADebugData=TRUE >> /vendor/etc/camera/camxoverridesettings.txt"
cmd "mkdir -p /data/capture/"
cmd "rm -fr /data/capture/*"

# Start gst app
cmd "$gst"

# Make expect buffer larger (menus are big)
match_max 200000
set timeout 180

# ---------------- Go to PLAYING, then immediately enter Plugin Mode ----------------
set max_tries 40
set tries 0
set playing 0
set sent_p 0

while {$tries < $max_tries && !$sent_p} {
  expect {
    -re "Already in PLAYING state" {
      set playing 1
      exp_continue
    }
    -re "Pipeline state changed.*to PLAYING" {
      set playing 1
      exp_continue
    }
    -re "pending: VOID_PENDING" {
      set playing 1
      exp_continue
    }

    # This prompt can appear many times. Decide what to send based on state.
    -re "Choose an option:\\s*$" {
      if {!$playing} {
        send -- "3\r"
        incr tries
        after 200
      } else {
        send -- "p\r"
        set sent_p 1
      }
    }

    timeout {
      incr tries
      after 300
    }
  }
}

if {!$sent_p} {
  send_user "\nERROR: Never managed to enter Plugin Mode after reaching PLAYING.\n"
  exit 31
}


# ---------------- Plugin selection: choose camsrc (9) ----------------
expect {
  -re "Enter plugin name or its index.*:" {
    after 200
    send -- "9\r"
  }
  timeout { send_user "\nERROR: No plugin selection prompt.\n"; exit 32 }
}

# ---------------- Trigger capture-image signal (36) ----------------
expect {
  -re "Choose an option:\\s*$" {
    after 200
    send -- "36\r"
  }
  timeout { send_user "\nERROR: No plugin menu after selecting camsrc.\n"; exit 33 }
}

expect {
  -re "Enter 'GstImageCaptureMode' value for arg0:" {
    after 200
    send -- "0\r"
  }
  timeout { send_user "\nERROR: No arg0 prompt.\n"; exit 34 }
}

expect {
  -re "Enter 'guint' value for arg1:" {
    after 200
    # how many frames to capture
    send -- "20\r"
  }
  timeout { send_user "\nERROR: No arg1 prompt.\n"; exit 35 }
}

# ---------------- Back to pipeline menu, then quit ----------------
expect {
  -re "Choose an option:\\s*$" {
    # wait longer for the images to be actually captured and saved
    after 2000
    send -- "b\r"
  }
  timeout { send_user "\nERROR: No prompt for back.\n"; exit 36 }
}

expect {
  -re "Choose an option:\\s*$" {
    after 200
    send -- "q\r"
  }
  timeout { send_user "\nERROR: No prompt for quit.\n"; exit 37 }
}

# Wait for return to shell (best effort)
expect {
  -re {[$#] $} { }
  timeout { }
}

cmd "exit"
expect eof
EXPECT_EOF
chmod +x "$RUN_EXP"

# ---------------- expect script: SCP back ----------------
cat > "$SCP_EXP" <<'EXPECT_EOF'
#!/usr/bin/expect -f
set timeout 180

# argv: host user pass port remote local
set host  [lindex $argv 0]
set user  [lindex $argv 1]
set pass  [lindex $argv 2]
set port  [lindex $argv 3]
set rfile [lindex $argv 4]
set lfile [lindex $argv 5]

spawn scp -P $port -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/dev/null $user@$host:$rfile $lfile

expect {
  -re "Are you sure you want to continue connecting.*\\?" {
    send -- "yes\r"
    exp_continue
  }
  -re "(P|p)assword:" {
    if {$pass eq ""} {
      send_user "\nERROR: SCP password prompt received but PASS is empty.\n"
      exit 2
    }
    send -- "$pass\r"
    exp_continue
  }
  -re "No such file or directory" {
    send_user "\nERROR: Remote file not found: $rfile\n"
    exit 21
  }
  eof
}
EXPECT_EOF
chmod +x "$SCP_EXP"

# Run
expect "$RUN_EXP" "$HOST" "$SSH_USER" "$PASS" "$PORT" "$GST_CMD"
expect "$SCP_EXP" "$HOST" "$SSH_USER" "$PASS" "$PORT" "$REMOTE_FILE" "$LOCAL_FILE"

echo "Done. Pulled: $REMOTE_FILE -> $LOCAL_FILE"
