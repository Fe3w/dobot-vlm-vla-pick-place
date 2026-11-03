#!/usr/bin/env bash
# Every session with the real robot: phone link, Dobot bridge, then Isaac Sim with the task window.
#
#   bash scripts/start_real.sh
#
# - USB phone: re-creates the adb port forward (needed after every reconnect).
# - Starts the Dobot bridge (dobot_link.py serve) in the background; its log is ~/dobot_real/bridge.log.
# - Starts Isaac Sim through start_dobot.sh. When Isaac Sim closes, the bridge is stopped (robot queue cleared).

set -u
. "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
P="$RPY"
LOG="$HOME/dobot_real/bridge.log"
mkdir -p "$(dirname "$LOG")"
ok()   { printf "  [ok] %s\n" "$1"; }
bad()  { printf "  [!!] %s\n" "$1"; }
info() { printf "  [i]  %s\n" "$1"; }

[ -x "$P" ] || { bad "run scripts/setup_real.sh first"; exit 1; }
PHONE_MODE="$(envget DOBOT_PHONE_MODE wifi)"
PHONE_URL="$(envget DOBOT_PHONE_URL)"

echo "== Phone camera"
if [ "${PHONE_MODE:-}" = "usb" ]; then
  if adb devices 2>/dev/null | awk 'NR>1 && $2=="device"' | grep -q .; then
    adb forward tcp:8080 tcp:8080 >/dev/null && ok "adb forward tcp:8080"
  else
    bad "phone not seen by adb (cable, USB debugging prompt)"
  fi
fi
if [ -n "${PHONE_URL:-}" ] && curl -s --noproxy '*' --max-time 5 -o /dev/null -w '%{http_code}' "$PHONE_URL" | grep -q 200; then
  ok "phone camera answers ($PHONE_URL)"
else
  info "no image from ${PHONE_URL:-<not set>} yet: start IP Webcam's server (the Phone camera window keeps retrying)"
fi

echo "== Dobot bridge"
BR_PID=""
state() { curl -s --noproxy '*' --max-time 1 http://127.0.0.1:5055/state 2>/dev/null; }
if state | grep -q '"connected": true'; then
  ok "bridge already running"
else
  if id -nG | grep -qw dialout; then
    nohup "$P" "$N/dobot_link.py" serve >"$LOG" 2>&1 &
  else
    nohup sg dialout -c "$(printf '%q ' "$P" "$N/dobot_link.py" serve)" >"$LOG" 2>&1 &
  fi
  BR_PID=$!
  for _ in $(seq 1 20); do state | grep -q '"connected": true' && break; sleep 0.5; done
  if state | grep -q '"connected": true'; then
    ok "bridge started (pid $BR_PID, log $LOG)"
  else
    bad "bridge did not connect to the Dobot; last log lines:"; tail -n 5 "$LOG" | sed 's/^/       /'
    info "continuing without the robot (the simulation still works). After fixing it, in another terminal:"
    info "  $P $N/dobot_link.py serve"
  fi
fi

stop_bridge() {
  if [ -n "$BR_PID" ] && kill -0 "$BR_PID" 2>/dev/null; then
    curl -s --noproxy '*' --max-time 2 -X POST -H 'Content-Type: application/json' -d '{}' \
      http://127.0.0.1:5055/stop >/dev/null 2>&1
    kill -INT "$BR_PID" 2>/dev/null; sleep 1; kill "$BR_PID" 2>/dev/null
    pkill -INT -f "dobot_link.py serve" 2>/dev/null
    echo "  [ok] bridge stopped"
  fi
}
trap stop_bridge EXIT

echo "== Isaac Sim"
bash "$N/start_dobot.sh" "$@"
