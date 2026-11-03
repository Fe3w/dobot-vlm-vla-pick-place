#!/usr/bin/env bash
# One-time setup for the REAL robot side on Ubuntu 24.04: phone camera + Dobot Magician.
#
#   bash scripts/setup_real.sh
#
# Safe to run again (it skips what is already done). It asks before every sudo step.
# Creates the real-robot Python environment (DOBOT_REAL_PY, default ~/dobot_real/venv) and writes the phone
# camera address to .env (DOBOT_PHONE_URL / DOBOT_PHONE_MODE).

set -u
U="${USER:-$(id -un)}"
. "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
P="$RPY"
V="$(dirname "$(dirname "$P")")"
HOME_REAL="$(dirname "$V")"
ENVF="$ROOT/.env"
setenv() {   # setenv KEY VALUE: replace or append KEY=VALUE in .env (created from .env.example)
  [ -f "$ENVF" ] || cp "$ROOT/.env.example" "$ENVF"
  if grep -q -E "^[[:space:]]*#?[[:space:]]*$1=" "$ENVF"; then
    sed -i -E "s|^[[:space:]]*#?[[:space:]]*$1=.*|$1=$2|" "$ENVF"
  else
    printf '%s=%s\n' "$1" "$2" >> "$ENVF"
  fi
}

ok()   { printf "  [ok] %s\n" "$1"; }
bad()  { printf "  [!!] %s\n" "$1"; }
info() { printf "  [i]  %s\n" "$1"; }
ask()  { local a; read -r -p "  ?    $1 [y/N] " a; [[ "$a" =~ ^[Yy] ]]; }

is_cam() {   # $1 = ip: does http://ip:8080/shot.jpg return a JPEG?
  local t; t="$(curl -s --noproxy '*' --max-time 1.5 "http://$1:8080/shot.jpg" 2>/dev/null | head -c 2 | od -An -tx1 | tr -d ' \n')"
  [ "$t" = "ffd8" ]
}

gateway_ip() {   # default gateway (the phone, when the laptop is on its hotspot)
  local h
  h="$(awk '$2=="00000000" && $3!="00000000" {print $3; exit}' /proc/net/route 2>/dev/null)"
  [ -n "$h" ] && printf '%d.%d.%d.%d\n' "0x${h:6:2}" "0x${h:4:2}" "0x${h:2:2}" "0x${h:0:2}"
}

laptop_ips() {   # this laptop's IPv4 addresses (not loopback, not 192.0.0.x)
  { hostname -I 2>/dev/null | tr ' ' '\n'; ip -4 -o addr show scope global 2>/dev/null | awk '{print $4}' | cut -d/ -f1; } \
    | grep -E '^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$' | grep -v -E '^(127\.|192\.0\.0\.)' | sort -u
}

find_phone() {  # prints the phone's IP: the gateway first (laptop on the phone's hotspot), then each local /24
  local gw net i
  gw="$(gateway_ip)"
  if [ -n "$gw" ] && is_cam "$gw"; then echo "$gw"; return 0; fi
  for net in $(laptop_ips | awk -F. '{print $1"."$2"."$3}' | sort -u); do
    for i in $(seq 1 254); do
      ( is_cam "$net.$i" && echo "$net.$i" ) &
      [ $((i % 64)) -eq 0 ] && wait
    done
    wait
  done | head -n 1
}

echo "== 1. Files in $N"
missing=0
for f in dobot_link.py dobot_real.py dobot_run_task.py; do
  [ -f "$N/$f" ] && ok "$f" || { bad "$f missing"; missing=1; }
done
[ $missing -eq 0 ] || { bad "run this from a complete copy of the repository"; exit 1; }

echo "== 2. Python environment ($V)"
if ! python3 -c "import venv, ensurepip" >/dev/null 2>&1; then
  info "python3-venv is not installed"
  if ask "install it (sudo apt install -y python3-venv)?"; then sudo apt install -y python3-venv; fi
fi
mkdir -p "$HOME_REAL"
[ -x "$P" ] || python3 -m venv "$V" || { bad "could not create the environment"; exit 1; }
CHECK='import serial, numpy, cv2, PIL; cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)'
if ! "$P" -c "$CHECK" >/dev/null 2>&1; then
  "$P" -m pip install -q --upgrade pip
  "$P" -m pip install -q pyserial opencv-python numpy pillow || { bad "pip install failed (internet?)"; exit 1; }
fi
if "$P" -c "$CHECK" 2>/dev/null; then
  ok "pyserial, numpy, opencv (with ArUco), pillow installed"
else
  bad "the packages did not import; see the pip output above"; exit 1
fi

echo "== 3. Serial-port access for the Dobot"
if id -nG | grep -qw dialout; then
  ok "you are in the 'dialout' group"
else
  info "you are not in 'dialout' (needed to open /dev/ttyUSB*)"
  if ask "add yourself (sudo usermod -aG dialout $U)? Takes effect after you log out and in"; then
    sudo usermod -aG dialout "$U" && ok "added; this script uses it already, log out/in for everything else"
  fi
fi
if dpkg -s brltty >/dev/null 2>&1; then
  info "brltty (braille display support) is installed; it often grabs USB-serial chips like the Dobot's"
  if ask "remove it (sudo apt remove -y brltty)? Say no if you use a braille display"; then sudo apt remove -y brltty; fi
else
  ok "brltty not installed"
fi
RULE=/etc/udev/rules.d/99-dobot.rules
if [ -f "$RULE" ]; then
  ok "udev rule present ($RULE: /dev/dobot, ModemManager ignores it)"
else
  info "optional: a udev rule gives the Dobot a fixed name /dev/dobot and stops ModemManager probing it"
  if ask "add it (Silicon Labs CP210x 10c4:ea60; check with lsusb that your Dobot shows these ids)?"; then
    echo 'SUBSYSTEM=="tty", ATTRS{idVendor}=="10c4", ATTRS{idProduct}=="ea60", SYMLINK+="dobot", GROUP="dialout", MODE="0660", ENV{ID_MM_DEVICE_IGNORE}="1"' \
      | sudo tee "$RULE" >/dev/null && sudo udevadm control --reload && sudo udevadm trigger && ok "rule added (replug the Dobot)"
  fi
fi

echo "== 4. Phone camera (Android app 'IP Webcam': Start server; 1280x720; main lens; no auto-rotate)"
cur="$(envget DOBOT_PHONE_URL)"
PHONE_MODE="$(envget DOBOT_PHONE_MODE)"
[ -n "$cur" ] && info "current: $cur (mode ${PHONE_MODE:-?})"
echo "     1) USB cable + adb  (most reliable)"
echo "     2) Wi-Fi / phone hotspot (laptop on the phone's hotspot or the same Wi-Fi)"
echo "     3) keep the current setting"
read -r -p "  ?    choose 1/2/3: " ch
case "$ch" in
  1)
    if ! command -v adb >/dev/null 2>&1; then
      if ask "install adb (sudo apt install -y adb)?"; then sudo apt install -y adb; fi
    fi
    command -v adb >/dev/null 2>&1 || { bad "adb not installed"; exit 1; }
    info "on the phone: Settings > About > tap Build number 7 times; Developer options > USB debugging ON; plug in; accept the prompt"
    read -r -p "  ?    press Enter when the phone is plugged in and IP Webcam's server is started ... " _
    if adb devices | awk 'NR>1 && $2=="device"' | grep -q .; then
      adb forward tcp:8080 tcp:8080 >/dev/null && ok "adb forward tcp:8080 -> phone"
    else
      bad "no authorised phone in 'adb devices' (accept the USB debugging prompt on the phone)"; adb devices
    fi
    PHONE_MODE=usb; PHONE_URL="http://127.0.0.1:8080/shot.jpg" ;;
  2)
    info "laptop addresses: $(laptop_ips | tr '\n' ' ') gateway: $(gateway_ip)"
    info "searching for the phone camera (IP Webcam server must be running) ..."
    found="$(find_phone)"
    if [ -n "$found" ]; then
      ok "phone camera found at $found"
      read -r -p "  ?    use $found? Enter = yes, or type another IP: " ip
      ip="${ip:-$found}"
    else
      bad "not found on this network. Laptop and phone must be on the same Wi-Fi, or the laptop on the phone's hotspot"
      info "(college / guest Wi-Fi often blocks phone-to-laptop traffic: use the hotspot or USB instead)"
      read -r -p "  ?    phone IP to use anyway (NOT a 192.0.0.x address; Enter = skip): " ip
    fi
    case "$ip" in
      192.0.0.*) bad "$ip is the phone's internal mobile-data address, not reachable from the laptop"; ip="" ;;
    esac
    PHONE_MODE=wifi; PHONE_URL="${ip:+http://$ip:8080/shot.jpg}" ;;
  *)
    PHONE_MODE="${PHONE_MODE:-wifi}"; PHONE_URL="${cur:-}"
    [ -n "$PHONE_URL" ] || bad "no phone address set yet: run again and choose 1 or 2" ;;
esac
if [ -n "${PHONE_URL:-}" ]; then
  setenv DOBOT_PHONE_MODE "$PHONE_MODE"
  setenv DOBOT_PHONE_URL "$PHONE_URL"
  ok "phone address saved in .env (DOBOT_PHONE_URL=$PHONE_URL)"
  tmp="$(mktemp --suffix=.jpg)"
  if curl -s --noproxy '*' --max-time 6 -o "$tmp" "$PHONE_URL" && [ "$(head -c 2 "$tmp" | od -An -tx1 | tr -d ' \n')" = "ffd8" ]; then
    ok "phone camera answers: $(stat -c %s "$tmp") byte JPEG from $PHONE_URL"
  else
    bad "no image from $PHONE_URL (app server started? same network / adb forward?)"
  fi
  rm -f "$tmp"
fi

echo "== 5. Dobot (power on, USB cable in)"
run_dobot() {   # use the dialout group right away, even before logging out and in
  if id -nG | grep -qw dialout; then "$P" "$N/dobot_link.py" "$@"
  else sg dialout -c "$(printf '%q ' "$P" "$N/dobot_link.py" "$@")"; fi
}
run_dobot ports
if run_dobot pose; then
  ok "the Dobot answers"
  if ask "home the arm now (it swings to its limit switches; clear the area)?"; then run_dobot home; fi
else
  bad "no answer from the Dobot: power, cable, dialout (log out/in), or another program on the port"
fi

echo
echo "== Setup done. Next (once):"
echo "   (from the repo folder: cd $ROOT)"
echo "   $P $N/dobot_real.py check        camera + robot"
echo "   $P $N/dobot_real.py view         live phone view (q quits) - line up the stand"
echo "   $P $N/dobot_real.py marker       print aruco_0.png (~6 cm square)"
echo "   $P $N/dobot_real.py calibrate    8 marker spots: 5 on the table, 3 on a box"
echo "   $P $N/dobot_real.py background   empty table, arm parked out of view"
echo "Then every session:  bash scripts/start_real.sh"
