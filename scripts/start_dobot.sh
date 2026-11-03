#!/usr/bin/env bash
# start_dobot.sh - run after every restart: checks everything, then opens Isaac Sim with the task window.
#
#   bash scripts/start_dobot.sh            check + launch
#   bash scripts/start_dobot.sh --check    check only
#   bash scripts/start_dobot.sh --force    launch even if a check fails
#
# Settings come from .env (ISAACSIM_CMD, DOBOT_ASSETS, VLM_PROVIDER, ...; see .env.example).

. "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
PROVIDER="$(envget VLM_PROVIDER claude_code)"
MODE="${1:-}"
FAIL=0
ok()   { printf "  \033[32m[ok]\033[0m %s\n" "$1"; }
bad()  { printf "  \033[31m[!!]\033[0m %s\n" "$1"; FAIL=1; }
info() { printf "  [i]  %s\n" "$1"; }

echo "Dobot pick-and-place - startup check"

echo "Files (code: $N, assets: $ASSETS)"
for f in dobot_run_task.py dobot_capture_overhead.py dobot_vlm_keypoints.py dobot_pnp_run_simulation_v3.py \
         dobot_pnp_run_simulation_isaacsim6_v2.py; do
  [ -f "$N/$f" ] && ok "$f" || bad "missing $N/$f"
done
for f in stage3_dobot_gripper_test.usd load_dobot.urdf dobot_description.yaml; do
  [ -f "$ASSETS/$f" ] && ok "$f" || bad "missing $ASSETS/$f (see docs/SETUP.md, 'Scene assets'; or set DOBOT_ASSETS in .env)"
done
if [ -f "$ASSETS/dobot_description.yaml" ] && ! grep -q cspace "$ASSETS/dobot_description.yaml"; then
  bad "dobot_description.yaml has no 'cspace' (wrong file?)"
fi
grep -q 'def _routes_from_env' "$N/dobot_vlm_keypoints.py" 2>/dev/null \
  && ok "VLM script reads VLM_PROVIDER from .env" || bad "dobot_vlm_keypoints.py is an old copy"
for f in dobot_capture_overhead.py:'dobot.capture/2' dobot_vlm_keypoints.py:'def scene_cfg' \
         dobot_pnp_run_simulation_v3.py:'class MotionMonitor' dobot_run_task.py:'class EditWatch'; do
  grep -q "${f#*:}" "$N/${f%%:*}" 2>/dev/null && ok "${f%%:*}: scene-objects version" \
    || bad "${f%%:*} is an old copy (no added/deleted object support); save the latest one into $N"
done

echo "VLM provider: $PROVIDER"
if [[ ",$PROVIDER," == *",claude_code,"* || ",$PROVIDER," == *",claude-code,"* ]]; then
CLAUDE="$(command -v claude 2>/dev/null || echo "$HOME/.local/bin/claude")"
if [ -x "$CLAUDE" ]; then
  ok "claude CLI: $CLAUDE ($("$CLAUDE" --version 2>/dev/null | head -1))"
  AUTH="$(env -u ANTHROPIC_API_KEY "$CLAUDE" auth status 2>/dev/null)"
  if echo "$AUTH" | grep -q '"authMethod": "claude.ai"'; then
    ok "signed in with the Claude subscription ($(echo "$AUTH" | grep -o '"subscriptionType": "[a-z]*"' | cut -d'"' -f4))"
  else
    bad "not signed in with the Claude subscription: run 'claude', log in with your Claude.ai account, then /exit"
  fi
else
  bad "claude CLI not installed: curl -fsSL https://claude.ai/install.sh | bash   (then open a new terminal)"
fi
[ -n "${ANTHROPIC_API_KEY:-}" ] && info "ANTHROPIC_API_KEY is set here; the Claude Code route removes it and uses the subscription login"
fi

echo "Network"
case ",$PROVIDER," in
  *,none,*) URL="" ;;
  *,openai,*) URL="$(envget VLM_API_BASE https://api.openai.com/v1)" ;;
  *) URL="https://api.anthropic.com" ;;
esac
if [ -z "$URL" ]; then
  info "VLM_PROVIDER=none: no model is called (tasks are decided from their wording)"
elif curl -s -o /dev/null -m 8 --noproxy 'localhost,127.0.0.1' "$URL"; then
  ok "$URL reachable"
else
  bad "cannot reach $URL (Wi-Fi login page, proxy, or local server not started?). Without it the engine falls back to the task wording"
fi

echo "VLM engine"
if python3 "$N/dobot_vlm_keypoints.py" --check >/tmp/dobot_vlm_check.txt 2>&1; then
  ok "VLM engine online"
else
  bad "VLM engine check failed:"; sed 's/^/        /' /tmp/dobot_vlm_check.txt
fi

echo "Real robot + phone camera (optional; only for 'Use real scene')"
if [ -x "$RPY" ] && [ -f "$N/dobot_real.py" ]; then
  ok "real-robot environment present"
  [ -f "$DATA/output/real/calib.json" ] \
    && ok "camera-robot calibration present" || info "not calibrated yet: $RPY $N/dobot_real.py calibrate"
  ls /dev/ttyUSB* /dev/ttyACM* >/dev/null 2>&1 && ok "USB serial port present (Dobot)" || info "no USB serial port: Dobot off or unplugged"
  groups | grep -qw dialout || info "add yourself to 'dialout' for the Dobot port: sudo usermod -aG dialout \$USER (then log out/in)"
  [ -f "$N/dobot_link.py" ] || bad "dobot_link.py missing in $N (the Dobot driver)"
  if curl -s --max-time 1 --noproxy '*' http://127.0.0.1:5055/state 2>/dev/null | grep -q '"connected": true'; then
    ok "robot bridge running and the Dobot answers"
  else
    info "robot bridge not running: in another terminal  $RPY $N/dobot_link.py serve"
  fi
else
  info "not set up (see docs/REAL_ROBOT.md: phone camera + Dobot); simulation-only use is unaffected"
fi

echo "GPU"
TO=$(gsettings get org.gnome.mutter check-alive-timeout 2>/dev/null | grep -o '[0-9]*$')
if [ -n "$TO" ] && [ "$TO" -lt 30000 ]; then
  info "Ubuntu shows 'Wait or Force Quit' after $((TO / 1000)) s of a busy app. Isaac Sim is busy, not dead, during"
  info "capture and route planning. To allow 60 s: gsettings set org.gnome.mutter check-alive-timeout 60000"
fi
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  ok "$(nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader | head -1)"
else
  bad "NVIDIA driver not responding (nvidia-smi). Isaac Sim needs the GPU"
fi

echo
if [ "$FAIL" -ne 0 ] && [ "$MODE" != "--force" ]; then
  echo "Fix the [!!] items above, then run this again (or add --force to launch anyway)."
  exit 1
fi
[ "$MODE" = "--check" ] && { echo "All checks passed."; exit 0; }

ISAAC="$(envget ISAACSIM_CMD "$HOME/env_isaacsim/bin/isaacsim")"; ISAAC="${ISAAC/#\~/$HOME}"
if [ ! -x "${ISAAC%% *}" ]; then
  echo "Isaac Sim launcher not found at $ISAAC."
  echo "Start Isaac Sim the way you normally do, then: Window > Script Editor > open $N/dobot_run_task.py > Run."
  exit 1
fi
echo "Starting Isaac Sim; the 'Dobot pick-and-place' window opens when it has loaded ..."
if [ "${ISAAC##*/}" = "isaacsim" ]; then
  exec $ISAAC isaacsim.exp.full.kit --/app/runLoops/main/rateLimitEnabled=true \
    --/app/runLoops/main/rateLimitFrequency=30 --exec "$N/dobot_run_task.py"
else
  exec $ISAAC --exec "$N/dobot_run_task.py"
fi
