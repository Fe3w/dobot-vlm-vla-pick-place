# Troubleshooting

## Simulation

| Symptom | Cause / fix |
|---|---|
| `missing assets/...` in `start_dobot.sh --check` | Put the scene files in `assets/`, or set `DOBOT_ASSETS` in `.env` |
| `route openai DOWN key rejected (HTTP 401)` | Wrong `VLM_API_KEY`, or a key for another service |
| `model '...' not offered` | `VLM_MODEL` is not in that endpoint's `/models` list; the closest match is printed |
| `not reachable` | Wrong `VLM_API_BASE`, local server not started, or a Wi-Fi login page / proxy |
| `no grasp point lands on a pickable object` | The pick object has no rigid body + collider, or it is hidden from the camera |
| Ubuntu shows "Wait or Force Quit" | Isaac Sim is busy (capture, route planning), not hung: `gsettings set org.gnome.mutter check-alive-timeout 60000` |
| Run refused: low memory | Fewer than 2.5 GB RAM or 0.9 GB GPU free; close other apps (`MIN_FREE_*` in `dobot_run_task.py`) |
| `differs from the reviewed V2 (SHA-256 ...)` | The controller file was edited; update `V2_SHA256` in `dobot_pnp_run_simulation_v3.py` after reviewing the change |

## Phone camera

| Symptom | Cause / fix |
|---|---|
| HTTP 404 | Wrong address or path: it must end in `/shot.jpg` |
| HTTP 401 | IP Webcam has a login set: remove it (Connection settings → Login/password) |
| Timeouts | The phone slept or the screen locked: keep it on its charger with the screen on |
| Address is `192.0.0.x` | That is the phone's mobile-data interface; use the hotspot address or USB + adb |
| Phantom objects | The arm or shadows differ from `background`; park the arm and retake `background` in the current light |
| Stand-ins mirrored | Set `YAW_DEG` in `src/dobot_real.py` |

## Dobot

| Symptom | Cause / fix |
|---|---|
| No `/dev/ttyUSB*` | Power, cable; `sudo apt remove brltty`; check `lsusb` for `10c4:ea60` |
| Permission denied on the port | `sudo usermod -aG dialout $USER`, then log out and in |
| Port busy | Only one program may hold it: stop the bridge (`pkill -f "dobot_link.py serve"`) or teleop first |
| Red LED, no motion | Alarm, often a joint beyond its limit: move the arm back by hand (UNLOCK button), `src/dobot_link.py clear-alarms`, then `home` |
| Moves accepted but no motion | Check that `LINK["BACKEND"]` in `src/dobot_link.py` is `"teleop"` (the default); `src/dobot_move_diag.py` tests three tiny moves and decodes the alarms |
| "arm is N mm from its park pose" | Auto-park could not reach the park pose: check alarms, home, then `src/dobot_real.py park` |
