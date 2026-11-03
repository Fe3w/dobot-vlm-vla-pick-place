# Real Dobot Magician with a phone camera

The simulation stays the planner and the safety check. The real arm only replays a path that passed in
simulation, after a fresh scan shows the table is unchanged.

```
phone (IP Webcam) --Wi-Fi / hotspot / USB--> laptop
  dobot_real.py scan     photo vs. empty-table photo -> objects -> robot millimetres (calibrated camera model)
  task window            "Use real scene": stand-ins replace the simulated objects -> VLM -> plan -> simulated run
  dobot_real.py execute  re-scan, compare, confirm, replay on the Dobot over USB (dobot_link.py)
```

All commands below are run from the repository folder. In the commands:

```bash
P=$(grep -s '^DOBOT_REAL_PY=' .env | cut -d= -f2); P=${P/#\~/$HOME}; P=${P:-$HOME/dobot_real/venv/bin/python}
R="$P src/dobot_real.py"
L="$P src/dobot_link.py"
```

## 1. One-time setup

```bash
bash scripts/setup_real.sh
```

The script asks before every `sudo` step. It does the following:

1. Creates the Python environment (`pyserial`, `numpy`, `opencv-python`, `pillow`).
2. Adds you to the `dialout` group for the serial port. Log out and in afterwards.
3. Offers to remove `brltty`, which grabs USB-serial chips, and to add a udev rule for `/dev/dobot`.
4. Finds the phone camera and writes `DOBOT_PHONE_URL` to `.env`.
5. Checks the Dobot and offers to home it.

### Phone

- App: **IP Webcam** (Android). Press *Start server*. The snapshot URL is `http://<phone>:8080/shot.jpg`.
- Settings: 1280×720 (keep it fixed after calibration), main 1x lens, no auto-rotate, no login, screen kept on,
  phone on its charger.
- Network: put the laptop on the phone's hotspot. Campus Wi-Fi often blocks traffic between devices. Or use USB:
  `adb forward tcp:8080 tcp:8080` with `DOBOT_PHONE_URL=http://127.0.0.1:8080/shot.jpg`. Ignore `192.0.0.x`
  addresses.
- Mount the phone face down, 50–70 cm above the table, seeing the robot base and the reach area. It must not move
  after calibration.

### Dobot

12 V power, USB to the laptop. Check the connection, then home after every power-on:

```bash
$L ports
$L pose
$L home        # the arm swings to its limit switches: clear the area
```

`$P src/dobot_teleop.py` gives keyboard control of the arm and the suction cup. It is the quickest way to
see that the arm responds.

## 2. Calibrate the camera to the arm

```bash
$R check             # frame size from the phone, pose from the robot
$R view              # live view (q quits): adjust the phone until the work area is in view
$R marker            # print aruco_0.png at about 6 cm and glue it on card
$R calibrate         # 8 points: 5 flat on the table, 3 on top of a 30-80 mm box
```

For each point:

1. Put the marker down, move the arm out of view, and press Enter.
2. Hold the forearm UNLOCK button, set the suction tip on the marker centre, let go, and press Enter.

Aim for an error under 4 mm. `calibrate-all` does the same with several markers laid out at once: one photo,
numbered, then you touch them one by one. Calibration also fits the suction tip's offset along the arm.

Then clear the table, park the arm, and take the empty-table photo:

```bash
$R park
$R background        # retake if the lighting changes
```

## 3. Tell it about your objects

A phone sees colour, not height. Heights come from `REAL["OBJECTS"]` in `src/dobot_real.py`, one line per object:

```python
{"colour": "red", "shape": "cylinder", "height_mm": 45, "pickable": True},
{"colour": "green", "shape": "sphere", "height_mm": 40, "pickable": False},   # suction needs a flat top
```

Check with `$R view` or `$R scan`. Each object should get the right name, outline and height, and nothing else
(shadows, the arm) should be detected.

Accuracy check without Isaac Sim: `$R hovertest` hovers the cup 15 mm above each object's top centre.

## 4. Every session

```bash
bash scripts/start_real.sh
```

The script checks the phone, starts the Dobot bridge (`dobot_link.py serve`, the only program that holds the USB
port), and opens Isaac Sim. In the task window:

1. Tick **Use real scene**, type the task, and press **Run task**. The arm is parked automatically before the scan.
2. Check that the stand-ins sit on the same side of the robot as the real objects. If they are mirrored or
   rotated, set `YAW_DEG` in `src/dobot_real.py`.
3. The Phone camera window shows the detected objects, the pick point, the place point and the planned path.
4. **Send to real Dobot** (click twice within 6 s) re-scans. It refuses if anything moved more than 10 mm,
   appeared or disappeared, then replays the checked path.

The same from a terminal: `$R plan` (dry run: prints every move), then `$R execute`.

## Safety limits

| Setting (`REAL` in `src/dobot_real.py`) | Default |
|---|---|
| Travel / final approach speed | 40 / 15 mm/s |
| Reach ring | 135–315 mm from the base |
| Park pose | x 200, y 0, 120 mm above the table |
| Scan only when parked | within 30 mm (`AUTO_PARK` moves it there first) |

The Magician has no emergency stop. Keep a hand near the power switch, and use **Stop** in the window or Ctrl+C
in the terminal. Either one force-stops and clears the robot's queue.

## Where the data goes

All of it is under `$DOBOT_DATA/output/real/`:

| Data | File |
|---|---|
| Calibration, empty-table photo, marker | `calib.json`, `background.png`, `aruco_0.png` |
| Scans | `scans/<id>/real_scene.json`, `rgb.png`, `view.png` |
| Hand-over to the arm | `latest_plan.json`, `real_plan_moves.json` (`$R export`) |

`tools/windows/dobot_replay_windows.py` replays an exported plan from Windows with Dobot's official DLL.
