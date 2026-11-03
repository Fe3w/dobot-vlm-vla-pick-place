# Scene assets

The simulation needs three files here (or in the folder named by `DOBOT_ASSETS` in `.env`):

| File | What it is |
|---|---|
| `stage3_dobot_gripper_test.usd` | Isaac Sim stage: table, Dobot Magician at `/magician`, overhead camera `/overhead_camera`, objects |
| `load_dobot.urdf` | Dobot Magician URDF (joints `joint_1`, `joint_2`, `joint_5`, `joint_6`, link `link_suction`) |
| `dobot_description.yaml` | Lula robot description for the URDF (`cspace: [joint_1, joint_2, joint_5, joint_6]`) |

Optional:

- `stage3_dobot_repaired_test_20261021_040412.usd`: clean baseline the controller reloads before a run, if present.
- `dobot_wrist_calib.json`: written by `src/dobot_wrist_level.py` (wrist calibration of the simulated arm).

These files are not in git: USD scenes are large binaries. Share them separately (a release asset, or Git LFS) and
place them here. Scene expectations: Z-up, 1 metre per unit, robot root `/magician`, end effector
`/magician/link_suction`, camera `/overhead_camera` looking down at the table. Objects with a rigid body and
collider are pickable; others are treated as obstacles or references.
