# Setup (simulation)

## 1. Isaac Sim 6.0

Install Isaac Sim 6.0.x in its own Python virtual environment, following NVIDIA's pip installation guide.
The result should look like this (the exact package extras and index URL are in NVIDIA's guide):

```bash
python3.12 -m venv ~/env_isaacsim
~/env_isaacsim/bin/pip install "isaacsim[all,extscache]==6.0.1" --extra-index-url https://pypi.nvidia.com
~/env_isaacsim/bin/isaacsim        # first start takes a while (shader cache)
```

If you start Isaac Sim another way, set `ISAACSIM_CMD` in `.env`.

## 2. The repository and `.env`

```bash
git clone https://github.com/<you>/dobot-vlm-pick-place.git
cd dobot-vlm-pick-place
cp .env.example .env
chmod 600 .env                     # it will hold your API key
```

Open `.env` and set the VLM (see [VLM.md](VLM.md)). Every setting can also be given as an environment variable,
which wins over the file.

Isaac Sim's Script Editor runs a script without knowing where it lives on disk. So that the scripts still find
`.env` when you open them there, link it once:

```bash
mkdir -p ~/.config/dobot-pnp && ln -sf "$PWD/.env" ~/.config/dobot-pnp/.env
```

## 3. Scene assets

Put the three files listed in [../assets/README.md](../assets/README.md) in `assets/`, or point `DOBOT_ASSETS` at
the folder that holds them. `DOBOT_DATA` is where `output/` is created (default: the repo folder).

## 4. Check and start

```bash
bash scripts/start_dobot.sh --check
```

This checks the scripts, the assets, the VLM settings and connection, the network and the GPU. Then:

```bash
bash scripts/start_dobot.sh
```

Isaac Sim opens the stage and the **Dobot pick-and-place** window. Alternatively, start Isaac Sim yourself and
open `src/dobot_run_task.py` in Window → Script Editor, then press Run with no text selected.

## 5. Run

Type a task and press **Run task**. The steps are VLM check → scene reset → capture → VLM decision → controller.
After the placement, the window checks the result against the task: the object sits next to the reference on the
asked side, stands upright, and nothing else moved. It prints `[ok] check passed` or what failed.

- **Reset scene** reloads the stage's saved state.
- **Run test list** runs every line of `output/isaac_capture/test_tasks.txt`. Then
  `python3 src/dobot_run_report.py` summarises the runs.
- To add objects: Reset scene, add shapes in the stage (table-sized, 8–300 mm), and give pickable ones a rigid body
  with colliders. Objects are named by their rendered colour and shape.

## 6. Wrist calibration (optional)

If the simulated suction cup does not point straight down the way the real one does, run `src/dobot_wrist_level.py`
in the Script Editor. It measures the wrist rule `joint_6 = s·(joint_2 + joint_5) + b` and saves
`dobot_wrist_calib.json` in the assets folder. The controller picks it up on the next run.
