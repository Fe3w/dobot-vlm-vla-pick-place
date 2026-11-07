# Guide: setup, configuration and usage

Detailed instructions for running the project. For the overview, see the [README](../README.md).
Step-by-step pages: [SETUP.md](SETUP.md) (simulation), [VLM.md](VLM.md) (model providers),
[REAL_ROBOT.md](REAL_ROBOT.md) (phone camera and physical Dobot), [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

## Requirements

- Ubuntu 22.04 or 24.04 with an NVIDIA RTX GPU (8 GB VRAM is enough), NVIDIA Isaac Sim 6.0.x
- The scene assets (USD stage, URDF, Lula YAML). See [assets/README.md](../assets/README.md).
- A vision-language model: any OpenAI-compatible API with image input, a local model, the Claude API, or a
  signed-in Claude Code CLI
- Real robot only: Dobot Magician with suction cup, USB cable, and an Android phone with the IP Webcam app

## Quick start (simulation)

```bash
git clone https://github.com/Fe3w/dobot-vlm-vla-pick-place.git
cd dobot-vlm-vla-pick-place
cp .env.example .env            # then set VLM_PROVIDER, VLM_API_BASE, VLM_MODEL, VLM_API_KEY
# put the scene assets in assets/ (see assets/README.md)
bash scripts/start_dobot.sh --check   # checks files, VLM access, GPU
bash scripts/start_dobot.sh           # opens Isaac Sim with the "Dobot pick-and-place" window
```

Type a task in the window and press **Run task**. Full setup: [docs/SETUP.md](SETUP.md).

## Choosing the VLM

Set `VLM_PROVIDER` in `.env`:

| Provider | Settings | Notes |
|---|---|---|
| `openai` | `VLM_API_BASE`, `VLM_MODEL`, `VLM_API_KEY` | OpenAI, OpenRouter, Gemini's OpenAI endpoint, Groq, Ollama, vLLM, LM Studio, local routers |
| `anthropic` | `ANTHROPIC_API_KEY`, `VLM_MODEL` | Claude API, billed per request |
| `claude_code` | `VLM_CLAUDE_CODE_MODEL` | Claude Code CLI on a Claude subscription; API keys in the environment are removed for the call |
| `none` | | No model: unambiguous tasks are decided from their wording |

Several providers at once (`VLM_PROVIDER=openai,claude_code`) are asked in parallel, and their answers must agree.
Test the connection with `python3 src/dobot_vlm_keypoints.py --check`. Details: [docs/VLM.md](VLM.md).

### Running with a local model (no API key, no internet)

The model only has to say which numbered tag is which. It never estimates coordinates, so a small local vision
model on the same laptop is enough. With [Ollama](https://ollama.com):

```bash
ollama pull qwen2.5vl:7b          # any vision model; smaller ones answer faster
```

```ini
# .env
VLM_PROVIDER=openai
VLM_API_BASE=http://127.0.0.1:11434/v1
VLM_MODEL=qwen2.5vl:7b
VLM_API_KEY=
```

Then run `python3 src/dobot_vlm_keypoints.py --check`. It should print `route openai up ok (model offered)`.

- A 7B vision model in 4-bit needs about 6 GB of GPU memory. Isaac Sim needs most of an 8 GB card, so on a laptop
  you may have to run the model on the CPU (slower), or use a smaller model.
- Raise `VLM_DEADLINE_S` (default 40 s) if the first answer is slow while the model loads.
- The answer is still cross-checked against the task wording, and every pixel is verified with depth rays before
  the arm moves. With a weaker model, name objects clearly in the task ("the red cylinder"): the wording check
  can then catch a wrong answer.

## Real robot

```bash
bash scripts/setup_real.sh      # once: Python environment, serial access, phone camera, Dobot check
# once: calibrate the phone camera against the arm (docs/REAL_ROBOT.md)
bash scripts/start_real.sh      # every session: phone link, Dobot bridge, Isaac Sim
```

Tick **Use real scene**, press **Run task**, check the plan in the Phone camera window, then click
**Send to real Dobot** twice. Step-by-step guide: [docs/REAL_ROBOT.md](REAL_ROBOT.md).

## Repository layout

```text
src/            runtime scripts (Isaac Sim side and real-robot side), all in one folder because they load each other
scripts/        start_dobot.sh, start_real.sh, setup_real.sh (+ _env.sh, shared settings loader)
diagnostics/    read-only Isaac Sim scene and gripper audits
tools/          git hooks, Windows replay of a plan
docs/           setup, VLM, real robot, architecture, troubleshooting
assets/         scene assets (not in git)
output/         run data: captures, VLM decisions, calibration, run log (created at run time, not in git)
```

| File | Role |
|---|---|
| `src/dobot_run_task.py` | Task window in Isaac Sim: Run task, Reset scene, Stop, test lists, real-robot controls |
| `src/dobot_capture_overhead.py` | Renders the overhead camera, checks the camera model, writes `capture.json` |
| `src/dobot_vlm_keypoints.py` | VLM engine: tagged image → target / reference / relation → `keypoints_2d_vlm.json` |
| `src/dobot_pnp_run_simulation_v3.py` | Checks the keypoints against the capture, then runs the controller |
| `src/dobot_pnp_run_simulation_isaacsim6_v2.py` | Controller: depth rays, IK, route planning, kinematic suction latch |
| `src/dobot_real.py` | Real side: phone camera, calibration, scan, plan, execute |
| `src/dobot_link.py` | Dobot serial driver and localhost bridge (`serve`) |
| `src/dobot_teleop.py` | Keyboard teleop of the real arm and suction (standalone) |
| `src/dobot_run_report.py` | Terminal summary of recorded runs |

## Safety

- Every plan runs in simulation first. The real arm only replays a path that passed the result check, after a
  re-scan shows nothing moved.
- Moves are limited to a reach ring (135–315 mm from the base), never below the table, at reduced speed.
- The Dobot Magician has no emergency stop. Keep a hand near the power switch, and use **Stop** in the task
  window or Ctrl+C in a terminal.
- The robot bridge only listens on `127.0.0.1`.
