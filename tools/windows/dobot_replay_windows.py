"""
dobot_replay_windows.py - run a plan made on Ubuntu (Isaac Sim + phone camera) on the real Dobot from WINDOWS,
with the official DobotDllType library that already moves this robot (your C:\\dobot_fence setup).

On Ubuntu:   ~/dobot_real/venv/bin/python ~/Downloads/n/dobot_real.py export
             -> .../output/real/real_plan_moves.json  (copy it to the Windows laptop, e.g. by USB stick)
On Windows:  put this file and real_plan_moves.json in C:\\dobot_fence (next to DobotDllType.py and the DLLs),
             close DobotStudio, then
                 py -3.8-32 dobot_replay_windows.py real_plan_moves.json
             It prints the current pose and every move, asks "yes", then runs them: joint moves (PTPMOVJXYZMode),
             suction on/off, pauses, and back to the park pose. Ctrl+C stops the robot.

The coordinates are the Dobot's own (as reported by GetPose on Ubuntu when the camera was calibrated). Check: with
the arm parked, the pose printed here must match `dobot_link.py pose` on Ubuntu within a few mm; if not, the
end-effector settings differ between the two computers and the cup will miss.
"""
import json
import sys

import DobotDllType as dType


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "real_plan_moves.json"
    with open(path) as f:
        plan = json.load(f)
    steps = plan["steps"]
    api = dType.load()
    state = dType.ConnectDobot(api, "", 115200)[0]
    if state != dType.DobotConnect.DobotConnect_NoError:
        print("cannot connect to the Dobot (DobotStudio still open? cable?):", state)
        return 1
    try:
        pose = dType.GetPose(api)
        r = pose[3]
        print(f"connected; pose x {pose[0]:.1f} y {pose[1]:.1f} z {pose[2]:.1f} r {pose[3]:.1f}")
        print(f'task: "{plan.get("task")}"  ({sum(1 for s in steps if s["type"] == "move")} moves)')
        for s in steps:
            if s["type"] == "move":
                x, y, z = s["xyz"]
                print(f"   {s['label']:18s} {s['speed']:4s}  x {x:7.1f}  y {y:7.1f}  z {z:7.1f}")
            elif s["type"] == "suck":
                print(f"   suction {'ON' if s['on'] else 'OFF'}")
            else:
                print(f"   wait {s['seconds']:.1f} s")
        if input("Move the REAL robot? Hand near the power switch. Type yes: ").strip().lower() != "yes":
            print("cancelled")
            return 1
        dType.SetQueuedCmdStopExec(api)
        dType.SetQueuedCmdClear(api)
        dType.SetPTPJointParams(api, 40, 40, 40, 40, 40, 40, 40, 40, isQueued=1)
        dType.SetPTPCommonParams(api, 100, 100, isQueued=1)
        speeds = {"fast": plan["speed"], "slow": plan["slow_speed"]}
        cur, last = None, None
        for s in steps:
            if s["type"] == "move":
                if s["speed"] != cur:
                    v, a = speeds[s["speed"]]
                    dType.SetPTPCoordinateParams(api, v, v, a, a, isQueued=1)
                    jv = 40 if s["speed"] == "fast" else 20
                    dType.SetPTPJointParams(api, jv, jv, jv, jv, jv, jv, jv, jv, isQueued=1)
                    cur = s["speed"]
                x, y, z = s["xyz"]
                last = dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, x, y, z, r, isQueued=1)[0]
            elif s["type"] == "suck":
                last = dType.SetEndEffectorSuctionCup(api, 1, 1 if s["on"] else 0, isQueued=1)[0]
            else:
                last = dType.SetWAITCmd(api, int(s["seconds"] * 1000), isQueued=1)[0]
        px, py, pz = plan["park"]
        last = dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, px, py, pz, r, isQueued=1)[0]
        dType.SetQueuedCmdStartExec(api)
        try:
            while last > dType.GetQueuedCmdCurrentIndex(api)[0]:
                dType.dSleep(100)
        except KeyboardInterrupt:
            dType.SetQueuedCmdForceStopExec(api)
            dType.SetQueuedCmdClear(api)
            print("STOPPED (suction left as it was)")
            return 130
        dType.SetQueuedCmdStopExec(api)
        print("done; arm parked")
        return 0
    finally:
        dType.DisconnectDobot(api)


if __name__ == "__main__":
    sys.exit(main())
