#!/usr/bin/env python3
"""
dobot_move_diag.py - why does the Dobot accept move commands but not move?

Talks to the robot DIRECTLY over USB (stop the bridge / Isaac Sim first: only one program can use the port).
Reads the alarms, then tries three tiny moves, reading the pose and the alarms after each:
  A  joint move: base joint +5 deg            (MOVJ_ANGLE)
  B  10 mm up, joint-interpolated             (MOVJ_XYZ)
  C  10 mm back down, straight line           (MOVL_XYZ)
Each asks first. Hand on the power switch.

    ~/dobot_real/venv/bin/python src/dobot_move_diag.py
"""
import os
import struct
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dobot_link as L  # noqa: E402

# Dobot Magician alarm bit numbers -> meaning (Communication Protocol, alarm list)
ALARMS = {
    0x00: "common: reset", 0x01: "common: undefined instruction", 0x02: "common: file system",
    0x03: "common: MCU/FPGA communication", 0x04: "common: angle sensor",
    0x10: "planning: inverse kinematics has no solution (target out of reach)",
    0x11: "planning: inverse kinematics out of joint limits", 0x12: "planning: data repeated",
    0x13: "planning: arc input parameters", 0x15: "planning: JUMP parameters",
    0x20: "kinematics: inverse kinematics has no solution", 0x21: "kinematics: out of joint limits",
    0x40: "limit: joint 1 positive", 0x41: "limit: joint 1 negative", 0x42: "limit: joint 2 positive",
    0x43: "limit: joint 2 negative", 0x44: "limit: joint 3 positive", 0x45: "limit: joint 3 negative",
    0x46: "limit: joint 4 positive", 0x47: "limit: joint 4 negative",
    0x48: "limit: joint 2 and 3 parallelogram (upper/lower arm angle) positive",
    0x49: "limit: joint 2 and 3 parallelogram negative",
    0x50: "lose step: joint 1", 0x51: "lose step: joint 2", 0x52: "lose step: joint 3", 0x53: "lose step: joint 4",
}


def show(link, label):
    p = link.get_pose()
    al = link.alarms()
    names = [ALARMS.get(a, f"alarm 0x{a:02X}") for a in al]
    print(f"  {label}: x {p[0]:.1f} y {p[1]:.1f} z {p[2]:.1f} r {p[3]:.1f} | joints "
          f"{p[4]:.1f} {p[5]:.1f} {p[6]:.1f} {p[7]:.1f} | queue {link.queued_index()} | alarms: "
          f"{names if names else 'none'}")
    return p, al


def ask(msg):
    return input(f"\n{msg} Type yes (anything else skips): ").strip().lower() == "yes"


def main():
    try:
        link = L.DobotLink()
    except Exception as e:
        print(f"cannot open the robot: {e}\nStop the bridge first: close Isaac Sim / Terminal 1, then "
              f"pkill -f 'dobot_link.py serve'")
        return 1
    try:
        print(f"robot on {link.port}")
        p0, al = show(link, "start")
        link.clear_alarms()
        link.stop_now()                       # clear any stuck queue, restart execution
        show(link, "after clearing alarms + queue")
        link.speed(30, 30)
        link._xfer(L.CMD["PTP_JOINT_PARAMS"], True, False, struct.pack("<8f", *([40.0] * 4 + [40.0] * 4)))
        print("  speeds set (Cartesian 30 mm/s, joints 40 deg/s)")
        results = {}
        if ask("A: turn the base joint +5 deg?"):
            p, _ = show(link, "before A")
            link.ptp(p[4] + 5.0, p[5], p[6], p[7], mode="MOVJ_ANGLE", wait=True)
            time.sleep(0.5)
            q, al = show(link, "after A")
            results["A joint move"] = abs(q[4] - p[4]) > 2.0
            if results["A joint move"]:
                link.ptp(p[4], p[5], p[6], p[7], mode="MOVJ_ANGLE", wait=True)
        if ask("B: 10 mm straight up (joint-interpolated)?"):
            p, _ = show(link, "before B")
            link.ptp(p[0], p[1], p[2] + 10.0, p[3], mode="MOVJ_XYZ", wait=True)
            time.sleep(0.5)
            q, al = show(link, "after B")
            results["B MOVJ_XYZ"] = abs(q[2] - p[2]) > 5.0
        if ask("C: 10 mm back down (straight line)?"):
            p, _ = show(link, "before C")
            link.ptp(p[0], p[1], p[2] - 10.0, p[3], mode="MOVL_XYZ", wait=True)
            time.sleep(0.5)
            q, al = show(link, "after C")
            results["C MOVL_XYZ"] = abs(q[2] - p[2]) > 5.0
        print("\nRESULT:", ", ".join(f"{k}: {'MOVED' if v else 'did NOT move'}" for k, v in results.items()))
        print("Send these lines to Claude.")
        return 0
    finally:
        link.close()


if __name__ == "__main__":
    sys.exit(main())
