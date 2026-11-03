#!/usr/bin/env python3
"""
dobot_teleop.py - drive a Dobot Magician and its suction cup from the keyboard (Ubuntu).

Stand-alone: not part of the pick-and-place project. Needs only Python 3 and pyserial:
    sudo apt install python3-serial          (or: pip install pyserial)

    python3 dobot_teleop.py                  find the Dobot on USB and start
    python3 dobot_teleop.py --port /dev/ttyUSB0
    python3 dobot_teleop.py --fake           try the keys with a pretend robot (no hardware)

Keys (shown on screen too)
    W / S  or  Up / Down        X  + / -   (away from / towards the base)
    A / D  or  Left / Right     Y  + / -   (left / right, seen from behind the robot)
    R / F  or  PgUp / PgDn      Z  up / down
    Z / C                       rotate the cup (R) - / +
    Space                       suction on / off
    1 2 3 4                     step size 1 / 5 / 10 / 25 (mm, or degrees in joint mode)
    - / +                       slower / faster
    T                           Cartesian (X Y Z R) <-> joint mode (J1 J2 J3 J4):
                                  A/D swing the base, W/S lean, R/F tip up/down, Z/C rotate the cup
    O                           go to the ready pose (200, 0, 50)
    H                           home (asks first; the arm swings to its limit switches)
    V                           set a floor at the current height (no move goes below it); V again clears it
    P                           save the current pose to dobot_points.csv
    X                           STOP (cancel every queued move)
    L                           clear alarms
    Q                           quit

Holding a key keeps moving; releasing it stops within one step. Close DobotStudio and anything else that
uses the robot's USB port first (only one program can own it).
"""
import argparse
import curses
import math
import os
import struct
import sys
import threading
import time

# ---------------------------------------------------------------------------------------------------------
# Settings (edit these if you like)
# ---------------------------------------------------------------------------------------------------------
BAUD = 115200
READY_POSE = (200.0, 0.0, 50.0, 0.0)        # X Y Z R the O key goes to
SPEEDS = [10, 20, 30, 50, 80, 120]          # mm/s (and deg/s in joint mode); - / + steps through these
START_SPEED_INDEX = 2                        # 30
STEPS = {"1": 1.0, "2": 5.0, "3": 10.0, "4": 25.0}
START_STEP = 5.0
# Reach limits checked before every Cartesian move (Dobot frame, mm). The robot also refuses unreachable
# points itself (it raises an alarm); these just catch them earlier with a clear message.
RADIUS_MIN, RADIUS_MAX = 120.0, 320.0
Z_MIN, Z_MAX = -80.0, 160.0
R_MIN, R_MAX = -135.0, 135.0
JOINT_LIMITS = [(-90.0, 90.0), (0.0, 85.0), (-10.0, 90.0), (-135.0, 135.0)]   # J1..J4, degrees
MAX_PENDING = 1                              # queued moves allowed ahead of the robot (keeps stops short)
POINTS_FILE = "dobot_points.csv"

# ---------------------------------------------------------------------------------------------------------
# Dobot Magician serial protocol: AA AA | len | id | ctrl | params | checksum
# ---------------------------------------------------------------------------------------------------------
GET_POSE, GET_ALARMS, CLEAR_ALARMS, SET_HOME = 10, 20, 21, 31
SUCTION_CUP, PTP_JOINT_PARAMS, PTP_COORD_PARAMS, PTP_COMMON_PARAMS, PTP_CMD = 62, 80, 81, 83, 84
Q_START, Q_FORCE_STOP, Q_CLEAR, Q_INDEX = 240, 242, 245, 246
MOVJ_XYZ, MOVL_XYZ, MOVJ_ANGLE = 1, 2, 4


class DobotError(RuntimeError):
    pass


def packet(cmd_id, ctrl, params=b""):
    body = bytes([cmd_id, ctrl]) + params
    return b"\xAA\xAA" + bytes([len(body)]) + body + bytes([(-sum(body)) & 0xFF])


def find_port():
    from serial.tools import list_ports
    ports = list(list_ports.comports())
    for p in ports:                                   # the Magician's USB chip (Silicon Labs CP210x)
        if p.vid == 0x10C4 or "CP210" in (p.description or ""):
            return p.device
    for p in ports:
        if "ttyUSB" in p.device or "ttyACM" in p.device:
            return p.device
    return None


class Dobot:
    def __init__(self, port=None, serial_obj=None, timeout=0.3):
        self.timeout = timeout
        self.lock = threading.Lock()
        if serial_obj is None:
            try:
                import serial
            except ImportError:
                raise DobotError("pyserial is missing: sudo apt install python3-serial   (or pip install pyserial)")
            port = port or find_port()
            if not port:
                raise DobotError("no Dobot found on USB. Is it switched on and plugged in? (ls /dev/ttyUSB*)")
            try:
                serial_obj = serial.Serial(port, BAUD, timeout=timeout)
            except Exception as e:
                msg = str(e)
                if "ermission" in msg:
                    msg += "\n  -> sudo usermod -aG dialout $USER   then log out and back in"
                elif "busy" in msg.lower() or "ock" in msg:
                    msg += "\n  -> another program has the port (DobotStudio, the robot bridge, an old script)"
                raise DobotError(f"cannot open {port}: {msg}")
        self.ser, self.port = serial_obj, port or "fake"
        time.sleep(0.05)
        try:
            self.ser.reset_input_buffer()
        except Exception:
            pass
        self.pose()                                   # proves the robot answers
        self._x(Q_START, True, False)

    def _read(self):
        end, prev = time.time() + self.timeout * 2, b""
        while time.time() < end:
            b = self.ser.read(1)
            if not b:
                continue
            if prev == b"\xAA" and b == b"\xAA":
                n = self.ser.read(1)
                if not n:
                    break
                body, chk = self.ser.read(n[0]), self.ser.read(1)
                if len(body) == n[0] and chk and (sum(body) + chk[0]) & 0xFF == 0:
                    return body[0], body[2:]
                break
            prev = b
        raise DobotError("no reply from the Dobot (power? cable? another program on the port?)")

    def _x(self, cmd, rw, queued, params=b""):
        ctrl = (1 if rw else 0) | (2 if queued else 0)
        with self.lock:
            for attempt in range(3):
                try:
                    self.ser.reset_input_buffer()     # drop any stale bytes so the reply matches
                except Exception:
                    pass
                self.ser.write(packet(cmd, ctrl, params))
                try:
                    rid, data = self._read()
                except DobotError:
                    if attempt == 2:
                        raise
                    continue
                if rid == cmd:
                    return data
        raise DobotError(f"unexpected reply to command {cmd}")

    def _queued(self, cmd, params):
        return struct.unpack("<Q", self._x(cmd, True, True, params)[:8])[0]

    # reading
    def pose(self):
        """[x, y, z, r, j1, j2, j3, j4]"""
        return list(struct.unpack("<8f", self._x(GET_POSE, False, False)[:32]))

    def done_index(self):
        return struct.unpack("<Q", self._x(Q_INDEX, False, False)[:8])[0]

    def alarms(self):
        d = self._x(GET_ALARMS, False, False)
        return [i * 8 + b for i, byte in enumerate(d) for b in range(8) if byte >> b & 1]

    # commands
    def move_xyz(self, x, y, z, r, linear=True):
        return self._queued(PTP_CMD, struct.pack("<B4f", MOVL_XYZ if linear else MOVJ_XYZ, x, y, z, r))

    def move_joints(self, j1, j2, j3, j4):
        return self._queued(PTP_CMD, struct.pack("<B4f", MOVJ_ANGLE, j1, j2, j3, j4))

    def suction(self, on):
        return self._queued(SUCTION_CUP, struct.pack("<BB", 1, 1 if on else 0))

    def speed(self, v):
        self._x(PTP_COORD_PARAMS, True, False, struct.pack("<4f", v, v, v, v))
        self._x(PTP_JOINT_PARAMS, True, False, struct.pack("<8f", *([v] * 8)))
        self._x(PTP_COMMON_PARAMS, True, False, struct.pack("<2f", 100.0, 100.0))

    def home(self):
        return self._queued(SET_HOME, struct.pack("<I", 0))

    def stop(self):
        self._x(Q_FORCE_STOP, True, False)
        self._x(Q_CLEAR, True, False)
        self._x(Q_START, True, False)

    def clear_alarms(self):
        self._x(CLEAR_ALARMS, True, False)

    def close(self):
        try:
            self.ser.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------------------------------------
# Pretend robot for --fake (behaves like the real protocol, moves instantly)
# ---------------------------------------------------------------------------------------------------------
class FakeSerial:
    def __init__(self):
        self.buf, self.queue, self.next, self.done = bytearray(), [], 0, 0
        self.p = [200.0, 0.0, 50.0, 0.0, *self._ik(200.0, 0.0, 50.0), 0.0]
        self.last_step = time.time()

    def reset_input_buffer(self):
        self.buf.clear()

    def _ik(self, x, y, z):
        """rough Magician geometry so joint mode shows believable numbers"""
        j1 = math.degrees(math.atan2(y, x))
        rr, zz, a, b = math.hypot(x, y) - 60.0, z, 135.0, 147.0
        d = min(max((rr * rr + zz * zz - a * a - b * b) / (2 * a * b), -1.0), 1.0)
        q3 = math.acos(d)
        j2 = 90.0 - math.degrees(math.atan2(zz, rr) + math.atan2(b * math.sin(q3), a + b * math.cos(q3)))
        return j1, j2, math.degrees(q3) - 90.0 + j2

    def _fk(self, j1, j2, j3):
        a, b = 135.0, 147.0
        rr = 60.0 + a * math.sin(math.radians(j2)) + b * math.cos(math.radians(j3))
        z = a * math.cos(math.radians(j2)) - b * math.sin(math.radians(j3))
        return rr * math.cos(math.radians(j1)), rr * math.sin(math.radians(j1)), z

    def write(self, data):
        n = data[2]
        cid, ctrl, prm = data[3], data[4], bytes(data[5:3 + n])
        out = b""
        if cid == GET_POSE:
            if self.queue and time.time() - self.last_step > 0.15:     # one queued move per 0.15 s
                self.last_step = time.time()
                kind, a, self.done = self.queue.pop(0)
                if kind == "xyz":
                    self.p[:4] = a
                    self.p[4:7] = self._ik(*a[:3])
                    self.p[7] = a[3]
                elif kind == "joint":
                    self.p[4:8] = a
                    self.p[:3] = self._fk(*a[:3])
                    self.p[3] = a[3]
                elif kind == "home":
                    self.p = [200.0, 0.0, 50.0, 0.0] + [0.0, *self._ik(200, 0, 50)[1:], 0.0]
            out = struct.pack("<8f", *self.p)
        elif cid == Q_INDEX:
            out = struct.pack("<Q", self.done if self.queue else self.next)
        elif cid == GET_ALARMS:
            out = bytes(16)
        elif cid in (Q_FORCE_STOP, Q_CLEAR):
            self.queue.clear()
            self.done = self.next
        elif ctrl & 2:
            self.next += 1
            if cid == PTP_CMD:
                mode, *a = struct.unpack("<B4f", prm)
                self.queue.append(("joint" if mode == MOVJ_ANGLE else "xyz", list(a), self.next))
            elif cid == SET_HOME:
                self.queue.append(("home", None, self.next))
            else:
                self.queue.append(("other", None, self.next))
            out = struct.pack("<Q", self.next)
        self.buf += packet(cid, ctrl, out)
        return len(data)

    def read(self, n):
        out = bytes(self.buf[:n])
        del self.buf[:n]
        return out

    def close(self):
        pass


# ---------------------------------------------------------------------------------------------------------
# Teleop logic (no curses here, so it can be tested)
# ---------------------------------------------------------------------------------------------------------
MOVE_KEYS = {  # key -> (axis index, sign)
    "w": (0, +1), "KEY_UP": (0, +1), "s": (0, -1), "KEY_DOWN": (0, -1),
    "a": (1, +1), "KEY_LEFT": (1, +1), "d": (1, -1), "KEY_RIGHT": (1, -1),
    "r": (2, +1), "KEY_PPAGE": (2, +1), "f": (2, -1), "KEY_NPAGE": (2, -1),
    "z": (3, -1), "c": (3, +1),
}


class Teleop:
    def __init__(self, bot, points_file=POINTS_FILE):
        self.bot = bot
        self.points_file = points_file
        self.joint_mode = False
        self.step = START_STEP
        self.speed_i = START_SPEED_INDEX
        self.suck = False
        self.floor = None
        self.pose = bot.pose()
        self.target = None            # last commanded [x,y,z,r] or [j1..j4]; None = follow the robot
        self.last_index = bot.done_index()
        self.done = self.last_index
        self.alarm_list = []
        self.msg = "ready"
        self.confirm = None           # pending yes/no question: (text, action)
        bot.speed(SPEEDS[self.speed_i])

    # ---- state ------------------------------------------------------------------------------------------
    def refresh(self):
        self.pose = self.bot.pose()
        self.done = self.bot.done_index()
        if self.idle():
            self.target = None        # robot caught up: next move starts from where it really is
        self.alarm_list = self.bot.alarms()
        # joint moves cannot be checked against the floor in advance, so watch the real height instead
        if self.joint_mode and self.floor is not None and self.pose[2] < self.floor - 2.0 and not self.idle():
            self.stop()
            self.msg = f"STOPPED: tip went below the floor (z {self.pose[2]:.1f} < {self.floor:.1f})"

    def idle(self):
        return self.done >= self.last_index

    def pending(self):
        return max(0, self.last_index - self.done)

    def base(self):
        if self.target is not None:
            return list(self.target)
        return list(self.pose[4:8] if self.joint_mode else self.pose[:4])

    # ---- checks -----------------------------------------------------------------------------------------
    def check_xyz(self, t):
        x, y, z, r = t
        rad = math.hypot(x, y)
        if not RADIUS_MIN <= rad <= RADIUS_MAX:
            return f"out of reach: {rad:.0f} mm from the base (allowed {RADIUS_MIN:.0f}-{RADIUS_MAX:.0f})"
        if not Z_MIN <= z <= Z_MAX:
            return f"height {z:.0f} mm outside {Z_MIN:.0f}..{Z_MAX:.0f}"
        if self.floor is not None and z < self.floor - 0.01:
            return f"floor: not going below z = {self.floor:.1f} (V clears it)"
        if not R_MIN <= r <= R_MAX:
            return f"rotation {r:.0f} deg outside {R_MIN:.0f}..{R_MAX:.0f}"
        return None

    def check_joints(self, t):
        for i, (v, (lo, hi)) in enumerate(zip(t, JOINT_LIMITS)):
            if not lo <= v <= hi:
                return f"J{i + 1} = {v:.0f} deg outside {lo:.0f}..{hi:.0f}"
        return None

    # ---- actions ----------------------------------------------------------------------------------------
    def jog(self, axis, sign):
        if self.pending() >= MAX_PENDING + 1:
            return                     # robot still busy: drop this key repeat (so release stops quickly)
        if self.joint_mode:            # A/D swing the base (J1), W/S lean (J2), R/F raise/lower (J3), Z/C cup (J4)
            axis, sign = {0: (1, sign), 1: (0, sign), 2: (2, -sign), 3: (3, sign)}[axis]
        t = self.base()
        t[axis] += sign * self.step
        if self.joint_mode:
            err = self.check_joints(t)
            if err:
                self.msg = "refused: " + err
                return
            self.last_index = self.bot.move_joints(*t)
        else:
            err = self.check_xyz(t)
            if err:
                self.msg = "refused: " + err
                return
            self.last_index = self.bot.move_xyz(*t)
        self.target = t
        names = ["J1", "J2", "J3", "J4"] if self.joint_mode else ["X", "Y", "Z", "R"]
        self.msg = f"{names[axis]} {'+' if sign > 0 else '-'}{self.step:g} -> {t[axis]:.1f}"

    def toggle_suction(self):
        self.suck = not self.suck
        self.last_index = self.bot.suction(self.suck)
        self.msg = "suction ON" if self.suck else "suction OFF"

    def set_step(self, key):
        self.step = STEPS[key]
        self.msg = f"step {self.step:g} {'deg' if self.joint_mode else 'mm'}"

    def change_speed(self, d):
        self.speed_i = min(max(self.speed_i + d, 0), len(SPEEDS) - 1)
        self.bot.speed(SPEEDS[self.speed_i])
        self.msg = f"speed {SPEEDS[self.speed_i]}"

    def toggle_mode(self):
        if not self.idle():
            self.msg = "wait until the arm stops before switching mode"
            return
        self.joint_mode = not self.joint_mode
        self.target = None
        self.msg = "JOINT mode (J1 J2 J3 J4)" if self.joint_mode else "CARTESIAN mode (X Y Z R)"

    def ready(self):
        err = self.check_xyz(list(READY_POSE))
        if err:
            self.msg = "refused: " + err
            return
        self.bot.stop()
        self.last_index = self.bot.move_xyz(*READY_POSE, linear=False)
        self.target = list(READY_POSE) if not self.joint_mode else None
        self.msg = "going to the ready pose"

    def ask_home(self):
        self.confirm = ("HOME: the arm swings to its limit switches. Area clear? (y/n)", self._home, None)

    def _home(self):
        self.bot.stop()
        self.last_index = self.bot.home()
        self.target = None
        self.msg = "homing (takes ~20 s)"

    def toggle_floor(self):
        if self.floor is None:
            self.floor = self.pose[2]
            self.msg = f"floor set at z = {self.floor:.1f} mm"
        else:
            self.floor = None
            self.msg = "floor cleared"

    def save_point(self):
        new = not os.path.exists(self.points_file)
        with open(self.points_file, "a") as f:
            if new:
                f.write("x,y,z,r,j1,j2,j3,j4,suction\n")
            f.write(",".join(f"{v:.2f}" for v in self.pose) + f",{int(self.suck)}\n")
        x, y, z, r = self.pose[:4]
        self.msg = f"saved ({x:.1f}, {y:.1f}, {z:.1f}, {r:.1f}) to {self.points_file}"

    def stop(self):
        self.bot.stop()
        self.last_index = self.bot.done_index()
        self.target = None
        self.msg = "STOPPED (queue cleared)"

    def clear_alarms(self):
        self.bot.clear_alarms()
        self.msg = "alarms cleared"

    def ask_quit(self):
        if self.suck:
            self.confirm = ("Suction is ON. Turn it off before quitting? (y = off, n = leave on)",
                            self._quit_off, lambda: "quit")
            return None
        return "quit"

    def _quit_off(self):
        self.bot.suction(False)
        self.suck = False
        return "quit"

    def key(self, k):
        """Handle one key name; returns 'quit' to exit."""
        if self.confirm:
            text, yes, no = self.confirm
            self.confirm = None
            if k in ("y", "Y"):
                return yes()
            self.msg = "cancelled"
            return no() if no else None
        kl = k.lower() if len(k) == 1 else k
        if kl in MOVE_KEYS:
            self.jog(*MOVE_KEYS[kl])
        elif k == " ":
            self.toggle_suction()
        elif k in STEPS:
            self.set_step(k)
        elif k in ("-", "_"):
            self.change_speed(-1)
        elif k in ("+", "="):
            self.change_speed(+1)
        elif kl == "t":
            self.toggle_mode()
        elif kl == "o":
            self.ready()
        elif kl == "h":
            self.ask_home()
        elif kl == "v":
            self.toggle_floor()
        elif kl == "p":
            self.save_point()
        elif kl == "x":
            self.stop()
        elif kl == "l":
            self.clear_alarms()
        elif kl == "q":
            return self.ask_quit()
        return None


# ---------------------------------------------------------------------------------------------------------
# Screen
# ---------------------------------------------------------------------------------------------------------
HELP = [
    "W/S or Up/Down: X +/-     A/D or Left/Right: Y +/-     R/F or PgUp/PgDn: Z up/down     Z/C: rotate",
    "Space: suction   1-4: step 1/5/10/25   -/+: speed   T: joint/cartesian   O: ready pose   H: home",
    "V: floor at this height   P: save point   X: STOP   L: clear alarms   Q: quit",
]
HELP_JOINT = "Joint mode: A/D swing base (J1)   W/S lean forward/back (J2)   R/F tip up/down (J3)   Z/C cup (J4)"


def draw(scr, tp):
    scr.erase()
    p = tp.pose

    def put(y, text, attr=0):
        try:
            scr.addstr(y, 0, text[: max(0, curses.COLS - 1)], attr)
        except curses.error:
            pass

    mode = "JOINT" if tp.joint_mode else "CARTESIAN"
    unit = "deg" if tp.joint_mode else "mm"
    put(0, f" DOBOT TELEOP   port {tp.bot.port}   mode {mode}   step {tp.step:g} {unit}   "
           f"speed {SPEEDS[tp.speed_i]}", curses.A_REVERSE)
    put(2, f" Tip     X {p[0]:7.1f}   Y {p[1]:7.1f}   Z {p[2]:7.1f}   R {p[3]:7.1f}   "
           f"(radius {math.hypot(p[0], p[1]):5.1f} mm)")
    put(3, f" Joints  J1 {p[4]:6.1f}   J2 {p[5]:6.1f}   J3 {p[6]:6.1f}   J4 {p[7]:6.1f}")
    suck_attr = curses.A_BOLD | (curses.color_pair(2) if curses.has_colors() else 0)
    put(5, f" Suction {'ON ' if tp.suck else 'off'}", suck_attr if tp.suck else 0)
    put(6, f" Moving  {'yes (' + str(tp.pending()) + ' queued)' if not tp.idle() else 'no'}")
    put(7, f" Floor   {'z >= %.1f mm' % tp.floor if tp.floor is not None else 'none'}")
    al = tp.alarm_list
    put(8, f" Alarms  {'none' if not al else str(al) + '  -> press L to clear (often: point out of reach)'}",
        (curses.A_BOLD | (curses.color_pair(1) if curses.has_colors() else 0)) if al else 0)
    for i, h in enumerate(HELP):
        put(10 + i, " " + h, curses.A_DIM)
    if tp.joint_mode:
        put(13, " " + HELP_JOINT, curses.A_BOLD)
    if tp.confirm:
        put(14, " " + tp.confirm[0], curses.A_BOLD | (curses.color_pair(1) if curses.has_colors() else 0))
    else:
        put(14, " > " + tp.msg)
    scr.refresh()


def run(scr, bot):
    try:
        curses.curs_set(0)
    except curses.error:
        pass
    if curses.has_colors():
        try:
            curses.start_color()
            try:
                curses.use_default_colors()
                bg = -1
            except curses.error:
                bg = curses.COLOR_BLACK
            curses.init_pair(1, curses.COLOR_RED, bg)
            curses.init_pair(2, curses.COLOR_GREEN, bg)
        except curses.error:
            pass                      # plain text if the terminal has no colours
    scr.keypad(True)
    scr.timeout(100)
    tp = Teleop(bot)
    last = 0.0
    while True:
        if time.time() - last > 0.2:
            try:
                tp.refresh()
            except DobotError as e:
                tp.msg = f"lost contact: {e}"
            last = time.time()
        draw(scr, tp)
        try:
            k = scr.get_wch()
        except curses.error:
            continue
        except KeyboardInterrupt:
            k = "x"
        name = curses.keyname(k).decode() if isinstance(k, int) else k
        if name == "\x03":              # Ctrl+C
            tp.stop()
            name = "q"
        try:
            if tp.key(name) == "quit":
                return tp
        except DobotError as e:
            tp.msg = f"robot error: {e}"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Keyboard control of a Dobot Magician + suction cup")
    ap.add_argument("--port", help="serial port, e.g. /dev/ttyUSB0 (default: find it)")
    ap.add_argument("--fake", action="store_true", help="pretend robot, to try the keys without hardware")
    a = ap.parse_args(argv)
    try:
        bot = Dobot(serial_obj=FakeSerial()) if a.fake else Dobot(port=a.port)
    except DobotError as e:
        print(f"[teleop] {e}")
        return 1
    print(f"[teleop] connected on {bot.port}")
    try:
        curses.wrapper(run, bot)
    except DobotError as e:
        print(f"[teleop] {e}")
    finally:
        try:
            bot.stop()
        except Exception:
            pass
        bot.close()
    print("[teleop] stopped. Points you saved (P) are in", os.path.abspath(POINTS_FILE))
    return 0


if __name__ == "__main__":
    sys.exit(main())
