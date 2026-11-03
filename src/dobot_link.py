#!/usr/bin/env python3
"""
Dobot Magician over USB serial on Ubuntu, in pure Python (no Windows DLLs), plus a local bridge so that Isaac Sim
and the scripts can share the robot.

It speaks the Dobot Magician communication protocol directly (115200 baud, 8N1), the same protocol DobotDll.dll
uses under DobotDllType.py. The operations match what your Windows controller uses:

    DobotDllType (Windows)                         dobot_link (Ubuntu)
    dType.ConnectDobot(api, '', 115200)        ->  DobotLink()              (finds /dev/ttyUSB0 itself)
    dType.GetPose(api)                         ->  link.get_pose()          [x, y, z, r, j1, j2, j3, j4]
    dType.SetPTPCmd(api, PTPMOVJXYZMode, ...)  ->  link.ptp(x, y, z, r, mode="MOVJ_XYZ")   (returns the queue index)
    dType.SetPTPCmd(api, PTPMOVLXYZMode, ...)  ->  link.ptp(x, y, z, r, mode="MOVL_XYZ")   (straight line)
    dType.SetEndEffectorSuctionCup(...)        ->  link.suction(True / False)
    dType.SetQueuedCmdStartExec(api)           ->  link.start_exec()
    dType.SetQueuedCmdForceStopExec(api)       ->  link.force_stop()
    dType.SetQueuedCmdClear(api)               ->  link.clear_queue()
    dType.GetQueuedCmdCurrentIndex(api)        ->  link.queued_index()      (link.wait(index) waits for a move)
    dType.SetPTPCoordinateParams / CommonParams->  link.speed(mm_per_s, mm_per_s2)
    dType.SetHOMECmd / GetAlarmsState / Clear..->  link.home(), link.alarms(), link.clear_alarms()

Only one program can own the serial port (as with DobotStudio on Windows). Run the bridge and everything else goes
through it:

    python3 dobot_link.py ports                  list serial ports (the Dobot is usually /dev/ttyUSB0, Silicon Labs CP210x)
    python3 dobot_link.py pose                   connect directly, print the pose, disconnect
    python3 dobot_link.py serve                  bridge on http://127.0.0.1:5055 (owns the port; Ctrl+C ends it)
    python3 dobot_link.py jog --dz 10            small relative test move (asks first)
    python3 dobot_link.py home                   homing after power-on (replaces DobotStudio's Home; asks first)

Bridge API (JSON, localhost only): GET /state; POST /ptp {"points": [[x,y,z,r],...], "mode": "MOVL_XYZ", "wait": true},
/suck {"on": true}, /speed {"v": 40, "a": 40}, /wait {"index": n}, /stop, /clear_alarms, /home {"confirm": true}.

Backends: "serial" (default, this file, needs pyserial) or "dll" (the official DobotDllType.py with the Linux
libDobotDll.so from Dobot's SDK, if you have it). See 05_Dobot_On_Ubuntu.md.
"""

import argparse
import json
import os
import struct
import sys
import threading
import time

LINK = {
    "PORT": None,                 # None = auto (Silicon Labs CP210x first, then any ttyUSB / ttyACM)
    "BAUD": 115200,
    "TIMEOUT_S": 1.0,
    "BACKEND": "teleop",          # "teleop" (the protocol code of dobot_teleop.py, proven on this robot),
                                  # "serial" (this file's own) or "dll" (Dobot's official library)
    "DLL_DIR": os.path.expanduser("~/dobot_sdk"),     # folder with DobotDllType.py + libDobotDll.so ("dll" backend)
    "BRIDGE_HOST": "127.0.0.1",   # never expose the robot on the network
    "BRIDGE_PORT": 5055,
    "END_EFFECTOR_BIAS": None,    # e.g. (59.7, 0, 0) to make pose() report the suction tip; None = leave as set
    "WAIT_TIMEOUT_S": 60.0,
}

# Dobot Magician protocol command ids (Communication Protocol V1.1.x)
CMD = {"GET_POSE": 10, "GET_ALARMS": 20, "CLEAR_ALARMS": 21, "SET_HOME": 31, "END_EFFECTOR_PARAMS": 60,
       "SUCTION_CUP": 62, "PTP_JOINT_PARAMS": 80, "PTP_COORD_PARAMS": 81, "PTP_JUMP_PARAMS": 82,
       "PTP_COMMON_PARAMS": 83, "PTP_CMD": 84, "QUEUE_START": 240, "QUEUE_STOP": 241, "QUEUE_FORCE_STOP": 242,
       "QUEUE_CLEAR": 245, "QUEUE_INDEX": 246}
PTP_MODE = {"JUMP_XYZ": 0, "MOVJ_XYZ": 1, "MOVL_XYZ": 2, "JUMP_ANGLE": 3, "MOVJ_ANGLE": 4, "MOVL_ANGLE": 5,
            "MOVJ_INC": 6, "MOVL_INC": 7, "MOVJ_XYZ_INC": 8, "JUMP_MOVL_XYZ": 9}


def _say(msg):
    print(f"[DobotLink] {msg}", flush=True)


def packet(cmd_id, ctrl, params=b""):
    """AA AA | len | id | ctrl | params | checksum, with len = 2 + len(params), checksum = -(id+ctrl+params) mod 256."""
    body = bytes([cmd_id, ctrl]) + params
    return b"\xAA\xAA" + bytes([len(body)]) + body + bytes([(-sum(body)) & 0xFF])


class DobotError(RuntimeError):
    pass


# =============================================================================
# Serial backend (pure Python)
# =============================================================================
def find_port():
    from serial.tools import list_ports
    ports = list(list_ports.comports())
    for p in ports:                                   # the Magician's USB-UART chip
        if (p.vid == 0x10C4) or "CP210" in (p.description or "") or "Silicon Labs" in (p.manufacturer or ""):
            return p.device
    for p in ports:
        if "ttyUSB" in p.device or "ttyACM" in p.device:
            return p.device
    return None


class DobotLink:
    def __init__(self, port=None, cfg=LINK, serial_obj=None):
        self.cfg = cfg
        self.lock = threading.RLock()
        if serial_obj is None:
            import serial
            port = port or cfg["PORT"] or find_port()
            if not port:
                raise DobotError("no Dobot serial port found: is it powered and plugged in? (`dobot_link.py ports`)")
            try:
                serial_obj = serial.Serial(port, cfg["BAUD"], timeout=cfg["TIMEOUT_S"], bytesize=8, parity="N",
                                           stopbits=1)
            except Exception as e:
                hint = " Add yourself to 'dialout' (and log out/in)." if "ermission" in str(e) else \
                    " Is another program (the bridge, an old script) holding the port?" if "busy" in str(e).lower() \
                    else ""
                raise DobotError(f"cannot open {port}: {e}.{hint}")
        self.ser = serial_obj
        self.port = port or getattr(serial_obj, "port", "?")
        time.sleep(0.05)
        try:
            self.ser.reset_input_buffer()
        except Exception:
            pass
        self.get_pose()                               # proves the robot answers
        # Start clean, as DobotStudio does: empty queue running, every PTP speed set (joint moves use the joint
        # parameters, which a fresh robot may hold at 0).
        try:
            self.force_stop()
            self.clear_queue()
        except DobotError:
            pass
        self.start_exec()
        try:
            self.joint_speed(cfg.get("JOINT_SPEED_DEG_S", 40.0))
            self.speed(cfg.get("INIT_SPEED_MM_S", 40.0), cfg.get("INIT_SPEED_MM_S", 40.0))
        except DobotError:
            pass
        if cfg.get("END_EFFECTOR_BIAS") is not None:
            self.end_effector_params(*cfg["END_EFFECTOR_BIAS"])

    # ---- protocol --------------------------------------------------------------------------------------
    def _read_packet(self):
        deadline = time.time() + self.cfg["TIMEOUT_S"] * 2
        prev = b""
        while time.time() < deadline:
            b = self.ser.read(1)
            if not b:
                continue
            if prev == b"\xAA" and b == b"\xAA":
                n = self.ser.read(1)
                if not n:
                    break
                n = n[0]
                payload = self.ser.read(n)
                chk = self.ser.read(1)
                if len(payload) != n or not chk:
                    break
                if (sum(payload) + chk[0]) & 0xFF != 0:
                    raise DobotError("checksum error on a reply from the Dobot")
                return payload[0], payload[1], payload[2:]
            prev = b
        raise DobotError("no reply from the Dobot (power, cable, or another program on the port?)")

    def _xfer(self, cmd_id, rw, queued, params=b""):
        ctrl = (1 if rw else 0) | (2 if queued else 0)
        with self.lock:
            # Like the working teleop script: drop stale bytes, send once, then read until THIS command's reply.
            # A queued command (a move) is never re-sent: a second copy would run twice.
            for attempt in range(1 if queued else 3):
                try:
                    self.ser.reset_input_buffer()
                except Exception:
                    pass
                self.ser.write(packet(cmd_id, ctrl, params))
                for _ in range(4):
                    try:
                        rid, _, data = self._read_packet()
                    except DobotError:
                        break
                    if rid == cmd_id:
                        return data
            raise DobotError(f"no reply to command {cmd_id} (power, cable, or another program on the port?)")

    # ---- reading -----------------------------------------------------------------------------------------
    def get_pose(self):
        d = self._xfer(CMD["GET_POSE"], False, False)
        return list(struct.unpack("<8f", d[:32]))

    def queued_index(self):
        return struct.unpack("<Q", self._xfer(CMD["QUEUE_INDEX"], False, False)[:8])[0]

    def alarms(self):
        d = self._xfer(CMD["GET_ALARMS"], False, False)
        return [i * 8 + b for i, byte in enumerate(d) for b in range(8) if byte >> b & 1]

    # ---- queue -------------------------------------------------------------------------------------------
    def start_exec(self):
        self._xfer(CMD["QUEUE_START"], True, False)

    def stop_exec(self):
        self._xfer(CMD["QUEUE_STOP"], True, False)

    def force_stop(self):
        self._xfer(CMD["QUEUE_FORCE_STOP"], True, False)

    def clear_queue(self):
        self._xfer(CMD["QUEUE_CLEAR"], True, False)

    def stop_now(self):
        """Force-stop, empty the queue, and re-arm it so the next command runs. Suction is left as it is."""
        self.force_stop()
        self.clear_queue()
        self.start_exec()

    def wait(self, index, timeout=None):
        t_end = time.time() + (timeout or self.cfg["WAIT_TIMEOUT_S"])
        while time.time() < t_end:
            if self.queued_index() >= index:
                return True
            time.sleep(0.05)
        al = []
        try:
            al = self.alarms()
        except Exception:
            pass
        raise DobotError(f"move {index} did not finish in time" + (f"; robot alarms {al} (clear_alarms after "
                                                                     f"fixing the cause)" if al else ""))

    # ---- commands ----------------------------------------------------------------------------------------
    def _queued_index_reply(self, d):
        return struct.unpack("<Q", d[:8])[0] if len(d) >= 8 else None

    def ptp(self, x, y, z, r=0.0, mode="MOVL_XYZ", wait=False):
        d = self._xfer(CMD["PTP_CMD"], True, True, struct.pack("<B4f", PTP_MODE[mode], x, y, z, r))
        idx = self._queued_index_reply(d)
        if wait and idx is not None:
            self.wait(idx)
        return idx

    def suction(self, on, wait=True):
        d = self._xfer(CMD["SUCTION_CUP"], True, True, struct.pack("<BB", 1, 1 if on else 0))
        idx = self._queued_index_reply(d)
        if wait and idx is not None:
            self.wait(idx)
        return idx

    def speed(self, v, a, ratio=100.0):
        """Cartesian speed (mm/s) and acceleration (mm/s^2) for PTP moves."""
        self._xfer(CMD["PTP_COORD_PARAMS"], True, False, struct.pack("<4f", v, v, a, a))
        # joint limits too (as the teleop script does): with them at 0 the robot "finishes" moves without moving
        self.joint_speed(max(20.0, float(v)), max(20.0, float(a)))
        self._xfer(CMD["PTP_COMMON_PARAMS"], True, False, struct.pack("<2f", ratio, ratio))

    def joint_speed(self, v_deg_s, a_deg_s2=None):
        """Joint velocity / acceleration for joint-interpolated moves (MOVJ_*), all four joints."""
        a = a_deg_s2 if a_deg_s2 is not None else v_deg_s
        self._xfer(CMD["PTP_JOINT_PARAMS"], True, False, struct.pack("<8f", *([float(v_deg_s)] * 4 + [float(a)] * 4)))

    def home(self):
        idx = self._queued_index_reply(self._xfer(CMD["SET_HOME"], True, True, struct.pack("<I", 0)))
        return idx

    def clear_alarms(self):
        self._xfer(CMD["CLEAR_ALARMS"], True, False)

    def end_effector_params(self, x_bias, y_bias=0.0, z_bias=0.0):
        self._xfer(CMD["END_EFFECTOR_PARAMS"], True, False, struct.pack("<3f", x_bias, y_bias, z_bias))

    def close(self):
        try:
            self.ser.close()
        except Exception:
            pass


# =============================================================================
# DLL backend (official DobotDllType.py + libDobotDll.so), same interface
# =============================================================================
class DllLink:
    def __init__(self, port=None, cfg=LINK):
        sys.path.insert(0, cfg["DLL_DIR"])
        import DobotDllType as dType
        self.d, self.cfg = dType, cfg
        self.lock = threading.RLock()
        self.api = dType.load()
        r = dType.ConnectDobot(self.api, port or "", cfg["BAUD"])
        if r[0] != 0:
            raise DobotError(f"ConnectDobot returned {r}")
        self.port = r[1] if len(r) > 1 else "?"
        dType.SetQueuedCmdClear(self.api)
        dType.SetQueuedCmdStartExec(self.api)
        if cfg.get("END_EFFECTOR_BIAS") is not None:
            dType.SetEndEffectorParams(self.api, *cfg["END_EFFECTOR_BIAS"], isQueued=0)

    def get_pose(self):
        return list(self.d.GetPose(self.api))

    def queued_index(self):
        return self.d.GetQueuedCmdCurrentIndex(self.api)[0]

    def alarms(self):
        return list(self.d.GetAlarmsState(self.api)[0]) if hasattr(self.d, "GetAlarmsState") else []

    def start_exec(self):
        self.d.SetQueuedCmdStartExec(self.api)

    def stop_exec(self):
        self.d.SetQueuedCmdStopExec(self.api)

    def force_stop(self):
        self.d.SetQueuedCmdForceStopExec(self.api)

    def clear_queue(self):
        self.d.SetQueuedCmdClear(self.api)

    stop_now = DobotLink.stop_now
    wait = DobotLink.wait

    def ptp(self, x, y, z, r=0.0, mode="MOVL_XYZ", wait=False):
        idx = self.d.SetPTPCmd(self.api, PTP_MODE[mode], x, y, z, r, isQueued=1)[0]
        if wait:
            self.wait(idx)
        return idx

    def suction(self, on, wait=True):
        idx = self.d.SetEndEffectorSuctionCup(self.api, 1, 1 if on else 0, isQueued=1)[0]
        if wait:
            self.wait(idx)
        return idx

    def speed(self, v, a, ratio=100.0):
        self.d.SetPTPCoordinateParams(self.api, v, v, a, a, isQueued=0)
        j, ja = max(20.0, float(v)), max(20.0, float(a))
        self.d.SetPTPJointParams(self.api, j, ja, j, ja, j, ja, j, ja, isQueued=0)
        self.d.SetPTPCommonParams(self.api, ratio, ratio, isQueued=0)

    def home(self):
        return self.d.SetHOMECmd(self.api, temp=0, isQueued=1)[0]

    def clear_alarms(self):
        self.d.ClearAllAlarmsState(self.api)

    def close(self):
        try:
            self.d.DisconnectDobot(self.api)
        except Exception:
            pass


class TeleopLink:
    """DobotLink's interface on top of dobot_teleop.py's Dobot class: the serial code that moves this robot in the
    lab (the teleop keys work where this file's own driver did not)."""

    def __init__(self, port=None, cfg=LINK):
        import importlib.util
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, "dobot_teleop.py")
        if not os.path.isfile(path):
            raise DobotError(f"dobot_teleop.py not found next to dobot_link.py ({path}); set LINK BACKEND to 'serial'")
        spec = importlib.util.spec_from_file_location("dobot_teleop_core", path)
        T = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(T)
        self.T, self.cfg = T, cfg
        self.lock = threading.RLock()
        try:
            self.bot = T.Dobot(port=port or cfg.get("PORT"))
        except T.DobotError as e:
            raise DobotError(str(e))
        self.port = self.bot.port
        self.speed(cfg.get("INIT_SPEED_MM_S", 40.0), cfg.get("INIT_SPEED_MM_S", 40.0))

    def _call(self, fn, *a):
        with self.lock:
            try:
                return fn(*a)
            except self.T.DobotError as e:
                raise DobotError(str(e))

    def get_pose(self):
        return self._call(self.bot.pose)

    def queued_index(self):
        return self._call(self.bot.done_index)

    def alarms(self):
        return self._call(self.bot.alarms)

    def start_exec(self):
        self._call(self.bot._x, self.T.Q_START, True, False)

    def stop_exec(self):
        self._call(self.bot._x, 241, True, False)

    def force_stop(self):
        self._call(self.bot._x, self.T.Q_FORCE_STOP, True, False)

    def clear_queue(self):
        self._call(self.bot._x, self.T.Q_CLEAR, True, False)

    def stop_now(self):
        self._call(self.bot.stop)

    def wait(self, index, timeout=None):
        t_end = time.time() + (timeout or self.cfg["WAIT_TIMEOUT_S"])
        while time.time() < t_end:
            if self.queued_index() >= index:
                return True
            self.get_pose()                   # same polling the teleop loop does
            time.sleep(0.05)
        raise DobotError(f"move {index} did not finish in time; robot alarms {self.alarms()}")

    def ptp(self, x, y, z, r=0.0, mode="MOVL_XYZ", wait=False):
        if mode in ("MOVL_XYZ", "MOVJ_XYZ"):
            idx = self._call(self.bot.move_xyz, x, y, z, r, mode == "MOVL_XYZ")
        elif mode == "MOVJ_ANGLE":
            idx = self._call(self.bot.move_joints, x, y, z, r)
        else:
            idx = self._call(self.bot._queued, self.T.PTP_CMD, struct.pack("<B4f", PTP_MODE[mode], x, y, z, r))
        if wait:
            self.wait(idx)
        return idx

    def suction(self, on, wait=True):
        idx = self._call(self.bot.suction, on)
        if wait:
            self.wait(idx)
        return idx

    def speed(self, v, a=None, ratio=100.0):
        self._call(self.bot.speed, float(v))

    def home(self):
        idx = self._call(self.bot.home)
        self.wait(idx, timeout=90)
        return idx

    def clear_alarms(self):
        self._call(self.bot.clear_alarms)

    def end_effector_params(self, x_bias, y_bias=0.0, z_bias=0.0):
        self._call(self.bot._x, CMD["END_EFFECTOR_PARAMS"], True, False, struct.pack("<3f", x_bias, y_bias, z_bias))

    def close(self):
        self.bot.close()


def connect(cfg=LINK, port=None):
    if cfg.get("BACKEND") == "dll":
        return DllLink(port, cfg)
    if cfg.get("BACKEND") == "teleop":
        return TeleopLink(port, cfg)
    return DobotLink(port, cfg)


# =============================================================================
# Bridge: one process owns the port, others use HTTP on localhost
# =============================================================================
def make_server(link, cfg=LINK):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    state = {"last_error": None, "busy_index": None}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self):
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}") if n else {}

        def do_GET(self):
            if self.path.rstrip("/") != "/state":
                return self._send(404, {"error": "unknown path"})
            try:
                pose = link.get_pose()
                self._send(200, {"connected": True, "port": link.port, "pose": [round(v, 3) for v in pose],
                                 "queue_index": link.queued_index(), "last_error": state["last_error"]})
            except Exception as e:
                self._send(503, {"connected": False, "error": str(e)})

        def do_POST(self):
            p = self.path.rstrip("/")
            try:
                b = self._body()
                if p == "/ptp":
                    mode = b.get("mode", "MOVL_XYZ")
                    if mode not in PTP_MODE:
                        return self._send(400, {"error": f"mode must be one of {list(PTP_MODE)}"})
                    idx = None
                    for pt in b.get("points", []):
                        x, y, z = (float(v) for v in pt[:3])
                        idx = link.ptp(x, y, z, float(pt[3]) if len(pt) > 3 else 0.0, mode)
                    if b.get("wait") and idx is not None:
                        link.wait(idx, b.get("timeout"))
                    return self._send(200, {"index": idx})
                if p == "/suck":
                    return self._send(200, {"index": link.suction(bool(b.get("on")), wait=bool(b.get("wait", True)))})
                if p == "/speed":
                    link.speed(float(b["v"]), float(b.get("a", b["v"])), float(b.get("ratio", 100)))
                    return self._send(200, {"ok": True})
                if p == "/wait":
                    link.wait(int(b["index"]), b.get("timeout"))
                    return self._send(200, {"ok": True})
                if p == "/stop":
                    link.stop_now()
                    return self._send(200, {"ok": True})
                if p == "/clear_alarms":
                    link.clear_alarms()
                    return self._send(200, {"ok": True})
                if p == "/home":
                    if not b.get("confirm"):
                        return self._send(400, {"error": "homing moves the arm a lot: send {\"confirm\": true}"})
                    return self._send(200, {"index": link.home()})
                return self._send(404, {"error": "unknown path"})
            except Exception as e:
                state["last_error"] = str(e)
                return self._send(500, {"error": str(e)})

    return ThreadingHTTPServer((cfg["BRIDGE_HOST"], cfg["BRIDGE_PORT"]), H)


class BridgeClient:
    """The bridge's API with the same method names as DobotLink (for dobot_real.py and Isaac Sim)."""

    def __init__(self, url=None, cfg=LINK, timeout=3.0):
        import urllib.request
        self.url = (url or f"http://{cfg['BRIDGE_HOST']}:{cfg['BRIDGE_PORT']}").rstrip("/")
        self._op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.timeout = timeout
        self.port = f"bridge {self.url}"
        st = self.state()
        if not st.get("connected"):
            raise DobotError(f"bridge up but robot not answering: {st.get('error')}")

    def _req(self, path, body=None, timeout=None):
        import urllib.error
        import urllib.request
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.url + path, data=data, headers={"Content-Type": "application/json"})
        try:
            with self._op.open(req, timeout=timeout or self.timeout) as r:
                return json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            try:
                msg = json.loads(e.read()).get("error")
            except Exception:
                msg = str(e)
            raise DobotError(msg)

    def state(self):
        return self._req("/state")

    def get_pose(self):
        return self.state()["pose"]

    def queued_index(self):
        return self.state()["queue_index"]

    def ptp(self, x, y, z, r=0.0, mode="MOVL_XYZ", wait=False):
        return self._req("/ptp", {"points": [[x, y, z, r]], "mode": mode, "wait": wait},
                         timeout=LINK["WAIT_TIMEOUT_S"] + 5 if wait else None)["index"]

    def suction(self, on, wait=True):
        return self._req("/suck", {"on": bool(on), "wait": wait}, timeout=LINK["WAIT_TIMEOUT_S"] + 5)["index"]

    def speed(self, v, a, ratio=100.0):
        self._req("/speed", {"v": v, "a": a, "ratio": ratio})

    def wait(self, index, timeout=None):
        self._req("/wait", {"index": index, "timeout": timeout}, timeout=(timeout or LINK["WAIT_TIMEOUT_S"]) + 5)

    def stop_now(self):
        self._req("/stop", {})

    def clear_alarms(self):
        self._req("/clear_alarms", {})

    def close(self):
        pass


def bridge_or_direct(cfg=LINK):
    """The bridge if it is running (it owns the port), else a direct connection."""
    try:
        return BridgeClient(cfg=cfg)
    except Exception:
        return connect(cfg)


# =============================================================================
# CLI
# =============================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["ports", "pose", "serve", "jog", "home", "stop", "clear-alarms"])
    ap.add_argument("--port", default=None)
    ap.add_argument("--dx", type=float, default=0.0)
    ap.add_argument("--dy", type=float, default=0.0)
    ap.add_argument("--dz", type=float, default=0.0)
    a = ap.parse_args(argv)
    if a.command == "ports":
        from serial.tools import list_ports
        for p in list_ports.comports():
            print(f"{p.device:16s} {p.description}  vid:pid={p.vid and hex(p.vid)}:{p.pid and hex(p.pid)}")
        print(f"auto-selected: {find_port()}")
        return 0
    try:
        if a.command == "serve":
            link = connect(LINK, a.port)
            srv = make_server(link, LINK)
            _say(f"robot on {link.port}, pose {[round(v, 1) for v in link.get_pose()[:4]]}; bridge on "
                 f"http://{LINK['BRIDGE_HOST']}:{LINK['BRIDGE_PORT']} (Ctrl+C stops it)")
            try:
                srv.serve_forever()
            except KeyboardInterrupt:
                _say("stopping: clearing the robot's queue")
                try:
                    link.stop_now()
                except Exception:
                    pass
            finally:
                srv.server_close()
                link.close()
            return 0
        link = bridge_or_direct(LINK)
        try:
            if a.command == "pose":
                x, y, z, r, j1, j2, j3, j4 = link.get_pose()
                _say(f"via {link.port}: x {x:.1f}  y {y:.1f}  z {z:.1f}  r {r:.1f} mm/deg | joints {j1:.1f} {j2:.1f} "
                     f"{j3:.1f} {j4:.1f} deg | queue index {link.queued_index()}")
            elif a.command == "stop":
                link.stop_now()
                _say("queue force-stopped and cleared")
            elif a.command == "home":
                if input("Homing swings the arm through its whole range to its limit switches. Clear the area. "
                         "Type yes: ").strip().lower() != "yes":
                    return 1
                if isinstance(link, BridgeClient):
                    idx = link._req("/home", {"confirm": True})["index"]
                else:
                    idx = link.home()
                link.wait(idx, timeout=90)
                _say(f"homed; pose {[round(v, 1) for v in link.get_pose()[:4]]}")
            elif a.command == "clear-alarms":
                link.clear_alarms()
                _say("alarms cleared")
            elif a.command == "jog":
                x, y, z, r = link.get_pose()[:4]
                tgt = (x + a.dx, y + a.dy, z + a.dz)
                if max(abs(a.dx), abs(a.dy), abs(a.dz)) > 30:
                    _say("jog is limited to 30 mm per axis")
                    return 1
                if input(f"move from {[round(v, 1) for v in (x, y, z)]} to {[round(v, 1) for v in tgt]}? "
                         f"type yes: ").strip().lower() != "yes":
                    return 1
                link.speed(20, 20)
                link.ptp(*tgt, r, mode="MOVL_XYZ", wait=True)
                _say(f"now at {[round(v, 1) for v in link.get_pose()[:3]]}")
        finally:
            link.close()
        return 0
    except Exception as e:
        _say(f"ERROR: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
