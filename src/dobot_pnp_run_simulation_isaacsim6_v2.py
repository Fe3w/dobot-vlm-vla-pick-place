"""
Integrated Dobot Magician pick-and-place controller for NVIDIA Isaac Sim 6.0.

This combines the newer uploaded script (base controller + viewport visualization
+ depth-image diagnostics). It is adapted to the user's scene/worktree paths.

RUN MODE: "SIMULATION" first stops the timeline with frame updates and reloads the
saved test USD to reset in-memory object movement from an earlier failed attempt.
It then applies known in-memory preparation, reruns preflight, and starts the
reduced-speed controller only if blocking checks pass. It discards unsaved edits by
reloading the saved test USD and does not save the stage. It includes a camera-model
projected fallback grasp pixel on the configured red cylinder and fallback placements
around the cone. Keep Isaac Sim visible and be ready to stop playback. Missing robot-description/keypoint files are never fabricated.

The pick/release mechanism here is a kinematic latch, not the Isaac Surface
Gripper API. The separate SurfaceGripper/D6 prims are not used by this controller.


Pipeline (Phase 2: solid payload, obstacle-aware transit)
    1. Make sure the timeline is playing (physics must step for raycasts / IK).
    2. Read keypoints_2d.json: grasp_points, placement_candidates and
       obstacle_points (pixel_xy, image size from the file).
    3. Depth: back-project each pixel through the overhead camera and cast it
       into the scene (PhysX colliders + visible USD geometry; only the robot's
       own links are ignored). No replicator, no step_async, no render products.
    4. The object under the grasp pixel is resolved to its rigid body; a ring of
       depth samples gives the support height and the object height.
    5. Obstacles: every obstacle_points pixel is depth-cast; hits that land on a
       compact scene object become an obstacle (convex footprint from its real
       geometry at its live pose, plus top height).
       Labels for the robot / platform, and hits on the support, are ignored.
    6. For each placement candidate in order: plan a route. Short lift, then a
       shortest-path search over a polar (radius, azimuth) grid in which the tip
       rides as high as the arm can reach at each radius, keeping the held
       object's real footprint clear of every obstacle (or above it), then a
       vertical descent. First feasible candidate wins.
    7. Execute: HOME -> above pick -> hover -> touchdown -> kinematic vacuum ->
       route -> descent -> release -> back off -> HOME.
    8. Visual debugging (Phase 3, display only): depth hits, obstacle outlines,
       the planned tip path and a live tip trail are drawn in the viewport with
       Isaac Sim's debug-draw interface (no prims at all), or, if that extension
       is missing, as collider-free curves under VIZ_ROOT in the session layer.
       Marker prims (VIZ_ROOT and anything named like DEPTH_IGNORE_NAME_PATTERNS,
       e.g. the stage's DEBUG_* spheres) are excluded from every depth ray.
    9. Depth view (Phase 4, display only): once, before any motion, a depth image
       of /overhead_camera is built from the same rays the pipeline uses, the
       keypoints are drawn on it (grasp / placements / obstacles, then marked
       used or rejected), and it is saved as a PNG in DEPTH_OUT_DIR and shown in
       a "Dobot depth" window. Optionally (DEPTH_RENDERED) the camera's rendered
       depth is grabbed once from a temporary render product, saved beside it
       and compared with the ray-cast one; the render product is destroyed
       straight after. Nothing here feeds back into perception or planning.

Design notes
    * The vacuum is a kinematic latch, never a joint: on engage the object's
      RigidBodyAPI is switched to kinematic and its USD pose is rewritten every
      physics step from link_suction (sync_vacuum_payload). The robot's
      articulation is never touched. The payload keeps its colliders; a
      FilteredPairsAPI makes it ignore only the suction link (and its parent).
    * A kinematic body driven by pose writes is never stopped by contacts: it
      shoves dynamic bodies and passes through static/kinematic ones. Avoidance
      therefore comes from planning; a contact watchdog stops the arm if the
      held payload touches anything during transit anyway.
    * IK: Lula provides forward kinematics and a seed; the final solve is a damped
      least-squares on joint_1/joint_2/joint_5 with joint_6 = s * (joint_2 + joint_5)
      enforced at every iteration. The sign s is checked against the URDF at
      start-up (the one that keeps the suction cup's orientation fixed).
    * Geometry: the Lula base frame is the static USD transform of base_link
      (or the robot prim). At start-up the suction nozzle is measured from the
      simulated link_suction mesh: its axis and its tip are expressed in Lula's
      link_suction frame. Every target is for that tip, and joint_6's bias is
      chosen so the nozzle axis points straight down (world -Z) in every pose.
    * NumPy arrays are never used in boolean context; every optional value is
      checked with `is None`.
    * Joint I/O is layered so a tensor-API failure cannot stop the run:
        1. a CPU (numpy) PhysX tensor view created with the *explicit* stage id,
           only after PhysX has attached the stage and stepped;
        2. the view owned by isaacsim.core.simulation_manager, if one exists;
        3. USD drive targets for commands + live PhysX link poses for joint
           angles (no tensor API at all).
"""

import asyncio
import builtins
import gc
import json
import math
import os
import time
import traceback

import numpy as np

import carb
import omni.kit.app
import omni.physx
import omni.timeline
import omni.usd
from pxr import Gf, PhysicsSchemaTools, Sdf, Usd, UsdGeom, UsdPhysics


# =============================================================================
# Configuration
# =============================================================================
_HOME = os.path.expanduser("~")
# >>> dobot-settings (generated by tools/patch_paths.py; same block in every script)
def _dobot_settings():
    """Folders and settings: environment variables, else the repo's .env, else ~/.config/dobot-pnp/.env."""
    import os
    f = globals().get("__file__")
    src = os.path.dirname(os.path.abspath(f)) if f else ""
    vals = {}
    for c in (os.environ.get("DOBOT_ENV_FILE"), src and os.path.join(src, "..", ".env"),
              os.path.expanduser("~/.config/dobot-pnp/.env")):
        if not c or not os.path.isfile(c):
            continue
        with open(c) as fh:
            for line in fh:
                line = line.strip()
                if line.startswith("export "):
                    line = line[7:].strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                v = v.strip()
                if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
                    v = v[1:-1]
                else:
                    v = v.split(" #")[0].strip()
                vals[k.strip()] = v
        vals.setdefault("DOBOT_HOME", os.path.dirname(os.path.abspath(c)))
        break
    vals.update({k: v for k, v in os.environ.items() if k.startswith(("DOBOT_", "VLM_"))})
    home = os.path.expanduser(vals.get("DOBOT_HOME") or (os.path.dirname(src) if src else "~/dobot-vlm-pick-place"))

    def path(key, default):
        p = os.path.expanduser(vals.get(key) or default)
        return os.path.normpath(p if os.path.isabs(p) else os.path.join(home, p))
    vals["DOBOT_HOME"] = home
    vals["DOBOT_SRC"] = path("DOBOT_SRC", src or "src")
    vals["DOBOT_ASSETS"] = path("DOBOT_ASSETS", "assets")
    vals["DOBOT_DATA"] = path("DOBOT_DATA", ".")
    return vals


_DOBOT_ENV = _dobot_settings()
_DOBOT_SRC, _DOBOT_ASSETS, _DOBOT_DATA = _DOBOT_ENV["DOBOT_SRC"], _DOBOT_ENV["DOBOT_ASSETS"], _DOBOT_ENV["DOBOT_DATA"]
# <<< dobot-settings

_ASSET_DIR = _DOBOT_ASSETS
_PROJECT_DIR = _DOBOT_DATA
_KP_DIR = os.path.join(_PROJECT_DIR, "output", "new_scene_red_cylinder")

CFG = {
    # --- Paths / prims -------------------------------------------------------
    "ROBOT_PRIM": "/magician",
    "EE_FRAME": "link_suction",
    "EE_PRIM": "/magician/link_suction",
    "URDF_PATH": os.path.join(_ASSET_DIR, "load_dobot.urdf"),
    # Verify this path in the preflight output; edit it if your YAML is elsewhere.
    "YAML_PATH": os.path.join(_ASSET_DIR, "dobot_description.yaml"),
    "JSON_DIR": _KP_DIR,
    "JSON_FILE": "keypoints_2d.json",
    "CAMERA_PRIM": "/overhead_camera",
    "RED_PRIM": "/magician/color",
    "CONE_PRIM": "/magician/Cone",  # scene diagnostic found it nested under /magician
    "SPHERE_PRIM": "/magician/Sphere",  # scene diagnostic found it nested under /magician
    "AUTO_IMPORT_URDF": False,         # never import/replace the current repaired robot
    "RUN_MODE": "SIMULATION",         # prepares known issues, then runs only if preflight passes

    # --- Kinematics ----------------------------------------------------------
    "JOINT_BASE": "joint_1",
    "JOINT_SHOULDER": "joint_2",
    "JOINT_ELBOW": "joint_5",
    "JOINT_PITCH": "joint_6",          # joint_6 = s * (joint_2 + joint_5)
    "PITCH_SIGN": 1.0,                 # s in joint_6 = s*(joint_2 + joint_5) + b; +1 for this URDF
                                       # ("auto" picks the sign that keeps the cup orientation fixed)
    "PITCH_BIAS": "auto",              # b (rad); "auto" = the bias that points the nozzle at world -Z
    "HOME_JOINTS": None,               # dict {name: rad}; None = pose at script start
    "IK_TOL_MM": 0.2,
    "IK_MAX_ITERS": 80,
    "MAX_JOINT_JUMP_RAD": 0.35,        # between consecutive Cartesian waypoints
    "BASE_PRIM": None,                 # Lula base frame in USD; None = ROBOT_PRIM/base_link, else ROBOT_PRIM
    "TOOL_OFFSET_M": "auto",           # suction tip in Lula's link_suction frame (metres); "auto" = measured
                                       # from the simulated link_suction mesh (nozzle end) at start-up
    "MATCH_SIM_EE": True,              # measure the tool on the simulated link_suction (absorbs the 40 mm
                                       # gap between the simulated link and Lula's URDF link)

    # --- Camera / perception -------------------------------------------------
    "DEFAULT_IMAGE_SIZE": (1280, 720), # used only if the JSON carries no size
    "PIXEL_CENTER_OFFSET": 0.5,        # integer pixel index -> pixel centre
    "GRASP_INDEX": 0,                  # preferred grasp point; others tried if it fails
    "RAY_MAX_DIST_M": 50.0,
    "TABLE_RING_RADII_PX": (10, 16, 24, 34, 48, 66, 90, 120),
    "TABLE_RING_SAMPLES": 16,
    "TABLE_MIN_SAMPLES": 12,
    "USD_DEPTH": True,                 # also cast against visible geometry that has no collider
    "USD_DEPTH_MAX_TRIS": 300000,      # per mesh
    "ROBOT_EXTRA_EXCLUDE": (),         # extra robot prim paths to hide from depth, if any
    "OBJECT_MAX_SIZE_M": 0.35,         # larger hit prims are treated as support surfaces

    # --- Workspace (millimetres) --------------------------------------------
    "REACH_MAX_MM": 282.0,             # L1 + L2
    "HOVER_CLEARANCE_MM": 25.0,        # tip above the pick surface before touchdown
    "CARRY_CLEARANCE_MM": 30.0,        # held object's bottom above the higher of pick / place surfaces
    "CARRY_MIN_CLEARANCE_MM": 10.0,    # lowest acceptable carry clearance if the arm can't reach higher

    # --- Touchdown / release (millimetres) ----------------------------------
    "APPROACH_GAP_MM": 6.0,
    "TOUCH_STEP_MM": 0.4,              # creep per control tick
    "TOUCH_TOL_MM": 1.0,
    "TOUCH_MAX_OVERDRIVE_MM": 12.0,    # how far below the surface the command may go
    "PLACE_MAX_OVERDRIVE_MM": 6.0,
    "XY_CORRECTION_GAIN": 0.3,
    "XY_CORRECTION_LIMIT_MM": 3.0,
    "TOUCH_TIMEOUT_S": 8.0,
    "RELEASE_CLEARANCE_MM": 1.5,
    "RELEASE_BACKOFF_MM": 8.0,

    # --- Held payload collisions / obstacle avoidance -----------------------
    "HOLD_COLLISION": "filter",        # "filter": payload stays solid, ignores only FILTER_LINKS
                                       # "disable": payload colliders off while held (old behaviour)
    "FILTER_LINKS": "auto",            # "auto" = link_suction + its URDF parent link; or a list of prim paths
    "CONTACT_WATCHDOG": True,          # stop the arm if the held payload touches anything in transit
    "AVOID_OBSTACLES": True,           # False = Phase 1 direct joint-space move
    "IGNORE_OBSTACLE_LABELS": ("robot", "platform", "table", "ground", "floor"),
    "OBSTACLE_MARGIN_MM": 15.0,        # horizontal gap between payload and obstacle footprints
    "OBSTACLE_Z_MARGIN_MM": 15.0,      # vertical gap when passing over an obstacle
    "LIFT_MIN_MM": 5.0,                # lift / drop height before the route may move sideways
    "LIFT_INWARD_MM": (0.0, 5.0, 10.0, 15.0, 20.0),  # allowed inward lean of that lift at the reach edge
    "ROUTE_GRID_MM": 5.0,              # radial cell size of the transit search grid
    "ROUTE_GRID_DEG": 2.0,             # azimuth cell size of the transit search grid
    "CEILING_MARGIN_MM": 5.0,          # stay this far below the highest tip height the arm reaches

    # --- Visual debugging (display only; never affects planning) -----------
    "VISUAL_DEBUG": True,
    "VIZ_ROOT": "/World/VisualDebug",  # fallback prims (session layer) if debug-draw is unavailable
    "DEPTH_IGNORE_NAME_PATTERNS": ("debug",),  # prims named like this are markers: depth rays skip them
    "VIZ_CAMERA_RAYS": True,           # rays from the camera to each perceived hit
    "VIZ_KEEP_RAYS": False,            # False: rays are cleared when the arm starts moving
    "VIZ_LIVE_TRAIL": True,            # measured tip, every few physics steps
    "VIZ_TRAIL_EVERY": 4,
    "VIZ_POINT_SIZE": 12.0,            # pixels
    "VIZ_LINE_WIDTH": 2.0,             # pixels (debug-draw) / scaled to metres in the prim fallback

    # --- Depth view (display only; never affects planning) -----------------
    "DEPTH_VIZ": True,                 # depth image of CAMERA_PRIM once per run, keypoints drawn on it
    "DEPTH_OUT_DIR": _KP_DIR,
    "DEPTH_STRIDE_PX": 4,              # one depth ray per N x N pixel block (4 -> ~52k rays at 1245x672)
    "DEPTH_COLORMAP": "turbo",         # "turbo" (near = red, far = blue) or "gray" (near = white)
    "DEPTH_EDGE_MM": 20.0,             # outline depth jumps larger than this (object silhouettes); 0 = off
    "DEPTH_WINDOW": True,              # show the image in an Isaac Sim window ("Dobot depth")
    "DEPTH_SAVE_NPZ": True,            # raw depth (metres, distance to the image plane) as .npz
    "DEPTH_RENDERED": False,           # also grab the camera's rendered depth once (Replicator annotator on a
                                       # temporary render product, plain app updates, destroyed afterwards)
    "DEPTH_RENDER_FRAMES": 30,         # app frames to wait for the rendered depth before giving up

    # --- Motion --------------------------------------------------------------
    "JOINT_SPEED_RAD_S": 0.45,         # reduced speed for initial simulation validation
    "CART_SPEED_MM_S": 35.0,          # Cartesian moves (was 60); touchdown creep is separate
    "TRACK_TOL_MM": 2.0,               # pause the trajectory clock while the tip lags more than this
    "CART_STEP_MM": 1.0,
    "ARC_STEP_DEG": 2.0,
    "SETTLE_TOL_RAD": 0.004,
    "SETTLE_TIMEOUT_S": 2.5,
    "STARTUP_SETTLE_S": 0.5,
}

_TAG = "[DobotPnP]"


def _log(msg):
    print(f"{_TAG} {msg}")
    carb.log_info(f"{_TAG} {msg}")


def _warn(msg):
    print(f"{_TAG} WARNING: {msg}")
    carb.log_warn(f"{_TAG} {msg}")


def _mm(v):
    return float(v) * 1e-3


class _Abort(Exception):
    """Raised to unwind the pipeline (shutdown, timeline stop)."""


class _PlanningError(Exception):
    pass


class _ContactStop(RuntimeError):
    """The held payload touched something during transit."""


# =============================================================================
# Math helpers: column-vector 4x4 homogeneous transforms, metres
# =============================================================================
def _quat_wxyz_to_R(w, x, y, z):
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _R_to_quat_wxyz(R):
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q, dtype=float)
    return q / np.linalg.norm(q)


def _orthonormalize(R):
    U, _, Vt = np.linalg.svd(R)
    Rn = U @ Vt
    if np.linalg.det(Rn) < 0.0:
        U[:, -1] *= -1.0
        Rn = U @ Vt
    return Rn


def _make_T(R, p):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def _inv_T(T):
    R = T[:3, :3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ T[:3, 3]
    return Ti


def _apply(T, p):
    return T[:3, :3] @ np.asarray(p, dtype=float) + T[:3, 3]


def _gf_to_np(m):
    """Gf.Matrix4d (row-vector convention) -> numpy column-vector 4x4."""
    return np.array([[m[i][j] for j in range(4)] for i in range(4)], dtype=float).T


def _wrap_pi(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _path_str(p):
    if p is None:
        return ""
    if isinstance(p, int):
        try:
            return str(PhysicsSchemaTools.intToSdfPath(p))
        except Exception:
            return ""
    return str(p)


def _under(path, prefix):
    if not path or not prefix:
        return False
    return path == prefix or path.startswith(prefix.rstrip("/") + "/")


def _to_np(x):
    """numpy / torch / warp array -> flat float numpy array (None passes through)."""
    if x is None:
        return None
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    elif not isinstance(x, np.ndarray) and hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x, dtype=float).reshape(-1)


def _gf_quat_to_R(q):
    if q is None:
        return np.eye(3)
    im = q.GetImaginary()
    return _quat_wxyz_to_R(float(q.GetReal()), float(im[0]), float(im[1]), float(im[2]))


def _min_jerk(tau):
    tau = min(1.0, max(0.0, tau))
    return tau ** 3 * (10.0 - 15.0 * tau + 6.0 * tau * tau)


def _hull2d(P):
    """Convex hull (counter-clockwise, no repeated end point) of 2D points."""
    P = np.unique(np.round(np.asarray(P, dtype=float)[:, :2], 6), axis=0)
    if len(P) < 3:
        return P
    P = P[np.lexsort((P[:, 1], P[:, 0]))]

    def half(pts):
        h = []
        for p in pts:
            while len(h) >= 2 and ((h[-1][0] - h[-2][0]) * (p[1] - h[-2][1])
                                   - (h[-1][1] - h[-2][1]) * (p[0] - h[-2][0])) <= 0.0:
                h.pop()
            h.append(p)
        return h[:-1]
    return np.array(half(P) + half(P[::-1]))


def _seg_dist(P, A, B):
    """Smallest distance from points P (n,d) to segments A->B (m,d)."""
    d = B - A
    w = P[:, None, :] - A[None, :, :]
    t = np.clip(np.sum(w * d[None], axis=2) / np.maximum(np.sum(d * d, axis=1), 1e-18)[None], 0.0, 1.0)
    return float(np.min(np.linalg.norm(w - t[..., None] * d[None], axis=2)))


def _poly_gap(A, B):
    """Gap between two convex polygons (metres); negative when they overlap."""
    for P in (A, B):
        e = np.roll(P, -1, axis=0) - P
        nrm = np.stack([e[:, 1], -e[:, 0]], axis=1)
        if np.any(np.max(A @ nrm.T, axis=0) < np.min(B @ nrm.T, axis=0)) or \
                np.any(np.max(B @ nrm.T, axis=0) < np.min(A @ nrm.T, axis=0)):
            break
    else:
        return -1.0
    return min(_seg_dist(A, B, np.roll(B, -1, axis=0)), _seg_dist(B, A, np.roll(A, -1, axis=0)))


def _shape_points(prim, t):
    """Representative local-frame points of a Gprim (exact outline for implicit shapes)."""
    if prim.IsA(UsdGeom.Mesh):
        pts = UsdGeom.Mesh(prim).GetPointsAttr().Get(t)
        return None if pts is None or len(pts) == 0 else np.array(pts, dtype=float)
    a = np.linspace(0.0, 2.0 * math.pi, 32, endpoint=False)
    ring = np.stack([np.cos(a), np.sin(a), np.zeros_like(a)], axis=1)

    def along(axis, pts):                     # shape built along local Z -> the prim's axis token
        return pts[:, {"X": [2, 0, 1], "Y": [1, 2, 0]}.get(str(axis), [0, 1, 2])]

    def sphere(r):
        el = np.linspace(-0.5 * math.pi, 0.5 * math.pi, 9)
        return np.concatenate([np.stack([r * math.cos(e) * ring[:, 0], r * math.cos(e) * ring[:, 1],
                                         np.full(len(a), r * math.sin(e))], axis=1) for e in el])
    try:
        if prim.IsA(UsdGeom.Sphere):
            return sphere(float(UsdGeom.Sphere(prim).GetRadiusAttr().Get(t)))
        if prim.IsA(UsdGeom.Cube):
            h = 0.5 * float(UsdGeom.Cube(prim).GetSizeAttr().Get(t))
            return np.array([[x, y, z] for x in (-h, h) for y in (-h, h) for z in (-h, h)])
        if prim.IsA(UsdGeom.Cylinder) or prim.IsA(UsdGeom.Cone) or prim.IsA(UsdGeom.Capsule):
            g = UsdGeom.Cylinder(prim) if prim.IsA(UsdGeom.Cylinder) else (
                UsdGeom.Cone(prim) if prim.IsA(UsdGeom.Cone) else UsdGeom.Capsule(prim))
            r, h = float(g.GetRadiusAttr().Get(t)), float(g.GetHeightAttr().Get(t))
            if prim.IsA(UsdGeom.Capsule):
                pts = np.concatenate([sphere(r) + [0, 0, -0.5 * h], sphere(r) + [0, 0, 0.5 * h]])
            elif prim.IsA(UsdGeom.Cone):
                pts = np.concatenate([r * ring + [0, 0, -0.5 * h], [[0, 0, 0.5 * h]]])
            else:
                pts = np.concatenate([r * ring + [0, 0, -0.5 * h], r * ring + [0, 0, 0.5 * h]])
            return along(g.GetAxisAttr().Get(t), pts)
    except Exception:
        pass
    ext = UsdGeom.Boundable.ComputeExtentFromPlugins(UsdGeom.Boundable(prim), t)
    if ext is None or len(ext) < 2:
        return None
    lo, hi = np.array(ext[0], dtype=float), np.array(ext[1], dtype=float)
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])


# =============================================================================
# Keypoint JSON parsing (tolerant of several common layouts)
# =============================================================================
def _is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _pair(v):
    if isinstance(v, (list, tuple)) and len(v) >= 2 and _is_num(v[0]) and _is_num(v[1]):
        return float(v[0]), float(v[1])
    return None


# Keys that mark a dict as one keypoint record. Records are never searched
# recursively, so alternate encodings in the same record (pixel_yx, *_crop)
# can never be mistaken for extra points.
_XY_KEYS = ("pixel_xy", "uv", "xy", "pixel", "point", "center", "centre", "coords", "keypoint")
_YX_KEYS = ("pixel_yx", "yx")
_PAIR_KEYS = (("u", "v"), ("x", "y"), ("px", "py"))


def _is_record(d):
    return (any(k in d for k in _XY_KEYS + _YX_KEYS)
            or any(a in d and b in d for a, b in _PAIR_KEYS)
            or "type" in d)


def _record_xy(d):
    for k in _XY_KEYS:
        if k in d:
            p = _pair(d[k])
            if p is not None:
                return p
    for a, b in _PAIR_KEYS:
        if a in d and b in d and _is_num(d[a]) and _is_num(d[b]):
            return float(d[a]), float(d[b])
    for k in _YX_KEYS:
        if k in d:
            p = _pair(d[k])
            if p is not None:
                return p[1], p[0]
    return None


def _points_from(obj):
    """List of {"uv": (x, y), "label": str} from a keypoint list / dict, in image (x, y) order."""
    out = []
    if obj is None:
        return out
    p = _pair(obj)
    if p is not None:
        return [{"uv": p, "label": ""}]
    if isinstance(obj, dict):
        if _is_record(obj):
            if obj.get("visible", True) is False:
                return out
            xy = _record_xy(obj)
            if xy is not None:
                out.append({"uv": xy, "label": str(obj.get("object", obj.get("label", "")))})
            return out
        for key in ("points", "keypoints", "candidates", "items"):
            if key in obj:
                return _points_from(obj[key])
        for name, v in obj.items():            # {"cube": [x, y], ...}
            for rec in _points_from(v):
                rec["label"] = rec["label"] or str(name)
                out.append(rec)
        return out
    if isinstance(obj, (list, tuple)):
        for it in obj:
            out.extend(_points_from(it))
    return out


def _find_key(d, names):
    if isinstance(d, dict):
        for n in names:
            if n in d:
                return d[n]
        for v in d.values():
            r = _find_key(v, names)
            if r is not None:
                return r
    return None


def _image_size_from(data):
    w = _find_key(data, ("image_width", "img_width", "width"))
    h = _find_key(data, ("image_height", "img_height", "height"))
    if isinstance(w, (int, float)) and isinstance(h, (int, float)):
        return int(w), int(h)
    sz = _find_key(data, ("image_size", "resolution", "img_size", "image_shape"))
    if isinstance(sz, dict):
        w, h = sz.get("width"), sz.get("height")
        if w is not None and h is not None:
            return int(w), int(h)
    if isinstance(sz, (list, tuple)) and len(sz) >= 2:
        if _find_key(data, ("image_shape",)) is sz:   # numpy-style (H, W[, C])
            return int(sz[1]), int(sz[0])
        return int(sz[0]), int(sz[1])                # (W, H)
    return None


def _intrinsics_from(data):
    K = _find_key(data, ("K", "camera_matrix", "intrinsic_matrix", "intrinsics"))
    if K is None:
        return None
    if isinstance(K, dict):
        if all(k in K for k in ("fx", "fy", "cx", "cy")):
            return float(K["fx"]), float(K["fy"]), float(K["cx"]), float(K["cy"])
        return None
    try:
        Km = np.asarray(K, dtype=float).reshape(3, 3)
        return float(Km[0, 0]), float(Km[1, 1]), float(Km[0, 2]), float(Km[1, 2])
    except Exception:
        return None


# =============================================================================
# USD geometry depth (for visible geometry that has no PhysX collider)
# =============================================================================
class _UsdGeomRaycaster:
    """CPU ray casting against visible, static USD geometry.

    PhysX only sees colliders. A depth camera sees everything that renders, so
    visible Gprims without colliders (e.g. decorative obstacles) are cast here.
    Dynamic rigid bodies are left to PhysX because their USD pose can be stale
    while Fabric drives rendering. Meshes use their real triangles; implicit
    shapes (Cube, Sphere, Capsule, Cylinder, Cone, ...) use their local extent box.
    """

    def __init__(self, stage, mpu, exclude_prefixes, max_tris):
        self.geoms = []
        t = Usd.TimeCode.Default()
        xcache = UsdGeom.XformCache(t)
        skipped_big = []
        for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
            if not prim.IsA(UsdGeom.Gprim):
                continue
            path = str(prim.GetPath())
            if any(_under(path, e) for e in exclude_prefixes):
                continue
            img = UsdGeom.Imageable(prim)
            if img.ComputeVisibility(t) == UsdGeom.Tokens.invisible:
                continue
            if img.ComputePurpose() in (UsdGeom.Tokens.guide, UsdGeom.Tokens.proxy):
                continue
            if self._in_rigid_body(prim):
                continue
            tris = self._triangles(prim, t)
            if tris is None or len(tris) == 0:
                continue
            if len(tris) > max_tris:
                skipped_big.append(path)
                continue
            M = _gf_to_np(xcache.GetLocalToWorldTransform(prim))
            P = tris.reshape(-1, 3)
            Pw = ((P @ M[:3, :3].T) + M[:3, 3]) * mpu
            T = Pw.reshape(-1, 3, 3)
            self.geoms.append({
                "path": path,
                "v0": T[:, 0], "e1": T[:, 1] - T[:, 0], "e2": T[:, 2] - T[:, 0],
                "bmin": Pw.min(axis=0), "bmax": Pw.max(axis=0),
            })
        if skipped_big:
            _warn(f"USD depth skips very large meshes: {skipped_big}")

    @staticmethod
    def _in_rigid_body(prim):
        p = prim
        while p.IsValid() and not p.IsPseudoRoot():
            if p.HasAPI(UsdPhysics.RigidBodyAPI):
                return True
            p = p.GetParent()
        return False

    @staticmethod
    def _triangles(prim, t):
        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            pts = mesh.GetPointsAttr().Get(t)
            counts = mesh.GetFaceVertexCountsAttr().Get(t)
            idx = mesh.GetFaceVertexIndicesAttr().Get(t)
            if pts is None or counts is None or idx is None or len(pts) == 0 or len(counts) == 0:
                return None
            pts = np.array(pts, dtype=float)
            counts = np.array(counts, dtype=np.int64)
            idx = np.array(idx, dtype=np.int64)
            starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
            tris = []
            for k in range(1, int(counts.max()) - 1):
                s = starts[counts > k + 1]
                tris.append(np.stack([idx[s], idx[s + k], idx[s + k + 1]], axis=1))
            if len(tris) == 0:
                return None
            tri = np.concatenate(tris)
            tri = tri[np.all(tri < len(pts), axis=1)]
            return pts[tri]
        ext = UsdGeom.Boundable.ComputeExtentFromPlugins(UsdGeom.Boundable(prim), t)
        if ext is None or len(ext) < 2:
            return None
        lo, hi = np.array(ext[0], dtype=float), np.array(ext[1], dtype=float)
        c = np.array([[lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]], [hi[0], hi[1], lo[2]], [lo[0], hi[1], lo[2]],
                      [lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]], [hi[0], hi[1], hi[2]], [lo[0], hi[1], hi[2]]])
        f = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
                      [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])
        return c[f]

    @staticmethod
    def _hits_aabb(o, d, bmin, bmax, tmax):
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = 1.0 / d
            t1, t2 = (bmin - o) * inv, (bmax - o) * inv
        t1 = np.where(np.isnan(t1), -np.inf, t1)
        t2 = np.where(np.isnan(t2), np.inf, t2)
        tn = np.max(np.minimum(t1, t2))
        tf = np.min(np.maximum(t1, t2))
        return tf >= max(tn, 0.0) and tn <= tmax

    def cast(self, origin, direction, max_dist, skip_prefixes=()):
        o = np.asarray(origin, dtype=float)
        d = np.asarray(direction, dtype=float)
        best_t, best = None, None
        for g in self.geoms:
            if any(_under(g["path"], s) for s in skip_prefixes):
                continue
            if not self._hits_aabb(o, d, g["bmin"] - 1e-6, g["bmax"] + 1e-6, max_dist):
                continue
            e1, e2 = g["e1"], g["e2"]
            p = np.cross(d, e2)
            det = np.einsum("ij,ij->i", e1, p)
            ok = np.abs(det) > 1e-14
            if not np.any(ok):
                continue
            inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
            s = o - g["v0"]
            u = np.einsum("ij,ij->i", s, p) * inv
            q = np.cross(s, e1)
            v = (q @ d) * inv
            tt = np.einsum("ij,ij->i", e2, q) * inv
            valid = ok & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0) & (tt > 1e-6) & (tt < max_dist)
            if not np.any(valid):
                continue
            k = int(np.argmin(np.where(valid, tt, np.inf)))
            if best_t is None or tt[k] < best_t:
                n = np.cross(e1[k], e2[k])
                nn = np.linalg.norm(n)
                n = n / nn if nn > 0.0 else -d
                if n @ d > 0.0:
                    n = -n
                best_t, best = float(tt[k]), (g["path"], n)
        if best is None:
            return None
        return {"position": o + best_t * d, "normal": best[1], "collision": best[0], "body": "",
                "distance": best_t, "source": "usd"}


# =============================================================================
# Joint I/O backends
# =============================================================================
class _TensorJoints:
    """Joint state / targets through the PhysX tensor API (SingleArticulation)."""

    kind = "PhysX tensor view"

    def __init__(self, art, action_cls, own_view):
        self._art = art
        self._ctrl = art.get_articulation_controller()
        self._Action = action_cls
        self._own_view = own_view          # only views we created are invalidated by us
        self.dof_names = list(art.dof_names)

    def index(self, name):
        return int(self._art.get_dof_index(name))

    def positions(self):
        return _to_np(self._art.get_joint_positions())

    def command(self, idx, vals):
        self._ctrl.apply_action(self._Action(joint_positions=np.array(vals, dtype=np.float32),
                                             joint_indices=np.array(idx, dtype=np.int32)))

    def limits(self, idx):
        props = self._art.dof_properties
        return [(float(props["lower"][i]), float(props["upper"][i])) for i in idx]

    def close(self):
        self._ctrl = None
        self._art = None
        if self._own_view is not None:
            try:
                self._own_view.invalidate()
            except Exception:
                pass
            self._own_view = None


class _UsdDriveJoints:
    """Fallback with no tensor API: USD drive targets in, live PhysX link poses out.

    The joint angle is recovered from the relative rotation of the two joint
    frames (body pose * localRot) about the joint axis, which is exactly what the
    PhysX revolute joint measures.
    """

    kind = "USD drive targets + PhysX link poses"
    _AXIS = {"X": 0, "Y": 1, "Z": 2}

    def __init__(self, stage, robot_path, pose_fn):
        self._pose_fn = pose_fn
        self._joints = []
        for prim in Usd.PrimRange(stage.GetPrimAtPath(robot_path)):
            if not prim.IsA(UsdPhysics.RevoluteJoint):
                continue
            j = UsdPhysics.RevoluteJoint(prim)
            b0 = j.GetBody0Rel().GetTargets()
            b1 = j.GetBody1Rel().GetTargets()
            target = None
            if prim.HasAPI(UsdPhysics.DriveAPI, "angular"):
                drive = UsdPhysics.DriveAPI(prim, "angular")
                target = drive.GetTargetPositionAttr()
                if not target:
                    target = drive.CreateTargetPositionAttr(0.0)
            lo, hi = j.GetLowerLimitAttr().Get(), j.GetUpperLimitAttr().Get()
            self._joints.append({
                "name": prim.GetName(),
                "body0": str(b0[0]) if len(b0) > 0 else "",
                "body1": str(b1[0]) if len(b1) > 0 else "",
                "r0": _gf_quat_to_R(j.GetLocalRot0Attr().Get()),
                "r1": _gf_quat_to_R(j.GetLocalRot1Attr().Get()),
                "axis": self._AXIS.get(str(j.GetAxisAttr().Get()), 0),
                "target": target,
                "lo": math.radians(lo) if lo is not None and math.isfinite(lo) else -math.pi,
                "hi": math.radians(hi) if hi is not None and math.isfinite(hi) else math.pi,
            })
        self.dof_names = [j["name"] for j in self._joints]

    def index(self, name):
        return self.dof_names.index(name)

    def _R(self, body):
        if not body:
            return np.eye(3)                 # joint to world
        T = self._pose_fn(body)
        return None if T is None else T[:3, :3]

    def positions(self):
        out = np.zeros(len(self._joints))
        for k, j in enumerate(self._joints):
            R0, R1 = self._R(j["body0"]), self._R(j["body1"])
            if R0 is None or R1 is None:
                return None
            Rr = (R0 @ j["r0"]).T @ (R1 @ j["r1"])
            if j["axis"] == 0:
                out[k] = math.atan2(Rr[2, 1], Rr[1, 1])
            elif j["axis"] == 1:
                out[k] = math.atan2(Rr[0, 2], Rr[2, 2])
            else:
                out[k] = math.atan2(Rr[1, 0], Rr[0, 0])
        return out

    def command(self, idx, vals):
        with Sdf.ChangeBlock():
            for i, v in zip(idx, vals):
                t = self._joints[int(i)]["target"]
                if t is not None:
                    t.Set(float(math.degrees(v)))

    def missing_drives(self, names):
        return [j["name"] for j in self._joints if j["name"] in names and j["target"] is None]

    def limits(self, idx):
        return [(self._joints[int(i)]["lo"], self._joints[int(i)]["hi"]) for i in idx]

    def close(self):
        self._pose_fn = None
        self._joints = []


# =============================================================================
# Main controller
# =============================================================================
class _Viz:
    """Viewport overlay. Display only: every call is best effort and can never raise.

    Groups of points / polylines are kept by name. The preferred backend is the
    debug-draw interface (isaacsim.util.debug_draw), which draws directly in the
    viewport and creates no prims, so neither PhysX nor the USD depth caster can
    ever see it. Without it, each group becomes a UsdGeom.BasisCurves / Points
    prim under VIZ_ROOT, authored in the session layer (never saved with the
    stage), with no physics APIs, and VIZ_ROOT is excluded from depth.
    """

    COLORS = {
        "pick": (0.1, 1.0, 0.2, 1.0), "place": (0.0, 0.9, 1.0, 1.0), "place_rej": (0.45, 0.45, 0.5, 1.0),
        "obstacle": (1.0, 0.15, 0.1, 1.0), "ray": (1.0, 1.0, 1.0, 0.35), "ray_miss": (1.0, 0.4, 0.4, 0.35),
        "carry": (1.0, 0.85, 0.0, 1.0), "free": (1.0, 0.5, 0.0, 1.0), "trail": (1.0, 0.2, 1.0, 1.0),
    }

    def __init__(self, owner):
        self.o = owner
        self.cfg = owner.cfg
        self.groups = {}             # name -> ("points"|"strips", data, color, size)
        self.trail = []
        self.dd = None
        self.root = self.cfg["VIZ_ROOT"]
        try:
            owner._enable_ext("isaacsim.util.debug_draw")
            try:
                from isaacsim.util.debug_draw import _debug_draw
            except ImportError:
                from omni.isaac.debug_draw import _debug_draw
            self.dd = _debug_draw.acquire_debug_draw_interface()
            self.dd.clear_points()
            self.dd.clear_lines()
            _log("visual debug: debug-draw overlay (no prims)")
        except Exception as e:
            self.dd = None
            _log(f"visual debug: debug-draw unavailable ({e}); using collider-free curves under {self.root}")
        self._reset_prims()

    # -- units: world metres -> stage units
    def _u(self, p):
        p = np.asarray(p, dtype=float) / self.o._mpu
        return (float(p[0]), float(p[1]), float(p[2]))

    def _reset_prims(self):
        try:
            st = self.o._stage
            with Usd.EditContext(st, st.GetSessionLayer()):
                if st.GetPrimAtPath(self.root).IsValid():
                    st.RemovePrim(self.root)
        except Exception:
            pass

    # -- public API (world-frame metres)
    def points(self, name, pts, color, size=None):
        self.groups[name] = ("points", [np.asarray(p, dtype=float) for p in pts], color,
                             self.cfg["VIZ_POINT_SIZE"] if size is None else size)
        self._redraw()

    def strips(self, name, strips, color, width=None):
        """strips: list of polylines (each a list of points)."""
        self.groups[name] = ("strips", [[np.asarray(p, dtype=float) for p in s] for s in strips if len(s) >= 2],
                             color, self.cfg["VIZ_LINE_WIDTH"] if width is None else width)
        self._redraw()

    def drop(self, prefix):
        gone = [k for k in self.groups if k.startswith(prefix)]
        for k in gone:
            del self.groups[k]
        if gone:
            self._redraw(removed=gone)

    def trail_point(self, p):
        self.trail.append(np.asarray(p, dtype=float))
        try:
            if self.dd is not None:
                self.dd.draw_points([self._u(p)], [self.COLORS["trail"]], [0.5 * self.cfg["VIZ_POINT_SIZE"]])
            elif len(self.trail) % 10 == 0:
                self._usd_group("trail", ("points", self.trail, self.COLORS["trail"],
                                          0.5 * self.cfg["VIZ_POINT_SIZE"]))
        except Exception:
            pass

    # -- backends
    def _redraw(self, removed=()):
        try:
            if self.dd is not None:
                self.dd.clear_points()
                self.dd.clear_lines()
                for kind, data, color, size in self.groups.values():
                    if kind == "points" and data:
                        self.dd.draw_points([self._u(p) for p in data], [color] * len(data), [size] * len(data))
                    elif kind == "strips":
                        a = [self._u(p) for s in data for p in s[:-1]]
                        b = [self._u(p) for s in data for p in s[1:]]
                        if a:
                            self.dd.draw_lines(a, b, [color] * len(a), [size] * len(a))
                if self.trail:
                    n = len(self.trail)
                    self.dd.draw_points([self._u(p) for p in self.trail], [self.COLORS["trail"]] * n,
                                        [0.5 * self.cfg["VIZ_POINT_SIZE"]] * n)
            else:
                for name in removed:
                    self._usd_remove(name)
                for name, g in self.groups.items():
                    self._usd_group(name, g)
        except Exception as e:
            carb.log_warn(f"{_TAG} visual debug draw failed: {e}")

    def _usd_path(self, name):
        return f"{self.root}/{''.join(ch if ch.isalnum() else '_' for ch in name)}"

    def _usd_remove(self, name):
        st = self.o._stage
        with Usd.EditContext(st, st.GetSessionLayer()):
            if st.GetPrimAtPath(self._usd_path(name)).IsValid():
                st.RemovePrim(self._usd_path(name))

    def _usd_group(self, name, g):
        kind, data, color, size = g
        st = self.o._stage
        path = self._usd_path(name)
        mpu = self.o._mpu
        with Usd.EditContext(st, st.GetSessionLayer()):
            UsdGeom.Scope.Define(st, self.root)
            if kind == "points":
                if not data:
                    return
                g_ = UsdGeom.Points.Define(st, path)
                g_.CreatePointsAttr().Set([Gf.Vec3f(*self._u(p)) for p in data])
                g_.CreateWidthsAttr().Set([0.006 / mpu] * len(data))
            else:
                if not data:
                    return
                g_ = UsdGeom.BasisCurves.Define(st, path)
                g_.CreateTypeAttr().Set(UsdGeom.Tokens.linear)
                g_.CreateCurveVertexCountsAttr().Set([len(s) for s in data])
                g_.CreatePointsAttr().Set([Gf.Vec3f(*self._u(p)) for s in data for p in s])
                g_.CreateWidthsAttr().Set([0.0012 * size / mpu])
                g_.SetWidthsInterpolation(UsdGeom.Tokens.constant)
            g_.CreateDisplayColorAttr().Set([Gf.Vec3f(*color[:3])])
            g_.CreateDisplayOpacityAttr().Set([float(color[3])])


# =============================================================================
# Depth view helpers (display only)
# =============================================================================
def _turbo(x):
    """Turbo colour map (polynomial fit), x in [0, 1] -> uint8 RGB."""
    x = np.clip(x, 0.0, 1.0)
    r = 0.13572138 + x * (4.61539260 + x * (-42.66032258 + x * (132.13108234 + x * (-152.94239396 + x * 59.28637943))))
    g = 0.09140261 + x * (2.19418839 + x * (4.84296658 + x * (-14.18503333 + x * (4.27729857 + x * 2.82956604))))
    b = 0.10667330 + x * (12.64194608 + x * (-60.58204836 + x * (110.36276771 + x * (-89.90310912 + x * 27.34824973))))
    return (np.clip(np.stack([r, g, b], axis=-1), 0.0, 1.0) * 255.0).astype(np.uint8)


def _colorize_depth(z, lo, hi, cmap):
    """Depth (metres) -> uint8 RGB; pixels with no hit are black."""
    ok = np.isfinite(z) & (z > 0.0)
    x = np.where(ok, 1.0 - (np.where(ok, z, hi) - lo) / max(hi - lo, 1e-6), 0.0)   # 1 = near
    if cmap == "gray":
        rgb = np.repeat((np.clip(x, 0.0, 1.0) * 215.0 + 40.0).astype(np.uint8)[..., None], 3, axis=-1)
    else:
        rgb = _turbo(x)
    rgb[~ok] = 0
    return rgb


def _depth_edges(z, jump):
    """Pixels where depth jumps by more than `jump` metres to a neighbour (no-hit pixels count as far)."""
    zz = np.where(np.isfinite(z), z, 1e6)
    e = np.zeros(z.shape, dtype=bool)
    dx = np.abs(np.diff(zz, axis=1)) > jump
    dy = np.abs(np.diff(zz, axis=0)) > jump
    e[:, 1:] |= dx
    e[1:, :] |= dy
    return e


def _draw_mark(img, u, v, kind, color, r=9, t=2):
    """ring / plus / cross centred on pixel (u, v), black outline under the colour."""
    h, w = img.shape[:2]
    x0, x1 = int(max(0, math.floor(u - r - t - 1))), int(min(w, math.ceil(u + r + t + 2)))
    y0, y1 = int(max(0, math.floor(v - r - t - 1))), int(min(h, math.ceil(v + r + t + 2)))
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dx, dy = xx + 0.5 - u, yy + 0.5 - v
    for col, half in (((0, 0, 0), 0.5 * t + 1.0), (color, 0.5 * t)):
        if kind == "ring":
            m = np.abs(np.hypot(dx, dy) - r) <= half
        elif kind == "plus":
            m = ((np.abs(dx) <= half) | (np.abs(dy) <= half)) & (np.maximum(np.abs(dx), np.abs(dy)) <= r + half)
        else:
            m = ((np.abs(dx - dy) <= half * 1.41) | (np.abs(dx + dy) <= half * 1.41)) & \
                (np.maximum(np.abs(dx), np.abs(dy)) <= r * 0.75 + half)
        img[y0:y1, x0:x1][m] = col


def _draw_text(img, items):
    """items: (x, y, text, rgb). Needs PIL (bundled with Isaac Sim); silently skipped without it."""
    try:
        from PIL import Image, ImageDraw
        im = Image.fromarray(img)
        dr = ImageDraw.Draw(im)
        for x, y, txt, col in items:
            try:
                dr.text((x, y), txt, fill=tuple(col), stroke_width=2, stroke_fill=(0, 0, 0))
            except TypeError:
                dr.text((x, y), txt, fill=tuple(col))
        return np.array(im)
    except Exception:
        return img


def _write_png(path, rgb):
    """8-bit RGB PNG with the standard library only (no PIL / OpenCV / Matplotlib needed)."""
    import struct
    import zlib
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    h, w = rgb.shape[:2]
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


class DobotPickPlace:
    def __init__(self, cfg):
        self.cfg = dict(cfg)
        self._app = omni.kit.app.get_app()
        self._timeline = omni.timeline.get_timeline_interface()
        self._stage = None
        self._mpu = 1.0
        self._task = None
        self._physics_sub = None
        self._timeline_sub = None
        self._joints = None
        self._rt_stage = None
        self._robot_links = []
        self._usd_rc = None
        self._aperture_warned = False
        self._pitch_sign = -1.0
        self._pitch_bias = 0.0
        self._filter_state = None
        self._contact_state = None
        self._contact_sub = None
        self._contact_hit = None
        self._contact_ignore = set()
        self._watch_contacts = False
        self._lula = None
        self._Articulation = None
        self._ArticulationAction = None
        self._sim_clock = 0.0
        self._physics_steps = 0
        self._shutting_down = False
        self._latch = None
        self._sync_errors = 0
        self._q_cmd = None
        self._T_world_base = np.eye(4)
        self._T_base_world = np.eye(4)
        self._tip_local = np.zeros(3)
        self._viz = None
        self._viz_skip = []
        self._viz_ticks = 0
        self._depth = None

    # ------------------------------------------------------------------ life
    def start(self):
        self._task = asyncio.ensure_future(self._main())

    def shutdown(self, reason="shutdown", from_task=False):
        if self._shutting_down:
            return
        self._shutting_down = True
        _log(f"tearing down ({reason})")
        if not from_task and self._task is not None and not self._task.done():
            self._task.cancel()
        self._task = None
        self._restore_payload_immediately()
        self._restore_payload_filter()
        self._stop_contact_watch()
        self._physics_sub = None      # dropping the handle unsubscribes
        self._timeline_sub = None
        if self._joints is not None:
            self._joints.close()
            self._joints = None
        self._lula = None
        self._rt_stage = None
        self._usd_rc = None
        gc.collect()
        if getattr(builtins, "_DOBOT_PNP_INSTANCE", None) is self:
            builtins._DOBOT_PNP_INSTANCE = None

    async def _main(self):
        try:
            await self._setup()
            if self.cfg["VISUAL_DEBUG"]:
                try:
                    self._viz = _Viz(self)
                except Exception as e:
                    _warn(f"visual debug disabled: {e}")
                    self._viz = None
            await self._pipeline()
            _log("pick-and-place finished")
        except asyncio.CancelledError:
            _log("task cancelled")
        except _Abort as e:
            _warn(f"aborted: {e}")
        except Exception as e:
            _warn(f"pipeline failed: {e}\n{traceback.format_exc()}")
        finally:
            # Unsubscribe/restore first; then stop physics and give Kit a frame to process
            # the timeline STOP event. This avoids leaving the timeline playing after a
            # perception/planning exception, which otherwise makes the next run's preflight fail.
            self.shutdown("pipeline ended", from_task=True)
            try:
                if not self._timeline.is_stopped():
                    self._timeline.stop()
                    for _ in range(90):
                        await self._app.next_update_async()
                        if self._timeline.is_stopped():
                            break
                if not self._timeline.is_stopped():
                    _warn("timeline did not report stopped after 90 app updates; stop it from the UI before retrying")
            except Exception as stop_exc:
                _warn(f"timeline shutdown after pipeline exit raised: {stop_exc}")

    # ----------------------------------------------------------------- setup
    def _enable_ext(self, name):
        try:
            mgr = self._app.get_extension_manager()
            if not mgr.is_extension_enabled(name):
                mgr.set_extension_enabled_immediate(name, True)
        except Exception as e:
            _warn(f"could not enable extension {name}: {e}")

    def _import_isaac_apis(self):
        self._enable_ext("isaacsim.robot_motion.motion_generation")
        try:
            from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver
        except ImportError:
            from omni.isaac.motion_generation import LulaKinematicsSolver
        try:
            from isaacsim.core.prims import SingleArticulation as Articulation
        except ImportError:
            from omni.isaac.core.articulations import Articulation
        try:
            from isaacsim.core.utils.types import ArticulationAction
        except ImportError:
            from omni.isaac.core.utils.types import ArticulationAction
        return LulaKinematicsSolver, Articulation, ArticulationAction

    async def _setup(self):
        c = self.cfg
        self._stage = omni.usd.get_context().get_stage()
        if self._stage is None:
            raise RuntimeError("no USD stage is open")
        mpu = UsdGeom.GetStageMetersPerUnit(self._stage)
        self._mpu = float(mpu) if mpu else 1.0
        if UsdGeom.GetStageUpAxis(self._stage) != UsdGeom.Tokens.z:
            raise RuntimeError("stage up-axis must be Z")

        for key in ("URDF_PATH", "YAML_PATH"):
            if not os.path.isfile(c[key]):
                raise FileNotFoundError(f"{key} not found: {c[key]}")

        if not self._stage.GetPrimAtPath(c["ROBOT_PRIM"]).IsValid():
            if not c["AUTO_IMPORT_URDF"]:
                raise RuntimeError(f"robot prim {c['ROBOT_PRIM']} not found")
            await self._import_urdf()

        self._ensure_physics_scene()
        cam = self._stage.GetPrimAtPath(c["CAMERA_PRIM"])
        if not cam.IsValid() or not cam.IsA(UsdGeom.Camera):
            raise RuntimeError(f"camera prim {c['CAMERA_PRIM']} not found or not a Camera")

        ee_prim = self._resolve_ee_prim()
        self._robot_links = self._robot_link_paths()
        self._viz_skip = self._marker_paths()
        if c["USD_DEPTH"]:
            self._usd_rc = _UsdGeomRaycaster(self._stage, self._mpu,
                                             self._robot_links + self._viz_skip + [c["CAMERA_PRIM"]],
                                             c["USD_DEPTH_MAX_TRIS"])
            _log(f"USD depth geometry (no collider): {[g['path'] for g in self._usd_rc.geoms]}")
        self._keypoints = self._load_keypoints()

        LulaKinematicsSolver, self._Articulation, self._ArticulationAction = self._import_isaac_apis()

        # Lula: base pose is identity, so FK/IK live in the robot base frame (metres).
        self._lula = LulaKinematicsSolver(robot_description_path=c["YAML_PATH"], urdf_path=c["URDF_PATH"])
        self._lula.set_robot_base_pose(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
        self._cs_names = list(self._lula.get_joint_names())
        for j in (c["JOINT_BASE"], c["JOINT_SHOULDER"], c["JOINT_ELBOW"]):
            if j not in self._cs_names:
                raise RuntimeError(f"{j} is not in the Lula c-space {self._cs_names}")
        self._i1 = self._cs_names.index(c["JOINT_BASE"])
        self._i2 = self._cs_names.index(c["JOINT_SHOULDER"])
        self._i5 = self._cs_names.index(c["JOINT_ELBOW"])
        self._i6 = self._cs_names.index(c["JOINT_PITCH"]) if c["JOINT_PITCH"] in self._cs_names else None
        self._free = [self._i1, self._i2, self._i5]
        self._pitch_sign = self._detect_pitch_sign()

        # Subscribe before Play so we can tell when PhysX has really started stepping.
        self._physics_sub = omni.physx.get_physx_interface().subscribe_physics_step_events(self._on_physics_step)
        await self._ensure_playing()
        self._timeline_sub = self._timeline.get_timeline_event_stream().create_subscription_to_pop_by_type(
            int(omni.timeline.TimelineEventType.STOP), self._on_timeline_stop)

        self._joints = await self._create_joint_backend()
        dof_names = self._joints.dof_names
        missing = [n for n in self._cs_names if n not in dof_names]
        if missing:
            raise RuntimeError(f"Lula joints {missing} not found in articulation joints {dof_names}")
        self._cs_to_dof = np.array([self._joints.index(n) for n in self._cs_names], dtype=np.int32)
        self._pitch_dof_extra = None
        if self._i6 is None and c["JOINT_PITCH"] in dof_names:
            self._pitch_dof_extra = int(self._joints.index(c["JOINT_PITCH"]))
            _warn(f"{c['JOINT_PITCH']} is not in the Lula c-space; commanding it from the constraint only")
        self._load_joint_limits()

        await self._wait_sim(c["STARTUP_SETTLE_S"])
        q0 = self._q_meas()
        if q0 is None:
            raise RuntimeError("could not read joint positions")
        self._q_cmd = self._constrain(q0)

        self._set_base_and_tool(q0, ee_prim)
        self._q_cmd = None
        self._q_cmd = self._constrain(q0)

        home = self.cfg["HOME_JOINTS"]
        if home is None:
            self._q_home = self._constrain(q0)
        else:
            self._q_home = self._constrain(np.array([float(home.get(n, q0[i])) for i, n in enumerate(self._cs_names)]))
        _log(f"setup complete; c-space={self._cs_names}, metres/unit={self._mpu}")

    async def _import_urdf(self):
        c = self.cfg
        if not self._timeline.is_stopped():
            self._timeline.stop()
            await self._frames(5)
        self._enable_ext("isaacsim.asset.importer.urdf")
        import omni.kit.commands
        ok, icfg = omni.kit.commands.execute("URDFCreateImportConfig")
        if not ok or icfg is None:
            raise RuntimeError("URDF importer unavailable; import the robot manually")
        icfg.merge_fixed_joints = False
        icfg.fix_base = True
        icfg.make_default_prim = False
        icfg.create_physics_scene = False
        omni.kit.commands.execute("URDFParseAndImportFile", urdf_path=c["URDF_PATH"], import_config=icfg)
        await self._frames(5)
        if not self._stage.GetPrimAtPath(c["ROBOT_PRIM"]).IsValid():
            raise RuntimeError(f"URDF imported but {c['ROBOT_PRIM']} does not exist; set CFG['ROBOT_PRIM']")

    def _ensure_physics_scene(self):
        for prim in self._stage.Traverse():
            if prim.IsA(UsdPhysics.Scene):
                return
        if self._timeline.is_playing():
            _warn("no PhysicsScene in stage; stop the timeline and re-run")
            return
        scene = UsdPhysics.Scene.Define(self._stage, Sdf.Path("/physicsScene"))
        scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
        scene.CreateGravityMagnitudeAttr().Set(9.81 / self._mpu)
        _log("created /physicsScene")

    def _resolve_ee_prim(self):
        c = self.cfg
        prim = self._stage.GetPrimAtPath(c["EE_PRIM"])
        if prim.IsValid():
            return prim
        for p in Usd.PrimRange(self._stage.GetPrimAtPath(c["ROBOT_PRIM"])):
            if p.GetName() == c["EE_FRAME"]:
                c["EE_PRIM"] = str(p.GetPath())
                _warn(f"end-effector found at {c['EE_PRIM']}")
                return p
        raise RuntimeError(f"end-effector prim {c['EE_FRAME']} not found under {c['ROBOT_PRIM']}")

    def _load_keypoints(self):
        c = self.cfg
        path = os.path.join(c["JSON_DIR"], c["JSON_FILE"])
        if not os.path.isfile(path):
            path = None
            for root, _, files in os.walk(c["JSON_DIR"]):
                if c["JSON_FILE"] in files:
                    path = os.path.join(root, c["JSON_FILE"])
                    break
        if path is None:
            raise FileNotFoundError(f"{c['JSON_FILE']} not found in {c['JSON_DIR']}")
        with open(path, "r") as f:
            data = json.load(f)
        grasps = _points_from(_find_key(data, ("grasp_points", "grasp_point", "grasps")))
        places = _points_from(_find_key(data, ("placement_candidates", "placement_points", "place_points")))
        obstacles = _points_from(_find_key(data, ("obstacle_points", "obstacles")))
        if len(grasps) == 0:
            raise RuntimeError("no grasp_points in keypoints JSON")
        if len(places) == 0:
            raise RuntimeError("no placement_candidates in keypoints JSON")
        size = _image_size_from(data)
        if size is None:
            size = tuple(c["DEFAULT_IMAGE_SIZE"])
            _warn(f"image size not in JSON, assuming {size[0]}x{size[1]}")
        W, H = size
        allrec = grasps + places + obstacles
        crop = data.get("crop") if isinstance(data, dict) else None
        space = str(data.get("coordinate_space", "")).lower() if isinstance(data, dict) else ""
        if "crop" in space and isinstance(crop, dict):
            x0, y0 = float(crop.get("x0", 0.0)), float(crop.get("y0", 0.0))
            for r in allrec:
                r["uv"] = (r["uv"][0] + x0, r["uv"][1] + y0)
            _log(f"keypoints are in crop space; shifted by ({x0:.0f}, {y0:.0f})")
        elif all(0.0 <= r["uv"][0] <= 1.0 and 0.0 <= r["uv"][1] <= 1.0 for r in allrec):
            for r in allrec:
                r["uv"] = (r["uv"][0] * W, r["uv"][1] * H)
            _log("keypoints look normalised; scaled to pixels")
        for r in allrec:
            if not (0.0 <= r["uv"][0] < W and 0.0 <= r["uv"][1] < H):
                _warn(f"keypoint {r['uv']} ('{r['label']}') is outside the {W}x{H} image")
        _log(f"loaded {len(grasps)} grasp point(s), {len(places)} placement candidate(s), "
             f"{len(obstacles)} obstacle point(s) from {path} (image {W}x{H})")
        return {"grasps": grasps, "places": places, "obstacles": obstacles, "size": (W, H),
                "K": _intrinsics_from(data)}

    def _load_joint_limits(self):
        n = len(self._cs_names)
        lo = np.full(n, -math.pi)
        hi = np.full(n, math.pi)
        self._pitch_limits = (-math.pi, math.pi)
        try:
            lim = self._joints.limits(list(self._cs_to_dof))
            lo = np.array([a for a, _ in lim], dtype=float)
            hi = np.array([b for _, b in lim], dtype=float)
            if self._pitch_dof_extra is not None:
                self._pitch_limits = tuple(self._joints.limits([self._pitch_dof_extra])[0])
        except Exception as e:
            _warn(f"could not read joint limits ({e}); using +/-pi")
        bad = ~np.isfinite(lo) | ~np.isfinite(hi) | (hi <= lo)
        lo[bad], hi[bad] = -math.pi, math.pi
        self._lo, self._hi = lo, hi

    # ------------------------------------------------------- timeline helpers
    def _stage_id(self):
        try:
            sid = int(omni.usd.get_context().get_stage_id())
            if sid > 0:
                return sid
        except Exception:
            pass
        from pxr import UsdUtils
        cache = UsdUtils.StageCache.Get()
        sid = cache.GetId(self._stage)
        if not sid.IsValid():
            sid = cache.Insert(self._stage)
        return int(sid.ToLongInt())

    def _physx_attached_stage(self):
        try:
            return int(omni.physx.get_physx_simulation_interface().get_attached_stage())
        except Exception:
            return None                      # query not available in this build

    async def _wait_physics_steps(self, n, max_frames):
        start = self._physics_steps
        for _ in range(int(max_frames)):
            if self._physics_steps - start >= n:
                return True
            await self._app.next_update_async()
            if self._shutting_down:
                raise _Abort("shutdown requested")
        return self._physics_steps - start >= n

    async def _ensure_playing(self):
        if not self._timeline.is_playing():
            _log("starting timeline")
            self._timeline.play()
        for _ in range(600):
            await self._app.next_update_async()
            if self._timeline.is_playing():
                break
        if not self._timeline.is_playing():
            raise RuntimeError("timeline did not start")
        # Play is asynchronous: PhysX parses the stage, cooks colliders and attaches
        # the stage over the next frames. Wait for real physics steps, not app frames.
        if not await self._wait_physics_steps(10, 1800):
            raise RuntimeError("timeline is playing but PhysX is not stepping; check that the stage "
                               "has a PhysicsScene and that simulation is enabled")
        _log(f"PhysX is stepping (attached stage id {self._physx_attached_stage()}, "
             f"USD context stage id {self._stage_id()})")

    def _create_own_view(self, stage_id):
        import omni.physics.tensors as tensors
        try:
            view = tensors.create_simulation_view("numpy", stage_id=stage_id)
        except TypeError:                    # older signature without stage_id
            view = tensors.create_simulation_view("numpy")
        if view is None:
            raise RuntimeError("create_simulation_view returned None")
        view.set_subspace_roots("/")
        return view

    async def _tensor_backend(self, view, own, label):
        art = self._Articulation(prim_path=self.cfg["ROBOT_PRIM"], name=f"dobot_pnp_{label}")
        art.initialize(physics_sim_view=view)
        await self._wait_physics_steps(2, 120)
        backend = _TensorJoints(art, self._ArticulationAction, view if own else None)
        q = backend.positions()
        if q is None or q.size == 0:
            backend.close()
            raise RuntimeError("articulation returned no joint state")
        return backend

    async def _create_joint_backend(self):
        stage_id = self._stage_id()
        errors = []

        # 1. Our own CPU tensor view, bound to the explicit stage id once PhysX has attached it.
        for attempt in range(4):
            attached = self._physx_attached_stage()
            if attached is not None and attached != stage_id:
                errors.append(f"PhysX attached stage {attached} != {stage_id}")
                await self._wait_physics_steps(30, 300)
                continue
            view = None
            try:
                view = self._create_own_view(stage_id)
                backend = await self._tensor_backend(view, True, f"own{attempt}")
                _log(f"joint backend: {backend.kind} (own numpy view, stage id {stage_id})")
                return backend
            except Exception as e:
                errors.append(str(e))
                if view is not None:
                    try:
                        view.invalidate()
                    except Exception:
                        pass
                await self._wait_physics_steps(30, 300)

        # 2. The view Isaac Sim's SimulationManager creates on Play.
        try:
            from isaacsim.core.simulation_manager import SimulationManager
            view = SimulationManager.get_physics_sim_view()
            if view is not None:
                backend = await self._tensor_backend(view, False, "simmgr")
                _log(f"joint backend: {backend.kind} (SimulationManager view)")
                return backend
            errors.append("SimulationManager has no physics view")
        except Exception as e:
            errors.append(f"SimulationManager: {e}")

        # 3. No tensor API: USD drive targets + live link poses.
        _warn("tensor API unavailable (" + "; ".join(errors[-3:]) + "); falling back to USD drives")
        backend = _UsdDriveJoints(self._stage, self.cfg["ROBOT_PRIM"], self._physx_pose)
        if len(backend.dof_names) == 0:
            raise RuntimeError("no revolute joints found under the robot prim")
        no_drive = backend.missing_drives(self._cs_names + [self.cfg["JOINT_PITCH"]])
        if no_drive:
            raise RuntimeError(f"joints without an angular drive: {no_drive}")
        if backend.positions() is None:
            raise RuntimeError("cannot read live link poses from PhysX or Fabric")
        _log(f"joint backend: {backend.kind}")
        return backend

    def _on_timeline_stop(self, _event):
        self.shutdown("timeline stopped")

    def _on_physics_step(self, dt):
        self._sim_clock += float(dt)
        self._physics_steps += 1
        if self._latch is not None and self._latch["active"]:
            try:
                self.sync_vacuum_payload()
            except Exception as e:
                self._sync_errors += 1
                if self._sync_errors <= 3:
                    _warn(f"sync_vacuum_payload failed: {e}")

    async def _tick(self):
        await self._app.next_update_async()
        if self._shutting_down:
            raise _Abort("shutdown requested")
        if self._viz is not None and self.cfg["VIZ_LIVE_TRAIL"] and self._q_cmd is not None:
            self._viz_ticks += 1
            if self._viz_ticks % max(1, int(self.cfg["VIZ_TRAIL_EVERY"])) == 0:
                try:
                    q = self._q_meas()
                    if q is not None:
                        self._viz.trail_point(self._b2w(self._fk_tip(q)))
                except Exception:
                    pass
        if self._contact_hit is not None and self._watch_contacts:
            other, self._contact_hit = self._contact_hit, None
            q = self._q_meas()
            if q is not None:
                self._command(q)                 # freeze where we are
            raise _ContactStop(f"held payload touched {other} during transit; arm stopped")
        while not self._timeline.is_playing():
            if self._timeline.is_stopped():
                raise _Abort("timeline stopped")
            await self._app.next_update_async()
            if self._shutting_down:
                raise _Abort("shutdown requested")

    async def _frames(self, n):
        for _ in range(int(n)):
            await self._app.next_update_async()
            if self._shutting_down:
                raise _Abort("shutdown requested")

    async def _wait_sim(self, seconds):
        t0 = self._sim_clock
        while self._sim_clock - t0 < seconds:
            await self._tick()

    # ------------------------------------------------------ pose sources
    def _physx_pose(self, path):
        """Live world pose (metres) of a rigid body / link: PhysX first, then Fabric."""
        T = self._pose_from_physx(path)
        if T is None:
            T = self._pose_from_fabric(path)
        return T

    def _pose_from_physx(self, path):
        try:
            r = omni.physx.get_physx_interface().get_rigidbody_transformation(path)
        except Exception:
            return None
        if r is None or not r.get("ret_val", False):
            return None
        p, q = r["position"], r["rotation"]       # q is (x, y, z, w)
        R = _quat_wxyz_to_R(q[3], q[0], q[1], q[2])
        return _make_T(R, np.array([p[0], p[1], p[2]], dtype=float) * self._mpu)

    def _pose_from_fabric(self, path):
        try:
            from usdrt import Rt
            if self._rt_stage is None:
                from usdrt import Usd as RtUsd
                self._rt_stage = RtUsd.Stage.Attach(self._stage_id())
            prim = self._rt_stage.GetPrimAtPath(path)
            if not prim or not prim.IsValid():
                return None
            xf = Rt.Xformable(prim)
            if hasattr(xf, "GetFabricHierarchyWorldMatrixAttr"):
                attr = xf.GetFabricHierarchyWorldMatrixAttr()
                m = attr.Get() if attr and attr.IsValid() else None
                if m is not None:
                    M = np.array([[m[i][j] for j in range(4)] for i in range(4)], dtype=float).T
                    M[:3, 3] *= self._mpu
                    M[:3, :3] = _orthonormalize(M[:3, :3])
                    return M
            pa, oa = xf.GetWorldPositionAttr(), xf.GetWorldOrientationAttr()
            if pa and oa and pa.IsValid() and oa.IsValid():
                p, q = pa.Get(), oa.Get()
                if p is not None and q is not None:
                    im = q.GetImaginary()
                    R = _quat_wxyz_to_R(float(q.GetReal()), float(im[0]), float(im[1]), float(im[2]))
                    return _make_T(R, np.array([p[0], p[1], p[2]], dtype=float) * self._mpu)
        except Exception:
            return None
        return None

    def _usd_world(self, prim, rigid=True):
        M = _gf_to_np(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
        M[:3, 3] *= self._mpu
        if rigid:
            M[:3, :3] = _orthonormalize(M[:3, :3])
        return M

    def _q_meas(self):
        q = self._joints.positions() if self._joints is not None else None
        if q is None:
            return None
        return q[self._cs_to_dof]

    def _fk_frame(self, q):
        pos, rot = self._lula.compute_forward_kinematics(self.cfg["EE_FRAME"], np.asarray(q, dtype=float))
        return _make_T(np.asarray(rot, dtype=float), np.asarray(pos, dtype=float))

    def _fk_tip(self, q):
        return _apply(self._fk_frame(q), self._tip_local)

    def _w2b(self, p):
        return _apply(self._T_base_world, p)

    def _b2w(self, p):
        return _apply(self._T_world_base, p)

    def _ee_world(self):
        """World pose of Lula's link_suction from the measured joints (same frame the IK targets)."""
        q = self._q_meas()
        if q is None:
            return None
        return self._T_world_base @ self._fk_frame(q)

    def _tip_base_meas(self):
        q = self._q_meas()
        if q is None:
            return None
        return self._fk_tip(q)

    def _set_base_and_tool(self, q0, ee_prim):
        c = self.cfg
        base_path = c["BASE_PRIM"]
        if base_path is None:
            cand = c["ROBOT_PRIM"].rstrip("/") + "/base_link"
            base_path = cand if self._stage.GetPrimAtPath(cand).IsValid() else c["ROBOT_PRIM"]
        T_base = self._usd_world(self._stage.GetPrimAtPath(base_path))
        self._T_world_base = T_base
        self._T_base_world = _inv_T(T_base)
        _log(f"robot base frame = static USD transform of {base_path} at {np.round(T_base[:3, 3] * 1e3, 1)} mm")

        # Lula's link_suction at the start pose vs the simulated one.
        T_fk = T_base @ self._fk_frame(q0)
        T_sim = self._physx_pose(c["EE_PRIM"])
        src = "PhysX"
        if T_sim is None:
            T_sim, src = self._usd_world(ee_prim), "USD"
        gap = T_sim[:3, 3] - T_fk[:3, 3]
        _log(f"simulated link_suction ({src}) vs Lula FK: {np.round(gap * 1e3, 1)} mm "
             f"(|d| = {np.linalg.norm(gap) * 1e3:.1f} mm)")
        T_link = T_sim if c["MATCH_SIM_EE"] else T_fk

        # Tool point and nozzle axis, both in Lula's link_suction frame.
        axis_L = None
        if isinstance(c["TOOL_OFFSET_M"], str):
            nozzle = self._measure_nozzle(ee_prim, T_link)
            if nozzle is not None:
                tip_w, axis_w, length = nozzle
                self._tip_local = _apply(_inv_T(T_fk), tip_w)
                axis_L = T_fk[:3, :3].T @ axis_w
                _log(f"suction nozzle from the mesh: length {length * 1e3:.1f} mm, axis "
                     f"{np.round(axis_L, 3)} and tip {np.round(self._tip_local * 1e3, 1)} mm in Lula's "
                     f"{c['EE_FRAME']} frame")
            else:
                self._tip_local = _apply(_inv_T(T_fk), T_link[:3, 3])
                _warn("could not measure the suction mesh; targeting the link_suction origin")
        else:
            self._tip_local = np.asarray(c["TOOL_OFFSET_M"], dtype=float)
            n = np.linalg.norm(self._tip_local)
            axis_L = self._tip_local / n if n > 1e-6 else None
            _log(f"tool offset from CFG: {np.round(self._tip_local * 1e3, 1)} mm in Lula's {c['EE_FRAME']} frame")

        # joint_6 bias that makes the nozzle point straight down.
        if self._i6 is not None:
            j6_axis = self._fk_axis_local_j6()
            if axis_L is not None and j6_axis is not None:
                off = math.degrees(math.asin(min(1.0, abs(float(axis_L @ j6_axis)))))
                if off > 5.0:
                    _warn(f"the nozzle axis is {off:.0f} deg out of the plane joint_6 can rotate it in, so it "
                          f"cannot point exactly straight down")
            if c["PITCH_BIAS"] == "auto":
                if axis_L is not None:
                    self._pitch_bias, tilt = self._solve_pitch_bias(axis_L)
                    _log(f"joint_6 = {self._pitch_sign:+.0f} * (joint_2 + joint_5) {math.degrees(self._pitch_bias):+.1f} deg "
                         f"keeps the nozzle within {tilt:.1f} deg of straight down")
                else:
                    self._pitch_bias = q0[self._i6] - self._pitch_sign * (q0[self._i2] + q0[self._i5])
                    _warn(f"no nozzle axis; keeping the start pose's cup orientation "
                          f"(joint_6 bias {math.degrees(self._pitch_bias):+.1f} deg)")
            else:
                self._pitch_bias = float(c["PITCH_BIAS"])

    def _fk_axis_local_j6(self):
        """joint_6 rotation axis in Lula's link_suction frame (from FK finite difference)."""
        q = np.zeros(len(self._cs_names))
        R0 = self._fk_frame(q)[:3, :3]
        q[self._i6] = 1e-3
        Rr = R0.T @ self._fk_frame(q)[:3, :3]
        w = np.array([Rr[2, 1] - Rr[1, 2], Rr[0, 2] - Rr[2, 0], Rr[1, 0] - Rr[0, 1]])
        n = np.linalg.norm(w)
        return w / n if n > 0.0 else None

    def _measure_nozzle(self, ee_prim, T_link):
        """(tip_world, axis_world, length) of the suction nozzle mesh placed on T_link, or None."""
        t = Usd.TimeCode.Default()
        xc = UsdGeom.XformCache(t)
        M_link_inv = np.linalg.inv(_gf_to_np(xc.GetLocalToWorldTransform(ee_prim)))
        pts = []
        for p in Usd.PrimRange(ee_prim, Usd.TraverseInstanceProxies()):
            if not p.IsA(UsdGeom.Gprim):
                continue
            tris = _UsdGeomRaycaster._triangles(p, t)
            if tris is None:
                continue
            M = M_link_inv @ _gf_to_np(xc.GetLocalToWorldTransform(p))
            v = tris.reshape(-1, 3)
            pts.append((v @ M[:3, :3].T + M[:3, 3]) * self._mpu)     # link-local, metres
        if len(pts) == 0:
            return None
        V = np.unique(np.round(np.concatenate(pts), 6), axis=0)
        Vw = V @ T_link[:3, :3].T + T_link[:3, 3]
        o = T_link[:3, 3]
        _, _, Vt = np.linalg.svd(Vw - Vw.mean(axis=0), full_matrices=False)
        a = Vt[0]
        proj = (Vw - o) @ a
        if abs(proj.min()) > abs(proj.max()):
            a, proj = -a, -proj
        length = float(proj.max())
        if not (0.01 <= length <= 0.3):
            return None
        tip = Vw[proj >= length - 0.003].mean(axis=0)
        axis = tip - o
        axis = axis / np.linalg.norm(axis)
        return tip, axis, length

    def _solve_pitch_bias(self, axis_L):
        """Bias b in joint_6 = s*(j2 + j5) + b minimising the nozzle's tilt from base -Z."""
        down = np.array([0.0, 0.0, -1.0])
        n = len(self._cs_names)
        samples = [(0.3, 0.6), (0.8, 1.2), (1.2, 0.4), (0.5, 1.8), (1.0, 1.0)]

        def tilt(b):
            worst = 0.0
            for a2, a5 in samples:
                q = np.zeros(n)
                q[self._i2], q[self._i5] = a2, a5
                q[self._i6] = self._pitch_sign * (a2 + a5) + b
                d = self._fk_frame(q)[:3, :3] @ axis_L
                worst = max(worst, math.degrees(math.acos(max(-1.0, min(1.0, float(d @ down))))))
            return worst

        grid = np.radians(np.arange(-180.0, 180.0, 1.0))
        vals = [tilt(b) for b in grid]
        b = float(grid[int(np.argmin(vals))])
        for step in (0.25, 0.05, 0.01):
            cands = b + np.radians(np.arange(-1.0, 1.0 + 1e-9, step))
            vals = [tilt(x) for x in cands]
            b = float(cands[int(np.argmin(vals))])
        return _wrap_pi(b), min(vals)

    # ------------------------------------------------------------ perception
    def _intrinsics(self):
        W, H = self._keypoints["size"]
        K = self._keypoints["K"]
        if K is not None:
            return K
        cam = UsdGeom.Camera(self._stage.GetPrimAtPath(self.cfg["CAMERA_PRIM"]))
        f = float(cam.GetFocalLengthAttr().Get())
        ha = float(cam.GetHorizontalApertureAttr().Get())
        va = cam.GetVerticalApertureAttr().Get()
        hoff = cam.GetHorizontalApertureOffsetAttr().Get() or 0.0
        voff = cam.GetVerticalApertureOffsetAttr().Get() or 0.0
        fx = f / ha * W
        fy = fx                                  # square pixels (Isaac Sim render products)
        cx = W * 0.5 + float(hoff) / ha * W
        cy = H * 0.5 - float(voff) / ha * W
        if va is not None and abs(float(va) / ha - H / W) > 0.01 and not self._aperture_warned:
            self._aperture_warned = True
            _log("camera vertical aperture differs from the image aspect; using square pixels "
                 "(horizontal-aperture fit, as Isaac Sim render products do)")
        return fx, fy, cx, cy

    def _pixel_ray(self, u, v):
        fx, fy, cx, cy = self._intrinsics()
        off = self.cfg["PIXEL_CENTER_OFFSET"]
        # USD camera looks down -Z with +Y up; image v grows downward.
        d_cam = np.array([(u + off - cx) / fx, -(v + off - cy) / fy, -1.0])
        T_cam = self._usd_world(self._stage.GetPrimAtPath(self.cfg["CAMERA_PRIM"]))
        d = T_cam[:3, :3] @ d_cam
        return T_cam[:3, 3].copy(), d / np.linalg.norm(d)

    def _project_world_to_pixel(self, p_world):
        """Inverse of _pixel_ray: project a world-space point into this camera's pixel frame."""
        cam_prim = self._stage.GetPrimAtPath(self.cfg["CAMERA_PRIM"])
        if not cam_prim.IsValid():
            return None
        T_cam = self._usd_world(cam_prim)
        p_cam = T_cam[:3, :3].T @ (np.asarray(p_world, dtype=float) - T_cam[:3, 3])
        if not np.all(np.isfinite(p_cam)) or p_cam[2] >= -1e-6:
            return None
        fx, fy, cx, cy = self._intrinsics()
        off = float(self.cfg["PIXEL_CENTER_OFFSET"])
        u = fx * p_cam[0] / (-p_cam[2]) + cx - off
        v = cy - fy * p_cam[1] / (-p_cam[2]) - off
        W, H = self._keypoints["size"]
        if not (0.0 <= u < W and 0.0 <= v < H):
            return None
        return float(u), float(v)

    def _projected_top_pixel(self, prim_path):
        """Pixel corresponding to the centre of a prim's top AABB face."""
        prim = self._stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return None
        box = self._bbox_world(prim)
        if box is None:
            return None
        lo, hi = box
        extent = hi - lo
        # Back off a fraction from the exact extreme to avoid numerical edge hits.
        p_world = np.array([(lo[0] + hi[0]) * 0.5,
                            (lo[1] + hi[1]) * 0.5,
                            hi[2] - max(0.0005, min(0.002, extent[2] * 0.02))])
        return self._project_world_to_pixel(p_world)

    def _auto_placement_pixels_near_cone(self, support_z_base):
        """Generate projected tabletop pixels beside the cone as fallback candidates."""
        cone = self._stage.GetPrimAtPath(self.cfg["CONE_PRIM"])
        red = self._stage.GetPrimAtPath(self.cfg["RED_PRIM"])
        if not cone.IsValid() or not red.IsValid():
            return []
        cbox, rbox = self._bbox_world(cone), self._bbox_world(red)
        if cbox is None or rbox is None:
            return []
        clo, chi = cbox
        rlo, rhi = rbox
        c_world = np.array([(clo[0] + chi[0]) * .5, (clo[1] + chi[1]) * .5,
                            (clo[2] + chi[2]) * .5])
        c_base = self._w2b(c_world)
        c_half = (chi - clo) * .5
        r_half = (rhi - rlo) * .5
        dx = max(0.055, float(c_half[0] + r_half[0] + 0.020))
        dy = max(0.055, float(c_half[1] + r_half[1] + 0.020))
        offsets = [(dx, 0.0, "+x"), (-dx, 0.0, "-x"),
                   (0.0, dy, "+y"), (0.0, -dy, "-y"),
                   (dx, dy, "+x+y"), (-dx, dy, "-x+y"),
                   (dx, -dy, "+x-y"), (-dx, -dy, "-x-y")]
        out = []
        used = set()
        for ox, oy, label in offsets:
            p_base = np.array([c_base[0] + ox, c_base[1] + oy, float(support_z_base)])
            uv = self._project_world_to_pixel(self._b2w(p_base))
            if uv is None:
                continue
            key = (round(uv[0], 1), round(uv[1], 1))
            if key in used:
                continue
            used.add(key)
            out.append({"uv": uv, "label": "auto-near-cone-" + label})
        return out

    def _robot_link_paths(self):
        """Prims of the robot's own links, from the URDF link/joint names (never the robot root)."""
        c = self.cfg
        import xml.etree.ElementTree as ET
        root = ET.parse(c["URDF_PATH"]).getroot()
        link_names = {l.get("name") for l in root.findall("link")}
        joint_names = {j.get("name") for j in root.findall("joint")}
        robot = self._stage.GetPrimAtPath(c["ROBOT_PRIM"])
        paths = set()
        for p in Usd.PrimRange(robot, Usd.TraverseInstanceProxies()):
            if p == robot:
                continue
            if p.GetName() in link_names:
                paths.add(str(p.GetPath()))
            if p.GetName() in joint_names and p.IsA(UsdPhysics.Joint):
                j = UsdPhysics.Joint(p)
                for rel in (j.GetBody0Rel(), j.GetBody1Rel()):
                    for t in rel.GetTargets():
                        if str(t) != c["ROBOT_PRIM"]:
                            paths.add(str(t))
        paths.update(c["ROBOT_EXTRA_EXCLUDE"])
        # Keep only the outermost paths.
        paths = sorted(paths, key=len)
        out = []
        for p in paths:
            if not any(_under(p, q) for q in out):
                out.append(p)
        if len(out) == 0:
            raise RuntimeError(f"no URDF link prims found under {c['ROBOT_PRIM']}; is the robot payload loaded?")
        _log(f"robot links excluded from depth: {out}")
        return out

    # ------------------------------------------------------- visual debugging
    def _viz_call(self, fn, *a):
        """Run a display-only helper; it can never affect the run."""
        if self._viz is None:
            return
        try:
            fn(*a)
        except Exception as e:
            carb.log_warn(f"{_TAG} visual debug skipped: {e}")

    def _viz_ray(self, name, u, v, hit, ok=True):
        if not self.cfg["VIZ_CAMERA_RAYS"]:
            return
        o, d = self._pixel_ray(u, v)
        end = hit["position"] if hit is not None else o + d * 1.0
        g = self._viz.groups.get(name)
        strips = (g[1] if g is not None else []) + [[o, end]]
        self._viz.strips(name, strips, _Viz.COLORS["ray"] if (hit is not None and ok) else _Viz.COLORS["ray_miss"], 1.0)

    def _viz_marker(self, name, p_b, color, size=None):
        self._viz.points(name, [self._b2w(p_b)], color, size)

    def _viz_obstacles(self, obstacles):
        strips, tops = [], []
        for o in obstacles:
            hull = o["hull"]
            top = [self._b2w(np.array([x, y, o["top"]])) for x, y in hull]
            bot = [self._b2w(np.array([x, y, max(o["bottom"], 0.0)])) for x, y in hull]
            strips.append(top + [top[0]])
            strips.append(bot + [bot[0]])
            for k in range(0, len(hull), max(1, len(hull) // 8)):
                strips.append([bot[k], top[k]])
            tops.append(self._b2w(np.array([o["c"][0], o["c"][1], o["top"]])))
        self._viz.strips("obstacles", strips, _Viz.COLORS["obstacle"])
        self._viz.points("obstacle_tops", tops, _Viz.COLORS["obstacle"])

    def _viz_plan(self, p_pick, place):
        w = self._b2w
        if "route" in place:
            approach = list(reversed(place["ramp_out"])) + list(reversed(place["lift"])) + [p_pick]
            carry = [p_pick] + list(place["lift"]) + list(place["route"]) + [place["p_rel"]]
            leave = [place["p_rel"], place["route"][-1]] + list(reversed(place["ramp_in"]))
        else:
            up = np.array([p_pick[0], p_pick[1], place["z_carry"]])
            over = np.array([place["p_b"][0], place["p_b"][1], place["z_carry"]])
            approach, carry, leave = [up, p_pick], [p_pick, up, over, place["p_b"]], []
        self._viz.strips("path_free", [[w(p) for p in approach], [w(p) for p in leave]], _Viz.COLORS["free"])
        self._viz.strips("path_carry", [[w(p) for p in carry]], _Viz.COLORS["carry"], 1.5 * self.cfg["VIZ_LINE_WIDTH"])

    def _marker_paths(self):
        """Debug / marker prims that depth must never see: VIZ_ROOT plus anything named like a marker."""
        pats = tuple(p.lower() for p in self.cfg["DEPTH_IGNORE_NAME_PATTERNS"])
        found = []
        for p in Usd.PrimRange.Stage(self._stage, Usd.TraverseInstanceProxies()):
            path = str(p.GetPath())
            if any(_under(path, q) for q in found) or self._is_robot(path):
                continue
            if any(x in p.GetName().lower() for x in pats) and p.IsA(UsdGeom.Imageable):
                found.append(path)
        if found:
            _log(f"marker prims excluded from depth: {found}")
        return [self.cfg["VIZ_ROOT"]] + found

    # ------------------------------------------------------------ depth view
    async def _depth_safe(self, coro):
        """Run a display-only depth step; a failure is logged and never stops the run."""
        try:
            await coro
        except (_Abort, asyncio.CancelledError):
            raise
        except Exception as e:
            _warn(f"depth view skipped: {e}")

    async def _depth_capture(self):
        """Once, before any motion: depth of CAMERA_PRIM (ray-cast, optionally rendered), saved + shown."""
        c = self.cfg
        W, H = self._keypoints["size"]
        t0 = time.time()
        z_ray = await self._depth_raycast(int(W), int(H))
        t_ray = time.time() - t0
        z_ren = None
        if c["DEPTH_RENDERED"]:
            try:
                z_ren = await self._depth_rendered(int(W), int(H))
            except (_Abort, asyncio.CancelledError):
                raise
            except Exception as e:
                _warn(f"rendered depth unavailable ({e}); keeping the ray-cast depth only")
        self._depth = {"raycast": z_ray, "rendered": z_ren, "status": {}, "saved": False}
        ok = np.isfinite(z_ray["full"])
        if np.any(ok):
            lo, hi = np.percentile(z_ray["full"][ok], [1.0, 99.0])
        else:
            lo, hi = 0.0, 1.0
        self._depth["range"] = (float(lo), float(max(hi, lo + 1e-3)))
        _log(f"depth view: {z_ray['grid'][1]}x{z_ray['grid'][0]} rays (every {z_ray['stride']} px) in "
             f"{t_ray:.1f} s; depth {lo:.3f}..{hi:.3f} m, {100.0 * np.mean(~ok):.0f}% no hit; "
             f"the robot's own links are invisible to these rays, as they are to perception")
        if z_ren is not None:
            S = z_ray["sample_px"]
            zr = z_ren[S[0][:, None], S[1][None, :]]
            both = np.isfinite(zr) & np.isfinite(z_ray["grid_z"]) & (zr > 0.0)
            if np.any(both):
                e = np.abs(zr[both] - z_ray["grid_z"][both]) * 1e3
                _log(f"depth view: rendered vs ray-cast depth at the same pixels: median {np.median(e):.1f} mm, "
                     f"{100.0 * np.mean(e <= 5.0):.0f}% within 5 mm (the robot only appears in the rendered "
                     f"image, so its pixels differ by design)")
        self._depth_publish()

    async def _depth_raycast(self, W, H):
        """The pipeline's own depth model on a pixel grid: same intrinsics, same rays, same exclusions."""
        c = self.cfg
        s = max(1, int(c["DEPTH_STRIDE_PX"]))
        us = np.minimum(np.arange(0, W, s) + 0.5 * (s - 1), W - 1)   # block centres (pixel index, like
        vs = np.minimum(np.arange(0, H, s) + 0.5 * (s - 1), H - 1)   # the keypoints); last block may be partial
        fx, fy, cx, cy = self._intrinsics()
        off = c["PIXEL_CENTER_OFFSET"]
        T_cam = self._usd_world(self._stage.GetPrimAtPath(c["CAMERA_PRIM"]))
        o = T_cam[:3, 3].copy()
        fwd = T_cam[:3, :3] @ np.array([0.0, 0.0, -1.0])
        uu, vv = np.meshgrid(us, vs)
        d_cam = np.stack([(uu + off - cx) / fx, -(vv + off - cy) / fy, -np.ones_like(uu)], axis=-1).reshape(-1, 3)
        D = d_cam @ T_cam[:3, :3].T
        D /= np.linalg.norm(D, axis=1, keepdims=True)
        max_d = c["RAY_MAX_DIST_M"]
        dist = await self._depth_usd(o, D, max_d) if self._usd_rc is not None else np.full(len(D), np.inf)
        skip = tuple(self._robot_links) + tuple(self._viz_skip)
        per_yield = len(us) * 8                           # let the app update every 8 rows
        for i in range(len(D)):
            h = self._physx_ray(o, D[i], max_d, skip)
            if h is not None and h["distance"] < dist[i]:
                dist[i] = h["distance"]
            if (i + 1) % per_yield == 0:
                await self._frames(1)
        z = (dist * (D @ fwd)).reshape(len(vs), len(us))  # distance to the image plane, like a depth camera
        full = np.repeat(np.repeat(z, s, axis=0), s, axis=1)[:H, :W]
        sample_px = (np.clip(np.round(vs).astype(int), 0, H - 1), np.clip(np.round(us).astype(int), 0, W - 1))
        return {"full": full, "grid_z": z, "grid": z.shape, "stride": s, "sample_px": sample_px}

    async def _depth_usd(self, o, D, max_d):
        """Batched version of _UsdGeomRaycaster.cast for many rays from one origin (same triangles)."""
        out = np.full(len(D), np.inf)
        with np.errstate(divide="ignore", invalid="ignore"):
            inv_d = 1.0 / D
        for g in self._usd_rc.geoms:
            with np.errstate(invalid="ignore"):
                t1 = (g["bmin"] - 1e-6 - o) * inv_d
                t2 = (g["bmax"] + 1e-6 - o) * inv_d
            t1 = np.where(np.isnan(t1), -np.inf, t1)
            t2 = np.where(np.isnan(t2), np.inf, t2)
            tn, tf = np.minimum(t1, t2).max(axis=1), np.maximum(t1, t2).min(axis=1)
            idx = np.nonzero((tf >= np.maximum(tn, 0.0)) & (tn <= max_d))[0]
            if len(idx) == 0:
                continue
            e1, e2 = g["e1"], g["e2"]
            sv = o - g["v0"]
            A = np.cross(e2, e1)                          # det       = D . (e2 x e1)
            B = np.cross(e2, sv)                          # u * det   = D . (e2 x s)
            Q = np.cross(sv, e1)                          # v * det   = D . (s x e1)
            tq = np.einsum("ij,ij->i", e2, Q)             # t * det   = e2 . (s x e1)
            step = max(1, int(1e6 // max(1, len(e1))))
            for k in range(0, len(idx), step):
                ii = idx[k:k + step]
                Dk = D[ii]
                det = Dk @ A.T
                ok = np.abs(det) > 1e-14
                inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
                u = (Dk @ B.T) * inv
                v = (Dk @ Q.T) * inv
                t = tq[None, :] * inv
                valid = ok & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0) & (t > 1e-6) & (t < max_d)
                out[ii] = np.minimum(out[ii], np.where(valid, t, np.inf).min(axis=1))
                await self._frames(1)
        return out

    async def _depth_rendered(self, W, H):
        """The camera's rendered depth, once: temporary render product + distance_to_image_plane annotator.

        Only plain app updates are awaited (no orchestrator step, no step_async), so physics and the
        control loop keep their own pace; the render product is destroyed before returning.
        """
        c = self.cfg
        self._enable_ext("omni.replicator.core")
        import omni.replicator.core as rep
        rp, ann = None, None
        try:
            rp = rep.create.render_product(c["CAMERA_PRIM"], (W, H))
            ann = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
            ann.attach([rp])
            for k in range(int(c["DEPTH_RENDER_FRAMES"])):
                await self._frames(1)
                if k < 2:
                    continue                              # first frames of a new render product are empty
                d = ann.get_data()
                if isinstance(d, dict):
                    d = d.get("data")
                if d is None:
                    continue
                d = np.asarray(d, dtype=float)
                if d.ndim == 3:
                    d = d[..., 0]
                if d.shape == (H, W) and np.any(np.isfinite(d) & (d > 0.0)):
                    _log(f"depth view: rendered depth after {k + 1} frames")
                    return np.where(np.isfinite(d) & (d > 0.0), d * self._mpu, np.inf)
            _warn(f"rendered depth stayed empty for {c['DEPTH_RENDER_FRAMES']} frames")
            return None
        finally:
            for f in ((lambda: ann.detach([rp])) if ann is not None else None,
                      (lambda: rp.destroy()) if rp is not None else None):
                if f is not None:
                    try:
                        f()
                    except Exception:
                        pass

    def _depth_mark(self, kind, tried, chosen):
        """Record which keypoints were used / rejected (candidates are tried in order) and redraw."""
        if self._depth is None:
            return
        for i in tried:
            if chosen is not None and i == chosen:
                self._depth["status"][(kind, i)] = "ok"
                break
            self._depth["status"][(kind, i)] = "bad"
        try:
            self._depth_publish()
        except Exception as e:
            _warn(f"depth view update skipped: {e}")

    def _depth_overlay(self, rgb):
        kp, st = self._keypoints, self._depth["status"]
        grey = (150, 150, 160)
        cols = {"grasp": (40, 255, 40), "place": (255, 255, 255), "obstacle": (255, 60, 255)}
        text = []
        for kind, key, mark in (("obstacle", "obstacles", "cross"), ("place", "places", "plus"),
                                ("grasp", "grasps", "ring")):
            for i, rec in enumerate(kp[key]):
                u, v = rec["uv"]
                u, v = u + self.cfg["PIXEL_CENTER_OFFSET"], v + self.cfg["PIXEL_CENTER_OFFSET"]
                state = st.get((kind, i))
                col = grey if state == "bad" else cols[kind]
                big = 1.5 if state == "ok" else 1.0
                _draw_mark(rgb, u, v, mark, col, r=(6 if kind == "obstacle" else 9) * big, t=2 * big)
                if state == "ok":
                    _draw_mark(rgb, u, v, "ring", (255, 230, 0), r=17, t=2)
                if kind != "obstacle":
                    tag = {"ok": " used", "bad": " rejected"}.get(state, "")
                    text.append((u + 12, v - 16, f"{'G' if kind == 'grasp' else 'P'}{i}{tag}", col))
        return _draw_text(rgb, text)

    def _depth_legend(self, rgb, title):
        lo, hi = self._depth["range"]
        w = rgb.shape[1]
        bar = np.zeros((34, w, 3), dtype=np.uint8)
        n = min(256, max(16, w // 3))
        ramp = np.linspace(hi, lo, n)[None, :].repeat(12, axis=0)        # far (left) -> near (right)
        bar[4:16, 8:8 + n] = _colorize_depth(ramp, lo, hi, self.cfg["DEPTH_COLORMAP"])
        bar = _draw_text(bar, [(8, 18, f"far {hi:.3f} m", (220, 220, 220)),
                               (max(8, 8 + n - 90), 18, f"near {lo:.3f} m", (220, 220, 220)),
                               (16 + n, 2, f"{title}; black = no hit.  ring G = grasp, + P = placement, "
                                          f"x = obstacle; grey = rejected, yellow ring = used", (220, 220, 220)),
                               (16 + n, 18, f"white lines = depth jumps over {self.cfg['DEPTH_EDGE_MM']:.0f} mm "
                                            f"(object outlines)" if self.cfg["DEPTH_EDGE_MM"] > 0.0 else "",
                                (220, 220, 220))])
        return np.concatenate([rgb, bar], axis=0)

    def _depth_publish(self):
        c, d = self.cfg, self._depth
        lo, hi = d["range"]
        out_dir = os.path.expanduser(c["DEPTH_OUT_DIR"])
        os.makedirs(out_dir, exist_ok=True)
        views = []
        for key, title in (("raycast", "ray-cast depth (what the pipeline perceives; robot links hidden)"),
                           ("rendered", "rendered depth (distance_to_image_plane)")):
            z = d[key]["full"] if key == "raycast" else d[key]
            if z is None:
                continue
            rgb = _colorize_depth(z, lo, hi, c["DEPTH_COLORMAP"])
            if c["DEPTH_EDGE_MM"] > 0.0:
                rgb[_depth_edges(z, _mm(c["DEPTH_EDGE_MM"]))] = (255, 255, 255) if c["DEPTH_COLORMAP"] != "gray" \
                    else (255, 200, 0)
            rgb = self._depth_legend(self._depth_overlay(rgb), title)
            path = os.path.join(out_dir, f"dobot_depth_{key}.png")
            _write_png(path, rgb)
            views.append(rgb)
            if not d["saved"]:
                _log(f"depth view: saved {path}")
        if c["DEPTH_SAVE_NPZ"] and not d["saved"]:
            path = os.path.join(out_dir, "dobot_depth.npz")
            arrs = {"raycast_m": d["raycast"]["full"].astype(np.float32)}
            if d["rendered"] is not None:
                arrs["rendered_m"] = d["rendered"].astype(np.float32)
            np.savez_compressed(path, **arrs)
            _log(f"depth view: raw depth (metres, distance to the image plane, inf = no hit) in {path}")
        d["saved"] = True
        if c["DEPTH_WINDOW"] and views:
            try:
                self._depth_window(np.concatenate(views, axis=1))
            except Exception as e:
                _warn(f"depth window unavailable ({e}); the PNGs are still saved")

    def _depth_window(self, rgb):
        """A plain omni.ui window showing the latest depth view; it stays open after the run."""
        import omni.ui as ui
        h, w = rgb.shape[:2]
        rgba = np.ascontiguousarray(np.concatenate([rgb, np.full((h, w, 1), 255, np.uint8)], axis=-1))
        st = getattr(builtins, "_DOBOT_DEPTH_WINDOW", None)
        if st is None:
            ww = min(w, 1200)
            win = ui.Window("Dobot depth", width=ww, height=int(ww * h / w) + 40)
            with win.frame:
                prov = ui.ByteImageProvider()
                try:
                    ui.ImageWithProvider(prov, fill_policy=ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_FIT)
                except Exception:
                    ui.ImageWithProvider(prov)
            st = {"win": win, "prov": prov}
            builtins._DOBOT_DEPTH_WINDOW = st
        try:
            st["prov"].set_data_array(rgba, [w, h])
        except Exception:
            st["prov"].set_bytes_data(rgba.flatten().tolist(), [w, h])
        st["win"].visible = True

    def _is_robot(self, path):
        return any(_under(path, r) for r in self._robot_links)

    def _physx_ray(self, origin, direction, max_dist, skip_prefixes):
        sq = omni.physx.get_physx_scene_query_interface()
        o = np.asarray(origin, dtype=float).copy()
        d = np.asarray(direction, dtype=float)
        travelled = 0.0
        for _ in range(32):
            hit = sq.raycast_closest(carb.Float3(*(o / self._mpu)), carb.Float3(*d),
                                     (max_dist - travelled) / self._mpu)
            if hit is None or not hit.get("hit", False):
                return None
            pos = np.array([hit["position"][0], hit["position"][1], hit["position"][2]], dtype=float) * self._mpu
            coll = _path_str(hit.get("collision"))
            body = _path_str(hit.get("rigidBody"))
            if any(_under(coll, s) or _under(body, s) for s in skip_prefixes):
                travelled += float(np.linalg.norm(pos - o)) + 1e-4
                if travelled >= max_dist:
                    return None
                o = pos + d * 1e-4
                continue
            n = hit.get("normal")
            normal = np.array([n[0], n[1], n[2]], dtype=float) if n is not None else -d
            return {"position": pos, "normal": normal, "collision": coll, "body": body,
                    "distance": travelled + float(np.linalg.norm(pos - o)), "source": "physx"}
        return None

    def _raycast(self, origin, direction, skip_prefixes=()):
        """Closest surface along a ray (world, metres): PhysX colliders + visible USD geometry."""
        max_dist = self.cfg["RAY_MAX_DIST_M"]
        skip = tuple(self._robot_links) + tuple(self._viz_skip) + tuple(skip_prefixes)
        a = self._physx_ray(origin, direction, max_dist, skip)
        b = self._usd_rc.cast(origin, direction, max_dist, skip) if self._usd_rc is not None else None
        if a is None:
            return b
        if b is None:
            return a
        return a if a["distance"] <= b["distance"] + 1e-4 else b

    def _measure_pixel(self, u, v, skip=()):
        o, d = self._pixel_ray(u, v)
        return self._raycast(o, d, skip)

    def _bbox_world(self, prim):
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
        rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if rng.IsEmpty():
            return None
        return np.array(rng.GetMin(), dtype=float) * self._mpu, np.array(rng.GetMax(), dtype=float) * self._mpu

    def _object_root(self, prim):
        """Smallest-to-largest walk up from a hit prim to the compact object it belongs to.

        Returns None for support surfaces (ground, table) and robot parts.
        """
        limit = self.cfg["OBJECT_MAX_SIZE_M"]
        stop = {self.cfg["ROBOT_PRIM"], "/World", "/"}
        best = None
        p = prim
        while p.IsValid() and not p.IsPseudoRoot() and str(p.GetPath()) not in stop:
            if self._is_robot(str(p.GetPath())):
                return None
            bb = self._bbox_world(p)
            if bb is None or float(np.max(bb[1] - bb[0])) > limit:
                break
            best = p
            if p.HasAPI(UsdPhysics.RigidBodyAPI):
                return p
            p = p.GetParent()
        return best

    def _make_dynamic(self, prim):
        """Give a visible object without physics a rigid body + convex colliders so it can be picked."""
        with Sdf.ChangeBlock():
            UsdPhysics.RigidBodyAPI.Apply(prim)
            if not prim.HasAPI(UsdPhysics.MassAPI):
                UsdPhysics.MassAPI.Apply(prim)
            n = 0
            for g in Usd.PrimRange(prim):
                if not g.IsA(UsdGeom.Gprim) or g.HasAPI(UsdPhysics.CollisionAPI):
                    continue
                UsdPhysics.CollisionAPI.Apply(g)
                if g.IsA(UsdGeom.Mesh):
                    UsdPhysics.MeshCollisionAPI.Apply(g).CreateApproximationAttr().Set(UsdPhysics.Tokens.convexHull)
                n += 1
        _log(f"{prim.GetPath()} had no rigid body; added RigidBodyAPI and {n} collider(s)")

    async def _resolve_target(self, hit):
        """Rigid-body path of the object a depth hit belongs to, creating physics if needed."""
        if hit["body"]:
            return hit["body"]
        prim = self._stage.GetPrimAtPath(hit["collision"])
        if not prim.IsValid():
            return None
        obj = self._object_root(prim)
        if obj is None:
            return None
        path = str(obj.GetPath())
        if not obj.HasAPI(UsdPhysics.RigidBodyAPI):
            self._make_dynamic(obj)
            self._usd_rc.geoms = [g for g in self._usd_rc.geoms if not _under(g["path"], path)]
            await self._wait_physics_steps(10, 600)
            await self._wait_sim(0.5)               # let it settle under gravity before measuring
        if self._pose_from_physx(path) is None:
            _warn(f"PhysX does not report a pose for {path} yet")
        return path

    def _measure_support(self, u, v, target_body, z_top_b):
        """Support-surface height (base frame) from depth samples around the pick pixel."""
        c = self.cfg
        W, H = self._keypoints["size"]
        zs, all_z = [], []
        for r in c["TABLE_RING_RADII_PX"]:
            for k in range(c["TABLE_RING_SAMPLES"]):
                a = 2.0 * math.pi * k / c["TABLE_RING_SAMPLES"]
                uu, vv = u + r * math.cos(a), v + r * math.sin(a)
                if not (0.0 <= uu < W and 0.0 <= vv < H):
                    continue
                hit = self._measure_pixel(uu, vv)
                if hit is None:
                    continue
                z = self._w2b(hit["position"])[2]
                all_z.append(z)
                if _under(hit["body"], target_body) or _under(hit["collision"], target_body):
                    continue
                if z < z_top_b - _mm(1.0):
                    zs.append(z)
            if len(zs) >= c["TABLE_MIN_SAMPLES"]:
                break
        if len(zs) >= 3:
            arr = np.sort(np.array(zs))
            bins = np.round(arr / _mm(2.0))
            vals, counts = np.unique(bins, return_counts=True)
            mode = vals[int(np.argmax(counts))] * _mm(2.0)
            sel = arr[np.abs(arr - mode) <= _mm(2.0)]
            return float(np.median(sel)), all_z, "surrounding depth ring"
        # Fallback 1: the same ray, passing through the target, hits what supports it.
        hit = self._measure_pixel(u, v, skip=(target_body,))
        if hit is not None:
            return float(self._w2b(hit["position"])[2]), all_z, "ray through target"
        # Fallback 2: bottom of the target's bounding box.
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
        rng = cache.ComputeWorldBound(self._stage.GetPrimAtPath(target_body)).ComputeAlignedRange()
        if rng.IsEmpty():
            raise RuntimeError("could not determine the support surface under the object")
        p = np.array(rng.GetMin(), dtype=float) * self._mpu
        return float(self._w2b(p)[2]), all_z, "target bounding box"

    # ---------------------------------------------------------- kinematics
    def _detect_pitch_sign(self):
        """Sign s in joint_6 = s * (joint_2 + joint_5) that keeps link_suction's orientation fixed."""
        cfg = self.cfg["PITCH_SIGN"]
        if cfg != "auto":
            return float(cfg)
        if self._i6 is None:
            _warn("joint_6 not in the Lula c-space; using joint_6 = -(joint_2 + joint_5)")
            return -1.0
        n = len(self._cs_names)
        samples = [(0.3, 0.6), (0.8, 1.2), (1.2, 0.4), (0.5, 1.8)]
        errs = {}
        for s in (-1.0, 1.0):
            q0 = np.zeros(n)
            R0 = self._fk_frame(q0)[:3, :3]
            e = 0.0
            for a, b in samples:
                q = np.zeros(n)
                q[self._i2], q[self._i5] = a, b
                q[self._i6] = s * (a + b)
                Rr = self._fk_frame(q)[:3, :3] @ R0.T
                e = max(e, math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(Rr) - 1.0) / 2.0)))))
            errs[s] = e
        s = min(errs, key=errs.get)
        if errs[s] > 2.0:
            _warn(f"neither joint_6 sign keeps the suction cup orientation fixed (errors {errs}); using {s:+.0f}")
        elif s > 0:
            _warn("in this URDF joint_6 turns the opposite way to joint_2/joint_5, so the level-cup "
                  "constraint is joint_6 = +(joint_2 + joint_5); using that (set CFG['PITCH_SIGN'] to override)")
        _log(f"pitch constraint: joint_6 = {s:+.0f} * (joint_2 + joint_5) "
             f"(cup tilt over test poses: -1 -> {errs[-1.0]:.1f} deg, +1 -> {errs[1.0]:.1f} deg)")
        return s

    def _pitch_value(self, q2, q5, lo=-math.pi, hi=math.pi, ref=None):
        """joint_6 from the parallelogram rule, wrapped by 2*pi into its limits (closest to ref)."""
        v = self._pitch_sign * (q2 + q5) + self._pitch_bias
        options = [v + k for k in (0.0, -2.0 * math.pi, 2.0 * math.pi) if lo - 1e-9 <= v + k <= hi + 1e-9]
        if len(options) == 0:
            return v
        if ref is None:
            return options[0]
        return min(options, key=lambda x: abs(x - ref))

    def _constrain(self, q):
        q = np.array(q, dtype=float)
        if self._i6 is not None:
            ref = self._q_cmd[self._i6] if self._q_cmd is not None else None
            q[self._i6] = self._pitch_value(q[self._i2], q[self._i5], self._lo[self._i6], self._hi[self._i6], ref)
        return q

    def _within_limits(self, q):
        if np.any(q < self._lo - 1e-6) or np.any(q > self._hi + 1e-6):
            return False
        if self._pitch_dof_extra is not None:
            p = self._pitch_value(q[self._i2], q[self._i5], *self._pitch_limits)
            if p < self._pitch_limits[0] - 1e-6 or p > self._pitch_limits[1] + 1e-6:
                return False
        return True

    def _dls(self, target_b, seed):
        tol = _mm(self.cfg["IK_TOL_MM"])
        q = self._constrain(seed)
        free = self._free
        lam2 = 0.005 ** 2
        h = 1e-5
        for _ in range(self.cfg["IK_MAX_ITERS"]):
            p = self._fk_tip(q)
            e = target_b - p
            if np.linalg.norm(e) < tol:
                return q if self._within_limits(q) else None
            J = np.zeros((3, len(free)))
            for k, i in enumerate(free):
                qd = q.copy()
                qd[i] += h
                J[:, k] = (self._fk_tip(self._constrain(qd)) - p) / h
            dq = J.T @ np.linalg.solve(J @ J.T + lam2 * np.eye(3), e)
            n = np.linalg.norm(dq)
            if n > 0.25:
                dq *= 0.25 / n
            q[free] = np.clip(q[free] + dq, self._lo[free], self._hi[free])
            q = self._constrain(q)
        return None

    def _ik_tip_base(self, target_b, seed=None):
        """Joint solution placing the suction tip at target_b (base frame, metres)."""
        target_b = np.asarray(target_b, dtype=float)
        if seed is None:
            seed = self._q_cmd
        seeds = [self._constrain(seed)]
        # Seed with joint_1 rotated toward the target azimuth.
        p = self._fk_tip(seeds[0])
        d_az = _wrap_pi(math.atan2(target_b[1], target_b[0]) - math.atan2(p[1], p[0]))
        qa = seeds[0].copy()
        qa[self._i1] += 1e-3
        pa = self._fk_tip(self._constrain(qa))
        sign = 1.0 if _wrap_pi(math.atan2(pa[1], pa[0]) - math.atan2(p[1], p[0])) >= 0.0 else -1.0
        qa = seeds[0].copy()
        j1 = qa[self._i1] + sign * d_az
        for k in (0.0, -2.0 * math.pi, 2.0 * math.pi):   # keep joint_1 inside its limits
            if self._lo[self._i1] <= j1 + k <= self._hi[self._i1]:
                j1 += k
                break
        qa[self._i1] = j1
        seeds.append(self._constrain(qa))
        # Same azimuth with the arm at mid-range, then the home pose.
        qm = qa.copy()
        for i in (self._i2, self._i5):
            qm[i] = 0.5 * (self._lo[i] + self._hi[i])
        seeds.append(self._constrain(qm))
        if getattr(self, "_q_home", None) is not None:
            seeds.append(self._constrain(self._q_home))
        # Lula's own IK as a last seed (frame target approximated from the seed's tip offset).
        try:
            R = self._fk_frame(seeds[0])[:3, :3]
            q_l, ok = self._lula.compute_inverse_kinematics(
                self.cfg["EE_FRAME"], target_b - R @ self._tip_local, warm_start=seeds[0])
            if ok and q_l is not None:
                seeds.append(self._constrain(q_l))
        except Exception:
            pass
        for s in seeds:
            q = self._dls(target_b, s)
            if q is not None:
                return q
        return None

    def _plan_points(self, pts_b, seed, label):
        qs, q = [], self._constrain(seed)
        for p in pts_b:
            qn = self._ik_tip_base(p, q)
            if qn is None:
                raise _PlanningError(f"{label}: no IK solution for tip at {np.round(np.asarray(p) * 1e3, 1)} mm (base frame)")
            if np.max(np.abs(qn - q)) > self.cfg["MAX_JOINT_JUMP_RAD"]:
                raise _PlanningError(f"{label}: IK branch jump near {np.round(np.asarray(p) * 1e3, 1)} mm")
            qs.append(qn)
            q = qn
        return qs

    def _line_points(self, p0, p1):
        p0, p1 = np.asarray(p0, dtype=float), np.asarray(p1, dtype=float)
        n = max(1, int(math.ceil(np.linalg.norm(p1 - p0) / _mm(self.cfg["CART_STEP_MM"]))))
        return [p0 + (p1 - p0) * (i / n) for i in range(1, n + 1)]

    def _footprint(self, path):
        """Convex XY footprint (base frame), top and bottom of an object from its real geometry.

        Meshes use their vertices, implicit shapes their exact outline; the USD
        pose is corrected to the live PhysX / Fabric pose (objects settle or get
        pushed after Play).
        """
        prim = self._stage.GetPrimAtPath(path)
        if not prim.IsValid():
            return None
        t = Usd.TimeCode.Default()
        xcache = UsdGeom.XformCache(t)
        pts = []
        for p in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
            if not p.IsA(UsdGeom.Gprim):
                continue
            img = UsdGeom.Imageable(p)
            if img.ComputeVisibility(t) == UsdGeom.Tokens.invisible or \
                    img.ComputePurpose() in (UsdGeom.Tokens.guide, UsdGeom.Tokens.proxy):
                continue
            L = _shape_points(p, t)
            if L is None or len(L) == 0:
                continue
            if len(L) > 20000:
                L = L[::len(L) // 20000 + 1]
            M = _gf_to_np(xcache.GetLocalToWorldTransform(p))
            pts.append((L @ M[:3, :3].T + M[:3, 3]) * self._mpu)
        if len(pts) == 0:
            return None
        W = np.concatenate(pts)
        T_live = self._physx_pose(path)
        if T_live is not None:
            C = T_live @ _inv_T(self._usd_world(prim))
            W = W @ C[:3, :3].T + C[:3, 3]
        Tb = self._T_base_world
        B = W @ Tb[:3, :3].T + Tb[:3, 3]
        hull = _hull2d(B[:, :2])
        if len(hull) < 3:
            return None
        c = hull.mean(axis=0)
        return {"hull": hull, "c": c, "r": float(np.max(np.linalg.norm(hull - c, axis=1))),
                "top": float(B[:, 2].max()), "bottom": float(B[:, 2].min()),
                "size": hull.max(axis=0) - hull.min(axis=0)}

    def _build_obstacles(self, target):
        """Obstacle footprints (base frame) from the obstacle_points pixels, via depth."""
        ignore = tuple(x.lower() for x in self.cfg["IGNORE_OBSTACLE_LABELS"])
        groups, skipped = {}, set()
        for rec in self._keypoints["obstacles"]:
            label = rec["label"]
            if any(x in label.lower() for x in ignore):
                skipped.add(label)
                continue
            hit = self._measure_pixel(*rec["uv"])
            self._viz_call(self._viz_ray, "rays_obstacles", rec["uv"][0], rec["uv"][1], hit)
            if hit is None or _under(hit["collision"], target) or _under(hit["body"], target):
                continue
            root = self._object_root(self._stage.GetPrimAtPath(hit["collision"]))
            if root is None:
                continue                            # boundary pixel fell on the table / robot
            g = groups.setdefault(str(root.GetPath()), {"labels": set(), "z": []})
            g["labels"].add(label)
            g["z"].append(float(self._w2b(hit["position"])[2]))
        if skipped:
            _log(f"not treated as obstacles (robot / support labels): {sorted(skipped)}")
        out = []
        for path, g in groups.items():
            fp = self._footprint(path)
            if fp is None:
                continue
            fp.update({"path": path, "label": "/".join(sorted(g["labels"])), "top": max(fp["top"], max(g["z"]))})
            out.append(fp)
            _log(f"obstacle '{fp['label']}' -> {path}: footprint {fp['size'][0] * 1e3:.0f} x "
                 f"{fp['size'][1] * 1e3:.0f} mm centred {np.round(fp['c'] * 1e3, 0)} mm "
                 f"(enclosing radius {fp['r'] * 1e3:.0f} mm), top {fp['top'] * 1e3:.0f} mm")
        return out

    def _ceiling_table(self, rs, az, z_cap, ceil_m, seed):
        """Tip height to ride at each radius in rs: CEILING_MARGIN_MM under the arm's ceiling, capped.

        The ceiling is first estimated by a forward-kinematics sweep of joint_2 /
        joint_5 (cheap), then each radius is confirmed by IK at the ceiling itself,
        stepping down 1 mm at a time, so every height returned has the full margin
        of reach above it. NaN where nothing is reachable.
        """
        i2, i5 = self._i2, self._i5
        dr = float(rs[1] - rs[0])
        est = np.full(len(rs), -np.inf)
        q = self._constrain(seed).copy()
        step = math.radians(1.0)
        for a2 in np.arange(self._lo[i2], self._hi[i2] + 1e-9, step):
            for a5 in np.arange(self._lo[i5], self._hi[i5] + 1e-9, step):
                q[i2], q[i5] = a2, a5
                qc = self._constrain(q)
                if not self._within_limits(qc):
                    continue
                p = self._fk_tip(qc)
                k = int(round((math.hypot(p[0], p[1]) - rs[0]) / dr))
                if 0 <= k < len(rs) and p[2] > est[k]:
                    est[k] = p[2]
        zs = np.full(len(rs), np.nan)
        for k, r in enumerate(rs):
            if not np.isfinite(est[k]):
                continue
            z = min(z_cap + ceil_m, est[k] + _mm(2.0))
            for _ in range(12):
                if self._ik_tip_base(np.array([r * math.cos(az), r * math.sin(az), z]), seed) is not None:
                    zs[k] = min(z_cap, z - ceil_m)
                    break
                z -= _mm(1.0)
        return zs

    def _plan_route(self, p_pick, p_place, hold, payload, obstacles, q_touch, z_floor):
        """Collision-free tip route (base frame) from the touch point to above the release point.

        The Dobot's reach envelope is a surface of revolution: the highest tip
        height depends only on the radius. So the transit is searched on a polar
        (radius, azimuth) grid where the tip rides CEILING_MARGIN_MM under that
        ceiling (capped at the height needed to pass over the tallest obstacle):
        the held object is always as high as it can be, so every cell gets the
        best vertical clearance the arm can give. A cell is free when the held
        object's real footprint, rotated with joint_1, keeps OBSTACLE_MARGIN_MM
        from every obstacle footprint, or its bottom is OBSTACLE_Z_MARGIN_MM above
        that obstacle. Dijkstra finds the shortest free path (it can swing in to
        a small radius to go over an obstacle, or around it, either way round
        within joint_1's limits); the path is then shortcut-smoothed and checked
        with IK point by point.

        Returns (lift_pts, transit_pts, info) or (None, None, reason).
        """
        import heapq
        c = self.cfg
        m, mz = _mm(c["OBSTACLE_MARGIN_MM"]), _mm(c["OBSTACLE_Z_MARGIN_MM"])
        lift_min, ceil_m = _mm(c["LIFT_MIN_MM"]), _mm(c["CEILING_MARGIN_MM"])
        az0 = math.atan2(p_pick[1], p_pick[0])
        hull0, c0, rp = payload["hull0"], payload["c0"], payload["r"]
        z_over = max([z_floor] + [o["top"] + mz + hold for o in obstacles])
        z_cap = max(z_over, z_floor + _mm(c["CARRY_CLEARANCE_MM"] - c["CARRY_MIN_CLEARANCE_MM"]))

        def gap(p, margin):
            """(smallest gap, obstacle) of the held object at tip p, for obstacles it is not above."""
            bottom = p[2] - hold
            a = math.atan2(p[1], p[0]) - az0
            Rm = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
            cc = p[:2] + Rm @ c0
            worst = (np.inf, None)
            for o in obstacles:
                if bottom >= o["top"] + mz:
                    continue
                if np.linalg.norm(cc - o["c"]) >= rp + o["r"] + margin:
                    continue
                g = _poly_gap(p[:2] + hull0 @ Rm.T, o["hull"])
                if g < worst[0]:
                    worst = (g, o)
            return worst

        # Ceiling z(r), cached on the instance (it depends only on the arm).
        dr = _mm(c["ROUTE_GRID_MM"])
        if getattr(self, "_ceil_cache", None) is None or self._ceil_cache[0] != (round(z_cap, 4), round(ceil_m, 4)):
            rs = np.arange(0.06, _mm(c["REACH_MAX_MM"]) + 1e-9, dr)
            zs = self._ceiling_table(rs, az0, z_cap, ceil_m, q_touch)
            self._ceil_cache = ((round(z_cap, 4), round(ceil_m, 4)), rs, zs)
        _, rs, zs = self._ceil_cache

        def z_at(r):
            if r < rs[0] or r > rs[-1]:
                return None
            z = float(np.interp(r, rs, zs))
            k = min(len(rs) - 2, int((r - rs[0]) / dr))
            if np.isnan(zs[k]) or np.isnan(zs[k + 1]):
                return None
            return z

        # Lift / drop: straight up if the arm can, else leaning toward the base.
        def lift_point(p):
            r = math.hypot(p[0], p[1])
            for d_in in c["LIFT_INWARD_MM"]:
                xy = p[:2] * max(0.0, 1.0 - _mm(d_in) / max(r, 1e-6))
                z = z_at(float(np.linalg.norm(xy)))  # from the table, already IK-checked with margin
                if z is not None and z >= p[2] + lift_min and \
                        self._ik_tip_base(np.array([xy[0], xy[1], z]), q_touch) is not None:
                    return np.array([xy[0], xy[1], z])
            for d_in in c["LIFT_INWARD_MM"]:               # fallback: search the vertical line directly
                xy = p[:2] * max(0.0, 1.0 - _mm(d_in) / max(r, 1e-6))
                z, _ = self._highest_reachable(xy[0], xy[1], z_cap, p[2] + lift_min, q_touch)
                if z is not None:
                    return np.array([xy[0], xy[1], max(p[2] + lift_min, z - ceil_m)])
            return None

        top_a = lift_point(p_pick)
        if top_a is None:
            return None, None, (f"the arm cannot lift the object {c['LIFT_MIN_MM']:.0f} mm from the pick point "
                                f"(radius {math.hypot(p_pick[0], p_pick[1]) * 1e3:.0f} mm), even leaning in "
                                f"{max(c['LIFT_INWARD_MM']):.0f} mm")
        top_b = lift_point(p_place)
        if top_b is None:
            return None, None, (f"the arm cannot rise {c['LIFT_MIN_MM']:.0f} mm above the release point "
                                f"(radius {math.hypot(p_place[0], p_place[1]) * 1e3:.0f} mm)")
        lift = self._line_points(p_pick, top_a)
        drop = self._line_points(top_b, p_place)
        # Lowest tip height in transit: object bottom at least half LIFT_MIN_MM above the lower of
        # the pick / place surfaces (z_floor asks for more, but at the reach edge the arm may not
        # give it; obstacles are checked separately).
        z_low = min(p_pick[2], p_place[2]) + 0.5 * lift_min


        # Azimuth range allowed by joint_1 (unwrapped, relative to the pick).
        i1 = self._i1
        qa = q_touch.copy()
        qa[i1] += 1e-3
        pa = self._fk_tip(self._constrain(qa))
        sgn = 1.0 if _wrap_pi(math.atan2(pa[1], pa[0]) - az0) >= 0.0 else -1.0
        lo_d = (self._lo[i1] + 0.02 - q_touch[i1]) * sgn
        hi_d = (self._hi[i1] - 0.02 - q_touch[i1]) * sgn
        lo_d, hi_d = min(lo_d, hi_d), max(lo_d, hi_d)
        da = math.radians(c["ROUTE_GRID_DEG"])
        ka = np.arange(int(math.ceil(lo_d / da)), int(math.floor(hi_d / da)) + 1)
        az1 = math.atan2(p_place[1], p_place[0])
        goals = [d for d in (_wrap_pi(az1 - az0) + k * 2.0 * math.pi for k in (-1, 0, 1)) if lo_d <= d <= hi_d]
        if len(goals) == 0:
            return None, None, "the placement azimuth is outside joint_1's range"

        def polar(r, d):
            z = z_at(r)
            return None if z is None else np.array([r * math.cos(az0 + d), r * math.sin(az0 + d), z])

        def plan(margin):
            free = {}

            def is_free(ir, ia):
                key = (ir, ia)
                if key not in free:
                    p = polar(rs[ir], ia * da)
                    free[key] = p is not None and p[2] >= z_low and gap(p, margin)[0] >= margin
                return free[key]

            def seg_free(pa_, pb_):
                (r0, d0), (r1, d1) = pa_, pb_
                n = max(2, int(math.ceil(math.hypot(r1 - r0, 0.5 * (r0 + r1) * (d1 - d0)) / _mm(2.0))))
                for k in range(n + 1):
                    r, d = r0 + (r1 - r0) * k / n, d0 + (d1 - d0) * k / n
                    p = polar(r, d)
                    if p is None or p[2] < z_low or gap(p, margin)[0] < margin:
                        return False
                return True

            r_a, r_b = math.hypot(top_a[0], top_a[1]), math.hypot(top_b[0], top_b[1])
            ir_a = int(np.argmin(np.abs(rs - r_a)))
            starts = [(ir_a, 0)]
            if not seg_free((r_a, 0.0), (rs[ir_a], 0.0)):
                return None
            goal_cells = {}
            for g in goals:
                ir_b, ia_b = int(np.argmin(np.abs(rs - r_b))), int(round(g / da))
                if ka[0] <= ia_b <= ka[-1] and seg_free((rs[ir_b], ia_b * da), (r_b, g)):
                    goal_cells[(ir_b, ia_b)] = g
            if len(goal_cells) == 0:
                return None
            dist, prev, heap = {starts[0]: 0.0}, {}, [(0.0, starts[0])]
            end = None
            while heap:
                dcur, (ir, ia) = heapq.heappop(heap)
                if dcur > dist.get((ir, ia), np.inf):
                    continue
                if (ir, ia) in goal_cells:
                    end = (ir, ia)
                    break
                for dir_, dia in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                    nr, na = ir + dir_, ia + dia
                    if nr < 0 or nr >= len(rs) or na < ka[0] or na > ka[-1] or not is_free(nr, na):
                        continue
                    if not seg_free((rs[ir], ia * da), (rs[nr], na * da)):
                        continue                    # e.g. a corner cut past an obstacle
                    rm = 0.5 * (rs[ir] + rs[nr])
                    step = math.hypot(rs[nr] - rs[ir], rm * da * dia)
                    nd = dcur + step
                    if nd < dist.get((nr, na), np.inf):
                        dist[(nr, na)], prev[(nr, na)] = nd, (ir, ia)
                        heapq.heappush(heap, (nd, (nr, na)))
            if end is None:
                return None
            cells = [end]
            while cells[-1] in prev:
                cells.append(prev[cells[-1]])
            way = [(r_a, 0.0)] + [(rs[ir], ia * da) for ir, ia in reversed(cells)] + [(r_b, goal_cells[end])]
            # Shortcut smoothing: jump to the furthest waypoint reachable in a free straight (r, az) line.
            out, i = [way[0]], 0
            while i < len(way) - 1:
                j = len(way) - 1
                while j > i + 1 and not seg_free(way[i], way[j]):
                    j -= 1
                out.append(way[j])
                i = j
            return out

        way = plan(m)
        if way is None:
            # Diagnose: what margin would have worked, and what the vertical limit is.
            ok_m = None
            for mm_ in np.arange(c["OBSTACLE_MARGIN_MM"] - 1.0, -0.5, -1.0):
                if plan(_mm(mm_)) is not None:
                    ok_m = mm_
                    break
            blocking = [o for o in obstacles if o["top"] + mz + hold > z_floor]
            why = []
            for o in blocking:
                ro = float(np.linalg.norm(o["c"]))
                zc = z_at(ro)
                need = o["top"] + mz + hold
                r_ok = [r for r, z in zip(rs, zs) if not np.isnan(z) and z >= need - 1e-9]
                why.append(f"passing over '{o['label']}' needs the tip at {need * 1e3:.0f} mm; the arm keeps "
                           f"{'no height' if zc is None else format(zc * 1e3, '.0f') + ' mm'} over its centre "
                           f"(radius {ro * 1e3:.0f} mm) and {need * 1e3:.0f} mm only inside radius "
                           f"{(max(r_ok) * 1e3 if r_ok else 0):.0f} mm")
            hint = (f"a route exists only with OBSTACLE_MARGIN_MM <= {ok_m:.0f}" if ok_m is not None
                    else "no route exists even with zero horizontal margin")
            return None, None, f"{hint}; " + "; ".join(why)

        # Dense tip points along the smoothed (r, az) polyline, z on the ceiling.
        pts = []
        for (r0, d0), (r1, d1) in zip(way[:-1], way[1:]):
            n = max(1, int(math.ceil(math.hypot(r1 - r0, 0.5 * (r0 + r1) * (d1 - d0)) / _mm(c["CART_STEP_MM"]))))
            for k in range(1, n + 1):
                pts.append(polar(r0 + (r1 - r0) * k / n, d0 + (d1 - d0) * k / n))
        pts = [top_a] + pts[:-1] + [top_b]
        bad = [p for p in lift + pts + drop if gap(p, m)[0] < m]
        if bad:
            return None, None, (f"the lift, transit or final descent comes within the margin of an obstacle "
                                f"near {np.round(bad[0] * 1e3, 0)} mm")
        try:
            self._plan_points(lift + pts, q_touch, "route")
        except _PlanningError as e:
            return None, None, f"the planned route failed IK ({e})"
        g_min = min(gap(p, np.inf)[0] for p in lift + pts + drop)
        r_min = min(math.hypot(p[0], p[1]) for p in pts)
        z_max = max(p[2] for p in pts)
        turn = math.degrees(way[-1][1])
        info = (f"{len(way) - 2} waypoint(s), turns {turn:+.0f} deg about joint_1, comes in to radius "
                f"{r_min * 1e3:.0f} mm, tip up to {z_max * 1e3:.0f} mm (payload bottom {(z_max - hold) * 1e3:.0f} mm), "
                f"closest side clearance {('none needed' if not np.isfinite(g_min) else format(g_min * 1e3, '.1f') + ' mm')}; "
                f"lift {(top_a[2] - p_pick[2]) * 1e3:.0f} mm")
        return lift, pts, info

    def _highest_reachable(self, x, y, z_hi, z_lo, seed):
        z = z_hi
        while z >= z_lo - 1e-9:
            q = self._ik_tip_base(np.array([x, y, z]), seed)
            if q is not None:
                return z, q
            z -= _mm(5.0)
        return None, None

    # ------------------------------------------------------------- motion
    def _command(self, q):
        q = self._constrain(q)
        idx = list(self._cs_to_dof)
        vals = list(q)
        if self._pitch_dof_extra is not None:
            idx.append(self._pitch_dof_extra)
            vals.append(self._pitch_value(q[self._i2], q[self._i5], *self._pitch_limits))
        self._joints.command(idx, vals)
        self._q_cmd = q

    def _duration(self, qs, cart_len_m=0.0):
        Q = np.vstack([self._q_cmd] + list(qs))
        joint_len = float(np.max(np.sum(np.abs(np.diff(Q, axis=0)), axis=0)))
        t = max(joint_len / self.cfg["JOINT_SPEED_RAD_S"], cart_len_m / _mm(self.cfg["CART_SPEED_MM_S"]))
        return max(0.3, 1.875 * t)            # 1.875 = min-jerk peak/average velocity

    async def _execute(self, qs, duration, label="", settle=True, guard=False):
        """Time-scaled (min-jerk) tracking of a joint path.

        With guard (Cartesian moves), the trajectory clock slows to a quarter
        whenever the measured tip strays more than TRACK_TOL_MM sideways from the
        planned tip path (beyond the static droop seen at the start), so faster
        moves cannot pull the tip, or a held payload, off the clearance-checked
        path. Lag along the path is harmless and is not penalised. The clock never
        stops outright, so a steady PD droop cannot stall a move.
        """
        Q = np.vstack([self._q_cmd] + [self._constrain(q) for q in qs])
        seg = np.linalg.norm(np.diff(Q, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        L = float(cum[-1])
        tol = _mm(self.cfg["TRACK_TOL_MM"])
        P = np.array([self._fk_tip(q) for q in Q]) if guard else None
        if L > 1e-9:
            t, last = 0.0, self._sim_clock
            k, e_ref = 0, None
            while True:
                await self._tick()
                dt, last = self._sim_clock - last, self._sim_clock
                e = None
                if guard:
                    q_m = self._q_meas()
                    if q_m is not None:
                        lo, hi = max(0, k - 40), min(len(P) - 1, k + 2)
                        e = _seg_dist(self._fk_tip(q_m)[None, :], P[lo:hi], P[lo + 1:hi + 1]) \
                            if hi > lo else float(np.linalg.norm(self._fk_tip(q_m) - P[lo]))
                        if e_ref is None:
                            e_ref = e
                t += dt if (e is None or e - e_ref <= tol) else 0.25 * dt
                tau = t / duration
                dist = _min_jerk(tau) * L
                k = int(np.searchsorted(cum, dist, side="right")) - 1
                k = min(max(k, 0), len(seg) - 1)
                a = 0.0 if seg[k] < 1e-12 else (dist - cum[k]) / seg[k]
                q_c = Q[k] + min(1.0, a) * (Q[k + 1] - Q[k])
                self._command(q_c)
                if tau >= 1.0:
                    break
        self._command(Q[-1])
        if settle:
            await self._settle(Q[-1], label)

    async def _settle(self, q_goal, label=""):
        tol = self.cfg["SETTLE_TOL_RAD"]
        t0, good = self._sim_clock, 0
        err = float("inf")
        while self._sim_clock - t0 < self.cfg["SETTLE_TIMEOUT_S"]:
            await self._tick()
            q = self._q_meas()
            if q is None:
                continue
            err = float(np.max(np.abs(q - q_goal)))
            good = good + 1 if err < tol else 0
            if good >= 3:
                return True
        _warn(f"{label}: settle timeout, residual {err:.4f} rad (PD droop)")
        return False

    async def _move_joint(self, q_goal, label, settle=True):
        _log(f"joint move: {label}")
        await self._execute([q_goal], self._duration([q_goal]), label, settle)

    async def _move_tip_path(self, pts_b, label, settle=True):
        _log(f"cartesian move: {label}")
        qs = self._plan_points(pts_b, self._q_cmd, label)
        start = self._fk_tip(self._q_cmd)
        length = sum(np.linalg.norm(b - a) for a, b in zip([start] + pts_b[:-1], pts_b))
        await self._execute(qs, self._duration(qs, length), label, settle, guard=True)

    async def descend_cartesian_to_touch(self, xy_b, z_goal_b, label, contact_expected=True, lean=None):
        """Closed-loop vertical descent of the suction tip onto z_goal_b at exactly xy_b.

        The commanded tip is stepped downward until the measured tip (FK of the
        measured joints) reaches the goal within TOUCH_TOL_MM, or, when contact is
        expected, until the arm stalls on the surface. XY droop is corrected with a
        bounded integral term so the measured tip lands on the perception XY.
        lean (XY metres per metre of height) tilts the descent line, for points at
        the edge of the workspace that cannot be reached from straight above.
        """
        c = self.cfg
        xy_b = np.asarray(xy_b, dtype=float)
        lean = np.zeros(2) if lean is None else np.asarray(lean, dtype=float)

        def xy_at(z):
            return xy_b + lean * max(0.0, z - z_goal_b)
        tol = _mm(c["TOUCH_TOL_MM"])
        floor = z_goal_b - _mm(c["TOUCH_MAX_OVERDRIVE_MM"] if contact_expected else c["PLACE_MAX_OVERDRIVE_MM"])
        z_start = z_goal_b + _mm(c["APPROACH_GAP_MM"])
        start = np.array([*xy_at(z_start), z_start])
        tip = self._tip_base_meas()
        if tip is not None and tip[2] > start[2]:
            await self._move_tip_path(self._line_points(self._fk_tip(self._q_cmd), start), f"{label} approach")

        _log(f"{label}: touchdown creep toward z={z_goal_b * 1e3:.1f} mm (base)")
        cmd_z = start[2]
        xy_corr = np.zeros(2)
        lim = _mm(c["XY_CORRECTION_LIMIT_MM"])
        last_z, stall, t0 = None, 0, self._sim_clock
        t_floor = None
        reason = "timeout"
        while self._sim_clock - t0 < c["TOUCH_TIMEOUT_S"]:
            await self._tick()
            tip = self._tip_base_meas()
            if tip is None:
                continue
            if tip[2] - z_goal_b <= tol:
                reason = "reached surface"
                break
            if contact_expected and last_z is not None:
                lagging = (tip[2] - cmd_z) > 2.0 * tol
                stall = stall + 1 if (lagging and abs(tip[2] - last_z) < _mm(0.03)) else 0
                if stall >= 10 and tip[2] - z_goal_b < _mm(c["APPROACH_GAP_MM"]):
                    reason = "stalled on contact"
                    break
            last_z = tip[2]
            if cmd_z <= floor:
                t_floor = self._sim_clock if t_floor is None else t_floor
                if self._sim_clock - t_floor > 1.5:
                    reason = "overdrive limit reached"
                    break
            cmd_z = max(cmd_z - _mm(c["TOUCH_STEP_MM"]), floor)
            xy_corr = np.clip(xy_corr + c["XY_CORRECTION_GAIN"] * (xy_at(tip[2]) - tip[:2]), -lim, lim)
            xy_cmd = xy_at(cmd_z) + xy_corr
            q = self._ik_tip_base(np.array([xy_cmd[0], xy_cmd[1], cmd_z]), self._q_cmd)
            if q is None:
                reason = "IK limit"
                break
            self._command(q)
        tip = self._tip_base_meas()
        # Hold where we are so the PD loop stops pressing on the surface.
        q_now = self._q_meas()
        if q_now is not None:
            self._command(q_now)
        if tip is None:
            raise RuntimeError(f"{label}: lost joint state during touchdown")
        gap = (tip[2] - z_goal_b) * 1e3
        lat = np.linalg.norm(tip[:2] - xy_b) * 1e3
        _log(f"{label}: {reason}; tip gap {gap:+.2f} mm, lateral error {lat:.2f} mm")
        if gap > 3.0 * c["TOUCH_TOL_MM"]:
            _warn(f"{label}: tip still {gap:.1f} mm above the surface")
        return tip

    # ------------------------------------------------------ kinematic vacuum
    def _collision_prims(self, root):
        return [p for p in Usd.PrimRange(root) if p.HasAPI(UsdPhysics.CollisionAPI)]

    @staticmethod
    def _snapshot(attr):
        return (attr, attr.HasAuthoredValue(), attr.Get())

    @staticmethod
    def _restore(snap):
        attr, authored, value = snap
        if authored:
            attr.Set(value)
        else:
            attr.Clear()

    def _prepare_xform(self, prim):
        xf = UsdGeom.Xformable(prim)
        ops = xf.GetOrderedXformOps()
        types = [op.GetOpType() for op in ops]
        t_op = next((op for op in ops if op.GetOpType() == UsdGeom.XformOp.TypeTranslate), None)
        o_op = next((op for op in ops if op.GetOpType() == UsdGeom.XformOp.TypeOrient), None)
        allowed = {UsdGeom.XformOp.TypeTranslate, UsdGeom.XformOp.TypeOrient, UsdGeom.XformOp.TypeScale}
        if t_op is not None and o_op is not None and all(t in allowed for t in types):
            return {"mode": "to", "t": t_op, "o": o_op}
        local = xf.GetLocalTransformation(Usd.TimeCode.Default())
        scale = np.array(Gf.Transform(local).GetScale(), dtype=float)
        m_op = xf.MakeMatrixXform()
        m_op.Set(local)
        return {"mode": "matrix", "m": m_op, "scale": scale}

    def _write_body_world(self, L, T_obj_m):
        T = T_obj_m.copy()
        T[:3, 3] /= self._mpu
        local = L["parent_inv"] @ T
        ops = L["ops"]
        if ops["mode"] == "to":
            R = _orthonormalize(local[:3, :3])
            w, x, y, z = _R_to_quat_wxyz(R)
            t = local[:3, 3]
            if ops["o"].GetPrecision() == UsdGeom.XformOp.PrecisionFloat:
                ops["o"].Set(Gf.Quatf(float(w), float(x), float(y), float(z)))
            else:
                ops["o"].Set(Gf.Quatd(float(w), float(x), float(y), float(z)))
            if ops["t"].GetPrecision() == UsdGeom.XformOp.PrecisionFloat:
                ops["t"].Set(Gf.Vec3f(float(t[0]), float(t[1]), float(t[2])))
            else:
                ops["t"].Set(Gf.Vec3d(float(t[0]), float(t[1]), float(t[2])))
        else:
            M = local @ np.diag(np.append(ops["scale"], 1.0))
            ops["m"].Set(Gf.Matrix4d(*M.T.flatten().tolist()))

    def _zero_velocity(self, rb):
        rb.GetVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        rb.GetAngularVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))

    def engage_vacuum(self, body_path):
        prim = self._stage.GetPrimAtPath(body_path)
        if not prim.IsValid() or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise RuntimeError(f"{body_path} is not a rigid body")
        rb = UsdPhysics.RigidBodyAPI(prim)
        T_ee = self._ee_world()
        T_obj = self._physx_pose(body_path)
        if T_obj is None:
            T_obj = self._usd_world(prim)
        if T_ee is None:
            raise RuntimeError("end-effector pose unavailable")
        parent = prim.GetParent()
        P = _gf_to_np(UsdGeom.Xformable(parent).ComputeLocalToWorldTransform(Usd.TimeCode.Default())) \
            if parent.IsValid() and not parent.IsPseudoRoot() else np.eye(4)
        coll = self._collision_prims(prim) if self.cfg["HOLD_COLLISION"] == "disable" else []
        L = {
            "path": body_path,
            "prim": prim,
            "rb": rb,
            "T_obj_in_ee": _inv_T(T_ee) @ T_obj,     # T_target_ee
            "parent_inv": np.linalg.inv(P),
            "kin_snap": self._snapshot(rb.GetKinematicEnabledAttr() if rb.GetKinematicEnabledAttr()
                                       else rb.CreateKinematicEnabledAttr(False)),
            "coll_snaps": [],
            "active": False,
        }
        for cp in coll:
            api = UsdPhysics.CollisionAPI(cp)
            attr = api.GetCollisionEnabledAttr() if api.GetCollisionEnabledAttr() else api.CreateCollisionEnabledAttr(True)
            L["coll_snaps"].append(self._snapshot(attr))
        L["ops"] = self._prepare_xform(prim)
        with Sdf.ChangeBlock():
            rb.GetKinematicEnabledAttr().Set(True)
            for attr, _, _ in L["coll_snaps"]:
                attr.Set(False)
            self._zero_velocity(rb)
        self._latch = L
        self._write_body_world(L, T_obj)
        L["active"] = True
        how = f"{len(coll)} collider(s) disabled" if len(coll) > 0 else "colliders stay on, suction link filtered"
        _log(f"vacuum ON: {body_path} latched kinematically ({how})")

    def sync_vacuum_payload(self):
        """Called every physics step while the vacuum is on: object follows link_suction."""
        L = self._latch
        if L is None or not L["active"]:
            return
        T_ee = self._ee_world()
        if T_ee is None:
            return
        with Sdf.ChangeBlock():
            self._write_body_world(L, T_ee @ L["T_obj_in_ee"])

    async def release_vacuum(self, lean=None):
        L = self._latch
        if L is None:
            return
        # 1. Stop following; the body stays kinematic and frozen where it was released.
        L["active"] = False
        # 2. Back the cup off so restoring collision cannot create cup/object overlap.
        tip = self._fk_tip(self._q_cmd)
        up = _mm(self.cfg["RELEASE_BACKOFF_MM"])
        off = np.zeros(2) if lean is None else np.asarray(lean, dtype=float) * up
        await self._move_tip_path(self._line_points(tip, tip + np.array([off[0], off[1], up])), "release back-off")
        # 3. Hand the object back to dynamics at rest, and stop ignoring the suction link.
        self._watch_contacts = False
        self._restore_payload_filter()
        with Sdf.ChangeBlock():
            for snap in L["coll_snaps"]:
                self._restore(snap)
            self._restore(L["kin_snap"])
            self._zero_velocity(L["rb"])
        self._latch = None
        await self._frames(2)
        try:
            self._zero_velocity(L["rb"])
        except Exception:
            pass
        _log(f"vacuum OFF: {L['path']} released")

    def _filter_link_paths(self):
        c = self.cfg
        if c["FILTER_LINKS"] != "auto":
            return [str(p) for p in c["FILTER_LINKS"]]
        out = [c["EE_PRIM"]]
        try:
            import xml.etree.ElementTree as ET
            for j in ET.parse(c["URDF_PATH"]).getroot().findall("joint"):
                child, parent = j.find("child"), j.find("parent")
                if child is not None and parent is not None and child.get("link") == c["EE_FRAME"]:
                    name = parent.get("link")
                    for p in self._robot_links:
                        if p.rstrip("/").split("/")[-1] == name:
                            out.append(p)
        except Exception as e:
            _warn(f"could not read link_suction's parent from the URDF: {e}")
        return out

    async def _apply_payload_filter(self, body_path):
        """Payload ignores contacts with the suction link(s) only; solid to everything else."""
        if self.cfg["HOLD_COLLISION"] != "filter":
            return
        prim = self._stage.GetPrimAtPath(body_path)
        had = prim.HasAPI(UsdPhysics.FilteredPairsAPI)
        api = UsdPhysics.FilteredPairsAPI.Apply(prim)
        rel = api.GetFilteredPairsRel()
        if not rel:
            rel = api.CreateFilteredPairsRel()
        self._filter_state = {"prim": prim, "had": had, "old": list(rel.GetTargets())}
        links = self._filter_link_paths()
        for p in links:
            rel.AddTarget(Sdf.Path(p))
        await self._wait_physics_steps(3, 300)
        _log(f"collision filter: {body_path} ignores {links}; still solid to the table and obstacles")

    def _restore_payload_filter(self):
        st = self._filter_state
        if st is None:
            return
        self._filter_state = None
        try:
            api = UsdPhysics.FilteredPairsAPI(st["prim"])
            rel = api.GetFilteredPairsRel()
            if st["had"]:
                rel.SetTargets(st["old"])
            else:
                rel.ClearTargets(True)
                st["prim"].RemoveAPI(UsdPhysics.FilteredPairsAPI)
        except Exception as e:
            _warn(f"could not restore the collision filter: {e}")

    def _start_contact_watch(self, body_path, ignore):
        if not self.cfg["CONTACT_WATCHDOG"]:
            return
        try:
            from pxr import PhysxSchema
            prim = self._stage.GetPrimAtPath(body_path)
            had = prim.HasAPI(PhysxSchema.PhysxContactReportAPI)
            api = PhysxSchema.PhysxContactReportAPI.Apply(prim)
            api.CreateThresholdAttr().Set(0.0)
            self._contact_state = {"prim": prim, "had": had, "path": body_path}
            self._contact_ignore = set(p for p in ignore if p)
            self._contact_sub = omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(
                self._on_contact)
        except Exception as e:
            _warn(f"contact watchdog unavailable: {e}")

    def _on_contact(self, headers, *_):
        st = self._contact_state
        if st is None or not self._watch_contacts:
            return
        try:
            for h in headers:
                a0, a1 = _path_str(h.actor0), _path_str(h.actor1)
                if _under(a0, st["path"]):
                    other = a1
                elif _under(a1, st["path"]):
                    other = a0
                else:
                    continue
                if not other or self._is_robot(other) or any(_under(other, i) for i in self._contact_ignore):
                    continue
                self._contact_hit = other
        except Exception:
            pass

    def _stop_contact_watch(self):
        self._watch_contacts = False
        self._contact_sub = None
        st = self._contact_state
        self._contact_state = None
        if st is not None and not st["had"]:
            try:
                from pxr import PhysxSchema
                st["prim"].RemoveAPI(PhysxSchema.PhysxContactReportAPI)
            except Exception:
                pass

    def _restore_payload_immediately(self):
        L = self._latch
        if L is None:
            return
        L["active"] = False
        try:
            with Sdf.ChangeBlock():
                for snap in L["coll_snaps"]:
                    self._restore(snap)
                self._restore(L["kin_snap"])
                self._zero_velocity(L["rb"])
            _log(f"restored physics state of {L['path']}")
        except Exception as e:
            _warn(f"could not restore {L['path']}: {e}")
        self._latch = None

    # ------------------------------------------------------------- pipeline
    async def _pipeline(self):
        c = self.cfg
        kp = self._keypoints
        if c["DEPTH_VIZ"]:
            await self._depth_safe(self._depth_capture())

        # 3. Resolve grasp pixels, requiring the configured red cylinder rather than an arbitrary object.
        order = [c["GRASP_INDEX"]] + [i for i in range(len(kp["grasps"])) if i != c["GRASP_INDEX"]]
        pick, target = None, None
        expected_pick = c["RED_PRIM"]

        def is_expected_target(path):
            return bool(path and (_under(path, expected_pick) or _under(expected_pick, path)))

        for i in order:
            if i < 0 or i >= len(kp["grasps"]):
                continue
            rec = kp["grasps"][i]
            u, v = rec["uv"]
            hit = self._measure_pixel(u, v)
            self._viz_call(self._viz_ray, "rays_grasp", u, v, hit)
            if hit is None:
                _warn(f"grasp {i} ({u:.1f}, {v:.1f}): depth ray hit nothing")
                continue
            tgt = await self._resolve_target(hit)
            if tgt is None:
                _warn(f"grasp {i}: lands on {hit['collision']}, not a pickable object")
                continue
            if not is_expected_target(tgt):
                _warn(f"grasp {i}: ray hit {tgt}, but the intended pick target is {expected_pick}; skipping")
                continue
            if hit["source"] != "physx" or not _under(hit["body"], tgt):
                hit = self._measure_pixel(u, v)
                if hit is None or not (_under(hit["body"], tgt) or _under(hit["collision"], tgt)):
                    _warn(f"grasp {i}: object {tgt} no longer under pixel after physics setup")
                    continue
            pick = {"index": i, "uv": (u, v), "label": rec["label"], "hit": hit,
                    "p_b": self._w2b(hit["position"])}
            target = tgt
            break

        # Fallback: the keypoints may have been captured before the object moved or with
        # a different viewport crop. Project the actual red mesh's top centre using the
        # same camera model that _pixel_ray uses. This is still a depth/raycast hit test.
        if pick is None:
            uv = self._projected_top_pixel(expected_pick)
            if uv is not None:
                u, v = uv
                hit = self._measure_pixel(u, v)
                self._viz_call(self._viz_ray, "rays_grasp_auto", u, v, hit)
                if hit is None:
                    _warn(f"auto-projected red-cylinder top at ({u:.1f}, {v:.1f}) hit nothing")
                else:
                    tgt = await self._resolve_target(hit)
                    if tgt is not None and is_expected_target(tgt):
                        if hit["source"] != "physx" or not _under(hit["body"], tgt):
                            hit = self._measure_pixel(u, v)
                        if hit is not None and (_under(hit["body"], tgt) or _under(hit["collision"], tgt)):
                            pick = {"index": -1, "uv": (u, v),
                                    "label": "auto-projected red-cylinder top", "hit": hit,
                                    "p_b": self._w2b(hit["position"])}
                            target = tgt
                            _warn(f"using auto-projected red-cylinder grasp pixel ({u:.1f}, {v:.1f})")
                    else:
                        _warn(f"auto-projected pixel hit {hit.get('collision')}, not {expected_pick}")

        self._depth_mark("grasp", order, pick["index"] if pick is not None and pick["index"] >= 0 else None)
        if pick is None:
            raise RuntimeError(
                "no grasp point hit the configured red cylinder, including the auto-projected top-centre fallback. "
                "Check the overhead-camera view/keypoint resolution and confirm /magician/color is visible and "
                "has a working collider. The controller stopped without commanding a pick motion.")
        p_pick = pick["p_b"]
        self._viz_call(self._viz_marker, "pick", p_pick, _Viz.COLORS["pick"], 1.6 * self.cfg["VIZ_POINT_SIZE"])
        _log(f"grasp {pick['index']} '{pick['label']}' px={np.round(pick['uv'], 1)} -> {target} "
             f"({pick['hit']['source']}), surface {np.round(p_pick * 1e3, 1)} mm (base frame), "
             f"radius {math.hypot(p_pick[0], p_pick[1]) * 1e3:.0f} mm")

        # 4. Support surface and object height, from depth only.
        z_support, _, src = self._measure_support(pick["uv"][0], pick["uv"][1], target, p_pick[2])
        obj_h = p_pick[2] - z_support
        if obj_h <= _mm(1.0):
            raise RuntimeError(f"measured object height {obj_h * 1e3:.1f} mm is not plausible")
        _log(f"support surface z={z_support * 1e3:.1f} mm ({src}); object height {obj_h * 1e3:.1f} mm")

        # Payload stays solid; it only ignores the suction link. Watch its contacts in transit.
        await self._apply_payload_filter(target)
        under = self._measure_pixel(pick["uv"][0], pick["uv"][1], skip=(target,))
        self._start_contact_watch(target, [under["body"] if under is not None else ""])
        # Ensure the known cone and sphere are represented among obstacle ray samples even
        # if the perception JSON was generated with a different crop/pose.
        for obstacle_path in (c["CONE_PRIM"], c["SPHERE_PRIM"]):
            obstacle_uv = self._projected_top_pixel(obstacle_path)
            if obstacle_uv is not None:
                kp["obstacles"].append({"uv": obstacle_uv, "label": "auto-obstacle:" + obstacle_path})
        obstacles = self._build_obstacles(target) if c["AVOID_OBSTACLES"] else []
        self._viz_call(self._viz_obstacles, obstacles)
        fp = self._footprint(target)
        if fp is None:
            sq = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]) * 0.05
            fp = {"hull": p_pick[:2] + sq, "c": p_pick[:2].copy(), "r": 0.0707}
            _warn("no geometry found for the held object; assuming a 100 mm square footprint")
        payload = {"hull0": fp["hull"] - p_pick[:2], "c0": fp["c"] - p_pick[:2], "r": fp["r"]}

        q_touch = self._ik_tip_base(p_pick, self._q_home)
        if q_touch is None:
            raise RuntimeError(f"pick point {np.round(p_pick * 1e3, 1)} mm (radius "
                               f"{math.hypot(p_pick[0], p_pick[1]) * 1e3:.0f} mm) is outside the arm's reach")

        # Add projected placements around the cone after perception candidates. These are
        # only fallback candidates; each is still raycast against the actual support surface
        # and passed through the same IK/reachability and obstacle-aware route planner.
        generated_places = self._auto_placement_pixels_near_cone(z_support)
        if generated_places:
            _log(f"generated {len(generated_places)} fallback placement pixels around the purple cone")
            kp["places"].extend(generated_places)

        # 5. First reachable placement candidate.
        place = None
        for j, rec in enumerate(kp["places"]):
            u, v = rec["uv"]
            why = self._check_placement(j, u, v, target, p_pick, obj_h, q_touch,
                                        need_carry=not c["AVOID_OBSTACLES"])
            if not isinstance(why, str) and c["AVOID_OBSTACLES"]:
                p_b = why["p_b"]
                z_floor = max(p_pick[2], p_b[2] + obj_h) + _mm(c["CARRY_MIN_CLEARANCE_MM"])
                p_rel = np.array([p_b[0], p_b[1], p_b[2] + obj_h + _mm(c["RELEASE_CLEARANCE_MM"])])
                lift, pts, info = self._plan_route(p_pick, p_rel, obj_h, payload, obstacles, q_touch, z_floor)
                if lift is None:
                    why = f"no collision-free route ({info})"
                else:
                    # Approach / retract follow the route's first / last 40 mm (clear of the objects).
                    far = _mm(40.0)
                    k0 = next((k for k, p in enumerate(pts) if np.linalg.norm(p[:2] - p_pick[:2]) >= far),
                              len(pts) - 1)
                    k1 = next((k for k in range(len(pts) - 1, -1, -1)
                               if np.linalg.norm(pts[k][:2] - p_b[:2]) >= far), 0)
                    why.update({"lift": lift, "route": pts, "route_info": info, "z_carry": lift[-1][2], "p_rel": p_rel,
                                "ramp_out": pts[:k0 + 1], "ramp_in": pts[k1:]})
            if isinstance(why, str):
                _warn(f"placement {j} px=({u:.0f}, {v:.0f}): {why}; trying the next candidate")
                if self._viz is not None:
                    h = self._measure_pixel(u, v)
                    self._viz_call(self._viz_ray, "rays_place", u, v, h, False)
                    if h is not None:
                        self._viz_call(self._viz_marker, f"place_rej_{j}", self._w2b(h["position"]),
                                       _Viz.COLORS["place_rej"])
                continue
            place = why
            break
        self._depth_mark("place", range(len(kp["places"])), None if place is None else place["index"])
        if place is None:
            raise RuntimeError(f"none of the {len(kp['places'])} placement candidates has a reachable, "
                               f"collision-free route; the arm did not move")
        p_place, z_carry = place["p_b"], place["z_carry"]
        self._viz_call(self._viz_ray, "rays_place", place["uv"][0], place["uv"][1], place["hit"])
        self._viz_call(self._viz_marker, "place", p_place, _Viz.COLORS["place"], 1.6 * self.cfg["VIZ_POINT_SIZE"])
        self._viz_call(self._viz_plan, p_pick, place)
        if self._viz is not None and not c["VIZ_KEEP_RAYS"]:
            self._viz_call(self._viz.drop, "rays_")
        _log(f"placement {place['index']} px={np.round(place['uv'], 1)} -> "
             f"{'stacking on' if place['stack'] else 'tabletop'} {place['on']}, surface "
             f"{np.round(p_place * 1e3, 1)} mm (base frame)"
             + ("" if "route" in place else f"; carry height {z_carry * 1e3:.0f} mm"))
        if "route_info" in place:
            _log(f"route: {place['route_info']}")

        # 6. HOME -> above pick -> hover -> touchdown -> vacuum.
        await self._move_joint(self._q_home, "HOME")
        if "route" in place and len(place["ramp_out"]) > 0:
            # Come in the way the object will leave: high point of the route, then down its ramp.
            ramp = place["ramp_out"]
            q_up = self._ik_tip_base(ramp[-1], self._q_cmd)
            if q_up is None:
                raise RuntimeError("start of the planned route is unreachable")
            await self._move_joint(q_up, "route start (above pick side)", settle=False)
            await self._move_tip_path(list(reversed(ramp[:-1])) + [place["lift"][-1]], "ramp to above pick",
                                      settle=False)
        else:
            q_up = self._ik_tip_base(np.array([p_pick[0], p_pick[1], z_carry]), self._q_cmd)
            if q_up is None:
                raise RuntimeError("pose above the pick point is unreachable")
            await self._move_joint(q_up, "above pick")
        lean_pick, lean_place = None, None
        if "route" in place:
            # The lift / drop lines (vertical, or leaning in at the reach edge) are reused for touchdown.
            ta, tb = place["lift"][-1], place["route"][-1]
            lean_pick = (ta[:2] - p_pick[:2]) / max(ta[2] - p_pick[2], 1e-6)
            lean_place = (tb[:2] - p_place[:2]) / max(tb[2] - place["p_rel"][2], 1e-6)
        z_hover = min(p_pick[2] + _mm(c["HOVER_CLEARANCE_MM"]), z_carry)
        lp = np.zeros(2) if lean_pick is None else lean_pick
        hover_b = np.array([*(p_pick[:2] + lp * (z_hover - p_pick[2])), z_hover])
        await self._move_tip_path(self._line_points(self._fk_tip(self._q_cmd), hover_b), "hover above pick")
        tip = await self.descend_cartesian_to_touch(p_pick[:2], p_pick[2], "pick", contact_expected=True,
                                                    lean=lean_pick)
        self.engage_vacuum(target)
        hold_drop = tip[2] - z_support            # tip-to-object-bottom distance as actually grasped
        await self._wait_sim(0.1)

        self._watch_contacts = True
        tip = self._fk_tip(self._q_cmd)
        if "route" in place:
            # Vertical lift, then the planned obstacle-free route over the placement.
            lift_top = place["lift"][-1]
            await self._move_tip_path(self._line_points(tip, lift_top), "lift", settle=False)
            await self._move_tip_path(place["route"], "obstacle-free transit")
        else:
            # Phase 1: vertical lift, then a smooth joint-space move over the placement.
            await self._move_tip_path(self._line_points(tip, np.array([p_pick[0], p_pick[1], z_carry])),
                                      "vertical lift")
            q_over = self._ik_tip_base(np.array([p_place[0], p_place[1], z_carry]), self._q_cmd)
            if q_over is None:
                q_over = place["q_over"]
            await self._move_joint(q_over, "over placement")

        # Descend, release, retract.
        z_release = p_place[2] + hold_drop + _mm(c["RELEASE_CLEARANCE_MM"])
        await self.descend_cartesian_to_touch(p_place[:2], z_release, "place", contact_expected=False,
                                              lean=lean_place)
        await self.release_vacuum(lean=lean_place)
        await self._wait_sim(0.3)
        self._stop_contact_watch()
        tip = self._fk_tip(self._q_cmd)
        if "route" in place:
            # Leave the way the object came in: back up the drop, then back up the ramp.
            ramp = place["ramp_in"]
            await self._move_tip_path(self._line_points(tip, place["route"][-1]), "retract up",
                                      settle=len(ramp) <= 1)
            if len(ramp) > 1:
                await self._move_tip_path(list(reversed(ramp[:-1])), "retract along route", settle=False)
        else:
            await self._move_tip_path(self._line_points(tip, np.array([tip[0], tip[1], z_carry])), "retract up")
        await self._move_joint(self._q_home, "HOME")

        T_final = self._physx_pose(target)
        if T_final is not None:
            _log(f"object origin settled at {np.round(self._w2b(T_final[:3, 3]) * 1e3, 1)} mm (base frame); "
                 f"placement point {np.round(p_place * 1e3, 1)} mm (the suction point lands on the placement "
                 f"point, so the origin is off by the grip offset, turned with joint_1)")
            # Remembered across re-runs so a stale scene is reported as such.
            builtins._DOBOT_PNP_LAST_PLACED = {"path": target, "from": p_pick.copy(),
                                               "to": self._w2b(T_final[:3, 3])}

    def _check_placement(self, j, u, v, target, p_pick, obj_h, q_touch, need_carry=True):
        """Placement dict if candidate j is usable, else a string saying why not."""
        c = self.cfg
        hit = self._measure_pixel(u, v)
        if hit is None:
            return "depth ray hit nothing"
        if _under(hit["body"], target) or _under(hit["collision"], target):
            return "lands on the object being picked"
        if hit["normal"][2] < 0.9:
            return f"surface is tilted (normal z={hit['normal'][2]:.2f})"
        p_b = self._w2b(hit["position"])
        r = math.hypot(p_b[0], p_b[1]) * 1e3
        z_release = p_b[2] + obj_h + _mm(c["RELEASE_CLEARANCE_MM"])
        q_rel = self._ik_tip_base(np.array([p_b[0], p_b[1], z_release]), q_touch)
        if q_rel is None:
            return f"release pose unreachable (radius {r:.0f} mm, tip z {z_release * 1e3:.0f} mm)"
        base_obj = self._object_root(self._stage.GetPrimAtPath(hit["collision"])) if hit["collision"] else None
        out = {"index": j, "uv": (u, v), "hit": hit, "p_b": p_b, "stack": base_obj is not None,
               "on": str(base_obj.GetPath()) if base_obj is not None else hit["collision"]}
        if not need_carry:
            return out                              # the route planner picks the heights
        top = max(p_pick[2], p_b[2] + obj_h)
        z_want = top + _mm(c["CARRY_CLEARANCE_MM"])
        z_lo = top + _mm(c["CARRY_MIN_CLEARANCE_MM"])
        z_a, _ = self._highest_reachable(p_pick[0], p_pick[1], z_want, z_lo, q_touch)
        z_b, _ = self._highest_reachable(p_b[0], p_b[1], z_want, z_lo, q_rel)
        if z_a is None or z_b is None:
            return (f"no carry height at least {c['CARRY_MIN_CLEARANCE_MM']:.0f} mm above the surfaces is "
                    f"reachable over both pick and place")
        z_carry = min(z_a, z_b)
        q_up = self._ik_tip_base(np.array([p_pick[0], p_pick[1], z_carry]), q_touch)
        q_over = self._ik_tip_base(np.array([p_b[0], p_b[1], z_carry]), q_rel)
        if q_up is None or q_over is None:
            return "carry poses unreachable"
        out.update({"z_carry": z_carry, "q_over": q_over})
        return out

# =============================================================================
# Read-only preflight: NEVER starts the timeline or commands robot motion
# =============================================================================
def _run_preflight(cfg):
    """Inspect the open stage and required input files without modifying anything."""
    print("\n" + "=" * 76)
    print("DOBOT PICK-AND-PLACE PREFLIGHT (READ-ONLY)")
    print("No timeline start, no stage edits/saves, no robot commands.")
    print("=" * 76)
    results = []

    def report(name, ok, detail="", blocking=True):
        status = "PASS" if ok else ("BLOCK" if blocking else "WARN")
        print(f"[{status}] {name}" + (f": {detail}" if detail else ""))
        results.append((name, bool(ok), blocking))

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        report("USD stage", False, "No stage is open")
        print("PREFLIGHT RESULT: BLOCKED")
        return False

    try:
        root_layer = stage.GetRootLayer().identifier
    except Exception:
        root_layer = "(unavailable)"
    report("Current stage", True, root_layer, blocking=False)

    # Block if another copy of this controller is still active or the timeline is
    # already running; do not stop it automatically because that can alter payload state.
    prev_instance = getattr(builtins, "_DOBOT_PNP_INSTANCE", None)
    prev_active = prev_instance is not None and not getattr(prev_instance, "_shutting_down", False)
    report("no previous active controller", not prev_active,
           "a previous Dobot controller instance is still active; stop it manually first"
           if prev_active else "none detected")
    timeline = omni.timeline.get_timeline_interface()
    timeline_stopped = timeline.is_stopped()
    report("timeline stopped", timeline_stopped,
           "preflight does not stop an already-running timeline")

    up_axis = UsdGeom.GetStageUpAxis(stage)
    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    report("Stage up axis", up_axis == UsdGeom.Tokens.z, str(up_axis))
    report("Meters per unit", abs(float(mpu or 1.0) - 1.0) < 1e-6,
           f"{float(mpu or 1.0):g} (controller assumes metres)")

    expected = [
        ("robot", cfg["ROBOT_PRIM"]),
        ("end-effector link", cfg["EE_PRIM"]),
        ("overhead camera", cfg["CAMERA_PRIM"]),
        ("expected red cylinder", cfg["RED_PRIM"]),
        ("expected purple cone", cfg["CONE_PRIM"]),
        ("expected green sphere", cfg["SPHERE_PRIM"]),
    ]
    prims = {}
    for label, path in expected:
        prim = stage.GetPrimAtPath(path)
        valid = prim.IsValid()
        prims[path] = prim
        extra = ""
        if valid and label == "overhead camera":
            valid = prim.IsA(UsdGeom.Camera)
            extra = f" (type={prim.GetTypeName()})"
        report(f"{label} {path}", valid, extra.strip(), blocking=True)

    for joint_name in (cfg["JOINT_BASE"], cfg["JOINT_SHOULDER"],
                       cfg["JOINT_ELBOW"], cfg["JOINT_PITCH"]):
        joint_path = f'{cfg["ROBOT_PRIM"]}/joints/{joint_name}'
        joint = stage.GetPrimAtPath(joint_path)
        report(f"robot joint {joint_name}", joint.IsValid() and joint.IsA(UsdPhysics.Joint),
               f"{joint_path} (type={joint.GetTypeName() if joint.IsValid() else 'missing'})")

    physics_scene = [p for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)]
    report("PhysicsScene present", bool(physics_scene),
           ", ".join(str(p.GetPath()) for p in physics_scene) if physics_scene else "none found")

    # Existing drive targets can move the robot as soon as play is started. Compare
    # authored degree targets with live angular state (stored in radians in this USD).
    drive_mismatches = []
    for joint_name in (cfg["JOINT_BASE"], cfg["JOINT_SHOULDER"],
                       cfg["JOINT_ELBOW"], cfg["JOINT_PITCH"]):
        path = f'{cfg["ROBOT_PRIM"]}/joints/{joint_name}'
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            continue
        target_attr = prim.GetAttribute("drive:angular:physics:targetPosition")
        state_attr = prim.GetAttribute("state:angular:physics:position")
        target_deg = target_attr.Get() if target_attr.IsValid() else None
        state_rad = state_attr.Get() if state_attr.IsValid() else None
        if target_deg is not None and state_rad is not None:
            delta_deg = abs(math.degrees(float(state_rad)) - float(target_deg))
            if delta_deg > 5.0:
                drive_mismatches.append(f"{joint_name}: target {float(target_deg):.1f} deg vs state "
                                        f"{math.degrees(float(state_rad)):.1f} deg (delta {delta_deg:.1f} deg)")
    report("drive targets agree with authored joint state", not drive_mismatches,
           "; ".join(drive_mismatches) if drive_mismatches else "no >5 degree mismatches detected")

    # The controller latches the object kinematically; it does not use an Isaac
    # SurfaceGripper/D6 attachment joint. Flag the manually created prim if present.
    extra_joint_path = f'{cfg["EE_PRIM"]}/surface_gripper/suction_joint'
    extra_joint = stage.GetPrimAtPath(extra_joint_path)
    if extra_joint.IsValid():
        rel0 = extra_joint.GetRelationship("physics:body0")
        rel1 = extra_joint.GetRelationship("physics:body1")
        body0 = rel0.GetTargets() if rel0.IsValid() else []
        body1 = rel1.GetTargets() if rel1.IsValid() else []
        enabled_attr = extra_joint.GetAttribute("physics:jointEnabled")
        enabled = enabled_attr.Get() if enabled_attr.IsValid() else True
        detail = f"type={extra_joint.GetTypeName()}, body0={list(map(str, body0))}, " \
                 f"body1={list(map(str, body1))}, enabled={enabled}; controller will not use it"
        report("manually created D6 joint", True, detail, blocking=False)
        if enabled and (body0 or body1):
            report("unused D6 joint is inert", False,
                   "it has a body relationship and is enabled; disable/remove it in the test copy before motion", True)
    else:
        report("optional D6 attachment prim", True, "not present; not required by kinematic-latch mode", False)

    # Resolve/validate configured file paths. Search recursively only for the exact
    # configured JSON filename, matching the controller's own lookup behaviour.
    file_ok = {}
    for key in ("URDF_PATH", "YAML_PATH"):
        path = os.path.expanduser(str(cfg[key]))
        file_ok[key] = os.path.isfile(path)
        report(f"{key}", file_ok[key], path)

    json_dir = os.path.expanduser(str(cfg["JSON_DIR"]))
    json_path = os.path.join(json_dir, cfg["JSON_FILE"])
    if not os.path.isfile(json_path) and os.path.isdir(json_dir):
        for root, _, names in os.walk(json_dir):
            if cfg["JSON_FILE"] in names:
                json_path = os.path.join(root, cfg["JSON_FILE"])
                break
    json_exists = os.path.isfile(json_path)
    report("keypoint JSON", json_exists, json_path)
    if not json_exists:
        candidates = []
        if os.path.isdir(json_dir):
            candidates = [os.path.join(json_dir, f) for f in os.listdir(json_dir)
                          if f.lower().endswith(".json")]
        report("JSON files in configured directory", bool(candidates),
               ", ".join(candidates[:12]) if candidates else f"none; directory exists={os.path.isdir(json_dir)}",
               blocking=False)
    else:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            grasps = _points_from(_find_key(data, ("grasp_points", "grasp_point", "grasps")))
            places = _points_from(_find_key(data, ("placement_candidates", "placement_points", "place_points")))
            obstacles = _points_from(_find_key(data, ("obstacle_points", "obstacles")))
            size = _image_size_from(data) or tuple(cfg["DEFAULT_IMAGE_SIZE"])
            counts_ok = bool(grasps) and bool(places)
            report("keypoint schema", counts_ok,
                   f"grasps={len(grasps)}, placements={len(places)}, obstacles={len(obstacles)}, image={size[0]}x{size[1]}")
            all_points = grasps + places + obstacles
            all_norm = bool(all_points) and all(0 <= p["uv"][0] <= 1 and 0 <= p["uv"][1] <= 1
                                                for p in all_points)
            if all_norm:
                coord_ok = True
                detail = "coordinates appear normalized; controller scales to image pixels"
            else:
                coord_ok = all(0 <= p["uv"][0] < size[0] and 0 <= p["uv"][1] < size[1]
                               for p in all_points)
                detail = "pixel coordinates in bounds" if coord_ok else "one or more points are outside image bounds"
            report("keypoint coordinates", coord_ok, detail)
        except Exception as e:
            report("keypoint JSON parse", False, repr(e))

    # Parse the URDF and cross-check the selected joint/link names without importing it.
    urdf_path = os.path.expanduser(str(cfg["URDF_PATH"]))
    if os.path.isfile(urdf_path):
        try:
            import xml.etree.ElementTree as ET
            urdf = ET.parse(urdf_path).getroot()
            link_names = {x.get("name") for x in urdf.findall("link")}
            joint_names = {x.get("name") for x in urdf.findall("joint")}
            required_joints = {cfg["JOINT_BASE"], cfg["JOINT_SHOULDER"], cfg["JOINT_ELBOW"]}
            missing_joints = sorted(required_joints - joint_names)
            report("URDF joint names", not missing_joints,
                   f"missing={missing_joints}; found {len(joint_names)} joints")
            report("URDF end-effector link", cfg["EE_FRAME"] in link_names,
                   f"{cfg['EE_FRAME']} in URDF links={cfg['EE_FRAME'] in link_names}")
            mesh_refs = [m.get("filename", "") for m in urdf.findall(".//mesh") if m.get("filename")]
            unresolved = []
            for ref in mesh_refs:
                if ref.startswith("file://"):
                    candidate = ref[7:]
                elif os.path.isabs(ref):
                    candidate = ref
                elif ref.startswith("package://"):
                    # Package URIs need package-path mapping; do not silently guess it.
                    candidate = ""
                else:
                    candidate = os.path.join(os.path.dirname(urdf_path), ref)
                if not candidate or not os.path.exists(candidate):
                    unresolved.append(ref)
            report("URDF mesh references", not unresolved,
                   "all resolvable" if not unresolved else f"unresolved references (may break Lula/import): {unresolved[:10]}",
                   blocking=False)
        except Exception as e:
            report("URDF parse", False, repr(e))

    # Confirm object body APIs are present somewhere on each object's ancestry.
    def _has_rigid_body_ancestor(prim):
        p = prim
        while p.IsValid() and not p.IsPseudoRoot():
            if p.HasAPI(UsdPhysics.RigidBodyAPI):
                return True
            p = p.GetParent()
        return False

    red = stage.GetPrimAtPath(cfg["RED_PRIM"])
    if red.IsValid():
        has_red_body = _has_rigid_body_ancestor(red)
        report("red cylinder rigid body", has_red_body,
               "RigidBodyAPI found on prim/ancestor" if has_red_body
               else "no RigidBodyAPI on the cylinder or its ancestors; kinematic latch requires one")
        report("red cylinder collision API", red.HasAPI(UsdPhysics.CollisionAPI),
               "collision enabled schema present" if red.HasAPI(UsdPhysics.CollisionAPI)
               else "red cylinder needs collision geometry for grasp/contact checks")

    # These scene props are meant to remain static obstacles/landmarks. Their
    # custom USD raycasts can see render geometry without collision, but PhysX
    # contact monitoring cannot reliably report contact unless collision APIs exist.
    for label, path in (("purple cone", cfg["CONE_PRIM"]), ("green sphere", cfg["SPHERE_PRIM"])):
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            continue
        is_mesh = prim.IsA(UsdGeom.Mesh)
        has_collision = prim.HasAPI(UsdPhysics.CollisionAPI)
        has_mesh_collision = prim.HasAPI(UsdPhysics.MeshCollisionAPI) if hasattr(UsdPhysics, "MeshCollisionAPI") else True
        has_body_ancestor = _has_rigid_body_ancestor(prim)
        report(f"{label} is a USD mesh", is_mesh, f"type={prim.GetTypeName()}")
        report(f"{label} collision API", has_collision and has_mesh_collision,
               f"CollisionAPI={has_collision}, MeshCollisionAPI={has_mesh_collision}; static obstacle needs a collider")
        report(f"{label} remains static", not has_body_ancestor,
               "no RigidBodyAPI on prim/ancestor" if not has_body_ancestor
               else "unexpected rigid body on prim/ancestor; review before autonomous mode")

    # List candidate YAML files if the configured name was not found. No automatic
    # substitution is made because selecting the wrong robot-description YAML is unsafe.
    if not file_ok.get("YAML_PATH", False):
        yaml_roots = [_DOBOT_ASSETS, _PROJECT_DIR]
        yaml_candidates = []
        for root in yaml_roots:
            if os.path.isdir(root):
                for walk_root, _, names in os.walk(root):
                    for name in names:
                        if name.lower().endswith((".yaml", ".yml")) and ("dobot" in name.lower() or "robot" in name.lower()):
                            yaml_candidates.append(os.path.join(walk_root, name))
                    if len(yaml_candidates) >= 30:
                        break
        report("possible robot YAML files", bool(yaml_candidates),
               "; ".join(yaml_candidates[:15]) if yaml_candidates else "none found under Downloads/n or project worktree",
               blocking=False)

    blocking_failures = [name for name, ok, blocking in results if blocking and not ok]
    print("-" * 76)
    if blocking_failures:
        print("PREFLIGHT RESULT: BLOCKED")
        print("Resolve these blocking checks before autonomous mode: " + "; ".join(blocking_failures))
        print("Reminder: the controller starts the timeline and commands motion in AUTONOMOUS mode.")
        return False
    print("PREFLIGHT RESULT: PASS (static checks only; no motion tested)")
    print("Autonomous motion has NOT been tested or authorized by this preflight.")
    return True


# =============================================================================
# Simulation-run preparation: repair known blockers in memory, never save
# =============================================================================
def _walk_files(roots, suffixes, max_files=20000):
    """Yield plausible candidate files while avoiding huge/generated folders."""
    suffixes = tuple(s.lower() for s in suffixes)
    skip_dirs = {
        ".git", "node_modules", "__pycache__", ".venv", "venv", "env",
        "site-packages", "build", "dist", "models", "checkpoints", "cache",
    }
    count = 0
    seen = set()
    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        root = os.path.realpath(os.path.expanduser(root))
        for walk_root, dirs, names in os.walk(root):
            dirs[:] = [d for d in dirs if d not in skip_dirs]
            for name in names:
                if name.lower().endswith(suffixes):
                    path = os.path.realpath(os.path.join(walk_root, name))
                    if path not in seen:
                        seen.add(path)
                        yield path
                        count += 1
                        if count >= max_files:
                            return


def _valid_dobot_urdf(path):
    try:
        import xml.etree.ElementTree as ET
        root = ET.parse(path).getroot()
        joints = {j.get("name") for j in root.findall("joint")}
        links = {l.get("name") for l in root.findall("link")}
        required_joints = {"joint_1", "joint_2", "joint_5", "joint_6"}
        return required_joints.issubset(joints) and "link_suction" in links
    except Exception:
        return False


def _valid_dobot_yaml(path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read(1_000_000).lower()
        # This must be a Lula-style description for this joint set, not any random YAML.
        return ("cspace" in content and all(n in content for n in
                ("joint_1", "joint_2", "joint_5")) and
                ("root_link" in content or "base_link" in content))
    except Exception:
        return False


def _valid_keypoint_json(path, cfg):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        grasps = _points_from(_find_key(data, ("grasp_points", "grasp_point", "grasps")))
        places = _points_from(_find_key(data, ("placement_candidates", "placement_points", "place_points")))
        return bool(grasps) and bool(places)
    except Exception:
        return False


def _resolve_required_files(cfg):
    """Resolve configured files, or find exact-content candidates in the project roots."""
    home = os.path.expanduser("~")
    asset_root = _DOBOT_ASSETS
    project_root = cfg.get("_PROJECT_DIR", _PROJECT_DIR) if "_PROJECT_DIR" in cfg else _PROJECT_DIR
    roots = [asset_root, project_root, _DOBOT_DATA]

    urdf = os.path.expanduser(str(cfg.get("URDF_PATH", "")))
    if not os.path.isfile(urdf) or not _valid_dobot_urdf(urdf):
        matches = [p for p in _walk_files(roots, (".urdf",)) if _valid_dobot_urdf(p)]
        matches.sort(key=lambda p: (0 if os.path.basename(p).lower() == "load_dobot.urdf" else 1,
                                    0 if asset_root in p else 1, len(p)))
        if matches:
            urdf = matches[0]
    if not os.path.isfile(urdf) or not _valid_dobot_urdf(urdf):
        print("[BLOCK] Could not find a URDF containing joint_1, joint_2, joint_5, joint_6 and link_suction.")
        print("        Looked under:", roots)
        return False
    cfg["URDF_PATH"] = urdf
    print("[PATH] URDF:", urdf)

    yaml = os.path.expanduser(str(cfg.get("YAML_PATH", "")))
    if not os.path.isfile(yaml) or not _valid_dobot_yaml(yaml):
        matches = [p for p in _walk_files(roots, (".yaml", ".yml")) if _valid_dobot_yaml(p)]
        matches.sort(key=lambda p: (0 if "dobot" in os.path.basename(p).lower() else 1,
                                    0 if asset_root in p else 1, len(p)))
        if matches:
            yaml = matches[0]
    if not os.path.isfile(yaml) or not _valid_dobot_yaml(yaml):
        print("[BLOCK] Could not find a Lula-style YAML for joint_1, joint_2 and joint_5.")
        print("        No YAML will be fabricated. Add/correct the Dobot Lula robot-description YAML.")
        return False
    cfg["YAML_PATH"] = yaml
    print("[PATH] Lula YAML:", yaml)

    json_dir = os.path.expanduser(str(cfg.get("JSON_DIR", "")))
    json_path = os.path.join(json_dir, str(cfg.get("JSON_FILE", "keypoints_2d.json")))
    if not os.path.isfile(json_path) or not _valid_keypoint_json(json_path, cfg):
        candidates = []
        # Search the configured subtree first; then the known project roots.
        for p in _walk_files([json_dir, project_root,
                              _DOBOT_DATA], (".json",)):
            if os.path.basename(p).lower() == "keypoints_2d.json" and _valid_keypoint_json(p, cfg):
                candidates.append(p)
        # Prefer a capture-specific output directory if multiple copies exist.
        candidates.sort(key=lambda p: (0 if "new_scene_red_cylinder" in p else 1, len(p)))
        if candidates:
            json_path = candidates[0]
    if not os.path.isfile(json_path) or not _valid_keypoint_json(json_path, cfg):
        print("[BLOCK] No usable keypoints_2d.json with grasp and placement points was found.")
        print("        Expected at:", os.path.join(json_dir, str(cfg.get("JSON_FILE", "keypoints_2d.json"))))
        return False
    cfg["JSON_DIR"] = os.path.dirname(json_path)
    cfg["JSON_FILE"] = os.path.basename(json_path)
    cfg["DEPTH_OUT_DIR"] = os.path.dirname(json_path)
    print("[PATH] Keypoints:", json_path)
    return True


def _has_rigid_body_ancestor_for_run(prim):
    p = prim
    while p.IsValid() and not p.IsPseudoRoot():
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            return True
        p = p.GetParent()
    return False


def _prepare_scene_for_simulation(cfg):
    """Apply only known in-memory fixes on the named test scene; never save/start physics here."""
    print("\n" + "=" * 76)
    print("DOBOT SIMULATION RUNNER — SAFE PREPARATION")
    print("Edits, if any, are in memory only. No file is saved by this script.")
    print("The controller may move the robot only after all remaining preflight checks pass.")
    print("=" * 76)

    prev = getattr(builtins, "_DOBOT_PNP_INSTANCE", None)
    if prev is not None and not getattr(prev, "_shutting_down", False):
        print("[BLOCK] A previous Dobot controller is still active. Stop it before running this file again.")
        return False

    timeline = omni.timeline.get_timeline_interface()
    if not timeline.is_stopped():
        print("[INFO] Stopping the timeline before preparation (no controller instance is active).")
        timeline.stop()
        if not timeline.is_stopped():
            print("[BLOCK] Timeline could not be stopped. No scene preparation or robot commands issued.")
            return False

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        print("[BLOCK] No USD stage is open.")
        return False
    root_id = stage.GetRootLayer().identifier
    root_name = os.path.basename(root_id).lower()
    allowed = ("stage3_dobot_gripper_test" in root_name or
               "stage3_dobot_gripper_safe_prepped" in root_name or
               "stage3_dobot_repaired_test" in root_name)
    if not allowed:
        print("[BLOCK] Refusing to prepare a stage that does not look like the Dobot test copy:", root_id)
        print("        Open stage3_dobot_gripper_test.usd (or a safe-prepped test copy).")
        return False
    if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
        print("[BLOCK] Stage must be Z-up.")
        return False
    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    if mpu is not None and abs(float(mpu) - 1.0) > 1e-6:
        print("[BLOCK] Stage metersPerUnit must be 1.0 for this controller; actual:", mpu)
        return False

    # Required prims must exist before we edit anything.
    required = [cfg["ROBOT_PRIM"], cfg["EE_PRIM"], cfg["CAMERA_PRIM"],
                cfg["RED_PRIM"], cfg["CONE_PRIM"], cfg["SPHERE_PRIM"]]
    missing = [p for p in required if not stage.GetPrimAtPath(p).IsValid()]
    if missing:
        print("[BLOCK] Required scene prims are missing:", missing)
        return False
    for path in (cfg["CONE_PRIM"], cfg["SPHERE_PRIM"]):
        prim = stage.GetPrimAtPath(path)
        if not prim.IsA(UsdGeom.Mesh):
            print(f"[BLOCK] Expected a USD Mesh at {path}, got {prim.GetTypeName()}.")
            return False
        if _has_rigid_body_ancestor_for_run(prim):
            print(f"[BLOCK] {path} has a rigid-body ancestor; refusing to auto-make it a static obstacle.")
            return False
        if not UsdPhysics.CollisionAPI.CanApply(prim) or not UsdPhysics.MeshCollisionAPI.CanApply(prim):
            print(f"[BLOCK] Cannot apply static mesh-collision schemas to {path}.")
            return False

    # Resolve all external files before changing the stage.
    if not _resolve_required_files(cfg):
        print("SIMULATION RUN ABORTED before scene edits or robot motion because required files are missing/invalid.")
        return False

    # Author all intended changes to the root layer, but do not save the USD file.
    old_edit_target = stage.GetEditTarget()
    stage.SetEditTarget(stage.GetRootLayer())
    changed = []
    try:
        # Correct stale drive targets so physics startup does not drive toward the old authored pose.
        for name in (cfg["JOINT_BASE"], cfg["JOINT_SHOULDER"], cfg["JOINT_ELBOW"], cfg["JOINT_PITCH"]):
            path = f'{cfg["ROBOT_PRIM"]}/joints/{name}'
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid() or not prim.IsA(UsdPhysics.Joint):
                print("[BLOCK] Required joint missing/not a PhysicsJoint:", path)
                return False
            target = prim.GetAttribute("drive:angular:physics:targetPosition")
            state = prim.GetAttribute("state:angular:physics:position")
            if target.IsValid() and state.IsValid():
                target_deg = target.Get()
                state_rad = state.Get()
                if target_deg is not None and state_rad is not None:
                    state_deg = math.degrees(float(state_rad))
                    if abs(float(target_deg) - state_deg) > 0.1:
                        target.Set(float(state_deg))
                        changed.append(f"{name} drive target {float(target_deg):.2f} -> {state_deg:.2f} deg")

        # The autonomous controller uses a kinematic latch, not the manual D6 joint.
        d6_path = f'{cfg["EE_PRIM"]}/surface_gripper/suction_joint'
        d6 = stage.GetPrimAtPath(d6_path)
        if d6.IsValid():
            en = d6.GetAttribute("physics:jointEnabled")
            if not en.IsValid():
                en = d6.CreateAttribute("physics:jointEnabled", Sdf.ValueTypeNames.Bool, True)
            if en.Get() is not False:
                en.Set(False)
                changed.append(f"disabled unused D6 joint {d6_path}")

        # Make cone and sphere real static PhysX obstacles for contact monitoring.
        try:
            from pxr import PhysxSchema
        except Exception:
            PhysxSchema = None
        for path in (cfg["CONE_PRIM"], cfg["SPHERE_PRIM"]):
            prim = stage.GetPrimAtPath(path)
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI.Apply(prim)
                changed.append(f"added CollisionAPI to {path}")
            if not prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                UsdPhysics.MeshCollisionAPI.Apply(prim)
                changed.append(f"added MeshCollisionAPI to {path}")
            if PhysxSchema is not None and not prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
                PhysxSchema.PhysxCollisionAPI.Apply(prim)
                changed.append(f"added PhysxCollisionAPI to {path}")
            collision_api = UsdPhysics.CollisionAPI(prim)
            en = collision_api.GetCollisionEnabledAttr()
            if not en.IsValid():
                en = collision_api.CreateCollisionEnabledAttr()
            if en.Get() is not True:
                en.Set(True)
                changed.append(f"enabled collision on {path}")
            mesh_api = UsdPhysics.MeshCollisionAPI(prim)
            approx = mesh_api.GetApproximationAttr()
            if not approx.IsValid():
                approx = mesh_api.CreateApproximationAttr()
            if approx.Get() != "convexHull":
                approx.Set("convexHull")
                changed.append(f"set convexHull collision on {path}")
    except Exception as e:
        print("[BLOCK] Safe scene preparation failed:", repr(e))
        print("        The USD was not saved. Inspect the stage before continuing.")
        return False
    finally:
        stage.SetEditTarget(old_edit_target)

    if changed:
        for line in changed:
            print("[PREP]", line)
    else:
        print("[PREP] No known in-memory repairs were necessary.")
    print("[PREP] Done; stage remains unsaved. Controller preflight follows.")
    return True


# =============================================================================
# Safe simulation entry point (async timeline stop + clean-stage reload)
# =============================================================================
async def _stop_timeline_and_wait(max_frames=90):
    tl = omni.timeline.get_timeline_interface()
    if not tl.is_stopped():
        print("[INFO] Requesting timeline stop and waiting for Kit updates...")
        tl.stop()
    app = omni.kit.app.get_app()
    for _ in range(max_frames):
        if tl.is_stopped():
            return True
        await app.next_update_async()
    return bool(tl.is_stopped())


async def _run_simulation_from_clean_stage(cfg):
    """Stop timeline, reload the saved test USD to restore moved objects, then prepare and run."""
    print("\n" + "=" * 76)
    print("DOBOT SIMULATION RUNNER V2 — RESET + SAFE START")
    print("This reloads the saved test USD, discarding unsaved stage edits.")
    print("Controller uses a kinematic latch; the manually created D6 joint is not required.")
    print("=" * 76)
    app = omni.kit.app.get_app()
    ctx = omni.usd.get_context()

    # Tear down a previous controller cleanly before stopping/reloading the stage.
    prev = getattr(builtins, "_DOBOT_PNP_INSTANCE", None)
    if prev is not None and not getattr(prev, "_shutting_down", False):
        try:
            prev.shutdown("reset before retry")
        except Exception as exc:
            print("[WARN] Previous controller shutdown reported:", repr(exc))
    for _ in range(2):
        await app.next_update_async()

    stage = ctx.get_stage()
    if stage is None:
        print("[BLOCK] No USD stage is open; open stage3_dobot_gripper_test.usd first.")
        return
    root_layer = stage.GetRootLayer()
    root_path = getattr(root_layer, "realPath", "") or root_layer.identifier
    root_name = os.path.basename(root_path).lower()
    allowed = ("stage3_dobot_gripper_test" in root_name or
               "stage3_dobot_gripper_safe_prepped" in root_name or
               "stage3_dobot_repaired_test" in root_name)
    if not allowed:
        print("[BLOCK] Refusing to reset a non-test scene:", root_path)
        return
    if not os.path.isfile(root_path):
        print("[BLOCK] The active stage root is not a local file that can be reloaded:", root_path)
        print("        Stop playback manually, then reopen the saved stage3_dobot_gripper_test.usd copy.")
        return

    # Prefer the known clean repaired baseline if it exists. That scene predates any
    # successful/partial controller run and therefore restores the original object poses.
    clean_baseline = os.path.join(_ASSET_DIR, "stage3_dobot_repaired_test_20261021_040412.usd")
    reset_path = clean_baseline if os.path.isfile(clean_baseline) else root_path

    if not await _stop_timeline_and_wait(90):
        print("[BLOCK] Timeline did not stop after waiting for 90 Kit frames.")
        print("        Do not continue. Stop the timeline from the UI; if it still won't stop, restart Isaac Sim.")
        return

    # Reload the saved test copy to reset any in-memory object movement from a prior attempt.
    try:
        print("[RESET] Reloading saved clean stage:", reset_path)
        if reset_path != root_path:
            print("[RESET] Using the original repaired baseline to restore pre-run object poses.")
        await ctx.open_stage_async(reset_path)
        for _ in range(30):
            await app.next_update_async()
            try:
                loading = ctx.is_stage_loading()
            except Exception:
                loading = False
            if not loading:
                break
        stage = ctx.get_stage()
        if stage is None:
            print("[BLOCK] USD stage is None after reload.")
            return
        new_root = getattr(stage.GetRootLayer(), "realPath", "") or stage.GetRootLayer().identifier
        if os.path.basename(new_root) != os.path.basename(reset_path):
            print("[BLOCK] Reloaded stage root does not match the selected reset file:", new_root)
            return
    except Exception as exc:
        print("[BLOCK] Could not reload the saved test stage:", repr(exc))
        print("        Manually reopen the saved test USD before retrying.")
        return

    # The saved USD is the known starting configuration; old in-memory run history is stale now.
    builtins._DOBOT_PNP_LAST_PLACED = None
    builtins._DOBOT_PNP_INSTANCE = None

    if not await _stop_timeline_and_wait(90):
        print("[BLOCK] Timeline resumed or did not remain stopped after USD reload.")
        return

    if not _prepare_scene_for_simulation(cfg):
        print("\nSIMULATION NOT STARTED: preparation was blocked; no controller motion was issued.")
        return
    if not _run_preflight(cfg):
        print("\nSIMULATION NOT STARTED: remaining blocking preflight checks are listed above.")
        return

    print("\n[RUN] Preflight passed. Starting reduced-speed pick-and-place simulation.")
    print("[RUN] If the saved keypoints miss the cylinder, the controller now tries a projected top-centre grasp;")
    print("[RUN] it also adds fallback placement pixels around the purple cone and auto-projected obstacle points.")
    print("[RUN] Timeline is stopped automatically if the pipeline errors. Keep Isaac Sim visible.")
    inst = DobotPickPlace(cfg)
    builtins._DOBOT_PNP_INSTANCE = inst
    inst.start()


_MODE = str(CFG.get("RUN_MODE", "SIMULATION")).strip().upper()
if _MODE == "PREFLIGHT":
    _run_preflight(CFG)
elif _MODE == "AUTONOMOUS":
    # Strict mode, intentionally does not reset/reload the scene.
    if _run_preflight(CFG):
        _prev = getattr(builtins, "_DOBOT_PNP_INSTANCE", None)
        if _prev is not None:
            _prev.shutdown("script re-run")
        builtins._DOBOT_PNP_INSTANCE = DobotPickPlace(CFG)
        builtins._DOBOT_PNP_INSTANCE.start()
    else:
        print("AUTONOMOUS MODE NOT STARTED: preflight is blocked.")
elif _MODE == "SIMULATION":
    _task = asyncio.ensure_future(_run_simulation_from_clean_stage(CFG))
    builtins._DOBOT_PNP_RUNNER_TASK = _task
else:
    print(f"Unknown RUN_MODE={_MODE!r}; choose 'PREFLIGHT', 'AUTONOMOUS', or 'SIMULATION'. No motion commanded.")
