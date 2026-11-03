"""
dobot_rgb_view.py - what the robot "sees" with an RGB (phone) camera, drawn for the demo.

Without a depth camera there is no measured depth picture. What the RGB pipeline does know, and what this draws
as two panels side by side (same layout as the simulation's depth view):

  LEFT  "height map (estimated)"  gray picture in the camera's view: table dark, each detected object filled with
        a gray level for its height (lighter = taller, same idea as the depth view). Heights are not measured:
        they come from the OBJECTS catalogue in dobot_real.py (an RGB camera cannot see height), so each object
        is labelled with its height and where it came from.
  RIGHT "phone camera"  the live photo with: each object's outline, name and position in robot coordinates (mm);
        the robot's reach ring on the table; the pick point (green ring), the place point (white cross), the
        planned carry path (yellow) from the checked simulation run; the real suction tip (orange dot + trail)
        while the robot moves.

Pure numpy + Pillow (both in Isaac Sim's Python; `pip install pillow` in the real-robot environment). Used by
the task window ("Phone camera" window) and by `dobot_real.py scan` / `plan` (saved as PNGs).
"""
import math

import numpy as np

VIEW = {
    "REACH_MM": (135.0, 315.0),       # drawn reach ring on the table (match LIMIT_R_MIN / LIMIT_R_MAX)
    "TABLE_GRAY": 45,                 # left panel: table level
    "TOP_GRAY": 245,                  # left panel: the tallest object
    "MIN_SCALE_MM": 60.0,             # gray scale spans at least this many mm
    "OUTLINE": {"pick": (60, 230, 60), "place": (80, 220, 255), "other": (255, 80, 255)},
    "PATH": (255, 215, 0),
    "TIP": (255, 140, 0),
    "RING": (200, 200, 200),
    "RING_PHOTO": (30, 30, 30),       # reach ring on the photo (dark, the table is usually light)
}


def _font(size):
    from PIL import ImageFont
    for name in ("DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _text(dr, xy, s, col, size=16):
    try:
        dr.text(xy, s, fill=tuple(col), font=_font(size), stroke_width=2, stroke_fill=(0, 0, 0))
    except TypeError:
        dr.text(xy, s, fill=tuple(col))


# ---- geometry (pure) -------------------------------------------------------------------------------------
def to_pixel(calib, P_mm):
    """Robot-frame points (mm, tip positions) -> pixels with the phone calibration (projection or homography)."""
    P = np.atleast_2d(np.asarray(P_mm, float))
    if calib["model"] == "homography":
        h = np.c_[P[:, :2], np.ones(len(P))] @ np.linalg.inv(np.asarray(calib["H"], float)).T
    else:
        h = np.c_[P[:, :3], np.ones(len(P))] @ np.asarray(calib["P"], float).T
    uv = h[:, :2] / h[:, 2:3]
    k1 = float(calib.get("k1") or 0.0)
    if k1 == 0.0:
        return uv
    c, sc = np.asarray(calib["dist_center"], float), float(calib["dist_scale"])     # same lens model as dobot_real
    u = (uv - c) / sc
    ru = np.linalg.norm(u, axis=1)
    rmax = math.sqrt(-1.0 / (3.0 * k1)) if k1 < 0 else np.inf     # the model folds back beyond this radius
    rd = np.minimum(ru, 0.999 * rmax)
    for _ in range(25):                                            # Newton on rd * (1 + k1 rd^2) = ru
        f = rd * (1.0 + k1 * rd ** 2) - ru
        rd = np.clip(rd - f / (1.0 + 3.0 * k1 * rd ** 2), 0.0, 0.999 * rmax)
    scale = np.where(ru > 1e-12, rd / np.maximum(ru, 1e-12), 1.0)[:, None]
    return c + u * scale * sc


def tip_of(reported, b):
    """Real suction tip from a reported Dobot position (b = tool offset along the arm, from calib.json)."""
    p = np.array(reported, float)
    n = np.linalg.norm(p[..., :2], axis=-1, keepdims=True)
    p[..., :2] += b * p[..., :2] / np.maximum(n, 1e-9)
    return p


def ring_px(calib, r, z, n=96):
    a = np.linspace(0.0, 2.0 * math.pi, n)
    return to_pixel(calib, np.c_[r * np.cos(a), r * np.sin(a), np.full(n, z)])


def plan_points(steps, calib):
    """(path tip points mm, pick tip, place tip) from dobot_real.build_real_plan steps (reported positions)."""
    b = float(calib.get("tool_radial_mm", 0.0))
    path, pick, place, holding = [], None, None, False
    for s in steps:
        if s[0] == "suck":
            holding = bool(s[1])
            if not holding and path:
                place = path[-1]
            continue
        if s[0] != "move":
            continue
        t = tip_of(s[1], b)
        if s[3] == "touch the object":
            pick = t
        if holding or s[3] in ("touch the object",):
            path.append(t)
    return np.array(path) if path else np.zeros((0, 3)), pick, place


def outline(o):
    """Object outline in pixels: the stored contour, or its bounding box for older scans."""
    if o.get("outline_px"):
        return np.asarray(o["outline_px"], float)
    u0, v0, u1, v1 = o["bbox_px"]
    return np.array([[u0, v0], [u1, v0], [u1, v1], [u0, v1]], float)


def roles(objs, pick, place):
    """Which detected object is picked (nearest to the pick point) and which is the reference (nearest other
    object to the place point)."""
    out = {}
    if pick is not None and objs:
        i = min(range(len(objs)), key=lambda k: math.dist(objs[k]["center_mm"], pick[:2]))
        out[i] = "pick"
        if place is not None and len(objs) > 1:
            j = min((k for k in range(len(objs)) if k != i), key=lambda k: math.dist(objs[k]["center_mm"], place[:2]))
            out[j] = "place"
    return out


# ---- drawing -----------------------------------------------------------------------------------------------
def render(rgb, scan, calib, steps=None, tip_trail=None, title="", note=""):
    """Two panels (estimated height map | phone photo with overlays) + a legend strip. Returns uint8 RGB."""
    from PIL import Image, ImageDraw
    rgb = np.ascontiguousarray(np.asarray(rgb, np.uint8)[..., :3])
    h, w = rgb.shape[:2]
    objs = list((scan or {}).get("objects") or [])
    tz = float(calib.get("table_z_mm", (scan or {}).get("table_z_dobot_mm", 0.0)))
    path, pick, place = plan_points(steps, calib) if steps else (np.zeros((0, 3)), None, None)
    role = roles(objs, pick, place)
    fs = max(12, w // 80)

    # left: estimated height map
    hmax = max([VIEW["MIN_SCALE_MM"]] + [float(o["size_mm"][2]) for o in objs])
    g0, g1 = VIEW["TABLE_GRAY"], VIEW["TOP_GRAY"]
    L = Image.new("RGB", (w, h), (g0, g0, g0))
    dl = ImageDraw.Draw(L)
    for o in sorted(objs, key=lambda o: o["size_mm"][2]):
        g = int(g0 + (g1 - g0) * min(1.0, float(o["size_mm"][2]) / hmax))
        dl.polygon([tuple(p) for p in outline(o)], fill=(g, g, g))
    R = Image.fromarray(rgb)
    dr = ImageDraw.Draw(R)
    for d, rc in ((dl, VIEW["RING"]), (dr, VIEW["RING_PHOTO"])):
        for r in VIEW["REACH_MM"]:
            px = ring_px(calib, r, tz)
            ok = np.all(np.isfinite(px), axis=1)
            pts = [tuple(p) for p in px[ok]]
            for a, b in list(zip(pts[:-1], pts[1:]))[::2]:          # dashed
                d.line([a, b], fill=rc, width=2)
    for k, o in enumerate(objs):
        col = VIEW["OUTLINE"][role.get(k, "other")]
        poly = [tuple(p) for p in outline(o)]
        dr.line(poly + [poly[0]], fill=col, width=3)
        dl.line(poly + [poly[0]], fill=col, width=2)
        u0, v0 = outline(o).min(axis=0)
        hs = "catalogue" if o.get("height_from") == "catalogue" else "default"
        tag = {"pick": "  PICK", "place": "  REFERENCE"}.get(role.get(k), "")
        _text(dr, (u0, max(0, v0 - 2.6 * fs)), f"{o['name']}{tag}", col, fs)
        _text(dr, (u0, max(0, v0 - 1.4 * fs)), f"x {o['center_mm'][0]:.0f}  y {o['center_mm'][1]:.0f} mm", col, fs - 2)
        _text(dl, (u0, max(0, v0 - 1.4 * fs)), f"h {o['size_mm'][2]:.0f} mm ({hs})", (255, 255, 255), fs - 2)
        if not o.get("pickable", True):
            _text(dl, (u0, max(0, v0 - 2.6 * fs)), "not pickable (suction)", (255, 120, 120), fs - 2)
    for d, width in ((dr, 4), (dl, 3)):
        if len(path) >= 2:
            px = to_pixel(calib, path)
            d.line([tuple(p) for p in px], fill=VIEW["PATH"], width=width)
        if pick is not None:
            u, v = to_pixel(calib, pick)[0]
            d.ellipse([u - 12, v - 12, u + 12, v + 12], outline=VIEW["OUTLINE"]["pick"], width=4)
        if place is not None:
            u, v = to_pixel(calib, place)[0]
            d.line([u - 12, v, u + 12, v], fill=(255, 255, 255), width=4)
            d.line([u, v - 12, u, v + 12], fill=(255, 255, 255), width=4)
    if tip_trail:
        px = to_pixel(calib, np.asarray(tip_trail, float))
        if len(px) >= 2:
            dr.line([tuple(p) for p in px], fill=VIEW["TIP"], width=2)
        u, v = px[-1]
        dr.ellipse([u - 8, v - 8, u + 8, v + 8], fill=VIEW["TIP"], outline=(0, 0, 0), width=2)
    _text(dl, (8, 6), "height map (estimated: RGB outlines + catalogue heights)", (255, 255, 255), fs)
    _text(dr, (8, 6), "phone camera" + (f": {title}" if title else ""), (255, 255, 255), fs)

    # legend strip
    leg = Image.new("RGB", (2 * w, 3 * fs), (0, 0, 0))
    dg = ImageDraw.Draw(leg)
    n = min(256, w // 3)
    for i in range(n):
        g = int(g0 + (g1 - g0) * i / max(1, n - 1))
        dg.line([(8 + i, 4), (8 + i, fs + 4)], fill=(g, g, g))
    _text(dg, (8, fs + 6), f"table 0 mm", (220, 220, 220), fs - 3)
    _text(dg, (max(8, n - 40), fs + 6), f"{hmax:.0f} mm", (220, 220, 220), fs - 3)
    _text(dg, (16 + n, 4), "green = object to pick, blue = reference, magenta = other objects; green ring = pick "
                           "point, white + = place point, yellow = planned path, orange = real tip, dashed = reach",
          (220, 220, 220), fs - 3)
    if note:
        _text(dg, (16 + n, fs + 6), note, (220, 220, 220), fs - 3)
    out = Image.new("RGB", (2 * w, h + 3 * fs))
    out.paste(L, (0, 0))
    out.paste(R, (w, 0))
    out.paste(leg, (0, h))
    return np.asarray(out)
