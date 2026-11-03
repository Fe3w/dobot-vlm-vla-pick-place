# Read-only Isaac Sim scene diagnostic. Does not edit USD, start physics, or command motion.
import os
import math
import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics

lines = []
def out(s=""):
    lines.append(str(s))

stage = omni.usd.get_context().get_stage()
out("ISAAC SIM READ-ONLY SCENE PATH DIAGNOSTIC")
out("=" * 78)
if stage is None:
    out("ERROR: no USD stage is open")
else:
    try:
        out("Root layer: " + str(stage.GetRootLayer().identifier))
    except Exception as e:
        out("Root layer unavailable: " + repr(e))
    out("Up axis: " + str(UsdGeom.GetStageUpAxis(stage)))
    out("Meters per unit: " + str(UsdGeom.GetStageMetersPerUnit(stage)))
    out()
    out("TOP-LEVEL PRIMS")
    for prim in stage.GetPseudoRoot().GetChildren():
        out(f"  {prim.GetPath()} | type={prim.GetTypeName()}")
    out()
    out("EXPECTED PATHS")
    expected = ["/magician", "/magician/link_suction", "/magician/color",
                "/overhead_camera", "/Cone", "/Sphere"]
    for path in expected:
        p = stage.GetPrimAtPath(path)
        out(f"  {path}: {'FOUND type=' + p.GetTypeName() if p.IsValid() else 'MISSING'}")
    out()
    out("PRIMS WHOSE NAME OR PATH CONTAINS OBJECT KEYWORDS")
    tokens = ("cone", "sphere", "purple", "green", "cylinder", "color", "target", "obstacle")
    hits = []
    for p in stage.Traverse():
        label = (p.GetName() + " " + str(p.GetPath())).lower()
        if any(t in label for t in tokens):
            hits.append(p)
    if not hits:
        out("  No keyword matches found")
    else:
        for p in hits:
            try:
                rb = p.HasAPI(UsdPhysics.RigidBodyAPI)
            except Exception:
                rb = False
            try:
                col = p.HasAPI(UsdPhysics.CollisionAPI)
            except Exception:
                col = False
            out(f"  {p.GetPath()} | type={p.GetTypeName()} | RigidBodyAPI={rb} | CollisionAPI={col}")
    out()
    out("ROBOT JOINT TARGETS VERSUS AUTHORED ANGULAR STATE")
    for name in ("joint_1", "joint_2", "joint_5", "joint_6"):
        path = f"/magician/joints/{name}"
        p = stage.GetPrimAtPath(path)
        if not p.IsValid():
            out(f"  {path}: MISSING")
            continue
        ta = p.GetAttribute("drive:angular:physics:targetPosition")
        sa = p.GetAttribute("state:angular:physics:position")
        target = ta.Get() if ta.IsValid() else None
        state = sa.Get() if sa.IsValid() else None
        if target is not None and state is not None:
            state_deg = math.degrees(float(state))
            delta = abs(float(target) - state_deg)
            out(f"  {name}: target={float(target):.3f} deg; state={state_deg:.3f} deg; delta={delta:.3f} deg")
        else:
            out(f"  {name}: target={target!r}; state_rad={state!r}")
    out()
    out("No stage edits, saves, playback, or robot commands were performed.")

report = "\n".join(lines)
print(report)
try:
    target_path = os.path.expanduser("~/dobot_reports/isaacsim_scene_path_diagnostic.txt")
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print("Diagnostic report saved to: " + target_path)
except Exception as e:
    print("Could not save diagnostic report: " + repr(e))
