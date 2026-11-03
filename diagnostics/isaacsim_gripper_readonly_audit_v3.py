# Isaac Sim 6.0.1 — READ-ONLY Dobot gripper audit V3
# Run in Isaac Sim's Script Editor with the repaired test scene open.
# This script does not edit/save the stage, start physics, or command motion.

import traceback
from pathlib import Path
import omni.usd
from pxr import Usd, UsdGeom, Sdf

GRIPPER_ROOT = "/magician/link_suction"
ASSET_DIR = Path.home() / "Downloads" / "n"
REPORT_PATH = Path.home() / "dobot_reports" / "isaacsim_gripper_readonly_audit_v3.txt"
lines = []

def log(value=""):
    value = str(value)
    lines.append(value)
    print(value)

def names(prim):
    try:
        return sorted(str(name) for name in prim.GetPropertyNames())
    except Exception:
        try:
            return sorted(str(p.GetName()) for p in prim.GetProperties())
        except Exception:
            return []

def schemas(prim):
    try:
        return list(prim.GetAppliedSchemas())
    except Exception:
        return []

def read_prop(prim, name):
    # Do not call IsRelationship/GetTypeName on a UsdRelationship wrapper.
    try:
        rel = prim.GetRelationship(name)
        if rel and rel.IsValid():
            return f"REL {name} -> {[str(t) for t in rel.GetTargets()]}"
    except Exception:
        pass
    try:
        attr = prim.GetAttribute(name)
        if attr and attr.IsValid():
            return f"ATTR {name} = {repr(attr.Get())} (type={attr.GetTypeName()})"
    except Exception as e:
        return f"PROPERTY {name} <read error: {e}>"
    return f"PROPERTY {name} <not readable as attribute or relationship>"

def world_translation(prim):
    try:
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        t = cache.GetLocalToWorldTransform(prim).ExtractTranslation()
        return f"({t[0]:.9f}, {t[1]:.9f}, {t[2]:.9f})"
    except Exception as e:
        return f"<error: {e}>"

def relevant(name):
    n = name.lower()
    return any(k in n for k in (
        "attachment", "gripped", "grip", "force", "retry", "status",
        "physics:body0", "physics:body1", "physics:jointenabled",
        "physics:excludefromarticulation", "physics:localpos",
        "physics:localrot", "forwardaxis", "clearanceoffset",
        "enabledselfcollisions"
    ))

def inspect_prim_properties(prim, filter_relevant=True):
    for name in names(prim):
        if not filter_relevant or relevant(name):
            log("  " + read_prop(prim, name))

def inspect_asset(path):
    log()
    log("=" * 78)
    log(f"ASSET: {path}")
    log("=" * 78)
    if not path.exists():
        log("MISSING")
        return
    try:
        layer = Sdf.Layer.FindOrOpen(str(path))
        if not layer:
            log("Could not open layer")
            return
        log(f"Layer identifier: {layer.identifier}")
        log(f"Sublayers: {list(layer.subLayerPaths)}")
        asset_stage = Usd.Stage.Open(layer)
        if not asset_stage:
            log("Could not open stage")
            return
        count = 0
        for prim in asset_stage.Traverse():
            ptype = str(prim.GetTypeName())
            prop_names = names(prim)
            schema_hits = [s for s in schemas(prim)
                           if "Attachment" in s or "Gripper" in s]
            if ("Joint" in ptype or "SurfaceGripper" in ptype
                or schema_hits or any(relevant(n) for n in prop_names)):
                count += 1
                log(f"PRIM {prim.GetPath()} | type={ptype}")
                if schema_hits:
                    log(f"  Schema hits: {schema_hits}")
                inspect_prim_properties(prim, filter_relevant=True)
        log(f"Matching prim count: {count}")
    except Exception:
        log("ASSET ERROR")
        log(traceback.format_exc())

try:
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("No stage is open. Open the repaired test scene first.")

    log("ISAAC SIM READ-ONLY GRIPPER AUDIT V3")
    log("=" * 78)
    log(f"Root layer: {stage.GetRootLayer().identifier}")
    log(f"Up axis: {UsdGeom.GetStageUpAxis(stage)}")
    log(f"Meters per unit: {UsdGeom.GetStageMetersPerUnit(stage)}")
    log(f"Asset directory: {ASSET_DIR}")

    log()
    log("=" * 78)
    log("A. GRIPPER PRIMS")
    log("=" * 78)
    root = stage.GetPrimAtPath(GRIPPER_ROOT)
    if not root or not root.IsValid():
        log(f"ERROR: Missing {GRIPPER_ROOT}")
    else:
        for prim in Usd.PrimRange(root):
            path = str(prim.GetPath())
            if prim == root or "SurfaceGripper" in path or "SurfaceGripper" in str(prim.GetTypeName()):
                log()
                log(f"PRIM {path} | type={prim.GetTypeName()}")
                log(f"  Applied schemas: {schemas(prim)}")
                log(f"  World translation: {world_translation(prim)}")
                inspect_prim_properties(prim, filter_relevant=False)

    log()
    log("=" * 78)
    log("B. ALL JOINTS IN ACTIVE STAGE")
    log("=" * 78)
    joint_count = 0
    for prim in stage.Traverse():
        ptype = str(prim.GetTypeName())
        if "Joint" in ptype:
            joint_count += 1
            log()
            log(f"JOINT {prim.GetPath()} | type={ptype}")
            log(f"  Applied schemas: {schemas(prim)}")
            inspect_prim_properties(prim, filter_relevant=False)
    log(f"\nJoint-like prim count: {joint_count}")

    log()
    log("=" * 78)
    log("C. STAGE-WIDE ATTACHMENT / GRIPPER SCAN")
    log("=" * 78)
    count = 0
    for prim in stage.Traverse():
        ptype = str(prim.GetTypeName())
        prop_names = names(prim)
        schema_hits = [s for s in schemas(prim)
                       if "Attachment" in s or "Gripper" in s]
        matches = [n for n in prop_names if relevant(n)]
        if "SurfaceGripper" in ptype or schema_hits or matches:
            count += 1
            log(f"PRIM {prim.GetPath()} | type={ptype}")
            if schema_hits:
                log(f"  Schema hits: {schema_hits}")
            inspect_prim_properties(prim, filter_relevant=True)
    log(f"Matching prim count: {count}")

    for filename in (
        "load_dobot_base.usd",
        "load_dobot_physics.usd",
        "load_dobot_robot.usd",
        "load_dobot_sensor.usd",
    ):
        inspect_asset(ASSET_DIR / filename)

    log()
    log("=" * 78)
    log("END OF READ-ONLY AUDIT")
    log("No stage edits, saves, physics start, or robot motion were issued.")

except Exception:
    log("FATAL AUDIT ERROR")
    log(traceback.format_exc())

try:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nReport written to: {REPORT_PATH}")
except Exception as e:
    print(f"\nCould not write report: {e}")
    print("Copy the Script Editor output instead.")
