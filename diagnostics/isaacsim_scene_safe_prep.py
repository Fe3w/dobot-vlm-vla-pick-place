"""Read-only by default helper to prepare the Dobot test scene safely.

Run in Isaac Sim 6.0.1 Script Editor with stage3_dobot_gripper_test.usd open
and the timeline STOPPED. First run previews only. To apply the listed in-memory
USD edits, change APPLY_CHANGES to True and run again. This script never starts
physics, commands robot motion, or saves the stage.
"""

import builtins
import math
import os

import omni.timeline
import omni.usd
from pxr import UsdGeom, UsdPhysics, PhysxSchema

# Explicit opt-in: first run is always a preview.
APPLY_CHANGES = False

ROBOT_ROOT = "/magician"
STATIC_MESH_PATHS = ("/magician/Cone", "/magician/Sphere")
JOINT_NAMES = ("joint_1", "joint_2", "joint_5", "joint_6")
EXPECTED_SCENE_PREFIX = "stage3_dobot_gripper_test"


def has_rigid_body_ancestor(prim):
    p = prim
    while p.IsValid() and not p.IsPseudoRoot():
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            return True
        p = p.GetParent()
    return False


def attr_value(prim, name):
    attr = prim.GetAttribute(name)
    if not attr.IsValid():
        raise RuntimeError(f"Required attribute missing: {prim.GetPath()}.{name}")
    value = attr.Get()
    if value is None:
        raise RuntimeError(f"Required attribute has no value: {prim.GetPath()}.{name}")
    return attr, value


def run():
    print("\n" + "=" * 74)
    print("DOBOT TEST SCENE SAFE PREP — no playback, no robot commands, no save")
    print("Mode:", "APPLY IN MEMORY" if APPLY_CHANGES else "DRY RUN / PREVIEW ONLY")
    print("=" * 74)

    timeline = omni.timeline.get_timeline_interface()
    if not timeline.is_stopped():
        raise RuntimeError("Timeline is not stopped. Stop playback manually; no changes made.")

    active = getattr(builtins, "_DOBOT_PNP_INSTANCE", None)
    if active is not None and not getattr(active, "_shutting_down", False):
        raise RuntimeError("A prior Dobot controller is active. Shut it down manually; no changes made.")

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("No USD stage is open.")

    root_path = stage.GetRootLayer().identifier
    root_name = os.path.basename(root_path)
    print("Stage:", root_path)
    if EXPECTED_SCENE_PREFIX not in root_name:
        raise RuntimeError(
            f"Unexpected stage filename {root_name!r}. Open your saved test copy whose filename "
            f"contains {EXPECTED_SCENE_PREFIX!r}; no changes made."
        )

    if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
        raise RuntimeError("Expected Z-up stage; no changes made.")
    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    if mpu and abs(float(mpu) - 1.0) > 1e-6:
        raise RuntimeError(f"Expected metersPerUnit=1.0, got {mpu}; no changes made.")

    # Validate all target meshes first, before any edits.
    meshes = []
    for path in STATIC_MESH_PATHS:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            raise RuntimeError(f"Required static mesh not found: {path}; no changes made.")
        if not prim.IsA(UsdGeom.Mesh):
            raise RuntimeError(f"Expected a Mesh at {path}, got {prim.GetTypeName()}; no changes made.")
        if prim.IsInstanceProxy():
            raise RuntimeError(f"{path} is an instance proxy and cannot be edited safely; no changes made.")
        if has_rigid_body_ancestor(prim):
            raise RuntimeError(
                f"{path} has a RigidBodyAPI on itself or an ancestor. This prep expects a static obstacle; "
                "no changes made."
            )
        if not UsdPhysics.CollisionAPI.CanApply(prim):
            raise RuntimeError(f"CollisionAPI cannot be applied to {path}; no changes made.")
        if not UsdPhysics.MeshCollisionAPI.CanApply(prim):
            raise RuntimeError(f"MeshCollisionAPI cannot be applied to {path}; no changes made.")

        already_collision = prim.HasAPI(UsdPhysics.CollisionAPI)
        already_mesh_collision = prim.HasAPI(UsdPhysics.MeshCollisionAPI)
        print(f"[OBJECT] {path}: Mesh; static (no RigidBodyAPI ancestor); "
              f"CollisionAPI={already_collision}; MeshCollisionAPI={already_mesh_collision}")
        meshes.append((prim, already_collision, already_mesh_collision))

    # Validate all joint values and compute target positions from authored state.
    drive_updates = []
    for name in JOINT_NAMES:
        path = f"{ROBOT_ROOT}/joints/{name}"
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid() or not prim.IsA(UsdPhysics.Joint):
            raise RuntimeError(f"Required robot joint missing/not a PhysicsJoint: {path}; no changes made.")
        target_attr, target_deg = attr_value(prim, "drive:angular:physics:targetPosition")
        state_attr, state_rad = attr_value(prim, "state:angular:physics:position")
        state_deg = math.degrees(float(state_rad))
        delta = abs(float(target_deg) - state_deg)
        print(f"[JOINT] {name}: target={float(target_deg):.3f} deg; "
              f"authored state={state_deg:.3f} deg; delta={delta:.3f} deg")
        drive_updates.append((prim, target_attr, float(target_deg), state_deg))

    print("\nProposed edits:")
    for prim, was_collision, was_mesh_collision in meshes:
        if not was_collision:
            print(f"  - Add UsdPhysics.CollisionAPI to {prim.GetPath()}")
        if not was_mesh_collision:
            print(f"  - Add UsdPhysics.MeshCollisionAPI to {prim.GetPath()}")
        print(f"  - Use convexHull approximation for static mesh {prim.GetPath()}")
    for prim, _, old_target, state_deg in drive_updates:
        if abs(old_target - state_deg) > 1e-6:
            print(f"  - Set {prim.GetPath()} drive target {old_target:.3f} -> {state_deg:.3f} degrees")

    if not APPLY_CHANGES:
        print("\nDRY RUN ONLY: no changes were made.")
        print("Review the proposed edits. If correct, set APPLY_CHANGES = True and run again.")
        return

    # Author to the root layer so the edits can be saved with File > Save As.
    old_target_layer = stage.GetEditTarget()
    stage.SetEditTarget(stage.GetRootLayer())
    try:
        for prim, _, _ in meshes:
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI.Apply(prim)
            if not prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                UsdPhysics.MeshCollisionAPI.Apply(prim)
            # PhysX collision schema is part of the Isaac Sim static-collider setup.
            if not prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
                PhysxSchema.PhysxCollisionAPI.Apply(prim)
            collision_api = UsdPhysics.CollisionAPI(prim)
            enabled_attr = collision_api.GetCollisionEnabledAttr()
            if not enabled_attr.IsValid():
                enabled_attr = collision_api.CreateCollisionEnabledAttr()
            enabled_attr.Set(True)

            mesh_api = UsdPhysics.MeshCollisionAPI(prim)
            approx_attr = mesh_api.GetApproximationAttr()
            old_approx = approx_attr.Get() if approx_attr.IsValid() else None
            if not approx_attr.IsValid():
                approx_attr = mesh_api.CreateApproximationAttr()
                old_approx = None
            if old_approx in (None, "", "none"):
                approx_attr.Set("convexHull")

        # USD Physics angular DriveAPI targetPosition is stored in degrees;
        # authored state:angular:physics:position is read in radians here.
        for _, target_attr, _, state_deg in drive_updates:
            target_attr.Set(float(state_deg))
    finally:
        stage.SetEditTarget(old_target_layer)

    print("\n[APPLIED IN MEMORY] Static collision APIs and drive targets were updated.")
    print("No physics was started, no robot motion was commanded, and the stage was NOT saved.")
    print("Next: File > Save As -> stage3_dobot_gripper_safe_prepped.usd")
    print("Then reopen that saved copy and run integrated_preflight_v2.py in PREFLIGHT mode.")


try:
    run()
except Exception as exc:
    print("\nSAFE PREP ABORTED:", repr(exc))
    print("If this was a dry run, no changes were made. If APPLY_CHANGES=True, inspect the stage before saving.")
