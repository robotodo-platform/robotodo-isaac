- USD commands: see [here](/home/sysadmin/lab/robotodo/.conda/lib/python3.11/site-packages/omni/data/Kit/kit/107.3/exts/3/omni.usd-1.13.10+8131b85d.lx64.r.cp311/omni/usd/commands/usd_commands.py)
- physics: /home/sysadmin/lab/robotodo/.conda/lib/python3.11/site-packages/omni/data/Kit/Isaac-Sim Full/5.0/exts/3/omni.physx-107.3.18+107.3.1.lx64.r.cp311.u353/omni/physx/scripts/deformableUtils.py

-
```
def usd_physics_body_get_kinematic_enabled(prim):
    import pxr

    value_omni = False
    if "OmniPhysicsBodyAPI" in prim.GetAppliedSchemas():
        value_omni = prim.GetAttribute("omniphysics:kinematicEnabled").Get()

    value = False
    rigid_body_api = pxr.UsdPhysics.RigidBodyAPI(prim)
    if rigid_body_api:
        value = rigid_body_api.GetKinematicEnabledAttr().Get()

    return value or value_omni
```