- TODO: license:
    ```
    # SPDX-License-Identifier: Apache-2.0
    ```

- TODO: doc:
    ```
    .conda/bin/../lib/libstdc++.so.6: version `GLIBCXX_3.4.30' not found (required by /home/sysadmin/lab/robotodo/.conda/lib/python3.11/site-packages/omni/libcarb.so)
    ```
    ```
    conda install -c conda-forge gcc=12 -y
    ```

- TODO:
    ```
    import omni.kit.app

    # extension enablement
    app = omni.kit.app.get_app()
    em = app.get_extension_manager()
    em.set_extension_enabled_immediate("isaacsim.app.about", True)
    em.set_extension_enabled_immediate("omni.services.livestream.nvcf", True)

    # extension settings
    import carb
    settings = carb.settings.get_settings()
    settings.get("exts/omni.services.transport.server.http/port")
    # settings.set("/app/runLoops/main/rateLimitEnabled", False)          # or True + raise Hz
    settings.get("/app/runLoops/main/rateLimitEnabled")
    settings.get_settings_dictionary()
    settings.get("/app/python/logSysStdOutput")
    settings.get("/app/enableStdoutOutput")


    ext_manager = kernel._app_framework.app.get_extension_manager()
    ext_manager.set_extension_enabled_immediate("isaacsim.exp.base", True)
    ```

- TODO: agilex curobo demo
    0. `Path`
        - `Path("/some/expression/**")`
    0. `Scene`
        - `.copy("/**", target=["/a", "/b"])`
    0. `Group`
        - `.pose`
        - `.bounding_box`
    1. `Camera("/**")`
        - `.pose`: read/write
    2. `Articulation("/**")`
        - ensure homogenous
        - `.dof_positions`: read/write
    
```python
    scene = Scene()

    camera = Camera(
        "/World/Camera{1..128}", 
        resolution=(32, 16),
        scene=scene,
    )

    scene.copy(
        "/World/Camera1",
        target="/World/Camera_0_{1..128}",
    )
```


```
# def todo_get_physx():
#     s = kernel.omni.physx.get_physx_simulation_interface()
#     # TODO NOTE valid stage is REQUIRED!!!
#     s.attach_stage(kernel.omni.usd.get_context().get_stage_id())

# def todo_sim():
#     s.simulate(0, 1)
#     s.fetch_results()

# def todo_new_stage():
#     kernel.omni.usd.get_context().new_stage()


# # kernel.submit(todo_sim).result()
# kernel.submit(todo_new_stage).result()
# kernel.submit(todo_sim).result()

# # TODO NOTE valid stage is REQUIRED!!!
# s.attach_stage(kernel.omni.usd.get_context().get_stage_id())
# s.get_attached_stage()
# s.simulate(0, 1)
# # s.fetch_results()
# s.detach_stage()
# kernel.omni.usd.get_context().new_stage()
# s.get_attached_stage()
# dir(kernel._omni.physx.get_physx_simulation_interface())

def todo():
    context = kernel.omni.usd.create_context("sfsa")
    context.can_open_stage()
    context.get_stage()
    context.new_stage()
    return context.get_stage_id()

kernel.submit(todo).result()
kernel.omni.usd.get_context().get_stage()
todo()
kernel.omni.usd.get_context("fafs")
kernel.omni.usd.get_context().can_open_stage()
kernel.omni.usd.get_context_from_stage_id(0)

dir(kernel.omni.usd.get_context())

stage = (kernel.omni.usd.get_context().get_stage())
# dir(stage)

kernel.omni.usd.get_context().get_stage_id()
dir(kernel.pxr.Usd)
# kernel.pxr.UsdUtils.StageCache.Get().Find(kernel.pxr.Usd.Id(0))
dir(kernel.pxr.UsdUtils.StageCache.Get())
kernel.pxr.UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
def todo():
    return kernel.omni.usd.get_context().new_stage()

kernel.submit(todo).result()
dir(kernel._omni.physx)
dir(kernel._omni.physx.get_physx_interface())



# def todo():
#     physx = kernel.omni.physx.get_physx_interface()
#     physx.update_simulation(0, 0)
#     return

# kernel.submit(todo).result()
physx = kernel.omni.physx.acquire_physx_interface()
physx
physx.is_running()
# physx.start_simulation()
# physx.is_running()
# physx.update_simulation(0, 0)

s = kernel.omni.physx.get_physx_simulation_interface()
# s.get_full_contact_report()
# s.simulate(0, 0)
s.simulate(1/60, 1/60)
s.fetch_results()
dir(kernel.omni.physx)
kernel.omni.physx.get_physx_simulation_interface?
s = kernel.omni.physx.get_physx_simulation_interface()
s.get_attached_stage()

list(kernel.omni.usd.get_context().get_stage().Traverse())
```

.conda/lib/python3.11/site-packages/isaacsim/apps/isaacsim.exp.base.kit
.conda/lib/python3.11/site-packages/isaacsim/exts/isaacsim.core.cloner/isaacsim/core/cloner/impl/cloner.py




```
kernel.omni.physx.get_physx_simulation_interface?

stage = kernel.omni.usd.get_context().get_stage()
for p in (stage.Traverse()):
    p

dir(kernel.pxr.Usd.SchemaRegistry)
p.GetTypeName()
p.GetAppliedSchemas()
p.GetProperties()
import omni.usd

ctx = omni.usd.get_context()
dir(ctx)

kernel.omni.physics.tensors.create_simulation_view("warp", stage_id=-1)
```


```
articulation = Articulation("/Franka", scene=scene)

articulation.dof_types
# articulation._physics_tensor_get_articulation_view().get_dof_drive_model_properties()
articulation.driver.dof_drive_types
articulation.driver.dof_target_positions = articulation.dof_positions * 2
pos_error = articulation.dof_positions - articulation.driver.dof_target_positions
pos_error

articulation.driver.dof_target_velocities
articulation.dof_positions

# %timeit -n 10 articulation._physics_tensor_get_articulation_view().check()
# %timeit -n 100 articulation._physics_tensor_get_articulation_view().get_dof_velocity_targets()
articulation.root_path
articulation.link_paths
```


`Pose` api


FIXME clone tensors before returning!!!!


```
# event = AsyncEvent()
# event.stream()
# await event.future()
```

```
from robotodo.engines.isaac import engine

engine

scene = engine.get_default_scene()
scene

```

```python

# robot.controller.drive
# robot.controller.reach
# robot.controller.grasp


class BaseDriveController:
    def drive(self):
        ...


class BaseMotionController:

    def reach(self, local_pose):
        ...

    def grasp(self, local_poses):
        ...


```


```python
import trimesh
# TODO
from isaacsim.replicator.grasping.sampler_utils import sample_antipodal


sample_antipodal?
```


```
articulation.driver.enable()
articulation.driver.disable()
```


.conda/lib/python3.11/site-packages/omni/kernel/py/omni/ext/_impl/fast_importer.py

import importlib
importlib.invalidate_caches()


```
from isaacsim.replicator.grasping.ui import grasping_ui_utils

# TODO
grasping_ui_utils.clear_debug_draw()
grasping_ui_utils.draw_grasp_samples_as_lines?

```


```
import sapien

pose_a = sapien.Pose(p=[-2, 3, 4], q=[.231, .321, .231, .23])
pose_b = sapien.Pose(p=[1, 2, 3], q=[.23, .21, .12, .32])

pose_a.inv().to_transformation_matrix()
pose_a = Pose(p=[-2, 3, 4], q=[.23, .231, .321, .231])
pose_b = Pose(p=[1, 2, 3], q=[.32, .23, .21, .12])

pose_a.inv().to_matrix()
pose_a.inv() * pose_a
Pose(p=[-2, 3, 4], q=[.23, .231, .321, .231]).inv() #* Pose(p=[1, 2, 3], q=[.32, .23, .21, .12])
```



```python

class FrankaPanda:
    @property
    def pending_tasks(self) -> deque[asyncio.Future]:
        ...

    def drive(self, dof_positions):
        ...

    def reach(self, pose):
        ...

    # TODO
    def grasp(self, candidate_poses):
        ...

    # TODO
    def compute_finger_aperture(self):
        ...    



```


collision sphere gen:
`isaacsim.robot_setup.xrdf_editor.extension`
https://docs.isaacsim.omniverse.nvidia.com/4.5.0/py/source/extensions/isaacsim.robot_motion.lula/docs/index.html#lula.create_collision_sphere_generator



```

import omni
_, config = omni.kit.commands.execute("URDFCreateImportConfig")
dir(config)

```


```
from isaacsim.core.utils.stage import add_reference_to_stage, get_stage_units
from isaacsim.storage.native import get_assets_root_path

# asset_path = get_assets_root_path() + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
# robot1 = add_reference_to_stage(usd_path=asset_path, prim_path="/World/Franka_1")
mug = add_reference_to_stage(usd_path="https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0/Isaac/Props/Mugs/SM_Mug_A2.usd", prim_path="/World/Mug")
mug
```

```


entity = Entity("/Path/To/SomeCreatedRigidBody", scene=scene)
entity.physics
entity.visuals


# returns a view of the body
entity = RigidBody("/Path/To/SomeCreatedRigidBody", scene=scene)
# creates the body and returns a view
entity = RigidBody(
    "/Path/To/SomeTODORigidBody", 
    scene=scene,
)

entity.geometry
entity.pose
entity.contacts

```


```
[op.GetOpType() for op in x.GetOrderedXformOps()]


prim.GetProperties()
import pxr

x = pxr.UsdGeom.Xformable(prim)
%timeit -n 100 x.ClearXformOpOrder()
x.AddTransformOp
# o = x.GetTransformOp("xformOp:transform")

x.GetXformOpOrderAttr()
x.GetOrderedXformOps()
scene._kernel.omni.physx.scripts.physicsUtils.set_or_add_translate_op
%%timeit -n 10
# %%prun

import numpy
from omni.physx.scripts import physicsUtils
from pxr import Gf



with pxr.Sdf.ChangeBlock():
    for _ in range(1024):
        xformable = pxr.UsdGeom.Xformable(prim)

        physicsUtils.set_or_add_translate_op(
            xformable,
            # Gf.Vec3f(.0, .0, 1.0),
            numpy.array([.0, .0, 1.0]),
        )
        physicsUtils.set_or_add_orient_op(
            xformable,
            Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)),  # identity quat
        )

        # physicsUtils.setup_transform_as_scale_orient_translate(xformable)
        # physicsUtils.set_or_add_scale_orient_translate(
        #     xformable,
        #     Gf.Vec3f(.5, 1.0, 1.0),             # scale
        #     Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)),  # orient (identity quat)
        #     Gf.Vec3f(.0, .0, 1.0)              # translate
        # )
        # physicsUtils.set_or_add_scale_orient_translate(
        #     xformable,
        #     Gf.Vec3f(1., 1.0, 1.0),             # scale
        #     Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)),  # orient (identity quat)
        #     Gf.Vec3f(.0, 1.0, 2.0)              # translate
        # )
import numpy
Gf.Vec3f
a = numpy.array([[1, 2, 3], [1, 2, 3]])
%timeit -n 1000 pxr.Vt.Vec3fArray.FromNumpy(a)
%timeit -n 1000 pxr.Gf.Vec3f(a.tolist()[0])
list(pxr.Vt.Vec3fArray.FromNumpy(a))

q = numpy.array([1, 2, 3, 4])

pxr.Vt.QuatfArrayFromBuffer(q)
%timeit -n 1000 pxr.Vt.Vec3fArrayFromBuffer(a)
%timeit -n 10 pxr.UsdGeom.Xformable(prim)
xformable.GetOrientOp()
xformable.GetOrientOp()
physicsUtils.set_or_add_scale_orient_translate?
%prun --help
physicsUtils.set_or_add_scale_orient_translate(
    pxr.UsdGeom.Xformable(prim),
    Gf.Vec3f(.1, 1.0, 1.0),             # scale
    Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)),  # orient (identity quat)
    Gf.Vec3f(.0, .0, 1.0)              # translate
)
```

```
# NOTE example: passive force compensation

async for _ in scene.on_update:
    forces = (
        art.driver.compute_dof_passive_gravity_forces()
        + art.driver.compute_dof_passive_coriolis_and_centrifugal_forces()
    )

    art.driver.dof_forces = forces
```

```
pxr.UsdPhysics.DriveAPI.Get(joint_driveable_prim, "angular")
```


```


# TODO doc
_ = """


planner = MotionPlanner(panda)
action = planner.compute_action({
    "dof_positions": panda.dof_positions,
    "target_poses": panda.root_pose.inv() * Pose(p=[1, 1, 1]),
})


for a in action.iter(dim="time"):
    panda.dof_positions = a["dof_positions"]
    # TODO
    await anext(scene.on_update)


"""

```




necesito??
```
class BasePlanner:
    observation_spec: ...
    action_spec: ...
    
    def compute_action(self, observation):
        ...

```


usd layers
```
from pxr import Usd, Sdf

# Get the root layer
# stage: Usd.Stage = Usd.Stage.CreateInMemory()
root_layer: Sdf.Layer = scene._stage.GetRootLayer()
root_layer.GetLoadedLayers()

# s = scene._usd_current_stage
# e = s.GetEditTarget()
# e.GetLayer()

# # s.SetEditTarget(scene._usd_current_stage.GetSessionLayer())
import pxr

editable_layer = pxr.Sdf.Layer.CreateAnonymous()

stage = scene._usd_current_stage
root_layer = stage.GetRootLayer()
root_layer.subLayerPaths.append(editable_layer.identifier)

editable_layer.SetPermissionToEdit(True)
stage.SetEditTarget(editable_layer)

```


```
from isaacsim.core.utils.stage import open_stage

open_stage("https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0/Isaac/Environments/Grid/default_environment.usd")
```


- timeline
```
omni.timeline.get_timeline_interface

# rewind doesnt seem to work
timeline = omni.timeline.get_timeline_interface()
timeline.play()
timeline.is_auto_updating()
timeline.get_ticks_per_second(), timeline.get_ticks_per_frame()
timeline
import omni

# Obtain the main timeline object
timeline = omni.timeline.get_timeline_interface()

timeline.get_current_tick(), timeline.get_current_time()

%timeit -n 1_000 timeline.forward_one_frame()
timeline.get_current_time()
timeline.play()
%%timeit -n 10

timeline.forward_one_frame()
timeline.commit()
%timeit -n 100

timeline.rewind_one_frame()
timeline.commit()
```

- batch api:
```
with batch_changes:
    some_entity.pose = ...
    ...

```