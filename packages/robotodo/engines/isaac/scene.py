
import functools
import contextlib
import asyncio

import numpy
from robotodo.engines.core.error import InvalidReferenceError
from robotodo.engines.core.path import PathExpression, PathExpressionLike
from robotodo.engines.core.scene import ProtoScene
# TODO
from robotodo.utils.event import BaseSubscriptionPartialAsyncEventStream
from robotodo.engines.isaac._kernel import Kernel, get_default_kernel
from robotodo.engines.isaac._utils.usd import (
    USDStageRef,
    is_usd_stage_ref,
    usd_get_stage_id, 
    usd_create_stage, 
    usd_load_stage,
)
from robotodo.engines.isaac._utils.ui import (
    omni_enable_editing_experience,
)


# TODO
# _USD_PHYSICSSCENE_PATH_DEFAULT = "/PhysicsScene"
# def usd_ensure_physics_scene(stage: "pxr.Usd.Stage", kernel: Kernel):
#     pxr = kernel.pxr
#     omni = kernel.omni
#     kernel.enable_extension("omni.usd")

#     has_physics_scene = False
#     for prim in stage.Traverse():
#         if prim.IsA(pxr.UsdPhysics.Scene):
#             has_physics_scene = True
#             break

#     if not has_physics_scene:
#         path = omni.usd.get_stage_next_free_path(
#             stage, 
#             path=self._USD_PHYSICSSCENE_PATH_DEFAULT,
#             prepend_default_prim=False,
#         )
#         pxr.UsdPhysics.Scene.Define(stage, path)


# TODO !!!! per stage???
class PhysicsStepAsyncEventStream(BaseSubscriptionPartialAsyncEventStream[float]):
    def __init__(self, scene: "Scene"):
        self._scene = scene

    @functools.cached_property
    def _isaac_physx_interface(self):
        self._scene._kernel.enable_extension("omni.physx")
        return self._scene._kernel.omni.physx.get_physx_interface()

    # TODO
    @contextlib.contextmanager
    def subscribe(self, callable):
        # TODO cache?
        def physx_callback(timestep: float):
            result = callable(timestep)
            if asyncio.iscoroutine(result):
                asyncio.create_task(result)
        sub = self._isaac_physx_interface.subscribe_physics_step_events(physx_callback)
        yield
        sub.unsubscribe()


# TODO multiple scenes are not supported!!!
# TODO ? default scene: scene = Scene(); new scene: scene = engine.add(Scene())
class Scene(ProtoScene):

    @classmethod
    def create(cls, _kernel: Kernel | None = None):
        stage = usd_create_stage(kernel=_kernel)
        return Scene(lambda: stage, _kernel=_kernel)
    
    # TODO
    @classmethod
    def load_usd(cls, source: str, _kernel: Kernel | None = None):
        stage = usd_load_stage(source, kernel=_kernel)
        return Scene(lambda: stage, _kernel=_kernel)
    
    # TODO
    @classmethod
    def load(cls, source: str, _kernel: Kernel | None = None):
        # TODO !!!! check extension
        return cls.load_usd(source=source, _kernel=_kernel)

    class _USDKernelDefaultStageRef(USDStageRef):
        def __init__(self, kernel: Kernel):
            self._kernel = kernel

        def __call__(self):
            omni = self._kernel.omni
            self._kernel.enable_extension("omni.usd")

            # TODO !!!
            stage = omni.usd.get_context().get_stage()
            # TODO rm
            if stage is None:
                # TODO !!!!!  Stage opening or closing already in progress so async???
                omni.usd.get_context().new_stage()
                stage = omni.usd.get_context().get_stage()
            # TODO check None
            assert stage is not None
            return stage

    _usd_stage_ref: USDStageRef
    _kernel: Kernel

    def __init__(
        self,
        ref: USDStageRef | None = None,
        _kernel: Kernel | None = None
    ):
        # TODO !!!!!
        if _kernel is None:
            _kernel = get_default_kernel()
        match ref:
            case ref if is_usd_stage_ref(ref):
                self._usd_stage_ref = ref
                self._kernel = _kernel
            case None:
                self._usd_stage_ref = Scene._USDKernelDefaultStageRef(
                    kernel=_kernel,
                )
                self._kernel = _kernel
                ...
            case _:
                raise InvalidReferenceError(ref)

    # TODO !!!!!
    # def __init__(self, _usd_stage_ref: ... = None, _kernel: Kernel | None = None):
    #     # TODO !!!!!
    #     if _kernel is None:
    #         _kernel = get_default_kernel()
    #     self._kernel = _kernel
    #     self._usd_stage_ref = _usd_stage_ref

    def save(self, source: str | None = None):
        if source is None:
            stage = self._usd_stage_ref()
            stage.Save()
            stage.SaveSessionLayers()
        else:
            # TODO
            raise NotImplementedError("TODO")

    # TODO rm?
    @property
    def _usd_stage(self):
        return self._usd_stage_ref()
    
    @property
    def _isaac_physx(self):
        self._kernel.enable_extension("omni.physx")

        physx = self._kernel.omni.physx.get_physx_interface()
        # TODO attach stage

        return physx
    
    # TODO FIXME perf
    # TODO cache and ensure current stage !!!!
    @property
    def _isaac_physx_simulation(self):
        self._kernel.enable_extension("omni.physx")

        physx_sim = self._kernel.omni.physx.get_physx_simulation_interface()
        # TODO util: usd_get_stage_id
        # TODO FIXME: this causes timeline to stop working: 
        # see: https://forums.developer.nvidia.com/t/omniverse-physx-time-stepping-blocking/284664/10
        if False:
            current_stage_id = usd_get_stage_id(self._usd_stage, kernel=self._kernel).ToLongInt()
            if physx_sim.get_attached_stage() != current_stage_id:
                if not physx_sim.attach_stage(current_stage_id):
                    raise RuntimeError(
                        f"Failed to attach USD stage {self._usd_stage} "
                        f"to the PhysX simulation interface {physx_sim}. Is it valid?"
                    )
        return physx_sim
    
        # TODO check stage attachment
        # stage_id = physx_sim.get_attached_stage()
        # if stage_id == 0:
        #     pass # get stage from omni
        # stage_cache = scene._kernel.pxr.UsdUtils.StageCache.Get()
        # stage_cache.Find(stage_cache.Id.FromLongInt(9223003))

    @functools.cached_property
    def _isaac_physics_tensor_view_cache(self):
        omni = self._kernel.omni

        try:
            self._kernel.enable_extension("omni.physics.tensors")

            stage = self._usd_stage
            # TODO NOTE this also starts the simulation
            self._isaac_physx_simulation.flush_changes()
            # TODO
            res = omni.physics.tensors.create_simulation_view(
                "torch",
                # TODO FIXME this breaks urdf importer bc of multiple stages in cache
                # stage_id=usd_get_stage_id(stage, kernel=self._kernel).ToLongInt(),
            )
        except Exception as error:
            # TODO
            raise RuntimeError(
                "Failed to create physics view."
                # f"Does a prim of type `PhysicsScene` exist on {self._usd_stage}?"
                # TODO rm
                # f"Make sure the physics simulation is running: call {self.timeline_play}"
            ) from error
        if not res.is_valid:
            raise RuntimeError(f"Created physics view is invalid: {res}")
        return res

    @property
    def _isaac_physics_tensor_view(self):
        if not self._isaac_physics_tensor_view_cache.is_valid:
            del self._isaac_physics_tensor_view_cache
        return self._isaac_physics_tensor_view_cache
    
    # TODO see https://github.com/isaac-sim/IsaacSim/issues/223
    def _isaac_physics_tensor_ensure_sync(self):
        if not self._isaac_physx.is_running():
            self._isaac_physx.start_simulation()
            # TODO call at least once: physx.update_simulation with non-zero dt

        self._isaac_physx.update_transformations(
            updateToFastCache=True,
            updateToUsd=True,
            updateVelocitiesToUsd=True,
            outputVelocitiesLocalSpace=True,
        )

    def has(self, path: PathExpressionLike):    
        return len(self.resolve(path)) > 0

    def traverse(self, path: PathExpressionLike | None = None):
        # TODO
        if path is not None:
            raise NotImplementedError(f"TODO {path}")

        root_prim = (
            self._usd_stage.GetPseudoRoot()
            # if path is None else
            # # TODO FIXME
            # self._usd_stage.GetPrimAtPath(self.resolve(path))
        )

        return (
            prim.GetPath().pathString
            for prim in self._kernel.pxr.Usd.PrimRange(root_prim)
        )

    def resolve(self, path: PathExpressionLike):
        return PathExpression(path).resolve(self.traverse())

    # TODO stage !!!!
    def copy(self, path: PathExpressionLike, target_path: PathExpressionLike):
        self._kernel.enable_extension("isaacsim.core.cloner")
        self._kernel.import_module("isaacsim.core.cloner")
        isaacsim = self._kernel.import_module("isaacsim")

        path = PathExpression(path)
        target_path = PathExpression(target_path)

        # TODO check if dir
        cloner = isaacsim.core.cloner.Cloner(stage=self._usd_stage)
        source_prim_path = self.resolve(path)
        if len(source_prim_path) != 1:
            raise NotImplementedError("TODO")
        [source_prim_path] = source_prim_path

        cloner.clone(
            source_prim_path=source_prim_path,
            prim_paths=PathExpression(target_path).expand(),
        )

        # raise NotImplementedError

    # TODO
    def rename(self, path: PathExpressionLike, target_path: PathExpressionLike):
        omni = self._kernel.omni
        # TODO
        self._kernel.enable_extension("omni.usd")
        self._kernel.enable_extension("omni.kit.commands")

        paths = self.resolve(path)
        target_paths = self.resolve(target_path)

        if len(paths) != len(target_paths):
            raise NotImplementedError("TODO")

        # TODO
        move_prims_command = omni.usd.commands.MovePrimsCommand(
            paths_to_move={
                path_single: target_path_single
                for path_single, target_path_single in zip(paths, target_paths, strict=True)
            },
            stage_or_context=self._usd_stage,
        )
        move_prims_command.do()

        # for path_single, target_path_single in zip(path, target_path, strict=True):
        #     # TODO
        #     is_success, _ = omni.kit.commands.execute(
        #         "MovePrim",
        #         path_from=path_single,
        #         path_to=target_path_single,
        #         stage_or_context=self._usd_stage,
        #     )
        #     if not is_success:
        #         raise RuntimeError("TODO")

    @functools.cached_property
    def _isaac_timeline(self):
        # TODO ensure current stage !!!!
        omni = self._kernel.omni
        self._kernel.enable_extension("omni.timeline")
        self._kernel.import_module("omni.timeline")
        return omni.timeline.acquire_timeline_interface()

    @property
    def autostepping(self):
        timeline = self._isaac_timeline
        return timeline.is_playing() and timeline.is_auto_updating()
    
    @autostepping.setter
    def autostepping(self, value):
        timeline = self._isaac_timeline
        if value:
            timeline.set_auto_update(True)
            timeline.play()
        else:
            timeline.set_auto_update(False)
            timeline.pause()

    # TODO
    # TODO ref https://docs.omniverse.nvidia.com/kit/docs/omni_physics/108.0/dev_guide/simulation_control/simulation_control.html
    # TODO cannot reproduce: ~~this messes up the timeline for some reason !!!!!!!!~~
    # TODO api: support time??
    async def step(self, timestep: float | None = None):
        if timestep is None:
            timestep = 1 / 60

        if timestep != 0:
            # TODO
            physx_sim = self._isaac_physx_simulation
            physx_sim.simulate(elapsed_time=float(timestep), current_time=0.)
            # NOTE this ensures the enqueued `.simulate` gets processed
            # and yes, this is non-async!
            physx_sim.fetch_results()

        return timestep
    
    # TODO 
    async def reset(self):
        raise NotImplementedError
    
    # TODO
    @functools.cached_property
    def on_step(self):
        return PhysicsStepAsyncEventStream(scene=self)

    @functools.cached_property
    def viewer(self):
        return SceneViewer(scene=self)

    # TODO FIXME unit
    @property
    def gravity(self):
        # TODO tensor backend
        return numpy.asarray(self._isaac_physics_tensor_view.get_gravity())
    
    @gravity.setter
    def gravity(self, value: ...):
        # TODO tensor backend
        self._isaac_physics_tensor_view.set_gravity(numpy.broadcast_to(value, shape=3))


from typing import TypedDict, Unpack

import numpy
import einops
from robotodo.utils.pose import Pose


class SceneViewer:
    def __init__(self, scene: Scene):
        self._scene = scene

    # TODO
    def show(self):
        pass

    # TODO
    @property
    def mode(self):
        settings = self._scene._kernel.get_settings()
        match settings.get("/app/window/hideUi"):
            case True:
                return "viewing"
            case False:
                return "editing"
            
    @mode.setter
    def mode(self, value: ...):
        settings = self._scene._kernel.get_settings()
        match value:
            case "viewing":
                settings.set("/app/window/hideUi", True)
            case "editing":
                # TODO
                settings.set("/app/window/hideUi", False)
                omni_enable_editing_experience(kernel=self._scene._kernel)
                # TODO start editing extension
            case _:
                raise ValueError(f"TODO")

    # TODO
    @property
    def selected_entity(self):
        from robotodo.engines.isaac.entity import Entity

        omni = self._scene._kernel.omni
        # TODO ensure stage
        selection = omni.usd.get_context().get_selection()
        return Entity(selection.get_selected_prim_paths(), scene=self._scene)
    
    # TODO
    @selected_entity.setter
    def selected_entity(self, value):
        from robotodo.engines.isaac.entity import Entity
        # TODO
        entity: Entity = value

        # TODO ensure stage
        omni = self._scene._kernel.omni
        selection = omni.usd.get_context().get_selection()
        selection.set_selected_prim_paths(entity.path)

    # TODO
    @property
    def _isaac_debug_draw_interface(self):
        # TODO
        kernel = self._scene._kernel
        kernel.enable_extension("isaacsim.util.debug_draw")
        kernel.import_module("isaacsim.util.debug_draw")
        isaacsim = kernel.import_module("isaacsim")

        return (
            isaacsim.util.debug_draw._debug_draw
            .acquire_debug_draw_interface()
        )
    
        # TODO lifecycle
        # isaacsim = self._scene._kernel.isaacsim
        # isaacsim.util.debug_draw._debug_draw.release_debug_draw_interface

    def clear_drawings(self):
        iface = self._isaac_debug_draw_interface
        iface.clear_points()
        iface.clear_lines()

    class DrawPoseOptions(TypedDict, total=False):
        scale: float
        line_thickness: float
        line_opacity: float

    def draw_pose(
        self, 
        pose: Pose, 
        options: DrawPoseOptions = DrawPoseOptions(),
        **options_kwds: Unpack[DrawPoseOptions],
    ):
        """
        TODO doc


        """

        options = self.DrawPoseOptions(options, **options_kwds)

        scale: float = options.get("scale", 1.)
        line_thickness: float = options.get("line_thickness", 2)
        line_opacity: float = options.get("line_opacity", .5)

        # TODO x y z
        for mask in (
            numpy.asarray([1., 0., 0.]),
            numpy.asarray([0., 1., 0.]),
            numpy.asarray([0., 0., 1.]),
        ):
            start_points = pose.p
            # TODO
            end_points = (pose * Pose(p=mask * [scale, scale, scale])).p

            start_points, _ = einops.pack([start_points], "* xyz")
            end_points, _ = einops.pack([end_points], "* xyz")

            colors = einops.repeat(
                numpy.asarray([*mask, line_opacity]),
                "rgba -> b rgba",
                **einops.parse_shape(start_points, "b _"),
            )
            thicknesses = einops.repeat(
                numpy.asarray(line_thickness),
                "-> b",
                **einops.parse_shape(start_points, "b _"),
            )

            self._isaac_debug_draw_interface.draw_lines(
                numpy.asarray(start_points).tolist(), 
                numpy.asarray(end_points).tolist(), 
                numpy.asarray(colors).tolist(), 
                numpy.asarray(thicknesses).tolist(),
            )
