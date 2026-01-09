# SPDX-License-Identifier: Apache-2.0

"""
Scene.
"""


import io
import warnings
import functools
import contextlib
import asyncio
from typing import IO

import numpy
import torch
from robotodo.engines.core.error import InvalidReferenceError
from robotodo.engines.core.path import PathExpression, PathExpressionLike
from robotodo.engines.core.scene import ProtoScene
# TODO
from robotodo.utils.event import BaseSubscriptionPartialAsyncEventStream
from robotodo.engines.isaac.kernel import Kernel, get_running_kernel
from robotodo.engines.isaac._utils.usd import (
    USDStageRef,
    is_usd_stage_ref,
    usd_get_stage_id, 
    usd_create_stage, 
    usd_load_stage,
    usd_save_stage,
    usd_physics_ensure_physics_scene,
)
from robotodo.engines.isaac._utils.ui import (
    omni_enable_editing_experience,
    omni_enable_viewing_experience,
    omni_disable_ui,
)


# TODO !!!! per stage???
class PhysicsStepAsyncEventStream(BaseSubscriptionPartialAsyncEventStream[float]):
    def __init__(self, scene: "Scene"):
        self._scene = scene

    @functools.cached_property
    def _isaac_physx_interface(self):
        self._scene._kernel._omni_enable_extension("omni.physx")
        return self._scene._kernel._omni.physx.get_physx_interface()

    # TODO
    @contextlib.contextmanager
    def subscribe(self, callable):
        # TODO cache?
        def physx_callback(timestep: float):
            result = callable(timestep)
            if asyncio.iscoroutine(result):
                asyncio.ensure_future(result)
        sub = self._isaac_physx_interface.subscribe_physics_step_events(physx_callback)
        yield
        sub.unsubscribe()


# TODO FIXME multiple scenes are not supported!!!
class Scene(ProtoScene):

    @classmethod
    def create(cls, kernel: Kernel | None = None):
        stage = usd_create_stage(kernel=kernel)
        return Scene(lambda: stage, kernel=kernel)
    
    # TODO
    @classmethod
    def load_usd(cls, source: str | IO, kernel: Kernel | None = None):
        stage = usd_load_stage(source, kernel=kernel)
        return Scene(lambda: stage, kernel=kernel)
    
    # TODO
    @classmethod
    def load(cls, source: str | IO, kernel: Kernel | None = None):
        # TODO !!!! check extension
        return cls.load_usd(source=source, kernel=kernel)

    class _USDKernelDefaultStageRef(USDStageRef):
        def __init__(self, kernel: Kernel):
            self._kernel = kernel

        def __call__(self):
            omni = self._kernel._omni
            self._kernel._omni_enable_extension("omni.usd")

            # TODO !!!
            stage = omni.usd.get_context().get_stage()
            if stage is None:
                raise ValueError(
                    f"Default scene (USD stage) does not exist. "
                    f"Create or load one using {Scene.create} or {Scene.load}"
                )
            # # TODO rm
            # if stage is None:
            #     # TODO !!!!!  Stage opening or closing already in progress so async???
            #     omni.usd.get_context().new_stage()
            #     stage = omni.usd.get_context().get_stage()
            # TODO check None
            assert stage is not None
            return stage

    _usd_stage_ref: USDStageRef
    _kernel: Kernel

    def __init__(
        self,
        ref: USDStageRef | None = None,
        kernel: Kernel | None = None
    ):
        # TODO !!!!!
        if kernel is None:
            kernel = get_running_kernel()
        match ref:
            case ref if is_usd_stage_ref(ref):
                self._usd_stage_ref = ref
                self._kernel = kernel
            case None:
                self._usd_stage_ref = Scene._USDKernelDefaultStageRef(
                    kernel=kernel,
                )
                self._kernel = kernel
            case _:
                raise InvalidReferenceError(ref)

    @functools.cached_property
    def viewer(self):
        return SceneViewer(scene=self)            

    @property
    def kernel(self):
        return self._kernel

    # TODO buffer
    def save(self, destination: str | IO | None = None):
        stage = self._usd_stage_ref()
        usd_save_stage(stage, resource=destination, kernel=self.kernel)

    # TODO rm?
    @property
    def _usd_stage(self):
        return self._usd_stage_ref()
    
    def _omni_ensure_current_stage(self):
        omni = self._kernel._omni
        self._kernel._omni_enable_extension("omni.usd")
        stage = self._usd_stage_ref()

        usd_context = omni.usd.get_context()
        if usd_context.get_stage() != stage:
            # TODO rm due to BUG
            # self._kernel._omni_run_coroutine(
            #     usd_context.attach_stage_async(stage),
            # ).result()

            import concurrent.futures
            future = concurrent.futures.Future()
            def on_finish_fn(is_success: bool, message: str):
                if not is_success:
                    future.set_exception(RuntimeError(message))
                else:
                    future.set_result(is_success)
            # TODO does this require `app.update`?? no??
            usd_context.attach_stage_with_callback(
                usd_get_stage_id(stage, kernel=self._kernel), 
                on_finish_fn=on_finish_fn,
            )
            # TODO
            assert future.result()

        return usd_context
    
    # TODO !!!!
    @property
    def _omni_usd_context(self):
        return self._omni_ensure_current_stage()
    
    @property
    def _omni_physx(self):
        self._kernel._omni_enable_extensions(["omni.physx"])

        self._omni_ensure_current_stage()
        physx = self._kernel._omni.physx.get_physx_interface()
        # TODO attach stage

        return physx
    
    # TODO FIXME perf
    # TODO cache and ensure current stage !!!!
    @property
    def _omni_physx_simulation(self):
        self._kernel._omni_enable_extensions(["omni.physx"])

        self._omni_ensure_current_stage()
        physx_sim = self._kernel._omni.physx.get_physx_simulation_interface()
        # TODO FIXME: this causes timeline to stop working: 
        # see: https://forums.developer.nvidia.com/t/omniverse-physx-time-stepping-blocking/284664/10
        # TODO use usd context instead
        # if False:
        #     current_stage_id = usd_get_stage_id(self._usd_stage, kernel=self._kernel)
        #     if physx_sim.get_attached_stage() != current_stage_id:
        #         if not physx_sim.attach_stage(current_stage_id):
        #             raise RuntimeError(
        #                 f"Failed to attach USD stage {self._usd_stage} "
        #                 f"to the PhysX simulation interface {physx_sim}. Is it valid?"
        #             )
        return physx_sim
    
        # TODO check stage attachment
        # stage_id = physx_sim.get_attached_stage()
        # if stage_id == 0:
        #     pass # get stage from omni
        # stage_cache = scene._kernel.pxr.UsdUtils.StageCache.Get()
        # stage_cache.Find(stage_cache.Id.FromLongInt(9223003))

    @functools.cached_property
    def _omni_physics_tensor_view_cache(self):
        omni = self._kernel._omni

        try:
            self._kernel._omni_enable_extensions([
                "omni.physics.tensors",
                "omni.physx.tensors",
            ])

            stage = self._usd_stage
            # TODO
            usd_physics_ensure_physics_scene(stage, kernel=self._kernel)
            # TODO NOTE this also starts the simulation
            self._omni_physx_simulation.flush_changes()
            self._omni_ensure_current_stage()
            res = omni.physics.tensors.create_simulation_view(
                "torch",
                # TODO
                # stage_id=usd_get_stage_id(stage, kernel=self._kernel),
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
    def _omni_physics_tensor_view(self):
        if not self._omni_physics_tensor_view_cache.is_valid:
            del self._omni_physics_tensor_view_cache
        return self._omni_physics_tensor_view_cache
    
    # TODO see https://github.com/isaac-sim/IsaacSim/issues/223
    def _omni_physics_tensor_ensure_sync(self):
        if not self._omni_physx.is_running():
            self._omni_physx.start_simulation()
            # TODO call at least once: physx.update_simulation with non-zero dt

        self._omni_physx.update_transformations(
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
            for prim in self._kernel._pxr.Usd.PrimRange(root_prim)
        )

    def resolve(self, path: PathExpressionLike):
        return PathExpression(path).resolve(self.traverse())

    # TODO stage !!!!
    def copy(self, path: PathExpressionLike, target_path: PathExpressionLike):
        self._kernel._omni_enable_extension("isaacsim.core.cloner")
        self._kernel._omni_import_module("isaacsim.core.cloner")
        isaacsim = self._kernel._omni_import_module("isaacsim")

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
        omni = self._kernel._omni
        # TODO
        self._kernel._omni_enable_extension("omni.usd")
        self._kernel._omni_enable_extension("omni.kit.commands")

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
    
    class _PlayController:
        _enabled: bool | None = None
        _future: asyncio.Future | None = None

        def __init__(
            self,
            scene: "Scene",
        ):
            self._scene = scene

        async def _enable(self):
            timeline = self._scene._omni_usd_context.get_timeline()
            timeline.set_auto_update(True)
            timeline.play()
            timeline.commit()
            if self._future is None or self._future.done():
                self._future = self._scene._kernel._omni_ensure_future(
                    asyncio.Event().wait()
                )

        async def _disable(self):    
            timeline = self._scene._omni_usd_context.get_timeline()
            timeline.set_auto_update(False)
            timeline.pause()
            timeline.commit()
            if self._future is not None:
                self._future.cancel()

        def __await__(self):
            if self._enabled is None:
                return
            yield from (
                self._enable().__await__()
                if self._enabled else
                self._disable().__await__()
            )

        async def __aenter__(self):
            if self._enabled is None:
                return
            await (
                self._enable()
                if self._enabled else
                self._disable()
            )

        async def __aexit__(self, *_):
            if self._enabled is None:
                return
            await (
                self._enable()
                if not self._enabled else
                self._disable()
            )

    # TODO
    @functools.cached_property
    def _play_controller(self):
        return self._PlayController(self)

    def play(self, enabled: bool = True):
        controller = self._play_controller
        controller._enabled = enabled
        return controller

    # TODO
    # TODO ref https://docs.omniverse.nvidia.com/kit/docs/omni_physics/108.0/dev_guide/simulation_control/simulation_control.html
    async def step(
        self, 
        timestep: float | None = None,
        time: float | None = None,
    ):
        if timestep is None:
            timestep = 1 / 60

        if timestep != 0:
            # TODO
            physx_sim = self._omni_physx_simulation
            physx_sim.simulate(
                current_time=float(time) if time is not None else 0.,
                elapsed_time=float(timestep), 
            )
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

    @property
    def gravity(self):
        pxr = self._kernel._pxr
        # TODO tensor backend
        return (
            torch.asarray(self._omni_physics_tensor_view.get_gravity()) 
            * pxr.UsdGeom.GetStageMetersPerUnit(self._usd_stage)
        )
    
    @gravity.setter
    def gravity(self, value: ...):
        pxr = self._kernel._pxr
        # TODO tensor backend
        self._omni_physics_tensor_view.set_gravity(
            torch.asarray(
                torch.broadcast_to(torch.asarray(value), size=(3, ))
                / pxr.UsdGeom.GetStageMetersPerUnit(self._usd_stage)                
            )
            .numpy(force=True)
        )


from typing import TypedDict, Unpack, Literal

import numpy
import einops
from robotodo.utils.pose import Pose


class SceneViewer:
    def __init__(self, scene: Scene):
        self._scene = scene

    @property
    def _omni_app_window(self):
        kernel = self._scene._kernel
        kernel._omni_enable_extension("omni.appwindow")
        omni = kernel._omni

        app_window_factory = omni.appwindow.acquire_app_window_factory_interface()
        app_window = app_window_factory.get_app_window()
        # TODO
        assert app_window is not None
        return app_window

    def _omni_set_app_window_visible(self, visible: bool = True):
        carb = self._scene._kernel._carb
        self._scene._kernel._omni_enable_extension("carb.windowing.plugins")

        try:
            carb_windowing_iface = carb.windowing.acquire_windowing_interface()
        except Exception as error:
            warnings.warn(f"TODO: {error}")
            carb_windowing_iface = None
        
        if carb_windowing_iface is not None:
            carb_app_window = self._omni_app_window.get_window()
            if carb_app_window is not None:
                if visible:
                    carb_windowing_iface.show_window(carb_app_window)
                else:
                    carb_windowing_iface.hide_window(carb_app_window)

    class _ShowController:
        def __init__(
            self, 
            viewer: "SceneViewer",
        ):
            self._viewer = viewer

        _enabled: bool | None = None
        _mode: Literal["viewing", "editing"] | None = None
        _future: asyncio.Future | None = None            

        async def _enable(self):
            # TODO rm better ways to synchronize
            # self._viewer.mode = self._viewer.mode

            scene = self._viewer._scene

            settings = scene._kernel._carb.settings.get_settings()
            match self._mode:
                # TODO
                case "viewing":
                    settings.set("/app/window/hideUi", True)
                    # await scene._kernel._app.next_update_async()
                    # TODO this causes issues with the isaacsim registry in content browser. why????
                    omni_enable_viewing_experience(kernel=scene._kernel)
                case "editing":
                    # TODO
                    settings.set("/app/window/hideUi", False)
                    # await scene._kernel._app.next_update_async()
                    omni_enable_editing_experience(kernel=scene._kernel)
                case None:
                    pass
                case _:
                    raise ValueError(f"TODO")

            # TODO
            scene._omni_ensure_current_stage()
            self._viewer._omni_set_app_window_visible(True)

            if self._future is None or self._future.done():
                self._future = scene._kernel._omni_ensure_future(
                    asyncio.Event().wait()
                )

        async def _disable(self):
            # TODO
            self._viewer._omni_set_app_window_visible(False)
            # TODO destroy windows to save resource
            omni_disable_ui(kernel=self._viewer._scene.kernel)

            if self._future is not None:
                self._future.cancel()

        def __await__(self):
            if self._enabled is None:
                return
            yield from (
                self._enable().__await__()
                if self._enabled else
                self._disable().__await__()
            )

        async def __aenter__(self):
            if self._enabled is None:
                return
            await (
                self._enable()
                if self._enabled else
                self._disable()
            )

        async def __aexit__(self, *_):
            if self._enabled is None:
                return
            await (
                self._enable()
                if not self._enabled else
                self._disable()
            )

    # TODO
    @functools.cached_property
    def _show_controller(self):
        return self._ShowController(self)

    # TODO cache???
    def show(
        self, 
        enabled: bool = True, 
        mode: Literal["viewing", "editing"] = "viewing",
    ):
        controller = self._show_controller
        controller._enabled = enabled
        controller._mode = mode
        return controller

    # TODO
    @property
    def selected_entity(self):
        from robotodo.engines.isaac.entity import Entity

        omni = self._scene._kernel._omni
        # TODO ensure stage
        self._scene._omni_ensure_current_stage()
        selection = omni.usd.get_context().get_selection()
        return Entity(selection.get_selected_prim_paths(), scene=self._scene)
    
    # TODO
    @selected_entity.setter
    def selected_entity(self, value):
        from robotodo.engines.isaac.entity import Entity
        # TODO
        entity: Entity = value

        # TODO ensure stage
        omni = self._scene._kernel._omni
        self._scene._omni_ensure_current_stage()
        selection = omni.usd.get_context().get_selection()
        selection.set_selected_prim_paths(entity.path)

    # TODO deprecate: in tiled rendering this shows a single gizmo in the frame which is incorrect
    @property
    def _isaac_debug_draw_interface(self):
        # TODO
        kernel = self._scene._kernel
        kernel._omni_enable_extension("isaacsim.util.debug_draw")
        kernel._omni_import_module("isaacsim.util.debug_draw")
        isaacsim = kernel._omni_import_module("isaacsim")

        self._scene._omni_ensure_current_stage()
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
