# SPDX-License-Identifier: Apache-2.0

"""
Sensor.
"""


import warnings
import sys
import functools
from typing import Any, Literal, NamedTuple

# TODO
import warp
import torch
import numpy
import einops
from robotodo.utils.pose import Pose
from robotodo.engines.core.path import (
    PathExpressionLike,
    PathExpression,
    is_path_expression_like,
)
from robotodo.engines.core.sensor import (
    ProtoCamera,
    ProtoCameraImager,
    ProtoCameraOptics,
)
from robotodo.engines.isaac.scene import Scene
from robotodo.engines.isaac.entity import Entity
from robotodo.engines.isaac._utils.usd import (
    USDPrimRef,
    is_usd_prim_ref,
    USDXformView,
    USDPrimPathExpressionRef,
    usd_prims_get_meters_per_unit,
)


# TODO
# TODO ref isaacsim.sensors.camera
class Camera(ProtoCamera):
    class Resolution(NamedTuple):
        height: int
        width: int

    _usd_prims_ref: USDPrimRef
    _scene: Scene

    _usd_clipping_range_default = None  # (1e-2, 1e+5)

    @classmethod
    def create(cls, ref, scene: Scene):
        pxr = scene._kernel._pxr
        prims = [
            pxr.UsdGeom.Camera.Define(scene._usd_stage, path).GetPrim()
            for path in PathExpression(ref).expand()
        ]
        if cls._usd_clipping_range_default is not None:
            for prim in prims:
                pxr.UsdGeom.Camera(prim).GetClippingRangeAttr().Set(
                    cls._usd_clipping_range_default
                )
        # TODO set clipping range to something more reasonable??
        return cls(lambda: prims, scene=scene)

    @classmethod
    def load_usd(cls, ref: PathExpressionLike, source: str, scene: Scene):
        return cls(Entity.load_usd(ref, source=source, scene=scene))

    @classmethod
    def load(cls, ref: PathExpressionLike, source: str, scene: Scene):
        # TODO
        return cls.load_usd(ref, source=source, scene=scene)

    # TODO
    def __init__(
        self,
        ref: "Camera | Entity | USDPrimRef | PathExpressionLike",
        scene: Scene | None = None,
    ):
        match ref:
            case Camera() as camera:
                assert scene is None
                self._usd_prims_ref = camera._usd_prims_ref
                self._scene = camera._scene
            case Entity() as entity:
                assert scene is None
                self._usd_prims_ref = entity._usd_prim_ref
                self._scene = entity._scene
            case ref if is_usd_prim_ref(ref):
                # TODO
                assert scene is not None
                self._usd_prims_ref = ref
                self._scene = scene
            case expr if is_path_expression_like(ref):
                # TODO
                assert scene is not None
                self._usd_prims_ref = USDPrimPathExpressionRef(
                    expr,
                    stage_ref=lambda: scene._usd_stage,
                )
                self._scene = scene
            case _:
                raise ValueError("TODO")

    @property
    def viewer(self):
        return CameraViewer(self)

    @property
    def path(self):
        return [prim.GetPath().pathString for prim in self._usd_prims_ref()]

    @property
    def scene(self):
        return self._scene

    # TODO
    @functools.cached_property
    def _usd_xform_view(self):
        return USDXformView(
            self._usd_prims_ref,
            kernel=self._scene._kernel,
        )
    
    @property
    def pose(self):
        return self._usd_xform_view.pose

    @pose.setter
    def pose(self, value: Pose):
        self._usd_xform_view.pose = value

    @property
    def pose_in_parent(self):
        return self._usd_xform_view.pose_in_parent

    @pose_in_parent.setter
    def pose_in_parent(self, value: Pose):
        self._usd_xform_view.pose_in_parent = value

    _omni_renderer_default: Literal["rtx"] = "rtx"

    # TODO
    @functools.cache
    def _omni_ensure_rendering(self, renderer: Literal["rtx"] | None = None):
        if renderer is None:
            renderer = self._omni_renderer_default

        omni = self._scene._kernel._omni
        self._scene._kernel._omni_enable_extensions(
            [
                "omni.usd",
            ]
        )

        usd_context = self._scene._omni_usd_context

        hydra_engine_name = ...
        match renderer:
            case "rtx":
                hydra_engine_name = "rtx"
                self._scene._kernel._omni_enable_extension("omni.hydra.rtx")
            case _:
                raise NotImplementedError
        if hydra_engine_name not in usd_context.get_attached_hydra_engine_names():
            omni.usd.create_hydra_engine(hydra_engine_name, usd_context)

    @functools.cache
    def _omni_get_render_product(self, resolution: Resolution):
        self._omni_ensure_rendering()

        self._scene._kernel._omni_enable_extension("omni.replicator.core")
        # TODO
        # # TODO Run a preview to ensure the replicator graph is initialized??
        # omni.replicator.core.orchestrator.preview()
        # TODO customizable!!!!!!!
        self._scene._omni_ensure_current_stage()
        render_product = (
            self._scene._kernel._omni.replicator.core.create.render_product_tiled(
                [prim.GetPath().pathString for prim in self._usd_prims_ref()],
                tile_resolution=(resolution.width, resolution.height),
            )
        )

        # TODO NOTE ux: hide unnecessary stuff such as grid lines
        carb = self._scene._kernel._carb
        settings = carb.settings.get_settings()
        settings.set("/exts/omni.kit.hydra_texture/gizmos/enabled", False)
        settings.set(
            f"/exts/omni.kit.hydra_texture/{render_product.hydra_texture.get_name()}/gizmos/enabled",
            False,
        )

        match render_product.hydra_texture.hydra_engine:
            case "rtx":
                match settings.get("/rtx/rendermode"):
                    case "RaytracedLighting":
                        warnings.warn(
                            f"RTX real-time mode has a bug that prevents tile > 1 rendering "
                            f"from working properly. Coercing to path-tracing mode. "
                            f"For more information, see https://github.com/isaac-sim/IsaacSim/issues/367"
                        )
                        settings.set("/rtx/rendermode", "PathTracing")
                    case _:
                        ...
            case _:
                pass

        return render_product

    # TODO caching
    # @functools.cache
    def _omni_get_render_targets(self, resolution: Resolution):
        return (
            self._scene._usd_stage.GetPrimAtPath(
                self._omni_get_render_product(resolution=resolution).path
            )
            .GetRelationship("camera")
            .GetTargets()
        )

    _OmniRenderName = Literal["rgb", "distance_to_image_plane"] | str

    # TODO
    # TODO perf: auto detach when not in use?
    @functools.cache
    def _omni_get_render_annotator(
        self,
        name: _OmniRenderName,
        resolution: Resolution,
        device: str = "cuda",
        copy: bool = False,
    ):
        """
        TODO doc

        omni.replicator.core.AnnotatorRegistry.get_registered_annotators()

        """

        omni = self._scene._kernel._omni
        self._scene._kernel._omni_enable_extension("omni.replicator.core")

        annotator = omni.replicator.core.AnnotatorRegistry.get_annotator(
            name, device=device, do_array_copy=copy
        )
        annotator.attach(self._omni_get_render_product(resolution=resolution))

        return annotator

    _omni_use_synchronization: bool = False

    async def _omni_get_frame(
        self,
        name: _OmniRenderName,
        resolution: Resolution,
    ):

        # TODO ref why do_array_copy? https://docs.omniverse.nvidia.com/py/replicator/1.10.10/source/extensions/omni.replicator.core/docs/API.html#omni.replicator.core.scripts.annotators.Annotator.get_data
        render_annotator = self._omni_get_render_annotator(
            name,
            resolution=resolution,
            copy=True,
        )
        frame_tiled = ...

        while True:
            # TODO
            if self._omni_use_synchronization:
                omni = self._scene._kernel._omni
                self._scene._kernel._omni_enable_extension("omni.replicator.core")
                settings = self._scene._kernel._carb.settings.get_settings()
                # TODO NOTE enable frame gate
                settings.set("/exts/omni.replicator.core/Orchestrator/enabled", True)

                self._scene._omni_ensure_current_stage()
                # TODO
                # self._scene._kernel._omni_ensure_future
                # TODO
                # await omni.replicator.core.orchestrator.step_async(
                #     delta_time=0,
                #     pause_timeline=False,
                #     wait_for_render=True,
                # )
                await self._scene._kernel._omni_ensure_future(
                    omni.replicator.core.orchestrator.step_async(
                        delta_time=0,
                        pause_timeline=False,
                        wait_for_render=True,
                    )
                )
            else:
                # TODO
                settings = self._scene._kernel._carb.settings.get_settings()
                # TODO restore
                settings.set("/exts/omni.replicator.core/Orchestrator/enabled", False)
                ...

            frame_tiled = render_annotator.get_data(
                # TODO
                do_array_copy=True,
            )
            # TODO check data type: this can be numpy if device="cpu" !!!!
            # TODO better way to handle this??
            if frame_tiled.size != 0:
                break

            await self._scene._kernel._omni_ensure_future(
                self._scene._kernel._app.next_update_async()
            )

        # TODO NOTE ensure dim
        if frame_tiled.ndim == 2:
            frame_tiled = frame_tiled.reshape((*frame_tiled.shape, 1))

        res = einops.rearrange(
            warp.to_torch(frame_tiled),
            "(num_tiles_height height) (num_tiles_width width) channel -> (num_tiles_height num_tiles_width) height width channel",
            height=resolution.height,
            width=resolution.width,
        )
        render_targets = self._omni_get_render_targets(resolution=resolution)
        res = res[: len(render_targets)]

        return res

    _resolution_default = Resolution(256, 256)

    # TODO
    async def read_rgba(
        self, resolution: Resolution | tuple[int, int] = _resolution_default
    ):
        resolution = self.Resolution._make(resolution)
        return (
            torch.asarray(await self._omni_get_frame(name="rgb", resolution=resolution))
            / 255
        )

    # TODO FIXME upstream: tiled output channel optional
    async def read_depth(
        self, resolution: Resolution | tuple[int, int] = _resolution_default
    ):
        resolution = self.Resolution._make(resolution)
        return torch.asarray(
            await self._omni_get_frame(
                name="distance_to_image_plane", resolution=resolution
            )
        )

    @property
    def projection_matrix(self):
        # TODO
        raise NotImplementedError

    @functools.cached_property
    def imager(self):
        return CameraImager(self)

    @functools.cached_property
    def optics(self):
        return CameraOptics(self)


class CameraImager(ProtoCameraImager):
    def __init__(self, camera: Camera):
        self._camera = camera

    @property
    def size(self):
        pxr = self._camera._scene._kernel._pxr
        prims = self._camera._usd_prims_ref()
        value = numpy.asarray(
            [
                [
                    pxr.UsdGeom.Camera(prim).GetVerticalApertureAttr().Get(),
                    pxr.UsdGeom.Camera(prim).GetHorizontalApertureAttr().Get(),
                ]
                for prim in prims
            ],
            dtype=numpy.float_,
        )
        value /= 10.0
        value *= numpy.reshape(
            usd_prims_get_meters_per_unit(
                prims,
                kernel=self._camera._scene._kernel,
            ),
            (len(prims), 1),
        )
        return value

    @size.setter
    def size(self, value):
        pxr = self._camera._scene._kernel._pxr
        prims = self._camera._usd_prims_ref()
        value = numpy.asarray(value)
        value /= numpy.reshape(
            usd_prims_get_meters_per_unit(
                prims,
                kernel=self._camera._scene._kernel,
            ),
            (len(prims), 1),
        )
        value *= 10.0
        value = numpy.broadcast_to(value, (len(prims), 2))
        for prim, [v_vaperture, v_haperture] in zip(prims, value):
            api = pxr.UsdGeom.Camera(prim)
            api.GetVerticalApertureAttr().Set(float(v_vaperture))
            api.GetHorizontalApertureAttr().Set(float(v_haperture))


class CameraOptics(ProtoCameraOptics):
    def __init__(self, camera: Camera):
        self._camera = camera

    @property
    def focal_length(self):
        pxr = self._camera._scene._kernel._pxr
        prims = self._camera._usd_prims_ref()
        value = numpy.asarray(
            [pxr.UsdGeom.Camera(prim).GetFocalLengthAttr().Get() for prim in prims],
            dtype=numpy.float_,
        )
        value /= 10.0
        value *= usd_prims_get_meters_per_unit(
            prims,
            kernel=self._camera._scene._kernel,
        )
        return value

    @focal_length.setter
    def focal_length(self, value):
        pxr = self._camera._scene._kernel._pxr
        prims = self._camera._usd_prims_ref()
        value /= usd_prims_get_meters_per_unit(
            prims,
            kernel=self._camera._scene._kernel,
        )
        value *= 10.0
        value = numpy.broadcast_to(value, len(prims))
        for prim, v in zip(prims, value):
            pxr.UsdGeom.Camera(prim).GetFocalLengthAttr().Set(float(v))

    @property
    def focus_distance(self):
        pxr = self._camera._scene._kernel._pxr
        prims = self._camera._usd_prims_ref()
        value = numpy.asarray(
            [pxr.UsdGeom.Camera(prim).GetFocusDistanceAttr().Get() for prim in prims],
            dtype=numpy.float_,
        )
        value *= usd_prims_get_meters_per_unit(
            prims,
            kernel=self._camera._scene._kernel,
        )
        return value

    @focus_distance.setter
    def focus_distance(self, value):
        pxr = self._camera._scene._kernel._pxr
        prims = self._camera._usd_prims_ref()
        value = numpy.asarray(value)
        value /= usd_prims_get_meters_per_unit(
            prims,
            kernel=self._camera._scene._kernel,
        )
        value = numpy.broadcast_to(value, len(prims))
        for prim, v in zip(prims, value):
            pxr.UsdGeom.Camera(prim).GetFocusDistanceAttr().Set(float(v))

    @property
    def f_stop(self):
        pxr = self._camera._scene._kernel._pxr
        prims = self._camera._usd_prims_ref()
        value = numpy.asarray(
            [pxr.UsdGeom.Camera(prim).GetFStopAttr().Get() for prim in prims],
            dtype=numpy.float_,
        )
        value *= usd_prims_get_meters_per_unit(
            prims,
            kernel=self._camera._scene._kernel,
        )
        return value

    @f_stop.setter
    def f_stop(self, value):
        pxr = self._camera._scene._kernel._pxr
        prims = self._camera._usd_prims_ref()
        value /= usd_prims_get_meters_per_unit(
            prims,
            kernel=self._camera._scene._kernel,
        )
        value = numpy.broadcast_to(value, len(prims))
        for prim, v in zip(prims, value):
            pxr.UsdGeom.Camera(prim).GetFStopAttr().Set(float(v))


class CameraViewer:
    def __init__(self, camera: Camera):
        self._camera = camera

    def show(self):
        # TODO
        import asyncio

        import matplotlib.pyplot as plt

        async def todo():
            rgbas = await self._camera.read_rgba()
            for rgba in rgbas:
                plt.imshow(rgba.numpy(force=True))
                plt.show()

        asyncio.ensure_future(todo())
