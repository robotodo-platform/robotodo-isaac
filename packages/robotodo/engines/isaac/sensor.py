# SPDX-License-Identifier: Apache-2.0

"""
Sensor.
"""


import functools
from typing import Any, Literal, NamedTuple

# TODO
import warp
import torch
import einops
from robotodo.utils.pose import Pose
from robotodo.engines.core.path import (
    PathExpressionLike, 
    PathExpression, 
    is_path_expression_like,
)
from robotodo.engines.core.sensor import ProtoCamera
from robotodo.engines.isaac.scene import Scene
from robotodo.engines.isaac.entity import Entity
from robotodo.engines.isaac._utils.usd import (
    USDPrimRef,
    is_usd_prim_ref,
    USDXformView,
    USDPrimPathExpressionRef,
)


# TODO
# TODO ref isaacsim.sensors.camera
class Camera(ProtoCamera):
    class Resolution(NamedTuple):
        height: int
        width: int

    _usd_prims_ref: USDPrimRef
    _scene: Scene

    @classmethod
    def create(cls, ref, scene: Scene):
        pxr = scene._kernel._pxr
        prims = [
            pxr.UsdGeom.Camera.Define(scene._usd_stage, path).GetPrim()
            for path in PathExpression(ref).expand()
        ]
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
            
    # TODO
    @functools.cached_property
    def _usd_xform_view(self):
        return USDXformView(
            self._usd_prims_ref, 
            kernel=self._scene._kernel,
        )

    @property
    def path(self):
        return [
            prim.GetPath().pathString
            for prim in self._usd_prims_ref()
        ]

    @property
    def scene(self):
        return self._scene
    
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
    

    # # TODO NOTE usd uses diff coords for cams: impl in __usd_prim_helper instead??
    # # TODO https://docs.isaacsim.omniverse.nvidia.com/5.0.0/reference_material/reference_conventions.html#world-axes
    # # TODO https://docs.isaacsim.omniverse.nvidia.com/5.0.0/reference_material/reference_conventions.html#default-camera-axes
    # @property
    # def pose(self):
    #     pose_w_cam_u = self._usd_prim_helper.pose

    #     # TODO
    #     # from World camera convention to USD camera convention
    #     # TODO https://github.com/isaac-sim/IsaacSim/blob/21bbdbad07ba31687f2ff71f414e9d21a08e16b8/source/extensions/isaacsim.sensors.camera/isaacsim/sensors/camera/camera_view.py#L35
    #     import numpy
    #     U_W_TRANSFORM = numpy.asarray([
    #         [0, -1, 0, 0], 
    #         [0, 0, 1, 0], 
    #         [-1, 0, 0, 0], 
    #         [0, 0, 0, 1],
    #     ])

    #     return pose_w_cam_u * Pose.from_matrix(U_W_TRANSFORM)

    # # TODO bug upstream in Pose??
    # @pose.setter
    # def pose(self, value: Pose):
    #     pose_w_cam_w = value

    #     # TODO
    #     # from USD camera convention to World camera convention
    #     # TODO https://github.com/isaac-sim/IsaacSim/blob/21bbdbad07ba31687f2ff71f414e9d21a08e16b8/source/extensions/isaacsim.sensors.camera/isaacsim/sensors/camera/camera_view.py#L32
    #     import numpy
    #     W_U_TRANSFORM = numpy.asarray([
    #         [0, 0, -1, 0], 
    #         [-1, 0, 0, 0], 
    #         [0, 1, 0, 0], 
    #         [0, 0, 0, 1],
    #     ])
        
    #     self._usd_prim_helper.pose = pose_w_cam_w * Pose.from_matrix(W_U_TRANSFORM)


    # TODO mv to _Kernel and handle caching
    # TODO caching
    @functools.cache
    def _isaac_get_render_product(self, resolution: Resolution):
        self._scene._kernel._omni_enable_extension("omni.replicator.core")
        # TODO

        # # TODO Run a preview to ensure the replicator graph is initialized??
        # omni.replicator.core.orchestrator.preview()
        # TODO customizable!!!!!!!
        return (
            self._scene._kernel._omni.replicator.core.create
            .render_product_tiled(
                [
                    prim.GetPath().pathString 
                    for prim in self._usd_prims_ref()
                ], 
                tile_resolution=(resolution.width, resolution.height),
            )
        )

    # TODO caching
    # @functools.cache
    def _isaac_get_render_targets(self, resolution: Resolution):
        return (
            self._scene._usd_stage.GetPrimAtPath(
                self._isaac_get_render_product(resolution=resolution).path
            )
            .GetRelationship("camera")
            .GetTargets()
        )
    
    _IsaacRenderName = Literal["rgb", "distance_to_image_plane"] | str

    # TODO
    @functools.cache
    def _isaac_get_render_annotator(
        self, 
        name: _IsaacRenderName, 
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

        return (
            omni.replicator.core.AnnotatorRegistry
            .get_annotator(name, device=device, do_array_copy=copy)
            .attach(self._isaac_get_render_product(resolution=resolution))
        )

        # annotator = (
        #     omni.replicator.core.AnnotatorRegistry
        #     .get_annotator(name, device=device, do_array_copy=copy)
        #     .attach(self._isaac_get_render_product(resolution=resolution))
        # )

        # async def result_fn():
        #     # TODO rm??
        #     # NOTE ensure the data is available IMMEDIATELY after this function call
        #     # TODO ref omni.replicator.core.scripts.annotators.Annotator.get_data
        #     # await omni.replicator.core.orchestrator.step_async(
        #     #     # rt_subframes=1, 
        #     #     delta_time=0, 
        #     #     pause_timeline=False,
        #     #     wait_for_render=True,
        #     # )
        #     return annotator

        # # TODO
        # return self._scene._kernel._loop.create_task(result_fn())
    
    async def _isaac_get_frame(
        self, 
        name: _IsaacRenderName,
        resolution: Resolution,
    ):
        
        # TODO ref why do_array_copy? https://docs.omniverse.nvidia.com/py/replicator/1.10.10/source/extensions/omni.replicator.core/docs/API.html#omni.replicator.core.scripts.annotators.Annotator.get_data
        render_targets = self._isaac_get_render_targets(resolution=resolution)
        render_annotator = self._isaac_get_render_annotator(
            name, 
            resolution=resolution,
            copy=True,
        )
        frame_tiled = ...

        while True:
            # TODO
            omni = self._scene._kernel._omni
            self._scene._kernel._omni_enable_extension("omni.replicator.core")

            # TODO ensure kernel running?
            self._scene._kernel.run_forever()
            await omni.replicator.core.orchestrator.step_async(
                # rt_subframes=1, 
                delta_time=0, 
                pause_timeline=False,
                wait_for_render=True,
            )

            frame_tiled = render_annotator.get_data(
                # TODO
                do_array_copy=True,
            )
            # TODO better way to handle this??
            if frame_tiled.size != 0:
                break

        # TODO rm
        # frame_tiled = (
        #     (await self._isaac_get_render_annotator(
        #         name, 
        #         resolution=resolution,
        #     ))
        #     .get_data()
        # )
        # # TODO better way to handle this??
        # if frame_tiled.size == 0:
        #     omni = self._scene._kernel.omni
        #     self._scene._kernel.enable_extension("omni.replicator.core")
        #     # TODO
        #     await omni.replicator.core.orchestrator.step_async(
        #         # rt_subframes=1, 
        #         delta_time=0, 
        #         pause_timeline=False,
        #         wait_for_render=True,
        #     )
        #     # NOTE maybe next time
        #     return await self._isaac_get_frame(name=name, resolution=resolution)
        
        # TODO NOTE ensure dim
        if frame_tiled.ndim == 2:
            frame_tiled = frame_tiled.reshape((*frame_tiled.shape, 1))

        res = einops.rearrange(
            warp.to_torch(frame_tiled), 
            "(num_tiles_height height) (num_tiles_width width) channel -> (num_tiles_height num_tiles_width) height width channel", 
            height=resolution.height, width=resolution.width,
        )
        res = res[:len(render_targets)]

        return res

        # res = untile_image(
        #     tiled_image=frame_tiled,
        #     shape=(
        #         len(self._isaac_get_render_targets(resolution=resolution)),
        #         resolution.height, 
        #         resolution.width,
        #     ),
        # )
        # # res = _reshape_tiled_image(
        # #     frame_tiled, 
        # #     shape=(
        # #         len(self._isaac_get_render_targets(resolution=resolution)), 
        # #         resolution.height, 
        # #         resolution.width, 
        # #         # TODO optional !!!!!!!!!!!!!!!!!!!!!!!!
        # #         frame_tiled.shape[-1],
        # #     ),
        # # )
        # # TODO perf: this has a constant us-level overhead!!!!
        # return warp.to_torch(res)
    
    _RESOLUTION_DEFAULT = Resolution(256, 256)

    # TODO
    async def read_rgba(self, resolution: Resolution | tuple[int, int] = _RESOLUTION_DEFAULT):
        resolution = self.Resolution._make(resolution)
        return torch.as_tensor(
            await self._isaac_get_frame(name="rgb", resolution=resolution)
        ) / 255

    # TODO FIXME upstream: tiled output channel optional 
    async def read_depth(self, resolution: Resolution | tuple[int, int] = _RESOLUTION_DEFAULT):
        resolution = self.Resolution._make(resolution)
        return await self._isaac_get_frame(name="distance_to_image_plane", resolution=resolution)

    @property
    def viewer(self):
        return CameraViewer(self)
    

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
                plt.imshow(rgba.to(device="cpu"))
                plt.show()

        asyncio.ensure_future(todo())
