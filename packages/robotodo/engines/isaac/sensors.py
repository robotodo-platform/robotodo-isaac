"""
TODO

"""


import functools
from typing import Any, Literal, NamedTuple

# TODO
import warp
# import torch
from robotodo.utils.pose import Pose
from robotodo.engines.core.path import PathExpressionLike, PathExpression

from .scene import Scene
from ._utils import USDPrimHelper


# TODO FIXME illegal mem: sanity check!!!!!
@warp.kernel
def _reshape_tiled_image_kernel(
    tiled_image: Any,
    batched_image: Any,
    image_height: int,
    image_width: int,
    num_channels: int,
    num_output_channels: int,
    num_tiles_x: int,
    offset: int,
):
    """
    TODO doc

    Reshape a tiled image (height*width*num_channels*num_cameras,) to a batch of images (num_cameras, height, width, num_channels).

    Args:
        tiled_image_buffer: The input image buffer. Shape is ((height*width*num_channels*num_cameras,).
        batched_image: The output image. Shape is (num_cameras, height, width, num_channels).
        image_width: The width of the image.
        image_height: The height of the image.
        num_channels: The number of channels in the image.
        num_tiles_x: The number of tiles in x direction.
        offset: The offset in the image buffer. This is used when multiple image types are concatenated in the buffer.
    """

    # get the thread id
    camera_id, height_id, width_id = warp.tid()
    # resolve the tile indices
    tile_x_id = camera_id % num_tiles_x
    tile_y_id = camera_id // num_tiles_x
    # compute the start index of the pixel in the tiled image buffer
    pixel_start = (
        offset
        + num_channels * num_tiles_x * image_width * (image_height * tile_y_id + height_id)
        + num_channels * tile_x_id * image_width
        + num_channels * width_id
    )
    # copy the pixel values into the batched image
    for i in range(num_output_channels):
        batched_image[camera_id, height_id, width_id, i] = batched_image.dtype(tiled_image[pixel_start + i])


# wp.overload(
#     reshape_tiled_image,
#     {"tiled_image_buffer": wp.array(dtype=wp.uint8), "batched_image": wp.array(dtype=wp.uint8, ndim=4)},
# )
# wp.overload(
#     reshape_tiled_image,
#     {"tiled_image_buffer": wp.array(dtype=wp.float32), "batched_image": wp.array(dtype=wp.float32, ndim=4)},
# )


# TODO tests
def _reshape_tiled_image(
    tiled_image: warp.array,
    shape: tuple[int, int, int, int],
):
    """
    TODO doc

    """

    height, width, channels = tiled_image.shape
    n, tile_height, tile_width, tile_channels = shape
    num_tiles_x = width // tile_width

    out = warp.empty(shape, dtype=tiled_image.dtype)

    warp.launch(
        kernel=_reshape_tiled_image_kernel,
        dim=(n, height, width),
        inputs=[
            tiled_image.flatten(),
            out,
            tile_height,
            tile_width,
            channels,
            tile_channels,
            num_tiles_x,
            0,  # offset is always 0 since we sliced the data
        ],
        device=tiled_image.device,
    )

    return out



# TODO
# TODO ref isaacsim.sensors.camera
class Camera:
    """
    TODO doc
    
    """

    class Resolution(NamedTuple):
        height: int
        width: int

    def __init__(
        self, 
        path: PathExpressionLike,
        scene: Scene,
    ):
        """
        TODO doc        
        """

        self._path = PathExpression(path)
        self._scene = scene

    # TODO
    @functools.cached_property
    def _usd_prim_helper(self):
        # TODO
        return USDPrimHelper(path=self._path, scene=self._scene)
    
    @property
    def _usd_prims(self):
        return self._usd_prim_helper._usd_prims
    
    @property
    def pose(self):
        return self._usd_prim_helper.pose
    
    @pose.setter
    def pose(self, value: Pose):
        self._usd_prim_helper.pose = value

    @property
    def pose_in_parent(self):
        return self._usd_prim_helper.pose_in_parent
    
    @pose_in_parent.setter
    def pose_in_parent(self, value: Pose):
        self._usd_prim_helper.pose_in_parent = value
    

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
        self._scene._kernel.enable_extension("omni.replicator.core")
        # TODO customizable!!!!!!!
        return (
            self._scene._kernel.omni.replicator.core.create
            .render_product_tiled(
                self._scene.resolve(self._path), 
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
        
        omni = self._scene._kernel.omni
        self._scene._kernel.enable_extension("omni.replicator.core")

        annotator = (
            omni.replicator.core.AnnotatorRegistry
            .get_annotator(name, device=device, do_array_copy=copy)
            .attach(self._isaac_get_render_product(resolution=resolution))
        )

        async def result_fn():
            # NOTE ensure the data is available IMMEDIATELY after this function call
            # TODO ref omni.replicator.core.scripts.annotators.Annotator.get_data
            await omni.replicator.core.orchestrator.step_async(
                # rt_subframes=1, 
                delta_time=0, 
                wait_for_render=True,
            )
            return annotator

        # TODO
        return self._scene._kernel._loop.create_task(result_fn())
    
    async def _isaac_get_frame(
        self, 
        name: _IsaacRenderName,
        resolution: Resolution,
    ):

        frame_tiled = (
            (await self._isaac_get_render_annotator(
                name, 
                resolution=resolution,
            ))
            .get_data()
        )
        # TODO better way to handle this??
        if frame_tiled.size == 0:
            # NOTE maybe next time
            return None
        
        # TODO NOTE ensure dim
        if frame_tiled.ndim == 2:
            frame_tiled = frame_tiled.reshape((*frame_tiled.shape, 1))

        res = _reshape_tiled_image(
            frame_tiled, 
            shape=(
                len(self._isaac_get_render_targets(resolution=resolution)), 
                resolution.height, 
                resolution.width, 
                # TODO optional !!!!!!!!!!!!!!!!!!!!!!!!
                frame_tiled.shape[-1],
            ),
        )
        # TODO perf: this has a constant us-level overhead!!!!
        return warp.to_torch(res)
    
    _RESOLUTION_DEFAULT = Resolution(256, 256)

    # TODO
    async def read_rgba(self, resolution: Resolution | tuple[int, int] = _RESOLUTION_DEFAULT):
        resolution = self.Resolution._make(resolution)
        return await self._isaac_get_frame(name="rgb", resolution=resolution)

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

        asyncio.get_running_loop().run_until_complete(todo())
