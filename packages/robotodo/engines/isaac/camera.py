"""
TODO

"""


import functools
from typing import Any

# TODO
import warp
# import torch
from robotodo.engines.core.entity_selector import PathExpressionLike, PathExpression

from .scene import Scene


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


from typing import NamedTuple

# TODO
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
        _todo_resolution: Resolution | tuple[int, int],
        scene: Scene,
    ):
        """
        TODO doc

        :param _todo_resolution: (height, width)
        
        """

        # TODO
        self._path = PathExpression(path)

        # TODO !!!!!!
        self._todo_resolution = self.Resolution._make(_todo_resolution)
        # self._todo_resolution = _todo_resolution

        # TODO
        self._scene = scene

    # TODO
    @property
    def _todo_kernel(self):
        return self._scene._kernel

    # TODO
    @property
    def _todo_stage(self):
        return self._scene._usd_stage

    # TODO mv to _Kernel and handle caching
    # TODO caching
    @functools.cache
    def _get_render_product(self, resolution: Resolution):
        # TODO customizable!!!!!!!
        return (
            self._todo_kernel.omni.replicator.core.create
            .render_product_tiled(
                # TODO FIXME this MUST be a list of concrete paths!!!!
                self._scene.resolve(self._path), 
                tile_resolution=(resolution.width, resolution.height),
            )
        )

    # TODO caching
    @functools.cache
    def _get_render_targets(self, resolution: Resolution):
        return (
            self._todo_stage.GetPrimAtPath(
                self._get_render_product(resolution=resolution).path
            )
            .GetRelationship("camera")
            .GetTargets()
        )

    @functools.cache
    def _get_render_annotator(
        self, 
        name: str, 
        resolution: Resolution,
        device: str = "cuda", 
        copy: bool = False,
    ):
        """
        TODO doc

        omni.replicator.core.AnnotatorRegistry.get_registered_annotators()

        """      

        return (
            self._todo_kernel.omni.replicator.core.AnnotatorRegistry
            .get_annotator(name, device=device, do_array_copy=copy)
            .attach(self._get_render_product(resolution=resolution))
        )

    @property
    def image(self):
        # TODO !!!!!

        rgba_tiled = self._get_render_annotator(
            "rgb", 
            resolution=self._todo_resolution,
        ).get_data()
        # TODO better way to handle this??
        if rgba_tiled.size == 0:
            return None

        res = _reshape_tiled_image(
            rgba_tiled, 
            shape=(
                len(self._get_render_targets(resolution=self._todo_resolution)), 
                self._todo_resolution.height, 
                self._todo_resolution.width, 
                rgba_tiled.shape[-1],
            ),
        )
        # TODO perf: this has a constant us-level overhead!!!!
        return warp.to_torch(res)
    
    # TODO pointcloud