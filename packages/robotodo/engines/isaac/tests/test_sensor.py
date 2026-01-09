import asyncio

import pytest
import numpy

from robotodo.engines.isaac.scene import Scene
from robotodo.engines.isaac.sensor import Camera
from robotodo.utils.pose import Pose


class TestCamera:
    @pytest.mark.asyncio(loop_scope="session")
    async def test_read_rgba(self):
        # TODO
        scene = Scene.create()

        camera = Camera.create("/CameraSingle", scene=scene)
        assert (await camera.read_rgba()) is not None
        # TODO auto validate using tensorspecs 
        ...
        camera = Camera.create("/Camera{0..16}", scene=scene)
        assert (await camera.read_rgba()) is not None

    @pytest.mark.asyncio(loop_scope="session")
    async def test_read_rgba_content(self):
        scene = Scene.load(
            "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0/Isaac/Environments/Grid/default_environment.usd"
        )

        camera = Camera.create("/CameraSingle", scene=scene)

        camera.pose = Pose.from_lookat([1, 1, 1], [0, 0, 0])
        rgba_image = await camera.read_rgba()
        assert not numpy.allclose(rgba_image[..., [0, 1, 2]].numpy(force=True), 0.)

        camera.pose = Pose()
        rgba_image = await camera.read_rgba()
        assert numpy.allclose(rgba_image[..., [0, 1, 2]].numpy(force=True), 0.)

        camera = Camera.create("/Camera{1..2}", scene=scene)

        camera.pose = Pose.from_lookat([1, 1, 1], [0, 0, 0])
        rgba_image = await camera.read_rgba()
        assert not numpy.allclose(rgba_image[..., [0, 1, 2]].numpy(force=True), 0.)

        camera.pose = Pose()
        rgba_image = await camera.read_rgba()
        assert numpy.allclose(rgba_image[..., [0, 1, 2]].numpy(force=True), 0.)

    @pytest.mark.asyncio(loop_scope="session")
    async def test_read_depth(self):
        scene = Scene.create()

        camera = Camera.create("/CameraSingle", scene=scene)
        assert (await camera.read_depth()) is not None
        # TODO auto validate using tensorspecs 
        ...
        camera = Camera.create("/Camera{0..16}", scene=scene)
        assert (await camera.read_depth()) is not None

    @pytest.mark.asyncio(loop_scope="session")
    async def test_properties(self):
        scene = Scene.create()

        camera = Camera.create("/Camera{0..4}", scene=scene)

        # TODO auto test
        camera.imager.size
        camera.optics.f_stop
        camera.optics.focal_length
        camera.optics.focus_distance

        ...

    # TODO
    # @pytest.mark.benchmark
    # @pytest.mark.asyncio(loop_scope="session")
    # async def test_benchmark_read_rgba(self, benchmark):
    #     scene = Scene.load(
    #         "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0/Isaac/Environments/Grid/default_environment.usd"
    #     )

    #     camera = Camera.create("/Camera{0..16}", scene=scene)
    #     camera.pose = Pose.from_lookat([1, 1, 1], [0, 0, 0])

    #     @benchmark
    #     def _():
    #         loop = asyncio.get_running_loop()
    #         asyncio.run_coroutine_threadsafe(camera.read_rgba(), loop=loop).result()

    #     ...