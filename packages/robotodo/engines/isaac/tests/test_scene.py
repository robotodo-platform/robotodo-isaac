
import importlib.resources

import pytest
import numpy

# from robotodo.engines.isaac.kernel import default_kernel
# default_kernel.configure(
#     _omni_extra_argv=["--/app/enableStdoutOutput=true"],
# )
from robotodo.engines.isaac.tests import assets
from robotodo.engines.isaac.scene import Scene


class TestScene:
    @pytest.mark.asyncio(loop_scope="session")
    async def test_load(self):
        with (importlib.resources.files(assets) / "empty-scene.usda").open() as io:
            scene = Scene.load(io)
            assert numpy.allclose(
                scene.gravity,
                [0, 0, -9.81],
            )
            
    @pytest.mark.asyncio(loop_scope="session")
    async def test_viewer(self):
        scene = Scene.create()
        # await scene.viewer.show()
        async with scene.viewer.show():
            ...

    @pytest.mark.asyncio(loop_scope="session")
    async def test_play(self):
        scene = Scene.create()
        # await scene.play()
        # TODO test
        async with scene.play():
            ...

    @pytest.mark.asyncio(loop_scope="session")
    async def test_step(self):
        scene = Scene.create()
        assert (await scene.step(time=0, timestep=.5)) == .5
        ...

    @pytest.mark.asyncio(loop_scope="session")
    async def test_properties(self):
        scene = Scene.create()

        # TODO validate tensor
        scene.gravity

        # TODO
        gravity_expected = 1.
        scene.gravity = gravity_expected
        
        assert numpy.array_equiv(scene.gravity, gravity_expected)