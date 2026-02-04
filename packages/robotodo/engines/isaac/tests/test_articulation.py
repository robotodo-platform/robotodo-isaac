import pytest

from robotodo.engines.isaac.scene import Scene
from robotodo.engines.isaac.articulation import Articulation


class TestArticulation:

    @pytest.mark.asyncio(loop_scope="session")
    async def test_load(self):
        scene = Scene.create()

        articulation = Articulation.load(
            "/World/Panda",
            "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd",
            scene=scene,
        )

        # TODO
        articulation.path


    @pytest.mark.asyncio(loop_scope="session")
    async def test_properties(self):

        # TODO use fixtures

        scene = Scene.create()

        articulation = Articulation.load(
            "/World/Panda",
            "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd",
            scene=scene,
        )
        
        articulation.dof_names
        articulation.dof_kinds
        articulation.dof_position_limits
        articulation.dof_positions
        articulation.dof_velocities

    # TODO
    # @pytest.mark.asyncio(loop_scope="session")
    # async def test_driver(self):
    #     scene = Scene.create()

    #     articulation = Articulation.load(
    #         "/World/Panda",
    #         "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd",
    #         scene=scene,
    #     )

    #     async with scene.play():
    #         await articulation.driver.execute_action({
    #             "dof_positions": articulation.dof_positions
    #         })