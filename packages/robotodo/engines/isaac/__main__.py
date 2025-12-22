# SPDX-License-Identifier: Apache-2.0

"""
Entry point - CLI.
"""


import asyncio


async def main():
    from robotodo.engines.isaac.scene import Scene

    # TODO load from source specified in cli
    scene = Scene.create()
    scene.viewer.mode = "editing"
    await scene.viewer.show()

    await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
