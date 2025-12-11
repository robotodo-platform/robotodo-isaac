from robotodo.engines.isaac.kernel import default_kernel
default_kernel.configure(
    _omni_extra_argv=["--/app/enableStdoutOutput=true"],
)
from robotodo.engines.isaac.scene import Scene


class TestScene:
    def test_gravity(self):
        scene = Scene.create()
        # TODO validate tensor
        scene.gravity