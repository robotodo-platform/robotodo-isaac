
# TODO
from ._kernel import _Kernel


# TODO
class Scene:
    @classmethod
    def create(cls):
        # TODO
        raise RuntimeError("TODO Not supported")

    # TODO 
    @classmethod
    def load(cls, resource: str):
        ...
    
    def __init__(self, _todo_kernel: _Kernel):
        self._kernel = _todo_kernel

        # TODO
        usd_context = _todo_kernel._omni.usd.get_context()
        stage = usd_context.get_stage()
        self._stage = stage

    def traverse(self):
        ...

    def copy(self, source, target):
        ...