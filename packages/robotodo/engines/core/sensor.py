
import abc

from tensorspecs import TensorLike


class ProtoCamera(abc.ABC):
    @abc.abstractmethod
    async def read_rgba(self, resolution: TensorLike["xy:2"] | None = None) -> TensorLike["* x y rgba:4", "float"]:
        ...

    @property
    @abc.abstractmethod
    def viewer(self):
        ...