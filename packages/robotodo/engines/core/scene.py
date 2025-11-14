import abc
import enum

from tensorspecs import TensorLike
from robotodo.utils.pose import Pose


class ProtoScene(abc.ABC):
    """
    TODO doc
    """

    @property
    @abc.abstractmethod
    def autostepping(self):
        r"""
        TODO doc
        """
        ...

    @abc.abstractmethod
    async def step(self, timestep: float | None = None):
        r"""
        TODO doc
        """
        ...

    @property
    @abc.abstractmethod
    def gravity(self) -> TensorLike["xyz:3"]:
        """
        Gravity vector in :math:`\mathrm{m / s^2}`.
        TODO doc clarity: xyz direction
        """
        ...


# TODO
class ProtoSceneBuilder(abc.ABC):
    @abc.abstractmethod
    def build(self) -> ProtoScene:
        ...

    @abc.abstractmethod
    def destroy(self, scene: ProtoScene):
        ...