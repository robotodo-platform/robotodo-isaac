
import abc
import enum
import asyncio
from typing import TypedDict, Unpack

from tensorspecs import TensorLike
from robotodo.engines.core.path import PathExpressionLike
from robotodo.engines.core.entity import ProtoEntity
from robotodo.engines.core.body import ProtoBody
from robotodo.engines.core.scene import ProtoScene
from robotodo.utils.pose import Pose


class Axis(enum.IntEnum):
    UNKNOWN = -1
    X = 0
    Y = 1
    Z = 2


class JointKind(enum.IntEnum):
    UNKNOWN = -1
    FIXED = 0
    REVOLUTE = 1
    PRISMATIC = 2
    SPHERICAL = 3


# TODO
class ProtoJoint(ProtoEntity, abc.ABC):
    # @classmethod
    # @abc.abstractmethod
    # def create(cls):
    #     ...

    # TODO
    @abc.abstractmethod
    def __init__(
        self, 
        ref: "ProtoJoint | ProtoEntity | PathExpressionLike", 
        scene: ProtoScene | None = None,
    ):
        ...

    @property
    @abc.abstractmethod
    def kind(self) -> TensorLike["*", JointKind]:
        ...

    @property
    @abc.abstractmethod
    def body0(self) -> ProtoBody:
        ...
    
    @property
    @abc.abstractmethod
    def pose_in_body0(self) -> Pose:
        ...

    @property
    @abc.abstractmethod
    def body1(self) -> ProtoBody:
        ...
    
    @property
    @abc.abstractmethod
    def pose_in_body1(self) -> Pose:
        ...


class ProtoFixedJoint(ProtoJoint, abc.ABC):
    @property
    @abc.abstractmethod
    def kind(self) -> TensorLike["*", JointKind.FIXED]:
        ...


class ProtoRevoluteJoint(ProtoJoint, abc.ABC):
    @property
    @abc.abstractmethod
    def kind(self) -> TensorLike["*", JointKind.REVOLUTE]:
        ...

    @property
    @abc.abstractmethod
    def axis(self) -> TensorLike["*", Axis]:
        ...

    @property
    @abc.abstractmethod
    def position_limit(self) -> TensorLike["* minmax:2"]:
        r"""
        Joint angular position limit (min-max).
        Unit: radians.
        """
        ...


class ProtoPrismaticJoint(ProtoJoint, abc.ABC):
    @property
    @abc.abstractmethod
    def kind(self) -> TensorLike["*", JointKind.PRISMATIC]:
        ...

    @property
    @abc.abstractmethod
    def axis(self) -> TensorLike["*", Axis]:
        ...

    @property
    @abc.abstractmethod
    def position_limit(self) -> TensorLike["* minmax:2"]:
        r"""
        Joint linear position limit (min-max).
        Unit: TODO.
        """
        ...


class ProtoSphericalJoint(ProtoJoint, abc.ABC):
    @property
    @abc.abstractmethod
    def kind(self) -> TensorLike["*", JointKind.SPHERICAL]:
        ...

    # TODO ...


class DOFKind(enum.IntEnum):
    NONE = -1
    ROTATION = 0
    TRANSLATION = 1


class ProtoArticulation(ProtoEntity, abc.ABC):
    """
    TODO doc
    """

    # TODO ArticulationSpec
    # @classmethod
    # @abc.abstractmethod
    # def create(cls, ref: PathExpressionLike, scene: ProtoScene):
    #     ...

    # TODO
    # @abc.abstractmethod
    # def from_entity(self, entity: ProtoBody):
    #     ...

    # TODO
    @abc.abstractmethod
    def __init__(
        self, 
        ref: "ProtoArticulation | ProtoEntity | PathExpressionLike", 
        scene: ProtoScene | None = None,
    ):
        ...

    @property
    @abc.abstractmethod
    def joints(self) -> dict[str, ProtoJoint]:
        ...

    # TODO infer from joints??
    @property
    @abc.abstractmethod
    def links(self) -> dict[str, ProtoBody]:
        ...

    # TODO
    @property
    @abc.abstractmethod
    def pose(self) -> Pose:
        ...

    # TODO
    @property
    @abc.abstractmethod
    def pose_in_parent(self) -> Pose:
        ...

    @property
    @abc.abstractmethod
    def dof_names(self) -> TensorLike["*? dof", str]:
        r"""
        TODO doc
        no batch dim when same model
        """
        ...

    @property
    @abc.abstractmethod
    def dof_kinds(self) -> TensorLike["* dof", DOFKind]:
        ...

    @property
    @abc.abstractmethod
    def dof_positions(self) -> TensorLike["* dof", float]:
        ...

    @property
    @abc.abstractmethod
    def dof_position_limits(self) -> TensorLike["* dof minmax:2", float]:
        ...

    # TODO make optional?
    @property
    @abc.abstractmethod
    def driver(self) -> "ProtoArticulationDriver":
        r"""
        The builtin driver for this articulation.
        """
        ...

    # TODO
    @abc.abstractmethod
    def planner(
        self, 
        **config_kwds: Unpack["ProtoArticulationPlanner.Config"],
    ) -> "ProtoArticulationPlanner":
        ...


class DriveMode(enum.IntEnum):
    NONE = 0
    FORCE = 1
    ACCELERATION = 2


# TODO necesito?
ArticulationAction = ...


class ProtoArticulationDriver(abc.ABC):
    # TODO
    @property
    @abc.abstractmethod
    def enabled(self) -> TensorLike["*", bool]:
        ...

    @property
    @abc.abstractmethod
    def dof_drive_modes(self) -> TensorLike["* dof", DriveMode]:
        ...

    @property
    @abc.abstractmethod
    def dof_stiffnesses(self) -> TensorLike["* dof", float]:
        ...

    @property
    @abc.abstractmethod
    def dof_dampings(self) -> TensorLike["* dof", float]:
        ...

    @property
    @abc.abstractmethod
    def dof_forces(self) -> TensorLike["* dof", float]:
        ...

    @property
    @abc.abstractmethod
    def dof_force_limits(self) -> TensorLike["* dof", float]:
        r"""
        TODO doc positive
        """
        ...

    @property
    @abc.abstractmethod
    def dof_target_positions(self) -> TensorLike["* dof", float]:
        ...

    @property
    @abc.abstractmethod
    def dof_target_velocities(self) -> TensorLike["* dof", float]:
        ...

    # TODO timestep
    @abc.abstractmethod
    async def execute_action(
        self, 
        action: ..., 
        # TODO
        position_error_limit: float = 1e-1,
        velocity_error_limit: float = 1e-1,
    ):
        ...


class ProtoArticulationPlanner(abc.ABC):
    class Config(TypedDict):
        base_link: str
        end_link: str

    # TODO
    @abc.abstractmethod
    def compute_action(self, observation: ...):
        ...