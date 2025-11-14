
import abc
import enum
import asyncio

from tensorspecs import TensorLike
from robotodo.engines.core.entity import ProtoEntity
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
class ProtoJoint(abc.ABC):
    # TODO mv ProtoEntity #####
    @classmethod
    def __class_getitem__(cls, label: TensorLike["*", str] | None):
        return cls

    # TODO mv ProtoEntity or necesito??
    @property
    def label(self) -> TensorLike["*", str] | None:
        return None
    # TODO mv ProtoEntity #####

    # TODO
    @abc.abstractmethod
    def __init__(self, ref: ..., scene: ... = None):
        ...

    # TODO mv ProtoEntity
    def __repr__(self):
        return (
            f"""{self.__class__.__qualname__}"""
            f"""{f"[{self.label!r}]" if self.label is not None else ""}"""
            # TODO
            f"""({...}, scene={...})"""
        )    

    # TODO
    kind: TensorLike["*", JointKind]    
    
    body0: ProtoEntity
    pose_in_body0: Pose

    body1: ProtoEntity
    pose_in_body1: Pose


class ProtoFixedJoint(ProtoJoint, abc.ABC):
    kind: TensorLike["*", JointKind.FIXED]


class ProtoRevoluteJoint(ProtoJoint, abc.ABC):
    kind: TensorLike["*", JointKind.REVOLUTE]
    axis: TensorLike["*", Axis]

    @property
    def position_limit(self) -> TensorLike["* minmax:2"]:
        """
        Joint angular position limit (min-max).
        Unit: radians.
        """
        ...


class ProtoPrismaticJoint(ProtoJoint, abc.ABC):
    kind: TensorLike["*", JointKind.PRISMATIC]
    axis: TensorLike["*", Axis]

    @property
    def position_limit(self) -> TensorLike["* minmax:2"]:
        """
        Joint linear position limit (min-max).
        Unit: TODO.
        """
        ...


class ProtoSphericalJoint(ProtoJoint, abc.ABC):
    kind: TensorLike["*", JointKind.SPHERICAL]
    # TODO ...


class DOFKind(enum.IntEnum):
    NONE = -1
    ROTATION = 0
    TRANSLATION = 1


class ProtoArticulation(abc.ABC):
    """
    TODO doc
    """

    # TODO rm
    # body_names: TensorLike["*? body", str]
    # joint_names: TensorLike["*? joint", str]
    # joint_dof_counts: TensorLike["*? joint", int]
    # joint_body0_index: TensorLike["*? joint", str]
    # joint_poses_in_body0: Pose["*? joint", float]

    # TODO
    @classmethod
    @abc.abstractmethod
    def create(self, scene: ProtoScene):
        ...

    @abc.abstractmethod
    def from_entity(self, entity: ProtoEntity):
        ...

    # TODO
    @abc.abstractmethod
    def __init__(self, path: TensorLike["*?", str], scene: ProtoScene):
        ...

    @property
    @abc.abstractmethod
    def joints(self) -> dict[str, ProtoJoint]:
        ...

    @property
    @abc.abstractmethod
    def links(self) -> dict[str, ProtoEntity]:
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

    @property
    @abc.abstractmethod
    def driver(self) -> "ProtoArticulationDriver":
        r"""
        The builtin driver for this articulation.
        """
        ...


class DriveMode(enum.IntEnum):
    NONE = 0
    FORCE = 1
    ACCELERATION = 2


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
    def execute_action(
        self, 
        action: ..., 
        # tolerance: ...,
        # timesteps: TensorLike["*", float] | None = None,
    ) -> asyncio.Task:
        ...


class ProtoArticulationPlanner(abc.ABC):
    # TODO
    @abc.abstractmethod
    def compute_action(self, observation: ...):
        ...