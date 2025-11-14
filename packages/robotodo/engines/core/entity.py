import abc
import enum

from tensorspecs import TensorLike
from robotodo.utils.pose import Pose

from robotodo.engines.core.material import ProtoMaterial


# TODO deprecate
# class EntityMotionKind(enum.IntEnum):
#     """
#     TODO doc
#     https://maniskill.readthedocs.io/en/latest/user_guide/concepts/simulation_101.html#actor-types-dynamic-kinematic-static
#     """
#     NONE = -1
#     STATIC = 0
#     KINEMATIC = 1
#     DYNAMIC = 2
# TODO deprecate


class EntityBodyKind(enum.IntEnum):
    NONE = -1
    RIGID = 0
    DEFORMABLE_VOLUME = 1
    DEFORMABLE_SURFACE = 2


class ProtoCollision(abc.ABC):
    enabled: TensorLike["* value", bool]
    on_contact: ...


# TODO mv
class ProtoElement(abc.ABC):
    path: ...
    parent: "ProtoElement"
    children: list["ProtoElement"]


# TODO mv ProtoBody???
class ProtoEntity(abc.ABC):
    label: ...
    # path: Path

    pose: Pose
    # TODO
    velocity: Pose
    
    # parent: ProtoEntity
    # pose_in_parent: Pose

    # TODO 
    # kinematic: TensorLike["* value", bool]
    # TODO deprecate
    # motion_kind: TensorLike["* value", EntityMotionKind]

    collision: ProtoCollision

    inertia: ...

    # TODO dedicated EntityGeometry??
    geometry: ...
    material: ProtoMaterial | None
    body_kind: TensorLike["* value", EntityBodyKind]
    mass: TensorLike["* value"] | None
    mass_center: Pose | None


