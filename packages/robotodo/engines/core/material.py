import abc
import enum

from tensorspecs import TensorLike
from robotodo.utils.pose import Pose


class ProtoMaterial(abc.ABC):
    # TODO
    static_friction: TensorLike["* value"]
    dynamic_friction: TensorLike["* value"]
    density: TensorLike["* value"]

    young: TensorLike["* value"] | None
    poisson: TensorLike["* value"] | None

    surface_thickness: TensorLike["* value"] | None