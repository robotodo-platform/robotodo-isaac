
import contextlib
import warnings
from typing import Optional, TypedDict, Unpack

import numpy
from robotodo.engines.core.path import PathExpressionLike
from robotodo.utils.geometry import Plane, Box, Sphere, PolygonMesh

from ._kernel import Kernel, get_default_kernel


def usd_ensure_free_path(
    stage: "pxr.Usd.Stage",
    path: str | None = None,
    remove_existing: bool = False,
    # TODO
    kernel: Kernel | None = None,
):
    if kernel is None:
        kernel = get_default_kernel()

    omni = kernel.omni

    if path is not None:
        if remove_existing:
            kernel.enable_extension("omni.usd")
            kernel.enable_extension("omni.kit.commands")
            kernel.import_module("omni.usd")

            # TODO call only when not exists
            # TODO check success??
            delete_prims_command = omni.usd.commands.DeletePrimsCommand(
                [path], 
                stage=stage,
            )
            delete_prims_command.do()

            # is_success = stage.RemovePrim(config.get("path"))
            # if not is_success:
            #     raise RuntimeError(
            #         f"""Failed to remove loaded USD prim at {config.get("path")}"""
            #     )

        if stage.GetPrimAtPath(path).IsValid():
            raise RuntimeError(f"Path already exists: {path}")

    return omni.usd.get_stage_next_free_path(
        stage, 
        path="/" if path is None else path,
        prepend_default_prim=False,
    )


# TODO
import functools
from typing import Callable

import numpy
from robotodo.utils.pose import Pose
from robotodo.engines.core.path import PathExpression
from robotodo.engines.isaac._kernel import Kernel


USDPrimRef = Callable[[], list["pxr.Usd.Prim"]]


# TODO 
def is_usd_prim_ref(ref: USDPrimRef):
    return isinstance(ref, Callable)


USDStageRef = Callable[[], "pxr.Usd.Stage"]



# TODO FIXME perf:
# TODO cache
class USDPrimPathRef(USDPrimRef):
    __slots__ = ["_paths", "_stage_ref"]

    def __init__(self, paths: list[str], stage_ref: USDStageRef):
        self._paths = paths
        self._stage_ref = stage_ref

    def __call__(self):
        stage = self._stage_ref()
        return [
            stage.GetPrimAtPath(path)
            for path in self._paths
        ]


# TODO FIXME perf:
# TODO cache
# TODO keep order
class USDPrimPathExpressionRef(USDPrimRef):
    __slots__ = ["_path_expr", "_stage_ref"]

    def __init__(self, expr: PathExpressionLike, stage_ref: USDStageRef):
        self._path_expr = PathExpression(expr)
        self._stage_ref = stage_ref

    def __call__(self):
        stage = self._stage_ref()

        # TODO 
        import pxr
        # TODO DO NOT dedup !!!!
        # TODO avoid roundtrip
        paths = self._path_expr.resolve((
            prim.GetPath().pathString 
            for prim in stage.Traverse(pxr.Usd.TraverseInstanceProxies())
        ))
        return [
            stage.GetPrimAtPath(path)
            for path in paths
        ]


class USDPrimHelper:
    __slots__ = ["_prims_ref", "_kernel"]

    def __init__(self, ref: USDPrimRef, kernel: Kernel):
        self._prims_ref = ref
        self._kernel = kernel

    # TODO invalidate!!!!
    @functools.lru_cache
    def _xform_cache(self, stage: "pxr.Usd.Stage"):
        pxr = self._kernel.pxr

        # TODO Usd.TimeCode.Default()
        cache = pxr.UsdGeom.XformCache()
        def _on_changed(notice, sender):
            # TODO
            cache.Clear()
        # TODO NOTE life cycle
        cache._notice_handler = _on_changed
        # TODO
        cache._notice_token = pxr.Tf.Notice.Register(
            pxr.Usd.Notice.ObjectsChanged, 
            cache._notice_handler, 
            stage,
        )

        return cache

    @property
    def prims(self):
        return self._prims_ref()

    # TODO mv prim_paths
    @property
    def prim_paths(self):
        return [
            prim.GetPath().pathString
            for prim in self.prims
        ]

    # TODO special handling for cameras??
    @property
    def pose(self):
        # TODO 
        # pxr = self._kernel.pxr

        return Pose.from_matrix(
            numpy.stack([
                # NOTE matrices in USD are in col-major hence transpose
                numpy.transpose(
                    # TODO
                    # pxr.UsdGeom.Imageable(prim)
                    # .ComputeLocalToWorldTransform(pxr.Usd.TimeCode.Default())
                    self._xform_cache(prim.GetStage())
                    .GetLocalToWorldTransform(prim)
                    .RemoveScaleShear()
                )
                for prim in self.prims
            ])
            # TODO mv
            # @ self._world_convention_transform
        )

    # TODO FIXME BUG: _world_convention_transform!!!!
    # TODO special handling for cameras??
    @pose.setter
    def pose(self, value: Pose):
        # TODO 
        # pxr = self._kernel.pxr

        pose_parent = Pose.from_matrix(
            numpy.stack([
                numpy.asarray(
                    # TODO
                    # pxr.UsdGeom.Imageable(prim)
                    # .ComputeParentToWorldTransform(pxr.Usd.TimeCode.Default())
                    self._xform_cache(prim.GetStage())
                    .GetParentToWorldTransform(prim)
                    .RemoveScaleShear()
                ).T
                for prim in self.prims
            ])
        )
        self.pose_in_parent = pose_parent.inv() * value
    
    @property
    def pose_in_parent(self):
        def get_local_transform(prim: "pxr.Usd.Prim"):
            transform, _ = self._xform_cache(prim.GetStage()).GetLocalTransformation(prim)
            # NOTE matrices in USD are in col-major hence transpose
            return numpy.transpose(transform.RemoveScaleShear())
        return Pose.from_matrix(
            numpy.stack([
                get_local_transform(prim)
                for prim in self.prims
            ])
            # TODO mv
            # @ self._world_convention_transform
        )
    
    @pose_in_parent.setter
    def pose_in_parent(self, value: Pose):
        pxr = self._kernel.pxr
        omni = self._kernel.omni
        # TODO
        self._kernel.enable_extension("omni.physx")
        self._kernel.import_module("omni.physx.scripts.physicsUtils")
        
        # TODO mv
        # value = Pose.from_matrix(
        #     numpy.linalg.inv(self._world_convention_transform) 
        #     @ value.to_matrix()
        # )

        p_vec3s = pxr.Vt.Vec3fArrayFromBuffer(value.p)
        # NOTE this auto-converts from xyzw to wxyz
        q_quats = pxr.Vt.QuatfArrayFromBuffer(value.q)
        
        with pxr.Sdf.ChangeBlock():
            for prim, p_vec3, q_quat in zip(self.prims, p_vec3s, q_quats):
                xformable = pxr.UsdGeom.Xformable(prim)
                omni.physx.scripts.physicsUtils \
                    .set_or_add_translate_op(xformable, p_vec3)
                omni.physx.scripts.physicsUtils \
                    .set_or_add_orient_op(xformable, q_quat)

    # TODO rm
    # # TODO !!!
    # # TODO cache
    # @property
    # def _world_convention_transform(self):
    #     raise NotImplementedError
    #     # TODO for cams: necesito??
    #     # return Pose(q=[1., -1., -1., 1.]) 




# TODO
import warnings


# TODO doc
def usd_get_stage_id(stage: "pxr.Usd.Stage", kernel: Kernel) -> int:
    """Get the stage ID of a USD stage.

    Backends: :guilabel:`usd`.

    Args:
        stage: The stage to get the ID of.

    Returns:
        The stage ID.

    Example:

    .. code-block:: python

        >>> import isaacsim.core.experimental.utils.stage as stage_utils
        >>>
        >>> stage = stage_utils.get_current_stage()
        >>> stage_utils.get_stage_id(stage)  # doctest: +NO_CHECK
        9223006
    """

    # TODO
    pxr = kernel.pxr

    stage_cache = pxr.UsdUtils.StageCache.Get()
    stage_id = stage_cache.GetId(stage).ToLongInt()
    if stage_id < 0:
        # TODO better msg
        warnings.warn("TODO usd stage cache")
        stage_id = stage_cache.Insert(stage).ToLongInt()
    return stage_id


# TODO
def usd_physx_query_articulation_properties(prims: list["pxr.Usd.Prim"], kernel: Kernel):
    """
    Query articulation properties.

    TODO doc
    """

    omni = kernel.omni
    kernel.enable_extension("omni.physx")
    pxr = kernel.pxr
    
    result = []

    cached_usd_get_stage_id = functools.lru_cache(usd_get_stage_id)
    physx_property_query_interface = omni.physx.get_physx_property_query_interface()
    for prim in prims:
        v = None
        def articulation_fn(response: ...):
            nonlocal v
            v = response
        physx_property_query_interface.query_prim(
            stage_id=cached_usd_get_stage_id(prim.GetStage(), kernel=kernel),
            query_mode=omni.physx.bindings._physx.PhysxPropertyQueryMode.QUERY_ARTICULATION,
            prim_id=pxr.PhysicsSchemaTools.sdfPathToInt(prim.GetPath()),
            articulation_fn=articulation_fn,
        )
        assert v is not None
        result.append(v)

    return result
    

