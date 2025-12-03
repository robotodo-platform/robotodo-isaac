
import contextlib
import warnings
from typing import Optional, TypedDict, Unpack

import numpy
from robotodo.engines.core.path import PathExpressionLike
from robotodo.utils.geometry import ProtoGeometry, Plane, Box, Sphere, PolygonMesh
from robotodo.engines.isaac._kernel import Kernel, get_default_kernel


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


def usd_create_stage(
    # TODO
    kernel: Kernel | None = None,
) -> "pxr.Usd.Stage":
    if kernel is None:
        kernel = get_default_kernel()

    pxr = kernel.pxr
    kernel.import_module("pxr.Usd")
    kernel.import_module("pxr.UsdGeom")

    stage = pxr.Usd.Stage.CreateInMemory()
    # TODO customizable!!!  also necesito???
    pxr.UsdGeom.SetStageUpAxis(stage, pxr.UsdGeom.Tokens.z)

    # TODO this attaches the stage to the viewer and inits rendering+physics?
    # TODO FIXME rm in the future when: 1) urdf stage= bug has been fixed 2) users are informed of the behavior
    kernel.enable_extension("omni.usd")
    # TODO
    import asyncio
    asyncio.ensure_future(kernel.omni.usd.get_context().attach_stage_async(stage))

    return stage


def usd_load_stage(
    resource: str,
    as_sublayer: bool = False,
    kernel: Kernel | None = None,
) -> "pxr.Usd.Stage":
    if kernel is None:
        kernel = get_default_kernel()

    pxr = kernel.pxr
    kernel.import_module("pxr.Usd")
    kernel.import_module("pxr.UsdGeom")

    # TODO NOTE this allows remote urls to work?
    kernel.enable_extension("omni.usd_resolver")

    if as_sublayer:
        stage = pxr.Usd.Stage.CreateInMemory()
        # TODO customizable!!!  also necesito???
        pxr.UsdGeom.SetStageUpAxis(stage, pxr.UsdGeom.Tokens.z)
        stage.GetRootLayer().subLayerPaths.append(resource)
    else:
        stage = pxr.Usd.Stage.Open(resource)

    # TODO this attaches the stage to the viewer and inits rendering+physics?
    # TODO FIXME rm in the future when: 1) urdf stage= bug has been fixed 2) users are informed of the behavior
    kernel.enable_extension("omni.usd")
    kernel.import_module("omni.usd")
    # TODO
    import asyncio
    asyncio.ensure_future(kernel.omni.usd.get_context().attach_stage_async(stage))

    return stage

    # TODO alt impl: add stage directly but stage maybe readonly!!!!
    # is_success, message = await ctx.open_stage_async(resource_or_model)
    # if not is_success:
    #     raise RuntimeError(f"Failed to load USD scene {resource_or_model}: {message}")
    # stage = ctx.get_stage()
    # # TODO check None
    # return Scene(kernel=kernel, _usd_stage=stage)


# TODO
from typing import Iterable


def usd_add_reference(
    stage: "pxr.Usd.Stage",
    paths: Iterable[str],
    resource: str,
    # TODO
    kernel: Kernel | None = None,
) -> list["pxr.Usd.Prim"]:
    if kernel is None:
        kernel = get_default_kernel()

    pxr = kernel.pxr
    omni = kernel.omni
    # TODO
    kernel.enable_extension("omni.usd")
    kernel.enable_extension("omni.usd.metrics.assembler")
    # TODO NOTE this allows remote urls to work?
    kernel.enable_extension("omni.usd_resolver")

    sdf_layer = pxr.Sdf.Layer.FindOrOpen(resource)
    if not sdf_layer:
        raise RuntimeError(f"Could not get {pxr.Sdf.Layer} for {resource}")
    
    stage_id = pxr.UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
    ret_val = omni.metrics.assembler.core.get_metrics_assembler_interface().check_layers(
        stage.GetRootLayer().identifier, sdf_layer.identifier, stage_id,
    )
    should_use_add_reference_command = ret_val["ret_val"] != 0

    prims = []

    # TODO
    # with pxr.Sdf.ChangeBlock():
    if True:
        for path in paths:
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid():
                # TODO make prim_type customizable??
                prim = stage.DefinePrim(path, "Xform")    

            if should_use_add_reference_command:
                try:
                    omni.usd.commands.AddReferenceCommand(
                        stage=stage,
                        prim_path=prim.GetPath(), 
                        reference=pxr.Sdf.Reference(resource),
                    ).do()
                except Exception as error:
                    warnings.warn(
                        f"USD reference {resource} may have divergent units, "
                        f"please either enable extension `omni.usd.metrics.assembler` "
                        f"or convert into right units: {error}"
                    )
                    is_success = prim.GetReferences().AddReference(resource)
                    if not is_success:
                        raise RuntimeError(f"Invalid USD reference: {resource}")
            else:
                is_success = prim.GetReferences().AddReference(resource)
                if not is_success:
                    raise RuntimeError(f"Invalid USD reference: {resource}")
            
            prims.append(prim)
        
    return prims


import os
import tempfile


def usd_import_urdf(
    stage: "pxr.Usd.Stage",
    paths: Iterable[str],
    resource_or_model: str,
    kernel: Kernel,
):
    pxr = kernel.pxr
    # TODO
    # TODO NOTE BUG isaacsim.asset.importer.urdf prior to 2.4.27 has `.rotateMeshX` for `.dae` files which causes the links to be rotated!!!
    # https://github.com/isaac-sim/IsaacSim/commit/e680e71274626b275d5dbe755f04ccdea7bbe97c#diff-8f3f54a270af13942c9904d254c27992c5eed50f6addf95ce31060ea97c1c0ffL267
    kernel.enable_extension("isaacsim.asset.importer.urdf")
    isaacsim = kernel.import_module("isaacsim.asset.importer.urdf")

    with tempfile.TemporaryDirectory() as tmpdir:
        usd_path = os.path.join(tmpdir, "todo.usd")

        urdf_import_config = isaacsim.asset.importer.urdf.URDFCreateImportConfig().do()
        urdf_import_config.make_default_prim = True  # Make the robot the default prim in the scene
        # import_config.fix_base = config.get("fix_root_link", False) # Fix the base of the robot to the ground
        urdf_import_config.merge_fixed_joints = False
        # import_config.convex_decomp = False  # Disable convex decomposition for simplicity
        # import_config.self_collision = False  # Disable self-collision for performance

        isaacsim.asset.importer.urdf.URDFParseAndImportFile(
            urdf_path=resource_or_model,
            # TODO
            import_config=urdf_import_config,
            dest_path=usd_path,
        ).do()

        # TODO NOTE this alt impl does not respect merge_fixed_joints due to a bug?
        # urdf_robot_model = isaacsim.asset.importer.urdf.URDFParseFile(
        #     resource_or_model,
        #     import_config=urdf_import_config,
        # ).do()
        # isaacsim.asset.importer.urdf.URDFImportRobot(
        #     urdf_robot=urdf_robot_model,
        #     dest_path=usd_path,
        # ).do()

        layer = pxr.Usd.Stage.Open(usd_path).Flatten()

        return usd_add_reference(
            stage,
            paths=paths,
            # TODO rm
            # resource=usd_path,
            resource=layer.identifier,
            kernel=kernel,
        )
    


# TODO
import functools
from typing import Any, Callable, Protocol

import numpy
from robotodo.utils.pose import Pose
from robotodo.engines.core.path import PathExpression
from robotodo.engines.isaac._kernel import Kernel


class USDPrimRef(Protocol):
    def __call__(self) -> list["pxr.Usd.Prim"]:
        ...


# TODO 
def is_usd_prim_ref(ref: USDPrimRef | Any):
    return isinstance(ref, Callable)


class USDStageRef(Protocol):
    def __call__(self) -> "pxr.Usd.Stage":
        ...


# TODO 
def is_usd_stage_ref(ref: USDStageRef | Any):
    return isinstance(ref, Callable)


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


# TODO FIXME: matrices in USD are NOT in col-major!!!
class USDXformView:
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
        pxr = self._kernel.pxr

        return Pose.from_matrix(
            numpy.stack([
                # NOTE matrices in USD are in col-major hence transpose
                numpy.transpose(
                    # TODO
                    pxr.UsdGeom.Imageable(prim)
                    .ComputeLocalToWorldTransform(pxr.Usd.TimeCode.Default())
                    # self._xform_cache(prim.GetStage())
                    # .GetLocalToWorldTransform(prim)
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
        pxr = self._kernel.pxr

        pose_parent = Pose.from_matrix(
            numpy.stack([
                numpy.transpose(
                    # TODO
                    pxr.UsdGeom.Imageable(prim)
                    .ComputeParentToWorldTransform(pxr.Usd.TimeCode.Default())
                    # self._xform_cache(prim.GetStage())
                    # .GetParentToWorldTransform(prim)
                    .RemoveScaleShear()
                )
                for prim in self.prims
            ])
        )
        self.pose_in_parent = pose_parent.inv() * value
    
    @property
    def pose_in_parent(self):
        pxr = self._kernel.pxr
        def get_local_transform(prim: "pxr.Usd.Prim"):
            xformable = pxr.UsdGeom.Xformable(prim)
            transform = xformable.GetLocalTransformation(pxr.Usd.TimeCode.Default())
            # TODO BUG cache invalidation issues or no ??
            # transform, _ = self._xform_cache(prim.GetStage()).GetLocalTransformation(prim)
            transform = transform.RemoveScaleShear()
            # NOTE matrices in USD are in col-major hence transpose
            return numpy.transpose(transform)
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
        
        # TODO
        # with pxr.Sdf.ChangeBlock():
        if True:
            for prim, p_vec3, q_quat in zip(self.prims, p_vec3s, q_quats):
                xformable = pxr.UsdGeom.Xformable(prim)
                omni.physx.scripts.physicsUtils \
                    .setup_transform_as_scale_orient_translate(xformable)
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
# TODO scale: ...
def usd_compute_geometry(
    prims: list["pxr.Usd.Prim"], 
    kernel: Kernel,
) -> list[ProtoGeometry | PolygonMesh | None]:
    # TODO
    pxr = kernel.pxr

    # TODO
    geometries = []

    for prim in prims:
        # TODO rm
        # scale = (
        #     pxr.Gf.Transform(
        #         pxr.UsdGeom.Imageable(prim)
        #         .ComputeLocalToWorldTransform(pxr.Usd.TimeCode.Default())                
        #     )
        #     .GetScale()
        # )

        # TODO ComputeRelativeTransform

        geometry = None
        
        match prim:
            # case _ if prim.IsA(pxr.UsdGeom.Plane):
            #     # TODO
            #     api = pxr.UsdGeom.Plane(prim)
            #     api.GetWidthAttr().Get()
            #     api.GetLengthAttr().Get()
            #     api.GetAxisAttr().Get()
            #     pass
            case _ if prim.IsA(pxr.UsdGeom.Cube):
                # TODO scaling
                api = pxr.UsdGeom.Cube(prim)
                size = prim.GetSizeAttr().Get()
                # TODO
                geometry = Box(size=numpy.asarray([size, size, size]))
            # case _ if prim.IsA(pxr.UsdGeom.Sphere):
            #     # TODO * scale
            #     api = pxr.UsdGeom.Sphere(prim)
            #     Sphere(radius=api.GetRadiusAttr().Get())
            case _ if prim.IsA(pxr.UsdGeom.Mesh):
                api = pxr.UsdGeom.Mesh(prim)
                geometry = PolygonMesh(
                    vertices=numpy.asarray(api.GetPointsAttr().Get()),
                    face_vertex_counts=numpy.asarray(api.GetFaceVertexCountsAttr().Get()),
                    face_vertex_indices=numpy.asarray(api.GetFaceVertexIndicesAttr().Get()),
                )
            case _:
                # TODO
                pass

        geometries.append(geometry)

    return geometries
        

# TODO doc
def usd_get_stage_id(stage: "pxr.Usd.Stage", kernel: Kernel) -> int:
    r"""
    TODO doc

    Get the stage ID of a USD stage.

    Backends: :guilabel:`usd`.

    Args:
        stage: The stage to get the ID of.

    Returns:
        The stage ID.
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
def usd_physx_query_articulation_properties(
    prims: list["pxr.Usd.Prim"], 
    kernel: Kernel,
):
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
    

# TODO
# usd_physics_remove_rigid


def usd_physics_make_rigid(
    prims: list["pxr.Usd.Prim"], 
    kernel: Kernel,
    deep: bool = True,
):
    omni = kernel.omni
    pxr = kernel.pxr
    kernel.enable_extension("omni.physx")
    kernel.import_module("omni.physx.scripts.deformableUtils")

    if deep:
        prims = [
            child_prim 
            for prim in prims
            for child_prim in pxr.Usd.PrimRange(prim) 
        ]

    for prim in prims:       
        omni.physx.scripts.deformableUtils.remove_deformable_body(
            prim.GetStage(), 
            prim_path=prim.GetPath(),
        )

        # NOTE pxr.UsdGeom.Xformable should work 
        # however we use pxr.UsdGeom.Gprim to ensure consistency with deformables
        if not prim.IsA(pxr.UsdGeom.Gprim):
            continue

        omni.physx.scripts.utils.setRigidBody(
            prim, 
            approximationShape=None, 
            kinematic=False,
        )


# TODO
from robotodo.engines.isaac._kernel import enable_physx_deformable_beta


# TODO
# usd_physics_remove_deformable


def usd_physics_make_surface_deformable(
    prims: list["pxr.Usd.Prim"], 
    kernel: Kernel,
    deep: bool = True,
):
    omni = kernel.omni
    pxr = kernel.pxr

    kernel.enable_extension("omni.physx")
    kernel.import_module("omni.physx.scripts.deformableUtils")
    enable_physx_deformable_beta(kernel)
    
    if deep:
        prims = [
            child_prim 
            for prim in prims
            for child_prim in pxr.Usd.PrimRange(prim) 
        ]

    for prim in prims:
        omni.physx.scripts.utils.removeRigidBody(prim)

        if not prim.IsA(pxr.UsdGeom.Gprim):
            continue

        if prim.IsA(pxr.UsdGeom.Mesh):
            api = pxr.UsdGeom.Mesh(prim)

            # TODO check
            if not numpy.array_equiv(api.GetFaceVertexCountsAttr().Get(), 3):
                warnings.warn(f"Mesh USD prim is not a triangle mesh, converting: {api}")
                # TODO do not use this
                face_vertex_indices = omni.physx.scripts.deformableUtils.triangulate_mesh(api)
                # TODO
                api.GetFaceVertexIndicesAttr().Set(face_vertex_indices)
                api.GetFaceVertexCountsAttr().Set(
                    numpy.full(len(face_vertex_indices) // 3, 3)
                )

            is_success = omni.physx.scripts.deformableUtils.set_physics_surface_deformable_body(
                prim.GetStage(), 
                prim_path=prim.GetPath(),
            )
            if not is_success:
                raise RuntimeError("TODO")
            
            # TODO
            prim.ApplyAPI("PhysxSurfaceDeformableBodyAPI")
            if prim.HasAPI("PhysxSurfaceDeformableBodyAPI"):
                prim.GetAttribute("physxDeformableBody:selfCollision").Set(True)

            return prim
        
        warnings.warn(f"Non-mesh USD prim cannot be surface deformable: {prim}")
