import asyncio
import contextlib
import dataclasses
import functools
import warnings

import numpy
import torch

from robotodo.utils.pose import Pose
from robotodo.utils.geometry import Plane, Box, Sphere, PolygonMesh
from robotodo.engines.core.path import PathExpression, PathExpressionLike

from .scene import Scene
# TODO
from ._kernel import Kernel, enable_physx_deformable_beta


# TODO
# USDPrimRefFunction = Callable[[Scene], "list[pxr.Usd.Prim]"]


# TODO
class USDPrimRef:
    def __init__(
        self,
        path: PathExpressionLike, 
        stage: "pxr.Usd.Stage",
        # kernel: Kernel,
    ):
        raise NotImplementedError


def usd_get_world_convention_transform(prim: "pxr.Usd.Prim"):
    r"""
    Obtain the transformation matrix for USD-to-world conversion.

    TODO NOTE camera
    TODO https://docs.isaacsim.omniverse.nvidia.com/5.0.0/reference_material/reference_conventions.html#world-axes
    TODO https://docs.isaacsim.omniverse.nvidia.com/5.0.0/reference_material/reference_conventions.html#default-camera-axes
    """

    # TODO
    import pxr

    if prim.IsA(pxr.UsdGeom.Camera):
        return numpy.asarray([
            [0, -1, 0, 0], 
            [0, 0, 1, 0], 
            [-1, 0, 0, 0], 
            [0, 0, 0, 1],
        ])
    
    return None


class USDPrimHelper:
    # TODO drop dep on Scene; pass usd stage directly
    # TODO support usd prims directly??
    def __init__(
        self, 
        path: PathExpressionLike, 
        scene: Scene, 
        _usd_prims_ref: "list[pxr.Usd.Prim] | None" = None,
    ):
        self._scene = scene
        self._path = PathExpression(path)

        self._usd_prims_ref = _usd_prims_ref

    # TODO FIXME performance thru prim obj caching
    @property
    # TODO invalidate !!!!!
    # @functools.cached_property
    def _usd_prims(self):
        # TODO
        if self._usd_prims_ref is not None:
            return self._usd_prims_ref

        return [
            self._scene._usd_stage.GetPrimAtPath(p)
            for p in self._scene.resolve(self._path)
        ]
        
    # TODO invalidate!!!!
    @functools.cached_property
    def _usd_xform_cache(self):
        # TODO Usd.TimeCode.Default()
        cache = self._scene._kernel.pxr.UsdGeom.XformCache()

        def _on_changed(notice, sender):
            # TODO
            cache.Clear()

        # TODO NOTE life cycle
        cache._notice_handler = _on_changed
        # TODO
        cache._notice_token = self._scene._kernel.pxr.Tf.Notice.Register(
            self._scene._kernel.pxr.Usd.Notice.ObjectsChanged, 
            cache._notice_handler, 
            self._scene._usd_stage,
        )

        return cache
    
    # TODO
    # TODO https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/usd/transforms/compute-prim-bounding-box.html
    @functools.cached_property
    def _usd_bbox_cache(self):
        raise NotImplementedError
        return self._scene._kernel.pxr.UsdGeom.BBoxCache(
            self._scene._kernel.pxr.Usd.TimeCode.Default(),
            [self._scene._kernel.pxr.UsdGeom.Tokens.default_,],
        )
    
    # TODO cache
    @property
    def _usd_world_convention_transform(self):
        res = []

        for prim in self._usd_prims:
            transform = usd_get_world_convention_transform(prim)
            if transform is None:
                transform = numpy.eye(4)
            res.append(transform)

        return numpy.asarray(res)
    
    @property
    def path(self):
        # TODO
        return [
            prim.GetPath().pathString
            for prim in self._usd_prims
        ]

    # TODO special handling for cameras??
    @property
    def pose(self):
        # TODO 
        return Pose.from_matrix(
            numpy.stack([
                # NOTE matrices in USD are in col-major hence transpose
                numpy.transpose(
                    self._usd_xform_cache
                    .GetLocalToWorldTransform(prim)
                    .RemoveScaleShear()
                )
                for prim in self._usd_prims
            ])
            @ self._usd_world_convention_transform
        )
    
    # TODO special handling for cameras??
    @pose.setter
    def pose(self, value: Pose):
        pose_parent = Pose.from_matrix(
            numpy.stack([
                numpy.asarray(
                    self._usd_xform_cache
                    .GetParentToWorldTransform(prim)
                    .RemoveScaleShear()
                ).T
                for prim in self._usd_prims
            ])
        )
        self.pose_in_parent = pose_parent.inv() * value
    
    # TODO rm
    # # TODO special handling for cameras??
    # @pose.setter
    # def pose(self, value: Pose):
    #     pxr = self._scene._kernel.pxr
    #     omni = self._scene._kernel.omni
    #     # TODO
    #     self._scene._kernel.enable_extension("omni.physx")
    #     self._scene._kernel.import_module("omni.physx.scripts.physicsUtils")

    #     prims = self._usd_prims

    #     p = numpy.broadcast_to(value.p, (len(prims), 3))
    #     q = numpy.broadcast_to(value.q, (len(prims), 4))

    #     pose = Pose(p=p, q=q)
    #     pose_parent = Pose.from_matrix(
    #         numpy.stack([
    #             numpy.asarray(
    #                 self._usd_xform_cache
    #                 .GetParentToWorldTransform(prim)
    #                 .RemoveScaleShear()
    #             ).T
    #             for prim in prims         
    #         ])
    #     )
        
    #     # TODO why?
    #     # pose_local = pose * pose_parent.inv()
    #     pose_local = pose_parent.inv() * pose

    #     p_vec3s = pxr.Vt.Vec3fArrayFromBuffer(pose_local.p)
    #     # NOTE this auto-converts from xyzw to wxyz
    #     q_quats = pxr.Vt.QuatfArrayFromBuffer(pose_local.q)

    #     with pxr.Sdf.ChangeBlock():
    #         for prim, p_vec3f, q_quatf in zip(prims, p_vec3s, q_quats):
    #             xformable = pxr.UsdGeom.Xformable(prim)
    #             omni.physx.scripts.physicsUtils \
    #                 .set_or_add_translate_op(xformable, p_vec3f)
    #             omni.physx.scripts.physicsUtils \
    #                 .set_or_add_orient_op(xformable, q_quatf)
                

    @property
    def pose_in_parent(self):
        def get_local_transform(prim: "pxr.Usd.Prim"):
            transform, _ = self._usd_xform_cache.GetLocalTransformation(prim)
            # NOTE matrices in USD are in col-major hence transpose
            return numpy.transpose(transform.RemoveScaleShear())
        return Pose.from_matrix(
            numpy.stack([
                get_local_transform(prim)
                for prim in self._usd_prims
            ])
            @ self._usd_world_convention_transform
        )
    
    @pose_in_parent.setter
    def pose_in_parent(self, value: Pose):
        pxr = self._scene._kernel.pxr
        omni = self._scene._kernel.omni
        # TODO
        self._scene._kernel.enable_extension("omni.physx")
        self._scene._kernel.import_module("omni.physx.scripts.physicsUtils")

        prims = self._usd_prims
        
        value = Pose.from_matrix(
            numpy.linalg.inv(self._usd_world_convention_transform) 
            @ value.to_matrix()
        )

        p_vec3s = pxr.Vt.Vec3fArrayFromBuffer(value.p)
        # NOTE this auto-converts from xyzw to wxyz
        q_quats = pxr.Vt.QuatfArrayFromBuffer(value.q)
        
        with pxr.Sdf.ChangeBlock():
            for prim, p_vec3, q_quat in zip(prims, p_vec3s, q_quats):
                xformable = pxr.UsdGeom.Xformable(prim)
                omni.physx.scripts.physicsUtils \
                    .set_or_add_translate_op(xformable, p_vec3)
                omni.physx.scripts.physicsUtils \
                    .set_or_add_orient_op(xformable, q_quat)

                
    # TODO rm
    # def compute_geometry(self):
    #     # TODO
    #     import pxr

    #     geometries = []

    #     # TODO scale
    #     for prim in self._usd_prims:
    #         geometry = None

    #         # TODO apply world xform as well????
    #         prim_scale = (
    #             pxr.Gf.Transform(
    #                 self._usd_xform_cache
    #                 .GetLocalToWorldTransform(prim)
    #             )
    #             .GetScale()
    #         )

    #         match prim:
    #             case _ if prim.IsA(pxr.UsdGeom.Plane):
    #                 # TODO
    #                 api = pxr.UsdGeom.Plane(prim)
    #                 api.GetWidthAttr().Get()
    #                 api.GetLengthAttr().Get()
    #                 api.GetAxisAttr().Get()
    #                 pass
    #             case _ if prim.IsA(pxr.UsdGeom.Cube):
    #                 # TODO scaling
    #                 api = pxr.UsdGeom.Cube(prim)
    #                 size = prim.GetSizeAttr().Get()
    #                 # TODO
    #                 geometry = Box(size=numpy.asarray([size, size, size]) * prim_scale)
    #             case _ if prim.IsA(pxr.UsdGeom.Sphere):
    #                 api = pxr.UsdGeom.Sphere(prim)
    #                 # TODO scale xyz
    #                 geometry = Sphere(radius=api.GetRadiusAttr().Get() * prim_scale)
    #             case _ if prim.IsA(pxr.UsdGeom.Mesh):
    #                 api = pxr.UsdGeom.Mesh(prim)
    #                 geometry = PolygonMesh(
    #                     vertices=numpy.asarray(api.GetPointsAttr().Get()) * prim_scale,
    #                     face_vertex_counts=numpy.asarray(api.GetFaceVertexCountsAttr().Get()),
    #                     face_vertex_indices=numpy.asarray(api.GetFaceVertexIndicesAttr().Get()),
    #                 )

    #         geometries.append(geometry)
                
    #     return geometries


# TODO
# TODO scale: ...
def usd_compute_geometry(prim: "pxr.Usd.Prim"):
    # TODO
    import pxr

    match prim:
        case _ if prim.IsA(pxr.UsdGeom.Plane):
            # TODO
            api = pxr.UsdGeom.Plane(prim)
            api.GetWidthAttr().Get()
            api.GetLengthAttr().Get()
            api.GetAxisAttr().Get()
            pass
        case _ if prim.IsA(pxr.UsdGeom.Cube):
            # TODO scaling
            api = pxr.UsdGeom.Cube(prim)
            size = prim.GetSizeAttr().Get()
            # TODO
            return Box(size=numpy.asarray([size, size, size]))
        case _ if prim.IsA(pxr.UsdGeom.Sphere):
            api = pxr.UsdGeom.Sphere(prim)
            return Sphere(radius=api.GetRadiusAttr().Get())
        case _ if prim.IsA(pxr.UsdGeom.Mesh):
            api = pxr.UsdGeom.Mesh(prim)
            return PolygonMesh(
                vertices=numpy.asarray(api.GetPointsAttr().Get()),
                face_vertex_counts=numpy.asarray(api.GetFaceVertexCountsAttr().Get()),
                face_vertex_indices=numpy.asarray(api.GetFaceVertexIndicesAttr().Get()),
            )
        
    return None


class USDRigidHelper:
    def __init__(
        self,
        prims: list["pxr.Usd.Prim"],
        kernel: Kernel,
    ):
        # TODO
        self._usd_prims = prims
        self._kernel = kernel

    # TODO 
    def can_apply(self):
        raise NotImplementedError

    def apply(self):
        omni = self._kernel.omni
        pxr = self._kernel.pxr
        self._kernel.enable_extension("omni.physx")

        for prim in self._usd_prims:
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


class USDSurfaceDeformableHelper:
    def __init__(
        self,
        prims: list["pxr.Usd.Prim"],
        kernel: Kernel,
    ):
        # TODO
        self._usd_prims = prims
        self._kernel = kernel

    # TODO 
    def can_apply(self):
        raise NotImplementedError

    def apply(self):
        omni = self._kernel.omni
        pxr = self._kernel.pxr

        self._kernel.enable_extension("omni.physx")
        enable_physx_deformable_beta(self._kernel)
        
        for prim in self._usd_prims:
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
