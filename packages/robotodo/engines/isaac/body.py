# SPDX-License-Identifier: Apache-2.0

"""
Body.
"""


import asyncio
import contextlib
import dataclasses
import functools
import warnings
from typing import NotRequired, TypedDict

import numpy
import torch

from robotodo.utils.pose import Pose
from robotodo.utils.geometry import Plane, Box, Sphere, PolygonMesh
from robotodo.utils.event import BaseSubscriptionPartialAsyncEventStream
from robotodo.engines.core.error import InvalidReferenceError
from robotodo.engines.core.body import BodyKind, BodySpec, ProtoBody
from robotodo.engines.core.path import PathExpression, PathExpressionLike, is_path_expression_like
from robotodo.engines.isaac.scene import Scene
from robotodo.engines.isaac.entity import Entity
from robotodo.engines.isaac.material import Material
from robotodo.engines.isaac.kernel import Kernel
from robotodo.engines.isaac._utils.usd import (
    USDPrimRef, 
    USDPrimPathExpressionRef, 
    USDXformView,
    is_usd_prim_ref,
    usd_compute_geometry,
    usd_physics_make_rigid,
    usd_physics_make_surface_deformable,
)


# TODO find all bodies under the path instead???
class Body(ProtoBody):

    class _USDBodyPrimRef:
        def __init__(self, ref: USDPrimRef, kernel: Kernel):
            self._ref = ref
            self._kernel = kernel

        def __call__(self):
            pxr = self._kernel._pxr
            return [
                child_prim
                for prim in self._ref()
                # NOTE this already includes the prim itself
                for child_prim in pxr.Usd.PrimRange(
                    prim, 
                    # TODO rm???
                    # pxr.Usd.TraverseInstanceProxies(
                    #     pxr.Usd.PrimAllPrimsPredicate
                    # ),
                )
                if child_prim.HasAPI(pxr.UsdPhysics.RigidBodyAPI)
                    or pxr.UsdPhysics.RigidBodyAPI.CanApply(child_prim)
                    or "OmniPhysicsBodyAPI" in prim.GetAppliedSchemas()
            ]

    # TODO
    _usd_prim_ref: USDPrimRef
    _scene: Scene

    _impl_label: str | None = None

    @classmethod
    def create(
        cls, 
        ref: PathExpressionLike, 
        scene: Scene, 
        spec: BodySpec = BodySpec(),
    ):
        pxr = scene._kernel._pxr

        prims = []

        # TODO instancing???
        for prim_path in PathExpression(ref).expand():
            prim = ...

            match spec.get("geometry", None):
                case None:
                    prim = scene._usd_stage.GetPrimAtPath(prim_path)
                    if not prim:
                        api = pxr.UsdGeom.Xform.Define(scene._usd_stage, prim_path)
                        prim = api.GetPrim()

                case Plane() as plane:
                    # NOTE not using pxr.UsdGeom.Plane because it is not renderable in Omniverse!!!
                    _use_usd_plane = False
                    if _use_usd_plane:
                        api = pxr.UsdGeom.Plane.Define(scene._usd_stage, prim_path)
                        # TODO batching !!!!!
                        api.CreateWidthAttr().Set(plane.size[0])
                        api.CreateLengthAttr().Set(plane.size[1])
                        api.CreateAxisAttr().Set("Z")
                    else:
                        api = pxr.UsdGeom.Mesh.Define(scene._usd_stage, prim_path)

                        polygon_mesh = PolygonMesh.from_plane(plane)
                        # TODO cast float/double
                        api.CreatePointsAttr().Set(polygon_mesh.vertices)
                        api.CreateFaceVertexCountsAttr().Set(polygon_mesh.face_vertex_counts)
                        api.CreateFaceVertexIndicesAttr().Set(polygon_mesh.face_vertex_indices)
                    prim = api.GetPrim()

                # TODO
                case Box() as box:
                    api = pxr.UsdGeom.Cube.Define(scene._usd_stage, prim_path)
                    if len(numpy.unique(box.size)) == 1:
                        [size] = numpy.unique(box.size)
                        api.CreateSizeAttr().Set(float(size))
                        (api.GetScaleOp() or api.AddScaleOp()) \
                            .Set(pxr.Gf.Vec3f(1., 1., 1.))
                    else:
                        api.CreateSizeAttr().Set(1.)
                        # TODO upstream Box already assumes xyz so validate??
                        (api.GetScaleOp() or api.AddScaleOp()) \
                            .Set(pxr.Gf.Vec3f(*numpy.broadcast_to(box.size, 3)))
                    prim = api.GetPrim()
                    
                case Sphere() as sphere:
                    api = pxr.UsdGeom.Sphere.Define(scene._usd_stage, prim_path)
                    # TODO batch 
                    api.CreateRadiusAttr().Set(sphere.radius)
                    prim = api.GetPrim()
                    
                case PolygonMesh() as polygon_mesh:
                    # TODO
                    api = pxr.UsdGeom.Mesh.Define(scene._usd_stage, prim_path)
                    # TODO cast float/double
                    api.CreatePointsAttr().Set(polygon_mesh.vertices)
                    api.CreateFaceVertexCountsAttr().Set(polygon_mesh.face_vertex_counts)
                    api.CreateFaceVertexIndicesAttr().Set(polygon_mesh.face_vertex_indices)
                    prim = api.GetPrim()

                # TODO
                case _ as unknown:
                    raise ValueError(f"Unknown geometry: {unknown}")
                
            prims.append(prim)

        match spec.get("kind", None):
            case None:
                pass
            case BodyKind.RIGID:
                usd_physics_make_rigid(prims, kernel=scene._kernel)
            case BodyKind.DEFORMABLE_SURFACE:
                usd_physics_make_surface_deformable(prims, kernel=scene._kernel)
            case BodyKind.DEFORMABLE_VOLUME:
                # TODO
                raise NotImplementedError("TODO")
            case _ as unknown:
                raise ValueError(f"Unknown body kind: {unknown}")
                
        return cls(lambda: prims, scene=scene)
    
    @classmethod
    def load_usd(
        cls, 
        ref: PathExpressionLike, 
        source: str, 
        scene: Scene,
        spec_overrides: BodySpec = BodySpec(),
    ):
        entity = Entity.load_usd(ref, source=source, scene=scene)

        prims = entity._usd_prim_ref()
        match spec_overrides.get("kind", None):
            case None:
                pass
            case BodyKind.RIGID:
                usd_physics_make_rigid(prims, kernel=scene._kernel)
            case BodyKind.DEFORMABLE_SURFACE:
                usd_physics_make_surface_deformable(prims, kernel=scene._kernel)
            case BodyKind.DEFORMABLE_VOLUME:
                # TODO
                raise NotImplementedError("TODO")
            case _ as unknown:
                raise ValueError(f"Unknown body kind: {unknown}")

        return cls(entity)
    
    @classmethod
    def load(
        cls, 
        ref: PathExpressionLike, 
        source: str, 
        scene: Scene,
        spec_overrides: BodySpec = BodySpec(),
    ):
        # TODO
        return cls.load_usd(
            ref, 
            source=source, 
            scene=scene,
            spec_overrides=spec_overrides,
        )

    # TODO next
    @classmethod
    def find(
        cls,
        ref: "Body | Entity | USDPrimRef | PathExpressionLike",
        scene: Scene | None = None,
    ):
        # TODO this feels hacky
        instance = cls(ref=ref, scene=scene)
        instance._usd_prim_ref = Body._USDBodyPrimRef(
            instance._usd_prim_ref, 
            kernel=instance._scene._kernel,
        )
        return instance

    # TODO
    def __init__(
        self,
        ref: "Body | Entity | USDPrimRef | PathExpressionLike",
        scene: Scene | None = None,
    ):
        match ref:
            case Body() as body:
                assert scene is None
                self._usd_prim_ref = body._usd_prim_ref
                self._scene = body._scene
            case Entity() as entity:
                assert scene is None
                self._usd_prim_ref = entity._usd_prim_ref
                self._scene = entity._scene
            case ref if is_usd_prim_ref(ref):
                # TODO
                assert scene is not None
                self._usd_prim_ref = ref
                self._scene = scene
            case expr if is_path_expression_like(ref):
                # TODO
                assert scene is not None
                self._usd_prim_ref = USDPrimPathExpressionRef(
                    expr,
                    stage_ref=lambda: scene._usd_stage,
                )
                self._scene = scene
            case _:
                raise InvalidReferenceError(ref)

    # TODO
    @functools.cached_property
    def _usd_xform_view(self):
        return USDXformView(
            self._usd_prim_ref, 
            kernel=self._scene._kernel,
        )
    
    @functools.cached_property
    def viewer(self):
        return BodyViewer(self)

    # TODO
    @property
    def prototypes(self):
        raise NotImplementedError

    # TODO
    def astype(self, prototype: ...):
        raise NotImplementedError

    # TODO
    @property
    def path(self):
        return [
            prim.GetPath().pathString
            for prim in self._usd_prim_ref()
        ]
    
    @property
    def scene(self):
        return self._scene

    @property
    def label(self):
        return self._impl_label

    @property
    def pose(self):
        return self._usd_xform_view.pose
    
    @pose.setter
    def pose(self, value: Pose):
        self._usd_xform_view.pose = value

    @property
    def pose_in_parent(self):
        return self._usd_xform_view.pose_in_parent
    
    @pose_in_parent.setter
    def pose_in_parent(self, value: Pose):
        self._usd_xform_view.pose_in_parent = value

    # TODO
    @functools.cached_property
    def material(self):
        pxr = self._scene._kernel._pxr

        # TODO
        def _material_prims_ref():
            prims = self._usd_prim_ref()
            material_apis, _ = pxr.UsdShade.MaterialBindingAPI.ComputeBoundMaterials(prims)
            
            material_prims = []
            for prim, material_api in zip(prims, material_apis):
                if not material_api:
                    # TODO
                    warnings.warn(f"USD prim does not have a material applied: {prim}")
                material_prims.append(material_api.GetPrim())

            return material_prims

        return Material(_material_prims_ref, scene=self._scene)

    @functools.cached_property
    def collision(self):
        return Collision(self)
    
    # TODO better repr format??>
    # TODO not all children are included???
    # TODO optimize: instanceable assets may have shared geoms
    @property
    def geometry(self):
        kernel = self._scene._kernel
        pxr = kernel._pxr

        res = []
        for prim in self._usd_prim_ref():

            # TODO
            assert prim.IsA(pxr.UsdGeom.Imageable)
            world_transform_parent = numpy.asarray(
                pxr.UsdGeom.Imageable(prim)
                .ComputeLocalToWorldTransform(pxr.Usd.TimeCode.Default())
                # TODO NOTE robotodo uses the col-vector convention
                .GetTranspose()
            )

            child_prims = list(
                prim
                # NOTE the first child_prim would be the prim itself
                for prim in pxr.Usd.PrimRange(
                    prim, 
                    pxr.Usd.TraverseInstanceProxies(
                        pxr.Usd.PrimAllPrimsPredicate
                    ),
                )
                if prim.IsA(pxr.UsdGeom.Imageable)
            )

            world_transform_children = numpy.asarray([
                pxr.UsdGeom.Imageable(child_prim)
                .ComputeLocalToWorldTransform(pxr.Usd.TimeCode.Default())
                # TODO NOTE robotodo uses the col-vector convention
                .GetTranspose()
                for child_prim in child_prims
            ])

            transforms = numpy.linalg.inv(world_transform_parent) @ world_transform_children
            geometries = usd_compute_geometry(
                child_prims, 
                kernel=kernel,
            )

            def set_transform(geometry: ..., transform: ...):
                geometry.transform = transform
                return geometry
            res.append([
                set_transform(geometry, transform)
                for geometry, transform in zip(geometries, transforms, strict=True)
                if geometry is not None
            ])

        return res

    # TODO deep?
    @property
    def kind(self):
        pxr = self._scene._kernel._pxr

        res = []

        for prim in self._usd_prim_ref():
            v = BodyKind.NONE

            for schema in prim.GetAppliedSchemas():
                match schema:
                    case "PhysicsRigidBodyAPI":
                        if (
                            pxr.UsdPhysics.RigidBodyAPI(prim)
                            .GetRigidBodyEnabledAttr()
                            .Get()
                        ):
                            v = BodyKind.RIGID
                    case "OmniPhysicsVolumeDeformableSimAPI":
                        if (
                            prim
                            .GetAttribute("omniphysics:deformableBodyEnabled")
                            .Get()
                        ):
                            v = BodyKind.DEFORMABLE_VOLUME
                    case "OmniPhysicsSurfaceDeformableSimAPI":
                        if (
                            prim
                            .GetAttribute("omniphysics:deformableBodyEnabled")
                            .Get()
                        ):
                            v = BodyKind.DEFORMABLE_SURFACE
                    case _:
                        pass

            res.append(v.value)

        return numpy.asarray(res)
    
    # TODO
    @kind.setter
    def kind(self, value: ...):
        prims = numpy.asarray(self._usd_prim_ref())
        value = numpy.broadcast_to(value, shape=len(prims))

        usd_physics_make_rigid(
            prims[numpy.argwhere(value == BodyKind.RIGID.value)].flatten(), 
            kernel=self._scene._kernel,
            deep=False,
        )

        usd_physics_make_surface_deformable(
            prims[numpy.argwhere(value == BodyKind.DEFORMABLE_SURFACE.value)].flatten(), 
            kernel=self._scene._kernel,
            deep=False,
        )
        
        # TODO deformable volume
    
    @property
    def fixed(self):
        pxr = self._scene._kernel._pxr

        res = []

        for prim in self._usd_prim_ref():
            v = False

            for schema in prim.GetAppliedSchemas():
                match schema:
                    case "PhysicsRigidBodyAPI":
                        if (
                            pxr.UsdPhysics.RigidBodyAPI(prim)
                            .GetKinematicEnabledAttr()
                            .Get()
                        ):
                            v = True
                    case "OmniPhysicsBodyAPI":
                        if (
                            prim
                            .GetAttribute("omniphysics:kinematicEnabled")
                            .Get()
                        ):
                            v = True
                    case _:
                        pass

            res.append(v)
                
        return numpy.asarray(res)

    # TODO
    @fixed.setter
    def fixed(self, value):
        pxr = self._scene._kernel._pxr

        prims = self._usd_prim_ref()
        value = numpy.broadcast_to(value, shape=len(prims))

        for prim, v in zip(prims, value):
            for schema in prim.GetAppliedSchemas():
                match schema:
                    case "PhysicsRigidBodyAPI":
                        (
                            pxr.UsdPhysics.RigidBodyAPI(prim)
                            .GetKinematicEnabledAttr()
                            .Set(bool(v))
                        )
                    case "OmniPhysicsBodyAPI":
                        (
                            prim
                            .GetAttribute("omniphysics:kinematicEnabled")
                            .Set(bool(v))
                        )
                    case _:
                        pass


# TODO 
class RigidBody(Body):

    @classmethod
    def load_usd(
        cls, 
        ref: PathExpressionLike, 
        source: str, 
        scene: Scene,
        spec_overrides: BodySpec = BodySpec(),
    ):
        spec_overrides = BodySpec(spec_overrides)

        match spec_overrides.get("kind", None):
            case None:
                spec_overrides["kind"] = BodyKind.RIGID
            case BodyKind.RIGID:
                pass
            case _ as unsupported_kind:
                raise ValueError(f"Inconsistent non-rigid body kind: {unsupported_kind}")
            
        return super().load_usd(
            ref,
            source=source,
            scene=scene,
            spec_overrides=spec_overrides,
        )
    
    @classmethod
    def load(
        cls, 
        ref: PathExpressionLike, 
        source: str, 
        scene: Scene,
        spec_overrides: BodySpec = BodySpec(),
    ):
        # TODO
        return cls.load_usd(
            ref, 
            source=source, 
            scene=scene,
            spec_overrides=spec_overrides,
        )

    @classmethod
    def create(
        cls, 
        ref: PathExpressionLike, 
        scene: Scene, 
        spec: BodySpec = BodySpec(),
    ):
        spec_n = BodySpec(spec)

        match spec_n.get("kind", None):
            case None:
                spec_n["kind"] = BodyKind.RIGID
            case BodyKind.RIGID:
                pass
            case _ as unsupported_kind:
                raise ValueError(f"Inconsistent non-rigid body kind: {unsupported_kind}")

        return super().create(
            ref=ref,
            scene=scene,
            spec=spec_n,
        )

    @functools.cached_property
    def _omni_physics_rigid_body_view_cache(self):
        try:
            paths = [
                prim.GetPath().pathString
                for prim in self._usd_prim_ref()
            ]
            self._scene._omni_physx_simulation.flush_changes()
            omni_physics_tensor_view = self._scene._omni_physics_tensor_view
            rigid_body_view = (
                omni_physics_tensor_view
                .create_rigid_body_view(paths)
            )
            assert rigid_body_view is not None
            assert rigid_body_view.check()
            def should_invalidate():
                return not (
                    omni_physics_tensor_view.is_valid
                    and rigid_body_view.check()
                )
        except Exception as error:
            raise RuntimeError(
                f"Failed to create rigid body physics view from "
                f"resolved USD paths (are they valid?): {paths}"
            ) from error
        return rigid_body_view, should_invalidate
    
    @property
    def _omni_physics_rigid_body_view(self):
        while True:
            rigid_body_view, should_invalidate = self._omni_physics_rigid_body_view_cache
            if should_invalidate():
                del self._omni_physics_rigid_body_view_cache
            else:
                return rigid_body_view
        
    # TODO is this relevant?? https://docs.omniverse.nvidia.com/kit/docs/omni_physics/108.1/dev_guide/rigid_bodies_articulations/rigid_bodies.html#automatic-computation-of-rigid-body-mass-distribution
    @property
    def mass(self):
        return self._omni_physics_rigid_body_view.get_masses()
    
    # TODO BUG upstream: physics tensor api: changes not written to usd until .step
    @mass.setter
    def mass(self, value):
        view = self._omni_physics_rigid_body_view
        value_ = torch.broadcast_to(torch.asarray(value), (view.count, 1))
        view.set_masses(value_, indices=torch.arange(view.count))

    @property
    def mass_center_pose(self):
        coms = self._omni_physics_rigid_body_view.get_coms()
        return Pose(
            p=coms[..., [0, 1, 2]],
            q=coms[..., [3, 4, 5, 6]],
        )
    
    # TODO BUG upstream: physics tensor api: changes not written to usd until .step
    @mass_center_pose.setter
    def mass_center_pose(self, value: Pose):
        view = self._omni_physics_rigid_body_view
        value_ = torch.broadcast_to(
            torch.concat((torch.asarray(value.p), torch.asarray(value.q))),
            size=(view.count, 7),
        )
        view.set_coms(value_, indices=torch.arange(view.count))


# TODO
# TODO https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/deformables_beta/physx_deformable_schema.html
class DeformableBody(Body):
    @classmethod
    def create(
        cls, 
        ref: PathExpressionLike, 
        scene: Scene, 
        spec: BodySpec = BodySpec(),
    ):
        spec = BodySpec(spec)

        match spec.get("kind", None):
            case None:
                # TODO DEFORMABLE_VOLUME should be default??
                spec["kind"] = BodyKind.DEFORMABLE_SURFACE
            case BodyKind.DEFORMABLE_SURFACE:
                pass
            case BodyKind.DEFORMABLE_VOLUME:
                pass
            case _ as unsupported_kind:
                raise ValueError(f"Inconsistent non-deformable body kind: {unsupported_kind}")

        return super().create(
            ref=ref,
            scene=scene,
            spec=spec,
        )

    @classmethod
    def load_usd(
        cls, 
        ref: PathExpressionLike, 
        source: str, 
        scene: Scene,
        spec_overrides: BodySpec = BodySpec(),
    ):
        spec_overrides = BodySpec(spec_overrides)

        match spec_overrides.get("kind", None):
            case None:
                # TODO DEFORMABLE_VOLUME should be default??
                spec_overrides["kind"] = BodyKind.DEFORMABLE_SURFACE
            case BodyKind.DEFORMABLE_SURFACE:
                pass
            case BodyKind.DEFORMABLE_VOLUME:
                pass
            case _ as unsupported_kind:
                raise ValueError(f"Inconsistent non-deformable body kind: {unsupported_kind}")
            
        return super().load_usd(
            ref,
            source=source,
            scene=scene,
            spec_overrides=spec_overrides,
        )
    
    @classmethod
    def load(
        cls, 
        ref: PathExpressionLike, 
        source: str, 
        scene: Scene,
        spec_overrides: BodySpec = BodySpec(),
    ):
        # TODO
        return cls.load_usd(
            ref, 
            source=source, 
            scene=scene,
            spec_overrides=spec_overrides,
        )


# TODO TensorTable
@dataclasses.dataclass
class ContactPoint:
    """
    TODO doc units
    """

    position: ...
    """The position of the contact point in world frame."""
    impulse: ...
    """TODO The impulse applied."""
    normal: ...
    """The direction of impulse."""
    separation: ...
    """The minimum distance between two shapes involved in the contact."""

# TODO TensorTable
@dataclasses.dataclass
class ContactAnchor:
    """
    TODO doc units
    """

    position: ...
    """The position of the contact friction anchor in world frame."""
    impulse: ...
    """TODO The impulse applied."""

@dataclasses.dataclass
class Contact:
    """
    TODO doc units
    """

    body0: "Body"
    """TODO doc"""
    body1: "Body"
    """TODO doc"""
    points: ContactPoint | None = None
    """Contact points. `None` to indicate contact loss."""
    anchors: ContactAnchor | None = None
    """Contact friction anchors. `None` to indicate contact loss."""


class ContactAsyncEventStream(
    BaseSubscriptionPartialAsyncEventStream[Contact]
):
    def __init__(self, body: "Body"):
        self._body = body

    # TODO
    # def _omni_physx_contact_report_callback(
    #     self,
    #     contact_headers: list, 
    #     contact_datas: list, 
    #     friction_anchors: list | None = None,
    # ):
    #     pass

    @contextlib.contextmanager
    def subscribe(self, callable):
        scene = self._body._scene
        pxr = scene._kernel._pxr
        
        # TODO
        scene._kernel._omni_enable_extension("omni.usd.schema.physx")

        for prim in self._body._usd_prim_ref():
            if prim.HasAPI(pxr.PhysxSchema.PhysxContactReportAPI):
                api = pxr.PhysxSchema.PhysxContactReportAPI(prim)
            else:
                api = pxr.PhysxSchema.PhysxContactReportAPI.Apply(prim)
            # TODO necesito???
            api.CreateThresholdAttr().Set(0)

        # TODO cache?
        def contact_report_callback(
            contact_headers, 
            contact_datas, 
            friction_anchors,
        ):
            """
            TODO doc

            """

            # TODO
            prim_paths = set(
                prim.GetPath().pathString
                for prim in self._body._usd_prim_ref()
            )

            for contact_header in contact_headers:
                path_entity0 = pxr.PhysicsSchemaTools.intToSdfPath(contact_header.collider0)
                path_entity1 = pxr.PhysicsSchemaTools.intToSdfPath(contact_header.collider1)


                # TODO FIXME perf
                # TODO match ancesters as well?
                if any(
                    str(path) not in prim_paths
                    for path in (path_entity0, path_entity1)
                ):
                    continue

                entity0 = Body(path_entity0, scene=scene)
                entity1 = Body(path_entity1, scene=scene)

                # TODO https://docs.omniverse.nvidia.com/kit/docs/omni_physics/108.0/extensions/runtime/source/omni.physx/docs/api/python.html#omni.physx.bindings._physx.ContactEventType
                match contact_header.type:
                    case contact_header.type.CONTACT_LOST:
                        contact = Contact(
                            body0=entity0,
                            body1=entity1,
                        )
                    case contact_header.type.CONTACT_FOUND | contact_header.type.CONTACT_PERSIST:
                        # TODO 
                        # actor_pair = (pxr.PhysicsSchemaTools.intToSdfPath(contact_header.actor0), pxr.PhysicsSchemaTools.intToSdfPath(contact_header.actor1))
                        # collider_pair = (pxr.PhysicsSchemaTools.intToSdfPath(contact_header.collider0), pxr.PhysicsSchemaTools.intToSdfPath(contact_header.collider1))


                        contact_data_offset = contact_header.contact_data_offset
                        num_contact_data = contact_header.num_contact_data

                        positions = []
                        impulses = []
                        normals = []
                        separations = []

                        for i in range(contact_data_offset, contact_data_offset + num_contact_data):
                            contact_data = contact_datas[i]
                            # TODO ref https://docs.omniverse.nvidia.com/kit/docs/omni_physics/108.0/extensions/runtime/source/omni.physx/docs/api/python.html#omni.physx.bindings._physx.ContactData
                            # TODO this belongs to the header???
                            # pxr.PhysicsSchemaTools.intToSdfPath(contact_data.material0), pxr.PhysicsSchemaTools.intToSdfPath(contact_data.material1)
                            # TODO only valid for mesh; necesito? positions should be enough?>??
                            # contact_data.face_index0, contact_data.face_index1
                            normals.append(contact_data.normal)
                            impulses.append(contact_data.impulse)
                            positions.append(contact_data.position)
                            separations.append(contact_data.separation)

                        contact_point = ContactPoint(
                            position=numpy.asarray(positions),
                            impulse=numpy.asarray(impulses),
                            normal=numpy.asarray(normals),
                            separation=numpy.asarray(separations),
                        )


                        impulses = []
                        positions = []

                        if friction_anchors is not None:
                            friction_anchors_offset = contact_header.friction_anchors_offset
                            num_friction_anchors_data = contact_header.num_friction_anchors_data

                            for i in range(friction_anchors_offset, friction_anchors_offset + num_friction_anchors_data):
                                friction_anchor = friction_anchors[i]
                                # TODO ref https://docs.omniverse.nvidia.com/kit/docs/omni_physics/108.0/extensions/runtime/source/omni.physx/docs/api/python.html#omni.physx.bindings._physx.FrictionAnchor
                                impulses.append(friction_anchor.impulse)
                                positions.append(friction_anchor.position)

                        contact_anchor = ContactAnchor(
                            position=numpy.asarray(positions),
                            impulse=numpy.asarray(impulses),
                        )

                        contact = Contact(
                            body0=entity0,
                            body1=entity1,
                            points=contact_point,
                            anchors=contact_anchor,
                        )
                    case _:
                        # TODO
                        continue

                result = callable(contact)
                if asyncio.iscoroutine(result):
                    asyncio.create_task(result)

        sub = (
            scene._omni_physx_simulation
            .subscribe_full_contact_report_events(
                contact_report_callback
            )
        )
        yield
        sub.unsubscribe()


class Collision:
    def __init__(self, body: "Body"):
        self._body = body

    @property
    def enabled(self):
        pxr = self._body._scene._kernel._pxr

        value = []

        for prim in self._body._usd_prim_ref():
            if not prim.HasAPI(pxr.UsdPhysics.CollisionAPI):
                value.append(False)
            else:
                api = pxr.UsdPhysics.CollisionAPI(prim)
                value.append(bool(
                    api.CreateCollisionEnabledAttr().Get()
                ))

        return numpy.asarray(value)

    @enabled.setter
    def enabled(self, value):
        pxr = self._body._scene._kernel._pxr

        prims = self._body._usd_prim_ref()

        for prim, v in zip(
            prims,
            numpy.broadcast_to(value, shape=len(prims)),
        ):
            if not prim.HasAPI(pxr.UsdPhysics.CollisionAPI):
                api = pxr.UsdPhysics.CollisionAPI.Apply(prim)
            else:
                api = pxr.UsdPhysics.CollisionAPI(prim)
            api.CreateCollisionEnabledAttr().Set(bool(v))

    @functools.cached_property
    def on_contact(self):
        # TODO
        return ContactAsyncEventStream(self._body)



# TODO
from robotodo.utils.geometry import export_trimesh

# TODO
class BodyViewer:
    def __init__(self, body: Body):
        self._body = body

    def show(self):
        # TODO
        import itertools

        import trimesh
        import trimesh.viewer

        return (
            trimesh.Scene([
                export_trimesh(g)
                for g in itertools.chain(*self._body.geometry)
            ])
            .show()
        )

