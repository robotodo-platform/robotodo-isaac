import asyncio
import contextlib
import dataclasses
import functools
import warnings

import numpy
import torch

from robotodo.utils.pose import Pose
from robotodo.utils.geometry import PolygonMesh
from robotodo.utils.event import BaseSubscriptionPartialAsyncEventStream
from robotodo.engines.core.entity import EntityBodyKind
from robotodo.engines.core.path import PathExpression, PathExpressionLike

from robotodo.engines.isaac.scene import Scene
from robotodo.engines.isaac.material import Material
# from robotodo.engines.isaac._utils import USDPrimHelper
# TODO
from robotodo.engines.isaac._utils_next import USDPrimRef, USDPrimPathExpressionRef, USDPrimHelper



# TODO TensorTable
@dataclasses.dataclass
class ContactPoint:
    """
    TODO doc
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
    TODO doc
    """

    position: ...
    """The position of the contact friction anchor in world frame."""
    impulse: ...
    """TODO The impulse applied."""

@dataclasses.dataclass
class Contact:
    """
    TODO doc

    """

    entity0: "Entity"
    """TODO doc"""
    entity1: "Entity"
    """TODO doc"""
    points: ContactPoint | None = None
    """Contact points. `None` to indicate contact loss."""
    anchors: ContactAnchor | None = None
    """Contact friction anchors. `None` to indicate contact loss."""


class EntityContactAsyncEventStream(
    BaseSubscriptionPartialAsyncEventStream[Contact]
):
    def __init__(self, entity: "Entity"):
        self._entity = entity

    # TODO
    # def _isaac_physx_contact_report_callback(
    #     self,
    #     contact_headers: list, 
    #     contact_datas: list, 
    #     friction_anchors: list | None = None,
    # ):
    #     pass

    @contextlib.contextmanager
    def subscribe(self, callable):
        scene = self._entity._scene
        pxr = scene._kernel.pxr
        
        # TODO
        scene._kernel.enable_extension("omni.usd.schema.physx")

        for prim in self._entity._usd_prims:
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

            for contact_header in contact_headers:
                path_entity0 = pxr.PhysicsSchemaTools.intToSdfPath(contact_header.collider0)
                path_entity1 = pxr.PhysicsSchemaTools.intToSdfPath(contact_header.collider1)

                # TODO FIXME perf
                # TODO match ancesters as well?
                if any(
                    str(path) not in self._entity._usd_prim_helper.prim_paths
                    for path in (path_entity0, path_entity1)
                ):
                    continue

                entity0 = Entity(path_entity0, scene=scene)
                entity1 = Entity(path_entity1, scene=scene)

                # TODO https://docs.omniverse.nvidia.com/kit/docs/omni_physics/108.0/extensions/runtime/source/omni.physx/docs/api/python.html#omni.physx.bindings._physx.ContactEventType
                match contact_header.type:
                    case contact_header.type.CONTACT_LOST:
                        contact = Contact(
                            entity0=entity0,
                            entity1=entity1,
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
                            entity0=entity0,
                            entity1=entity1,
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
            scene._isaac_physx_simulation
            .subscribe_full_contact_report_events(
                contact_report_callback
            )
        )
        yield
        sub.unsubscribe()


class EntityCollision:
    def __init__(self, entity: "Entity"):
        self._entity = entity

    @property
    def enabled(self):
        pxr = self._entity._scene._kernel.pxr

        value = []

        for prim in self._entity._usd_prims:
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
        pxr = self._entity._scene._kernel.pxr

        prims = self._entity._usd_prims

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
        return EntityContactAsyncEventStream(self._entity)


# TODO deprecate
class EntityRigidBody:
    def __init__(self, entity: "Entity"):
        self._entity = entity
    
    @functools.cached_property
    def _isaac_physics_rigid_body_view_cache(self):
        try:
            paths = self._entity._usd_prim_helper.prim_paths
            self._entity._scene._isaac_physx_simulation.flush_changes()
            isaac_physics_tensor_view = self._entity._scene._isaac_physics_tensor_view
            rigid_body_view = (
                isaac_physics_tensor_view
                .create_rigid_body_view(paths)
            )
            assert rigid_body_view is not None
            assert rigid_body_view.check()
            def should_invalidate():
                return not (
                    isaac_physics_tensor_view.is_valid
                    and rigid_body_view.check()
                )
        except Exception as error:
            raise RuntimeError(
                f"Failed to create rigid body physics view from "
                f"resolved USD paths (are they valid?): {paths}"
            ) from error
        return rigid_body_view, should_invalidate
    
    @property
    def _isaac_physics_rigid_body_view(self):
        while True:
            rigid_body_view, should_invalidate = self._isaac_physics_rigid_body_view_cache
            if should_invalidate():
                del self._isaac_physics_rigid_body_view_cache
            else:
                return rigid_body_view

    @property
    def enabled(self):
        pxr = self._entity._scene._kernel.pxr

        prims = self._entity._usd_prims

        # TODO use numpy.empty for performance??
        value = []
        for prim in prims:
            if not prim.HasAPI(pxr.UsdPhysics.RigidBodyAPI):
                value.append(False)
            else:
                value.append(bool(
                    pxr.UsdPhysics.RigidBodyAPI(prim)
                    .CreateRigidBodyEnabledAttr()
                    .Get()
                ))

        return numpy.asarray(value)

    @enabled.setter
    def enabled(self, value: bool):
        pxr = self._entity._scene._kernel.pxr

        prims = self._entity._usd_prims

        for prim, v in zip(
            prims,
            numpy.broadcast_to(value, shape=len(prims)),
        ):
            if not prim.HasAPI(pxr.UsdPhysics.RigidBodyAPI):
                api = pxr.UsdPhysics.RigidBodyAPI.Apply(prim)
            else:
                api = pxr.UsdPhysics.RigidBodyAPI(prim)
            api.CreateRigidBodyEnabledAttr().Set(bool(v))
        
    # TODO is this relevant?? https://docs.omniverse.nvidia.com/kit/docs/omni_physics/108.1/dev_guide/rigid_bodies_articulations/rigid_bodies.html#automatic-computation-of-rigid-body-mass-distribution
    @property
    def mass(self):
        return self._isaac_physics_rigid_body_view.get_masses()
    
    # TODO BUG upstream: physics tensor api: changes not written to usd until .step
    @mass.setter
    def mass(self, value):
        view = self._isaac_physics_rigid_body_view
        value_ = torch.broadcast_to(torch.asarray(value), (view.count, 1))
        view.set_masses(value_, indices=torch.arange(view.count))

    @property
    def mass_center_pose(self):
        coms = self._isaac_physics_rigid_body_view.get_coms()

        return Pose(
            p=coms[..., [0, 1, 2]],
            q=coms[..., [3, 4, 5, 6]],
        )
    
    # TODO BUG upstream: physics tensor api: changes not written to usd until .step
    @mass_center_pose.setter
    def mass_center_pose(self, value: Pose):
        view = self._isaac_physics_rigid_body_view
        value_ = torch.broadcast_to(
            torch.concat((torch.asarray(value.p), torch.asarray(value.q))),
            size=(view.count, 7),
        )
        view.set_coms(value_, indices=torch.arange(view.count))


# TODO
# TODO https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/deformables_beta/physx_deformable_schema.html

# TODO deprecate
class EntityDeformableBody:
    def __init__(self, entity: "Entity"):
        self._entity = entity
        raise NotImplementedError


# TODO
class Entity:
    # TODO
    @classmethod
    def _from_usd_prim_ref(cls, ref: USDPrimRef, scene: Scene, _label: ... = None):
        self = cls.__new__(cls)
        self._label = _label
        self._scene = scene
        self._usd_prim_helper = USDPrimHelper(
            ref=ref,
            kernel=scene._kernel,
        )
        return self

    def __init__(self, path: PathExpressionLike, scene: Scene, _label: ... = None):
        self._label = _label
        self._scene = scene
        # TODO
        # self._usd_prim_ref = ...
        self._usd_prim_helper = USDPrimHelper(
            ref=USDPrimPathExpressionRef(path, stage_ref=lambda: scene._usd_stage),
            kernel=scene._kernel,
        )

    @property
    def label(self):
        return self._label

    # TODO rm
    # def __init__(self, path: PathExpressionLike, scene: Scene, _usd_prims_ref: ... = None):
    #     if _usd_prims_ref is not None:
    #         # TODO
    #         raise NotImplementedError
    #     self._scene = scene
    #     self._path = PathExpression(path)
    #     self._usd_prims_ref = _usd_prims_ref
    #
    # TODO rm
    # @functools.cached_property
    # def _usd_prim_helper(self):
    #     # TODO
    #     return USDPrimHelper(
    #         path=self._path, 
    #         scene=self._scene, 
    #         _usd_prims_ref=self._usd_prims_ref,
    #     )

    # TODO
    # def __repr__(self):
    #     # TODO
    #     self._usd_prim_helper._prims_ref
    #     return f"{Entity.__qualname__}({str(self._path)!r}, scene={self._scene!r})"

    @functools.cached_property
    def viewer(self):
        # TODO mv here !!!
        from .viewer import EntityViewer
        return EntityViewer(self)
    
    # TODO
    @property
    def _usd_prims_ref(self):
        return self._usd_prim_helper._prims_ref

    @property
    def _usd_prims(self):
        return self._usd_prim_helper.prims
    
    # TODO
    @property
    def path(self):
        return self._usd_prim_helper.prim_paths
    
    @property
    def pose(self):
        return self._usd_prim_helper.pose
    
    @pose.setter
    def pose(self, value: Pose):
        self._usd_prim_helper.pose = value

    @property
    def pose_in_parent(self):
        return self._usd_prim_helper.pose_in_parent
    
    @pose_in_parent.setter
    def pose_in_parent(self, value: Pose):
        self._usd_prim_helper.pose_in_parent = value

    # TODO not all children are included???
    # TODO scaling
    # TODO optimize: instanceable assets may have shared geoms
    @property
    def geometry(self):
        # TODO
        pxr = self._scene._kernel.pxr

        # TODO geom TensorView("n? geom")
        # -or- {Mesh: <Mesh>}??? -or- <collection>.find_by_type(Mesh)???

        geoms = []

        for prim in self._usd_prim_helper.prims:
            prim_geoms = []

            # TODO apply world xform as well????
            prim_scale_factors = (
                pxr.Gf.Transform(
                    self._usd_prim_helper._xform_cache(prim.GetStage())
                    .GetLocalToWorldTransform(prim)
                )
                .GetScale()
            )

            # NOTE the first child_prim would be the prim itself
            for child_prim in pxr.Usd.PrimRange(
                prim, 
                pxr.Usd.TraverseInstanceProxies(
                    pxr.Usd.PrimAllPrimsPredicate
                ),
            ):
                # TODO restrict to collision only?
                # if not child_prim.HasAPI(pxr.UsdPhysics.CollisionAPI):
                #     continue

                match child_prim:
                    case _ if child_prim.IsA(pxr.UsdGeom.Plane):
                        pass
                    case _ if child_prim.IsA(pxr.UsdGeom.Mesh):
                        api = pxr.UsdGeom.Mesh(child_prim)
                        # TODO lazy??
                        prim_geoms.append(
                            PolygonMesh(
                                vertices=numpy.asarray(api.GetPointsAttr().Get()) * prim_scale_factors,
                                face_vertex_counts=numpy.asarray(api.GetFaceVertexCountsAttr().Get()),
                                face_vertex_indices=numpy.asarray(api.GetFaceVertexIndicesAttr().Get()),
                            )
                        )
                    case _ if child_prim.IsA(pxr.UsdGeom.Cube):
                        pass
                    case _ if child_prim.IsA(pxr.UsdGeom.Sphere):
                        pass
                    case _:
                        # TODO
                        pass

            geoms.append(prim_geoms)

        return geoms
    
    # TODO
    @functools.cached_property
    def material(self):
        pxr = self._scene._kernel.pxr

        def _material_prims_ref(scene: Scene):
            prims = self._usd_prims
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
        return EntityCollision(self)

    @property
    def body_kind(self):
        pxr = self._scene._kernel.pxr

        res = []

        for prim in self._usd_prims:
            v = EntityBodyKind.NONE

            for schema in prim.GetAppliedSchemas():
                match schema:
                    case "PhysicsRigidBodyAPI":
                        if (
                            pxr.UsdPhysics.RigidBodyAPI(prim)
                            .GetRigidBodyEnabledAttr()
                            .Get()
                        ):
                            v = EntityBodyKind.RIGID
                    case "OmniPhysicsVolumeDeformableSimAPI":
                        if (
                            prim
                            .GetAttribute("omniphysics:deformableBodyEnabled")
                            .Get()
                        ):
                            v = EntityBodyKind.DEFORMABLE_VOLUME
                    case "OmniPhysicsSurfaceDeformableSimAPI":
                        if (
                            prim
                            .GetAttribute("omniphysics:deformableBodyEnabled")
                            .Get()
                        ):
                            v = EntityBodyKind.DEFORMABLE_SURFACE

            res.append(v.value)

        return numpy.asarray(res)


    # TODO deprecate
    @functools.cached_property
    def rigid_body(self):
        return EntityRigidBody(self)
    @functools.cached_property
    def soft_body(self):
        return EntityDeformableBody(self)
    # TODO deprecate
