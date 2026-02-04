# SPDX-License-Identifier: Apache-2.0

"""
Articulation.
"""


import functools
import warnings
import dataclasses
from typing import Type, Unpack

# TODO
import torch
import numpy

from robotodo.utils.pose import Pose
from robotodo.engines.core.path import (
    PathExpression, 
    PathExpressionLike, 
    is_path_expression_like,
)
from robotodo.engines.core.error import InvalidReferenceError
from robotodo.engines.core.entity import ProtoEntity
from robotodo.engines.core.articulation import (
    Axis,
    JointKind, 
    ProtoJoint, 
    ProtoFixedJoint,
    ProtoRevoluteJoint,
    ProtoPrismaticJoint,
    ProtoSphericalJoint,
    ProtoArticulation,
)
from robotodo.engines.isaac.kernel import Kernel
from robotodo.engines.isaac.body import Body
from robotodo.engines.isaac.entity import Entity
from robotodo.engines.isaac.scene import Scene
# TODO
from robotodo.engines.isaac._utils.usd import (
    USDPrimRef, 
    is_usd_prim_ref,
    USDPrimPathRef, 
    USDPrimPathExpressionRef,
    usd_physx_query_articulation_properties,
    usd_import_urdf,
    USDXformView,
)


class Joint(ProtoJoint):
    # __slots__ = ["_usd_prim_ref", "_scene"]

    _usd_prim_ref: USDPrimRef
    _scene: Scene

    _impl_label: str | None = None
    _impl_label_body0: str | None = None
    _impl_label_body1: str | None = None

    # TODO
    @classmethod
    def create(cls):
        raise NotImplementedError("TODO")
    
    @classmethod
    def load(cls, ref: PathExpressionLike, source: str, scene: Scene):
        # TODO convert? check?
        return cls(Entity.load(ref, source=source, scene=scene))

    # TODO use _kernel
    def __init__(
        self, 
        ref: "Joint | Entity | USDPrimRef | PathExpressionLike", 
        scene: Scene | None = None,
    ):
        # TODO
        match ref:
            # TODO merge with Entity as ProtoEntity
            case Joint() as joint:
                assert scene is None
                self._usd_prim_ref = joint._usd_prim_ref
                self._scene = joint._scene
            case Entity() as entity:
                # TODO
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
    @property
    def prototypes(self):
        raise NotImplementedError

    # TODO !!!! 
    def astype(self, prototype):
        match prototype:
            case _ if issubclass(prototype, ProtoFixedJoint):
                return FixedJoint(self)
            case _ if issubclass(prototype, ProtoRevoluteJoint):
                return RevoluteJoint(self)
            case _ if issubclass(prototype, ProtoPrismaticJoint):
                return PrismaticJoint(self)
            case _ if issubclass(prototype, ProtoSphericalJoint):
                return SphericalJoint(self)
            case _ if issubclass(prototype, ProtoEntity):
                # TODO
                raise NotImplementedError("TODO")        
            case _:
                raise ValueError("TODO")
    
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
        if self._impl_label is not None:
            return self._impl_label
        return super().label
    
    @property
    def kind(self):
        # TODO
        pxr = self._scene._kernel._pxr
        value = []
        for prim in self._usd_prim_ref():
            v = JointKind.UNKNOWN
            match prim:
                case _ if prim.IsA(pxr.UsdPhysics.FixedJoint):
                    v = JointKind.FIXED
                case _ if prim.IsA(pxr.UsdPhysics.PrismaticJoint):
                    v = JointKind.PRISMATIC
                case _ if prim.IsA(pxr.UsdPhysics.RevoluteJoint):
                    v = JointKind.REVOLUTE
                case _ if prim.IsA(pxr.UsdPhysics.SphericalJoint):
                    v = JointKind.SPHERICAL
                case _:
                    # TODO
                    warnings.warn(f"USD joint currently not supported: {prim}")
            value.append(v)
        return numpy.asarray(value)

    @property
    def body0(self):
        # TODO
        pxr = self._scene._kernel._pxr

        body_prim_paths = []

        for prim in self._usd_prim_ref():
            api = pxr.UsdPhysics.Joint(prim)
            targets = api.GetBody0Rel().GetTargets()
            # TODO
            [target] = targets
            body_prim_paths.append(target.pathString)

        # TODO
        body = Body(
            USDPrimPathRef(
                paths=body_prim_paths,
                stage_ref=lambda: self._scene._usd_stage,
            ),
            scene=self._scene,
        )
        body._impl_label = self._impl_label_body0

        return body

    @property
    def pose_in_body0(self):
        pxr = self._scene._kernel._pxr

        positions = []
        rotations = []

        for prim in self._usd_prim_ref():
            api = pxr.UsdPhysics.Joint(prim)
            pos_vec3 = api.GetLocalPos0Attr().Get()
            rot_quat = api.GetLocalRot0Attr().Get()
            positions.append(pos_vec3)
            # NOTE wxyz
            rotations.append((*rot_quat.GetImaginary(), rot_quat.GetReal()))

        return Pose(
            p=numpy.asarray(positions),
            q=numpy.asarray(rotations),
        )
    
    @property
    def body1(self):
        pxr = self._scene._kernel._pxr

        body_prim_paths = []

        for prim in self._usd_prim_ref():
            api = pxr.UsdPhysics.Joint(prim)
            targets = api.GetBody1Rel().GetTargets()
            # TODO
            [target] = targets
            body_prim_paths.append(target.pathString)

        # TODO
        body = Body(
            USDPrimPathRef(
                paths=body_prim_paths,
                stage_ref=lambda: self._scene._usd_stage,
            ),
            scene=self._scene,
        )
        body._impl_label = self._impl_label_body1

        return body

    @property
    def pose_in_body1(self):
        pxr = self._scene._kernel._pxr

        positions = []
        rotations = []

        for prim in self._usd_prim_ref():
            api = pxr.UsdPhysics.Joint(prim)
            pos_vec3 = api.GetLocalPos1Attr().Get()
            rot_quat = api.GetLocalRot1Attr().Get()
            positions.append(pos_vec3)
            # NOTE wxyz
            rotations.append((*rot_quat.GetImaginary(), rot_quat.GetReal()))

        return Pose(
            p=numpy.asarray(positions),
            q=numpy.asarray(rotations),
        )


# TODO
class FixedJoint(Joint, ProtoFixedJoint):
    pass


class RevoluteJoint(Joint, ProtoRevoluteJoint):
    @property
    def axis(self):
        pxr = self._scene._kernel._pxr

        value = []
        for prim in self._usd_prim_ref():
            api = pxr.UsdPhysics.RevoluteJoint(prim)
            v = Axis.UNKNOWN
            if not api:
                warnings.warn(f"TODO: {prim}")
            else:
                match api.GetAxisAttr().Get():
                    case "X":
                        v = Axis.X
                    case "Y":
                        v = Axis.Y
                    case "Z":
                        v = Axis.Z
            value.append(v)

        return numpy.asarray(value)
    
    @property
    def position_limit(self):
        pxr = self._scene._kernel._pxr

        value = []
        for prim in self._usd_prim_ref():
            api = pxr.UsdPhysics.RevoluteJoint(prim)
            v = [numpy.nan, numpy.nan]
            if not api:
                warnings.warn(f"TODO: {prim}")
            else:
                v = [api.GetLowerLimitAttr().Get(), api.GetUpperLimitAttr().Get()]
            value.append(v)

        # TODO
        return numpy.deg2rad(value)


# TODO
class PrismaticJoint(Joint, ProtoPrismaticJoint):
    @property
    def axis(self):
        pxr = self._scene._kernel._pxr

        value = []
        for prim in self._usd_prim_ref():
            api = pxr.UsdPhysics.PrismaticJoint(prim)
            v = Axis.UNKNOWN
            if not api:
                warnings.warn(f"TODO: {prim}")
            else:
                match api.GetAxisAttr().Get():
                    case "X":
                        v = Axis.X
                    case "Y":
                        v = Axis.Y
                    case "Z":
                        v = Axis.Z
            value.append(v)

        return numpy.asarray(value)
    
    @property
    def position_limit(self):
        pxr = self._scene._kernel._pxr

        value = []
        for prim in self._usd_prim_ref():
            api = pxr.UsdPhysics.PrismaticJoint(prim)
            v = [numpy.nan, numpy.nan]
            if not api:
                warnings.warn(f"TODO: {prim}")
            else:
                v = [api.GetLowerLimitAttr().Get(), api.GetUpperLimitAttr().Get()]
            value.append(v)

        return numpy.asarray(value)
    

# TODO
class SphericalJoint(Joint, ProtoSphericalJoint):
    pass


# TODO NOTE must be homogenous
# TODO FIXME write operations may not sync to the USD stage unless .step called
class Articulation(ProtoArticulation):
    # TODO
    _usd_prim_ref: USDPrimRef
    _scene: Scene

    # TODO
    @classmethod
    def create(cls, ref: PathExpressionLike, source: str, scene: Scene):
        raise NotImplementedError

    @classmethod
    def load(cls, ref: PathExpressionLike, source: str, scene: Scene):
        # TODO
        import pathlib
        import urllib.parse

        # TODO use pxr.sdf? 
        # TODO upstream: util to standardize url handling?
        match pathlib.Path(urllib.parse.urlparse(source).path).suffixes:
            case [".urdf"]:
                expr = PathExpression(ref)
                prims = usd_import_urdf(
                    stage=scene._usd_stage,
                    paths=expr.expand(),
                    resource_or_model=source,
                    kernel=scene._kernel,
                )
                return cls(lambda: prims, scene=scene)
    
            # NOTE other formats are handled through USD
            case _:
                return cls(Entity.load(ref, source=source, scene=scene))

    def __init__(
        self,
        ref: "Articulation | Entity | PathExpressionLike | USDPrimRef",
        scene: Scene | None = None,
    ):
        match ref:
            case Articulation() as articulation:
                assert scene is None
                self._usd_prim_ref = articulation._usd_prim_ref
                self._scene = articulation._scene
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

    @property
    def _usd_articulation_root_prims(self):
        pxr = self._scene._kernel._pxr

        return [
            maybe_root_prim
            for prim in self._usd_prim_ref()
            # NOTE this already includes the prim itself
            for maybe_root_prim in pxr.Usd.PrimRange(
                prim, 
                pxr.Usd.TraverseInstanceProxies(
                    pxr.Usd.PrimAllPrimsPredicate
                ),
            )
            if maybe_root_prim.HasAPI(pxr.UsdPhysics.ArticulationRootAPI)
        ]
            
    @functools.cached_property
    def _isaac_physics_articulation_view_cache(self):
        # TODO
        resolved_root_paths = None
        try:
            resolved_root_paths = [
                prim.GetPath().pathString
                for prim in self._usd_articulation_root_prims
            ]
            self._scene._omni_physx_simulation.flush_changes()
            isaac_physics_tensor_view = self._scene._omni_physics_tensor_view
            articulation_view = (
                isaac_physics_tensor_view
                .create_articulation_view(resolved_root_paths)
            )
            assert articulation_view is not None
            assert articulation_view.check()
            def should_invalidate():
                return not (
                    isaac_physics_tensor_view.is_valid
                    and articulation_view.check()
                )
        except Exception as error:
            raise RuntimeError(
                f"Failed to create articulation physics view from "
                f"resolved USD prim paths (are they valid?): "
                f"{resolved_root_paths}"
            ) from error
        return articulation_view, should_invalidate

    @property
    def _isaac_physics_articulation_view(self):
        while True:
            articulation_view, should_invalidate = self._isaac_physics_articulation_view_cache
            if should_invalidate():
                del self._isaac_physics_articulation_view_cache
            else:
                return articulation_view
            
    @property
    def _isaac_physx_articulation_properties(self):
        return usd_physx_query_articulation_properties(
            self._usd_articulation_root_prims,
            kernel=self._scene._kernel,
        )

    # TODO
    @functools.cached_property
    def _usd_xform_view(self):
        return USDXformView(
            self._usd_prim_ref, 
            kernel=self._scene._kernel,
        )

    @property
    def path(self):
        return [
            prim.GetPath().pathString
            if prim else
            None
            for prim in self._usd_prim_ref()
        ]

    # TODO rm
    # TODO clarify UsdPhysicsArticulationRootAPI
    # @property
    # def path(self):
    #     return self._isaac_physics_articulation_view.prim_paths

    @property
    def scene(self):
        return self._scene

    # # TODO cache
    # # @functools.cached_property
    # @property
    # def joints(self):    
    #     # TODO
    #     if not self._isaac_physics_articulation_view.is_homogeneous:
    #         raise NotImplementedError("TODO")

    #     joint_names = self._isaac_physics_articulation_view.shared_metatype.joint_names
    #     # TODO wrong !!!! doesnt contain fixed joints
    #     joint_paths = numpy.asarray(self._isaac_physics_articulation_view.dof_paths)[
    #         ...,
    #         self._isaac_physics_articulation_view.shared_metatype.joint_dof_offsets
    #     ]

    #     *_, n_joints = joint_paths.shape
    #     return {
    #         joint_names[joint_index]: Joint(
    #             joint_paths[..., joint_index], 
    #             scene=self._scene,
    #         )
    #         for joint_index in range(n_joints)
    #     }

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

    # @property
    # def pose(self):
    #     view = self._isaac_physics_articulation_view
    #     root_trans = view.get_root_transforms()
    #     return Pose(
    #         p=root_trans[..., [0, 1, 2]],
    #         q=root_trans[..., [3, 4, 5, 6]],
    #     )
    
    # @pose.setter
    # def pose(self, value: Pose):
    #     view = self._isaac_physics_articulation_view
    #     value_ = torch.broadcast_to(
    #         torch.concat((torch.asarray(value.p), torch.asarray(value.q)), dim=-1), 
    #         size=(view.count, 7),
    #     )
    #     view.set_root_transforms(value_, indices=torch.arange(view.count))
    #     # TODO
    #     # self._scene._isaac_physics_tensor_ensure_sync()

    @property
    def joints(self):
        pxr = self._scene._kernel._pxr

        joint_names = self._isaac_physics_articulation_view.shared_metatype.joint_names
        # NOTE shift by one since the world joint is not included
        body0_names = self._isaac_physics_articulation_view.shared_metatype.link_parents[1:]
        body1_names = self._isaac_physics_articulation_view.shared_metatype.link_names[1:]

        articulation_properties = self._isaac_physx_articulation_properties

        joint_name_path_mapping = {
            joint_name: numpy.full(len(articulation_properties), fill_value=None, dtype=object)
            for joint_name in joint_names
        }

        for articulation_index, articulation_prop in enumerate(articulation_properties):
            for link in articulation_prop.links:
                joint_path = link.joint_name
                if not joint_path:
                    # NOTE world link, skip
                    continue
                joint_name = pxr.Sdf.Path(joint_path).name

                if joint_name_path_mapping[joint_name][articulation_index] is not None:
                    # TODO better msg
                    raise RuntimeError(
                        f"A joint with name {joint_name} already exists "
                        f"in articulation {self.path[articulation_index]}, this is currently unsupported. "
                        f"Existing and to-be-assigned paths are: "
                        f"{joint_path}, {joint_name_path_mapping[joint_name][articulation_index]}"
                    )
                joint_name_path_mapping[joint_name][articulation_index] = joint_path

        joints = dict[str, Joint]()
        for joint_index, joint_name in enumerate(joint_names):
            joint = Joint(
                joint_name_path_mapping[joint_name],
                scene=self._scene,
            )
            joint._impl_label = joint_name
            joint._impl_label_body0 = body0_names[joint_index]
            joint._impl_label_body1 = body1_names[joint_index]
            joints[joint_name] = joint
        return joints

    # TODO
    @property
    def links(self):
        # TODO
        if not self._isaac_physics_articulation_view.is_homogeneous:
            raise NotImplementedError("TODO")
        
        link_names = self._isaac_physics_articulation_view.shared_metatype.link_names
        link_paths = numpy.asarray(self._isaac_physics_articulation_view.link_paths)

        *_, n_links = link_paths.shape
        return {
            link_names[link_index]: Body(
                link_paths[..., link_index],
                scene=self._scene,
            )
            for link_index in range(n_links)
        }

    @property
    def dof_count(self):
        return self._isaac_physics_articulation_view.shared_metatype.dof_count

    @property
    def dof_names(self):
        # TODO warn !!!
        if not self._isaac_physics_articulation_view.is_homogeneous:
            pass
        return self._isaac_physics_articulation_view.shared_metatype.dof_names

    # TODO
    @property
    def dof_kinds(self):
        view = self._isaac_physics_articulation_view
        return view.get_dof_types()

    @property
    def dof_positions(self):
        view = self._isaac_physics_articulation_view
        dof_positions = view.get_dof_positions()
        return dof_positions

    # TODO maybe async???
    # TODO typing
    # TODO support masking !!!!!!
    @dof_positions.setter
    def dof_positions(self, value):
        # TODO doc: rotation convention https://github.com/NVIDIAGameWorks/PhysX/issues/126#issuecomment-503007721
        view = self._isaac_physics_articulation_view
        device = self._scene._omni_physics_tensor_view.device
        value_ = torch.asarray(value, device=device)
        value_ = torch.broadcast_to(value_, (view.count, view.shared_metatype.dof_count))
        indices = torch.arange(view.count, device=device)
        view.set_dof_positions(value_, indices=indices)
        # TODO FIXME: perf .change_block to defer result fetching?
        # self._scene._isaac_physics_tensor_ensure_sync()

    @property
    def dof_position_limits(self):
        view = self._isaac_physics_articulation_view
        # TODO
        return view.get_dof_limits()

    @property
    def dof_velocities(self):
        view = self._isaac_physics_articulation_view
        velocities = view.get_dof_velocities()
        return velocities

    @dof_velocities.setter
    def dof_velocities(self, value):
        view = self._isaac_physics_articulation_view
        device = self._scene._omni_physics_tensor_view.device
        value_ = torch.asarray(value, device=device)
        value_ = torch.broadcast_to(value_, (view.count, view.shared_metatype.dof_count))
        indices = torch.arange(view.count, device=device)
        view.set_dof_velocities(value_, indices=indices)
        # self._scene._isaac_physics_tensor_ensure_sync()

    # TODO dof_velocity_limits
    # view.get_dof_max_velocities

    @functools.cached_property
    def driver(self):
        return ArticulationDriver(articulation=self)
    
    # @functools.lru_cache
    def planner(self, **planner_kwds: Unpack["ArticulationPlanner.Config"]):
        return ArticulationPlanner(self, **planner_kwds)


# TODO
import functools
from typing import TypedDict, Unpack

from tensorspecs import TensorLike, TensorTableLike, TensorSpec, TensorTableSpec


# TODO mv
class ArticulationAction(TypedDict, total=False):
    """
    TODO doc articulation action protocol type
    """
    # TODO -or- namedtensor??
    dof_names: TensorLike["dof"]
    # TODO waypoint optional?
    dof_positions: TensorLike["n? timestep dof"]
    dof_velocities: TensorLike["n? timestep dof"]


class ArticulationDriver:
    def __init__(self, articulation: Articulation):
        self._articulation = articulation

    def compute_dof_passive_gravity_forces(self):
        view = self._articulation._isaac_physics_articulation_view
        # TODO extra +6 dofs when floating base !!!!!!
        # TODO indices: [*base_6dof, *dof]
        return view.get_gravity_compensation_forces()

    def compute_dof_passive_coriolis_and_centrifugal_forces(self):
        view = self._articulation._isaac_physics_articulation_view
        # TODO extra +6 dofs when floating base !!!!!!
        return view.get_coriolis_and_centrifugal_compensation_forces()

    @property
    def dof_stiffnesses(self):
        view = self._articulation._isaac_physics_articulation_view
        return view.get_dof_stiffnesses()

    @dof_stiffnesses.setter
    def dof_stiffnesses(self, value):
        view = self._articulation._isaac_physics_articulation_view
        device = self._articulation._scene._omni_physics_tensor_view.device
        value_ = torch.asarray(value, device=device)
        value_ = torch.broadcast_to(value_, (view.count, view.max_dofs))
        indices = torch.arange(view.count, device=device)
        view.set_dof_stiffnesses(value_, indices=indices)

    @property
    def dof_dampings(self):
        view = self._articulation._isaac_physics_articulation_view
        return view.get_dof_dampings()
    
    @dof_dampings.setter
    def dof_dampings(self, value):
        view = self._articulation._isaac_physics_articulation_view
        device = self._articulation._scene._omni_physics_tensor_view.device
        value_ = torch.asarray(value, device=device)
        value_ = torch.broadcast_to(value_, (view.count, view.max_dofs))
        indices = torch.arange(view.count, device=device)
        view.set_dof_dampings(value_, indices=indices)
    
    # TODO
    @property
    def dof_max_forces(self):
        view = self._articulation._isaac_physics_articulation_view
        return view.get_dof_max_forces()
    
    # TODO
    @property
    def dof_forces(self):
        view = self._articulation._isaac_physics_articulation_view
        return view.get_dof_actuation_forces()
    
    @dof_forces.setter
    def dof_forces(self, value):
        view = self._articulation._isaac_physics_articulation_view
        device = self._articulation._scene._omni_physics_tensor_view.device
        value_ = torch.asarray(value, device=device)
        value_ = torch.broadcast_to(value_, (view.count, view.max_dofs))
        indices = torch.arange(view.count, device=device)
        view.set_dof_actuation_forces(value_, indices=indices)

    # TODO typing: DriveType
    @property
    def dof_drive_types(self):
        view = self._articulation._isaac_physics_articulation_view
        return view.get_drive_types()

    @property
    def dof_target_positions(self):
        view = self._articulation._isaac_physics_articulation_view
        return view.get_dof_position_targets()

    @dof_target_positions.setter
    def dof_target_positions(self, value):
        view = self._articulation._isaac_physics_articulation_view
        device = self._articulation._scene._omni_physics_tensor_view.device
        value_ = torch.asarray(value, device=device)
        value_ = torch.broadcast_to(value_, (view.count, view.max_dofs))
        indices = torch.arange(view.count, device=device)
        view.set_dof_position_targets(value_, indices=indices)

    @property
    def dof_target_velocities(self):
        view = self._articulation._isaac_physics_articulation_view
        return view.get_dof_velocity_targets()

    @dof_target_velocities.setter
    def dof_target_velocities(self, value):
        view = self._articulation._isaac_physics_articulation_view
        device = self._articulation._scene._omni_physics_tensor_view.device
        value_ = torch.asarray(value, device=device)
        value_ = torch.broadcast_to(value_, (view.count, view.max_dofs))
        indices = torch.arange(view.count, device=device)
        view.set_dof_velocity_targets(value_, indices=indices)

    # TODO
    async def execute_action(
        self, 
        # TODO
        action: "ArticulationAction",
        # TODO
        position_error_limit: float = 1e-1,
        velocity_error_limit: float = 1e-1,
        # TODO
        # iteration_callback: ... = None,
    ):
        dof_names = action.get("dof_names", None)
        if dof_names is not None:
            # TODO support native .index(<batch_dof_names>)
            dof_indices = [
                self._articulation.dof_names.index(dof_name)
                for dof_name in dof_names
            ]
            dof_count = len(dof_indices)
        else:
            dof_indices = numpy.s_[:]
            dof_count = self._articulation._isaac_physics_articulation_view.shared_metatype.dof_count

        dof_positions = action.get("dof_positions", None)
        if dof_positions is None:
            dof_positions = self._articulation.dof_positions        
        dof_velocities = action.get("dof_velocities", None)
        if dof_velocities is None:
            dof_velocities = self._articulation.dof_velocities
        
        dof_positions = torch.asarray(dof_positions)
        dof_velocities = torch.asarray(dof_velocities)
        dof_positions, dof_velocities = (
            torch.broadcast_tensors(dof_positions, dof_velocities)
        )

        shape = (
            self._articulation._isaac_physics_articulation_view.count,
            -1,
            dof_count,
        )
        dof_positions = torch.broadcast_to(dof_positions, size=shape)
        dof_velocities = torch.broadcast_to(dof_velocities, size=shape)

        _, n_timesteps, _ = torch.broadcast_shapes(
            dof_positions.shape, 
            dof_velocities.shape,
        )

        for timestep in range(n_timesteps):
            dof_target_positions = self.dof_target_positions            
            dof_target_positions[..., dof_indices] = torch.asarray(
                dof_positions[:, timestep, :], 
                device=dof_target_positions.device,
            )
            self.dof_target_positions = dof_target_positions

            dof_target_velocities = self.dof_target_velocities
            dof_target_velocities[..., dof_indices] = torch.asarray(
                dof_velocities[:, timestep, :],
                device=dof_target_positions.device,
            )
            self.dof_target_velocities = dof_target_velocities

            # TODO use dt for timeout
            async for _ in self._articulation._scene.on_step:
                position_err = (
                    self._articulation.dof_positions[..., dof_indices] 
                        - self.dof_target_positions[..., dof_indices]
                )
                velocity_err = (
                    self._articulation.dof_velocities[..., dof_indices] 
                        - self.dof_target_velocities[..., dof_indices]
                )

                # TODO
                # iteration_callback(position_err, velocity_err)

                if all([
                    torch.allclose(position_err, torch.asarray(0.), atol=position_error_limit),
                    torch.allclose(velocity_err, torch.asarray(0.), atol=velocity_error_limit),
                ]):
                    break


# TODO
import functools
from typing import TypedDict, Unpack

from tensorspecs import TensorTableLike, TensorSpec, TensorTableSpec
# TODO
# from robotodo.algos.motion_planning.reach import MotionPlanner


# TODO impl protoplanner
# TODO merge with robotodo.algos.motion_planning!!!
class ArticulationPlanner:
    class Config(TypedDict, total=False):
        base_link: str
        end_link: str
        # TODO
        use_self_collision: bool
        use_world_collision: bool

    # TODO
    def __init__(
        self, 
        articulation: Articulation, 
        config: Config | dict = dict(),
        **config_kwds: Unpack[Config],
    ):
        self._articulation = articulation
        self._config = ArticulationPlanner.Config(config, **config_kwds)

    # TODO
    @functools.cached_property
    def _backend(self):
        # TODO
        from robotodo.algos.motion_planning.reach import MotionPlanner

        return MotionPlanner(
            # TODO articulation.kinematics?
            list(self._articulation.joints.values()),
            config=MotionPlanner.Config({
                "base_link": self._config["base_link"],
                "end_link": self._config["end_link"],
                **self._config,
            }),
        )
    
    @property
    def observation_spec(self):
        n_dofs = len(self._backend.dof_names)
        # TODO named indexing
        return TensorTableSpec({
            # TODO also dof_names???
            "dof_positions": TensorSpec("n? dof", shape={"dof": n_dofs}),
            # "target_pose": PoseSpec("n?"),
            # "target_pose_from_base": PoseSpec("n?"),
            # "target_pose_candidates": PoseSpec("n? candidate"),
            # "target_pose_from_base_candidates": PoseSpec("n? candidate"),
            # "obstacles": ...,
        })
        
    @property
    def action_spec(self):
        n_dofs = len(self._backend.dof_names)
        return TensorTableSpec({
            # TODO -or- namedtensor??
            "dof_names": TensorSpec("dof", shape={"dof": n_dofs}),
            "dof_positions": TensorSpec("n? timestep dof", shape={"dof": n_dofs}),
            "dof_velocities": TensorSpec("n? timestep dof", shape={"dof": n_dofs}),
        })

    # TODO
    def compute_action(
        self, 
        observation: TensorTableLike[observation_spec],
    ) -> "TensorTableLike[action_spec] | ArticulationAction":
        
        dof_positions = observation.get("dof_positions", None)
        if dof_positions is None:
            # TODO cache index
            dof_positions = self._articulation.dof_positions[..., [
                self._articulation.dof_names.index(name)
                for name in self._backend.dof_names
            ]]

        target_pose_from_base = observation.get("target_pose_from_base", None)
        if target_pose_from_base is None:
            target_pose = observation.get("target_pose", None)
            if target_pose is None:
                # TODO msg
                raise ValueError("TODO")
            target_pose_from_base = (
                self._articulation.links[self._backend.base_link].pose.inv()
                * target_pose
            )
            
        return self._backend.compute_action({
            "dof_positions": dof_positions,
            "target_pose": target_pose_from_base,
        })

