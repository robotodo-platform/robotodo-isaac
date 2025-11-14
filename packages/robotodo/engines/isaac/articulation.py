
import enum
import functools
import warnings
import dataclasses

# TODO
import torch
import numpy

from robotodo.utils.pose import Pose
from robotodo.engines.core.path import (
    PathExpression, 
    PathExpressionLike, 
    is_path_expression_like,
)
from robotodo.engines.core.articulation import (
    Axis,
    JointKind, 
    ProtoJoint, 
    ProtoFixedJoint,
    ProtoRevoluteJoint,
    ProtoPrismaticJoint,
    ProtoSphericalJoint,
)
from robotodo.engines.isaac.entity import Entity
from robotodo.engines.isaac.scene import Scene
# TODO
from robotodo.engines.isaac._utils_next import (
    USDPrimRef, 
    is_usd_prim_ref,
    USDPrimPathRef, 
    USDPrimPathExpressionRef,
    usd_physx_query_articulation_properties,
)


class Joint(ProtoJoint):
    @dataclasses.dataclass(slots=True)
    class _ImplMetadata:
        label: str
        label_body0: str
        label_body1: str

    __slots__ = ["_usd_prims_ref", "_scene"]

    # TODO use _kernel
    def __init__(
        self, 
        ref: "Joint | Entity | PathExpressionLike | USDPrimRef", 
        scene: Scene | None = None,
        _impl_metadata: _ImplMetadata | None = None,
    ):
        # TODO
        match ref:
            case Joint() as joint:
                assert scene is None
                self._usd_prims_ref = joint._usd_prims_ref
                self._scene = joint._scene
                if _impl_metadata is None:
                    _impl_metadata = joint._impl_metadata
            case _ if is_usd_prim_ref(ref):
                # TODO
                assert scene is not None
                self._usd_prims_ref = ref
                self._scene = scene
            case Entity() as entity:
                # TODO
                assert scene is None
                self._usd_prims_ref = entity._usd_prims_ref
                self._scene = entity._scene
            case expr if is_path_expression_like(ref):
                # TODO
                assert scene is not None
                self._usd_prims_ref = USDPrimPathExpressionRef(
                    expr,
                    stage_ref=lambda: scene._usd_stage,
                )
                self._scene = scene
            case _:
                raise ValueError(f"Invalid reference type {type(ref)}: {ref}")
        self._impl_metadata = _impl_metadata

    @property
    def _usd_prims(self):
        # TODO
        return self._usd_prims_ref()
    
    @property
    def path(self):
        return [
            prim.GetPath().pathString
            for prim in self._usd_prims
        ]
    
    @property
    def label(self):
        if self._impl_metadata is not None:
            return self._impl_metadata.label
        return super().label
    
    @property
    def kind(self):
        # TODO
        pxr = self._scene._kernel.pxr
        value = []
        for prim in self._usd_prims:
            v = JointKind.UNKNOWN
            match prim:
                case _ if prim.IsA(pxr.UsdPhysics.FixedJoint):
                    v = JointKind.FIXED
                case _ if prim.IsA(pxr.UsdPhysics.PrismaticJoint):
                    v = JointKind.PRISMATIC
                case _ if prim.IsA(pxr.UsdPhysics.RevoluteJoint):
                    v = JointKind.REVOLUTE
                case _:
                    # TODO
                    warnings.warn(f"USD joint currently not supported: {prim}")
            value.append(v)
        return numpy.asarray(value)

    @property
    def body0(self):
        # TODO
        pxr = self._scene._kernel.pxr

        body_prim_paths = []

        for prim in self._usd_prims:
            api = pxr.UsdPhysics.Joint(prim)
            targets = api.GetBody0Rel().GetTargets()
            # TODO
            [target] = targets
            body_prim_paths.append(target.pathString)

        # TODO
        return Entity._from_usd_prim_ref(
            ref=USDPrimPathRef(
                paths=body_prim_paths,
                stage_ref=lambda: self._scene._usd_stage,
            ),
            scene=self._scene,
            # TODO
            _label=self._impl_metadata.label_body0,
        )

    @property
    def pose_in_body0(self):
        pxr = self._scene._kernel.pxr

        positions = []
        rotations = []

        for prim in self._usd_prims:
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
        pxr = self._scene._kernel.pxr

        body_prim_paths = []

        for prim in self._usd_prims:
            api = pxr.UsdPhysics.Joint(prim)
            targets = api.GetBody1Rel().GetTargets()
            # TODO
            [target] = targets
            body_prim_paths.append(target.pathString)

        # TODO
        return Entity._from_usd_prim_ref(
            ref=USDPrimPathRef(
                paths=body_prim_paths,
                stage_ref=lambda: self._scene._usd_stage
            ),
            scene=self._scene,
            # TODO
            _label=self._impl_metadata.label_body1,
        )

    @property
    def pose_in_body1(self):
        pxr = self._scene._kernel.pxr

        positions = []
        rotations = []

        for prim in self._usd_prims:
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
class FixedJoint(ProtoFixedJoint, Joint):
    pass


class RevoluteJoint(ProtoRevoluteJoint, Joint):
    @property
    def axis(self):
        pxr = self._scene._kernel.pxr

        value = []
        for prim in self._usd_prims:
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
    
    # TODO deg or rad???
    @property
    def position_limit(self):
        pxr = self._scene._kernel.pxr

        value = []
        for prim in self._usd_prims:
            api = pxr.UsdPhysics.RevoluteJoint(prim)
            v = [numpy.nan, numpy.nan]
            if not api:
                warnings.warn(f"TODO: {prim}")
            else:
                v = [api.GetLowerLimitAttr().Get(), api.GetUpperLimitAttr().Get()]
            value.append(v)

        # TODO
        return numpy.deg2rad(value)
        # return numpy.asarray(value)


# TODO
class PrismaticJoint(ProtoPrismaticJoint, Joint):
    @property
    def axis(self):
        pxr = self._scene._kernel.pxr

        value = []
        for prim in self._usd_prims:
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
        pxr = self._scene._kernel.pxr

        value = []
        for prim in self._usd_prims:
            api = pxr.UsdPhysics.PrismaticJoint(prim)
            v = [numpy.nan, numpy.nan]
            if not api:
                warnings.warn(f"TODO: {prim}")
            else:
                v = [api.GetLowerLimitAttr().Get(), api.GetUpperLimitAttr().Get()]
            value.append(v)

        return numpy.asarray(value)
    

# TODO NOTE must be homogenous
# TODO FIXME write operations may not sync to the USD stage unless .step called
class Articulation:
    # TODO next
    @classmethod
    def create(cls):
        raise NotImplementedError

    # TODO next
    def __init__(
        self,
        ref: Entity | PathExpressionLike,
        scene: Scene | None = None,
    ):
        match ref:
            case Entity():
                pass
            case _:
                pass
        raise NotImplementedError
        ...

    # TODO accept entity as well !!!!
    def __init__(self, path: PathExpressionLike, scene: Scene):
        self._scene = scene
        # TODO
        self._path = PathExpression(path)

    # TODO FIXME: perf
    @property
    def _usd_prims(self):
        return [
            self._scene._usd_stage.GetPrimAtPath(path)
            for path in self._scene.resolve(self._path)
        ]

    # TODO FIXME: perf
    @property
    def _usd_articulation_root_prims(self):
        pxr = self._scene._kernel.pxr

        return [
            maybe_root_prim
            for prim in self._usd_prims
            # NOTE this already includes the prim itself
            for maybe_root_prim in pxr.Usd.PrimRange(
                prim, 
                # TODO rm???
                # pxr.Usd.TraverseInstanceProxies(
                #     pxr.Usd.PrimAllPrimsPredicate
                # ),
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
            self._scene._isaac_physx_simulation.flush_changes()
            isaac_physics_tensor_view = self._scene._isaac_physics_tensor_view
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
                f"resolved USD root joint paths (are they valid?): "
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
            
    # TODO clarify UsdPhysicsArticulationRootAPI
    @property
    def path(self):
        return self._isaac_physics_articulation_view.prim_paths

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
    def joints(self):
        pxr = self._scene._kernel.pxr

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

        return {
            joint_name: Joint(
                joint_name_path_mapping[joint_name],
                scene=self._scene,
                # TODO !!!
                _impl_metadata=Joint._ImplMetadata(
                    label=joint_name,
                    label_body0=body0_names[joint_index],
                    label_body1=body1_names[joint_index],
                ),
            )
            for joint_index, joint_name in enumerate(joint_names)
        }

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
            link_names[link_index]: Entity(
                link_paths[..., link_index],
                scene=self._scene,
            )
            for link_index in range(n_links)
        }

    # TODO ###############
    @property
    def link_count(self):
        return self._isaac_physics_articulation_view.shared_metatype.link_count

    # TODO
    @property
    def link_names(self):
        return self._isaac_physics_articulation_view.shared_metatype.link_names

    # TODO !!!!! necesito???
    @property
    def link_paths(self):
        return self._isaac_physics_articulation_view.link_paths

    # TODO necesito?
    @property
    def link_poses(self):
        """
        TODO doc

        """

        view = self._isaac_physics_articulation_view
        link_transforms = view.get_link_transforms()

        return Pose(
            p=link_transforms[..., [0, 1, 2]],
            q=link_transforms[..., [3, 4, 5, 6]],
        )
    # TODO ###############

    # TODO ###################
    @property
    def joint_names(self):
        return self._isaac_physics_articulation_view.shared_metatype.joint_names
    # #################

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
    def dof_types(self):
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
        value_ = torch.broadcast_to(torch.asarray(value), (view.count, view.max_dofs))
        view.set_dof_positions(value_, indices=torch.arange(view.count))
        # TODO FIXME: perf .change_block to defer result fetching?
        self._scene._isaac_physics_tensor_ensure_sync()

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
        value_ = torch.broadcast_to(torch.asarray(value), (view.count, view.max_dofs))
        view.set_dof_velocities(value_, indices=torch.arange(view.count))
        self._scene._isaac_physics_tensor_ensure_sync()

    # TODO dof_velocity_limits
    # view.get_dof_max_velocities

    @functools.cached_property
    def driver(self):
        return ArticulationDriver(articulation=self)

    # TODO deprecate ############
    @property
    def root_pose(self):
        view = self._isaac_physics_articulation_view
        root_trans = view.get_root_transforms()
        return Pose(
            p=root_trans[..., [0, 1, 2]],
            q=root_trans[..., [3, 4, 5, 6]],
        )
    
    @root_pose.setter
    def root_pose(self, value: Pose):
        view = self._isaac_physics_articulation_view
        value_ = torch.broadcast_to(
            torch.concat((torch.asarray(value.p), torch.asarray(value.q)), dim=-1), 
            size=(view.count, 7),
        )
        view.set_root_transforms(value_, indices=torch.arange(view.count))
        self._scene._isaac_physics_tensor_ensure_sync()
    # TODO deprecate ############


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
        value_ = torch.broadcast_to(torch.asarray(value), (view.count, view.max_dofs))
        view.set_dof_stiffnesses(value_, indices=torch.arange(view.count))

    @property
    def dof_dampings(self):
        view = self._articulation._isaac_physics_articulation_view
        return view.get_dof_dampings()
    
    @dof_dampings.setter
    def dof_dampings(self, value):
        view = self._articulation._isaac_physics_articulation_view
        value_ = torch.broadcast_to(torch.asarray(value), (view.count, view.max_dofs))
        view.set_dof_dampings(value_, indices=torch.arange(view.count))
    
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
        value_ = torch.broadcast_to(torch.asarray(value), (view.count, view.max_dofs))
        view.set_dof_actuation_forces(value_, indices=torch.arange(view.count))

    # TODO mv and stdize
    class DriveType(enum.Enum):
        Disabled = 0
        Force = 1
        Acceleration = 2

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
        value_ = torch.broadcast_to(torch.asarray(value), (view.count, view.max_dofs))
        view.set_dof_position_targets(value_, indices=torch.arange(view.count))

    @property
    def dof_target_velocities(self):
        view = self._articulation._isaac_physics_articulation_view
        return view.get_dof_velocity_targets()

    @dof_target_velocities.setter
    def dof_target_velocities(self, value):
        view = self._articulation._isaac_physics_articulation_view
        value_ = torch.broadcast_to(torch.asarray(value), (view.count, view.max_dofs))
        view.set_dof_velocity_targets(value_, indices=torch.arange(view.count))