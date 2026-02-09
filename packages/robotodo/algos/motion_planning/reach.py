# TODO
import os
# TODO NOTE seealso https://curobo.org/notes/07_environment_variables.html
os.environ.setdefault("CUROBO_TORCH_CUDA_GRAPH_RESET", "1")

# TODO
import functools
import warnings
from typing import Callable, TypedDict, Unpack, NotRequired, Literal

import numpy
import einops
import torch
# TODO
import trimesh
# TODO
import morphitx
from tensorspecs import TensorSpec, TensorTableSpec, TensorTableLike
from curobo.types.state import JointState as CuroboJointState
from curobo.types.robot import RobotConfig as CuroboRobotConfig
from curobo.types.math import Pose as CuroboPose
from curobo.cuda_robot_model.cuda_robot_generator import (
    CudaRobotGenerator,
    CudaRobotGeneratorConfig as CuroboCudaRobotGeneratorConfig,
)
from curobo.cuda_robot_model.cuda_robot_model import (
    CudaRobotModelConfig as CuroboCudaRobotModelConfig,
)
from curobo.wrap.reacher.motion_gen import (
    MotionGen as CuroboMotionGen,
    MotionGenConfig as CuroboMotionGenConfig,
)
from curobo.cuda_robot_model.types import JointType as CuroboJointType
from curobo.cuda_robot_model.kinematics_parser import (
    KinematicsParser as CuroboKinematicsParser,
    LinkParams as CuroboLinkParams,
)
from robotodo.utils.pose import Pose
# TODO
from robotodo.utils.geometry import export_trimesh
from robotodo.engines.core.body import ProtoBody
from robotodo.engines.core.articulation import (
    Axis,
    JointKind,
    ProtoJoint,
    ProtoRevoluteJoint,
    ProtoPrismaticJoint,
)


# TODO
def _array_reduce_single(x: numpy.ndarray, shape: tuple[int, ...]):
    x = numpy.reshape(x, (-1, *shape))
    x = numpy.unique(x, axis=0)
    try:
        x = numpy.squeeze(x, axis=0)
    except Exception as error:
        # TODO better msg
        raise ValueError(
            f"Failed to reduce batched array from shape {x.shape} to {shape}, "
            f"are the elements along the leading batch dimension unique?: {x}"
        ) from error
    return x

# TODO
def test_array_reduce_single():
    # TODO
    # _array_reduce_single(["a", "a"], shape=())
    _array_reduce_single(numpy.full((2, 3, 4), fill_value=1), shape=(3, 4))
    _array_reduce_single(numpy.random.rand(*(2, 3, 4)), shape=(3, 4))


# TODO
def _compute_axis_vector(axis: Axis):
    axis_vector = numpy.full((*numpy.shape(axis), 3), fill_value=0)
    # TODO handle Axis.UNKNOWN !!!!!
    axis_vector[..., axis] = 1
    axis_vector = axis_vector[..., [Axis.X, Axis.Y, Axis.Z]]
    return axis_vector


# TODO
def _compute_axis(axis_vector: ...):
    axis_vector = numpy.asarray(axis_vector)
    dominant_axis_index = numpy.argmax(numpy.abs(axis_vector))
    sign = numpy.sign(axis_vector[..., dominant_axis_index])
    # TODO handle Axis.UNKNOWN !!!!!
    axis = numpy.asarray([Axis.X, Axis.Y, Axis.Z])[dominant_axis_index]
    return axis, sign


class _CuroboKinematicsParser(CuroboKinematicsParser):
    # TODO rm
    # class Config(TypedDict):
    #     joints: list[Joint]
    #     joint_name_fn: Callable[[Joint], str] | None
    #     link_name_fn: Callable[[Entity], str] | None

    def __init__(
        self, 
        # TODO ProtoJoint
        robotodo_joints: list[ProtoJoint],
        extra_links: dict[str, CuroboLinkParams] | None = None,
    ):
        # TODO do something
        self._robotodo_joints = robotodo_joints
        super().__init__(extra_links=extra_links)

    # TODO
    def _robotodo_derive_entity_name(self, entity: ProtoJoint | ProtoBody):
        if entity.label is not None:
            return _array_reduce_single(entity.label, shape=()).item()
        # TODO is this the only way to make it hashable???
        return str(entity.path)

    # TODO NOTE override
    def build_link_parent(self):
        for joint in self._robotodo_joints:
            # TODO rm
            # parent_link_name = joint.body0.path[0]
            # child_link_name = joint.body1.path[0]
            parent_link_name = self._robotodo_derive_entity_name(joint.body0)
            link_name = self._robotodo_derive_entity_name(joint.body1)            
            self._parent_map[link_name] = {"parent": parent_link_name}
    
    @functools.lru_cache
    def _build_link_geometries(self) -> dict[str, trimesh.Trimesh]:
        link_geometries = dict()

        for joint in self._robotodo_joints:
            for body in (joint.body0, joint.body1):
                link_name = self._robotodo_derive_entity_name(body)
                if link_name in link_geometries:
                    continue

                trimesh_objs = []
                for subgeoms in body.geometry:
                    for subgeom in subgeoms:
                        # TODO upstream batch support!!!!
                        trimesh_obj = export_trimesh(subgeom)
                        if trimesh_obj is None:
                            # TODO
                            warnings.warn(f"TODO: body contains non-convertible geometry, skipping the non-convertible geometry: {body}")
                            continue
                        # TODO
                        trimesh_objs.append(trimesh_obj.convex_hull)
                if len(trimesh_objs) == 0:
                    warnings.warn(f"TODO: body has no geometry, skipping: {body}")
                    continue
                trimesh_obj_combined: trimesh.Trimesh = trimesh.boolean.union(trimesh_objs)
                trimesh_obj_combined.metadata["_curobo_kinematics_parser_link_name"] = link_name
                if trimesh_obj_combined.is_empty:
                    warnings.warn(f"TODO: body has empty geometry, skipping: {body}")
                    continue

                link_geometries[link_name] = trimesh_obj_combined

        return link_geometries

    # TODO
    @functools.lru_cache
    def _build_link_parameters(self) -> dict[str, CuroboLinkParams]:
        link_params = dict()

        # TODO
        for joint in self._robotodo_joints:
            # TODO rm
            # TODO relative to articulation?
            # parent_link_name = joint.body0.path[0]
            # link_name = joint.body1.path[0]
            parent_link_name = self._robotodo_derive_entity_name(joint.body0)
            link_name = self._robotodo_derive_entity_name(joint.body1)     

            joint_pose_in_parent_link = joint.pose_in_body0
            joint_pose_in_link = joint.pose_in_body1

            # TODO this is correct???
            joint_frame = _array_reduce_single(
                (joint_pose_in_parent_link * joint_pose_in_link.inv()).to_matrix(),
                # (joint.pose_in_body1[0].inv() * joint.pose_in_body0[0])
                shape=(4, 4),
            )
            # TODO ensure homogenous

            # TODO rm
            # joint_name = joint.path[0]
            joint_name = self._robotodo_derive_entity_name(joint)
            joint_limits = None
            joint_offset_multiplier = 1.
            joint_offset_bias = 0.

            # TODO !!!!!
            joint_type = CuroboJointType.FIXED
            match _array_reduce_single(joint.kind, shape=()):
                case JointKind.FIXED:
                    # TODO
                    joint_type = CuroboJointType.FIXED

                case JointKind.REVOLUTE:
                    # TODO
                    revolute_joint = joint.astype(ProtoRevoluteJoint)
                    # TODO
                    # revolute_joint = RevoluteJoint(joint)
                    #

                    axis_vector = _compute_axis_vector(revolute_joint.axis)
                    # TODO upstream pose: better way to get rotation
                    axis_vector = einops.einsum(
                        joint_pose_in_link.inv().to_matrix()[..., :3, :3],
                        axis_vector,
                        # TODO transpose necesito?
                        "... xyz_b xyz_a, ... xyz_b -> ... xyz_a",
                    )
                    # TODO
                    axis, sign = _compute_axis(axis_vector)
                    axis = _array_reduce_single(axis, shape=())
                    sign = _array_reduce_single(sign, shape=())

                    joint_offset_multiplier = sign
                    match axis:
                        case Axis.X:
                            joint_type = CuroboJointType.X_ROT
                        case Axis.Y:
                            joint_type = CuroboJointType.Y_ROT
                        case Axis.Z:
                            joint_type = CuroboJointType.Z_ROT
                        case _:
                            # TODO
                            raise ValueError(f"TODO: {axis}")

                    # TODO use common range instead?
                    joint_limits = _array_reduce_single(
                        revolute_joint.position_limit,
                        shape=(2, ),
                    )
                    # TODO
                    # joint_limits *= sign

                case JointKind.PRISMATIC:
                    prismatic_joint = joint.astype(ProtoPrismaticJoint)

                    axis_vector = _compute_axis_vector(prismatic_joint.axis)
                    # TODO upstream pose: better way to get rotation
                    axis_vector = einops.einsum(
                        joint_pose_in_link.inv().to_matrix()[..., :3, :3],
                        axis_vector,
                        # TODO
                        "... xyz_b xyz_a, ... xyz_b -> ... xyz_a",
                    )
                    # TODO
                    axis, sign = _compute_axis(axis_vector)
                    axis = _array_reduce_single(axis, shape=())
                    sign = _array_reduce_single(sign, shape=())

                    joint_offset_multiplier = sign
                    match axis:
                        case Axis.X:
                            joint_type = CuroboJointType.X_ROT
                        case Axis.Y:
                            joint_type = CuroboJointType.Y_ROT
                        case Axis.Z:
                            joint_type = CuroboJointType.Z_ROT
                        case _:
                            # TODO
                            raise ValueError(f"TODO: {axis}")
                        
                    # TODO use common range instead?
                    joint_limits = _array_reduce_single(
                        prismatic_joint.position_limit,
                        shape=(2, ),
                    )
                    # TODO
                    # joint_limits *= sign
                case _:
                    joint_type = CuroboJointType.FIXED
                    warnings.warn(f"Unsupported joint kind {joint.kind}, planning as fixed: {joint}")

            link_params[link_name] = CuroboLinkParams(
                link_name=link_name,
                joint_name=joint_name,
                joint_type=joint_type,
                fixed_transform=joint_frame,
                parent_link_name=parent_link_name,
                # TODO
                joint_limits=joint_limits,
                # TODO
                joint_offset=[joint_offset_multiplier, joint_offset_bias],
            )

        return link_params

    def get_link_parameters(self, link_name: str, base: bool = False) -> CuroboLinkParams:
        # TODO
        if base:
            return CuroboLinkParams(
                link_name=link_name,
                # TODO !!!! ensure unique
                joint_name="",
                joint_type=CuroboJointType.FIXED,
                fixed_transform=numpy.eye(4),
            )
        return self._build_link_parameters()[link_name]


class MotionPlanningError(Exception):
    pass


class MotionPlanner:
    # TODO
    class Config(TypedDict, total=False):
        base_link: str
        end_link: str
        # TODO
        use_self_collision: bool
        use_world_collision: bool
        #
        _curobo_motiongen_config: NotRequired[dict]
        r"""TODO doc: see https://github.com/NVlabs/curobo/blob/ebb71702f3f70e767f40fd8e050674af0288abe8/src/curobo/wrap/reacher/motion_gen.py#L88"""

    # TODO
    def __init__(
        self, 
        # TODO ArticulationKinematics instead
        # target: Articulation | list[Joint], 
        target: list[ProtoJoint], 
        config: Config = Config(),
        **config_kwds: Unpack[Config],
    ):
        match target:
            # TODO rm
            # case Articulation() as articulation:
            #     # TODO
            #     curobo_kinematics_parser = (
            #         _CuroboKinematicsParser(
            #             list(articulation.joints.values())
            #         )
            #     )
            case list() as joints:
                # TODO
                curobo_kinematics_parser = (
                    _CuroboKinematicsParser(joints)        
                )
            case _:
                # TODO
                raise NotImplementedError("TODO")
            
        config = MotionPlanner.Config(config, **config_kwds)

        # TODO
        collision_link_names = list()
        collision_spheres: dict[str, list[dict]] = dict()

        # TODO
        if (
            config.get("use_self_collision", False)
            or config.get("use_world_collision", False)
        ):
            link_geometries = curobo_kinematics_parser._build_link_geometries()

            collision_link_names = list(link_geometries.keys())
            collision_spheres: dict[str, list[dict]] = dict()
            spheresets = morphitx.approximate_spheres(
                # TODO
                mesh=[g.convex_hull for g in link_geometries.values()],
                num_spheres=16,
            )
            for key, sphereset in zip(link_geometries.keys(), spheresets, strict=True):           
                collision_spheres[key] = [
                    # TODO
                    dict(center=center, radius=radius)
                    for center, radius in zip(sphereset.centers, sphereset.radii, strict=True)
                ]

        # TODO https://github.com/NVlabs/curobo/blob/ebb71702f3f70e767f40fd8e050674af0288abe8/src/curobo/cuda_robot_model/cuda_robot_generator.py#L319
        self._curobo_motion_gen = CuroboMotionGen(
            CuroboMotionGenConfig.load_from_robot_config(
                robot_cfg=CuroboRobotConfig(
                    kinematics=CuroboCudaRobotModelConfig.from_config(
                        CuroboCudaRobotGeneratorConfig(
                            # TODO infer from dof_names
                            base_link=config["base_link"],
                            ee_link=config["end_link"],
                            custom_kinematics_parser=curobo_kinematics_parser,
                            # TODO
                            collision_link_names=collision_link_names,
                            collision_spheres=collision_spheres,
                            collision_sphere_buffer=0.,
                            self_collision_buffer={
                                collision_link_name: 0.
                                for collision_link_name in collision_link_names
                            },
                            # TODO does this make sense?
                            self_collision_ignore={
                                parent_link_params["parent"]: [child_link_name, ]
                                for child_link_name, parent_link_params in 
                                curobo_kinematics_parser._parent_map.items()
                            },
                        ),
                    ),
                ),
                self_collision_check=config.get("use_self_collision", False),
                **(
                    config.get("_curobo_motiongen_config", dict())
                )
            ),
        )

    @property
    def base_link(self):
        return self._curobo_motion_gen.kinematics.base_link

    @property
    def end_link(self):
        return self._curobo_motion_gen.kinematics.ee_link

    @property
    def joints(self):
        return self._curobo_motion_gen.joint_names
    
    @property
    def dof_names(self):
        return self._curobo_motion_gen.joint_names
    
    @property
    def observation_spec(self):
        n_dofs = len(self.dof_names)
        # TODO named indexing
        return TensorTableSpec({
            "dof_positions": TensorSpec("n? dof", shape={"dof": n_dofs}),
            # "target_pose": PoseSpec("n?"),
            # "target_pose_candidates": PoseSpec("n? candidate"),
            # "obstacles": ...,
        })
        
    @property
    def action_spec(self):
        n_dofs = len(self.dof_names)
        return TensorTableSpec({
            # TODO -or- namedtensor??
            # "dof_names": TensorSpec("dof"),
            "dof_positions": TensorSpec("n? time dof", shape={"dof": n_dofs}),
            "dof_velocities": TensorSpec("n? time dof", shape={"dof": n_dofs}),
        })

    def compute_action(
        self, 
        observation: TensorTableLike[observation_spec],
    ) -> TensorTableLike[action_spec]:
        """
        TODO doc

        """

        # TODO
        dof_positions = observation["dof_positions"]
        target_pose: Pose = observation["target_pose"]

        # TODO
        # observation.get("target_pose")
        # observation.get("target_pose_candidates")

        # TODO broadcast
        dof_positions, [batch_shape] = einops.pack([dof_positions], "* dof")

        # TODO
        target_pose_p, _ = einops.pack([target_pose.p], "* p")
        target_pose_q, _ = einops.pack([target_pose.q], "* q")

        # TODO broadcast target_pose
        target_pose_p = torch.broadcast_to(
            torch.as_tensor(target_pose_p), 
            size=(*batch_shape, 3),
        )
        target_pose_q = torch.broadcast_to(
            torch.as_tensor(target_pose_q), 
            size=(*batch_shape, 4),
        )

        try:
            # TODO .plan_batch_env for batch collisions
            plan_result = self._curobo_motion_gen.plan_batch(
                start_state=CuroboJointState.from_position(
                    position=torch.as_tensor(dof_positions, dtype=torch.float32).cuda(),
                ),
                goal_pose=CuroboPose(
                    position=torch.as_tensor(target_pose_p, dtype=torch.float32).cuda(), 
                    # TODO use named wxyz
                    quaternion=torch.as_tensor(target_pose_q, dtype=torch.float32).cuda()
                        [..., [3, 0, 1, 2]],
                ),
                # plan_config=MotionGenPlanConfig(num_graph_seeds=1, enable_graph=False, max_attempts=100),
                # plan_config=MotionGenPlanConfig(
                #     # num_graph_seeds=1,
                #     # TODO this doesnt seem to do anything
                #     # check_start_validity=False
                # ),
            )
        except Exception as error:
            raise RuntimeError(
                f"An error occurred while planning: are the DOF positions valid?"
            ) from error

        # TODO
        # plan_result.interpolated_plan.position
        # plan_result.interpolated_plan.velocity

        if plan_result.optimized_plan is None:
            raise MotionPlanningError(
                f"Failed to generate optimized plan, "
                f"run {self.report} for more information: {plan_result.status}"
            )

        if plan_result.success is None or not torch.all(plan_result.success):
            warnings.warn(f"Motion planning is partially successful: {plan_result.status}")

        dof_positions_result = torch.asarray(plan_result.optimized_plan.position)
        # NOTE curobo auto-squeezes the result which must be undone!
        dof_positions_result = dof_positions_result.reshape(
            -1, *dof_positions_result.shape[-2:],
        )
        [dof_positions_result] = einops.unpack(
            dof_positions_result,
            [batch_shape], "* time dof",
        )

        dof_velocities_result = torch.asarray(plan_result.optimized_plan.velocity)
        # NOTE curobo auto-squeezes the result which must be undone!
        dof_velocities_result = dof_velocities_result.reshape(
            -1, *dof_velocities_result.shape[-2:],
        )
        # TODO
        [dof_velocities_result] = einops.unpack(
            dof_velocities_result,
            [batch_shape], "* time dof",
        )

        action = {
            "dof_names": self.dof_names,
            "dof_positions": dof_positions_result,
            "dof_velocities": dof_velocities_result,
        }

        return action
    
    # TODO experimental
    def report(self):
        return dict(
            self_collisions=self._report_self_collisions(),
        )

    # TODO experimental
    def _report_self_collisions(self):
        import torch
        from curobo.rollout.arm_base import ArmBase as CuroboArmBase

        # TODO see https://github.com/NVlabs/curobo/discussions/223

        mg = self._curobo_motion_gen
        # TODO set to True?? or check?
        mg.kinematics.robot_spheres.requires_grad

        link_sphere_idx_map = mg.kinematics.kinematics_config.link_sphere_idx_map
        link_sphere_num_collisions_total_map = torch.full((len(link_sphere_idx_map), ), fill_value=0, dtype=torch.int)

        for rollout_instance in mg.get_all_rollout_instances():
            if not isinstance(rollout_instance, CuroboArmBase):
                continue
            sparse_sphere_idx = getattr(rollout_instance.robot_self_collision_constraint, "_sparse_sphere_idx", None)
            if sparse_sphere_idx is None:
                continue

            colliding_sphere_indicators = torch.squeeze(sparse_sphere_idx, dim=1)
            # TODO NOTE tensor shape (n_link_spheres,), element: num of collisions
            link_sphere_num_collisions_map = colliding_sphere_indicators.sum(dim=0)

            link_sphere_num_collisions_total_map += (
                link_sphere_num_collisions_map
                .to(link_sphere_num_collisions_total_map.device)
            )


        colliding_link_indices = link_sphere_idx_map[
            torch.argwhere(link_sphere_num_collisions_total_map).squeeze()
        ]

        link_idx_to_name_map = {v: k for k, v in mg.kinematics.kinematics_config.link_name_to_idx_map.items()}
        colliding_link_names = set([link_idx_to_name_map[idx.item()] for idx in colliding_link_indices])

        return dict(
            colliding_link_names=colliding_link_names,
            link_sphere_num_collisions=link_sphere_num_collisions_total_map,
            link_spheres=mg.kinematics.kinematics_config.link_spheres,
            link_sphere_poses=mg.kinematics.get_all_link_transforms()[
                mg.kinematics.kinematics_config.link_sphere_idx_map.tolist()
            ],
        )
