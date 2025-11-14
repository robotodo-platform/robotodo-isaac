import os
import functools
import warnings

# TODO NOTE seealso https://curobo.org/notes/07_environment_variables.html
os.environ.setdefault("CUROBO_TORCH_CUDA_GRAPH_RESET", "1")

import torch

import curobo.geom.types
# from curobo.types.robot import RobotConfig
# from curobo.geom.types import WorldConfig, Mesh
from curobo.types.math import Pose as CuroboPose
from curobo.types.robot import JointState
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
    PoseCostMetric,
    WorldConfig,
)
from curobo.util.torch_utils import is_cuda_graph_available, is_cuda_graph_reset_available


import einops

# TODO
from robotodo.utils.pose import Pose

from tensorspecs import TensorSpec, TensorTableSpec, TensorTableLike



class MotionPlanningError(Exception):
    pass


class ReachPlanner:

    observation_spec = TensorTableSpec({
        "dof_positions": TensorSpec("n? dof"),
        # "target_pose": PoseSpec("n?"),
        # "target_pose_candidates": PoseSpec("n? candidate"),
        # "obstacles": ...,
    })

    action_spec = TensorTableSpec({
        # TODO -or- namedtensor??
        # "dof_names": TensorSpec("dof"),
        "dof_positions": TensorSpec("time n? dof"),
        "dof_velocities": TensorSpec("time n? dof"),
    })

    # TODO cache with purpose to avoid cuda error: "plan", "plan_goalset"
    # TODO allow urdf input
    # TODO allow custom ee link
    def __init__(self, _todo_robot_config: dict | MotionGen):
        ...

        match _todo_robot_config:
            case dict():
                # TODO
                self._curobo_motion_gen = MotionGen(
                    MotionGenConfig.load_from_robot_config(
                        # robot_cfg=RobotConfig.from_basic(
                        #     # TODO '/home/sysadmin/lab/robotodo/.conda/lib/python3.11/site-packages/isaacsim/exts/isaacsim.asset.importer.urdf/data/urdf/robots/franka_description/robots/panda_arm_hand.urdf'
                        #     urdf_path=panda_arm_hand_urdf_path,
                        #     base_link="panda_link0",
                        #     ee_link="panda_link7",
                        #     # ee_link="panda_leftfinger",
                        # ),
                        robot_cfg=_todo_robot_config,
                        # world_model=WorldConfig(
                        #     # TODO
                        #     # mesh=curobo_meshes,
                        # ),
                        # world_model=obstacles.get_collision_check_world(),
                        # interpolation_dt=1 / 250,
                        interpolation_dt=1 / 60,
                        # num_trajopt_seeds=1,
                        num_graph_seeds=1,
                        # TODO requires %env CUROBO_TORCH_CUDA_GRAPH_RESET=1
                        # TODO NOTE seealso https://curobo.org/notes/07_environment_variables.html
                        use_cuda_graph=is_cuda_graph_available() and is_cuda_graph_reset_available(),
                        # use_cuda_graph=False,
                    )
                )                
            case MotionGen():
                # TODO
                self._curobo_motion_gen = _todo_robot_config

    def compute_action(
        self, 
        observation: TensorTableLike[observation_spec],
        *,
        include_raw_result: bool = False,
    ) -> TensorTableLike[action_spec]:
        """
        TODO doc

        """

        dof_positions = observation["dof_positions"]
        target_pose: Pose = observation["target_pose"]

        dof_positions, batch_shapes = einops.pack([dof_positions], "* dof")

        # TODO
        target_pose_p, _ = einops.pack([target_pose.p], "* p")
        target_pose_q, _ = einops.pack([target_pose.q], "* q")

        # TODO broadcast target_pose?? necesito??
        # einops.broadcast
        # einops.unpack(target_pose.p, batch_shapes, "* p")
        # einops.unpack(target_pose.q, batch_shapes, "* q")

        try:
            # TODO .plan_batch_env for batch collisions
            plan_result = self._curobo_motion_gen.plan_batch(
                start_state=JointState.from_position(
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
            raise MotionPlanningError(f"Failed to generate optimized plan: {plan_result.status}")

        if plan_result.success is None or not torch.all(plan_result.success):
            warnings.warn(f"Motion planning is partially successful: {plan_result.status}")

        dof_positions_result = plan_result.optimized_plan.position
        # NOTE curobo auto-squeezes the result which must be undone!
        dof_positions_result = dof_positions_result.reshape(
            -1, *dof_positions_result.shape[-2:],
        )
        [dof_positions_result] = einops.unpack(
            dof_positions_result,
            batch_shapes, "* time dof",
        )

        dof_velocities_result = plan_result.optimized_plan.velocity
        # NOTE curobo auto-squeezes the result which must be undone!
        dof_velocities_result = dof_velocities_result.reshape(
            -1, *dof_velocities_result.shape[-2:],
        )
        # TODO
        [dof_velocities_result] = einops.unpack(
            dof_velocities_result,
            batch_shapes, "* time dof",
        )

        action = {
            "dof_positions": dof_positions_result,
            "dof_velocities": dof_velocities_result,
        }

        if include_raw_result:
            return action, plan_result
        return action


# TODO
class ReachSamplePlanner:

    observation_spec = TensorTableSpec({
        "dof_positions": TensorSpec("n? dof"),
        # "candidate_target_poses": PoseSpec("n? sample"),
        # "obstacles": ...,
    })

    action_spec = TensorTableSpec({
        "dof_positions": TensorSpec("time n? dof"),
        "dof_velocities": TensorSpec("time n? dof"),
    })


    # TODO allow custom ee link
    def __init__(self, _todo_robot_config: dict):

        # TODO
        self._curobo_motion_gen = MotionGen(
            MotionGenConfig.load_from_robot_config(
                # robot_cfg=RobotConfig.from_basic(
                #     # TODO '/home/sysadmin/lab/robotodo/.conda/lib/python3.11/site-packages/isaacsim/exts/isaacsim.asset.importer.urdf/data/urdf/robots/franka_description/robots/panda_arm_hand.urdf'
                #     urdf_path=panda_arm_hand_urdf_path,
                #     base_link="panda_link0",
                #     ee_link="panda_link7",
                #     # ee_link="panda_leftfinger",
                # ),
                robot_cfg=_todo_robot_config,
                # TODO
                world_model=WorldConfig(),
                # world_model=WorldConfig(
                #     # TODO
                #     # mesh=curobo_meshes,
                # ),
                # world_model=obstacles.get_collision_check_world(),
                # interpolation_dt=1 / 250,
                # interpolation_dt=1 / 60,
                # num_trajopt_seeds=1,
                # num_graph_seeds=1,
                # TODO requires %env CUROBO_TORCH_CUDA_GRAPH_RESET=1
                # TODO NOTE seealso https://curobo.org/notes/07_environment_variables.html
                use_cuda_graph=is_cuda_graph_available() and is_cuda_graph_reset_available(),
                # use_cuda_graph=False,
            )
        )

    # TODO invalidate when world_model changes
    @functools.cached_property
    def _todo_curobo_motion_gen(self, _todo_config: ...):
        ...

    def compute_action(
        self, 
        observation: TensorTableLike[observation_spec],
        *,
        include_raw_result: bool = False,
    ):

        dof_positions = observation["dof_positions"]
        target_pose_samples: Pose = observation["candidate_target_poses"]

        # TODO
        observation.get("obstacle_geometries")
        observation.get("obstacle_poses")

        dof_positions, batch_shapes = einops.pack([dof_positions], "* dof")
        # TODO
        target_pose_samples_p, _ = einops.pack([target_pose_samples.p], "* sample p")
        target_pose_samples_q, _ = einops.pack([target_pose_samples.q], "* sample q")

        try:
            plan_result = self._curobo_motion_gen.plan_batch_goalset(
                start_state=JointState.from_position(
                    position=torch.as_tensor(dof_positions, dtype=torch.float32).cuda(),
                ),
                goal_pose=CuroboPose(
                    position=torch.as_tensor(target_pose_samples_p, dtype=torch.float32).cuda(), 
                    # TODO use named wxyz
                    quaternion=torch.as_tensor(target_pose_samples_q, dtype=torch.float32).cuda()
                        [..., [3, 0, 1, 2]],
                ),
            )
        except Exception as error:
            raise RuntimeError("TODO") from error

        if plan_result.optimized_plan is None:
            raise MotionPlanningError(f"Failed to generate optimized plan: {plan_result.status}")

        dof_positions_result = plan_result.optimized_plan.position
        # NOTE curobo auto-squeezes the result which must be undone!
        dof_positions_result = dof_positions_result.reshape(
            -1, *dof_positions_result.shape[-2:],
        )
        [dof_positions_result] = einops.unpack(
            dof_positions_result,
            batch_shapes, "* time dof",
        )

        dof_velocities_result = plan_result.optimized_plan.velocity
        # NOTE curobo auto-squeezes the result which must be undone!
        dof_velocities_result = dof_velocities_result.reshape(
            -1, *dof_velocities_result.shape[-2:],
        )
        # TODO
        [dof_velocities_result] = einops.unpack(
            dof_velocities_result,
            batch_shapes, "* time dof",
        )

        action = {
            "dof_positions": dof_positions_result,
            "dof_velocities": dof_velocities_result,
        }

        if include_raw_result:
            return action, plan_result
        return action