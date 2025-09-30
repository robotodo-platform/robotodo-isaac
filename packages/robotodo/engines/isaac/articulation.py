
import enum
import functools

# TODO
import torch
from robotodo.utils import Pose
from robotodo.engines.core import PathExpression, PathExpressionLike

from .scene import Scene


# TODO use kernel thread for everything
# TODO NOTE must be homogenous
# TODO FIXME write operations may not sync to the USD stage unless .step called
class Articulation:
    def __init__(self, root_path: PathExpressionLike, scene: Scene):
        self._scene = scene
        # TODO
        self._root_path = PathExpression(root_path)

    @functools.cached_property
    def _isaac_physics_articulation_view_cache(self):
        try:
            resolved_root_paths = self._scene.resolve(self._root_path)
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
                f"Failed to create physics view from root joint paths "
                f"(are they valid?): {resolved_root_paths}"
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

    # TODO
    @property
    def root_joint_path(self):
        # TODO
        return self._isaac_physics_articulation_view.prim_paths

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

    # TODO
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

    # TODO rm    
    # @link_poses.setter
    # def link_poses(self, value: Pose):
    #     # TODO
    #     view = self._physics_tensor_get_articulation_view()
        
    #     value_ = torch.broadcast_to(
    #         torch.concat((value["p"], value["q"]), dim=-1), 
    #         size=(view.count, view.max_links, 7),
    #     )
    #     view.set_link_transforms(value_, indices=torch.arange(view.count))

    @property
    def dof_count(self):
        return self._isaac_physics_articulation_view.shared_metatype.dof_count

    @property
    def dof_names(self):
        # TODO !!!
        return self._isaac_physics_articulation_view.shared_metatype.dof_names

    # TODO mv and standardize
    class DOFType(enum.Enum):
        Rotation = 0
        Translation = 1

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
        view = self._isaac_physics_articulation_view
        value_ = torch.broadcast_to(torch.asarray(value), (view.count, view.max_dofs))
        view.set_dof_positions(value_, indices=torch.arange(view.count))
        # TODO FIXME: perf .change_block to defer result fetching?
        self._scene._isaac_physx_simulation.fetch_results()

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

    # TODO dof_velocity_limits
    # view.get_dof_max_velocities

    @functools.cached_property
    def driver(self):
        return ArticulationDriver(articulation=self)

    @property
    def root_pose(self):
        view = self._isaac_physics_articulation_view
        root_trans = view.get_root_transforms()
        return Pose(
            p=root_trans[..., [0, 1, 2]],
            q=root_trans[..., [3, 4, 5, 6]],
        )
    
    # TODO this doesnt set the pose immediately!!!!
    @root_pose.setter
    def root_pose(self, value: Pose):
        view = self._isaac_physics_articulation_view
        value_ = torch.broadcast_to(
            torch.concat((torch.asarray(value.p), torch.asarray(value.q)), dim=-1), 
            size=(view.count, 7),
        )
        view.set_root_transforms(value_, indices=torch.arange(view.count))


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