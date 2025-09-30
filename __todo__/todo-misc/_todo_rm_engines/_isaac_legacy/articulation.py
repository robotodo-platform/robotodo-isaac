"""
TODO doc

"""


import enum
import functools

# TODO rm?
# import jax
# import xarray
# TODO !!!!!
from robotodo.utils import Pose

from .scene import Scene


class Articulation:
    """
    TODO doc

    """

    # TODO
    def __init__(self, _todo_selector: str | list[str], scene: Scene):
        self._scene = scene
        self._kernel = scene._kernel
        # TODO
        self._todo_selector = _todo_selector

    @functools.cached_property
    def __physics_articulation_view(self):
        # TODO
        self._kernel.flush_physics_changes()
        try:
            res = self._kernel.get_physics_view().create_articulation_view(
                self._todo_selector
            )
            assert res is not None
            assert res.check()
        except Exception as error:
            raise RuntimeError(
                f"Failed to create physics view from selector (is it valid?): {self._todo_selector}"
            ) from error
        return res
    
    def _get_physics_articulation_view(self):
        if not self.__physics_articulation_view.check():
            del self.__physics_articulation_view
        return self.__physics_articulation_view
    
    @property
    def is_root_fixed(self):
        return self._get_physics_articulation_view().shared_metatype.fixed_base
    
    @property
    def link_names(self):
        # TODO LinksView?
        return self._get_physics_articulation_view().shared_metatype.link_names

    @property
    def link_poses(self):
        """
        TODO doc

        """

        view = self._get_physics_articulation_view()
        link_transforms = view.get_link_transforms()

        return Pose(
            p=link_transforms[..., [0, 1, 2]],
            q=link_transforms[..., [3, 4, 5, 6]],
        )

        # TODO FIXME jax indexing overhead due to non-jit 
        # link_transforms = jax.dlpack.from_dlpack(link_transforms)
        # return xarray.Dataset(
        #     Pose(
        #         p=xarray.DataArray(
        #             jax.dlpack.from_dlpack(link_transforms[..., [0, 1, 2]]),
        #             dims=("articulation", "link", "pos"),
        #             coords={
        #                 "articulation": physics_articulation_view.prim_paths,
        #                 "link": physics_articulation_view.shared_metatype.link_names,
        #                 "pos": ["x", "y", "z"],
        #             },
        #         ),
        #         q=xarray.DataArray(
        #             jax.dlpack.from_dlpack(link_transforms[..., [3, 4, 5, 6]]),
        #             dims=("articulation", "link", "quat"),
        #             coords={
        #                 "articulation": physics_articulation_view.prim_paths,
        #                 "link": physics_articulation_view.shared_metatype.link_names,
        #                 "quat": ["x", "y", "z", "w"],
        #             },
        #         ),
        #     )
        # )

    @property
    def dof_positions(self):
        view = self._get_physics_articulation_view()
        dof_positions = view.get_dof_positions()
        return dof_positions
    
    @dof_positions.setter
    def dof_positions(self, value):
        view = self._get_physics_articulation_view()
        # TODO
        # view.set_dof_positions()

        raise NotImplementedError

    class DOFType(enum.Enum):
        Rotation = 0
        Translation = 1

    # TODO
    @property
    def dof_types(self):
        view = self._get_physics_articulation_view()
        return view.get_dof_types()
