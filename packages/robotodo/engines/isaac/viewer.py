
from typing import TypedDict, Unpack

import numpy
import einops
from robotodo.utils.pose import Pose

from .scene import Scene
from .entity import Entity


class SceneViewer:
    def __init__(self, scene: Scene):
        self._scene = scene

    # TODO
    def show(self):
        pass

    # TODO
    @property
    def selected_entity(self):
        omni = self._scene._kernel.omni
        # TODO
        selection = omni.usd.get_context().get_selection()
        return Entity(selection.get_selected_prim_paths(), scene=self._scene)
    
    # TODO
    @selected_entity.setter
    def selected_entity(self, value: ...):
        raise NotImplementedError

    # TODO
    @property
    def _isaac_debug_draw_interface(self):
        # TODO
        self._scene._kernel.enable_extension("isaacsim.util.debug_draw")
        return (
            self._scene._kernel.isaacsim.util.debug_draw._debug_draw
            .acquire_debug_draw_interface()
        )
    
        # TODO lifecycle
        # isaacsim = self._scene._kernel.isaacsim
        # isaacsim.util.debug_draw._debug_draw.release_debug_draw_interface

    def clear_drawings(self):
        iface = self._isaac_debug_draw_interface
        iface.clear_points()
        iface.clear_lines()

    class DrawPoseOptions(TypedDict):
        scale: float | None
        line_thickness: float | None
        line_opacity: float | None

    def draw_pose(
        self, 
        pose: Pose, 
        options: DrawPoseOptions = DrawPoseOptions(),
        **options_kwds: Unpack[DrawPoseOptions],
    ):
        """
        TODO doc


        """

        options = self.DrawPoseOptions(options, **options_kwds)

        scale: float = options.get("scale", 1.)
        line_thickness: float = options.get("line_thickness", 2)
        line_opacity: float = options.get("line_opacity", .5)

        # TODO x y z
        for mask in (
            numpy.asarray([1., 0., 0.]),
            numpy.asarray([0., 1., 0.]),
            numpy.asarray([0., 0., 1.]),
        ):
            start_points = pose.p
            # TODO
            end_points = (pose * Pose(p=mask * [scale, scale, scale])).p

            start_points, _ = einops.pack([start_points], "* xyz")
            end_points, _ = einops.pack([end_points], "* xyz")

            colors = einops.repeat(
                numpy.asarray([*mask, line_opacity]),
                "rgba -> b rgba",
                **einops.parse_shape(start_points, "b _"),
            )
            thicknesses = einops.repeat(
                numpy.asarray(line_thickness),
                "-> b",
                **einops.parse_shape(start_points, "b _"),
            )

            self._isaac_debug_draw_interface.draw_lines(
                numpy.asarray(start_points).tolist(), 
                numpy.asarray(end_points).tolist(), 
                numpy.asarray(colors).tolist(), 
                numpy.asarray(thicknesses).tolist(),
            )


# TODO
from robotodo.utils.geometry import export_trimesh

# TODO
class EntityViewer:
    def __init__(self, entity: Entity):
        self._entity = entity

    def show(self):
        # TODO
        import itertools

        import trimesh
        import trimesh.viewer

        return (
            trimesh.Scene([
                export_trimesh(g)
                for g in itertools.chain(*self._entity.geometry)
            ])
            .show()
        )

