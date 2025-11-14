# TODO
# https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/deformables_beta/deformable_beta.html#create-and-assign-materials

import functools
import warnings
from typing import Callable

# TODO
import numpy

from robotodo.engines.isaac.scene import Scene
from robotodo.engines.isaac._utils import USDPrimHelper

from robotodo.engines.core.path import PathExpression, PathExpressionLike


class Material:

    _usd_prims_ref: Callable[[Scene], "list[pxr.Usd.Prim]"]

    def __init__(
        self,
        ref: PathExpressionLike | Callable[[Scene], "list[pxr.Usd.Prim]"],
        scene: Scene,
    ):
        match ref:
            case _ if isinstance(ref, Callable):
                self._usd_prims_ref = ref
            case _:
                # TODO cache
                class _USDPrimsRef:
                    def __init__(self, path: PathExpressionLike):
                        self._path = path
                    
                    def __repr__(self):
                        return repr(self._path)
                    
                    def __call__(self, scene: Scene):
                        pxr = scene._kernel.pxr
                        return [
                            maybe_material_prim
                            for prim_path in scene.resolve(self._path)
                            for maybe_material_prim in pxr.Usd.PrimRange(
                                scene._usd_stage.GetPrimAtPath(prim_path), 
                                # pxr.Usd.TraverseInstanceProxies(
                                #     pxr.Usd.PrimAllPrimsPredicate
                                # ),
                            )
                            if maybe_material_prim.IsA(pxr.UsdShade.Material)
                        ]
                self._usd_prims_ref = _USDPrimsRef(ref)
        # TODO maybe _kernel?
        self._scene = scene

    def __repr__(self):
        return f"{self.__class__.__qualname__}({self._usd_prims_ref!r}, scene={self._scene})"

    @property
    def _usd_prims(self):
        return self._usd_prims_ref(self._scene)
    
    @property
    def path(self):
        return [
            material_prim.GetPath().pathString
            if material_prim else
            None
            for material_prim in self._usd_prims
        ]

    @property
    def static_friction(self):
        pxr = self._scene._kernel.pxr
        # TODO FIXME perf
        res = []
        for material_prim in self._usd_prims:
            value = numpy.nan
            if material_prim:
                if material_prim.HasAPI(pxr.UsdPhysics.MaterialAPI):
                    value = (
                        pxr.UsdPhysics.MaterialAPI(material_prim)
                        .GetStaticFrictionAttr()
                        .Get()
                    )
                if material_prim.HasAPI("OmniPhysicsBaseMaterialAPI"):
                    value = material_prim.GetAttribute("omniphysics:staticFriction").Get()
            res.append(value)
        return numpy.asarray(res)

    @static_friction.setter
    def static_friction(self, value):
        pxr = self._scene._kernel.pxr
        material_prims = self._usd_prims
        for material_prim, v in zip(
            material_prims,
            numpy.broadcast_to(value, len(material_prims)).astype(numpy.float_),
            strict=True,
        ):
            if not material_prim:
                continue
            # TODO
            pxr.UsdPhysics.MaterialAPI.Apply(material_prim) \
                .GetStaticFrictionAttr().Set(v)
            # TODO
            material_prim.ApplyAPI("OmniPhysicsBaseMaterialAPI")
            material_prim.GetAttribute("omniphysics:staticFriction").Set(v)

    @property
    def dynamic_friction(self):
        pxr = self._scene._kernel.pxr
        # TODO FIXME perf
        res = []
        for material_prim in self._usd_prims:
            value = numpy.nan
            if material_prim:
                if material_prim.HasAPI(pxr.UsdPhysics.MaterialAPI):
                    value = (
                        pxr.UsdPhysics.MaterialAPI(material_prim)
                        .GetDynamicFrictionAttr()
                        .Get()
                    )
                if material_prim.HasAPI("OmniPhysicsBaseMaterialAPI"):
                    value = material_prim.GetAttribute("omniphysics:dynamicFriction").Get()
            res.append(value)
        return numpy.asarray(res)

    @dynamic_friction.setter
    def dynamic_friction(self, value):
        pxr = self._scene._kernel.pxr
        material_prims = self._usd_prims
        for material_prim, v in zip(
            material_prims,
            numpy.broadcast_to(value, len(material_prims)).astype(numpy.float_),
            strict=True,
        ):
            if not material_prim:
                continue
            # TODO
            pxr.UsdPhysics.MaterialAPI.Apply(material_prim) \
                .GetDynamicFrictionAttr().Set(v)
            # TODO
            material_prim.ApplyAPI("OmniPhysicsBaseMaterialAPI")
            material_prim.GetAttribute("omniphysics:dynamicFriction").Set(v)

    @property
    def density(self):
        pxr = self._scene._kernel.pxr
        # TODO FIXME perf
        res = []
        for material_prim in self._usd_prims:
            value = numpy.nan
            if material_prim:
                if material_prim.HasAPI(pxr.UsdPhysics.MaterialAPI):
                    value = (
                        pxr.UsdPhysics.MaterialAPI(material_prim)
                        .GetDensityAttr()
                        .Get()
                    )
                if material_prim.HasAPI("OmniPhysicsBaseMaterialAPI"):
                    value = material_prim.GetAttribute("omniphysics:density").Get()
            res.append(value)
        return numpy.asarray(res)

    @density.setter
    def density(self, value):
        pxr = self._scene._kernel.pxr
        material_prims = self._usd_prims
        for material_prim, v in zip(
            material_prims,
            numpy.broadcast_to(value, len(material_prims)).astype(numpy.float_),
            strict=True,
        ):
            if not material_prim:
                continue
            # TODO
            pxr.UsdPhysics.MaterialAPI.Apply(material_prim) \
                .GetDensityAttr().Set(v)
            # TODO
            material_prim.ApplyAPI("OmniPhysicsBaseMaterialAPI")
            material_prim.GetAttribute("omniphysics:density").Set(v)

    @property
    def youngs_modulus(self):
        # TODO FIXME perf
        res = []
        for material_prim in self._usd_prims:
            value = numpy.nan
            if material_prim:
                if material_prim.HasAPI("OmniPhysicsDeformableMaterialAPI"):
                    value = material_prim.GetAttribute("omniphysics:youngsModulus").Get()
            res.append(value)
        return numpy.asarray(res)

    @youngs_modulus.setter
    def youngs_modulus(self, value):
        material_prims = self._usd_prims
        for material_prim, v in zip(
            material_prims,
            numpy.broadcast_to(value, len(material_prims)).astype(numpy.float_),
            strict=True,
        ):
            if not material_prim:
                continue
            # TODO
            material_prim.ApplyAPI("OmniPhysicsDeformableMaterialAPI")
            material_prim.GetAttribute("omniphysics:youngsModulus").Set(v)

    @property
    def poissons_ratio(self):
        # TODO FIXME perf
        res = []
        for material_prim in self._usd_prims:
            value = numpy.nan
            if material_prim:
                if material_prim.HasAPI("OmniPhysicsDeformableMaterialAPI"):
                    value = material_prim.GetAttribute("omniphysics:poissonsRatio").Get()
            res.append(value)
        return numpy.asarray(res)
    
    @poissons_ratio.setter
    def poissons_ratio(self, value):
        material_prims = self._usd_prims
        for material_prim, v in zip(
            material_prims,
            numpy.broadcast_to(value, len(material_prims)).astype(numpy.float_),
            strict=True,
        ):
            if not material_prim:
                continue
            # TODO
            material_prim.ApplyAPI("OmniPhysicsDeformableMaterialAPI")
            material_prim.GetAttribute("omniphysics:poissonsRatio").Set(v)

    @property
    def surface_thickness(self):
        # TODO FIXME perf
        res = []
        for material_prim in self._usd_prims:
            value = numpy.nan
            if material_prim:
                if material_prim.HasAPI("OmniPhysicsSurfaceDeformableMaterialAPI"):
                    value = material_prim.GetAttribute("omniphysics:surfaceThickness").Get()
            res.append(value)
        return numpy.asarray(res)

    @surface_thickness.setter
    def surface_thickness(self, value):
        material_prims = self._usd_prims
        for material_prim, v in zip(
            material_prims,
            numpy.broadcast_to(value, len(material_prims)).astype(numpy.float_),
            strict=True,
        ):
            if not material_prim:
                continue
            # TODO
            material_prim.ApplyAPI("OmniPhysicsSurfaceDeformableMaterialAPI")
            material_prim.GetAttribute("omniphysics:surfaceThickness").Set(v)


    # TODO
    """
    # TODO
    legacy_api = pxr.UsdPhysics.MaterialAPI.Apply(todo_material)
    print(legacy_api.CreateDynamicFrictionAttr().Get())
    todo_material.ApplyAPI("OmniPhysicsBaseMaterialAPI")
    print(todo_material.GetAttribute("omniphysics:dynamicFriction").Get())

    # TODO
    legacy_api = pxr.UsdPhysics.MaterialAPI.Apply(todo_material)
    print(legacy_api.CreateDensityAttr().Get())
    todo_material.ApplyAPI("OmniPhysicsBaseMaterialAPI")
    print(todo_material.GetAttribute("omniphysics:density").Get())
    """

    # TODO
    """
    # TODO
    prim.ApplyAPI("OmniPhysicsDeformableMaterialAPI")
    prim.GetAttribute("omniphysics:youngsModulus").Set(youngs)
    prim.GetAttribute("omniphysics:poissonsRatio").Set(poissons)

    # TODO
    prim.ApplyAPI("OmniPhysicsSurfaceDeformableMaterialAPI")
    prim.GetAttribute("omniphysics:surfaceThickness").Set(thickness)
    # Estimate bend stiffness from Young's modulus and Poisson's ratio
    # surfaceThickness is incorporated in PhysX
    bendStiffness = youngs/(12.0*(1.0-poissons**2))
    prim.GetAttribute("omniphysics:surfaceBendStiffness").Set(bendStiffness)
    """

