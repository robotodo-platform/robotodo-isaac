
import contextlib
import warnings
from typing import Optional, TypedDict, Unpack

import numpy
from robotodo.engines.core.path import PathExpressionLike
from robotodo.utils.geometry import Plane, Box, Sphere, PolygonMesh

from ._kernel import Kernel, get_default_kernel
from .scene import Scene
from .articulation import Articulation
from .entity import Entity
from .sensors import Camera


def usd_ensure_free_path(
    stage: "pxr.Usd.Stage",
    path: str | None = None,
    remove_existing: bool = False,
    # TODO
    kernel: Kernel | None = None,
):
    if kernel is None:
        kernel = get_default_kernel()

    omni = kernel.omni

    if path is not None:
        if remove_existing:
            kernel.enable_extension("omni.usd")
            kernel.enable_extension("omni.kit.commands")
            kernel.import_module("omni.usd")

            # TODO call only when not exists
            # TODO check success??
            delete_prims_command = omni.usd.commands.DeletePrimsCommand(
                [path], 
                stage=stage,
            )
            delete_prims_command.do()

            # is_success = stage.RemovePrim(config.get("path"))
            # if not is_success:
            #     raise RuntimeError(
            #         f"""Failed to remove loaded USD prim at {config.get("path")}"""
            #     )

        if stage.GetPrimAtPath(path).IsValid():
            raise RuntimeError(f"Path already exists: {path}")

    return omni.usd.get_stage_next_free_path(
        stage, 
        path="/" if path is None else path,
        prepend_default_prim=False,
    )


# TODO deprecate
def _usd_get_or_use_free_path(
    scene: Scene, 
    path: str | None = None,
):
    omni = scene._kernel.omni

    # TODO
    scene._kernel.enable_extension("omni.usd")

    if path is None:
        path = omni.usd.get_stage_next_free_path(
            scene._usd_stage, 
            # TODO behavior should be similar to dir??
            path="/",
            prepend_default_prim=False,
        )
    else:
        if scene._usd_stage.GetPrimAtPath(path).IsValid():
            raise RuntimeError(f"Path already exists: {path}")
        
    return path
# TODO deprecate


class USDSceneLoader:
    async def __call__(
        self,
        resource_or_model: ... = None,
        # TODO mv engine??
        _kernel: Kernel = None,
    ) -> Scene:
        if _kernel is None:
            _kernel = get_default_kernel()

        # TODO rm???
        # omni = _kernel.omni
        # _kernel.enable_extension("omni.usd")

        # ctx = omni.usd.get_context()
        # while True:
        #     if ctx.can_open_stage():
        #         break
        #     # TODO
        #     # _kernel.step_app_loop_soon()
        #     # TODO cancel pending stage opening??
        #     await ctx.next_stage_event_async()

        # # TODO NOTE current impl opens model as sublayer: prob safest??
        # is_success, message = await ctx.new_stage_async()
        # if not is_success:
        #     raise RuntimeError(f"Failed to create empty USD scene: {message}")
        # stage = ctx.get_stage()
        # if stage is None:
        #     # TODO
        #     raise RuntimeError("TODO")

        pxr = _kernel.pxr
        _kernel.import_module("pxr.Usd")
        _kernel.import_module("pxr.UsdGeom")

        stage = pxr.Usd.Stage.CreateInMemory()
        # TODO customizable!!!  also necesito???
        pxr.UsdGeom.SetStageUpAxis(stage, pxr.UsdGeom.Tokens.z)

        stage.GetRootLayer().subLayerPaths.append(resource_or_model)

        # TODO this attaches the stage to the viewer and inits rendering+physics?
        # TODO FIXME rm in the future when: 1) urdf stage= bug has been fixed 2) users are informed of the behavior
        _kernel.enable_extension("omni.usd")
        _kernel.import_module("omni.usd")
        await _kernel.omni.usd.get_context().attach_stage_async(stage)

        return Scene(_kernel=_kernel, _usd_stage_ref=stage)

        # TODO alt impl: add stage directly but stage maybe readonly!!!!
        # is_success, message = await ctx.open_stage_async(resource_or_model)
        # if not is_success:
        #     raise RuntimeError(f"Failed to load USD scene {resource_or_model}: {message}")
        # stage = ctx.get_stage()
        # # TODO check None
        # return Scene(_kernel=_kernel, _usd_stage=stage)
    

async def load_usd_scene(
    resource_or_model: ...,
    _kernel: Kernel | None = None,
):
    return await USDSceneLoader()(
        resource_or_model=resource_or_model,
        _kernel=_kernel,
    )


class USDSceneBuilder:
    async def __call__(
        self,
        # TODO mv engine??
        _kernel: Kernel = None,
    ) -> Scene:
        if _kernel is None:
            _kernel = get_default_kernel()

        pxr = _kernel.pxr
        stage = pxr.Usd.Stage.CreateInMemory()
        # TODO customizable!!!  also necesito???
        pxr.UsdGeom.SetStageUpAxis(stage, pxr.UsdGeom.Tokens.z)


        # TODO this attaches the stage to the viewer and inits rendering+physics?
        # TODO FIXME rm in the future when: 1) urdf stage= bug has been fixed 2) users are informed of the behavior
        _kernel.enable_extension("omni.usd")
        await _kernel.omni.usd.get_context().attach_stage_async(stage)

        return Scene(_kernel=_kernel, _usd_stage_ref=stage)


async def build_usd_scene(
    _kernel: Kernel | None = None,
):
    return await USDSceneBuilder()(
        _kernel=_kernel,
    )


class USDLoader:
    class Config(TypedDict):
        path: Optional[PathExpressionLike]
        """Path to the created entity."""
        replace: Optional[bool]
        """Whether to replace the existing entity."""
    
    # TODO ref isaacsim.core.utils.stage.add_reference_to_stage
    async def __call__(
        self,
        resource_or_model: ...,
        scene: Scene,
        config: Config = Config(),
    ):
        pxr = scene._kernel.pxr
        omni = scene._kernel.omni        

        # TODO
        # scene._kernel.enable_extension("omni.usd")
        scene._kernel.enable_extension("omni.usd.metrics.assembler")

        match resource_or_model:
            case str() as resource:
                pass
            # TODO support for Usd prims directly??
            case _:
                raise NotImplementedError(f"TODO {resource_or_model}")
            
        stage = scene._usd_stage

        if config.get("replace", False):
            if config.get("path", None) is not None:
                scene._kernel.enable_extension("omni.usd")
                scene._kernel.enable_extension("omni.kit.commands")

                # TODO call only when not exists
                # TODO check success??
                delete_prims_command = omni.usd.commands.DeletePrimsCommand(
                    [config.get("path")], 
                    stage=scene._usd_stage,
                )
                delete_prims_command.do()

                # is_success = stage.RemovePrim(config.get("path"))
                # if not is_success:
                #     raise RuntimeError(
                #         f"""Failed to remove loaded USD prim at {config.get("path")}"""
                #     )

        prim_path = _usd_get_or_use_free_path(
            scene=scene, 
            path=config.get("path", None),
        )

        sdf_layer = pxr.Sdf.Layer.FindOrOpen(resource_or_model)
        if not sdf_layer:
            raise RuntimeError(f"Could not get Sdf layer for {resource_or_model}")
        
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            # TODO make prim_type customizable??
            prim = stage.DefinePrim(prim_path, "Xform")

        stage_id = pxr.UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
        ret_val = omni.metrics.assembler.core.get_metrics_assembler_interface().check_layers(
            stage.GetRootLayer().identifier, sdf_layer.identifier, stage_id,
        )
        if ret_val["ret_val"] != 0:
            try:
                scene._kernel.enable_extension("omni.kit.commands")

                payref = pxr.Sdf.Reference(resource_or_model)
                omni.kit.commands.execute("AddReference", stage=stage, prim_path=prim.GetPath(), reference=payref)
            except Exception as error:
                warnings.warn(
                    f"USD reference {resource_or_model} may have divergent units, "
                    f"please either enable extension`omni.usd.metrics.assembler` "
                    f"or convert into right units: {error}"
                )
                is_success = prim.GetReferences().AddReference(resource_or_model)
                if not is_success:
                    raise RuntimeError(f"Invalid USD reference: {resource_or_model}")
        else:
            is_success = prim.GetReferences().AddReference(resource_or_model)
            if not is_success:
                raise RuntimeError(f"Invalid USD reference: {resource_or_model}")

        # TODO FIXME entity: no path roundtrip; ref underlying prim directly
        return Entity(
            path=prim.GetPath().pathString,
            scene=scene,
        )
    
    # TODO
    async def destroy(self, entity: Entity):
        raise NotImplementedError


# TODO usd has two modes: reference and sublayer; 
# this func should handle both; maybe infer from paths? 
async def load_usd(
    resource_or_model: ...,
    scene: Scene,
    config: USDLoader.Config = USDLoader.Config(),
    **config_kwds: Unpack[USDLoader.Config],
) -> Entity:
    return await USDLoader()(
        resource_or_model=resource_or_model, 
        scene=scene, 
        config=USDLoader.Config(config, **config_kwds),
    )


# TODO FIXME reimpl the urdf loader
class URDFLoader:
    class Config(TypedDict):
        path: Optional[PathExpressionLike]
        fix_root_link: Optional[bool]
        num_copies: Optional[int]

    # TODO use scene._usd_current_stage
    async def __call__(
        self, 
        resource_or_model: ...,
        scene: Scene,
        config: Config = Config(),
    ) -> Articulation:
        """
        Load a URDF model into the scene and return its articulation view.

        :param resource_or_model: The URDF resource (file path or model object).
        :param scene: The scene to load the URDF model into.
        :param config: Configuration options for loading the URDF model.
        :return: An :class:`Articulation` representing the loaded URDF model.
        """
        
        omni = scene._kernel.omni
        isaacsim = scene._kernel.isaacsim

        # TODO
        scene._kernel.enable_extension("omni.kit.commands")
        scene._kernel.enable_extension("omni.usd")
        scene._kernel.enable_extension("isaacsim.asset.importer.urdf")
        scene._kernel.enable_extension("isaacsim.core.utils")

        import_config = isaacsim.asset.importer.urdf._urdf.ImportConfig()
        import_config.make_default_prim = False  # Make the robot the default prim in the scene
        import_config.fix_base = config.get("fix_root_link", False) # Fix the base of the robot to the ground
        import_config.merge_fixed_joints = False
        # import_config.convex_decomp = False  # Disable convex decomposition for simplicity
        # import_config.self_collision = False  # Disable self-collision for performance

        if config.get("num_copies", None) is not None:
            raise NotImplementedError("TODO")

        # TODO
        match resource_or_model:
            case str() as resource:
                pass
            case _:
                raise NotImplementedError(f"TODO {resource_or_model}")

        # TODO use interface
        is_success, robot_model = omni.kit.commands.execute(
            "URDFParseFile",
            # TODO
            urdf_path=resource,
            import_config=import_config,
        )
        assert is_success

        prim_path = _usd_get_or_use_free_path(
            scene=scene, 
            path=config.get("path", None),
        )

        stage = scene._usd_stage

        stage_context = omni.usd.get_context_from_stage(scene._usd_stage)
        if stage_context is None:
            warnings.warn(
                f"Failed to acquire context from the USD stage. "
                f"Stage: {scene._usd_stage}"
            )
        else:
            if not stage_context.is_writable():
                warnings.warn(
                    f"The USD stage does not appear to be writable. Crash may result! "
                    f"Stage: {scene._usd_stage}"
                )
        
        # TODO
        # import os
        # import tempfile
        # with tempfile.TemporaryDirectory() as tmpdir:
        #     root_path, filename = os.path.split(os.path.abspath(resource))
        #     # TODO find free
        #     robot_model.name = omni.usd.get_stage_next_free_path(
        #         stage, 
        #         # TODO behavior should be similar to dir??
        #         path="/tmp",
        #         prepend_default_prim=False,
        #     ).strip("/")
        #     urdf_interface = isaacsim.asset.importer.urdf._urdf.acquire_urdf_interface()
        #     prim_path_temp = urdf_interface.import_robot(
        #         assetRoot=root_path,
        #         assetName=filename,
        #         robot=robot_model,
        #         importConfig=import_config,
        #         # TODO this doesnt do anything AT ALL!!!! still the opened stage!!!
        #         # stage=stage.GetEditTarget().GetLayer().identifier,
        #         stage=os.path.join(tmpdir, "robot.usd"),
        #     )
        #     pxr = scene._kernel.pxr
        #     sub_layer = pxr.Sdf.Layer.CreateAnonymous()
        #     sub_layer.TransferContent(
        #         pxr.Sdf.Layer.FindOrOpen(os.path.join(tmpdir, "robot.usd")),
        #     )
        #     stage.GetRootLayer().subLayerPaths.append(sub_layer.identifier)

        @contextlib.contextmanager
        def _undo_unnecessary_stage_changes():
            default_prim_orig = stage.GetDefaultPrim()
            yield
            stage.SetDefaultPrim(default_prim_orig)

        urdf_interface = isaacsim.asset.importer.urdf._urdf.acquire_urdf_interface()
        # FIXME BUG:default-prim: 
        # for some reason when the stage identifier is passed, 
        # `import_config.make_default_prim` is always `True`!
        # seealso: https://github.com/isaac-sim/IsaacSim/blob/21bbdbad07ba31687f2ff71f414e9d21a08e16b8/source/extensions/isaacsim.asset.importer.urdf/plugins/isaacsim.asset.importer.urdf/PluginInterface.cpp#L297
        # FIXME BUG:write-stage: 
        # this also attempts to output the converted assets to `/configuration` which isn't valid 
        # when the identifier is a non-fs path!!
        with _undo_unnecessary_stage_changes():
            prim_path_temp = urdf_interface.import_robot(
                assetRoot="",
                assetName="",
                robot=robot_model,
                importConfig=import_config,
                # TODO this doesnt do anything AT ALL!!!! still the opened stage!!!
                # stage=stage.GetEditTarget().GetLayer().identifier,
            )

        # TODO rm?
        # is_success, prim_path_temp = omni.kit.commands.execute(
        #     "URDFImportRobot",
        #     urdf_robot=robot_model,
        #     import_config=import_config,
        #     # get_articulation_root=True,
        #     # TODO
        #     dest_path=scene._usd_current_stage.GetEditTarget().GetLayer().identifier,
        # )
        # assert is_success

        # TODO stage
        if prim_path is None:
            prim_path = prim_path_temp
        else:
            # TODO FIXME: this also requires an active stage context??
            is_success, _ = omni.kit.commands.execute(
                "MovePrim",
                path_from=prim_path_temp,
                path_to=prim_path,
                stage_or_context=stage,
            )
            assert is_success        

        # TODO
        # TODO use this instead of get_articulation_root??
        """
        from isaacsim.core.utils.prims import (
            get_articulation_root_api_prim_path,
            get_prim_at_path,
            get_prim_parent,
            get_prim_property,
            set_prim_property,
        )
        """
        return Articulation(
            # TODO stage
            scene._kernel.isaacsim.core.utils.prims
                .get_articulation_root_api_prim_path(prim_path), 
            scene=scene,
        )


async def load_urdf(
    resource_or_model: ...,
    scene: Scene,
    config: URDFLoader.Config = URDFLoader.Config(),
    **config_kwds: Unpack[URDFLoader.Config]
) -> Articulation:
    return await URDFLoader()(
        scene=scene, 
        resource_or_model=resource_or_model, 
        config=URDFLoader.Config(config, **config_kwds),
    )


# TODO impl
class MaterialBuilder:
    async def create(self):
        raise NotImplementedError


# TODO
from robotodo.engines.core.entity import EntityBodyKind
from robotodo.engines.isaac._utils import USDRigidHelper, USDSurfaceDeformableHelper


BuildableGeometry = Plane | Box | Sphere | PolygonMesh

class EntityBuilder:
    # TODO batch
    class Config(TypedDict):
        # TODO
        body_kind: Optional[EntityBodyKind]
        geometry: Optional[BuildableGeometry]
        # material: ...

    class CreateConfig(Config, TypedDict):
        path: Optional[PathExpressionLike]
        replace: Optional[bool]
        num_copies: Optional[int]

    async def create(
        self,
        scene: Scene,
        config: CreateConfig = CreateConfig(),
        **config_kwds: Unpack[CreateConfig],
    ):
        config = self.CreateConfig(config, **config_kwds)

        pxr = scene._kernel.pxr
        
        # TODO
        if config.get("num_copies", None) is not None:
            raise NotImplementedError("TODO")
        
        prim_path = usd_ensure_free_path(
            stage=scene._usd_stage,
            path=config.get("path", None),
            remove_existing=config.get("replace", False),
            kernel=scene._kernel,
        )

        api = ...
        # TODO batching !!!!
        match config.get("geometry", None):
            case None:
                api = pxr.UsdGeom.Xform.Define(scene._usd_stage, prim_path)

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
                    
            case Sphere() as sphere:
                api = pxr.UsdGeom.Sphere.Define(scene._usd_stage, prim_path)
                # TODO batch 
                api.CreateRadiusAttr().Set(sphere.radius)
                
            case PolygonMesh() as polygon_mesh:
                # TODO
                api = pxr.UsdGeom.Mesh.Define(scene._usd_stage, prim_path)
                # TODO cast float/double
                api.CreatePointsAttr().Set(polygon_mesh.vertices)
                api.CreateFaceVertexCountsAttr().Set(polygon_mesh.face_vertex_counts)
                api.CreateFaceVertexIndicesAttr().Set(polygon_mesh.face_vertex_indices)

            case _ as unknown:
                raise NotImplementedError(f"Unsupported geometry: {unknown}")
            
        prim = api.GetPrim()

        match config.get("body_kind", None):
            case None:
                pass
            case EntityBodyKind.RIGID:
                USDRigidHelper([prim], kernel=scene._kernel).apply()
            # TODO ensure mesh
            case EntityBodyKind.DEFORMABLE_SURFACE:
                USDSurfaceDeformableHelper([prim], kernel=scene._kernel).apply()
            case EntityBodyKind.DEFORMABLE_VOLUME:
                raise NotImplementedError
            case _:
                raise NotImplementedError("TODO")        
            
        # TODO FIXME entity: no path roundtrip; ref underlying prim directly
        return Entity(path=prim.GetPath().pathString, scene=scene)
    
    # TODO
    async def destroy(self, entity: Entity):
        raise NotImplementedError
    
    class ResetConfig(Config, TypedDict):
        path: Optional[PathExpressionLike]
        num_copies: Optional[int]

    # TODO async def
    def reset(
        self, 
        entity: Entity,
        config: ResetConfig = ResetConfig(),
        **config_kwds: Unpack[ResetConfig],
    ):
        config = self.ResetConfig(config, **config_kwds)

        # TODO
        if config.get("geometry", None) is not None:
            raise NotImplementedError("TODO")

        match config.get("body_kind", None):
            case None:
                pass

            case EntityBodyKind.RIGID:
                pxr = entity._scene._kernel.pxr

                rigid_helper = USDRigidHelper([
                    child_prim
                    for prim in entity._usd_prims
                    for child_prim in pxr.Usd.PrimRange(
                        prim, 
                        # TODO
                        # pxr.Usd.TraverseInstanceProxies(
                        #     pxr.Usd.PrimAllPrimsPredicate
                        # ),
                    )
                ], kernel=entity._scene._kernel)
                rigid_helper.apply()

            case EntityBodyKind.DEFORMABLE_SURFACE:
                pxr = entity._scene._kernel.pxr

                # TODO convert to mesh if non-mesh
                deformable_helper = USDSurfaceDeformableHelper([
                    child_prim
                    for prim in entity._usd_prims
                    for child_prim in pxr.Usd.PrimRange(
                        prim, 
                        # TODO
                        # pxr.Usd.TraverseInstanceProxies(
                        #     pxr.Usd.PrimAllPrimsPredicate
                        # ),
                    )
                ], kernel=entity._scene._kernel)

                deformable_helper.apply()

            case EntityBodyKind.DEFORMABLE_VOLUME:
                raise NotImplementedError("TODO")
            
            case unknown:
                raise ValueError(f"Unknown body kind specified for {entity}: {unknown}")


# TODO
builder = EntityBuilder()


# TODO deprecate????
async def build(
    scene: Scene,
    config: EntityBuilder.CreateConfig = EntityBuilder.CreateConfig(),
    **config_kwds: Unpack[EntityBuilder.CreateConfig]
):
    return await EntityBuilder().create(
        scene=scene,
        config=EntityBuilder.CreateConfig(config, **config_kwds),
    )


class CameraBuilder:
    class Config(TypedDict):
        path: Optional[PathExpressionLike]
        num_copies: Optional[int]

    async def __call__(
        self,
        scene: Scene,
        config: Config = Config(),
    ):
        pxr = scene._kernel.pxr

        prim_path = _usd_get_or_use_free_path(
            scene=scene, 
            path=config.get("path", None),
        )
        prim = pxr.UsdGeom.Camera.Define(scene._usd_stage, prim_path)

        # TODO FIXME entity: no path roundtrip; ref underlying prim directly
        return Camera(prim.GetPath().pathString, scene=scene)

    # TODO
    async def destroy(self, camera: Camera):
        raise NotImplementedError


# TODO ref omni.replicator.core.create.camera
async def build_camera(
    scene: Scene,
    config: CameraBuilder.Config = CameraBuilder.Config(),
    **config_kwds: Unpack[CameraBuilder.Config],
):
    return await CameraBuilder()(
        scene=scene,
        config=CameraBuilder.Config(config, **config_kwds),
    )