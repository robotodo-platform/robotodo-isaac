
import contextlib
import warnings
from typing import Optional, TypedDict, Unpack

from robotodo.engines.core import PathExpressionLike

from ._kernel import Kernel
from .scene import Scene
from .articulation import Articulation
from .entity import Entity


class USDLoader:
    class Config(TypedDict):
        path: Optional[PathExpressionLike]

    # TODO rm
    # # TODO use scene._usd_current_stage
    # async def __call__(
    #     self,
    #     resource_or_model: ...,
    #     scene: Scene,
    #     config: Config = Config(),
    # ):
        
    #     match resource_or_model:
    #         case str() as resource:
    #             pass
    #         # TODO support for Usd prims directly??
    #         case _:
    #             raise NotImplementedError(f"TODO {resource_or_model}")
        
    #     prim_path = config.get("path", None)
    #     if prim_path is None:
    #         prim_path = scene._kernel.omni.usd.get_stage_next_free_path(
    #             scene._usd_current_stage, 
    #             path="/",
    #             prepend_default_prim=False,
    #         )

    #     if prim_path is not None:
    #         if scene._usd_current_stage.GetPrimAtPath(prim_path).IsValid():
    #             raise RuntimeError(f"Path already exists: {prim_path}")

    #     # TODO add to scene._usd_current_stage!!!!
    #     prim = scene._kernel.isaacsim.core.utils.stage \
    #         .add_reference_to_stage(
    #             usd_path=resource,
    #             prim_path=prim_path,
    #         )
        
    #     # TODO FIXME upstream entity: no path roundtrip; ref underlying prim directly
    #     return Entity(
    #         path=prim.GetPath().pathString,
    #         scene=scene,
    #     )
    
    async def __call__(
        self,
        resource_or_model: ...,
        scene: Scene,
        config: Config = Config(),
    ):
        pxr = scene._kernel.pxr
        omni = scene._kernel.omni        

        # TODO
        scene._kernel.enable_extension("omni.usd")
        scene._kernel.enable_extension("omni.usd.metrics.assembler")

        match resource_or_model:
            case str() as resource:
                pass
            # TODO support for Usd prims directly??
            case _:
                raise NotImplementedError(f"TODO {resource_or_model}")
        
        prim_path = config.get("path", None)
        if prim_path is None:
            prim_path = omni.usd.get_stage_next_free_path(
                scene._usd_stage, 
                path="/",
                prepend_default_prim=False,
            )

        if prim_path is not None:
            if scene._usd_stage.GetPrimAtPath(prim_path).IsValid():
                raise RuntimeError(f"Path already exists: {prim_path}")

        stage = scene._usd_stage
            
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

        # TODO FIXME upstream entity: no path roundtrip; ref underlying prim directly
        return Entity(
            path=prim.GetPath().pathString,
            scene=scene,
        )    


# TODO usd has two modes: reference and sublayer; 
# this func should handle both; maybe infer from paths? 
async def load_usd(
    resource_or_model: ...,
    scene: Scene,
    config: USDLoader.Config = USDLoader.Config(),
    **config_kwds: Unpack[USDLoader.Config],
):
    return await USDLoader()(
        resource_or_model=resource_or_model, 
        scene=scene, 
        config=USDLoader.Config(config, **config_kwds),
    )



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

        prim_path = config.get("path", None)
        if prim_path is None:
            prim_path = omni.usd.get_stage_next_free_path(
                scene._usd_stage, 
                path="/",
                prepend_default_prim=False,
            )

        if prim_path is not None:
            if scene._usd_stage.GetPrimAtPath(prim_path).IsValid():
                raise RuntimeError(f"Path already exists: {prim_path}")
            
        stage_context = omni.usd.get_context_from_stage(scene._usd_stage)
        if stage_context is None:
            raise RuntimeError(
                f"The USD stage is invalid. "
                f"Stage: {scene._usd_stage}"
            )
        if not stage_context.is_writable():
            raise RuntimeError(
                f"The USD stage does not appear to be writable. Crash may result! "
                f"Stage: {scene._usd_stage}"
            )
        
        @contextlib.contextmanager
        def _undo_unnecessary_stage_changes():
            stage = stage_context.get_stage()
            
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
                stage=stage_context.get_stage().GetEditTarget().GetLayer().identifier,
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

        if prim_path is None:
            prim_path = prim_path_temp
        else:
            is_success, _ = omni.kit.commands.execute(
                "MovePrim",
                path_from=prim_path_temp,
                path_to=prim_path,
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


class USDSceneLoader:
    async def __call__(
        self,
        resource_or_model: ... = None,
        # TODO mv engine??
        _kernel: Kernel = None,
    ) -> Scene:
        if _kernel is None:
            raise NotImplementedError("TODO")

        # TODO
        omni = _kernel.omni
        _kernel.enable_extension("omni.usd")

        ctx = omni.usd.get_context()
        while True:
            if ctx.can_open_stage():
                break
            await ctx.next_stage_event_async()

        # TODO NOTE current impl opens model as sublayer: prob safest??
        is_success, message = await ctx.new_stage_async()
        if not is_success:
            raise RuntimeError(f"Failed to create empty USD scene: {message}")
        stage = ctx.get_stage()
        if stage is None:
            # TODO
            raise RuntimeError("TODO")
        stage.GetRootLayer().subLayerPaths.append(resource_or_model)

        return Scene(_kernel=_kernel, _usd_stage_ref=stage)


        # TODO
        # is_success, message = await ctx.open_stage_async(resource_or_model)
        # if not is_success:
        #     raise RuntimeError(f"Failed to load USD scene {resource_or_model}: {message}")

        # stage = ctx.get_stage()
        # # TODO check None
        # # TODO
        # return Scene(_kernel=_kernel, _usd_stage=stage)
    

async def load_usd_scene(
    resource_or_model: ...,
    _kernel: Kernel,
):
    return await USDSceneLoader()(
        resource_or_model=resource_or_model,
        _kernel=_kernel,
    )