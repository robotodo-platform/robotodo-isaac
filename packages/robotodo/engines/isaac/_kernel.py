import os
import contextlib
import asyncio
import functools

import nest_asyncio


# TODO next
class Kernel:
    """
    TODO doc
    
    """

    def __init__(
        self, 
        # TODO
        kit_path: str | None = None,         
        extra_argv: list[str] | None = [],
        loop: asyncio.AbstractEventLoop | None = None,
    ):
        """
        Create an Isaac Sim "kernel".
        At its core, this starts an Isaac Sim Omniverse App in the background.
        
        :param extra_argv:
            Extra CLI arguments to pass to the Isaac Sim Omniverse App.

            Common arguments include:

            * :code:`--help`: Show help message.
            * :code:`--list-exts`: List available extensions that can be `--enable`d.
            * :code:`--no-window`: Switch to headless mode (disables window).
            * :code:`--/app/window/hideUi=True`: Hide all UI elements (still opens a window).

            .. seealso::
                * :class:`isaacsim.simulation_app.AppFramework`
                * :class:`isaacsim.simulation_app.SimulationApp`
                * https://docs.omniverse.nvidia.com/kit/docs/kit-manual/108.0.0/guide/configuring.html#kit-kernel-settings

        :param kit_path: 
            Path to the kit file (ends in `.kit`, aka app config) that defines an Omniverse App.
            Unspecified by default.
            
            .. seealso::
                * https://docs.omniverse.nvidia.com/kit/docs/kit-manual/108.0.0/guide/creating_kit_apps.html

        :param loop:
            The `asyncio` loop to use for all async operations.
            Defaults to the current running loop.

        Examples:

        .. code-block:: python
            
            # selectively enable some extensions during startup
            Kernel(["--enable", "isaacsim.exp.full"])

        """

        if loop is None:
            try:
                # NOTE this ensures a loop is present beforehand 
                # to prevent isaacsim from "stealing" the default loop                
                loop = asyncio.get_running_loop()
            except Exception as error:
                raise RuntimeError(
                    "Failed to get running asyncio loop. "
                    "Start one with `asyncio.new_event_loop()`"
                ) from error
        self._loop = loop

        nest_asyncio.apply()

        @contextlib.contextmanager
        def _undo_omni_monkeypatching():
            """
            Undo the unnecessary monkey-patching 
            of builtin Python modules done by `omni.kit.app`.

            .. seealso::
                * :func:`_startup_kit_scripting` in 
                `site-packages/omni/kernel/py/omni/kit/app/_impl/__init__.py`
            """

            import sys
            import logging

            meta_path_orig = list(sys.meta_path)
            logging_handlers_orig = list(logging.root.handlers)
            logging_level_orig = int(logging.root.level)

            yield

            sys.meta_path = [*meta_path_orig, *sys.meta_path]
            logging.root.handlers = logging_handlers_orig
            logging.root.level = logging_level_orig

        with _undo_omni_monkeypatching():
            # import isaacsim
            # isaacsim.bootstrap_kernel()
            # from isaacsim.simulation_app import AppFramework

            import isaacsim.kit.kit_app

            # import omni.kit_app
            # omni.kit_app.bootstrap_kernel()

            # TODO
            # kit_app = omni.kit_app.KitApp()
            kit_app = isaacsim.kit.kit_app.KitApp()

            argv = []
            if kit_path is not None:
                # argv.append(os.path.abspath(os.path.join(os.environ["EXP_PATH"], kit_path)))
                argv.append(os.path.abspath(kit_path))
            argv.extend([
                # TODO necesito?
                # extensions: so we can reference other kit files  
                # "--ext-folder", os.path.abspath(os.path.join(os.environ["ISAAC_PATH"], "exts")),
                # "--ext-folder", os.path.abspath(os.path.join(os.environ["ISAAC_PATH"], "apps")),
                #
                "--/app/name=robotodo.engines.isaac",
                # 
                "--/app/installSignalHandlers=false",
                "--/app/python/interceptSysStdOutput=false",
                "--/app/python/interceptSysExit=false",
                "--/app/python/logSysStdOutput=false",
                "--/app/python/enableGCProfiling=false",
                "--/app/python/disableGCDuringStartup=false",
                # NOTE required for GUI and MUST be enabled during startup (NOT later!!)
                # "isaacsim.exp.*" extensions seem to rely on this for `.update` to work
                "--enable", "omni.kit.loop-isaac",
                # NOTE required for functionality due to some unknown bugs.
                # Some must be enabled during startup
                "--enable", "omni.physics.physx",
                "--enable", "omni.physics.stageupdate",
                "--enable", "omni.physx.tensors",
                "--enable", "omni.hydra.rtx",
                "--enable", "omni.replicator.core",
                "--enable", "omni.kit.manipulator.camera", # NOTE prevent error messages when missing
                # optional ux improvement
                "--portable", # run as portable to prevent writing extra files to user directory
                "--/app/window/hideUi=true", # TODO
                "--/app/enableStdoutOutput=false",
                # logging
                "--/log/file=",
                "--/log/outputStreamLevel=Error",
                "--/log/async=true",
                # TODO NOTE ensure rendering syncd with .step
                "--/app/asyncRendering=false",
                # TODO NOTE make users reset manually??
                # TODO https://docs.omniverse.nvidia.com/kit/docs/omni_physics/108.0/dev_guide/settings.html#_CPPv419kSettingResetOnStop
                # "--/physics/resetOnStop=false",
                "--/persistent/renderer/startupMessageDisplayed=true",
            ])
            if extra_argv is not None:
                argv.extend(extra_argv)

            kit_app.startup(argv)
            # kit_app = AppFramework(argv=argv).app

            # TODO
            while not kit_app._app.is_app_ready():
                kit_app._app.update()

            self._app = kit_app._app

        # TODO 
        self._should_run_app_loop = None

    @functools.cached_property
    def carb(self):
        return __import__("carb")

    # @functools.cached_property
    # def isaacsim(self):
    #     # TODO enable ext on demand !!!!
    #     return __import__("isaacsim")

    @functools.cached_property
    def omni(self):
        # TODO !!!!
        return __import__("omni")

    @functools.cached_property
    def pxr(self):
        # TODO !!!!
        self.enable_extension("omni.usd.libs")
        return __import__("pxr")
    
    @property
    def loop(self):
        return self._loop
    
    def start_app_loop_soon(self):
        def f():
            if not self._should_run_app_loop:
                return
            try:
                self._app.update()
            finally:
                self._loop.call_soon_threadsafe(f)
        
        self._should_run_app_loop = True
        self._loop.call_soon_threadsafe(f)

    def stop_app_loop_soon(self):
        self._should_run_app_loop = False

    def step_app_loop_soon(self):
        def f():
            if not self._should_run_app_loop:
                return
            self._app.update()
        self._loop.call_soon_threadsafe(f)
        
    def step_app_loop(self):
        self._app.update()

    def shutdown_app(self):
        self._app.shutdown()

    @property
    def app(self):
        return self._app
    
    def get_settings(self):
        # TODO
        return self.carb.settings.get_settings()

    def enable_extension(
        self, 
        name: str, 
        enabled: bool = True,
        immediate: bool = True,
    ):
        ext_manager = self._app.get_extension_manager()
        if ext_manager.is_extension_enabled(name) is enabled:
            return
        if immediate:
            ext_manager.set_extension_enabled_immediate(name, enabled)
        else:
            ext_manager.set_extension_enabled(name, enabled)

    def enable_extensions(
        self, 
        names: list[str], 
        enabled: bool = True,
        immediate: bool = True,
    ):
        ext_manager = self._app.get_extension_manager()
        for name in names:
            ext_manager.set_extension_enabled(name, enabled)
        if immediate:
            ext_manager.process_and_apply_all_changes()

    def import_module(self, module: str):
        return __import__(module)



# TODO
def enable_physx_deformable_beta(kernel: Kernel):
    """
    TODO doc

    :param kernel: ...
    """

    import warnings

    omni = kernel.omni
    kernel.enable_extension("omni.physx")

    settings = kernel.get_settings()
    SETTING_ENABLE_DEFORMABLE_BETA = omni.physx.bindings._physx.SETTING_ENABLE_DEFORMABLE_BETA

    if not settings.get(SETTING_ENABLE_DEFORMABLE_BETA):
        settings.set(SETTING_ENABLE_DEFORMABLE_BETA, True)
        warnings.warn(
            f"Deformable Schema Beta was requested to be enabled in Omniverse. "
            f"It has now been enabled (Restart may be required for the changes to take effect). "
            f"For details see https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/deformables_beta/deformable_authoring.html#enable-deformable-schema-beta"
        )


# TODO
import functools
from typing import TypedDict, NotRequired, Unpack


class KernelConfig(TypedDict):
    kit_path: NotRequired[str | None]
    extra_argv: NotRequired[list[str] | None]


__kernel_config_default = KernelConfig(
    # kit_path="isaacsim.exp.full.kit",
)
def configure_default_kernel(
    config: KernelConfig = KernelConfig(),
    **config_kwds: Unpack[KernelConfig],
):
    __kernel_config_default.update({
        **config,
        **config_kwds,
    })


@functools.cache
def get_default_kernel():
    # TODO
    kernel = Kernel(
        extra_argv=__kernel_config_default.get("extra_argv", []),
        kit_path=__kernel_config_default.get("kit_path", None),
    )
    kernel.start_app_loop_soon()
    return kernel