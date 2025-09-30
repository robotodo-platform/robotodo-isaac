import os
import contextlib
import asyncio
import functools

import nest_asyncio


class Kernel:
    """
    TODO doc
    
    """

    def __init__(
        self, 
        extra_argv: list[str] = [],
        # TODO
        kit_path: str | None = None, 
        loop: asyncio.AbstractEventLoop = None,
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

        # NOTE this ensures a loop is present beforehand 
        # to prevent isaacsim from "stealing" the default loop
        self._loop = loop
        if self._loop is None:
            try:
                self._loop = asyncio.get_running_loop()
            except Exception as error:
                raise RuntimeError(
                    "Failed to get running asyncio loop. "
                    "Start one with `asyncio.new_event_loop()`"
                ) from error

        nest_asyncio.apply()

        @contextlib.contextmanager
        def _undo_isaacsim_monkeypatching():
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

        with _undo_isaacsim_monkeypatching():
            import isaacsim
            isaacsim.bootstrap_kernel()

            # # TODO
            # import carb
            # carb_logger = carb.logging.acquire_logging()
            # carb_logger.set_level_threshold(carb.logging.LEVEL_WARN)

            from isaacsim.simulation_app import AppFramework

            # TODO
            # if kit_path is None:
            #     for p in [
            #         "isaacsim.exp.full.kit",
            #         "omni.isaac.sim.python.kit",
            #         "isaacsim.exp.base.python.kit",
            #         "isaacsim.exp.base.kit",
            #     ]:
            #         p = os.path.join(os.environ["EXP_PATH"], p)
            #         if os.path.isfile(p):
            #             kit_path = p
            #             break
            exe_path = os.environ["CARB_APP_PATH"]

            argv = []
            if kit_path is not None:
                argv.append(os.path.abspath(os.path.join(os.environ["EXP_PATH"], kit_path)))
            argv.extend([
                # run as portable to prevent writing extra files to user directory
                "--portable",
                # extensions
                # extensions: adding to json doesn't work
                "--ext-folder", os.path.abspath(os.path.join(os.environ["ISAAC_PATH"], "exts")),
                # extensions: so we can reference other kit files  
                "--ext-folder", os.path.abspath(os.path.join(os.environ["ISAAC_PATH"], "apps")),      
                # ...
                # TODO this is needed so dlss lib is found? 
                f"--/app/tokens/exe-path={os.path.abspath(exe_path)}",
                # "--/app/fastShutdown=true",
                "--/app/installSignalHandlers=false",
                "--/app/python/interceptSysStdOutput=false",
                "--/app/python/interceptSysExit=false",
                "--/app/python/logSysStdOutput=false",
                "--/app/python/enableGCProfiling=false",
                "--/app/python/disableGCDuringStartup=false",
                # TODO
                # "--/app/extensions/fastImporter/enabled=false",
                # logging
                # logging: disable default log file creation
                # "--/log/enabled=false",
                "--/log/file=",
                "--/log/outputStreamLevel=Error",
                "--/log/async=true",
                # NOTE required for GUI and MUST be enabled during startup (NOT later!!)
                # "isaacsim.exp.*" extensions seem to rely on this for `.update` to work
                "--enable", "omni.kit.loop-isaac",
                *extra_argv,
            ])

            self._app_framework = AppFramework(argv=argv)

            # import isaacsim
            # import omni
            # # TODO enable extensions
            # import pxr

            # self._isaacsim = isaacsim
            # self._omni = omni
            # self._pxr = pxr

            # TODO load extensions

        # TODO !!!!!
        self._should_run_app_loop = None

    @functools.cached_property
    def carb(self):
        return __import__("carb")

    @functools.cached_property
    def isaacsim(self):
        # TODO enable ext on demand !!!!
        return __import__("isaacsim")

    @functools.cached_property
    def omni(self):
        # TODO !!!!
        return __import__("omni")

    @functools.cached_property
    def pxr(self):
        # TODO !!!!
        return __import__("pxr")
    
    def start_app_loop_soon(self):
        def f():
            if not self._should_run_app_loop:
                return
            self._app_framework.update()
            self._loop.call_soon(f)
        
        self._should_run_app_loop = True
        self._loop.call_soon(f)

    def stop_app_loop_soon(self):
        self._should_run_app_loop = False

    def step_app_loop(self):
        self._app_framework.update()

    def close_app(self):
        self._app_framework.close()

    @property
    def app(self):
        return self._app_framework.app
    
    def get_settings(self):
        # TODO
        return self.carb.settings.get_settings()

    def enable_extension(self, name: str, enabled: bool = True):
        ext_manager = self._app_framework.app.get_extension_manager()
        if ext_manager.is_extension_enabled(name) is enabled:
            return
        ext_manager.set_extension_enabled_immediate(name, enabled)


import functools


# TODO cache
@functools.cache
def get_default_kernel():
    return Kernel(kit_path="isaacsim.exp.full.kit")