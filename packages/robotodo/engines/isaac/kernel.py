# SPDX-License-Identifier: Apache-2.0

"""
Kernel.
"""


import warnings
import os
import contextlib
import asyncio
import functools
import threading
import concurrent.futures
import queue
from typing import TypedDict, NotRequired, Unpack, Awaitable, Callable, Any

import nest_asyncio
from robotodo.engines.isaac._utils.io import redirect_stdout_libc


def _omni_create_app(argv: list[str] = []) -> "omni.kit.app.IApp":
    """
    TODO doc
    """

    @contextlib.contextmanager
    def _omni_undo_monkeypatching():
        """
        Undo the unnecessary monkey-patching 
        of builtin Python modules done by `omni.kit.app`.

        .. seealso::
            * :func:`_startup_kit_scripting` in 
            `site-packages/omni/kernel/py/omni/kit/app/_impl/__init__.py`
        """

        import sys
        import logging
        import asyncio

        meta_path_orig = list(sys.meta_path)
        logging_handlers_orig = list(logging.root.handlers)
        logging_level_orig = int(logging.root.level)
        asyncio_run_orig = asyncio.run

        yield

        sys.meta_path = [*meta_path_orig, *sys.meta_path]
        logging.root.handlers = logging_handlers_orig
        logging.root.level = logging_level_orig
        # asyncio.run = asyncio_run_orig

    with _omni_undo_monkeypatching():
        try:
            import isaacsim
            isaacsim.bootstrap_kernel()
        except Exception as error:
            raise RuntimeError(
                f"Failed to load the isaacsim kernel. To possibly resolve the problem: "
                f"- If you are in a Conda environment, run `conda install 'libstdcxx>11'`. "
                f"- If you are using Linux, also run `ldd --version` to check your GLIC version "
                f"and upgrade your Linux distribution if the version is below 2.35. "
                f"For more information, see https://robotodo-isaac.readthedocs.io/en/latest/content/manuals/tutorial-000_installation.html"
            ) from error

        try:
            import isaacsim.kit.kit_app
            kit_app = isaacsim.kit.kit_app.KitApp()
        except ImportError as error:
            import warnings
            warnings.warn(f"TODO: {error}")
            import omni.kit_app
            omni.kit_app.bootstrap_kernel()
            kit_app = omni.kit_app.KitApp()

        # TODO
        # nest_asyncio.apply()
        nest_asyncio._patch_loop(asyncio.get_event_loop())

        kit_app.startup([
            *argv,
            # TODO necesito?
            # extensions: so we can reference other kit files  
            # "--ext-folder", os.path.abspath(os.path.join(os.environ["ISAAC_PATH"], "exts")),
            # "--ext-folder", os.path.abspath(os.path.join(os.environ["ISAAC_PATH"], "apps")),
            # 
            "--allow-root",
            "--/app/installSignalHandlers=false",
            "--/app/python/interceptSysStdOutput=false",
            "--/app/python/interceptSysExit=false",
            "--/app/python/logSysStdOutput=false",
            "--/app/python/enableGCProfiling=false",
            "--/app/python/disableGCDuringStartup=false",
            # NOTE required for GUI and MUST be enabled during startup (NOT later!!)
            # "isaacsim.exp.*" extensions seem to rely on this for `.update` to work
            "--enable", "omni.kit.loop-isaac",
        ])

        # TODO
        app = kit_app._app

        return app


class KernelConfig(TypedDict):
    verbose: NotRequired[bool]
    _omni_config_path: NotRequired[str]
    _omni_extra_argv: NotRequired[list[str]]


class Kernel:
    """
    TODO doc
    """

    _config: KernelConfig

    def __init__(self):
        self._config = KernelConfig()

    def configure(
        self,
        config: KernelConfig = KernelConfig(), 
        **config_kwds: Unpack[KernelConfig],
    ):
        self._config = KernelConfig({
            **self._config,
            **config,
            **config_kwds,
        })
        return self
    
    @property
    def config(self):
        return self._config

    @functools.cached_property
    def _app_cache(self):
        """
        TODO doc
        """

        # NOTE HACK to suppress GUI window during startup
        def _omni_suppress_appwindow(app: "omni.kit.app.IApp"):
            extension_manager = app.get_extension_manager()

            # TODO
            import carb.settings
            settings = carb.settings.get_settings()
            settings.set("/exts/omni.appwindow/autocreateAppWindow", False)

            for extension_name in [
                "omni.appwindow",
                "carb.windowing.plugins",
                "omni.kit.renderer.core",
            ]:
                extension_manager.set_extension_enabled(extension_name, True)
            extension_manager.process_and_apply_all_changes()

            import carb.windowing
            import omni.appwindow

            try:
                carb_windowing_iface = carb.windowing.acquire_windowing_interface()
            except Exception as error:
                warnings.warn(f"TODO: {error}")
                settings.set("/app/window/enabled", False)
                carb_windowing_iface = None
            omni_app_window_factory = omni.appwindow.acquire_app_window_factory_interface()

            app_window = omni_app_window_factory.get_app_window()
            if app_window is None:
                app_window = omni_app_window_factory.create_window_from_settings()
                app_window.startup()
                omni_app_window_factory.set_default_window(app_window)

            renderer = omni.kit.renderer.bind.acquire_renderer_interface()
            renderer.startup()
            # TODO NOTE this flashes the window (by 1 frame?)
            renderer.attach_app_window(app_window)
            # renderer.freeze_app_window(app_window, True)

            carb_app_window = app_window.get_window()
            if carb_app_window is not None:
                if carb_windowing_iface is not None:
                    carb_windowing_iface.hide_window(carb_app_window)

        # NOTE HACK to close all windows in workspace post startup
        def _omni_suppress_subwindows():
            import omni.kit.app

            app = omni.kit.app.acquire_app_interface()
            extension_manager = app.get_extension_manager()
            extension_manager.set_extension_enabled_immediate("omni.ui", True)

            # TODO
            import omni.ui

            for window in omni.ui.Workspace.get_windows():
                window.visible = False

        kit_path = self._config.get("_omni_config_path", None)
        extra_argv = self._config.get("_omni_extra_argv", None)

        argv = []
        if kit_path is not None:
            # argv.append(os.path.abspath(os.path.join(os.environ["EXP_PATH"], kit_path)))
            argv.append(os.path.abspath(kit_path))
        argv.extend([
            #
            "--/app/name=robotodo.engines.isaac",

            #
            "--portable", # run as portable to prevent writing extra files to user directory
            "--reset-user",

            # "--/app/runLoops/main/rateLimitEnabled=false",
            # "--/app/runLoops/rendering_0/rateLimitEnabled=false",
            # "--/app/runLoops/present/rateLimitEnabled=false",

            # TODO
            "--/exts/omni.appwindow/autocreateAppWindow=false",

            "--/app/docks/disabled=true",
            "--/app/window/hideUi=true", # TODO

            # logging
            "--/log/file=",
            "--/log/outputStream=stderr",
            "--/log/outputStreamLevel=fatal",
            "--/log/async=true",
            "--/app/enableStdoutOutput=false",

            # TODO NOTE ensure rendering syncd with .step
            # "--/app/asyncRendering=false",
            # TODO NOTE make users reset manually??
            # TODO https://docs.omniverse.nvidia.com/kit/docs/omni_physics/108.0/dev_guide/settings.html#_CPPv419kSettingResetOnStop
            # "--/physics/resetOnStop=false",

            "--/foundation/verifyOsVersion/enabled=false",
            
            "--/persistent/renderer/startupMessageDisplayed=true",
            
            "--/app/content/emptyStageOnStart=false",

            # TODO
            "--/app/viewport/defaults/fillViewport=true",
            "--/app/viewport/defaults/hud/renderResolution/visible=true",
            "--/app/viewport/defaults/hud/renderFPS/visible=true",                
            # "--/persistent/app/viewport/pickOccluded=true",
            "--/persistent/app/viewport/camMoveVelocity=0.05",
            "--/persistent/app/viewport/gizmo/scale=0.01",
            "--/persistent/app/viewport/grid/scale=1.0",
            "--/persistent/app/primCreation/typedDefaults/camera/clippingRange=[0.01,10000000.0]",
            "--/persistent/app/primCreation/typedDefaults/camera/focusDistance=10000000.0",
            "--/persistent/app/primCreation/typedDefaults/orthoCamera/clippingRange=[0.01,10000000.0]",
            "--/persistent/app/primCreation/typedDefaults/orthoCamera/focusDistance=10000000.0",

            "--/exts/omni.kit.viewport.window/startup/showOnLaunch=false",
            "--/exts/omni.kit.viewport.window/startup/disableWindowOnLoad=true",
            "--/exts/omni.kit.viewport.window/startup/dockTabInvisible=false",
            "--/exts/omni.kit.viewport.window/startup/singleTabGroup=true",

            # TODO
            # "--/exts/omni.kit.async_engine/updateSubscriptionOrder=0",
            "--/exts/omni.kit.async_engine/keep_loop_running=false",
            # TODO NOTE unpatched `asyncio.run` will not return when this is false?
            "--/exts/omni.kit.async_engine/spinLoopOncePerUpdate=true",

            # TODO
            # "--/app/asyncRendering=true",
            # "--/app/asyncRenderingLowLatency=true",
            # "--/app/asyncRendering=false",
            # "--/app/asyncRenderingLowLatency=false",
            # TODO NOTE causes issues with omni.replicator.core
            # "--/renderer/asyncInit=true",

            # # "--/exts/omni.replicator.core/numFrames=1",
            "--/omni/replicator/captureOnPlay=false",
            "--/app/settings/fabricDefaultStageFrameHistoryCount=3",
            "--/exts/omni.replicator.core/Orchestrator/enabled=false",

            # TODO
            "--enable", "usdrt.scenegraph",
            "--enable", "omni.hydra.usdrt_delegate",
            "--/app/useFabricSceneDelegate=true",
            "--/rtx/hydra/readTransformsFromFabricInRenderDelegate=true",
        ])
        if extra_argv is not None:
            argv.extend(extra_argv)

        with contextlib.ExitStack() as contexts:
            if not self._config.get("verbose", False):
                contexts.enter_context(redirect_stdout_libc(None))
                contexts.enter_context(contextlib.redirect_stdout(None))

            app = _omni_create_app(argv)
            # TODO
            # _omni_suppress_appwindow()

            extension_manager = app.get_extension_manager()

            # TODO
            def _on_appwindow_requested():
                _omni_suppress_appwindow(app)
                # TODO ui: menu needs to be created before everything else??
                extension_manager.set_extension_enabled("omni.kit.menu.utils", True)
            # TODO
            self._todo = extension_manager.subscribe_to_extension_enable(
                on_enable_fn=lambda _: _on_appwindow_requested(),
                ext_name="omni.appwindow",
            )

            for extension_name in [
                # NOTE required for functionality due to some unknown bugs.
                # Some must be enabled during startup
                "omni.physics.physx",
                "omni.physics.stageupdate",
                # "omni.physx.tensors",
                # "omni.hydra.rtx",
                # "omni.hydra.pxr",
                # TODO
                # "omni.replicator.core",
                # TODO why?
                "omni.graph.core",
                # TODO ui: menu needs to be created before everything else??
                # "omni.kit.menu.utils",
                # "omni.kit.manipulator.camera", # NOTE prevent error messages when missing
                # TODO HACK right now this just keeps the GUI layout working
                # "isaacsim.asset.importer.urdf",
                # TODO NOTE required for docking to work?
                # "omni.kit.mainwindow",
                # TODO necesito?
                "omni.usd.schema.metrics.assembler",
            ]:
                extension_manager.set_extension_enabled(extension_name, True)
            extension_manager.process_and_apply_all_changes()

            # TODO
            # _omni_suppress_subwindows()

            # TODO
            while not app.is_app_ready():
                app.update()

        # TODO
        def should_invalidate():
            return not app.is_running()
        return app, should_invalidate

    @property
    def _app(self):
        while True:
            app, should_invalidate = self._app_cache
            if should_invalidate():
                del self._app_cache
            else:
                return app

    @functools.cached_property
    def _carb(self):
        # TODO
        self._app
        return __import__("carb")

    @functools.cached_property
    def _omni(self):
        # TODO !!!!
        self._app
        return __import__("omni")

    @functools.cached_property
    def _pxr(self):
        # TODO !!!!
        self._app
        self._omni_enable_extension("omni.usd.libs")
        return __import__("pxr")
    
    @functools.cached_property
    def _usdrt(self):
        # TODO !!!!
        self._app
        self._omni_enable_extensions([
            "usdrt.scenegraph",
            "omni.hydra.usdrt_delegate",
        ])
        return __import__("usdrt")

    def _omni_enable_extension(
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

    def _omni_enable_extensions(
        self, 
        names: list[str], 
        enabled: bool = True,
        immediate: bool = True,
    ):
        ext_manager = self._app.get_extension_manager()
        for name in names:
            ext_manager.set_extension_enabled(name, enabled)
        if immediate:
            # TODO FIXME perf: this is expensive ~2ms
            ext_manager.process_and_apply_all_changes()

    def _omni_import_module(self, module: str):
        return __import__(module)

    class _Runner:
        def __init__(self, app: ...):
            self._app = app
            self._ref_count = 0

        # TODO
        # @functools.cached_property
        # def _cached_future(self):
        #     async def _impl():
        #         while self._ref_count > 0:
        #             await asyncio.sleep(0)
        #             self._app.update()
        #     return asyncio.create_task(_impl())
        
        # TODO
        @functools.cached_property
        def _cached_future(self):
            future = asyncio.Future()
            loop = future.get_loop()

            def f():
                if self._ref_count <= 0:
                    future.set_result(None)
                if future.done():
                    return
                # if not app.is_running():
                #     return
                self._app.update()
                loop.call_soon(f)
            
            loop.call_soon_threadsafe(f)
            return future
            
        def _ensure_future(self):
            future = self._cached_future
            if future.done():
                del self._cached_future
            return self._cached_future

        def acquire(self):
            self._ref_count += 1
            self._ensure_future()

        def release(self):
            if self._ref_count > 0:
                self._ref_count -= 1

    # TODO invalidate
    @functools.cached_property
    def _runner(self):
        return self._Runner(self._app)

    # TODO
    def _omni_ensure_future(
        self,
        coroutine: Awaitable,
    ) -> asyncio.Future:
        omni = self._omni
        self._omni_enable_extension("omni.kit.async_engine")

        # TODO
        task_or_future = omni.kit.async_engine.run_coroutine(coroutine)
        self._runner.acquire()
        def _done_callback(_future: ...):
            self._runner.release()
            task_or_future.remove_done_callback(_done_callback)
        task_or_future.add_done_callback(_done_callback)

        match task_or_future:
            case _ if asyncio.isfuture(task_or_future):
                return task_or_future
            case concurrent.futures.Future():
                return asyncio.wrap_future(task_or_future)
            case _:
                # TODO
                raise ValueError(f"TODO: {task_or_future}")


default_kernel = Kernel()
"""
TODO doc
Default kernel.
"""


# TODO deprecate
def get_running_kernel():
    # default_kernel.run_forever()
    return default_kernel