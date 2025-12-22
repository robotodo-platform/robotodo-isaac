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
        import isaacsim
        isaacsim.bootstrap_kernel()
        # from isaacsim.simulation_app import AppFramework

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

        # while not app.is_app_ready():
        #     app.update()

        return app


class KernelConfig(TypedDict):
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

            # TODO necesito??
            # carb_app_window = app_window.get_window()
            # if carb_app_window is not None:
            #     if carb_windowing_iface is not None:
            #         carb_windowing_iface.hide_window(carb_app_window)

            renderer = omni.kit.renderer.bind.acquire_renderer_interface()
            renderer.startup()
            # TODO NOTE this flashes the window (by 1 frame?)
            renderer.attach_app_window(app_window)

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

            "--/app/content/emptyStageOnStart=false",

            # TODO
            # "--/exts/omni.replicator.core/Orchestrator/enabled=false",

            # "--/app/runLoops/main/rateLimitEnabled=false",
            # "--/app/runLoops/rendering_0/rateLimitEnabled=false",
            # "--/app/runLoops/present/rateLimitEnabled=false",

            # "--/app/window/hideUi=true", # TODO
            "--/app/enableStdoutOutput=false",
            # logging
            "--/log/file=",
            "--/log/outputStreamLevel=Error",
            "--/log/async=true",

            # TODO NOTE ensure rendering syncd with .step
            # "--/app/asyncRendering=false",
            # TODO NOTE make users reset manually??
            # TODO https://docs.omniverse.nvidia.com/kit/docs/omni_physics/108.0/dev_guide/settings.html#_CPPv419kSettingResetOnStop
            # "--/physics/resetOnStop=false",
            "--/persistent/renderer/startupMessageDisplayed=true",
            
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
            "--/exts/omni.kit.async_engine/spinLoopOncePerUpdate=true",

            # TODO
            # "--/app/asyncRendering=true",
            # "--/app/asyncRenderingLowLatency=true",
            # "--/app/asyncRendering=false",
            # "--/app/asyncRenderingLowLatency=false",
            # TODO NOTE causes issues with omni.replicator.core
            # "--/renderer/asyncInit=true",
            # "--/exts/omni.replicator.core/Orchestrator/enabled=false",
            # # "--/exts/omni.replicator.core/numFrames=1",
            # "--/omni/replicator/captureOnPlay=false",
            # "--/app/settings/fabricDefaultStageFrameHistoryCount=3",
        ])
        if extra_argv is not None:
            argv.extend(extra_argv)

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

    # @functools.cached_property
    # def _run_forever_cache(self):
    #     app = self._app
    #     async def _impl():
    #         # loop = asyncio.get_running_loop()
    #         # TODO 
    #         # while app.is_running():
    #         while True:
    #             # TODO
    #             # print("TODO")
    #             await asyncio.sleep(0)
    #             app.update()
    #             # TODO just app.update() ??
    #             # loop.call_soon_threadsafe(app.update)
    #             # try: app.update()
    #             # finally: continue
    #     # TODO check loop
    #     if not asyncio.get_event_loop().is_running():
    #         warnings.warn(
    #             f"{self.run_forever} called without a running asyncio event loop. "
    #             f"Updates are paused until an event loop is running"
    #             # f"For more information, see TODO"
    #         )
    #     omni = self._omni
    #     self._omni_enable_extension("omni.kit.async_engine")
    #     task = omni.kit.async_engine.run_coroutine(_impl())
    #     # app.update()
    #     # task = asyncio.create_task(_impl())
    #     def should_invalidate():
    #         return task.done()
    #     return task, should_invalidate

    # TODO
    @functools.cached_property
    def _run_forever_cache(self):
        app = self._app
        loop = asyncio.get_event_loop()
        if not loop.is_running():
            warnings.warn(
                f"{self.run_forever} called without a running asyncio event loop. "
                f"Updates are paused until an event loop is running. "
                f"For more information, see TODO"
            )

        future = asyncio.Future(loop=loop)

        def f():
            if future.done():
                return
            # if not app.is_running():
            #     # TODO
            #     print("TODO")
            app.update()
            loop.call_soon_threadsafe(f)
        
        loop.call_soon_threadsafe(f)

        def should_invalidate():
            return future.done()
        return future, should_invalidate

    # TODO spin app loop only when necessary; exit loop when idle: i.e. no tasks beside internal omni tasks
    def run_forever(self):
        while True:
            task, should_invalidate = self._run_forever_cache
            if should_invalidate():
                del self._run_forever_cache
            else:
                return task

    # TODO
    # async def stop(self):
    #     # TODO
    #     # self._app.shutdown()
    #     self.run_forever().cancel()

    # def step(self):
    #     # TODO
    #     self._app.update()

    # 
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
            ext_manager.process_and_apply_all_changes()

    def _omni_import_module(self, module: str):
        return __import__(module)

    # TODO better lifecycle mgmt: detect if kernel is autostepping
    # TODO seealso: https://github.com/isaac-sim/IsaacSim/blob/aa503a9bbf92405bbbcfe5361e1c4a74fe10d689/source/extensions/isaacsim.simulation_app/isaacsim/simulation_app/simulation_app.py#L717
    def _omni_run_coroutine(
        self,
        coroutine: Awaitable,
    ):
        omni = self._omni
        self._omni_enable_extension("omni.kit.async_engine")

        # task_or_future = omni.kit.async_engine.run_coroutine(coroutine)
        # if run_until_complete:
        #     while not task_or_future.done():
        #         # self._app.update()
        #         # TODO
        #         self._callback_queue.put(self._app.update)
        #     # # TODO
        #     # self.run_forever()
        #     # match task_or_future:
        #     #     case _ if asyncio.isfuture(task_or_future):
        #     #         asyncio.get_running_loop() \
        #     #             .run_until_complete(task_or_future)
        #     #     case _:
        #     #         # TODO handle concurrent.futures.Future??
        #     #         raise ValueError(f"TODO: {task_or_future}")
        # return task_or_future

        future = concurrent.futures.Future()

        task_or_future = omni.kit.async_engine.run_coroutine(coroutine)
        # TODO
        if True:
            while not task_or_future.done():
                self._app.update()
                # TODO
            # # TODO
            # self.run_forever()
            # match task_or_future:
            #     case _ if asyncio.isfuture(task_or_future):
            #         asyncio.get_running_loop() \
            #             .run_until_complete(task_or_future)
            #     case _:
            #         # TODO handle concurrent.futures.Future??
            #         raise ValueError(f"TODO: {task_or_future}")
        future.set_result(task_or_future.result())

        # TODO
        return future
    
    # TODO
    def _omni_ensure_future(
        self,
        coroutine: Awaitable,
    ) -> asyncio.Future:
        omni = self._omni
        self._omni_enable_extension("omni.kit.async_engine")

        # TODO add to list instead
        # self.run_forever()

        task_or_future = omni.kit.async_engine.run_coroutine(coroutine)
        # task_or_future = asyncio.ensure_future(coroutine, loop=loop)

        # return task_or_future

        app = self._app
        loop = asyncio.get_event_loop()
        # TODO use ref count instead
        def f():
            if task_or_future.done():
                return
            # TODO 
            # if not app.is_running():
            #     # TODO
            #     print("TODO")
            try: app.update()
            finally: pass
            loop.call_soon_threadsafe(f)
        loop.call_soon_threadsafe(f)

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


# TODO
def get_running_kernel():
    # default_kernel.run_forever()
    return default_kernel