import functools


class _Kernel:
    def __init__(self, config: dict | None = None):
        # TODO !!!! allow external
        import nest_asyncio
        nest_asyncio.apply()

        from isaacsim import bootstrap_kernel
        bootstrap_kernel()

        from isaacsim.simulation_app import SimulationApp
        # TODO
        import os
        self._simulation_app = SimulationApp(
            config, 
            # TODO 
            experience=os.path.join(os.environ["EXP_PATH"], "isaacsim.exp.full.kit"),
        ) 

        from isaacsim.core.api import SimulationContext
        self._simulation_context = SimulationContext()
        self._simulation_context.initialize_physics()

        import omni
        self._omni = omni
        # TODO
        self._omni_physx_simulation_interface = omni.physx.get_physx_simulation_interface()

    @functools.cached_property
    def __physics_view(self):
        if not self._simulation_context.is_playing():
            raise RuntimeError(
                f"Accessing physics view when simulation is stopped. "
                f"Start the simulation first through {self.play}"
            )
        res = self._omni.physics.tensors.create_simulation_view("torch")
        if not res.is_valid:
            raise RuntimeError(f"Created physics view is invalid: {res}")
        return res

    def get_physics_view(self):
        if not self.__physics_view.is_valid:
            del self.__physics_view
        return self.__physics_view
    
    def flush_physics_changes(self):
        self._omni_physx_simulation_interface.flush_changes()

    # TODO async
    def play(self):
        self._simulation_context.play()

    def pause(self):
        self._simulation_context.pause()

    def step(self):
        self._simulation_context.step(render=False)
        # TODO instead: async
        # self._physx_sim_interface.simulate(self.get_physics_dt(), current_time)
        # self._physx_sim_interface.fetch_results() # blocking

    def render(self):
        self._simulation_context.render()

    def stop(self):
        self._simulation_context.stop()

    def is_playing(self):
        return self._simulation_context.is_playing()


# TODO
class TestKernel:
    def test_physics_view(self):
        kernel = _Kernel()

        assert kernel.get_physics_view().is_valid

        kernel.stop()
        # TODO assert raise
        # kernel.physics_view

        kernel.play()
        assert kernel.get_physics_view().is_valid






# TODO
import contextlib
import threading
import queue
import concurrent.futures
from typing import Any, Awaitable, Callable, TypeVar

# TODO
_T = TypeVar("T")


# from .utils.futures import as_asyncio_future, as_concurrent_future


class Kernel:
    """
    TODO doc
    
    """

    def __init__(self, argv: list[str] = [], wait: bool = True):
        """
        Create an Isaac Sim kernel.

        :param argv: Additional arguments to pass to the kernel.

            Common arguments include:

            * :code:`--help`: show help message and shut down.
            * :code:`--no-window`: switch to headless mode (disables window).
            * :code:`--/app/window/hideUi=True`: hide all UI elements (still opens a window).

            .. seealso::
                * :class:`isaacsim.simulation_app.AppFramework`
                * :class:`isaacsim.simulation_app.SimulationApp`
                * https://docs.omniverse.nvidia.com/kit/docs/kit-manual/108.0.0/guide/configuring.html#kit-kernel-settings
            
        :param wait: Whether to block until the kernel is initialized.
        """

        super().__init__()

        self._should_run_update_loop = False

        # TODO FIXME performance: 25us overhead per .get call
        self._callbacks = queue.SimpleQueue[Callable[[], Any]]()

        self._init_event = threading.Event()
        self._thread = threading.Thread(
            target=self.__thread_target, 
            kwargs=dict(argv=argv), 
            daemon=False,
        )
        self._thread.start()
        if wait: 
            self._init_event.wait()

    def __thread_target(
        self, 
        argv: list[str],
        # on_ready: Callable[[], None] | None = None,
    ):
        import os

        @contextlib.contextmanager
        def _undo_monkeypatching():
            """
            Undo the unnecessary monkey-patching 
            of builtin Python modules done by `omni.kit.app`.

            .. seealso::
                * :func:`_startup_kit_scripting` in 
                `site-packages/omni/kernel/py/omni/kit/app/_impl/__init__.py`
            """

            import logging

            logging_handlers_orig = list(logging.root.handlers)
            logging_level_orig = int(logging.root.level)

            yield

            logging.root.handlers = logging_handlers_orig
            logging.root.level = logging_level_orig

        with _undo_monkeypatching():
            import isaacsim
            isaacsim.bootstrap_kernel()
            from isaacsim.simulation_app import AppFramework

            # TODO
            experience = ""
            if experience == "":
                # TODO use proper os.path.join
                for exp in [
                    "isaacsim.exp.full.kit",
                    "omni.isaac.sim.python.kit",
                    "isaacsim.exp.base.python.kit",
                    "isaacsim.exp.base.kit",
                ]:
                    exp = os.path.join(os.environ["EXP_PATH"], exp)
                    if os.path.isfile(exp):
                        experience = exp
                        break

            exe_path = os.environ["CARB_APP_PATH"]

            argv = [
                os.path.abspath(experience),
                # run as portable to prevent writing extra files to user directory
                "--portable",
                # extensions
                # extensions: adding to json doesn't work
                "--ext-folder", os.path.abspath(os.path.join(os.environ["ISAAC_PATH"], "exts")),
                # extensions: so we can reference other kit files  
                "--ext-folder", os.path.abspath(os.path.join(os.environ["ISAAC_PATH"], "apps")),      
                # ...      
                f"--/app/tokens/exe-path={os.path.abspath(exe_path)}",  # this is needed so dlss lib is found
                # "--/app/fastShutdown=true",
                "--/app/installSignalHandlers=false",
                "--/app/python/interceptSysStdOutput=false",
                "--/app/python/interceptSysExit=false",
                "--/app/python/logSysStdOutput=false",
                "--/app/python/enableGCProfiling=false",
                "--/app/python/disableGCDuringStartup=false",
                # logging
                # logging: disable default log file creation
                "--/log/file=",
                # "--/log/enabled=false",
                *argv,
            ]

            self._app_framework = AppFramework(argv=argv)

            import isaacsim
            import omni
            import pxr

            self._isaacsim = isaacsim
            self._omni = omni
            self._pxr = pxr

            # TODO load extensions

        # TODO
        self._init_event.set()

        try:
            while True:
                if not self._should_run_update_loop:
                    self._callbacks.get()()
                else:
                    # TODO FIXME performance
                    try: 
                        if not self._callbacks.empty():
                            self._callbacks.get(block=False)()
                    except queue.Empty: 
                        pass
                    self._app_framework.update()
        finally:
            # TODO FIXME crash? may have nothing to do with carb
            # [Error] [carb.settings.plugin] initializeFromDictionary: already initialized
            # import carb
            # _logging = carb.logging.acquire_logging()
            # _logging.set_log_enabled(False)
            self._app_framework.close()

    def call_soon(self, func: Callable[[], Any]):
        """
        Schedule function to be called in the kernel thread.
        Useful for operations requiring thread safety. 

        :param func: Function to be scheduled.
        """

        self._callbacks.put(func, block=False)

    # TODO mv??
    def submit(self, func: Callable[[], _T]):
        """
        Schedule function to be called in the kernel thread
        and get the return value of the scheduled function.
        Useful for operations requiring thread safety. 

        :param func: Function to be scheduled.
        :return: A future object that represents the return value.
        """

        future = concurrent.futures.Future[_T]()
        def wrapped_func():
            try: future.set_result(func())
            except Exception as error:
                future.set_exception(error)
        self.call_soon(wrapped_func)
        return future

    def step_update_loop(self):
        def f():
            self._app_framework.update()
        return self.submit(f)

    def run_update_loop(self):
        def f():
            self._should_run_update_loop = True
        return self.submit(f)

    def stop_update_loop(self):
        def f():
            self._should_run_update_loop = False
        return self.submit(f)

    @property
    def isaacsim(self):
        # TODO !!!!
        return self._isaacsim

    @property
    def omni(self):
        # TODO !!!!
        return self._omni

    @property
    def pxr(self):
        # TODO !!!!
        return self._pxr
    
    # TODO rm
    # @property
    # def pre_step_callbacks(self):
    #     return self._pre_step_callbacks

    # def start(self):
    #     self._thread.start()

    # def wait_until_initialized(self):
    #     self._is_initialized.wait()

    # def play(self):
    #     self._should_play.set()

    # def pause(self):
    #     self._should_play.clear()

    # def stop(self):
    #     self._should_stop.set()

    # def run_coroutine(self, awaitable: Awaitable):
    #     return as_asyncio_future(
    #         self._omni.kit.async_engine.run_coroutine(awaitable)
    #     )
    
    # def run_coroutine_concurrent(self, awaitable: Awaitable):
    #     return as_concurrent_future(
    #         self._omni.kit.async_engine.run_coroutine(awaitable)
    #     )

    # def timeline_play(self):
    #     self._omni.timeline.get_timeline_interface().play()

    # def timeline_pause(self):
    #     self._omni.timeline.get_timeline_interface().pause()

    # def timeline_is_playing(self):
    #     return self._omni.timeline.get_timeline_interface().is_playing()

    # TODO deprecate #######
    # def physics_flush_changes(self):
    #     self._omni.physx.get_physx_simulation_interface().flush_changes()

    # @functools.cached_property
    # def __physics_view(self):
    #     try:
    #         res = self._omni.physics.tensors.create_simulation_view("torch")
    #     except Exception as error:
    #         # TODO
    #         raise RuntimeError(
    #             "Failed to create physics view. "
    #             f"Does a prim of type `PhysicsScene` exist on the stage?"
    #             # TODO rm
    #             # f"Make sure the physics simulation is running: call {self.timeline_play}"
    #         ) from error
    #     if not res.is_valid:
    #         raise RuntimeError(f"Created physics view is invalid: {res}")
    #     return res

    # def physics_get_view(self):
    #     if not self.__physics_view.is_valid:
    #         del self.__physics_view
    #     return self.__physics_view
    
    # def physics_get_articulation_view(self, selector: str | list[str]):
    #     # TODO
    #     self._omni.physx.get_physx_simulation_interface().flush_changes()
    #     try:
    #         res = self.physics_get_view().create_articulation_view(selector)
    #         assert res is not None
    #         assert res.check()
    #     except Exception as error:
    #         raise RuntimeError(
    #             f"Failed to create physics view from selector (is it valid?): {selector}"
    #         ) from error
    #     return res
    # TODO deprecate #######
