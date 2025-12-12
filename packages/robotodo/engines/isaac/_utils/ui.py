import asyncio

from robotodo.engines.isaac.kernel import Kernel


def omni_enable_viewing_experience(kernel: Kernel):

    extension_names = [
        #
        "omni.kit.viewport.window",
    ]

    workspace_layout = [
        {
            "dock_id": 3358485147,
            "children": [{
                "dock_id": 3358485147,
                "dock_tab_bar_enabled": True,
                "dock_tab_bar_visible": True,
                "height": 772.0,
                # "position_x": 43.333335876464844,
                # "position_y": 22.666667938232422,
                # "selected_in_dock": True,
                "title": "Viewport",
                "visible": True,
                "width": 1823.3333740234375,
            }]
        }
    ]

    omni = kernel._omni
    kernel._omni_enable_extension("omni.ui")
    kernel._omni_import_module("omni.ui")

    async def _restore_workspace_task():
        # await _ensure_window_titles(
        #     window_titles,
        #     kernel=kernel,
        # )
        # # TODO
        # print("TODO", window_titles)
        kernel._omni_enable_extensions(extension_names)
        # TODO
        await kernel._app.next_update_async()
        omni.ui.Workspace.restore_workspace(
            workspace_dump=workspace_layout,
            keep_windows_open=False,
        )

    kernel._omni_run_coroutine(_restore_workspace_task(), run_until_complete=False)



def omni_enable_editing_experience(kernel: Kernel):

    # TODO rm?
    # async def _ensure_window_titles(window_titles: set[str], kernel: Kernel):
    #     window_titles = set(window_titles)

    #     omni = kernel.omni
    #     kernel.enable_extension("omni.ui")
    #     kernel.import_module("omni.ui")

    #     while True:
    #         # TODO
    #         print("TODO", set(window.dock_id for window in omni.ui.Workspace.get_windows()))

    #         window_titles_current = set(window.title for window in omni.ui.Workspace.get_windows())
    #         if window_titles.issubset(window_titles_current):
    #             break
    #         # TODO
    #         await kernel.app.next_update_async()

    # TODO rm?
    # extension_names = [
    #     # Isaac Sim Extra
    #     "semantics.schema.editor",
    #     "semantics.schema.property",
    #     # Kit based editor extensions
    #     "omni.graph.ui",
    #     "omni.hydra.engine.stats",
    #     "omni.kit.mainwindow",
    #     "omni.kit.manipulator.camera",
    #     "omni.kit.manipulator.prim",
    #     "omni.kit.manipulator.selection",
    #     "omni.kit.material.library",
    #     # TODO
    #     "omni.kit.menu.common",
    #     "omni.kit.menu.create",
    #     "omni.kit.menu.stage",
    #     "omni.kit.menu.utils",
    #     "omni.kit.primitive.mesh",
    #     "omni.kit.property.bundle",
    #     "omni.kit.stage_template.core",
    #     "omni.kit.tool.asset_importer",
    #     "omni.kit.tool.collect",
    #     "omni.kit.viewport.legacy_gizmos",
    #     "omni.kit.viewport.menubar.camera",
    #     "omni.kit.viewport.menubar.display",
    #     "omni.kit.viewport.menubar.lighting",
    #     "omni.kit.viewport.menubar.render",
    #     "omni.kit.viewport.menubar.settings",
    #     "omni.kit.viewport.scene_camera_model",
    #     "omni.kit.viewport.window",
    #     "omni.kit.window.console",
    #     "omni.kit.window.content_browser",
    #     "omni.kit.window.property",
    #     "omni.kit.window.stage",
    #     "omni.kit.window.status_bar",
    #     "omni.kit.window.toolbar",
    #     "omni.uiaudio",
    #     "omni.usd.metrics.assembler.ui",
    #     # Additional Kit extensions
    #     "omni.anim.curve.bundle",
    #     "omni.physx.asset_validator",
    #     "omni.asset_validator.ui",
    #     "omni.anim.shared.core",
    #     "omni.graph.bundle.action",
    #     "omni.graph.visualization.nodes",
    #     "omni.graph.window.action",
    #     "omni.graph.window.generic",
    #     "omni.importer.onshape",
    #     "omni.kit.actions.window",
    #     "omni.kit.asset_converter",
    #     "omni.kit.browser.asset",
    #     "omni.kit.browser.material",
    #     "omni.kit.collaboration.channel_manager",
    #     "omni.kit.context_menu",
    #     "omni.kit.widget.schema_api",
    #     "omni.kit.graph.delegate.default",
    #     "omni.kit.hotkeys.window",
    #     "omni.kit.manipulator.transform",
    #     "omni.kit.mesh.raycast",
    #     "omni.kit.preferences.animation",
    #     "omni.kit.profiler.window",
    #     "omni.kit.property.collection",
    #     "omni.kit.property.layer",
    #     "omni.kit.renderer.capture",
    #     "omni.kit.renderer.core",
    #     "omni.kit.scripting",
    #     "omni.kit.search.files",
    #     "omni.kit.selection",
    #     "omni.kit.stage.copypaste",
    #     "omni.kit.stage.mdl_converter",
    #     "omni.kit.stage_column.payload",
    #     "omni.kit.stage_column.variant",
    #     "omni.kit.stage_templates",
    #     "omni.kit.stagerecorder.bundle",
    #     "omni.kit.tool.asset_exporter",
    #     "omni.kit.tool.remove_unused.controller",
    #     "omni.kit.tool.remove_unused.core",
    #     "omni.kit.uiapp",
    #     "omni.kit.usda_edit",
    #     "omni.kit.variant.editor",
    #     "omni.kit.variant.presenter",
    #     "omni.kit.viewport.actions",
    #     "omni.kit.viewport.bundle",
    #     "omni.kit.viewport.rtx",
    #     "omni.kit.viewport_widgets_manager",
    #     "omni.kit.widget.cache_indicator",
    #     "omni.kit.widget.collection",
    #     "omni.kit.widget.extended_searchfield",
    #     "omni.kit.widget.filebrowser",
    #     "omni.kit.widget.layers",
    #     "omni.kit.widget.live",
    #     "omni.kit.widget.timeline",
    #     "omni.kit.widget.versioning",
    #     "omni.kit.widgets.custom",
    #     "omni.kit.window.collection",
    #     "omni.kit.window.commands",
    #     "omni.kit.window.cursor",
    #     "omni.kit.window.extensions",
    #     "omni.kit.window.file",
    #     "omni.kit.window.filepicker",
    #     "omni.kit.window.material",
    #     "omni.kit.window.material_graph",
    #     "omni.kit.window.preferences",
    #     "omni.kit.window.quicksearch",
    #     "omni.kit.window.script_editor",
    #     "omni.kit.window.stats",
    #     "omni.kit.window.title",
    #     "omni.kit.window.usd_paths",
    #     "omni.resourcemonitor",
    #     "omni.kit.quicklayout",
    # ]

    extension_names = [
        #
        # TODO
        "omni.app.dev",
        #
        "omni.kit.viewport.legacy_gizmos",
        "omni.kit.viewport.menubar.camera",
        "omni.kit.viewport.menubar.display",
        "omni.kit.viewport.menubar.lighting",
        "omni.kit.viewport.menubar.render",
        "omni.kit.viewport.menubar.settings",
        "omni.kit.viewport.scene_camera_model",
        "omni.kit.viewport.window",
        # TODO
        "omni.usdphysics.ui",
        # TODO
        "isaacsim.gui.content_browser",
        "omni.simready.explorer",
    ]

    workspace_layout = [
        {
            "children": [
                {
                    "children": [
                        {
                            "children": [
                                {
                                    "children": [
                                        {
                                            "dock_id": 5,
                                            "height": 772.0,
                                            "position_x": 0.0,
                                            "position_y": 22.666667938232422,
                                            "selected_in_dock": False,
                                            "title": "Main ToolBar",
                                            "visible": True,
                                            "width": 42.0,
                                        }
                                    ],
                                    "dock_id": 5,
                                    "position": "LEFT",
                                },
                                {
                                    "children": [
                                        {
                                            "dock_id": 6,
                                            "dock_tab_bar_enabled": True,
                                            "dock_tab_bar_visible": True,
                                            "height": 772.0,
                                            "position_x": 43.333335876464844,
                                            "position_y": 22.666667938232422,
                                            "selected_in_dock": True,
                                            "title": "Viewport",
                                            "visible": True,
                                            "width": 1823.3333740234375,
                                        }
                                    ],
                                    "dock_id": 6,
                                    "position": "RIGHT",
                                },
                            ],
                            "dock_id": 3,
                            "position": "TOP",
                        },
                        {
                            "children": [
                                {
                                    "dock_id": 4,
                                    "dock_tab_bar_enabled": True,
                                    "dock_tab_bar_visible": True,
                                    "height": 554.6666870117188,
                                    "position_x": 0.0,
                                    "position_y": 796.0,
                                    "selected_in_dock": True,
                                    "title": "Content",
                                    "visible": True,
                                    "width": 1866.666748046875,
                                },
                                {
                                    "dock_id": 4,
                                    "height": 554.6666870117188,
                                    "position_x": 0.0,
                                    "position_y": 796.0,
                                    "selected_in_dock": False,
                                    "title": "Console",
                                    "visible": True,
                                    "width": 1866.666748046875,
                                },
                            ],
                            "dock_id": 4,
                            "position": "BOTTOM",
                        },
                    ],
                    "dock_id": 1,
                    "position": "LEFT",
                },
                {
                    "children": [
                        {
                            "children": [
                                {
                                    "dock_id": 7,
                                    "dock_tab_bar_enabled": True,
                                    "dock_tab_bar_visible": True,
                                    "height": 655.3333740234375,
                                    "position_x": 1868.0,
                                    "position_y": 22.666667938232422,
                                    "selected_in_dock": True,
                                    "title": "Stage",
                                    "visible": True,
                                    "width": 692.0,
                                },
                                {
                                    "dock_id": 7,
                                    "height": 655.3333740234375,
                                    "position_x": 1868.0,
                                    "position_y": 22.666667938232422,
                                    "selected_in_dock": False,
                                    "title": "Layer",
                                    "visible": True,
                                    "width": 692.0,
                                },
                                # TODO
                                # {
                                #     "dock_id": 7,
                                #     "height": 655.3333740234375,
                                #     "position_x": 1868.0,
                                #     "position_y": 22.666667938232422,
                                #     "selected_in_dock": False,
                                #     "title": "Render Settings",
                                #     "visible": True,
                                #     "width": 692.0,
                                # },
                            ],
                            "dock_id": 7,
                            "position": "TOP",
                        },
                        {
                            "children": [
                                {
                                    "dock_id": 8,
                                    "dock_tab_bar_enabled": True,
                                    "dock_tab_bar_visible": True,
                                    "height": 671.3333740234375,
                                    "position_x": 1868.0,
                                    "position_y": 679.3333740234375,
                                    "selected_in_dock": True,
                                    "title": "Property",
                                    "visible": True,
                                    "width": 692.0,
                                }
                            ],
                            "dock_id": 8,
                            "position": "BOTTOM",
                        },
                    ],
                    "dock_id": 2,
                    "position": "RIGHT",
                },
            ],
            "dock_id": 3358485147,
        }
    ]

    omni = kernel._omni
    kernel._omni_enable_extension("omni.ui")
    kernel._omni_import_module("omni.ui")
    kernel._omni_enable_extension("omni.kit.mainwindow")
    kernel._omni_import_module("omni.kit.mainwindow")

    main_window = omni.kit.mainwindow.get_main_window()
    main_menu_bar = main_window.get_main_menu_bar()
    main_menu_bar.visible = True
    main_status_bar = main_window.get_status_bar_frame()
    main_status_bar.visible = True

    # TODO rm?
    # window_titles = set([
    #     "Main ToolBar",
    #     "Viewport",
    #     "Content",
    #     "Console",
    #     "Stage",
    #     "Layer",
    #     "Property",
    # ])

    async def _restore_workspace_task():
        # await _ensure_window_titles(
        #     window_titles,
        #     kernel=kernel,
        # )
        # # TODO
        # print("TODO", window_titles)
        kernel._omni_enable_extensions(extension_names)
        # TODO
        await kernel._app.next_update_async()
        omni.ui.Workspace.restore_workspace(
            workspace_dump=workspace_layout,
            keep_windows_open=False,
        )

    kernel._omni_run_coroutine(_restore_workspace_task(), run_until_complete=False)

