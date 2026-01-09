# SPDX-License-Identifier: Apache-2.0

"""
Rendering utilities.
"""


import asyncio

from robotodo.engines.isaac.kernel import Kernel


# TODO perf: use async iterator? also .next_event is slow?
async def omni_usd_wait_next_render(
    kernel: Kernel,
    usd_context: "omni.usd.UsdContext",
    target_render_product_path: str | None = None,
):
    # TODO FIXME perf: _omni_enable_extensions is expensive
    kernel._omni_enable_extensions([
        "omni.usd",
        "omni.syntheticdata",
    ])
    # TODO FIXME perf: memoize
    omni = kernel._omni
    carb = kernel._carb
    _sdg_iface = omni.syntheticdata.helpers._get_syntheticdata_iface()

    preupdate_event_stream = kernel._app.get_pre_update_event_stream()
    rendering_event_stream = usd_context.get_rendering_event_stream()

    # TODO 
    preupdate_event = await preupdate_event_stream.next_event()
    swh_frame_number = preupdate_event.payload["SWHFrameNumber"]

    # TODO thread safety
    future = asyncio.Future()
    
    def rendering_new_frame_callback(rendering_new_frame_event: ...):
        if future.done():
            return

        render_product_path, reftime_nom, reftime_denom = _sdg_iface.parse_rendered_simulation_event(
            rendering_new_frame_event.payload["product_path_handle"], 
            rendering_new_frame_event.payload["results"],
        )
        if target_render_product_path is not None:
            if render_product_path != target_render_product_path:
                return
            
        render_swh_frame_number = rendering_new_frame_event.payload.get("swh_frame_number", -1)
        if render_swh_frame_number == -1:
            # TODO assert? warn? disable `/app/asyncRendering`
            settings = carb.settings.get_settings()
            # if settings.get("/app/asyncRendering") is True:
            #     # TODO warn???
            #     # future.set_exception(NotImplementedError("TODO"))
            #     ...
            settings.set("/app/asyncRendering", False)
        
        offset = render_swh_frame_number - swh_frame_number
        if offset >= 0:
            future.set_result(offset)

    rendering_event_stream = usd_context.get_rendering_event_stream()
    rendering_event_sub = rendering_event_stream.create_subscription_to_pop_by_type(omni.usd.StageRenderingEventType.NEW_FRAME, rendering_new_frame_callback)

    future.add_done_callback(lambda _, rendering_event_sub=rendering_event_sub: rendering_event_sub.unsubscribe())
    return await future