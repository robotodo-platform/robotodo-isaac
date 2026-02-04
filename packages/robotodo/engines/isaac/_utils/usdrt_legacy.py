# SPDX-License-Identifier: Apache-2.0

"""
USDRT utilities.
"""


import warnings
import functools
from typing import Any

import warp
import torch

from robotodo.engines.isaac.kernel import Kernel
from robotodo.engines.isaac._utils.usd import usd_get_stage_id, USDPrimRef


@warp.kernel(enable_backward=False)
def _fabric_ragged_compute_lengths_kernel(
    a: warp.fabricarrayarray(), 
    indices: Any,
    lengths: warp.array(),
):
    i = warp.tid()
    lengths[i] = lengths.dtype(len(a[indices[i]]))


@warp.kernel(enable_backward=False)
def _fabric_pack_kernel(
    a: warp.fabricarrayarray(), 
    indices: Any,
    lengths: warp.array(),
    offsets: warp.array(),
    a_packed: warp.array(),
):
    i, j = warp.tid()
    if j < lengths[i]:
        a_packed[offsets[i] + offsets[i].dtype(j)] = a[indices[i]][j]


def fabric_read_torch_jagged(
    a: warp.fabricarrayarray(),
    indices: warp.array(),
):
    lengths = warp.empty(len(a), dtype=warp.int64)
    warp.launch(
        _fabric_ragged_compute_lengths_kernel, 
        dim=len(a), 
        inputs=[a, indices], 
        outputs=[lengths],
        stream=warp.stream_from_torch(),
    )
    lengths_pt = warp.to_torch(lengths)
    offsets_pt = torch.nn.functional.pad(torch.cumsum(lengths_pt, dim=0), (1, 0))
    
    a_packed = warp.empty(torch.sum(lengths_pt).reshape(1), dtype=a.dtype)
    warp.launch(
        _fabric_pack_kernel,
        dim=[len(a), torch.max(lengths_pt)],
        inputs=[a, indices, lengths, warp.from_torch(offsets_pt)],
        outputs=[a_packed],
        stream=warp.stream_from_torch(),
    )

    return torch.nested.nested_tensor_from_jagged(
        values=warp.to_torch(a_packed),
        # lengths=lengths_pt,
        offsets=offsets_pt,
    )


@warp.kernel(enable_backward=False)
def _fabric_copy_kernel(
    a: warp.fabricarray(), 
    indices: Any,
    a_out: warp.array(),
):
    i = warp.tid()
    a_out[i] = a[indices[i]]


def fabric_read_torch(
    a: warp.fabricarray(),
    indices: warp.array(),
):
    return warp.to_torch(a.contiguous())[warp.to_torch(indices.contiguous()).to(torch.long)]
    # TODO why is warp slower????
    # a_out = warp.empty_like(a)
    # warp.launch(
    #     _fabric_copy_kernel,
    #     dim=[len(a)],
    #     inputs=[a, indices],
    #     outputs=[a_out],
    # )
    # return warp.to_torch(a_out)


def fabric_write_torch(
    a: warp.fabricarray(),
    indices: warp.array(),
    a_in: torch.Tensor,
):
    return a.assign(
        warp.from_torch(
            a_in[warp.to_torch(indices.contiguous()).to(torch.long)],
            dtype=a.dtype,
        )
    )


def usdrt_get_stage(stage: "pxr.Usd.Stage", kernel: Kernel):
    # TODO
    usdrt = kernel._usdrt
    omni = kernel._omni
    kernel._omni_enable_extension("omni.usd")

    stage_id = usd_get_stage_id(stage, kernel=kernel)

    # TODO BUG ensure a valid usd_context; otherwise crash may result for some unknown reason!!!!
    context = omni.usd.get_context()
    if context.get_stage_id() != stage_id:
        context.attach_stage_with_callback(stage_id)

    res = usdrt.Usd.Stage.Attach(stage_id)
    # TODO necesito?
    # res.SynchronizeToFabric()
    return res


class USDRTUSDPrimRef:
    def __init__(self, usd_ref: USDPrimRef, kernel: Kernel):
        self._usd_ref = usd_ref
        self._kernel = kernel

    def __call__(self):
        prims = self._usd_ref()

        memoized_usdrt_get_stage = functools.cache(usdrt_get_stage)
        return [
            memoized_usdrt_get_stage(
                prim.GetStage(), 
                kernel=self._kernel,
            )
            .GetPrimAtPath(prim.GetPath().pathString)
            for prim in prims
        ]


class USDRTPrimView:
    def __init__(self, ref: ..., kernel: Kernel, device: str | None = None):
        self._prims_ref = ref
        self._kernel = kernel
        self._device = device

    _AttrSpec = tuple["usdrt.Sdf.ValueTypeName", str, "usdrt.Usd.Access"]

    @functools.cached_property
    def _prim_index_attribute(self):
        # TODO NOTE private
        return f"_robotodo:{id(self)}:index"
    
    @functools.cached_property
    def _requested_attributes(self):
        usdrt = self._kernel._usdrt

        return dict[str, USDRTPrimView._AttrSpec]({
            self._prim_index_attribute:
            (usdrt.Sdf.ValueTypeNames.UInt, self._prim_index_attribute, usdrt.Usd.Access.Read),
        })
    
    class _Selections(dict):
        def __hash__(self):
            return hash(id(self))

    @functools.cached_property
    def _cached_selections(self):
        usdrt = self._kernel._usdrt

        res = dict[int, "usdrt.Rt.RtPrimSelection"]()

        # TODO invalidate on prim selection change
        prims = self._prims_ref()
        for i, prim in enumerate(prims):
            # TODO NOTE BUG a new USDRT stage object is always returned!!!
            res.setdefault(prim.GetStage().GetStageId())
            prim.CreateAttribute(self._prim_index_attribute, usdrt.Sdf.ValueTypeNames.UInt, custom=True).Set(i)

        # TODO
        for stage_id in res:
            stage = usdrt.Usd.Stage.Attach(stage_id)
            stage.SynchronizeToFabric()
            res[stage_id] = stage.SelectPrims(
                require_attrs=list(self._requested_attributes.values()),
                # TODO
                device=warp.get_device().alias if self._device is None else self._device,
                # want_paths=True,
            )
            res[stage_id].PrepareForReuse()

        return USDRTPrimView._Selections(res)
    
    _should_invalidate_selections: bool = False

    @property
    def _selections(self):
        if self._should_invalidate_selections:
            if "_cached_selections" in self.__dict__:
                del self._cached_selections
            self._should_invalidate_selections = False
        return self._cached_selections
    
    @functools.cache
    def _fabric_hierachies_from_selections(self, selections: _Selections):
        usdrt = self._kernel._usdrt

        fabric_hierarchy_iface = usdrt.hierarchy.IFabricHierarchy()

        fabric_hierarchies = dict()
        for stage_id, _ in selections.items():
            stage_rt = usdrt.Usd.Stage.Attach(stage_id)
            fabric_hierarchies[stage_id] = fabric_hierarchy_iface.get_fabric_hierarchy(
                stage_rt.GetFabricId(), 
                stage_rt.GetStageIdAsStageId(),
            )

        return fabric_hierarchies
    
    def sync_from_usd(self):
        usdrt = self._kernel._usdrt
        for stage_id, _ in self._selections.items():
            stage_rt = usdrt.Usd.Stage.Attach(stage_id)
            stage_rt.SynchronizeToFabric()

    # TODO upstream: this isnt fully working
    def sync_to_usd(self):
        usdrt = self._kernel._usdrt
        for stage_id, _ in self._selections.items():
            stage_rt = usdrt.Usd.Stage.Attach(stage_id)
            stage_rt.WriteToStage(
                includePrivateFabricProperties=False, 
                convertFabricXforms=True,
            )

    def request_attribute(self, spec: _AttrSpec):
        _, name, _ = spec
        if self._requested_attributes.get(name) == spec:
            return
        self._requested_attributes[name] = spec
        self._should_invalidate_selections = True

    def get_attribute(self, name: str):        
        if name not in self._requested_attributes:
            # TODO
            warnings.warn(f"TODO call `{self.request_attribute}`: {name}")
        
        # TODO NOTE ensure single stage
        # TODO NOTE support for multiple USDRT stages is not planned for now
        assert len(self._selections) <= 1
        for _, selection in self._selections.items():
            return warp.fabricarray(data=selection, attrib=name)
        
        # TODO
        raise ValueError("TODO empty selection")
        # TODO NOTE support for multiple USDRT stages is not planned for now
        # return {
        #     stage_id: warp.fabricarray(data=selection, attrib=name)
        #     for stage_id, selection in self._cached_selections.items()
        # }

    @functools.cached_property
    def xform(self):
        res = USDRTPrimXFormView(self)
        # TODO HACK ensure sync??? why???
        res.local_matrix
        res.world_matrix
        return res


class USDRTPrimXFormView:
    def __init__(
        self,
        rt_view: USDRTPrimView,
    ):
        self._rt_view = rt_view

    @functools.cached_property
    def _attribute_specs(self):
        # TODO
        usdrt = self._rt_view._kernel._usdrt

        return {
            "omni:fabric:worldMatrix": (
                usdrt.Sdf.ValueTypeNames.Matrix4d, 
                "omni:fabric:worldMatrix", 
                # usdrt.Usd.Access.ReadWrite,
                usdrt.Usd.Access.Read,
            ),
            "omni:fabric:localMatrix": (
                usdrt.Sdf.ValueTypeNames.Matrix4d, 
                "omni:fabric:localMatrix", 
                usdrt.Usd.Access.ReadWrite,
            ),
        }

    def _sync_world_xforms(self):
        for _, hierarchy in self._rt_view._fabric_hierachies_from_selections(self._rt_view._selections).items():
            hierarchy.track_local_xform_changes(True)
            hierarchy.update_world_xforms()
            # TODO FIXME perf: https://github.com/isaac-sim/IsaacSim/issues/391
            hierarchy.update_world_xforms_gpu(True)

    # TODO this is row-major?
    @property
    def world_matrix(self):
        self._rt_view.request_attribute(self._attribute_specs["omni:fabric:worldMatrix"])
        self._sync_world_xforms()
        return fabric_read_torch(
            self._rt_view.get_attribute("omni:fabric:worldMatrix"),
            indices=self._rt_view.get_attribute(self._rt_view._prim_index_attribute),
        )

    # TODO NOTE not supported    
    # @world_matrix.setter
    # def world_matrix(self, value: ...):
    #     self._rt_view.request_attribute(self._attribute_specs["omni:fabric:worldMatrix"])
    #     a = self._rt_view.get_attribute("omni:fabric:worldMatrix")
    #     fabric_write_torch(
    #         a,
    #         indices=self._rt_view.get_attribute(self._rt_view._prim_index_attribute),
    #         a_in=torch.broadcast_to(torch.asarray(value, dtype=torch.float64, device=a.device.alias), size=(len(a), 4, 4)),
    #     )

    # TODO this is row-major?
    @property
    def local_matrix(self):
        self._rt_view.request_attribute(self._attribute_specs["omni:fabric:localMatrix"])
        return fabric_read_torch(
            self._rt_view.get_attribute("omni:fabric:localMatrix"),
            indices=self._rt_view.get_attribute(self._rt_view._prim_index_attribute),
        )
    
    @local_matrix.setter
    def local_matrix(self, value: ...):
        self._rt_view.request_attribute(self._attribute_specs["omni:fabric:localMatrix"])
        # self._rt_view.request_attribute(self._attribute_specs["omni:fabric:worldMatrix"])
        a = self._rt_view.get_attribute("omni:fabric:localMatrix")
        fabric_write_torch(
            a,
            indices=self._rt_view.get_attribute(self._rt_view._prim_index_attribute),
            a_in=torch.broadcast_to(
                torch.asarray(
                    value, 
                    dtype=torch.float64, 
                    device=warp.device_to_torch(a.device),
                ), 
                size=(len(a), 4, 4),
            ),
        )
        # TODO enable opt-out using ctx mgr!!!
        if True:
            self._sync_world_xforms()
