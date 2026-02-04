import abc
import warnings
import functools
import weakref
import contextlib
from typing import Any, Literal

import warp
import torch

from robotodo.engines.isaac.kernel import Kernel
# TODO
from robotodo.engines.isaac._utils.usd import usd_get_stage_id
from robotodo.engines.core.path import PathExpression, PathExpressionLike


class ProtoPrimView(abc.ABC):
    @abc.abstractmethod
    def __init__(
        self,
        kernel: Kernel,
        stage: "pxr.Usd.Stage | usdrt.Usd.Stage",
        path: PathExpressionLike | None = None,
        # type: str | None = None,
        # applied_schemas: list[str] | None = None,
    ):
        ...

    @abc.abstractmethod
    def create_attribute(
        self, 
        name: str, 
        value_type: "pxr.Sdf.ValueTypeName | usdrt.Sdf.ValueTypeName",
        custom: bool = False,
    ):
        ...

    @abc.abstractmethod
    def request_attribute(
        self, 
        name: str, 
        value_type: str,
        value_access: Literal["read", "write", "readwrite"] | None = None,
    ):
        r"""
        Request a USD attribute to be available 
        for :meth:`read_attribute`, :meth:`write_attribute`, or both.

        :param name:
            Name of the attribute to request.
        :param value_type: 
            Type of the attribute. Must be one of :class:`pxr.Sdf.ValueTypeNames`.
        :param value_access:
            Access mode of the attribute. Determined by the backend if not provided.
        """
        ...

    # TODO value, indices
    # @abc.abstractmethod
    # def get_attribute(self, name: str) -> tuple[warp.array(), warp.array() | None]:
    #     ...

    # read_attribute
    # write_attribute

    @property
    @abc.abstractmethod
    def xform(self) -> "ProtoPrimXformView":
        ...


class ProtoPrimXformView(abc.ABC):
    @abc.abstractmethod
    def read_world_matrix(self):
        ...

    @abc.abstractmethod
    def read_local_matrix(self):
        ...

    @abc.abstractmethod
    def write_local_matrix(self, value: warp.array()):
        ...


@warp.kernel(enable_backward=False)
def _usdrt_kernel_compute_lengths(
    a: warp.fabricarrayarray(), 
    indices: Any,
    lengths: warp.array(),
):
    i = warp.tid()
    lengths[i] = lengths.dtype(len(a[indices[i]]))


@warp.kernel(enable_backward=False)
def _usdrt_kernel_pack_jagged(
    a: warp.fabricarrayarray(), 
    indices: Any,
    lengths: warp.array(),
    offsets: warp.array(),
    a_packed: warp.array(),
):
    i, j = warp.tid()
    if j < lengths[i]:
        a_packed[offsets[i] + offsets[i].dtype(j)] = a[indices[i]][j]


def usdrt_read_torch_jagged(
    a: warp.fabricarrayarray(),
    indices: warp.array(),
):
    lengths = warp.empty(len(a), device=a.device, dtype=warp.int64)
    warp.launch(
        _usdrt_kernel_compute_lengths, 
        dim=len(a), 
        inputs=[a, indices], 
        outputs=[lengths],
        device=a.device,
        stream=warp.stream_from_torch(),
    )
    lengths_pt = warp.to_torch(lengths)
    offsets_pt = torch.nn.functional.pad(torch.cumsum(lengths_pt, dim=0), (1, 0))
    
    a_packed = warp.empty(torch.sum(lengths_pt).reshape(1), device=a.device, dtype=a.dtype)
    warp.launch(
        _usdrt_kernel_pack_jagged,
        dim=[len(a), torch.max(lengths_pt)],
        inputs=[a, indices, lengths, warp.from_torch(offsets_pt)],
        outputs=[a_packed],
        device=a.device,
        stream=warp.stream_from_torch(),
    )

    return torch.nested.nested_tensor_from_jagged(
        values=warp.to_torch(a_packed),
        # lengths=lengths_pt,
        offsets=offsets_pt,
    )


@warp.kernel(enable_backward=False)
def _usdrt_kernel_copy(
    a: warp.fabricarray(), 
    indices: Any,
    a_out: warp.array(),
):
    i = warp.tid()
    a_out[i] = a[indices[i]]


def usdrt_is_jagged(
    a: warp.fabricarray(),
):
    # TODO https://github.com/NVIDIA/warp/blob/25d0c7e6b2e6ddd9d2394d501e66589d46762bcf/warp/_src/context.py#L8144
    return a.ndim > 1


def usdrt_read_torch(
    a: warp.fabricarray(),
    indices: warp.array(),
):
    return warp.to_torch(a.contiguous())[warp.to_torch(indices.contiguous()).to(torch.long)]
    # TODO why is warp slower????
    # a_out = warp.empty_like(a)
    # warp.launch(
    #     _usdrt_kernel_copy,
    #     dim=[len(a)],
    #     inputs=[a, indices],
    #     outputs=[a_out],
    # )
    # return warp.to_torch(a_out)


def usdrt_write_torch(
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


def usdrt_get_stage(
    kernel: Kernel,
    stage: "pxr.Usd.Stage | usdrt.Usd.Stage", 
) -> "usdrt.Usd.Stage":
    # TODO
    pxr = kernel._pxr
    usdrt = kernel._usdrt
    omni = kernel._omni
    kernel._omni_enable_extension("omni.usd")

    stage_rt: "usdrt.Usd.Stage"
    match stage:
        case pxr.Usd.Stage():
            stage_id = usd_get_stage_id(stage, kernel=kernel)

            # TODO BUG ensure a valid usd_context; otherwise crash may result for some unknown reason!!!!
            context = omni.usd.get_context()
            if context.get_stage_id() != stage_id:
                context.attach_stage_with_callback(stage_id)

            stage_rt = usdrt.Usd.Stage.Attach(stage_id)
            # TODO necesito?
            # stage_rt.SynchronizeToFabric()

        case usdrt.Usd.Stage():
            stage_rt = stage

        case _ as stage_id:
            # TODO
            stage_rt = usdrt.Usd.Stage.Attach(stage_id)

    return stage_rt


_USDRTAttrSpec = tuple["usdrt.Sdf.ValueTypeName", str, "usdrt.Usd.Access"]


# TODO allow raw prims?
@contextlib.contextmanager
def usdrt_index_prims(
    kernel: Kernel,
    stage: "usdrt.Usd.Stage",
    path: PathExpressionLike,
    _custom_attribute_namespace: str = "__robotodo",
):
    usdrt = kernel._usdrt

    lifecycle = object()
    prim_index_attribute = f"{_custom_attribute_namespace}:{id(lifecycle)}:index"

    prims_rt: list["usdrt.Usd.Prim"] = []

    for i, resolved_path in enumerate(
        PathExpression(path).filter(
            prim.GetPath().pathString
            for prim in stage.Traverse()
        )
    ):
        prim_rt = stage.GetPrimAtPath(resolved_path)
        prim_rt.CreateAttribute(
            prim_index_attribute, 
            usdrt.Sdf.ValueTypeNames.UInt, 
            custom=True,
        ).Set(i)
        prims_rt.append(prim_rt)

    prim_index_attribute_spec: _USDRTAttrSpec = (
        usdrt.Sdf.ValueTypeNames.UInt, 
        prim_index_attribute, 
        usdrt.Usd.Access.Read,
    )

    try:
        yield prims_rt, prim_index_attribute_spec
    finally:
        for prim_rt in prims_rt:
            prim_rt.RemoveProperty(prim_index_attribute)


from robotodo.engines.isaac._utils.cache import CachableRef


class USDRTPrimView(ProtoPrimView):
    def __init__(
        self, 
        kernel: Kernel,
        stage: "pxr.Usd.Stage | usdrt.Usd.Stage", 
        path: PathExpressionLike | None = None,
    ):
        self._kernel = kernel
        self._requested_stage = usdrt_get_stage(kernel, stage=stage)
        self._requested_path = path

    class _CacheToken:
        pass

    # TODO
    @functools.lru_cache(maxsize=1)
    def _cached_index_attribute_spec(
        self, 
        stage_ref: CachableRef["usdrt.Usd.Stage"], 
        path_ref: CachableRef[PathExpressionLike | None], 
    ):
        if path_ref.value is None:
            return None, None
        ctx = usdrt_index_prims(
            kernel=self._kernel,
            stage=stage_ref.value, 
            path=path_ref.value,
        )
        _, result = ctx.__enter__()
        token = self._CacheToken()
        # NOTE token is guaranteed to be not GC-d when in cache
        weakref.finalize(token, ctx.__exit__, None, None, None)
        return result, token

    @functools.lru_cache(maxsize=1)
    def _cached_selection_and_indices(
        self,
        stage_ref: CachableRef["usdrt.Usd.Stage"], 
        attribute_specs_ref: CachableRef[_USDRTAttrSpec],
        path_ref: CachableRef[PathExpressionLike | None], 
        device: str | None = None,
    ):
        require_attrs = []

        index_attribute_spec, _ = self._cached_index_attribute_spec(
            stage_ref=stage_ref, 
            path_ref=path_ref, 
        )
        if index_attribute_spec is not None:
            require_attrs.append(index_attribute_spec)

        if attribute_specs_ref.value is not None:
            require_attrs.extend(attribute_specs_ref.value)

        stage_ref.value.SynchronizeToFabric()
        selection = stage_ref.value.SelectPrims(
            require_attrs=require_attrs,
            # require_applied_schemas=applied_schemas,
            # require_prim_type=type,
            # NOTE this defaults to cpu if unspecified
            device=warp.get_device().alias if device is None else device,
            # TODO ondemand
            want_paths=True,
        )
        selection.PrepareForReuse()

        indices = None
        if index_attribute_spec is not None:
            _, prim_index_attribute, _ = index_attribute_spec
            indices = warp.fabricarray(data=selection, attrib=prim_index_attribute)

        return selection, indices    

    @functools.cached_property
    def _requested_attributes(self):
        return dict[str, _USDRTAttrSpec]()
    
    _requested_attributes_committed: bool = True

    @property
    def _requested_selection_and_indices(self):
        if not self._requested_attributes_committed:
            self._cached_selection_and_indices.cache_clear()
            self._requested_attributes_committed = True
        return self._cached_selection_and_indices(
            stage_ref=CachableRef(
                self._requested_stage, 
                lambda stage_rt: stage_rt.GetStageId(),
            ),
            attribute_specs_ref=CachableRef(
                self._requested_attributes.values(), 
                lambda _: id(self._requested_attributes),
            ),
            path_ref=CachableRef(self._requested_path, id),
            # TODO
            # device=warp.get_device().alias,
        )
    
    def create_attribute(
        self, 
        name: str, 
        value_type: str, 
        custom: bool = False,
    ):
        # TODO
        raise NotImplementedError

    def request_attribute(
        self, 
        name: str, 
        value_type: str,
        value_access: Literal["read", "write", "readwrite"] | None = None,
    ):
        usdrt = self._kernel._usdrt
        if value_access is None:
            value_access = "read"

        # TODO
        match value_type:
            case str():
                value_type_rt = getattr(usdrt.Sdf.ValueTypeNames, value_type)
            case _:
                raise ValueError("TODO")

        match value_access:
            case "read":
                value_access_rt = usdrt.Usd.Access.Read
            case "readwrite":
                value_access_rt = usdrt.Usd.Access.ReadWrite
            case "write":
                value_access_rt = usdrt.Usd.Access.Overwrite
            case _:
                raise ValueError("TODO")

        spec = _USDRTAttrSpec((value_type_rt, name, value_access_rt))
        if self._requested_attributes.get(name) == spec:
            return
        self._requested_attributes[name] = spec
        self._requested_attributes_committed = False

    def _get_attribute(self, name: str):
        if name not in self._requested_attributes:
            # TODO
            warnings.warn(f"TODO call `{self.request_attribute}`: {name}")
        # TODO torch tensor
        selection, indices = self._requested_selection_and_indices
        return warp.fabricarray(data=selection, attrib=name), indices
    
    def read_attribute(self, name: str):
        attr, indices = self._get_attribute(name)
        if usdrt_is_jagged(attr):
            return usdrt_read_torch_jagged(attr, indices=indices)
        return usdrt_read_torch(attr, indices=indices)
    
    def write_attribute(self, name: str, value: ...):
        attr, indices = self._get_attribute(name)
        if usdrt_is_jagged(attr):
            # TODO
            raise NotImplementedError("TODO")
            return
        usdrt_write_torch(attr, indices=indices, a_in=value)

    @functools.cached_property
    def xform(self):
        return USDRTPrimXFormView(self)


class USDRTPrimXFormView(ProtoPrimXformView):
    def __init__(self, rt_view: USDRTPrimView):
        self._rt_view = rt_view

    @functools.lru_cache(maxsize=1)
    def _cached_stage_fabric_hierachy(
        self, 
        stage_ref: CachableRef["usdrt.Usd.Stage"], 
    ):
        usdrt = self._rt_view._kernel._usdrt
        fabric_hierarchy_iface = usdrt.hierarchy.IFabricHierarchy()
        return fabric_hierarchy_iface.get_fabric_hierarchy(
            stage_ref.value.GetFabricId(), 
            stage_ref.value.GetStageIdAsStageId(),
        )

    def _sync_world_xforms(self):        
        hierarchy = self._cached_stage_fabric_hierachy(
            stage_ref=CachableRef(
                self._rt_view._requested_stage, 
                lambda stage_rt: stage_rt.GetStageId(),
            ),
        )
        hierarchy.track_local_xform_changes(True)
        hierarchy.update_world_xforms()
        # TODO FIXME perf: https://github.com/isaac-sim/IsaacSim/issues/391
        hierarchy.update_world_xforms_gpu(True)

    def _get_world_matrix_attribute(self):
        self._rt_view.request_attribute(
            "omni:fabric:worldMatrix",
            value_type="Matrix4d",
            value_access="read",
        )
        self._sync_world_xforms()
        return self._rt_view._get_attribute("omni:fabric:worldMatrix")
    
    def _get_local_matrix_attribute(
        self, 
        value_access: Literal["read", "write", "readwrite"] | None = None,
    ):
        self._rt_view.request_attribute(
            "omni:fabric:localMatrix",
            value_type="Matrix4d",
            value_access=value_access,
        )
        return self._rt_view._get_attribute("omni:fabric:localMatrix")

    # TODO doc row-major or convert to col-major
    def read_world_matrix(self):
        attr, indices = self._get_world_matrix_attribute()
        self._sync_world_xforms()
        # TODO allow sparse when indices non-cont?
        return usdrt_read_torch(attr, indices=indices)
    
    # TODO doc row-major or convert to col-major
    def read_local_matrix(self):
        attr, indices = self._get_local_matrix_attribute()
        return usdrt_read_torch(attr, indices=indices)
    
    # TODO doc row-major or convert to col-major
    def write_local_matrix(self, value):
        attr, indices = self._get_local_matrix_attribute()
        value = torch.broadcast_to(
            torch.asarray(
                value, 
                dtype=torch.float64, 
                device=warp.device_to_torch(attr.device),
            ), 
            size=(len(attr), 4, 4),
        )
        usdrt_write_torch(
            attr,
            indices=indices,
            a_in=value,
        )
        # TODO enable opt-out using ctx mgr!!!
        if True:
            self._sync_world_xforms()
