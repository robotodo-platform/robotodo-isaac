# SPDX-License-Identifier: Apache-2.0

"""
Entity.
"""

import io
from typing import IO

import numpy
from tensorspecs import TensorLike
from robotodo.engines.core.error import InvalidReferenceError
from robotodo.engines.core.path import PathExpressionLike, is_path_expression_like, PathExpression
from robotodo.engines.core.entity import ProtoEntity
from robotodo.engines.isaac.scene import Scene
from robotodo.engines.isaac._utils.usd import (
    usd_add_reference,
    USDPrimRef, 
    is_usd_prim_ref,
    USDPrimPathRef, 
    USDPrimPathExpressionRef,
)


class Entity(ProtoEntity):
    # TODO
    _usd_prim_ref: USDPrimRef
    _scene: Scene

    @classmethod
    def load(cls, ref: PathExpressionLike, source: "str | IO | pxr.Sdf.Layer", scene: Scene):
        expr = PathExpression(ref)

        # TODO
        prims = usd_add_reference(
            stage=scene._usd_stage,
            paths=expr.expand(),
            resource=source,
            kernel=scene._kernel,
        )         

        return cls(lambda: prims, scene=scene)

    def __init__(
        self, 
        ref: "Entity | USDPrimRef | PathExpressionLike", 
        scene: Scene | None = None,
    ):
        match ref:
            case Entity() as entity:
                # TODO
                assert scene is None
                self._usd_prim_ref = entity._usd_prim_ref
                self._scene = entity._scene
            case ref if is_usd_prim_ref(ref):
                assert scene is not None
                self._usd_prim_ref = ref
                # TODO create scene on the fly??
                self._scene = scene                
            case expr if is_path_expression_like(ref):
                # TODO
                assert scene is not None
                self._usd_prim_ref = USDPrimPathExpressionRef(
                    expr,
                    stage_ref=lambda: scene._usd_stage,
                )
                self._scene = scene
            case _:
                # TODO
                raise InvalidReferenceError(ref)
            
    # TODO
    @property
    def prototypes(self):
        return set([ProtoEntity])

    # TODO
    def astype(self, prototype):
        match prototype:
            case _ if issubclass(prototype, ProtoEntity):
                return Entity(self)
            case _:
                raise ValueError("TODO")
        
    @property
    def path(self):
        return numpy.asarray([
            prim.GetPath().pathString
            for prim in self._usd_prim_ref()
        ])
    
    @property
    def scene(self):
        return self._scene
