
import functools

import numpy
from robotodo.utils.pose import Pose
from robotodo.utils.geometry import PolygonMesh
from robotodo.engines.core.entity_selector import PathExpression, PathExpressionLike

from .scene import Scene


class Entity:
    # TODO support usd prims directly??
    def __init__(self, path: PathExpressionLike, scene: Scene):
        self._scene = scene
        self._path = PathExpression(path)

    # TODO invalidate!!!!
    @functools.cached_property
    def _isaac_usdgeom_xform_cache(self):
        # TODO Usd.TimeCode.Default()
        cache = self._scene._kernel.pxr.UsdGeom.XformCache()

        def _on_changed(notice, sender):
            # TODO
            cache.Clear()

        # TODO NOTE life cycle
        cache._notice_handler = _on_changed
        # TODO
        cache._notice_token = self._scene._kernel.pxr.Tf.Notice.Register(
            self._scene._kernel.pxr.Usd.Notice.ObjectsChanged, 
            cache._notice_handler, 
            self._scene._usd_stage,
        )

        return cache
    
    # TODO
    # TODO https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/usd/transforms/compute-prim-bounding-box.html
    @functools.cached_property
    def _isaac_usdgeom_bbox_cache(self):
        raise NotImplementedError
        return self._scene._kernel.pxr.UsdGeom.BBoxCache(
            self._scene._kernel.pxr.Usd.TimeCode.Default(),
            [self._scene._kernel.pxr.UsdGeom.Tokens.default_,],
        )

    # TODO FIXME performance thru prim obj caching
    @property
    # TODO invalidate !!!!!
    # @functools.cached_property
    def _isaac_prims(self):
        return [
            self._scene._usd_stage.GetPrimAtPath(p)
            for p in self._scene.resolve(self._path)
        ]
        
    @property
    def pose(self):
        # TODO 
        return Pose.from_matrix(
            numpy.stack([
                numpy.asarray(
                    self._isaac_usdgeom_xform_cache
                    .GetLocalToWorldTransform(prim)
                    .RemoveScaleShear()
                ).T # TODO NOTE col-major
                for prim in self._isaac_prims
            ])
        )
    
    @pose.setter
    def pose(self, value: Pose):
        pxr = self._scene._kernel.pxr
        omni = self._scene._kernel.omni

        p = numpy.broadcast_to(value.p, (len(self._isaac_prims), 3))
        q = numpy.broadcast_to(value.q, (len(self._isaac_prims), 4))

        p_vec3fs = pxr.Vt.Vec3fArrayFromBuffer(p)
        # NOTE this auto-converts from xyzw to wxyz
        q_quatfs = pxr.Vt.QuatfArrayFromBuffer(q)

        with pxr.Sdf.ChangeBlock():
            for prim, p_vec3f, q_quatf in zip(self._isaac_prims, p_vec3fs, q_quatfs):
                xformable = pxr.UsdGeom.Xformable(prim)
                omni.physx.scripts.physicsUtils \
                    .set_or_add_translate_op(xformable, p_vec3f)
                omni.physx.scripts.physicsUtils \
                    .set_or_add_orient_op(xformable, q_quatf)

    # TODO scaling
    # TODO optimize: instanceable assets may have shared geoms
    @property
    def geometry(self):
        """
        TODO doc

        The *physical* geometry of the entity.
        If the entity has no collision then this would be empty.
        
        """

        # TODO
        pxr = self._scene._kernel.pxr

        # TODO geom TensorView("n? geom")
        # -or- {Mesh: <Mesh>}??? -or- <collection>.find_by_type(Mesh)???

        geoms = []

        for prim in self._isaac_prims:
            prim_geoms = []

            prim_scale_factors = (
                pxr.Gf.Transform(
                    self._isaac_usdgeom_xform_cache
                    .GetLocalToWorldTransform(prim)
                )
                .GetScale()
            )

            # NOTE the first child_prim would be the prim itself
            for child_prim in pxr.Usd.PrimRange(
                prim, 
                pxr.Usd.TraverseInstanceProxies(
                    pxr.Usd.PrimAllPrimsPredicate
                ),
            ):
                # TODO
                # if not child_prim.HasAPI(pxr.UsdPhysics.CollisionAPI):
                #     continue

                match child_prim:
                    case _ if child_prim.IsA(pxr.UsdGeom.Mesh):
                        api = pxr.UsdGeom.Mesh(child_prim)
                        # TODO lazy??
                        prim_geoms.append(
                            PolygonMesh(
                                vertices=numpy.asarray(api.GetPointsAttr().Get()) * prim_scale_factors,
                                face_vertex_counts=numpy.asarray(api.GetFaceVertexCountsAttr().Get()),
                                face_vertex_indices=numpy.asarray(api.GetFaceVertexIndicesAttr().Get()),
                            )
                        )
                    case _:
                        # TODO
                        pass

            geoms.append(prim_geoms)

        return geoms