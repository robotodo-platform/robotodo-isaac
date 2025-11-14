r"""
TODO doc
"""


import dataclasses
import warnings

# TODO
import numpy
import einops
# TODO
from tensorspecs import TensorLike


@dataclasses.dataclass(slots=True)
class Plane:
    size: TensorLike["*? xy:2", "float"] \
        = dataclasses.field(default_factory=lambda: [1., 1.])


@dataclasses.dataclass(slots=True)
class Box:
    size: TensorLike["*? xyz:3", "float"] \
        = dataclasses.field(default_factory=lambda: [1., 1., 1.])


@dataclasses.dataclass(slots=True)
class Sphere:
    radius: TensorLike["*?", "float"] \
        = dataclasses.field(default_factory=lambda: 1.)


@dataclasses.dataclass(slots=True)
class PolygonMesh:
    vertices: TensorLike["*? vertex xyz:3", "float"]
    face_vertex_counts: TensorLike["*? face", "int"]
    # TODO doc len(face_vertex_indices) == sum(face_vertex_counts)
    face_vertex_indices: TensorLike["*? face_vertex", "int"]

    # TODO patches
    @classmethod
    def from_plane(cls, plane: Plane = Plane()):
        signs = numpy.asarray(
            [
                [-1, -1, 0],
                [+1, -1, 0],
                [+1, +1, 0],
                [-1, +1, 0],
            ],
            dtype=float,
        )

        face_template = numpy.asarray(
            [
                [0, 1, 2, 3],
            ],
            dtype=numpy.int64,
        )

        # TODO handle xy:1
        vertices = einops.einsum(
            signs,
            numpy.insert(plane.size, 2, values=0, axis=-1),
            "vertex xyz, ... xyz -> ... vertex xyz",
        )
        *s_batch, _, _ = numpy.shape(vertices)
        num_faces, num_vertices_per_face = face_template.shape
        face_vertex_counts = numpy.broadcast_to(
            num_vertices_per_face, 
            shape=(*s_batch, num_faces),
        )
        face_vertex_indices = numpy.broadcast_to(
            face_template.flatten(),
            shape=(*s_batch, *face_template.flatten().shape)
        )

        return cls(
            vertices=vertices,
            face_vertex_counts=face_vertex_counts,
            face_vertex_indices=face_vertex_indices,
        )

    # TODO ref omni.physx.scripts.physicsUtils.create_mesh_cube
    @classmethod
    def from_box(cls, box: Box = Box()):
        signs = numpy.asarray(
            [
                [+1, -1, -1],
                [+1, +1, -1],
                [+1, +1, +1],
                [+1, -1, +1],
                [-1, -1, -1],
                [-1, +1, -1],
                [-1, +1, +1],
                [-1, -1, +1],
            ],
            dtype=float,
        )

        face_template = numpy.asarray(
            [
                [0, 1, 2, 3], 
                [1, 5, 6, 2], 
                [3, 2, 6, 7], 
                [0, 3, 7, 4], 
                [1, 0, 4, 5], 
                [5, 4, 7, 6],
            ],
            dtype=numpy.int64,
        )

        vertices = einops.einsum(
            signs,
            numpy.asarray(box.size) / 2,
            "vertex xyz, ... xyz -> ... vertex xyz",
        )
        *s_batch, _, _ = numpy.shape(vertices)
        num_faces, num_vertices_per_face = face_template.shape
        face_vertex_counts = numpy.broadcast_to(
            num_vertices_per_face, 
            shape=(*s_batch, num_faces),
        )
        face_vertex_indices = numpy.broadcast_to(
            face_template.flatten(),
            shape=(*s_batch, *face_template.flatten().shape)
        )

        return cls(
            vertices=vertices,
            face_vertex_counts=face_vertex_counts,
            face_vertex_indices=face_vertex_indices,  
        )
    
    @classmethod
    def from_sphere(cls, sphere: Sphere = Sphere()):
        # TODO
        raise NotImplementedError

    # TODO FIXME batching
    # TODO FIXME perf
    # TODO NOTE ref: from isaacsim.replicator.grasping.sampler_utils import usd_mesh_to_trimesh
    def triangulate(self, copy: bool = False):
        # TODO !!!
        if copy: 
            raise NotImplementedError("TODO")

        vertices = self.vertices
        # TODO FIXME this should be flat
        tri_face_vertex_indices = []

        offset = 0
        # TODO batch
        # Iterate over the face vertex counts where 'count' is the number of vertices in each face
        # TODO rm: 
        for count in numpy.array(self.face_vertex_counts, ndmin=1):
        # for count in numpy.broadcast_to(self.face_vertex_counts, numpy.asarray(numpy.shape(self.vertices))[:-1]):
            # Current face indices using the offset and count
            indices = self.face_vertex_indices[offset : offset + count]
            if count == 3:
                # If the face is a triangle, add it directly to the faces list
                tri_face_vertex_indices.extend(indices)
            elif count == 4:
                # If the face is a quad, split it into two triangles
                tri_face_vertex_indices.extend([indices[0], indices[1], indices[2]])
                tri_face_vertex_indices.extend([indices[0], indices[2], indices[3]])
            else:
                # Fan triangulation for polygons with more than 4 vertices
                # NOTE: This approach works for convex polygons but may not be optimal for concave ones
                for i in range(1, count - 1):
                    tri_face_vertex_indices.extend([indices[0], indices[i], indices[i + 1]])
            offset += count

        return PolygonMesh(
            vertices=vertices,
            face_vertex_counts=3,
            face_vertex_indices=numpy.asarray(tri_face_vertex_indices),
        )


# TODO 
@dataclasses.dataclass(slots=True)
class GeometryCollection:
    ...


# TODO FIXME batching
def export_trimesh(geometry: Plane | Box | PolygonMesh):
    try:
        import trimesh
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "Install `trimesh` to use this feature"
        ) from error

    match geometry:
        case Plane():
            return export_trimesh(geometry=PolygonMesh.from_plane(geometry))
        
        case Box():
            return export_trimesh(geometry=PolygonMesh.from_box(geometry))
        
        case PolygonMesh():
            for face_vertex_count in (3, 4):
                if numpy.array_equiv(geometry.face_vertex_counts, face_vertex_count):
                    # TODO batch???
                    return trimesh.Trimesh(
                        vertices=geometry.vertices,
                        faces=numpy.reshape(geometry.face_vertex_indices, (-1, face_vertex_count)),
                    )
            warnings.warn(
                f"{PolygonMesh} with non-triangular or non-quad faces are "
                f"not supported by trimesh, converting: {geometry}"
            )
            return export_trimesh(geometry=geometry.triangulate())
        
    # TODO raise
    warnings.warn(f"TODO WIP: {geometry}")


# TODO
def import_trimesh(obj: "trimesh.Trimesh"):
    vertices = obj.vertices
    num_faces, num_vertices_per_face = obj.faces.shape
    face_vertex_counts = numpy.broadcast_to(num_vertices_per_face, (num_faces, ))
    face_vertex_indices = obj.faces.reshape(-1)

    return PolygonMesh(
        vertices=vertices,
        face_vertex_counts=face_vertex_counts,
        face_vertex_indices=face_vertex_indices,
    )