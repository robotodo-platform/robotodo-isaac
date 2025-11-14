"""
TODO

"""


import numpy
import scipy.spatial.transform
import einops


# TODO typing
# TODO ref https://graphicscompendium.com/opengl/18-lookat-matrix
# def lookat(eye: ..., center: ..., up_axis: ...):
#     """
#     TODO doc

#     # >>> lookat([1, 1, 1], [0, 0, 0], up=[0, 0, 1])
#     # >>> lookat([0, 1, 0], [0, 0, 0], up=[0, 0, 1])

#     """

#     # TODO float cast necesito?
#     eye = numpy.asarray(eye, dtype="float")
#     center = numpy.asarray(center, dtype="float")
#     up_axis = numpy.asarray(up_axis, dtype="float")

#     forward = center - eye
#     forward /= numpy.linalg.norm(forward)

#     u = numpy.cross(forward, up_axis)
#     v = numpy.cross(u, forward)
#     w = -forward

#     # TODO +Z up +X forward !!!! instead of +Y up -Z forward
#     return numpy.stack((u, v, w), axis=-1)

# TODO https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/lookat-function/framing-lookat-function.html
def lookat(eye: ..., center: ..., up_axis: ...):
    """
    TODO doc: right-handed coords: +X right, +Y up, +Z forward

    https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/viewport/camera.html#look-at-a-prim
    
    """

    # TODO float cast necesito?
    eye = numpy.asarray(eye, dtype="float")
    center = numpy.asarray(center, dtype="float")
    up_axis = numpy.asarray(up_axis, dtype="float")

    forward = center - eye
    forward /= numpy.linalg.norm(forward)
    right = numpy.cross(up_axis, forward)
    up = numpy.cross(forward, right)

    # TODO
    return numpy.stack((forward, right, up), axis=-1)
    # return numpy.stack((right, up, forward), axis=-1)


# TODO
class ProtoPose:
    p: ...  # TODO xyz
    q: ...  # TODO xyzw
    # radians: ... # TODO radian xyz


# TODO FIXME perf
# TODO support (*batch_dims, data)
# TODO dataclass???
# TODO testing
class Pose:
    p: ...  # TODO xyz
    q: ...  # TODO xyzw
    # randians: ... # TODO radian xyz
    # degrees: ... # TODO radian xyz

    @classmethod
    def from_lookat(cls, p: ..., p_target: ..., up_axis: ... = [0, 0, 1]):
        """
        TODO doc
        """

        return cls(
            p=p,
            q=scipy.spatial.transform.Rotation.from_matrix(
                lookat(p, p_target, up_axis=up_axis)
            ).as_quat(),
        )

    # TODO support for rotation matrix as well?
    @classmethod
    def from_matrix(cls, matrix: ...):
        # TODO !!!!
        matrix = numpy.asarray(matrix)
        return cls(
            p=matrix[..., :3, 3],
            q=scipy.spatial.transform.Rotation.from_matrix(matrix[..., :3, :3]).as_quat(),
        )

    def __init__(
        self,
        # TODO
        # other: ProtoPose,
        p: ... = [0, 0, 0],
        q: ... = [0, 0, 0, 1],
        # radians: ... = [0, 0, 0],
        # degrees: ... = [0, 0, 0],
    ):
        # TODO broadcast
        self.p = numpy.asarray(p, dtype=numpy.float_)
        self.q = numpy.asarray(q, dtype=numpy.float_)

    def __repr__(self):
        return f"{Pose.__qualname__}(p={self.p!r}, q={self.q!r})"

    # TODO better ux!!!
    def __getitem__(self, index):
        if not isinstance(index, tuple):
            index = tuple((index, ))
        return Pose(
            p=self.p[*index, :],
            q=self.q[*index, :],
        )

    # TODO
    def facing(self, p: ..., up_axis: ... = [0, 0, 1]):
        return Pose.from_lookat(
            p=self.p,
            p_target=p,
            up_axis=up_axis,
        )

    def rotated(self, angles):
        # raise NotImplementedError
        # TODO
        return Pose(
            p=self.p,
            q=(
                scipy.spatial.transform.Rotation.from_quat(self.q)
                * scipy.spatial.transform.Rotation.from_euler("XYZ", angles)
            ).as_quat(),
        )

    def translated(self, positions):
        raise NotImplementedError

    # TODO FIXME performance
    def inv(self):
        rot_inv = scipy.spatial.transform.Rotation.from_quat(self.q).inv()
        return Pose(
            # TODO
            p=-rot_inv.apply(self.p),
            q=rot_inv.as_quat(),
        )

    # TODO FIXME performance
    def __mul__(self, other: "Pose"):
        rot = scipy.spatial.transform.Rotation.from_quat(self.q)
        rot_other = scipy.spatial.transform.Rotation.from_quat(other.q)

        return Pose(
            p=self.p + rot.apply(other.p),
            # TODO this is wrong???
            q=(rot * rot_other).as_quat(),
            # q=other.q,
        )
    
    # TODO pack/unpack for all other methods!!!!!
    @property
    def radians(self):
        q_packed, ps = einops.pack([self.q], "* quat")
        [eulers] = einops.unpack(
            # TODO
            scipy.spatial.transform.Rotation.from_quat(q_packed).as_euler("XYZ"), 
            ps, 
            "* eulers",
        )
        return eulers
    
    @property
    def degrees(self):
        return numpy.rad2deg(self.radians)
    
    # TODO
    def to_matrix(self):
        p = numpy.asarray(self.p)
        q = numpy.asarray(self.q)

        # TODO
        q_packed, ps = einops.pack([q], "* quat")
        R = scipy.spatial.transform.Rotation.from_quat(q_packed).as_matrix()
        [R] = einops.unpack(R, ps, "* out_base in_base",)

        eye = numpy.eye(4)
        shape = p.shape[:-1] + (4, 4)
        T = numpy.broadcast_to(eye, shape).copy()
        T[..., :3, :3] = R
        T[..., :3, 3] = p
        return T
    

    def __eq__(self, other: "Pose"):
        return (
            numpy.array_equiv(self.p, other.p) 
            and numpy.array_equiv(self.q, other.q)
        )
    
    def isclose(self, other: "Pose", rtol: float = 1.e-5, atol: float = 1.e-8):
        return numpy.allclose(
            (self.inv() * other).to_matrix(),
            numpy.eye(4),
            rtol=rtol, atol=atol,
        )
        

