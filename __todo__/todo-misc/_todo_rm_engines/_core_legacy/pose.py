"""
TODO

"""


from typing import TypedDict

import numpy
import scipy.spatial.transform
import einops


# TODO
class ProtoPose(TypedDict):
    pass


# TODO support (*batch_dims, data)
# TODO dataclass???
# TODO testing
# TODO from/to 4x4 trans matrix !!!!!!
class Pose:
    p: ...  # TODO xyz
    q: ...  # TODO xyzw
    # angles: ... # TODO radian xyz

    @classmethod
    def from_matrix(cls, matrix):
        # TODO !!!!
        matrix = numpy.asarray(matrix)
        return Pose(
            p=matrix[..., :3, 3],
            q=scipy.spatial.transform.Rotation.from_matrix(matrix[..., :3, :3]).as_quat(),
        )

    def __init__(
        self,
        # TODO
        # other: ProtoPose,
        *,
        p: ... = [0, 0, 0],
        q: ... = [0, 0, 0, 1],
        # angles: ... = [0, 0, 0],
    ):
        self.p = numpy.asarray(p)
        self.q = numpy.asarray(q)

    # TODO better ux!!!
    def __getitem__(self, *index):
        return Pose(
            p=self.p[*index, :],
            q=self.q[*index, :],
        )

    # TODO
    def facing(self, pose: "Pose"):
        # TODO
        raise NotImplementedError

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
    def angles(self):
        q_packed, ps = einops.pack([self.q], "* quat")
        [eulers] = einops.unpack(
            # TODO
            scipy.spatial.transform.Rotation.from_quat(q_packed).as_euler("XYZ"), 
            ps, 
            "* eulers",
        )
        return eulers
    

    # TODO !!!
    def to_matrix(self):
        p = numpy.asarray(self.p)
        q = numpy.asarray(self.q)

        # TODO
        q_packed, ps = einops.pack([self.q], "* quat")
        R = scipy.spatial.transform.Rotation.from_quat(q_packed).as_matrix()
        [R] = einops.unpack(R, ps, "* out_base in_base",)

        # handle batching
        # if p.ndim == 1:  # single pose
        #     T = np.eye(4)
        #     T[:3, :3] = R
        #     T[:3, 3] = p
        # else:  # batched poses [..., 3] / [..., 4]
        eye = numpy.eye(4)
        shape = p.shape[:-1] + (4, 4)
        T = numpy.broadcast_to(eye, shape).copy()
        T[..., :3, :3] = R
        T[..., :3, 3] = p
        return T


# TODO
def _todo_test_pose():
    a = Pose.from_matrix(
        numpy.array([
            [[0.0, -1.0, 0.0, 1.0],
            [1.0,  0.0, 0.0, 2.0],
            [0.0,  0.0, 1.0, 3.0],
            [0.0,  0.0, 0.0, 1.0]],
            
            [[1.0, 0.0, 0.0, 4.0],
            [0.0, 1.0, 0.0, 5.0],
            [0.0, 0.0, 1.0, 6.0],
            [0.0, 0.0, 0.0, 1.0]]
        ])    
    )

    a.to_matrix()

    Pose.from_matrix(
        numpy.array([
            [0.0, -1.0, 0.0, 1.0],
            [1.0,  0.0, 0.0, 2.0],
            [0.0,  0.0, 1.0, 3.0],
            [0.0,  0.0, 0.0, 1.0],
        ])
    ).to_matrix()
    # TODO
    # Pose().inv()
    # Pose(p=[1, 0, 0], q=[0, 0, 0, 1]) * Pose(p=[0, 0, 0], q=[0, 0, 0, 1])
    # Pose(p=[0, 0, 0], q=[0, 0, 0, 1]).inv()