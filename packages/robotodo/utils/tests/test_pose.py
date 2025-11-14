
import numpy
from robotodo.utils.pose import Pose


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