# TODO
class _TODOTestUSDPrimHelper:
    def _todo_test_pose(self):
        _ = """
        # TODO add as test
        camera.pose = camera.pose
        camera.pose
        camera.pose = Pose(p=[.1, .3, .5]).rotated([.1, .2, .3])
        camera.pose
        scene.viewer.clear_drawings()
        scene.viewer.draw_pose(camera.pose)
        scene.viewer.draw_pose(Pose(p=[.1, .3, .5]).rotated([.1, .2, .3]))
        """
        raise NotImplementedError
