# Scene: Construction

TODO WIP 60%

Proposed code example:

Scene:
```python
from robotodo.engines.core import Scene
scene = Scene(backend="sapien")
# -or-
from robotodo.engines.sapien import Scene
scene = Scene()
# -or-
import robotodo.engines.core
robotodo.engines.core.defaults.backend = "sapien"
scene = Scene()
```

Robot:
```python
from robotodo.engines.core import Articulation
robot = Articulation.from_file(
    "path/to/some/robot.urdf", 
    scene=scene,
    batch=10,
)

robot.root_link.pose
robot.links
robot.joints[...].qpos
robot # __repr__ or _ipython_display_ a tree view
```

Object:
```python
from robotodo.engines.core import Actor
obj = Actor.from_file("path/to/some/object.usd", scene=scene)
```

Camera:
```python
from robotodo.engines.core import Camera
camera = Camera(scene=scene)
camera.image # returns a stream
```

Primitives:
```python
from robotodo.engines.core import Pose
Pose.spec # DictSpec({"p": TensorSpec("batch? xyz"), "q": TensorSpec("batch? 1ijk")})
Pose().rotate(rpy=...)
Pose().matrix
Pose().inv() * Pose()
Pose() * Pose()
Pose() == Pose()
```