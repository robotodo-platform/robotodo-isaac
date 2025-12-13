Introduction
============

Overview
--------

``robotodo.engines.isaac`` is a member simulation engine of the ``robotodo`` platform.
It is based on NVIDIA Omniverse and Isaac Sim. As a result, when using the engine, you gain
additional features such as photo-realistic rendering, interoperation with CAD tools, and
collaborative scene editing.

``robotodo`` is an open platform aimed at making robotics truly accessible by unifying sim+real
control and algorithm prototyping. ``robotodo`` has the following characteristics:

Hackable
   Instead of using layers upon layers of abstractions, we define only an essential set of
   abstract interfaces for developers to implement. Inheritance more than three layers deep is
   generally also not permitted unless absolutely necessary. Exploring the codebase can generally
   be done with a few mouse clicks. This is very different from ROS and IsaacLab, which are often
   frowned upon for their framework-y architecture.

Batteries included
   ``robotodo.utils`` provides the tooling foundation for common robotics-related workflows such
   as 3D spatial transformations and asynchronous streaming. Different from ROS and ``pxr.Tf``,
   which are largely geared towards the C++ community, ``robotodo.utils`` is closer to the Python
   ecosystem in that PyTorch and NumPy are exclusively used for tensor data structures.
   ``robotodo.engines`` also asks more from the developers than other simulators: collision-free IK
   for articulations, for instance, must be implemented by the backend. High performance computation
   is also much easier because our batteries are vectorized by default: batch spatial transforms and
   IK, for instance, can be done via single function calls.

Intuitive API = simplified mental model = lower barrier of entry = faster iteration
   The ``robotodo.engines`` API abstracts away the complexity of sim/real state management and
   parallel data processing while keeping the core functionality intact. You have the freedom to
   control your simulation procedurally. Modifying a simulated subject's state often only involves
   a single property assignment operation. The data format is often the familiar PyTorch/NumPy
   tensor with a leading batch dimension. Visualization features are also included. Simulation
   frameworks such as IsaacLab, on the other hand, are overwhelmingly declarative and much less
   ergonomic: debugging and experimentation, as a result, are painful.

Features
--------

.. list-table::
   :header-rows: 1

   * - Feature
     - SAPIEN
     - MuJoCo
     - Genesis
     - IsaacLab
     - ``robotodo.engines.isaac``
   * - Physics engine
     - PhysX
     - MuJoCo
     - Taichi
     - PhysX, Newton [#newton]_
     - PhysX, Newton [#newton]_
   * - Graphics engine
     - ?
     - OpenGL, Custom
     - ?
     - NVIDIA RTX, Pixar Storm, Custom
     - NVIDIA RTX, Pixar Storm, Custom
   * - Deformable body support
     - —
     - ✅
     - ✅
     - ✅
     - ✅
   * - 3D interchange formats [#formats]_
     - Baseline
     - Baseline
     - Baseline, USD (Partial), MJCF (Partial)
     - Baseline, USD, MJCF
     - Baseline, USD, MJCF
   * - Robotics interchange formats
     - URDF
     - URDF
     - URDF
     - URDF
     - URDF
   * - Batch API
     - ❌
     - ✅
     - ✅
     - ✅
     - ✅
   * - Procedural API
     - ✅
     - —
     - ✅
     - —
     - ✅
   * - Async API [#async]_
     - —
     - —
     - —
     - ☑️ Partial
     - ✅

TODOs
-----

- doc: improve documentation clarity (e.g. SI unit); add more examples.
- api: ensure consistent use of PyTorch tensors across implementations.
- ``robotodo.utils.pose``: replace SciPy implementation with Torch for better batch-processing performance.
- ``robotodo.engines.core.body``: revamp ``Body.geometry`` for better batch-processing performance.
- ``robotodo.engines.isaac``: use USDRT backend for better batch-processing performance.
- ``robotodo.engines.isaac.scene``: ``SceneViewer``: implement remote viewer.
- ``robotodo.engines.isaac.articulation``: implement builtin collision-free IK.
- ``robotodo.engines.core.material``: add support for visual features (e.g. color, reflectivity).
- ``robotodo.engines.isaac.sensor``: ``Camera``: implement projection matrix.
- ``robotodo.engines.core.sensor``: ``Camera``: add support for camera lens model.
- ``robotodo.engines.core.sensor``: add support for additional sensor types (e.g. lidar, imu).


.. rubric:: Notes
  
.. [#newton] Newton is experimental and will eventually replace PhysX.
.. [#formats] Baseline formats: STL, OBJ, FBX, GLB.
.. [#async] Async API is useful for interactive debugging and human-in-the-loop control.
