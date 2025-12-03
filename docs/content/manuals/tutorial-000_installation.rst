Installation
============

Before you begin, complete the following checklist:

- **Python**

  - Ensure Python **3.11** is installed and available as ``python``.
  - Ensure ``pip`` is available; if not, run ``python3 -m ensurepip``.

- **Operating system**

  - **Linux**: Check your GLIBC version with ``ldd --version``.
    Upgrade your distribution if GLIBC is below **2.35**. 
    Find supported versions of common distributions 
    `here <https://gist.github.com/richardlau/6a01d7829cc33ddab35269dacc127680>`_.

  - **Windows**: You may need to enable `long path support <https://pip.pypa.io/warnings/enable-long-paths>`_ to avoid
    installation errors caused by path length limits.

.. seealso::

   Isaac Sim Python installation via ``pip``:
   https://docs.isaacsim.omniverse.nvidia.com/5.0.0/installation/install_python.html#install-isaac-sim-using-pip


TODO