Installation
============

Before you begin, complete the following check list:

- **Internet**
  The installation requires internet to complete.
  Offline installation is possible but is outside the scope of this document.

- **Python**

  - Ensure **Python 3.11** [#ref-isaacsim_pip]_ is installed and available as ``python3``. 
    Run ``python3 --version`` to check.

  - Ensure Python package manager pip is installed and available as ``python3 -m pip``.
    Run ``python3 -m ensurepip`` to install `pip` if not exists.

- **Operating system**

  - **Linux**: 
    - Check your GLIBC version with ``ldd --version``.
      Upgrade your distribution if GLIBC is below **2.35**. 
      Find supported versions of common distributions 
      `here <https://gist.github.com/richardlau/6a01d7829cc33ddab35269dacc127680>`_. [#ref-isaacsim_pip]_
    - If you are in a conda environment, it may be necessary to run ``conda install 'libstdcxx>11'``
      to get rid of ``libstdc++``-related errors.

  - **Windows**: 
    You may need to enable `long path support <https://pip.pypa.io/warnings/enable-long-paths>`_ 
    to avoid installation errors caused by path length limits. [#ref-isaacsim_pip]_

- **Legal matters**
  ``robotodo-isaac`` uses components from NVIDIA Omniverse. 
  By installing or using Omniverse Kit, you agree to the terms of 
  `NVIDIA OMNIVERSE LICENSE AGREEMENT (EULA) <https://docs.omniverse.nvidia.com/platform/latest/common/NVIDIA_Omniverse_License_Agreement.html>`_.


Getting Started
---------------

The easiest way to install ``robotodo-isaac`` is through ``pip``:

.. code-block:: shell

  python3 -m pip install 'robotodo-isaac @ git+https://github.com/robotodo-platform/robotodo-isaac.git'

.. note:: 
  Effort to publish to `PyPI <https://pypi.org>`_ is underway.

.. note::
  Find alternative ways to install ``robotodo-isaac`` 
  `here <https://pip.pypa.io/en/stable/cli/pip_install/#examples>`_.

(Optional) After the installation is done, verify your installation 
by running the following command:

.. code-block:: shell

  python3 -m robotodo.engines.isaac

Allow a few minutes for the app to initialize. If nothing goes wrong
you should see a window that looks much similar to this: [#ref-omniverse-kit-ui]_

.. image:: https://docs.omniverse.nvidia.com/kit/docs/kit-app-template/108.0/_images/graphical_kit_app_stack.png


References
----------

.. [#ref-isaacsim_pip] https://docs.isaacsim.omniverse.nvidia.com/5.0.0/installation/install_python.html#install-isaac-sim-using-pip
.. [#ref-omniverse-kit-ui] https://docs.omniverse.nvidia.com/kit/docs/kit-app-template/108.0/docs/kit_sdk_overview.html

