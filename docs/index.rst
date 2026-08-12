BayesISOLA documentation
========================

BayesISOLA is an open-source Python package for seismic source inversion using a point-source centroid moment tensor representation.

The method is described in:

J. Vackář, J. Burjánek, F. Gallovič, J. Zahradník, and J. Clinton (2017),
*Bayesian ISOLA: new tool for automated centroid moment tensor inversion*,
Geophysical Journal International, 210(2), 693--705.

.. toctree::
   :hidden:
   :maxdepth: 3

   BayesISOLA.load_data
   BayesISOLA.grid
   BayesISOLA.process_data
   BayesISOLA.covariance_matrix
   BayesISOLA.resolve_MT
   BayesISOLA.plot
   BayesISOLA.axitra
   BayesISOLA.fileformats
   BayesISOLA.helpers
   BayesISOLA.histogram
   BayesISOLA.inverse_problem
   BayesISOLA.MouseTrap
   BayesISOLA.MT_comps
   BayesISOLA.gf_helpers
   BayesISOLA.syngine
   BayesISOLA.workflows

Installation
------------

Source installations compile the bundled Axitra Fortran programs automatically through CMake and ``scikit-build-core``. A Fortran compiler is required when installing directly from source or Git.

On Ubuntu/WSL::

   sudo apt update
   sudo apt install -y gfortran git
   python -m pip install "git+https://github.com/AaronJR474/BayesISOLA.git"

From a local checkout::

   python -m pip install .

The compiled ``gr_xyz`` and ``elemse`` executables are installed inside the Python package. Mutable Axitra inputs and Green's-function outputs are kept in an event-specific ``<outdir>/green`` workspace by default.

Requirements
------------

Runtime Python dependencies are declared in ``pyproject.toml`` and installed by pip. The main scientific dependencies include NumPy, SciPy, matplotlib, ObsPy, pyproj, pandas, and requests. ``tqdm`` and ``threadpoolctl`` are included for progress reporting and multiprocessing control.

Examples
--------

All inputs required by the SAC example are included under ``examples/input``. In a source checkout, the example scripts resolve their input and output paths relative to the script location and can therefore be launched from any working directory::

   python examples/example_2_SAC.py
   python examples/example_2_fdsnws.py

Automated workflow
------------------

The automated centroid-moment-tensor workflow is available through ``BayesISOLA.workflows.run_auto_cmt``. Reusable velocity-model and Green's-function preparation utilities are provided by ``BayesISOLA.gf_helpers``.

License
-------

The repository ``LICENSE`` file contains the GNU General Public License, version 3.

Module summary
--------------

.. currentmodule:: BayesISOLA

.. autosummary::

   BayesISOLA.load_data
   BayesISOLA.grid
   BayesISOLA.process_data
   BayesISOLA.covariance_matrix
   BayesISOLA.resolve_MT
   BayesISOLA.plot
   BayesISOLA.axitra
   BayesISOLA.fileformats
   BayesISOLA.helpers
   BayesISOLA.histogram
   BayesISOLA.inverse_problem
   BayesISOLA.MouseTrap
   BayesISOLA.MT_comps
   BayesISOLA.gf_helpers
   BayesISOLA.syngine
   BayesISOLA.workflows

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
