BayesISOLA documentation
========================

BayesISOLA is an independently maintained open-source Python package for seismic source inversion using a point-source centroid moment tensor representation. It preserves the original BayesISOLA inversion and Axitra algorithms while adding automated workflows, reusable Green's-function utilities, model-specific Axitra support, multiprocessing improvements, and installable platform wheels.

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

Binary release wheels are built for Linux x86_64 and Windows x86_64. They contain the compiled Axitra executables, so a Fortran compiler is not required at runtime when installing a release wheel. A downloaded wheel can be installed with::

   python -m pip install /path/to/bayesisola-<version>-<platform>.whl

Source and Git installations compile the bundled Axitra Fortran programs automatically through CMake and ``scikit-build-core`` and therefore require a working GNU Fortran compiler.

On Ubuntu/WSL::

   sudo apt update
   sudo apt install -y gfortran git
   python -m pip install "git+https://github.com/AaronJR474/BayesISOLA.git"

From a local checkout::

   python -m pip install .

On Windows, source installation requires a GNU Fortran/MinGW toolchain available to CMake.

The installed ``gr_xyz`` and ``elemse`` executables live under ``BayesISOLA/_bin``. Mutable Axitra inputs and Green's-function outputs are kept in an event-specific ``<outdir>/green`` workspace by default.

Requirements
------------

BayesISOLA requires Python 3.10 or newer. Runtime Python dependencies are declared in ``pyproject.toml`` and installed by pip. The main scientific dependencies include NumPy, SciPy, matplotlib, ObsPy, pyproj, pandas, and requests. ``tqdm`` and ``threadpoolctl`` are included for progress reporting and multiprocessing control.

Examples
--------

All inputs required by the SAC example are included under ``examples/input``. In a source checkout, the example scripts resolve their input and output paths relative to the script location and can therefore be launched from any working directory::

   python examples/example_2_SAC.py
   python examples/example_2_fdsnws.py

Automated workflow
------------------

The automated centroid-moment-tensor workflow is available through ``BayesISOLA.workflows.run_auto_cmt``. Reusable velocity-model and Green's-function preparation utilities are provided by ``BayesISOLA.gf_helpers``. Axitra remains the default Green's-function backend; the corrected EarthScope Syngine backend is available through the workflow's explicit ``gf_source`` interface.

Project history and attribution
-------------------------------

This repository is an independently maintained development derived from the
original BayesISOLA project developed by Jiří Vackář and collaborators. The
inherited Git history is preserved for scientific attribution and provenance.
The original repository is `vackar/BayesISOLA
<https://github.com/vackar/BayesISOLA>`_, and the original documentation and
examples remain available through `Jiří Vackář's legacy BayesISOLA
documentation <https://geo.mff.cuni.cz/~vackar/BayesISOLA/>`_.

License
-------

The repository distributes the GNU General Public License, version 3 text in ``LICENSE``.

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
