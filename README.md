# BayesISOLA

BayesISOLA is a Python implementation of centroid moment-tensor inversion based on the Bayesian ISOLA methodology. This fork retains the existing inversion and Axitra algorithms while adding automated workflows, reusable Green's-function helpers, model-specific Axitra support, improved multiprocessing, and install-time compilation of the bundled Axitra Fortran programs.

## Installation from source

The source installation builds the bundled `gr_xyz` and `elemse` Fortran programs automatically with CMake through `scikit-build-core`. A working Fortran compiler is therefore required when installing directly from Git.

On Ubuntu/WSL:

```bash
sudo apt update
sudo apt install -y gfortran git
python -m pip install "git+https://github.com/AaronJR474/BayesISOLA.git"
```

From a local checkout:

```bash
python -m pip install .
```

No manual compilation inside a repository `green/` directory is required. The compiled Axitra executables are installed under `BayesISOLA/_bin`, while every inversion uses its own writable Green's-function workspace under `<outdir>/green` by default.

Pre-built platform wheels are intentionally not configured yet; Linux/WSL and Windows wheel production will be handled separately after the source package is stable.

## Core interface

The historical interface is preserved:

```python
import BayesISOLA

inputs = BayesISOLA.load_data(outdir="output/event")
inputs.read_event_info("event.isl")
inputs.set_source_time_function("triangle", 2.0)
inputs.read_network_coordinates("network.stn")
inputs.read_crust("crustal.dat")
```

`load_data` now creates an event-specific Axitra workspace at `<outdir>/green`. A different writable workspace can be supplied with `green_dir=` when required.

## Automated workflow

The automated centroid-moment-tensor workflow is available from:

```python
from BayesISOLA.workflows import run_auto_cmt
```

Reusable velocity-model and Green's-function preparation utilities are available from:

```python
from BayesISOLA import gf_helpers
```

## Examples

The repository examples use paths relative to their own files, so they can be launched from any working directory in a source checkout:

```bash
python examples/example_2_SAC.py
python examples/example_2_fdsnws.py
```

The SAC example includes its waveform and response files under `examples/input/example_2_SAC`.

## Repository layout

```text
BayesISOLA/                 Python package
BayesISOLA/resources/       Runtime static resources
fortran/axitra/             Axitra Fortran build sources
examples/                   Runnable examples and example inputs
docs/                       Sphinx documentation
tests/                      Packaging and path regression tests
CMakeLists.txt               Axitra build definition
pyproject.toml               Python build and package metadata
```

The installed package contains the Python sources, runtime resources, and compiled Axitra executables. Mutable Axitra files such as `crustal.dat`, `station.dat`, `grdat*.hed`, and `elemse*.dat` are written only to the event-specific workspace.

## Reference

J. Vackář, J. Burjánek, F. Gallovič, J. Zahradník, and J. Clinton (2017), *Bayesian ISOLA: new tool for automated centroid moment tensor inversion*, Geophysical Journal International, 210(2), 693–705.

## License

The repository `LICENSE` file contains the GNU General Public License, version 3.
