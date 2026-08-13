# BayesISOLA

BayesISOLA is an independently maintained Python package for centroid moment-tensor inversion based on the Bayesian ISOLA methodology of Vackář et al. (2017). It preserves the original BayesISOLA inversion and Axitra Green's-function algorithms while adding automated workflows, reusable Green's-function utilities, model-specific Axitra support, multiprocessing improvements, and installable platform wheels.

## Installation

BayesISOLA requires Python 3.10 or newer.

### Binary wheels

Release wheels are built for Linux x86_64 and Windows x86_64. The wheels contain the compiled Axitra `gr_xyz` and `elemse` executables, so a Fortran compiler is not required at runtime when installing a release wheel.

Install a downloaded wheel with:

```bash
python -m pip install /path/to/bayesisola-<version>-<platform>.whl
```

The Linux wheel vendors the required non-system Fortran runtime libraries during the manylinux repair step. The Windows wheel installs the required MinGW runtime DLLs beside `gr_xyz.exe` and `elemse.exe` under `BayesISOLA/_bin` so the Axitra programs can run without MinGW on `PATH`.

### Installation from source

Source and Git installations compile the bundled Axitra Fortran programs automatically through CMake and `scikit-build-core`. A working GNU Fortran compiler is therefore required.

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

On Windows, source installation requires a GNU Fortran/MinGW toolchain available to CMake. No manual compilation inside a repository `green/` directory is required.

## Axitra runtime layout

The immutable Axitra build sources are kept under:

```text
fortran/axitra/
```

Installed Axitra executables are kept under:

```text
BayesISOLA/_bin/
```

Every inversion uses a separate writable Green's-function workspace under `<outdir>/green` by default. Mutable files such as `crustal.dat`, `station.dat`, `grdat*.hed`, `gr*.hea`, `gr*.hes`, and `elemse*.dat` are written only to that event-specific workspace. A different writable workspace can be supplied with `green_dir=`.

## Core interface

The historical BayesISOLA interface is preserved:

```python
import BayesISOLA

inputs = BayesISOLA.load_data(outdir="output/event")
inputs.read_event_info("event.isl")
inputs.set_source_time_function("triangle", 2.0)
inputs.read_network_coordinates("network.stn")
inputs.read_crust("crustal.dat")
```

## Automated workflow

The automated centroid-moment-tensor workflow is available from:

```python
from BayesISOLA.workflows import run_auto_cmt
```

Reusable velocity-model and Green's-function preparation utilities are available from:

```python
from BayesISOLA import gf_helpers
```

The automated workflow retains Axitra as the default Green's-function backend and also supports the corrected EarthScope Syngine backend through its explicit `gf_source` interface.

## Examples

The repository examples resolve their input and output paths relative to the script location and can therefore be launched from any working directory in a source checkout:

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
tests/                      Packaging, path, and numerical regression tests
CMakeLists.txt              Axitra build definition
pyproject.toml              Python build and package metadata
```

## Project history and attribution

This repository is an independently maintained development derived from the original BayesISOLA project developed by Jiří Vackář and collaborators. The inherited Git history is preserved for scientific attribution and provenance. The original project repository is [vackar/BayesISOLA](https://github.com/vackar/BayesISOLA), and Jiří Vackář's [legacy BayesISOLA documentation](https://geo.mff.cuni.cz/~vackar/BayesISOLA/) preserves the original documentation, installation notes, module summary, and examples.

## Reference

J. Vackář, J. Burjánek, F. Gallovič, J. Zahradník, and J. Clinton (2017), *Bayesian ISOLA: new tool for automated centroid moment tensor inversion*, Geophysical Journal International, 210(2), 693–705.

## License

The repository distributes the GNU General Public License, version 3 text in `LICENSE`.
