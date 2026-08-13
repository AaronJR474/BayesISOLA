# BayesISOLA

BayesISOLA is an independently maintained Python package for centroid moment-tensor inversion based on the Bayesian ISOLA methodology of Vackář et al. (2017). The original BayesISOLA inversion and Axitra algorithms are retained, while the maintained version adds modern packaging, automated inversion workflows, reusable velocity-model and Green's-function utilities, improved results handling, and tested binary distributions for current Python versions.

## Installation

BayesISOLA requires Python 3.10 or newer. The 0.1.0 release wheels are tested on Python 3.10–3.14.

### Recommended: install a release wheel

For Windows x86_64 and Linux x86_64, the recommended installation is a pre-built wheel from the [BayesISOLA Releases](https://github.com/AaronJR474/BayesISOLA/releases) page.

Download the wheel for your platform and install it with:

```bash
python -m pip install /path/to/bayesisola-0.1.0-<platform>.whl
```

The release wheels already contain the compiled Axitra `gr_xyz` and `elemse` executables and their required non-system runtime libraries. **A Fortran compiler, CMake, MinGW, and manual Axitra compilation are not required for normal wheel installation.**

The release wheels are regression-tested on Python 3.10, 3.11, 3.12, 3.13 and 3.14 on both supported platforms.

### Installation from source

Source installation is mainly intended for development, modification of the Fortran sources, or platforms for which a release wheel is not provided. A working GNU Fortran compiler is required.

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

The bundled Axitra programs are compiled automatically through CMake and `scikit-build-core`; the legacy manual compilation step inside a repository `green/` directory is no longer required.

## What is new in this maintained version?

The scientific basis remains the original Bayesian ISOLA formulation and the inherited Git history is preserved. The maintained version adds the following user-facing and infrastructure improvements.

### Packaging and portability

- Modern `pyproject.toml` packaging using `scikit-build-core` and CMake.
- Pre-built Windows x86_64 and manylinux x86_64 wheels with Axitra included.
- Axitra executables are installed under `BayesISOLA/_bin` rather than expected in a writable source-tree `green/` directory.
- Each inversion uses an event-specific writable Green's-function workspace, `<outdir>/green`, preventing mutable inversion products from being written into the installed package.
- Runtime Python dependencies are installed by pip.
- Example scripts use paths relative to their own location and can be launched from any working directory in a source checkout.

### Automated CMT workflow

`BayesISOLA.workflows` adds a higher-level operational layer while retaining the native BayesISOLA objects and inversion routines. The main entry point is:

```python
from BayesISOLA.workflows import run_auto_cmt
```

`run_auto_cmt()` can coordinate the complete inversion sequence:

1. event definition and source-time function;
2. station discovery or loading of an existing station table;
3. FDSN waveform/response acquisition or reuse of local miniSEED + StationXML;
4. magnitude-dependent station-radius and waveform-window selection;
5. BayesISOLA waveform quality control and preprocessing;
6. automatic space-time centroid-grid construction;
7. velocity-model preparation and Green's-function calculation/reuse;
8. noise covariance weighting or the unweighted BayesISOLA branch;
9. full or deviatoric moment-tensor inversion;
10. curated result tables, grid-edge diagnostics and figures.

Remote acquisition supports ordered FDSN-client fallback. Local and remote waveform branches converge on the same local-data loading and inversion path.

### Velocity models and Green's functions

`BayesISOLA.gf_helpers` provides reusable tools for working with depth-sampled 1-D profiles and tabular 3-D velocity models. A 3-D model can be prepared once and sampled repeatedly at individual locations or along event-to-station paths.

The automated workflow supports:

- an existing native BayesISOLA `crustal.dat`;
- a 1-D `pandas.DataFrame` velocity profile;
- a `gf_helpers.VelocityGrid3D` object;
- the native Axitra backend, which remains the default;
- the EarthScope Syngine backend through `gf_source="syngine"`.

For a 3-D velocity grid, Axitra may use:

```python
gf_options={"grid": "station"}
```

to extract a vertical model at each station, or:

```python
gf_options={
    "grid": "path",
    "path_spacing_km": 2.0,
    "path_profile": "mean",
}
```

to form a representative depth profile along each event-to-station path. `path_profile` may be `"mean"`, `"median"`, `"p05"` or `"p95"`.

These station/path models are **station-dependent 1-D approximations to a 3-D medium**, not fully 3-D Green's-function calculations.

Green's-function reuse is controlled consistently by `use_precalculated_Green`:

- `False`: force regeneration;
- `"auto"`: reuse compatible cached Green's functions and regenerate otherwise;
- `True`: require a complete compatible cache.

### Results and diagnostics

The automated workflow retains the native BayesISOLA solution objects but also exposes a curated results layer containing:

- centroid location and time-shift information;
- moment-tensor/source summary;
- per-station and per-component fit information;
- grid-edge diagnostics;
- optional posterior uncertainty samples and diagnostics;
- output paths and figure paths;
- Green's-function backend/cache metadata in `run["gf"]`.

The `summary` plotting preset adds a compact CMT summary while keeping native BayesISOLA diagnostics available through the full plotting path.

### Validation and release testing

The release build is tested using installed-wheel numerical regression rather than import tests alone. Linux and Windows wheels are tested independently on Python 3.10–3.14, including execution of the bundled Axitra programs and comparison against a stored numerical reference.

## Quick start

### Native BayesISOLA interface

The historical interface remains available:

```python
import BayesISOLA

inputs = BayesISOLA.load_data(outdir="output/event")
inputs.read_event_info("event.isl")
inputs.set_source_time_function("triangle", 2.0)
inputs.read_network_coordinates("network.stn")
inputs.read_crust("crustal.dat")
```

### Automated workflow

A minimal Axitra-backed automated run using an existing BayesISOLA crustal model has the form:

```python
from BayesISOLA.workflows import run_auto_cmt

run = run_auto_cmt(
    event_id="event_id",
    origin_time="2026-01-01T00:00:00Z",
    event_lon=0.0,
    event_lat=0.0,
    event_depth_km=10.0,
    magnitude=5.0,
    output_dir="output/event_id",
    crust_file="crustal.dat",
)
```

The returned dictionary retains the native BayesISOLA objects together with acquisition, Green's-function, processing, result and output metadata.

For a more complete example, including `run_auto_cmt()` and multiple velocity-model/Green's-function choices, see:

```text
examples/run_auto_cmt_example.ipynb
```

## `BayesISOLA.workflows`

`workflows.py` is the operational automation layer. It does not replace the BayesISOLA inversion; it prepares inputs, calls the native processing/inversion machinery and organizes the outputs.

Its public helpers cover four main areas:

- **Station and waveform acquisition:** `get_max_radius`, `discover_stations`, `get_network_file`, `write_network_file`, `get_mseed_stationxml`, `get_waveform_window`, `load_streams_fdsnws_auto`, and `load_streams_local`.
- **Quality control and grid diagnostics:** `plot_waveform_section`, `plot_station_section`, `suggest_depth_limits`, and `diagnose_grid_edge`.
- **Results extraction and reporting:** `extract_station_fit_df`, `extract_centroid_location`, `extract_solution_summary`, `extract_uncertainty_df`, `write_solution_outputs`, and `plot_cmt_summary`.
- **End-to-end orchestration:** `run_auto_cmt` and the `PLOT_PRESETS` configuration.

See [`docs/BayesISOLA.workflows.rst`](docs/BayesISOLA.workflows.rst) for a fuller description and the generated API reference.

## `BayesISOLA.gf_helpers`

`gf_helpers.py` is the velocity-model preparation and interoperability layer used by the automated Green's-function workflow.

The principal public interfaces are:

- `VelocityGrid3D`: prepared rectilinear, triangulated or rotated-rectilinear 3-D velocity model with repeated profile extraction.
- `build_regular_velocity_grid`: validates a tabular 3-D model and constructs the appropriate interpolation representation.
- `VelocityGrid3D.extract_profile`: extracts a vertical profile at a physical location.
- `get_profile_from_path`: samples a path through the 3-D model and returns the individual samples, depth-wise summary statistics and a layered model.
- `profile_to_pyfk_layers`: converts a depth-sampled profile to piecewise-constant layers.
- `layers_to_pyfk_array` and `make_pyfk_model`: lower-level pyFK model conversion utilities.
- `write_mttime_herrmann_sac`: interoperability utility for writing Herrmann-basis Green's functions.

For ordinary BayesISOLA use, the velocity-grid/profile utilities are the most relevant; the pyFK/MTtime conversion functions are lower-level interoperability helpers retained by the module.

See [`docs/BayesISOLA.gf_helpers.rst`](docs/BayesISOLA.gf_helpers.rst) for details and the generated API reference.

## Examples

The source repository contains:

```bash
python examples/example_2_SAC.py
python examples/example_2_fdsnws.py
```

The SAC example includes its waveform and response files under `examples/input/example_2_SAC`.

The notebook:

```text
examples/run_auto_cmt_example.ipynb
```

demonstrates the higher-level automated CMT workflow and the available velocity-model and Green's-function configurations.

## Repository layout

```text
BayesISOLA/                 Python package
BayesISOLA/_bin/            Installed Axitra executables in wheel/source builds
BayesISOLA/resources/       Runtime static resources
fortran/axitra/             Axitra Fortran build sources
examples/                   Runnable scripts, notebook and example inputs
docs/                       Sphinx documentation
tests/                      Packaging, path and numerical regression tests
CMakeLists.txt              Axitra build definition
pyproject.toml              Python build and package metadata
```

## Project history and attribution

This repository is an independently maintained development derived from the original BayesISOLA project developed by Jiří Vackář and collaborators. The inherited Git history is preserved for scientific attribution and provenance.

- [Original BayesISOLA repository](https://github.com/vackar/BayesISOLA)
- [Jiří Vackář's legacy BayesISOLA documentation](https://geo.mff.cuni.cz/~vackar/BayesISOLA/)

The legacy documentation is useful for the original method, module organization and historical source-install procedure. Installation and maintained-version features should follow this repository's current documentation.

## Reference

J. Vackář, J. Burjánek, F. Gallovič, J. Zahradník, and J. Clinton (2017), *Bayesian ISOLA: new tool for automated centroid moment tensor inversion*, Geophysical Journal International, 210(2), 693–705.

## License

The repository distributes the GNU General Public License, version 3 text in `LICENSE`.
