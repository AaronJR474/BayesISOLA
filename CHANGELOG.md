# Changelog

All notable changes to BayesISOLA are documented in this file.

## [0.2.0] - 2026-08-19

BayesISOLA 0.2.0 extends the automated workflow around the validated native
BayesISOLA point-source moment-tensor inversion. The release adds exact
workflow-level posterior products, bounded adaptive centroid searches,
station-selection and robustness diagnostics, richer reporting, and faster
Green-function/inversion orchestration while preserving the native Axitra
algorithms and moment-tensor parameterization.

### Posterior and uncertainty

- Added an exact discrete space-time posterior table over every valid final-grid
  point and source-time shift.
- Retained the numerically stable log-space posterior formulation introduced in
  0.1.1 and exposed normalized cell probabilities, posterior rank, cumulative
  probability, and 68%/95% highest-posterior-density membership.
- Added posterior diagnostics including entropy, normalized entropy, effective
  occupied cells, modal probability, and posterior probability on active
  spatial/time boundaries.
- Replaced rounded per-cell uncertainty allocation with exactly `n` categorical
  posterior-cell draws followed by conditional Gaussian moment-tensor sampling.
- Added optional preferred-cell residual-variance calibration through
  `uncertainty_scale="residual"`; `"fixed"` retains the native covariance scale.
  The residual option is a scalar empirical calibration, not structural/model
  covariance.
- Added deterministic `uncertainty_random_state` support for reproducible
  posterior sampling.
- Aligned the two physically interchangeable nodal-plane labels to the preferred
  mechanism when reporting uncertainty so arbitrary NP1/NP2 label switching does
  not inflate strike/dip/rake intervals. The sampled moment tensors themselves
  are unchanged.

### Adaptive centroid search

- Added optional bounded Axitra adaptive-grid search through
  `adaptive_grid_search`.
- Added automatic expansion when the preferred solution or sufficient posterior
  mass reaches an active spatial boundary.
- Added optional grid refinement around an interior preferred solution with
  explicit refinement limits and grid-point budgets.
- Kept waveform and Green-function time support fixed across adaptive stages so
  changing the depth search window does not silently change the likelihood.
- Added `adaptive_history`, adaptive-grid result tables, and plotting of search
  evolution.
- Improved grid construction around floating-point step boundaries and removed
  the historical single-letter horizontal-grid identifier limit.
- Fixed fixed-horizontal searches so a one-point horizontal grid is not reported
  as a horizontal boundary merely because it contains one location.

### Station acquisition and selection

- Retained the historical sequence form of `channel_priority`, for example
  `("HH", "BH", "LH")`.
- Added optional magnitude- and distance-dependent channel-family precedence
  rules through the same `channel_priority` argument, avoiding a second public
  channel-rule interface.
- Rule intervals are half-open `[min, max)` and the first matching rule wins;
  unmatched/temporarily unavailable families fall back to the configured default
  order rather than discarding the station.
- Channel families appearing only in rule mappings are automatically included in
  the FDSN metadata query.
- Added explicit `drop_stations` handling and optional azimuth-sector thinning
  with channel priority applied before distance within a sector.
- Added a complete `station_selection` audit table describing selected, manually
  dropped, and azimuth-excluded stations.

### Station robustness

- Added exact leave-one-station-out robustness diagnostics on the final converged
  full-solution grid.
- The station jackknife reports changes in Mw, depth, centroid position/time,
  Kagan angle, decomposition, condition number, variance reduction, held-out
  whitened residuals, azimuthal geometry, and grid-boundary status.
- Reworked the jackknife to retain station-wise normal-equation sufficient
  statistics while each grid point is inverted. The final leave-one-out scan
  subtracts those cached blocks instead of rereading and refiltering Green
  functions, reducing a previously expensive second pass to small matrix solves.
- The cached sufficient statistics are internal and are removed from the public
  grid after the jackknife is complete.
- Removed the jackknife progress bar because the cached calculation is now a
  short final diagnostic rather than a second long-running inversion stage.

### Green functions and inversion performance

- Added a unified `use_precalculated_Green` policy across supported Green-function
  backends: `False` forces regeneration, `"auto"` reuses compatible output and
  repairs incomplete/incompatible caches, and `True` requires a complete
  compatible cache.
- Added station-dependent and representative path-dependent Axitra 1-D models
  derived from a 3-D velocity grid while retaining the native Axitra solver.
- Added the EarthScope Syngine Green-function backend and conversion to the same
  six elementary source bases consumed by BayesISOLA's unchanged inverse
  operator.
- Added backend/model/cache metadata under `run["gf"]`.
- Reduced multiprocessing serialization by initializing invariant inversion data
  once per worker and limiting nested BLAS threading.
- Replaced an inherited repeated `todo.index(...)` result lookup with direct
  ordered result assignment, removing unnecessary quadratic Python overhead for
  large grids.
- Added optional station-wise normal-equation caching as a side-car to the native
  inversion; synthetic regression confirms cache-on/cache-off inversion results
  are unchanged for unweighted and factorized-noise covariance cases.

### Diagnostics and reporting

- Added `BayesISOLA._diagnostics` with maintained workflow-level figures for the
  CMT solution, exact posterior, adaptive-grid history, station geometry/QC,
  uncertainty, and station/component fit.
- `plot_preset="summary"` now produces the curated diagnostic suite; `"full"`
  additionally requests the native BayesISOLA plotting suite.
- Station-fit plots now preserve negative component variance reduction rather
  than clipping poor fits at 0%.
- Added `BayesISOLA._report.write_html_report()` for a curated workflow-level
  `report.html` containing posterior, uncertainty, adaptive-search, station-fit,
  jackknife, and result-table summaries.
- Kept the historical `BayesISOLA._html` renderer intact.
- `html_output=True` is independent of `plot_preset` and builds the historical
  native `index.html` using BayesISOLA's original complete native plotting
  defaults. HTML-only native figures are not displayed merely because
  `show=True`.
- Ensured adaptive covariance reconstruction retains the non-inverted covariance
  matrix when required by the native HTML covariance-matrix figure.

### Results and workflow API

- Added structured `posterior_cells`, `posterior_diagnostics`,
  `uncertainty_diagnostics`, `station_jackknife`, `adaptive_history`, and
  `station_selection` outputs.
- Added optional `save_posterior_cells` because complete posterior-cell tables can
  be large; the table remains available in memory regardless of CSV persistence.
- Added independent `html_output` (native BayesISOLA `index.html`) and
  `write_report` (curated workflow `report.html`) controls.
- Preserved the native BayesISOLA objects in the returned `run` mapping for
  direct scientific inspection and custom plotting.

### Testing and validation

- Added/updated synthetic regression coverage for exact posterior normalization,
  uncertainty allocation/reproducibility, adaptive-grid proposals, dynamic
  channel rules, azimuth controls, grid boundaries, station normal-equation
  partitioning, cached jackknife equivalence, diagnostic plotting, native HTML
  independence, and curated report generation.
- Verified cached station normal-equation collection does not alter the native
  inversion solution in unweighted or factorized-noise covariance branches.
- Retained numerical Axitra regression coverage; this release does not modify the
  validated Axitra Fortran algorithms or source basis.

### Compatibility

- Python >= 3.10
- ObsPy >= 1.5.0
- Windows and Linux
- Native BayesISOLA point-source MT parameterization and Axitra algorithms remain
  unchanged.

---

## [0.1.1] - 2026-08-16

BayesISOLA 0.1.1 is a patch release focused on numerical robustness,
cross-platform reliability, automated workflow improvements, and reporting.

### Numerical robustness

- Reworked posterior grid-point weighting to operate in log space, preventing
  overflow and underflow in the BayesISOLA posterior probability calculation.
- Added stable log-determinant handling for moment-tensor covariance matrices.
- Preserved the relative posterior probabilities while avoiding numerical
  failure for large dynamic ranges.
- Added explicit handling for cases where no finite posterior weights or valid
  inversion grid points remain.
- Improved handling of failed Green-function calculations in serial Axitra runs.
- Strengthened validation of cached Axitra Green-function files by checking the
  corresponding binary payloads.

### Moment-tensor plotting and reporting

- Improved beachball plotting robustness for full and deviatoric moment tensors.
- Removed reliance on the obsolete ObsPy `plot_zerotrace` argument.
- Normalized plotting tensors before beachball construction to improve numerical
  stability.
- Improved handling of moment tensors for which a nodal-plane representation is
  undefined.
- Fixed UTF-8 HTML output on Windows, including names and symbols containing
  non-ASCII characters.
- Fixed HTML resource-path handling when the Python environment and BayesISOLA
  output directory are located on different Windows drives.

### Automated workflow

- Added the `progress` argument to `run_auto_cmt()` so progress reporting can be
  enabled or disabled explicitly.
- Improved propagation of Green-function generation failures through the
  automated workflow.
- Added clearer errors when no valid inversion grid points remain.
- Improved recursive output-directory creation.
- Removed the possibility of silently importing an unrelated top-level
  `gf_helpers` module instead of the packaged BayesISOLA implementation.
- Added an explicit `load_data.close()` method for releasing BayesISOLA-owned
  file handles, particularly important when deleting event workspaces on
  Windows.
- `load_data.__exit__()` and object cleanup now use the explicit close method.

### Data handling

- Improved station-table loading so SEED identifiers such as station and
  location codes retain leading zeros and string values.
- Improved preservation of identifier fields when restarting workflows from
  saved station CSV files.

### Dependencies

- Raised the minimum supported ObsPy version to:

  `ObsPy >= 1.5.0`

  This avoids known FDSN error-handling issues in older ObsPy versions and
  provides improved moment-tensor plotting support.

### Testing

- Added regression tests for numerically stable posterior weighting.
- Added tests for posterior probability equivalence in well-conditioned cases.
- Added plotting and beachball robustness tests.
- Added tests for the `run_auto_cmt()` progress interface.
- Added tests for UTF-8 HTML generation and Windows-style relative resource
  paths.
- Added tests for preservation of SEED station identifiers.
- Added tests for Green-function failure and cache validation.
- Added tests for empty/invalid inversion-grid handling.
- Added tests for explicit `load_data.close()` resource cleanup.
- Extended portability and package-level regression coverage.

### Compatibility

- Python >= 3.10
- ObsPy >= 1.5.0
- Windows and Linux
- Axitra Green functions remain numerically unchanged; this release does not
  modify the validated Axitra algorithms or convergence behaviour.

---

## [0.1.0] - 2026-08-13

First standalone BayesISOLA release.

### Highlights

- Modern standalone Python package for BayesISOLA.
- Cross-platform Windows and Linux packaging.
- Bundled and validated Axitra executables.
- Automated CMT workflows through `BayesISOLA.workflows`.
- Automated waveform acquisition and Green-function generation.
- Support for full and deviatoric moment-tensor inversion.
- Noise covariance and optional cross-component covariance.
- Centroid location and time grid search.
- Automated result tables and CMT summary plots.
- Python 3.10–3.14 support.
- GitHub Actions build, numerical regression, release, and PyPI publishing.
