# Changelog

All notable changes to BayesISOLA are documented in this file.

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
