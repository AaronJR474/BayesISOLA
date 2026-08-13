BayesISOLA.workflows
===================

Overview
--------

``BayesISOLA.workflows`` provides the high-level operational layer used to run
repeatable centroid-moment-tensor inversions.  It does not replace the native
BayesISOLA inversion.  Instead, it coordinates acquisition, preparation,
Green's functions, covariance weighting, inversion, diagnostics and reporting,
while returning the native BayesISOLA objects for further inspection.

The principal entry point is ``run_auto_cmt``.

Automated inversion sequence
----------------------------

A ``run_auto_cmt`` call can coordinate the following sequence:

#. define the catalogue event and source-time function;
#. determine the station search radius;
#. discover stations through one or more FDSN providers, or use a supplied
   local station table;
#. determine a BayesISOLA-safe origin-centred waveform window;
#. download and validate miniSEED + StationXML, or load existing local files;
#. run BayesISOLA waveform screening and preprocessing;
#. construct the space-time centroid grid;
#. prepare the velocity model and Green's-function backend;
#. calculate, verify or reuse compatible Green's functions;
#. construct noise covariance weighting, or select the unweighted branch;
#. solve the native BayesISOLA full or deviatoric moment-tensor problem;
#. extract deterministic result tables, optional uncertainty samples,
   grid-edge diagnostics and figures.

Minimal example
---------------

Using the native Axitra backend and an existing BayesISOLA ``crustal.dat``::

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

The returned dictionary contains the native BayesISOLA objects together with
acquisition, processing, Green's-function, result and output metadata.

Station and waveform acquisition
--------------------------------

The public acquisition helpers are reusable independently of
``run_auto_cmt``:

``get_max_radius``
   Estimates a magnitude-dependent maximum station radius.

``discover_stations``
   Discovers candidate stations for an event using the configured FDSN
   provider/client sequence and the requested component/channel rules.

``get_network_file`` and ``write_network_file``
   Prepare BayesISOLA ``network.stn`` input from the authoritative station
   table.

``get_mseed_stationxml``
   Performs the complete remote acquisition path, including ordered
   multi-client fallback, waveform/response validation and storage of one
   miniSEED + StationXML pair per accepted station.

``get_waveform_window``
   Computes the origin-centred waveform interval used by acquisition and
   processing.  The window accounts for source depth, magnitude, distance and
   the covariance/noise requirements of the inversion.

``load_streams_fdsnws_auto`` and ``load_streams_local``
   Load remote or previously acquired data into BayesISOLA.  Remote and local
   branches ultimately converge on the same local-data contract.

Grid and quality-control helpers
--------------------------------

``suggest_depth_limits``
   Resolves the automated shallow/deep centroid-search bounds from the catalogue
   depth and configured multipliers.

``diagnose_grid_edge``
   Reports whether the preferred centroid lies against the realised search-grid
   limits, providing a simple warning that the search domain may need expansion.

``plot_waveform_section`` and ``plot_station_section``
   Provide acquisition/record-section diagnostics before or alongside the
   inversion.

Green's-function backends
-------------------------

Axitra is the default and validated historical backend::

   gf_source="axitra"

With ``gf_options=None``, an existing ``crustal.dat`` or a 1-D velocity profile
is used as one model.  A ``gf_helpers.VelocityGrid3D`` may also be sampled once
at the catalogue epicentre.

Station-dependent vertical models are enabled with::

   gf_options={"grid": "station"}

A representative event-to-station path profile is enabled with::

   gf_options={
       "grid": "path",
       "path_spacing_km": 2.0,
       "path_profile": "mean",
   }

``path_profile`` may be ``"mean"``, ``"median"``, ``"p05"`` or ``"p95"``.
These are station-dependent 1-D approximations to a 3-D medium; they are not
fully 3-D Green's-function calculations.

The alternative EarthScope Syngine backend is selected with::

   gf_source="syngine"

The default model is ``ak135f_5s`` and can be changed through
``gf_options={"model": ...}``.  Syngine output is converted to the same six
elementary source bases consumed by the unchanged BayesISOLA inverse operator.

Green's-function cache policy
-----------------------------

``use_precalculated_Green`` has the same meaning for every backend:

``False``
   Force regeneration.

``"auto"``
   Reuse complete compatible output and regenerate missing or incompatible
   output.

``True``
   Require a complete compatible cache; raise instead of regenerating.

The resolved backend, physical/model mode, storage path, cache policy, reuse
status, normalized backend options and manifest information are returned under
``run["gf"]``.

Covariance and inversion
------------------------

``covariance="noise"`` retains BayesISOLA noise weighting.
``crosscovariance=False`` uses component-wise covariance blocks, while ``True``
retains the full three-component station covariance.

``covariance="none"`` avoids the long pre-event noise interval and invokes
BayesISOLA's native unweighted ordinary least-squares branch.

``deviatoric=False`` performs the full six-component moment-tensor inversion;
``deviatoric=True`` selects the deviatoric branch.

Results and plotting
--------------------

The workflow always prepares deterministic scientific outputs and retains the
native solution object.  Public result helpers include:

``extract_centroid_location``
   Preferred centroid location/time information.

``extract_solution_summary``
   Source and moment-tensor summary.

``extract_station_fit_df``
   Tidy per-station/per-component geometry, weighting and fit information.

``extract_uncertainty_df``
   Optional native BayesISOLA posterior samples when uncertainty sampling is
   requested.

``write_solution_outputs``
   Writes the curated result tables.

``plot_cmt_summary``
   Produces the compact maintained-version CMT summary.

``PLOT_PRESETS``
   Provides ``"none"``, ``"summary"`` and ``"full"`` plotting presets.

``run["results"]`` contains the curated result tables, while the returned
``run`` dictionary also contains output and figure paths and the underlying
BayesISOLA objects.

Worked example
--------------

``examples/run_auto_cmt_example.ipynb`` demonstrates ``run_auto_cmt`` together
with multiple velocity-model and Green's-function configurations.

API reference
-------------

.. automodule:: BayesISOLA.workflows
   :members:
   :undoc-members:
   :show-inheritance:
