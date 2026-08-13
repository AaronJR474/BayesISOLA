BayesISOLA.gf_helpers
======================

Overview
--------

``BayesISOLA.gf_helpers`` provides reusable velocity-model preparation,
interpolation and interoperability utilities.  The automated BayesISOLA
workflow uses these helpers to turn 1-D profiles or tabular 3-D velocity models
into the layered receiver models required by the Green's-function calculation.

The module is intentionally separate from the inversion itself: it prepares
velocity information, while BayesISOLA/Axitra remains responsible for the
Green's-function and moment-tensor calculations.

Supported velocity-model representations
----------------------------------------

``build_regular_velocity_grid`` accepts a tabular 3-D model and prepares one of
three horizontal interpolation representations.

Rectilinear
   Complete separable x/y axes in a projected metre-based CRS.  Interpolation
   is performed on depth, y and x using a regular-grid interpolator.

Triangulated
   Geographic, curvilinear or otherwise non-separable horizontal nodes.  The
   horizontal node set is projected and triangulated once, and the same
   barycentric geometry is reused at each depth.

Rotated rectilinear
   A regular computational x/y grid described by an origin, CRS, rotation and
   central meridian.  Physical query coordinates are transformed into the model
   frame before interpolation.

Vp and Vs are expressed in km/s, depth in km and density, when supplied, in
g/cm3.  Qs and Qp may also be supplied.  Missing density or attenuation values
are not silently synthesized by these helpers.

``VelocityGrid3D``
------------------

``VelocityGrid3D`` is the prepared model returned by
``build_regular_velocity_grid``.  It stores the model depth levels and
interpolation representation so repeated profile extraction does not rebuild
the spatial geometry.

The most important method for BayesISOLA use is
``VelocityGrid3D.extract_profile``.  It extracts the model quantities at one
horizontal position for every stored depth level.  Horizontal interpolation is
linear; the helper does not smooth or vertically resample the model.

Typical preparation
-------------------

A tabular model is prepared once::

   from BayesISOLA.gf_helpers import build_regular_velocity_grid

   grid = build_regular_velocity_grid(
       velocity_df,
       x_col="X",
       y_col="Y",
       depth_col="Depth_km",
       vp_col="Vp",
       vs_col="Vs",
       density_col="Density",
       qs_col="Qs",
       qp_col="Qp",
       coordinate_crs="EPSG:2193",
   )

A vertical profile can then be extracted repeatedly::

   profile = grid.extract_profile(
       x=station_x,
       y=station_y,
       crs="EPSG:2193",
   )

Path sampling
-------------

``get_profile_from_path`` samples the prepared model between two physical
locations at a requested path spacing.  It returns:

#. all sampled depth profiles along the path;
#. depth-wise summary statistics;
#. the selected representative layered model.

The available representative path statistics are ``mean``, ``median``,
``p05`` and ``p95``.

In ``run_auto_cmt``, this functionality underlies::

   gf_options={
       "grid": "path",
       "path_spacing_km": 2.0,
       "path_profile": "mean",
   }

For::

   gf_options={"grid": "station"}

the workflow instead extracts one vertical profile at each station.

Both modes produce station-dependent 1-D models.  They should not be described
as fully 3-D Green's-function calculations because the selected layered model
is fixed for each receiver across the centroid search grid.

Layer conversion
----------------

``profile_to_pyfk_layers``
   Converts a depth-sampled profile to piecewise-constant layers.  Repeated
   depths may represent interfaces, and adjacent equivalent layers may be
   compressed.

``layers_to_pyfk_array``
   Converts the validated layer table to pyFK's positional model array while
   retaining only properties explicitly supplied where pyFK permits this.

``make_pyfk_model``
   Creates a pyFK ``SeisModel`` from the prepared layers.

Although these functions retain their pyFK naming, the profile/layer
construction is also used by the BayesISOLA workflow when translating supplied
velocity information to the layered model required by native Axitra.

Interoperability helpers
------------------------

The module also retains basis-order/sign constants and
``write_mttime_herrmann_sac`` for pyFK/MTtime interoperability.  These are
lower-level utilities and are not required for ordinary BayesISOLA
``run_auto_cmt`` use.

Relationship to ``BayesISOLA.workflows``
-----------------------------------------

The division of responsibility is:

``gf_helpers``
   Prepare and sample velocity models.

``workflows``
   Decide which model/profile is required for each inversion or station, write
   the BayesISOLA/Axitra model inputs, manage the GF backend/cache policy and
   execute the operational inversion sequence.

``BayesISOLA`` native modules
   Perform waveform processing, Green's-function consumption and moment-tensor
   inversion.

Worked example
--------------

``examples/run_auto_cmt_example.ipynb`` demonstrates how the velocity-model
choices feed into ``run_auto_cmt``.

API reference
-------------

.. automodule:: BayesISOLA.gf_helpers
   :members:
   :undoc-members:
   :show-inheritance:
