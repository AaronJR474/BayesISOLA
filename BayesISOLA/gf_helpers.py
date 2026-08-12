"""Velocity-model helpers for pyFK and MTtime.

The module supports two input types:

1. A depth-sampled 1-D velocity profile, converted to the piecewise-constant
   layer format required by :class:`pyfk.SeisModel`.
2. A tabular 3-D velocity model, from which vertically sampled 1-D profiles
   can be extracted at individual locations or along a projected path.

Horizontal interpolation
------------------------
``build_regular_velocity_grid`` selects the interpolation representation from
how the supplied horizontal coordinates are organised:

``rectilinear``
    Used when ``x_col`` and ``y_col`` form complete, separable axes in a
    projected metre-based CRS. Interpolation is trilinear on
    ``(depth, y, x)`` using :class:`scipy.interpolate.RegularGridInterpolator`.
    This is the preferred representation for ordinary UTM, NZTM, NAP1955 and
    similar rectangular projected grids.

``triangulated``
    Used when the horizontal coordinates are structured or unstructured nodes
    rather than separable axes. Geographic coordinates are first projected to
    ``interpolation_crs`` and then triangulated once with
    :class:`scipy.spatial.Delaunay`. The same triangle and barycentric weights
    are reused at every depth. This supports rotated or curvilinear grids
    supplied only as longitude/latitude or projected node coordinates.

``rotated rectilinear``
    Used when the table contains a regular arbitrary computational grid in
    metres. The user supplies ``origin``, ``origin_crs``, ``rotation_deg`` and
    ``central_meridian``. Query coordinates are transformed internally into
    the model grid before regular-grid interpolation.

The optional ``interpolation_crs`` must be projected and metre-based. It is
required for geographic node coordinates and is also the CRS in which path
lengths and path sampling increments are evaluated.

Units
-----
Vp, Vs
    km/s.
Depth and layer thickness
    km, increasing downward.
Density
    g/cm3, optional.
Projected and arbitrary-grid coordinates
    metres.
Path spacing
    km.

Optional density, Qs and Qp profiles are supported. Values are passed to pyFK
only when supplied; the helper does not synthesize density or attenuation.
Omitted properties are left to :class:`pyfk.SeisModel` through its native
three-, four- or five-column model conventions.

Version 0.5.1
-------------
``make_pyfk_model`` explicitly exposes pyFK's model ``flattening`` switch as
``pyfk_flattening`` while retaining the previous ``flattening=...`` keyword route
for backward compatibility. The default remains ``False``. No profile extraction,
layer construction, property handling or Herrmann-basis conversion is changed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import Delaunay


__version__ = "0.5.1"

PYFK_DC_FK_ORDER = (
    "ZDD", "RDD", "TDD",
    "ZDS", "RDS", "TDS",
    "ZSS", "RSS", "TSS",
)
PYFK_EP_FK_ORDER = ("ZEX", "REX", "TEX")
MTTIME_HERRMANN_ORDER = (
    "ZSS", "ZDS", "ZDD", "ZEX",
    "RSS", "RDS", "RDD", "REX",
    "TSS", "TDS",
)

# Relative basis conversion between pyFK and MTtime's Herrmann forward operator.
# pyFK's native SS and DS bases have the opposite relative sign from MTtime;
# DD and explosion bases retain their relative signs. Keep this map separate
# from the absolute/global polarity so either convention can be tested directly.
PYFK_TO_MTTIME_HERRMANN_RELATIVE_SIGN = {
    "ZSS": -1.0, "ZDS": -1.0, "ZDD": 1.0, "ZEX": 1.0,
    "RSS": -1.0, "RDS": -1.0, "RDD": 1.0, "REX": 1.0,
    "TSS": -1.0, "TDS": -1.0,
}

# Absolute polarity applied after the relative Herrmann-basis conversion.
# Set to +1.0 to test the previous convention without changing the basis map.
PYFK_TO_MTTIME_GLOBAL_POLARITY = 1.0

# Maps Zhu's FK to Herrmann's CPS moment definition
PYFK_TO_MTTIME_AMPLITUDE_SCALE = 1.0

# Backward-compatible snapshot of the effective signs for callers that imported
# the original constant. New code should use the two factors above explicitly.
PYFK_TO_MTTIME_HERRMANN_SIGN = {
    component: PYFK_TO_MTTIME_GLOBAL_POLARITY * relative_sign
    for component, relative_sign in PYFK_TO_MTTIME_HERRMANN_RELATIVE_SIGN.items()
}

__all__ = [
    "__version__",
    "PYFK_DC_FK_ORDER",
    "PYFK_EP_FK_ORDER",
    "MTTIME_HERRMANN_ORDER",
    "PYFK_TO_MTTIME_HERRMANN_RELATIVE_SIGN",
    "PYFK_TO_MTTIME_GLOBAL_POLARITY",
    "PYFK_TO_MTTIME_HERRMANN_SIGN",
    "VelocityGrid3D",
    "build_regular_velocity_grid",
    "get_profile_from_path",
    "profile_to_pyfk_layers",
    "layers_to_pyfk_array",
    "make_pyfk_model",
    "write_mttime_herrmann_sac",
]


@dataclass(slots=True)
class VelocityGrid3D:
    """Prepared 3-D velocity model for repeated profile extraction.

    Instances are created by :func:`build_regular_velocity_grid`; users should
    not normally instantiate this class directly.

    Parameters stored publicly
    --------------------------
    depth
        Sorted depth levels in kilometres.
    values
        Model quantities keyed by their original column names. Rectilinear
        models store arrays with shape ``(n_depth, n_y, n_x)``; triangulated
        models store arrays with shape ``(n_depth, n_horizontal_nodes)``.
    interpolation_crs
        Projected metre-based CRS used for physical query locations and path
        distances.

    Notes
    -----
    ``extract_profile`` always returns values at the model's stored depth
    levels. Horizontal interpolation is linear. No smoothing or vertical
    resampling is performed.
    """

    depth: np.ndarray
    values: dict[str, np.ndarray]
    interpolation_crs: CRS
    _mode: str
    _x_axis: np.ndarray | None
    _y_axis: np.ndarray | None
    _interpolators: dict[str, RegularGridInterpolator] | None
    _triangulation: Delaunay | None
    _coordinate_crs: CRS | None
    _longitude_center: float | None
    _local_tm_crs: CRS | None
    _origin_tm_x: float | None
    _origin_tm_y: float | None
    _rotation_deg: float | None
    _tolerance: float

    @property
    def interpolation_mode(self) -> str:
        """Return ``'rectilinear'`` or ``'triangulated'``."""

        return self._mode

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the storage shape of one model quantity."""

        return next(iter(self.values.values())).shape

    @property
    def x(self) -> np.ndarray:
        """Return the rectilinear x axis.

        Raises
        ------
        AttributeError
            If the model uses triangulated horizontal nodes rather than a
            separable x axis.
        """

        if self._x_axis is None:
            raise AttributeError("A triangulated model does not have a separable x axis.")
        return self._x_axis

    @property
    def y(self) -> np.ndarray:
        """Return the rectilinear y axis.

        Raises
        ------
        AttributeError
            If the model uses triangulated horizontal nodes rather than a
            separable y axis.
        """

        if self._y_axis is None:
            raise AttributeError("A triangulated model does not have a separable y axis.")
        return self._y_axis

    def extract_profile(
        self,
        x: float,
        y: float,
        *,
        crs: str | int | CRS,
        value_names: Sequence[str] | None = None,
        bounds_error: bool = True,
    ) -> pd.DataFrame:
        """Extract one vertically sampled profile at a physical location.

        Parameters
        ----------
        x, y
            Horizontal query coordinates.
        crs
            CRS of ``x`` and ``y``. Geographic and projected CRSs are accepted;
            coordinates are transformed internally to ``interpolation_crs``.
        value_names
            Optional subset of model quantities to return. By default all
            stored values are returned.
        bounds_error
            If ``True``, raise when the point lies outside the horizontal model
            domain. If ``False``, return NaN values for an outside point.

        Returns
        -------
        pandas.DataFrame
            One row per stored depth level. The first column is ``Depth_km``;
            remaining columns retain their source model names.
        """

        metric_x, metric_y = self._coordinates_to_metric(x, y, crs)
        arrays = self._extract_profiles_metric(
            np.array([[metric_x, metric_y]], dtype=float),
            value_names=value_names,
            bounds_error=bounds_error,
        )
        output = {"Depth_km": self.depth.copy()}
        output.update({name: array[0] for name, array in arrays.items()})
        return pd.DataFrame(output)

    def _coordinates_to_metric(
        self,
        x: float | np.ndarray,
        y: float | np.ndarray,
        crs: str | int | CRS,
    ) -> tuple[np.ndarray, np.ndarray] | tuple[float, float]:
        query_crs = CRS.from_user_input(crs)
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)

        if query_crs.is_geographic and self._longitude_center is not None:
            x_arr = _unwrap_longitude(x_arr, self._longitude_center)

        if query_crs == self.interpolation_crs:
            out_x, out_y = x_arr, y_arr
        else:
            transformer = Transformer.from_crs(
                query_crs,
                self.interpolation_crs,
                always_xy=True,
                force_over=True,
            )
            out_x, out_y = transformer.transform(x_arr, y_arr)

        if np.ndim(out_x) == 0:
            return float(out_x), float(out_y)
        return np.asarray(out_x, dtype=float), np.asarray(out_y, dtype=float)

    def _metric_to_grid_xy(self, metric_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._rotation_deg is None:
            return metric_points[:, 0].copy(), metric_points[:, 1].copy()

        x_metric = metric_points[:, 0]
        y_metric = metric_points[:, 1]
        if self.interpolation_crs != self._local_tm_crs:
            transformer = Transformer.from_crs(
                self.interpolation_crs,
                self._local_tm_crs,
                always_xy=True,
                force_over=True,
            )
            x_metric, y_metric = transformer.transform(x_metric, y_metric)
            x_metric = np.asarray(x_metric, dtype=float)
            y_metric = np.asarray(y_metric, dtype=float)

        de = x_metric - self._origin_tm_x
        dn = y_metric - self._origin_tm_y
        angle = np.radians(self._rotation_deg)

        # SIMUL-style convention used by the supplied NZ velocity model.
        grid_x = -np.cos(angle) * de - np.sin(angle) * dn
        grid_y = -np.sin(angle) * de + np.cos(angle) * dn
        return grid_x, grid_y

    def _extract_profiles_metric(
        self,
        metric_points: np.ndarray,
        *,
        value_names: Sequence[str] | None,
        bounds_error: bool,
    ) -> dict[str, np.ndarray]:
        names = list(self.values) if value_names is None else list(value_names)
        unknown = set(names).difference(self.values)
        if unknown:
            raise KeyError(f"Unknown grid values: {sorted(unknown)}")

        points = np.asarray(metric_points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("metric_points must have shape (n_points, 2).")

        if self._mode == "rectilinear":
            return self._extract_rectilinear(points, names, bounds_error)
        return self._extract_triangulated(points, names, bounds_error)

    def _extract_rectilinear(
        self,
        metric_points: np.ndarray,
        names: Sequence[str],
        bounds_error: bool,
    ) -> dict[str, np.ndarray]:
        grid_x, grid_y = self._metric_to_grid_xy(metric_points)
        outside = (
            (grid_x < self._x_axis[0] - self._tolerance)
            | (grid_x > self._x_axis[-1] + self._tolerance)
            | (grid_y < self._y_axis[0] - self._tolerance)
            | (grid_y > self._y_axis[-1] + self._tolerance)
        )
        if bounds_error and np.any(outside):
            index = int(np.flatnonzero(outside)[0])
            raise ValueError(
                "Query lies outside the rectilinear model domain: "
                f"x={grid_x[index]:g}, y={grid_y[index]:g}."
            )

        grid_x = np.clip(grid_x, self._x_axis[0], self._x_axis[-1])
        grid_y = np.clip(grid_y, self._y_axis[0], self._y_axis[-1])
        n_points = metric_points.shape[0]
        query = np.column_stack((
            np.tile(self.depth, n_points),
            np.repeat(grid_y, self.depth.size),
            np.repeat(grid_x, self.depth.size),
        ))

        output = {}
        for name in names:
            result = self._interpolators[name](query).reshape(n_points, self.depth.size)
            if not bounds_error and np.any(outside):
                result[outside] = np.nan
            output[name] = np.asarray(result, dtype=float)
        return output

    def _extract_triangulated(
        self,
        metric_points: np.ndarray,
        names: Sequence[str],
        bounds_error: bool,
    ) -> dict[str, np.ndarray]:
        simplex = self._triangulation.find_simplex(metric_points)
        outside = simplex < 0
        if bounds_error and np.any(outside):
            index = int(np.flatnonzero(outside)[0])
            raise ValueError(
                "Query lies outside the triangulated model domain in "
                f"{self.interpolation_crs.to_string()}: "
                f"x={metric_points[index, 0]:g}, y={metric_points[index, 1]:g}."
            )

        output = {
            name: np.full((metric_points.shape[0], self.depth.size), np.nan, dtype=float)
            for name in names
        }
        inside_indices = np.flatnonzero(~outside)

        for point_index in inside_indices:
            simplex_index = int(simplex[point_index])
            transform = self._triangulation.transform[simplex_index]
            barycentric = transform[:2] @ (metric_points[point_index] - transform[2])
            weights = np.r_[barycentric, 1.0 - barycentric.sum()]
            vertices = self._triangulation.simplices[simplex_index]

            for name in names:
                output[name][point_index] = self.values[name][:, vertices] @ weights

        return output


def build_regular_velocity_grid(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    depth_col: str,
    vp_col: str,
    vs_col: str,
    density_col: str | None = None,
    qs_col: str | None = None,
    qp_col: str | None = None,
    coordinate_crs: str | int | CRS | None = None,
    interpolation_crs: str | int | CRS | None = None,
    origin: tuple[float, float] | None = None,
    origin_crs: str | int | CRS | None = None,
    rotation_deg: float | None = None,
    central_meridian: float | None = None,
    scale_factor: float = 0.9996,
) -> VelocityGrid3D:
    """Prepare a tabular 3-D velocity model for profile interpolation.

    Parameters
    ----------
    df
        Table containing one row per depth and horizontal node.
    x_col, y_col
        Horizontal coordinate columns. Their interpretation depends on the
        spatial description selected below.
    depth_col
        Depth coordinate in kilometres, increasing downward. Negative values
        are allowed, for example elevations above mean sea level.
    vp_col, vs_col
        P- and S-wave velocity columns in km/s.
    density_col
        Optional density column in g/cm3.
    qs_col, qp_col
        Optional dimensionless S- and P-wave quality-factor columns. ``qp_col``
        requires ``qs_col``. Missing density or attenuation values are not
        derived by this helper; they are left to pyFK whenever its positional
        model format can represent the supplied subset unambiguously.
    coordinate_crs
        CRS of ``x_col`` and ``y_col`` for an ordinary geographic or projected
        model. Do not provide this together with the rotated-grid parameters.
    interpolation_crs
        Projected metre-based CRS used for horizontal interpolation and path
        distances. It is required when ``coordinate_crs`` is geographic. For a
        projected rectilinear grid it defaults to ``coordinate_crs``. Supplying
        a different CRS causes the transformed horizontal nodes to be handled
        by Delaunay triangulation because the transformed grid is generally no
        longer separable.
    origin
        ``(x, y)`` location of the arbitrary-grid origin in ``origin_crs``.
        Required only for a rotated computational grid.
    origin_crs
        CRS of ``origin``.
    rotation_deg
        Counter-clockwise rotation metadata for the arbitrary grid, in degrees.
        For the supplied NZ model this is ``140``.
    central_meridian
        Central meridian used by the local Transverse Mercator construction.
        For the supplied NZ model this is ``173``.
    scale_factor
        Transverse Mercator scale factor for the rotated-grid mapping. The
        default, ``0.9996``, matches the supplied NZ model.

    Returns
    -------
    VelocityGrid3D
        Prepared object supporting :meth:`VelocityGrid3D.extract_profile` and
        :func:`get_profile_from_path`.

    Interpolation selection
    -----------------------
    A projected grid that contains every combination of its unique x, y and
    depth values uses regular trilinear interpolation. This is the recommended
    route for ordinary UTM, NZTM, NAP1955 and similar rectangular grids.

    Geographic coordinates, curvilinear projected coordinates, or any table
    whose horizontal coordinates are not separable are projected to
    ``interpolation_crs`` and triangulated once. Each depth must contain the
    same horizontal node set.

    An arbitrary rotated model must contain a complete regular x-y-depth grid
    in metres. Query coordinates are converted internally to that computational
    frame before regular interpolation.
    """

    rotated_parameters = (origin, origin_crs, rotation_deg, central_meridian)
    use_rotation = any(value is not None for value in rotated_parameters)

    if coordinate_crs is not None and use_rotation:
        raise ValueError("Use coordinate_crs or rotated-grid parameters, not both.")
    if coordinate_crs is None and not use_rotation:
        raise ValueError(
            "Provide coordinate_crs, or origin, origin_crs, rotation_deg and "
            "central_meridian."
        )
    if use_rotation and not all(value is not None for value in rotated_parameters):
        raise ValueError(
            "A rotated grid requires origin, origin_crs, rotation_deg and "
            "central_meridian together."
        )
    if origin is not None and len(origin) != 2:
        raise ValueError("origin must be a two-value (x, y) tuple.")
    if scale_factor <= 0:
        raise ValueError("scale_factor must be positive.")

    if qp_col is not None and qs_col is None:
        raise ValueError("qp_col requires qs_col.")

    value_cols = [vp_col, vs_col]
    for name in (density_col, qs_col, qp_col):
        if name is not None:
            value_cols.append(name)

    required = {x_col, y_col, depth_col, *value_cols}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"Missing velocity-model columns: {sorted(missing)}")

    table = df.loc[:, [depth_col, y_col, x_col, *value_cols]].copy()
    for column in table.columns:
        table[column] = pd.to_numeric(table[column], errors="raise")
    if table.isna().any().any():
        raise ValueError("Velocity-model table contains missing values.")
    if table.duplicated([depth_col, y_col, x_col]).any():
        raise ValueError("Duplicate (depth, y, x) nodes were found.")

    depth = np.sort(table[depth_col].unique().astype(float))
    _validate_velocity_values(
        table, vp_col, vs_col, density_col, qs_col, qp_col
    )

    if use_rotation:
        return _build_rotated_rectilinear_grid(
            table,
            x_col=x_col,
            y_col=y_col,
            depth_col=depth_col,
            value_cols=value_cols,
            depth=depth,
            interpolation_crs=interpolation_crs,
            origin=origin,
            origin_crs=origin_crs,
            rotation_deg=float(rotation_deg),
            central_meridian=float(central_meridian),
            scale_factor=float(scale_factor),
        )

    source_crs = CRS.from_user_input(coordinate_crs)
    target_crs = CRS.from_user_input(interpolation_crs or source_crs)
    if source_crs.is_geographic and interpolation_crs is None:
        raise ValueError(
            "A projected metre-based interpolation_crs is required when "
            "coordinate_crs is geographic."
        )
    _require_projected_metre_crs(target_crs, "interpolation_crs")

    is_rectilinear = _is_complete_rectilinear_grid(
        table,
        x_col=x_col,
        y_col=y_col,
        depth_col=depth_col,
        depth=depth,
    )
    same_projected_crs = source_crs.is_projected and source_crs == target_crs

    if is_rectilinear and same_projected_crs:
        return _build_standard_rectilinear_grid(
            table,
            x_col=x_col,
            y_col=y_col,
            depth_col=depth_col,
            value_cols=value_cols,
            depth=depth,
            coordinate_crs=source_crs,
        )

    return _build_triangulated_grid(
        table,
        x_col=x_col,
        y_col=y_col,
        depth_col=depth_col,
        value_cols=value_cols,
        depth=depth,
        coordinate_crs=source_crs,
        interpolation_crs=target_crs,
    )



def get_profile_from_path(
    grid: VelocityGrid3D,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    crs: str | int | CRS,
    spacing_km: float = 1.0,
    value_names: Sequence[str] | None = None,
    bounds_error: bool = True,
    profile: str | int = "median",
    vp_name: str = "Vp",
    vs_name: str = "Vs",
    density_name: str | None = "Density",
    qs_name: str | None = None,
    qp_name: str | None = None,
    surface_depth_km: float = 0.0,
    max_depth_km: float | None = None,
    compress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Sample path profiles and construct a layered pyFK model.

    Velocities are km/s, density is g/cm3, and Q values are dimensionless.
    ``qp_name`` requires ``qs_name``. The returned array uses pyFK's shortest
    positional model format for the properties actually supplied, leaving
    omitted density or attenuation values to pyFK.
    """
    if qp_name is not None and qs_name is None:
        raise ValueError("qp_name requires qs_name.")

    spacing_m = float(spacing_km) * 1000.0
    if spacing_m <= 0:
        raise ValueError("spacing_km must be positive.")

    start_x, start_y = grid._coordinates_to_metric(x1, y1, crs)
    end_x, end_y = grid._coordinates_to_metric(x2, y2, crs)
    delta_x = end_x - start_x
    delta_y = end_y - start_y
    total_m = float(np.hypot(delta_x, delta_y))

    if total_m == 0:
        distances_m = np.array([0.0])
    else:
        distances_m = np.arange(0.0, total_m, spacing_m)
        if distances_m.size == 0 or not np.isclose(distances_m[-1], total_m):
            distances_m = np.r_[distances_m, total_m]

    fractions = np.zeros_like(distances_m) if total_m == 0 else distances_m / total_m
    metric_points = np.column_stack((
        start_x + fractions * delta_x,
        start_y + fractions * delta_y,
    ))
    arrays = grid._extract_profiles_metric(
        metric_points, value_names=value_names, bounds_error=bounds_error
    )

    n_points = metric_points.shape[0]
    n_depth = grid.depth.size
    profiles = pd.DataFrame({
        "Path_index": np.repeat(np.arange(n_points), n_depth),
        "Distance_km": np.repeat(distances_m / 1000.0, n_depth),
        "Path_x_m": np.repeat(metric_points[:, 0], n_depth),
        "Path_y_m": np.repeat(metric_points[:, 1], n_depth),
        "Depth_km": np.tile(grid.depth, n_points),
    })
    for name, array in arrays.items():
        profiles[name] = array.reshape(-1)

    summary = pd.DataFrame({"Depth_km": grid.depth.copy()})
    for name, array in arrays.items():
        summary[f"{name}_mean"] = np.nanmean(array, axis=0)
        summary[f"{name}_median"] = np.nanmedian(array, axis=0)
        summary[f"{name}_p05"] = np.nanpercentile(array, 5, axis=0)
        summary[f"{name}_p95"] = np.nanpercentile(array, 95, axis=0)

    profiles.attrs["interpolation_crs"] = grid.interpolation_crs.to_string()
    summary.attrs["interpolation_crs"] = grid.interpolation_crs.to_string()
    summary.attrs["n_profiles"] = n_points
    summary.attrs["path_length_km"] = total_m / 1000.0

    statistic_aliases = {
        "mean": "mean", "median": "median",
        "5": "p05", "05": "p05", "5th": "p05", "5%": "p05",
        "p5": "p05", "p05": "p05",
        "95": "p95", "95th": "p95", "95%": "p95", "p95": "p95",
    }
    profile_key = str(profile).strip().lower()
    if profile_key not in statistic_aliases:
        raise ValueError("profile must be 'mean', 'median', 'p05' or 'p95'.")
    statistic = statistic_aliases[profile_key]

    required_names = [vp_name, vs_name]
    required_optional = [name for name in (qs_name, qp_name) if name is not None]
    missing = [
        f"{name}_{statistic}" for name in required_names + required_optional
        if f"{name}_{statistic}" not in summary.columns
    ]
    if missing:
        raise KeyError(f"The path summary lacks requested columns: {missing}.")

    selected_profile = pd.DataFrame({
        "Depth_km": summary["Depth_km"].to_numpy(copy=True),
        vp_name: summary[f"{vp_name}_{statistic}"].to_numpy(copy=True),
        vs_name: summary[f"{vs_name}_{statistic}"].to_numpy(copy=True),
    })
    selected_density_name = None
    if density_name is not None and f"{density_name}_{statistic}" in summary.columns:
        selected_profile[density_name] = summary[
            f"{density_name}_{statistic}"
        ].to_numpy(copy=True)
        selected_density_name = density_name
    for name in required_optional:
        selected_profile[name] = summary[f"{name}_{statistic}"].to_numpy(copy=True)

    layers = profile_to_pyfk_layers(
        selected_profile,
        depth_col="Depth_km",
        vp_col=vp_name,
        vs_col=vs_name,
        density_col=selected_density_name,
        qs_col=qs_name,
        qp_col=qp_name,
        surface_depth_km=surface_depth_km,
        max_depth_km=max_depth_km,
        compress=compress,
    )
    return profiles, summary, layers_to_pyfk_array(layers)


def profile_to_pyfk_layers(
    profile: pd.DataFrame,
    *,
    depth_col: str,
    vp_col: str,
    vs_col: str,
    density_col: str | None = None,
    qs_col: str | None = None,
    qp_col: str | None = None,
    surface_depth_km: float = 0.0,
    max_depth_km: float | None = None,
    compress: bool = True,
    property_rtol: float = 1e-7,
    property_atol: float = 1e-9,
) -> pd.DataFrame:
    """Convert a depth-sampled profile to piecewise-constant pyFK layers.

    Depth and velocities use km and km/s; density is g/cm3 and Q is
    dimensionless. Repeated depths represent interfaces. ``qp_col`` requires
    ``qs_col``. No density or Q relationship is applied by this helper.
    """
    if qp_col is not None and qs_col is None:
        raise ValueError("qp_col requires qs_col.")
    if qp_col is not None and density_col is None:
        raise ValueError(
            "Explicit Qp requires density_col because pyFK's six-column model "
            "format is thickness, Vs, Vp, density, Qs, Qp. Omit qp_col to let "
            "pyFK supply Qp, or provide density_col explicitly."
        )

    source_cols = [depth_col, vp_col, vs_col]
    property_cols = [vs_col, vp_col]
    output_cols = ["Vs_km_s", "Vp_km_s"]
    optional = (
        (density_col, "Density_g_cm3"),
        (qs_col, "Qs"),
        (qp_col, "Qp"),
    )
    for source_name, output_name in optional:
        if source_name is not None:
            source_cols.append(source_name)
            property_cols.append(source_name)
            output_cols.append(output_name)

    missing = set(source_cols).difference(profile.columns)
    if missing:
        raise KeyError(f"Missing profile columns: {sorted(missing)}")

    table = profile.loc[:, source_cols].copy()
    for column in source_cols:
        table[column] = pd.to_numeric(table[column], errors="raise")
    if table.isna().any().any():
        raise ValueError("Profile contains missing values.")

    table.sort_values(depth_col, kind="mergesort", inplace=True, ignore_index=True)
    depths = table[depth_col].to_numpy(dtype=float)
    surface = float(surface_depth_km)
    bottom = float(depths[-1] if max_depth_km is None else max_depth_km)
    if surface < depths[0] or surface > depths[-1]:
        raise ValueError("surface_depth_km lies outside the profile.")
    if bottom <= surface or bottom > depths[-1]:
        raise ValueError("max_depth_km must lie within and below the selected surface.")

    surface_values = _sample_properties(table, surface, depth_col, property_cols)
    bottom_values = _sample_properties(table, bottom, depth_col, property_cols)
    interior = table.loc[(table[depth_col] > surface) & (table[depth_col] < bottom)]
    clipped = pd.DataFrame([
        {depth_col: surface, **dict(zip(property_cols, surface_values))},
        *interior.to_dict("records"),
        {depth_col: bottom, **dict(zip(property_cols, bottom_values))},
    ])
    clipped.sort_values(depth_col, kind="mergesort", inplace=True, ignore_index=True)
    below = clipped.groupby(depth_col, sort=True, as_index=False).tail(1).copy()
    below.sort_values(depth_col, inplace=True, ignore_index=True)

    tops = below[depth_col].to_numpy(dtype=float)
    properties = below[property_cols].to_numpy(dtype=float)
    thickness = np.r_[np.diff(tops), 0.0]
    if np.any(thickness[:-1] <= 0):
        raise ValueError("Non-positive finite layer thickness encountered.")
    if np.any(properties[:, 0] < 0) or np.any(properties[:, 1] <= 0):
        raise ValueError("Vs must be non-negative and Vp must be positive.")

    for index, output_name in enumerate(output_cols[2:], start=2):
        if np.any(properties[:, index] <= 0):
            label = "Density" if output_name == "Density_g_cm3" else output_name
            raise ValueError(f"{label} must be positive.")

    layers = pd.DataFrame({
        "Top_depth_km": tops - surface,
        "Thickness_km": thickness,
    })
    for index, name in enumerate(output_cols):
        layers[name] = properties[:, index]

    if compress:
        layers = _compress_layers(layers, output_cols, property_rtol, property_atol)
    return layers


def layers_to_pyfk_array(layers: pd.DataFrame) -> np.ndarray:
    """Return a validated model array using pyFK's native optional defaults.

    The helper never derives density, Qs or Qp. Instead it emits the shortest
    pyFK model representation consistent with the supplied columns:

    - 3 columns: ``thickness, Vs, Vp``; pyFK supplies density, Qs and Qp.
    - 4 columns: density only, or Qs only; pyFK supplies the omitted properties.
    - 5 columns: ``density, Qs``; pyFK supplies Qp.
    - 6 columns: ``density, Qs, Qp``; all optional properties are explicit.

    pyFK distinguishes a four-column Qs model from a density model by whether
    the fourth column contains a value greater than 20. Explicit Qp therefore
    requires explicit density so the six-column format is unambiguous.
    """
    base = ["Thickness_km", "Vs_km_s", "Vp_km_s"]
    missing = set(base).difference(layers.columns)
    if missing:
        raise KeyError(f"Missing pyFK layer columns: {sorted(missing)}")

    has_density = "Density_g_cm3" in layers.columns
    has_qs = "Qs" in layers.columns
    has_qp = "Qp" in layers.columns
    if has_qp and not has_qs:
        raise ValueError("A Qp column requires a Qs column.")
    if has_qp and not has_density:
        raise ValueError(
            "Explicit Qp requires Density_g_cm3 because pyFK's six-column "
            "model format cannot encode Qs and Qp while leaving density omitted."
        )

    table = layers.loc[:, base].astype(float)
    if table.shape[0] < 2:
        raise ValueError("A finite layer and a half-space are required.")
    thickness = table["Thickness_km"].to_numpy(dtype=float)
    if np.any(thickness[:-1] <= 0) or not np.isclose(thickness[-1], 0):
        raise ValueError("Finite layers must be positive and the half-space zero.")
    if not np.isfinite(table.to_numpy()).all():
        raise ValueError("pyFK layer values must be finite.")

    vs = table["Vs_km_s"].to_numpy(dtype=float)
    vp = table["Vp_km_s"].to_numpy(dtype=float)
    if np.any(vs < 0) or np.any(vp <= 0):
        raise ValueError("Vs must be non-negative and Vp must be positive.")

    optional_columns = [
        name for name in ("Density_g_cm3", "Qs", "Qp")
        if name in layers.columns
    ]
    if optional_columns:
        optional_values = layers.loc[:, optional_columns].to_numpy(dtype=float)
        if not np.isfinite(optional_values).all():
            raise ValueError("Supplied density and Q values must be finite.")
        if np.any(optional_values <= 0):
            raise ValueError("Supplied density and Q values must be positive.")

    if not has_density and not has_qs:
        return layers.loc[:, base].to_numpy(dtype=float)

    if has_density and not has_qs:
        density = layers["Density_g_cm3"].to_numpy(dtype=float)
        if np.any(density > 20.0):
            raise ValueError(
                "A four-column pyFK density model requires density <= 20 so "
                "pyFK does not interpret the fourth column as Qs."
            )
        return layers.loc[:, [*base, "Density_g_cm3"]].to_numpy(dtype=float)

    if has_qs and not has_density:
        qs = layers["Qs"].to_numpy(dtype=float)
        if not np.any(qs > 20.0):
            raise ValueError(
                "A Qs-only four-column pyFK model is ambiguous when every Qs "
                "value is <= 20; provide density explicitly or use Qs values "
                "that pyFK can identify as attenuation."
            )
        return layers.loc[:, [*base, "Qs"]].to_numpy(dtype=float)

    if has_density and has_qs and not has_qp:
        return layers.loc[:, [*base, "Density_g_cm3", "Qs"]].to_numpy(dtype=float)

    return layers.loc[:, [*base, "Density_g_cm3", "Qs", "Qp"]].to_numpy(dtype=float)

def make_pyfk_model(
    layers: pd.DataFrame,
    *,
    pyfk_flattening: bool = False,
    **seis_model_kwargs,
):
    """Instantiate :class:`pyfk.SeisModel` from a validated layer table.

    Parameters
    ----------
    layers
        Layer table accepted by :func:`layers_to_pyfk_array`. The table is
        converted to pyFK's positional numeric model format immediately before
        constructing :class:`pyfk.SeisModel`.
    pyfk_flattening
        Whether pyFK should apply its native spherical-Earth flattening
        transformation when constructing the seismic model. The default is
        ``False``, preserving the existing flat layered-Earth behaviour.
    **seis_model_kwargs
        Additional keyword arguments forwarded unchanged to
        :class:`pyfk.SeisModel`, for example ``use_kappa`` or ``r_planet``. For
        backward compatibility, an existing ``flattening=...`` keyword is also
        accepted here and takes precedence over ``pyfk_flattening``.

    Returns
    -------
    pyfk.SeisModel
        Constructed pyFK seismic model.

    Notes
    -----
    The pyFK import remains local so the coordinate and profile utilities can be
    imported in environments where pyFK itself is not installed. Only model
    construction is affected by ``pyfk_flattening``; the input layer table and
    array returned by :func:`layers_to_pyfk_array` are not modified in place.
    """

    if not isinstance(pyfk_flattening, (bool, np.bool_)):
        raise TypeError("pyfk_flattening must be a boolean.")

    kwargs = dict(seis_model_kwargs)
    if "flattening" in kwargs:
        legacy_flattening = kwargs["flattening"]
        if not isinstance(legacy_flattening, (bool, np.bool_)):
            raise TypeError("flattening must be a boolean when supplied.")
        kwargs["flattening"] = bool(legacy_flattening)
    else:
        kwargs["flattening"] = bool(pyfk_flattening)

    from pyfk import SeisModel

    return SeisModel(model=layers_to_pyfk_array(layers), **kwargs)


def write_mttime_herrmann_sac(
    dc_streams: Sequence,
    ep_streams: Sequence,
    *,
    stations: Sequence[str],
    source_depth_km: float,
    output_dir: str | Path,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Write pyFK Green's functions using MTtime Herrmann SAC filenames.

    Parameters
    ----------
    dc_streams, ep_streams
        Receiver-ordered pyFK streams for double-couple and explosion source
        calculations. Each double-couple stream must contain nine traces in
        ``PYFK_DC_FK_ORDER``; each explosion stream must contain three traces in
        ``PYFK_EP_FK_ORDER``.
    stations
        Station codes in the same receiver order as both stream sequences.
    source_depth_km
        Source depth used in the MTtime filename, formatted to four decimals.
    output_dir
        Output directory, created when necessary.
    overwrite
        Replace existing files when ``True``.

    Returns
    -------
    pandas.DataFrame
        Manifest containing station, source depth, basis name, output path,
        sample count and time step for each written file.

    Notes
    -----
    Ten files are written per station. Symmetry-zero ``TDD`` and ``TEX`` are
    omitted because they are not part of MTtime's Herrmann input basis.

    The polarity conversion is deliberately factored into two exported controls.
    ``PYFK_TO_MTTIME_HERRMANN_RELATIVE_SIGN`` handles the relative Herrmann-basis
    mapping: pyFK's native ``SS`` and ``DS`` bases have the opposite relative sign
    from MTtime, while ``DD`` and explosion bases retain their relative signs.
    ``PYFK_TO_MTTIME_GLOBAL_POLARITY`` then applies one common absolute polarity
    to every basis. Keeping these factors separate permits controlled polarity
    tests without editing the relative basis map.
    """

    from obspy.core.util.attribdict import AttribDict

    stations = [str(station) for station in stations]
    if not (len(dc_streams) == len(ep_streams) == len(stations)):
        raise ValueError("dc_streams, ep_streams and stations must have equal length.")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    depth = float(source_depth_km)
    manifest = []

    for station, dc_stream, ep_stream in zip(stations, dc_streams, ep_streams):
        if len(dc_stream) != 9 or len(ep_stream) != 3:
            raise ValueError(f"Station {station}: expected 9 DC and 3 EP traces.")
        _check_streams(station, dc_stream, ep_stream)
        basis = {
            **dict(zip(PYFK_DC_FK_ORDER, dc_stream)),
            **dict(zip(PYFK_EP_FK_ORDER, ep_stream)),
        }

        for component in MTTIME_HERRMANN_ORDER:
            trace = basis[component].copy()
            relative_sign = PYFK_TO_MTTIME_HERRMANN_RELATIVE_SIGN[component]
            effective_sign = PYFK_TO_MTTIME_GLOBAL_POLARITY * relative_sign * PYFK_TO_MTTIME_AMPLITUDE_SCALE
            trace.data = np.asarray(trace.data) * effective_sign
            path = output / f"{station}.{depth:.4f}.{component}"
            if path.exists() and not overwrite:
                raise FileExistsError(f"Output exists: {path}")

            trace.stats.station = station
            trace.stats.channel = component
            if not hasattr(trace.stats, "sac"):
                trace.stats.sac = AttribDict()
            trace.stats.sac.kstnm = station[:8]
            trace.stats.sac.kcmpnm = component[:8]
            trace.stats.sac.evdp = depth
            trace.write(str(path), format="SAC")

            manifest.append({
                "station": station,
                "source_depth_km": depth,
                "basis": component,
                "path": str(path),
                "npts": int(trace.stats.npts),
                "dt_s": float(trace.stats.delta),
            })

    return pd.DataFrame(manifest)


def _build_standard_rectilinear_grid(
    table: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    depth_col: str,
    value_cols: Sequence[str],
    depth: np.ndarray,
    coordinate_crs: CRS,
) -> VelocityGrid3D:
    x = np.sort(table[x_col].unique().astype(float))
    y = np.sort(table[y_col].unique().astype(float))
    sorted_table = table.sort_values(
        [depth_col, y_col, x_col],
        kind="mergesort",
    ).reset_index(drop=True)
    shape = (depth.size, y.size, x.size)
    values = {
        column: sorted_table[column].to_numpy(dtype=float).reshape(shape)
        for column in value_cols
    }
    interpolators = {
        name: RegularGridInterpolator(
            (depth, y, x),
            array,
            method="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
        for name, array in values.items()
    }
    tolerance = _coordinate_tolerance(x, y)
    return VelocityGrid3D(
        depth=depth,
        values=values,
        interpolation_crs=coordinate_crs,
        _mode="rectilinear",
        _x_axis=x,
        _y_axis=y,
        _interpolators=interpolators,
        _triangulation=None,
        _coordinate_crs=coordinate_crs,
        _longitude_center=None,
        _local_tm_crs=None,
        _origin_tm_x=None,
        _origin_tm_y=None,
        _rotation_deg=None,
        _tolerance=tolerance,
    )


def _build_rotated_rectilinear_grid(
    table: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    depth_col: str,
    value_cols: Sequence[str],
    depth: np.ndarray,
    interpolation_crs: str | int | CRS | None,
    origin: tuple[float, float],
    origin_crs: str | int | CRS,
    rotation_deg: float,
    central_meridian: float,
    scale_factor: float,
) -> VelocityGrid3D:
    if not _is_complete_rectilinear_grid(
        table,
        x_col=x_col,
        y_col=y_col,
        depth_col=depth_col,
        depth=depth,
    ):
        raise ValueError(
            "A rotated computational grid must contain every combination of "
            "its unique x, y and depth coordinates."
        )

    source_crs = CRS.from_user_input(origin_crs)
    local_tm = _local_tm_crs(
        source_crs,
        central_meridian=central_meridian,
        scale_factor=scale_factor,
    )
    metric_crs = CRS.from_user_input(interpolation_crs or local_tm)
    _require_projected_metre_crs(metric_crs, "interpolation_crs")

    transformer = Transformer.from_crs(
        source_crs,
        local_tm,
        always_xy=True,
        force_over=True,
    )
    origin_e, origin_n = transformer.transform(float(origin[0]), float(origin[1]))

    grid = _build_standard_rectilinear_grid(
        table,
        x_col=x_col,
        y_col=y_col,
        depth_col=depth_col,
        value_cols=value_cols,
        depth=depth,
        coordinate_crs=metric_crs,
    )
    grid.interpolation_crs = metric_crs
    grid._coordinate_crs = None
    grid._local_tm_crs = local_tm
    grid._origin_tm_x = float(origin_e)
    grid._origin_tm_y = float(origin_n)
    grid._rotation_deg = rotation_deg
    grid._longitude_center = float(central_meridian)
    return grid


def _build_triangulated_grid(
    table: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    depth_col: str,
    value_cols: Sequence[str],
    depth: np.ndarray,
    coordinate_crs: CRS,
    interpolation_crs: CRS,
) -> VelocityGrid3D:
    first_depth = depth[0]
    nodes = table.loc[
        table[depth_col] == first_depth,
        [x_col, y_col],
    ].copy()
    nodes.sort_values([y_col, x_col], kind="mergesort", inplace=True, ignore_index=True)
    if nodes.duplicated([x_col, y_col]).any():
        raise ValueError("Duplicate horizontal nodes were found at the first depth.")

    node_index = pd.MultiIndex.from_frame(nodes[[x_col, y_col]])
    value_arrays = {
        name: np.empty((depth.size, len(nodes)), dtype=float)
        for name in value_cols
    }

    for depth_index, depth_value in enumerate(depth):
        layer = table.loc[table[depth_col] == depth_value].copy()
        layer.set_index([x_col, y_col], inplace=True)
        if layer.index.has_duplicates:
            raise ValueError(f"Duplicate horizontal nodes at depth {depth_value:g} km.")
        if len(layer) != len(nodes) or not layer.index.isin(node_index).all():
            raise ValueError(
                "Every depth must contain the same horizontal coordinate pairs "
                "for triangulated interpolation."
            )
        layer = layer.reindex(node_index)
        if layer[value_cols].isna().any().any():
            raise ValueError(
                f"The horizontal node set differs at depth {depth_value:g} km."
            )
        for name in value_cols:
            value_arrays[name][depth_index] = layer[name].to_numpy(dtype=float)

    longitude_center = None
    node_x = nodes[x_col].to_numpy(dtype=float)
    node_y = nodes[y_col].to_numpy(dtype=float)
    if coordinate_crs.is_geographic:
        longitude_center = _circular_mean_longitude(node_x)
        node_x = _unwrap_longitude(node_x, longitude_center)

    transformer = Transformer.from_crs(
        coordinate_crs,
        interpolation_crs,
        always_xy=True,
        force_over=True,
    )
    projected_x, projected_y = transformer.transform(node_x, node_y)
    projected_nodes = np.column_stack((
        np.asarray(projected_x, dtype=float),
        np.asarray(projected_y, dtype=float),
    ))
    if not np.isfinite(projected_nodes).all():
        raise ValueError("Coordinate transformation produced non-finite nodes.")
    if pd.DataFrame(projected_nodes).duplicated().any():
        raise ValueError("Distinct input nodes collapse to duplicate projected coordinates.")

    triangulation = Delaunay(projected_nodes)
    return VelocityGrid3D(
        depth=depth,
        values=value_arrays,
        interpolation_crs=interpolation_crs,
        _mode="triangulated",
        _x_axis=None,
        _y_axis=None,
        _interpolators=None,
        _triangulation=triangulation,
        _coordinate_crs=coordinate_crs,
        _longitude_center=longitude_center,
        _local_tm_crs=None,
        _origin_tm_x=None,
        _origin_tm_y=None,
        _rotation_deg=None,
        _tolerance=0.0,
    )


def _is_complete_rectilinear_grid(
    table: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    depth_col: str,
    depth: np.ndarray,
) -> bool:
    x_count = table[x_col].nunique()
    y_count = table[y_col].nunique()
    return len(table) == depth.size * x_count * y_count



def _validate_velocity_values(
    table: pd.DataFrame,
    vp_col: str,
    vs_col: str,
    density_col: str | None,
    qs_col: str | None = None,
    qp_col: str | None = None,
) -> None:
    if qp_col is not None and qs_col is None:
        raise ValueError("qp_col requires qs_col.")
    if np.any(table[vp_col].to_numpy(dtype=float) <= 0):
        raise ValueError("Vp must be positive.")
    if np.any(table[vs_col].to_numpy(dtype=float) < 0):
        raise ValueError("Vs must be non-negative.")
    for column, label in ((density_col, "Density"), (qs_col, "Qs"), (qp_col, "Qp")):
        if column is not None and np.any(table[column].to_numpy(dtype=float) <= 0):
            raise ValueError(f"{label} must be positive.")

def _coordinate_tolerance(x: np.ndarray, y: np.ndarray) -> float:
    return max(1e-8, 1e-10 * max(float(np.ptp(x)), float(np.ptp(y)), 1.0))


def _require_projected_metre_crs(crs: CRS, name: str) -> None:
    if not crs.is_projected:
        raise ValueError(f"{name} must be projected rather than geographic.")
    for axis in crs.axis_info[:2]:
        factor = axis.unit_conversion_factor
        if factor is None or not np.isclose(factor, 1.0):
            raise ValueError(f"{name} must use metre axes; {axis.name} uses {axis.unit_name}.")


def _local_tm_crs(source_crs: CRS, *, central_meridian: float, scale_factor: float) -> CRS:
    geodetic = source_crs.geodetic_crs
    if geodetic is None:
        raise ValueError("origin_crs must have an associated geodetic CRS.")
    ellipsoid = geodetic.ellipsoid
    return CRS.from_proj4(
        "+proj=tmerc +lat_0=0 "
        f"+lon_0={central_meridian:g} +k={scale_factor:g} +x_0=0 +y_0=0 "
        f"+a={ellipsoid.semi_major_metre:.12g} "
        f"+rf={ellipsoid.inverse_flattening:.12g} +units=m +no_defs"
    )


def _circular_mean_longitude(values: np.ndarray) -> float:
    angles = np.radians(np.asarray(values, dtype=float))
    return float(np.degrees(np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))))


def _unwrap_longitude(values, center: float | None):
    array = np.asarray(values, dtype=float)
    if center is None:
        return array
    return center + ((array - center + 180.0) % 360.0 - 180.0)


def _sample_properties(
    table: pd.DataFrame,
    target: float,
    depth_col: str,
    property_cols: Sequence[str],
) -> np.ndarray:
    depth = table[depth_col].to_numpy(dtype=float)
    exact = np.flatnonzero(np.isclose(depth, target, atol=1e-10, rtol=0))
    if exact.size:
        return table.loc[exact[-1], property_cols].to_numpy(dtype=float)

    lower = np.flatnonzero(depth < target)
    upper = np.flatnonzero(depth > target)
    if not lower.size or not upper.size:
        raise ValueError(f"Cannot interpolate profile at depth {target:g} km.")

    d0, d1 = depth[lower[-1]], depth[upper[0]]
    i0 = np.flatnonzero(np.isclose(depth, d0, atol=1e-10, rtol=0))[-1]
    i1 = np.flatnonzero(np.isclose(depth, d1, atol=1e-10, rtol=0))[0]
    v0 = table.loc[i0, property_cols].to_numpy(dtype=float)
    v1 = table.loc[i1, property_cols].to_numpy(dtype=float)
    return v0 + (target - d0) / (d1 - d0) * (v1 - v0)


def _compress_layers(
    layers: pd.DataFrame,
    property_cols: Sequence[str],
    rtol: float,
    atol: float,
) -> pd.DataFrame:
    rows = []
    for _, row in layers.iloc[:-1].iterrows():
        if rows and np.allclose(
            row.loc[list(property_cols)].to_numpy(dtype=float),
            rows[-1].loc[list(property_cols)].to_numpy(dtype=float),
            rtol=rtol,
            atol=atol,
        ):
            rows[-1]["Thickness_km"] += float(row["Thickness_km"])
        else:
            rows.append(row.copy())

    compressed = pd.DataFrame(rows)
    if compressed.empty:
        raise ValueError("No finite layers remain after compression.")

    compressed["Top_depth_km"] = np.r_[
        0.0,
        np.cumsum(compressed["Thickness_km"].to_numpy(dtype=float))[:-1],
    ]
    halfspace = layers.iloc[-1].copy()
    halfspace["Top_depth_km"] = float(compressed["Thickness_km"].sum())
    halfspace["Thickness_km"] = 0.0
    return pd.concat([compressed, halfspace.to_frame().T], ignore_index=True)


def _check_streams(station: str, dc_stream, ep_stream) -> None:
    reference = dc_stream[0]
    for label, stream in (("DC", dc_stream), ("EP", ep_stream)):
        for index, trace in enumerate(stream):
            if trace.stats.npts != reference.stats.npts:
                raise ValueError(f"Station {station}: {label} trace {index} has different npts.")
            if not np.isclose(trace.stats.delta, reference.stats.delta):
                raise ValueError(f"Station {station}: {label} trace {index} has different dt.")
            if abs(trace.stats.starttime - reference.stats.starttime) > 1e-6:
                raise ValueError(f"Station {station}: {label} trace {index} has different starttime.")
            if not np.isfinite(trace.data).all():
                raise ValueError(f"Station {station}: {label} trace {index} is non-finite.")
