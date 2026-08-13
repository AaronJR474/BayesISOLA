"""Workflow helpers for automated BayesISOLA centroid-moment-tensor inversion.

Version 0.1.8 makes ``use_precalculated_Green`` the single backend-independent
Green's-function cache policy. ``False`` forces regeneration, ``"auto"`` reuses
compatible outputs and regenerates missing/incompatible outputs, and ``True``
requires a complete compatible cache and raises rather than generating anything.
Syngine's backend-specific ``gf_options["overwrite"]`` control is therefore removed;
cache/reuse policy is no longer duplicated inside backend options. ``run["gf"]`` now
records the resolved cache policy explicitly. No Green's-function mathematics,
station/path model construction, acquisition, covariance weighting or inversion
algorithm is changed.

Version 0.1.7 adds an explicit Green's-function backend contract while preserving
``gf_source='axitra', gf_options=None`` as the validated historical path. Native
Axitra can now derive station-dependent layered models from a ``gf_helpers`` 3-D
velocity grid using either a vertical profile at each receiver (``grid='station'``)
or a representative catalogue-event-to-station path profile (``grid='path'``).
Model identifiers are written through BayesISOLA's native optional network field,
exact duplicate profiles are grouped, and model-specific cache metadata are checked
against the actual crust/station files used by Axitra. The corrected EarthScope
Syngine backend is available through ``gf_source='syngine'`` and writes the same six
elementary-seismogram interface consumed by the unchanged BayesISOLA inversion.
Every run returns first-class backend metadata in ``run['gf']``.

Version 0.1.7 also removes posterior ``±`` values from the custom CMT summary.
Optional BayesISOLA uncertainty sampling remains available as a separate diagnostic
table/output when ``n_uncertainty`` is requested, but it is not mixed into the
deterministic source-summary figure and native uncertainty plotting remains disabled
in the helper presets.

The 0.1.6 acquisition/results path is otherwise retained: ordered multi-client FDSN
fallback, magnitude-based station radii, origin-centred waveform windows, reusable
local miniSEED/StationXML input, pre-inversion waveform screening, explicit
noise/no-covariance branches, curated result tables and deterministic plotting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence
import math
import shutil
import hashlib
import io

import numpy as np
import pandas as pd


__version__ = "0.1.8"

__all__ = [
    "__version__",
    "get_max_radius",
    "discover_stations",
    "write_network_file",
    "get_network_file",
    "get_mseed_stationxml",
    "get_waveform_window",
    "load_streams_fdsnws_auto",
    "load_streams_local",
    "plot_waveform_section",
    "plot_station_section",
    "suggest_depth_limits",
    "diagnose_grid_edge",
    "extract_station_fit_df",
    "extract_centroid_location",
    "extract_solution_summary",
    "extract_uncertainty_df",
    "write_solution_outputs",
    "plot_cmt_summary",
    "PLOT_PRESETS",
    "run_auto_cmt",
]


_COMPONENT_SCHEMES = ("ZNE", "Z12", "123")
_GROUND_LEVEL_DEPTH_ATOL_M = 1e-6
_CHANNEL_COORDINATE_ATOL_DEG = 1e-4
_CHANNEL_ELEVATION_ATOL_M = 1.0

_MAGNITUDES = np.array([
    3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4,
    4.5, 4.6, 4.7, 4.8, 4.9, 5.0, 5.1, 5.2, 5.3, 5.4,
    5.5, 5.6, 5.7, 5.8, 5.9, 6.0, 6.1, 6.2, 6.3, 6.4,
    6.5, 6.6, 6.7, 6.8, 6.9, 7.0, 7.1, 7.2, 7.3, 7.4,
    7.5, 7.6, 7.7, 7.8, 7.9, 8.0,
], dtype=float)

_MAX_RADII = np.array([
    96.001584, 95.9631833664, 98.0, 102.0, 108.026902688,
    114.868566765, 123.445238634, 128.689599653, 134.586833832,
    145.681117253, 157.689926419, 170.688647664, 188.45405921,
    192.223140394, 203.988734372, 216.474476825, 233.196675756,
    248.661128941, 258.190038239, 268.728407146, 280.032818545,
    297.173067302, 315.362436406, 334.665140413, 361.817781899,
    384.425050255, 407.954938731, 439.468601405, 468.049881681,
    497.294793922, 517.385503596, 538.287877942, 560.03470821,
    589.576829967, 615.854723363, 643.04652117, 669.293317953,
    696.332767998, 724.464611825, 753.732982143, 793.492788462,
    845.098386021, 897.18404165, 949.816128972, 994.534662958,
    1055.40814061,
], dtype=float)


def get_max_radius(magnitude, scale_factor=1.66):
    """
    Estimate the maximum radius from earthquake magnitude.

    The unscaled radius is obtained by linear interpolation through the
    tabulated magnitude-radius relationship and is then multiplied by
    ``scale_factor``.

    Parameters
    ----------
    magnitude : float or array-like
        Earthquake magnitude. Values must lie within 3.5 to 8.0.
    scale_factor : float, default=1.66
        Multiplicative factor applied to the interpolated radius.

    Returns
    -------
    float or numpy.ndarray
        Scaled maximum radius. A scalar input returns a float, while an
        array-like input returns a NumPy array.
    """
    magnitude = np.asarray(magnitude, dtype=float)
    scale_factor = float(scale_factor)

    if not np.all(np.isfinite(magnitude)):
        raise ValueError("magnitude must contain only finite values.")

    if not np.isfinite(scale_factor) or scale_factor <= 0.0:
        raise ValueError("scale_factor must be a positive finite value.")

    mag_min = _MAGNITUDES[0]
    mag_max = _MAGNITUDES[-1]

    if np.any((magnitude < mag_min) | (magnitude > mag_max)):
        raise ValueError(
            f"magnitude must lie within [{mag_min:.1f}, {mag_max:.1f}]."
        )

    radius = np.interp(magnitude, _MAGNITUDES, _MAX_RADII)
    scaled_radius = radius * scale_factor

    return float(scaled_radius) if magnitude.ndim == 0 else scaled_radius


def _resolve_max_radius_km(magnitude: float | None, max_radius_km: float | None, radius_scale_factor: float) -> float:
    """Resolve the station-search radius, using ``get_max_radius`` only when requested."""
    if max_radius_km is None:
        if magnitude is None:
            raise ValueError("magnitude is required when max_radius_km=None.")
        return float(get_max_radius(float(magnitude), scale_factor=float(radius_scale_factor)))
    value = float(max_radius_km)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("max_radius_km must be a positive finite value or None.")
    return value


def _bayesisola_rupture_length_m(magnitude: float) -> float:
    """Return the exact rupture-length proxy used by ``BayesISOLA.set_event_info``."""
    magnitude = float(magnitude)
    if not np.isfinite(magnitude):
        raise ValueError("magnitude must be finite.")
    return float(math.sqrt(111.0 * 10.0**magnitude))


def _depth_bounds_km(
    catalog_depth_km: float,
    *,
    min_depth_km: float,
    min_depth_multiplier: float,
    max_depth_multiplier: float,
) -> tuple[float, float]:
    """Resolve the explicit BayesISOLA depth limits used by both grid and acquisition."""
    catalog_depth_km = float(catalog_depth_km)
    min_depth_km = float(min_depth_km)
    min_depth_multiplier = float(min_depth_multiplier)
    max_depth_multiplier = float(max_depth_multiplier)
    if not np.isfinite([catalog_depth_km, min_depth_km, min_depth_multiplier, max_depth_multiplier]).all():
        raise ValueError("Depth controls must be finite.")
    if catalog_depth_km <= 0:
        raise ValueError("catalog_depth_km must be positive.")
    if min_depth_km < 0:
        raise ValueError("min_depth_km cannot be negative.")
    if not 0.0 <= min_depth_multiplier <= 1.0:
        raise ValueError("min_depth_multiplier must lie within [0, 1].")
    if max_depth_multiplier <= 1.0:
        raise ValueError("max_depth_multiplier must be > 1.")

    grid_min_depth_km = max(min_depth_km, catalog_depth_km * min_depth_multiplier)
    grid_max_depth_km = catalog_depth_km * max_depth_multiplier
    if grid_max_depth_km <= grid_min_depth_km:
        raise ValueError(
            "The requested depth controls give grid_max_depth_km <= grid_min_depth_km. "
            "Increase max_depth_multiplier or reduce min_depth_km/min_depth_multiplier."
        )
    return grid_min_depth_km, grid_max_depth_km


def _waveform_window_from_bounds(
    max_distance_km: float,
    depth_max_km: float,
    shift_min_s: float,
    shift_max_s: float,
    *,
    velocity_slowest_m_s: float,
    covariance: str,
    noise_factor: float,
    edge_margin_s: float,
    minimum_pre_event_s: float,
) -> dict[str, float]:
    """Apply BayesISOLA's time-window equations to already resolved bounds."""
    max_distance_km = float(max_distance_km)
    depth_max_km = float(depth_max_km)
    shift_min_s = float(shift_min_s)
    shift_max_s = float(shift_max_s)
    velocity_slowest_m_s = float(velocity_slowest_m_s)
    noise_factor = float(noise_factor)
    edge_margin_s = float(edge_margin_s)
    minimum_pre_event_s = float(minimum_pre_event_s)
    covariance = str(covariance).lower().strip()
    if covariance not in {"none", "noise"}:
        raise ValueError("covariance must be 'none' or 'noise'.")
    if max_distance_km < 0 or depth_max_km <= 0 or velocity_slowest_m_s <= 0:
        raise ValueError("Distance/depth must be non-negative/positive and velocity_slowest_m_s must be positive.")
    if covariance == "noise" and noise_factor < 1.1:
        raise ValueError("noise_factor must be >= 1.1 when covariance='noise'; 4.0 matches BayesISOLA's default noise slice.")
    if edge_margin_s < 0 or minimum_pre_event_s < 0:
        raise ValueError("edge_margin_s and minimum_pre_event_s cannot be negative.")

    t_min = 0.0
    hypocentral_max_m = math.hypot(max_distance_km * 1000.0, depth_max_km * 1000.0)
    t_max = hypocentral_max_m / velocity_slowest_m_s
    processing_length = t_max - t_min + shift_max_s + 10.0
    if covariance == "noise":
        t_before = noise_factor * processing_length - shift_min_s - t_min
    else:
        t_before = max(0.0, -(shift_min_s + t_min))
    t_before = max(t_before, minimum_pre_event_s)
    trim_end = shift_min_s + t_min + processing_length
    shifted_end = t_max + shift_max_s + 1.0
    t_after = max(0.0, trim_end, shifted_end)
    t_before = float(math.ceil(t_before + edge_margin_s))
    t_after = float(math.ceil(t_after + edge_margin_s))
    return {
        "max_distance_km": max_distance_km,
        "depth_max_km": depth_max_km,
        "shift_min_s": shift_min_s,
        "shift_max_s": shift_max_s,
        "t_min_s": t_min,
        "t_max_s": t_max,
        "processing_length_s": processing_length,
        "noise_factor": noise_factor if covariance == "noise" else 0.0,
        "minimum_pre_event_s": minimum_pre_event_s,
        "t_before_s": t_before,
        "t_after_s": t_after,
    }

def _client_specs(client) -> tuple[Any, ...]:
    specs = tuple(client) if isinstance(client, (list, tuple)) else (client,)
    if not specs:
        raise ValueError("client cannot be an empty list or tuple.")
    return specs


def _client_label(client_spec: Any) -> str:
    if isinstance(client_spec, str):
        label = client_spec.strip()
        if not label:
            raise ValueError("FDSN client provider names cannot be empty.")
        return label
    base_url = getattr(client_spec, "base_url", None)
    return str(base_url).strip() if base_url is not None and str(base_url).strip() else client_spec.__class__.__name__


def _normalize_location(value) -> str:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return ""
    value = str(value).strip()
    return "" if value == "--" else value


def _as_bool(value, *, default: bool = False) -> bool:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return bool(default)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in {0, 1}:
        return bool(int(value))
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    raise ValueError(f"Cannot interpret {value!r} as a boolean.")


_STATION_TEXT_COLUMNS = (
    "network", "station", "location", "channel_prefix", "station_id",
    "component_scheme", "selected_channels", "channels", "gf_model",
)


def _coerce_station_df(station_df: pd.DataFrame | str | Path) -> pd.DataFrame:
    """Return a station table while preserving literal SEED identifier fields.

    pandas type inference can otherwise coerce identifiers such as location
    ``"00"`` or station ``"0123"`` to integers when a saved CSV is reloaded.
    Converters are used rather than ``dtype=str`` so literal codes such as
    network ``"NA"`` are not interpreted as missing values.
    """
    if isinstance(station_df, (str, Path)):
        converters = {column: lambda value: str(value).strip() for column in _STATION_TEXT_COLUMNS}
        table = pd.read_csv(Path(station_df).expanduser(), converters=converters)
    else:
        table = station_df.copy()
    if table.empty:
        raise ValueError("station_df is empty.")
    return table


def _parse_channel_codes(value: Sequence[str] | str | None) -> tuple[str, ...]:
    """Normalize channel codes from a sequence or comma-separated string.

    ``value`` is commonly a Python list during StationXML discovery, so missing
    value handling is restricted to scalar inputs. Applying ``pd.isna`` directly
    to a sequence returns a boolean array and cannot be used as a scalar truth
    value.
    """
    if value is None:
        return ()

    if isinstance(value, str):
        values = value.split(",")
    elif value is pd.NA or (
        isinstance(value, (float, np.floating)) and np.isnan(value)
    ):
        return ()
    else:
        values = value

    return tuple(
        sorted({str(item).strip() for item in values if str(item).strip()})
    )


def _component_selection_from_codes(channel_codes: Sequence[str]) -> tuple[str, tuple[str, str, str]]:
    """Choose one explicit three-component family, preferring ZNE, Z12, then 123."""
    codes = _parse_channel_codes(channel_codes)
    by_suffix: dict[str, list[str]] = {}
    for code in codes:
        by_suffix.setdefault(code[-1].upper(), []).append(code)
    for scheme in _COMPONENT_SCHEMES:
        if all(suffix in by_suffix for suffix in scheme):
            return scheme, tuple(sorted(by_suffix[suffix])[0] for suffix in scheme)
    raise ValueError(f"No complete ZNE, Z12 or 123 family was found in {list(codes)}.")


def _row_component_selection(row: Mapping[str, Any]) -> tuple[str, tuple[str, str, str]]:
    selected = _parse_channel_codes(row.get("selected_channels"))
    scheme = str(row.get("component_scheme", "")).upper().strip()
    if selected:
        inferred_scheme, inferred = _component_selection_from_codes(selected)
        if scheme and scheme != inferred_scheme:
            raise ValueError(f"component_scheme={scheme!r} conflicts with selected_channels={selected}.")
        return inferred_scheme, inferred
    return _component_selection_from_codes(_parse_channel_codes(row.get("channels")))


def _inventory_channel_families(inventory, channel_priority: Sequence[str], *, ground_level: bool) -> pd.DataFrame:
    """Select one complete three-component family per station from StationXML.

    Candidates are grouped by network, station, location and two-character
    channel prefix. Prefix priority is user-controlled; within a prefix the
    orientation preference is ZNE, Z12, then 123. Selected components must have
    one common positive sample rate, sensor depth, sensor location and elevation.
    When ``ground_level=True``, StationXML ``Channel.depth`` must be zero.
    """
    rows: list[dict[str, Any]] = []
    priority = {str(prefix): index for index, prefix in enumerate(channel_priority)}

    for network in inventory:
        for station in network:
            grouped: dict[tuple[str, str], list[Any]] = {}
            for channel in station.channels:
                prefix = str(channel.code)[:2]
                if prefix in priority:
                    grouped.setdefault((channel.location_code or "", prefix), []).append(channel)

            candidates = []
            for (location, prefix), channels in grouped.items():
                try:
                    scheme, selected_codes = _component_selection_from_codes([channel.code for channel in channels])
                except ValueError:
                    continue

                selected_objects = []
                for code in selected_codes:
                    matches = [channel for channel in channels if channel.code == code]
                    matches = [channel for channel in matches if np.isfinite(float(channel.sample_rate)) and float(channel.sample_rate) > 0]
                    if not matches:
                        selected_objects = []
                        break
                    selected_objects.append(max(matches, key=lambda channel: float(channel.sample_rate)))
                if len(selected_objects) != 3:
                    continue

                rates = np.asarray([float(channel.sample_rate) for channel in selected_objects])
                depths = np.asarray([float(channel.depth) for channel in selected_objects])
                latitudes = np.asarray([float(channel.latitude) for channel in selected_objects])
                longitudes = np.asarray([float(channel.longitude) for channel in selected_objects])
                elevations = np.asarray([float(channel.elevation) for channel in selected_objects])

                if not np.isfinite(np.r_[rates, depths, latitudes, longitudes, elevations]).all():
                    continue
                if not np.allclose(rates, rates[0], atol=1e-9, rtol=0.0):
                    continue
                if not np.allclose(depths, depths[0], atol=_GROUND_LEVEL_DEPTH_ATOL_M, rtol=0.0):
                    continue
                if ground_level and not np.allclose(depths, 0.0, atol=_GROUND_LEVEL_DEPTH_ATOL_M, rtol=0.0):
                    continue
                if not np.allclose(latitudes, latitudes[0], atol=_CHANNEL_COORDINATE_ATOL_DEG, rtol=0.0):
                    continue
                if not np.allclose(longitudes, longitudes[0], atol=_CHANNEL_COORDINATE_ATOL_DEG, rtol=0.0):
                    continue
                if not np.allclose(elevations, elevations[0], atol=_CHANNEL_ELEVATION_ATOL_M, rtol=0.0):
                    continue

                candidates.append({
                    "network": network.code, "station": station.code, "location": location, "channel_prefix": prefix,
                    "component_scheme": scheme, "selected_channels": ",".join(selected_codes),
                    "channels": ",".join(sorted({channel.code for channel in channels})), "sample_rate": float(rates[0]),
                    "channel_depth_m": float(depths.mean()), "station_lat": float(latitudes.mean()),
                    "station_lon": float(longitudes.mean()), "station_elevation_m": float(elevations.mean()),
                    "site_lat": float(station.latitude), "site_lon": float(station.longitude),
                    "site_elevation_m": float(station.elevation), "priority": priority[prefix],
                })

            if candidates:
                candidates.sort(key=lambda item: (item["priority"], _COMPONENT_SCHEMES.index(item["component_scheme"]), -item["sample_rate"], item["location"]))
                rows.append(candidates[0])

    columns = [
        "network", "station", "location", "channel_prefix", "component_scheme", "selected_channels", "channels",
        "sample_rate", "channel_depth_m", "station_lat", "station_lon", "station_elevation_m", "site_lat", "site_lon",
        "site_elevation_m", "priority",
    ]
    return pd.DataFrame(rows, columns=columns).sort_values(["network", "station"], ignore_index=True) if rows else pd.DataFrame(columns=columns)


def _compute_arrivals(taup, depth_km: float, event_lat: float, event_lon: float, station_lat: float, station_lon: float) -> tuple[float, float]:
    """Return first common regional P and S arrivals for plotting metadata."""
    p = taup.get_travel_times_geo(source_depth_in_km=depth_km, source_latitude_in_deg=event_lat, source_longitude_in_deg=event_lon,
                                  receiver_latitude_in_deg=station_lat, receiver_longitude_in_deg=station_lon, phase_list=["P", "p", "Pn", "Pg"])
    s = taup.get_travel_times_geo(source_depth_in_km=depth_km, source_latitude_in_deg=event_lat, source_longitude_in_deg=event_lon,
                                  receiver_latitude_in_deg=station_lat, receiver_longitude_in_deg=station_lon, phase_list=["S", "s", "Sn", "Sg"])
    return (float(p[0].time) if p else np.nan, float(s[0].time) if s else np.nan)


def discover_stations(
    event_id: str,
    origin_time,
    event_lon: float,
    event_lat: float,
    event_depth_km: float,
    *,
    magnitude: float | None = None,
    client: str | Sequence[str | Any] | Any = "GEONET",
    min_radius_km: float = 0.0,
    max_radius_km: float | None = None,
    radius_scale_factor: float = 1.66,
    ground_level: bool = True,
    channels: Sequence[str] = ("HH?", "BH?", "LH?"),
    channel_priority: Sequence[str] = ("HH", "BH", "LH"),
    taup_model: str = "iasp91",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Discover and select regional three-component stations without waveforms.

    ``client`` may be one ObsPy FDSN provider/client or an ordered sequence. Each
    provider is queried independently and the resulting station families are
    combined. One family is retained per network/station/location according to
    ``channel_priority`` and the orientation preference ZNE -> Z12 -> 123.

    ``max_radius_km`` is a user override. When it is ``None``, ``magnitude`` is
    required and the radius is computed with :func:`get_max_radius` using
    ``radius_scale_factor`` (default 1.66, matching the latest MTtime helper).
    Sensor coordinates from the retained StationXML channels define distance and
    azimuth; TauP P/S arrivals are diagnostic metadata only and never determine
    the BayesISOLA waveform length.

    Returns ``event_df``, ``station_df`` and ``discovery_log``. This function is
    metadata-only; use :func:`get_mseed_stationxml` for validated waveform and
    response acquisition with ordered provider fallback.
    """
    from obspy import UTCDateTime
    from obspy.clients.fdsn import Client
    from obspy.geodetics.base import gps2dist_azimuth, kilometers2degrees
    from obspy.taup import TauPyModel

    event_id = str(event_id).strip()
    event_lon, event_lat, event_depth_km = float(event_lon), float(event_lat), float(event_depth_km)
    min_radius_km = float(min_radius_km)
    max_radius_km = _resolve_max_radius_km(magnitude, max_radius_km, radius_scale_factor)
    if not event_id:
        raise ValueError("event_id cannot be empty.")
    if not np.isfinite([event_lon, event_lat, event_depth_km, min_radius_km, max_radius_km]).all():
        raise ValueError("Event coordinates/depth and station radii must be finite.")
    if not -90.0 <= event_lat <= 90.0 or not -180.0 <= event_lon <= 180.0 or event_depth_km < 0:
        raise ValueError("Invalid event longitude, latitude or depth.")
    if min_radius_km < 0 or min_radius_km >= max_radius_km:
        raise ValueError("Require 0 <= min_radius_km < max_radius_km.")
    if not channels or not channel_priority:
        raise ValueError("channels and channel_priority cannot be empty.")

    channel_patterns = tuple(str(value).strip() for value in channels)
    priority_prefixes = tuple(str(value).strip() for value in channel_priority)
    if any(len(value) != 2 for value in priority_prefixes):
        raise ValueError("channel_priority entries must be two-character prefixes.")

    origin = UTCDateTime(origin_time)
    specs = _client_specs(client)
    channel_query = ",".join(channel_patterns)
    padding_km = max(1.0, 0.01 * max_radius_km)
    query_min_deg = float(kilometers2degrees(max(0.0, min_radius_km - padding_km)))
    query_max_deg = float(kilometers2degrees(max_radius_km + padding_km))
    if query_max_deg > 180.0:
        raise ValueError("max_radius_km is too large for an FDSN radial query after padding.")

    candidate_tables, log_rows, labels = [], [], []
    for client_index, client_spec in enumerate(specs):
        try:
            label = _client_label(client_spec)
            labels.append(label)
            fdsn = Client(client_spec) if isinstance(client_spec, str) else client_spec
            if not callable(getattr(fdsn, "get_stations", None)):
                raise TypeError("Each client must provide get_stations().")
            inventory = fdsn.get_stations(latitude=event_lat, longitude=event_lon, minradius=query_min_deg, maxradius=query_max_deg,
                                          starttime=origin, endtime=origin + 1.0, channel=channel_query, level="channel")
            candidates = _inventory_channel_families(inventory, priority_prefixes, ground_level=ground_level)
        except Exception as exc:
            if len(specs) == 1:
                raise
            log_rows.append({"event_id": event_id, "fdsn_client": str(client_spec), "status": "client_failed", "reason": str(exc)})
            continue
        if not candidates.empty:
            candidates = candidates.copy()
            candidates["fdsn_client"] = label
            candidates["fdsn_client_index"] = client_index
            candidate_tables.append(candidates)

    candidate_df = pd.concat(candidate_tables, ignore_index=True, sort=False) if candidate_tables else pd.DataFrame()
    taup = TauPyModel(model=str(taup_model))
    selected_rows, selected_ids = [], set()
    distance_tolerance_km = max(1e-6, 1e-9 * max_radius_km)

    for row in candidate_df.itertuples(index=False):
        location = _normalize_location(row.location)
        station_id = f"{row.network}.{row.station}.{location or '--'}"
        distance_m, azimuth, back_azimuth = gps2dist_azimuth(event_lat, event_lon, float(row.station_lat), float(row.station_lon))
        distance_km = float(distance_m) / 1000.0
        base = {
            "event_id": event_id, "fdsn_client": row.fdsn_client, "fdsn_client_index": int(row.fdsn_client_index),
            "station_id": station_id, "network": row.network, "station": row.station, "location": location,
            "channel_prefix": row.channel_prefix, "component_scheme": row.component_scheme,
            "selected_channels": row.selected_channels, "channels": row.channels, "sample_rate": float(row.sample_rate),
            "channel_depth_m": float(row.channel_depth_m), "ground_level": bool(ground_level),
            "station_lat": float(row.station_lat), "station_lon": float(row.station_lon),
            "station_elevation_m": float(row.station_elevation_m), "site_lat": float(row.site_lat),
            "site_lon": float(row.site_lon), "site_elevation_m": float(row.site_elevation_m),
            "distance_km": distance_km, "azimuth_deg": float(azimuth) % 360.0,
            "back_azimuth_deg": float(back_azimuth) % 360.0,
        }
        if station_id in selected_ids:
            log_rows.append({**base, "status": "duplicate_client_station", "reason": "Station already selected from an earlier FDSN client."})
            continue
        if distance_km < min_radius_km - distance_tolerance_km or distance_km > max_radius_km + distance_tolerance_km:
            log_rows.append({**base, "status": "distance_excluded", "reason": f"Distance {distance_km:.3f} km outside [{min_radius_km:.3f}, {max_radius_km:.3f}] km."})
            continue
        p_time, s_time = _compute_arrivals(taup, event_depth_km, event_lat, event_lon, float(row.station_lat), float(row.station_lon))
        base.update({"p_arrival_s": p_time, "s_arrival_s": s_time})
        selected_rows.append(base)
        selected_ids.add(station_id)
        log_rows.append({**base, "status": "selected", "reason": ""})

    station_df = pd.DataFrame(selected_rows).sort_values("distance_km", ignore_index=True) if selected_rows else pd.DataFrame()
    event_df = pd.DataFrame([{
        "event_id": event_id, "origin_time": str(origin), "event_lon": event_lon, "event_lat": event_lat,
        "event_depth_km": event_depth_km, "magnitude": magnitude, "min_radius_km": min_radius_km, "max_radius_km": max_radius_km,
        "radius_scale_factor": float(radius_scale_factor), "ground_level": bool(ground_level), "taup_model": str(taup_model),
        "fdsn_clients": ",".join(labels),
    }])
    return event_df, station_df, pd.DataFrame(log_rows)


def write_network_file(station_df: pd.DataFrame | str | Path, filename: str | Path) -> Path:
    """Write BayesISOLA ``network.stn`` from the authoritative station table.

    When ``station_df`` contains ``gf_model``, its non-empty value is written as
    BayesISOLA's optional fourth network-field. Native Axitra then associates that
    receiver with ``crustal-<gf_model>.dat`` / ``station-<gf_model>.dat`` and the
    matching model-specific elementary seismograms. Without ``gf_model`` the
    historical three-field network format is preserved exactly.
    """
    table = _coerce_station_df(station_df)
    required = {"network", "station", "location", "channel_prefix", "station_lat", "station_lon"}
    missing = required.difference(table.columns)
    if missing:
        raise KeyError(f"station_df is missing network-file columns: {sorted(missing)}")
    filename = Path(filename).expanduser()
    filename.parent.mkdir(parents=True, exist_ok=True)
    with filename.open("w", encoding="utf-8", newline="\n") as f:
        for row in table.to_dict("records"):
            location = _normalize_location(row["location"])
            station_code = f"{row['network']}:{row['station']}:{location}:{row['channel_prefix']}"
            model_value = row.get("gf_model", "")
            model = "" if model_value is None or (not isinstance(model_value, str) and pd.isna(model_value)) else str(model_value).strip()
            suffix = f" {model}" if model else ""
            f.write(f"{station_code:<24s} {float(row['station_lat']):12.6f} {float(row['station_lon']):12.6f}{suffix}\n")
    return filename


def get_network_file(
    event_id: str,
    origin_time,
    event_lon: float,
    event_lat: float,
    event_depth_km: float,
    *,
    magnitude: float | None = None,
    client: str | Sequence[str | Any] | Any = "GEONET",
    min_radius_km: float = 0.0,
    max_radius_km: float | None = None,
    radius_scale_factor: float = 1.66,
    ground_level: bool = True,
    channels: Sequence[str] = ("HH?", "BH?", "LH?"),
    channel_priority: Sequence[str] = ("HH", "BH", "LH"),
    taup_model: str = "iasp91",
    output_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Discover stations and write ``input/network.stn`` without waveforms.

    This remains a useful metadata-only utility. ``max_radius_km=None`` selects
    the magnitude-dependent radius from :func:`get_max_radius`; otherwise the
    supplied radius is respected exactly. ``run_auto_cmt`` no longer needs this
    preliminary pass because :func:`get_mseed_stationxml` can now determine its
    own safe waveform window directly from discovered station geometry.
    """
    root = Path(output_dir).expanduser()
    event_df, station_df, discovery_log = discover_stations(
        event_id, origin_time, event_lon, event_lat, event_depth_km, magnitude=magnitude, client=client,
        min_radius_km=min_radius_km, max_radius_km=max_radius_km, radius_scale_factor=radius_scale_factor,
        ground_level=ground_level, channels=channels, channel_priority=channel_priority, taup_model=taup_model,
    )
    (root / "input").mkdir(parents=True, exist_ok=True)
    (root / "metadata").mkdir(parents=True, exist_ok=True)
    write_network_file(station_df, root / "input" / "network.stn")
    event_df.to_csv(root / "metadata" / "event.csv", index=False)
    station_df.to_csv(root / "metadata" / "stations_network.csv", index=False)
    discovery_log.to_csv(root / "metadata" / "network_log.csv", index=False)
    return event_df, station_df, discovery_log


def _raw_inputs_match_request(
    waveform_path: Path,
    stationxml_path: Path,
    *,
    row: Mapping[str, Any],
    request_start,
    request_end,
    response_time,
    read,
    read_inventory,
) -> tuple[bool, str]:
    """Validate cached miniSEED/StationXML against the selected station contract."""
    try:
        network, station = str(row["network"]), str(row["station"])
        location, prefix = _normalize_location(row.get("location")), str(row["channel_prefix"])
        stream = read(str(waveform_path), format="MSEED").select(network=network, station=station, location=location, channel=f"{prefix}?")
        if len(stream) == 0:
            return False, "no matching traces in existing miniSEED"
        _, requested = _row_component_selection(row)
        selected = stream.__class__()
        for channel in requested:
            matches = stream.select(channel=channel)
            if len(matches) == 0:
                return False, f"missing selected channel {channel}"
            selected += matches
        gaps = [gap for gap in selected.get_gaps() if gap[-1] > 0]
        if gaps:
            return False, f"internal gaps in existing miniSEED: {gaps}"
        selected.merge(method=0)
        if len(selected) != 3:
            return False, f"selected channel set does not reduce to three traces: {[trace.id for trace in selected]}"
        common_start = max(trace.stats.starttime for trace in selected)
        common_end = min(trace.stats.endtime for trace in selected)
        tolerance = max(float(trace.stats.delta) for trace in selected) + 1e-6
        if common_start > request_start + tolerance or common_end < request_end - tolerance:
            return False, "existing miniSEED does not cover the requested common window"

        inventory = read_inventory(str(stationxml_path), format="STATIONXML")
        depths, latitudes, longitudes, elevations = [], [], [], []
        for trace in selected:
            inventory.get_response(trace.id, response_time)
            metadata = inventory.get_channel_metadata(trace.id, response_time)
            depth = float(metadata.get("local_depth", np.nan))
            latitude = float(metadata.get("latitude", np.nan))
            longitude = float(metadata.get("longitude", np.nan))
            elevation = float(metadata.get("elevation", np.nan))
            if not np.isfinite([float(metadata["azimuth"]), float(metadata["dip"]), depth, latitude, longitude, elevation]).all():
                return False, f"non-finite StationXML metadata for {trace.id}"
            depths.append(depth); latitudes.append(latitude); longitudes.append(longitude); elevations.append(elevation)
        if not np.allclose(depths, depths[0], atol=_GROUND_LEVEL_DEPTH_ATOL_M, rtol=0.0):
            return False, f"selected channels have inconsistent sensor depths: {depths}"
        if _as_bool(row.get("ground_level"), default=False) and not np.allclose(depths, 0.0, atol=_GROUND_LEVEL_DEPTH_ATOL_M, rtol=0.0):
            return False, f"selected channels are not at ground level: {depths}"
        checks = [
            (depths, float(row.get("channel_depth_m", np.mean(depths))), _GROUND_LEVEL_DEPTH_ATOL_M),
            (latitudes, float(row.get("station_lat", np.mean(latitudes))), _CHANNEL_COORDINATE_ATOL_DEG),
            (longitudes, float(row.get("station_lon", np.mean(longitudes))), _CHANNEL_COORDINATE_ATOL_DEG),
            (elevations, float(row.get("station_elevation_m", np.mean(elevations))), _CHANNEL_ELEVATION_ATOL_M),
        ]
        for values, expected, atol in checks:
            if not np.allclose(values, values[0], atol=atol, rtol=0.0) or not np.isclose(float(np.mean(values)), expected, atol=atol, rtol=0.0):
                return False, "StationXML sensor geometry differs from discovery metadata"
        return True, "existing raw inputs satisfy the selected station and time-window contract"
    except Exception as exc:
        return False, f"existing raw-input validation failed: {exc}"


def get_mseed_stationxml(
    event_id: str,
    origin_time,
    event_lon: float,
    event_lat: float,
    event_depth_km: float,
    *,
    magnitude: float | None = None,
    t_before: float | None = None,
    t_after: float | None = None,
    output_dir: str | Path,
    station_df: pd.DataFrame | str | Path | None = None,
    client: str | Sequence[str | Any] | Any = "GEONET",
    min_radius_km: float = 0.0,
    max_radius_km: float | None = None,
    radius_scale_factor: float = 1.66,
    ground_level: bool = True,
    channels: Sequence[str] = ("HH?", "BH?", "LH?"),
    channel_priority: Sequence[str] = ("HH", "BH", "LH"),
    taup_model: str = "iasp91",
    min_depth_km: float = 5.0,
    min_depth_multiplier: float = 0.5,
    max_depth_multiplier: float = 3.0,
    time_unc_s: float = 2.0,
    rupture_velocity_m_s: float = 1000.0,
    velocity_slowest_m_s: float = 1000.0,
    covariance: str = "noise",
    noise_factor: float = 4.0,
    edge_margin_s: float = 1.0,
    minimum_pre_event_s: float = 20.0,
    overwrite: bool = False,
    plot: bool = False,
    show: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Discover/acquire BayesISOLA-ready miniSEED and StationXML.

    This function is intentionally usable *without* ``run_auto_cmt``. When
    ``t_before`` and/or ``t_after`` are omitted, it first obtains station metadata,
    determines the farthest in-radius candidate, and calls
    :func:`get_waveform_window` using the same depth/time/covariance assumptions as
    the automated inversion. The resulting common origin-centred interval is then
    used for every station. Explicit ``t_before``/``t_after`` values override only
    the corresponding automatically calculated side.

    ``max_radius_km`` behaves similarly: ``None`` uses :func:`get_max_radius` from
    the MTtime workflow (default scale factor 1.66), while a numerical value is a
    direct user override. Automatic radius or automatic record length requires
    ``magnitude``.

    Multiple clients/providers are preserved exactly as in the MTtime acquisition
    path. A network/station/location identifier is considered duplicated only
    after an earlier provider has supplied a *valid* miniSEED/StationXML pair.
    Thus a later provider can rescue a station whose metadata were advertised by
    an earlier service but whose waveform or response request failed. Stations
    unique to later providers are retained normally.

    Supplying ``station_df`` restricts acquisition to those station/channel rows.
    Successful files are validated for component completeness, common continuous
    coverage, response availability and sensor geometry, then stored under
    ``raw/waveforms`` and ``raw/stationxml``. ``plot=True`` writes the normalized,
    response-corrected unfiltered ZNE record section used for visual screening.

    Returns
    -------
    event_df, station_df, download_log, waveform_window
        ``station_df`` contains the exact local miniSEED/StationXML paths and is
        directly reusable with ``run_auto_cmt(waveform_source='local')``.
        ``waveform_window`` records both the calculated bounds and the final
        ``t_before_s``/``t_after_s`` actually requested.
    """
    from obspy import UTCDateTime, read, read_inventory
    from obspy.clients.fdsn import Client
    from obspy.geodetics.base import gps2dist_azimuth, kilometers2degrees
    from obspy.taup import TauPyModel

    event_id = str(event_id).strip()
    event_lon, event_lat, event_depth_km = float(event_lon), float(event_lat), float(event_depth_km)
    min_radius_km = float(min_radius_km)
    max_radius_km = _resolve_max_radius_km(magnitude, max_radius_km, radius_scale_factor)
    covariance = str(covariance).lower().strip()
    if not event_id:
        raise ValueError("event_id cannot be empty.")
    if not np.isfinite([event_lon, event_lat, event_depth_km, min_radius_km, max_radius_km]).all():
        raise ValueError("Event coordinates/depth and station radii must be finite.")
    if not -90.0 <= event_lat <= 90.0 or not -180.0 <= event_lon <= 180.0 or event_depth_km < 0:
        raise ValueError("Invalid event longitude, latitude or depth.")
    if min_radius_km < 0 or min_radius_km >= max_radius_km:
        raise ValueError("Require 0 <= min_radius_km < max_radius_km.")
    if covariance not in {"none", "noise"}:
        raise ValueError("covariance must be 'none' or 'noise'.")
    if not channels or not channel_priority:
        raise ValueError("channels and channel_priority cannot be empty.")
    if t_before is not None and (not np.isfinite(float(t_before)) or float(t_before) < 0):
        raise ValueError("t_before must be None or a finite value >= 0.")
    if t_after is not None and (not np.isfinite(float(t_after)) or float(t_after) <= 0):
        raise ValueError("t_after must be None or a finite value > 0.")

    channel_patterns = tuple(str(value).strip() for value in channels)
    priority_prefixes = tuple(str(value).strip() for value in channel_priority)
    if any(not value for value in channel_patterns):
        raise ValueError("channels cannot contain empty patterns.")
    if any(len(value) != 2 for value in priority_prefixes):
        raise ValueError("channel_priority entries must be two-character prefixes.")
    if len(set(priority_prefixes)) != len(priority_prefixes):
        raise ValueError("channel_priority cannot contain duplicate prefixes.")

    root = Path(output_dir).expanduser()
    waveform_dir, stationxml_dir, metadata_dir, figure_dir = root / "raw" / "waveforms", root / "raw" / "stationxml", root / "metadata", root / "figures"
    for directory in (waveform_dir, stationxml_dir, metadata_dir, root / "input"):
        directory.mkdir(parents=True, exist_ok=True)
    if plot:
        figure_dir.mkdir(parents=True, exist_ok=True)

    origin = UTCDateTime(origin_time)
    specs = _client_specs(client)
    requested_client_labels: list[str] = []
    fdsn_clients: list[Any] = []
    candidate_tables: list[pd.DataFrame] = []
    log_rows: list[dict[str, Any]] = []

    if station_df is None:
        channel_query = ",".join(channel_patterns)
        padding_km = max(1.0, 0.01 * max_radius_km)
        query_min_deg = float(kilometers2degrees(max(0.0, min_radius_km - padding_km)))
        query_max_deg = float(kilometers2degrees(max_radius_km + padding_km))
        if query_max_deg > 180.0:
            raise ValueError("max_radius_km is too large for an FDSN radial query after padding.")

        for client_spec in specs:
            try:
                client_label = _client_label(client_spec)
            except Exception as exc:
                if len(specs) == 1:
                    raise
                client_label = f"client_{len(requested_client_labels) + 1}"
                requested_client_labels.append(client_label)
                log_rows.append({"event_id": event_id, "fdsn_client": client_label, "status": "client_failed", "reason": str(exc), "cache_reason": "not_checked"})
                continue
            requested_client_labels.append(client_label)
            try:
                fdsn = Client(client_spec) if isinstance(client_spec, str) else client_spec
                if not callable(getattr(fdsn, "get_stations", None)) or not callable(getattr(fdsn, "get_waveforms", None)):
                    raise TypeError("Each client must provide callable get_stations() and get_waveforms() methods.")
                inventory = fdsn.get_stations(latitude=event_lat, longitude=event_lon, minradius=query_min_deg, maxradius=query_max_deg,
                                              starttime=origin, endtime=origin + 1.0, channel=channel_query, level="channel")
                candidates = _inventory_channel_families(inventory, priority_prefixes, ground_level=ground_level)
            except Exception as exc:
                if len(specs) == 1:
                    raise
                log_rows.append({"event_id": event_id, "fdsn_client": client_label, "status": "client_failed", "reason": str(exc), "cache_reason": "not_checked"})
                continue
            if candidates.empty:
                continue
            fdsn_index = len(fdsn_clients)
            fdsn_clients.append(fdsn)
            candidates = candidates.copy()
            candidates["fdsn_client_index"] = int(fdsn_index)
            candidates["fdsn_client"] = client_label
            candidate_tables.append(candidates)
        candidate_df = pd.concat(candidate_tables, ignore_index=True, sort=False) if candidate_tables else pd.DataFrame()
    else:
        candidate_df = _coerce_station_df(station_df)
        for client_spec in specs:
            requested_client_labels.append(_client_label(client_spec))
            fdsn_clients.append(Client(client_spec) if isinstance(client_spec, str) else client_spec)
        if "fdsn_client_index" not in candidate_df.columns:
            candidate_df = candidate_df.copy(); candidate_df["fdsn_client_index"] = 0
        if "fdsn_client" not in candidate_df.columns:
            candidate_df = candidate_df.copy(); candidate_df["fdsn_client"] = requested_client_labels[0]

    if candidate_df.empty:
        raise ValueError("No candidate stations were discovered or supplied.")

    distance_tolerance_km = max(1e-6, 1e-9 * max_radius_km)
    candidate_distances = []
    for row in candidate_df.to_dict("records"):
        if "distance_km" in row and pd.notna(row["distance_km"]) and np.isfinite(float(row["distance_km"])):
            distance_km = float(row["distance_km"])
        else:
            distance_m, _, _ = gps2dist_azimuth(event_lat, event_lon, float(row["station_lat"]), float(row["station_lon"]))
            distance_km = float(distance_m) / 1000.0
        if min_radius_km - distance_tolerance_km <= distance_km <= max_radius_km + distance_tolerance_km:
            candidate_distances.append(distance_km)
    if not candidate_distances:
        raise ValueError("No candidate stations lie within the requested epicentral annulus.")

    if t_before is None or t_after is None:
        if magnitude is None:
            raise ValueError("magnitude is required when t_before or t_after is determined automatically.")
        waveform_window = get_waveform_window(
            event_depth_km, float(magnitude), max_distance_km=max(candidate_distances), min_depth_km=min_depth_km,
            min_depth_multiplier=min_depth_multiplier, max_depth_multiplier=max_depth_multiplier, time_unc_s=time_unc_s,
            rupture_velocity_m_s=rupture_velocity_m_s, velocity_slowest_m_s=velocity_slowest_m_s, covariance=covariance,
            noise_factor=noise_factor, edge_margin_s=edge_margin_s, minimum_pre_event_s=minimum_pre_event_s,
        )
        waveform_window["window_source"] = "automatic"
    else:
        waveform_window = {"window_source": "explicit"}

    if t_before is not None:
        waveform_window["t_before_s"] = float(t_before)
        waveform_window["t_before_source"] = "explicit"
    else:
        waveform_window["t_before_source"] = "automatic"
    if t_after is not None:
        waveform_window["t_after_s"] = float(t_after)
        waveform_window["t_after_source"] = "explicit"
    else:
        waveform_window["t_after_source"] = "automatic"
    final_t_before = float(waveform_window["t_before_s"])
    final_t_after = float(waveform_window["t_after_s"])
    request_start, request_end = origin - final_t_before, origin + final_t_after

    taup = TauPyModel(model=str(taup_model))
    successful_station_ids: set[str] = set()
    successful_rows: list[dict[str, Any]] = []

    for row in candidate_df.to_dict("records"):
        client_index = int(row.get("fdsn_client_index", 0))
        if client_index < 0 or client_index >= len(fdsn_clients):
            raise ValueError(f"Station row references fdsn_client_index={client_index}, but only {len(fdsn_clients)} active clients exist.")
        fdsn = fdsn_clients[client_index]
        client_label = str(row.get("fdsn_client", requested_client_labels[client_index]))
        location = _normalize_location(row.get("location"))
        location_request = location if location else "--"
        station_id = str(row.get("station_id") or f"{row['network']}.{row['station']}.{location or '--'}")

        if station_id in successful_station_ids:
            log_rows.append({**row, "station_id": station_id, "fdsn_client": client_label, "status": "duplicate_client_station",
                             "reason": "The same network/station/location was already acquired successfully from an earlier client.", "cache_reason": "not_checked"})
            continue

        if "distance_km" in row and pd.notna(row["distance_km"]) and np.isfinite(float(row["distance_km"])):
            distance_km = float(row["distance_km"])
            azimuth = float(row.get("azimuth_deg", np.nan)); back_azimuth = float(row.get("back_azimuth_deg", np.nan))
        else:
            distance_m, azimuth, back_azimuth = gps2dist_azimuth(event_lat, event_lon, float(row["station_lat"]), float(row["station_lon"]))
            distance_km = float(distance_m) / 1000.0
        if distance_km < min_radius_km - distance_tolerance_km or distance_km > max_radius_km + distance_tolerance_km:
            log_rows.append({**row, "station_id": station_id, "fdsn_client": client_label, "distance_km": distance_km,
                             "status": "distance_excluded", "reason": f"Distance {distance_km:.6f} km lies outside [{min_radius_km:.6f}, {max_radius_km:.6f}] km.",
                             "cache_reason": "not_checked"})
            continue

        p_time, s_time = _compute_arrivals(taup, event_depth_km, event_lat, event_lon, float(row["station_lat"]), float(row["station_lon"]))
        base = {**row, "event_id": event_id, "station_id": station_id, "fdsn_client": client_label, "location": location,
                "distance_km": distance_km, "azimuth_deg": float(azimuth) % 360.0 if np.isfinite(azimuth) else np.nan,
                "back_azimuth_deg": float(back_azimuth) % 360.0 if np.isfinite(back_azimuth) else np.nan,
                "p_arrival_s": p_time, "s_arrival_s": s_time, "ground_level": bool(ground_level)}
        _, selected_channels = _row_component_selection(base)
        selected_query = ",".join(selected_channels)
        sample_interval_s = 1.0 / float(base["sample_rate"])
        padding_s = 2.0 * sample_interval_s
        waveform_request_start, waveform_request_end = request_start - padding_s, request_end + padding_s
        waveform_path = waveform_dir / f"{station_id}.{base['channel_prefix']}.mseed"
        stationxml_path = stationxml_dir / f"{station_id}.{base['channel_prefix']}.xml"
        existed_before = waveform_path.exists() or stationxml_path.exists()
        cache_reason, reuse = "overwrite=True", False

        if not overwrite and waveform_path.exists() and stationxml_path.exists():
            reuse, cache_reason = _raw_inputs_match_request(waveform_path, stationxml_path, row=base, request_start=request_start,
                                                            request_end=request_end, response_time=origin, read=read, read_inventory=read_inventory)

        if reuse:
            status = "existing"
        else:
            tmp_waveform = waveform_path.with_name(waveform_path.name + ".tmp")
            tmp_stationxml = stationxml_path.with_name(stationxml_path.name + ".tmp")
            try:
                inventory = fdsn.get_stations(network=base["network"], station=base["station"], location=location_request, channel=selected_query,
                                              starttime=waveform_request_start, endtime=waveform_request_end, level="response")
                stream = fdsn.get_waveforms(network=base["network"], station=base["station"], location=location_request, channel=selected_query,
                                            starttime=waveform_request_start, endtime=waveform_request_end, attach_response=False)
                if len(stream) == 0:
                    raise ValueError("Waveform request returned an empty stream.")
                stream.write(str(tmp_waveform), format="MSEED")
                inventory.write(str(tmp_stationxml), format="STATIONXML")
                valid, reason = _raw_inputs_match_request(tmp_waveform, tmp_stationxml, row=base, request_start=request_start,
                                                          request_end=request_end, response_time=origin, read=read, read_inventory=read_inventory)
                if not valid:
                    raise ValueError(f"Downloaded files failed validation: {reason}")
                tmp_waveform.replace(waveform_path); tmp_stationxml.replace(stationxml_path)
                status = "overwritten" if overwrite and existed_before else "redownloaded" if existed_before else "downloaded"
                cache_reason = reason
            except Exception as exc:
                for temporary in (tmp_waveform, tmp_stationxml):
                    if temporary.exists():
                        temporary.unlink()
                log_rows.append({**base, "request_starttime": str(request_start), "request_endtime": str(request_end),
                                 "status": "download_failed", "reason": str(exc), "cache_reason": cache_reason})
                continue

        successful = {**base, "origin_time": str(origin), "event_lon": event_lon, "event_lat": event_lat, "event_depth_km": event_depth_km,
                      "magnitude": magnitude, "min_radius_km": min_radius_km, "max_radius_km": max_radius_km,
                      "request_starttime": str(request_start), "request_endtime": str(request_end), "waveform_path": str(waveform_path),
                      "stationxml_path": str(stationxml_path), "download_status": status}
        successful_rows.append(successful); successful_station_ids.add(station_id)
        log_rows.append({**base, "request_starttime": str(request_start), "request_endtime": str(request_end),
                         "status": status, "reason": "", "cache_reason": cache_reason})

    downloaded = pd.DataFrame(successful_rows).sort_values("distance_km", ignore_index=True) if successful_rows else pd.DataFrame()
    if downloaded.empty:
        raise ValueError("No station waveforms were acquired successfully.")

    event_df = pd.DataFrame([{
        "event_id": event_id, "origin_time": str(origin), "event_lon": event_lon, "event_lat": event_lat, "event_depth_km": event_depth_km,
        "magnitude": magnitude, "min_radius_km": min_radius_km, "max_radius_km": max_radius_km,
        "radius_scale_factor": float(radius_scale_factor), "ground_level": bool(ground_level), "taup_model": str(taup_model),
        "t_before_s": final_t_before, "t_after_s": final_t_after, "fdsn_clients": ",".join(requested_client_labels),
    }])
    download_log = pd.DataFrame(log_rows)
    write_network_file(downloaded, root / "input" / "network.stn")
    event_df.to_csv(metadata_dir / "event.csv", index=False)
    downloaded.to_csv(metadata_dir / "stations_downloaded.csv", index=False)
    download_log.to_csv(metadata_dir / "download_log.csv", index=False)
    pd.DataFrame([waveform_window]).to_csv(metadata_dir / "waveform_window.csv", index=False)

    if plot:
        plot_waveform_section(downloaded, origin, figure_dir / "waveform_record_section_unfiltered.png", show=show)

    return event_df, downloaded, download_log, waveform_window


def get_waveform_window(
    event_depth_km: float,
    magnitude: float,
    *,
    max_distance_km: float | None = None,
    station_df: pd.DataFrame | str | Path | None = None,
    radius_scale_factor: float = 1.66,
    min_depth_km: float = 5.0,
    min_depth_multiplier: float = 0.5,
    max_depth_multiplier: float = 3.0,
    time_unc_s: float = 2.0,
    rupture_velocity_m_s: float = 1000.0,
    velocity_slowest_m_s: float = 1000.0,
    covariance: str = "noise",
    noise_factor: float = 4.0,
    edge_margin_s: float = 1.0,
    minimum_pre_event_s: float = 0.0,
) -> dict[str, Any]:
    """Compute a BayesISOLA-safe origin-centred waveform request window.

    Unlike the 0.1.2 helper, this calculation is independent of instantiated
    BayesISOLA ``inputs``/``grid`` objects and is therefore usable before any
    inversion is constructed. It reproduces the same equations later used by
    BayesISOLA's grid/time-window machinery:

    * the deepest trial source is ``catalog_depth * max_depth_multiplier``;
    * the shallow trial bound is ``max(min_depth_km, catalog_depth *
      min_depth_multiplier)`` (returned for transparency although only the deep
      bound enters the longest travel-time estimate);
    * BayesISOLA's rupture-length proxy is exactly ``sqrt(111 * 10**Mw)`` metres;
    * the source-time grid spans ``-time_unc_s`` to ``time_unc_s +
      rupture_length / rupture_velocity_m_s``. This temporal rupture term is
      always present in native BayesISOLA and is independent of the spatial
      ``add_rupture_length`` grid option;
    * ``t_max`` is the farthest hypocentral distance divided by
      ``velocity_slowest_m_s``;
    * ``covariance='noise'`` requests ``noise_factor`` processing lengths before
      origin (4.0 matches BayesISOLA's default noise slice), while ``'none'``
      requests only the negative source-time-shift coverage;
    * ``minimum_pre_event_s`` may impose an independent QC requirement such as
      the 20 s used by ``detect_mouse``.

    Distance can be supplied three ways. ``max_distance_km`` has highest
    precedence. Otherwise a supplied ``station_df`` must contain ``distance_km``
    and its farthest station is used. If neither is supplied, the planning value
    from :func:`get_max_radius(magnitude, radius_scale_factor)` is used.
    """
    event_depth_km = float(event_depth_km)
    magnitude = float(magnitude)
    if max_distance_km is not None:
        resolved_distance = float(max_distance_km)
        distance_source = "max_distance_km"
    elif station_df is not None:
        table = _coerce_station_df(station_df)
        if "distance_km" not in table.columns:
            raise KeyError("station_df must contain distance_km when max_distance_km is not supplied.")
        distances = pd.to_numeric(table["distance_km"], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(distances).all() or len(distances) == 0:
            raise ValueError("station_df['distance_km'] must contain finite values.")
        resolved_distance = float(np.max(distances))
        distance_source = "station_df"
    else:
        resolved_distance = float(get_max_radius(magnitude, scale_factor=float(radius_scale_factor)))
        distance_source = "magnitude_radius"

    grid_min_depth_km, grid_max_depth_km = _depth_bounds_km(
        event_depth_km, min_depth_km=min_depth_km, min_depth_multiplier=min_depth_multiplier,
        max_depth_multiplier=max_depth_multiplier,
    )
    rupture_length_m = _bayesisola_rupture_length_m(magnitude)
    rupture_velocity_m_s = float(rupture_velocity_m_s)
    time_unc_s = float(time_unc_s)
    if rupture_velocity_m_s <= 0:
        raise ValueError("rupture_velocity_m_s must be positive.")
    if time_unc_s < 0 or not np.isfinite(time_unc_s):
        raise ValueError("time_unc_s must be finite and non-negative.")
    shift_min_s = -time_unc_s
    shift_max_s = time_unc_s + rupture_length_m / rupture_velocity_m_s

    window = _waveform_window_from_bounds(
        resolved_distance, grid_max_depth_km, shift_min_s, shift_max_s,
        velocity_slowest_m_s=velocity_slowest_m_s, covariance=covariance, noise_factor=noise_factor,
        edge_margin_s=edge_margin_s, minimum_pre_event_s=minimum_pre_event_s,
    )
    window.update({
        "distance_source": distance_source,
        "grid_min_depth_km": grid_min_depth_km,
        "grid_max_depth_km": grid_max_depth_km,
        "min_depth_km": float(min_depth_km),
        "min_depth_multiplier": float(min_depth_multiplier),
        "max_depth_multiplier": float(max_depth_multiplier),
        "rupture_length_km": rupture_length_m / 1000.0,
        "rupture_velocity_m_s": rupture_velocity_m_s,
        "velocity_slowest_m_s": float(velocity_slowest_m_s),
        "covariance": str(covariance).lower().strip(),
    })
    return window


def load_streams_fdsnws_auto(
    inputs,
    grid,
    hosts,
    *,
    save_to: str | Path,
    velocity_slowest_m_s: float = 1000.0,
    noise: bool = True,
    noise_factor: float = 4.0,
    edge_margin_s: float = 1.0,
) -> dict[str, float]:
    """Backward-compatible wrapper around BayesISOLA's native FDSN loader.

    This utility remains for direct package testing. The main automated workflow
    uses :func:`get_mseed_stationxml` followed by :func:`load_streams_local` so
    FDSN and local runs share the identical downstream path.
    """
    if not inputs.stations:
        raise ValueError("No stations are loaded in inputs.")
    rupture_length = inputs.rupture_length if grid.add_rupture_length else 0.0
    depth_max_m = float(grid.grid_max_depth) if grid.grid_max_depth else float(inputs.event["depth"] + grid.depth_unc + rupture_length)
    shift_max = float(grid.grid_max_time) if grid.grid_max_time else float(grid.time_unc + inputs.rupture_length / grid.rupture_velocity)
    shift_min = float(grid.grid_min_time) if grid.grid_min_time else float(-grid.time_unc)
    max_distance_km = max(float(station["dist"]) for station in inputs.stations) / 1000.0
    window = _waveform_window_from_bounds(
        max_distance_km, depth_max_m / 1000.0, shift_min, shift_max,
        velocity_slowest_m_s=velocity_slowest_m_s, covariance="noise" if noise else "none",
        noise_factor=noise_factor, edge_margin_s=edge_margin_s, minimum_pre_event_s=0.0,
    )
    Path(save_to).mkdir(parents=True, exist_ok=True)
    inputs.load_streams_fdsnws(hosts, t_before=window["t_before_s"], t_after=window["t_after_s"], save_to=str(save_to))
    return window


def load_streams_local(
    inputs,
    station_df: pd.DataFrame | str | Path,
    *,
    t_before: float | None = None,
    t_after: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Populate ``inputs.data_raw`` from miniSEED/StationXML station records.

    The station table must identify network, station, location, channel prefix,
    the selected three-component family, and ``waveform_path``/
    ``stationxml_path``. This is exactly the table written by
    ``get_mseed_stationxml``. Each miniSEED is reduced to the explicit selected
    triplet, checked for gaps/common coverage, attached to its StationXML
    response, and reordered to BayesISOLA's required Z/N/E stream order. Z12 or
    123 installations are rotated to ZNE using StationXML orientation metadata,
    matching BayesISOLA's native local-file behaviour.

    When ``t_before``/``t_after`` are supplied, the common three-component
    interval must cover the automatic origin-centred window before the station is
    admitted. Stations that cannot be loaded are removed in the same spirit as
    BayesISOLA's native FDSN/file loaders; the returned load log records why.
    """
    from obspy import Stream, read, read_inventory
    from BayesISOLA.fileformats import attach_xml_paz

    table = _coerce_station_df(station_df)
    required = {"network", "station", "location", "channel_prefix", "waveform_path", "stationxml_path"}
    missing = required.difference(table.columns)
    if missing:
        raise KeyError(f"station_df is missing local-waveform columns: {sorted(missing)}")

    origin = inputs.event["t"]
    expected_start = origin - float(t_before) if t_before is not None else None
    expected_end = origin + float(t_after) if t_after is not None else None
    lookup = {}
    for row in table.to_dict("records"):
        key = (str(row["network"]), str(row["station"]), _normalize_location(row.get("location")), str(row["channel_prefix"]))
        lookup[key] = row

    inputs.data_raw = []; inputs.data_deltas = []
    retained_stations, loaded_rows, log_rows = [], [], []

    for station in list(inputs.stations):
        key = (str(station["network"]), str(station["code"]), _normalize_location(station.get("location")), str(station["channelcode"]))
        row = lookup.get(key)
        station_id = f"{key[0]}.{key[1]}.{key[2] or '--'}"
        if row is None:
            log_rows.append({"station_id": station_id, "status": "missing_station_metadata", "reason": "No matching station_df row."})
            continue

        try:
            waveform_path, stationxml_path = Path(row["waveform_path"]).expanduser(), Path(row["stationxml_path"]).expanduser()
            if not waveform_path.is_file() or not stationxml_path.is_file():
                raise FileNotFoundError(f"Missing waveform or StationXML file: {waveform_path}, {stationxml_path}")
            available = read(str(waveform_path), format="MSEED").select(network=key[0], station=key[1], location=key[2], channel=f"{key[3]}?")
            scheme, selected_channels = _row_component_selection(row)
            selected = Stream()
            for channel in selected_channels:
                matching = available.select(channel=channel)
                if len(matching) == 0:
                    raise ValueError(f"Selected input channel {channel} is absent from {waveform_path}.")
                selected += matching
            gaps = [gap for gap in selected.get_gaps() if gap[-1] > 0]
            if gaps:
                raise ValueError(f"Internal waveform gaps: {gaps}")
            selected.merge(method=0)
            if len(selected) != 3:
                raise ValueError(f"Selected channel set does not reduce to exactly three traces: {[trace.id for trace in selected]}")
            common_start = max(trace.stats.starttime for trace in selected)
            common_end = min(trace.stats.endtime for trace in selected)
            if common_end <= common_start:
                raise ValueError("The selected components have no common recorded interval.")
            tolerance = max(float(trace.stats.delta) for trace in selected) + 1e-6
            if expected_start is not None and common_start > expected_start + tolerance:
                raise ValueError(f"Record starts at {common_start}, after required start {expected_start}.")
            if expected_end is not None and common_end < expected_end - tolerance:
                raise ValueError(f"Record ends at {common_end}, before required end {expected_end}.")
            selected.trim(common_start, common_end, pad=False)

            inventory = read_inventory(str(stationxml_path), format="STATIONXML")
            attach_xml_paz(selected, inventory=inventory)
            if scheme != "ZNE":
                selected.rotate(method="->ZNE", inventory=inventory, components=(scheme,))

            zne = Stream()
            for component in "ZNE":
                matches = [trace for trace in selected if str(trace.stats.channel).upper().endswith(component)]
                if len(matches) != 1:
                    raise ValueError(f"Expected one {component} component after orientation handling; found {len(matches)}.")
                zne += matches[0]
            deltas = np.asarray([float(trace.stats.delta) for trace in zne])
            if not np.allclose(deltas, deltas[0], atol=1e-9, rtol=0.0):
                raise ValueError(f"ZNE sample intervals differ: {deltas.tolist()}")

            station["useZ"] = station["useN"] = station["useE"] = True
            station["accelerograph"] = False
            retained_stations.append(station); inputs.data_raw.append(zne)
            if float(deltas[0]) not in inputs.data_deltas:
                inputs.data_deltas.append(float(deltas[0]))
            loaded_rows.append(row)
            log_rows.append({"station_id": station_id, "status": "loaded", "reason": "", "waveform_path": str(waveform_path), "stationxml_path": str(stationxml_path)})
        except Exception as exc:
            inputs.log(f"{key[0]}:{key[1]}: Local waveform loading unsuccessful. Removing station from further processing: {exc}")
            log_rows.append({"station_id": station_id, "status": "load_failed", "reason": str(exc)})

    inputs.stations = retained_stations
    inputs.create_station_index()
    inputs.data_are_corrected = False
    inputs.logtext["data"] = f"Loaded {len(inputs.data_raw)} local miniSEED/StationXML station streams."
    inputs.check_a_station_present()
    inputs.write_stations()

    loaded_df = pd.DataFrame(loaded_rows)
    if not loaded_df.empty and "distance_km" in loaded_df.columns:
        loaded_df = loaded_df.sort_values("distance_km", ignore_index=True)
    return loaded_df, pd.DataFrame(log_rows)


def plot_waveform_section(
    station_df: pd.DataFrame | str | Path,
    origin_time,
    output_file: str | Path,
    *,
    show: bool = False,
    water_level: float | None = 20.0,
    amplitude_scale: float = 1.0,
) -> Path:
    """Plot the acquired/local waveforms before BayesISOLA bandpass filtering.

    The function reads the explicit miniSEED/StationXML paths in ``station_df``,
    reproduces BayesISOLA's input orientation convention, and applies response
    removal to plotting copies as velocity. The default ``water_level=20``
    matches ``BayesISOLA.process_data.correct_data``. Z12/123 installations are
    rotated to ZNE with StationXML metadata. No filtering, decimation or file
    modification is performed, so the figure is a direct QC view of the data
    that will enter BayesISOLA before its inversion-band processing.

    Every trace is normalized independently for display and offset by epicentral
    distance. The event origin and any finite theoretical P/S arrivals stored in
    the station table are marked explicitly. Plot normalization is diagnostic
    only and has no effect on inversion amplitudes.
    """
    from obspy import Stream, UTCDateTime, read, read_inventory
    from BayesISOLA.fileformats import attach_xml_paz
    import matplotlib.pyplot as plt

    table = _coerce_station_df(station_df)
    required = {"network", "station", "location", "channel_prefix", "waveform_path", "stationxml_path", "distance_km"}
    missing = required.difference(table.columns)
    if missing:
        raise KeyError(f"station_df is missing waveform-section columns: {sorted(missing)}")
    amplitude_scale = float(amplitude_scale)
    if not np.isfinite(amplitude_scale) or amplitude_scale <= 0:
        raise ValueError("amplitude_scale must be finite and positive.")

    origin = UTCDateTime(origin_time)
    prepared: list[tuple[dict[str, Any], Any]] = []
    for row in table.to_dict("records"):
        waveform_path, stationxml_path = Path(row["waveform_path"]).expanduser(), Path(row["stationxml_path"]).expanduser()
        inventory = read_inventory(str(stationxml_path), format="STATIONXML")
        location = _normalize_location(row.get("location"))
        available = read(str(waveform_path), format="MSEED").select(network=str(row["network"]), station=str(row["station"]),
                                                                    location=location, channel=f"{row['channel_prefix']}?")
        scheme, selected_channels = _row_component_selection(row)
        selected = Stream()
        for channel in selected_channels:
            matches = available.select(channel=channel)
            if len(matches) == 0:
                raise ValueError(f"Selected input channel {channel} is absent from {waveform_path}.")
            selected += matches
        gaps = [gap for gap in selected.get_gaps() if gap[-1] > 0]
        if gaps:
            raise ValueError(f"Internal waveform gaps for {row.get('station_id', row['station'])}: {gaps}")
        selected.merge(method=0)
        if len(selected) != 3:
            raise ValueError(f"Selected channels do not reduce to exactly three traces for {row.get('station_id', row['station'])}.")
        common_start = max(trace.stats.starttime for trace in selected)
        common_end = min(trace.stats.endtime for trace in selected)
        selected.trim(common_start, common_end, pad=False)

        attach_xml_paz(selected, inventory=inventory)
        if scheme != "ZNE":
            selected.rotate(method="->ZNE", inventory=inventory, components=(scheme,))
        zne = Stream()
        for component in "ZNE":
            matches = [trace for trace in selected if str(trace.stats.channel).upper().endswith(component)]
            if len(matches) != 1:
                raise ValueError(f"Expected one {component} component after orientation handling; found {len(matches)}.")
            trace = matches[0].copy()
            trace.detrend(type="demean")
            kwargs = {"output": "VEL"}
            if water_level is not None:
                kwargs["water_level"] = float(water_level)
            trace.remove_response(**kwargs)
            zne += trace
        prepared.append((row, zne))

    output_file = Path(output_file).expanduser()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 9), dpi=200, sharey=True, constrained_layout=True)
    for row, stream in prepared:
        distance_km = float(row["distance_km"])
        p_time, s_time = float(row.get("p_arrival_s", np.nan)), float(row.get("s_arrival_s", np.nan))
        for axis, component, trace in zip(axes, "ZNE", stream):
            amplitude = float(np.nanmax(np.abs(trace.data))) if len(trace.data) else 0.0
            values = amplitude_scale * np.asarray(trace.data) / amplitude if amplitude > 0 else np.asarray(trace.data)
            time = trace.times() + float(trace.stats.starttime - origin)
            axis.plot(time, values + distance_km, linewidth=0.7)
            if np.isfinite(p_time):
                axis.plot(p_time, distance_km, marker="|", markersize=5, linestyle="none")
            if np.isfinite(s_time):
                axis.plot(s_time, distance_km, marker="|", markersize=5, linestyle="none")
    for axis, component in zip(axes, "ZNE"):
        axis.axvline(0.0, linestyle="--", linewidth=0.8)
        axis.set_title(component)
        axis.set_xlabel("Time relative to event origin (s)")
        axis.grid(axis="x", alpha=0.15)
    axes[0].set_ylabel("Epicentral distance (km)")
    fig.suptitle("Response-corrected ZNE waveforms before BayesISOLA filtering")
    fig.savefig(output_file, bbox_inches="tight")
    if show:
        try:
            from IPython import get_ipython
            from IPython.display import display

            if get_ipython() is not None:
                display(fig)
            else:
                plt.show()
        except ImportError:
            plt.show()

    plt.close(fig)
    return output_file


def plot_station_section(
    data,
    station_df: pd.DataFrame | str | Path | None,
    output_file: str | Path,
    *,
    show: bool = False,
    amplitude_scale: float = 1.0,
) -> Path:
    """Plot BayesISOLA's inversion-ready filtered Z/N/E record section.

    When available, the zero/nearest-zero member of ``data.data_shifts`` is used
    because those streams have undergone BayesISOLA's final inversion filter and
    working-rate decimation. ``data.data`` is used only as a fallback. Traces are
    independently normalized for visualization and offset by epicentral distance;
    no inversion data are modified. Finite P/S arrivals from ``station_df`` are
    overlaid as timing references.
    """
    import matplotlib.pyplot as plt

    amplitude_scale = float(amplitude_scale)
    if not np.isfinite(amplitude_scale) or amplitude_scale <= 0:
        raise ValueError("amplitude_scale must be finite and positive.")
    metadata = _coerce_station_df(station_df) if station_df is not None else pd.DataFrame()
    metadata_lookup = {}
    if not metadata.empty:
        for row in metadata.to_dict("records"):
            metadata_lookup[(str(row["network"]), str(row["station"]), _normalize_location(row.get("location")))] = row

    streams = data.data
    if getattr(data, "data_shifts", None) and getattr(data, "shifts", None):
        zero_index = int(np.argmin(np.abs(np.asarray(data.shifts, dtype=float))))
        streams = data.data_shifts[zero_index]

    output_file = Path(output_file).expanduser()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 9), dpi=200, sharey=True, constrained_layout=True)
    origin = data.d.event["t"]

    for station, stream in zip(data.d.stations, streams):
        distance_km = float(station["dist"]) / 1000.0
        row = metadata_lookup.get((str(station["network"]), str(station["code"]), _normalize_location(station.get("location"))), {})
        p_time, s_time = float(row.get("p_arrival_s", np.nan)), float(row.get("s_arrival_s", np.nan))
        for axis, component, trace in zip(axes, "ZNE", stream):
            amplitude = float(np.nanmax(np.abs(trace.data))) if len(trace.data) else 0.0
            values = amplitude_scale * np.asarray(trace.data) / amplitude if amplitude > 0 else np.asarray(trace.data)
            time = trace.times() + float(trace.stats.starttime - origin)
            axis.plot(time, values + distance_km, linewidth=0.7)
            if np.isfinite(p_time):
                axis.plot(p_time, distance_km, marker="|", markersize=5, linestyle="none")
            if np.isfinite(s_time):
                axis.plot(s_time, distance_km, marker="|", markersize=5, linestyle="none")
    for axis, component in zip(axes, "ZNE"):
        axis.axvline(0.0, linestyle="--", linewidth=0.8)
        axis.set_title(component)
        axis.set_xlabel("Time relative to event origin (s)")
        axis.grid(axis="x", alpha=0.15)
    axes[0].set_ylabel("Epicentral distance (km)")
    fig.suptitle("BayesISOLA inversion-ready filtered ZNE waveforms")
    fig.savefig(output_file, bbox_inches="tight")
    if show:
        try:
            from IPython import get_ipython
            from IPython.display import display

            if get_ipython() is not None:
                display(fig)
            else:
                plt.show()
        except ImportError:
            plt.show()

    plt.close(fig)
    return output_file

def _get_gf_helpers_module():
    """Import the packaged ``BayesISOLA.gf_helpers`` module only."""
    from BayesISOLA import gf_helpers as module
    return module


def _prepare_crust_file(
    output_file: Path,
    *,
    event_lon: float,
    event_lat: float,
    crust_file: str | Path | None,
    velocity_model,
    profile_crs: str,
    depth_col: str,
    vp_col: str,
    vs_col: str,
    density_col: str,
    qp_col: str,
    qs_col: str,
    surface_depth_km: float,
) -> tuple[Path, pd.DataFrame | None, pd.DataFrame | None]:
    """Prepare the six-column BayesISOLA/Axitra ``crustal.dat`` model.

    Exactly one model source is used:

    * ``crust_file``: copied verbatim and handed to ``inputs.read_crust``;
    * 1-D ``pandas.DataFrame``: interpreted directly as one depth profile;
    * ``gf_helpers`` velocity-grid object: sampled at the event longitude/latitude
      through ``extract_profile`` before layer conversion.

    The DataFrame/grid dispatch is automatic. For a 1-D DataFrame, common column
    aliases are recognized when the requested default name is absent. The model
    must still supply depth, Vp, Vs, density, Qp and Qs because Axitra reads all six
    values for every layer; the helper does not invent attenuation or density.
    ``gf_helpers.profile_to_pyfk_layers`` is reused only for its generic layered
    profile conversion and compression; the final file written here is the native
    BayesISOLA top-depth/Vp/Vs/rho/Qp/Qs format.
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if crust_file is not None and velocity_model is not None:
        raise ValueError("Supply crust_file or velocity_model, not both.")
    if crust_file is not None:
        source = Path(crust_file).expanduser()
        if not source.is_file():
            raise FileNotFoundError(source)
        if source.resolve() != output_file.resolve():
            shutil.copyfile(source, output_file)
        return output_file, None, None
    if velocity_model is None:
        raise ValueError("Supply either crust_file or velocity_model.")

    profile_to_pyfk_layers = _get_gf_helpers_module().profile_to_pyfk_layers

    value_names = [vp_col, vs_col, density_col, qp_col, qs_col]
    if isinstance(velocity_model, pd.DataFrame):
        profile = velocity_model.copy()
        aliases = {
            "depth": (depth_col, "Depth_km", "depth_km", "Depth", "depth", "Depth(km_BSL)", "depth(km)"),
            "vp": (vp_col, "Vp", "vp", "VP", "Vp_km_s", "vp_km_s"),
            "vs": (vs_col, "Vs", "vs", "VS", "Vs_km_s", "vs_km_s"),
            "density": (density_col, "Density", "density", "Rho", "rho", "Density_g_cm3", "rho_g_cm3"),
            "qp": (qp_col, "Qp", "qp", "QP"),
            "qs": (qs_col, "Qs", "qs", "QS"),
        }
        def pick(candidates, label):
            for candidate in dict.fromkeys(candidates):
                if candidate in profile.columns:
                    return candidate
            raise KeyError(f"1-D velocity_model is missing {label}; tried columns {list(dict.fromkeys(candidates))}.")
        depth_col = pick(aliases["depth"], "depth")
        vp_col = pick(aliases["vp"], "Vp")
        vs_col = pick(aliases["vs"], "Vs")
        density_col = pick(aliases["density"], "density")
        qp_col = pick(aliases["qp"], "Qp")
        qs_col = pick(aliases["qs"], "Qs")
        model_description = "1-D velocity profile"
    elif callable(getattr(velocity_model, "extract_profile", None)):
        profile = velocity_model.extract_profile(float(event_lon), float(event_lat), crs=profile_crs, value_names=value_names)
        depth_col = "Depth_km"
        model_description = "event-centred profile"
    else:
        raise TypeError("velocity_model must be a 1-D pandas DataFrame or a gf_helpers velocity-grid object with extract_profile().")

    layers = profile_to_pyfk_layers(
        profile, depth_col=depth_col, vp_col=vp_col, vs_col=vs_col, density_col=density_col,
        qs_col=qs_col, qp_col=qp_col, surface_depth_km=float(surface_depth_km), compress=True,
    )
    required = ["Top_depth_km", "Vp_km_s", "Vs_km_s", "Density_g_cm3", "Qp", "Qs"]
    missing = [column for column in required if column not in layers.columns]
    if missing:
        raise KeyError(f"Layer conversion did not produce BayesISOLA-required columns: {missing}")
    table = layers[required]
    with output_file.open("w", encoding="utf-8", newline="\n") as f:
        f.write(f"Crustal model                {model_description}\n")
        f.write("number of layers\n")
        f.write(f"{len(table)}\n")
        f.write("Parameters of the layers\n")
        f.write("depth of layer top(km)   Vp(km/s)   Vs(km/s)   Rho(g/cm**3)   Qp   Qs\n")
        table.to_csv(f, sep=" ", header=False, index=False, float_format="%.5f")
    return output_file, profile, layers


def _normalize_gf_options(
    gf_source: str,
    gf_options: Mapping[str, Any] | None,
    *,
    surface_depth_km: float,
) -> tuple[str, dict[str, Any]]:
    """Normalize backend-specific Green's-function options.

    ``gf_options=None`` is intentionally meaningful for ``gf_source='axitra'``:
    it preserves the historical one-model workflow in which a supplied 1-D
    profile is used directly or a 3-D velocity grid is sampled once at the
    catalogue epicentre. Station/path-dependent Axitra models are opt-in.

    ``gf_options`` contains only backend-specific physical/numerical/service
    configuration. Green's-function reuse is controlled solely by
    ``use_precalculated_Green`` in :func:`run_auto_cmt`; in particular, Syngine
    ``overwrite`` is not accepted here from version 0.1.8 onward.
    """
    source = str(gf_source).strip().lower()
    if source not in {"axitra", "syngine"}:
        raise ValueError("gf_source must be 'axitra' or 'syngine'.")
    if gf_options is None:
        raw = {}
    elif isinstance(gf_options, Mapping):
        raw = dict(gf_options)
    else:
        raise TypeError("gf_options must be a mapping or None.")

    if source == "axitra":
        if "grid" in raw and "grid_mode" in raw:
            raise ValueError("Use only one of gf_options['grid'] or ['grid_mode'].")
        grid_mode = raw.pop("grid", raw.pop("grid_mode", None))
        if grid_mode is not None:
            grid_mode = str(grid_mode).strip().lower()
            if grid_mode not in {"path", "station"}:
                raise ValueError("Axitra gf_options['grid'] must be 'path', 'station' or None.")

        allowed = {"path_spacing_km", "path_profile", "surface_depth_km", "max_depth_km", "compress"}
        unknown = set(raw).difference(allowed)
        if unknown:
            raise KeyError(f"Unknown Axitra gf_options: {sorted(unknown)}")
        if grid_mode != "path" and ({"path_spacing_km", "path_profile"} & set(raw)):
            raise ValueError("path_spacing_km/path_profile are valid only with gf_options['grid']='path'.")

        options = {
            "grid": grid_mode,
            "surface_depth_km": float(raw.get("surface_depth_km", surface_depth_km)),
            "max_depth_km": None if raw.get("max_depth_km") is None else float(raw["max_depth_km"]),
            "compress": bool(raw.get("compress", True)),
        }
        if grid_mode == "path":
            options["path_spacing_km"] = float(raw.get("path_spacing_km", 2.0))
            profile_key = str(raw.get("path_profile", "mean")).strip().lower()
            profile_aliases = {
                "mean": "mean", "median": "median",
                "5": "p05", "05": "p05", "5th": "p05", "5%": "p05", "p5": "p05", "p05": "p05",
                "95": "p95", "95th": "p95", "95%": "p95", "p95": "p95",
            }
            if options["path_spacing_km"] <= 0:
                raise ValueError("gf_options['path_spacing_km'] must be positive.")
            if profile_key not in profile_aliases:
                raise ValueError("gf_options['path_profile'] must select mean, median, p05 or p95.")
            options["path_profile"] = profile_aliases[profile_key]
        return source, options

    if "overwrite" in raw:
        raise ValueError(
            "Syngine gf_options['overwrite'] was removed in bayesisola_helpers 0.1.8. "
            "Use the backend-independent use_precalculated_Green=False, True or 'auto' "
            "cache policy instead."
        )
    allowed = {
        "model", "output_dir", "url", "syngine_dt", "kernelwidth", "timeout",
        "request_padding_s", "max_workers", "progress",
    }
    unknown = set(raw).difference(allowed)
    if unknown:
        raise KeyError(f"Unknown Syngine gf_options: {sorted(unknown)}")
    options = {
        "model": str(raw.get("model", "ak135f_5s")).strip(),
        "output_dir": raw.get("output_dir"),
        "url": raw.get("url"),
        "syngine_dt": raw.get("syngine_dt"),
        "kernelwidth": int(raw.get("kernelwidth", 12)),
        "timeout": float(raw.get("timeout", 120.0)),
        "request_padding_s": float(raw.get("request_padding_s", 60.0)),
        "max_workers": int(raw.get("max_workers", 4)),
        "progress": bool(raw.get("progress", True)),
    }
    if not options["model"]:
        raise ValueError("gf_options['model'] cannot be empty for Syngine.")
    if options["syngine_dt"] is not None:
        options["syngine_dt"] = float(options["syngine_dt"])
        if options["syngine_dt"] <= 0:
            raise ValueError("gf_options['syngine_dt'] must be positive or None.")
    if options["kernelwidth"] < 1 or options["timeout"] <= 0 or options["request_padding_s"] < 0 or options["max_workers"] < 1:
        raise ValueError("Syngine kernelwidth/max_workers must be positive, timeout > 0 and request_padding_s >= 0.")
    return source, options


def _axitra_table_from_layers(layers: pd.DataFrame) -> pd.DataFrame:
    """Return the six columns required by BayesISOLA/Axitra."""
    required = ["Top_depth_km", "Vp_km_s", "Vs_km_s", "Density_g_cm3", "Qp", "Qs"]
    missing = [column for column in required if column not in layers.columns]
    if missing:
        raise KeyError(f"Axitra requires explicit density/Q columns; missing {missing}.")
    table = layers.loc[:, required].astype(float).copy()
    if not np.isfinite(table.to_numpy()).all():
        raise ValueError("Axitra layered model contains non-finite values.")
    if (table[["Vp_km_s", "Density_g_cm3", "Qp", "Qs"]] <= 0).any().any() or (table["Vs_km_s"] < 0).any():
        raise ValueError("Axitra layered model contains invalid velocity/density/Q values.")
    return table


def _write_axitra_crust_model(path: Path, layers: pd.DataFrame, description: str) -> Path:
    """Write one native BayesISOLA/Axitra layered model file."""
    table = _axitra_table_from_layers(layers)
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(f"Crustal model                {description}\n")
        f.write("number of layers\n")
        f.write(f"{len(table)}\n")
        f.write("Parameters of the layers\n")
        f.write("depth of layer top(km)   Vp(km/s)   Vs(km/s)   Rho(g/cm**3)   Qp   Qs\n")
        table.to_csv(f, sep=" ", header=False, index=False, float_format="%.5f")
    return path


def _axitra_model_filename(base: Path, model: str) -> Path:
    """Return ``crustal-<model>.dat`` using BayesISOLA's native naming rule."""
    base = Path(base)
    return base.with_name(f"{base.stem}-{model}{base.suffix}") if model else base


def _layers_from_pyfk_array(model_array: np.ndarray) -> pd.DataFrame:
    """Convert a six-column gf_helpers/pyFK model array back to named layers."""
    array = np.asarray(model_array, dtype=float)
    if array.ndim != 2 or array.shape[1] != 6:
        raise ValueError(
            "Station/path Axitra models require explicit density, Qs and Qp so "
            "gf_helpers must produce a six-column model array."
        )
    thickness = array[:, 0]
    if len(thickness) < 2 or np.any(thickness[:-1] <= 0) or not np.isclose(thickness[-1], 0.0):
        raise ValueError("Invalid finite-layer/half-space thickness convention.")
    tops = np.r_[0.0, np.cumsum(thickness[:-1])]
    return pd.DataFrame({
        "Top_depth_km": tops,
        "Thickness_km": thickness,
        "Vs_km_s": array[:, 1],
        "Vp_km_s": array[:, 2],
        "Density_g_cm3": array[:, 3],
        "Qs": array[:, 4],
        "Qp": array[:, 5],
    })


def _axitra_model_signature(layers: pd.DataFrame) -> str:
    """Hash the numerical model as it will actually be written to Axitra."""
    table = _axitra_table_from_layers(layers)
    buffer = io.StringIO()
    table.to_csv(buffer, sep=" ", header=False, index=False, float_format="%.5f")
    return hashlib.sha256(buffer.getvalue().encode()).hexdigest()[:12]


def _prepare_axitra_station_models(
    station_df: pd.DataFrame,
    *,
    event_lon: float,
    event_lat: float,
    velocity_model,
    output_file: Path,
    options: Mapping[str, Any],
    vp_col: str,
    vs_col: str,
    density_col: str,
    qp_col: str,
    qs_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create native station-dependent Axitra models and station model tags.

    ``grid='station'`` extracts a vertical profile at each authoritative receiver
    coordinate. ``grid='path'`` samples the catalogue-origin-to-receiver path and
    collapses it depth-by-depth using ``path_profile``. BayesISOLA associates a
    crust model with a receiver, not with each source grid point; consequently a
    path model is fixed for that station across the complete centroid grid. This
    is the native station-dependent Axitra parameterization and is intentionally
    distinct from a fully 3-D source-path solver such as a future SW4 backend.
    Native Axitra evaluates the complete source grid once per unique receiver-model
    group, so station/path modes can be substantially more expensive than the
    default single-model calculation; exact duplicate layered models are grouped.
    """
    gf_helpers_module = _get_gf_helpers_module()
    get_profile_from_path = gf_helpers_module.get_profile_from_path
    profile_to_pyfk_layers = gf_helpers_module.profile_to_pyfk_layers

    if not callable(getattr(velocity_model, "extract_profile", None)):
        raise TypeError(
            "Axitra gf_options['grid'] requires a gf_helpers velocity-grid object "
            "with extract_profile(); a single 1-D profile cannot vary by station/path."
        )

    table = _coerce_station_df(station_df).copy()
    mode = str(options["grid"])
    value_names = [vp_col, vs_col, density_col, qp_col, qs_col]
    model_layers: dict[str, pd.DataFrame] = {}
    model_rows: list[dict[str, Any]] = []
    model_ids: list[str] = []

    for row in table.to_dict("records"):
        station_lon = float(row["station_lon"])
        station_lat = float(row["station_lat"])
        path_length_km = np.nan

        if mode == "station":
            profile = velocity_model.extract_profile(
                station_lon, station_lat, crs="EPSG:4326", value_names=value_names,
            )
            layers = profile_to_pyfk_layers(
                profile, depth_col="Depth_km", vp_col=vp_col, vs_col=vs_col,
                density_col=density_col, qs_col=qs_col, qp_col=qp_col,
                surface_depth_km=float(options["surface_depth_km"]),
                max_depth_km=options["max_depth_km"], compress=bool(options["compress"]),
            )
        else:
            _, summary, model_array = get_profile_from_path(
                velocity_model,
                float(event_lon), float(event_lat), station_lon, station_lat,
                crs="EPSG:4326", spacing_km=float(options["path_spacing_km"]),
                profile=str(options["path_profile"]), vp_name=vp_col, vs_name=vs_col,
                density_name=density_col, qs_name=qs_col, qp_name=qp_col,
                surface_depth_km=float(options["surface_depth_km"]),
                max_depth_km=options["max_depth_km"], compress=bool(options["compress"]),
            )
            layers = _layers_from_pyfk_array(model_array)
            path_length_km = float(summary.attrs.get("path_length_km", np.nan))

        signature = _axitra_model_signature(layers)
        model = f"m{signature}"
        model_layers.setdefault(model, layers)
        model_ids.append(model)
        model_rows.append({
            "station_id": row.get("station_id", f"{row['network']}.{row['station']}.{_normalize_location(row.get('location')) or '--'}"),
            "network": row["network"], "station": row["station"], "location": _normalize_location(row.get("location")),
            "station_lat": station_lat, "station_lon": station_lon,
            "gf_model": model, "model_mode": mode,
            "path_length_km": path_length_km, "n_layers": int(len(layers)),
            "model_signature": signature,
        })

    table["gf_model"] = model_ids
    for model, layers in model_layers.items():
        model_path = _axitra_model_filename(output_file, model)
        _write_axitra_crust_model(model_path, layers, f"Axitra {mode} model {model}")
        for record in model_rows:
            if record["gf_model"] == model:
                record["crust_file"] = str(model_path)

    return table, pd.DataFrame(model_rows)


def _prune_inactive_bayesisola_models(inputs) -> None:
    """Drop crust-model groups whose stations failed waveform loading."""
    active = {str(station.get("model", "")) for station in inputs.stations}
    inputs.models = {model: 0 for model in inputs.models if model in active}
    inputs.write_stations()


def _axitra_green_state(data) -> dict[str, tuple[int, int]]:
    """Snapshot Axitra elementary-seismogram metadata for reuse diagnostics."""
    return {
        path.name: (path.stat().st_mtime_ns, path.stat().st_size)
        for path in Path(data.d.green_dir).glob("elemse*.txt") if path.is_file()
    }


def _axitra_expected_grdat(data, model: str) -> str:
    return (
        "&input\n"
        f"nc=99\nnfreq={int(data.freq)}\ntl={float(data.tl):1.2f}\naw=0.5\n"
        f"nr={int(data.d.models[model])}\nns=1\nxl={float(data.xl):1.1f}\n"
        "ikmax=100000\nuconv=0.1E-06\nfref=1.\n/end\n"
    )


def _axitra_multimodel_cache_valid(data) -> bool:
    """Verify model-specific Axitra cache files without BayesISOLA's base-model shortcut."""
    from BayesISOLA._paths import green_path

    soutype = Path(green_path(data.d.green_dir, "soutype.dat"))
    if not soutype.is_file():
        return False
    txt_soutype = soutype.read_text().strip().replace("\n", "_")

    for model in data.d.models:
        suffix = f"-{model}" if model else ""
        grdat = Path(green_path(data.d.green_dir, f"grdat{suffix}.hed"))
        crust = Path(green_path(data.d.green_dir, f"crustal{suffix}.dat"))
        station = Path(green_path(data.d.green_dir, f"station{suffix}.dat"))
        if not grdat.is_file() or not crust.is_file() or not station.is_file():
            return False
        if grdat.read_text() != _axitra_expected_grdat(data, model):
            return False
        md5_crust = hashlib.md5(crust.read_bytes()).hexdigest()
        md5_station = hashlib.md5(station.read_bytes()).hexdigest()

        for i, gp in enumerate(data.grid.grid):
            point_id = str(i).zfill(4) + suffix
            elemse = Path(green_path(data.d.green_dir, f"elemse{point_id}.dat"))
            meta = Path(green_path(data.d.green_dir, f"elemse{point_id}.txt"))
            if not elemse.is_file() or elemse.stat().st_size == 0 or not meta.is_file():
                return False
            expected = (
                f"{gp['x']/1e3:1.3f} {gp['y']/1e3:1.3f} {gp['z']/1e3:1.3f} "
                f"{md5_crust} {md5_station} {txt_soutype}"
            )
            if meta.read_text() != expected:
                return False
    return True


def _rewrite_axitra_multimodel_metadata(data) -> None:
    """Write model-specific hashes after native Axitra finishes each model group."""
    from BayesISOLA._paths import green_path

    txt_soutype = Path(green_path(data.d.green_dir, "soutype.dat")).read_text().strip().replace("\n", "_")
    for model in data.d.models:
        suffix = f"-{model}" if model else ""
        crust = Path(green_path(data.d.green_dir, f"crustal{suffix}.dat"))
        station = Path(green_path(data.d.green_dir, f"station{suffix}.dat"))
        md5_crust = hashlib.md5(crust.read_bytes()).hexdigest()
        md5_station = hashlib.md5(station.read_bytes()).hexdigest()
        for i, gp in enumerate(data.grid.grid):
            point_id = str(i).zfill(4) + suffix
            meta = Path(green_path(data.d.green_dir, f"elemse{point_id}.txt"))
            if not meta.is_file():
                continue
            meta.write_text(
                f"{gp['x']/1e3:1.3f} {gp['y']/1e3:1.3f} {gp['z']/1e3:1.3f} "
                f"{md5_crust} {md5_station} {txt_soutype}"
            )


def _ensure_axitra_metadata_base_files(data) -> None:
    """Provide base files required by the legacy Axitra metadata writer.

    The Fortran calculation itself uses the model-suffixed files. The legacy
    Python wrapper hashes unsuffixed ``crustal.dat``/``station.dat`` even for a
    named model, so a compatibility copy is required during calculation. The
    metadata are immediately rewritten with the correct model-specific hashes.
    """
    from BayesISOLA._paths import green_path

    if "" in data.d.models:
        return
    models = list(data.d.models)
    if not models:
        raise ValueError("No active Axitra crust models remain after waveform loading.")
    model = models[0]
    shutil.copyfile(green_path(data.d.green_dir, f"crustal-{model}.dat"), green_path(data.d.green_dir, "crustal.dat"))
    shutil.copyfile(green_path(data.d.green_dir, f"station-{model}.dat"), green_path(data.d.green_dir, "station.dat"))


def _calculate_or_verify_axitra_multimodel(data, use_precalculated_Green) -> bool:
    """Calculate/reuse station-dependent Axitra GFs with model-aware cache checks.

    Returns ``True`` only when the complete compatible cache was reused.
    """
    mode = use_precalculated_Green
    if mode not in {False, True, "auto"}:
        raise ValueError("use_precalculated_Green must be False, True or 'auto'.")

    if mode is not False and _axitra_multimodel_cache_valid(data):
        data.log("Using verified model-specific pre-calculated Green's functions.")
        return True
    if mode is True:
        raise ValueError("Model-specific pre-calculated Axitra Green's functions are missing or incompatible.")

    data.write_Greens_parameters()
    _ensure_axitra_metadata_base_files(data)
    data.calculate_Green()
    _rewrite_axitra_multimodel_metadata(data)
    return False


def _syngine_cache_record(
    syngine_module,
    query,
    *,
    output_root: Path,
    model: str,
    source_lat: float,
    source_lon: float,
    source_depth_m: float,
    origin,
    end,
    target_npts: int,
    target_sampling_rate: float,
    source_time_function: str,
) -> dict[str, Any]:
    """Return the exact corrected-Syngine cache signature and validity state.

    This is the non-mutating counterpart of ``BayesISOLA.syngine.generate_query``
    ``do_query_simple``. It deliberately uses the corrected backend's own station
    geometry, signature and cache validators so ``use_precalculated_Green=True``
    can require a compatible cache *before* any network request or file rewrite.
    The payload mirrors the backend's v0.2 cache contract; if those internal cache
    primitives are unavailable, strict-cache mode fails explicitly rather than
    silently degrading to ``'auto'`` behaviour.
    """
    required = (
        "_CACHE_SCHEMA", "_BAYESISOLA_SOURCES_USE", "_station_contexts",
        "_short_signature", "_cache_is_valid",
    )
    missing = [name for name in required if not hasattr(syngine_module, name)]
    if missing:
        raise RuntimeError(
            "The installed BayesISOLA.syngine backend does not expose the corrected "
            f"cache contract required by use_precalculated_Green=True: missing {missing}."
        )

    stf = str(source_time_function).strip().lower()
    if stf not in {"step", "heaviside", "step in displacement"}:
        raise NotImplementedError(
            "The Syngine GF backend currently supports BayesISOLA's step-in-"
            "displacement source-time function only."
        )

    source_lat = float(source_lat)
    source_lon = float(source_lon)
    source_depth_m = int(round(float(source_depth_m)))
    requested_end = float(end - origin)
    target_npts = int(target_npts)
    target_sampling_rate = float(target_sampling_rate)
    target_end = (target_npts - 1) / target_sampling_rate
    request_end_s = max(requested_end, target_end) + float(query.request_padding_s)
    contexts = syngine_module._station_contexts(
        query.bulk, source_lat=source_lat, source_lon=source_lon
    )
    payload = {
        "schema": syngine_module._CACHE_SCHEMA,
        "version": syngine_module.__version__,
        "model": str(model),
        "url": query.url,
        "source_lat": source_lat,
        "source_lon": source_lon,
        "source_depth_m": source_depth_m,
        "origin": str(origin),
        "requested_end_s": requested_end,
        "request_end_s": request_end_s,
        "syngine_dt": query.syngine_dt,
        "kernelwidth": query.kernelwidth,
        "units": "velocity",
        "source_time_function": "step in displacement",
        "tensor_coordinates": "NED_x_north_y_east_z_down",
        "tensor_rotation": "alpha_deg=(180-azimuth_deg)%360",
        "receiver_rotation": "ObsPy RT->NE using source-to-station back azimuth",
        "bayesisola_sources_USE": [
            list(source) for source in syngine_module._BAYESISOLA_SOURCES_USE
        ],
        "stations": contexts,
    }
    signature = syngine_module._short_signature(payload)
    return {
        "signature": signature,
        "valid": bool(syngine_module._cache_is_valid(Path(output_root), signature, contexts)),
    }


def _prepare_syngine_greens(
    data,
    *,
    output_path: Path,
    metadata_path: Path,
    source_time_function: str,
    options: Mapping[str, Any],
    use_precalculated_Green,
) -> dict[str, Any]:
    """Prepare corrected EarthScope Syngine GFs under the common cache policy.

    ``use_precalculated_Green`` has the same public meaning as for Axitra:

    - ``False`` forces regeneration of every source-grid Green's-function set;
    - ``'auto'`` reuses a compatible point cache and regenerates only missing or
      incompatible point caches;
    - ``True`` requires every source-grid point to have a complete compatible
      cache and raises before requesting/replacing an incompatible point.

    Syngine's native ``overwrite`` flag is therefore an internal implementation
    detail and is intentionally absent from ``gf_options``. Compatibility is based
    on the corrected backend's deterministic signature, which includes source and
    receiver geometry, model/service configuration, timing/sampling requirements,
    source-time-function convention and cache schema, plus existence/readability of
    the six BayesISOLA elementary-seismogram outputs for every receiver.
    """
    import BayesISOLA.syngine as syngine_module
    from obspy import UTCDateTime
    try:
        from tqdm.auto import tqdm
    except ImportError:
        tqdm = None

    mode = use_precalculated_Green
    if mode not in {False, True, "auto"}:
        raise ValueError("use_precalculated_Green must be False, True or 'auto'.")

    root_value = options.get("output_dir")
    root = Path(root_value).expanduser() if root_value is not None else output_path / "greens" / "syngine"
    if not root.is_absolute():
        root = output_path / root
    root.mkdir(parents=True, exist_ok=True)

    init_kwargs = {
        "syngine_dt": options["syngine_dt"],
        "kernelwidth": int(options["kernelwidth"]),
        "timeout": float(options["timeout"]),
        "request_padding_s": float(options["request_padding_s"]),
        "max_workers": int(options["max_workers"]),
        "progress": False,
    }
    if options.get("url") is not None:
        init_kwargs["url"] = str(options["url"])

    force = mode is False
    query = syngine_module.generate_query(overwrite=force, **init_kwargs)
    query.bulk = [
        {
            "networkcode": station["network"],
            "stationcode": station["code"],
            "locationcode": station.get("location", ""),
            "latitude": float(station["lat"]),
            "longitude": float(station["lon"]),
        }
        for station in data.d.stations
    ]

    grid = data.grid.grid
    iterator = range(len(grid))
    if tqdm is not None:
        iterator = tqdm(
            iterator, total=len(grid), desc="Syngine source grid", unit="pt",
            disable=not bool(options["progress"]),
        )

    rows = []
    origin = UTCDateTime(data.d.event["t"])
    start = origin + min(0.0, float(data.t_min))
    end = origin + float(data.t_max)
    for i in iterator:
        gp = grid[i]
        point_root = root / str(gp["z_id"]) / f"{gp['x_id']}{gp['y_id']}"

        if mode is True:
            cache = _syngine_cache_record(
                syngine_module, query, output_root=point_root,
                model=options["model"], source_lat=float(gp["lat"]),
                source_lon=float(gp["lon"]), source_depth_m=float(gp["z"]),
                origin=origin, end=end,
                target_npts=int(data.npts_elemse),
                target_sampling_rate=float(data.samprate),
                source_time_function=source_time_function,
            )
            if not cache["valid"]:
                raise ValueError(
                    "use_precalculated_Green=True requires complete compatible Syngine "
                    f"Green's functions, but grid point {i} ({point_root}) is missing "
                    "or incompatible. Use 'auto' to regenerate only incompatible points "
                    "or False to force a complete regeneration."
                )
            manifest = {"status": "existing", "signature": cache["signature"]}
        else:
            manifest = query.do_query_simple(
                options["model"], float(gp["lat"]), float(gp["lon"]), float(gp["z"]),
                origin, start, end, point_root,
                target_npts=int(data.npts_elemse),
                target_sampling_rate=float(data.samprate),
                overwrite=force, progress=False,
                max_workers=int(options["max_workers"]),
                source_time_function=source_time_function,
            )

        rows.append({
            "grid_index": i, "grid_point_id": str(i).zfill(4),
            "source_lat": float(gp["lat"]), "source_lon": float(gp["lon"]),
            "source_depth_km": float(gp["z"]) / 1000.0,
            "status": manifest.get("status", "unknown"),
            "signature": manifest.get("signature", ""), "path": str(point_root),
        })

    manifest_df = pd.DataFrame(rows)
    manifest_path = metadata_path / "gf_syngine_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)
    data.use_elemse_from_files(str(root))
    reused = bool(len(manifest_df)) and bool((manifest_df["status"] == "existing").all())
    status_counts = manifest_df["status"].value_counts().to_dict() if not manifest_df.empty else {}
    normalized_options = dict(options)
    normalized_options["output_dir"] = str(root)
    return {
        "source": "syngine",
        "model": str(options["model"]),
        "path": str(root),
        "reused": reused,
        "cache_policy": mode,
        "options": normalized_options,
        "manifest": str(manifest_path),
        "cache_status": status_counts,
    }


def _png_state(root: Path) -> dict[Path, tuple[int, int]]:
    return {path.resolve(): (path.stat().st_mtime_ns, path.stat().st_size) for path in root.rglob("*.png") if path.is_file()}


def _changed_pngs(root: Path, before: Mapping[Path, tuple[int, int]]) -> list[Path]:
    changed = []
    for path in root.rglob("*.png"):
        if not path.is_file():
            continue
        resolved = path.resolve(); state = (path.stat().st_mtime_ns, path.stat().st_size)
        if resolved not in before or before[resolved] != state:
            changed.append(path)
    return sorted(changed)


def _display_saved_figures(paths: Sequence[Path]) -> None:
    from IPython.display import Image, display
    for path in paths:
        if "mouse" not in {part.lower() for part in path.parts}:
            display(Image(filename=str(path)))
# ---------------------------------------------------------------------------
# Depth-grid automation
# ---------------------------------------------------------------------------

def suggest_depth_limits(
    catalog_depth_km: float,
    *,
    min_depth_km: float = 5.0,
    min_depth_multiplier: float = 0.5,
    max_depth_multiplier: float = 3.0,
    step_z_km: float,
    step_x_km: float,
    radius_km: float,
    max_points: int,
) -> dict[str, Any]:
    """Resolve the explicit BayesISOLA depth range and estimate grid rescaling.

    The shallow bound is no longer forced to the global 5-km floor for every
    event. Instead,

    ``grid_min_depth = max(min_depth_km, catalog_depth * min_depth_multiplier)``.

    With the defaults, a 40-km event therefore begins its search at 20 km, while
    a shallow event is never searched above the 5-km resolution floor. The deep
    bound remains ``catalog_depth * max_depth_multiplier``. ``depth_unc_km`` is
    the symmetric half-width BayesISOLA needs around the catalogue depth so its
    catalogue-anchored candidate-depth loop can reach both explicit bounds; it is
    not a statistical hypocentral uncertainty.

    ``step_x_km``/``step_z_km`` and ``max_points`` reproduce BayesISOLA's own
    approximate cubic rescaling estimate. The realised grid remains authoritative
    because BayesISOLA subsequently rounds the number of horizontal/vertical
    steps to integers.
    """
    catalog_depth_km = float(catalog_depth_km)
    step_z_km = float(step_z_km); step_x_km = float(step_x_km); radius_km = float(radius_km); max_points = int(max_points)
    grid_min_depth_km, grid_max_depth_km = _depth_bounds_km(
        catalog_depth_km, min_depth_km=min_depth_km, min_depth_multiplier=min_depth_multiplier,
        max_depth_multiplier=max_depth_multiplier,
    )
    if step_z_km <= 0 or step_x_km <= 0:
        raise ValueError("step_z_km and step_x_km must be positive.")
    if radius_km < 0:
        raise ValueError("radius_km cannot be negative.")
    if max_points <= 0:
        raise ValueError("max_points must be positive.")

    depth_unc_km = max(abs(catalog_depth_km - grid_min_depth_km), abs(grid_max_depth_km - catalog_depth_km))
    k_min = math.ceil((grid_min_depth_km - catalog_depth_km) / step_z_km)
    k_max = math.floor((grid_max_depth_km - catalog_depth_km) / step_z_km)
    n_depth_levels_estimate = max(0, k_max - k_min + 1)
    depth_span_km = grid_max_depth_km - grid_min_depth_km
    n_points_raw = np.pi * (radius_km / step_x_km) ** 2 * depth_span_km / step_z_km

    rescale_factor = 1.0
    warnings = []
    if n_points_raw > max_points:
        rescale_factor = (n_points_raw / max_points) ** 0.333
        warnings.append(
            f"Estimated pre-rescaling grid size is {n_points_raw:.0f} points, which exceeds max_points={max_points}. "
            f"BayesISOLA will increase step_x and step_z together by approximately {rescale_factor:.3f}x "
            f"(step_x ~{step_x_km * rescale_factor:.3f} km; step_z ~{step_z_km * rescale_factor:.3f} km)."
        )

    return {
        "grid_min_depth_km": grid_min_depth_km,
        "grid_max_depth_km": grid_max_depth_km,
        "depth_unc_km": depth_unc_km,
        "n_depth_levels_estimate": n_depth_levels_estimate,
        "n_points_estimate": max(n_depth_levels_estimate if radius_km == 0 else 0, int(math.ceil(n_points_raw))),
        "rescale_factor_estimate": rescale_factor,
        "warnings": warnings,
    }


def diagnose_grid_edge(grid, centroid: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Disambiguate a grid point's `edge=True` flag: depth-range edge vs.
    horizontal-radius edge vs. horizontal-index edge. Pass the constructed
    `BayesISOLA.grid` instance and, optionally, `solution.centroid` to
    check that specific point; otherwise reports on every edge point found.

    BayesISOLA stores depth- and horizontal-boundary conditions in one
    ``edge`` flag. This helper separates those causes so an edge solution can
    be interpreted correctly before deciding whether the depth range or the
    horizontal search radius needs to be widened.
    """
    depths = sorted({gp["z"] for gp in grid.grid})
    depth_lo, depth_hi = (depths[0], depths[-1]) if depths else (None, None)
    n_steps = int(grid.radius / grid.step_x) if grid.step_x else 0

    def _classify(gp: dict[str, Any]) -> list[str]:
        reasons = []
        if gp["z"] == depth_lo:
            reasons.append(f"depth at grid floor ({depth_lo / 1e3:.3f} km)")
        if gp["z"] == depth_hi:
            reasons.append(f"depth at grid ceiling ({depth_hi / 1e3:.3f} km)")
        i = round(gp["x"] / grid.step_x) if grid.step_x else 0
        j = round(gp["y"] / grid.step_x) if grid.step_x else 0
        if max(abs(i), abs(j)) == n_steps:
            reasons.append(
                f"horizontal index at max (i={i}, j={j}, n_steps={n_steps}); "
                "driven by location_unc + rupture_length, not depth"
            )
        return reasons or ["horizontal distance exceeds radius by >1 step"]

    report: dict[str, Any] = {
        "realized_radius_km": grid.radius / 1e3,
        "realized_step_x_km": grid.step_x / 1e3,
        "realized_step_z_km": grid.step_z / 1e3,
        "realized_depth_range_km": (
            (depth_lo / 1e3, depth_hi / 1e3) if depths else None
        ),
        "n_depth_levels_realized": len(depths),
        "n_steps_horizontal": n_steps,
    }
    if centroid is not None:
        report["centroid_edge_reasons"] = (
            _classify(centroid) if centroid.get("edge") else []
        )
    else:
        edge_points = [gp for gp in grid.grid if gp.get("edge") and not gp.get("err")]
        report["n_edge_points"] = len(edge_points)
    return report
 

# ---------------------------------------------------------------------------
# Structured output extraction
# ---------------------------------------------------------------------------

def extract_station_fit_df(solution) -> pd.DataFrame:
    """Return station/component geometry, weighting and fit information.

    One row is returned for each Z/N/E component of every station retained by
    BayesISOLA. Distances are in kilometres; azimuths are in degrees; filter
    limits are in hertz. ``variance_reduction`` is BayesISOLA's native VR
    fraction (1.0 = 100%, and values may be negative for a poor component fit),
    not a percentage.

    Per-component VR values exist only after ``solution.VR_of_components()`` has
    run. ``run_auto_cmt`` requests them directly through
    ``resolve_MT(..., VR_of_components=True)``; solutions constructed separately
    may therefore need ``solution.VR_of_components()`` before calling this helper.
    """
    rows = []
    for stn in solution.inp.stations:
        for comp in ("Z", "N", "E"):
            rows.append({
                "network": stn.get("network"),
                "station": stn.get("code"),
                "location": stn.get("location"),
                "channelcode": stn.get("channelcode"),
                "component": comp,
                "distance_km": stn.get("dist", np.nan) / 1e3 if stn.get("dist") is not None else np.nan,
                "azimuth_deg": stn.get("az"),
                "fmin_hz": stn.get("fmin"),
                "fmax_hz": stn.get("fmax"),
                "used": stn.get(f"use{comp}"),
                "weight": stn.get(f"weight{comp}"),
                "variance_reduction": stn.get(f"VR_{comp}"),
            })
    df = pd.DataFrame(rows)
    if not df.empty and df["variance_reduction"].isna().all():
        import warnings as _warnings
        _warnings.warn(
            "All variance_reduction values are missing -- call solution.VR_of_components() "
            "before extract_station_fit_df() if per-component fit quality is needed.",
            stacklevel=2,
        )
    return df


def extract_centroid_location(solution) -> dict[str, Any]:
    """Return catalogue hypocentre and preferred BayesISOLA centroid location.

    The returned mapping is designed to form one row of a results table. Event
    and centroid depths are in kilometres; north/east centroid offsets are in
    metres; ``centroid_time_shift_s`` is relative to the catalogue origin time.
    ``variance_reduction`` is BayesISOLA's native fractional VR (1.0 = 100%),
    while ``condition_number`` is the native inversion condition number.
    """
    event = solution.event
    c = solution.centroid
    centroid_time = event["t"] + c["shift"]
    return {
        "origin_time": event["t"].datetime,
        "origin_lat": event["lat"],
        "origin_lon": event["lon"],
        "origin_depth_km": event["depth"] / 1e3,
        "centroid_time": centroid_time.datetime,
        "centroid_time_shift_s": c["shift"],
        "centroid_lat": c["lat"],
        "centroid_lon": c["lon"],
        "centroid_depth_km": c["z"] / 1e3,
        "offset_north_m": c["x"],
        "offset_east_m": c["y"],
        "on_grid_edge": bool(c["edge"]),
        "variance_reduction": c["VR"],
        "condition_number": c["CN"],
    }


def extract_solution_summary(solution) -> dict[str, Any]:
    """Return the preferred moment-tensor solution as a flat mapping.

    The six tensor components are converted with BayesISOLA ``a2mt`` to the USE
    convention and retain BayesISOLA's native moment units. ``M0_Nm`` and ``Mw``
    come from the native tensor decomposition, as do DC/CLVD/ISO percentages and
    the two nodal planes. ``variance_reduction`` is the native fractional VR
    (1.0 = 100%), not a percentage. The solution must have been resolved with
    decomposition enabled (the BayesISOLA default and the ``run_auto_cmt`` path).
    """
    from BayesISOLA.MT_comps import a2mt

    if not solution.mt_decomp:
        raise RuntimeError(
            "solution.mt_decomp is empty -- resolve_MT must be constructed "
            "with decompose=True (the default) for this function to work."
        )
    c = solution.centroid
    mt = a2mt(c["a"], system="USE")
    d = solution.mt_decomp
    return {
        "variance_reduction": c["VR"],
        "condition_number": c["CN"],
        "Mrr": mt[0], "Mtt": mt[1], "Mpp": mt[2],
        "Mrt": mt[3], "Mrp": mt[4], "Mtp": mt[5],
        "M0_Nm": d["mom"],
        "Mw": d["Mw"],
        "DC_percent": d["dc_perc"],
        "CLVD_percent": d["clvd_perc"],
        "ISO_percent": d["iso_perc"],
        "NP1_strike_deg": d["s1"], "NP1_dip_deg": d["d1"], "NP1_rake_deg": d["r1"],
        "NP2_strike_deg": d["s2"], "NP2_dip_deg": d["d2"], "NP2_rake_deg": d["r2"],
    }


def extract_uncertainty_df(solution, n: int = 400) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return BayesISOLA mechanism realizations with their space-time support.

    Moment-tensor realizations are generated by BayesISOLA's native
    ``plot_uncertainty(..., just_return_histogram_data=True)`` routine. This
    helper does not replace that sampler. It replays only BayesISOLA's deterministic
    allocation loop -- ``round(GP['c'] / solution.sum_c * n)`` -- to attach the
    same grid point and source-time shift to each returned mechanism realization,
    in the same iteration order used by the native routine.

    The space-time columns are therefore *discrete posterior support labels*.
    Within one occupied grid-point/time-shift cell, depth, latitude/longitude,
    horizontal offsets and time are repeated while BayesISOLA samples moment
    tensors from that grid point's ``GtGinv`` covariance. Zero spread in a
    location/time column consequently means the rounded realizations occupied
    only one sampled value at the chosen grid resolution; it is not evidence of
    zero sub-grid uncertainty.

    ``n`` is a target realization count, not an exact row count. BayesISOLA
    rounds each space-time cell independently, so ``n_allocated``/``n_sampled``
    can differ slightly from ``n_requested``. The returned diagnostics distinguish
    unique spatial grid points, unique time shifts and occupied space-time cells.

    Existing mechanism column names from earlier helper versions are retained for
    compatibility. Added columns include grid/shift identifiers, absolute and
    relative centroid time, centroid coordinates/depth/offsets, edge status,
    normalized space-time-cell weight and the rounded number of draws assigned to the
    originating cell.
    """
    import BayesISOLA

    n = int(n)
    if n <= 0:
        raise ValueError("n must be a positive integer.")
    sum_c = float(solution.sum_c)
    if not np.isfinite(sum_c) or sum_c <= 0.0:
        raise ValueError("solution.sum_c must be positive and finite for uncertainty sampling.")

    shim = BayesISOLA.plot(solution, **PLOT_PRESETS["none"])

    allocation_rows: list[dict[str, Any]] = []
    used_grid_points: set[int] = set()
    used_shift_indices: set[int] = set()
    used_cells: set[tuple[int, int]] = set()

    for grid_index, gp in enumerate(solution.grid):
        if gp["err"]:
            continue
        for shift_index, shift_entry in gp["shifts"].items():
            n_gp = int(round(float(shift_entry["c"]) / sum_c * n))
            if n_gp <= 0:
                continue

            shift_index_int = int(shift_index)
            shift_s = float(shim.data.shifts[shift_index])
            centroid_time = solution.event["t"] + shift_s
            weight = float(shift_entry["c"]) / sum_c
            label = {
                "grid_point_index": int(grid_index),
                "grid_point_id": gp.get("id", str(grid_index).zfill(4)),
                "shift_index": shift_index_int,
                "centroid_time_shift_s": shift_s,
                "centroid_time": centroid_time.datetime if hasattr(centroid_time, "datetime") else centroid_time,
                "centroid_lat": float(gp.get("lat", np.nan)),
                "centroid_lon": float(gp.get("lon", np.nan)),
                "centroid_depth_km": float(gp["z"]) / 1e3,
                "offset_north_m": float(gp["x"]),
                "offset_east_m": float(gp["y"]),
                "on_grid_edge": bool(gp.get("edge", False)),
                "space_time_cell_weight": weight,
                "space_time_cell_n_draws": n_gp,
            }
            allocation_rows.extend(dict(label) for _ in range(n_gp))
            used_grid_points.add(int(grid_index))
            used_shift_indices.add(shift_index_int)
            used_cells.add((int(grid_index), shift_index_int))

    n_allocated = len(allocation_rows)
    sampled = shim.plot_uncertainty(n=n, just_return_histogram_data=True)

    diagnostics = {
        "n_requested": n,
        "n_allocated": n_allocated,
        "n_sampled": 0 if not sampled else len(sampled["Mw"]),
        "allocation_difference": n_allocated - n,
        "n_space_time_cells_used": len(used_cells),
        "n_grid_points_used": len(used_grid_points),
        "n_time_shifts_used": len(used_shift_indices),
        "spatially_degenerate": len(used_grid_points) <= 1,
        "temporally_degenerate": len(used_shift_indices) <= 1,
        "degenerate_allocation": len(used_cells) <= 1,
    }

    if not sampled:
        diagnostics["note"] = (
            "BayesISOLA returned no uncertainty table because the rounded posterior allocation "
            "contained at most one realization. Increase n before interpreting mechanism or "
            "space-time uncertainty."
        )
        return pd.DataFrame(), diagnostics

    n_draws = len(sampled["Mw"])
    if n_allocated != n_draws:
        raise RuntimeError(
            "BayesISOLA uncertainty allocation and returned mechanism draws have different lengths "
            f"({n_allocated} vs {n_draws}); refusing to attach potentially misaligned space-time labels."
        )

    scalar_keys = ("dc", "clvd", "iso", "moment", "Mw")
    if any(len(sampled[key]) != n_draws for key in scalar_keys):
        raise RuntimeError("BayesISOLA uncertainty scalar arrays do not share a common draw length.")
    if any(len(sampled[key]) < 2 * n_draws for key in ("strike", "dip", "rake")):
        raise RuntimeError("BayesISOLA uncertainty nodal-plane arrays are shorter than two values per draw.")

    strike = np.asarray(sampled["strike"])
    dip = np.asarray(sampled["dip"])
    rake = np.asarray(sampled["rake"])
    df = pd.DataFrame(allocation_rows)
    df.insert(0, "draw", np.arange(1, n_draws + 1, dtype=int))
    df["dc_percent"] = sampled["dc"]
    df["clvd_percent"] = sampled["clvd"]
    df["iso_percent"] = sampled["iso"]
    df["moment_Nm"] = sampled["moment"]
    df["Mw"] = sampled["Mw"]
    df["NP1_strike_deg"] = strike[0::2][:n_draws]
    df["NP1_dip_deg"] = dip[0::2][:n_draws]
    df["NP1_rake_deg"] = rake[0::2][:n_draws]
    df["NP2_strike_deg"] = strike[1::2][:n_draws]
    df["NP2_dip_deg"] = dip[1::2][:n_draws]
    df["NP2_rake_deg"] = rake[1::2][:n_draws]

    if diagnostics["degenerate_allocation"]:
        diagnostics["note"] = (
            "All returned realizations occupy one discrete space-time cell. Location/time columns "
            "therefore repeat exactly; mechanism spread is conditional on that cell's GtGinv covariance."
        )
    elif diagnostics["spatially_degenerate"] or diagnostics["temporally_degenerate"]:
        fixed = []
        if diagnostics["spatially_degenerate"]:
            fixed.append("space")
        if diagnostics["temporally_degenerate"]:
            fixed.append("time")
        diagnostics["note"] = (
            "The rounded posterior allocation occupies only one sampled value in " + " and ".join(fixed) +
            "; zero spread there reflects the discrete grid/time resolution, not zero sub-grid uncertainty."
        )

    return df, diagnostics


def _build_results(
    solution,
    *,
    n_uncertainty: int | None = None,
    grid_edge_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the curated in-memory results layer used by ``run_auto_cmt``."""
    centroid = pd.DataFrame([extract_centroid_location(solution)])
    summary = pd.DataFrame([extract_solution_summary(solution)])
    station_fit = extract_station_fit_df(solution)

    uncertainty = None
    uncertainty_diagnostics = None
    if n_uncertainty is not None:
        n_uncertainty = int(n_uncertainty)
        if n_uncertainty <= 0:
            raise ValueError("n_uncertainty must be a positive integer or None.")
        uncertainty, diagnostics = extract_uncertainty_df(solution, n=n_uncertainty)
        uncertainty_diagnostics = pd.DataFrame([diagnostics])

    return {
        "centroid": centroid,
        "summary": summary,
        "station_fit": station_fit,
        "uncertainty": uncertainty,
        "uncertainty_diagnostics": uncertainty_diagnostics,
        "grid_edge_report": pd.DataFrame([dict(grid_edge_report)]) if grid_edge_report is not None else None,
    }


def _write_result_tables(results: Mapping[str, Any], output_dir: str | Path) -> dict[str, Path]:
    """Write an already-built results mapping without repeating stochastic sampling."""
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    paths["station_df"] = output_dir / "station_df.csv"
    results["station_fit"].to_csv(paths["station_df"], index=False)
    paths["centroid_location"] = output_dir / "centroid_location.csv"
    results["centroid"].to_csv(paths["centroid_location"], index=False)
    paths["solution_summary"] = output_dir / "solution_summary.csv"
    results["summary"].to_csv(paths["solution_summary"], index=False)

    if results.get("uncertainty") is not None:
        paths["solution_unc_df"] = output_dir / "solution_unc_df.csv"
        results["uncertainty"].to_csv(paths["solution_unc_df"], index=False)
        paths["solution_unc_diagnostics"] = output_dir / "solution_unc_diagnostics.csv"
        results["uncertainty_diagnostics"].to_csv(paths["solution_unc_diagnostics"], index=False)

    return paths


def write_solution_outputs(
    solution,
    output_dir: str | Path,
    *,
    n_uncertainty: int | None = None,
) -> dict[str, Path]:
    """Extract and write the standard BayesISOLA scientific result tables.

    Station/component fit, preferred centroid and moment-tensor summary are always
    written. ``n_uncertainty=None`` skips posterior sampling and uncertainty CSVs.
    A positive integer requests a *target* number of BayesISOLA uncertainty
    realizations. Because the native routine rounds each space-time cell's
    allocation independently, the actual number of rows can differ slightly; the
    saved diagnostics report both requested and allocated/sample counts.

    ``run_auto_cmt`` builds its in-memory ``run["results"]`` mapping first and
    writes those same objects through an internal writer so stochastic uncertainty
    sampling occurs only once. This public function remains useful for a separately
    constructed ``resolve_MT`` solution and retains its original return contract of
    a mapping from result name to CSV path.
    """
    results = _build_results(solution, n_uncertainty=n_uncertainty)
    return _write_result_tables(results, output_dir)


# ---------------------------------------------------------------------------
# CMT summary figure
# ---------------------------------------------------------------------------

def plot_cmt_summary(
    summary,
    centroid,
    uncertainty=None,
    *,
    tensor_mode="full",
    facecolor="white",
    bgcolor="red",
    show_dc_overlay=True,
    figsize=(11.5, 6.2),
    dpi=300,
    output_file=None,
    show=False,
):
    """
    Plot a publication-style summary of a BayesISOLA CMT solution.

    This is the operational replacement for BayesISOLA's basic single-beachball
    figure in the helper's ``summary`` plot preset. It does not alter the native
    inversion or decomposition. The figure combines the preferred solution with
    a zero-trace beachball, P/T labels, the DC nodal-plane overlay, source and
    quality metrics, moment-tensor components, nodal planes and principal axes.

    BayesISOLA's native ``plot_MT`` passes the preferred six-component tensor to
    ObsPy ``beach`` without overriding ``plot_zerotrace``; ObsPy therefore plots
    the zero-trace/deviatoric representation even when ``deviatoric=False`` was
    used for a full six-component inversion. This helper follows that zero-trace
    plotting convention explicitly. For ``tensor_mode='full'`` the source table
    and component table still report the full six-component solution, including
    ISO, while the beachball and principal axes represent its zero-trace part.
    For ``tensor_mode='deviatoric'`` the inversion itself is five-component and
    ISO is reported as constrained to zero.

    The beachball fill colours are controlled independently through ``facecolor``
    and ``bgcolor``. Their defaults are white and red, respectively, providing
    the preferred visual polarity convention for this summary figure without
    altering or negating the moment tensor.

    Parameters
    ----------
    summary : mapping, pandas.Series, or one-row pandas.DataFrame
        BayesISOLA solution summary, normally ``results["summary"]``. Required
        fields are ``Mrr``, ``Mtt``, ``Mpp``, ``Mrt``, ``Mrp``, ``Mtp``,
        ``M0_Nm``, ``Mw``, ``DC_percent``, ``CLVD_percent``, ``ISO_percent``,
        ``variance_reduction``, ``condition_number`` and both NP1/NP2
        strike/dip/rake triplets. Tensor components are in the USE
        (Up-South-East; r-theta-phi) convention and moments are in N·m.

    centroid : mapping, pandas.Series, or one-row pandas.DataFrame
        Preferred centroid information, normally ``results["centroid"]``.
        ``centroid_depth_km`` is required.

    uncertainty : optional
        Retained only for backward call compatibility with version 0.1.6. It is
        intentionally ignored in 0.1.7; the deterministic CMT summary no longer
        displays uncertainty-derived ``±`` values.

    tensor_mode : {"full", "deviatoric"}
        Inversion type that produced ``summary``. ``"full"`` denotes the native
        six-component inversion, whereas ``"deviatoric"`` denotes the
        five-component zero-trace inversion. This controls source reporting and
        decomposition interpretation. The beachball itself is plotted zero-trace
        in both modes, matching BayesISOLA/ObsPy's native tensor-display
        convention; its colour assignment is controlled separately by
        ``facecolor`` and ``bgcolor``.

    facecolor : matplotlib color, default="white"
        Foreground beachball colour passed directly to ObsPy ``beach`` as
        ``facecolor``. Together with the default red ``bgcolor``, this defines
        the preferred fill convention used by the summary figure. The moment
        tensor itself is not negated or otherwise modified.

    bgcolor : matplotlib color, default="red"
        Background beachball colour passed directly to ObsPy ``beach`` as
        ``bgcolor``. The default red background combined with white
        ``facecolor`` intentionally reverses the native ObsPy/BayesISOLA colour
        assignment while leaving the moment-tensor geometry and P/T axes
        unchanged.

    show_dc_overlay : bool, default=True
        Overlay the double-couple nodal-plane geometry derived from NP1. For a
        non-pure-DC solution these curves need not coincide exactly with the
        zero-trace moment-tensor beachball boundaries; the overlay is retained as
        a useful visual comparison with the decomposed DC mechanism.

    figsize : tuple, default=(11.5, 6.2)
        Matplotlib figure size in inches.

    dpi : int, default=300
        Figure resolution.

    output_file : str or pathlib.Path, optional
        Save the figure to this path. Parent directories are created as needed.

    show : bool, default=False
        Display the figure. In IPython/Jupyter, ``display(fig)`` is used so an
        Agg backend does not emit a non-interactive ``plt.show`` warning.

    Returns
    -------
    matplotlib.figure.Figure
        Generated CMT summary figure.
    """
    import matplotlib.pyplot as plt
    from obspy.imaging.beachball import beach, MomentTensor, mt2axes

    def as_record(obj, name):
        if isinstance(obj, pd.DataFrame):
            if len(obj) != 1:
                raise ValueError(f"{name} must contain exactly one row.")
            return obj.iloc[0].to_dict()
        if isinstance(obj, pd.Series):
            return obj.to_dict()
        return dict(obj)

    def format_value(value, fmt, unit=""):
        return f"{format(float(value), fmt)}{unit}"

    def format_scientific(value, unit=""):
        return f"{float(value):.3e}{unit}"

    def project_axis(axis):
        azimuth = np.deg2rad(axis.strike)
        radius = np.sqrt(2.0) * np.sin(np.deg2rad((90.0 - axis.dip) / 2.0))
        return radius * np.sin(azimuth), radius * np.cos(azimuth)

    def rim_xy(azimuth, radius=1.0):
        azimuth = np.deg2rad(azimuth)
        return radius * np.sin(azimuth), radius * np.cos(azimuth)

    def choose_plane_label_azimuths(strike1, strike2):
        candidates1 = (strike1 % 360.0, (strike1 + 180.0) % 360.0)
        candidates2 = (strike2 % 360.0, (strike2 + 180.0) % 360.0)
        best_pair, best_distance = None, -np.inf
        for az1 in candidates1:
            for az2 in candidates2:
                distance = np.linalg.norm(np.asarray(rim_xy(az1)) - np.asarray(rim_xy(az2)))
                if distance > best_distance:
                    best_pair, best_distance = (az1, az2), distance
        return best_pair

    def style_table(table, *, first_col_bold=False, header=True, fontsize=9):
        table.auto_set_font_size(False)
        table.set_fontsize(fontsize)
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor("0.72")
            cell.set_linewidth(0.55)
            cell.visible_edges = "B"
            if header and row == 0:
                cell.get_text().set_fontweight("bold")
            if first_col_bold and col == 0 and (not header or row > 0):
                cell.get_text().set_fontweight("bold")

    summary = as_record(summary, "summary")
    centroid = as_record(centroid, "centroid")
    tensor_mode = str(tensor_mode).lower().strip()
    if tensor_mode not in {"full", "deviatoric"}:
        raise ValueError("tensor_mode must be 'full' or 'deviatoric'.")

    mt = np.array([
        summary["Mrr"], summary["Mtt"], summary["Mpp"],
        summary["Mrt"], summary["Mrp"], summary["Mtp"],
    ], dtype=float)

    # BayesISOLA's native beachball is zero-trace even after a six-component
    # inversion because ObsPy beach() defaults to plot_zerotrace=True. Make that
    # behavior explicit and use the same tensor for the plotted principal axes.
    mt_plot = mt.copy()
    mt_plot[:3] -= np.mean(mt_plot[:3])

    if tensor_mode == "deviatoric":
        tensor_label = "Deviatoric (5-component)"
        mt_table_values = mt_plot
    else:
        tensor_label = "Full MT (6-component)"
        mt_table_values = mt

    mt_obj = MomentTensor(*mt_plot, 0)
    T, N, P = mt2axes(mt_obj)

    nodal_values = [
        summary.get("NP1_strike_deg"), summary.get("NP1_dip_deg"), summary.get("NP1_rake_deg"),
        summary.get("NP2_strike_deg"), summary.get("NP2_dip_deg"), summary.get("NP2_rake_deg"),
    ]
    has_nodal_planes = all(
        value is not None and np.isfinite(float(value)) for value in nodal_values
    )
    draw_dc_overlay = bool(show_dc_overlay and has_nodal_planes)

    fig = plt.figure(figsize=figsize, dpi=dpi, constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=(1.05, 1.25))
    ax_ball = fig.add_subplot(gs[0])
    ax_info = fig.add_subplot(gs[1])
    ax_info.axis("off")

    ball = beach(
        mt_plot,
        xy=(0, 0),
        width=2.0,
        size=300,
        linewidth=1.15,
        facecolor=facecolor,
        bgcolor=bgcolor,
        edgecolor="black",
        zorder=1,
    )
    ax_ball.add_collection(ball)

    if draw_dc_overlay:
        planes = beach(
            (summary["NP1_strike_deg"], summary["NP1_dip_deg"], summary["NP1_rake_deg"]),
            xy=(0, 0), width=2.0, size=300, linewidth=0.65,
            nofill=True, edgecolor="0.35", zorder=2,
        )
        ax_ball.add_collection(planes)

    for label, axis in (("P", P), ("T", T)):
        x, y = project_axis(axis)
        ax_ball.text(x, y, label, ha="center", va="center", fontsize=15, zorder=5)

    if draw_dc_overlay:
        np1_az, np2_az = choose_plane_label_azimuths(summary["NP1_strike_deg"], summary["NP2_strike_deg"])
        for label, azimuth in (("NP1", np1_az), ("NP2", np2_az)):
            x0, y0 = rim_xy(azimuth, 1.00)
            x1, y1 = rim_xy(azimuth, 1.11)
            ax_ball.annotate(
                label, xy=(x0, y0), xytext=(x1, y1),
                ha="left" if x1 >= 0 else "right", va="center", fontsize=9,
                annotation_clip=False,
                arrowprops={"arrowstyle": "-", "linewidth": 0.7, "shrinkA": 2, "shrinkB": 0},
                zorder=6,
            )

    ax_ball.set_xlim(-1.18, 1.18)
    ax_ball.set_ylim(-1.18, 1.18)
    ax_ball.set_aspect("equal")
    ax_ball.axis("off")
    ax_ball.set_title("Centroid Moment Tensor", fontsize=13, pad=10)

    depth_text = format_value(centroid["centroid_depth_km"], ".1f", " km")
    moment_text = format_scientific(summary["M0_Nm"], " N·m")
    magnitude_text = format_value(summary["Mw"], ".2f", " Mw")
    dc_text = format_value(summary["DC_percent"], ".1f", "%")
    clvd_text = format_value(summary["CLVD_percent"], ".1f", "%")
    iso_text = (
        format_value(summary["ISO_percent"], ".1f", "%")
        if tensor_mode == "full" else "0.0% (constrained)"
    )
    vr = float(summary["variance_reduction"])

    source_rows = [
        ["Tensor", tensor_label],
        ["Moment", moment_text],
        ["Magnitude", magnitude_text],
        ["Centroid depth", depth_text],
        ["DC", dc_text],
        ["CLVD", clvd_text],
        ["ISO", iso_text],
        ["Variance reduction", f"{100.0 * vr:.1f}%"],
        ["Condition number", f"{summary['condition_number']:.2f}"],
    ]

    ax_info.text(0.0, 0.98, "Source", fontsize=12, fontweight="bold", transform=ax_info.transAxes)
    source_table = ax_info.table(
        cellText=source_rows, cellLoc="left", colWidths=[0.42, 0.56],
        bbox=[0.0, 0.62, 1.0, 0.32],
    )
    style_table(source_table, first_col_bold=True, header=False, fontsize=9.2)

    ax_info.text(
        0.0, 0.575, "Moment Tensor Components (N·m)",
        fontsize=11, fontweight="bold", transform=ax_info.transAxes,
    )
    mt_rows = [
        ["Mrr", f"{mt_table_values[0]:.2e}", "Mtt", f"{mt_table_values[1]:.2e}", "Mpp", f"{mt_table_values[2]:.2e}"],
        ["Mrt", f"{mt_table_values[3]:.2e}", "Mrp", f"{mt_table_values[4]:.2e}", "Mtp", f"{mt_table_values[5]:.2e}"],
    ]
    mt_table = ax_info.table(
        cellText=mt_rows, cellLoc="center",
        colWidths=[0.09, 0.23, 0.09, 0.23, 0.09, 0.23],
        bbox=[0.0, 0.448, 1.0, 0.10],
    )
    mt_table.auto_set_font_size(False)
    mt_table.set_fontsize(8.6)
    for (row, col), cell in mt_table.get_celld().items():
        cell.set_edgecolor("0.75")
        cell.set_linewidth(0.5)
        cell.visible_edges = "B"
        if col in (0, 2, 4):
            cell.get_text().set_fontweight("bold")

    ax_info.text(0.0, 0.375, "Nodal Planes", fontsize=11, fontweight="bold", transform=ax_info.transAxes)
    def format_angle(value):
        return f"{float(value):.1f}°" if value is not None and np.isfinite(float(value)) else "—"

    nodal_rows = [
        ["NP1", format_angle(summary.get("NP1_strike_deg")), format_angle(summary.get("NP1_dip_deg")), format_angle(summary.get("NP1_rake_deg"))],
        ["NP2", format_angle(summary.get("NP2_strike_deg")), format_angle(summary.get("NP2_dip_deg")), format_angle(summary.get("NP2_rake_deg"))],
    ]
    nodal_table = ax_info.table(
        cellText=nodal_rows, colLabels=["Plane", "Strike", "Dip", "Rake"],
        cellLoc="center", colLoc="center", bbox=[0.0, 0.075, 0.47, 0.265],
    )
    style_table(nodal_table, header=True, fontsize=8.4)

    ax_info.text(0.52, 0.375, "Principal Axes", fontsize=11, fontweight="bold", transform=ax_info.transAxes)
    axis_rows = [
        ["T", f"{T.val:.2e}", f"{T.dip:.1f}°", f"{T.strike:.1f}°"],
        ["N", f"{N.val:.2e}", f"{N.dip:.1f}°", f"{N.strike:.1f}°"],
        ["P", f"{P.val:.2e}", f"{P.dip:.1f}°", f"{P.strike:.1f}°"],
    ]
    axes_table = ax_info.table(
        cellText=axis_rows, colLabels=["Axis", "Value (N·m)", "Plunge", "Azimuth"],
        cellLoc="center", colLoc="center", bbox=[0.52, 0.075, 0.48, 0.265],
    )
    style_table(axes_table, header=True, fontsize=8.0)

    if output_file is not None:
        output_file = Path(output_file).expanduser()
        output_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_file, bbox_inches="tight")

    if show:
        try:
            from IPython import get_ipython
            from IPython.display import display
            if get_ipython() is not None:
                display(fig)
            else:
                plt.show()
        except ImportError:
            plt.show()

    return fig


# Deterministic figure presets. Native uncertainty plotting is intentionally
# disabled; optional n_uncertainty sampling remains a separate table-only diagnostic.
PLOT_PRESETS: dict[str, dict[str, Any]] = {
    "none": dict(
        maps=False, slices=False, maps_sum=False, MT=False, uncertainty=0,
        seismo=False, seismo_sharey=False, seismo_cova=False, noise=False,
        spectra=False, stations=False, covariance_matrix=False,
        covariance_function=False,
    ),
    "summary": dict(
        maps=False, slices=False, maps_sum=True, MT=False, uncertainty=0,
        seismo=True, seismo_sharey=False, seismo_cova=False, noise=False,
        spectra=False, stations=True, covariance_matrix=False,
        covariance_function=False,
    ),
    "full": dict(
        maps=True, slices=True, maps_sum=True, MT=True, uncertainty=0,
        seismo=True, seismo_sharey=True, seismo_cova=True, noise=True,
        spectra=True, stations=True, covariance_matrix=True,
        covariance_function=False,
    ),
}


def run_auto_cmt(
    event_id: str,
    origin_time,
    event_lon: float,
    event_lat: float,
    event_depth_km: float,
    magnitude: float,
    *,
    output_dir: str | Path,
    velocity_model=None,
    crust_file: str | Path | None = None,
    gf_source: str = "axitra",
    gf_options: Mapping[str, Any] | None = None,
    source_time_function: str = "step",
    waveform_source: str = "fdsn",
    station_df: pd.DataFrame | str | Path | None = None,
    client: str | Sequence[str | Any] | Any = "GEONET",
    min_radius_km: float = 0.0,
    max_radius_km: float | None = None,
    radius_scale_factor: float = 1.66,
    ground_level: bool = True,
    channels: Sequence[str] = ("HH?", "BH?", "LH?"),
    channel_priority: Sequence[str] = ("HH", "BH", "LH"),
    taup_model: str = "iasp91",
    overwrite_waveforms: bool = False,
    location_unc_km: float = 0.0,
    time_unc_s: float = 2.0,
    min_depth_km: float = 5.0,
    min_depth_multiplier: float = 0.5,
    max_depth_multiplier: float = 3.0,
    step_x_km: float = 2.0,
    step_z_km: float = 1.0,
    max_grid_points: int = 5000,
    add_rupture_length: bool = True,
    rupture_velocity_m_s: float = 1000.0,
    velocity_slowest_m_s: float = 1000.0,
    noise_factor: float = 4.0,
    edge_margin_s: float = 1.0,
    freqmin: float = 0.02,
    freqmax: float = 0.05,
    threads: int = 2,
    progress: bool = True,
    invert_displacement: bool = False,
    use_precalculated_Green: bool | str = "auto",
    covariance: str = "noise",
    crosscovariance: bool = False,
    deviatoric: bool = False,
    detect_mouse: bool = True,
    n_uncertainty: int | None = None,
    plot: bool = True,
    plot_preset: str = "summary",
    show: bool = True,
    profile_crs: str = "EPSG:4326",
    velocity_depth_col: str = "Depth_km",
    vp_col: str = "Vp",
    vs_col: str = "Vs",
    density_col: str = "Density",
    qp_col: str = "Qp",
    qs_col: str = "Qs",
    surface_depth_km: float = 0.0,
) -> dict[str, Any]:
    """Run the complete automated BayesISOLA centroid-moment-tensor workflow.

    The wrapper keeps acquisition, grid construction, Green's-function selection,
    covariance weighting and inversion choices explicit while routing all supported
    Green's-function backends into the same BayesISOLA inverse problem.

    Green's-function backend
    ------------------------
    ``gf_source='axitra'`` is the default and preserves the historical behaviour
    when ``gf_options=None``. ``crust_file`` may then provide an existing native
    BayesISOLA model, a 1-D ``velocity_model`` is converted directly, and a
    ``gf_helpers`` 3-D velocity grid is sampled once at the catalogue epicentre.
    The resulting six-column Vp/Vs/density/Qp/Qs model is used by native Axitra.

    A 3-D ``gf_helpers`` velocity grid can opt into station-dependent native
    Axitra models through ``gf_options``::

        gf_options={"grid": "station"}

    extracts one vertical model at each authoritative station coordinate, while::

        gf_options={
            "grid": "path",
            "path_spacing_km": 2.0,
            "path_profile": "mean",
        }

    samples each catalogue-origin-to-station path and forms one depth-wise
    representative model per station. ``path_profile`` may be mean, median, p05
    or p95. BayesISOLA/Axitra attaches the layered model to the receiver, not to
    each source grid point, so a station/path model is fixed across the centroid
    search grid. This is a station-dependent 1-D approximation to a 3-D medium,
    not a fully 3-D source-path Green's-function calculation. Exact duplicate
    layered profiles are grouped under one Axitra model identifier.

    ``gf_source='syngine'`` uses the corrected ``BayesISOLA.syngine`` backend.
    The default EarthScope model is ``ak135f_5s`` and may be changed with
    ``gf_options={'model': ...}``. The backend requests the ten native
    SeisComP/Instaseis Green's functions, converts them to BayesISOLA's six
    elementary source bases in ZNE, and writes the external ``GFs1``...``GFs6``
    layout consumed by BayesISOLA's unchanged inverse operator. ``velocity_model``
    and ``crust_file`` are Axitra inputs and are not used by Syngine. Supported
    Syngine options are ``model``, ``output_dir``, ``url``, ``syngine_dt``,
    ``kernelwidth``, ``timeout``, ``request_padding_s``, ``max_workers`` and
    ``progress``. Cache policy is intentionally not a backend option.

    Green's-function reuse policy
    ------------------------------
    ``use_precalculated_Green`` is the single cache/reuse control for every GF
    backend. ``False`` forces regeneration, ``'auto'`` reuses compatible output
    and regenerates missing/incompatible output, and ``True`` requires a complete
    compatible cache and raises rather than regenerating it. For native one-model
    Axitra this delegates to BayesISOLA's verified-cache path; station/path Axitra
    uses this helper's model-specific hash checks; Syngine uses its deterministic
    manifest signature and verifies the expected elementary-seismogram files.
    Backend-specific options therefore describe only physical/numerical/service
    configuration and cannot silently override this public cache policy.

    ``run['gf']`` records the selected backend, physical model/model mode, GF
    storage path, resolved cache policy, whether the complete compatible cache was
    reused, normalized backend options and any backend manifest. This backend
    contract is kept independent of the inversion so a future solver such as SW4
    can populate the
    same elementary-seismogram interface without changing ``resolve_MT``.

    For station/path-dependent Axitra models, this helper performs model-aware
    cache verification because the legacy BayesISOLA cache verifier only checks
    the unsuffixed base model. The numerical Axitra calculation itself remains
    native: model-specific ``crustal-<model>.dat`` and ``station-<model>.dat``
    files are passed to the existing ``gr_xyz``/``elemse`` programs.

    Progress reporting
    ------------------
    ``progress`` controls the native BayesISOLA progress bars used for Axitra
    Green's-function calculation and moment-tensor inversion.  Syngine retains
    its backend-specific ``gf_options['progress']`` control.

    Station radius and waveforms
    ----------------------------
    ``max_radius_km=None`` calls :func:`get_max_radius` with ``magnitude`` and
    ``radius_scale_factor``. In FDSN mode :func:`get_mseed_stationxml` performs
    ordered multi-client station discovery, determines the BayesISOLA-safe
    origin-centred waveform window, downloads/validates miniSEED+StationXML and
    returns the authoritative station table. In local mode ``station_df`` must
    already contain those files. Both branches converge through
    :func:`load_streams_local`.

    Depth grid
    ----------
    The shallow grid bound is
    ``max(min_depth_km, event_depth_km * min_depth_multiplier)`` and the deep
    bound is ``event_depth_km * max_depth_multiplier``. ``max_grid_points`` is
    BayesISOLA's approximate pre-construction rescaling target; the realised
    discrete grid remains authoritative.

    Waveform window and covariance
    ------------------------------
    ``covariance='noise'`` requests the long pre-event noise interval.
    ``crosscovariance=False`` retains only component-wise noise covariance,
    whereas ``True`` retains the full three-component station covariance.
    ``covariance='none'`` avoids the noise interval and invokes BayesISOLA's
    unweighted ordinary least-squares branch. ``velocity_slowest_m_s`` is passed
    to ``process_data`` so acquisition and BayesISOLA use the same timing bound.

    Outputs and uncertainty
    -----------------------
    Deterministic centroid, moment-tensor summary, station/component fit and
    grid-edge tables are always written. ``n_uncertainty`` remains available for
    the native BayesISOLA posterior sampler and is returned/written separately
    when requested, but uncertainty-derived ``±`` values are deliberately not
    mixed into :func:`plot_cmt_summary` in version 0.1.7. Native uncertainty plots
    also remain disabled in all helper plot presets. ``plot=True`` saves figures;
    ``plot_preset`` is ``'none'``, ``'summary'`` or ``'full'``; ``show=True``
    additionally displays saved figures in the notebook.

    Returns
    -------
    dict
        Native BayesISOLA objects plus acquisition diagnostics, ``run['gf']``
        backend metadata, crust/model information where applicable, curated
        ``run['results']`` tables, result paths and figure paths.
    """
    import BayesISOLA

    waveform_source = str(waveform_source).lower().strip()
    covariance = str(covariance).lower().strip()
    plot_preset = str(plot_preset).lower().strip()
    gf_source, normalized_gf_options = _normalize_gf_options(
        gf_source, gf_options, surface_depth_km=float(surface_depth_km)
    )
    if waveform_source not in {"fdsn", "local"}:
        raise ValueError("waveform_source must be 'fdsn' or 'local'.")
    if covariance not in {"none", "noise"}:
        raise ValueError("covariance must be 'none' or 'noise'.")
    if plot_preset not in PLOT_PRESETS:
        raise ValueError(f"plot_preset must be one of {tuple(PLOT_PRESETS)}.")
    if float(freqmin) < 0 or float(freqmax) <= float(freqmin):
        raise ValueError("Require 0 <= freqmin < freqmax.")
    if waveform_source == "local" and station_df is None:
        raise ValueError("waveform_source='local' requires station_df from get_mseed_stationxml or an equivalent local-file table.")
    if use_precalculated_Green not in {False, True, "auto"}:
        raise ValueError("use_precalculated_Green must be False, True or 'auto'.")

    resolved_max_radius_km = _resolve_max_radius_km(float(magnitude), max_radius_km, radius_scale_factor)
    output_path = Path(output_dir).expanduser()
    raw_path, metadata_path, input_path, results_path, figure_path = (
        output_path / "raw", output_path / "metadata", output_path / "input",
        output_path / "results", output_path / "figures",
    )
    for directory in (output_path, raw_path, metadata_path, input_path, results_path):
        directory.mkdir(parents=True, exist_ok=True)
    if plot and plot_preset != "none":
        figure_path.mkdir(parents=True, exist_ok=True)
    network_path = input_path / "network.stn"
    case_crust_file = input_path / "crustal.dat"

    inputs = BayesISOLA.load_data(outdir=str(output_path))
    inputs.set_event_info(
        lat=float(event_lat), lon=float(event_lon), depth=float(event_depth_km),
        mag=float(magnitude), t=origin_time,
    )
    inputs.set_source_time_function(str(source_time_function))

    minimum_pre_event_s = 20.0 if detect_mouse else 0.0
    acquisition_window = None
    if waveform_source == "fdsn":
        event_df, stations, download_log, acquisition_window = get_mseed_stationxml(
            event_id, origin_time, event_lon, event_lat, event_depth_km,
            magnitude=float(magnitude), output_dir=output_path, station_df=station_df,
            client=client, min_radius_km=min_radius_km,
            max_radius_km=resolved_max_radius_km,
            radius_scale_factor=radius_scale_factor, ground_level=ground_level,
            channels=channels, channel_priority=channel_priority,
            taup_model=taup_model, min_depth_km=min_depth_km,
            min_depth_multiplier=min_depth_multiplier,
            max_depth_multiplier=max_depth_multiplier, time_unc_s=time_unc_s,
            rupture_velocity_m_s=rupture_velocity_m_s,
            velocity_slowest_m_s=velocity_slowest_m_s, covariance=covariance,
            noise_factor=noise_factor, edge_margin_s=edge_margin_s,
            minimum_pre_event_s=minimum_pre_event_s,
            overwrite=bool(overwrite_waveforms), plot=False, show=False,
        )
    else:
        stations = _coerce_station_df(station_df)
        download_log = pd.DataFrame()
        event_df = pd.DataFrame([{
            "event_id": str(event_id), "origin_time": str(origin_time),
            "event_lon": float(event_lon), "event_lat": float(event_lat),
            "event_depth_km": float(event_depth_km), "magnitude": float(magnitude),
            "min_radius_km": float(min_radius_km),
            "max_radius_km": resolved_max_radius_km,
            "radius_scale_factor": float(radius_scale_factor),
            "ground_level": bool(ground_level),
        }])

    crust_profile = None
    crust_layers = None
    axitra_model_manifest = None
    axitra_model_manifest_path = None

    if gf_source == "axitra":
        axitra_grid_mode = normalized_gf_options["grid"]
        if axitra_grid_mode is None:
            case_crust_file, crust_profile, crust_layers = _prepare_crust_file(
                case_crust_file, event_lon=float(event_lon), event_lat=float(event_lat),
                crust_file=crust_file, velocity_model=velocity_model,
                profile_crs=profile_crs, depth_col=velocity_depth_col,
                vp_col=vp_col, vs_col=vs_col, density_col=density_col,
                qp_col=qp_col, qs_col=qs_col,
                surface_depth_km=float(normalized_gf_options["surface_depth_km"]),
            )
        else:
            if crust_file is not None:
                raise ValueError(
                    "crust_file cannot be combined with station/path-dependent Axitra models; "
                    "supply the 3-D velocity_model instead."
                )
            stations, axitra_model_manifest = _prepare_axitra_station_models(
                stations, event_lon=float(event_lon), event_lat=float(event_lat),
                velocity_model=velocity_model, output_file=case_crust_file,
                options=normalized_gf_options, vp_col=vp_col, vs_col=vs_col,
                density_col=density_col, qp_col=qp_col, qs_col=qs_col,
            )
            axitra_model_manifest_path = metadata_path / "gf_axitra_models.csv"
            axitra_model_manifest.to_csv(axitra_model_manifest_path, index=False)

    write_network_file(stations, network_path)
    inputs.read_network_coordinates(
        str(network_path), min_distance=float(min_radius_km) * 1000.0,
        max_distance=resolved_max_radius_km * 1000.0, max_n_of_stations=None,
    )
    if not inputs.stations:
        raise ValueError("No stations remain after BayesISOLA network-distance filtering.")

    # BayesISOLA determines model filenames from the station-model tags populated
    # by read_network_coordinates, so crust copying must occur after the network
    # has been read. The 0.1.6 ordering copied before those model keys existed.
    if gf_source == "axitra":
        inputs.read_crust(str(case_crust_file))

    grid_radius_km = float(location_unc_km) + (
        inputs.rupture_length / 1000.0 if add_rupture_length else 0.0
    )
    depth_spec = suggest_depth_limits(
        float(event_depth_km), min_depth_km=float(min_depth_km),
        min_depth_multiplier=float(min_depth_multiplier),
        max_depth_multiplier=float(max_depth_multiplier),
        step_z_km=float(step_z_km), step_x_km=float(step_x_km),
        radius_km=grid_radius_km, max_points=int(max_grid_points),
    )
    grid = BayesISOLA.grid(
        inputs, location_unc=float(location_unc_km) * 1000.0,
        depth_unc=depth_spec["depth_unc_km"] * 1000.0,
        time_unc=float(time_unc_s), step_x=float(step_x_km) * 1000.0,
        step_z=float(step_z_km) * 1000.0, max_points=int(max_grid_points),
        grid_min_depth=depth_spec["grid_min_depth_km"] * 1000.0,
        grid_max_depth=depth_spec["grid_max_depth_km"] * 1000.0,
        circle_shape=True, add_rupture_length=bool(add_rupture_length),
        rupture_velocity=float(rupture_velocity_m_s),
    )

    waveform_window = get_waveform_window(
        float(event_depth_km), float(magnitude), station_df=stations,
        radius_scale_factor=radius_scale_factor, min_depth_km=min_depth_km,
        min_depth_multiplier=min_depth_multiplier,
        max_depth_multiplier=max_depth_multiplier, time_unc_s=time_unc_s,
        rupture_velocity_m_s=rupture_velocity_m_s,
        velocity_slowest_m_s=velocity_slowest_m_s, covariance=covariance,
        noise_factor=noise_factor, edge_margin_s=edge_margin_s,
        minimum_pre_event_s=minimum_pre_event_s,
    )
    loaded_stations, load_log = load_streams_local(
        inputs, stations, t_before=waveform_window["t_before_s"],
        t_after=waveform_window["t_after_s"],
    )
    write_network_file(loaded_stations, network_path)
    loaded_stations.to_csv(metadata_path / "stations_loaded.csv", index=False)
    load_log.to_csv(metadata_path / "load_log.csv", index=False)
    pd.DataFrame([waveform_window]).to_csv(
        metadata_path / "waveform_window_required.csv", index=False
    )

    if gf_source == "axitra":
        _prune_inactive_bayesisola_models(inputs)
        if axitra_model_manifest is not None:
            active_ids = set(loaded_stations.get("station_id", pd.Series(dtype=str)).astype(str))
            if active_ids:
                axitra_model_manifest["active"] = axitra_model_manifest["station_id"].astype(str).isin(active_ids)
            else:
                active_keys = {
                    (str(row["network"]), str(row["station"]), _normalize_location(row.get("location")))
                    for row in loaded_stations.to_dict("records")
                }
                axitra_model_manifest["active"] = [
                    (str(row["network"]), str(row["station"]), _normalize_location(row.get("location"))) in active_keys
                    for row in axitra_model_manifest.to_dict("records")
                ]
            axitra_model_manifest.to_csv(axitra_model_manifest_path, index=False)

    custom_figures: list[Path] = []
    if plot and plot_preset != "none":
        custom_figures.append(
            plot_waveform_section(
                loaded_stations, origin_time,
                figure_path / "waveform_record_section_unfiltered.png", show=False,
            )
        )

    mouse_figures = output_path / "mouse" if plot and plot_preset == "full" else False
    if detect_mouse:
        inputs.detect_mouse(figures=mouse_figures)

    use_noise = covariance == "noise"
    data = BayesISOLA.process_data(
        inputs, grid, threads=int(threads), progress=bool(progress),
        invert_displacement=bool(invert_displacement),
        use_precalculated_Green=use_precalculated_Green,
        velocity_ot_the_slowest_wave=float(velocity_slowest_m_s),
        fmax=float(freqmax), fmin=float(freqmin),
        min_depth=depth_spec["grid_min_depth_km"] * 1000.0,
        calculate_or_verify_Green=False,
        trim_filter_data=False, decimate_shift=False,
    )

    if gf_source == "axitra":
        if normalized_gf_options["grid"] is None:
            before_green = _axitra_green_state(data)
            data.calculate_or_verify_Green()
            after_green = _axitra_green_state(data)
            reused = bool(use_precalculated_Green is not False and before_green and before_green == after_green)
            model_name = Path(case_crust_file).name if crust_file is not None else "event_profile"
        else:
            reused = _calculate_or_verify_axitra_multimodel(data, use_precalculated_Green)
            model_name = f"{normalized_gf_options['grid']}_profiles"
        gf_info = {
            "source": "axitra",
            "model": model_name,
            "path": str(inputs.green_dir),
            "reused": bool(reused),
            "cache_policy": use_precalculated_Green,
            "options": dict(normalized_gf_options),
            "n_models": int(len(inputs.models)),
            "manifest": str(axitra_model_manifest_path) if axitra_model_manifest_path is not None else None,
        }
    else:
        gf_info = _prepare_syngine_greens(
            data, output_path=output_path, metadata_path=metadata_path,
            source_time_function=str(source_time_function),
            options=normalized_gf_options,
            use_precalculated_Green=use_precalculated_Green,
        )

    data.trim_filter_data(noise_slice=use_noise)
    data.decimate_shift()

    if plot and plot_preset != "none":
        custom_figures.append(
            plot_station_section(
                data, loaded_stations,
                figure_path / "waveform_record_section.png", show=False,
            )
        )

    cova = BayesISOLA.covariance_matrix(data)
    if use_noise:
        cova.covariance_matrix_noise(
            crosscovariance=bool(crosscovariance),
            save_non_inverted=bool(plot and plot_preset == "full"),
            save_covariance_function=False,
        )

    solution = BayesISOLA.resolve_MT(
        data, cova, deviatoric=bool(deviatoric), VR_of_components=True
    )
    grid_edge_report = diagnose_grid_edge(grid, centroid=solution.centroid)
    results = _build_results(
        solution, n_uncertainty=n_uncertainty,
        grid_edge_report=grid_edge_report,
    )
    result_paths = _write_result_tables(results, results_path)
    result_paths["grid_edge_diagnostic"] = results_path / "grid_edge_diagnostic.csv"
    results["grid_edge_report"].to_csv(
        result_paths["grid_edge_diagnostic"], index=False
    )

    if plot and plot_preset != "none":
        cmt_summary_path = figure_path / "cmt_summary.png"
        cmt_figure = plot_cmt_summary(
            results["summary"], results["centroid"],
            tensor_mode="deviatoric" if deviatoric else "full",
            output_file=cmt_summary_path, show=False,
        )
        import matplotlib.pyplot as plt
        plt.close(cmt_figure)
        custom_figures.insert(0, cmt_summary_path)

    plot_object = None
    native_figures: list[Path] = []
    if plot and plot_preset != "none":
        before = _png_state(output_path)
        preset = dict(PLOT_PRESETS[plot_preset])
        if not use_noise:
            preset.update(
                seismo_cova=False, noise=False, spectra=False,
                covariance_matrix=False, covariance_function=False,
            )
        plot_object = BayesISOLA.plot(solution, **preset)
        plot_object.html_log(
            h1=f"BayesISOLA CMT — {event_id}",
            mouse_figures="mouse/" if detect_mouse and plot_preset == "full" else None,
        )
        native_figures = _changed_pngs(output_path, before)

    figure_paths = list(dict.fromkeys([*custom_figures, *native_figures]))
    if show and figure_paths:
        _display_saved_figures(figure_paths)

    return {
        "inputs": inputs, "grid": grid, "data": data, "cova": cova,
        "solution": solution, "gf": gf_info,
        "event_df": event_df, "station_df": loaded_stations,
        "download_log": download_log, "load_log": load_log,
        "max_radius_km": resolved_max_radius_km, "depth_spec": depth_spec,
        "acquisition_window": acquisition_window,
        "waveform_window": waveform_window,
        "grid_edge_report": grid_edge_report,
        "crust_file": case_crust_file if gf_source == "axitra" else None,
        "crust_profile": crust_profile, "crust_layers": crust_layers,
        "gf_model_manifest": axitra_model_manifest,
        "results": results, "result_paths": result_paths,
        "figure_paths": figure_paths, "plot": plot_object,
    }

