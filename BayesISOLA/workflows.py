"""High-level workflows for automated BayesISOLA CMT inversion.

Version 0.2.0 extends the validated native BayesISOLA/Axitra inversion with
bounded adaptive centroid searches, exact discrete posterior outputs, calibrated
conditional moment-tensor sampling, station-selection controls, fast exact
fixed-grid station jackknife diagnostics, workflow-level diagnostic figures and
curated HTML reporting.  Native BayesISOLA plotting and ``plot.html_log()`` are
retained independently for backward-compatible scientific inspection.

The workflow layer coordinates these capabilities without changing the native
point-source moment-tensor parameterization or Axitra Green-function algorithms.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import math
import shutil
import hashlib
import io

import numpy as np
import pandas as pd

from BayesISOLA._diagnostics import (
    plot_adaptive_history,
    plot_cmt_summary,
    plot_posterior_summary,
    plot_station_fit_summary,
    plot_station_qc,
    plot_uncertainty_summary,
    summarize_uncertainty,
)
from BayesISOLA._report import write_html_report


__version__ = "0.2.0"

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
    "compute_grid_expansion",
    "compute_grid_refinement",
    "build_posterior_cells",
    "compute_posterior_diagnostics",
    "extract_station_fit_df",
    "extract_centroid_location",
    "extract_solution_summary",
    "extract_uncertainty_df",
    "write_solution_outputs",
    "plot_cmt_summary",
    "plot_posterior_summary",
    "plot_adaptive_history",
    "plot_station_qc",
    "summarize_uncertainty",
    "plot_uncertainty_summary",
    "plot_station_fit_summary",
    "write_html_report",
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
    grid_min_depth_km: float | None = None,
    grid_max_depth_km: float | None = None,
) -> tuple[float, float]:
    """Resolve BayesISOLA depth limits, with optional explicit overrides.

    ``None`` preserves the automated 0.1.1 depth bounds.  Explicit bounds are
    useful for controlled/adaptive reruns and are still constrained by the
    absolute ``min_depth_km`` floor on the shallow side.
    """
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

    auto_min = max(min_depth_km, catalog_depth_km * min_depth_multiplier)
    auto_max = catalog_depth_km * max_depth_multiplier

    if grid_min_depth_km is None:
        resolved_min = auto_min
    else:
        resolved_min = max(min_depth_km, float(grid_min_depth_km))

    if grid_max_depth_km is None:
        resolved_max = auto_max
    else:
        resolved_max = float(grid_max_depth_km)

    if not np.isfinite([resolved_min, resolved_max]).all():
        raise ValueError("Explicit grid depth bounds must be finite or None.")
    if resolved_max <= resolved_min:
        raise ValueError(
            "The requested depth controls give grid_max_depth_km <= grid_min_depth_km."
        )
    return resolved_min, resolved_max

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





ChannelPriority = Sequence[str] | Mapping[str, Any]

_CHANNEL_PRIORITY_RULE_KEYS = {"mag_range", "dist_range", "channels", "default"}


def _channel_prefixes_from_patterns(channels: Sequence[str]) -> tuple[str, ...]:
    """Return ordered two-character channel families implied by FDSN patterns.

    The first two characters of each channel pattern define the family used by
    BayesISOLA (for example ``HH?`` -> ``HH`` and ``HN?`` -> ``HN``). Duplicate
    families are removed while preserving the caller's order.
    """
    prefixes: list[str] = []
    for pattern in channels:
        text = str(pattern).strip()
        if len(text) < 2 or any(char in "?*" for char in text[:2]):
            raise ValueError(
                f"Channel pattern {pattern!r} must begin with a literal two-character family prefix."
            )
        prefix = text[:2]
        if prefix not in prefixes:
            prefixes.append(prefix)
    if not prefixes:
        raise ValueError("channels cannot be empty.")
    return tuple(prefixes)


def _validate_priority_prefixes(values: Sequence[str], *, name: str) -> tuple[str, ...]:
    """Normalize and validate one ordered channel-family priority sequence."""
    if isinstance(values, str):
        raise TypeError(f"{name} must be a sequence of two-character prefixes, not a string.")
    prefixes = tuple(str(value).strip() for value in values)
    if not prefixes:
        raise ValueError(f"{name} cannot be empty.")
    if any(len(prefix) != 2 for prefix in prefixes):
        raise ValueError(f"{name} entries must be two-character prefixes.")
    if len(set(prefixes)) != len(prefixes):
        raise ValueError(f"{name} cannot contain duplicate prefixes.")
    return prefixes


def _normalize_channel_priority(
    channel_priority: ChannelPriority,
    channels: Sequence[str],
) -> dict[str, Any]:
    """Normalize static or magnitude/distance-dependent channel priority.

    ``channel_priority`` retains its historical ordered-sequence form, e.g.::

        ("HH", "BH", "LH")

    It may alternatively be a mapping containing parallel ``mag_range``,
    ``dist_range`` and ``channels`` rule lists, with an optional ``default``
    priority::

        {
            "mag_range": [[4.0, 5.0], [5.0, 6.0]],
            "dist_range": [[10, 250], [40, 300]],
            "channels": [["BH", "HH"], ["HN", "BN"]],
        }

    Rule ranges use half-open intervals ``[minimum, maximum)`` and the first
    matching rule wins. A rule changes precedence, not exclusivity: if its named
    families are unavailable, the station falls back through the default order.
    When ``default`` is omitted, that fallback is inferred from the caller's
    ordered FDSN ``channels`` patterns.

    In rule mode only, family prefixes named by a rule/default but absent from the
    outer ``channels`` patterns are automatically added to the metadata query as
    ``XX?``. This lets the mapping be self-contained. Historical sequence mode is
    left unchanged and does not alter the caller's FDSN query patterns.
    """
    channel_patterns = tuple(str(value).strip() for value in channels)
    if not channel_patterns or any(not value for value in channel_patterns):
        raise ValueError("channels cannot be empty or contain empty patterns.")

    if not isinstance(channel_priority, Mapping):
        priority = _validate_priority_prefixes(channel_priority, name="channel_priority")
        return {
            "mode": "static",
            "eligible": priority,
            "default": priority,
            "rules": (),
            "query_patterns": channel_patterns,
        }

    explicit_prefixes = _channel_prefixes_from_patterns(channel_patterns)
    unknown = sorted(set(channel_priority) - _CHANNEL_PRIORITY_RULE_KEYS)
    if unknown:
        raise ValueError(
            f"Unknown channel_priority rule key(s): {', '.join(map(str, unknown))}. "
            f"Allowed keys are: {', '.join(sorted(_CHANNEL_PRIORITY_RULE_KEYS))}."
        )

    required = {"mag_range", "dist_range", "channels"}
    missing_keys = sorted(required - set(channel_priority))
    if missing_keys:
        raise ValueError(
            "Rule-based channel_priority requires key(s): " + ", ".join(missing_keys)
        )

    mag_ranges = list(channel_priority["mag_range"])
    dist_ranges = list(channel_priority["dist_range"])
    rule_priorities = list(channel_priority["channels"])
    if not mag_ranges:
        raise ValueError("Rule-based channel_priority must contain at least one rule.")
    if not (len(mag_ranges) == len(dist_ranges) == len(rule_priorities)):
        raise ValueError(
            "channel_priority mag_range, dist_range and channels must have equal lengths."
        )

    default = _validate_priority_prefixes(
        channel_priority.get("default", explicit_prefixes),
        name="channel_priority['default']",
    )

    parsed_rules: list[dict[str, Any]] = []
    eligible_order = list(default)
    for index, (mag_range, dist_range, priority_values) in enumerate(
        zip(mag_ranges, dist_ranges, rule_priorities)
    ):
        if len(mag_range) != 2 or len(dist_range) != 2:
            raise ValueError(
                f"channel_priority rule {index} magnitude and distance ranges must each contain [min, max]."
            )
        mag_min, mag_max = map(float, mag_range)
        dist_min, dist_max = map(float, dist_range)
        if not np.isfinite([mag_min, mag_max, dist_min, dist_max]).all():
            raise ValueError(f"channel_priority rule {index} ranges must be finite.")
        if mag_min >= mag_max:
            raise ValueError(f"channel_priority rule {index} requires mag_min < mag_max.")
        if dist_min < 0.0 or dist_min >= dist_max:
            raise ValueError(
                f"channel_priority rule {index} requires 0 <= dist_min < dist_max."
            )

        priority = _validate_priority_prefixes(
            priority_values,
            name=f"channel_priority['channels'][{index}]",
        )
        for prefix in priority:
            if prefix not in eligible_order:
                eligible_order.append(prefix)
        parsed_rules.append({
            "index": index,
            "mag_min": mag_min,
            "mag_max": mag_max,
            "dist_min": dist_min,
            "dist_max": dist_max,
            "requested_priority": priority,
        })

    rules: list[dict[str, Any]] = []
    for rule in parsed_rules:
        priority = tuple(rule["requested_priority"])
        resolved = priority + tuple(prefix for prefix in default if prefix not in priority)
        rules.append({
            "index": int(rule["index"]),
            "mag_min": float(rule["mag_min"]),
            "mag_max": float(rule["mag_max"]),
            "dist_min": float(rule["dist_min"]),
            "dist_max": float(rule["dist_max"]),
            "priority": resolved,
        })

    query_patterns = list(channel_patterns)
    queried_prefixes = list(explicit_prefixes)
    for prefix in eligible_order:
        if prefix not in queried_prefixes:
            query_patterns.append(f"{prefix}?")
            queried_prefixes.append(prefix)

    return {
        "mode": "rules",
        "eligible": tuple(eligible_order),
        "default": default,
        "rules": tuple(rules),
        "query_patterns": tuple(query_patterns),
    }

def _resolve_channel_priority(
    config: Mapping[str, Any],
    *,
    magnitude: float | None,
    distance_km: float | None,
) -> tuple[tuple[str, ...], int | None]:
    """Return the priority order and matching rule index for one station.

    Rule evaluation requires both a finite event magnitude and station distance.
    If either is unavailable, the normalized default order is used.
    """
    default = tuple(config["default"])
    if config.get("mode") != "rules" or magnitude is None or distance_km is None:
        return default, None

    magnitude = float(magnitude)
    distance_km = float(distance_km)
    if not np.isfinite([magnitude, distance_km]).all():
        return default, None

    for rule in config["rules"]:
        if (
            float(rule["mag_min"]) <= magnitude < float(rule["mag_max"])
            and float(rule["dist_min"]) <= distance_km < float(rule["dist_max"])
        ):
            return tuple(rule["priority"]), int(rule["index"])
    return default, None

_AZIMUTH_CONTROL_DEFAULTS: dict[str, Any] = {
    "azimuth_selection": True,
    "azimuth_min_sectors": 3,
    "azimuth_max_stations_per_sector": 2,
    "minimum_stations": 4,
}


def _normalize_azimuth_control(
    azimuth_control: bool | Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Normalize optional GISOLA-style azimuthal station selection.

    ``None``/``False`` disables selection, ``True`` uses the defaults, and a
    mapping enables selection while overriding only named defaults.  The sector
    width is deliberately fixed at GISOLA's 45 degrees (eight sectors); channel
    family preference remains the caller's existing ``channel_priority``.
    """
    if azimuth_control is None or azimuth_control is False:
        config = dict(_AZIMUTH_CONTROL_DEFAULTS)
        config["azimuth_selection"] = False
        return config
    if azimuth_control is True:
        config = dict(_AZIMUTH_CONTROL_DEFAULTS)
    elif isinstance(azimuth_control, Mapping):
        unknown = sorted(set(azimuth_control) - set(_AZIMUTH_CONTROL_DEFAULTS))
        if unknown:
            allowed = ", ".join(_AZIMUTH_CONTROL_DEFAULTS)
            raise ValueError(
                f"Unknown azimuth_control option(s): {', '.join(map(str, unknown))}. "
                f"Allowed keys are: {allowed}."
            )
        config = dict(_AZIMUTH_CONTROL_DEFAULTS)
        config.update(dict(azimuth_control))
    else:
        raise TypeError("azimuth_control must be None, a boolean, or a mapping.")

    config["azimuth_selection"] = bool(config["azimuth_selection"])
    config["azimuth_min_sectors"] = int(config["azimuth_min_sectors"])
    config["azimuth_max_stations_per_sector"] = int(config["azimuth_max_stations_per_sector"])
    config["minimum_stations"] = int(config["minimum_stations"])
    if not 1 <= config["azimuth_min_sectors"] <= 8:
        raise ValueError("azimuth_min_sectors must lie within [1, 8].")
    if config["azimuth_max_stations_per_sector"] < 1:
        raise ValueError("azimuth_max_stations_per_sector must be >= 1.")
    if config["minimum_stations"] < 1:
        raise ValueError("minimum_stations must be >= 1.")
    return config


def _station_identity(row: Mapping[str, Any]) -> str:
    """Return a normalized ``NET.STA.LOC`` identity for one station row."""
    location = _normalize_location(row.get("location"))
    return f"{str(row.get('network', '')).strip()}.{str(row.get('station', '')).strip()}.{location or '--'}"


def _station_drop_mask(table: pd.DataFrame, specification: str) -> np.ndarray:
    """Return the unique station-row mask addressed by one drop specification.

    Accepted forms are ``STA``, ``NET.STA`` and ``NET.STA.LOC``.  Bare or
    two-field identifiers must resolve to one network/station/location identity;
    this avoids silently dropping multiple unrelated stations when codes collide.
    """
    specification = str(specification).strip()
    if not specification:
        raise ValueError("drop_stations cannot contain empty station identifiers.")
    parts = specification.split(".")
    network = table["network"].astype(str).str.strip()
    station = table["station"].astype(str).str.strip()
    locations = table["location"].map(_normalize_location)

    if len(parts) == 1:
        mask = station.eq(parts[0]).to_numpy()
    elif len(parts) == 2:
        mask = (network.eq(parts[0]) & station.eq(parts[1])).to_numpy()
    elif len(parts) == 3:
        location = _normalize_location(parts[2])
        mask = (network.eq(parts[0]) & station.eq(parts[1]) & locations.eq(location)).to_numpy()
    else:
        raise ValueError(
            f"Invalid drop_stations identifier {specification!r}; use STA, NET.STA or NET.STA.LOC."
        )

    if not np.any(mask):
        raise ValueError(f"Requested drop station {specification!r} did not match any retained station.")
    identities = {_station_identity(row) for row in table.loc[mask].to_dict("records")}
    if len(identities) != 1:
        raise ValueError(
            f"Requested drop station {specification!r} is ambiguous across {sorted(identities)}; "
            "use NET.STA.LOC."
        )
    return mask


def _ensure_station_geometry(
    table: pd.DataFrame,
    *,
    event_lat: float,
    event_lon: float,
) -> pd.DataFrame:
    """Ensure finite distance/azimuth columns using authoritative station coordinates."""
    table = table.copy()
    if "distance_km" not in table:
        table["distance_km"] = np.nan
    if "azimuth_deg" not in table:
        table["azimuth_deg"] = np.nan

    for index, row in table.iterrows():
        distance = pd.to_numeric(pd.Series([row.get("distance_km")]), errors="coerce").iloc[0]
        azimuth = pd.to_numeric(pd.Series([row.get("azimuth_deg")]), errors="coerce").iloc[0]
        if np.isfinite(distance) and np.isfinite(azimuth):
            table.at[index, "distance_km"] = float(distance)
            table.at[index, "azimuth_deg"] = float(azimuth) % 360.0
            continue
        if "station_lat" not in row or "station_lon" not in row:
            raise KeyError(
                "Azimuth selection requires distance_km/azimuth_deg or station_lat/station_lon."
            )
        from obspy.geodetics.base import gps2dist_azimuth
        station_lat = float(row["station_lat"])
        station_lon = float(row["station_lon"])
        if not np.isfinite([station_lat, station_lon]).all():
            raise ValueError("Station coordinates must be finite for azimuth selection.")
        distance_m, azimuth, _ = gps2dist_azimuth(
            float(event_lat), float(event_lon), station_lat, station_lon
        )
        table.at[index, "distance_km"] = float(distance_m) / 1000.0
        table.at[index, "azimuth_deg"] = float(azimuth) % 360.0
    return table


def _apply_station_controls(
    station_df: pd.DataFrame | str | Path,
    *,
    drop_stations: Sequence[str] | str | None,
    azimuth_config: Mapping[str, Any],
    channel_priority: ChannelPriority,
    channels: Sequence[str],
    magnitude: float | None,
    event_lat: float,
    event_lon: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply manual drops first, then optional eight-sector azimuthal thinning.

    Channel-family ranking is evaluated per station so magnitude/distance-based
    ``channel_priority`` rules are respected consistently by the azimuth selector.
    """
    table = _coerce_station_df(station_df).reset_index(drop=True)
    table = _ensure_station_geometry(table, event_lat=event_lat, event_lon=event_lon)
    audit = table.copy()
    audit["selection_status"] = "selected"
    audit["selection_reason"] = ""
    audit["azimuth_sector"] = np.floor((audit["azimuth_deg"].astype(float) % 360.0) / 45.0).astype(int)

    priority_config = _normalize_channel_priority(channel_priority, channels)
    prefixes = audit.get("channel_prefix", pd.Series("", index=audit.index)).astype(str).str[:2]
    priority_ranks = []
    priority_rules = []
    for prefix, distance_km in zip(prefixes, audit["distance_km"]):
        order, rule_index = _resolve_channel_priority(
            priority_config, magnitude=magnitude, distance_km=float(distance_km)
        )
        lookup = {value: rank for rank, value in enumerate(order)}
        priority_ranks.append(lookup.get(prefix, len(lookup)))
        priority_rules.append(rule_index)
    audit["channel_priority_rank"] = priority_ranks
    audit["channel_priority_rule"] = priority_rules

    dropped_indices: set[int] = set()
    if drop_stations is not None:
        specifications = [drop_stations] if isinstance(drop_stations, str) else list(drop_stations)
        for specification in specifications:
            active = audit.loc[~audit.index.isin(dropped_indices)]
            mask_local = _station_drop_mask(active, specification)
            matched_indices = active.index[np.asarray(mask_local, dtype=bool)].tolist()
            dropped_indices.update(matched_indices)
            audit.loc[matched_indices, "selection_status"] = "manual_drop"
            audit.loc[matched_indices, "selection_reason"] = f"drop_stations={str(specification).strip()}"

    candidate = audit.loc[~audit.index.isin(dropped_indices)].copy()
    if candidate.empty:
        raise ValueError("No stations remain after applying drop_stations.")

    if bool(azimuth_config["azimuth_selection"]):
        occupied = int(candidate["azimuth_sector"].nunique())
        if occupied < int(azimuth_config["azimuth_min_sectors"]):
            raise ValueError(
                f"Azimuthal coverage has {occupied} occupied 45-degree sectors, below "
                f"azimuth_min_sectors={int(azimuth_config['azimuth_min_sectors'])}."
            )

        selected_indices: list[int] = []
        max_per_sector = int(azimuth_config["azimuth_max_stations_per_sector"])
        for _, sector in candidate.groupby("azimuth_sector", sort=True):
            ordered = sector.sort_values(
                ["channel_priority_rank", "distance_km", "network", "station", "location"],
                kind="stable",
            )
            selected_indices.extend(ordered.index[:max_per_sector].tolist())

        excluded = candidate.index.difference(selected_indices)
        audit.loc[excluded, "selection_status"] = "azimuth_excluded"
        audit.loc[excluded, "selection_reason"] = (
            f"sector cap={max_per_sector} after channel_priority then distance ranking"
        )
        candidate = audit.loc[selected_indices].copy()
        if len(candidate) < int(azimuth_config["minimum_stations"]):
            raise ValueError(
                f"Azimuth selection retained {len(candidate)} stations, below "
                f"minimum_stations={int(azimuth_config['minimum_stations'])}."
            )

    selected = table.loc[candidate.index].copy().sort_values("distance_km", ignore_index=True)
    audit = audit.sort_values(["azimuth_sector", "channel_priority_rank", "distance_km", "network", "station"], ignore_index=True)
    return selected, audit


def _validate_selected_azimuth_geometry(
    station_df: pd.DataFrame,
    azimuth_config: Mapping[str, Any],
) -> None:
    """Validate the actually loaded subset after waveform-level station failures."""
    if not bool(azimuth_config["azimuth_selection"]):
        return
    if len(station_df) < int(azimuth_config["minimum_stations"]):
        raise ValueError(
            f"Only {len(station_df)} stations remain after waveform loading, below "
            f"azimuth_control minimum_stations={int(azimuth_config['minimum_stations'])}."
        )
    azimuth = pd.to_numeric(station_df["azimuth_deg"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(azimuth).all():
        raise ValueError("Loaded station azimuths must be finite when azimuth_control is enabled.")
    sectors = np.floor((azimuth % 360.0) / 45.0).astype(int)
    occupied = int(np.unique(sectors).size)
    if occupied < int(azimuth_config["azimuth_min_sectors"]):
        raise ValueError(
            f"Waveform loading reduced azimuthal coverage to {occupied} occupied sectors, below "
            f"azimuth_min_sectors={int(azimuth_config['azimuth_min_sectors'])}."
        )


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


def _inventory_channel_families(
    inventory,
    channel_priority: ChannelPriority | Mapping[str, Any],
    *,
    ground_level: bool,
    channels: Sequence[str] = ("HH?", "BH?", "LH?"),
    magnitude: float | None = None,
    event_lat: float | None = None,
    event_lon: float | None = None,
) -> pd.DataFrame:
    """Select one complete three-component family per station from StationXML.

    ``channel_priority`` may be the historical ordered family sequence or the
    normalized rule mapping returned by :func:`_normalize_channel_priority`.
    Rule-based priorities are evaluated separately for each station using the
    event magnitude and epicentral distance. This must happen at metadata
    selection time, before waveform download, because only the selected family
    is subsequently requested and processed.

    Within a channel family the orientation preference is ZNE, Z12, then 123.
    Selected components must have one common positive sample rate, sensor depth,
    sensor location and elevation. When ``ground_level=True``, StationXML
    ``Channel.depth`` must be zero within the package tolerance.
    """
    if isinstance(channel_priority, Mapping) and "mode" in channel_priority:
        config = dict(channel_priority)
    else:
        config = _normalize_channel_priority(channel_priority, channels)

    if config["mode"] == "rules" and (event_lat is None or event_lon is None):
        raise ValueError(
            "Rule-based channel_priority requires event_lat and event_lon during StationXML selection."
        )

    rows: list[dict[str, Any]] = []

    if config["mode"] == "rules":
        from obspy.geodetics.base import gps2dist_azimuth

    for network in inventory:
        for station in network:
            if config["mode"] == "rules":
                site_lat = float(station.latitude)
                site_lon = float(station.longitude)
                if not np.isfinite([site_lat, site_lon]).all():
                    raise ValueError(
                        f"Station {network.code}.{station.code} has non-finite site coordinates required by channel_priority rules."
                    )
                distance_m, _, _ = gps2dist_azimuth(
                    float(event_lat), float(event_lon), site_lat, site_lon
                )
                rule_distance_km = float(distance_m) / 1000.0
            else:
                rule_distance_km = None

            priority_order, rule_index = _resolve_channel_priority(
                config, magnitude=magnitude, distance_km=rule_distance_km
            )
            priority_lookup = {value: rank for rank, value in enumerate(priority_order)}

            grouped: dict[tuple[str, str], list[Any]] = {}
            for channel in station.channels:
                prefix = str(channel.code)[:2]
                # Metadata queries include every family needed by any rule, but
                # family eligibility is station-specific. A family that appears
                # only in a nonmatching rule must not leak into the default or a
                # different rule merely because it was present in StationXML.
                if prefix in priority_lookup:
                    grouped.setdefault((channel.location_code or "", prefix), []).append(channel)

            candidates = []
            for (location, prefix), family_channels in grouped.items():
                try:
                    scheme, selected_codes = _component_selection_from_codes(
                        [channel.code for channel in family_channels]
                    )
                except ValueError:
                    continue

                selected_objects = []
                for code in selected_codes:
                    matches = [channel for channel in family_channels if channel.code == code]
                    matches = [
                        channel for channel in matches
                        if np.isfinite(float(channel.sample_rate)) and float(channel.sample_rate) > 0
                    ]
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

                station_lat = float(latitudes.mean())
                station_lon = float(longitudes.mean())
                priority_rank = priority_lookup.get(prefix, len(priority_lookup))

                candidates.append({
                    "network": network.code,
                    "station": station.code,
                    "location": location,
                    "channel_prefix": prefix,
                    "component_scheme": scheme,
                    "selected_channels": ",".join(selected_codes),
                    "channels": ",".join(sorted({channel.code for channel in family_channels})),
                    "sample_rate": float(rates[0]),
                    "channel_depth_m": float(depths.mean()),
                    "station_lat": station_lat,
                    "station_lon": station_lon,
                    "station_elevation_m": float(elevations.mean()),
                    "site_lat": float(station.latitude),
                    "site_lon": float(station.longitude),
                    "site_elevation_m": float(station.elevation),
                    "priority": int(priority_rank),
                    "channel_priority_rule": rule_index,
                })

            if candidates:
                candidates.sort(
                    key=lambda item: (
                        item["priority"],
                        _COMPONENT_SCHEMES.index(item["component_scheme"]),
                        -item["sample_rate"],
                        item["location"],
                    )
                )
                rows.append(candidates[0])

    columns = [
        "network", "station", "location", "channel_prefix", "component_scheme",
        "selected_channels", "channels", "sample_rate", "channel_depth_m",
        "station_lat", "station_lon", "station_elevation_m", "site_lat", "site_lon",
        "site_elevation_m", "priority", "channel_priority_rule",
    ]
    return (
        pd.DataFrame(rows, columns=columns).sort_values(["network", "station"], ignore_index=True)
        if rows else pd.DataFrame(columns=columns)
    )

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
    channel_priority: ChannelPriority = ("HH", "BH", "LH"),
    taup_model: str = "iasp91",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Discover and select regional three-component stations without waveforms.

    ``client`` may be one ObsPy FDSN provider/client or an ordered sequence. Each
    provider is queried independently and the resulting station families are
    combined. One family is retained per network/station/location according to
    ``channel_priority`` and the orientation preference ZNE -> Z12 -> 123. The
    historical ordered prefix sequence remains supported. A mapping may instead
    provide parallel ``mag_range``, ``dist_range`` and ``channels`` rules; these
    are evaluated per station before waveform acquisition. Rule ranges are
    half-open ``[min, max)`` and the first matching rule wins.

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
    if any(not value for value in channel_patterns):
        raise ValueError("channels cannot contain empty patterns.")
    priority_config = _normalize_channel_priority(channel_priority, channel_patterns)
    channel_patterns = tuple(priority_config["query_patterns"])

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
            candidates = _inventory_channel_families(
                inventory, priority_config, ground_level=ground_level, channels=channel_patterns,
                magnitude=magnitude, event_lat=event_lat, event_lon=event_lon,
            )
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
    channel_priority: ChannelPriority = ("HH", "BH", "LH"),
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
    channel_priority: ChannelPriority = ("HH", "BH", "LH"),
    taup_model: str = "iasp91",
    min_depth_km: float = 5.0,
    min_depth_multiplier: float = 0.5,
    max_depth_multiplier: float = 3.0,
    grid_min_depth_km: float | None = None,
    grid_max_depth_km: float | None = None,
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

    ``channel_priority`` accepts either the historical ordered prefix sequence or
    the same magnitude/distance rule mapping as :func:`discover_stations`. Because
    the selected three-component family determines the actual FDSN request, rules
    are applied during StationXML discovery, before miniSEED is downloaded. Every
    family referenced only by a rule is automatically added to the FDSN metadata
    query as ``XX?``; no duplicate outer ``channels`` entry is required.

    Supplying ``station_df`` restricts acquisition to those authoritative
    station/channel rows; rule-based priority cannot substitute an alternate family
    that is not present in the supplied table. Successful files are validated for
    component completeness, common continuous
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
    if any(not value for value in channel_patterns):
        raise ValueError("channels cannot contain empty patterns.")
    priority_config = _normalize_channel_priority(channel_priority, channel_patterns)
    channel_patterns = tuple(priority_config["query_patterns"])

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
                candidates = _inventory_channel_families(
                    inventory, priority_config, ground_level=ground_level, channels=channel_patterns,
                    magnitude=magnitude, event_lat=event_lat, event_lon=event_lon,
                )
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
            min_depth_multiplier=min_depth_multiplier, max_depth_multiplier=max_depth_multiplier,
            grid_min_depth_km=grid_min_depth_km, grid_max_depth_km=grid_max_depth_km, time_unc_s=time_unc_s,
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
    grid_min_depth_km: float | None = None,
    grid_max_depth_km: float | None = None,
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
        max_depth_multiplier=max_depth_multiplier, grid_min_depth_km=grid_min_depth_km,
        grid_max_depth_km=grid_max_depth_km,
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
    grid_min_depth_km: float | None = None,
    grid_max_depth_km: float | None = None,
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
        max_depth_multiplier=max_depth_multiplier, grid_min_depth_km=grid_min_depth_km,
        grid_max_depth_km=grid_max_depth_km,
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


def _grid_boundary_flags(grid, gp: Mapping[str, Any]) -> dict[str, bool]:
    """Return dimension-specific boundary flags for one spatial grid point.

    Native BayesISOLA stores depth and horizontal boundaries in one ``edge``
    boolean.  A one-point horizontal grid (radius/steps equal to zero) is a
    deliberate fixed-XY constraint, not a horizontal boundary failure.
    """
    depths = sorted({float(point["z"]) for point in grid.grid if not point.get("err")})
    depth_lo = depths[0] if depths else np.nan
    depth_hi = depths[-1] if depths else np.nan
    step_x = float(grid.step_x)
    radius = float(grid.radius)
    n_steps = int(radius / step_x) if step_x > 0 else 0

    xy = {(float(point["x"]), float(point["y"])) for point in grid.grid if not point.get("err")}
    horizontal_fixed = len(xy) <= 1 or radius <= 0.0 or n_steps <= 0

    z = float(gp["z"])
    x = float(gp["x"])
    y = float(gp["y"])
    depth_floor = bool(depths) and np.isclose(z, depth_lo)
    depth_ceiling = bool(depths) and np.isclose(z, depth_hi)

    horizontal = False
    north = south = east = west = False
    if not horizontal_fixed and step_x > 0:
        i = int(round(x / step_x))
        j = int(round(y / step_x))
        index_edge = max(abs(i), abs(j)) == n_steps
        if bool(getattr(grid, "circle_shape", True)):
            radial_edge = (
                math.sqrt((abs(x) + step_x) ** 2 + y ** 2) > radius
                or math.sqrt((abs(y) + step_x) ** 2 + x ** 2) > radius
            )
        else:
            radial_edge = False
        horizontal = bool(index_edge or radial_edge)
        if horizontal:
            tol = 0.25 * step_x
            north = x > tol
            south = x < -tol
            east = y > tol
            west = y < -tol
            # A boundary point can lie on a curved sector without either
            # coordinate being near its absolute maximum.  Preserve the
            # dominant sign so directional diagnostics remain informative.
            if not (north or south or east or west):
                if abs(x) >= abs(y):
                    north = x >= 0
                    south = x < 0
                else:
                    east = y >= 0
                    west = y < 0

    return {
        "horizontal_search_fixed": horizontal_fixed,
        "on_horizontal_boundary": horizontal,
        "on_north_boundary": north,
        "on_south_boundary": south,
        "on_east_boundary": east,
        "on_west_boundary": west,
        "on_depth_floor": depth_floor,
        "on_depth_ceiling": depth_ceiling,
        "on_active_spatial_boundary": bool(horizontal or depth_floor or depth_ceiling),
    }


def diagnose_grid_edge(grid, centroid: dict[str, Any] | None = None) -> dict[str, Any]:
    """Disaggregate BayesISOLA spatial-grid boundary diagnostics.

    The 0.1.1 native ``gp['edge']`` flag combines horizontal and depth edges and
    also labels a deliberately fixed one-point XY grid as an edge.  This helper
    treats fixed XY as a constraint rather than a failed search and reports the
    active dimensions separately.
    """
    depths = sorted({float(gp["z"]) for gp in grid.grid if not gp.get("err")})
    depth_lo, depth_hi = (depths[0], depths[-1]) if depths else (None, None)
    step_x = float(grid.step_x)
    radius = float(grid.radius)
    n_steps = int(radius / step_x) if step_x > 0 else 0
    xy = {(float(gp["x"]), float(gp["y"])) for gp in grid.grid if not gp.get("err")}
    horizontal_fixed = len(xy) <= 1 or radius <= 0.0 or n_steps <= 0

    report: dict[str, Any] = {
        "realized_radius_km": radius / 1e3,
        "realized_step_x_km": step_x / 1e3,
        "realized_step_z_km": float(grid.step_z) / 1e3,
        "realized_depth_range_km": (
            (depth_lo / 1e3, depth_hi / 1e3) if depths else None
        ),
        "n_depth_levels_realized": len(depths),
        "n_steps_horizontal": n_steps,
        "horizontal_search_fixed": horizontal_fixed,
    }

    if centroid is not None:
        flags = _grid_boundary_flags(grid, centroid)
        report.update({f"centroid_{key}": value for key, value in flags.items()})
        reasons: list[str] = []
        if flags["horizontal_search_fixed"]:
            reasons.append("horizontal search fixed (single XY point; not treated as a boundary)")
        elif flags["on_horizontal_boundary"]:
            directions = [
                name for name, key in (
                    ("north", "on_north_boundary"), ("south", "on_south_boundary"),
                    ("east", "on_east_boundary"), ("west", "on_west_boundary"),
                ) if flags[key]
            ]
            suffix = f" ({'/'.join(directions)})" if directions else ""
            reasons.append(f"horizontal search boundary{suffix}")
        if flags["on_depth_floor"]:
            reasons.append(f"depth at grid floor ({depth_lo / 1e3:.3f} km)")
        if flags["on_depth_ceiling"]:
            reasons.append(f"depth at grid ceiling ({depth_hi / 1e3:.3f} km)")
        report["centroid_edge_reasons"] = reasons
    else:
        report["n_edge_points"] = sum(
            _grid_boundary_flags(grid, gp)["on_active_spatial_boundary"]
            for gp in grid.grid if not gp.get("err")
        )
    return report


def _grid_point_estimate(radius_km: float, step_x_km: float, depth_min_km: float,
                         depth_max_km: float, step_z_km: float) -> float:
    """Continuous BayesISOLA-style estimate used only for adaptive cost control."""
    radius_km = float(radius_km)
    step_x_km = float(step_x_km)
    step_z_km = float(step_z_km)
    depth_span_km = max(0.0, float(depth_max_km) - float(depth_min_km))
    if step_x_km <= 0 or step_z_km <= 0:
        raise ValueError("Grid spacings must be positive.")
    if radius_km <= 0:
        return max(1.0, depth_span_km / step_z_km)
    return math.pi * (radius_km / step_x_km) ** 2 * depth_span_km / step_z_km


_ADAPTIVE_GRID_SEARCH_DEFAULTS: dict[str, Any] = {
    "adaptive_grid": True,
    "adaptive_expand_xy_steps": 2,
    "adaptive_expand_z_steps": 2,
    "adaptive_max_expansions": 1,
    "adaptive_max_refinements": 1,
    "adaptive_refine_factor": 0.5,
    "adaptive_min_step_fraction": 0.25,
    "adaptive_depth_window_parent_steps": 3,
    "adaptive_max_radius_factor": 2.0,
    "adaptive_max_depth_span_factor": 1.5,
    "adaptive_max_grid_points": 20000,
    "adaptive_max_total_reruns": 2,
    "adaptive_expand_on_posterior_boundary": True,
    "adaptive_boundary_probability_threshold": 0.05,
}


def _normalize_adaptive_grid_search(
    adaptive_grid_search: bool | Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return a validated adaptive-search configuration.

    ``None``/``False`` preserves the non-adaptive workflow, ``True`` enables the
    validated defaults, and a mapping enables adaptive search while overriding
    only named settings. ``adaptive_grid=False`` inside a mapping remains a
    supported explicit disable switch.
    """
    if adaptive_grid_search is None or adaptive_grid_search is False:
        config = dict(_ADAPTIVE_GRID_SEARCH_DEFAULTS)
        config["adaptive_grid"] = False
        return config
    if adaptive_grid_search is True:
        adaptive_grid_search = {}
    if not isinstance(adaptive_grid_search, Mapping):
        raise TypeError("adaptive_grid_search must be None, a boolean, or a mapping.")

    unknown = sorted(set(adaptive_grid_search) - set(_ADAPTIVE_GRID_SEARCH_DEFAULTS))
    if unknown:
        allowed = ", ".join(_ADAPTIVE_GRID_SEARCH_DEFAULTS)
        raise ValueError(
            f"Unknown adaptive_grid_search option(s): {', '.join(map(str, unknown))}. "
            f"Allowed keys are: {allowed}."
        )

    config = dict(_ADAPTIVE_GRID_SEARCH_DEFAULTS)
    config.update(dict(adaptive_grid_search))
    config["adaptive_grid"] = bool(config["adaptive_grid"])
    config["adaptive_expand_on_posterior_boundary"] = bool(
        config["adaptive_expand_on_posterior_boundary"]
    )

    for key in ("adaptive_expand_xy_steps", "adaptive_expand_z_steps"):
        value = int(config[key])
        if value < 1:
            raise ValueError(f"{key} must be >= 1.")
        config[key] = value

    config["adaptive_max_expansions"] = int(config["adaptive_max_expansions"])
    if config["adaptive_max_expansions"] < 0:
        raise ValueError("adaptive_max_expansions cannot be negative.")
    config["adaptive_max_refinements"] = int(config["adaptive_max_refinements"])
    if not 0 <= config["adaptive_max_refinements"] <= 2:
        raise ValueError("adaptive_max_refinements must be 0, 1 or 2.")

    config["adaptive_refine_factor"] = float(config["adaptive_refine_factor"])
    if not 0.0 <= config["adaptive_refine_factor"] < 1.0:
        raise ValueError("adaptive_refine_factor must lie within [0, 1); 0 disables refinement.")
    config["adaptive_min_step_fraction"] = float(config["adaptive_min_step_fraction"])
    if not 0.25 <= config["adaptive_min_step_fraction"] <= 1.0:
        raise ValueError("adaptive_min_step_fraction must lie within [0.25, 1].")

    depth_steps = config["adaptive_depth_window_parent_steps"]
    if depth_steps is not None:
        depth_steps = int(depth_steps)
        if depth_steps < 1:
            raise ValueError("adaptive_depth_window_parent_steps must be >= 1 or None.")
    config["adaptive_depth_window_parent_steps"] = depth_steps

    for key in ("adaptive_max_radius_factor", "adaptive_max_depth_span_factor"):
        value = float(config[key])
        if value < 1.0:
            raise ValueError(f"{key} must be >= 1.")
        config[key] = value

    max_points = config["adaptive_max_grid_points"]
    if max_points is not None:
        max_points = int(max_points)
        if max_points <= 0:
            raise ValueError("adaptive_max_grid_points must be positive or None.")
    config["adaptive_max_grid_points"] = max_points

    config["adaptive_max_total_reruns"] = int(config["adaptive_max_total_reruns"])
    if config["adaptive_max_total_reruns"] < 0:
        raise ValueError("adaptive_max_total_reruns cannot be negative.")
    config["adaptive_boundary_probability_threshold"] = float(
        config["adaptive_boundary_probability_threshold"]
    )
    if not 0.0 <= config["adaptive_boundary_probability_threshold"] <= 1.0:
        raise ValueError("adaptive_boundary_probability_threshold must lie within [0, 1].")
    return config


def compute_grid_expansion(
    grid,
    centroid: Mapping[str, Any],
    *,
    initial_radius_km: float,
    initial_depth_min_km: float,
    initial_depth_max_km: float,
    expand_xy_steps: int = 2,
    expand_z_steps: int = 2,
    max_radius_factor: float = 2.0,
    max_depth_span_factor: float = 1.5,
    min_depth_km: float = 0.0,
    grid_point_budget: int | None = None,
    posterior_diagnostics: Mapping[str, Any] | None = None,
    boundary_probability_threshold: float = 0.05,
) -> dict[str, Any]:
    """Propose one bounded, catalogue-centred grid expansion.

    XY expansion is always symmetric about the catalogue epicentre; the grid is
    never re-centred on the current centroid.  Depth expands only on the active
    floor/ceiling side.  This function only proposes a grid -- it performs no
    Green-function calculation or inversion.
    """
    if int(expand_xy_steps) < 1 or int(expand_z_steps) < 1:
        raise ValueError("expand_xy_steps and expand_z_steps must be >= 1.")
    if float(max_radius_factor) < 1.0 or float(max_depth_span_factor) < 1.0:
        raise ValueError("Adaptive extent factors must be >= 1.")

    flags = _grid_boundary_flags(grid, centroid)
    threshold = float(boundary_probability_threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("boundary_probability_threshold must lie within [0, 1].")
    posterior_diagnostics = dict(posterior_diagnostics or {})

    horizontal_probability = posterior_diagnostics.get(
        "posterior_horizontal_boundary_probability", np.nan
    )
    depth_floor_probability = posterior_diagnostics.get(
        "posterior_depth_floor_probability", np.nan
    )
    depth_ceiling_probability = posterior_diagnostics.get(
        "posterior_depth_ceiling_probability", np.nan
    )
    horizontal_active = bool(flags["on_horizontal_boundary"]) or (
        np.isfinite(horizontal_probability) and float(horizontal_probability) >= threshold
    )
    depth_floor_active = bool(flags["on_depth_floor"]) or (
        np.isfinite(depth_floor_probability) and float(depth_floor_probability) >= threshold
    )
    depth_ceiling_active = bool(flags["on_depth_ceiling"]) or (
        np.isfinite(depth_ceiling_probability) and float(depth_ceiling_probability) >= threshold
    )

    radius = float(grid.radius) / 1000.0
    step_x = float(grid.step_x) / 1000.0
    step_z = float(grid.step_z) / 1000.0
    depths = sorted({float(gp["z"]) / 1000.0 for gp in grid.grid if not gp.get("err")})
    if not depths:
        raise RuntimeError("Cannot expand an empty grid.")
    depth_min = depths[0]
    depth_max = depths[-1]

    new_radius = radius
    new_depth_min = depth_min
    new_depth_max = depth_max
    actions: list[str] = []

    if horizontal_active and not flags["horizontal_search_fixed"]:
        cap = float(initial_radius_km) * float(max_radius_factor)
        new_radius = min(radius + int(expand_xy_steps) * step_x, cap)
        if new_radius > radius + 1e-12:
            actions.append("expand_xy")

    initial_span = float(initial_depth_max_km) - float(initial_depth_min_km)
    extra_total = max(0.0, (float(max_depth_span_factor) - 1.0) * initial_span)
    both_depth_edges = depth_floor_active and depth_ceiling_active
    side_allowance = extra_total / 2.0 if both_depth_edges else extra_total

    if depth_floor_active:
        lower_cap = max(float(min_depth_km), float(initial_depth_min_km) - side_allowance)
        candidate = max(lower_cap, depth_min - int(expand_z_steps) * step_z)
        if candidate < depth_min - 1e-12:
            new_depth_min = candidate
            actions.append("expand_z_floor")

    if depth_ceiling_active:
        upper_cap = float(initial_depth_max_km) + side_allowance
        candidate = min(upper_cap, depth_max + int(expand_z_steps) * step_z)
        if candidate > depth_max + 1e-12:
            new_depth_max = candidate
            actions.append("expand_z_ceiling")

    estimate = _grid_point_estimate(new_radius, step_x, new_depth_min, new_depth_max, step_z)
    budget_ok = grid_point_budget is None or estimate <= int(grid_point_budget)
    apply = bool(actions) and budget_ok
    reason = "apply" if apply else ("grid_point_budget_exceeded" if actions and not budget_ok else "no_expandable_active_boundary")
    max_points = max(1, int(math.ceil(estimate * 1.05)))

    return {
        "apply": apply,
        "reason": reason,
        "actions": actions,
        "grid_radius_km": new_radius,
        "grid_min_depth_km": new_depth_min,
        "grid_max_depth_km": new_depth_max,
        "step_x_km": step_x,
        "step_z_km": step_z,
        "estimated_grid_points": estimate,
        "max_grid_points_required": max_points,
        "boundary_flags": flags,
        "posterior_boundary_probability_threshold": threshold,
        "posterior_horizontal_boundary_probability": horizontal_probability,
        "posterior_depth_floor_probability": depth_floor_probability,
        "posterior_depth_ceiling_probability": depth_ceiling_probability,
    }


def compute_grid_refinement(
    grid,
    centroid: Mapping[str, Any],
    *,
    initial_step_x_km: float,
    initial_step_z_km: float,
    refinement_level: int,
    max_refinement_levels: int = 2,
    refine_factor: float = 0.5,
    min_step_fraction: float = 0.25,
    depth_window_parent_steps: int | None = 3,
    grid_point_budget: int | None = None,
    posterior_diagnostics: Mapping[str, Any] | None = None,
    boundary_probability_threshold: float = 0.05,
) -> dict[str, Any]:
    """Propose one bounded x/y/z refinement on a symmetric XY domain.

    The horizontal search remains centred on the catalogue epicentre.  A fixed
    one-point XY search stays fixed.  Depth may be narrowed around an internal
    coarse-grid optimum to control cost; set ``depth_window_parent_steps=None``
    to retain the full depth range.  Automatic refinement is hard-limited to two
    levels (quarter of the initial spacing at the default factor 0.5).
    """
    refinement_level = int(refinement_level)
    max_refinement_levels = int(max_refinement_levels)
    if not 0 <= max_refinement_levels <= 2:
        raise ValueError("max_refinement_levels must be 0, 1 or 2 for the bounded adaptive search.")
    if refinement_level >= max_refinement_levels:
        return {"apply": False, "reason": "maximum_refinement_level_reached"}
    refine_factor = float(refine_factor)
    if np.isclose(refine_factor, 0.0):
        return {"apply": False, "reason": "refinement_disabled"}
    if not 0.0 < refine_factor < 1.0:
        raise ValueError("refine_factor must lie within [0, 1); 0 disables refinement.")
    if not 0.25 <= float(min_step_fraction) <= 1:
        raise ValueError("min_step_fraction must lie within [0.25, 1] for bounded 0.2 refinement.")

    flags = _grid_boundary_flags(grid, centroid)
    threshold = float(boundary_probability_threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("boundary_probability_threshold must lie within [0, 1].")
    posterior_diagnostics = dict(posterior_diagnostics or {})
    posterior_boundary_probability = posterior_diagnostics.get(
        "posterior_active_spatial_boundary_probability", np.nan
    )
    posterior_boundary_active = (
        np.isfinite(posterior_boundary_probability)
        and float(posterior_boundary_probability) >= threshold
    )
    if flags["on_active_spatial_boundary"] or posterior_boundary_active:
        return {
            "apply": False,
            "reason": "active_boundary_requires_expansion",
            "boundary_flags": flags,
            "posterior_active_spatial_boundary_probability": posterior_boundary_probability,
        }

    radius = float(grid.radius) / 1000.0
    current_step_x = float(grid.step_x) / 1000.0
    current_step_z = float(grid.step_z) / 1000.0
    horizontal_fixed = flags["horizontal_search_fixed"]

    min_step_x = float(initial_step_x_km) * float(min_step_fraction)
    min_step_z = float(initial_step_z_km) * float(min_step_fraction)
    new_step_x = current_step_x if horizontal_fixed else max(current_step_x * refine_factor, min_step_x)
    new_step_z = max(current_step_z * refine_factor, min_step_z)

    if np.isclose(new_step_x, current_step_x) and np.isclose(new_step_z, current_step_z):
        return {"apply": False, "reason": "minimum_spacing_reached", "boundary_flags": flags}

    depths = sorted({float(gp["z"]) / 1000.0 for gp in grid.grid if not gp.get("err")})
    depth_min = depths[0]
    depth_max = depths[-1]
    if depth_window_parent_steps is not None:
        steps = int(depth_window_parent_steps)
        if steps < 1:
            raise ValueError("depth_window_parent_steps must be >= 1 or None.")
        center_depth = float(centroid["z"]) / 1000.0
        depth_min = max(depth_min, center_depth - steps * current_step_z)
        depth_max = min(depth_max, center_depth + steps * current_step_z)

    estimate = _grid_point_estimate(radius, new_step_x, depth_min, depth_max, new_step_z)
    budget_ok = grid_point_budget is None or estimate <= int(grid_point_budget)
    apply = budget_ok
    reason = "apply" if apply else "grid_point_budget_exceeded"
    max_points = max(1, int(math.ceil(estimate * 1.05)))

    return {
        "apply": apply,
        "reason": reason,
        "grid_radius_km": radius,
        "grid_min_depth_km": depth_min,
        "grid_max_depth_km": depth_max,
        "step_x_km": new_step_x,
        "step_z_km": new_step_z,
        "estimated_grid_points": estimate,
        "max_grid_points_required": max_points,
        "boundary_flags": flags,
        "refinement_level": refinement_level + 1,
    }


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

    ``on_grid_edge`` is the corrected *active* spatial-boundary state.  A
    deliberately fixed one-point XY grid is reported through
    ``horizontal_search_fixed`` rather than being mislabelled as an edge.
    """
    event = solution.event
    c = solution.centroid
    centroid_time = event["t"] + c["shift"]
    flags = _grid_boundary_flags(solution.g, c)
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
        "on_grid_edge": bool(flags["on_active_spatial_boundary"]),
        "horizontal_search_fixed": bool(flags["horizontal_search_fixed"]),
        "on_horizontal_boundary": bool(flags["on_horizontal_boundary"]),
        "on_depth_floor": bool(flags["on_depth_floor"]),
        "on_depth_ceiling": bool(flags["on_depth_ceiling"]),
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


def _resolve_uncertainty_variance_scale(
    solution,
    uncertainty_scale: str | float = "fixed",
    *,
    minimum_scale: float = 1.0,
) -> tuple[float, dict[str, Any]]:
    """Resolve the scalar variance multiplier used for posterior/MT sampling.

    ``fixed`` reproduces the 0.1.1 likelihood/covariance exactly. ``residual``
    estimates a common discrepancy variance as chi-square / degrees-of-freedom
    at the preferred cell and, by default, does not allow the measured-noise
    covariance to be deflated below its original scale.
    """
    n_parameters = 5 if solution.deviatoric else 6
    n_data = int(solution.d.components * solution.d.npts_slice)
    dof = n_data - n_parameters
    if dof <= 0:
        raise ValueError("Not enough waveform values to estimate an uncertainty scale.")
    mode_misfit = float(solution.centroid["misfit"])
    reduced_chi2 = mode_misfit / dof

    if isinstance(uncertainty_scale, str):
        mode = uncertainty_scale.lower().strip()
        if mode in {"fixed", "noise", "native"}:
            scale = 1.0
            mode = "fixed"
        elif mode in {"residual", "scaled", "residual_scaled"}:
            scale = max(float(minimum_scale), reduced_chi2)
            mode = "residual"
        else:
            raise ValueError("uncertainty_scale must be 'fixed', 'residual', or a positive number.")
    else:
        scale = float(uncertainty_scale)
        mode = "explicit"
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("A numerical uncertainty_scale must be positive and finite.")

    diagnostics = {
        "uncertainty_scale_mode": mode,
        "variance_scale": scale,
        "sd_scale": math.sqrt(scale),
        "n_waveform_values": n_data,
        "n_mt_parameters": n_parameters,
        "degrees_of_freedom": dof,
        "preferred_misfit": mode_misfit,
        "reduced_chi_square": reduced_chi2,
    }
    return scale, diagnostics


def build_posterior_cells(solution, *, variance_scale: float = 1.0) -> pd.DataFrame:
    """Expose the complete discrete BayesISOLA space-time posterior.

    Cell likelihoods are reconstructed from the shift-specific misfit and
    ``log_det_Ca`` in log space.  ``variance_scale=1`` reproduces 0.1.1 exactly;
    a common scalar covariance inflation divides relative misfit differences by
    that variance factor.  The determinant's common scale factor cancels between
    cells, so the stored shift-specific log determinant remains sufficient.
    """
    from scipy.special import logsumexp

    variance_scale = float(variance_scale)
    if not np.isfinite(variance_scale) or variance_scale <= 0:
        raise ValueError("variance_scale must be positive and finite.")

    valid_grid = [(gi, gp) for gi, gp in enumerate(solution.grid) if not gp["err"]]
    n_cells = sum(len(gp["shifts"]) for _, gp in valid_grid)
    if n_cells == 0:
        raise RuntimeError("No valid BayesISOLA space-time cells are available.")

    rows: list[dict[str, Any]] = []
    for gi, gp in valid_grid:
        spatial_flags = _grid_boundary_flags(solution.g, gp)
        for si, GP in gp["shifts"].items():
            si = int(si)
            shift_s = float(solution.d.shifts[si])
            rows.append({
                "cell_index": len(rows),
                "grid_index": int(gi),
                "grid_point_id": str(gp.get("id", gi)),
                "x_id": str(gp.get("x_id", "")),
                "y_id": str(gp.get("y_id", "")),
                "z_id": str(gp.get("z_id", "")),
                "shift_index": si,
                "centroid_time_shift_s": shift_s,
                "centroid_lat": float(gp.get("lat", np.nan)),
                "centroid_lon": float(gp.get("lon", np.nan)),
                "centroid_depth_km": float(gp["z"]) / 1e3,
                "offset_north_m": float(gp["x"]),
                "offset_east_m": float(gp["y"]),
                "horizontal_search_fixed": bool(spatial_flags["horizontal_search_fixed"]),
                "on_horizontal_boundary": bool(spatial_flags["on_horizontal_boundary"]),
                "on_depth_floor": bool(spatial_flags["on_depth_floor"]),
                "on_depth_ceiling": bool(spatial_flags["on_depth_ceiling"]),
                "misfit": float(GP["misfit"]),
                "variance_reduction": float(GP["VR"]),
                "condition_number": float(GP["CN"]),
                "log_det_Ca": float(GP["log_det_Ca"]),
                "native_weight": float(GP["c"]),
            })

    df = pd.DataFrame(rows)
    time_min = float(df["centroid_time_shift_s"].min())
    time_max = float(df["centroid_time_shift_s"].max())
    time_fixed = np.isclose(time_min, time_max)
    df["time_search_fixed"] = bool(time_fixed)
    df["on_time_floor"] = False if time_fixed else np.isclose(df["centroid_time_shift_s"], time_min)
    df["on_time_ceiling"] = False if time_fixed else np.isclose(df["centroid_time_shift_s"], time_max)
    df["on_active_boundary"] = (
        df["on_horizontal_boundary"] | df["on_depth_floor"] | df["on_depth_ceiling"]
        | df["on_time_floor"] | df["on_time_ceiling"]
    )

    misfit = df["misfit"].to_numpy(dtype=float)
    log_det = df["log_det_Ca"].to_numpy(dtype=float)
    finite = np.isfinite(misfit) & np.isfinite(log_det)
    if not finite.any():
        raise RuntimeError("No finite posterior cells are available.")

    misfit_ref = float(np.min(misfit[finite]))
    log_weight = np.full(len(df), -np.inf, dtype=float)
    log_weight[finite] = 0.5 * log_det[finite] - 0.5 * (misfit[finite] - misfit_ref) / variance_scale
    log_norm = float(logsumexp(log_weight[finite]))
    log_posterior = log_weight - log_norm
    probability = np.exp(log_posterior)

    df["log_weight"] = log_weight
    df["log_posterior"] = log_posterior
    df["posterior_probability"] = probability
    sum_c = float(solution.sum_c)
    df["native_probability"] = df["native_weight"] / sum_c if np.isfinite(sum_c) and sum_c > 0 else np.nan

    order = np.argsort(log_posterior)[::-1]
    rank = np.empty(len(df), dtype=np.int64)
    rank[order] = np.arange(1, len(df) + 1, dtype=np.int64)
    sorted_p = probability[order]
    cumulative_sorted = np.cumsum(sorted_p)
    cumulative = np.empty(len(df), dtype=float)
    cumulative[order] = cumulative_sorted
    cumulative_before = cumulative_sorted - sorted_p
    hpd68 = np.zeros(len(df), dtype=bool)
    hpd95 = np.zeros(len(df), dtype=bool)
    hpd68[order] = cumulative_before < 0.68
    hpd95[order] = cumulative_before < 0.95
    df["posterior_rank"] = rank
    df["cumulative_probability"] = cumulative
    df["in_hpd68"] = hpd68
    df["in_hpd95"] = hpd95
    return df


def compute_posterior_diagnostics(
    solution,
    posterior_cells: pd.DataFrame,
    *,
    variance_scale_diagnostics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize posterior concentration, discretization and active boundaries."""
    p = posterior_cells["posterior_probability"].to_numpy(dtype=float)
    logp = posterior_cells["log_posterior"].to_numpy(dtype=float)
    positive = p > 0
    entropy = float(-np.sum(p[positive] * logp[positive])) if positive.any() else np.nan
    n_cells = len(p)
    normalized_entropy = entropy / math.log(n_cells) if n_cells > 1 and np.isfinite(entropy) else 0.0
    effective_cells = float(1.0 / np.sum(p ** 2))
    horizontal_fixed = bool(posterior_cells["horizontal_search_fixed"].iloc[0])
    time_fixed = bool(posterior_cells["time_search_fixed"].iloc[0])

    diagnostics: dict[str, Any] = {
        "n_space_time_cells": n_cells,
        "posterior_probability_sum": float(np.sum(p)),
        "posterior_mode_probability": float(np.max(p)),
        "posterior_entropy": entropy,
        "posterior_normalized_entropy": normalized_entropy,
        "posterior_effective_cells": effective_cells,
        "n_cells_hpd68": int(posterior_cells["in_hpd68"].sum()),
        "n_cells_hpd95": int(posterior_cells["in_hpd95"].sum()),
        "horizontal_search_fixed": horizontal_fixed,
        "time_search_fixed": time_fixed,
        "posterior_horizontal_boundary_probability": (
            np.nan if horizontal_fixed else float(p[posterior_cells["on_horizontal_boundary"].to_numpy()].sum())
        ),
        "posterior_depth_floor_probability": float(p[posterior_cells["on_depth_floor"].to_numpy()].sum()),
        "posterior_depth_ceiling_probability": float(p[posterior_cells["on_depth_ceiling"].to_numpy()].sum()),
        "posterior_time_floor_probability": (
            np.nan if time_fixed else float(p[posterior_cells["on_time_floor"].to_numpy()].sum())
        ),
        "posterior_time_ceiling_probability": (
            np.nan if time_fixed else float(p[posterior_cells["on_time_ceiling"].to_numpy()].sum())
        ),
        "posterior_active_spatial_boundary_probability": float(
            p[(
                posterior_cells["on_horizontal_boundary"].to_numpy()
                | posterior_cells["on_depth_floor"].to_numpy()
                | posterior_cells["on_depth_ceiling"].to_numpy()
            )].sum()
        ),
        "posterior_active_boundary_probability": float(p[posterior_cells["on_active_boundary"].to_numpy()].sum()),
    }
    if variance_scale_diagnostics is not None:
        diagnostics.update(dict(variance_scale_diagnostics))
    return diagnostics


def _circular_angle_difference_deg(value: float, reference: float) -> float:
    """Return the shortest signed difference between two angles in degrees."""
    return (float(value) - float(reference) + 180.0) % 360.0 - 180.0


def _align_nodal_planes(
    decomposition: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> dict[str, Any]:
    """Align an unordered nodal-plane pair to a reference plane ordering.

    Moment-tensor decomposition returns two physically interchangeable nodal
    planes.  Eigenvector sign choices can therefore exchange the numerical
    ``NP1``/``NP2`` labels between nearby posterior samples even when the
    mechanism changes smoothly.  This helper preserves the decomposition itself
    but chooses the direct or swapped labeling whose strike/dip/rake parameters
    are closest to the preferred solution.

    Strike and rake use their shortest circular differences; dip is compared
    linearly.  The normalized squared cost is used only to resolve the two-plane
    permutation and has no effect on posterior probabilities or MT samples.
    """
    out = dict(decomposition)
    sample = np.array([
        [out.get("s1"), out.get("d1"), out.get("r1")],
        [out.get("s2"), out.get("d2"), out.get("r2")],
    ], dtype=float)
    ref = np.array([
        [reference.get("s1"), reference.get("d1"), reference.get("r1")],
        [reference.get("s2"), reference.get("d2"), reference.get("r2")],
    ], dtype=float)

    if not np.isfinite(sample).all() or not np.isfinite(ref).all():
        return out

    def plane_cost(plane, target):
        ds = _circular_angle_difference_deg(plane[0], target[0]) / 180.0
        dd = (plane[1] - target[1]) / 90.0
        dr = _circular_angle_difference_deg(plane[2], target[2]) / 180.0
        return ds * ds + dd * dd + dr * dr

    direct = plane_cost(sample[0], ref[0]) + plane_cost(sample[1], ref[1])
    swapped = plane_cost(sample[0], ref[1]) + plane_cost(sample[1], ref[0])
    if swapped < direct:
        out["s1"], out["d1"], out["r1"] = sample[1]
        out["s2"], out["d2"], out["r2"] = sample[0]
    return out


def extract_uncertainty_df(
    solution,
    n: int = 400,
    *,
    variance_scale: float = 1.0,
    posterior_cells: pd.DataFrame | None = None,
    random_state: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Draw exactly ``n`` posterior realizations using categorical cell sampling.

    Nonlinear space-time cells are sampled categorically from the exact discrete
    posterior; moment-tensor coefficients are then drawn from the selected cell's
    conditional Gaussian covariance, optionally multiplied by a common residual
    variance scale.  Because the two nodal planes are physically interchangeable,
    sampled NP1/NP2 labels are finally aligned to the preferred solution before
    reporting strike/dip/rake uncertainty.  This relabeling does not modify the
    sampled moment tensor or its posterior probability.
    """
    from BayesISOLA.MT_comps import a2mt, decompose

    n = int(n)
    if n <= 0:
        raise ValueError("n must be a positive integer.")
    variance_scale = float(variance_scale)
    if not np.isfinite(variance_scale) or variance_scale <= 0:
        raise ValueError("variance_scale must be positive and finite.")
    if posterior_cells is None:
        posterior_cells = build_posterior_cells(solution, variance_scale=variance_scale)

    probabilities = posterior_cells["posterior_probability"].to_numpy(dtype=float)
    if not np.isclose(probabilities.sum(), 1.0, atol=1e-12):
        raise RuntimeError("Posterior cell probabilities do not sum to one.")

    rng = np.random.default_rng(random_state)
    selected = rng.choice(len(posterior_cells), size=n, replace=True, p=probabilities)
    counts = np.bincount(selected, minlength=len(posterior_cells))
    label_columns = [
        "cell_index", "grid_index", "grid_point_id", "shift_index",
        "centroid_time_shift_s", "centroid_lat", "centroid_lon", "centroid_depth_km",
        "offset_north_m", "offset_east_m", "horizontal_search_fixed",
        "on_horizontal_boundary", "on_depth_floor", "on_depth_ceiling",
        "on_time_floor", "on_time_ceiling", "on_active_boundary",
        "posterior_probability",
    ]
    sampled_labels = posterior_cells.iloc[selected][label_columns].reset_index(drop=True)
    sampled_labels.insert(0, "draw", np.arange(1, n + 1, dtype=int))
    sampled_labels["space_time_cell_n_draws"] = counts[selected]
    sampled_labels["centroid_time"] = [
        (solution.event["t"] + float(shift)).datetime
        for shift in sampled_labels["centroid_time_shift_s"].to_numpy()
    ]

    preferred_decomposition = solution.mt_decomp
    if not preferred_decomposition:
        raise RuntimeError(
            "solution.mt_decomp is empty -- resolve_MT must be constructed "
            "before uncertainty sampling."
        )

    mechanism_rows: list[dict[str, Any] | None] = [None] * n
    for cell_position in np.flatnonzero(counts):
        draw_positions = np.flatnonzero(selected == cell_position)
        row = posterior_cells.iloc[int(cell_position)]
        gp = solution.grid[int(row["grid_index"])]
        GP = gp["shifts"][int(row["shift_index"])]
        a_mean = np.asarray(GP["a"], dtype=float).reshape(-1)
        if solution.deviatoric:
            a_mean = a_mean[:5]
        cov = np.asarray(GP["GtGinv"], dtype=float)
        cov = cov[: len(a_mean), : len(a_mean)] * variance_scale
        cov = 0.5 * (cov + cov.T)
        a_draws = rng.multivariate_normal(a_mean, cov, size=len(draw_positions))

        for position, a_draw in zip(draw_positions, a_draws):
            a_col = np.asarray(a_draw, dtype=float)[:, None]
            if solution.deviatoric:
                a_col = np.vstack([a_col, [[0.0]]])
            mt = a2mt(a_col)
            dec = _align_nodal_planes(decompose(mt), preferred_decomposition)
            mechanism_rows[int(position)] = {
                "dc_percent": dec["dc_perc"],
                "clvd_percent": dec["clvd_perc"],
                "iso_percent": dec["iso_perc"],
                "moment_Nm": dec["mom"],
                "Mw": dec["Mw"],
                "NP1_strike_deg": dec["s1"], "NP1_dip_deg": dec["d1"], "NP1_rake_deg": dec["r1"],
                "NP2_strike_deg": dec["s2"], "NP2_dip_deg": dec["d2"], "NP2_rake_deg": dec["r2"],
            }

    mechanism = pd.DataFrame(mechanism_rows)
    df = pd.concat([sampled_labels, mechanism], axis=1)
    diagnostics = {
        "n_requested": n,
        "n_sampled": len(df),
        "n_space_time_cells_used": int(np.count_nonzero(counts)),
        "n_grid_points_used": int(df["grid_index"].nunique()),
        "n_time_shifts_used": int(df["shift_index"].nunique()),
        "spatially_degenerate": int(df["grid_index"].nunique()) <= 1,
        "temporally_degenerate": int(df["shift_index"].nunique()) <= 1,
        "degenerate_allocation": int(np.count_nonzero(counts)) <= 1,
        "variance_scale": variance_scale,
        "sd_scale": math.sqrt(variance_scale),
        "random_state": random_state,
        "sampling_method": "categorical_cells_conditional_gaussian_mt",
    }
    return df, diagnostics


_STATION_JACKKNIFE_DEFAULTS: dict[str, Any] = {
    "jackknife_min_stations": 4,
}


def _normalize_station_jackknife(
    station_jackknife: bool | Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Normalize optional single-station fixed-grid jackknife settings."""
    if station_jackknife is None or station_jackknife is False:
        return {"enabled": False, **_STATION_JACKKNIFE_DEFAULTS}
    if station_jackknife is True:
        config = dict(_STATION_JACKKNIFE_DEFAULTS)
    elif isinstance(station_jackknife, Mapping):
        unknown = sorted(set(station_jackknife) - set(_STATION_JACKKNIFE_DEFAULTS))
        if unknown:
            allowed = ", ".join(_STATION_JACKKNIFE_DEFAULTS)
            raise ValueError(
                f"Unknown station_jackknife option(s): {', '.join(map(str, unknown))}. "
                f"Allowed keys are: {allowed}."
            )
        config = dict(_STATION_JACKKNIFE_DEFAULTS)
        config.update(dict(station_jackknife))
    else:
        raise TypeError("station_jackknife must be None, a boolean, or a mapping.")
    config["jackknife_min_stations"] = int(config["jackknife_min_stations"])
    if config["jackknife_min_stations"] < 1:
        raise ValueError("jackknife_min_stations must be >= 1.")
    return {"enabled": True, **config}


def _solve_omission_normal_equations(
    A_total: np.ndarray,
    B_total: np.ndarray,
    q_total: np.ndarray,
    A_removed: np.ndarray,
    B_removed: np.ndarray,
    q_removed: np.ndarray,
) -> dict[str, Any]:
    """Solve all source-time shifts after subtracting one station's statistics.

    ``B`` has shape ``(n_parameters, n_shifts)`` and ``q`` contains the whitened
    data norm for each shift. This is algebraically identical to rebuilding the
    reduced design/data matrices and solving them directly. ``numpy.linalg.solve``
    is used instead of explicitly forming ``A^{-1}``, which is both faster and
    numerically preferable. The condition number is intentionally evaluated only
    for the final winning leave-one-out solution, not for every grid/station trial.
    """
    A = np.asarray(A_total, dtype=float) - np.asarray(A_removed, dtype=float)
    B = np.asarray(B_total, dtype=float) - np.asarray(B_removed, dtype=float)
    q = np.asarray(q_total, dtype=float) - np.asarray(q_removed, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A_total/A_removed must define square normal matrices.")
    if B.shape[0] != A.shape[0] or q.ndim != 1 or B.shape[1] != q.size:
        raise ValueError("B/q dimensions are inconsistent with the normal matrix.")
    if not np.isfinite(A).all() or not np.isfinite(B).all() or not np.isfinite(q).all():
        raise ValueError("Normal-equation statistics must be finite.")
    if np.any(q <= 0.0):
        raise np.linalg.LinAlgError("Station omission produced a non-positive data norm.")

    coefficients = np.linalg.solve(A, B)
    misfit = q - np.sum(B * coefficients, axis=0)
    tolerance = 1e-10 * np.maximum(q, 1.0)
    if np.any(misfit < -tolerance):
        raise np.linalg.LinAlgError(
            "Station-omission normal equations produced a negative minimized misfit."
        )
    misfit = np.maximum(misfit, 0.0)
    variance_reduction = 1.0 - misfit / q
    return {
        "A": A,
        "coefficients": coefficients,
        "misfit": misfit,
        "norm_d": q,
        "variance_reduction": variance_reduction,
    }

def _jackknife_design_matrix(solution, data, gp: Mapping[str, Any]) -> np.ndarray:
    """Reconstruct the exact filtered/weighted design matrix used by ``invert``."""
    from obspy import UTCDateTime
    from BayesISOLA.fileformats import read_elemse, read_elemse_from_files
    from BayesISOLA.helpers import my_filter
    from BayesISOLA._paths import green_path

    stations = solution.inp.stations
    nr = solution.inp.nr
    ne = 5 if solution.deviatoric else 6
    elemse_path = gp.get("path")
    if elemse_path:
        elemse = read_elemse_from_files(
            nr, elemse_path, stations, solution.inp.event["t"], data.samprate,
            data.npts_elemse, data.invert_displacement,
        )
    else:
        elemse = read_elemse(
            nr, data.npts_elemse,
            green_path(solution.inp.green_dir, "elemse" + str(gp["id"]) + ".dat"),
            stations, data.invert_displacement,
        )

    for r in range(nr):
        for e in range(ne):
            my_filter(elemse[r][e], stations[r]["fmin"], stations[r]["fmax"])
    if data.npts_slice != data.npts_elemse:
        for st6 in elemse:
            for trace in st6:
                trace.trim(UTCDateTime(0) + data.elemse_start_origin)
    npts = int(data.npts_slice)

    G = np.empty((int(data.components) * npts, ne), dtype=float)
    component_row = 0
    for r, station in enumerate(stations):
        for comp, use_key, weight_key in zip(
            range(3), ("useZ", "useN", "useE"), ("weightZ", "weightN", "weightE")
        ):
            if station[use_key]:
                sl = slice(component_row * npts, (component_row + 1) * npts)
                weight = float(station[weight_key])
                for e in range(ne):
                    G[sl, e] = np.asarray(elemse[r][e][comp].data[:npts], dtype=float) * weight
                component_row += 1
    if component_row != int(data.components):
        raise RuntimeError(
            f"Jackknife design matrix assembled {component_row} component blocks but data.components={data.components}."
        )
    return G


def _jackknife_station_blocks(stations: Sequence[Mapping[str, Any]], npts: int) -> list[dict[str, Any]]:
    """Return active station row slices in BayesISOLA's concatenated data order."""
    blocks: list[dict[str, Any]] = []
    offset = 0
    for station_index, station in enumerate(stations):
        used = [
            comp for comp, key in zip(("Z", "N", "E"), ("useZ", "useN", "useE"))
            if bool(station[key])
        ]
        size = len(used) * int(npts)
        if size:
            blocks.append({
                "station_index": station_index,
                "station": station,
                "used_components": tuple(used),
                "slice": slice(offset, offset + size),
            })
        offset += size
    return blocks


def _moment_tensor_matrix_from_a(a: np.ndarray) -> np.ndarray:
    """Convert six elementary-source coefficients to a symmetric NED MT matrix."""
    from BayesISOLA.MT_comps import a2mt
    mt = np.asarray(a2mt(np.asarray(a, dtype=float).reshape(6, 1)), dtype=float)
    return np.array([
        [mt[0], mt[3], mt[4]],
        [mt[3], mt[1], mt[5]],
        [mt[4], mt[5], mt[2]],
    ], dtype=float)


def _azimuth_geometry_from_solution_stations(
    stations: Sequence[Mapping[str, Any]],
    omitted_index: int | None = None,
) -> tuple[int, float]:
    """Return occupied 45-degree sectors and maximum azimuthal gap after omission."""
    azimuths = [
        float(station["az"]) % 360.0
        for i, station in enumerate(stations)
        if i != omitted_index
        and any(bool(station[key]) for key in ("useZ", "useN", "useE"))
        and station.get("az") is not None
        and np.isfinite(float(station["az"]))
    ]
    if not azimuths:
        return 0, np.nan
    azimuths = np.sort(np.asarray(azimuths, dtype=float))
    sectors = np.floor(azimuths / 45.0).astype(int)
    gaps = np.diff(np.r_[azimuths, azimuths[0] + 360.0])
    return int(np.unique(sectors).size), float(np.max(gaps))


def _jackknife_grid_candidates(
    solution,
    data,
    gp: Mapping[str, Any],
    *,
    factorized_noise: bool,
    covariance_factors,
    stations: Sequence[Mapping[str, Any]],
    npts: int,
    D: np.ndarray,
    q_total: np.ndarray,
    blocks: Sequence[Mapping[str, Any]],
    q_station: Sequence[np.ndarray],
) -> dict[int, dict[str, Any]]:
    """Evaluate all station omissions at one existing source-grid point.

    The function is deliberately side-effect free so final-grid points can be
    evaluated concurrently by :class:`~concurrent.futures.ThreadPoolExecutor`.
    Each returned entry is the best source-time shift for one omitted station at
    this grid point; the caller then performs the unchanged global comparison over
    all grid points.
    """
    from BayesISOLA.inverse_problem import whiten_covariance_array

    G = _jackknife_design_matrix(solution, data, gp)
    if factorized_noise:
        G = whiten_covariance_array(G, covariance_factors, stations, npts)

    A_total = G.T @ G
    B_total = G.T @ D
    candidates: dict[int, dict[str, Any]] = {}

    for block, q_removed in zip(blocks, q_station):
        station_index = int(block["station_index"])
        sl = block["slice"]
        Gs = G[sl, :]
        Ds = D[sl, :]
        A_removed = Gs.T @ Gs
        B_removed = Gs.T @ Ds

        try:
            solved = _solve_omission_normal_equations(
                A_total, B_total, q_total, A_removed, B_removed, q_removed
            )
        except (np.linalg.LinAlgError, ValueError):
            continue

        vr = solved["variance_reduction"]
        if not np.isfinite(vr).any():
            continue
        shift_index = int(np.nanargmax(vr))
        candidate_vr = float(vr[shift_index])
        coeff = solved["coefficients"][:, shift_index]
        a_col = coeff[:, None]
        if solution.deviatoric:
            a_col = np.vstack([a_col, [[0.0]]])

        # ||d_s - G_s a||^2 from sufficient statistics. This avoids forming a
        # long held-out residual vector for every grid/station trial.
        b_removed = B_removed[:, shift_index]
        heldout_misfit = float(
            q_removed[shift_index]
            - 2.0 * np.dot(coeff, b_removed)
            + np.dot(coeff, A_removed @ coeff)
        )
        heldout_tolerance = 1e-10 * max(float(q_removed[shift_index]), 1.0)
        if heldout_misfit < -heldout_tolerance:
            continue
        heldout_misfit = max(heldout_misfit, 0.0)

        candidates[station_index] = {
            "gp": gp,
            "shift_index": shift_index,
            "a": a_col,
            "variance_reduction": candidate_vr,
            "misfit": float(solved["misfit"][shift_index]),
            "normal_matrix": solved["A"],
            "heldout_misfit": heldout_misfit,
            "heldout_n_values": int(Ds.shape[0]),
        }

    return candidates


def _cached_station_normal_equations(
    gp: Mapping[str, Any],
    blocks: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return cached station normal equations ordered like ``blocks``.

    ``inverse_problem.invert`` stores these arrays only when the workflow knows
    that a station jackknife will be requested.  Returning ``None`` preserves a
    backward-compatible fallback for solution objects that do not contain the
    0.2 station-wise normal-equation cache.
    """
    indices = gp.get("_station_normal_indices")
    station_GtG = gp.get("_station_GtG")
    station_Gtd = gp.get("_station_Gtd")
    if indices is None or station_GtG is None or station_Gtd is None:
        return None

    indices = np.asarray(indices, dtype=int).reshape(-1)
    station_GtG = np.asarray(station_GtG, dtype=float)
    station_Gtd = np.asarray(station_Gtd, dtype=float)
    expected = [int(block["station_index"]) for block in blocks]

    if station_GtG.ndim != 3 or station_Gtd.ndim != 3:
        raise ValueError("Cached station normal equations must be three-dimensional arrays.")
    if station_GtG.shape[0] != len(indices) or station_Gtd.shape[0] != len(indices):
        raise ValueError("Cached station normal-equation arrays have inconsistent station counts.")

    position = {int(station_index): i for i, station_index in enumerate(indices)}
    missing = [station_index for station_index in expected if station_index not in position]
    if missing:
        raise ValueError(
            "Cached station normal equations are missing active station index/indices: "
            + ", ".join(map(str, missing))
        )
    order = [position[station_index] for station_index in expected]
    return station_GtG[order], station_Gtd[order]


def _jackknife_grid_candidates_cached(
    gp: Mapping[str, Any],
    *,
    blocks: Sequence[Mapping[str, Any]],
    q_total: np.ndarray,
    q_station: Sequence[np.ndarray],
    deviatoric: bool,
) -> dict[int, dict[str, Any]]:
    """Evaluate all station omissions using inversion-cached sufficient statistics."""
    cached = _cached_station_normal_equations(gp, blocks)
    if cached is None:
        raise RuntimeError("Cached station normal equations are not available for this grid point.")
    station_GtG, station_Gtd = cached

    A_total = np.sum(station_GtG, axis=0)
    B_total = np.sum(station_Gtd, axis=0)
    candidates: dict[int, dict[str, Any]] = {}

    for j, (block, q_removed) in enumerate(zip(blocks, q_station)):
        station_index = int(block["station_index"])
        A_removed = station_GtG[j]
        B_removed = station_Gtd[j]

        try:
            solved = _solve_omission_normal_equations(
                A_total,
                B_total,
                q_total,
                A_removed,
                B_removed,
                q_removed,
            )
        except (np.linalg.LinAlgError, ValueError):
            continue

        vr = solved["variance_reduction"]
        if not np.isfinite(vr).any():
            continue
        shift_index = int(np.nanargmax(vr))
        candidate_vr = float(vr[shift_index])
        coeff = solved["coefficients"][:, shift_index]
        a_col = coeff[:, None]
        if deviatoric:
            a_col = np.vstack([a_col, [[0.0]]])

        b_removed = B_removed[:, shift_index]
        heldout_misfit = float(
            q_removed[shift_index]
            - 2.0 * np.dot(coeff, b_removed)
            + np.dot(coeff, A_removed @ coeff)
        )
        heldout_tolerance = 1e-10 * max(float(q_removed[shift_index]), 1.0)
        if heldout_misfit < -heldout_tolerance:
            continue
        heldout_misfit = max(heldout_misfit, 0.0)

        candidates[station_index] = {
            "gp": gp,
            "shift_index": shift_index,
            "a": a_col,
            "variance_reduction": candidate_vr,
            "misfit": float(solved["misfit"][shift_index]),
            "normal_matrix": solved["A"],
            "heldout_misfit": heldout_misfit,
            "heldout_n_values": int(block["slice"].stop - block["slice"].start),
        }

    return candidates


def compute_station_jackknife(
    solution,
    data,
    cova,
    *,
    jackknife_min_stations: int = 4,
    threads: int = 1,
) -> pd.DataFrame:
    """Evaluate exact single-station omissions on the final converged grid.

    No acquisition, Green-function generation, grid expansion or refinement is
    repeated.  At each existing spatial point the station block is subtracted
    from the full normal equations and every source-time shift is solved at once.
    The preferred leave-one-out source therefore follows BayesISOLA's native
    maximum-variance-reduction criterion, conditional on the final full-solution
    grid.  A leave-one-out mode on that grid boundary is reported, not refined.

    Parameters
    ----------
    solution, data, cova
        Final BayesISOLA inversion objects. When the solution was produced by
        ``run_auto_cmt`` with jackknife enabled, each grid point already contains
        small station-wise normal-equation contributions retained during the main
        inversion. The jackknife then requires no GF file I/O or filtering. Older
        solution objects transparently use the exact legacy reconstruction path.
    jackknife_min_stations : int, default=4
        Minimum number of active stations that must remain after one station is
        omitted.
    threads : int, default=1
        Number of concurrent grid evaluations used only by the backward-compatible
        GF-reconstruction fallback. The cached-normal-equation path uses inexpensive
        small-matrix solves and therefore runs serially to avoid executor overhead.
    """
    from BayesISOLA.inverse_problem import whiten_covariance_array
    from BayesISOLA.MT_comps import a2mt, decompose
    from BayesISOLA.kagan import kagan_angle_mt

    jackknife_min_stations = int(jackknife_min_stations)
    if jackknife_min_stations < 1:
        raise ValueError("jackknife_min_stations must be >= 1.")
    threads = int(threads)
    if threads < 1:
        raise ValueError("threads must be >= 1.")
    if getattr(cova, "Cd_inv_shifts", None):
        raise NotImplementedError("Station jackknife does not support shift-dependent ACF covariance.")
    factorized_noise = bool(getattr(cova, "factorized_noise", False))
    if bool(getattr(cova, "has_covariance", False)) and not factorized_noise:
        raise NotImplementedError(
            "Station jackknife currently supports no covariance or factorized noise covariance only."
        )

    stations = solution.inp.stations
    npts = int(data.npts_slice)
    blocks = _jackknife_station_blocks(stations, npts)
    n_active_stations = len(blocks)
    if not blocks:
        return pd.DataFrame()

    d_shifts = list(data.d_shifts)
    covariance_factors = {"LT": cova.LT, "LT3": cova.LT3} if factorized_noise else None
    if factorized_noise:
        d_columns = [
            whiten_covariance_array(d, covariance_factors, stations, npts).reshape(-1)
            for d in d_shifts
        ]
    else:
        d_columns = [np.asarray(d, dtype=float).reshape(-1) for d in d_shifts]
    D = np.column_stack(d_columns)
    q_total = np.sum(D * D, axis=0)
    q_station = [np.sum(D[block["slice"], :] ** 2, axis=0) for block in blocks]

    full_gp = solution.centroid
    full_shift_index = int(full_gp["shift_idx"])
    full_a = np.asarray(full_gp["a"], dtype=float).reshape(6, 1)
    full_mt = _moment_tensor_matrix_from_a(full_a)
    full_decomp = solution.mt_decomp
    valid_grid_points = [gp for gp in solution.grid if not gp.get("err")]

    # Fast 0.2 path: every grid-point inversion can retain each
    # station's contributions to G.T@G and G.T@d while the filtered design
    # matrix is already in memory.  If all final-grid points contain those
    # statistics, the jackknife never reopens an elementary-seismogram file.
    cached_fast_path = bool(valid_grid_points) and all(
        gp.get("_station_normal_indices") is not None
        and gp.get("_station_GtG") is not None
        and gp.get("_station_Gtd") is not None
        for gp in valid_grid_points
    )

    full_station_metrics: dict[int, dict[str, float]] = {}
    stored_full_misfit = float(full_gp["misfit"])

    if cached_fast_path:
        full_cached = _cached_station_normal_equations(full_gp, blocks)
        if full_cached is None:
            raise RuntimeError("Preferred grid point is missing cached jackknife statistics.")
        station_GtG, station_Gtd = full_cached
        coeff = full_a[: station_GtG.shape[1], 0]
        full_misfit = 0.0

        for j, block in enumerate(blocks):
            q = float(q_station[j][full_shift_index])
            A = station_GtG[j]
            b = station_Gtd[j, :, full_shift_index]
            misfit = float(q - 2.0 * np.dot(coeff, b) + np.dot(coeff, A @ coeff))
            tolerance = 1e-10 * max(q, 1.0)
            if misfit < -tolerance:
                raise RuntimeError(
                    "Cached station normal equations produced a negative preferred-solution "
                    f"misfit for station index {block['station_index']}."
                )
            misfit = max(misfit, 0.0)
            full_misfit += misfit
            n_values = int(block["slice"].stop - block["slice"].start)
            full_station_metrics[block["station_index"]] = {
                "misfit": misfit,
                "rms": float(np.sqrt(misfit / n_values)),
                "fraction": np.nan,
            }

        if full_misfit > 0.0:
            for metrics in full_station_metrics.values():
                metrics["fraction"] = metrics["misfit"] / full_misfit
    else:
        # Backward-compatible path for solution objects without the 0.2 cache.
        # This is exact but expensive because every GF must be reread/refiltered.
        full_G = _jackknife_design_matrix(solution, data, full_gp)
        if factorized_noise:
            full_G = whiten_covariance_array(full_G, covariance_factors, stations, npts)
        full_residual = D[:, full_shift_index] - full_G @ full_a[: full_G.shape[1], 0]
        full_misfit = float(np.dot(full_residual, full_residual))
        for block in blocks:
            sl = block["slice"]
            block_residual = full_residual[sl]
            misfit = float(np.dot(block_residual, block_residual))
            full_station_metrics[block["station_index"]] = {
                "misfit": misfit,
                "rms": float(np.sqrt(misfit / block_residual.size)),
                "fraction": misfit / full_misfit if full_misfit > 0 else np.nan,
            }

    if not np.isclose(
        full_misfit,
        stored_full_misfit,
        rtol=1e-8,
        atol=1e-8 * max(1.0, abs(stored_full_misfit)),
    ):
        raise RuntimeError(
            "Jackknife sufficient statistics do not reproduce the stored preferred-cell misfit: "
            f"recomputed={full_misfit:.12g}, stored={stored_full_misfit:.12g}."
        )

    best: dict[int, dict[str, Any] | None] = {block["station_index"]: None for block in blocks}

    # If the requested minimum cannot be met, the output rows are still useful
    # diagnostics but there is no reason to scan the final grid.
    if n_active_stations - 1 >= jackknife_min_stations and valid_grid_points:
        if cached_fast_path:
            grid_iterator = valid_grid_points

            try:
                from threadpoolctl import threadpool_limits
            except ImportError:
                blas_context = nullcontext()
            else:
                blas_context = threadpool_limits(limits=1, user_api="blas")

            with blas_context:
                for gp in grid_iterator:
                    candidates = _jackknife_grid_candidates_cached(
                        gp,
                        blocks=blocks,
                        q_total=q_total,
                        q_station=q_station,
                        deviatoric=bool(solution.deviatoric),
                    )
                    for station_index, candidate in candidates.items():
                        current = best[station_index]
                        if (
                            current is None
                            or float(candidate["variance_reduction"])
                            > float(current["variance_reduction"])
                        ):
                            best[station_index] = candidate
        else:
            def evaluate_grid_point(gp):
                return _jackknife_grid_candidates(
                    solution, data, gp,
                    factorized_noise=factorized_noise,
                    covariance_factors=covariance_factors,
                    stations=stations,
                    npts=npts,
                    D=D,
                    q_total=q_total,
                    blocks=blocks,
                    q_station=q_station,
                )

            n_workers = min(threads, len(valid_grid_points))
            if n_workers > 1:
                try:
                    from threadpoolctl import threadpool_limits
                except ImportError:
                    blas_context = nullcontext()
                else:
                    # Each jackknife task is already parallel at the Python level.
                    # Prevent small BLAS operations from spawning nested worker pools.
                    blas_context = threadpool_limits(limits=1, user_api="blas")

                executor = ThreadPoolExecutor(max_workers=n_workers)
                with blas_context, executor:
                    grid_iterator = executor.map(evaluate_grid_point, valid_grid_points)
                    for candidates in grid_iterator:
                        for station_index, candidate in candidates.items():
                            current = best[station_index]
                            if (
                                current is None
                                or float(candidate["variance_reduction"])
                                > float(current["variance_reduction"])
                            ):
                                best[station_index] = candidate
            else:
                grid_iterator = valid_grid_points
                for gp in grid_iterator:
                    candidates = evaluate_grid_point(gp)
                    for station_index, candidate in candidates.items():
                        current = best[station_index]
                        if (
                            current is None
                            or float(candidate["variance_reduction"])
                            > float(current["variance_reduction"])
                        ):
                            best[station_index] = candidate

    rows: list[dict[str, Any]] = []
    for block in blocks:
        station_index = block["station_index"]
        station = block["station"]
        n_remaining = n_active_stations - 1
        used_components = block["used_components"]
        native_vr = [
            station.get(f"VR_{comp}") for comp in used_components
            if station.get(f"VR_{comp}") is not None and np.isfinite(float(station.get(f"VR_{comp}")))
        ]
        base = {
            "network": station.get("network"),
            "station": station.get("code"),
            "location": station.get("location"),
            "n_components": len(used_components),
            "n_stations_remaining": n_remaining,
            "distance_km": float(station.get("dist", np.nan)) / 1000.0,
            "azimuth_deg": station.get("az", np.nan),
            "full_Mw": full_decomp.get("Mw", np.nan),
            "full_depth_km": float(full_gp["z"]) / 1000.0,
            "full_station_fit": float(np.mean(native_vr)) if native_vr else np.nan,
            "full_station_whitened_rms": full_station_metrics[station_index]["rms"],
            "full_station_misfit_fraction": full_station_metrics[station_index]["fraction"],
        }
        if n_remaining < jackknife_min_stations:
            rows.append({
                **base,
                "loo_Mw": np.nan, "delta_Mw": np.nan,
                "loo_depth_km": np.nan, "delta_depth_km": np.nan,
                "centroid_shift_km": np.nan, "delta_time_s": np.nan,
                "kagan_angle_deg": np.nan, "loo_condition_number": np.nan,
                "loo_variance_reduction": np.nan,
                "loo_DC_percent": np.nan, "loo_CLVD_percent": np.nan, "loo_ISO_percent": np.nan,
                "loo_NP1_strike_deg": np.nan, "loo_NP1_dip_deg": np.nan, "loo_NP1_rake_deg": np.nan,
                "loo_NP2_strike_deg": np.nan, "loo_NP2_dip_deg": np.nan, "loo_NP2_rake_deg": np.nan,
                "heldout_whitened_misfit": np.nan,
                "heldout_mean_squared_whitened_residual": np.nan,
                "heldout_rms_whitened_residual": np.nan,
                "loo_n_azimuth_sectors": np.nan, "loo_azimuthal_gap_deg": np.nan,
                "loo_on_grid_edge": np.nan,
                "qc_flags": "insufficient_remaining_stations",
            })
            continue

        candidate = best[station_index]
        if candidate is None:
            rows.append({
                **base,
                "loo_Mw": np.nan, "delta_Mw": np.nan,
                "loo_depth_km": np.nan, "delta_depth_km": np.nan,
                "centroid_shift_km": np.nan, "delta_time_s": np.nan,
                "kagan_angle_deg": np.nan, "loo_condition_number": np.nan,
                "loo_variance_reduction": np.nan,
                "loo_DC_percent": np.nan, "loo_CLVD_percent": np.nan, "loo_ISO_percent": np.nan,
                "loo_NP1_strike_deg": np.nan, "loo_NP1_dip_deg": np.nan, "loo_NP1_rake_deg": np.nan,
                "loo_NP2_strike_deg": np.nan, "loo_NP2_dip_deg": np.nan, "loo_NP2_rake_deg": np.nan,
                "heldout_whitened_misfit": np.nan,
                "heldout_mean_squared_whitened_residual": np.nan,
                "heldout_rms_whitened_residual": np.nan,
                "loo_n_azimuth_sectors": np.nan, "loo_azimuthal_gap_deg": np.nan,
                "loo_on_grid_edge": np.nan,
                "qc_flags": "no_valid_loo_solution",
            })
            continue

        gp = candidate["gp"]
        a = candidate["a"]
        dec = _align_nodal_planes(decompose(a2mt(a)), full_decomp)
        loo_mt = _moment_tensor_matrix_from_a(a)
        try:
            kagan = float(kagan_angle_mt(full_mt, loo_mt))
        except ValueError:
            kagan = np.nan
        loo_shift_s = float(data.shifts[int(candidate["shift_index"])])
        full_shift_s = float(full_gp["shift"])
        spatial_shift_km = math.sqrt(
            (float(gp["x"]) - float(full_gp["x"])) ** 2
            + (float(gp["y"]) - float(full_gp["y"])) ** 2
            + (float(gp["z"]) - float(full_gp["z"])) ** 2
        ) / 1000.0
        flags = _grid_boundary_flags(solution.g, gp)
        n_sectors, az_gap = _azimuth_geometry_from_solution_stations(stations, omitted_index=station_index)
        heldout_misfit = float(candidate["heldout_misfit"])
        heldout_n = int(candidate["heldout_n_values"])
        qc_flags = []
        if bool(flags["on_active_spatial_boundary"]):
            qc_flags.append("loo_grid_boundary")
        if n_remaining == jackknife_min_stations:
            qc_flags.append("minimum_station_count")

        rows.append({
            **base,
            "loo_Mw": dec["Mw"], "delta_Mw": dec["Mw"] - full_decomp["Mw"],
            "loo_depth_km": float(gp["z"]) / 1000.0,
            "delta_depth_km": float(gp["z"] - full_gp["z"]) / 1000.0,
            "centroid_shift_km": spatial_shift_km,
            "delta_time_s": loo_shift_s - full_shift_s,
            "kagan_angle_deg": kagan,
            "loo_condition_number": float(
                np.sqrt(np.linalg.cond(np.asarray(candidate["normal_matrix"], dtype=float)))
            ),
            "loo_variance_reduction": candidate["variance_reduction"],
            "loo_DC_percent": dec["dc_perc"],
            "loo_CLVD_percent": dec["clvd_perc"],
            "loo_ISO_percent": dec["iso_perc"],
            "loo_NP1_strike_deg": dec["s1"], "loo_NP1_dip_deg": dec["d1"], "loo_NP1_rake_deg": dec["r1"],
            "loo_NP2_strike_deg": dec["s2"], "loo_NP2_dip_deg": dec["d2"], "loo_NP2_rake_deg": dec["r2"],
            "heldout_whitened_misfit": heldout_misfit,
            "heldout_mean_squared_whitened_residual": heldout_misfit / heldout_n,
            "heldout_rms_whitened_residual": math.sqrt(heldout_misfit / heldout_n),
            "loo_n_azimuth_sectors": n_sectors,
            "loo_azimuthal_gap_deg": az_gap,
            "loo_on_grid_edge": bool(flags["on_active_spatial_boundary"]),
            "qc_flags": ";".join(qc_flags),
        })

    return pd.DataFrame(rows).sort_values(
        ["heldout_rms_whitened_residual", "kagan_angle_deg"],
        ascending=[False, False], na_position="last", ignore_index=True,
    )


def _clear_station_normal_equation_cache(solution) -> None:
    """Remove internal jackknife sufficient statistics from grid-point dictionaries."""
    for gp in solution.grid:
        gp.pop("_station_normal_indices", None)
        gp.pop("_station_GtG", None)
        gp.pop("_station_Gtd", None)


def _build_results(
    solution,
    *,
    n_uncertainty: int | None = None,
    uncertainty_scale: str | float = "fixed",
    uncertainty_scale_floor: float = 1.0,
    uncertainty_random_state: int | None = None,
    grid_edge_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the curated in-memory results layer used by ``run_auto_cmt``."""
    centroid = pd.DataFrame([extract_centroid_location(solution)])
    summary = pd.DataFrame([extract_solution_summary(solution)])
    station_fit = extract_station_fit_df(solution)

    variance_scale, scale_diag = _resolve_uncertainty_variance_scale(
        solution, uncertainty_scale, minimum_scale=float(uncertainty_scale_floor)
    )
    posterior_cells = build_posterior_cells(solution, variance_scale=variance_scale)
    posterior_diag = compute_posterior_diagnostics(
        solution, posterior_cells, variance_scale_diagnostics=scale_diag
    )

    uncertainty = None
    uncertainty_diagnostics = None
    if n_uncertainty is not None:
        n_uncertainty = int(n_uncertainty)
        if n_uncertainty <= 0:
            raise ValueError("n_uncertainty must be a positive integer or None.")
        uncertainty, diagnostics = extract_uncertainty_df(
            solution,
            n=n_uncertainty,
            variance_scale=variance_scale,
            posterior_cells=posterior_cells,
            random_state=uncertainty_random_state,
        )
        diagnostics.update(scale_diag)
        uncertainty_diagnostics = pd.DataFrame([diagnostics])

    return {
        "centroid": centroid,
        "summary": summary,
        "station_fit": station_fit,
        "posterior_cells": posterior_cells,
        "posterior_diagnostics": pd.DataFrame([posterior_diag]),
        "uncertainty": uncertainty,
        "uncertainty_diagnostics": uncertainty_diagnostics,
        "grid_edge_report": pd.DataFrame([dict(grid_edge_report)]) if grid_edge_report is not None else None,
    }


def _write_result_tables(
    results: Mapping[str, Any],
    output_dir: str | Path,
    *,
    save_posterior_cells: bool = False,
) -> dict[str, Path]:
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
    paths["posterior_diagnostics"] = output_dir / "posterior_diagnostics.csv"
    results["posterior_diagnostics"].to_csv(paths["posterior_diagnostics"], index=False)

    if save_posterior_cells:
        paths["posterior_cells"] = output_dir / "posterior_cells.csv"
        results["posterior_cells"].to_csv(paths["posterior_cells"], index=False)

    if results.get("uncertainty") is not None:
        paths["solution_unc_df"] = output_dir / "solution_unc_df.csv"
        results["uncertainty"].to_csv(paths["solution_unc_df"], index=False)
        paths["solution_unc_diagnostics"] = output_dir / "solution_unc_diagnostics.csv"
        results["uncertainty_diagnostics"].to_csv(paths["solution_unc_diagnostics"], index=False)

    if results.get("station_jackknife") is not None:
        paths["station_jackknife"] = output_dir / "station_jackknife.csv"
        results["station_jackknife"].to_csv(paths["station_jackknife"], index=False)

    return paths


def write_solution_outputs(
    solution,
    output_dir: str | Path,
    *,
    n_uncertainty: int | None = None,
    uncertainty_scale: str | float = "fixed",
    uncertainty_scale_floor: float = 1.0,
    uncertainty_random_state: int | None = None,
    save_posterior_cells: bool = False,
) -> dict[str, Path]:
    """Extract and write BayesISOLA scientific result tables.

    The 0.2 workflow uses exact log-space posterior cells and categorical
    nonlinear-cell sampling. ``uncertainty_scale='fixed'`` reproduces the 0.1.1
    covariance scale, while ``'residual'`` multiplies conditional MT covariance
    by the preferred-cell reduced chi-square and applies the same scalar scale to
    relative space-time likelihood differences.
    """
    results = _build_results(
        solution,
        n_uncertainty=n_uncertainty,
        uncertainty_scale=uncertainty_scale,
        uncertainty_scale_floor=uncertainty_scale_floor,
        uncertainty_random_state=uncertainty_random_state,
    )
    return _write_result_tables(results, output_dir, save_posterior_cells=save_posterior_cells)



# Historical native BayesISOLA keyword presets are retained as public workflow
# metadata for backward compatibility. ``run_auto_cmt`` now treats ``summary`` as
# a workflow-diagnostic preset and passes only the explicit ``full`` native preset
# to ``BayesISOLA.plot``; the dictionaries therefore remain native-keyword only.
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


# Native BayesISOLA defaults used specifically to populate ``index.html``.
# These mirror ``BayesISOLA.plot.__init__`` rather than any workflow preset.
# Keeping them separate is essential: ``plot_preset="summary"`` must not trim
# the historical HTML, and HTML-only figures must not be added to ``figure_paths``
# or displayed by ``show=True``.
_NATIVE_HTML_PLOT_PRESET: dict[str, Any] = dict(
    maps=True,
    slices=True,
    maps_sum=True,
    MT=True,
    uncertainty=400,
    seismo=False,
    seismo_sharey=True,
    seismo_cova=True,
    noise=True,
    spectra=True,
    stations=True,
    covariance_matrix=True,
    covariance_function=False,
)


def _native_plot_kwargs(options: Mapping[str, Any], *, use_noise: bool) -> dict[str, Any]:
    """Return native BayesISOLA plotting options safe for the covariance mode.

    Parameters
    ----------
    options : mapping
        Native keyword arguments accepted by :class:`BayesISOLA.plot`.
    use_noise : bool
        Whether the workflow estimated a noise covariance matrix. Native plots
        that require the saved noise/covariance products are disabled when this
        is false; all other requested native figures are preserved.

    Returns
    -------
    dict
        Independent copy of ``options`` suitable for ``BayesISOLA.plot``.
    """
    kwargs = dict(options)
    if not use_noise:
        kwargs.update(
            seismo_cova=False,
            noise=False,
            spectra=False,
            covariance_matrix=False,
            covariance_function=False,
        )
    return kwargs


def _render_native_outputs(
    solution,
    output_path: str | Path,
    *,
    event_id: str,
    plot: bool,
    plot_preset: str,
    html_output: bool,
    use_noise: bool,
    detect_mouse: bool,
):
    """Generate native BayesISOLA plot products without coupling HTML to presets.

    Workflow ``summary`` figures are handled by :mod:`BayesISOLA._diagnostics`
    and therefore do not instantiate the historical native plot suite. Native
    figures are generated in two cases only:

    * ``plot_preset='full'`` explicitly requests the native full plot products;
      those files are returned for normal ``figure_paths``/``show`` handling.
    * ``html_output=True`` requests the historical ``index.html``. In this case
      the complete native HTML figure suite is generated regardless of
      ``plot_preset``. HTML-only figures are intentionally *not* returned for
      notebook display.

    When both cases apply, the native full workflow preset is generated once,
    captured for normal display, and only the native uncertainty products missing
    from that preset are added silently before ``html_log()`` is written.

    Returns
    -------
    tuple
        ``(plot_object, displayable_native_figures, native_html_path)``.
    """
    import BayesISOLA

    output_path = Path(output_path).expanduser()
    request_full = bool(plot and plot_preset == "full")

    if not request_full and not html_output:
        return None, [], None

    plot_object = None
    displayable_native_figures: list[Path] = []

    if request_full:
        before = _png_state(output_path)
        kwargs = _native_plot_kwargs(PLOT_PRESETS["full"], use_noise=use_noise)
        plot_object = BayesISOLA.plot(solution, **kwargs)
        # Capture only products explicitly requested by ``plot_preset='full'``.
        # Any additional HTML-only products generated below remain silent.
        displayable_native_figures = _changed_pngs(output_path, before)

    if html_output:
        if plot_object is None:
            kwargs = _native_plot_kwargs(
                _NATIVE_HTML_PLOT_PRESET,
                use_noise=use_noise,
            )
            plot_object = BayesISOLA.plot(solution, **kwargs)
        elif not plot_object.plots.get("uncertainty"):
            # ``PLOT_PRESETS['full']`` deliberately disables the historical
            # uncertainty sampler because the workflow has its own diagnostics.
            # Native HTML, however, should retain the original BayesISOLA suite.
            plot_object.plot_uncertainty(n=int(_NATIVE_HTML_PLOT_PRESET["uncertainty"]))

        plot_object.html_log(
            h1=f"BayesISOLA CMT — {event_id}",
            mouse_figures="mouse/" if detect_mouse else None,
        )

        candidate = output_path / "index.html"
        if not candidate.is_file() or candidate.stat().st_size == 0:
            raise OSError(
                f"Native BayesISOLA HTML was not written correctly: {candidate}"
            )
        native_html_path = candidate
    else:
        native_html_path = None

    return plot_object, displayable_native_figures, native_html_path


_DIAGNOSTIC_PLOT_PRESETS: dict[str, tuple[str, ...]] = {
    "none": (),
    "summary": (
        "cmt",
        "posterior",
        "adaptive",
        "station_qc",
        "uncertainty",
        "station_fit",
    ),
    "full": (
        "cmt",
        "posterior",
        "adaptive",
        "station_qc",
        "uncertainty",
        "station_fit",
    ),
}


def _save_diagnostic_figure(fig, path: str | Path, *, dpi: int = 200) -> Path:
    """Save and close a workflow diagnostic figure, verifying the output file."""
    if fig is None or not hasattr(fig, "savefig"):
        raise TypeError("Diagnostic plotter did not return a Matplotlib figure.")
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt
    try:
        fig.savefig(path, dpi=int(dpi), bbox_inches="tight")
    finally:
        plt.close(fig)
    if not path.is_file() or path.stat().st_size == 0:
        raise OSError(f"Diagnostic figure was not written correctly: {path}")
    return path


def _write_diagnostic_preset(
    run: Mapping[str, Any],
    preset: str,
    output_dir: str | Path,
    *,
    tensor_mode: str,
    dpi: int = 200,
) -> list[Path]:
    """Write the workflow-level figures associated with a plot preset.

    This helper intentionally operates on the completed public ``run`` mapping,
    exactly like the standalone functions in :mod:`BayesISOLA._diagnostics`.
    Keeping this layer separate from ``PLOT_PRESETS`` prevents diagnostic-only
    keywords from leaking into the native ``BayesISOLA.plot`` constructor.

    Parameters
    ----------
    run : mapping
        Completed or nearly completed ``run_auto_cmt`` mapping containing
        ``results``, ``adaptive_history`` and ``station_selection``.
    preset : {"none", "summary", "full"}
        Workflow plotting preset. ``summary`` and ``full`` currently write the
        same curated diagnostic set; native BayesISOLA plots remain different.
    output_dir : str or pathlib.Path
        Directory receiving diagnostic PNG files.
    tensor_mode : {"full", "deviatoric"}
        Inversion tensor type passed to :func:`plot_cmt_summary`.
    dpi : int, default=200
        Saved figure resolution.

    Returns
    -------
    list of pathlib.Path
        Diagnostic files actually generated. Optional figures are omitted when
        their source data are unavailable (for example uncertainty samples).
    """
    preset = str(preset).lower().strip()
    if preset not in _DIAGNOSTIC_PLOT_PRESETS:
        raise ValueError(
            f"preset must be one of {tuple(_DIAGNOSTIC_PLOT_PRESETS)}."
        )
    tensor_mode = str(tensor_mode).lower().strip()
    if tensor_mode not in {"full", "deviatoric"}:
        raise ValueError("tensor_mode must be 'full' or 'deviatoric'.")
    if not isinstance(run, Mapping):
        raise TypeError("run must be the mapping returned by run_auto_cmt.")
    results = run.get("results")
    if not isinstance(results, Mapping):
        raise TypeError("run['results'] must be a mapping.")

    output_dir = Path(output_dir).expanduser()
    names = _DIAGNOSTIC_PLOT_PRESETS[preset]
    if not names:
        return []

    paths: list[Path] = []

    def save(name: str, fig) -> None:
        paths.append(_save_diagnostic_figure(fig, output_dir / name, dpi=dpi))

    if "cmt" in names:
        save(
            "cmt_summary.png",
            plot_cmt_summary(
                results["summary"], results["centroid"],
                tensor_mode=tensor_mode, show=False,
            ),
        )
    if "posterior" in names:
        save("posterior_summary.png", plot_posterior_summary(run))

    adaptive_history = run.get("adaptive_history")
    if (
        "adaptive" in names
        and isinstance(adaptive_history, pd.DataFrame)
        and not adaptive_history.empty
    ):
        save("adaptive_grid_summary.png", plot_adaptive_history(run))

    station_selection = run.get("station_selection")
    if (
        "station_qc" in names
        and isinstance(station_selection, pd.DataFrame)
        and not station_selection.empty
    ):
        save("station_qc_summary.png", plot_station_qc(run))

    uncertainty = results.get("uncertainty")
    if (
        "uncertainty" in names
        and isinstance(uncertainty, pd.DataFrame)
        and not uncertainty.empty
    ):
        save("uncertainty_summary.png", plot_uncertainty_summary(run))

    if "station_fit" in names:
        save("station_fit_summary.png", plot_station_fit_summary(run, show=False))

    return paths


def _adaptive_stage_record(
    stage_index: int,
    stage_type: str,
    grid,
    solution,
    proposal: Mapping[str, Any] | None = None,
    posterior_diagnostics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return one compact row describing an adaptive inversion stage."""
    report = diagnose_grid_edge(grid, centroid=solution.centroid)
    depths = sorted({float(gp["z"]) / 1000.0 for gp in grid.grid if not gp.get("err")})
    row = {
        "stage_index": int(stage_index),
        "stage_type": str(stage_type),
        "n_grid_points": int(len(grid.grid)),
        "radius_km": float(grid.radius) / 1000.0,
        "step_x_km": float(grid.step_x) / 1000.0,
        "step_z_km": float(grid.step_z) / 1000.0,
        "depth_min_km": depths[0] if depths else np.nan,
        "depth_max_km": depths[-1] if depths else np.nan,
        "centroid_north_km": float(solution.centroid["x"]) / 1000.0,
        "centroid_east_km": float(solution.centroid["y"]) / 1000.0,
        "centroid_depth_km": float(solution.centroid["z"]) / 1000.0,
        "centroid_time_shift_s": float(solution.centroid["shift"]),
        "variance_reduction": float(solution.centroid["VR"]),
        "condition_number": float(solution.centroid["CN"]),
        "horizontal_search_fixed": bool(report["horizontal_search_fixed"]),
        "on_horizontal_boundary": bool(report.get("centroid_on_horizontal_boundary", False)),
        "on_depth_floor": bool(report.get("centroid_on_depth_floor", False)),
        "on_depth_ceiling": bool(report.get("centroid_on_depth_ceiling", False)),
        "on_active_spatial_boundary": bool(report.get("centroid_on_active_spatial_boundary", False)),
        "centroid_edge_reasons": report.get("centroid_edge_reasons", []),
    }
    if posterior_diagnostics is not None:
        row.update({
            "posterior_mode_probability": posterior_diagnostics.get("posterior_mode_probability"),
            "posterior_effective_cells": posterior_diagnostics.get("posterior_effective_cells"),
            "posterior_horizontal_boundary_probability": posterior_diagnostics.get("posterior_horizontal_boundary_probability"),
            "posterior_depth_floor_probability": posterior_diagnostics.get("posterior_depth_floor_probability"),
            "posterior_depth_ceiling_probability": posterior_diagnostics.get("posterior_depth_ceiling_probability"),
            "posterior_active_spatial_boundary_probability": posterior_diagnostics.get("posterior_active_spatial_boundary_probability"),
            "uncertainty_variance_scale": posterior_diagnostics.get("uncertainty_variance_scale"),
            "uncertainty_sd_scale": posterior_diagnostics.get("uncertainty_sd_scale"),
            "reduced_chi_square": posterior_diagnostics.get("reduced_chi_square"),
        })
    if proposal is not None:
        row["applied_proposal_reason"] = proposal.get("reason")
        row["applied_grid_points_estimate"] = proposal.get("estimated_grid_points")
    return row


def _set_fixed_adaptive_time_window(data, grid, processing_depth_max_km: float | None) -> None:
    """Keep waveform/GF time support invariant across adaptive stages.

    BayesISOLA derives ``t_max`` from ``grid.depth_max``. A refinement that
    narrows the searched depth interval would otherwise shorten the data vector,
    Green-function duration and noise covariance, so different adaptive stages
    would no longer evaluate the same likelihood. Temporarily exposing the
    maximum admissible adaptive depth to ``set_time_window`` preserves the search
    grid while fixing the inversion time support.
    """
    if processing_depth_max_km is None:
        return
    processing_depth_max_m = float(processing_depth_max_km) * 1000.0
    if not np.isfinite(processing_depth_max_m) or processing_depth_max_m <= 0:
        raise ValueError("processing_depth_max_km must be positive and finite or None.")
    search_depth_max = float(grid.depth_max)
    grid.depth_max = max(search_depth_max, processing_depth_max_m)
    try:
        data.set_time_window()
        data.set_Greens_parameters()
    finally:
        grid.depth_max = search_depth_max


def _run_adaptive_axitra_stage(
    *,
    inputs,
    radius_km: float,
    depth_min_km: float,
    depth_max_km: float,
    step_x_km: float,
    step_z_km: float,
    max_grid_points: int,
    time_unc_s: float,
    rupture_velocity_m_s: float,
    velocity_slowest_m_s: float,
    freqmin: float,
    freqmax: float,
    threads: int,
    progress: bool,
    invert_displacement: bool,
    use_precalculated_Green: bool | str,
    use_noise: bool,
    crosscovariance: bool,
    deviatoric: bool,
    normalized_gf_options: Mapping[str, Any],
    processing_depth_max_km: float | None = None,
    save_non_inverted_covariance: bool = False,
    store_station_normal_equations: bool = False,
) -> tuple[Any, Any, Any, Any, bool]:
    """Rerun Axitra on an explicit adaptive grid without reacquiring waveforms.

    ``inputs.data_raw`` already contains response-corrected waveforms after the
    first stage, so ``correct_data=False`` is essential here.  A changed grid
    invalidates native Axitra's whole-grid cache metadata; ``'auto'`` therefore
    regenerates the required Green functions.  The 0.2 adaptive search intentionally
    does not attempt partial GF reuse across differently indexed grids.
    """
    import BayesISOLA

    event_depth_km = float(inputs.event["depth"]) / 1000.0
    depth_unc_km = max(abs(event_depth_km - float(depth_min_km)),
                       abs(float(depth_max_km) - event_depth_km))
    grid = BayesISOLA.grid(
        inputs,
        location_unc=0.0,
        depth_unc=depth_unc_km * 1000.0,
        time_unc=float(time_unc_s),
        step_x=float(step_x_km) * 1000.0,
        step_z=float(step_z_km) * 1000.0,
        max_points=int(max_grid_points),
        grid_radius=float(radius_km) * 1000.0 if float(radius_km) > 0 else 0.0,
        grid_min_depth=float(depth_min_km) * 1000.0,
        grid_max_depth=float(depth_max_km) * 1000.0,
        circle_shape=True,
        add_rupture_length=False,
        rupture_velocity=float(rupture_velocity_m_s),
    )

    data = BayesISOLA.process_data(
        inputs,
        grid,
        threads=int(threads),
        progress=bool(progress),
        invert_displacement=bool(invert_displacement),
        use_precalculated_Green=use_precalculated_Green,
        velocity_ot_the_slowest_wave=float(velocity_slowest_m_s),
        fmax=float(freqmax),
        fmin=float(freqmin),
        min_depth=float(depth_min_km) * 1000.0,
        correct_data=False,
        calculate_or_verify_Green=False,
        trim_filter_data=False,
        decimate_shift=False,
    )
    _set_fixed_adaptive_time_window(data, grid, processing_depth_max_km)

    if normalized_gf_options["grid"] is None:
        before_green = _axitra_green_state(data)
        data.calculate_or_verify_Green()
        after_green = _axitra_green_state(data)
        reused = bool(use_precalculated_Green is not False and before_green and before_green == after_green)
    else:
        reused = _calculate_or_verify_axitra_multimodel(data, use_precalculated_Green)

    data.trim_filter_data(noise_slice=use_noise)
    data.decimate_shift()

    # Retain small station-wise normal equations while G is already in memory
    # when the exact fixed-grid jackknife has been requested.
    data._store_station_normal_equations = bool(store_station_normal_equations)

    cova = BayesISOLA.covariance_matrix(data)
    if use_noise:
        cova.covariance_matrix_noise(
            crosscovariance=bool(crosscovariance),
            save_non_inverted=bool(save_non_inverted_covariance),
            save_covariance_function=False,
        )

    solution = BayesISOLA.resolve_MT(
        data,
        cova,
        deviatoric=bool(deviatoric),
        VR_of_components=True,
    )
    return grid, data, cova, solution, reused


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
    channel_priority: ChannelPriority = ("HH", "BH", "LH"),
    taup_model: str = "iasp91",
    overwrite_waveforms: bool = False,
    location_unc_km: float = 0.0,
    time_unc_s: float = 2.0,
    min_depth_km: float = 5.0,
    min_depth_multiplier: float = 0.5,
    max_depth_multiplier: float = 3.0,
    grid_radius_km: float | None = None,
    grid_min_depth_km: float | None = None,
    grid_max_depth_km: float | None = None,
    step_x_km: float = 2.0,
    step_z_km: float = 1.0,
    max_grid_points: int = 5000,
    add_rupture_length: bool = True,
    rupture_velocity_m_s: float = 1000.0,
    adaptive_grid_search: bool | Mapping[str, Any] | None = None,
    drop_stations: Sequence[str] | str | None = None,
    azimuth_control: bool | Mapping[str, Any] | None = None,
    station_jackknife: bool | Mapping[str, Any] | None = None,
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
    uncertainty_scale: str | float = "fixed",
    uncertainty_scale_floor: float = 1.0,
    uncertainty_random_state: int | None = None,
    save_posterior_cells: bool = False,
    plot: bool = True,
    plot_preset: str = "summary",
    show: bool = True,
    html_output: bool = True,
    write_report: bool = False,
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
    ``progress`` controls the native BayesISOLA progress indicators used for
    Axitra Green's-function calculation and moment-tensor inversion.  The 0.2
    station jackknife uses cached station-wise normal-equation contributions and
    is intentionally not given a separate progress bar because it no longer
    performs a second Green-function/filtering pass. Syngine retains its
    backend-specific ``gf_options['progress']`` control.

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
    By default the shallow grid bound is
    ``max(min_depth_km, event_depth_km * min_depth_multiplier)`` and the deep
    bound is ``event_depth_km * max_depth_multiplier``. ``grid_radius_km``,
    ``grid_min_depth_km`` and ``grid_max_depth_km`` expose the native explicit
    BayesISOLA grid controls; ``None`` preserves the 0.1.1 automatic behaviour.
    ``max_grid_points`` is BayesISOLA's approximate pre-construction rescaling
    target; the realised discrete grid remains authoritative.

    ``adaptive_grid_search=None`` preserves the non-adaptive behaviour. Passing a
    mapping enables the bounded Axitra adaptive search and keeps the public wrapper to
    one option. Unspecified mapping values use the documented 0.2 defaults, e.g.::

        adaptive_grid_search={
            "adaptive_grid": True,
            "adaptive_max_grid_points": 20000,
        }

    A genuine spatial boundary is expanded symmetrically about the catalogue
    epicentre before any refinement. ``adaptive_refine_factor=0`` skips the
    refinement stage while still allowing boundary expansion. Adaptive stages use
    one fixed waveform/GF time support so narrowing a refinement depth window does
    not silently change the likelihood.

    Waveform window and covariance
    ------------------------------
    ``covariance='noise'`` requests the long pre-event noise interval.
    ``crosscovariance=False`` retains only component-wise noise covariance,
    whereas ``True`` retains the full three-component station covariance.
    ``covariance='none'`` avoids the noise interval and invokes BayesISOLA's
    unweighted ordinary least-squares branch. ``velocity_slowest_m_s`` is passed
    to ``process_data`` so acquisition and BayesISOLA use the same timing bound.

    Station controls and robustness
    -------------------------------
    ``drop_stations`` is an explicit manual station exclusion list applied before
    any azimuthal thinning. ``azimuth_control=None``/``False`` leaves all retained
    stations untouched, ``True`` uses eight 45-degree GISOLA-style sectors with
    three occupied sectors and at most two stations per sector, and a mapping can
    override those defaults. Within each sector BayesISOLA honors
    ``channel_priority`` first and distance second. ``channel_priority`` may remain
    the historical sequence, e.g. ``('HH', 'BH', 'LH')``, or use station-specific
    magnitude/distance precedence rules, for example::

        channels=("HH?", "BH?", "LH?"),
        channel_priority={
            "mag_range": [[4.0, 5.0], [5.0, 6.0]],
            "dist_range": [[10, 250], [40, 300]],
            "channels": [["BH", "HH"], ["HN", "BN"]],
        }

    The rule intervals are ``[min, max)``; the first matching rule wins. Families
    not listed in a matching rule remain fallbacks in the default order inferred
    from ``channels`` (or supplied explicitly with ``default``). Rule-only families
    such as ``HN``/``BN`` above are automatically added to the FDSN metadata query.

    ``station_jackknife=True`` evaluates every active station omission on the final
    converged full-solution grid. A mapping currently exposes only
    ``jackknife_min_stations`` (default 4). The jackknife reuses existing Green
    functions and station-block normal equations; it never triggers an adaptive
    expansion/refinement, even when a leave-one-out mode lies on the grid boundary.
    Cached station-wise normal equations make this final-grid robustness scan
    inexpensive and avoid any second Green-function read/filter pass.

    Outputs and uncertainty
    -----------------------
    Deterministic centroid, moment-tensor summary, station/component fit, exact
    posterior diagnostics and grid-edge tables are always written. If
    ``n_uncertainty`` is requested, the 0.2 workflow draws nonlinear cells
    categorically and then samples the selected cell's conditional Gaussian MT.
    ``uncertainty_scale='fixed'`` preserves the native covariance scale, whereas
    ``'residual'`` applies the preferred-cell reduced-chi-square scale to the
    nonlinear likelihood and conditional MT covariance. This is a scalar residual
    calibration, not a substitute for structural/theory covariance. Native
    BayesISOLA uncertainty plotting remains disabled in the workflow-facing
    ``PLOT_PRESETS``; the workflow-level uncertainty diagnostic is generated when
    uncertainty samples exist. ``plot=True`` saves figures and ``plot_preset`` is
    ``'none'``, ``'summary'`` or ``'full'``. ``summary`` writes the curated CMT,
    exact-posterior, adaptive-history, station-QC, uncertainty (when sampled), and
    station-fit diagnostics from :mod:`BayesISOLA._diagnostics`; it does not request
    the historical native plot suite. ``full`` additionally requests BayesISOLA's
    native full plotting preset. ``show=True`` displays only these workflow-requested
    products.

    ``html_output=True`` is independent of ``plot_preset`` and writes the historical
    native BayesISOLA ``index.html`` using the complete native HTML figure suite.
    Those HTML-supporting figures are written to disk because ``html_log()`` needs
    them, but they are not added to ``run['figure_paths']`` and are therefore not
    displayed merely because ``show=True``. If ``plot_preset='full'`` explicitly
    requests the same native products, that request still controls their normal
    display/export semantics. ``write_report=True`` independently writes the curated
    workflow-level ``report.html`` from the completed ``run`` mapping.

    Returns
    -------
    dict
        Native BayesISOLA objects plus acquisition diagnostics, ``run['gf']``
        backend metadata, crust/model information where applicable, curated
        ``run['results']`` tables, result/figure paths, ``native_html_path``
        when the historical HTML renderer is enabled, and ``report_path`` when
        the curated workflow report is requested.
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
    _normalize_channel_priority(channel_priority, channels)
    for option_name, option_value in (("html_output", html_output), ("write_report", write_report)):
        if not isinstance(option_value, (bool, np.bool_)):
            raise TypeError(f"{option_name} must be True or False.")
    html_output = bool(html_output)
    write_report = bool(write_report)
    if float(freqmin) < 0 or float(freqmax) <= float(freqmin):
        raise ValueError("Require 0 <= freqmin < freqmax.")
    if waveform_source == "local" and station_df is None:
        raise ValueError("waveform_source='local' requires station_df from get_mseed_stationxml or an equivalent local-file table.")
    if use_precalculated_Green not in {False, True, "auto"}:
        raise ValueError("use_precalculated_Green must be False, True or 'auto'.")
    if grid_radius_km is not None and (not np.isfinite(float(grid_radius_km)) or float(grid_radius_km) < 0):
        raise ValueError("grid_radius_km must be None or a finite value >= 0.")
    if int(max_grid_points) <= 0:
        raise ValueError("max_grid_points must be positive.")
    adaptive_config = _normalize_adaptive_grid_search(adaptive_grid_search)
    azimuth_config = _normalize_azimuth_control(azimuth_control)
    jackknife_config = _normalize_station_jackknife(station_jackknife)
    adaptive_grid = bool(adaptive_config["adaptive_grid"])
    adaptive_expand_xy_steps = int(adaptive_config["adaptive_expand_xy_steps"])
    adaptive_expand_z_steps = int(adaptive_config["adaptive_expand_z_steps"])
    adaptive_max_expansions = int(adaptive_config["adaptive_max_expansions"])
    adaptive_max_refinements = int(adaptive_config["adaptive_max_refinements"])
    adaptive_refine_factor = float(adaptive_config["adaptive_refine_factor"])
    adaptive_min_step_fraction = float(adaptive_config["adaptive_min_step_fraction"])
    adaptive_depth_window_parent_steps = adaptive_config["adaptive_depth_window_parent_steps"]
    adaptive_max_radius_factor = float(adaptive_config["adaptive_max_radius_factor"])
    adaptive_max_depth_span_factor = float(adaptive_config["adaptive_max_depth_span_factor"])
    adaptive_max_grid_points = adaptive_config["adaptive_max_grid_points"]
    adaptive_max_total_reruns = int(adaptive_config["adaptive_max_total_reruns"])
    adaptive_expand_on_posterior_boundary = bool(
        adaptive_config["adaptive_expand_on_posterior_boundary"]
    )
    adaptive_boundary_probability_threshold = float(
        adaptive_config["adaptive_boundary_probability_threshold"]
    )

    if adaptive_grid and gf_source != "axitra":
        raise NotImplementedError("The 0.2 adaptive-grid search currently supports gf_source='axitra' only.")
    if adaptive_grid and use_precalculated_Green is True:
        raise ValueError("adaptive_grid_search requires use_precalculated_Green='auto' or False because a changed grid needs new Green functions.")
    adaptive_grid_point_budget = (
        None if adaptive_max_grid_points is None else int(adaptive_max_grid_points)
    )

    base_grid_min_depth_km, base_grid_max_depth_km = _depth_bounds_km(
        float(event_depth_km),
        min_depth_km=float(min_depth_km),
        min_depth_multiplier=float(min_depth_multiplier),
        max_depth_multiplier=float(max_depth_multiplier),
        grid_min_depth_km=grid_min_depth_km,
        grid_max_depth_km=grid_max_depth_km,
    )
    acquisition_grid_min_depth_km = base_grid_min_depth_km
    acquisition_grid_max_depth_km = base_grid_max_depth_km
    if adaptive_grid and float(adaptive_max_depth_span_factor) > 1.0:
        base_span = base_grid_max_depth_km - base_grid_min_depth_km
        extra_span = (float(adaptive_max_depth_span_factor) - 1.0) * base_span
        acquisition_grid_min_depth_km = max(float(min_depth_km), base_grid_min_depth_km - extra_span)
        acquisition_grid_max_depth_km = base_grid_max_depth_km + extra_span

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
            max_depth_multiplier=max_depth_multiplier,
            grid_min_depth_km=acquisition_grid_min_depth_km,
            grid_max_depth_km=acquisition_grid_max_depth_km, time_unc_s=time_unc_s,
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

    stations, station_selection = _apply_station_controls(
        stations,
        drop_stations=drop_stations,
        azimuth_config=azimuth_config,
        channel_priority=channel_priority,
        channels=channels,
        magnitude=float(magnitude),
        event_lat=float(event_lat),
        event_lon=float(event_lon),
    )
    station_selection_path = metadata_path / "station_selection.csv"
    station_selection.to_csv(station_selection_path, index=False)

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
    # has been read. Earlier workflow ordering copied the crust before those
    # model keys existed.
    if gf_source == "axitra":
        inputs.read_crust(str(case_crust_file))

    automatic_grid_radius_km = float(location_unc_km) + (
        inputs.rupture_length / 1000.0 if add_rupture_length else 0.0
    )
    resolved_grid_radius_km = (
        automatic_grid_radius_km if grid_radius_km is None else float(grid_radius_km)
    )
    depth_spec = suggest_depth_limits(
        float(event_depth_km), min_depth_km=float(min_depth_km),
        min_depth_multiplier=float(min_depth_multiplier),
        max_depth_multiplier=float(max_depth_multiplier),
        grid_min_depth_km=base_grid_min_depth_km,
        grid_max_depth_km=base_grid_max_depth_km,
        step_z_km=float(step_z_km), step_x_km=float(step_x_km),
        radius_km=resolved_grid_radius_km, max_points=int(max_grid_points),
    )
    explicit_radius = grid_radius_km is not None
    grid = BayesISOLA.grid(
        inputs,
        location_unc=0.0 if explicit_radius else float(location_unc_km) * 1000.0,
        depth_unc=depth_spec["depth_unc_km"] * 1000.0,
        time_unc=float(time_unc_s), step_x=float(step_x_km) * 1000.0,
        step_z=float(step_z_km) * 1000.0, max_points=int(max_grid_points),
        grid_radius=resolved_grid_radius_km * 1000.0 if explicit_radius and resolved_grid_radius_km > 0 else 0.0,
        grid_min_depth=depth_spec["grid_min_depth_km"] * 1000.0,
        grid_max_depth=depth_spec["grid_max_depth_km"] * 1000.0,
        circle_shape=True,
        add_rupture_length=False if explicit_radius else bool(add_rupture_length),
        rupture_velocity=float(rupture_velocity_m_s),
    )

    waveform_window = get_waveform_window(
        float(event_depth_km), float(magnitude), station_df=stations,
        radius_scale_factor=radius_scale_factor, min_depth_km=min_depth_km,
        min_depth_multiplier=min_depth_multiplier,
        max_depth_multiplier=max_depth_multiplier,
        grid_min_depth_km=acquisition_grid_min_depth_km,
        grid_max_depth_km=acquisition_grid_max_depth_km, time_unc_s=time_unc_s,
        rupture_velocity_m_s=rupture_velocity_m_s,
        velocity_slowest_m_s=velocity_slowest_m_s, covariance=covariance,
        noise_factor=noise_factor, edge_margin_s=edge_margin_s,
        minimum_pre_event_s=minimum_pre_event_s,
    )
    loaded_stations, load_log = load_streams_local(
        inputs, stations, t_before=waveform_window["t_before_s"],
        t_after=waveform_window["t_after_s"],
    )
    _validate_selected_azimuth_geometry(loaded_stations, azimuth_config)
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

    # Mouse diagnostics belong to the native HTML/full-native suite, not to the
    # workflow summary preset. Generate them whenever either consumer needs them.
    mouse_figures = (
        output_path / "mouse"
        if detect_mouse and (html_output or (plot and plot_preset == "full"))
        else False
    )
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
    if adaptive_grid:
        _set_fixed_adaptive_time_window(data, grid, acquisition_grid_max_depth_km)

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

    # Piggyback exact station-jackknife sufficient statistics on the inversion
    # pass, avoiding a second read/filter pass through every elementary seismogram.
    data._store_station_normal_equations = bool(jackknife_config["enabled"])

    cova = BayesISOLA.covariance_matrix(data)
    if use_noise:
        cova.covariance_matrix_noise(
            crosscovariance=bool(crosscovariance),
            # Native covariance-matrix plotting requires the non-inverted
            # covariance blocks to be retained. HTML generation must therefore
            # request them independently of the workflow plot preset.
            save_non_inverted=bool(html_output or (plot and plot_preset == "full")),
            save_covariance_function=False,
        )

    solution = BayesISOLA.resolve_MT(
        data, cova, deviatoric=bool(deviatoric), VR_of_components=True
    )

    initial_depths = sorted({float(gp["z"]) / 1000.0 for gp in grid.grid if not gp.get("err")})
    initial_radius_km = float(grid.radius) / 1000.0
    initial_step_x_km = float(grid.step_x) / 1000.0
    initial_step_z_km = float(grid.step_z) / 1000.0
    initial_depth_min_km = initial_depths[0]
    initial_depth_max_km = initial_depths[-1]
    def adaptive_stage_posterior(current_solution):
        stage_scale, stage_scale_diag = _resolve_uncertainty_variance_scale(
            current_solution, uncertainty_scale, minimum_scale=float(uncertainty_scale_floor)
        )
        stage_cells = build_posterior_cells(current_solution, variance_scale=stage_scale)
        stage_diag = compute_posterior_diagnostics(
            current_solution, stage_cells, variance_scale_diagnostics=stage_scale_diag
        )
        return stage_diag

    adaptive_stage_diag = adaptive_stage_posterior(solution) if adaptive_grid else {}
    adaptive_history_rows = [_adaptive_stage_record(
        0, "initial", grid, solution,
        posterior_diagnostics=adaptive_stage_diag if adaptive_grid else None,
    )]
    expansion_count = 0
    refinement_count = 0

    if adaptive_grid:
        while True:
            edge_report_now = diagnose_grid_edge(grid, centroid=solution.centroid)
            centroid_boundary = bool(edge_report_now.get("centroid_on_active_spatial_boundary", False))
            posterior_boundary_probabilities = [
                adaptive_stage_diag.get("posterior_horizontal_boundary_probability", np.nan),
                adaptive_stage_diag.get("posterior_depth_floor_probability", np.nan),
                adaptive_stage_diag.get("posterior_depth_ceiling_probability", np.nan),
            ]
            posterior_boundary_active = bool(adaptive_expand_on_posterior_boundary) and any(
                np.isfinite(value)
                and float(value) >= float(adaptive_boundary_probability_threshold)
                for value in posterior_boundary_probabilities
            )
            active_boundary = centroid_boundary or posterior_boundary_active
            decision_posterior_diagnostics = (
                adaptive_stage_diag if adaptive_expand_on_posterior_boundary else None
            )
            total_reruns = expansion_count + refinement_count
            if total_reruns >= int(adaptive_max_total_reruns):
                adaptive_history_rows[-1]["next_action"] = "stop_max_total_reruns"
                break

            if active_boundary:
                if expansion_count >= int(adaptive_max_expansions):
                    adaptive_history_rows[-1]["next_action"] = "stop_max_expansions"
                    break
                proposal = compute_grid_expansion(
                    grid,
                    solution.centroid,
                    initial_radius_km=initial_radius_km,
                    initial_depth_min_km=initial_depth_min_km,
                    initial_depth_max_km=initial_depth_max_km,
                    expand_xy_steps=int(adaptive_expand_xy_steps),
                    expand_z_steps=int(adaptive_expand_z_steps),
                    max_radius_factor=float(adaptive_max_radius_factor),
                    max_depth_span_factor=float(adaptive_max_depth_span_factor),
                    min_depth_km=float(min_depth_km),
                    grid_point_budget=adaptive_grid_point_budget,
                    posterior_diagnostics=decision_posterior_diagnostics,
                    boundary_probability_threshold=float(adaptive_boundary_probability_threshold),
                )
                adaptive_history_rows[-1]["next_action"] = "expand" if proposal.get("apply") else proposal.get("reason")
                adaptive_history_rows[-1]["next_proposal_reason"] = proposal.get("reason")
                adaptive_history_rows[-1]["estimated_next_grid_points"] = proposal.get("estimated_grid_points")
                if not proposal.get("apply"):
                    break

                grid, data, cova, solution, stage_reused = _run_adaptive_axitra_stage(
                    inputs=inputs,
                    radius_km=proposal["grid_radius_km"],
                    depth_min_km=proposal["grid_min_depth_km"],
                    depth_max_km=proposal["grid_max_depth_km"],
                    step_x_km=proposal["step_x_km"],
                    step_z_km=proposal["step_z_km"],
                    max_grid_points=max(int(max_grid_points), int(proposal["max_grid_points_required"])),
                    time_unc_s=float(time_unc_s),
                    rupture_velocity_m_s=float(rupture_velocity_m_s),
                    velocity_slowest_m_s=float(velocity_slowest_m_s),
                    freqmin=float(freqmin),
                    freqmax=float(freqmax),
                    threads=int(threads),
                    progress=bool(progress),
                    invert_displacement=bool(invert_displacement),
                    use_precalculated_Green=use_precalculated_Green,
                    use_noise=use_noise,
                    crosscovariance=bool(crosscovariance),
                    deviatoric=bool(deviatoric),
                    normalized_gf_options=normalized_gf_options,
                    processing_depth_max_km=acquisition_grid_max_depth_km,
                    save_non_inverted_covariance=bool(
                        html_output or (plot and plot_preset == "full")
                    ),
                    store_station_normal_equations=bool(jackknife_config["enabled"]),
                )
                expansion_count += 1
                adaptive_stage_diag = adaptive_stage_posterior(solution)
                adaptive_history_rows.append(
                    _adaptive_stage_record(
                        len(adaptive_history_rows), "expansion", grid, solution, proposal,
                        posterior_diagnostics=adaptive_stage_diag,
                    )
                )
                gf_info["reused"] = bool(stage_reused)
                continue

            if refinement_count >= int(adaptive_max_refinements):
                adaptive_history_rows[-1]["next_action"] = "accept"
                break

            proposal = compute_grid_refinement(
                grid,
                solution.centroid,
                initial_step_x_km=initial_step_x_km,
                initial_step_z_km=initial_step_z_km,
                refinement_level=refinement_count,
                max_refinement_levels=int(adaptive_max_refinements),
                refine_factor=float(adaptive_refine_factor),
                min_step_fraction=float(adaptive_min_step_fraction),
                depth_window_parent_steps=adaptive_depth_window_parent_steps,
                grid_point_budget=adaptive_grid_point_budget,
                posterior_diagnostics=decision_posterior_diagnostics,
                boundary_probability_threshold=float(adaptive_boundary_probability_threshold),
            )
            adaptive_history_rows[-1]["next_action"] = "refine" if proposal.get("apply") else proposal.get("reason")
            adaptive_history_rows[-1]["next_proposal_reason"] = proposal.get("reason")
            adaptive_history_rows[-1]["estimated_next_grid_points"] = proposal.get("estimated_grid_points")
            if not proposal.get("apply"):
                break

            grid, data, cova, solution, stage_reused = _run_adaptive_axitra_stage(
                inputs=inputs,
                radius_km=proposal["grid_radius_km"],
                depth_min_km=proposal["grid_min_depth_km"],
                depth_max_km=proposal["grid_max_depth_km"],
                step_x_km=proposal["step_x_km"],
                step_z_km=proposal["step_z_km"],
                max_grid_points=max(int(max_grid_points), int(proposal["max_grid_points_required"])),
                time_unc_s=float(time_unc_s),
                rupture_velocity_m_s=float(rupture_velocity_m_s),
                velocity_slowest_m_s=float(velocity_slowest_m_s),
                freqmin=float(freqmin),
                freqmax=float(freqmax),
                threads=int(threads),
                progress=bool(progress),
                invert_displacement=bool(invert_displacement),
                use_precalculated_Green=use_precalculated_Green,
                use_noise=use_noise,
                crosscovariance=bool(crosscovariance),
                deviatoric=bool(deviatoric),
                normalized_gf_options=normalized_gf_options,
                processing_depth_max_km=acquisition_grid_max_depth_km,
                save_non_inverted_covariance=bool(
                    html_output or (plot and plot_preset == "full")
                ),
                store_station_normal_equations=bool(jackknife_config["enabled"]),
            )
            refinement_count += 1
            adaptive_stage_diag = adaptive_stage_posterior(solution)
            adaptive_history_rows.append(
                _adaptive_stage_record(
                    len(adaptive_history_rows), "refinement", grid, solution, proposal,
                    posterior_diagnostics=adaptive_stage_diag,
                )
            )
            gf_info["reused"] = bool(stage_reused)

    adaptive_history = pd.DataFrame(adaptive_history_rows)
    gf_info["adaptive_grid"] = bool(adaptive_grid)
    gf_info["adaptive_stage_count"] = int(len(adaptive_history))
    gf_info["adaptive_expansions"] = int(expansion_count)
    gf_info["adaptive_refinements"] = int(refinement_count)
    gf_info["adaptive_max_total_reruns"] = int(adaptive_max_total_reruns)
    gf_info["adaptive_boundary_probability_threshold"] = float(adaptive_boundary_probability_threshold)
    gf_info["adaptive_expand_on_posterior_boundary"] = bool(adaptive_expand_on_posterior_boundary)
    gf_info["adaptive_grid_search"] = dict(adaptive_config)

    grid_edge_report = diagnose_grid_edge(grid, centroid=solution.centroid)
    results = _build_results(
        solution,
        n_uncertainty=n_uncertainty,
        uncertainty_scale=uncertainty_scale,
        uncertainty_scale_floor=uncertainty_scale_floor,
        uncertainty_random_state=uncertainty_random_state,
        grid_edge_report=grid_edge_report,
    )
    if bool(jackknife_config["enabled"]):
        results["station_jackknife"] = compute_station_jackknife(
            solution,
            data,
            cova,
            jackknife_min_stations=int(jackknife_config["jackknife_min_stations"]),
            threads=int(threads),
        )
        # The cached station normal equations are an internal acceleration aid,
        # not part of the public solution/grid API. Release them once consumed.
        _clear_station_normal_equation_cache(solution)
    else:
        results["station_jackknife"] = None
    result_paths = _write_result_tables(
        results, results_path, save_posterior_cells=bool(save_posterior_cells)
    )
    result_paths["grid_edge_diagnostic"] = results_path / "grid_edge_diagnostic.csv"
    results["grid_edge_report"].to_csv(result_paths["grid_edge_diagnostic"], index=False)
    result_paths["adaptive_grid_history"] = results_path / "adaptive_grid_history.csv"
    adaptive_history.to_csv(result_paths["adaptive_grid_history"], index=False)

    # Build the public run mapping before workflow-level diagnostics/reporting.
    # Diagnostic functions consume exactly this structure, so notebook calls and
    # automated plot-preset generation exercise the same API.
    run = {
        "inputs": inputs, "grid": grid, "data": data, "cova": cova,
        "solution": solution, "gf": gf_info,
        "event_df": event_df, "station_df": loaded_stations,
        "download_log": download_log, "load_log": load_log,
        "max_radius_km": resolved_max_radius_km, "depth_spec": depth_spec,
        "acquisition_window": acquisition_window,
        "waveform_window": waveform_window,
        "grid_edge_report": grid_edge_report,
        "adaptive_history": adaptive_history,
        "adaptive_grid": bool(adaptive_grid),
        "adaptive_grid_search": dict(adaptive_config),
        "drop_stations": None if drop_stations is None else ([drop_stations] if isinstance(drop_stations, str) else list(drop_stations)),
        "azimuth_control": dict(azimuth_config),
        "station_jackknife": dict(jackknife_config),
        "station_selection": station_selection,
        "station_selection_path": station_selection_path,
        "crust_file": case_crust_file if gf_source == "axitra" else None,
        "crust_profile": crust_profile, "crust_layers": crust_layers,
        "gf_model_manifest": axitra_model_manifest,
        "results": results, "result_paths": result_paths,
        "figure_paths": [], "plot": None,
        "native_html_path": None, "report_path": None,
    }

    # Workflow-level diagnostic figures. These are separate from PLOT_PRESETS
    # because the latter is passed directly to BayesISOLA.plot and must retain
    # its native keyword contract.
    diagnostic_names = _DIAGNOSTIC_PLOT_PRESETS[plot_preset]
    if plot and diagnostic_names:
        diagnostic_paths = _write_diagnostic_preset(
            run, plot_preset, figure_path,
            tensor_mode="deviatoric" if deviatoric else "full",
            dpi=200,
        )
        custom_figures = [*diagnostic_paths, *custom_figures]

    # Native plotting and native HTML are intentionally separate concerns.
    # ``summary`` is workflow-facing only; ``html_output`` always receives the
    # complete historical BayesISOLA figure suite but those HTML-only files stay
    # out of ``figure_paths``/``show``.
    plot_object, native_figures, native_html_path = _render_native_outputs(
        solution,
        output_path,
        event_id=event_id,
        plot=bool(plot),
        plot_preset=plot_preset,
        html_output=html_output,
        use_noise=use_noise,
        detect_mouse=bool(detect_mouse),
    )

    run["plot"] = plot_object
    run["native_html_path"] = native_html_path
    figure_paths = list(dict.fromkeys([*custom_figures, *native_figures]))
    run["figure_paths"] = figure_paths

    # The curated report is generated only after the run mapping is complete so
    # standalone notebook calls and automatic generation are identical.
    if write_report:
        run["report_path"] = write_html_report(
            run,
            output_file=output_path / "report.html",
            embed_images=False,
            reuse_existing_figures=bool(plot and diagnostic_names),
            dpi=200,
        )

    if show and figure_paths:
        _display_saved_figures(figure_paths)

    return run

