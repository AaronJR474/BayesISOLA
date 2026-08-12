"""EarthScope Syngine Green's-function backend for BayesISOLA.

Version 0.2.0 replaces the legacy six ordinary-seismogram requests with the
Syngine/Instaseis Green's-function endpoint.  The service's ten SeisComP Green's
functions are expanded to Cartesian north-east-down tensor Green's functions,
combined into BayesISOLA's six elementary moment-tensor bases, rotated from ZRT
to ZNE, and written in the legacy ``GFs1`` ... ``GFs6`` directory layout used by
``BayesISOLA.fileformats.read_elemse_from_files``.

The six BayesISOLA source tensors are defined by ``MT_comps.a2mt``.  In USE
ordering ``[Mrr, Mtt, Mpp, Mrt, Mrp, Mtp]`` they are::

    [0, 0, 0, 0, 0, -1]
    [0, 0, 0, 1, 0,  0]
    [0, 0, 0, 0, 1,  0]
    [1,-1, 0, 0, 0,  0]
    [1, 0,-1, 0, 0,  0]
    [1, 1, 1, 0, 0,  0]

The final line corrects the legacy BayesISOLA Syngine ``source6`` definition,
which incorrectly contained an additional ``Mrp=1`` term.

Unlike the MTtime tensor writer, no 1e15 amplitude factor is applied here.
BayesISOLA's response-corrected observations are in SI velocity units and the
Syngine Green's-function endpoint returns SI response to unit moment-tensor
components in N m, so the fitted BayesISOLA coefficients retain N m units.

The current backend is intended for BayesISOLA's default ``step in displacement``
source-time-function setting with velocity elementary seismograms. In Axitra this
combination multiplies ``1/(i*omega)`` by the single velocity derivative
``i*omega``, leaving a unit transfer function; the Syngine GF endpoint is therefore
the matching impulse-response branch. Other BayesISOLA source-time functions need
an explicit convolution step and are intentionally not implemented here yet.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue
from typing import Mapping, Sequence

import hashlib
import io
import json
import os
import zipfile

import numpy as np
import requests
from obspy import Stream, UTCDateTime, read
from obspy.geodetics import gps2dist_azimuth, locations2degrees

try:
    from tqdm.auto import tqdm
except ImportError:  # progress is optional
    tqdm = None


__version__ = "0.2.0"

DEFAULT_SYNGINE_URL = "https://service.earthscope.org/irisws/syngine/1/query"

_SYNGINE_BASES = (
    "TSS", "ZSS", "RSS", "TDS", "ZDS", "RDS", "ZDD", "RDD", "ZEP", "REP"
)
_TENSOR_BASES = tuple(
    f"{component}{basis}"
    for component in ("Z", "R", "T")
    for basis in ("xx", "yy", "zz", "xy", "xz", "yz")
)

# BayesISOLA elementary-source tensors in ObsPy/Syngine USE ordering:
# [Mrr, Mtt, Mpp, Mrt, Mrp, Mtp].
_BAYESISOLA_SOURCES_USE = (
    (0.0, 0.0, 0.0, 0.0, 0.0, -1.0),
    (0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    (1.0, -1.0, 0.0, 0.0, 0.0, 0.0),
    (1.0, 0.0, -1.0, 0.0, 0.0, 0.0),
    (1.0, 1.0, 1.0, 0.0, 0.0, 0.0),
)

_CACHE_SCHEMA = "bayesisola-syngine-gf-v2-geocentric-southward-seiscomp-six-zne"


def _wgs84_geocentric_latitude(latitude_deg: float) -> float:
    """Convert WGS84 geographic latitude to geocentric latitude in degrees."""
    latitude_deg = float(latitude_deg)
    if not np.isfinite(latitude_deg) or not -90.0 <= latitude_deg <= 90.0:
        raise ValueError("latitude_deg must be finite and lie within [-90, 90].")

    a = 6378137.0
    b = 6356752.314245
    latitude = np.deg2rad(latitude_deg)
    geocentric = np.arctan2(
        (b * b) * np.sin(latitude),
        (a * a) * np.cos(latitude),
    )
    return float(np.rad2deg(geocentric))


def _syngine_distance_degrees(
    source_lat: float,
    source_lon: float,
    station_lat: float,
    station_lon: float,
) -> float:
    """Return Syngine spherical distance after WGS84 geocentric conversion."""
    source_lat_gc = _wgs84_geocentric_latitude(source_lat)
    station_lat_gc = _wgs84_geocentric_latitude(station_lat)
    return float(
        locations2degrees(
            source_lat_gc,
            float(source_lon),
            station_lat_gc,
            float(station_lon),
        )
    )


def _request_syngine_gf_stream(
    *,
    session: requests.Session,
    url: str,
    model: str,
    source_depth_m: int,
    distance_deg: float,
    origin: UTCDateTime,
    request_end_s: float,
    syngine_dt: float | None,
    kernelwidth: int,
    timeout: float,
) -> tuple[Stream, dict[str, float | int]]:
    """Request one ten-component SeisComP Green's-function set."""
    params = {
        "model": str(model),
        "greensfunction": 1,
        "sourcedistanceindegrees": float(distance_deg),
        "sourcedepthinmeters": int(source_depth_m),
        "format": "saczip",
        "units": "velocity",
        "origintime": str(origin),
        "starttime": 0.0,
        "endtime": float(request_end_s),
        "nodata": 404,
    }
    if syngine_dt is not None:
        params["dt"] = float(syngine_dt)
        params["kernelwidth"] = int(kernelwidth)

    response = session.get(str(url), params=params, timeout=float(timeout))
    if response.status_code != 200:
        detail = response.text.strip()
        if len(detail) > 2000:
            detail = detail[:2000] + "..."
        raise RuntimeError(
            f"Syngine returned HTTP {response.status_code} for model {model}, "
            f"depth={source_depth_m} m, distance={distance_deg:.6f} deg. {detail}"
        )

    try:
        archive = zipfile.ZipFile(io.BytesIO(response.content))
    except Exception as exc:
        raise ValueError("Syngine response is not a readable SACZIP archive.") from exc

    stream = Stream()
    try:
        for name in archive.namelist():
            if not name.lower().endswith(".sac"):
                continue
            with archive.open(name) as file:
                loaded = read(io.BytesIO(file.read()), format="SAC")
            if len(loaded) != 1:
                raise ValueError(
                    f"Syngine archive member {name!r} contains {len(loaded)} traces."
                )
            stream += loaded
    finally:
        archive.close()

    channels = [str(trace.stats.channel).upper() for trace in stream]
    expected = set(_SYNGINE_BASES)
    if len(stream) != len(_SYNGINE_BASES) or set(channels) != expected:
        missing = sorted(expected.difference(channels))
        extra = sorted(set(channels).difference(expected))
        raise ValueError(
            "Syngine GF archive does not contain the expected ten SeisComP bases. "
            f"Missing={missing}, extra={extra}, channels={channels}."
        )
    if len(set(channels)) != len(channels):
        raise ValueError(f"Syngine GF archive contains duplicate channels: {channels}.")

    by_channel = {str(trace.stats.channel).upper(): trace for trace in stream}
    ordered = Stream([by_channel[basis].copy() for basis in _SYNGINE_BASES])

    dt_values = np.asarray([float(trace.stats.delta) for trace in ordered], dtype=float)
    npts_values = np.asarray([int(trace.stats.npts) for trace in ordered], dtype=int)
    start_offsets = np.asarray(
        [float(trace.stats.starttime - origin) for trace in ordered], dtype=float
    )
    if not np.isfinite(dt_values).all() or (dt_values <= 0).any():
        raise ValueError("Syngine returned an invalid sample interval.")
    if not np.allclose(dt_values, dt_values[0], atol=1e-8, rtol=1e-6):
        raise ValueError(f"Syngine bases have inconsistent dt values: {dt_values.tolist()}.")
    if len(set(npts_values.tolist())) != 1:
        raise ValueError(f"Syngine bases have inconsistent npts: {npts_values.tolist()}.")

    tolerance = max(1e-6, 0.01 * float(dt_values[0]))
    if not np.allclose(start_offsets, start_offsets[0], atol=tolerance, rtol=0.0):
        raise ValueError(
            f"Syngine bases have inconsistent start offsets: {start_offsets.tolist()}."
        )
    for trace in ordered:
        if not np.isfinite(np.asarray(trace.data, dtype=float)).all():
            raise ValueError(
                f"Syngine basis {trace.stats.channel} contains non-finite samples."
            )

    returned_dt = float(dt_values[0])
    returned_npts = int(npts_values[0])
    start_offset = float(start_offsets[0])
    end_offset = start_offset + (returned_npts - 1) * returned_dt
    return ordered, {
        "returned_dt_s": returned_dt,
        "returned_npts": returned_npts,
        "start_offset_s": start_offset,
        "end_offset_s": float(end_offset),
    }


def _seiscomp_to_tensor_stream(stream: Stream, azimuth_deg: float) -> Stream:
    """Expand Syngine's ten SeisComP bases to 18 NED Cartesian tensor bases."""
    by_channel = {
        str(trace.stats.channel).upper(): trace.copy()
        for trace in stream
    }
    missing = set(_SYNGINE_BASES).difference(by_channel)
    if missing:
        raise ValueError(f"Missing Syngine SeisComP bases: {sorted(missing)}.")

    reference = by_channel[_SYNGINE_BASES[0]]
    ref_dt = float(reference.stats.delta)
    ref_npts = int(reference.stats.npts)
    ref_start = reference.stats.starttime
    tolerance = max(1e-8, 0.01 * ref_dt)
    for basis in _SYNGINE_BASES:
        trace = by_channel[basis]
        if int(trace.stats.npts) != ref_npts:
            raise ValueError(f"Syngine basis {basis} has inconsistent npts.")
        if not np.isclose(float(trace.stats.delta), ref_dt, atol=1e-8, rtol=1e-6):
            raise ValueError(f"Syngine basis {basis} has inconsistent dt.")
        if abs(trace.stats.starttime - ref_start) > tolerance:
            raise ValueError(f"Syngine basis {basis} has inconsistent start time.")

    data = {
        basis: np.asarray(by_channel[basis].data, dtype=np.float64)
        for basis in _SYNGINE_BASES
    }

    # Validated SeisComP path frame: positive theta is southward.
    alpha = np.deg2rad((180.0 - float(azimuth_deg)) % 360.0)
    c1, s1 = np.cos(alpha), np.sin(alpha)
    c2, s2 = np.cos(2.0 * alpha), np.sin(2.0 * alpha)

    tensor = {
        "Zxx": 0.5 * data["ZSS"] * c2 - data["ZDD"] / 6.0 + data["ZEP"] / 3.0,
        "Zyy": -0.5 * data["ZSS"] * c2 - data["ZDD"] / 6.0 + data["ZEP"] / 3.0,
        "Zzz": data["ZDD"] / 3.0 + data["ZEP"] / 3.0,
        "Zxy": -data["ZSS"] * s2,
        "Zxz": data["ZDS"] * c1,
        "Zyz": -data["ZDS"] * s1,
        "Rxx": 0.5 * data["RSS"] * c2 - data["RDD"] / 6.0 + data["REP"] / 3.0,
        "Ryy": -0.5 * data["RSS"] * c2 - data["RDD"] / 6.0 + data["REP"] / 3.0,
        "Rzz": data["RDD"] / 3.0 + data["REP"] / 3.0,
        "Rxy": -data["RSS"] * s2,
        "Rxz": data["RDS"] * c1,
        "Ryz": -data["RDS"] * s1,
        "Txx": 0.5 * data["TSS"] * s2,
        "Tyy": -0.5 * data["TSS"] * s2,
        "Tzz": np.zeros_like(data["TSS"]),
        "Txy": data["TSS"] * c2,
        "Txz": data["TDS"] * s1,
        "Tyz": data["TDS"] * c1,
    }

    templates = {
        "Z": by_channel["ZSS"],
        "R": by_channel["RSS"],
        "T": by_channel["TSS"],
    }
    output = Stream()
    for basis in _TENSOR_BASES:
        trace = templates[basis[0]].copy()
        trace.data = np.asarray(tensor[basis], dtype=np.float64)
        trace.stats.channel = basis
        output += trace
    return output


def _tensor_to_bayesisola_zrt(tensor_stream: Stream) -> list[Stream]:
    """Combine 18 NED tensor bases into BayesISOLA's six Z/R/T bases."""
    tensor = {
        str(trace.stats.channel): trace.copy()
        for trace in tensor_stream
    }
    missing = set(_TENSOR_BASES).difference(tensor)
    if missing:
        raise ValueError(f"Missing Cartesian tensor bases: {sorted(missing)}.")

    six: list[Stream] = []
    for basis_index in range(6):
        stream = Stream()
        for component in ("Z", "R", "T"):
            Gxx = np.asarray(tensor[f"{component}xx"].data, dtype=np.float64)
            Gyy = np.asarray(tensor[f"{component}yy"].data, dtype=np.float64)
            Gzz = np.asarray(tensor[f"{component}zz"].data, dtype=np.float64)
            Gxy = np.asarray(tensor[f"{component}xy"].data, dtype=np.float64)
            Gxz = np.asarray(tensor[f"{component}xz"].data, dtype=np.float64)
            Gyz = np.asarray(tensor[f"{component}yz"].data, dtype=np.float64)

            arrays = (
                Gxy,
                Gxz,
                -Gyz,
                -Gxx + Gzz,
                -Gyy + Gzz,
                Gxx + Gyy + Gzz,
            )

            trace = tensor[f"{component}xx"].copy()
            trace.data = np.asarray(arrays[basis_index], dtype=np.float64)
            trace.stats.channel = f"MX{component}"
            stream += trace
        six.append(stream)
    return six


def _rotate_six_to_zne(
    six_zrt: Sequence[Stream],
    *,
    network: str,
    station: str,
    back_azimuth_deg: float,
) -> list[Stream]:
    """Rotate each BayesISOLA elementary stream from ZRT to ZNE."""
    output: list[Stream] = []
    for stream in six_zrt:
        zrt = Stream()
        for component in ("Z", "R", "T"):
            selected = stream.select(component=component)
            if len(selected) != 1:
                raise ValueError(
                    f"Expected one {component} trace in elementary ZRT stream; "
                    f"found {len(selected)}."
                )
            trace = selected[0].copy()
            trace.stats.network = str(network)
            trace.stats.station = str(station)
            trace.stats.location = "SE"
            trace.stats.channel = f"MX{component}"
            zrt += trace

        zrt.rotate(method="RT->NE", back_azimuth=float(back_azimuth_deg))
        zne = Stream()
        for component in ("Z", "N", "E"):
            selected = zrt.select(component=component)
            if len(selected) != 1:
                raise ValueError(
                    f"Expected one {component} trace after RT->NE rotation; "
                    f"found {len(selected)}."
                )
            trace = selected[0].copy()
            trace.stats.network = str(network)
            trace.stats.station = str(station)
            trace.stats.location = "SE"
            trace.stats.channel = f"MX{component}"
            zne += trace
        output.append(zne)
    return output


def _output_paths(root: Path, network: str, station: str) -> list[Path]:
    paths: list[Path] = []
    for index in range(1, 7):
        directory = root / f"GFs{index}"
        for component in ("Z", "N", "E"):
            paths.append(directory / f"{network}.{station}.SE.MX{component}")
    return paths


def _write_six_zne(root: Path, network: str, station: str, six_zne: Sequence[Stream]) -> None:
    """Write the legacy BayesISOLA Syngine file layout expected by fileformats.py."""
    if len(six_zne) != 6:
        raise ValueError(f"Expected six elementary streams; received {len(six_zne)}.")

    for index, stream in enumerate(six_zne, start=1):
        directory = root / f"GFs{index}"
        directory.mkdir(parents=True, exist_ok=True)
        for component in ("Z", "N", "E"):
            selected = stream.select(component=component)
            if len(selected) != 1:
                raise ValueError(
                    f"Expected one {component} trace for GFs{index}; found {len(selected)}."
                )
            path = directory / f"{network}.{station}.SE.MX{component}"
            selected[0].write(str(path), format="MSEED")


def _short_signature(payload: Mapping) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20].upper()


def _manifest_path(root: Path) -> Path:
    return root / "syngine_manifest.json"


def _cache_is_valid(root: Path, signature: str, stations: Sequence[Mapping]) -> bool:
    manifest_path = _manifest_path(root)
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception:
        return False
    if manifest.get("signature") != signature or manifest.get("schema") != _CACHE_SCHEMA:
        return False

    for station in stations:
        paths = _output_paths(root, station["network"], station["station"])
        if not all(path.is_file() for path in paths):
            return False
        try:
            for path in paths:
                stream = read(str(path), headonly=True)
                if len(stream) != 1 or int(stream[0].stats.npts) < 2:
                    return False
        except Exception:
            return False
    return True


def _station_contexts(
    bulk: Sequence[Mapping],
    *,
    source_lat: float,
    source_lon: float,
) -> list[dict]:
    contexts = []
    for receiver in bulk:
        network = str(receiver.get("networkcode", "SY"))
        station = str(receiver.get("stationcode", "STA"))
        station_lat = float(receiver["latitude"])
        station_lon = float(receiver["longitude"])

        distance_m, azimuth_deg, back_azimuth_deg = gps2dist_azimuth(
            float(source_lat), float(source_lon), station_lat, station_lon
        )
        distance_deg = _syngine_distance_degrees(
            float(source_lat), float(source_lon), station_lat, station_lon
        )
        contexts.append({
            "network": network,
            "station": station,
            "requested_location": str(receiver.get("locationcode", "")),
            "station_lat": station_lat,
            "station_lon": station_lon,
            "distance_km": float(distance_m) / 1000.0,
            "distance_deg": float(distance_deg),
            "azimuth_deg": float(azimuth_deg) % 360.0,
            "back_azimuth_deg": float(back_azimuth_deg) % 360.0,
        })
    return contexts


class generate_query:
    """Generate BayesISOLA elementary seismograms from EarthScope Syngine.

    ``bulk`` remains public for compatibility with the original BayesISOLA API.
    Existing callers may append receiver dictionaries containing ``networkcode``,
    ``stationcode``, ``locationcode``, ``latitude`` and ``longitude`` and then call
    :meth:`do_query_simple`.
    """

    def __init__(
        self,
        *,
        url: str = DEFAULT_SYNGINE_URL,
        syngine_dt: float | None = None,
        kernelwidth: int = 12,
        timeout: float = 120.0,
        request_padding_s: float = 60.0,
        max_workers: int = 4,
        progress: bool = True,
        overwrite: bool = False,
        legacy_coverage_factor: float = 2.0,
    ):
        self.bulk: list[dict] = []
        self.sources = [list(source) for source in _BAYESISOLA_SOURCES_USE]
        self.dir_names = [f"GFs{i}" for i in range(1, 7)]

        self.url = str(url)
        self.syngine_dt = None if syngine_dt is None else float(syngine_dt)
        self.kernelwidth = int(kernelwidth)
        self.timeout = float(timeout)
        self.request_padding_s = float(request_padding_s)
        self.max_workers = int(max_workers)
        self.progress = bool(progress)
        self.overwrite = bool(overwrite)
        self.legacy_coverage_factor = float(legacy_coverage_factor)

        if self.syngine_dt is not None and self.syngine_dt <= 0:
            raise ValueError("syngine_dt must be None or positive.")
        if self.kernelwidth < 1:
            raise ValueError("kernelwidth must be positive.")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive.")
        if self.request_padding_s < 0:
            raise ValueError("request_padding_s cannot be negative.")
        if self.max_workers < 1:
            raise ValueError("max_workers must be positive.")
        if self.legacy_coverage_factor < 1.0:
            raise ValueError("legacy_coverage_factor must be >= 1.")

    def do_query_simple(
        self,
        model,
        sourcelatitude: float,
        sourcelongitude: float,
        sourcedepthinmeters: float,
        origintime: UTCDateTime,
        starttime: UTCDateTime,
        endtime: UTCDateTime,
        output_root_path,
        *,
        target_npts: int | None = None,
        target_sampling_rate: float | None = None,
        overwrite: bool | None = None,
        progress: bool | None = None,
        max_workers: int | None = None,
        source_time_function: str = "step",
    ) -> dict:
        """Download one source-point Green's-function set for every receiver.

        ``target_npts`` and ``target_sampling_rate`` should be supplied by the new
        automated BayesISOLA workflow. They determine the exact minimum raw GF
        duration required by ``read_elemse_from_files``. The legacy BayesISOLA
        caller does not provide them, so a conservative two-times inversion-window
        coverage is used as a compatibility fallback.
        """
        if not self.bulk:
            raise ValueError("generate_query.bulk is empty; add at least one receiver.")

        model = str(model).strip()
        if not model:
            raise ValueError("model cannot be empty.")

        stf = str(source_time_function).strip().lower()
        if stf not in {"step", "heaviside", "step in displacement"}:
            raise NotImplementedError(
                "The Syngine GF backend currently supports BayesISOLA's step-in-"
                "displacement source-time function only. Other source-time functions "
                "require an explicit convolution step before inversion."
            )

        source_lat = float(sourcelatitude)
        source_lon = float(sourcelongitude)
        source_depth_m = int(round(float(sourcedepthinmeters)))
        if not np.isfinite([source_lat, source_lon, source_depth_m]).all():
            raise ValueError("Source coordinates/depth must be finite.")
        if not -90.0 <= source_lat <= 90.0 or not -180.0 <= source_lon <= 180.0:
            raise ValueError("Source latitude/longitude are outside valid bounds.")
        if source_depth_m < 0:
            raise ValueError("sourcedepthinmeters cannot be negative.")

        origin = UTCDateTime(origintime)
        requested_start = float(UTCDateTime(starttime) - origin)
        requested_end = float(UTCDateTime(endtime) - origin)
        if requested_end <= 0:
            raise ValueError("endtime must be later than origintime.")
        if requested_start > 1e-6:
            raise ValueError(
                "BayesISOLA Syngine Green's functions must include source-origin "
                "time zero; starttime cannot be later than origintime."
            )

        if (target_npts is None) != (target_sampling_rate is None):
            raise ValueError(
                "target_npts and target_sampling_rate must be supplied together."
            )
        if target_npts is not None:
            target_npts = int(target_npts)
            target_sampling_rate = float(target_sampling_rate)
            if target_npts < 2 or target_sampling_rate <= 0:
                raise ValueError(
                    "target_npts must be >= 2 and target_sampling_rate must be positive."
                )
            target_end = (target_npts - 1) / target_sampling_rate
        else:
            # Native set_time_window guarantees npts_elemse < 2*npts_slice.
            # This keeps the old _green.py caller usable until gf_source forwards
            # the exact npts/sampling rate explicitly.
            target_end = self.legacy_coverage_factor * requested_end

        request_end_s = max(requested_end, target_end) + self.request_padding_s
        contexts = _station_contexts(
            self.bulk,
            source_lat=source_lat,
            source_lon=source_lon,
        )

        output_root = Path(output_root_path).expanduser()
        output_root.mkdir(parents=True, exist_ok=True)

        payload = {
            "schema": _CACHE_SCHEMA,
            "version": __version__,
            "model": model,
            "url": self.url,
            "source_lat": source_lat,
            "source_lon": source_lon,
            "source_depth_m": source_depth_m,
            "origin": str(origin),
            "requested_end_s": requested_end,
            "request_end_s": request_end_s,
            "syngine_dt": self.syngine_dt,
            "kernelwidth": self.kernelwidth,
            "units": "velocity",
            "source_time_function": "step in displacement",
            "tensor_coordinates": "NED_x_north_y_east_z_down",
            "tensor_rotation": "alpha_deg=(180-azimuth_deg)%360",
            "receiver_rotation": "ObsPy RT->NE using source-to-station back azimuth",
            "bayesisola_sources_USE": [list(source) for source in _BAYESISOLA_SOURCES_USE],
            "stations": contexts,
        }
        signature = _short_signature(payload)
        force = self.overwrite if overwrite is None else bool(overwrite)

        if not force and _cache_is_valid(output_root, signature, contexts):
            return {
                **payload,
                "signature": signature,
                "status": "existing",
                "output_root": str(output_root),
            }

        show_progress = self.progress if progress is None else bool(progress)
        worker_count = self.max_workers if max_workers is None else int(max_workers)
        worker_count = max(1, min(worker_count, len(contexts)))

        session_pool = Queue()
        sessions = [requests.Session() for _ in range(worker_count)]
        for session in sessions:
            session_pool.put(session)

        def fetch(context):
            session = session_pool.get()
            try:
                raw, info = _request_syngine_gf_stream(
                    session=session,
                    url=self.url,
                    model=model,
                    source_depth_m=source_depth_m,
                    distance_deg=context["distance_deg"],
                    origin=origin,
                    request_end_s=request_end_s,
                    syngine_dt=self.syngine_dt,
                    kernelwidth=self.kernelwidth,
                    timeout=self.timeout,
                )
                return context, raw, info
            finally:
                session_pool.put(session)

        bar = None
        if tqdm is not None:
            bar = tqdm(
                total=len(contexts),
                desc="Syngine Green's functions",
                unit="sta",
                disable=not show_progress,
            )

        station_results = []
        try:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = {executor.submit(fetch, context): context for context in contexts}
                for future in as_completed(futures):
                    context, raw, info = future.result()
                    tensor = _seiscomp_to_tensor_stream(
                        raw,
                        azimuth_deg=context["azimuth_deg"],
                    )

                    # Deliberately no MTtime 1e15 amplitude scaling here.
                    six_zrt = _tensor_to_bayesisola_zrt(tensor)
                    six_zne = _rotate_six_to_zne(
                        six_zrt,
                        network=context["network"],
                        station=context["station"],
                        back_azimuth_deg=context["back_azimuth_deg"],
                    )
                    _write_six_zne(
                        output_root,
                        context["network"],
                        context["station"],
                        six_zne,
                    )
                    station_results.append({**context, **info})
                    if bar is not None:
                        bar.update(1)
        finally:
            if bar is not None:
                bar.close()
            for session in sessions:
                session.close()

        payload["stations"] = sorted(
            station_results,
            key=lambda item: (item["network"], item["station"]),
        )
        manifest = {
            **payload,
            "signature": signature,
            "status": "written",
            "output_root": str(output_root),
        }
        manifest_path = _manifest_path(output_root)
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True))
        os.replace(temporary, manifest_path)
        return manifest

    def do_query_grid(
        self,
        model,
        sourcelatitude: float,
        sourcelongitude: float,
        sourcedepthinmeters: float,
        origintime: UTCDateTime,
        starttime: UTCDateTime,
        endtime: UTCDateTime,
        output_root_path,
        Lx,
        Ly,
        Lz,
        dx=2,
        dy=2,
        dz=2,
        **kwargs,
    ):
        """Compatibility grid wrapper using the corrected GF-endpoint backend."""
        depths = np.arange(
            int(sourcedepthinmeters - Lz),
            int(sourcedepthinmeters + Lz),
            dz,
        )
        longs = np.arange(
            sourcelongitude - (Lx / 112.0),
            sourcelongitude + (Lx / 112.0),
            dx / 112.0,
        )
        lats = np.arange(
            sourcelatitude - (Ly / 112.0),
            sourcelatitude + (Ly / 112.0),
            dy / 112.0,
        )

        manifests = []
        for depth in depths:
            for latitude in lats:
                for longitude in longs:
                    path = Path(output_root_path) / (
                        f"{depth:.1f}_{latitude:.4f}_{longitude:.4f}"
                    )
                    manifests.append(
                        self.do_query_simple(
                            model,
                            latitude,
                            longitude,
                            depth,
                            origintime,
                            starttime,
                            endtime,
                            path,
                            **kwargs,
                        )
                    )
        return manifests
