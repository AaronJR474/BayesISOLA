"""Regression tests for the vectorised filter/elemse paths and the Green's function cache."""

import types
from pathlib import Path

import numpy as np
from obspy import Stream, Trace
from obspy.signal.filter import highpass, lowpass

from BayesISOLA import _green
from BayesISOLA.fileformats import read_elemse, read_elemse_array
from BayesISOLA.helpers import filter_array, my_filter
from BayesISOLA.inverse_problem import _design_matrix, _read_filter_trim_elemse


def _write_elemse(path: Path, data: np.ndarray, dt: float) -> None:
    """Write an Axitra-style elemse file: one Fortran record (t, N, E, Z) per sample."""
    nr, _, _, npts = data.shape
    record = np.dtype([("head", "=i4"), ("t", "=f4"), ("N", "=f4"), ("E", "=f4"), ("Z", "=f4"), ("tail", "=i4")])
    rec = np.zeros(nr * 6 * npts, dtype=record)
    rec["head"] = rec["tail"] = 16
    rec["t"] = np.tile(np.arange(npts) * dt, nr * 6)
    for c, comp in enumerate("ZNE"):
        rec[comp] = data[:, :, c, :].reshape(-1)
    rec.tofile(path)


def _stations(nr: int) -> list[dict]:
    """Station metadata with mixed bands and component flags."""
    return [
        {"model": "", "fmin": 0.02 if r % 2 else 0.03, "fmax": 0.06, "useZ": True, "useN": r != 1,
         "useE": True, "weightZ": 1.0, "weightN": 1.5, "weightE": 0.5}
        for r in range(nr)
    ]


def test_filter_array_matches_obspy_chain():
    """filter_array is bit-identical to the ObsPy lowpass + 2x highpass chain."""
    x = np.random.default_rng(0).standard_normal((4, 256))
    expected = np.array([
        highpass(highpass(lowpass(row, 0.06, 0.48, corners=4), 0.02, 0.48, corners=2), 0.02, 0.48, corners=2)
        for row in x
    ])
    assert np.array_equal(filter_array(x, 0.48, 0.02, 0.06), expected)


def test_my_filter_matches_trace_filter():
    """my_filter on a Stream equals ObsPy Trace.filter, including mixed sampling rates."""
    rng = np.random.default_rng(1)
    st = Stream([Trace(rng.standard_normal(200), header={"sampling_rate": sr}) for sr in (1.0, 1.0, 2.0)])
    reference = st.copy()
    for tr in reference:
        tr.filter("lowpass", freq=0.1)
        tr.filter("highpass", freq=0.02, corners=2)
        tr.filter("highpass", freq=0.02, corners=2)
    my_filter(st, 0.02, 0.1)
    assert all(np.array_equal(a.data, b.data) for a, b in zip(st, reference))


def test_elemse_array_matches_streams_and_design_matrix(tmp_path):
    """Array reader, filtering, trimming and G assembly match the Stream-based path."""
    nr, npts, dt = 3, 64, 2.08
    data = np.random.default_rng(2).standard_normal((nr, 6, 3, npts)).astype(np.float32)
    path = tmp_path / "elemse0000.dat"
    _write_elemse(path, data, dt)
    stations = _stations(nr)

    arr, deltas = read_elemse_array(nr, npts, path, stations)
    streams = read_elemse(nr, npts, path, stations)
    for r in range(nr):
        for j in range(6):
            for c in range(3):
                assert np.array_equal(streams[r][j][c].data, arr[r, j, c])
                assert streams[r][j][c].stats.delta == 1.0 / (1.0 / deltas[r, j])

    npts_slice, offset = 40, 10.4
    elemse_data = _read_filter_trim_elemse(nr, 6, npts, npts_slice, offset, stations, path)
    from obspy import UTCDateTime
    expected = []
    for r in range(nr):
        for j in range(6):
            my_filter(streams[r][j], stations[r]["fmin"], stations[r]["fmax"])
            streams[r][j].trim(UTCDateTime(0) + offset)
    for r in range(nr):
        for comp, key in enumerate(("useZ", "useN", "useE")):
            if stations[r][key]:
                weight = stations[r][key.replace("use", "weight")]
                expected.append(np.array([[streams[r][e][comp].data[i] * weight for e in range(6)] for i in range(npts_slice)]))
    G = _design_matrix(elemse_data, stations, npts_slice)
    assert G.flags.c_contiguous
    assert np.array_equal(G, np.concatenate(expected))


def test_green_cache_reuses_points_by_coordinates(tmp_path):
    """Points of a changed grid are restored from the coordinate cache, new points are not."""
    for name, text in (("crustal.dat", "model"), ("station.dat", "stations"), ("soutype.dat", "7\n4.0"), ("grdat.hed", "params")):
        (tmp_path / name).write_text(text, encoding="utf-8")
    logs = []
    self = types.SimpleNamespace(
        d=types.SimpleNamespace(green_dir=str(tmp_path), models={"": 2}),
        use_precalculated_Green="auto", log=logs.append,
    )
    md5c, md5s, sout = _green._green_signature(self)
    old_grid = [{"x": 0.0, "y": 0.0, "z": 5000.0}, {"x": 1000.0, "y": 0.0, "z": 5000.0}]
    for i, gp in enumerate(old_grid):
        (tmp_path / f"elemse{i:04d}.dat").write_bytes(b"payload%d" % i)
        (tmp_path / f"elemse{i:04d}.txt").write_text(
            f"{gp['x'] / 1e3:1.3f} {gp['y'] / 1e3:1.3f} {gp['z'] / 1e3:1.3f} {md5c} {md5s} {sout}", encoding="utf-8")

    assert _green.stash_reusable_Green(self) == 2
    # new grid: old point 1 now at index 0, plus a new point
    self.grid = types.SimpleNamespace(grid=[old_grid[1], {"x": 2000.0, "y": 0.0, "z": 5000.0}])
    todo = _green._restore_cached_Green(self, "")
    assert todo == [1]
    assert (tmp_path / "elemse0000.dat").read_bytes() == b"payload1"

    # a changed station set must not be reused
    assert _green.stash_reusable_Green(self) == 1
    (tmp_path / "station.dat").write_text("other stations", encoding="utf-8")
    assert _green._restore_cached_Green(self, "") == [0, 1]


def test_depth_bundled_axitra_matches_per_point(tmp_path):
    """One Axitra run per depth gives the same elemse files as one run per grid point."""
    import pytest

    import BayesISOLA
    from BayesISOLA._paths import axitra_executable

    try:
        axitra_executable("gr_xyz")
        axitra_executable("elemse")
    except FileNotFoundError:
        pytest.skip("Axitra executables are not installed in this source environment.")

    example = Path(__file__).resolve().parents[1] / "examples" / "input" / "example_2_SAC"
    outputs = {}
    for bundled in (False, True):
        inputs = BayesISOLA.load_data(outdir=tmp_path / f"bundled_{bundled}")
        inputs.read_event_info(str(example / "event.isl"))
        inputs.set_source_time_function("triangle", 2.0)
        inputs.read_network_coordinates(str(example / "network.stn"))
        inputs.read_crust(str(example / "crustal.dat"))
        inputs.load_files(str(example / "sac"), separator="", pz_dir=str(example / "pzfiles"),
                          pz_separator="", pz_suffix=".pz")
        grid = BayesISOLA.grid(inputs, location_unc=1000, depth_unc=1000, time_unc=1, step_x=1000,
                               step_z=1000, max_points=12, circle_shape=True, rupture_velocity=1000)
        BayesISOLA.process_data(inputs, grid, threads=2, use_precalculated_Green=False, fmax=0.15,
                                fmin=0.02, progress=False, gf_bundle_depths=bundled)
        files = sorted(Path(inputs.green_dir).glob("elemse????.dat"))
        outputs[bundled] = {f.name: np.fromfile(f, dtype=np.float32) for f in files}
        assert len({gp["z"] for gp in grid.grid}) > 1
        assert all(not gp.get("err") for gp in grid.grid)

    assert outputs[True].keys() == outputs[False].keys()
    for name, reference in outputs[False].items():
        scale = np.max(np.abs(reference))
        assert np.max(np.abs(outputs[True][name] - reference)) / scale < 1e-5, name


def test_green_store_round_trip_is_byte_identical(tmp_path):
    """Records assembled from the store equal the elemse file they came from."""
    from BayesISOLA.gf_store import GreenStore

    npts, n_recv = 32, 3
    data = np.random.default_rng(3).standard_normal((n_recv, 6, 3, npts)).astype(np.float32)
    path = tmp_path / "elemse_chunk.dat"
    _write_elemse(path, data, 1.5)
    store = GreenStore(tmp_path / "store")
    keys = [f"k{i}" for i in range(n_recv)]
    assert store.add(path, keys, npts)
    store.index()
    assert store.elemse_records(keys).tobytes() == path.read_bytes()
    # a subset in another order is the matching receiver blocks
    reordered = store.elemse_records(["k2", "k0"]).tobytes()
    block = 6 * npts * 24
    raw = path.read_bytes()
    assert reordered == raw[2 * block:3 * block] + raw[0:block]


def test_shared_green_store_skips_axitra_for_known_receivers(tmp_path):
    """A second run sharing the store computes no receivers and writes identical elemse files."""
    import pytest

    import BayesISOLA
    from BayesISOLA._paths import axitra_executable

    try:
        axitra_executable("gr_xyz")
    except FileNotFoundError:
        pytest.skip("Axitra executables are not installed in this source environment.")

    example = Path(__file__).resolve().parents[1] / "examples" / "input" / "example_2_SAC"
    store_dir = tmp_path / "shared_store"
    outputs = []
    for run in range(2):
        inputs = BayesISOLA.load_data(outdir=tmp_path / f"run{run}")
        inputs.read_event_info(str(example / "event.isl"))
        inputs.set_source_time_function("triangle", 2.0)
        inputs.read_network_coordinates(str(example / "network.stn"))
        inputs.read_crust(str(example / "crustal.dat"))
        inputs.load_files(str(example / "sac"), separator="", pz_dir=str(example / "pzfiles"),
                          pz_separator="", pz_suffix=".pz")
        grid = BayesISOLA.grid(inputs, location_unc=1000, depth_unc=1000, time_unc=1, step_x=1000,
                               step_z=1000, max_points=8, circle_shape=True, rupture_velocity=1000)
        BayesISOLA.process_data(inputs, grid, threads=1, use_precalculated_Green=False, fmax=0.15,
                                fmin=0.02, progress=False, gf_store_dir=store_dir)
        outputs.append({f.name: f.read_bytes() for f in sorted(Path(inputs.green_dir).glob("elemse????.dat"))})
        if run == 1:
            # no Axitra run was logged: every receiver came from the store
            assert "receivers" not in (Path(inputs.outdir) / "log_green.txt").read_text()
    assert outputs[0] == outputs[1]


def test_table_rotation_matches_axitra_at_any_azimuth(tmp_path):
    """Elementary seismograms rotated from azimuth 0 equal Axitra run at the true azimuth."""
    import pytest

    import BayesISOLA
    from BayesISOLA._paths import axitra_executable
    from BayesISOLA.axitra import Axitra_depth_chunk
    from BayesISOLA.fileformats import ELEMSE_RECORD
    from BayesISOLA.gf_table import rotate_from_azimuth_zero

    try:
        axitra_executable("gr_xyz")
    except FileNotFoundError:
        pytest.skip("Axitra executables are not installed in this source environment.")

    example = Path(__file__).resolve().parents[1] / "examples" / "input" / "example_2_SAC"
    inputs = BayesISOLA.load_data(outdir=tmp_path / "out")
    inputs.read_event_info(str(example / "event.isl"))
    inputs.set_source_time_function("triangle", 2.0)
    inputs.read_network_coordinates(str(example / "network.stn"))
    inputs.read_crust(str(example / "crustal.dat"))
    inputs.load_files(str(example / "sac"), separator="", pz_dir=str(example / "pzfiles"),
                      pz_separator="", pz_suffix=".pz")
    grid = BayesISOLA.grid(inputs, location_unc=0, depth_unc=0, time_unc=1, step_x=1000, step_z=1000,
                           max_points=1, circle_shape=True, rupture_velocity=1000)
    data = BayesISOLA.process_data(inputs, grid, threads=1, use_precalculated_Green=False, fmax=0.15,
                                   fmin=0.02, progress=False)

    distance = 37.5
    azimuths = np.radians([0.0, 33.0, 90.0, 147.0, 211.0, 300.0])
    receivers = [(distance * np.cos(a), distance * np.sin(a), 0.0) for a in azimuths] + [(distance, 0.0, 0.0)]
    elemse_file, message = Axitra_depth_chunk("rot", "", "8.000", receivers, data.npts_exp,
                                              data.elemse_start_origin, inputs.green_dir)
    assert elemse_file is not None, message
    npts = 2 ** data.npts_exp
    records = np.fromfile(elemse_file, dtype=ELEMSE_RECORD).reshape(len(receivers), 6, npts)
    g = np.stack([records["N"], records["E"], records["Z"]], axis=-1).astype(float)
    rotated = rotate_from_azimuth_zero(np.repeat(g[-1:], len(azimuths), axis=0), azimuths)
    scale = np.max(np.abs(g[:-1]))
    # exact at 0 and 90 degrees; elsewhere ~1e-5 from rounding receiver coordinates to 0.1 m
    assert np.max(np.abs(rotated - g[:-1])) / scale < 5e-5
