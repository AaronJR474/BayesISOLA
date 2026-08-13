from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "examples" / "input" / "example_2_SAC"
REFERENCE = Path(__file__).resolve().parent / "data" / "axitra_smoke_reference.npz"


def _run_axitra_case(output_root):
    import BayesISOLA
    from BayesISOLA.fileformats import read_elemse

    inputs = BayesISOLA.load_data(outdir=Path(output_root) / "output")
    inputs.read_event_info(str(INPUT / "event.isl"))
    inputs.set_source_time_function("triangle", 2.0)
    inputs.read_network_coordinates(str(INPUT / "network.stn"))
    inputs.read_crust(str(INPUT / "crustal.dat"))
    inputs.load_files(
        str(INPUT / "sac"),
        separator="",
        pz_dir=str(INPUT / "pzfiles"),
        pz_separator="",
        pz_suffix=".pz",
    )
    inputs.detect_mouse(figures=False)

    grid = BayesISOLA.grid(
        inputs,
        location_unc=1000,
        depth_unc=3000,
        time_unc=1,
        step_x=200,
        step_z=200,
        max_points=1,
        circle_shape=True,
        rupture_velocity=1000,
    )

    data = BayesISOLA.process_data(
        inputs,
        grid,
        threads=1,
        use_precalculated_Green=False,
        fmax=0.15,
        fmin=0.02,
        progress=False,
    )

    assert len(grid.grid) == 1
    point = grid.grid[0]
    elemse_file = Path(inputs.green_dir) / "elemse0000.dat"

    if not elemse_file.is_file():
        log_file = Path(inputs.outdir) / "log_green.txt"
        log_text = (
            log_file.read_text(encoding="utf-8", errors="replace")
            if log_file.is_file()
            else "<log_green.txt was not created>"
        )
        pytest.fail(
            f"Axitra did not create {elemse_file}.\n"
            f"Axitra log:\n{log_text}"
        )

    elemse = read_elemse(
        inputs.nr,
        data.npts_elemse,
        str(elemse_file),
        inputs.stations,
        data.invert_displacement,
    )
    values = np.concatenate([
        np.asarray(trace.data, dtype=np.float64)
        for station in elemse
        for source in station
        for trace in source
    ])

    return values, point, data


def test_axitra_numerical_regression(tmp_path):
    pytest.importorskip("obspy")
    from BayesISOLA._paths import axitra_executable

    try:
        axitra_executable("gr_xyz")
        axitra_executable("elemse")
    except FileNotFoundError:
        pytest.skip("Axitra executables are not installed in this source environment.")

    reference = np.load(REFERENCE)
    expected = reference["values"]
    actual, point, data = _run_axitra_case(tmp_path)

    assert actual.shape == expected.shape == (10368,)
    assert np.isfinite(actual).all()
    assert np.max(np.abs(actual)) > 0.0

    assert str(reference["point_id"]) == "0000"
    assert float(point["x"]) == float(reference["x"])
    assert float(point["y"]) == float(reference["y"])
    assert float(point["z"]) == float(reference["z"])
    assert int(data.npts_elemse) == int(reference["npts_elemse"])
    assert float(data.samprate) == float(reference["samprate"])

    difference = actual - expected
    peak = max(np.max(np.abs(expected)), np.max(np.abs(actual)))
    signal_rms = np.sqrt(np.mean(expected**2))

    max_scaled_error = np.max(np.abs(difference)) / peak
    normalized_rmse = np.sqrt(np.mean(difference**2)) / signal_rms

    assert max_scaled_error < 1e-6
    assert normalized_rmse < 1e-6
