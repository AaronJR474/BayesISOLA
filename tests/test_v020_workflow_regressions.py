"""Focused regressions for the public/scientific additions in BayesISOLA 0.2."""

from inspect import signature
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from BayesISOLA.MT_comps import a2mt, decompose
from BayesISOLA.plot import plot as NativePlot
from BayesISOLA import workflows as wf
from BayesISOLA._diagnostics import plot_station_fit_summary


class _FakeTime:
    def __init__(self, value=0.0):
        self.value = float(value)

    def __add__(self, value):
        return _FakeTime(self.value + float(value))

    @property
    def datetime(self):
        return f"T{self.value:.2f}"


def _synthetic_solution(deviatoric=False):
    shifts = np.array([-0.5, 0.0, 0.5])
    grid = []
    raw_weights = []
    for grid_index, (north, east, depth) in enumerate(
        [(0.0, 0.0, 9000.0), (2000.0, 0.0, 10000.0), (0.0, 2000.0, 11000.0)]
    ):
        shift_results = {}
        for shift_index, _ in enumerate(shifts):
            misfit = 2.0 + 1.3 * grid_index + 0.7 * (shift_index - 1) ** 2
            log_det = np.log(1.0 + 0.4 * grid_index + 0.2 * shift_index)
            weight = np.exp(0.5 * log_det - 0.5 * (misfit - 2.0))
            raw_weights.append(weight)
            a = np.array([1.2, -0.8, 0.3, 0.2, -0.1, 0.4], dtype=float)[:, None] * 1e15
            if deviatoric:
                a[5, 0] = 0.0
            n_parameters = 5 if deviatoric else 6
            shift_results[shift_index] = {
                "misfit": misfit,
                "VR": 1.0 - misfit / 50.0,
                "CN": 2.0 + grid_index,
                "log_det_Ca": log_det,
                "c": weight,
                "a": a,
                "GtGinv": np.eye(n_parameters) * 1e27,
            }
        grid.append(
            {
                "id": str(grid_index).zfill(4),
                "x": north,
                "y": east,
                "z": depth,
                "lat": -43.0 + 0.01 * grid_index,
                "lon": 172.0 + 0.01 * grid_index,
                "err": 0,
                "edge": False,
                "shifts": shift_results,
            }
        )

    grid_obj = SimpleNamespace(
        grid=grid,
        radius=2000.0,
        step_x=2000.0,
        step_z=1000.0,
        circle_shape=True,
    )
    data = SimpleNamespace(shifts=shifts, components=5, npts_slice=20)
    solution = SimpleNamespace(
        grid=grid,
        g=grid_obj,
        d=data,
        deviatoric=deviatoric,
        sum_c=float(sum(raw_weights)),
        event={"t": _FakeTime()},
    )
    centroid = dict(grid[0])
    centroid.update(grid[0]["shifts"][1])
    centroid.update({"shift_idx": 1, "shift": 0.0})
    solution.centroid = centroid
    solution.mt_decomp = decompose(a2mt(centroid["a"]))
    return solution


def test_exact_posterior_matches_native_weights_at_unit_scale():
    solution = _synthetic_solution(False)
    posterior = wf.build_posterior_cells(solution, variance_scale=1.0)

    np.testing.assert_allclose(
        posterior["posterior_probability"],
        posterior["native_probability"],
        rtol=1e-13,
        atol=1e-15,
    )
    assert np.isclose(posterior["posterior_probability"].sum(), 1.0)
    diagnostics = wf.compute_posterior_diagnostics(solution, posterior)
    assert np.isclose(diagnostics["posterior_probability_sum"], 1.0)
    assert diagnostics["posterior_effective_cells"] >= 1.0


def test_uncertainty_uses_exact_n_reproducible_draws_and_deviatoric_iso_zero():
    solution = _synthetic_solution(False)
    posterior = wf.build_posterior_cells(solution)
    first, first_diag = wf.extract_uncertainty_df(
        solution,
        n=125,
        posterior_cells=posterior,
        random_state=91,
    )
    second, second_diag = wf.extract_uncertainty_df(
        solution,
        n=125,
        posterior_cells=posterior,
        random_state=91,
    )
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 125
    assert first_diag["n_sampled"] == second_diag["n_sampled"] == 125

    deviatoric = _synthetic_solution(True)
    dev_posterior = wf.build_posterior_cells(deviatoric)
    dev_samples, _ = wf.extract_uncertainty_df(
        deviatoric,
        n=40,
        posterior_cells=dev_posterior,
        random_state=3,
    )
    np.testing.assert_allclose(dev_samples["iso_percent"], 0.0, atol=1e-8)


def test_nodal_plane_alignment_resolves_only_pair_label_switching():
    preferred = {
        "s1": 20.0,
        "d1": 30.0,
        "r1": 40.0,
        "s2": 210.0,
        "d2": 60.0,
        "r2": -120.0,
    }
    swapped = {
        "s1": 211.0,
        "d1": 59.0,
        "r1": -119.0,
        "s2": 19.0,
        "d2": 31.0,
        "r2": 41.0,
        "Mw": 5.1,
    }
    aligned = wf._align_nodal_planes(swapped, preferred)
    assert aligned["Mw"] == swapped["Mw"]
    assert abs(wf._circular_angle_difference_deg(aligned["s1"], preferred["s1"])) < 2.0
    assert abs(aligned["d1"] - preferred["d1"]) < 2.0
    assert abs(wf._circular_angle_difference_deg(aligned["r1"], preferred["r1"])) < 2.0


def test_native_html_preset_matches_native_plot_constructor_defaults():
    parameters = list(signature(NativePlot.__init__).parameters.values())[2:]
    defaults = {parameter.name: parameter.default for parameter in parameters}
    assert wf._NATIVE_HTML_PLOT_PRESET == defaults


def test_station_jackknife_no_longer_exposes_progress_parameter():
    assert "progress" not in signature(wf.compute_station_jackknife).parameters


def test_station_fit_plot_keeps_negative_variance_reduction_visible():
    station_fit = pd.DataFrame(
        [
            {
                "network": "NZ",
                "station": "AAA",
                "location": "",
                "component": "Z",
                "distance_km": 20.0,
                "used": True,
                "variance_reduction": -0.35,
            },
            {
                "network": "NZ",
                "station": "AAA",
                "location": "",
                "component": "N",
                "distance_km": 20.0,
                "used": True,
                "variance_reduction": 0.65,
            },
        ]
    )
    run = {"results": {"station_fit": station_fit, "station_jackknife": None}}
    figure = plot_station_fit_summary(run)
    try:
        xmin, xmax = figure.axes[0].get_xlim()
        assert xmin < -35.0
        assert xmax >= 100.0
    finally:
        plt.close(figure)
