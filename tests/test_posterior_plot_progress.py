import importlib.util
import inspect
from pathlib import Path
import sys
import types

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INVERSE_SOURCE = ROOT / "BayesISOLA" / "_inverse.py"
INVERSE_PROBLEM_SOURCE = ROOT / "BayesISOLA" / "inverse_problem.py"
PLOT_SOURCE = ROOT / "BayesISOLA" / "_plot_solution_maps.py"
WORKFLOWS_SOURCE = ROOT / "BayesISOLA" / "workflows.py"


def _load_inverse_module():
    package = types.ModuleType("BayesISOLA")
    package.__path__ = [str(INVERSE_SOURCE.parent)]
    inverse_problem = types.ModuleType("BayesISOLA.inverse_problem")
    inverse_problem.invert = lambda *args, **kwargs: None
    inverse_problem.whiten_covariance_array = lambda *args, **kwargs: None

    modules = {
        "BayesISOLA": package,
        "BayesISOLA.inverse_problem": inverse_problem,
    }
    previous = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    try:
        spec = importlib.util.spec_from_file_location("BayesISOLA._inverse", INVERSE_SOURCE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, old in previous.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


def _load_inverse_problem_module():
    package = types.ModuleType("BayesISOLA")
    package.__path__ = [str(INVERSE_PROBLEM_SOURCE.parent)]

    obspy = types.ModuleType("obspy")
    obspy.UTCDateTime = lambda value=0: value
    fileformats = types.ModuleType("BayesISOLA.fileformats")
    fileformats.read_elemse = lambda *args, **kwargs: None
    fileformats.read_elemse_from_files = lambda *args, **kwargs: None
    helpers = types.ModuleType("BayesISOLA.helpers")
    helpers.my_filter = lambda *args, **kwargs: None
    mt_comps = types.ModuleType("BayesISOLA.MT_comps")
    mt_comps.decompose = lambda value: {}
    mt_comps.a2mt = lambda value: value
    paths = types.ModuleType("BayesISOLA._paths")
    paths.green_path = lambda *parts: Path(*map(str, parts))

    modules = {
        "BayesISOLA": package,
        "obspy": obspy,
        "BayesISOLA.fileformats": fileformats,
        "BayesISOLA.helpers": helpers,
        "BayesISOLA.MT_comps": mt_comps,
        "BayesISOLA._paths": paths,
    }
    previous = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    try:
        spec = importlib.util.spec_from_file_location("BayesISOLA.inverse_problem", INVERSE_PROBLEM_SOURCE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, old in previous.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


def _load_plot_module():
    package = types.ModuleType("BayesISOLA")
    package.__path__ = [str(PLOT_SOURCE.parent)]
    mt_comps = types.ModuleType("BayesISOLA.MT_comps")
    mt_comps.a2mt = lambda value, system=None: value
    mt_comps.decompose = lambda value: {}

    obspy = types.ModuleType("obspy")
    imaging = types.ModuleType("obspy.imaging")
    beachball = types.ModuleType("obspy.imaging.beachball")
    beachball.beach = lambda *args, **kwargs: None
    imaging.beachball = beachball
    obspy.imaging = imaging

    modules = {
        "BayesISOLA": package,
        "BayesISOLA.MT_comps": mt_comps,
        "obspy": obspy,
        "obspy.imaging": imaging,
        "obspy.imaging.beachball": beachball,
    }
    previous = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    try:
        spec = importlib.util.spec_from_file_location("BayesISOLA._plot_solution_maps", PLOT_SOURCE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, old in previous.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


def test_covariance_determinant_metrics_remain_stable_when_det_overflows():
    module = _load_inverse_problem_module()
    matrix = np.diag([1.0e200, 1.0e200])

    det, log_det = module._covariance_determinant_metrics(matrix)

    assert np.isinf(det)
    assert np.isfinite(log_det)
    assert log_det == np.linalg.slogdet(matrix)[1]


def test_covariance_determinant_metrics_match_direct_det_in_moderate_range():
    module = _load_inverse_problem_module()
    matrix = np.diag([2.0, 3.0, 4.0])

    det, log_det = module._covariance_determinant_metrics(matrix)

    assert det == np.linalg.det(matrix)
    assert np.exp(log_det) == det


def test_posterior_weights_are_finite_under_extreme_dynamic_range():
    module = _load_inverse_module()
    grid = [
        {
            "shift_idx": 0,
            "shifts": {
                0: {"misfit": 1000.0, "log_det_Ca": 3000.0},
                1: {"misfit": 0.0, "log_det_Ca": 0.0},
            },
        },
        {
            "shift_idx": 0,
            "shifts": {
                0: {"misfit": 50.0, "log_det_Ca": 10.0},
            },
        },
    ]
    dummy = types.SimpleNamespace(grid=grid)

    module._assign_posterior_weights(dummy, [0, 1])

    weights = [entry["c"] for gp in grid for entry in gp["shifts"].values()]
    assert np.isfinite(weights).all()
    assert all(0.0 <= value <= 1.0 for value in weights)
    assert max(weights) == 1.0
    assert np.isfinite(dummy.sum_c) and dummy.sum_c > 0.0
    assert np.isfinite(dummy.max_c)
    assert np.isfinite(dummy.max_sum_c)


def test_posterior_normalized_probabilities_match_direct_formula_in_moderate_range():
    module = _load_inverse_module()
    entries = [
        {"misfit": 2.0, "log_det_Ca": np.log(4.0)},
        {"misfit": 3.5, "log_det_Ca": np.log(9.0)},
        {"misfit": 6.0, "log_det_Ca": np.log(2.0)},
    ]
    grid = [{"shift_idx": 0, "shifts": {i: dict(entry) for i, entry in enumerate(entries)}}]
    dummy = types.SimpleNamespace(grid=grid)

    module._assign_posterior_weights(dummy, [0])

    min_misfit = min(entry["misfit"] for entry in entries)
    direct = np.array([
        np.sqrt(np.exp(entry["log_det_Ca"])) * np.exp(-0.5 * (entry["misfit"] - min_misfit))
        for entry in entries
    ])
    direct /= direct.sum()
    stable = np.array([grid[0]["shifts"][i]["c"] for i in range(len(entries))])
    stable /= stable.sum()

    np.testing.assert_allclose(stable, direct, rtol=1e-13, atol=0.0)


def test_beachball_tensor_is_zero_trace_and_scale_normalized():
    module = _load_plot_module()
    mt = np.array([5.3545488233688625e14, 5.637197809267039e14, 5.052969884888224e14,
                   1.1742842085766798e15, 3.1065809330411656e14, 2.2948454242567238e14])

    normalized = module._normalized_zero_trace_mt(mt)

    assert normalized is not None
    assert np.isfinite(normalized).all()
    assert np.max(np.abs(normalized)) == 1.0
    assert abs(np.sum(normalized[:3])) < 1e-14
    assert module._normalized_zero_trace_mt([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]) is None


def test_beachball_failure_falls_back_once_without_aborting_map(tmp_path):
    module = _load_plot_module()
    calls = []

    def failing_beach(mt, **kwargs):
        calls.append(np.asarray(mt, dtype=float))
        raise ValueError("synthetic beach failure")

    module.beach = failing_beach
    dummy = types.SimpleNamespace(
        grid=types.SimpleNamespace(step_x=2000.0, shift_min=-1.0, shift_max=1.0),
        MT=types.SimpleNamespace(decompose=False, centroid={"VR": 0.8}),
    )
    output = tmp_path / "map.png"
    mts = [
        [5.0e14, 6.0e14, 4.0e14, 1.0e15, 3.0e14, 2.0e14],
        [4.0e14, 3.0e14, 5.0e14, -8.0e14, 2.0e14, 1.0e14],
    ]

    module.plot_map_backend(
        dummy,
        x=[-1.0, 1.0], y=[-1.0, 1.0], s=None, CN=None,
        MT=mts, color=["black", "black"], width=[1.0, 1.0],
        highlight=[False, False], xmin=-2.0, xmax=2.0, ymin=-2.0, ymax=2.0,
        beachball_size_c=False, outfile=str(output),
    )
    plt.close("all")

    assert output.is_file()
    assert len(calls) == 2  # one attempt per tensor; the legacy duplicate retry is gone
    assert all(np.max(np.abs(mt)) == 1.0 for mt in calls)


def test_run_auto_cmt_exposes_and_routes_progress():
    spec = importlib.util.spec_from_file_location("bayesisola_workflows_progress_test", WORKFLOWS_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    parameter = inspect.signature(module.run_auto_cmt).parameters["progress"]
    source = inspect.getsource(module.run_auto_cmt)

    assert parameter.default is True
    assert "progress=bool(progress)" in source
