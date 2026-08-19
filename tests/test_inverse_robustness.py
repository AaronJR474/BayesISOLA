import importlib.util
from pathlib import Path
import sys
import types

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "BayesISOLA" / "_inverse.py"


def _load_inverse_module():
    package = types.ModuleType("BayesISOLA")
    package.__path__ = [str(SOURCE.parent)]

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
        spec = importlib.util.spec_from_file_location("BayesISOLA._inverse", SOURCE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, old in previous.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


def test_inversion_reports_no_valid_grid_points_cleanly():
    module = _load_inverse_module()
    grid = [
        {"err": 1},
        {"err": True},
    ]
    dummy = types.SimpleNamespace(
        grid=grid,
        d=types.SimpleNamespace(d_shifts=[]),
        cova=types.SimpleNamespace(Cd_inv=[], Cd_inv_shifts=[]),
    )

    with pytest.raises(RuntimeError, match="No valid grid points remain"):
        module.run_inversion(dummy)

    assert grid[0]["id"] == "0000"
    assert grid[1]["id"] == "0001"


def test_serial_inversion_maps_results_to_grid_in_todo_order(monkeypatch):
    module = _load_inverse_module()

    def fake_invert(point_id, *args, **kwargs):
        value = float(int(point_id) + 1)
        shift = 0
        return {
            "shift": shift,
            "a": [[value]],
            "VR": value / 10.0,
            "misfit": 10.0 - value,
            "CN": 1.0,
            "GtGinv": [[1.0]],
            "det_Ca": 1.0,
            "log_det_Ca": 0.0,
            "shifts": {
                shift: {
                    "a": [[value]],
                    "VR": value / 10.0,
                    "misfit": 10.0 - value,
                    "CN": 1.0,
                    "GtGinv": [[1.0]],
                    "det_Ca": 1.0,
                    "log_det_Ca": 0.0,
                }
            },
        }

    monkeypatch.setattr(module, "invert", fake_invert)
    grid = [
        {"err": 0, "path": None},
        {"err": 1, "path": None},
        {"err": 0, "path": None},
    ]
    d_shift = __import__("numpy").ones((2, 1), dtype=float)
    dummy = types.SimpleNamespace(
        grid=grid,
        threads=1,
        deviatoric=False,
        decompose=False,
        cova=types.SimpleNamespace(
            Cd_inv=[],
            Cd_inv_shifts=[],
            factorized_noise=False,
        ),
        inp=types.SimpleNamespace(
            nr=1,
            stations=[{"useZ": True, "useN": False, "useE": False}],
            event={"t": 0.0},
            green_dir="green",
        ),
        d=types.SimpleNamespace(
            d_shifts=[d_shift],
            components=1,
            npts_elemse=2,
            npts_slice=2,
            elemse_start_origin=0.0,
            samprate=1.0,
            invert_displacement=False,
            shifts=[0.0],
            progress=False,
        ),
    )

    module.run_inversion(dummy)

    assert grid[0]["a"] == [[1.0]]
    assert "a" not in grid[1]
    assert grid[2]["a"] == [[3.0]]
