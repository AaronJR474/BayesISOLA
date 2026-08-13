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
