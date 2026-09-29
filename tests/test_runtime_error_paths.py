import importlib.util
from pathlib import Path
import sys
import types

import pytest


ROOT = Path(__file__).resolve().parents[1]
PROCESS_SOURCE = ROOT / "BayesISOLA" / "_process_data.py"
MT_SOURCE = ROOT / "BayesISOLA" / "MT_comps.py"


def _load_process_module():
    package = types.ModuleType("BayesISOLA")
    package.__path__ = [str(PROCESS_SOURCE.parent)]
    helpers = types.ModuleType("BayesISOLA.helpers")
    helpers.my_filter = lambda *args, **kwargs: None
    helpers.prefilter_data = lambda *args, **kwargs: None
    helpers.next_power_of_2 = lambda value: value

    modules = {"BayesISOLA": package, "BayesISOLA.helpers": helpers}
    previous = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    try:
        spec = importlib.util.spec_from_file_location("BayesISOLA._process_data", PROCESS_SOURCE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, old in previous.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


def test_missing_instrument_response_raises_runtime_error():
    module = _load_process_module()

    class Stats:
        response = None
        paz = None

    class Trace:
        stats = Stats()

    class Stream(list):
        def detrend(self, **kwargs):
            return None

    dummy = types.SimpleNamespace(
        d=types.SimpleNamespace(data_raw=[Stream([Trace()])]),
        log=lambda *args, **kwargs: None,
    )

    with pytest.raises(RuntimeError, match="No response in tr.stats"):
        module.correct_data(dummy)


def test_pure_isotropic_decomposition_uses_nan_for_undefined_nodal_planes():
    spec = importlib.util.spec_from_file_location("bayesisola_mt_comps_test", MT_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    result = module.decompose([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])

    for key in ("s1", "d1", "r1", "s2", "d2", "r2"):
        assert result[key] != result[key]  # NaN: undefined, but safely float-formattable
