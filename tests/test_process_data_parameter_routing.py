from pathlib import Path
import importlib.util
import sys
import types


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "BayesISOLA" / "process_data.py"


def _load_process_data(calls):
    package = types.ModuleType("BayesISOLA")
    package.__path__ = [str(SOURCE.parent)]

    green = types.ModuleType("BayesISOLA._green")
    for name in (
        "set_Greens_parameters", "write_Greens_parameters", "verify_Greens_parameters",
        "verify_Greens_headers", "calculate_or_verify_Green", "calculate_Green",
        "use_elemse_from_files", "use_elemse_from_syngine",
    ):
        setattr(green, name, lambda self, *args, _name=name, **kwargs: calls.setdefault(_name, (args, kwargs)))

    parameters = types.ModuleType("BayesISOLA._parameters")

    def set_parameters(self, fmax, fmin=0.0, wavelengths=5, min_depth=1000, log=True):
        calls["set_parameters"] = {
            "fmax": fmax,
            "fmin": fmin,
            "wavelengths": wavelengths,
            "min_depth": min_depth,
            "log": log,
        }

    def skip_short_records(self, noise=False):
        calls["skip_short_records"] = noise

    for name in ("set_frequencies", "set_working_sampling", "count_components", "min_time", "max_time", "set_time_window"):
        setattr(parameters, name, lambda self, *args, _name=name, **kwargs: calls.setdefault(_name, (args, kwargs)))
    parameters.set_parameters = set_parameters
    parameters.skip_short_records = skip_short_records

    processing = types.ModuleType("BayesISOLA._process_data")
    for name in ("correct_data", "trim_filter_data", "decimate_shift"):
        setattr(processing, name, lambda self, *args, _name=name, **kwargs: calls.setdefault(_name, (args, kwargs)))

    modules = {
        "BayesISOLA": package,
        "BayesISOLA._green": green,
        "BayesISOLA._parameters": parameters,
        "BayesISOLA._process_data": processing,
    }
    previous = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)

    try:
        spec = importlib.util.spec_from_file_location("BayesISOLA.process_data", SOURCE)
        module = importlib.util.module_from_spec(spec)
        sys.modules["BayesISOLA.process_data"] = module
        spec.loader.exec_module(module)
        return module.process_data
    finally:
        sys.modules.pop("BayesISOLA.process_data", None)
        for name, old in previous.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


def test_process_data_routes_min_depth_without_overwriting_wavelengths():
    calls = {}
    process_data = _load_process_data(calls)

    class Data:
        logtext = {}

        @staticmethod
        def log(*args, **kwargs):
            pass

    process_data(
        Data(), object(), correct_data=False,
        fmax=0.25, fmin=0.05, min_depth=7000,
        skip_short_records=12.5,
        calculate_or_verify_Green=False,
        trim_filter_data=False, decimate_shift=False,
    )

    assert calls["set_parameters"] == {
        "fmax": 0.25,
        "fmin": 0.05,
        "wavelengths": 5,
        "min_depth": 7000,
        "log": True,
    }
    assert calls["skip_short_records"] == 12.5


def test_process_data_false_skip_short_records_disables_check():
    calls = {}
    process_data = _load_process_data(calls)

    class Data:
        logtext = {}

        @staticmethod
        def log(*args, **kwargs):
            pass

    process_data(
        Data(), object(), correct_data=False, set_parameters=False,
        skip_short_records=False, calculate_or_verify_Green=False,
        trim_filter_data=False, decimate_shift=False,
    )

    assert "skip_short_records" not in calls
