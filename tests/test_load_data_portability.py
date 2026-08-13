import importlib.util
from pathlib import Path
import sys
import types


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "BayesISOLA" / "load_data.py"


def _stub_module(name, functions):
    module = types.ModuleType(name)
    for function in functions:
        setattr(module, function, lambda self, *args, **kwargs: None)
    return module


def _load_load_data_module():
    package = types.ModuleType("BayesISOLA")
    package.__path__ = [str(SOURCE.parent)]

    paths = types.ModuleType("BayesISOLA._paths")
    paths.default_green_dir = lambda outdir: Path(outdir) / "green"

    def prepare_green_workspace(path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "soutype.dat").write_text("step\n", encoding="utf-8")
        return path

    paths.prepare_green_workspace = prepare_green_workspace

    modules = {
        "BayesISOLA": package,
        "BayesISOLA._paths": paths,
        "BayesISOLA._input_crust": _stub_module("BayesISOLA._input_crust", ["read_crust"]),
        "BayesISOLA._input_event": _stub_module("BayesISOLA._input_event", ["read_event_info", "set_event_info", "set_source_time_function"]),
        "BayesISOLA._input_network": _stub_module("BayesISOLA._input_network", ["read_network_info_DB", "read_network_coordinates", "create_station_index", "write_stations"]),
        "BayesISOLA._input_seismo_files": _stub_module("BayesISOLA._input_seismo_files", ["add_NEZ", "add_SAC", "add_NIED", "load_files", "load_NIED_files", "check_a_station_present"]),
        "BayesISOLA._input_seismo_remote": _stub_module("BayesISOLA._input_seismo_remote", ["load_streams_ArcLink", "load_streams_fdsnws"]),
        "BayesISOLA._mouse": _stub_module("BayesISOLA._mouse", ["detect_mouse"]),
    }

    previous = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    try:
        spec = importlib.util.spec_from_file_location("BayesISOLA.load_data", SOURCE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, old in previous.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


def test_load_data_creates_nested_output_and_writes_utf8_log(tmp_path):
    module = _load_load_data_module()
    outdir = tmp_path / "nested" / "event" / "output"

    data = module.load_data(outdir=outdir)
    try:
        data.log("Vackář — Gallovič — Burjánek")
    finally:
        data.logfile.close()

    assert outdir.is_dir()
    assert (outdir / "green" / "soutype.dat").is_file()
    text = (outdir / "log.txt").read_text(encoding="utf-8")
    assert "Vackář — Gallovič — Burjánek" in text
