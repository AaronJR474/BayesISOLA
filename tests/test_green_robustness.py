import hashlib
import importlib.util
from pathlib import Path
import sys
import types


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "BayesISOLA" / "_green.py"


def _load_green_module():
    package = types.ModuleType("BayesISOLA")
    package.__path__ = [str(SOURCE.parent)]

    axitra = types.ModuleType("BayesISOLA.axitra")
    axitra.Axitra_wrapper = lambda *args, **kwargs: True

    syngine = types.ModuleType("BayesISOLA.syngine")

    paths = types.ModuleType("BayesISOLA._paths")
    paths.green_path = lambda green_dir, *parts: Path(green_dir).joinpath(*parts)
    paths.axitra_executable = lambda name: Path(name)

    modules = {
        "BayesISOLA": package,
        "BayesISOLA.axitra": axitra,
        "BayesISOLA.syngine": syngine,
        "BayesISOLA._paths": paths,
    }
    previous = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    try:
        spec = importlib.util.spec_from_file_location("BayesISOLA._green", SOURCE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, old in previous.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


def _green_case(tmp_path):
    green_dir = tmp_path / "green"
    green_dir.mkdir()
    (green_dir / "crustal.dat").write_text("crust\n", encoding="utf-8")
    (green_dir / "station.dat").write_text("station\n", encoding="utf-8")
    (green_dir / "soutype.dat").write_text("step\n", encoding="utf-8")

    gp = {"x": 0.0, "y": 0.0, "z": 5000.0, "err": 0, "VR": 0.0}
    logs = []
    dummy = types.SimpleNamespace(
        d=types.SimpleNamespace(green_dir=green_dir, outdir=str(tmp_path), models={"": 1}),
        grid=types.SimpleNamespace(grid=[gp]),
        threads=1,
        npts_exp=8,
        elemse_start_origin=0.0,
        progress=False,
        log=logs.append,
    )

    md5_crust = hashlib.md5((green_dir / "crustal.dat").read_bytes()).hexdigest()
    md5_station = hashlib.md5((green_dir / "station.dat").read_bytes()).hexdigest()
    expected = f"0.000 0.000 5.000 {md5_crust} {md5_station} step"
    (green_dir / "elemse0000.txt").write_text(expected, encoding="utf-8")
    return dummy, gp, green_dir, logs


def test_green_cache_requires_nonempty_elemse_payload(tmp_path):
    module = _load_green_module()
    dummy, _, green_dir, _ = _green_case(tmp_path)

    assert module.verify_Greens_headers(dummy) is False

    (green_dir / "elemse0000.dat").write_bytes(b"")
    assert module.verify_Greens_headers(dummy) is False

    (green_dir / "elemse0000.dat").write_bytes(b"payload")
    assert module.verify_Greens_headers(dummy) is True


def test_green_cache_rejects_empty_metadata(tmp_path):
    module = _load_green_module()
    dummy, _, green_dir, _ = _green_case(tmp_path)
    (green_dir / "elemse0000.dat").write_bytes(b"payload")
    (green_dir / "elemse0000.txt").write_text("", encoding="utf-8")

    assert module.verify_Greens_headers(dummy) is False


def test_serial_axitra_failure_marks_grid_point_invalid(tmp_path):
    module = _load_green_module()
    dummy, gp, green_dir, _ = _green_case(tmp_path)
    module.Axitra_wrapper = lambda *args, **kwargs: False
    module.axitra_executable = lambda name: green_dir / name

    module.calculate_Green(dummy)

    assert gp["err"] == 1
    assert gp["VR"] == -10
