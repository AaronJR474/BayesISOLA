import builtins
import importlib.util
import io
from pathlib import Path
import sys
import types

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "BayesISOLA" / "_html.py"


def _load_html_module():
    package = types.ModuleType("BayesISOLA")
    package.__path__ = [str(SOURCE.parent)]

    mt_comps = types.ModuleType("BayesISOLA.MT_comps")
    mt_comps.a2mt = lambda *args, **kwargs: [1, 1, 1, 0, 0, 0]

    paths = types.ModuleType("BayesISOLA._paths")
    paths.copy_html_resources = lambda destination: Path(destination)

    modules = {
        "BayesISOLA": package,
        "BayesISOLA.MT_comps": mt_comps,
        "BayesISOLA._paths": paths,
    }
    previous = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    try:
        spec = importlib.util.spec_from_file_location("BayesISOLA._html", SOURCE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, old in previous.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


def test_imgpath_preserves_explicit_and_relative_resource_paths(tmp_path):
    module = _load_html_module()
    html = tmp_path / "event" / "index.html"

    assert module.imgpath("figures/custom.png", "ignored.png", html) == "figures/custom.png"
    assert module.imgpath("", "html/style.css", html) == "html/style.css"
    assert module.imgpath(None, "html/css/lightbox.min.css", html) == "html/css/lightbox.min.css"


def test_imgpath_auto_uses_recorded_plot_path(tmp_path):
    module = _load_html_module()
    html = tmp_path / "event" / "index.html"
    plot = tmp_path / "event" / "centroid.png"

    assert module.imgpath("auto", plot, html) == "centroid.png"


def test_imgpath_absolute_cross_drive_fallback_uses_file_uri(tmp_path, monkeypatch):
    module = _load_html_module()
    html = tmp_path / "event" / "index.html"
    plot = (tmp_path / "other" / "centroid.png").resolve()

    def cross_drive(*args, **kwargs):
        raise ValueError("path is on mount 'C:', start on mount 'D:'")

    monkeypatch.setattr(module.os.path, "relpath", cross_drive)
    result = module.imgpath("auto", plot, html)

    assert result.startswith("file:")
    assert result.endswith("centroid.png")


def test_html_log_opens_output_as_utf8(tmp_path, monkeypatch):
    module = _load_html_module()
    output = tmp_path / "index.html"
    calls = []

    def fake_open(*args, **kwargs):
        calls.append((args, kwargs))
        return io.StringIO()

    monkeypatch.setattr(builtins, "open", fake_open)

    dummy = types.SimpleNamespace(outdir=str(tmp_path))
    with pytest.raises(AttributeError):
        module.html_log(dummy, outfile=str(output), h1="BayesISOLA CMT — Vackář")

    assert calls
    _, kwargs = calls[0]
    assert kwargs["encoding"] == "utf-8"
    assert kwargs["newline"] == "\n"
