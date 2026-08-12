import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("bayesisola_paths_test", ROOT / "BayesISOLA" / "_paths.py")
PATHS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PATHS)


def test_green_workspace_is_event_specific(tmp_path):
    outdir = tmp_path / "event"
    workspace = PATHS.prepare_green_workspace(PATHS.default_green_dir(outdir))

    assert workspace == outdir / "green"
    assert (workspace / "soutype.dat").read_bytes() == PATHS.DEFAULT_SOUTYPE_FILE.read_bytes()


def test_green_workspace_preserves_existing_soutype(tmp_path):
    workspace = tmp_path / "green"
    workspace.mkdir()
    soutype = workspace / "soutype.dat"
    soutype.write_text("custom\n")

    PATHS.prepare_green_workspace(workspace)

    assert soutype.read_text() == "custom\n"


def test_html_resources_copy_with_expected_layout(tmp_path):
    target = PATHS.copy_html_resources(tmp_path / "html")

    assert (target / "style.css").is_file()
    assert (target / "css" / "lightbox.min.css").is_file()
    assert (target / "js" / "lightbox-plus-jquery.min.js").is_file()
    assert (target / "images" / "loading.gif").is_file()
    assert PATHS.HTML_RESOURCE_DIR.is_dir()
