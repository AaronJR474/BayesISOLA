from pathlib import Path


def test_fortran_sources_are_build_only():
    root = Path(__file__).resolve().parents[1]
    source_dir = root / "fortran" / "axitra"

    assert (source_dir / "gr_xyz.for").is_file()
    assert (source_dir / "elemse.for").is_file()
    assert not (root / "green").exists()


def test_examples_use_script_relative_paths():
    root = Path(__file__).resolve().parents[1]
    for name in ("example_2_SAC.py", "example_2_fdsnws.py"):
        text = (root / "examples" / name).read_text()
        assert "Path(__file__).resolve().parent" in text
        assert "'input/" not in text
        assert '"input/' not in text

    sac = (root / "examples" / "example_2_SAC.py").read_text()
    assert 'inputs.load_files(str(INPUT / "sac")' in sac
    assert 'pz_dir=str(INPUT / "pzfiles")' in sac
