#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""Internal filesystem locations used by BayesISOLA."""

from pathlib import Path
import shutil

PACKAGE_DIR = Path(__file__).resolve().parent
RESOURCE_DIR = PACKAGE_DIR / "resources"
AXITRA_RESOURCE_DIR = RESOURCE_DIR / "axitra"
HTML_RESOURCE_DIR = RESOURCE_DIR / "html"
AXITRA_BIN_DIR = PACKAGE_DIR / "_bin"
DEFAULT_SOUTYPE_FILE = AXITRA_RESOURCE_DIR / "soutype.dat"


def default_green_dir(outdir):
    """Return the default event-specific Axitra workspace."""
    return Path(outdir).expanduser() / "green"


def prepare_green_workspace(green_dir):
    """Create an Axitra workspace and install its default source-time file."""
    workspace = Path(green_dir).expanduser()
    workspace.mkdir(parents=True, exist_ok=True)
    soutype = workspace / "soutype.dat"
    if not soutype.exists():
        shutil.copyfile(DEFAULT_SOUTYPE_FILE, soutype)
    return workspace


def green_path(green_dir, *parts):
    """Return a path inside an explicit Axitra working directory."""
    return Path(green_dir).expanduser().joinpath(*parts)


def copy_html_resources(destination):
    """Copy packaged HTML assets into a result directory."""
    target = Path(destination).expanduser()
    shutil.copytree(HTML_RESOURCE_DIR, target, dirs_exist_ok=True)
    return target


def axitra_executable(name):
    """Return the installed Axitra executable path."""
    name = str(name).strip()
    if not name:
        raise ValueError("Axitra executable name cannot be empty.")
    candidates = (AXITRA_BIN_DIR / name, AXITRA_BIN_DIR / f"{name}.exe")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Axitra executable {name!r} was not found under {AXITRA_BIN_DIR}. "
        "Install BayesISOLA through pip so the bundled Fortran sources are compiled."
    )
