# Configuration file for the Sphinx documentation builder.

from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

project = "BayesISOLA"
copyright = "Jiří Vackář and contributors"
author = "Jiří Vackář"

_pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
_match = re.search(r'^version\s*=\s*"([^"]+)"', _pyproject, flags=re.MULTILINE)
if _match is None:
    raise RuntimeError("Could not determine BayesISOLA version from pyproject.toml.")
release = _match.group(1)
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.imgmath",
    "sphinx.ext.todo",
]

templates_path = []
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_theme = "alabaster"
html_static_path = []
