#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""Internal filesystem locations used by BayesISOLA."""

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
GREEN_DIR = PROJECT_ROOT / "green"


def green_path(*parts):
    """Return a path inside BayesISOLA's Axitra working directory."""
    return GREEN_DIR.joinpath(*parts)
