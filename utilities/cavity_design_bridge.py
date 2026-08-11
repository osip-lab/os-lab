"""Reaching the external cavity-design project from os-lab.

The simulations that turn a measurement into a numerical aperture (mode spacing -> NA for the
PicoScope traces, camera spot size -> NA for the camera videos) live in the cavity-design project,
which is not a package this repo installs - it is a folder whose path sits in local_config.py. Every
analysis script that needs one goes through here, so that path juggling is written once.

Deliberately dependency-free (sys / pathlib / importlib only): mode_analysis.py imports it and must
stay importable without matplotlib, OpenCV or a clipboard.
"""

import importlib
import sys
from pathlib import Path


def add_cavity_design_to_path():
    """Put the repo root and the cavity-design project on sys.path; return the latter.

    The repo root goes on first because local_config.py - which is where the cavity-design path is
    written - lives there, and a script started from a subfolder does not have it on the path.
    """
    repo_root = str(Path(__file__).resolve().parents[1])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from local_config import PATH_CAVITY_DESIGN_PROJECT
    if PATH_CAVITY_DESIGN_PROJECT not in sys.path:
        sys.path.append(PATH_CAVITY_DESIGN_PROJECT)
    return PATH_CAVITY_DESIGN_PROJECT


def import_cavity_design_module(module_name):
    """Import a module of the cavity-design project, e.g. 'simple_analysis_scripts.mode_spacing_to_NA'.

    Raises (ImportError, and whatever the module raises while being imported) when the project is
    not where local_config.py says it is - the callers catch that and report the NA as unavailable
    rather than losing the measurement itself.
    """
    add_cavity_design_to_path()
    return importlib.import_module(module_name)
