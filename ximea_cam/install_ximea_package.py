"""
Install the `ximea` Python package into the active environment.

The `ximea` package is NOT on PyPI - it ships only inside the XIMEA
Software Package installer (from ximea.com) as a plain source folder
with no setup.py, meant to be dropped into site-packages. This script
wraps it in a minimal setup.py and pip-installs it properly so it shows
up in `pip list`, can be uninstalled, etc.

Usage:
    1. Install the XIMEA Software Package (includes camera drivers + xiAPI).
    2. Run this script with the target environment's Python active:
           python ximea_cam/install_ximea_package.py
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

XIMEA_PYTHON_API = Path(r"C:\XIMEA\API\Python\v3\ximea")

SETUP_PY = '''\
from setuptools import setup, find_packages

setup(
    name="ximea",
    version="0.0.0+local",
    packages=find_packages(),
    include_package_data=True,
    package_data={"ximea": ["libs/32bit/*.dll", "libs/64bit/*.dll"]},
)
'''


def main():
    if not XIMEA_PYTHON_API.is_dir():
        raise SystemExit(
            f"Could not find {XIMEA_PYTHON_API}. "
            "Install the XIMEA Software Package from ximea.com first."
        )

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        shutil.copytree(XIMEA_PYTHON_API, tmp / "ximea")
        (tmp / "setup.py").write_text(SETUP_PY)
        subprocess.check_call([sys.executable, "-m", "pip", "install", str(tmp)])


if __name__ == "__main__":
    main()
