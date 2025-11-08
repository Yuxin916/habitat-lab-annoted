#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from setuptools import find_packages, setup

# Directory containing this setup.py
_HERE = Path(__file__).resolve().parent

# Safely resolve README.md next to this setup.py.
# If it's missing (e.g. in build dir), just use an empty description.
try:
    _readme_path = _HERE / "README.md"
    if _readme_path.is_file():
        LONG_DESCRIPTION = _readme_path.read_text(encoding="utf8")
    else:
        LONG_DESCRIPTION = ""
except Exception:
    LONG_DESCRIPTION = ""

# Safely resolve requirements.txt next to this setup.py.
# If it's missing, fall back to an empty list.
try:
    _req_path = _HERE / "requirements.txt"
    if _req_path.is_file():
        INSTALL_REQUIRES = _req_path.read_text(encoding="utf8").strip().splitlines()
    else:
        INSTALL_REQUIRES = []
except Exception:
    INSTALL_REQUIRES = []


def get_package_version():
    import os.path as osp
    import sys

    sys.path.insert(0, osp.join(osp.dirname(__file__), "habitat_hitl"))
    from version import VERSION

    return VERSION


if __name__ == "__main__":
    setup(
        name="habitat-hitl",
        install_requires=INSTALL_REQUIRES,
        packages=find_packages(),
        version=get_package_version(),
        include_package_data=True,
        description=(
            "Habitat-HITL: bring real human users into Habitat virtual "
            "environments to collect interaction data"
        ),
        long_description=LONG_DESCRIPTION,
        long_description_content_type="text/markdown",
        author="Meta AI Research",
        license="MIT License",
        url="https://aihabitat.org",
        project_urls={
            "GitHub repo": "https://github.com/facebookresearch/habitat-lab/",
            "Bug Tracker": "https://github.com/facebookresearch/habitat-lab/issues",
        },
        classifiers=[
            "Intended Audience :: Science/Research",
            "Development Status :: 3 - Alpha",
            "License :: OSI Approved :: MIT License",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Intended Audience :: Developers",
            "Intended Audience :: Education",
            "Intended Audience :: Science/Research",
            "Operating System :: MacOS",
            "Operating System :: Unix",
        ],
        # no entry points
    )
