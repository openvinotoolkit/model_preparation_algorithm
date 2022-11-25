# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os
from glob import glob
from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()


def get_requirements(filename="requirements.txt"):
    here = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(here, filename), "r") as f:
        requires = [line.replace("\n", "") for line in f.readlines()]
    return requires


def get_mpa_version():
    version_file = "mpa/version.py"
    with open(version_file, "r") as f:
        exec(compile(f.read(), version_file, "exec"))
    return locals()["__version__"]


if __name__ == "__main__":
    setup(
        name="otxmpa",
        # version=get_mpa_version(),
        version="0.3.0",
        url="https://github.com/openvinotoolkit/model_preparation_algorithm",
        packages=find_packages(include=('mpa', 'mpa.*', 'recipes.*')),
        include_package_data=True,
        description="Model Preperation Algorithms",
        # long_description=long_description,
        long_description="Temporary PyPI packaging. Project to be merged into OpenVINO Training Extension",
        install_requires=get_requirements(),
    )
