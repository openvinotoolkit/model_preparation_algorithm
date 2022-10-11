# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import sys
import warnings
import os
from typing import List, Optional, Union

from pkg_resources import Requirement, parse_requirements
from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

CUDA_VERSION = None
TORCH_VERSION = None

def get_requirements(filename='requirements.txt'):
    here = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(here, filename), 'r') as f:
        requires = [line.replace('\n', '') for line in f.readlines()]
    return requires


def get_mpa_version():
    version_file = 'mpa/version.py'
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']


def get_cuda_version() -> Optional[str]:
    """Get CUDA version.

    Output of nvcc --version command would be as follows:

    b'nvcc: NVIDIA (R) Cuda compiler driver\n
    Copyright (c) 2005-2021 NVIDIA Corporation\n
    Built on Sun_Aug_15_21:14:11_PDT_2021\n
    Cuda compilation tools, release 11.4, V11.4.120\n
    Build cuda_11.4.r11.4/compiler.30300941_0\n'

    It would be possible to get the version by getting
    the version number after "release" word in the string.
    """
    cuda_version: Optional[str] = None
    cuda_path = "/usr/local/cuda"
    if "CUDA_HOME" in os.environ:
        cuda_path = os.environ["CUDA_HOME"]
    version_file = f"{cuda_path}/version.txt"
    if os.path.isfile(version_file):
        f = open(version_file, 'r')
        f = f.readline()
        cuda_version = f.strip().split()[-1]
    else:
        warnings.warn(f"CUDA is not installed on this {sys.platform} machine.")

    return cuda_version


def update_version_with_cuda_suffix(name: str, version: str) -> str:
    """Update the requirement version with the correct CUDA suffix.

    This function checks whether CUDA is installed on the system. If yes, it
    adds finds the torch cuda suffix to properly install the right version of
    torch or torchvision.

    Args:
        name (str): Name of the requirement. (torch or torchvision.)
        version (str): Version of the requirement.

    Examples:
        Below examples demonstrate how the function append the cuda version.
        Note that the appended cuda version might be different on your system.

        >>> update_version_with_cuda_suffix("torch", "1.8.1")
        '1.8.1+cu111'

        >>> update_version_with_cuda_suffix("torch", "1.12.1")
        '1.12.1+cu113'

    Returns:
        str: Updated version with the correct cuda suffix.
    """

    # version[cuda]: suffix.
    # For example torch 1.8.0 Cuda 10., suffix would be 102.
    supported_torch_cuda_versions = {
        "torch": {
            "1.8.2": {"10": "102", "11": "111"},
            "1.9.0": {"10": "102", "11": "111"},
            "1.9.1": {"10": "102", "11": "111"},
        },
        "torchvision": {
            "0.9.2": {"10": "102", "11": "111"},
            "0.10.0": {"10": "102", "11": "111"},
            "0.10.1": {"10": "102", "11": "111"},
        },
    }

    suffix: str = ""
    if sys.platform in ["linux", "win32"]:
        # ``get_cuda version()`` returns the exact version such as 11.2. Here
        # we only need the major CUDA version such as 10 or 11, not the minor
        # version. That's why we use [:2] to get the major version.
        cuda = get_cuda_version()
        if cuda is not None and cuda[:2] in ("10", "11"):
            cuda_version = supported_torch_cuda_versions[name][version][cuda[:2]]
            suffix = f"+cu{cuda_version}"

    return f"{version}{suffix}"

def update_requirement(requirement, cuda_version=None):
    """Update torch requirement with the corrected cuda suffix.

    Args:
        requirement (Requirement): Requirement object comprising requirement
            details.

    Examples:
        >>> from pkg_resources import Requirement
        >>> req = "torch>=1.8.1, <=1.9.1"
        >>> requirement = Requirement.parse(req)
        >>> requirement.name
        'torch'
        >>> requirement.specs
        [('>=', '1.8.1'), ('<=', '1.9.1')]
        >>> update_torch_requirement(requirement)
        'torch<=1.9.1+cu111, >=1.8.1+cu111'

        >>> from pkg_resources import Requirement
        >>> req = "torch>=1.8.1"
        >>> requirement = Requirement.parse(req)
        >>> requirement.name
        'torch'
        >>> requirement.specs
        [('>=', '1.8.1')]
        >>> update_torch_requirement(requirement)
        'torch>=1.8.1+cu111'

    Raises:
        ValueError: When the requirement has more than two version criterion.

    Returns:
        str: Updated torch package with the right cuda suffix.

    """
    name = requirement.name

    for i, (operator, version) in enumerate(requirement.specs):
        if name in ("torch", "torchvision"):
            updated_version = update_version_with_cuda_suffix(name, version)
            requirement.specs[i] = (operator, updated_version)

    # ``specs`` contains operators and versions as follows:
    # [('<=', '1.9.1+cu111'), ('>=', '1.8.1+cu111')]
    # These are to be concatenated again for the updated version.
    specs = [spec[0] + spec[1] for spec in requirement.specs]
    updated_requirement: str

    if specs:
        # This is the case when specs are e.g. ['<=1.9.1+cu111']
        if len(specs) == 1:
            updated_requirement = name + specs[0]
        # This is the case when specs are e.g., ['<=1.9.1+cu111', '>=1.8.1+cu111']
        elif len(specs) == 2:
            updated_requirement = name + specs[0] + ", " + specs[1]
        else:
            raise ValueError(
                f"Requirement version can be a single value or a range. \n"
                f"For example it could be torch>=1.8.1 or torch>=1.8.1, <=1.9.1\n"
                f"Got {specs} instead."
            )

    return updated_requirement


if __name__ == '__main__':
    setup(
        name='mpa',
        version=get_mpa_version(),
        url='https://github.com/openvinotoolkit/model_preparation_algorithm',
        # packages=find_packages(include=('mpa', 'recipes', 'samples')),
        packages=find_packages(exclude=('tests')),
        description='Model Preperation Algorithms',
        long_description=long_description,
        install_requires=get_requirements(),
    )
