# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os
from glob import glob
from setuptools import setup, find_packages


try:
    from torch.utils.cpp_extension import CppExtension, BuildExtension
    cmd_class = {'build_ext': BuildExtension}
except ModuleNotFoundError:
    cmd_class = {}
    print('Skip building ext ops due to the absence of torch.')

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


def get_extensions():
    extensions = []

    ext_name = 'mpa.modules._mpl'

    # prevent ninja from using too many resources
    os.environ.setdefault('MAX_JOBS', '4')
    extra_compile_args = {'cxx': []}

    print(f'Compiling {ext_name} without CUDA')
    op_files = glob("./mpa/csrc/mpl/*.cpp")
    include_path = os.path.abspath("./mpa/csrc/mpl")
    ext_ops = CppExtension(
        name=ext_name,
        sources=op_files,
        include_dirs=[include_path],
        define_macros=[],
        extra_compile_args=extra_compile_args)
    extensions.append(ext_ops)

    return extensions


if __name__ == "__main__":
    setup(
        name="mpa",
        version=get_mpa_version(),
        url="https://github.com/openvinotoolkit/model_preparation_algorithm",
        packages=find_packages(include=('mpa', 'mpa.*', 'recipes.*')),
        include_package_data=True,
        description="Model Preperation Algorithms",
        long_description=long_description,
        install_requires=get_requirements(),
        ext_modules=get_extensions(),
        cmdclass=cmd_class,
    )
