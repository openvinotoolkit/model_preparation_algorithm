# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os
import os.path as osp

import Cython.Compiler.Options
import numpy
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup

Cython.Compiler.Options.annotate = True

cython_aug_root = "mpa/modules/datasets/pipelines/transforms/cython_augments"
cython_aug_modl = cython_aug_root.replace("/", ".")
ext_modules = []

for fname in os.listdir(cython_aug_root):
    name, ext = osp.splitext(fname)
    if ext == ".pyx":
        ext_modules += [
            Extension(f"{cython_aug_modl}.{name}", [osp.join(cython_aug_root, fname)],
                      include_dirs=[numpy.get_include()], extra_compile_args=["-O3"])
        ]
ext_modules = cythonize(ext_modules, annotate=True)


with open("README.md", "r") as fh:
    long_description = fh.read()


def get_requirements(filename='requirements.txt'):
    here = osp.dirname(osp.realpath(__file__))
    with open(osp.join(here, filename), 'r') as f:
        requires = [line.replace('\n', '') for line in f.readlines()]
    return requires


def find_version():
    version_file = 'mpa/version.py'
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']


setup(
    name='mpa',
    version=find_version(),
    url='https://github.com/openvinotoolkit/model_preparation_algorithm',
    packages=find_packages(include=('mpa', 'recipes', 'samples')),
    description='Model Preperation Algorithms',
    long_description=long_description,
    install_requires=get_requirements(),
    ext_modules=ext_modules,
)
