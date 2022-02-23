import os.path as osp
from setuptools import setup, find_packages

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
    url='https://gitlab.devtools.intel.com/ava/mpa',
    packages=find_packages(include=('mpa', 'recipes', 'samples', 'tools')),
    description='Model Preperation Algorithms',
    long_description=long_description,
    install_requires=get_requirements()
)
