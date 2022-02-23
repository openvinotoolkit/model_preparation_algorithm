import importlib
import subprocess


def load_ext(name, funcs):
    try:
        ext = importlib.import_module('mpa.modules.utils.seg_sampler.sampler.' + name)
    except ModuleNotFoundError:
        subprocess.run(["python", "./mpa/ops/csrc/builder.py", "build"])
        ext = importlib.import_module('mpa.modules.utils.seg_sampler.sampler.' + name)
        print("we re-build a new _mpl file!")

    for fun in funcs:
        assert hasattr(ext, fun), f'{fun} miss in module {name}'

    return ext
