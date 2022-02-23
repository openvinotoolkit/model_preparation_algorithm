import glob
import os
from setuptools import find_packages, setup
import shutil
import subprocess
from torch.utils.cpp_extension import CppExtension, BuildExtension
cmd_class = {'build_ext': BuildExtension}

def get_extensions():
    extensions = []

    ext_name = 'mmseg._mpl'

    # prevent ninja from using too many resources
    os.environ.setdefault('MAX_JOBS', '4')
    extra_compile_args = {'cxx': []}

    print(f'Compiling {ext_name} without CUDA')
    op_files = glob.glob('./mpa/ops/csrc/*.cpp')
    include_path = os.path.abspath('./mpa/ops/csrc')
    ext_ops = CppExtension(
        name=ext_name,
        sources=op_files,
        include_dirs=[include_path],
        define_macros=[],
        extra_compile_args=extra_compile_args)
    extensions.append(ext_ops)

    return extensions

def find(name, path):
    for root, dirs, files in os.walk(path):
        if len(dirs) > 0:
            for dir in dirs:
                find(name,os.path.join(root, dir))
        if len(files) > 0:
            for file_name in files:
                if file_name.startswith(name):
                    return root, file_name

if __name__ == '__main__':
    setup(
        name='_mpl',
        license='Apache License 2.0',
        ext_modules=get_extensions(),
        cmdclass=cmd_class,
        zip_safe=False)
    file_path = './build'
    target_path = './mpa/modules/utils/seg_sampler/sampler/'
    source_path, file_name = find('_mpl', file_path)
    print("_mpl builded @ {} and move to {}".format(source_path, target_path))
    shutil.copy(os.path.join(source_path, file_name), os.path.join(target_path, file_name))
    subprocess.run(["rm", "-r", "./build"])
    print("remove temporal folder @ {}".format(file_path))
