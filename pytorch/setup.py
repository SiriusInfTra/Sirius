import pathlib
import setuptools
from setuptools import Extension, setup, find_packages
from Cython.Build import cythonize, build_ext

import torch
import shutil

def copy_lib():
    pass

def config_extension():
    torch_install_path = pathlib.Path(torch.__file__).parent
    torch_include_path = str(torch_install_path / "include")

    cmake_cache_path = pathlib.Path('../build/CMakeCache.txt')
    assert cmake_cache_path.exists()

    cuda_root_path = None
    for line in cmake_cache_path.read_text().splitlines():
        if line.startswith('CUDA_TOOLKIT_ROOT_DIR'):
            cuda_root_path = line.split('=')[1]
            break

    ext = Extension(
        name="torch_col._C",
        sources=[
            "torch_col/memory.pyx",
        ],
        language="c++",
        include_dirs=[
            "../",
            "torch_col/csrc",
            torch_include_path,
            f"{cuda_root_path}/include"
        ],
        extra_compile_args=["-std=c++17"],
    )
    return cythonize([ext]) 

setup(
    name="torch_col",
    packages=find_packages(),
    package_data={
        "torch_col" : ['lib/*.so']
    },
    ext_modules=config_extension(),
)