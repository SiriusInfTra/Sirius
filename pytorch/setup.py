import os, pathlib
import setuptools
from setuptools import Extension, setup, find_packages
from Cython.Build import cythonize
from Cython.Distutils import build_ext

import torch
import shutil


def cleanup_previous():
    for file in pathlib.Path("./torch_col/_C").glob("*.so"):
        file.unlink()

cleanup_previous()

# -------------------

def get_build_path():
    return pathlib.Path(os.environ.get('BUILD_DIR', '../build'))


def copy_lib():
    # build_path = pathlib.Path('../build')
    build_path = get_build_path()
    print(f"Copying lib files from {build_path} to torch_col/lib/")
    ext_build_path = build_path / 'pytorch'
    comm_build_path = build_path / 'common'
    lib_path = pathlib.Path('torch_col/lib')
    lib_path.mkdir(parents=True, exist_ok=True)
    shutil.copy(comm_build_path / 'libsta.so', lib_path)
    shutil.copy(ext_build_path / 'libnew_torch_col.so', lib_path)


def config_extension():
    torch_install_path = pathlib.Path(torch.__file__).parent
    torch_include_path = [
        str(torch_install_path / "include"),
        str(torch_install_path / "include" / "torch" / "csrc" / "api" / "include")
    ]

    cmake_cache_path = pathlib.Path(f'{get_build_path()}/CMakeCache.txt')
    assert cmake_cache_path.exists()

    glog_binary_dir = None
    glog_source_dir = None
    glog_include_path = None
    cuda_root_path = None
    boost_root = None
    conda_prefix = None
    for line in cmake_cache_path.read_text().splitlines():
        if line.startswith('CUDA_TOOLKIT_ROOT_DIR'):
            cuda_root_path = line.split('=')[1]
        if line.startswith('GLOG_INCLUDE_DIR'):
            glog_include_path = line.split('=')[1]
        if line.startswith('glog_BINARY_DIR'):
            glog_binary_dir = line.split('=')[1]
        if line.startswith('glog_SOURCE_DIR'):
            glog_source_dir = line.split('=')[1]
        if line.startswith('Boost_ROOT'):
            boost_root = line.split('=')[1]
        if line.startswith('CONDA_PREFIX'):
            conda_prefix = line.split('=')[1]
    copy_lib()

    compile_args = {
        'language': 'c++',
        'include_dirs': [
            "./",
            "../",
            "../third_party/mpool/allocator/include/",
            "../third_party/mpool/pages_pool/include/",
            *torch_include_path,
            f"{cuda_root_path}/include"
        ],
        'libraries': ['new_torch_col'],
        'library_dirs': ['torch_col/lib'],
        'extra_compile_args': [
            '-std=c++17',
            '-DMPOOL_VERBOSE_LEVEL=0',
            '-DMPOOL_CHECK_LEVEL=0',
        ],
        'extra_link_args': [
            '-Wl,-rpath,$ORIGIN/../lib',
        ],
    }

    if 'NOTFOUND' in glog_include_path:
        # use third_party glog
        assert glog_binary_dir is not None
        assert glog_source_dir is not None
        compile_args['include_dirs'].append(f"{glog_source_dir}/src")
        compile_args['include_dirs'].append(f"{glog_binary_dir}")

    if boost_root is not None or boost_root != conda_prefix:
        boost_root_path = pathlib.Path(boost_root)
        boost_install_path = boost_root_path.parent.parent
        compile_args['include_dirs'].append(f'{boost_install_path / "include"}')
        compile_args['library_dirs'].append(f'{boost_install_path / "lib"}')

    c_ext = Extension(
        name="torch_col._C._main",
        sources=[
            "torch_col/cython/main.pyx",
        ],
        **compile_args,
    )
    c_dist_ext = Extension(
        name="torch_col._C._dist",
        sources=[
            "torch_col/cython/dist.pyx",
        ],
        **compile_args
    )

    return cythonize([c_ext, c_dist_ext], language_level=3) 


setup(
    packages=find_packages(),
    package_data={
        "torch_col" : ['lib/*.so']
    },
    cmdclass={'build_ext': build_ext},
    ext_modules=config_extension(),
)