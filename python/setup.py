import setuptools
from setuptools import Extension, setup, find_packages
from Cython.Build import cythonize, build_ext
import shutil
import pathlib
import os
from torch import utils
from torch.utils.cpp_extension import BuildExtension, CppExtension

shutil.rmtree("build", ignore_errors=True)
shutil.rmtree("dist", ignore_errors=True)
shutil.rmtree("torch_col.egg-info", ignore_errors=True)


class TorchColExtention(setuptools.command.build_ext.build_ext):
    def run(self):
        self.copy_lib()
        super().run()            

    def copy_lib(self):
        lib_dir = os.path.join('torch_col', 'lib')
        if os.path.exists(lib_dir):
            shutil.rmtree(lib_dir)
        os.mkdir(lib_dir)
        shutil.copy('../build/python/libtorch_col.so', lib_dir)
        shutil.copy('../build/libsta.so', lib_dir)


def config_extension():
    cython_ext = Extension(
        name="torch_col._C", 
        sources=[
            "./torch_col/torch_col.pyx",
            "./torch_col/logging.cc"
        ],
        language="c++",
        include_dirs=[
            "../",
            "./torch_col",
        ],
        libraries=["rt", "sta", "torch_col"],
        library_dirs=[
            "../build",
            '../build/python'
        ],
        # extra_link_args=["-lglog", "-Wl,-Bstatic"],
#define 
        extra_compile_args=["-std=c++17", "-DPY_EXTENSION_LOGGING=\"logging.h\""],
    )

    return cythonize(cython_ext)

setup(
    name="torch_col",
    packages=find_packages(),
    package_data={
        'torch_col' : [
            'lib/*.so',
        ]
    },
    ext_modules=config_extension(),
    cmdclass={
        'build_ext': TorchColExtention,
    },
)