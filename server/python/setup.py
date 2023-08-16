from setuptools import Extension, setup
from Cython.Build import cythonize
import shutil
import os

shutil.rmtree("build", ignore_errors=True)
shutil.rmtree("dist", ignore_errors=True)
shutil.rmtree("pycolserve.egg-info", ignore_errors=True)

setup(
    name="pycolserve",
    ext_modules=cythonize(Extension(
        name="pycolserve", 
        sources=[
            "pycolserve.pyx", 
        ],
        language="c++",
        include_dirs=[
            "../../build/_deps/glog-src/src", 
            "../../build/_deps/glog-build/"
        ],
        extra_compile_args=["-std=c++17"],
        extra_link_args=["-lglog", "-lrt"]
    )
))