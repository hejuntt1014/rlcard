"""
A3 Dizhu C++ Engine build script

Local dev (Windows):
    pip install pybind11
    python setup_cpp.py build_ext --inplace

Server (Linux):
    pip install pybind11
    python setup_cpp.py build_ext --inplace

The compiled module `a3dizhu_cpp.*.so` (or .pyd on Windows)
will be placed in the current directory and can be imported directly.
"""

import os
import sys
from setuptools import setup, Extension

try:
    from pybind11.setup_helpers import Pybind11Extension, build_ext
except ImportError:
    raise RuntimeError("pybind11 is required: pip install pybind11")

cpp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cpp_engine")

ext_modules = [
    Pybind11Extension(
        "a3dizhu_cpp",
        sources=[
            os.path.join(cpp_dir, "a3dizhu.cpp"),
            os.path.join(cpp_dir, "bindings.cpp"),
        ],
        include_dirs=[cpp_dir],
        depends=[os.path.join(cpp_dir, 'a3dizhu.h')],
        cxx_std=17,
        define_macros=[("NDEBUG", "1")],
        extra_compile_args=['/utf-8'] if sys.platform == 'win32' else [],
    ),
]

v12_dir = os.path.join(cpp_dir, 'a3_v12')
ext_modules.append(Pybind11Extension(
    'a3dizhu_v12_cpp',
    sources=[os.path.join(v12_dir, 'a3dizhu.cpp'), os.path.join(v12_dir, 'bindings.cpp')],
    include_dirs=[v12_dir], depends=[os.path.join(v12_dir, 'a3dizhu.h')],
    cxx_std=17, define_macros=[('NDEBUG', '1')],
    extra_compile_args=['/utf-8'] if sys.platform == 'win32' else [],
))

setup(
    name="a3dizhu_cpp",
    version="1.0.0",
    description="A3 Dizhu C++ game engine for RL training",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
