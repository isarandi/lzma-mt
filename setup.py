import os
import sys
from setuptools import setup, Extension
from Cython.Build import cythonize

# Platform-specific configuration for liblzma
include_dirs = []
library_dirs = []
extra_compile_args = []
extra_link_args = []

if sys.platform == "darwin":
    # macOS: xz built from source and installed to /usr/local
    include_dirs = ["/usr/local/include"]
    library_dirs = ["/usr/local/lib"]
    extra_compile_args = ["-Wno-unreachable-code"]
elif sys.platform == "win32":
    # Windows: vcpkg paths via environment variables
    include = os.environ.get("INCLUDE", "")
    lib = os.environ.get("LIB", "")
    if include:
        include_dirs = include.split(";")
    if lib:
        library_dirs = lib.split(";")
else:
    # Linux: xz built from source and installed to /usr/local (for manylinux)
    include_dirs = ["/usr/local/include"]
    library_dirs = ["/usr/local/lib"]

ext_modules = [
    Extension(
        "lzma_mt.lzma_mt",
        sources=["src/lzma_mt/lzma_mt.pyx"],
        libraries=["lzma"],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
]

setup(ext_modules=cythonize(ext_modules))