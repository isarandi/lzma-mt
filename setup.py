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
    # macOS: xz built from source and installed to /usr/local (wheel builds),
    # with Homebrew paths as fallback (/opt/homebrew on arm64)
    include_dirs = ["/usr/local/include", "/opt/homebrew/include"]
    library_dirs = ["/usr/local/lib", "/opt/homebrew/lib"]
    extra_compile_args = ["-Wno-unreachable-code"]
elif sys.platform == "win32":
    # Windows: vcpkg paths via environment variables; drop empty entries so
    # trailing semicolons don't turn into bare /LIBPATH: arguments
    include = os.environ.get("INCLUDE", "")
    lib = os.environ.get("LIB", "")
    include_dirs = [p for p in include.split(";") if p]
    library_dirs = [p for p in lib.split(";") if p]
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