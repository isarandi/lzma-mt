import sys
from setuptools import setup, Extension
from Cython.Build import cythonize

# Platform-specific compiler arguments
if sys.platform == 'win32':
    extra_compile_args = ['/O2']
else:
    extra_compile_args = ['-O3']

ext_modules = [
    Extension(
        'lzma_mt.lzma_mt',
        sources=['src/lzma_mt/lzma_mt.pyx'],
        libraries=['lzma'],
        extra_compile_args=extra_compile_args,
    ),
]

setup(ext_modules=cythonize(ext_modules))
