#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    author: Simone Boglio
    mail: bogliosimone@gmail.com
"""

import glob
import logging
import os.path
import os
import platform
import sys

from setuptools import Extension, setup, find_packages

NAME = 'similaripy'
VERSION = [l.split("'")[1] for l in open("similaripy/__init__.py")
           if l.startswith("__version__ =")][0]

files_to_compile = ['s_plus']

try:
    from Cython.Build import cythonize
    use_cython = True
except ImportError:
    use_cython = False

if not use_cython:
    raise RuntimeError('Cython required to build %s packasge.' % NAME)

use_openmp = True

# compile args from https://msdn.microsoft.com/en-us/library/fwkeyyhe.aspx
# define_extensions from https://github.com/benfred/implicit
def define_extensions(use_cython=False):
    files = [f for f in os.listdir('.') if os.path.isfile(f)]
    for f in files:
        print(f)
    if sys.platform.startswith("win"):
        compile_args = ['/O2', '/openmp']
        link_args = []
    else:
        gcc = extract_gcc_binaries()
        if gcc is not None:
            rpath = '/usr/local/opt/gcc/lib/gcc/' + gcc[-1] + '/'
            link_args = ['-Wl,-rpath,' + rpath]
        else:
            link_args = []

        compile_args = ['-Wno-unused-function', '-Wno-maybe-uninitialized', '-O3', '-ffast-math']
        if use_openmp:
            compile_args.append("-fopenmp")
            link_args.append("-fopenmp")

        compile_args.append("-std=c++11")
        link_args.append("-std=c++11")

    src_ext = '.pyx' if use_cython else '.cpp'
    modules = [Extension("similaripy.cython_code." + name,
                         [os.path.join("similaripy", "cython_code", name + src_ext)],
                         language='c++',
                         extra_compile_args=compile_args, extra_link_args=link_args)
               for name in files_to_compile]

    if use_cython:
        return cythonize(modules)
    else:
        return modules


# extract_gcc_binaries and set_gcc copied from implicit project
# https://github.com/benfred/implicit

def extract_gcc_binaries():
    """Try to find GCC on OSX for OpenMP support."""
    patterns = ['/opt/local/bin/g++-mp-[0-9].[0-9]',
                '/opt/local/bin/g++-mp-[0-9]',
                '/usr/local/bin/g++-[0-9].[0-9]',
                '/usr/local/bin/g++-[0-9]']
    if 'darwin' in platform.platform().lower():
        gcc_binaries = []
        for pattern in patterns:
            gcc_binaries += glob.glob(pattern)
        gcc_binaries.sort()
        if gcc_binaries:
            _, gcc = os.path.split(gcc_binaries[-1])
            return gcc
        else:
            return None
    else:
        return None


def set_gcc():
    """Try to use GCC on OSX for OpenMP support."""
    # For macports and homebrew

    if 'darwin' in platform.platform().lower():
        gcc = extract_gcc_binaries()

        if gcc is not None:
            os.environ["CC"] = gcc
            os.environ["CXX"] = gcc

        else:
            global use_openmp
            use_openmp = False
            logging.warning('No GCC available. Install gcc from Homebrew '
                            'using brew install gcc.')

set_gcc()

print('ciao')

setup(name=NAME,
      version=VERSION,
      description='knn similarity package for sparse matrix',
      url='http://github.com/bogliosimone/similaripy',
      author='Simone Boglio',
      author_email='bogliosimone@gmail.com',
      license='MIT',
      packages=['similaripy'],
      install_requires=[
          'scipy',
          'numpy',
          'sklearn',
          'tqdm',
      ],
      include_package_data=True,
      setup_requires=["Cython>=0.24"],
      ext_modules= define_extensions(use_cython),
      zip_safe=False)