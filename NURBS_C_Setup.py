from setuptools import setup
from Cython.Build import cythonize

setup(ext_modules=cythonize('NURBS_C.pyx'))