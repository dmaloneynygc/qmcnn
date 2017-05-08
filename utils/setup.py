"""Setup script."""
from distutils.core import setup
from Cython.Build import cythonize
import numpy

ext = cythonize('utils.pyx', include_path=[numpy.get_include()])
ext[0].include_dirs = [numpy.get_include()]
setup(
    ext_modules=ext
)
