from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import os

ext_modules = [
               Extension("integrator", ["src/integrator.pyx"], include_dirs=['src/models']),
               Extension("base_model", ["src/models/base_model.pyx"]),
               Extension("sgy_neuron", ["src/models/sgy_neuron.pyx"])
               ]

setup(
  name = 'Limit cycle search',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules,
)

