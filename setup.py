from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import os

ext_modules = [
               Extension("integrator", ["src/integrator.pyx"]),
               Extension("fn_neuron", ["src/models/fn_neuron.pyx"], include_dirs=['src']),
               Extension("sgy_neuron", ["src/models/sgy_neuron.pyx"], include_dirs=['src'])
               ]

setup(
  name = 'Limit cycle search',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules,
)

