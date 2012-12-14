from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [
               Extension("integrator", ["src/integrator.pyx"]),
               Extension("neuron_models", ["src/neuron_models.pyx"])
               ]

setup(
  name = 'Limit cycle search',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules
)
