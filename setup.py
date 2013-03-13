from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [
               Extension("integrator", ["src/integrator.pyx"], include_dirs=['src/models']),
               Extension("msf_tools", ["src/msf_tools.pyx"], include_dirs=['src/models'], libraries=['cblas']),
               Extension("base_model", ["src/models/base_model.pyx"]),
               Extension("sgy_neuron", ["src/models/sgy_neuron.pyx"]),
               Extension("sgy_double_neuron", ["src/models/sgy_double_neuron.pyx"])
               ]

setup(
  name = 'Limit cycle search',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules,
)

