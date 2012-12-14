#from pyintegrator import Integrator as PyIntegrator
#from pyintegrator import Model as PyModel
#from integrator import Integrator, Model
import integrator
import pyintegrator
#from simple_model import SimpleModel
import pylab as plt

tmax = 80
dt = 0.000010

#m = SimpleModel()
pym = pyintegrator.Model()
m = integrator.Model()

pyi = pyintegrator.Integrator(pym, tmax, dt)
i = integrator.Integrator(m, tmax, dt)
