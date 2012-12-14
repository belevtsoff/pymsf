import integrator
import pylab as plt

tmax = 80
dt = 0.000010

m = integrator.Model()

i = integrator.Integrator(m, tmax, dt)
