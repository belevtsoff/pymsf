# setup script for SGY (golomb 2006) model

from integrator import Integrator
from sgy_neuron import SGY_neuron
from sgy_params import reference_set
import pylab as plt
import numpy as np

tmax = 5e3 # ms
dt = 0.0005 # ms

init = np.array([18, # V, mvolt
                 0.1, # h
                 0.7, # n
                 0.07, # z
                 0.1, # s_ampa
                 0.1, # x_nmda
                 0.8])# s_nmda 

#init = np.zeros(7)

m = SGY_neuron()
m.set_parameters(**reference_set)
m.par.w = 1.0

i = Integrator(m, tmax, dt)
i.set_initial_conditions(init)
