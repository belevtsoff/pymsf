# setup script for SGY (golomb 2006) model

from msf_tools import MSF_Integrator
from integrator import Integrator
from sgy_neuron import SGY_neuron
from sgy_params import reference_set
from sgy_params import unstable_set
import pylab as plt
import numpy as np

tmax = 2e3 # ms
dt = 0.00005 # ms

#init = np.array([30, # V, mvolt
                 #0.1, # h
                 #0.7, # n
                 #0.07, # z
                 #0.1, # s_ampa
                 #0.1, # x_nmda
                 #0.8])# s_nmda 

init = np.array([-50, # V, mvolt
                 0.83, # h
                 0.11, # n
                 0.002, # z
                 0.0, # s_ampa
                 0.0, # x_nmda
                 0.0])# s_nmda 

#traj = np.load('sgy_period_ref.npy')

m = SGY_neuron()
m.set_parameters(**reference_set)

# unstable parameters
m.par.g_ampa = 0
m.par.g_nmda = 0.09
m.par.w = 1.0

msfi = MSF_Integrator(m)
i = Integrator(m, tmax, dt)
i.set_initial_conditions(init)

# compute periodic trajectory
idx1, idx2 = i.find_limit_cycle(1e-5, test_var_thr=-55)
traj = i.data[:, idx1:idx2].copy()
del i

def floq(gamma):
    msfi.compute_Floquet(gamma, dt, traj) 
    evals,_ = np.linalg.eig(msfi.X[-1])
    return np.sort(np.abs(evals))[::-1]#import time

#t1 = time.time()
#msfer.compute_Floquet(1, 0.0005, traj)
#t2 = time.time()

#print t2-t1
