# test script for computing MSF for a Hindmarsh-Rose neuron model

from integrator import Integrator
from integrator import methods
from msf_tools import MSF_Integrator

from hr_neuron import HR_neuron
from hr_params import reference_set
import numpy as np
import time

tmax = 5e3 # ms
dt = 0.0005 # ms
n = 100

init = np.array([0.0, # x
                 0.0, # y
                 0.0])# z 

m = HR_neuron()
m.set_parameters(**reference_set)

m.par.I_ext = 2.5
m.par.eps = 0.7
m.par.row_sum = 1

i = Integrator(m, tmax, dt)
i.set_initial_conditions(init)
msfi = MSF_Integrator(m)

#def plot_traces(idx=0, skip = 10):
    #t = np.linspace(0, tmax, np.floor(i.data.shape[1] / skip) + 1)
    #plt.figure()
    #plt.plot(t, i.data[idx, ::skip])
    #plt.show()

def timeit(closure, n):
    total_t = 0
    for j in range(n):
        t1 = time.time()
        closure()
        t2 = time.time()
        total_t += t2 - t1
    return total_t / n

def floq(gamma, dt, traj):
    msfi.compute_Floquet(gamma, dt, traj) 
    evals,_ = np.linalg.eig(msfi.X[-1])
    
    return np.log(np.sort(np.abs(evals))[::-1])

print('Computing trajectory...\n')
idx1, idx2 = i.find_limit_cycle(1e-5, 0, -1.4, methods.EULER)
traj = i.data[:, idx1:idx2].copy()
del i

print('\nTesting MSF with lambda = row_sum...\n')

print(floq(-m.par.eps, dt, traj))

print('\nMeasuring computation time...\n')
f = lambda: msfi.compute_Floquet(m.par.eps, dt, traj)
print("{0} seconds per call, dt = {1}, ndim = {2}, niter = {3}".format(timeit(f, n), dt, m.ndim, traj.shape[1]))
