# setup script for SGY (golomb 2006) model

from integrator import Integrator
from sgy_double_neuron import SGY_double_neuron
from sgy_params import reference_set, unstable_set
import pylab as plt
import numpy as np

tmax = 200 # ms
#tmax = 0.001 # ms
dt = 0.0005 # ms
#dt = 0.001 # ms

#init = np.array([18, # V, mvolt
                 #0.1, # h
                 #0.7, # n
                 #0.07, # z
                 #0.1, # s_ampa
                 #0.1, # x_nmda
                 #0.8,

                 #18.5, # V, mvolt
                 #0.15, # h
                 #0.75, # n
                 #0.075, # z
                 #0.15, # s_ampa
                 #0.15, # x_nmda
                 #0.85])# s_nmda 
init = np.array([-50, # V, mvolt
                 0.8, # h
                 0.1, # n
                 0.002, # z
                 0.0, # s_ampa
                 0.0, # x_nmda
                 0.0,

                 -50, # V, mvolt
                 0.8, # h
                 0.1, # n
                 0.002, # z
                 0.0, # s_ampa
                 0.0, # x_nmda
                 0.0])# s_nmda 

def set_w(w, m):
    W = np.array(w).astype('double')
    m.par.W_arr[:, :] = W[:, :]
    #m.set_parameters(**reference_set)

def find_matrix(l1, l2, s):
    c = ((l1 + l2 -2*s) + np.sqrt((l1+l2-2*s)**2 + 8 * ((l1+l2)*s - s**2 - l1*l2))) / -4.                     
    d = s - c
    a = l1 + l2 - d
    b = c
    return np.array([[a,b],[c,d]])

def find_matrixb(b, l1, l2, s):
    d = 0.5 * ((b + l1 + l2) + np.sqrt((b + l1 + l2)**2 - 4 * (l1 * l2 + b * s)))
    a = l1 + l2 - d
    c = s - d
    return np.array([[a,b],[c,d]])

def plot_traces(idx=0):
    plt.plot(i.data[idx, ::10])
    plt.plot(i.data[idx + 7, ::10])
    plt.show()

w = np.array([[0, 1.2],
             [1.2, 0]]).astype('double')


m = SGY_double_neuron()
m.par.W_arr = np.empty((2,2))
set_w(w, m)
m.set_parameters(**reference_set)
m.par.g_ampa = 0
m.par.g_nmda = 0.09

i = Integrator(m, tmax, dt)
i.set_initial_conditions(init)
