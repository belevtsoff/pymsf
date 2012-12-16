# Integrates the model until it is settled to a limit cycle and then
# stores the last full period

#TODO:
# - limit cycle search
# - protection against uninitialized parameters

cimport numpy as np
import numpy as np
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef class Integrator:
    cdef Model model
    cdef double tmax, dt
    cdef int n_iter

    cdef readonly np.ndarray data
    cdef np.ndarray dx

    def __init__(self, Model model, double tmax, double dt):
        self.model = model
        self.tmax = tmax
        self.dt = dt
        self.n_iter = <int>(tmax / dt)
        self.data = np.empty((model.ndim, self.n_iter + 1), dtype = np.double)
        self.dx = np.empty(model.ndim, dtype = np.double)
    
    def set_initial_conditions(self, vector):
        self.data[:, 0] = vector

    def find_limit_cycle(self, double dev_thr, int test_var_idx = 0, double test_var_thr = 0.0):
        cdef int i, j
        cdef int start_idx
        cdef double dev
        cdef double[:, :] data
        cdef double[:] dx

        start_idx = 0
        dev = 1.0

        data = self.data
        dx = self.dx

        for i in range(self.n_iter):
            self.model.compute_dx(data[:, i], dx)

            # Euler integration step
            for j in range(self.model.ndim):
                data[j, i + 1] = data[j, i] + dx[j] * self.dt

            if (data[test_var_idx, i] <= test_var_thr) & (data[test_var_idx, i + 1] > test_var_thr):
                
                # safe to use numpy because this is only called once per burst
                dev = np.linalg.norm(self.data[:, start_idx] - data[:, i + 1])
                
                if dev < dev_thr:
                    break

                start_idx = i + 1

        return start_idx, i + 1



cdef class Model:
    '''Simplest possible model. It serves as an abstract class to be
    inherited by other model classes'''

    def __cinit__(self):
        self.ndim = 1

    cdef void compute_dx(self, double[:] x, double[:] out_dx):
        out_dx[0] = 0.0
