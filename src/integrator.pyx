# Integrates the model until it is settled to a limit cycle and then
# stores the last full period

cimport numpy as np
import numpy as np
cimport cython

cdef class Model:
    '''Simplest possible model. It serves as an abstract class to be
    inherited by other model classes'''

    def __cinit__(self):
        self.ndim = 1

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void compute_dx(self, double[:] in_x, double[:] out_dx):
        out_dx[0] = 0.0

cdef class Integrator:
    cdef Model model
    cdef double tmax, dt
    cdef int n_iter

    cdef readonly np.ndarray data
    cdef np.ndarray dx

    def __cinit__(self, Model model, double tmax, double dt):
        self.model = model
        self.tmax = tmax
        self.dt = dt
        self.n_iter = <int>(tmax / dt)
        self.data = np.empty((model.ndim, self.n_iter + 1), dtype = np.double)
        self.dx = np.empty(model.ndim, dtype = np.double)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def integrate(self):
        cdef int i, j
        cdef double[:, :] data # memory view on data
        cdef double[:] dx # memory view on dx

        data = self.data
        dx = self.dx

        for i in range(self.n_iter):
            self.model.compute_dx(data[:, i], dx)

            for j in range(self.model.ndim):
                data[j, i + 1] = data[j, i] + dx[j] * self.dt

