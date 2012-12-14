from integrator cimport Model
cimport cython

cdef class FN_neuron(Model):
    cdef double I_app

    def __cinit__(self, I_app = 0.5):
        #Model.__cinit_(self)
        self.ndim = 2
        self.I_app = I_app
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void compute_dx(self, double[:] in_x, double[:] out_dx):
        out_dx[0] = in_x[0] - in_x[0] ** 3 / 3 - in_x[1] + self.I_app
        out_dx[1] = 0.08 * (in_x[0] + 0.7 - 0.8 * in_x[1])
