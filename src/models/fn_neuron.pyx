from integrator cimport Model
cimport cython

cdef class FN_neuron(Model):

    cdef FN_parameters par

    def __cinit__(self):
        self.ndim = 2
        self.I_app = 0
        self.par = FN_parameters()

    def load_params(self, **parameters):
        for key, value in parameters.items():
            setattr(self.par, key, value)
    
    cdef void compute_dx(self, double[:] x, double[:] out_dx):
        out_dx[0] = x[0] - x[0] ** 3 / 3 - x[1] + self.par.I_app
        out_dx[1] = self.par.phi * (x[0] + 0.7 - 0.8 * x[1])


cdef class FN_parameters:
    cdef public double I_app
    cdef public double phi
