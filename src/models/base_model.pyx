cdef class Model:
    '''Base class for a neuron Model.
    All neuron models must inherit from this class'''

    cdef void compute_dx(self, double[:] x, double[:] out_dx):
        pass

    cdef void compute_DF(self, double[:] x, complex[:, :] out_DF):
        pass

    cdef void compute_DH(self, double[:] x, complex[:, :] out_DH):
        pass
