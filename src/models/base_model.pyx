cdef class Model:
    '''Base class for a neuron Model.
    All neuron models must inherit from this class'''

    cdef void compute_dx(self, double[:] x, double[:] out_dx):
        pass

