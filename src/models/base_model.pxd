cdef class Model:
    cdef int ndim
    #cdef double I_app # maybe drop this one

    cdef void compute_dx(self, double[:] x, double[:] out_dx)
