cdef class Model:
    cdef int ndim
    #cdef double I_app # maybe drop this one

    cdef void compute_dx(self, double[:] x, double[:] out_dx)
    cdef void compute_DF(self, double[:] x, complex[:, :] out_DF)
    cdef void compute_DH(self, double[:] x, complex[:, :] out_DH)
