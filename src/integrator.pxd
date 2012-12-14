cdef class Model:
    cdef int ndim
    cdef void compute_dx(self, double[:] in_x, double[:] out_dx)
