# To implement a new compatible model, cimport this header and inherit 
# from Model

cdef class Model:
    cdef int ndim
    cdef void compute_dx(self, double[:] in_x, double[:] out_dx)
