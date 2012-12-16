# To implement a new compatible model, cimport this header and inherit 
# from Model

cdef class Model:
    cdef int ndim
    cdef double I_app # maybe drop this one

    cdef void compute_dx(self, double[:] x, double[:] out_dx)

# math helper functions
cdef extern from "math.h":
    double exp(double x)
    double sqrt(double x)

cdef inline double sigm(double V, double theta, double sigma):
    return  1 / (1 + exp(-(V - theta) / sigma))
