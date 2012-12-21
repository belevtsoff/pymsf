cdef extern from "math.h":
    double exp(double x)

cdef inline double sigm(double V, double theta, double sigma):
    return  1 / (1 + exp(-(V - theta) / sigma))

