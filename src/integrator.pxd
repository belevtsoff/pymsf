cimport numpy as np
cimport cython
from base_model cimport Model

# callback function signature
ctypedef int (*Callback_t)(double[:, :] data, double dt, int i, void *user_data)

# check_deviation callback arguments
ctypedef struct check_deviation_args:
    int test_var_idx
    double test_var_thr
    double dev_thr
    int start_idx
    int end_idx

# enum for integration types
cdef enum IntegrationType:
    EULER
    RK2
    RK4

# To inherit Integrator in your Cython module, e.g. to write a custom
# callback function, cimport it in your Cython module and include 'src' and
# 'src/models' directories in your setup.py options
cdef class Integrator:
    cdef Model model
    cdef double dt
    cdef int n_iter

    # allocate all temporaty storage arrays
    cdef readonly np.ndarray data

    cdef void _integrate_model(self,
            const IntegrationType it,
            const Callback_t callback,
            void *user_data)

    cdef void _rk_step(self,
            double[:] x,
            double[:] k_in,
            double[:] k_out,
            double[:] tmp,
            double frac)
