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
