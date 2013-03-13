#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True

import numpy as np
cimport numpy as np
cimport cython
from base_model cimport Model


cdef class Integrator:
    cdef Model model
    cdef double dt
    cdef int n_iter

    # allocate all temporaty storage arrays
    cdef readonly np.ndarray data

    def __init__(self, Model model, double tmax, double dt):
        self.model = model
        self.dt = dt
        self.n_iter = int(tmax / dt)
        self.data = np.empty((model.ndim, self.n_iter + 1), dtype = np.double)
    
    def integrate(self, IntegrationType t = RK2):
        self._integrate_model(t, NULL, NULL)

    def integrate_with_progress(self, IntegrationType t = RK2):
        self._integrate_model(t, print_progress, NULL)

    def find_limit_cycle(self,
            double dev_thr,
            int test_var_idx = 0,
            double test_var_thr = 0.0,
            IntegrationType t = RK2):

        cdef check_deviation_args args

        args.dev_thr = dev_thr
        args.test_var_idx = test_var_idx
        args.test_var_thr = test_var_thr
        args.start_idx = 0

        self._integrate_model(t, check_deviation, &args)

    def set_initial_conditions(self, vector):
        self.data[:, 0] = vector

    # C functions ------------------------------------------------------------

    cdef void _integrate_model(self,
            const IntegrationType it,
            Callback_t callback,
            void *user_data):

        cdef int i

        cdef double[:, :] data
        cdef double[:] tmp, k1, k2, k3, k4
        cdef double[:] x


        data = self.data
        tmp = np.empty(self.model.ndim, dtype = np.double)
        k1 = np.empty(self.model.ndim, dtype = np.double)
        k2 = np.empty(self.model.ndim, dtype = np.double)
        k3 = np.empty(self.model.ndim, dtype = np.double)
        k4 = np.empty(self.model.ndim, dtype = np.double)

        for i in range(self.n_iter):
            x = data[:, i]

            if it == EULER:
                self._rk_step(x, x, k1, tmp, 0.0) # k1 <-- dt * dx

                for j in range(self.model.ndim):
                    data[j, i + 1] = data[j, i] + k1[j]

            elif it == RK2:
                self._rk_step(x, x, k1, tmp, 0.0)
                self._rk_step(x, k1, k2, tmp, 1.0)

                for j in range(self.model.ndim):
                    data[j, i + 1] = data[j, i] + (k1[j] + k2[j]) / 2.0

            elif it == RK4:
                self._rk_step(x, x, k1, tmp, 0.0)
                self._rk_step(x, k1, k2, tmp, 0.5)
                self._rk_step(x, k2, k3, tmp, 0.5)
                self._rk_step(x, k3, k4, tmp, 1.0)

                for j in range(self.model.ndim):
                    data[j, i + 1] = data[j, i] + (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j]) / 6.0

            else:
                pass

            if callback != NULL:
                if callback(data, self.dt, i, user_data):
                    break


    cdef void _rk_step(self, double[:] x, double[:] k_in, double[:] k_out, double[:] tmp, double frac):
        cdef int j

        for j in range(self.model.ndim):
            tmp[j] = x[j] + frac * k_in[j]

        self.model.compute_dx(tmp, k_out)

        for j in range(self.model.ndim):
            k_out[j] = self.dt * k_out[j]





# Callbacks

cdef int print_progress(double[:, :] data, double dt, int i, void *user_data):
    if i % 50000 == 0:
        print float(i) / float(data.shape[1]) * 100, ' %'

    return 0

cdef int check_deviation(double[:, :] data, double dt, int i, void *user_data):
    cdef check_deviation_args *args
    cdef double dev

    args = <check_deviation_args*>user_data # read arguments
    
    if (data[args.test_var_idx, i] <= args.test_var_thr) & \
        (data[args.test_var_idx, i + 1] > args.test_var_thr):
            
        # safe to use numpy because this is only called once per burst
        dev = np.linalg.norm(np.array(data[:, args.start_idx]) - np.array(data[:, i + 1]))
        print dev
        
        if dev < args.dev_thr:
            args.end_idx = i + 1
            return 1

        args.start_idx = i + 1

    return 0

# A workaround to expose IntegrationTypes enum to Python
class methods: pass
methods.EULER = EULER
methods.RK2 = RK2
methods.RK4 = RK4
