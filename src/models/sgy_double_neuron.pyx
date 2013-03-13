# self-coupled SGY neuron model (Golomb 2006). With reference parameters
# exhibits intrinsic bursting behaviour

#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True

from base_model cimport Model
cimport cython
cimport numpy as np

# constants
DEF NDIM = 14

cdef class SGY_double_neuron(Model):
    # currents and gating variables
    cdef double I_na, I_nap, I_kdr, I_kslow, I_l, I_ampa, I_nmda, I_app
    cdef double m_inf, h_inf, p_inf, n_inf, z_inf, s_inf
    cdef double tau_h, tau_n
    cdef double[:, :] W

    # parameters
    cdef public SGY_double_parameters par

    def __cinit__(self):
        self.ndim = NDIM
        self.par = SGY_double_parameters()

        # initialize I_app, DF and DH to zeros
        self.I_app = 0

    def set_parameters(self, **parameters):
        for key, value in parameters.items():
            setattr(self.par, key, value)

        self.W = self.par.W_arr

    cdef void compute_dx(self, double[:] x, double[:] out_dx):
        # neuron 1
        self.compute_one_dx(x[0:NDIM / 2], out_dx[0:NDIM / 2], x[NDIM / 2 + 4], x[NDIM / 2 + 6], self.W[0, 0], self.W[0, 1])

        # neuron 1
        self.compute_one_dx(x[NDIM / 2:NDIM], out_dx[NDIM / 2:NDIM], x[4], x[6], self.W[1, 1], self.W[1, 0])

    cdef void compute_one_dx(self, double[:] x, double[:] out_dx, double s_ampa_ext, double s_nmda_ext, double w_int, double w_ext):
        """computes the right side of the ODE, given the state vector"""
        # 0 : V
        # 1 : h
        # 2 : n
        # 3 : z
        # 4 : s_ampa
        # 5 : x_nmda
        # 6 : s_nmda

        self.m_inf = sigm(x[0], self.par.theta_m, self.par.sigma_m)
        self.h_inf = sigm(x[0], self.par.theta_h, self.par.sigma_h)
        self.p_inf = sigm(x[0], self.par.theta_p, self.par.sigma_p)
        self.n_inf = sigm(x[0], self.par.theta_n, self.par.sigma_n)
        self.z_inf = sigm(x[0], self.par.theta_z, self.par.sigma_z)
        self.s_inf = sigm(x[0], self.par.theta_s, self.par.sigma_s)

        self.tau_h = 0.1 + 0.75 * sigm(x[0], self.par.theta_th, self.par.sigma_th)
        self.tau_n = 0.1 + 0.5 * sigm(x[0], self.par.theta_tn, self.par.sigma_tn)

        self.I_na = self.par.g_na * self.m_inf ** 3 * x[1] * (x[0] - self.par.V_na)
        self.I_nap = self.par.g_nap * self.p_inf * (x[0] - self.par.V_na)
        self.I_kdr = self.par.g_kdr * x[2] ** 4 * (x[0] - self.par.V_k)
        self.I_kslow = self.par.g_kslow * x[3] * (x[0] - self.par.V_k)
        self.I_l = self.par.g_l * (x[0] - self.par.V_l)
        self.I_ampa = self.par.g_ampa * (x[0] - self.par.V_glu) * (w_int * x[4] + w_ext * s_ampa_ext)
        self.I_nmda = self.par.g_nmda * (x[0] - self.par.V_glu) * (w_int * x[6] + w_ext * s_nmda_ext)

        out_dx[0] = (- self.I_na - self.I_nap - self.I_kdr - self.I_kslow - self.I_l\
                    - self.I_ampa - self.I_nmda + self.I_app) / self.par.Cm
        out_dx[1] = (self.h_inf - x[1]) / self.tau_h
        out_dx[2] = (self.n_inf - x[2]) / self.tau_n
        out_dx[3] = (self.z_inf - x[3]) / self.par.tau_z
        out_dx[4] = self.par.k_fp * self.s_inf * (1 - x[4]) - x[4] / self.par.tau_ampa
        out_dx[5] = self.par.k_xn * self.s_inf * (1 - x[5])\
                    - (1 - self.s_inf) * x[5] / self.par.tau_hat_nmda
        out_dx[6] = self.par.k_fn * x[5] * (1 - x[6]) - x[6] / self.par.tau_nmda


cdef class SGY_double_parameters:
    cdef public double Cm
    cdef public double g_na, g_nap, g_kdr, g_kslow, g_l, g_ampa, g_nmda
    cdef public double V_na, V_k, V_l, V_glu
    cdef public double theta_m, theta_h, theta_th, theta_p, theta_n, theta_tn,\
                theta_z, theta_s
    cdef public double sigma_m, sigma_h, sigma_th, sigma_p, sigma_n, sigma_tn,\
                sigma_z, sigma_s
    cdef public double tau_z, tau_ampa, tau_hat_nmda, tau_nmda
    cdef public double k_fp, k_xn, k_fn
    cdef public np.ndarray W_arr 
    #cdef public double c # row-sum of the coupling matrix
