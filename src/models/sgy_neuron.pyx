# self-coupled SGY neuron model (Golomb 2006). With reference parameters
# exhibits intrinsic bursting behaviour

#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True

from base_model cimport Model
cimport cython

# constants
DEF NDIM = 7

cdef class SGY_neuron(Model):
    # temporary matrices (Jacobians) for MSF calculations
    #cdef complex DF[NDIM][NDIM]
    #cdef complex DH[NDIM][NDIM]

    # currents and gating variables
    cdef double I_na, I_nap, I_kdr, I_kslow, I_l, I_ampa, I_nmda, I_app
    cdef double m_inf, h_inf, p_inf, n_inf, z_inf, s_inf
    cdef double tau_h, tau_n

    # parameters
    cdef public SGY_parameters par

    def __cinit__(self):
        self.ndim = NDIM
        self.par = SGY_parameters()

        # initialize I_app, DF and DH to zeros
        self.I_app = 0

        #for i in range(NDIM):
            #for j in range(NDIM):
                #self.DF[i, j] = 0
                #self.DH[i, j] = 0

    def set_parameters(self, **parameters):
        for key, value in parameters.items():
            setattr(self.par, key, value)

    cdef void compute_dx(self, double[:] x, double[:] out_dx):
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
        self.I_ampa = self.par.w * self.par.g_ampa * x[4] * (x[0] - self.par.V_glu)
        self.I_nmda = self.par.w * self.par.g_nmda * x[6] * (x[0] - self.par.V_glu)

        out_dx[0] = (- self.I_na - self.I_nap - self.I_kdr - self.I_kslow - self.I_l\
                    - self.I_ampa - self.I_nmda + self.I_app) / self.par.Cm
        out_dx[1] = (self.h_inf - x[1]) / self.tau_h
        out_dx[2] = (self.n_inf - x[2]) / self.tau_n
        out_dx[3] = (self.z_inf - x[3]) / self.par.tau_z
        out_dx[4] = self.par.k_fp * self.s_inf * (1 - x[4]) - x[4] / self.par.tau_ampa
        out_dx[5] = self.par.k_xn * self.s_inf * (1 - x[5])\
                    - (1 - self.s_inf) * x[5] / self.par.tau_hat_nmda
        out_dx[6] = self.par.k_fn * x[5] * (1 - x[6]) - x[6] / self.par.tau_nmda

    #cdef void compute_MSF(self, complex[:, :] X, double alpha, double beta, complex[:, :] out_MSF):
        #"""computes the Master Stability Function for this model"""
        #elf.DF[0][0] = 

    cdef void compute_DF(self, double[:] x, complex[:, :] out_DF):
        """computes Jacobian of the internal dynamics of this model.
        It is assumed that all elements of out_DF that are not modified by
        this method are set to zeros.
        
        Derivatives are computed using Sympy with sgy_jacobian.py"""

        out_DF[0, 0] = (self.par.w*(-self.par.g_ampa*x[4] - self.par.g_nmda*x[6])/self.par.Cm + (-self.par.g_kdr*x[2]**4 - self.par.g_kslow*x[3] - self.par.g_l - self.par.g_na*x[1]/(exp((self.par.theta_m - x[0])/self.par.sigma_m) + 1)**3 - 3*self.par.g_na*x[1]*(-self.par.V_na + x[0])*exp((self.par.theta_m - x[0])/self.par.sigma_m)/(self.par.sigma_m*(exp((self.par.theta_m - x[0])/self.par.sigma_m) + 1)**4) - self.par.g_nap/(exp((self.par.theta_p - x[0])/self.par.sigma_p) + 1) - self.par.g_nap*(-self.par.V_na + x[0])*exp((self.par.theta_p - x[0])/self.par.sigma_p)/(self.par.sigma_p*(exp((self.par.theta_p - x[0])/self.par.sigma_p) + 1)**2))/self.par.Cm)
        out_DF[0, 1] = -self.par.g_na*(-self.par.V_na + x[0])/(self.par.Cm*(exp((self.par.theta_m - x[0])/self.par.sigma_m) + 1)**3)
        out_DF[0, 2] = -4*self.par.g_kdr*x[2]**3*(-self.par.V_k + x[0])/self.par.Cm
        out_DF[0, 3] = -self.par.g_kslow*(-self.par.V_k + x[0])/self.par.Cm

        out_DF[1, 0] = -0.75*(-x[1] + 1/(exp((self.par.theta_h - x[0])/self.par.sigma_h) + 1))*exp((self.par.theta_th - x[0])/self.par.sigma_th)/(self.par.sigma_th*(0.1 + 0.75/(exp((self.par.theta_th - x[0])/self.par.sigma_th) + 1))**2*(exp((self.par.theta_th - x[0])/self.par.sigma_th) + 1)**2) + exp((self.par.theta_h - x[0])/self.par.sigma_h)/(self.par.sigma_h*(0.1 + 0.75/(exp((self.par.theta_th - x[0])/self.par.sigma_th) + 1))*(exp((self.par.theta_h - x[0])/self.par.sigma_h) + 1)**2)
        out_DF[1, 1] = -1/(0.1 + 0.75/(exp((self.par.theta_th - x[0])/self.par.sigma_th) + 1))

        out_DF[2, 0] = -0.5*(-x[2] + 1/(exp((self.par.theta_n - x[0])/self.par.sigma_n) + 1))*exp((self.par.theta_tn - x[0])/self.par.sigma_tn)/(self.par.sigma_tn*(0.1 + 0.5/(exp((self.par.theta_tn - x[0])/self.par.sigma_tn) + 1))**2*(exp((self.par.theta_tn - x[0])/self.par.sigma_tn) + 1)**2) + exp((self.par.theta_n - x[0])/self.par.sigma_n)/(self.par.sigma_n*(0.1 + 0.5/(exp((self.par.theta_tn - x[0])/self.par.sigma_tn) + 1))*(exp((self.par.theta_n - x[0])/self.par.sigma_n) + 1)**2)
        out_DF[2, 2] = -1/(0.1 + 0.5/(exp((self.par.theta_tn - x[0])/self.par.sigma_tn) + 1))

        out_DF[3, 0] = exp((self.par.theta_z - x[0])/self.par.sigma_z)/(self.par.sigma_z*self.par.tau_z*(exp((self.par.theta_z - x[0])/self.par.sigma_z) + 1)**2)
        out_DF[3, 3] = -1/self.par.tau_z

        out_DF[4, 0] = self.par.k_fp*(-x[4] + 1)*exp((self.par.theta_s - x[0])/self.par.sigma_s)/(self.par.sigma_s*(exp((self.par.theta_s - x[0])/self.par.sigma_s) + 1)**2)
        out_DF[4, 4] = -self.par.k_fp/(exp((self.par.theta_s - x[0])/self.par.sigma_s) + 1) - 1/self.par.tau_ampa

        out_DF[5, 0] = self.par.k_xn*(-x[5] + 1)*exp((self.par.theta_s - x[0])/self.par.sigma_s)/(self.par.sigma_s*(exp((self.par.theta_s - x[0])/self.par.sigma_s) + 1)**2) + x[5]*exp((self.par.theta_s - x[0])/self.par.sigma_s)/(self.par.sigma_s*self.par.tau_hat_nmda*(exp((self.par.theta_s - x[0])/self.par.sigma_s) + 1)**2)
        out_DF[5, 5] = -self.par.k_xn/(exp((self.par.theta_s - x[0])/self.par.sigma_s) + 1) - (1 - 1/(exp((self.par.theta_s - x[0])/self.par.sigma_s) + 1))/self.par.tau_hat_nmda

        out_DF[6, 5] = self.par.k_fn*(-x[6] + 1)
        out_DF[6, 6] = -self.par.k_fn*x[5] - 1/self.par.tau_nmda

    cdef void compute_DH(self, double[:] x, complex[:, :] out_DH):
        """Computes Jacobian of the coupled dynamics for this model
        It is assumed that all elements of out_DH that are not modified by
        this method are set to zeros.
        
        Derivatives are computed using Sympy with sgy_jacobian.py"""

        out_DH[0, 4] = -self.par.g_ampa*(-self.par.V_glu + x[0])/self.par.Cm
        out_DH[0, 6] = -self.par.g_nmda*(-self.par.V_glu + x[0])/self.par.Cm

 

cdef class SGY_parameters:
    cdef public double Cm
    cdef public double g_na, g_nap, g_kdr, g_kslow, g_l, g_ampa, g_nmda
    cdef public double V_na, V_k, V_l, V_glu
    cdef public double theta_m, theta_h, theta_th, theta_p, theta_n, theta_tn,\
                theta_z, theta_s
    cdef public double sigma_m, sigma_h, sigma_th, sigma_p, sigma_n, sigma_tn,\
                sigma_z, sigma_s
    cdef public double tau_z, tau_ampa, tau_hat_nmda, tau_nmda
    cdef public double k_fp, k_xn, k_fn
    cdef public double w # self-coupling strength = row-sum of the coupling matrix
    #cdef public double c # row-sum of the coupling matrix
