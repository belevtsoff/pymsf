#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True

cimport numpy as np
import numpy as np
cimport cython
from base_model cimport Model

from libc.math cimport log, fabs, sqrt


cdef class MSF_Integrator:
    cdef Model model
    cdef readonly np.ndarray X

    def __init__(self, Model model):
        self.model = model
    
    def compute_Floquet(self, complex gamma, double dt, np.ndarray[double, ndim=2] np_traj):

        cdef int i
        cdef int ndim = self.model.ndim

        # create memoryviews
        self.X = np.empty((np_traj.shape[1] + 1, ndim, ndim), dtype = np.complex)

        cdef double[:, :] traj = np_traj
        cdef complex[:, :, ::1] X = self.X
        cdef complex[:, ::1] DF = np.zeros((ndim, ndim), dtype = np.complex)
        cdef complex[:, ::1] DH = np.zeros((ndim, ndim), dtype = np.complex)
        cdef complex[:, ::1] ACC = np.zeros((ndim, ndim), dtype = np.complex)

        # set X(0) = I
        self.X[0, :, :] = np.eye(ndim)

        # compute MSF
        for i in range(traj.shape[1]):
            euler_step_msf(self.model, i, gamma, dt, X, traj, DF, DH, ACC)

    def compute_Lyapunov(self,
            complex gamma,
            double dt,
            np.ndarray[double, ndim=2] np_traj,
            
            double max_diff = 1e-4,
            int max_iter = <int>1e4,
            int n = 100
            ):

        # declarations
        cdef int i, iteration # counters
        cdef int ndim
        cdef double lambda_diff
        cdef np.ndarray[double, ndim=2] np_lambdas

        cdef double[:, ::1] lambdas
        cdef double[:, :] traj
        cdef complex[:, :, ::1] X
        cdef complex[:, ::1] DF
        cdef complex[:, ::1] DH
        cdef complex[:, ::1] ACC

        ndim = self.model.ndim
        np_lambdas = np.zeros((ndim, max_iter))
        iteration = 0
        lambda_diff = 1

        # create memoryviews
        self.X = np.empty((np_traj.shape[1] + 1, ndim, ndim), dtype = np.complex)

        lambdas = np_lambdas
        traj = np_traj
        X = self.X
        DF = np.zeros((ndim, ndim), dtype = np.complex)
        DH = np.zeros((ndim, ndim), dtype = np.complex)
        ACC = np.zeros((ndim, ndim), dtype = np.complex)

        # set X(0) = I
        self.X[0, :, :] = np.eye(ndim)

        # main loop
        while lambda_diff > max_diff and iteration < max_iter:
            for i in range(traj.shape[1]):
                euler_step_msf(self.model, i, gamma, dt, X, traj, DF, DH, ACC)

                # compute LE's every n steps
                if not i % n:
                    compute_les(i + 1, iteration, dt*n, X, ACC, lambdas)
                    iteration += 1

                if iteration >= max_iter:
                    break

            # rare, using numpy
            self.X[0, :, :] = self.X[-1, :, :]
            print iteration

        return np_lambdas

# C functions

cdef void compute_les(
        int i,
        int iteration,
        double deltat,
        complex[:, :, :] X,
        complex[:, :] ACC,
        double[:, :] lambdas):

    cdef int icol
    cdef int ndim = X.shape[1]

    # Compute LE's and reorthonormalize X using QR factorization

    # get post-GSR norms via QR factorization
    # X <- QR
    # ACC (first row) <- tau
    LAPACKE_zgeqrf(LAPACK_ROW_MAJOR, ndim, ndim, &X[i, 0, 0], ndim, &ACC[0, 0])

    # calculate LE's
    for icol in range(ndim):
        lambdas[icol, iteration] = log(fabs(X[i, icol, icol].real)) / deltat

    # reorthogonalize by generating Q
    LAPACKE_zungqr(LAPACK_ROW_MAJOR, ndim, ndim, ndim, &X[i, 0, 0], ndim, &ACC[0, 0])


#cdef void compute_les(
        #int i,
        #int iteration,
        #double epsilon,
        #double deltat,
        #complex[:, :, :] X,
        #complex[:, :] ACC,
        #double[:, :] lambdas):

    #cdef int icol
    #cdef int ndim = X.shape[1]
    ##cdef double norm

    ## Compute LE's and reorthonormalize X using QR factorization

    ## get post-GSR norms via QR factorization
    ## X <- QR
    ## ACC (first row) <- tau
    #LAPACKE_zgeqrf(LAPACK_ROW_MAJOR, ndim, ndim, &X[i, 0, 0], ndim, &ACC[0, 0])

    ## calculate LE's
    #for icol in range(ndim):
        #lambdas[icol, iteration] = log(fabs(X[i, icol, icol].real) / epsilon) / deltat

    ## reorthogonalize by generating Q and scaling it with epsilon
    ## X <- epsilon * Q
    #LAPACKE_zungqr(LAPACK_ROW_MAJOR, ndim, ndim, ndim, &X[i, 0, 0], ndim, &ACC[0, 0])

def test_qr():
    cdef int i
    cdef int ndim = 3

    np_X = np.random.random((ndim, ndim)).astype('complex')
    np_Q = np_X.copy()
    np_norms = np.empty(ndim).astype('complex')
    cdef complex[:] norms = np_norms
    cdef complex[:, ::1] Q = np_Q
    cdef complex[:, ::1] ACC = np.empty((ndim, ndim)).astype('complex')

    LAPACKE_zgeqrf(LAPACK_ROW_MAJOR, ndim, ndim, &Q[0, 0], ndim, &ACC[0, 0])

    for i in range(ndim):
        norms[i] = Q[i, i]

    LAPACKE_zungqr(LAPACK_ROW_MAJOR, ndim, ndim, ndim, &Q[0, 0], ndim, &ACC[0, 0])

    return np_X, np_Q, np_norms



#    # for every column of X
    #for icol in range(ndim):
        ## compute its norm
        #norm = cblas_dznrm2(ndim, &X[i, 0, icol], ndim)
        ## write corresponding LE
        #lambdas[icol, iteration] = log(norm / epsilon) / deltat
        ## renormalize
        #for irow in range(ndim):
#            X[i, irow, icol] = X[i, irow, icol] * epsilon / norm


cdef void compute_labmda_diff():
    pass

cdef void euler_step_msf(Model model,
                        int i,
                        complex gamma,
                        complex dt,

                        complex[:, :, :] X,
                        double[:, :] traj,
                        complex[:, :] DF,
                        complex[:, :] DH,
                        complex[:, :] ACC):

    cdef int ndim = model.ndim
    cdef complex one = 1
    cdef complex zero = 0
    cdef complex gamma_times_dt = gamma * dt

    model.compute_DF(traj[:, i], DF)
    model.compute_DH(traj[:, i], DH)
    
    # One Euler integration step:
    #
    # X(t + 1) = (I + dt * DF + dt * gamma * DH) * X(t)
    #
    # ACC <- I
    # ACC <- ACC + dt * DF
    # ACC <- ACC + gamma * dt * DH 
    # X(t + 1) <- ACC * X(t)
    identity(ACC, ndim)
    cblas_zaxpy(ndim ** 2, &dt, &DF[0, 0], 1, &ACC[0, 0], 1)
    cblas_zaxpy(ndim ** 2, &gamma_times_dt, &DH[0, 0], 1, &ACC[0, 0], 1)
    cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                ndim, ndim, ndim,
                &one, &ACC[0, 0], ndim, &X[i, 0, 0], ndim,
                &zero, &X[i + 1, 0, 0], ndim)

cdef void identity(complex[:, :] M, int ndim):
    cdef int irow, icol

    for irow in range(ndim):
        for icol in range(ndim):
            if irow == icol: M[irow, icol] = 1
            else: M[irow, icol] = 0
