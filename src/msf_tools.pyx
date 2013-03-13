cimport numpy as np
import numpy as np
cimport cython
from base_model cimport Model


cdef class MSF_Integrator:
    cdef Model model
    cdef readonly np.ndarray X

    def __init__(self, Model model):
        self.model = model
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def compute_Floquet(self, complex gamma, double dt, np.ndarray[double, ndim=2] np_traj):

        cdef int i, irow, icol # indices
        cdef int ndim = self.model.ndim

        cdef complex one = 1
        cdef complex zero = 0
        cdef complex complex_dt = <complex>dt
        cdef complex gamma_dt = gamma * complex_dt

        # declare memoryviews
        cdef complex[:, :, ::1] X
        cdef complex[:, ::1] DF, DH, ACC
        cdef double[:, :] traj

        # declare and initialize temporary arrays
        cdef np.ndarray DF_arr = np.zeros((ndim, ndim), dtype = np.complex)
        cdef np.ndarray DH_arr = np.zeros((ndim, ndim), dtype = np.complex)
        cdef np.ndarray ACC_arr = np.zeros((ndim, ndim), dtype = np.complex)

        # initialize memoryviews
        DF, DH, ACC = DF_arr, DH_arr, ACC_arr
        traj = np_traj

        # allocate mem for fundamental solution
        self.X = np.empty((traj.shape[1] + 1, ndim, ndim), dtype = np.complex)
        X = self.X

        # initialize X[0] to identity
        self.X[0, :, :] = np.eye(ndim)

        # compute MSF

        for i in range(traj.shape[1]):
            self.model.compute_DF(traj[:, i], DF)
            self.model.compute_DH(traj[:, i], DH)
            
            ## ACC <- DF
            ## ACC <- ACC + gamma * DH
            ## X(t + 1) <- X(t) 
            ## X(t + 1) <- dt * ACC * X(t) + X(t + 1)
            #cblas_zcopy(ndim ** 2, &DF[0, 0], 1, &ACC[0, 0], 1)
            #cblas_zaxpy(ndim ** 2, &gamma, &DH[0, 0], 1, &ACC[0, 0], 1)
            #cblas_zcopy(ndim ** 2, &X[i, 0, 0], 1, &X[i + 1, 0, 0], 1)
            #cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        #ndim, ndim, ndim,
                        #&complex_dt, &ACC[0, 0], ndim, &X[i, 0, 0], ndim,
                        #&one, &X[i + 1, 0, 0], ndim)

            # X(t + 1) = (I + dt * DF + dt * gamma * DH) * X(t):
            #
            # ACC <- I
            # ACC <- ACC + dt * DF
            # ACC <- ACC + gamma * dt * DH 
            # X(t + 1) <- ACC * X(t)
            for irow in range(ndim):
                for icol in range(ndim):
                    if irow == icol: ACC[irow, icol] = 1
                    else: ACC[irow, icol] = 0
            cblas_zaxpy(ndim ** 2, &complex_dt, &DF[0, 0], 1, &ACC[0, 0], 1)
            cblas_zaxpy(ndim ** 2, &gamma_dt, &DH[0, 0], 1, &ACC[0, 0], 1)
            cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        ndim, ndim, ndim,
                        &one, &ACC[0, 0], ndim, &X[i, 0, 0], ndim,
                        &zero, &X[i + 1, 0, 0], ndim)
