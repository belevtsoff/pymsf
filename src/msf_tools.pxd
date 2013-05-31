cdef extern from "cblas.h":
    enum CBLAS_ORDER:
        CblasRowMajor
    enum CBLAS_TRANSPOSE:
        CblasNoTrans

    void cblas_zcopy(
                        const int N,
                        const void *X,
                        const int incX,
                        void *Y,
                        const int incY
                    )

    void cblas_zscal(
                     const int N, const void *alpha, void *X, const int incX)

    void cblas_zaxpy(
                        const int N,
                        const void *alpha,
                        const void *X,
                        const int incX,
                        void *Y,
                        const int incY
                    )

    void cblas_zgemm(
                        const CBLAS_ORDER Order,
                        const CBLAS_TRANSPOSE TransA,
                        const CBLAS_TRANSPOSE TransB,
                        const int M,
                        const int N,
                        const int K,
                        const void *alpha,
                        const void *A,
                        const int lda,
                        const void *B,
                        const int ldb,
                        const void *beta,
                        void *C,
                        const int ldc
                    )    

    void cblas_zdscal(
                        const int N,
                        const double alpha,
                        void *X,
                        const int incX
                     )

    double cblas_dznrm2(
                        const int N,
                        const void *X,
                        const int incX
                       )

#cdef extern from "clapack.h":
    #int clapack_zgeqrf(
                        #const enum CBLAS_ORDER Order,
                        #const int M,
                        #const int N,
                        #void *A,
                        #const int lda,
                        #void *TAU
                      #)

cdef extern from "lapacke.h":
    int LAPACK_ROW_MAJOR

    int LAPACKE_zgeqrf(
                       int matrix_order,
                       int m,
                       int n,
                       complex *a,
                       int lda,
                       complex *tau
                      )

    int LAPACKE_zungqr(
                       int matrix_order,
                       int m,
                       int n,
                       int k,
                       complex *a,
                       int lda,
                       const complex *tau
                      )
