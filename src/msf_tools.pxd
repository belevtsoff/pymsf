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
