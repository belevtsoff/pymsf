cdef extern from "cblas.h":
    enum CBLAS_ORDER:
        CblasRowMajor
    enum CBLAS_TRANSPOSE:
        CblasNoTrans

    void cblas_zcopy(
                        int N,
                        void *X,
                        int incX,
                        void *Y,
                        int incY
                    )

    void cblas_zaxpy(
                        int N,
                        void *alpha,
                        void *X,
                        int incX,
                        void *Y,
                        int incY
                    )

    void cblas_zgemm(
                        CBLAS_ORDER Order,
                        CBLAS_TRANSPOSE TransA,
                        CBLAS_TRANSPOSE TransB,
                        int M,
                        int N,
                        int K,
                        void *alpha,
                        void *A,
                        int lda,
                        void *B,
                        int ldb,
                        void *beta,
                        void *C,
                        int ldc
                    )    

    void cblas_zdscal(
                        int N,
                        double alpha,
                        void *X,
                        int incX
                    )
