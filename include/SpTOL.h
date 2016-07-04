#ifndef SPTOL_H_INCLUDED
#define SPTOL_H_INCLUDED

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Define sptScalar as 64-bit float
 * (or 32-bit float, if an old GPU is not good at 64-bit arithmetic
 */
typedef double sptScalar;

/**
 * Sparse matrix type
 */
typedef struct {
    size_t    nrows;   /// # rows
    size_t    ncols;   /// # colums
    size_t    nnz;     /// # non-zeros
    size_t    *rowind; /// row indices, length nnz
    size_t    *colind; /// column indices, length nnz
    sptScalar *values; /// non-zero values, length nnz
} sptSparseMatrix;

/**
 * Dense matrix type
 */
typedef struct {
    size_t    nrows;   /// # rows
    size_t    ncols;   /// # columns
    sptScalar *values; /// values, length nrows*ncols
} sptMatrix;

/**
 * Sparse tensor type
 */
typedef struct {
    size_t    nmodes;  /// # modes
    size_t    *ndims;  /// size of each mode, length nmodes
    size_t    nnz;     /// # non-zeros
    size_t    **inds;  /// indices of each element, length [nmodes][nnz]
    sptScalar *values; /// non-zero values, length nnz
} sptSparseTensor;

/**
 * Semi-sparse tensor type
 * The last mode is dense, while other modes are sparse.
 * Can be considered as "sparse tensor of dense fiber".
 * The "fiber" here can be defined as a vector of elements that have indices
 * only different in the last mode.
 */
typedef struct {
    size_t    nmodes;  /// # Modes, must >= 2
    size_t    *ndims;  /// size of each mode, length nmodes
    size_t    nnz;     /// # non-zero fibers
    size_t    **inds;  /// indices of each dense fiber, length [nmodes-1][nnz]
    sptScalar *fibers; /// dense fibers, length nnz*ndims[nmodes-1]
} sptSemiSparseTensor;

/**
 * Calls free() on a sparse matrix, useful to ensure all pointers are freed
 * Set free_func to free when calling this, unless stated.
 */
void sptFreeSparseMatrix(sptSparseMatrix *mtx, void (*free_func)(void *));

/**
 * Calls free() on a dense matrix, useful to ensure all pointers are freed
 * Set free_func to free when calling this, unless stated.
 */
void sptFreeMatrix(sptMatrix *mtx, void (*free_func)(void *));

/**
 * Calls free() on a sparse tensor, useful to ensure all pointers are freed
 * Set free_func to free when calling this, unless stated.
 */
void sptFreeSparseTensor(sptSparseTensor *mtx, void (*free_func)(void *));

/**
 * Calls free() on a semi-sparse tensor, useful to ensure all pointers are freed
 * Set free_func to free when calling this, unless stated.
 */
void sptFreeSemiSparseTensor(sptSemiSparseTensor *mtx, void (*free_func)(void *));

/**
 * Element-wise addition on a sparse tensor.
 */
int sptSparseTensorAdd(sptSparseTensor *Y, const sptSparseTensor *X);

/**
 * Element-wise subtraction on a sparse tensor.
 */
int sptSparseTensorSub(sptSparseTensor *Y, const sptSparseTensor *X);

/**
 * Scalar multiplication on a sparse tensor.
 */
int sptSparseTensorMulScalar(sptSparseTensor *Y, sptScalar a);

/**
 * Scalar division on a sparse tensor.
 */
int sptSparseTensorDivScalar(sptSparseTensor *Y, sptScalar a);

/**
 * Element-wise multiplication on a sparse tensor.
 */
int sptSparseTensorDotMul(sptSparseTensor *Y, const sptSparseTensor *X);

/**
 * Element-wise division on a sparse tensor.
 */
int sptSparseTensorDotDiv(sptSparseTensor *Y, const sptSparseTensor *X);

/**
 * Sparse tensor times a dense matrix (TTM)
 * Input: sparse tensor X[I][J][K], dense matrix U[I][R}, mode n={0, 1, 2}
 * Output: sparse tensor Y[I][J][R] (e.g. n=2)
 * Free Y with sptFreeSparseTensor(Y, NULL)
 */
int sptSparseTensorMulMatrix(sptSparseTensor *Y, const sptSparseTensor *X, const sptMatrix *U);

/**
 * Kronecker product
 * Free Y with sptFreeSparseTensor(Y, NULL)
 */
int sptSparseTensorKroneckerMul(sptSparseTensor *Y, const sptSparseTensor *A, const sptSparseTensor *B);

/**
 * Khatrio-Rao product
 * Free Y with sptFreeSparseTensor(Y, NULL)
 */
int sptSparseTensorKhatrioRaoMul(sptSparseTensor *Y, const sptSparseTensor *A, const sptSparseTensor *B);

#ifdef __cplusplus
}
#endif

#endif
