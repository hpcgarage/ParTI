#ifndef SPTOL_H_INCLUDED
#define SPTOL_H_INCLUDED

#include <stddef.h>
#include <stdio.h>
#include <omp.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Define sptScalar as 64-bit float
 * (or 32-bit float, if an old GPU is not good at 64-bit arithmetic
 */
typedef double sptScalar;

/**
 * Dynamic array of scalars
 */
typedef struct {
    size_t    len;   /// length
    size_t    cap;   /// capacity
    sptScalar *data; /// data
} sptVector;

/**
 * Dynamic array of size_t's
 */
typedef struct {
    size_t len;   /// length
    size_t cap;   /// capacity
    size_t *data; /// data
} sptSizeVector;

/**
 * Dense matrix type
 */
typedef struct {
    size_t    nrows;   /// # rows
    size_t    ncols;   /// # columns
    size_t    stride;  /// ncols rounded up to 8
    sptScalar *values; /// values, length nrows*stride
} sptMatrix;

/**
 * Sparse matrix type
 */
typedef struct {
    size_t        nrows;  /// # rows
    size_t        ncols;  /// # colums
    size_t        nnz;    /// # non-zeros
    sptSizeVector rowind; /// row indices, length nnz
    sptSizeVector colind; /// column indices, length nnz
    sptVector     values; /// non-zero values, length nnz
} sptSparseMatrix;

/**
 * Sparse tensor type
 */
typedef struct {
    size_t        nmodes; /// # modes
    size_t        *ndims; /// size of each mode, length nmodes
    size_t        nnz;    /// # non-zeros
    sptSizeVector *inds;  /// indices of each element, length [nmodes][nnz]
    sptVector     values; /// non-zero values, length nnz
} sptSparseTensor;

/**
 * Semi-sparse tensor type
 * The last mode is dense, while other modes are sparse.
 * Can be considered as "sparse tensor of dense fiber".
 * The "fiber" here can be defined as a vector of elements that have indices
 * only different in the last mode.
 */
typedef struct {
    size_t        nmodes; /// # Modes, must >= 2
    size_t        *ndims; /// size of each mode, length nmodes
    size_t        nnz;    /// # non-zero fibers
    sptSizeVector *inds;  /// indices of each dense fiber, length [nmodes-1][nnz]
    sptVector     values; /// dense fibers, length nnz*ndims[nmodes-1]
} sptSemiSparseTensor;

int sptNewVector(sptVector *vec, size_t len, size_t cap);
int sptCopyVector(sptVector *dest, const sptVector *src);
int sptAppendVector(sptVector *vec, sptScalar value);
int sptResizeVector(sptVector *vec, size_t size);
void sptFreeVector(sptVector *vec);

int sptNewSizeVector(sptSizeVector *vec, size_t len, size_t cap);
int sptCopySizeVector(sptSizeVector *dest, const sptSizeVector *src);
int sptAppendSizeVector(sptSizeVector *vec, size_t value);
int sptResizeSizeVector(sptSizeVector *vec, size_t value);
void sptFreeSizeVector(sptSizeVector *vec);

int sptNewMatrix(sptMatrix *mtx, size_t nrows, size_t ncols);
int sptCopyMatrix(sptMatrix *dest, const sptMatrix *src);
void sptFreeMatrix(sptMatrix *mtx);

int sptNewSparseMatrix(sptSparseMatrix *mtx, size_t nrows, size_t ncols);
int sptCopySparseMatrix(sptSparseMatrix *dest, const sptSparseMatrix *src);
void sptFreeSparseMatrix(sptSparseMatrix *mtx);

int sptNewSparseTensor(sptSparseTensor *tsr, size_t nmodes, const size_t ndims[]);
int sptCopySparseTensor(sptSparseTensor *dest, const sptSparseTensor *src);
void sptFreeSparseTensor(sptSparseTensor *tsr);
int sptLoadSparseTensor(sptSparseTensor *tsr, FILE *fp);
int sptDumpSparseTensor(const sptSparseTensor *tsr, FILE *fp);

int sptNewSemiSparseTensor(sptSemiSparseTensor *tsr, size_t nmodes, const size_t ndims[]);
int sptCopySemiSparseTensor(sptSemiSparseTensor *dest, const sptSemiSparseTensor *src);
void sptFreeSemiSparseTensor(sptSemiSparseTensor *tsr);

void sptSparseTensorSortIndex(sptSparseTensor *tsr);
int sptSparseTensorAdd(sptSparseTensor *Y, const sptSparseTensor *X);
int sptSparseTensorSub(sptSparseTensor *Y, const sptSparseTensor *X);
int sptSparseTensorMulScalar(sptSparseTensor *X, sptScalar a);
int sptSparseTensorDivScalar(sptSparseTensor *X, sptScalar a);
int sptSparseTensorDotMul(sptSparseTensor *Y, const sptSparseTensor *X);
int sptSparseTensorDotDiv(sptSparseTensor *Y, const sptSparseTensor *X);

/**
 * Sparse tensor times a dense matrix (TTM)
 * Input: sparse tensor X[I][J][K], dense matrix U[I][R}, mode n={0, 1, 2}
 * Output: sparse tensor Y[I][J][R] (e.g. n=2)
 */
int sptSparseTensorMulMatrix(sptSparseTensor *Y, const sptSparseTensor *X, const sptMatrix *U, size_t mode);

/**
 * Kronecker product
 */
int sptSparseTensorKroneckerMul(sptSparseTensor *Y, const sptSparseTensor *A, const sptSparseTensor *B);

/**
 * Khatri-Rao product
 */
int sptSparseTensorKhatriRaoMul(sptSparseTensor *Y, const sptSparseTensor *A, const sptSparseTensor *B);

#ifdef __cplusplus
}
#endif

#endif
