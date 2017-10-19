/*
    This file is part of ParTI!.

    ParTI! is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as
    published by the Free Software Foundation, either version 3 of
    the License, or (at your option) any later version.

    ParTI! is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with ParTI!.
    If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef PARTI_H_INCLUDED
#define PARTI_H_INCLUDED

#include <stddef.h>
#include <stdio.h>
#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#ifdef PARTI_USE_OPENMP
    #include <omp.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif


/**
 * Define sptScalar as 32-bit float (save to compatible with old ParTI code, TODO: delete it)
 *
 * You can adjust this type to suit the application
 */
typedef float sptScalar;


/**
 * Re-Define types, TODO: check the bit size of them, add branch for different settings
 */
// typedef uint16_t sptElementIndex;
// typedef uint32_t sptBlockMatrixIndex;
typedef uint_fast8_t sptElementIndex;
typedef uint_fast16_t sptBlockMatrixIndex;
// typedef uint8_t sptElementIndex;
// typedef uint16_t sptBlockMatrixIndex;
// typedef uint32_t sptElementIndex;
// typedef uint32_t sptBlockMatrixIndex;
typedef uint32_t sptBlockIndex;
typedef sptBlockIndex sptBlockNnzIndex;
typedef uint32_t sptIndex;
typedef uint64_t sptNnzIndex;
typedef float sptValue;
typedef unsigned __int128 sptMortonIndex;


#define SPT_PF_ELEMENTINDEX PRIu16
#define SPT_PF_BLOCKINDEX PRIu32
#define SPT_PF_BLOCKNNZINDEX SPT_PF_BLOCKINDEX
#define SPT_PF_INDEX PRIu32
#define SPT_PF_NNZINDEX PRIu64
#define SPT_PF_VALUE "f"


/**
 * Dense dynamic array of scalars, TODO: delete
 */
typedef struct {
    size_t    len;   /// length
    size_t    cap;   /// capacity
    sptScalar *data; /// data
} sptVector;

/**
 * Dense dynamic array of size_t's, TODO: delete
 */
typedef struct {
    size_t len;   /// length
    size_t cap;   /// capacity
    size_t *data; /// data
} sptSizeVector;


/**
 * Dense dynamic array of specified type of scalars
 */
typedef struct {
    sptNnzIndex    len;   /// length
    sptNnzIndex    cap;   /// capacity
    sptValue    *data; /// data
} sptValueVector;

/**
 * Dense dynamic array of different types of integers
 */
typedef struct {
    sptNnzIndex len;   /// length
    sptNnzIndex cap;   /// capacity
    sptElementIndex *data; /// data
} sptElementIndexVector;

typedef struct {
    sptNnzIndex len;   /// length
    sptNnzIndex cap;   /// capacity
    sptBlockIndex *data; /// data
} sptBlockIndexVector;

typedef struct {
    sptNnzIndex len;   /// length
    sptNnzIndex cap;   /// capacity
    sptIndex *data; /// data
} sptIndexVector;

typedef struct {
    sptNnzIndex len;   /// length
    sptNnzIndex cap;   /// capacity
    sptNnzIndex *data; /// data
} sptNnzIndexVector;


/**
 * Dense matrix type
 */
typedef struct {
    size_t    nrows;   /// # rows
    size_t    ncols;   /// # columns
    size_t    cap;     /// # of allocated rows
    size_t    stride;  /// ncols rounded up to 8
    sptScalar *values; /// values, length cap*stride
} sptMatrix;


/**
 * Dense matrix type, ncols = small rank (<= 256)
 */
typedef struct {
    sptIndex    nrows;   /// # rows
    sptElementIndex    ncols;   /// # columns, <= 256
    sptIndex    cap;     /// # of allocated rows
    sptElementIndex    stride;  /// ncols rounded up to 8, <= 256
    sptValue *values; /// values, length cap*stride
} sptRankMatrix;

/**
 * Sparse matrix type, COO format
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
 * Sparse matrix type, CSR format
 */
typedef struct {
    size_t        nrows;  /// # rows
    size_t        ncols;  /// # colums
    size_t        nnz;    /// # non-zeros
    sptSizeVector rowptr; /// row indices, length nnz
    sptSizeVector colind; /// column indices, length nnz
    sptVector     values; /// non-zero values, length nnz
} sptSparseMatrixCSR;


/**
 * Sparse tensor type
 */
typedef struct {
    size_t        nmodes;      /// # modes
    size_t        *sortorder;  /// the order in which the indices are sorted
    size_t        *ndims;      /// size of each mode, length nmodes
    size_t        nnz;         /// # non-zeros
    sptSizeVector *inds;       /// indices of each element, length [nmodes][nnz]
    sptVector     values;      /// non-zero values, length nnz
} sptSparseTensor;




/**
 * Sparse tensor type, Hierarchical COO format (HiCOO)
 */
typedef struct {
    /* Basic information */
    sptIndex            nmodes;      /// # modes
    sptIndex            *sortorder;  /// the order in which the indices are sorted
    sptIndex            *ndims;      /// size of each mode, length nmodes
    sptNnzIndex         nnz;         /// # non-zeros

    /* Parameters */
    sptElementIndex       sb_bits;         /// block size by nnz
    sptElementIndex       sk_bits;         /// kernel size by nnz
    sptElementIndex       sc_bits;         /// chunk size by blocks

    /* Scheduling information */
    sptNnzIndexVector         kptr;      /// Nonzero kernel pointers in 1-D array, indexing blocks. sptIndexVector may be enough
    sptIndexVector            **kschr;    /// Kernel scheduler
    sptIndex                  *nkiters;
    sptNnzIndexVector         cptr;      /// Chunk pointers to evenly split or combine blocks in a group, indexing blocks. sptIndexVector may be enough

    /* Index data arrays */
    sptNnzIndexVector         bptr;      /// Block pointers to all nonzeros
    sptBlockIndexVector       *binds;    /// Block indices within each group
    sptElementIndexVector     *einds;    /// Element indices within each block 
    sptValueVector            values;      /// non-zero values, length nnz
} sptSparseTensorHiCOO;



/**
 * Semi-sparse tensor type
 * The chosen mode is dense, while other modes are sparse.
 * Can be considered as "sparse tensor of dense fiber".
 * The "fiber" here can be defined as a vector of elements that have indices
 * only different in the last mode.
 */
typedef struct {
    size_t        nmodes; /// # Modes, must >= 2
    size_t        *ndims; /// size of each mode, length nmodes
    size_t        mode;   /// the mode where data is stored in dense format
    size_t        nnz;    /// # non-zero fibers
    sptSizeVector *inds;  /// indices of each dense fiber, length [nmodes][nnz], the mode-th value is ignored
    size_t        stride; /// ndims[mode] rounded up to 8
    sptMatrix     values; /// dense fibers, size nnz*ndims[mode]
} sptSemiSparseTensor;


/**
 * General Semi-sparse tensor type
 */
typedef struct {
    size_t        nmodes; /// # Modes, must >= 2
    size_t        *ndims; /// size of each mode, length nmodes
    size_t        ndmodes;
    size_t        *dmodes;   /// the mode where data is stored in dense format, allocate nmodes sized space
    size_t        nnz;    /// # non-zero fibers
    sptSizeVector *inds;  /// indices of each dense fiber, length [nmodes][nnz], the mode-th value is ignored
    size_t        *strides; /// ndims[mode] rounded up to 8
    sptMatrix     values; /// dense fibers, size nnz*ndims[mode]
} sptSemiSparseTensorGeneral;


/**
 * Kruskal tensor type, for CP decomposition result
 */
typedef struct {
  sptIndex nmodes;
  sptIndex rank;
  sptIndex * ndims;
  sptValue * lambda;
  double fit;
  sptMatrix ** factors;
} sptKruskalTensor;


/**
 * Kruskal tensor type, for CP decomposition result. 
 * ncols = small rank (<= 256)
 */
typedef struct {
  sptIndex nmodes;
  sptElementIndex rank;
  sptIndex * ndims;
  sptValue * lambda;
  double fit;
  sptRankMatrix ** factors;
} sptRankKruskalTensor;


/**
 * OpenMP lock pool.
 */
typedef struct
{
  bool initialized;
  sptIndex nlocks;
  sptIndex padsize;
  omp_lock_t * locks;
} sptMutexPool;

#ifndef PARTI_DEFAULT_NLOCKS
#define PARTI_DEFAULT_NLOCKS 1024
#endif

#ifndef PARTI_DEFAULT_LOCK_PAD_SIZE
#define PARTI_DEFAULT_LOCK_PAD_SIZE 16
#endif

/**
 * An opaque data type to store a specific time point, using either CPU or GPU clock.
 */
typedef struct sptTagTimer *sptTimer;

typedef enum {
    SPTERR_NO_ERROR       = 0,
    SPTERR_UNKNOWN        = 1,
    SPTERR_SHAPE_MISMATCH = 2,
    SPTERR_VALUE_ERROR    = 3,
    SPTERR_ZERO_DIVISION  = 4,
    SPTERR_NO_MORE        = 99,
    SPTERR_OS_ERROR       = 0x10000,
    SPTERR_CUDA_ERROR     = 0x20000,
} SptError;

int sptGetLastError(const char **module, const char **file, unsigned *line, const char **reason);
void sptClearLastError(void);
void spt_Panic(const char *file, unsigned line, const char *expr);
/**
 * The assert function that always execute even when `NDEBUG` is set
 *
 * Quick & dirty error checking. Useful when writing small programs.
 */
#define sptAssert(expr) ((expr) ? (void) 0 : spt_Panic(__FILE__, __LINE__, #expr))

/* Helper function for pure C module */
int sptCudaSetDevice(int device);
int sptCudaGetLastError(void);

/* Timer functions, using either CPU or GPU timer */
int sptNewTimer(sptTimer *timer, int use_cuda);
int sptStartTimer(sptTimer timer);
int sptStopTimer(sptTimer timer);
double sptElapsedTime(const sptTimer timer);
double sptPrintElapsedTime(const sptTimer timer, const char *name);
double sptPrintAverageElapsedTime(const sptTimer timer, const int niters, const char *name);
int sptFreeTimer(sptTimer timer);

/* Base functions */
size_t sptMaxSizeArray(size_t const * const indices, size_t const size);
char * sptBytesString(size_t const bytes);
sptValue sptRandomValue(void);

/* Dense vector, aka variable length array */
int sptNewVector(sptVector *vec, size_t len, size_t cap);
int sptConstantVector(sptVector * const vec, sptScalar const val);
int sptCopyVector(sptVector *dest, const sptVector *src);
int sptAppendVector(sptVector *vec, sptScalar value);
int sptAppendVectorWithVector(sptVector *vec, const sptVector *append_vec);
int sptResizeVector(sptVector *vec, size_t size);
void sptFreeVector(sptVector *vec);
int sptDumpVector(sptVector *vec, FILE *fp);

/* Dense vector, with size_t type */
int sptNewSizeVector(sptSizeVector *vec, size_t len, size_t cap);
int sptConstantSizeVector(sptSizeVector * const vec, size_t const num);
int sptCopySizeVector(sptSizeVector *dest, const sptSizeVector *src);
int sptAppendSizeVector(sptSizeVector *vec, size_t value);
int sptAppendSizeVectorWithVector(sptSizeVector *vec, const sptSizeVector *append_vec);
int sptResizeSizeVector(sptSizeVector *vec, size_t value);
void sptFreeSizeVector(sptSizeVector *vec);
int sptDumpSizeVector(sptSizeVector *vec, FILE *fp);

int sptNewElementIndexVector(sptElementIndexVector *vec, uint64_t len, uint64_t cap);
int sptAppendElementIndexVector(sptElementIndexVector *vec, sptElementIndex value);
void sptFreeElementIndexVector(sptElementIndexVector *vec);
int sptNewBlockIndexVector(sptBlockIndexVector *vec, uint64_t len, uint64_t cap);
int sptAppendBlockIndexVector(sptBlockIndexVector *vec, sptBlockIndex value);
void sptFreeBlockIndexVector(sptBlockIndexVector *vec);
int sptNewIndexVector(sptIndexVector *vec, uint64_t len, uint64_t cap);
int sptAppendIndexVector(sptIndexVector *vec, sptIndex value);
void sptFreeIndexVector(sptIndexVector *vec);
int sptNewNnzIndexVector(sptNnzIndexVector *vec, uint64_t len, uint64_t cap);
int sptAppendNnzIndexVector(sptNnzIndexVector *vec, sptNnzIndex value);
void sptFreeNnzIndexVector(sptNnzIndexVector *vec);
int sptNewValueVector(sptValueVector *vec, uint64_t len, uint64_t cap);
int sptAppendValueVector(sptValueVector *vec, sptValue value);
int sptConstantValueVector(sptValueVector * const vec, sptValue const val);
void sptFreeValueVector(sptValueVector *vec);
int sptDumpElementIndexVector(sptElementIndexVector *vec, FILE *fp);
int sptDumpBlockIndexVector(sptBlockIndexVector *vec, FILE *fp);
int sptDumpNnzIndexVector(sptNnzIndexVector *vec, FILE *fp);
int sptDumpIndexVector(sptIndexVector *vec, FILE *fp);
int sptDumpValueIndexVector(sptValueVector *vec, FILE *fp);
int sptDumpIndexArray(sptIndex *array, uint64_t n, FILE *fp);


/* Dense matrix */
int sptNewMatrix(sptMatrix *mtx, size_t nrows, size_t ncols);
int sptRandomizeMatrix(sptMatrix *mtx, size_t nrows, size_t ncols);
int sptIdentityMatrix(sptMatrix *mtx);
int sptConstantMatrix(sptMatrix * const mtx, sptScalar const val);
int sptCopyMatrix(sptMatrix *dest, const sptMatrix *src);
void sptFreeMatrix(sptMatrix *mtx);
int sptDumpMatrix(sptMatrix *mtx, FILE *fp);
int sptAppendMatrix(sptMatrix *mtx, const sptScalar values[]);
int sptResizeMatrix(sptMatrix *mtx, size_t newsize);
static inline size_t sptGetMatrixLength(const sptMatrix *mtx) {
    return mtx->nrows * mtx->stride;
}
int sptSparseTensorToMatrix(sptMatrix *dest, const sptSparseTensor *src);
int sptMatrixDotMul(sptMatrix const * A, sptMatrix const * B, sptMatrix const * C);
int sptMatrixDotMulSeq(size_t const mode, size_t const nmodes, sptMatrix ** mats);
int sptMatrixDotMulSeqCol(size_t const mode, size_t const nmodes, sptMatrix ** mats);
int sptCudaMatrixDotMulSeq(
    size_t const mode,
    size_t const nmodes,
    const size_t rank,
    const size_t stride,
    sptScalar ** dev_ata);
int sptMatrix2Norm(sptMatrix * const A, sptScalar * const lambda);
int sptCudaMatrix2Norm(
    size_t const nrows,
    size_t const ncols,
    size_t const stride,
    sptScalar * const dev_vals,
    sptScalar * const dev_lambda);
int sptMatrixMaxNorm(sptMatrix * const A, sptScalar * const lambda);
void GetFinalLambda(
  size_t const rank,
  size_t const nmodes,
  sptMatrix ** mats,
  sptScalar * const lambda);


/* Dense Rank matrix, ncols = small rank (<= 256) */
int sptNewRankMatrix(sptRankMatrix *mtx, sptIndex nrows, sptElementIndex ncols);
int sptRandomizeRankMatrix(sptRankMatrix *mtx, sptIndex nrows, sptElementIndex ncols);
int sptConstantRankMatrix(sptRankMatrix *mtx, sptValue const val);
void sptFreeRankMatrix(sptRankMatrix *mtx);
int sptDumpRankMatrix(sptRankMatrix *mtx, FILE *fp);
int sptRankMatrixDotMulSeqTriangle(sptIndex const mode, sptIndex const nmodes, sptRankMatrix ** mats);
int sptRankMatrix2Norm(sptRankMatrix * const A, sptValue * const lambda);
int sptRankMatrixMaxNorm(sptRankMatrix * const A, sptValue * const lambda);
void GetRankFinalLambda(
  sptElementIndex const rank,
  sptIndex const nmodes,
  sptRankMatrix ** mats,
  sptValue * const lambda);
int sptRankMatrixSolveNormals(
  sptIndex const mode,
  sptIndex const nmodes,
  sptRankMatrix ** aTa,
  sptRankMatrix * rhs);

/* Sparse matrix */
int sptNewSparseMatrix(sptSparseMatrix *mtx, size_t nrows, size_t ncols);
int sptCopySparseMatrix(sptSparseMatrix *dest, const sptSparseMatrix *src);
void sptFreeSparseMatrix(sptSparseMatrix *mtx);

/* Sparse tensor */
int sptNewSparseTensor(sptSparseTensor *tsr, size_t nmodes, const size_t ndims[]);
int sptCopySparseTensor(sptSparseTensor *dest, const sptSparseTensor *src);
void sptFreeSparseTensor(sptSparseTensor *tsr);
double SparseTensorFrobeniusNormSquared(sptSparseTensor const * const spten);
int sptLoadSparseTensor(sptSparseTensor *tsr, size_t start_index, FILE *fp);
int sptDumpSparseTensor(const sptSparseTensor *tsr, size_t start_index, FILE *fp);
void sptSparseTensorSortIndex(sptSparseTensor *tsr, int force);
void sptSparseTensorSortIndexAtMode(sptSparseTensor *tsr, size_t mode, int force);
void sptSparseTensorSortIndexCustomOrder(sptSparseTensor *tsr, const size_t sortkeys[], int force);
void sptSparseTensorSortIndexMorton(
    sptSparseTensor *tsr, 
    int force,
    const sptNnzIndex begin,
    const sptNnzIndex end,
    const sptElementIndex sb_bits);
void sptSparseTensorSortIndexRowBlock(
    sptSparseTensor *tsr, 
    int force,
    const sptNnzIndex begin,
    const sptNnzIndex end,
    const sptElementIndex sk_bits);
void sptSparseTensorSortIndexSingleMode(sptSparseTensor *tsr, int force, sptIndex mode);
void sptSparseTensorCalcIndexBounds(size_t inds_low[], size_t inds_high[], const sptSparseTensor *tsr);
int spt_ComputeSliceSizes(
    size_t * slice_sizes, 
    sptSparseTensor * const tsr,
    size_t const mode);
void sptSparseTensorStatus(sptSparseTensor *tsr, FILE *fp);
double sptSparseTensorDensity(sptSparseTensor const * const tsr);

/* Sparse tensor HiCOO */
int sptNewSparseTensorHiCOO(
    sptSparseTensorHiCOO *hitsr, 
    const sptIndex nmodes, 
    const sptIndex ndims[],
    const sptNnzIndex nnz,
    const sptElementIndex sb_bits,
    const sptElementIndex sk_bits,
    const sptElementIndex sc_bits);
int sptNewSparseTensorHiCOO_NoNnz(
    sptSparseTensorHiCOO *hitsr, 
    const sptIndex nmodes, 
    const sptIndex ndims[],
    const sptElementIndex sb_bits,
    const sptElementIndex sk_bits,
    const sptElementIndex sc_bits);
void sptFreeSparseTensorHiCOO(sptSparseTensorHiCOO *hitsr);
int sptSparseTensorToHiCOO(
    sptSparseTensorHiCOO *hitsr, 
    sptNnzIndex *max_nnzb,
    sptSparseTensor *tsr, 
    const sptElementIndex sb_bits,
    const sptElementIndex sk_bits,
    const sptElementIndex sc_bits);
int sptDumpSparseTensorHiCOO(sptSparseTensorHiCOO * const hitsr, FILE *fp);
void sptSparseTensorStatusHiCOO(sptSparseTensorHiCOO *hitsr, FILE *fp);
double SparseTensorFrobeniusNormSquaredHiCOO(sptSparseTensorHiCOO const * const hitsr);


/**
 * epsilon is a small positive value, every -epsilon < x < x would be considered as zero
 */
int sptSemiSparseTensorToSparseTensor(sptSparseTensor *dest, const sptSemiSparseTensor *src, sptScalar epsilon);

int sptNewSemiSparseTensor(sptSemiSparseTensor *tsr, size_t nmodes, size_t mode, const size_t ndims[]);
int sptCopySemiSparseTensor(sptSemiSparseTensor *dest, const sptSemiSparseTensor *src);
void sptFreeSemiSparseTensor(sptSemiSparseTensor *tsr);
int sptSparseTensorToSemiSparseTensor(sptSemiSparseTensor *dest, const sptSparseTensor *src, size_t mode);
int sptSemiSparseTensorSortIndex(sptSemiSparseTensor *tsr);

int sptNewSemiSparseTensorGeneral(sptSemiSparseTensorGeneral *tsr, size_t nmodes, const size_t ndims[], size_t ndmodes, const size_t dmodes[]);
void sptFreeSemiSparseTensorGeneral(sptSemiSparseTensorGeneral *tsr);

/**
 * Set indices of a semi-sparse according to a reference sparse
 * Call sptSparseTensorSortIndexAtMode on ref first
 */
int sptSemiSparseTensorSetIndices(sptSemiSparseTensor *dest, sptSizeVector *fiberidx, sptSparseTensor *ref);


/* Kruskal tensor */
int sptNewKruskalTensor(sptKruskalTensor *ktsr, sptIndex nmodes, const size_t ndims[], sptIndex rank);  // "size_t" is preserved for old coo tensor format
void sptFreeKruskalTensor(sptKruskalTensor *ktsr);
int sptDumpKruskalTensor(sptKruskalTensor *ktsr, sptIndex start_index, FILE *fp);
double KruskalTensorFit(
  sptSparseTensor const * const spten,
  sptValue const * const __restrict lambda,
  sptMatrix ** mats,
  sptMatrix ** ata);
double KruskalTensorFrobeniusNormSquared(
  sptIndex const nmodes,
  sptValue const * const __restrict lambda,
  sptMatrix ** ata);
double SparseKruskalTensorInnerProduct(
  sptIndex const nmodes,
  sptValue const * const __restrict lambda,
  sptMatrix ** mats);


/* Rank Kruskal tensor, ncols = small rank (<= 256)  */
int sptNewRankKruskalTensor(sptRankKruskalTensor *ktsr, sptIndex nmodes, const size_t ndims[], sptElementIndex rank);
void sptFreeRankKruskalTensor(sptRankKruskalTensor *ktsr);
int sptDumpRankKruskalTensor(sptRankKruskalTensor *ktsr, sptIndex start_index, FILE *fp);
double KruskalTensorFitHiCOO(
  sptSparseTensorHiCOO const * const hitsr,
  sptValue const * const __restrict lambda,
  sptRankMatrix ** mats,
  sptRankMatrix ** ata);
double KruskalTensorFrobeniusNormSquaredRank(
  sptIndex const nmodes,
  sptValue const * const __restrict lambda,
  sptRankMatrix ** ata);
double SparseKruskalTensorInnerProductRank(
  sptIndex const nmodes,
  sptValue const * const __restrict lambda,
  sptRankMatrix ** mats);


/* Sparse tensor unary operations */
int sptSparseTensorMulScalar(sptSparseTensor *X, sptScalar a);
int sptSparseTensorDivScalar(sptSparseTensor *X, sptScalar a);
/* Sparse tensor binary operations */
int sptSparseTensorAdd(sptSparseTensor *Z, const sptSparseTensor *X, const sptSparseTensor *Y);
int sptSparseTensorSub(sptSparseTensor *Z, const sptSparseTensor *X, const sptSparseTensor *Y);
int sptSparseTensorDotMul(sptSparseTensor *Z, const sptSparseTensor *X, const sptSparseTensor *Y);
int sptSparseTensorDotMulEq(sptSparseTensor *Z, const sptSparseTensor *X, const sptSparseTensor *Y);
int sptOmpSparseTensorDotMulEq(sptSparseTensor *Z, const sptSparseTensor *X, const sptSparseTensor *Y);
int sptCudaSparseTensorDotMulEq(sptSparseTensor *Z, const sptSparseTensor *X, const sptSparseTensor *Y);
int sptSparseTensorDotDiv(sptSparseTensor *Z, const sptSparseTensor *X, const sptSparseTensor *Y);

int sptSparseTensorMulMatrix(sptSemiSparseTensor *Y, sptSparseTensor *X, const sptMatrix *U, size_t mode);
int sptOmpSparseTensorMulMatrix(sptSemiSparseTensor *Y, sptSparseTensor *X, const sptMatrix *U, size_t mode);
int sptCudaSparseTensorMulMatrix(sptSemiSparseTensor *Y, sptSparseTensor *X, const sptMatrix *U, size_t mode);
int sptCudaSparseTensorMulMatrixOneKernel(sptSemiSparseTensor *Y, sptSparseTensor *X, const sptMatrix *U, size_t mode, size_t const impl_num, size_t const smen_size);

/**
 * Semi-sparse tensor times a dense matrix (TTM)
 * Input: semi-sparse tensor X[I][J][K], dense matrix U[I][R}, mode n={0, 1, 2}
 * Output: sparse tensor Y[I][J][R] (e.g. n=2)
 */
int sptSemiSparseTensorMulMatrix(sptSemiSparseTensor *Y, const sptSemiSparseTensor *X, const sptMatrix *U, size_t mode);
int sptCudaSemiSparseTensorMulMatrix(sptSemiSparseTensor *Y, const sptSemiSparseTensor *X, const sptMatrix *U, size_t mode);
/**
 * Kronecker product
 */
int sptSparseTensorKroneckerMul(sptSparseTensor *Y, const sptSparseTensor *A, const sptSparseTensor *B);

/**
 * Khatri-Rao product
 */
int sptSparseTensorKhatriRaoMul(sptSparseTensor *Y, const sptSparseTensor *A, const sptSparseTensor *B);

/**
 * Matricized tensor times Khatri-Rao product.
 */
int sptMTTKRP(
    sptSparseTensor const * const X,
    sptMatrix * mats[],     // mats[nmodes] as temporary space.
    size_t const mats_order[],    // Correspond to the mode order of X.
    size_t const mode);
int sptOmpMTTKRP(
    sptSparseTensor const * const X,
    sptMatrix * mats[],     // mats[nmodes] as temporary space.
    size_t const mats_order[],    // Correspond to the mode order of X.
    size_t const mode,
    const int tk);
int sptOmpMTTKRP_Reduce(sptSparseTensor const * const X,
    sptMatrix * mats[],     // mats[nmodes] as temporary space.
    sptMatrix * copy_mats[],    // temporary matrices for reduction
    size_t const mats_order[],    // Correspond to the mode order of X.
    size_t const mode,
    const int tk);
int sptOmpMTTKRP_Lock(sptSparseTensor const * const X,
    sptMatrix * mats[],     // mats[nmodes] as temporary space.
    size_t const mats_order[],    // Correspond to the mode order of X.
    size_t const mode,
    const int tk,
    sptMutexPool * lock_pool);
int sptCudaMTTKRP(
    sptSparseTensor const * const X,
    sptMatrix * mats[],     // mats[nmodes] as temporary space.
    size_t * const mats_order,    // Correspond to the mode order of X.
    size_t const mode,
    size_t const impl_num);
int sptCudaMTTKRPOneKernel(
    sptSparseTensor const * const X,
    sptMatrix ** const mats,     // mats[nmodes] as temporary space.
    size_t * const mats_order,    // Correspond to the mode order of X.
    size_t const mode,
    size_t const impl_num);
int sptCudaMTTKRPSM(
    sptSparseTensor const * const X,
    sptMatrix ** const mats,     // mats[nmodes] as temporary space.
    size_t * const mats_order,    // Correspond to the mode order of X.
    size_t const mode,
    size_t const impl_num);
int sptCudaMTTKRPDevice(
    const size_t mode,
    const size_t nmodes,
    const size_t nnz,
    const size_t rank,
    const size_t stride,
    const size_t * Xndims,
    size_t ** const Xinds,
    const sptScalar * Xvals,
    const size_t * dev_mats_order,
    sptScalar ** dev_mats,
    sptScalar * dev_scratch);
int sptSplittedMTTKRP(
    sptSparseTensor const *const X,
    sptMatrix *mats[],
    size_t const mats_order[],
    size_t const mode,
    sptVector *scratch,
    size_t const split_count[]
);

/* Coarse GPU */
int sptCudaCoarseMTTKRP(
    sptSparseTensor const * const X,
    sptMatrix ** const mats,     // mats[nmodes] as temporary space.
    sptSizeVector const * const mats_order,    // Correspond to the mode order of X.
    size_t const mode);

/*
   TODO: I'm not sure where to put this forward declaration.
   The "split" operation should not expose its interface to the application,
   while "MTTKRP" is a public function.
   Consider privatize the function or publicize the data type,
   or we need some sort of encapsulation?
*/
struct spt_TagSplitResult;

int sptPresplittedMTTKRP(
    struct spt_TagSplitResult const *splits,
    size_t const nsplits,
    sptMatrix *mats[],
    size_t const mats_order[],
    size_t const mode,
    sptVector *scratch
);
int sptCudaDistributedMTTKRP(
    double *queue_time,
    int const split_grain,
    struct spt_TagSplitResult const *splits,
    size_t const nsplits,
    size_t const batch_size,
    sptMatrix *mats[],
    size_t const mats_order[],
    size_t const mode,
    int const gpu_map[]);

int sptCudaOneMTTKRP(
    double *queue_time,
    double *queue_time_h2d,
    double *queue_time_d2h,
    double *queue_time_reduce,
    int const split_grain,
    sptSparseTensor * const tsr,
    struct spt_TagSplitResult const *splits,
    size_t const queue_size,
    size_t const nblocks,
    sptMatrix *mats[],
    size_t const mats_order[],
    size_t const mode,
    size_t const nnz_split_begin,
    size_t const max_nstreams,
    size_t const max_nthreadsy,
    size_t const smem_size,
    size_t const impl_num,
    size_t const cuda_dev_id);

int sptCudaOneMTTKRPAsync(
    double *queue_time,
    int const split_grain,
    sptSparseTensor * const tsr,
    struct spt_TagSplitResult const *splits,
    size_t const queue_size,
    size_t const nblocks,
    sptMatrix *mats[],
    size_t const mats_order[],
    size_t const mode,
    size_t const nnz_split_begin,
    size_t const max_nstreams,
    size_t const max_nthreadsy,
    size_t const impl_num,
    size_t const cuda_dev_id);

/**
 * Matricized tensor times Khatri-Rao product for HiCOO tensors
 */
int sptMTTKRPHiCOO(
    sptSparseTensorHiCOO const * const hitsr,
    sptMatrix * mats[],     // mats[nmodes] as temporary space.
    sptIndex const mats_order[],    // Correspond to the mode order of X.
    sptIndex const mode);
int sptMTTKRPHiCOO_MatrixTiling(
    sptSparseTensorHiCOO const * const hitsr,
    sptRankMatrix * mats[],     // mats[nmodes] as temporary space.
    sptIndex const mats_order[],    // Correspond to the mode order of X.
    sptIndex const mode);
int sptOmpMTTKRPHiCOO(
    sptSparseTensorHiCOO const * const hitsr,
    sptMatrix * mats[],     // mats[nmodes] as temporary space.
    sptIndex const mats_order[],    // Correspond to the mode order of X.
    sptIndex const mode,
    const int tk,
    const int tb);
int sptOmpMTTKRPHiCOO_MatrixTiling(
    sptSparseTensorHiCOO const * const hitsr,
    sptRankMatrix * mats[],     // mats[nmodes] as temporary space.
    sptIndex const mats_order[],    // Correspond to the mode order of X.
    sptIndex const mode,
    const int tk,
    const int tb);
int sptOmpMTTKRPHiCOO_MatrixTiling_Scheduled(
    sptSparseTensorHiCOO const * const hitsr,
    sptRankMatrix * mats[],     // mats[nmodes] as temporary space.
    sptIndex const mats_order[],    // Correspond to the mode order of X.
    sptIndex const mode,
    const int tk,
    const int tb);
int sptOmpMTTKRPHiCOO_MatrixTiling_Scheduled_Reduce(
    sptSparseTensorHiCOO const * const hitsr,
    sptRankMatrix * mats[],     // mats[nmodes] as temporary space.
    sptRankMatrix * copy_mats[],    // temporary matrices for reduction
    sptIndex const mats_order[],    // Correspond to the mode order of X.
    sptIndex const mode,
    const int tk,
    const int tb);
int sptOmpMTTKRPHiCOO_MatrixTiling_Scheduled_Reduce_Two(
    sptSparseTensorHiCOO const * const hitsr,
    sptRankMatrix * mats[],     // mats[nmodes] as temporary space.
    sptRankMatrix * copy_mats[],    // temporary matrices for reduction
    sptIndex const mats_order[],    // Correspond to the mode order of X.
    sptIndex const mode,
    const int tk,
    const int tb);
int sptCudaMTTKRPHiCOO(
    sptSparseTensorHiCOO const * const hitsr,
    sptMatrix ** const mats,     // mats[nmodes] as temporary space.
    sptIndex * const mats_order,    // Correspond to the mode order of X.
    sptIndex const mode,
    int const impl_num);
int sptMTTKRPKernelHiCOO(
    const sptIndex mode,
    const sptIndex nmodes,
    const sptNnzIndex nnz,
    const sptIndex R,
    const sptIndex stride,
    const sptElementIndex sb_bits,
    const sptElementIndex sc_bits,
    const int impl_num,
    const sptNnzIndex kptr_begin,
    const sptNnzIndex kptr_end,
    sptIndex * const dev_ndims,
    sptNnzIndex * const dev_cptr,
    sptNnzIndex * const dev_bptr,
    sptBlockIndex ** const dev_binds,
    sptElementIndex ** const dev_einds,
    sptValue * const dev_values,
    sptIndex * const dev_mats_order,
    sptValue ** const dev_mats);
int sptTTMHiCOO_MatrixTiling(
    sptSparseTensorHiCOO * const Y,
    sptSparseTensorHiCOO const * const X,
    sptRankMatrix * U,     // mats[nmodes] as temporary space.
    sptIndex const mode);

/**
 * CP-ALS
 */
int sptCpdAls(
  sptSparseTensor const * const spten,
  size_t const rank,
  size_t const niters,
  double const tol,
  sptKruskalTensor * ktensor);
int sptOmpCpdAls(
  sptSparseTensor const * const spten,
  size_t const rank,
  size_t const niters,
  double const tol,
  const int tk,
  const int use_reduce,
  sptKruskalTensor * ktensor);
int sptCudaCpdAls(
  sptSparseTensor const * const spten,
  size_t const rank,
  size_t const niters,
  double const tol,
  sptKruskalTensor * ktensor);
int sptCpdAlsHiCOO(
  sptSparseTensorHiCOO const * const hitsr,
  sptIndex const rank,
  sptIndex const niters,
  double const tol,
  sptRankKruskalTensor * ktensor);
int sptOmpCpdAlsHiCOO(
  sptSparseTensorHiCOO const * const hitsr,
  sptIndex const rank,
  sptIndex const niters,
  double const tol,
  const int tk,
  const int tb,
  sptRankKruskalTensor * ktensor);

/**
 * OMP functions
 */
int sptSparseTensorAddOMP(sptSparseTensor *Y, sptSparseTensor *X, size_t const nthreads);
int sptSparseTensorSubOMP(sptSparseTensor *Y, sptSparseTensor *X, size_t const nthreads);

/**
 * OMP Lock functions
 */
sptMutexPool * sptMutexAlloc();
sptMutexPool * SptMutexAllocCustom(
    sptIndex const num_locks,
    sptIndex const pad_size);
void sptMutexFree(sptMutexPool * pool);

static inline sptIndex sptMutexTranslateId(
    sptIndex const id,
    sptIndex const num_locks,
    sptIndex const pad_size)
{
  return (id % num_locks) * pad_size;
}

static inline void sptMutexSetLock(
    sptMutexPool * const pool,
    sptIndex const id)
{
  sptIndex const lock_id = sptMutexTranslateId(id, pool->nlocks, pool->padsize);
  omp_set_lock(pool->locks + lock_id);
}

static inline void sptMutexUnsetLock(
    sptMutexPool * const pool,
    sptIndex const id)
{
  sptIndex const lock_id = sptMutexTranslateId(id, pool->nlocks, pool->padsize);
  omp_unset_lock(pool->locks + lock_id);
}

#ifdef __cplusplus
}
#endif

#endif
