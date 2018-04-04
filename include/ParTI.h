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
#ifdef PARTI_USE_MPI
    #include <mpi.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif


/**
 * Define types, TODO: check the bit size of them, add branch for different settings
 */
#define PARTI_INDEX_TYPEWIDTH 32
#define PARTI_VALUE_TYPEWIDTH 32
#define PARTI_ELEMENT_INDEX_TYPEWIDTH 8

#if PARTI_INDEX_TYPEWIDTH == 32
  typedef uint32_t sptIndex;
  typedef uint32_t sptBlockIndex;
  #define PARTI_PRI_INDEX PRIu32
  #define PARTI_SCN_INDEX SCNu32
  #define PARTI_PRI_BLOCK_INDEX PRIu32
  #define PARTI_SCN_BLOCK_INDEX SCNu32
#elif PARTI_INDEX_TYPEWIDTH == 64
  typedef uint64_t sptIndex;
  typedef uint64_t sptBlockIndex;
  #define PARTI_PFI_INDEX PRIu64
  #define PARTI_SCN_INDEX SCNu64
  #define PARTI_PRI_BLOCK_INDEX PRIu64
  #define PARTI_SCN_BLOCK_INDEX SCNu64
#else
  #error "Unrecognized PARTI_INDEX_TYPEWIDTH."
#endif

#if PARTI_VALUE_TYPEWIDTH == 32
  typedef float sptValue;
  #define PARTI_PRI_VALUE "f"
  #define PARTI_SCN_VALUE "f"
#elif PARTI_VALUE_TYPEWIDTH == 64
  typedef double sptValue;
  #define PARTI_PRI_VALUE "lf"
  #define PARTI_SCN_VALUE "lf"
#else
  #error "Unrecognized PARTI_VALUE_TYPEWIDTH."
#endif

#if PARTI_ELEMENT_INDEX_TYPEWIDTH == 8
  typedef uint_fast8_t sptElementIndex;
  typedef uint_fast16_t sptBlockMatrixIndex;  // R < 256
  #define PARTI_PRI_ELEMENT_INDEX PRIuFAST8
  #define PARTI_SCN_ELEMENT_INDEX SCNuFAST8
  #define PARTI_PRI_BLOCKMATRIX_INDEX PRIuFAST16
  #define PARTI_SCN_BLOCKMATRIX_INDEX SCNuFAST16
  // typedef uint8_t sptElementIndex;
  // typedef uint16_t sptBlockMatrixIndex;  // R < 256
  // #define PARTI_PRI_ELEMENT_INDEX PRIu8
  // #define PARTI_SCN_ELEMENT_INDEX SCNu8
  // #define PARTI_PRI_BLOCKMATRIX_INDEX PRIu16
  // #define PARTI_SCN_BLOCKMATRIX_INDEX SCNu16
#elif PARTI_ELEMENT_INDEX_TYPEWIDTH == 16
  typedef uint16_t sptElementIndex;
  typedef uint32_t sptBlockMatrixIndex;
  #define PARTI_PFI_ELEMENT_INDEX PRIu16
  #define PARTI_SCN_ELEMENT_INDEX SCNu16
  #define PARTI_PRI_BLOCKMATRIX_INDEX PRIu32
  #define PARTI_SCN_BLOCKMATRIX_INDEX SCNu32
#elif PARTI_ELEMENT_INDEX_TYPEWIDTH == 32
  typedef uint32_t sptElementIndex;
  typedef uint32_t sptBlockMatrixIndex;
  #define PARTI_PFI_ELEMENT_INDEX PRIu32
  #define PARTI_SCN_ELEMENT_INDEX SCNu32
  #define PARTI_PRI_BLOCKMATRIX_INDEX PRIu32
  #define PARTI_SCN_BLOCKMATRIX_INDEX SCNu32
#else
  #error "Unrecognized PARTI_ELEMENT_INDEX_TYPEWIDTH."
#endif

typedef sptBlockIndex sptBlockNnzIndex;
#define PARTI_PRI_BLOCKNNZ_INDEX PARTI_PRI_BLOCK_INDEX
#define PARTI_SCN_BLOCKNNZ_INDEX PARTI_SCN_BLOCK_INDEX

typedef uint64_t sptNnzIndex;
#define PARTI_PRI_NNZ_INDEX PRIu64
#define PARTI_SCN_NNZ_INDEX PRIu64

typedef unsigned __int128 sptMortonIndex;
// typedef __uint128_t sptMortonIndex;


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
    sptIndex *data; /// data
} sptIndexVector;

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
    sptNnzIndex *data; /// data
} sptNnzIndexVector;


/**
 * Dense matrix type
 */
typedef struct {
    sptIndex nrows;   /// # rows
    sptIndex ncols;   /// # columns
    sptIndex cap;     /// # of allocated rows
    sptIndex stride;  /// ncols rounded up to 8
    sptValue *values; /// values, length cap*stride
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
    sptIndex nrows;  /// # rows
    sptIndex ncols;  /// # colums
    sptNnzIndex nnz;    /// # non-zeros
    sptIndexVector rowind; /// row indices, length nnz
    sptIndexVector colind; /// column indices, length nnz
    sptValueVector values; /// non-zero values, length nnz
} sptSparseMatrix;


/**
 * Sparse tensor type, COO format
 */
typedef struct {
    sptIndex nmodes;      /// # modes
    sptIndex *sortorder;  /// the order in which the indices are sorted
    sptIndex *ndims;      /// size of each mode, length nmodes
    sptNnzIndex nnz;         /// # non-zeros
    sptIndexVector *inds;       /// indices of each element, length [nmodes][nnz]
    sptValueVector values;      /// non-zero values, length nnz
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
 * Key-value pair structure
 */
typedef struct 
{
  sptIndex key;
  sptIndex value;
} sptKeyValuePair;

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


/* Timer functions, using either CPU or GPU timer */
int sptNewTimer(sptTimer *timer, int use_cuda);
int sptStartTimer(sptTimer timer);
int sptStopTimer(sptTimer timer);
double sptElapsedTime(const sptTimer timer);
double sptPrintElapsedTime(const sptTimer timer, const char *name);
double sptPrintAverageElapsedTime(const sptTimer timer, const int niters, const char *name);
int sptFreeTimer(sptTimer timer);

/* Base functions */
char * sptBytesString(uint64_t const bytes);
sptValue sptRandomValue(void);

/* Dense Array functions */
sptNnzIndex sptMaxNnzIndexArray(sptNnzIndex const * const indices, sptNnzIndex const size);
sptIndex sptMaxIndexArray(sptIndex const * const indices, sptNnzIndex const size);
void sptPairArraySort(sptKeyValuePair const * kvarray, sptIndex const length);
int sptDumpIndexArray(sptIndex *array, sptNnzIndex const n, FILE *fp);
int sptDumpNnzIndexArray(sptNnzIndex *array, sptNnzIndex const n, FILE *fp);

/* Dense vector, with sptValueVector type */
int sptNewValueVector(sptValueVector *vec, uint64_t len, uint64_t cap);
int sptConstantValueVector(sptValueVector * const vec, sptValue const val);
int sptCopyValueVector(sptValueVector *dest, const sptValueVector *src);
int sptAppendValueVector(sptValueVector *vec, sptValue const value);
int sptAppendValueVectorWithVector(sptValueVector *vec, const sptValueVector *append_vec);
int sptResizeValueVector(sptValueVector *vec, sptNnzIndex const size);
void sptFreeValueVector(sptValueVector *vec);
int sptDumpValueIndexVector(sptValueVector *vec, FILE *fp);

/* Dense vector, with sptIndexVector type */
int sptNewIndexVector(sptIndexVector *vec, uint64_t len, uint64_t cap);
int sptConstantIndexVector(sptIndexVector * const vec, sptIndex const num);
int sptCopyIndexVector(sptIndexVector *dest, const sptIndexVector *src);
int sptAppendIndexVector(sptIndexVector *vec, sptIndex const value);
int sptAppendIndexVectorWithVector(sptIndexVector *vec, const sptIndexVector *append_vec);
int sptResizeIndexVector(sptIndexVector *vec, sptNnzIndex const size);
void sptFreeIndexVector(sptIndexVector *vec);
int sptDumpIndexVector(sptIndexVector *vec, FILE *fp);

/* Dense vector, with sptElementIndexVector type */
int sptNewElementIndexVector(sptElementIndexVector *vec, uint64_t len, uint64_t cap);
int sptConstantElementIndexVector(sptElementIndexVector * const vec, sptElementIndex const num);
int sptCopyElementIndexVector(sptElementIndexVector *dest, const sptElementIndexVector *src);
int sptAppendElementIndexVector(sptElementIndexVector *vec, sptElementIndex const value);
int sptAppendElementIndexVectorWithVector(sptElementIndexVector *vec, const sptElementIndexVector *append_vec);
int sptResizeElementIndexVector(sptElementIndexVector *vec, sptNnzIndex const size);
void sptFreeElementIndexVector(sptElementIndexVector *vec);
int sptDumpElementIndexVector(sptElementIndexVector *vec, FILE *fp);

/* Dense vector, with sptBlockIndexVector type */
int sptNewBlockIndexVector(sptBlockIndexVector *vec, uint64_t len, uint64_t cap);
int sptConstantBlockIndexVector(sptBlockIndexVector * const vec, sptBlockIndex const num);
int sptCopyBlockIndexVector(sptBlockIndexVector *dest, const sptBlockIndexVector *src);
int sptAppendBlockIndexVector(sptBlockIndexVector *vec, sptBlockIndex const value);
int sptAppendBlockIndexVectorWithVector(sptBlockIndexVector *vec, const sptBlockIndexVector *append_vec);
int sptResizeBlockIndexVector(sptBlockIndexVector *vec, sptNnzIndex const size);
void sptFreeBlockIndexVector(sptBlockIndexVector *vec);
int sptDumpBlockIndexVector(sptBlockIndexVector *vec, FILE *fp);

/* Dense vector, with sptNnzIndexVector type */
int sptNewNnzIndexVector(sptNnzIndexVector *vec, uint64_t len, uint64_t cap);
int sptConstantNnzIndexVector(sptNnzIndexVector * const vec, sptNnzIndex const num);
int sptCopyNnzIndexVector(sptNnzIndexVector *dest, const sptNnzIndexVector *src);
int sptAppendNnzIndexVector(sptNnzIndexVector *vec, sptNnzIndex const value);
int sptAppendNnzIndexVectorWithVector(sptNnzIndexVector *vec, const sptNnzIndexVector *append_vec);
int sptResizeNnzIndexVector(sptNnzIndexVector *vec, sptNnzIndex const size);
void sptFreeNnzIndexVector(sptNnzIndexVector *vec);
int sptDumpNnzIndexVector(sptNnzIndexVector *vec, FILE *fp);


/* Dense matrix */
static inline sptNnzIndex sptGetMatrixLength(const sptMatrix *mtx) {
    return mtx->nrows * mtx->stride;
}
int sptNewMatrix(sptMatrix *mtx, sptIndex const nrows, sptIndex const ncols);
int sptRandomizeMatrix(sptMatrix *mtx, sptIndex const nrows, sptIndex const ncols);
int sptIdentityMatrix(sptMatrix *mtx);
int sptConstantMatrix(sptMatrix * const mtx, sptValue const val);
int sptCopyMatrix(sptMatrix *dest, const sptMatrix *src);
int sptAppendMatrix(sptMatrix *mtx, const sptValue values[]);
int sptResizeMatrix(sptMatrix *mtx, sptIndex const new_nrows);
void sptFreeMatrix(sptMatrix *mtx);
int sptDumpMatrix(sptMatrix *mtx, FILE *fp);

/* Dense matrix operations */
int sptMatrixDotMul(sptMatrix const * A, sptMatrix const * B, sptMatrix const * C);
int sptMatrixDotMulSeq(sptIndex const mode, sptIndex const nmodes, sptMatrix ** mats);
int sptMatrixDotMulSeqCol(sptIndex const mode, sptIndex const nmodes, sptMatrix ** mats);
int sptMatrix2Norm(sptMatrix * const A, sptValue * const lambda);
int sptMatrixMaxNorm(sptMatrix * const A, sptValue * const lambda);
void GetFinalLambda(
  sptIndex const rank,
  sptIndex const nmodes,
  sptMatrix ** mats,
  sptValue * const lambda);
int sptMatrixSolveNormals(
  sptIndex const mode,
  sptIndex const nmodes,
  sptMatrix ** aTa,
  sptMatrix * rhs);
int sptSparseTensorToMatrix(sptMatrix *dest, const sptSparseTensor *src);

/* Dense Rank matrix, ncols = small rank (<= 256) */
int sptNewRankMatrix(sptRankMatrix *mtx, sptIndex const nrows, sptElementIndex const ncols);
int sptRandomizeRankMatrix(sptRankMatrix *mtx, sptIndex const nrows, sptElementIndex const ncols);
int sptConstantRankMatrix(sptRankMatrix *mtx, sptValue const val);
void sptFreeRankMatrix(sptRankMatrix *mtx);
int sptDumpRankMatrix(sptRankMatrix *mtx, FILE *fp);

/* Dense rank matrix operations */
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


/* Sparse tensor */
int sptNewSparseTensor(sptSparseTensor *tsr, sptIndex nmodes, const sptIndex ndims[]);
int sptCopySparseTensor(sptSparseTensor *dest, const sptSparseTensor *src);
void sptFreeSparseTensor(sptSparseTensor *tsr);
double SparseTensorFrobeniusNormSquared(sptSparseTensor const * const spten);
int sptLoadSparseTensor(sptSparseTensor *tsr, sptIndex start_index, FILE *fp);
int sptDumpSparseTensor(const sptSparseTensor *tsr, sptIndex start_index, FILE *fp);
int sptMatricize(sptSparseTensor const * const X,
    sptIndex const m,
    sptSparseMatrix * const A,
    int const transpose);
void sptGetBestModeOrder(
    sptIndex * mode_order,
    sptIndex const mode,
    sptIndex const * ndims,
    sptIndex const nmodes);
void sptGetWorstModeOrder(
    sptIndex * mode_order,
    sptIndex const mode,
    sptIndex const * ndims,
    sptIndex const nmodes);
void sptGetRandomShuffleElements(sptSparseTensor *tsr);
void sptGetRandomShuffleIndices(sptSparseTensor *tsr, sptIndexVector *map_inds);
void sptSparseTensorSortIndex(sptSparseTensor *tsr, int force);
void sptSparseTensorSortIndexAtMode(sptSparseTensor *tsr, sptIndex const mode, int force);
void sptSparseTensorSortIndexCustomOrder(sptSparseTensor *tsr, sptIndex const *  mode_order, int force);
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
int sptSparseTensorMixedOrder(
    sptSparseTensor *tsr, 
    const sptElementIndex sb_bits,
    const sptElementIndex sk_bits);
void sptSparseTensorCalcIndexBounds(sptIndex inds_low[], sptIndex inds_high[], const sptSparseTensor *tsr);
int spt_ComputeSliceSizes(
    sptNnzIndex * slice_nnzs, 
    sptSparseTensor * const tsr,
    sptIndex const mode);
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
int sptSetKernelPointers(
    sptNnzIndexVector *kptr,
    sptSparseTensor *tsr, 
    const sptElementIndex sk_bits);



/* Kruskal tensor */
int sptNewKruskalTensor(sptKruskalTensor *ktsr, sptIndex nmodes, const sptIndex ndims[], sptIndex rank); 
void sptFreeKruskalTensor(sptKruskalTensor *ktsr);
int sptDumpKruskalTensor(sptKruskalTensor *ktsr, FILE *fp);
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
int sptNewRankKruskalTensor(sptRankKruskalTensor *ktsr, sptIndex nmodes, const sptIndex ndims[], sptElementIndex rank);
void sptFreeRankKruskalTensor(sptRankKruskalTensor *ktsr);
int sptDumpRankKruskalTensor(sptRankKruskalTensor *ktsr, FILE *fp);
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
int sptSparseTensorMulScalar(sptSparseTensor *X, sptValue const a);
int sptSparseTensorDivScalar(sptSparseTensor *X, sptValue const a);
/* Sparse tensor binary operations */
int sptSparseTensorAdd(sptSparseTensor *Z, const sptSparseTensor *X, const sptSparseTensor *Y);
int sptSparseTensorSub(sptSparseTensor *Z, const sptSparseTensor *X, const sptSparseTensor *Y);
int sptSparseTensorDotMul(sptSparseTensor *Z, const sptSparseTensor *X, const sptSparseTensor *Y);
int sptSparseTensorDotMulEq(sptSparseTensor *Z, const sptSparseTensor *X, const sptSparseTensor *Y);
#ifdef PARTI_USE_OPENMP
int sptOmpSparseTensorDotMulEq(sptSparseTensor *Z, const sptSparseTensor *X, const sptSparseTensor *Y);
#endif
int sptSparseTensorDotDiv(sptSparseTensor *Z, const sptSparseTensor *X, const sptSparseTensor *Y);



/**
 * Matricized tensor times Khatri-Rao product.
 */
int sptMTTKRP(
    sptSparseTensor const * const X,
    sptMatrix * mats[],     // mats[nmodes] as temporary space.
    sptIndex const mats_order[],    // Correspond to the mode order of X.
    sptIndex const mode);
#ifdef PARTI_USE_OPENMP
int sptOmpMTTKRP(
    sptSparseTensor const * const X,
    sptMatrix * mats[],     // mats[nmodes] as temporary space.
    sptIndex const mats_order[],    // Correspond to the mode order of X.
    sptIndex const mode,
    const int tk);
int sptOmpMTTKRP_Reduce(sptSparseTensor const * const X,
    sptMatrix * mats[],     // mats[nmodes] as temporary space.
    sptMatrix * copy_mats[],    // temporary matrices for reduction
    sptIndex const mats_order[],    // Correspond to the mode order of X.
    sptIndex const mode,
    const int tk);
int sptOmpMTTKRP_Lock(sptSparseTensor const * const X,
    sptMatrix * mats[],     // mats[nmodes] as temporary space.
    sptIndex const mats_order[],    // Correspond to the mode order of X.
    sptIndex const mode,
    const int tk,
    sptMutexPool * lock_pool);
#endif


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
#ifdef PARTI_USE_OPENMP
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
#endif

/**
 * CP-ALS
 */
int sptCpdAls(
  sptSparseTensor const * const spten,
  sptIndex const rank,
  sptIndex const niters,
  double const tol,
  sptKruskalTensor * ktensor);
#ifdef PARTI_USE_OPENMP
int sptOmpCpdAls(
  sptSparseTensor const * const spten,
  sptIndex const rank,
  sptIndex const niters,
  double const tol,
  const int tk,
  const int use_reduce,
  sptKruskalTensor * ktensor);
#endif
int sptCpdAlsHiCOO(
  sptSparseTensorHiCOO const * const hitsr,
  sptIndex const rank,
  sptIndex const niters,
  double const tol,
  sptRankKruskalTensor * ktensor);
#ifdef PARTI_USE_OPENMP
int sptOmpCpdAlsHiCOO(
  sptSparseTensorHiCOO const * const hitsr,
  sptIndex const rank,
  sptIndex const niters,
  double const tol,
  const int tk,
  const int tb,
  sptRankKruskalTensor * ktensor);
#endif

#ifdef PARTI_USE_OPENMP
/**
 * OMP functions
 */
int sptSparseTensorAddOMP(sptSparseTensor *Y, sptSparseTensor *X, int const nthreads);
int sptSparseTensorSubOMP(sptSparseTensor *Y, sptSparseTensor *X, int const nthreads);

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
#endif

#ifdef __cplusplus
}
#endif

#endif
