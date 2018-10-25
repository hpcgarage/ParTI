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

#ifndef PARTI_STRUCTS_H
#define PARTI_STRUCTS_H



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
 * Sparse matrix type, CSR format
 */
typedef struct {
    sptIndex nrows;  /// # rows
    sptIndex ncols;  /// # colums
    sptNnzIndex nnz;    /// # non-zeros
    sptNnzIndexVector rowptr; /// row indices, length nnz
    sptIndexVector colind; /// column indices, length nnz
    sptValueVector values; /// non-zero values, length nnz
} sptSparseMatrixCSR;


/**
 * Sparse tensor type, COO format
 */
typedef struct {
    sptIndex nmodes;      /// # modes
    sptIndex * sortorder;  /// the order in which the indices are sorted
    sptIndex * ndims;      /// size of each mode, length nmodes
    sptNnzIndex nnz;         /// # non-zeros
    sptIndexVector * inds;       /// indices of each element, length [nmodes][nnz]
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
 * Semi-sparse tensor type
 * The chosen mode is dense, while other modes are sparse.
 * Can be considered as "sparse tensor of dense fiber".
 * The "fiber" here can be defined as a vector of elements that have indices
 * only different in the last mode.
 */
typedef struct {
    sptIndex nmodes; /// # Modes, must >= 2
    sptIndex *ndims; /// size of each mode, length nmodes
    sptIndex mode;   /// the mode where data is stored in dense format
    sptNnzIndex nnz;    /// # non-zero fibers
    sptIndexVector *inds;  /// indices of each dense fiber, length [nmodes][nnz], the mode-th value is ignored
    sptIndex stride; /// ndims[mode] rounded up to 8
    sptMatrix     values; /// dense fibers, size nnz*ndims[mode]
} sptSemiSparseTensor;


/**
 * General Semi-sparse tensor type
 */
typedef struct {
    sptIndex nmodes; /// # Modes, must >= 2
    sptIndex *ndims; /// size of each mode, length nmodes
    sptIndex ndmodes;
    sptIndex *dmodes;   /// the mode where data is stored in dense format, allocate nmodes sized space
    sptNnzIndex nnz;    /// # non-zero fibers
    sptIndexVector *inds;  /// indices of each dense fiber, length [nmodes][nnz], the mode-th value is ignored
    sptIndex *strides; /// ndims[mode] rounded up to 8
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
 * Key-value pair structure
 */
typedef struct 
{
  sptIndex key;
  sptIndex value;
} sptKeyValuePair;

#ifdef PARTI_USE_OPENMP
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
#endif

#endif