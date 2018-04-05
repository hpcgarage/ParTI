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

#ifndef PARTI_SPTENSORS_H
#define PARTI_SPTENSORS_H

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


/* Sparse tensor unary operations */
int sptSparseTensorMulScalar(sptSparseTensor *X, sptValue const a);
int sptSparseTensorDivScalar(sptSparseTensor *X, sptValue const a);

/* Sparse tensor binary operations */
int sptSparseTensorAdd(sptSparseTensor *Z, const sptSparseTensor *X, const sptSparseTensor *Y);
int sptSparseTensorSub(sptSparseTensor *Z, const sptSparseTensor *X, const sptSparseTensor *Y);
#ifdef PARTI_USE_OPENMP
int sptSparseTensorAddOMP(sptSparseTensor *Y, sptSparseTensor *X, int const nthreads);
int sptSparseTensorSubOMP(sptSparseTensor *Y, sptSparseTensor *X, int const nthreads);
#endif
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


#endif