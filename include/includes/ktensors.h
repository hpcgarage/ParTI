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

#ifndef PARTI_KTENSORS_H
#define PARTI_KTENSORS_H

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

#endif