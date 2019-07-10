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

#ifndef PARTI_SPTENSOR_H
#define PARTI_SPTENSOR_H

#ifdef __cplusplus
extern "C" {
#endif

#include <ParTI.h>
#include "../error/error.h"

double spt_SparseTensorNorm(const sptSparseTensor *X);
int spt_SparseTensorCompareIndices(const sptSparseTensor *tsr1, sptNnzIndex ind1, const sptSparseTensor *tsr2, sptNnzIndex ind2);
void spt_SparseTensorCollectZeros(sptSparseTensor *tsr);
int spt_DistSparseTensor(sptSparseTensor * tsr,
    int const nthreads,
    sptNnzIndex * const dist_nnzs,
    sptIndex * dist_nrows);
int spt_DistSparseTensorFixed(sptSparseTensor * tsr,
    int const nthreads,
    sptNnzIndex * const dist_nnzs);
int spt_GetSubSparseTensor(sptSparseTensor *dest, const sptSparseTensor *tsr, const sptIndex limit_low[], const sptIndex limit_high[]);
int spt_SparseTensorCompareIndicesExceptSingleMode(const sptSparseTensor *tsr1, sptNnzIndex loc1, const sptSparseTensor *tsr2, sptNnzIndex loc2, sptIndex * mode_order);


#ifdef __cplusplus
}
#endif

#endif
