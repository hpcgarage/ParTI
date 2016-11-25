/*
    This file is part of SpTOL.

    SpTOL is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as
    published by the Free Software Foundation, either version 3 of
    the License, or (at your option) any later version.

    SpTOL is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with SpTOL.
    If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef SPTOL_SPTENSOR_H
#define SPTOL_SPTENSOR_H

#ifdef __cplusplus
extern "C" {
#endif

#include <SpTOL.h>
#include "../error/error.h"

int spt_SparseTensorCompareIndices(const sptSparseTensor *tsr1, size_t ind1, const sptSparseTensor *tsr2, size_t ind2);

void spt_SparseTensorCollectZeros(sptSparseTensor *tsr);

int spt_DistSparseTensor(sptSparseTensor * tsr,
    int const nthreads,
    size_t * const dist_nnzs,
    size_t * dist_nrows);

int spt_DistSparseTensorFixed(sptSparseTensor * tsr,
    int const nthreads,
    size_t * const dist_nnzs,
    size_t * dist_nrows);
    
int spt_SliceSparseTensor(sptSparseTensor *dest, const sptSparseTensor *tsr, const size_t limit_low[], const size_t limit_high[]);

#ifdef __cplusplus
}
#endif

#endif
