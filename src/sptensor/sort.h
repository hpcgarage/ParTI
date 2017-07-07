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
void spt_SwapValues(sptSparseTensor *tsr, size_t ind1, size_t ind2);
/* TODO: change the below two function, remove "_" */
int spt_SparseTensorCompareIndices(const sptSparseTensor *tsr1, size_t loc1, const sptSparseTensor *tsr2, size_t loc2);
int spt_SparseTensorCompareIndicesRange(const sptSparseTensor *tsr, size_t loc, size_t * const inds1, size_t * const inds2);

#ifdef __cplusplus
}
#endif

#endif