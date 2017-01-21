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

#include <string.h>
#include <ParTI.h>
#include "sptensor.h"
#include "../error/error.h"

int sptSplittedMTTKRP(
    sptSparseTensor const *const X,
    sptMatrix *mats[],
    size_t const mats_order[],
    size_t const mode,
    sptVector *scratch,
    size_t const split_count[]
) {
    int result;
    sptMatrix product;
    result = sptNewMatrix(&product, mats[mode]->nrows, mats[mode]->ncols);
    spt_CheckError(result, "CPU  SpTns SpltMTTKRP", NULL);
    memset(product.values, 0, product.nrows * product.stride * sizeof (sptScalar));

    spt_SplitHandle split_handle;
    result = spt_StartSplitSparseTensor(&split_handle, X, split_count);
    spt_CheckError(result, "CPU  SpTns SpltMTTKRP", NULL);

    for(;;) {
        sptSparseTensor subX;
        result = spt_SplitSparseTensor(&subX, split_handle);
        if(result == SPTERR_NO_MORE) {
            break;
        }
        spt_CheckError(result, "CPU  SpTns SpltMTTKRP", NULL);

        result = sptMTTKRP(&subX, mats, mats_order, mode, scratch);
        spt_CheckError(result, "CPU  SpTns SpltMTTKRP", NULL);

        sptFreeSparseTensor(&subX);

        size_t i;
        for(i = 0; i < product.nrows * product.stride; ++i) {
            product.values[i] += mats[X->nmodes]->values[i];
        }
    }

    spt_FinishSplitSparseTensor(split_handle);

    memcpy(mats[X->nmodes]->values, product.values, product.nrows * product.stride);
    sptFreeMatrix(&product);

    return 0;
}

int sptPresplittedMTTKRP(
    sptSparseTensor const splits[],
    size_t const nsplits,
    sptMatrix *mats[],
    size_t const mats_order[],
    size_t const mode,
    sptVector *scratch
) {
    int result;
    sptMatrix product;
    result = sptNewMatrix(&product, mats[mode]->nrows, mats[mode]->ncols);
    spt_CheckError(result, "CPU  SpTns SpltMTTKRP", NULL);
    memset(product.values, 0, product.nrows * product.stride * sizeof (sptScalar));

    size_t split_id;

    for(split_id = 0; split_id < nsplits; ++split_id) {
        result = sptMTTKRP(&splits[split_id], mats, mats_order, mode, scratch);
        spt_CheckError(result, "CPU  SpTns SpltMTTKRP", NULL);

        size_t i;
        for(i = 0; i < product.nrows * product.stride; ++i) {
            product.values[i] += mats[splits[split_id].nmodes]->values[i];
        }
    }

    memcpy(mats[splits[0].nmodes]->values, product.values, product.nrows * product.stride);
    sptFreeMatrix(&product);

    return 0;
}
