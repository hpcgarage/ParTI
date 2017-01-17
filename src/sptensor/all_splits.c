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

#include <stdlib.h>
#include <ParTI.h>
#include "sptensor.h"
#include "../error/error.h"

static size_t spt_CalcMaxSplitCount(const sptSparseTensor *tsr, const size_t cuts_by_mode[]) {
    size_t prod = 1;
    size_t m;
    for(m = 0; m < tsr->nmodes; ++m) {
        prod *= cuts_by_mode[m];
    }
    return prod;
}

/**
 * A convenient function to get all splits by repeatively calling `spt_SplitSparseTensor`
 *
 * @param[out] splits        Place to store all splits
 * @param[out] nsplits       Place to store the number of actual splits
 * @param[in]  tsr           The tensor to split
 * @param[in]  cuts_by_mode  The number of cuts at each mode, length `tsr->nmodes`
 */
int spt_SparseTensorGetAllSplits(sptSparseTensor **splits, size_t *nsplits, const sptSparseTensor *tsr, const size_t cuts_by_mode[]) {
    int result = 0;
    size_t nsplits_bak = 0;
    spt_SplitHandle split_handle;
    if(nsplits == NULL) {
        nsplits = &nsplits_bak;
    }
    *(sptSparseTensor **) splits = calloc(spt_CalcMaxSplitCount(tsr, cuts_by_mode), sizeof (*splits)[0]);
    result = spt_StartSplitSparseTensor(&split_handle, tsr, cuts_by_mode);
    spt_CheckError(result, "SpTns AllSplt", NULL);
    for(;;) {
        result = spt_SplitSparseTensor(&(*splits)[*nsplits], split_handle);
        if(result == SPTERR_NO_MORE) {
            break;
        }
        spt_CheckError(result, "SpTns AllSplt", NULL);
        ++*nsplits;
    }
    spt_FinishSplitSparseTensor(split_handle);
    return 0;
}

/**
 * Free all sub-tensors created by `spt_SparseTensorGetAllSplits`
 */
void spt_SparseTensorFreeAllSplits(sptSparseTensor splits[], size_t nsplits) {
    size_t i;
    for(i = 0; i < nsplits; ++i) {
        sptFreeSparseTensor(&splits[i]);
    }
    free(splits);
}
