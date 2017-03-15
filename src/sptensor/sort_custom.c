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

#include <ParTI.h>
#include "sptensor.h"

void sptSparseTensorSortIndexCustomOrder(sptSparseTensor *tsr, const size_t keymodes[]) {
    size_t nmodes = tsr->nmodes;
    size_t m;
    sptSparseTensor tsr_temp;
    tsr_temp.nmodes = nmodes;
    tsr_temp.sortkey = (size_t) -1;
    tsr_temp.ndims = malloc(nmodes * sizeof tsr_temp.ndims[0]);
    tsr_temp.nnz = tsr->nnz;
    tsr_temp.inds = malloc(nmodes * sizeof tsr_temp.inds[0]);
    tsr_temp.values = tsr->values;

    for(m = 0; m < nmodes; ++m) {
        tsr_temp.ndims[m] = tsr->ndims[keymodes[m]];
        tsr_temp.inds[m] = tsr->inds[keymodes[m]];
    }

    sptSparseTensorSortIndex(&tsr_temp);

    free(tsr_temp.inds);
    free(tsr_temp.ndims);
}
