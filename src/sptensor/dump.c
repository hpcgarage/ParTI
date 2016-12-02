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
#include <stdio.h>
#include "sptensor.h"

/**
 * Save the contents of a sparse tensor into a text file
 * @param tsr         th sparse tensor used to write
 * @param start_index the index of the first element in array. Set to 1 for MATLAB compability, else set to 0
 * @param fp          the file to write into
 */
int sptDumpSparseTensor(const sptSparseTensor *tsr, size_t start_index, FILE *fp) {
    int iores;
    size_t mode, i;
    iores = fprintf(fp, "%zu\n", tsr->nmodes);
    spt_CheckOSError(iores < 0, "SpTns Dump");
    for(mode = 0; mode < tsr->nmodes; ++mode) {
        if(mode != 0) {
            iores = fputs(" ", fp);
            spt_CheckOSError(iores < 0, "SpTns Dump");
        }
        iores = fprintf(fp, "%zu", tsr->ndims[mode]);
        spt_CheckOSError(iores < 0, "SpTns Dump");
    }
    fputs("\n", fp);
    for(i = 0; i < tsr->nnz; ++i) {
        for(mode = 0; mode < tsr->nmodes; ++mode) {
            iores = fprintf(fp, "%zu\t", tsr->inds[mode].data[i]+start_index);
            spt_CheckOSError(iores < 0, "SpTns Dump");
        }
        iores = fprintf(fp, "%lg\n", (double) tsr->values.data[i]);
        spt_CheckOSError(iores < 0, "SpTns Dump");
    }
    return 0;
}
