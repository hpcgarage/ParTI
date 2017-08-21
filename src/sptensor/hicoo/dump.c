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
#include "hicoo.h"

/**
 * Save the contents of a HiCOO sparse tensor into a text file
 * @param hitsr         th sparse tensor used to write
 * @param start_index the index of the first element in array. Set to 1 for MATLAB compability, else set to 0
 * @param fp          the file to write into
 */
int sptDumpSparseTensorHiCOO(sptSparseTensorHiCOO * const hitsr, FILE *fp) 
{
    int iores;
    sptIndex mode;
    sptIndex sk = (sptIndex)pow(2, hitsr->sk_bits);

    iores = fprintf(fp, "%u\n", hitsr->nmodes);
    spt_CheckOSError(iores < 0, "SpTns Dump");
    for(mode = 0; mode < hitsr->nmodes; ++mode) {
        if(mode != 0) {
            iores = fputs(" ", fp);
            spt_CheckOSError(iores < 0, "SpTns Dump");
        }
        iores = fprintf(fp, "%u", hitsr->ndims[mode]);
        spt_CheckOSError(iores < 0, "SpTns Dump");
    }
    fputs("\n", fp);
    fprintf(fp, "nkiters:\n");
    sptDumpIndexArray(hitsr->nkiters, hitsr->nmodes, fp);
    fprintf(fp, "kschr:\n");
    for(mode = 0; mode < hitsr->nmodes; ++mode) {
        fprintf(fp, "mode %u\n", mode);
        for(sptIndex i=0; i<(hitsr->ndims[mode] + sk - 1)/sk; ++i) {
            sptDumpIndexVector(&hitsr->kschr[mode][i], fp);
        }
    }
    fprintf(fp, "kptr:\n");
    sptDumpNnzIndexVector(&hitsr->kptr, fp);
    fprintf(fp, "cptr:\n");
    sptDumpNnzIndexVector(&hitsr->cptr, fp);
    fprintf(fp, "bptr:\n");
    sptDumpNnzIndexVector(&hitsr->bptr, fp);
    fprintf(fp, "binds:\n");
    for(mode = 0; mode < hitsr->nmodes; ++mode) {
        sptDumpBlockIndexVector(&hitsr->binds[mode], fp);
    }
    fprintf(fp, "einds:\n");
    for(mode = 0; mode < hitsr->nmodes; ++mode) {
        sptDumpElementIndexVector(&hitsr->einds[mode], fp);
    }
    fprintf(fp, "values:\n");
    sptDumpValueIndexVector(&hitsr->values, fp);

    return 0;
}
