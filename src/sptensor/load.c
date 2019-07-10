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
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * Load the contents of a sparse tensor fro a text file
 * @param tsr         th sparse tensor to store into
 * @param start_index the index of the first element in array. Set to 1 for MATLAB compability, else set to 0
 * @param fp          the file to read from
 */
int sptLoadSparseTensor(sptSparseTensor *tsr, sptIndex start_index, FILE *fp) {
    int iores, retval;
    sptIndex mode;
    iores = fscanf(fp, "%u", &tsr->nmodes);
    spt_CheckOSError(iores < 0, "SpTns Load");
    /* Only allocate space for sortorder, initialized to 0s. */
    tsr->sortorder = malloc(tsr->nmodes * sizeof tsr->sortorder[0]);
    spt_CheckOSError(!tsr->sortorder, "SpTns Load");
    memset(tsr->sortorder, 0, tsr->nmodes * sizeof tsr->sortorder[0]);
    tsr->ndims = malloc(tsr->nmodes * sizeof *tsr->ndims);
    spt_CheckOSError(!tsr->ndims, "SpTns Load");
    for(mode = 0; mode < tsr->nmodes; ++mode) {
        iores = fscanf(fp, "%u", &tsr->ndims[mode]);
        spt_CheckOSError(iores != 1, "SpTns Load");
    }
    tsr->nnz = 0;
    tsr->inds = malloc(tsr->nmodes * sizeof *tsr->inds);
    spt_CheckOSError(!tsr->inds, "SpTns Load");
    for(mode = 0; mode < tsr->nmodes; ++mode) {
        retval = sptNewIndexVector(&tsr->inds[mode], 0, 0);
        spt_CheckError(retval, "SpTns Load", NULL);
    }
    retval = sptNewValueVector(&tsr->values, 0, 0);
    spt_CheckError(retval, "SpTns Load", NULL);
    while(retval == 0) {
        double value;
        for(mode = 0; mode < tsr->nmodes; ++mode) {
            sptIndex index;
            iores = fscanf(fp, "%u", &index);
            if(iores != 1) {
                retval = -1;
                break;
            }
            if(index < start_index) {
                spt_CheckError(SPTERR_VALUE_ERROR, "SpTns Load", "index < start_index");
            }
            sptAppendIndexVector(&tsr->inds[mode], index-start_index);
        }
        if(retval == 0) {
            iores = fscanf(fp, "%lf", &value);
            if(iores != 1) {
                retval = -1;
                break;
            }
            sptAppendValueVector(&tsr->values, value);
            ++tsr->nnz;
        }
    }
    for(mode = 0; mode < tsr->nmodes; ++mode) {
        tsr->inds[mode].len = tsr->nnz;
    }
    spt_SparseTensorCollectZeros(tsr);
    
    return 0;
}

void sptLoadShuffleFile(sptSparseTensor *tsr, FILE *fs, sptIndex ** map_inds)
{
    sptNnzIndex line_count = 0;
    sptNnzIndex dim_count = 0;
    int iores;
    for(sptIndex mode = 0; mode < tsr->nmodes; ++mode) {
        dim_count += tsr->ndims[mode];
        for(sptIndex i = 0; i < tsr->ndims[mode]; ++i) {
            iores = fscanf(fs, "%u", &(map_inds[mode][i]));
            if(iores != 1) {
                return;
            }
            -- map_inds[mode][i];
            ++ line_count;
        }
    }
    sptAssert(dim_count == line_count);    
    
    return;
}