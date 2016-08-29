#include <SpTOL.h>
#include "sptensor.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

int sptLoadSparseTensor(sptSparseTensor *tsr, size_t start_index, FILE *fp) {
    int iores, retval;
    size_t mode;
    iores = fscanf(fp, "%zu", &tsr->nmodes);
    if(iores != 1) {
        return -1;
    }
    tsr->ndims = malloc(tsr->nmodes * sizeof *tsr->ndims);
    if(!tsr->ndims) {
        return -1;
    }
    for(mode = 0; mode < tsr->nmodes; ++mode) {
        iores = fscanf(fp, "%zu", &tsr->ndims[mode]);
        if(iores != 1) {
            return -1;
        }
    }
    tsr->nnz = 0;
    tsr->inds = malloc(tsr->nmodes * sizeof *tsr->inds);
    if(!tsr->inds) {
        return -1;
    }
    for(mode = 0; mode < tsr->nmodes; ++mode) {
        retval = sptNewSizeVector(&tsr->inds[mode], 0, 0);
        if(retval) {
            return retval;
        }
    }
    retval = sptNewVector(&tsr->values, 0, 0);
    if(retval) {
        return retval;
    }
    while(retval == 0) {
        double value;
        for(mode = 0; mode < tsr->nmodes; ++mode) {
            size_t index;
            iores = fscanf(fp, "%zu", &index);
            if(iores != 1) {
                retval = -1;
                break;
            }
            assert(index >= start_index);
            sptAppendSizeVector(&tsr->inds[mode], index-start_index);
        }
        if(retval == 0) {
            iores = fscanf(fp, "%lf", &value);
            if(iores != 1) {
                retval = -1;
                break;
            }
            sptAppendVector(&tsr->values, value);
            ++tsr->nnz;
        }
    }
    for(mode = 0; mode < tsr->nmodes; ++mode) {
        tsr->inds[mode].len = tsr->nnz;
    }
    spt_SparseTensorCollectZeros(tsr);
    sptSparseTensorSortIndex(tsr);
    return 0;
}
