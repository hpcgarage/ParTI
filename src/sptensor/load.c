#include <SpTOL.h>
#include "sptensor.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

int sptLoadSparseTensor(sptSparseTensor *tsr, size_t start_index, FILE *fp) {
    int iores, retval;
    size_t mode;
    iores = fscanf(fp, "%zu", &tsr->nmodes);
    spt_CheckOSError(iores < 0, "spTensor Load");
    tsr->ndims = malloc(tsr->nmodes * sizeof *tsr->ndims);
    spt_CheckOSError(!tsr->ndims, "spTensor Load");
    for(mode = 0; mode < tsr->nmodes; ++mode) {
        iores = fscanf(fp, "%zu", &tsr->ndims[mode]);
        spt_CheckOSError(iores != 1, "spTensor Load");
    }
    tsr->nnz = 0;
    tsr->inds = malloc(tsr->nmodes * sizeof *tsr->inds);
    spt_CheckOSError(!tsr->inds, "spTensor Load");
    for(mode = 0; mode < tsr->nmodes; ++mode) {
        retval = sptNewSizeVector(&tsr->inds[mode], 0, 0);
        spt_CheckError(retval, "spTensor Load", NULL);
    }
    retval = sptNewVector(&tsr->values, 0, 0);
    spt_CheckError(retval, "spTensor Load", NULL);
    while(retval == 0) {
        double value;
        for(mode = 0; mode < tsr->nmodes; ++mode) {
            size_t index;
            iores = fscanf(fp, "%zu", &index);
            spt_CheckOSError(iores != 1, "spTensor Load");
            assert(index >= start_index);
            sptAppendSizeVector(&tsr->inds[mode], index-start_index);
        }
        if(retval == 0) {
            iores = fscanf(fp, "%lf", &value);
            spt_CheckOSError(iores != 1, "spTensor Load");
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
