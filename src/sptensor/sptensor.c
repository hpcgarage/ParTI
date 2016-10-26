#include <SpTOL.h>
#include <stdlib.h>
#include <string.h>

int sptNewSparseTensor(sptSparseTensor *tsr, size_t nmodes, const size_t ndims[]) {
    size_t i;
    int result;
    tsr->nmodes = nmodes;
    tsr->ndims = malloc(nmodes * sizeof *tsr->ndims);
    spt_CheckOSError(!tsr->ndims, "SpTns New");
    memcpy(tsr->ndims, ndims, nmodes * sizeof *tsr->ndims);
    tsr->nnz = 0;
    tsr->inds = malloc(nmodes * sizeof *tsr->inds);
    spt_CheckOSError(!tsr->inds, "SpTns New");
    for(i = 0; i < nmodes; ++i) {
        result = sptNewSizeVector(&tsr->inds[i], 0, 0);
        spt_CheckError(result, "SpTns New", NULL);
    }
    result = sptNewVector(&tsr->values, 0, 0);
    spt_CheckError(result, "SpTns New", NULL);
    return 0;
}

int sptCopySparseTensor(sptSparseTensor *dest, const sptSparseTensor *src) {
    size_t i;
    int result;
    dest->nmodes = src->nmodes;
    dest->ndims = malloc(dest->nmodes * sizeof *dest->ndims);
    spt_CheckOSError(!tsr->ndims, "SpTns Copy");
    memcpy(dest->ndims, src->ndims, src->nmodes * sizeof *src->ndims);
    dest->nnz = src->nnz;
    dest->inds = malloc(dest->nmodes * sizeof *dest->inds);
    spt_CheckOSError(!tsr->inds, "SpTns Copy");
    for(i = 0; i < dest->nmodes; ++i) {
        result = sptCopySizeVector(&dest->inds[i], &src->inds[i]);
        spt_CheckError(result, "SpTns Copy", NULL);
    }
    result = sptCopyVector(&dest->values, &src->values);
    spt_CheckError(result, "SpTns Copy", NULL);
    return 0;
}

void sptFreeSparseTensor(sptSparseTensor *tsr) {
    size_t i;
    for(i = 0; i < tsr->nmodes; ++i) {
        sptFreeSizeVector(&tsr->inds[i]);
    }
    free(tsr->ndims);
    free(tsr->inds);
    sptFreeVector(&tsr->values);
}


void sptDistSparseTensor(sptSparseTensor * tsr, 
    int const nthreads,
    size_t * const dist_nnzs, 
    size_t * dist_nrows) {

    size_t global_nnz = tsr->nnz;
    size_t aver_nnz = global_nnz / nthreads;
    memset(dist_nnzs, 0, nthreads*sizeof(size_t));
    memset(dist_nrows, 0, nthreads*sizeof(size_t));

    sptSparseTensorSortIndex(tsr);
    size_t * ind0 = tsr->inds[0].data;

    int ti = 0;
    dist_nnzs[0] = 1;
    dist_nrows[0] = 1;
    for(size_t x=1; x<global_nnz; ++x) {
        if(ind0[x] == ind0[x-1]) {
            ++ dist_nnzs[ti];
        } else if (ind0[x] > ind0[x-1]) {
            if(dist_nnzs[ti] < aver_nnz || ti == nthreads-1) {
                ++ dist_nnzs[ti];
                ++ dist_nrows[ti];
            } else {
                ++ ti;
                ++ dist_nnzs[ti];
                ++ dist_nrows[ti];
            }
        } else {
            spt_CheckError(SPTERR_VALUE_ERROR, "SpTns Dist", "tensor unsorted on mode-0");
        }
    }

}


void sptDistSparseTensorFixed(sptSparseTensor * tsr, 
    int const nthreads,
    size_t * const dist_nnzs, 
    size_t * dist_nrows) {

    size_t global_nnz = tsr->nnz;
    size_t aver_nnz = global_nnz / nthreads;
    memset(dist_nnzs, 0, nthreads*sizeof(size_t));

    sptSparseTensorSortIndex(tsr);
    size_t * ind0 = tsr->inds[0].data;

    int ti = 0;
    dist_nnzs[0] = 1;
    for(size_t x=1; x<global_nnz; ++x) {
        if(ind0[x] == ind0[x-1]) {
            ++ dist_nnzs[ti];
        } else if (ind0[x] > ind0[x-1]) {
            if(dist_nnzs[ti] < aver_nnz || ti == nthreads-1) {
                ++ dist_nnzs[ti];
            } else {
                ++ ti;
                ++ dist_nnzs[ti];
            }
        } else {
            spt_CheckError(SPTERR_VALUE_ERROR, "SpTns Dist", "tensor unsorted on mode-0");
        }
    }

}
