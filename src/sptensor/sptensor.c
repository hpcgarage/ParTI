#include <SpTOL.h>
#include <stdlib.h>
#include <string.h>

int sptNewSparseTensor(sptSparseTensor *tsr, size_t nmodes, const size_t ndims[]) {
    size_t i;
    int result;
    tsr->nmodes = nmodes;
    tsr->ndims = malloc(nmodes * sizeof *tsr->ndims);
    if(!tsr->ndims) {
        return -1;
    }
    memcpy(tsr->ndims, ndims, nmodes * sizeof *tsr->ndims);
    tsr->nnz = 0;
    tsr->inds = malloc(nmodes * sizeof *tsr->inds);
    if(!tsr->inds) {
        return -1;
    }
    for(i = 0; i < nmodes; ++i) {
        result = sptNewSizeVector(&tsr->inds[i], 0, 0);
        if(result) {
            return result;
        }
    }
    result = sptNewVector(&tsr->values, 0, 0);
    if(result) {
        return result;
    }
    return 0;
}

int sptCopySparseTensor(sptSparseTensor *dest, const sptSparseTensor *src) {
    size_t i;
    int result;
    dest->nmodes = src->nmodes;
    dest->ndims = malloc(dest->nmodes * sizeof *dest->ndims);
    if(!dest->ndims) {
        return -1;
    }
    memcpy(dest->ndims, src->ndims, src->nmodes * sizeof *src->ndims);
    dest->nnz = src->nnz;
    dest->inds = malloc(dest->nmodes * sizeof *dest->inds);
    if(!dest->inds) {
        return -1;
    }
    for(i = 0; i < dest->nmodes; ++i) {
        result = sptCopySizeVector(&dest->inds[i], &src->inds[i]);
        if(result) {
            return result;
        }
    }
    result = sptCopyVector(&dest->values, &src->values);
    if(result) {
        return result;
    }
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
            fprintf(stderr, "SpTOL ERROR: tensor unsorted on mode-0.\n");
            exit(-1);
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
            fprintf(stderr, "SpTOL ERROR: tensor unsorted on mode-0.\n");
            exit(-1);
        }
    }

}
