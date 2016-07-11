#include <SpTOL.h>
#include <stdlib.h>
#include <string.h>

int sptSparseTensorKroneckerMul(sptSparseTensor *Y, const sptSparseTensor *A, const sptSparseTensor *B) {
    size_t nmodes;
    size_t mode;
    size_t *inds;
    size_t i, j;
    int result;
    if(A->nmodes != B->nmodes) {
        return -1;
    }
    nmodes = A->nmodes;
    inds = malloc(nmodes * sizeof *inds);
    if(!inds) {
        return -1;
    }
    for(mode = 0; mode < nmodes; ++mode) {
        inds[mode] = A->ndims[mode] * B->ndims[mode];
    }
    result = sptNewSparseTensor(Y, nmodes, inds);
    free(inds);
    if(result) {
        return -1;
    }
    /* For each element in A and B */
    for(i = 0; i < A->nnz; ++i) {
        for(j = 0; j < B->nnz; ++j) {
            /*
                Y[f(i1,j1), ..., f(i(N-1), j(N-1)] = a[i1, ..., i(N-1)] * b[j1, ..., j(N-1)]
                where f(in, jn) = jn + in * Jn
            */
            for(mode = 0; mode < nmodes; ++mode) {
                sptAppendSizeVector(&Y->inds[mode], A->inds[mode].data[i] * B->ndims[mode] + B->inds[mode].data[j]);
            }
            sptAppendVector(&Y->values, A->values.data[i] * B->values.data[j]);
            ++Y->nnz;
        }
    }
    sptSparseTensorSortIndex(Y);
    return 0;
}
