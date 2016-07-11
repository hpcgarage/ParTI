#include <SpTOL.h>
#include <stdlib.h>
#include <string.h>

int sptSparseTensorKhatrioRaoMul(sptSparseTensor *Y, const sptSparseTensor *A, const sptSparseTensor *B) {
    size_t nmodes;
    size_t mode;
    size_t *inds;
    size_t i, j;
    int result;
    if(A->nmodes != B->nmodes) {
        return -1;
    }
    nmodes = A->nmodes;
    if(nmodes == 0) {
        return -1;
    }
    if(A->ndims[nmodes-1] != B->ndims[nmodes-1]) {
        return -1;
    }
    inds = malloc(nmodes * sizeof *inds);
    if(!inds) {
        return -1;
    }
    for(mode = 0; mode < nmodes-1; ++mode) {
        inds[mode] = A->ndims[mode] * B->ndims[mode];
    }
    inds[nmodes-1] = A->ndims[mode];
    result = sptNewSparseTensor(Y, nmodes, inds);
    free(inds);
    if(result) {
        return -1;
    }
    /* For each element in A and B */
    for(i = 0; i < A->nnz; ++i) {
        for(j = 0; j < B->nnz; ++j) {
            if(A->inds[nmodes-1].data[i] == B->inds[nmodes-1].data[j]) {
                /*
                    Y[f(i0,j0), ..., f(i(N-2), j(N-2))] = a[i10 ..., i(N-2)] * b[j0, ..., j(N-2)]
                    where f(in, jn) = jn + in * Jn
                */
                for(mode = 0; mode < nmodes-1; ++mode) {
                    sptAppendSizeVector(&Y->inds[mode], A->inds[mode].data[i] * B->ndims[mode] + B->inds[mode].data[j]);
                }
                sptAppendSizeVector(&Y->inds[nmodes-1], A->inds[nmodes-1].data[i]);
                sptAppendVector(&Y->values, A->values.data[i] * B->values.data[j]);
                ++Y->nnz;
            }
        }
    }
    sptSparseTensorSortIndex(Y);
    return 0;
}
