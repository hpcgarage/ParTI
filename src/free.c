#include <stdlib.h>
#include <SpTOL.h>

void sptFreeSparseMatrix(sptSparseMatrix *mtx, void (*free_func)(void *)) {
    free_func(mtx->rowind);
    free_func(mtx->colind);
    free_func(mtx->values);
    free_func(mtx);
}

void sptFreeMatrix(sptMatrix *mtx, void (*free_func)(void *)) {
    free_func(mtx->values);
    free_func(mtx);
}

void sptFreeSparseTensor(sptSparseTensor *tsr, void (*free_func)(void *)) {
    size_t i;
    free_func(tsr->ndims);
    for(i = 0; i < tsr->nmodes; ++i) {
        free_func(tsr->inds[i]);
    }
    free_func(tsr->inds);
    free_func(tsr->values);
}

void sptFreeSemiSparseTensor(sptSemiSparseTensor *tsr, void (*free_func)(void *)) {
    size_t i;
    free_func(tsr->ndims);
    for(i = 0; i < tsr->nmodes-1; ++i) {
        free_func(tsr->inds[i]);
    }
    free_func(tsr->inds);
    free_func(tsr->fibers);
}
