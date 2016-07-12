#include <SpTOL.h>

int sptNewSparseMatrix(sptSparseMatrix *mtx, size_t nrows, size_t ncols) {
    int result;
    mtx->nrows = nrows;
    mtx->ncols = ncols;
    mtx->nnz = 0;
    result = sptNewSizeVector(&mtx->rowind, 0, 0);
    if(result) {
        return result;
    }
    result = sptNewSizeVector(&mtx->colind, 0, 0);
    if(result) {
        return result;
    }
    result = sptNewVector(&mtx->values, 0, 0);
    if(result) {
        return result;
    }
    return 0;
}

int sptCopySparseMatrix(sptSparseMatrix *dest, const sptSparseMatrix *src) {
    int result;
    dest->nrows = src->nrows;
    dest->ncols = src->ncols;
    dest->nnz = src->nnz;
    result = sptCopySizeVector(&dest->rowind, &src->rowind);
    if(result) {
        return result;
    }
    result = sptCopySizeVector(&dest->colind, &src->colind);
    if(result) {
        return result;
    }
    result = sptCopyVector(&dest->values, &src->values);
    if(result) {
        return result;
    }
    return 0;
}

void sptFreeSparseMatrix(sptSparseMatrix *mtx) {
    sptFreeSizeVector(&mtx->rowind);
    sptFreeSizeVector(&mtx->colind);
    sptFreeVector(&mtx->values);
}
