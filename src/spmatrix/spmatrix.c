#include <SpTOL.h>

/**
 * Initialize a new sparse matrix
 *
 * @param mtx   a valid pointer to an uninitialized sptSparseMatrix variable
 * @param nrows the number of rows
 * @param ncols the number of columns
 */
int sptNewSparseMatrix(sptSparseMatrix *mtx, size_t nrows, size_t ncols) {
    int result;
    mtx->nrows = nrows;
    mtx->ncols = ncols;
    mtx->nnz = 0;
    result = sptNewSizeVector(&mtx->rowind, 0, 0);
    spt_CheckError(result, "SpMtx New", NULL);
    result = sptNewSizeVector(&mtx->colind, 0, 0);
    spt_CheckError(result, "SpMtx New", NULL);
    result = sptNewVector(&mtx->values, 0, 0);
    spt_CheckError(result, "SpMtx New", NULL);
    return 0;
}

/**
 * Copy a sparse matrix to an uninitialized sparse matrix
 *
 * @param dest a pointer to an uninitialized sparse matrix
 * @param src  a pointer to an existing valid sparse matrix
 *
 * The contents of `src` will be copied to `dest`.
 */
int sptCopySparseMatrix(sptSparseMatrix *dest, const sptSparseMatrix *src) {
    int result;
    dest->nrows = src->nrows;
    dest->ncols = src->ncols;
    dest->nnz = src->nnz;
    result = sptCopySizeVector(&dest->rowind, &src->rowind);
    spt_CheckError(result, "SpMtx Copy", NULL);
    result = sptCopySizeVector(&dest->colind, &src->colind);
    spt_CheckError(result, "SpMtx Copy", NULL);
    result = sptCopyVector(&dest->values, &src->values);
    spt_CheckError(result, "SpMtx Copy", NULL);
    return 0;
}

/**
 * Release the memory buffer a sparse matrix is holding
 *
 * @param mtx a pointer to a valid sparse matrix
 *
 * By using `sptFreeSparseMatrix`, a valid sparse matrix would become
 * uninitialized and should not be used anymore prior to another initialization
 */
void sptFreeSparseMatrix(sptSparseMatrix *mtx) {
    sptFreeSizeVector(&mtx->rowind);
    sptFreeSizeVector(&mtx->colind);
    sptFreeVector(&mtx->values);
}
