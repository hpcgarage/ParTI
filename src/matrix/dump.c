#include <SpTOL.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "../error/error.h"


/**
 * Dum a dense matrix to file
 *
 * @param mtx   a valid pointer to a sptMatrix variable
 * @param fp a file pointer 
 *
 */
int sptDumpMatrix(sptMatrix *mtx, FILE *fp) {
    int iores;
    size_t nrows = mtx->nrows;
    size_t ncols = mtx->ncols;
    iores = fprintf(fp, "%zu x %zu matrix\n", nrows, ncols);
    spt_CheckOSError(iores < 0, "SpMtx Dump");
    for(size_t i=0; i < nrows; ++i) {
      for(size_t j=0; j < ncols; ++j) {
          iores = fprintf(fp, "%lf\t", mtx->values[i * ncols + j]);
          spt_CheckOSError(iores < 0, "SpMtx Dump");
      }
      iores = fprintf(fp, "\n");
    }
    iores = fprintf(fp, "\n");
    return 0;
}
