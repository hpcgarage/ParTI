/*
    This file is part of ParTI!.

    ParTI! is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as
    published by the Free Software Foundation, either version 3 of
    the License, or (at your option) any later version.

    ParTI! is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with ParTI!.
    If not, see <http://www.gnu.org/licenses/>.
*/

#include <ParTI.h>
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
          iores = fprintf(fp, "%.2lf\t", mtx->values[i * ncols + j]);
          spt_CheckOSError(iores < 0, "SpMtx Dump");
      }
      iores = fprintf(fp, "\n");
    }
    iores = fprintf(fp, "\n");
    return 0;
}
