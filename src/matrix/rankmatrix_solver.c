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
#include <math.h>
#include "../error/error.h"
// #include "magma_lapack.h"

int sptRankMatrixSolveNormals(
  sptIndex const mode,
  sptIndex const nmodes,
  sptRankMatrix ** aTa,
  sptRankMatrix * rhs)
{
  int rank = (int)(aTa[0]->ncols);
  int stride = (int)(aTa[0]->stride);

  sptRankMatrixDotMulSeqTriangle(mode, nmodes, aTa);

  int info;
  char uplo = 'L';
  int nrhs = (int) rhs->nrows;
  sptValue * const neqs = aTa[nmodes]->values;

  /* Cholesky factorization */
  bool is_spd = true;
  // lapackf77_spotrf(&uplo, &rank, neqs, &stride, &info);
  spotrf_(&uplo, &rank, neqs, &stride, &info);
  if(info) {
    printf("Gram matrix is not SPD. Trying `gesv`.\n");
    is_spd = false;
  }

  /* Continue with Cholesky */
  if(is_spd) {
    /* Solve against rhs */
    // lapackf77_spotrs(&uplo, &rank, &nrhs, neqs, &stride, rhs->values, &stride, &info);
    spotrs_(&uplo, &rank, &nrhs, neqs, &stride, rhs->values, &stride, &info);
    if(info) {
      printf("DPOTRS returned %d\n", info);
    }
  } 
  else {
    int * ipiv = (int*)malloc(rank * sizeof(int));  

    /* restore gram matrix */
    sptRankMatrixDotMulSeqTriangle(mode, nmodes, aTa);

    sgesv_(&rank, &nrhs, neqs, &stride, ipiv, rhs->values, &stride, &info);
    // lapackf77_sgesv(&rank, &nrhs, neqs, &stride, ipiv, rhs->values, &stride, &info);
    // magma_sgesv(rank, nrhs, neqs, stride, ipiv, rhs->values, stride, &info);
    if(info) {
      printf("magma_sgesv returned %d\n", info);
    }

    free(ipiv);
  }

  return 0;
}
