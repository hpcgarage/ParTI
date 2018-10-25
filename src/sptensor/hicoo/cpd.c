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
#include <math.h>
#ifdef PARTI_USE_MAGMA
  #include "magma_v2.h"
  #include "magma_lapack.h"
#endif
#include "hicoo.h"


double CpdAlsStepHiCOO(
  sptSparseTensorHiCOO const * const hitsr,
  sptIndex const rank,
  sptIndex const niters,
  double const tol,
  sptRankMatrix ** mats,
  sptValue * const lambda)
{
  sptIndex const nmodes = hitsr->nmodes;
  sptIndex const stride = mats[0]->stride;
  double fit = 0;

  for(sptIndex m=0; m < nmodes; ++m) {
    sptAssert(hitsr->ndims[m] == mats[m]->nrows);
    sptAssert(mats[m]->ncols == rank);
  }

  sptValue alpha = 1.0, beta = 0.0;
  char const notrans = 'N';
  char const trans = 'T';
  char const uplo = 'L';
  int blas_rank = (int) rank;
  int blas_stride = (int) stride;

  sptRankMatrix * tmp_mat = mats[nmodes];
  sptRankMatrix ** ata = (sptRankMatrix **)malloc((nmodes+1) * sizeof(*ata));
  for(sptIndex m=0; m < nmodes+1; ++m) {
    ata[m] = (sptRankMatrix *)malloc(sizeof(sptRankMatrix));
    sptAssert(sptNewRankMatrix(ata[m], rank, rank) == 0);
    sptAssert(mats[m]->stride == ata[m]->stride);
  }

  /* Compute all "ata"s */
  for(sptIndex m=0; m < nmodes; ++m) {
    /* ata[m] = mats[m]^T * mats[m]), actually do A * A' due to row-major mats, and output an upper triangular matrix. */
    int blas_nrows = (int)(mats[m]->nrows);
    ssyrk_(&uplo, &notrans, &blas_rank, &blas_nrows, &alpha,
      mats[m]->values, &blas_stride, &beta, ata[m]->values, &blas_stride);
  }

  double oldfit = 0;
  sptIndex * mats_order = (sptIndex*)malloc(nmodes * sizeof(*mats_order));

  for(sptIndex it=0; it < niters; ++it) {
    sptTimer timer;
    sptNewTimer(&timer, 0);
    sptStartTimer(timer);

    for(sptIndex m=0; m < nmodes; ++m) {
      printf("\nmode %u \n", m);
      tmp_mat->nrows = mats[m]->nrows;

      /* Factor Matrices order */
      mats_order[0] = m;
      for(sptIndex i=1; i<nmodes; ++i)
          mats_order[i] = (m+i) % nmodes;     

      sptAssert (sptMTTKRPHiCOO_MatrixTiling(hitsr, mats, mats_order, m) == 0);

      memcpy(mats[m]->values, tmp_mat->values, mats[m]->nrows * stride * sizeof(sptValue));
      /* Solve ? * ata[nmodes] = mats[nmodes] (tmp_mat) */
      sptAssert ( sptRankMatrixSolveNormals(m, nmodes, ata, mats[m]) == 0 );

      /* Normalized mats[m], store the norms in lambda. Use different norms to avoid precision explosion. */
      if (it == 0 ) {
        sptRankMatrix2Norm(mats[m], lambda);
      } else {
        sptRankMatrixMaxNorm(mats[m], lambda);
      }

      /* ata[m] = mats[m]^T * mats[m]) */
      int blas_nrows = (int)(mats[m]->nrows);
      ssyrk_(&uplo, &notrans, &blas_rank, &blas_nrows, &alpha,
        mats[m]->values, &blas_stride, &beta, ata[m]->values, &blas_stride);

    } // Loop nmodes

    fit = KruskalTensorFitHiCOO(hitsr, lambda, mats, ata);

    sptStopTimer(timer);
    double its_time = sptElapsedTime(timer);
    sptFreeTimer(timer);

    printf("  its = %3u ( %.3lf s ) fit = %0.5f  delta = %+0.4e\n",
        it+1, its_time, fit, fit - oldfit);
    if(it > 0 && fabs(fit - oldfit) < tol) {
      break;
    }
    oldfit = fit;
    
  } // Loop niters

  GetRankFinalLambda(rank, nmodes, mats, lambda);

  for(sptIndex m=0; m < nmodes+1; ++m) {
    sptFreeRankMatrix(ata[m]);
  }
  free(ata);
  free(mats_order);


  return fit;
}


/**
 * Sequential CANDECOMP/PARAFAC decomposition (CPD) using alternating least squares for HiCOO formatted sparse tensors.
 * @param[out] ktensor the Kruskal tensor
 * @param[in]  hitsr the HiCOO representation of a sparse tensor
 * @param[in]  rank the CPD rank
 * @param[in]  niters the maximum number of iterations
 * @param[in]  tol the tolerance value for convergence
 */
int sptCpdAlsHiCOO(
  sptSparseTensorHiCOO const * const hitsr,
  sptIndex const rank,
  sptIndex const niters,
  double const tol,
  sptRankKruskalTensor * ktensor)
{
  sptIndex nmodes = hitsr->nmodes;
#ifdef PARTI_USE_MAGMA
  magma_init();
#endif

  /* Initialize factor matrices */
  sptIndex max_dim = 0;
  for(sptIndex m=0; m < nmodes; ++m) {
    max_dim = (hitsr->ndims[m] > max_dim) ? hitsr->ndims[m] : max_dim;
  }
  sptRankMatrix ** mats = (sptRankMatrix **)malloc((nmodes+1) * sizeof(*mats));
  for(sptIndex m=0; m < nmodes+1; ++m) {
    mats[m] = (sptRankMatrix *)malloc(sizeof(sptRankMatrix));
  }
  for(sptIndex m=0; m < nmodes; ++m) {
    sptAssert(sptNewRankMatrix(mats[m], hitsr->ndims[m], rank) == 0);
    sptAssert(sptRandomizeRankMatrix(mats[m], hitsr->ndims[m], rank) == 0);
  }
  sptAssert(sptNewRankMatrix(mats[nmodes], max_dim, rank) == 0);
  sptAssert(sptConstantRankMatrix(mats[nmodes], 0) == 0);

  sptTimer timer;
  sptNewTimer(&timer, 0);
  sptStartTimer(timer);

  ktensor->fit = CpdAlsStepHiCOO(hitsr, rank, niters, tol, mats, ktensor->lambda);

  sptStopTimer(timer);
  sptPrintElapsedTime(timer, "CPU  HiCOO SpTns CPD-ALS");
  sptFreeTimer(timer);

  ktensor->factors = mats;

#ifdef PARTI_USE_MAGMA
  magma_finalize();
#endif
  sptFreeRankMatrix(mats[nmodes]);

  return 0;
}
