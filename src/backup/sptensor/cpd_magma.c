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

/**** !!! This uses MAGMA for all matrix functions. Can be changed to support CP-ALS on GPU ****/

#include <ParTI.h>
#include <assert.h>
#include <math.h>
#include "magma_v2.h"
#include "magma_lapack.h"
#include "sptensor.h"


double CpdAlsStep(
  sptSparseTensor const * const spten,
  size_t const rank,
  size_t const niters,
  double const tol,
  sptMatrix ** mats,
  sptScalar * const lambda)
{
  size_t const nmodes = spten->nmodes;
  double fit = 0;

  for(size_t m=0; m < nmodes; ++m) {
    assert(spten->ndims[m] == mats[m]->nrows);
    assert(mats[m]->ncols == rank);
    assert(mats[m]->stride == rank);  // for correct column-major magma functions
  }

  magma_init();
  magma_queue_t queue;
  magma_queue_create(0, &queue);


  sptMatrix * tmp_mat = mats[nmodes];
  sptMatrix ** ata = (sptMatrix **)malloc((nmodes+1) * sizeof(*ata));
  for(size_t m=0; m < nmodes+1; ++m) {
    ata[m] = (sptMatrix *)malloc(sizeof(sptMatrix));
    sptNewMatrix(ata[m], rank, rank);
  }

  for(size_t m=0; m < nmodes; ++m) {
    /* ata[m] = mats[m]^T * mats[m]) */
    // cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, rank, rank, mats[m]->nrows, 1.0, mats[m]->values, mats[m]->stride, mats[m]->values, mats[m]->stride, 0.0, ata[m]->values, ata[m]->stride);
    magma_sgemm(MagmaNoTrans, MagmaTrans, rank, rank, mats[m]->nrows, 1.0,
      mats[m]->values, mats[m]->stride, mats[m]->values, mats[m]->stride, 0.0, ata[m]->values, ata[m]->stride, queue);
  }
  printf("Initial mats:\n");
  for(size_t m=0; m < nmodes+1; ++m)
    sptDumpMatrix(mats[m], stdout);
  printf("Initial ata:\n");
  for(size_t m=0; m < nmodes+1; ++m)
    sptDumpMatrix(ata[m], stdout);


  double oldfit = 0;

  // timer_reset(&g_timers[TIMER_ATA]);
  // Timer itertime;
  // Timer * modetime = (Timer*)malloc(nmodes*sizeof(Timer));

  sptSizeVector mats_order;
  sptNewSizeVector(&mats_order, nmodes, nmodes);
  int * ipiv = (int*)malloc(rank * sizeof(int));
  int info;


  for(size_t it=0; it < niters; ++it) {
    // printf("  its = %3lu\n", it+1);
    sptTimer timer;
    sptNewTimer(&timer, 0);
    sptStartTimer(timer);

    for(size_t m=0; m < nmodes; ++m) {
      // printf("\nmode %lu \n", m);
      tmp_mat->nrows = mats[m]->nrows;

      /* Factor Matrices order */
      mats_order.data[0] = m;
      for(size_t i=1; i<nmodes; ++i)
          mats_order.data[i] = (m+i) % nmodes;
      // sptDumpSizeVector(&mats_order, stdout);

      assert (sptMTTKRP(spten, mats, mats_order.data, m) == 0);
      // printf("sptMTTKRP:\n");
      // sptDumpMatrix(mats[nmodes], stdout);

      sptMatrixDotMulSeq(m, nmodes, ata);
      // printf("sptMatrixDotMulSeq:\n");
      // sptDumpMatrix(ata[nmodes], stdout);

      /* Solve ? * ata[nmodes] = mats[nmodes] (tmp_mat) */
      // LAPACKE_sgesv(LAPACK_ROW_MAJOR, rank, rank, ata[nmodes]->values, ata[nmodes]->stride, ipiv, tmp_mat->values, tmp_mat->stride);
      magma_sgesv(rank, mats[m]->nrows, ata[nmodes]->values, ata[nmodes]->stride, ipiv, tmp_mat->values, tmp_mat->stride, &info);

      // printf("Inverse ata[nmodes] LU:\n");
      // sptDumpMatrix(ata[nmodes], stdout);
      // printf("Inverse ata[nmodes]:\n");
      // sptDumpMatrix(tmp_mat, stdout);

      memset(mats[m]->values, tmp_mat->values, mats[m]->nrows * mats[m]->stride * sizeof(sptScalar));
      /* sptMatrixMultiply(tmp_mat, ata[nmodes], mats[m]); */
      // cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
      //   tmp_mat->nrows, rank, mats[m]->ncols,
      //   1.0, tmp_mat->values, tmp_mat->stride,
      //   unitMat->values, unitMat->stride,
      //   0.0, mats[m]->values, mats[m]->stride);
      // printf("Update mats[m]:\n");
      // sptDumpMatrix(mats[m], stdout);

      /* Normalized mats[m], store the norms in lambda */
      sptMatrix2Norm(mats[m], lambda);
      // printf("Normalize mats[m]:\n");
      // sptDumpMatrix(mats[m], stdout);
      // printf("lambda:\n");
      // for(size_t i=0; i<rank; ++i)
      //   printf("%lf  ", lambda[i]);
      // printf("\n\n");

      /* ata[m] = mats[m]^T * mats[m]) */
      // cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, rank, rank, mats[m]->nrows, 1.0, mats[m]->values, mats[m]->stride, mats[m]->values, mats[m]->stride, 0.0, ata[m]->values, ata[m]->stride);
      magma_sgemm(MagmaNoTrans, MagmaTrans, rank, rank, mats[m]->nrows, 1.0,
        mats[m]->values, mats[m]->stride, mats[m]->values, mats[m]->stride, 0.0, ata[m]->values, ata[m]->stride, queue);
      // printf("Update ata[m]:\n");
      // sptDumpMatrix(ata[m], stdout);

      // timer_stop(&modetime[m]);

    } // Loop nmodes

    // PrintDenseValueVector(lambda, rank, "lambda", "debug.txt");
    fit = KruskalTensorFit(spten, lambda, mats, tmp_mat, ata);

    sptStopTimer(timer);
    sptPrintElapsedTime(timer, "Iteration");
    sptFreeTimer(timer);

    printf("  its = %3lu  fit = %0.5f  delta = %+0.4e\n",
        it+1, fit, fit - oldfit);
    // for(IndexType m=0; m < nmodes; ++m) {
    //   printf("     mode = %1"PF_INDEX" (%0.3fs)\n", m+1,
    //       modetime[m].seconds);
    // }

    if(it > 0 && fabs(fit - oldfit) < tol) {
      break;
    }
    oldfit = fit;

  } // Loop niters

  // GetFinalLambda(rank, nmodes, mats, lambda);

  for(size_t m=0; m < nmodes+1; ++m) {
    sptFreeMatrix(ata[m]);
  }
  free(ata);
  sptFreeSizeVector(&mats_order);
  free(ipiv);
  // free(modetime);

  magma_queue_destroy(queue);
  magma_finalize();

  return fit;
}


int sptCpdAls(
  sptSparseTensor const * const spten,
  size_t const rank,
  size_t const niters,
  double const tol,
  sptKruskalTensor * ktensor)
{
  size_t nmodes = spten->nmodes;

  /* Initialize factor matrices */
  size_t max_dim = sptMaxSizeArray(spten->ndims, nmodes);
  sptMatrix ** mats = (sptMatrix **)malloc((nmodes+1) * sizeof(*mats));
  for(size_t m=0; m < nmodes+1; ++m) {
    mats[m] = (sptMatrix *)malloc(sizeof(sptMatrix));
  }
  for(size_t m=0; m < nmodes; ++m) {
    assert(sptNewMatrix(mats[m], spten->ndims[m], rank) == 0);
    assert(sptConstantMatrix(mats[m], 1) == 0);
    // assert(sptRandomizeMatrix(mats[m], spten->ndims[m], rank) == 0);
  }
  sptNewMatrix(mats[nmodes], max_dim, rank);

  sptTimer timer;
  sptNewTimer(&timer, 0);
  sptStartTimer(timer);

  ktensor->fit = CpdAlsStep(spten, rank, niters, tol, mats, ktensor->lambda);

  sptStopTimer(timer);
  sptPrintElapsedTime(timer, "CPU  SpTns CPD-ALS");
  sptFreeTimer(timer);

  ktensor->factors = mats;

  sptFreeMatrix(mats[nmodes]);

  return 0;
}
