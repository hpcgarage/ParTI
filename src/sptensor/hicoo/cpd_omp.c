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
#include "magma_v2.h"
#include "magma_lapack.h"
#include "hicoo.h"


double OmpCpdAlsStepHiCOO(
  sptSparseTensorHiCOO const * const hitsr,
  sptIndex const rank,
  sptIndex const niters,
  double const tol,
  const int tk,
  const int tb,
  const int * par_iters,
  sptRankMatrix ** mats,
  sptRankMatrix *** copy_mats,
  sptValue * const lambda)
{
  sptIndex const nmodes = hitsr->nmodes;
  sptIndex const stride = mats[0]->stride;
  double fit = 0;
  omp_set_num_threads(tk);

  // sptAssert(stride == rank);  // for correct column-major magma functions
  for(sptIndex m=0; m < nmodes; ++m) {
    sptAssert(hitsr->ndims[m] == mats[m]->nrows);
    sptAssert(mats[m]->ncols == rank);
  }

  sptValue alpha = 1.0, beta = 0.0;

  sptRankMatrix * tmp_mat = mats[nmodes];
  sptRankMatrix ** ata = (sptRankMatrix **)malloc((nmodes+1) * sizeof(*ata));
  for(sptIndex m=0; m < nmodes+1; ++m) {
    ata[m] = (sptRankMatrix *)malloc(sizeof(sptRankMatrix));
    sptNewRankMatrix(ata[m], rank, rank);
    sptAssert(mats[m]->stride == ata[m]->stride);
  }

  /* Compute all "ata"s */
  for(sptIndex m=0; m < nmodes; ++m) {
    /* ata[m] = mats[m]^T * mats[m]) */
    blasf77_sgemm("N", "T", (magma_int_t*)&rank, (magma_int_t*)&rank, (magma_int_t*)&(mats[m]->nrows), &alpha,
      mats[m]->values, (magma_int_t*)&stride, mats[m]->values, (magma_int_t*)&stride, &beta, ata[m]->values, (magma_int_t*)&stride);
  }
  // printf("Initial mats:\n");
  // for(size_t m=0; m < nmodes+1; ++m)
  //   sptDumpRankMatrix(mats[m], stdout);
  // printf("Initial ata:\n");
  // for(sptIndex m=0; m < nmodes+1; ++m)
  //   sptDumpRankMatrix(ata[m], stdout);

  double oldfit = 0;

  // timer_reset(&g_timers[TIMER_ATA]);
  // Timer itertime;
  // Timer * modetime = (Timer*)malloc(nmodes*sizeof(Timer));

  /* For MttkrpHyperTensor with size rank. */
  sptIndex * mats_order = (sptIndex*)malloc(nmodes * sizeof(*mats_order));
  int * ipiv = (int*)malloc(rank * sizeof(int));
  int info;


  for(sptIndex it=0; it < niters; ++it) {
    // printf("  its = %3lu\n", it+1);
    sptTimer timer;
    sptNewTimer(&timer, 0);
    sptStartTimer(timer);

    for(sptIndex m=0; m < nmodes; ++m) {
      // printf("\nmode %lu \n", m);
      tmp_mat->nrows = mats[m]->nrows;

      /* Factor Matrices order */
      mats_order[0] = m;
      for(sptIndex i=1; i<nmodes; ++i)
          mats_order[i] = (m+i) % nmodes;     

      // sptAssert (sptOmpMTTKRPHiCOO_MatrixTiling(hitsr, mats, mats_order, m) == 0);  
      if(par_iters[m] == 1) {
        sptAssert (sptOmpMTTKRPHiCOO_MatrixTiling_Scheduled_Reduce(hitsr, mats, copy_mats[m], mats_order, m, tk, tb) == 0);
      } else {
        sptAssert (sptOmpMTTKRPHiCOO_MatrixTiling_Scheduled(hitsr, mats, mats_order, m, tk, tb) == 0);
      }
      // printf("sptMTTKRPHiCOO_MatrixTiling mats[nmodes]:\n");
      // sptDumpRankMatrix(mats[nmodes], stdout);

      // Column-major calculation
      sptRankMatrixDotMulSeqCol(m, nmodes, ata);
      // printf("sptRankMatrixDotMulSeqCol ata[nmodes]:\n");
      // sptDumpRankMatrix(ata[nmodes], stdout);

      memcpy(mats[m]->values, tmp_mat->values, mats[m]->nrows * stride * sizeof(sptValue));
      /* Solve ? * ata[nmodes] = mats[nmodes] (tmp_mat) */
      magma_sgesv(rank, mats[m]->nrows, ata[nmodes]->values, stride, ipiv, mats[m]->values, stride, &info);
      sptAssert ( info == 0 );
      // printf("Inverse mats[m]:\n");
      // sptDumpRankMatrix(mats[m], stdout);

      /* Normalized mats[m], store the norms in lambda. Use different norms to avoid precision explosion. */
      if (it == 0 ) {
        sptRankMatrix2Norm(mats[m], lambda);
      } else {
        sptRankMatrixMaxNorm(mats[m], lambda);
      }
      // printf("Normalize mats[m]:\n");
      // sptDumpRankMatrix(mats[m], stdout);
      // printf("lambda:\n");
      // for(size_t i=0; i<rank; ++i)
      //   printf("%lf  ", lambda[i]);
      // printf("\n\n");

      /* ata[m] = mats[m]^T * mats[m]) */
      blasf77_sgemm("N", "T", (magma_int_t*)&rank, (magma_int_t*)&rank, (magma_int_t*)&(mats[m]->nrows), &alpha, mats[m]->values, (magma_int_t*)&stride, mats[m]->values, (magma_int_t*)&stride, &beta, ata[m]->values, (magma_int_t*)&stride);
      // printf("Update ata[m]:\n");
      // sptDumpRankMatrix(ata[m], stdout);

      // timer_stop(&modetime[m]);

    } // Loop nmodes

    // PrintDenseValueVector(lambda, rank, "lambda", "debug.txt");
    fit = KruskalTensorFitHiCOO(hitsr, lambda, mats, ata);

    sptStopTimer(timer);
    double its_time = sptElapsedTime(timer);
    sptFreeTimer(timer);

    printf("  its = %3u ( %.3lf s ) fit = %0.5f  delta = %+0.4e\n",
        it+1, its_time, fit, fit - oldfit);
    // for(IndexType m=0; m < nmodes; ++m) {
    //   printf("     mode = %1"PF_INDEX" (%0.3fs)\n", m+1,
    //       modetime[m].seconds);
    // }
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
  free(ipiv);
  // free(modetime);

  return fit;
}


int sptOmpCpdAlsHiCOO(
  sptSparseTensorHiCOO const * const hitsr,
  sptIndex const rank,
  sptIndex const niters,
  double const tol,
  const int tk,
  const int tb,
  sptRankKruskalTensor * ktensor)
{
  sptIndex nmodes = hitsr->nmodes;
  magma_init();

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
    // assert(sptNewRankMatrix(mats[m], hitsr->ndims[m], rank) == 0);
    // assert(sptConstantRankMatrix(mats[m], 1) == 0);
    assert(sptRandomizeRankMatrix(mats[m], hitsr->ndims[m], rank) == 0);
  }
  sptNewRankMatrix(mats[nmodes], max_dim, rank);
  sptAssert(sptConstantRankMatrix(mats[nmodes], 0) == 0);

  /* determine niters or num_kernel_dim to be parallelized */
  int * par_iters = (int *)malloc(nmodes * sizeof(*par_iters));
  sptIndex sk = (sptIndex)pow(2, hitsr->sk_bits);
  for(sptIndex m=0; m < nmodes; ++m) {
    par_iters[m] = 0;
    sptIndex num_kernel_dim = (hitsr->ndims[m] + sk - 1) / sk;
    // printf("num_kernel_dim: %u, hitsr->nkiters[m] / num_kernel_dim: %u\n", num_kernel_dim, hitsr->nkiters[m]/num_kernel_dim);
    if(num_kernel_dim <= 24 && hitsr->nkiters[m] / num_kernel_dim >= 20) {
        par_iters[m] = 1;
    }
  }
  printf("par_iters:\n");
  for(sptIndex m=0; m < nmodes; ++m) {
    printf("%d, ", par_iters[m]);
  }
  printf("\n");

  sptRankMatrix *** copy_mats = (sptRankMatrix ***)malloc(nmodes * sizeof(*copy_mats));
  for(sptIndex m=0; m < nmodes; ++m) {
    if (par_iters[m] == 1) {
      copy_mats[m] = (sptRankMatrix **)malloc(tk * sizeof(sptRankMatrix*));
      for(int t=0; t<tk; ++t) {
        copy_mats[m][t] = (sptRankMatrix *)malloc(sizeof(sptRankMatrix));
        sptAssert(sptNewRankMatrix(copy_mats[m][t], hitsr->ndims[m], rank) == 0);
        sptAssert(sptConstantRankMatrix(copy_mats[m][t], 0) == 0);
      }
    }
  }

  sptTimer timer;
  sptNewTimer(&timer, 0);
  sptStartTimer(timer);

  ktensor->fit = OmpCpdAlsStepHiCOO(hitsr, rank, niters, tol, tk, tb, par_iters,  mats, copy_mats, ktensor->lambda);

  sptStopTimer(timer);
  sptPrintElapsedTime(timer, "CPU  HiCOO SpTns CPD-ALS");
  sptFreeTimer(timer);

  ktensor->factors = mats;

  magma_finalize();
  sptFreeRankMatrix(mats[nmodes]);
  for(sptIndex m=0; m < nmodes; ++m) {
    if(par_iters[m] == 1) {
      for(int t=0; t<tk; ++t) {
        sptFreeRankMatrix(copy_mats[m][t]);
        free(copy_mats[m][t]);
      }
      free(copy_mats[m]);
    }
  }
  free(copy_mats);

  return 0;
}
