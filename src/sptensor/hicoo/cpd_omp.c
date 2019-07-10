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
#else
  #include "clapack.h"
#endif
#include "hicoo.h"

#ifdef PARTI_USE_OPENMP


/*************************************************
 * PRIVATE FUNCTIONS
 *************************************************/
double spt_OmpCpdAlsStepHiCOO(
  sptSparseTensorHiCOO const * const hitsr,
  sptIndex const rank,
  sptIndex const niters,
  double const tol,
  const int tk,
  const int * par_iters,
  sptRankMatrix ** mats,
  sptRankMatrix *** copy_mats,
  sptValue * const lambda,
  int balanced)
{
  sptIndex const nmodes = hitsr->nmodes;
  sptIndex const stride = mats[0]->stride;
  double fit = 0;

  omp_set_num_threads(tk);
#ifdef PARTI_USE_MAGMA
  magma_set_omp_numthreads(tk);
  magma_set_lapack_numthreads(tk);
#endif

  // sptAssert(stride == rank);  // for correct column-major magma functions
  for(sptIndex m=0; m < nmodes; ++m) {
    sptAssert(hitsr->ndims[m] == mats[m]->nrows);
    sptAssert(mats[m]->ncols == rank);
  }

  sptValue alpha = 1.0, beta = 0.0;
  char notrans = 'N';
  // char trans = 'T';
  char uplo = 'L';
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

  sptTimer tmp_timer;
  sptNewTimer(&tmp_timer, 0);
  // double mttkrp_time, solver_time, norm_time, ata_time, fit_time;

  for(sptIndex it=0; it < niters; ++it) {
    sptTimer timer;
    sptNewTimer(&timer, 0);
    sptStartTimer(timer);

    for(sptIndex m=0; m < nmodes; ++m) {
      tmp_mat->nrows = mats[m]->nrows;

      /* Factor Matrices order */
      mats_order[0] = m;
      for(sptIndex i=1; i<nmodes; ++i)
          mats_order[i] = (m+i) % nmodes;     

      // sptAssert (sptOmpMTTKRPHiCOO_MatrixTiling(hitsr, mats, mats_order, m) == 0);  
      sptStartTimer(tmp_timer);
      if(par_iters[m] == 1) {
        sptAssert (sptOmpMTTKRPHiCOO_MatrixTiling_Scheduled_Reduce(hitsr, mats, copy_mats[m], mats_order, m, tk, balanced) == 0);
      } else {
        sptAssert (sptOmpMTTKRPHiCOO_MatrixTiling_Scheduled(hitsr, mats, mats_order, m, tk, balanced) == 0);
      }
      sptStopTimer(tmp_timer);
      sptPrintElapsedTime(tmp_timer, "MTTKRP");

      sptStartTimer(tmp_timer);
#ifdef PARTI_USE_OPENMP
      #pragma omp parallel for num_threads(tk)
#endif
      for(sptIndex i=0; i<mats[m]->nrows * stride; ++i)
        mats[m]->values[i] = tmp_mat->values[i];

      /* Solve ? * ata[nmodes] = mats[nmodes] (tmp_mat) */
      /* result is row-major, solve AT XT = BT */
      sptAssert ( sptRankMatrixSolveNormals(m, nmodes, ata, mats[m]) == 0 );
      sptStopTimer(tmp_timer);

      /* Normalized mats[m], store the norms in lambda. Use different norms to avoid precision explosion. */
      sptStartTimer(tmp_timer);
      if (it == 0 ) {
        sptRankMatrix2Norm(mats[m], lambda);
      } else {
        sptRankMatrixMaxNorm(mats[m], lambda);
      }
      sptStopTimer(tmp_timer);

      /* ata[m] = mats[m]^T * mats[m]) */
      sptStartTimer(tmp_timer);
      int blas_nrows = (int)(mats[m]->nrows);
      ssyrk_(&uplo, &notrans, &blas_rank, &blas_nrows, &alpha,
        mats[m]->values, &blas_stride, &beta, ata[m]->values, &blas_stride);
      sptStopTimer(tmp_timer);

    } // Loop nmodes

    sptStartTimer(tmp_timer);
    fit = sptKruskalTensorFitHiCOO(hitsr, lambda, mats, ata);
    sptStopTimer(tmp_timer);
    // fit_time = sptPrintElapsedTime(tmp_timer, "KruskalTensorFitHiCOO");

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



/*************************************************
 * PUBLIC FUNCTIONS
 *************************************************/
/**
 * OpenMP Parallel CANDECOMP/PARAFAC decomposition (CPD) using alternating least squares for HiCOO formatted sparse tensors.
 * @param[out] ktensor the Kruskal tensor
 * @param[in]  hitsr the HiCOO representation of a sparse tensor
 * @param[in]  rank the CPD rank
 * @param[in]  niters the maximum number of iterations
 * @param[in]  tol the tolerance value for convergence
 * @param[in]  tk the number of threads for superblock parallelism
 */
int sptOmpCpdAlsHiCOO(
  sptSparseTensorHiCOO const * const hitsr,
  sptIndex const rank,
  sptIndex const niters,
  double const tol,
  const int tk,
  int balanced,
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
    // assert(sptConstantRankMatrix(mats[m], 1) == 0);
    sptAssert(sptRandomizeRankMatrix(mats[m], hitsr->ndims[m], rank) == 0);
  }
  sptAssert(sptNewRankMatrix(mats[nmodes], max_dim, rank) == 0);
  sptAssert(sptConstantRankMatrix(mats[nmodes], 0) == 0);

  /* determine niters or num_kernel_dim to be parallelized */
  int * par_iters = (int *)malloc(nmodes * sizeof(*par_iters));
  sptIndex sk = (sptIndex)pow(2, hitsr->sk_bits);
  for(sptIndex m=0; m < nmodes; ++m) {
    par_iters[m] = 0;
    sptIndex num_kernel_dim = (hitsr->ndims[m] + sk - 1) / sk;
    if(num_kernel_dim <= PAR_MIN_DEGREE * NUM_CORES && hitsr->nkiters[m] / num_kernel_dim >= PAR_DEGREE_REDUCE) {
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

  ktensor->fit = spt_OmpCpdAlsStepHiCOO(hitsr, rank, niters, tol, tk, par_iters, mats, copy_mats, ktensor->lambda, balanced);

  sptStopTimer(timer);
  sptPrintElapsedTime(timer, "CPU  HiCOO SpTns CPD-ALS");
  sptFreeTimer(timer);

  ktensor->factors = mats;

#ifdef PARTI_USE_MAGMA
  magma_finalize();
#endif
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

#endif

