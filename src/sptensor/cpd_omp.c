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
#include "sptensor.h"


double OmpCpdAlsStep(
  sptSparseTensor const * const spten,
  sptIndex const rank,
  sptIndex const niters,
  double const tol,
  const int tk,
  const int use_reduce,
  sptMatrix ** mats,  // Row-major
  sptMatrix ** copy_mats, 
  sptValue * const lambda)
{
  sptIndex const nmodes = spten->nmodes;
  sptIndex const stride = mats[0]->stride;
  double fit = 0;
#ifdef PARTI_USE_OPENMP  
  omp_set_num_threads(tk);
#endif

  for(sptIndex m=0; m < nmodes; ++m) {
    sptAssert(spten->ndims[m] == mats[m]->nrows);
    sptAssert(mats[m]->ncols == rank);
    // assert(mats[m]->stride == rank);  // for correct column-major magma functions
  }

  sptValue alpha = 1.0, beta = 0.0;
  char const notrans = 'N';
  char const trans = 'T';
  char const uplo = 'L';
  int blas_rank = (int) rank;
  int blas_stride = (int) stride;

  sptMatrix * tmp_mat = mats[nmodes];
  sptMatrix ** ata = (sptMatrix **)malloc((nmodes+1) * sizeof(*ata)); // symmetric matrices, but in column-major
  for(sptIndex m=0; m < nmodes+1; ++m) {
    ata[m] = (sptMatrix *)malloc(sizeof(sptMatrix));
    sptAssert(sptNewMatrix(ata[m], rank, rank) == 0);
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
      tmp_mat->nrows = mats[m]->nrows;

      /* Factor Matrices order */
      mats_order[0] = m;
      for(sptIndex i=1; i<nmodes; ++i)
          mats_order[i] = (m+i) % nmodes;

      // mats[nmodes]: row-major
      if(use_reduce == 1) {
        sptAssert (sptOmpMTTKRP_Reduce(spten, mats, copy_mats, mats_order, m, tk) == 0);
      } else {
        sptAssert (sptOmpMTTKRP(spten, mats, mats_order, m, tk) == 0);
      }

      // Row-major
#ifdef PARTI_USE_OPENMP
    #pragma omp parallel for num_threads(tk)
#endif
      for(sptIndex i=0; i<mats[m]->nrows * stride; ++i)
        mats[m]->values[i] = tmp_mat->values[i];        

      /* Solve ? * ata[nmodes] = mats[nmodes] (tmp_mat) */
      sptAssert ( sptMatrixSolveNormals(m, nmodes, ata, mats[m]) == 0 );

      /* Normalized mats[m], store the norms in lambda. Use different norms to avoid precision explosion. */
      if (it == 0 ) {
        sptMatrix2Norm(mats[m], lambda);
      } else {
        sptMatrixMaxNorm(mats[m], lambda);
      }

      /* ata[m] = mats[m]^T * mats[m]) */
      int blas_nrows = (int)(mats[m]->nrows);
      ssyrk_(&uplo, &notrans, &blas_rank, &blas_nrows, &alpha,
        mats[m]->values, &blas_stride, &beta, ata[m]->values, &blas_stride);

    } // Loop nmodes

    // PrintDenseValueVector(lambda, rank, "lambda", "debug.txt");
    fit = KruskalTensorFit(spten, lambda, mats, ata);

    sptStopTimer(timer);
    double its_time = sptElapsedTime(timer);
    sptFreeTimer(timer);

    printf("  its = %3"PARTI_PRI_INDEX " ( %.3lf s ) fit = %0.5f  delta = %+0.4e\n",
        it+1, its_time, fit, fit - oldfit);
    if(it > 0 && fabs(fit - oldfit) < tol) {
      break;
    }
    oldfit = fit;

  } // Loop niters

  GetFinalLambda(rank, nmodes, mats, lambda);

  for(sptIndex m=0; m < nmodes+1; ++m) {
    sptFreeMatrix(ata[m]);
  }
  free(ata);
  free(mats_order);

  return fit;
}


/**
 * OpenMP Parallel CANDECOMP/PARAFAC decomposition (CPD) using alternating least squares for COO formatted sparse tensors.
 * @param[out] ktensor the Kruskal tensor
 * @param[in]  spten the COO representation of a sparse tensor
 * @param[in]  rank the CPD rank
 * @param[in]  niters the maximum number of iterations
 * @param[in]  tol the tolerance value for convergence
 * @param[in]  tk the number of threads
 * @param[in]  use_reduce =1: use privatization; =0: use OpenMP atomic.
 */
int sptOmpCpdAls(
  sptSparseTensor const * const spten,
  sptIndex const rank,
  sptIndex const niters,
  double const tol,
  const int tk,
  const int use_reduce,
  sptKruskalTensor * ktensor)
{
  sptIndex nmodes = spten->nmodes;
#ifdef PARTI_USE_MAGMA
  magma_init();
#endif

  /* Initialize factor matrices */
  sptIndex max_dim = sptMaxIndexArray(spten->ndims, nmodes);
  sptMatrix ** mats = (sptMatrix **)malloc((nmodes+1) * sizeof(*mats));
  for(sptIndex m=0; m < nmodes+1; ++m) {
    mats[m] = (sptMatrix *)malloc(sizeof(sptMatrix));
  }
  for(sptIndex m=0; m < nmodes; ++m) {
    sptAssert(sptNewMatrix(mats[m], spten->ndims[m], rank) == 0);
    sptAssert(sptRandomizeMatrix(mats[m], spten->ndims[m], rank) == 0);
  }
  sptAssert(sptNewMatrix(mats[nmodes], max_dim, rank) == 0);
  sptAssert(sptConstantMatrix(mats[nmodes], 0) == 0);

  sptMatrix ** copy_mats;
  if(use_reduce == 1) {
    copy_mats = (sptMatrix **)malloc(tk * sizeof(*copy_mats));
    for(int t=0; t<tk; ++t) {
      copy_mats[t] = (sptMatrix *)malloc(sizeof(sptMatrix));
      sptAssert(sptNewMatrix(copy_mats[t], max_dim, rank) == 0);
      sptAssert(sptConstantMatrix(copy_mats[t], 0) == 0);
    }
  }

  sptTimer timer;
  sptNewTimer(&timer, 0);
  sptStartTimer(timer);

  ktensor->fit = OmpCpdAlsStep(spten, rank, niters, tol, tk, use_reduce, mats, copy_mats, ktensor->lambda);

  sptStopTimer(timer);
  sptPrintElapsedTime(timer, "CPU  SpTns CPD-ALS");
  sptFreeTimer(timer);

  ktensor->factors = mats;

#ifdef PARTI_USE_MAGMA
  magma_finalize();
#endif
  sptFreeMatrix(mats[nmodes]);
  if(use_reduce == 1) {
    for(int t=0; t<tk; ++t) {
      sptFreeMatrix(copy_mats[t]);
      free(copy_mats[t]);
    }
    free(copy_mats);
  }

  return 0;
}
