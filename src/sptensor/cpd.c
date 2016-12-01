/*
    This file is part of SpTOL.

    SpTOL is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as
    published by the Free Software Foundation, either version 3 of
    the License, or (at your option) any later version.

    SpTOL is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with SpTOL.
    If not, see <http://www.gnu.org/licenses/>.
*/

#include <SpTOL.h>
#include "sptensor.h"
#include <cblas.h>


double CpdAlsStep(
  sptSparseTensor const * const spten,
  size_t const rank,
  size_t const niters,
  double const tol,
  sptMatrix ** mats,
  sptScalar * const lambda)
{
  size_t const nmodes = spten->nmodes;

  sptMatrix * tmp_mat = mats[nmodes];
  sptMatrix ** ata = (sptMatrix **)malloc((nmodes+1) * sizeof(*ata));
  for(size_t m=0; m < nmodes+1; ++m) {
    ata[m] = (sptMatrix *)malloc(sizeof(sptMatrix));
  }
  for(size_t m=0; m < nmodes; ++m) {
    sptNewMatrix(ata[m], rank, rank);
    // sptMatrixTransposeMultiply(mats[m], ata[m]);  /* The same storage order with mats. */
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, rank, mats[m]->nrows, rank, 1.0, 
      mats[m]->values, rank, mats[m]->values, mats[m]->nrows, 0.0, ata[m]->values, rank);
  }
  sptNewMatrix(ata[nmodes], rank, rank);
  for(size_t m=0; m < nmodes+1; ++m)
    sptDumpMatrix(ata[m], stdout);

  double oldfit = 0;
  double fit = 0;

  // timer_reset(&g_timers[TIMER_ATA]);
  // Timer itertime;
  // Timer * modetime = (Timer*)malloc(nmodes*sizeof(Timer));
  
  /* For MttkrpHyperTensor with size rank. */
  sptVector scratch;
  sptNewVector (&scratch, rank, rank);
  size_t nmats = nmodes - 1;
  sptSizeVector mats_order;
  sptNewSizeVector(&mats_order, nmats, nmats);

  for(size_t it=0; it < niters; ++it) {
    // timer_fstart(&itertime);
    for(size_t m=0; m < nmodes; ++m) {
      // printf("\nmode %lu \n", m);

      assert(spten->ndims[m] == mats[m]->nrows);
      tmp_mat->nrows = mats[m]->nrows;

      /* Factor Matrices order */
      size_t j = 0;
      for (int i=nmodes-1; i>=0; --i) {
        if (i != m) {
          mats_order.data[j] = i;
          ++ j;
        }
      }
      assert (j == nmats);
      sptDumpSizeVector(&mats_order, stdout);

      assert (sptMTTKRP(spten, mats, &mats_order, m, &scratch) == 0);
      sptDumpMatrix(mats[nmodes], stdout);

      // InverseHadamardGramMatrices(m, ata, nmodes);
      // PrintDenseMatrixArray(ata, nmodes+1, "ata after InverseHadamardGramMatrices", stdout);

      // memset(mats[m]->vals, 0, mats[m]->nrows * rank * sizeof(ValueType));
      // sptMatrixMultiply(tmp_mat, ata[nmodes], mats[m]);
      // PrintDenseMatrix(mats[m], "mats[m] after DenseMatrixMultiply", "debug.txt");

      // sptMatrix2Norm(mats[m], lambda);
      // PrintDenseMatrix(mats[m], "mats[m] after DenseMatrix2Norm", "debug.txt");

      // sptMatrixTransposeMultiply(mats[m], ata[m]);
      // PrintDenseMatrix(ata[m], "ata[m] after DenseMatrixTransposeMultiply", "debug.txt");
      // timer_stop(&modetime[m]);
    }

    // PrintDenseValueVector(lambda, rank, "lambda", "debug.txt");
    // fit = KruskalTensorFit(spten, lambda, mats, tmp_mat, ata);
    // timer_stop(&itertime);

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
  }

  // GetFinalLambda(rank, nmodes, mats, lambda);

  for(size_t m=0; m < nmodes+1; ++m) {
    sptFreeMatrix(ata[m]);
  }
  free(ata);
  sptFreeVector(&scratch);
  sptFreeSizeVector(&mats_order);
  // free(modetime);

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
    sptRandomizeMatrix(mats[m], spten->ndims[m], rank);
  }
  sptNewMatrix(mats[nmodes], max_dim, rank);

  for(size_t m=0; m < nmodes+1; ++m)
    sptDumpMatrix(mats[m], stdout);

  sptScalar * lambda = (sptScalar *) malloc(rank * sizeof(sptScalar));

  sptTimer timer;
  sptNewTimer(&timer, 0);
  sptStartTimer(timer);

  ktensor->fit = CpdAlsStep(spten, rank, niters, tol, mats, lambda);

  sptStopTimer(timer);
  sptPrintElapsedTime(timer, "CPU  SpTns CPD-ALS");
  sptFreeTimer(timer);

  ktensor->rank = rank;
  ktensor->nmodes = nmodes;
  ktensor->lambda = lambda;
  ktensor->factors = mats;

  sptFreeMatrix(mats[nmodes]);
  
  return 0;
}


