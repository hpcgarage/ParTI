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
#include <cublas_v2.h>
#include "magma.h"
#include "magma_lapack.h"
#include "sptensor.h"


double CudaCpdAlsStep(
    const size_t nmodes,
    const size_t nnz,
    const size_t rank,
    const size_t stride,
    size_t const niters,
    double const tol,
    const size_t * ndims,
    const size_t * Xndims,
    size_t ** const Xinds,
    const sptScalar * Xvals,
    size_t * dev_mats_order,
    sptScalar ** tmp_mats,
    sptScalar ** dev_mats,
    sptScalar ** tmp_ata,
    sptScalar ** dev_ata,
    sptScalar * dev_scratch,
    sptScalar * dev_unit,
    sptScalar * dev_lambda)
{
  size_t nmats = nmodes - 1;
  int result;

  cublasHandle_t handle;
  cublasCreate(&handle);
  float alpha = 1.0;
  float beta = 0.0;

  for(size_t m=0; m < nmodes; ++m) {
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, rank, rank, ndims[m], &alpha, tmp_mats[m], stride, tmp_mats[m], stride, &beta, tmp_ata[m], stride);
  }
  

  double oldfit = 0;
  double fit = 0;

  sptSizeVector mats_order;
  sptNewSizeVector(&mats_order, nmats, nmats);

  // timer_reset(&g_timers[TIMER_ATA]);
  // Timer itertime;
  // Timer * modetime = (Timer*)malloc(nmodes*sizeof(Timer));
  for(size_t it=0; it < niters; ++it) {
    printf("  its = %3lu\n", it+1);
    sptTimer timer;
    sptNewTimer(&timer, 0);
    sptStartTimer(timer);

    for(size_t m=0; m < nmodes; ++m) {
      // printf("\nmode %lu \n", m);

      /* Factor Matrices order */
      size_t j = 0;
      for (int i=nmodes-1; i>=0; --i) {
        if (i != m) {
          mats_order.data[j] = i;
          ++ j;
        }
      }
      assert (j == nmats);
      result = cudaMemcpy(dev_mats_order, mats_order.data, nmats * sizeof (size_t), cudaMemcpyHostToDevice);
      spt_CheckCudaError(result != 0, "CUDA SpTns CPD-ALS");

      assert (sptCudaMTTKRPDevice(m, nmodes, nnz, rank, stride, Xndims, Xinds, Xvals, dev_mats_order, dev_mats, dev_scratch) == 0);

      sptCudaMatrixDotMulSeq(m, nmodes, rank, stride, dev_ata);

      /* mat_syminv(ata[nmodes]); */
      int info;
      int * ipiv = (int*)malloc(rank * sizeof(int));
      // magma_sgesv_gpu(rank, rank, tmp_ata[nmodes], stride, ipiv, dev_unit, stride, &info);
      free(ipiv);


      result = cudaMemset(tmp_mats[m], 0, ndims[m] * stride * sizeof (sptScalar));
      spt_CheckCudaError(result != 0, "CUDA SpTns CPD-ALS");
      cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, rank, ndims[m], rank, &alpha, dev_unit, stride, tmp_mats[nmodes], stride, &beta, tmp_mats[m], stride);

      sptCudaMatrix2Norm(ndims[m], rank, stride, tmp_mats[m], dev_lambda);

      cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, rank, rank, ndims[m], &alpha, tmp_mats[m], stride, tmp_mats[m], stride, &beta, tmp_ata[m], stride);

      // timer_stop(&modetime[m]);
    }

    // fit = KruskalTensorFit(spten, lambda, mats, tmp_mat, ata);
    sptStopTimer(timer);
    sptPrintElapsedTime(timer, "Iteration");
    sptFreeTimer(timer);

    // printf("  its = %3lu  fit = %0.5f  delta = %+0.4e\n",
    //     it+1, fit, fit - oldfit);
    // for(IndexType m=0; m < nmodes; ++m) {
    //   printf("     mode = %1"PF_INDEX" (%0.3fs)\n", m+1,
    //       modetime[m].seconds);
    // }
    // if(it > 0 && fabs(fit - oldfit) < tol) {
    //   break;
    // }
    // oldfit = fit;
  }

  // GetFinalLambda(rank, nmodes, mats, lambda);

  cublasDestroy(handle);
  sptFreeSizeVector(&mats_order);
  // free(modetime);

  return fit;
}


int sptCudaCpdAls(
  sptSparseTensor const * const X,
  size_t const rank,
  size_t const niters,
  double const tol,
  sptKruskalTensor * ktensor)
{
  size_t nmodes = X->nmodes;
  size_t const nnz = X->nnz;
  size_t const * const ndims = X->ndims;
  size_t const nmats = nmodes - 1;
  int result;

  /* Initialize factor matrices */
  size_t max_dim = sptMaxSizeArray(ndims, nmodes);
  sptMatrix ** mats = (sptMatrix **)malloc((nmodes+1) * sizeof(*mats));
  for(size_t m=0; m < nmodes+1; ++m) {
    mats[m] = (sptMatrix *)malloc(sizeof(sptMatrix));
  }
  for(size_t m=0; m < nmodes; ++m) {
    assert(sptNewMatrix(mats[m], X->ndims[m], rank) == 0);
    assert(sptConstantMatrix(mats[m], 1) == 0);
    // assert(sptRandomizeMatrix(mats[m], X->ndims[m], rank) == 0);
  }
  sptNewMatrix(mats[nmodes], max_dim, rank);
  size_t const stride = mats[0]->stride;
  // printf("Initial mats:\n");
  // for(size_t m=0; m < nmodes+1; ++m)
  //   sptDumpMatrix(mats[m], stdout);


  /* Transfer tensor and matrices */
  size_t * Xndims = NULL;
  result = cudaMalloc((void **) &Xndims, nmodes * sizeof (size_t));
  spt_CheckCudaError(result != 0, "CUDA SpTns CPD-ALS");
  result = cudaMemcpy(Xndims, ndims, nmodes * sizeof (size_t), cudaMemcpyHostToDevice);
  spt_CheckCudaError(result != 0, "CUDA SpTns CPD-ALS");

  sptScalar * Xvals = NULL;
  result = cudaMalloc((void **) &Xvals, nnz * sizeof (sptScalar));
  spt_CheckCudaError(result != 0, "CUDA SpTns CPD-ALS");
  result = cudaMemcpy(Xvals, X->values.data, nnz * sizeof (sptScalar), cudaMemcpyHostToDevice);
  spt_CheckCudaError(result != 0, "CUDA SpTns CPD-ALS");

  size_t ** tmp_Xinds = NULL;
  tmp_Xinds = (size_t **)malloc(nmodes * sizeof(size_t*));
  for(size_t i=0; i<nmodes; ++i) {
    result = cudaMalloc((void **) &(tmp_Xinds[i]), nnz * sizeof(size_t));
    spt_CheckCudaError(result != 0, "CUDA SpTns CPD-ALS");
    result = cudaMemcpy(tmp_Xinds[i], X->inds[i].data, nnz * sizeof (size_t), cudaMemcpyHostToDevice);
    spt_CheckCudaError(result != 0, "CUDA SpTns CPD-ALS");
  }
  size_t ** Xinds = NULL;   // array of pointer to device memory
  result = cudaMalloc((void ***) &Xinds, nmodes * sizeof(size_t*));
  spt_CheckCudaError(result != 0, "CUDA SpTns CPD-ALS");
  result = cudaMemcpy(Xinds, tmp_Xinds, nmodes * sizeof (size_t*), cudaMemcpyHostToDevice);
  spt_CheckCudaError(result != 0, "CUDA SpTns CPD-ALS");  

  sptScalar ** tmp_mats = NULL;
  tmp_mats = (sptScalar **)malloc((nmodes+1) * sizeof(sptScalar*));
  for(size_t i=0; i<nmodes+1; ++i) {
    result = cudaMalloc((void **) &(tmp_mats[i]), mats[i]->nrows * mats[i]->stride * sizeof(sptScalar));
    spt_CheckCudaError(result != 0, "CUDA SpTns CPD-ALS");
    result = cudaMemcpy(tmp_mats[i], mats[i]->values, mats[i]->nrows * mats[i]->stride * sizeof(sptScalar), cudaMemcpyHostToDevice);
    spt_CheckCudaError(result != 0, "CUDA SpTns CPD-ALS");
  }
  result = cudaMemset(tmp_mats[nmodes], 0, mats[nmodes]->nrows * mats[nmodes]->stride * sizeof (sptScalar));
  spt_CheckCudaError(result != 0, "CUDA SpTns CPD-ALS");
  sptScalar ** dev_mats = NULL;   // array of pointer to device memory
  result = cudaMalloc((void ***) &dev_mats, (nmodes+1) * sizeof(sptScalar*));
  spt_CheckCudaError(result != 0, "CUDA SpTns CPD-ALS");
  result = cudaMemcpy(dev_mats, tmp_mats, (nmodes+1) * sizeof (sptScalar*), cudaMemcpyHostToDevice);
  spt_CheckCudaError(result != 0, "CUDA SpTns CPD-ALS");


  sptScalar * dev_lambda = NULL;
  result = cudaMalloc((void **) &dev_lambda, rank * sizeof (sptScalar));
  spt_CheckCudaError(result != 0, "CUDA SpTns CPD-ALS");
  result = cudaMemset(dev_lambda, 0, rank * sizeof (sptScalar));
  spt_CheckCudaError(result != 0, "CUDA SpTns CPD-ALS");


  size_t * dev_mats_order = NULL;
  result = cudaMalloc((void **) &dev_mats_order, nmats * sizeof (size_t));
  spt_CheckCudaError(result != 0, "CUDA SpTns CPD-ALS");
  result = cudaMemset(dev_mats_order, 0, nmats * sizeof (size_t));
  spt_CheckCudaError(result != 0, "CUDA SpTns CPD-ALS");

  sptScalar * dev_scratch = NULL;
  result = cudaMalloc((void **) &dev_scratch, nnz * rank * sizeof (sptScalar));
  spt_CheckCudaError(result != 0, "CUDA SpTns CPD-ALS");
  result = cudaMemset(dev_scratch, 0, nnz * rank * sizeof (sptScalar));
  spt_CheckCudaError(result != 0, "CUDA SpTns CPD-ALS");

  sptMatrix * unitMat = (sptMatrix*)malloc(sizeof(sptMatrix));
  sptNewMatrix(unitMat, rank, rank);
  sptIdentityMatrix(unitMat);
  sptScalar * dev_unit = NULL;
  result = cudaMalloc((void **) &dev_unit, rank * unitMat->stride * sizeof (sptScalar));
  spt_CheckCudaError(result != 0, "CUDA SpTns CPD-ALS");
  result = cudaMemcpy(dev_unit, unitMat->values, rank * unitMat->stride * sizeof (sptScalar), cudaMemcpyHostToDevice);
  spt_CheckCudaError(result != 0, "CUDA SpTns CPD-ALS");
  sptFreeMatrix(unitMat);
  free(unitMat);

  sptScalar ** tmp_ata = NULL;
  tmp_ata = (sptScalar **)malloc((nmodes+1) * sizeof(sptScalar*));
  for(size_t i=0; i<nmodes+1; ++i) {
    // Don't use stride for tmp_ata.
    result = cudaMalloc((void **) &(tmp_ata[i]), rank * rank * sizeof(sptScalar));
    spt_CheckCudaError(result != 0, "CUDA SpTns CPD-ALS");
  }
  result = cudaMemset(tmp_ata[nmodes], 0, rank * rank * sizeof (sptScalar));
  spt_CheckCudaError(result != 0, "CUDA SpTns CPD-ALS");
  sptScalar ** dev_ata = NULL;   // array of pointer to device memory
  result = cudaMalloc((void ***) &dev_ata, (nmodes+1) * sizeof(sptScalar*));
  spt_CheckCudaError(result != 0, "CUDA SpTns CPD-ALS");
  result = cudaMemcpy(dev_ata, tmp_ata, (nmodes+1) * sizeof (sptScalar*), cudaMemcpyHostToDevice);
  spt_CheckCudaError(result != 0, "CUDA SpTns CPD-ALS");

  magma_init();
  // sptTimer timer;
  // sptNewTimer(&timer, 0);
  // sptStartTimer(timer);

  ktensor->fit = CudaCpdAlsStep(nmodes, nnz, rank, stride, niters, tol,
    ndims, Xndims, Xinds, Xvals, dev_mats_order, 
    tmp_mats, dev_mats, tmp_ata, dev_ata, 
    dev_scratch, dev_unit,
    dev_lambda);

  // sptStopTimer(timer);
  // sptPrintElapsedTime(timer, "CUDA  SpTns CPD-ALS");
  // sptFreeTimer(timer);
  magma_finalize();

  for(size_t i=0; i<nmodes; ++i) {
    result = cudaMemcpy(mats[i]->values, tmp_mats[i], mats[i]->nrows * mats[i]->stride * sizeof (sptScalar), cudaMemcpyDeviceToHost);
    spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP copy back");
  }
  // sptScalar * lambda = (sptScalar *) malloc(rank * sizeof(sptScalar));
  result = cudaMemcpy(ktensor->lambda, dev_lambda, rank * sizeof (sptScalar), cudaMemcpyDeviceToHost);
  spt_CheckCudaError(result != 0, "CUDA SpTns CPD-ALS");


  // ktensor->rank = rank;
  // ktensor->nmodes = nmodes;
  // ktensor->lambda = lambda;
  ktensor->factors = mats;


  result = cudaFree(Xndims);
  spt_CheckCudaError(result != 0, "CUDA SpTns CPD-ALS");
  result = cudaFree(Xvals);
  spt_CheckCudaError(result != 0, "CUDA SpTns CPD-ALS");
  result = cudaFree(dev_mats_order);
  spt_CheckCudaError(result != 0, "CUDA SpTns CPD-ALS");
  result = cudaFree(dev_lambda);
  spt_CheckCudaError(result != 0, "CUDA SpTns CPD-ALS");
  result = cudaFree(dev_scratch);
  spt_CheckCudaError(result != 0, "CUDA SpTns CPD-ALS");
  for(size_t i=0; i<nmodes; ++i) {
    result = cudaFree(tmp_Xinds[i]);
    spt_CheckCudaError(result != 0, "CUDA SpTns CPD-ALS");
  }
  result = cudaFree(Xinds);
  spt_CheckCudaError(result != 0, "CUDA SpTns CPD-ALS");
  for(size_t i=0; i<nmodes+1; ++i) {
    result = cudaFree(tmp_mats[i]);
    spt_CheckCudaError(result != 0, "CUDA SpTns CPD-ALS");
  }
  result = cudaFree(dev_mats);
  spt_CheckCudaError(result != 0, "CUDA SpTns CPD-ALS");
  for(size_t i=0; i<nmodes+1; ++i) {
    result = cudaFree(tmp_ata[i]);
    spt_CheckCudaError(result != 0, "CUDA SpTns CPD-ALS");
  }
  result = cudaFree(dev_ata);
  spt_CheckCudaError(result != 0, "CUDA SpTns CPD-ALS");
  result = cudaFree(dev_unit);
  spt_CheckCudaError(result != 0, "CUDA SpTns CPD-ALS");
  free(tmp_Xinds);
  free(tmp_mats);
  free(tmp_ata);


  sptFreeMatrix(mats[nmodes]);
  
  return 0;
}


