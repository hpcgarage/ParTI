#include <SpTOL.h>
#include "sptensor.h"

__global__ static void spt_MTTKRPKernel(
    const size_t nmodes,
    const size_t nnz,
    const size_t * Xndims,
    const size_t * dev_mats_order,
    const sptScalar * Xvals,
    const size_t * Xinds,
    const sptScalar * m1vals,
    const sptScalar * m2vals,
    sptScalar * mvals
) {
    const size_t tidx = threadIdx.x;
    const size_t x = blockIdx.x * blockDim.x + tidx;

    if(x < nnz) {
      size_t k = Xinds[x * nmodes + dev_mats_order[0]];
      size_t j = Xinds[x * nmodes + dev_mats_order[1]];
      size_t i = Xinds[x * nmodes + mode];
      sptScalar val;
      for(size_t r=0; r<R; ++r) {
        val = Xvals[i] * m1vals[k * R + r] * m2vals[j * R + r];
        atomimcAdd(&(mvals[i * R + r]), val); 
    }
    __syncthreads();
}




/* Only for 3D tensors */
int sptCudaMTTKRP(sptSparseTensor const * const X,
	sptMatrix ** const mats, 	// mats[nmodes] as temporary space.
	size_t const * const mats_order,	// Correspond to the mode order of X.
	size_t const mode,
	sptScalar * const scratch) {

	size_t const nmodes = X->nmodes;
  if(nmodes != 3)
    spt_CheckError(SPTERR_SHAPE_MISMATCH, "CUDA SpTns MTTKRP", "nmodes ! =3 for
        GPU MTTKRP");
	size_t const nnz = X->nnz;
	size_t const * const ndims = X->ndims;
	size_t const nmats = nmodes - 1;

	/* Check the mats. */
	for(size_t i=0; i<nmodes; ++i) {
		if(mats[i]->ncols != mats[nmodes]->ncols) {
			spt_CheckError(SPTERR_SHAPE_MISMATCH, "CUDA SpTns MTTKRP", "mats[i]->cols != mats[nmodes]->ncols");
                }
		if(mats[i]->nrows != ndims[i]) {
			spt_CheckError(SPTERR_SHAPE_MISMATCH, "CUDA SpTns MTTKRP", "mats[i]->nrows != ndims[i]");
		}
	}

	size_t const I = mats[mode]->nrows;
	size_t const R = mats[mode]->ncols;

	/* Transfer tensor and matrices */
  size_t * Xndims = NULL;
  int result = cudaMalloc((void **) &Xndims, nmodes * sizeof (size_t));
  spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");
  cudaMemcpy(Xndims, ndims, nmodes * sizeof (size_t), cudaMemcpyHostToDevice);

  size_t * dev_mats_order = NULL;
  int result = cudaMalloc((void **) &dev_mats_order, nmats * sizeof (size_t));
  spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");
  cudaMemcpy(dev_mats_order, mats_order, nmats * sizeof (size_t), cudaMemcpyHostToDevice);

  sptScalar * Xvals = NULL;
  int result = cudaMalloc((void **) &Xvals, nnz * sizeof (sptScalar));
  spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");
  cudaMemcpy(Xvals, X->values.data, nnz * sizeof (sptScalar), cudaMemcpyHostToDevice);

  size_t * Xinds = NULL;
  size_t pitch;
  result = cudaMallocPitch((void **) &Xinds, &pitch, nnz * sizeof (sptScalar), nmodes);
  spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");
  result = cudaMemcpy2D(Xinds, pitch, X->inds, nnz * sizeof(sptScalar), nnz * sizeof(sptScalar), nmodes, cudaMemcpyHostToDevice);
  spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");

  /* Only for 3D tensors */
  size_t times_mat_index = mats_order->data[0];
  sptScalar * m1vals = NULL;
  result = cudaMalloc((void **) &m1vals, ndims[times_mat_index] * R * sizeof (sptScalar));
  spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");
  cudaMemcpy(m1vals, mats[times_mat_index]->values, ndims[times_mat_index] * R * sizeof (sptScalar), cudaMemcpyHostToDevice);
  times_mat_index = mats_order->data[1];
  sptScalar * m2vals = NULL;
  result = cudaMalloc((void **) &m2vals, ndims[times_mat_index] * R * sizeof (sptScalar));
  spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");
  cudaMemcpy(m2vals, mats[times_mat_index]->values, ndims[times_mat_index] * R * sizeof (sptScalar), cudaMemcpyHostToDevice);
  sptScalar * mvals = NULL;
  result = cudaMalloc((void **) &mvals, ndims[mode] * R * sizeof (sptScalar));
  spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");
  cudaMemset(mvals, 0, ndims[mode] * R * sizeof (sptScalar));

  sptScalar * dev_scratch = NULL;
  int result = cudaMalloc((void **) &dev_scratch, nnz * R * sizeof (sptScalar));
  spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");
  cudaMemcpy(dev_scratch, scratch, nnz * R * sizeof (sptScalar), cudaMemcpyHostToDevice);



  sptTimer timer;
  sptNewTimer(&timer, 0);
  sptStartTimer(timer);

  size_t nthreads = 32;
  size_t nblocks = (nnz + nthreads -1) / nthreads;

  spt_MTTKRPKernel<<<nblocks, nthreads>>>(
      nmodes,
      nnz,
      Xndims,
      dev_mats_order,
      Xvals,
      Xinds,
      m1vals,
      m2vals,
      mvals
      );


  sptStopTimer(timer);
  sptPrintElapsedTime(timer, "CUDA SpTns MTTKRP");
  sptFreeTimer(timer);

  cudaMemcpy(mats[nmodes]->values, mvals, ndims[mode] * R * sizeof (sptScalar), cudaMemcpyDeviceToHost);
  result = cudaFree(Xvals);
  spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");
  result = cudaFree(Xinds);
  spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");
  result = cudaFree(m1vals);
  spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");
  result = cudaFree(m2vals);
  spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");
  result = cudaFree(mvals);
  spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");

  return 0;
}
