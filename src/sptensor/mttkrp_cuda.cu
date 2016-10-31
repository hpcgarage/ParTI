#include <SpTOL.h>
#include "sptensor.h"
#include <cuda_runtime.h>


__device__ void lock(int* mutex) {
  /* compare mutex to 0.
     when it equals 0, set it to 1
     we will break out of the loop after mutex gets set to  */
     while (atomicCAS(mutex, 0, 1) != 0) {
    /* do nothing */
    }
}


__device__ void unlock(int* mutex) {
  atomicExch(mutex, 0);
}



__global__ static void spt_MTTKRPKernel(
    const size_t mode,
    const size_t nmodes,
    const size_t nnz,
    const size_t R,
    const size_t stride,
    const size_t * Xndims,
    size_t ** const Xinds,
    const sptScalar * Xvals,
    const size_t * dev_mats_order,
    sptScalar ** dev_mats,
    sptScalar * dev_scratch
) {
    const size_t tidx = threadIdx.x;
    const size_t x = blockIdx.x * blockDim.x + tidx;
    // __shared__ int mutex = 0;

    size_t const nmats = nmodes - 1;
    size_t const I = Xndims[mode];
    size_t const * const mode_ind = Xinds[mode];
    /* The 64-bit floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher. */
    sptScalar * const mvals = (sptScalar*)dev_mats[nmodes];

    /* nnz > I */
    // if(x < I) {
    //   for(size_t r=0; r<R; ++r)
    //     mvals[x * stride + r] = 0;
    // }

    
    if(x < nnz) {
      size_t times_mat_index = dev_mats_order[0];
      sptScalar * times_mat = dev_mats[times_mat_index];
      size_t * times_inds = Xinds[times_mat_index];
      size_t tmp_i = times_inds[x];
      sptScalar const entry = Xvals[x];
      for(size_t r=0; r<R; ++r) {
        dev_scratch[x * stride + r] = entry * times_mat[tmp_i * stride + r];
      }

      for(size_t i=1; i<nmats; ++i) {
        times_mat_index = dev_mats_order[i];
        times_mat = dev_mats[times_mat_index];
        times_inds = Xinds[times_mat_index];
        tmp_i = times_inds[x];
        for(size_t r=0; r<R; ++r) {
          dev_scratch[x * stride + r] *= times_mat[tmp_i * stride + r];
        }
      }

    }
    __syncthreads();

    if(x < nnz) {
      size_t const mode_i = mode_ind[x];
      // lock(&mutex);
      for(size_t r=0; r<R; ++r) {
        // mvals[mode_i * stride + r] += dev_scratch[x * stride + r];
        atomicAdd(&(mvals[mode_i * stride + r]), dev_scratch[x * stride + r]);
      }
      // unlock(&mutex);
    }
    __syncthreads();

}




/**
 * CUDA parallelized Matriced sparse tensor times a sequence of dense matrix Khatri-Rao products (MTTKRP) on a specified mode
 * @param[out] mats[nmodes]    the result of MTTKRP, a dense matrix, with size
 * ndims[mode] * R 
 * @param[in]  X    the sparse tensor input X
 * @param[in]  mats    (N+1) dense matrices, with mats[nmodes] as temporary
 * @param[in]  mats_order    the order of the Khatri-Rao products 
 * @param[in]  mode   the mode on which the MTTKRP is performed 
 * @param[in]  scratch an temporary array to store intermediate results, space assigned before this function
 *
 * This function uses support arbitrary-order sparse tensors with Khatri-Rao
 * products of dense factor matrices, the output is the updated dense matrix for the "mode".
 * In this version, atomic function to lock the global reduction and a large
 * scratch is used to maximize parallelism. (To be optimized)
 */
int sptCudaMTTKRP(sptSparseTensor const * const X,
	sptMatrix ** const mats, 	// mats[nmodes] as temporary space.
	sptSizeVector const * const mats_order,	// Correspond to the mode order of X.
	size_t const mode,
	sptVector * scratch) {

	size_t const nmodes = X->nmodes;
	size_t const nnz = X->nnz;
	size_t const * const ndims = X->ndims;
  size_t const R = mats[mode]->ncols;
  size_t const stride = mats[mode]->stride;
  size_t const nmats = nmodes - 1;
  int result;

	/* Check the mats. */
	for(size_t i=0; i<nmodes; ++i) {
		if(mats[i]->ncols != mats[nmodes]->ncols) {
			spt_CheckError(SPTERR_SHAPE_MISMATCH, "CUDA SpTns MTTKRP", "mats[i]->cols != mats[nmodes]->ncols");
                }
		if(mats[i]->nrows != ndims[i]) {
			spt_CheckError(SPTERR_SHAPE_MISMATCH, "CUDA SpTns MTTKRP", "mats[i]->nrows != ndims[i]");
		}
	}


	/* Transfer tensor and matrices */
  size_t * Xndims = NULL;
  result = cudaMalloc((void **) &Xndims, nmodes * sizeof (size_t));
  spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");
  result = cudaMemcpy(Xndims, ndims, nmodes * sizeof (size_t), cudaMemcpyHostToDevice);
  spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");

  sptScalar * Xvals = NULL;
  result = cudaMalloc((void **) &Xvals, nnz * sizeof (sptScalar));
  spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");
  result = cudaMemcpy(Xvals, X->values.data, nnz * sizeof (sptScalar), cudaMemcpyHostToDevice);
  spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");

  size_t ** tmp_Xinds = NULL;
  tmp_Xinds = (size_t **)malloc(nmodes * sizeof(size_t*));
  for(size_t i=0; i<nmodes; ++i) {
    result = cudaMalloc((void **) &(tmp_Xinds[i]), nnz * sizeof(size_t));
    spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");
    result = cudaMemcpy(tmp_Xinds[i], X->inds[i].data, nnz * sizeof (size_t), cudaMemcpyHostToDevice);
    spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");
  }
  size_t ** Xinds = NULL;   // array of pointer to device memory
  result = cudaMalloc((void ***) &Xinds, nmodes * sizeof(size_t*));
  spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");
  result = cudaMemcpy(Xinds, tmp_Xinds, nmodes * sizeof (size_t*), cudaMemcpyHostToDevice);
  spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");

  // size_t * Xinds = NULL;
  // result = cudaMallocPitch((void **) &Xinds, &pitch, nnz * sizeof (sptScalar), nmodes);
  // spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");
  // result = cudaMemcpy2D(Xinds, pitch, X->inds, nnz * sizeof(sptScalar), nnz * sizeof(sptScalar), nmodes, cudaMemcpyHostToDevice);
  // spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");



  size_t * dev_mats_order = NULL;
  result = cudaMalloc((void **) &dev_mats_order, nmats * sizeof (size_t));
  spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");
  result = cudaMemcpy(dev_mats_order, mats_order->data, nmats * sizeof (size_t), cudaMemcpyHostToDevice);
  spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");

  sptScalar ** tmp_mats = NULL;
  tmp_mats = (sptScalar **)malloc((nmodes+1) * sizeof(sptScalar*));
  for(size_t i=0; i<nmodes+1; ++i) {
    result = cudaMalloc((void **) &(tmp_mats[i]), mats[i]->nrows * mats[i]->stride * sizeof(sptScalar));
    spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");
    result = cudaMemcpy(tmp_mats[i], mats[i]->values, mats[i]->nrows * mats[i]->stride * sizeof(sptScalar), cudaMemcpyHostToDevice);
    spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");
  }
  result = cudaMemset(tmp_mats[nmodes], 0, mats[nmodes]->nrows * mats[nmodes]->stride * sizeof (sptScalar));
  spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");
  sptScalar ** dev_mats = NULL;   // array of pointer to device memory
  result = cudaMalloc((void ***) &dev_mats, (nmodes+1) * sizeof(sptScalar*));
  spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");
  result = cudaMemcpy(dev_mats, tmp_mats, (nmodes+1) * sizeof (sptScalar*), cudaMemcpyHostToDevice);
  spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");


  sptScalar * dev_scratch = NULL;
  result = cudaMalloc((void **) &dev_scratch, nnz * R * sizeof (sptScalar));
  spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");
  result = cudaMemset(dev_scratch, 0, nnz * R * sizeof (sptScalar));
  spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");



  size_t nthreads = 128;
  size_t nblocks = (nnz + nthreads -1) / nthreads;

  sptTimer timer;
  sptNewTimer(&timer, 0);
  sptStartTimer(timer);

  spt_MTTKRPKernel<<<nblocks, nthreads>>>(
      mode,
      nmodes,
      nnz,
      R,
      stride,
      Xndims,
      Xinds,
      Xvals,
      dev_mats_order,
      dev_mats,
      dev_scratch
      );
  result = cudaThreadSynchronize();
  spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");


  sptStopTimer(timer);
  sptPrintElapsedTime(timer, "CUDA SpTns MTTKRP");
  sptFreeTimer(timer);


  result = cudaMemcpy(mats[nmodes]->values, tmp_mats[nmodes], mats[nmodes]->nrows * mats[nmodes]->stride * sizeof (sptScalar), cudaMemcpyDeviceToHost);
  spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP copy back");

  result = cudaFree(Xndims);
  spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");
  result = cudaFree(Xvals);
  spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");
  result = cudaFree(dev_mats_order);
  spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");
  result = cudaFree(dev_scratch);
  spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");
  for(size_t i=0; i<nmodes; ++i) {
    result = cudaFree(tmp_Xinds[i]);
    spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");
  }
  result = cudaFree(Xinds);
  spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");
  for(size_t i=0; i<nmodes+1; ++i) {
    result = cudaFree(tmp_mats[i]);
    spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");
  }
  result = cudaFree(dev_mats);
  spt_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");
  free(tmp_Xinds);
  free(tmp_mats);

  return 0;
}
