#include <SpTOL.h>
#include <stdio.h>
#include <stdlib.h>

__global__ static void spt_TTMKernel(
    sptScalar *Y_val, size_t Y_stride, size_t Y_nnz,
    const sptScalar *X_val, size_t X_nnz, size_t *X_inds_m,
    size_t *fiberidx_val, size_t fiberidx_len,
    const sptScalar *U_val, size_t U_nrows, size_t U_ncols, size_t U_stride,
    size_t block_offset
) {
    extern __shared__ sptScalar mem_pool[];

    const size_t tidx = threadIdx.x;
    const size_t tidy = threadIdx.y;
    const size_t i = (blockIdx.x + block_offset) * blockDim.x + tidx;
    //const size_t off = blockIdx.x * blockDim.x + tidx;
    size_t inz_begin, inz_end;
    if(i < Y_nnz) {
        inz_begin = fiberidx_val[i];
        inz_end = fiberidx_val[i+1];
    }
    __syncthreads();

    //sptScalar * const Y_shr = (sptScalar *) &mem_pool[tidx*Y_stride]; // size U_ncols
    sptScalar * const Y_shr = (sptScalar *) mem_pool; // size U_ncols
    if(i < Y_nnz && tidy < U_ncols) {
        Y_shr[tidx * Y_stride + tidy] = 0;
    }
    __syncthreads();

    if(i < Y_nnz && tidy < U_ncols) {
        for(size_t j = inz_begin; j < inz_end; ++j) {
            const size_t r = X_inds_m[j];
            Y_shr[tidx * Y_stride + tidy] += X_val[j] * U_val[r*U_stride + tidy];
        }
    }
    __syncthreads();

    if(i < Y_nnz && tidy < U_ncols) {
        Y_val[i*Y_stride + tidy] = Y_shr[tidx*Y_stride + tidy];
    }
    __syncthreads();
}

#if 0
__global__ static void spt_TTMKernel(
    sptScalar *Y_val, size_t Y_stride, size_t Y_nnz,
    const sptScalar *X_val, size_t X_nnz, size_t *X_inds_m,
    size_t *fiberidx_val, size_t fiberidx_len,
    const sptScalar *U_val, size_t U_nrows, size_t U_ncols, size_t U_stride,
    size_t block_offset
) {
    extern __shared__ sptScalar mem_pool[];

    const size_t tidx = threadIdx.x;
    const size_t tidy = threadIdx.y;
    const size_t i = (blockIdx.x + block_offset) * blockDim.x + tidx;
    const size_t off = blockIdx.x * blockDim.x + tidx;
    const size_t inz_begin = fiberidx_val[i];
    const size_t inz_end = fiberidx_val[i+1];

    sptScalar *const Y_shr = (sptScalar *) &mem_pool[tidx*Y_stride]; // size U_ncols

    if(i < Y_nnz) {
        for(size_t k = tidy; k < U_ncols; k += blockDim.x) {
            Y_shr[off*Y_stride + k] = 0;
        }
    }

    __syncthreads();

    if(i < Y_nnz) {
        for(size_t j = inz_begin; j < inz_end; ++j) {
            const size_t r = X_inds_m[j];
            for(size_t k = tidy; k < U_ncols; k += blockDim.x) {
                Y_shr[off*Y_stride + k] += X_val[j] * U_val[r*U_stride + k];
            }
        }
    }

    __syncthreads();

    if(i < Y_nnz) {
        for(size_t k = tidy; k < U_ncols; k += blockDim.x) {
            Y_val[i*Y_stride + k] = Y_shr[off*Y_stride + k];
        }
    }
}
#endif


__global__ static void spt_TTMNaiveKernel(
    sptScalar *Y_val, size_t Y_stride, size_t Y_nnz,
    const sptScalar *X_val, size_t X_nnz, size_t *X_inds_m,
    size_t *fiberidx_val, size_t fiberidx_len,
    const sptScalar *U_val, size_t U_nrows, size_t U_ncols, size_t U_stride,
    size_t block_offset
) {
    const size_t tidx = threadIdx.x;
    const size_t tidy = threadIdx.y;
    const size_t i = (blockIdx.x + block_offset) * blockDim.x + tidx;

    if(i >= Y_nnz || tidy >= U_ncols) return;
    const size_t inz_begin = fiberidx_val[i];
    const size_t inz_end = fiberidx_val[i+1];

    Y_val[i*Y_stride + tidy] = 0;
    for(size_t j = inz_begin; j < inz_end; ++j) {
        const size_t r = X_inds_m[j];
        Y_val[i*Y_stride + tidy] += X_val[j] * U_val[r*U_stride + tidy];
    }
}


__global__ static void spt_TTMNaiveKernelBasic(
    sptScalar *Y_val, size_t Y_stride, size_t Y_nnz,
    const sptScalar *X_val, size_t X_nnz, size_t *X_inds_m,
    size_t *fiberidx_val, size_t fiberidx_len,
    const sptScalar *U_val, size_t U_nrows, size_t U_ncols, size_t U_stride,
    size_t block_offset
) {
    const size_t tidx = threadIdx.x;
    const size_t i = (blockIdx.x + block_offset) * blockDim.x + tidx;
    const size_t inz_begin = fiberidx_val[i];
    const size_t inz_end = fiberidx_val[i+1];
    for(size_t j = inz_begin; j < inz_end; ++j) {
        for(size_t k = 0; k < U_ncols; ++k) {
            size_t r = X_inds_m[j];
            Y_val[i*Y_stride + k] += X_val[j] * U_val[r*U_stride + k];
        }
    }
}


int sptCudaSparseTensorMulMatrix(
    sptSemiSparseTensor *Y,
    sptSparseTensor *X,
    const sptMatrix *U,
    size_t mode
) {
    int result;
    size_t *ind_buf;
    size_t m;
    sptSizeVector fiberidx;
    if(mode >= X->nmodes) {
        sptCheckError(-1, "CUDA SpTns * Mtx");
    }
    if(X->ndims[mode] != U->nrows) {
        sptCheckError(-1, "CUDA SpTns * Mtx");
    }
    if(X->sortkey != mode) {
        sptSparseTensorSortIndexAtMode(X, mode);
    }
    ind_buf = new size_t[X->nmodes * sizeof *ind_buf];
    if(!ind_buf) {
        sptCheckError(-1, "CUDA SpTns * Mtx");
    }
    for(m = 0; m < X->nmodes; ++m) {
        ind_buf[m] = X->ndims[m];
    }
    ind_buf[mode] = U->ncols;
    result = sptNewSemiSparseTensor(Y, X->nmodes, mode, ind_buf);
    delete[] ind_buf;
    sptCheckError(result, "CUDA SpTns * Mtx");
    sptSemiSparseTensorSetIndices(Y, &fiberidx, X);

    sptScalar *Y_val = NULL;
    result = cudaMalloc((void **) &Y_val, Y->nnz * Y->stride * sizeof (sptScalar));
    sptCheckError(result, "CUDA SpTns * Mtx"); // TODO: map error code?
    // jli: Add memset to Y.
    cudaMemset(Y_val, 0, Y->nnz * Y->stride * sizeof (sptScalar));
    sptScalar *X_val = NULL;
    result = cudaMalloc((void **) &X_val, X->nnz * sizeof (sptScalar));
    sptCheckError(result, "CUDA SpTns * Mtx");
    cudaMemcpy(X_val, X->values.data, X->nnz * sizeof (sptScalar), cudaMemcpyHostToDevice);
    size_t *X_inds_m = NULL;
    result = cudaMalloc((void **) &X_inds_m, X->nnz * sizeof (size_t));
    sptCheckError(result, "CUDA SpTns * Mtx");
    cudaMemcpy(X_inds_m, X->inds[mode].data, X->nnz * sizeof (size_t), cudaMemcpyHostToDevice);
    sptScalar *U_val = NULL;
    result = cudaMalloc((void **) &U_val, U->nrows * U->stride * sizeof (sptScalar));
    sptCheckError(result, "CUDA SpTns * Mtx");
    cudaMemcpy(U_val, U->values, U->nrows * U->stride * sizeof (sptScalar), cudaMemcpyHostToDevice);
    size_t *fiberidx_val = NULL;
    result = cudaMalloc((void **) &fiberidx_val, fiberidx.len * sizeof (size_t));
    sptCheckError(result, "CUDA SpTns * Mtx");
    cudaMemcpy(fiberidx_val, fiberidx.data, fiberidx.len * sizeof (size_t), cudaMemcpyHostToDevice);

    const char *env_SPTOL_TTM_KERNEL = getenv("SPTOL_TTM_KERNEL");
    const bool use_naive_kernel = env_SPTOL_TTM_KERNEL && !strcmp(env_SPTOL_TTM_KERNEL, "naive");

    const size_t max_nblocks = 32768;
    const size_t max_nthreads = 1024;
    // size_t sharedMem = (Y->ndims[mode] + X->ndims[mode])*sizeof (sptScalar) + X->ndims[mode]*sizeof (size_t);
    size_t nthreadsX = 32;
    size_t sharedMem = nthreadsX * U->ncols * sizeof (sptScalar);

    size_t all_nblocks = Y->nnz % nthreadsX == 0 ? Y->nnz / nthreadsX : Y->nnz / nthreadsX + 1;
    assert(U->ncols < max_nthreads);
    dim3 dimBlock(nthreadsX, U->ncols);
    // size_t nblocks = Y->nnz < max_nblocks ? Y->nnz : max_nblocks;

    if(!use_naive_kernel) {
        fprintf(stderr, "[CUDA SpTns * Mtx] spt_TTMKernel<<<%zu, (%u, %u), %zu>>>\n", all_nblocks, dimBlock.x, dimBlock.y, sharedMem);
    } else {
        fprintf(stderr, "[CUDA SpTns * Mtx] spt_TTMNaiveKernel<<<%zu, (%u, %u), 0>>>\n", all_nblocks, dimBlock.x, dimBlock.y);
    }

    sptTimer timer;
    sptNewTimer(&timer, 0);
    sptStartTimer(timer);

    for(size_t block_offset = 0; block_offset < all_nblocks; block_offset += max_nblocks) {
        size_t nblocks = all_nblocks - block_offset;
        if(nblocks > max_nblocks) {
            nblocks = max_nblocks;
        }
        printf("nblocks: %zu\n", nblocks);
        if(!use_naive_kernel) {
            spt_TTMKernel<<<nblocks, dimBlock, sharedMem>>>(
                Y_val, Y->stride, Y->nnz,
                X_val, X->nnz, X_inds_m,
                fiberidx_val, fiberidx.len,
                U_val, U->nrows, U->ncols, U->stride,
                block_offset
            );
        } else {
            spt_TTMNaiveKernel<<<nblocks, dimBlock>>>(
                Y_val, Y->stride, Y->nnz,
                X_val, X->nnz, X_inds_m,
                fiberidx_val, fiberidx.len,
                U_val, U->nrows, U->ncols, U->stride,
                block_offset
            );
        }
        result = cudaThreadSynchronize();
        sptCheckError(result, "CUDA SpTns * Mtx kernel");
    }

    sptStopTimer(timer);
    sptPrintElapsedTime(timer, "CUDA SpTns * Mtx");
    sptFreeTimer(timer);

    cudaMemcpy(Y->values.values, Y_val, Y->nnz * Y->stride * sizeof (sptScalar), cudaMemcpyDeviceToHost);
    result = cudaFree(fiberidx_val);
    sptCheckError(result, "CUDA SpTns * Mtx");
    result = cudaFree(U_val);
    sptCheckError(result, "CUDA SpTns * Mtx");
    result = cudaFree(X_inds_m);
    sptCheckError(result, "CUDA SpTns * Mtx");
    result = cudaFree(X_val);
    sptCheckError(result, "CUDA SpTns * Mtx");
    result = cudaFree(Y_val);
    sptCheckError(result, "CUDA SpTns * Mtx");
    sptFreeSizeVector(&fiberidx);

    return 0;
}
