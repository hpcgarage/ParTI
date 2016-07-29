#include <SpTOL.h>
#include "sptensor.h"

/* TODO: bug. */
int sptSparseTensorSubOMP(sptSparseTensor *Y, sptSparseTensor *X, int const nthreads) {
    /* Ensure X and Y are in same shape */
    if(Y->nmodes != X->nmodes) {
        return -1;
    }
    for(size_t i = 0; i < X->nmodes; ++i) {
        if(Y->ndims[i] != X->ndims[i]) {
            fprintf(stderr, "SpTOL ERROR: Substracting tensors in different shapes.\n");
            return -1;
        }
    }

    /* Determine partationing strategy. */
    size_t * dist_nnzs_X = (size_t*)malloc(nthreads*sizeof(size_t));
    size_t * dist_nnzs_Y = (size_t*)malloc(nthreads*sizeof(size_t));
    size_t * dist_nrows_Y = (size_t*)malloc(nthreads*sizeof(size_t));

    sptDistSparseTensor(Y, nthreads, dist_nnzs_Y, dist_nrows_Y);
    sptDistSparseTensorFixed(X, nthreads, dist_nnzs_X, dist_nnzs_Y);
    free(dist_nrows_Y);

    printf("dist_nnzs_Y:\n");
    for(int i=0; i<nthreads; ++i) {
        printf("%zu ", dist_nnzs_Y[i]);
    }
    printf("\n");
    printf("dist_nnzs_X:\n");
    for(int i=0; i<nthreads; ++i) {
        printf("%zu ", dist_nnzs_X[i]);
    }
    printf("\n");
    fflush(stdout);


    /* Build a private arrays to append values. */
    size_t nnz_gap = fabs(Y->nnz - X->nnz);
    size_t increase_size = 0;
    if(nnz_gap == 0) increase_size = 10;
    else increase_size = nnz_gap;

    sptSizeVector **local_inds = (sptSizeVector**)malloc(nthreads* sizeof *local_inds);
    for(size_t k=0; k<nthreads; ++k) {
        local_inds[k] = (sptSizeVector*)malloc(Y->nmodes* sizeof *(local_inds[k]));
        for(size_t m=0; m<Y->nmodes; ++m) {
            sptNewSizeVector(&(local_inds[k][m]), 0, increase_size);
        }
    }

    sptVector *local_vals = (sptVector*)malloc(nthreads* sizeof *local_vals);
    for(size_t k=0; k<nthreads; ++k) {
        sptNewVector(&(local_vals[k]), 0, increase_size);
    }


    /* Add elements one by one, assume indices are ordered */
    size_t Ynnz = 0;
    omp_set_dynamic(0);
    omp_set_num_threads(nthreads);
    #pragma omp parallel reduction(+:Ynnz)
    {
        int tid = omp_get_thread_num();
        size_t i=0, j=0;
        Ynnz = dist_nnzs_Y[tid];
        while(i < dist_nnzs_X[tid] && j < dist_nnzs_Y[tid]) {
            int compare = spt_SparseTensorCompareIndices(X, i, Y, j);
            if(compare > 0) {
                ++j;
            } else if(compare < 0) {
                size_t mode;
                int result;
                for(mode = 0; mode < X->nmodes; ++mode) {
                    result = sptAppendSizeVector(&(local_inds[tid][mode]), X->inds[mode].data[i]);
                    if(result) {
                        exit(result);
                    }
                }
                result = sptAppendVector(&(local_vals[tid]), -X->values.data[i]);
                if(result) {
                    exit(result);
                }
                ++Ynnz;
                ++i;
            } else {
                Y->values.data[j] -= X->values.data[i];
                ++i;
                ++j;
            }
        }
        /* Append remaining elements of X to Y */
        while(i < dist_nnzs_X[tid]) {
            size_t mode;
            int result;
            for(mode = 0; mode < X->nmodes; ++mode) {
                result = sptAppendSizeVector(&(local_inds[tid][mode]), X->inds[mode].data[i]);
                if(result) {
                    exit(result);
                }
            }
            result = sptAppendVector(&(local_vals[tid]), -X->values.data[i]);
            if(result) {
                exit(result);
            }
            ++Ynnz;
            ++i;
        }

    }

    /* Append all the local arrays to Y. */
    for(size_t k=0; k<nthreads; ++k) {
        for(size_t m=0; m<Y->nmodes; ++m) {
            sptAppendSizeVectorWithVector(&(Y->inds[m]), &(local_inds[k][m]));
        }
        sptAppendVectorWithVector(&(Y->values), &(local_vals[k]));
    }

    for(size_t k=0; k<nthreads; ++k) {
        for(size_t m=0; m<Y->nmodes; ++m) {
            sptFreeSizeVector(&(local_inds[k][m]));
        }
        free(local_inds[k]);
        sptFreeVector(&(local_vals[k]));
    }
    free(local_inds);
    free(local_vals);
    free(dist_nnzs_X);
    free(dist_nnzs_Y);

    /* Check whether elements become zero after adding.
       If so, fill the gap with the [nnz-1]'th element.
    */
    spt_SparseTensorCollectZeros(Y);
    /* Sort the indices */
    sptSparseTensorSortIndex(Y);
    return 0;
}
