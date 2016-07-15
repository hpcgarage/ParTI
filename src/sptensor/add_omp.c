#include <SpTOL.h>
#include "sptensor.h"

int sptSparseTensorAddOMP(sptSparseTensor *Y, const sptSparseTensor *X, int const nthreads) {
    /* Ensure X and Y are in same shape */
    if(Y->nmodes != X->nmodes) {
        return -1;
    }
    for(i = 0; i < X->nmodes; ++i) {
        if(Y->ndims[i] != X->ndims[i]) {
            fprintf(stderr, "SpTOL ERROR: Adding tensors in different shapes.\n");
            return -1;
        }
    }
    /* Add elements one by one, assume indices are ordered */
    size_t Ynnz = 0;

    /* Determine partationing strategy. */
    size_t * dist_nnzs_X = (size_t*)malloc(nthreads*sizeof(size_t));
    size_t * dist_nnzs_Y = (size_t*)malloc(nthreads*sizeof(size_t));

    /* Build a private arrays to append values. */


    omp_set_dynamic(0);
    omp_set_num_threads(nthreads);
    #pragma omp parallel private(Ynnz) reduction(+:Ynnz)
    {
        int tid = omp_get_thread_num();
        size_t i=0, j=0;
        Ynnz = dist_nnzs_Y[tid];

        while(i < dist_nnzs_X[tid] && j < dist_nnzs_Y[tid]) {
            int compare = spt_SparseTensorCompareIndices(X, i, Y, j);
            if(compare > 0) {    // X(i) > Y(j)
                ++j;
            } else if(compare < 0) {    // X(i) < Y(j)
                size_t mode;
                int result;
                for(mode = 0; mode < X->nmodes; ++mode) {
                    result = sptAppendSizeVector(&Y->inds[mode], X->inds[mode].data[i]);
                    if(result) {
                        return result;
                    }
                }
                result = sptAppendVector(&Y->values, X->values.data[i]);
                if(result) {
                    return result;
                }
                ++Ynnz;
                ++i;
            } else {    // X(i) = Y(j)
                Y->values.data[j] += X->values.data[i];
                ++i;
                ++j;
            }
        }
        /* Append remaining elements of X to Y */
        while(i < dist_nnzs_X[tid]) {
            size_t mode;
            int result;
            for(mode = 0; mode < X->nmodes; ++mode) {
                result = sptAppendSizeVector(&Y->inds[mode], X->inds[mode].data[i]);
                if(result) {
                    return result;
                }
            }
            result = sptAppendVector(&Y->values, X->values.data[i]);
            if(result) {
                return result;
            }
            ++Ynnz;
            ++i;
        }

    }
    Y->nnz = Ynnz;

    /* Append all the local arrays to Y. */
    

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
