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
#include "sptensor.h"

/* TODO: bug. */
int sptSparseTensorSubOMP(sptSparseTensor *Y, sptSparseTensor *X, int const nthreads) {
    /* Ensure X and Y are in same shape */
    if(Y->nmodes != X->nmodes) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "OMP SpTns Sub", "shape mismatch");
    }
    for(sptIndex i = 0; i < X->nmodes; ++i) {
        if(Y->ndims[i] != X->ndims[i]) {
            spt_CheckError(SPTERR_SHAPE_MISMATCH, "OMP SpTns Sub", "shape mismatch");
        }
    }

    /* Determine partationing strategy. */
    sptNnzIndex * dist_nnzs_X = (sptNnzIndex*)malloc(nthreads*sizeof(sptNnzIndex));
    sptNnzIndex * dist_nnzs_Y = (sptNnzIndex*)malloc(nthreads*sizeof(sptNnzIndex));
    sptIndex * dist_nrows_Y = (sptIndex*)malloc(nthreads*sizeof(sptIndex));

    spt_DistSparseTensor(Y, nthreads, dist_nnzs_Y, dist_nrows_Y);
    spt_DistSparseTensorFixed(X, nthreads, dist_nnzs_X);
    free(dist_nrows_Y);

    printf("dist_nnzs_Y:\n");
    for(int i=0; i<nthreads; ++i) {
        printf("%lu ", dist_nnzs_Y[i]);
    }
    printf("\n");
    printf("dist_nnzs_X:\n");
    for(int i=0; i<nthreads; ++i) {
        printf("%lu ", dist_nnzs_X[i]);
    }
    printf("\n");
    fflush(stdout);


    /* Build a private arrays to append values. */
    sptNnzIndex nnz_gap = llabs((long long) Y->nnz - (long long) X->nnz);
    sptNnzIndex increase_size = 0;
    if(nnz_gap == 0) increase_size = 10;
    else increase_size = nnz_gap;

    sptIndexVector **local_inds = (sptIndexVector**)malloc(nthreads* sizeof *local_inds);
    for(int k=0; k<nthreads; ++k) {
        local_inds[k] = (sptIndexVector*)malloc(Y->nmodes* sizeof *(local_inds[k]));
        for(sptIndex m=0; m<Y->nmodes; ++m) {
            sptNewIndexVector(&(local_inds[k][m]), 0, increase_size);
        }
    }

    sptValueVector *local_vals = (sptValueVector*)malloc(nthreads* sizeof *local_vals);
    for(int k=0; k<nthreads; ++k) {
        sptNewValueVector(&(local_vals[k]), 0, increase_size);
    }


    /* Add elements one by one, assume indices are ordered */
    sptNnzIndex Ynnz = 0;
    omp_set_dynamic(0);
    omp_set_num_threads(nthreads);
    #pragma omp parallel reduction(+:Ynnz)
    {
        int tid = omp_get_thread_num();
        sptNnzIndex i=0, j=0;
        Ynnz = dist_nnzs_Y[tid];
        while(i < dist_nnzs_X[tid] && j < dist_nnzs_Y[tid]) {
            int compare = spt_SparseTensorCompareIndices(X, i, Y, j);
            if(compare > 0) {
                ++j;
            } else if(compare < 0) {
                sptIndex mode;
                int result;
                for(mode = 0; mode < X->nmodes; ++mode) {
                    result = sptAppendIndexVector(&(local_inds[tid][mode]), X->inds[mode].data[i]);
                    spt_CheckOmpError(result, "OMP SpTns Sub", NULL);
                }
                result = sptAppendValueVector(&(local_vals[tid]), -X->values.data[i]);
                spt_CheckOmpError(result, "OMP SpTns Sub", NULL);
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
            sptIndex mode;
            int result;
            for(mode = 0; mode < X->nmodes; ++mode) {
                result = sptAppendIndexVector(&(local_inds[tid][mode]), X->inds[mode].data[i]);
                spt_CheckOmpError(result, "OMP SpTns Sub", NULL);
            }
            result = sptAppendValueVector(&(local_vals[tid]), -X->values.data[i]);
            spt_CheckOmpError(result, "OMP SpTns Sub", NULL);
            ++Ynnz;
            ++i;
        }

    }

    /* Append all the local arrays to Y. */
    for(int k=0; k<nthreads; ++k) {
        for(sptIndex m=0; m<Y->nmodes; ++m) {
            sptAppendIndexVectorWithVector(&(Y->inds[m]), &(local_inds[k][m]));
        }
        sptAppendValueVectorWithVector(&(Y->values), &(local_vals[k]));
    }

    for(int k=0; k<nthreads; ++k) {
        for(sptIndex m=0; m<Y->nmodes; ++m) {
            sptFreeIndexVector(&(local_inds[k][m]));
        }
        free(local_inds[k]);
        sptFreeValueVector(&(local_vals[k]));
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
    sptSparseTensorSortIndex(Y, 1, nthreads);

    return 0;
}
