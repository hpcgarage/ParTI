#include <SpTOL.h>
#include "sptensor.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

int sptSemiSparseTensorToSparseTensor(sptSparseTensor *dest, const sptSemiSparseTensor *src, sptScalar epsilon) {
    size_t i;
    int result;
    size_t nmodes = src->nmodes;
    assert(epsilon > 0);
    dest->nmodes = nmodes;
    dest->ndims = malloc(nmodes * sizeof *dest->ndims);
    spt_CheckOSError(!dest->ndims, "SspTns -> SpTns");
    memcpy(dest->ndims, src->ndims, nmodes * sizeof *dest->ndims);
    dest->nnz = 0;
    dest->inds = malloc(nmodes * sizeof *dest->inds);
    spt_CheckOSError(!dest->inds, "SspTns -> SpTns");
    for(i = 0; i < nmodes; ++i) {
        result = sptNewSizeVector(&dest->inds[i], 0, src->nnz);
        spt_CheckError(result, "SspTns -> SpTns", NULL);
    }
    result = sptNewVector(&dest->values, 0, src->nnz);
    spt_CheckError(result, "SspTns -> SpTns", NULL);
    for(i = 0; i < src->nnz; ++i) {
        size_t j;
        for(j = 0; j < src->ndims[src->mode]; ++j) {
            sptScalar data = src->values.values[i*src->stride + j];
            int data_class = fpclassify(data);
            if(
                data_class == FP_NAN ||
                data_class == FP_INFINITE ||
                (data_class == FP_NORMAL && !(data < epsilon && data > -epsilon))
            ) {
                size_t m;
                for(m = 0; m < nmodes; ++m) {
                    if(m != src->mode) {
                        result = sptAppendSizeVector(&dest->inds[m], src->inds[m].data[i]);
                    } else {
                        result = sptAppendSizeVector(&dest->inds[src->mode], j);
                    }
                    spt_CheckError(result, "SspTns -> SpTns", NULL);
                }
                result = sptAppendVector(&dest->values, data);
                spt_CheckError(result, "SspTns -> SpTns", NULL);
                ++dest->nnz;
            }
        }
    }
    sptSparseTensorSortIndex(dest);
    return 0;
}
