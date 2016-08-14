#include <SpTOL.h>
#include "ssptensor.h"
#include <string.h>

static void spt_SwapValues(sptSemiSparseTensor *tsr, size_t ind1, size_t ind2, sptScalar buffer[]);

int spt_SemiSparseTensorMergeValues(sptSemiSparseTensor *tsr) {
    int result;
    size_t i;
    sptSizeVector collided;
    sptScalar *buffer;

    if(tsr->nnz == 0) {
        return 0;
    }

    buffer = malloc(tsr->stride * sizeof (sptScalar));
    if(!buffer) {
        fprintf(stderr, "SpTOL: memory failure\n");
        return -1;
    }

    result = sptNewSizeVector(&collided, 0, 0);
    if(result) {
        free(buffer);
        return result;
    }

    for(i = 0; i < tsr->nnz-1; ++i) {
        // If two nnz has the same indices
        if(spt_SemiSparseTensorCompareIndices(tsr, i, tsr, i+1) == 0) {
            size_t col;
            for(col = 0; col < tsr->stride; ++col) {
                // Add them together
                tsr->values.values[(i+1)*tsr->stride + col] += tsr->values.values[i*tsr->stride + col];
            }
            sptAppendSizeVector(&collided, i);
        }
    }

    // Swap the last invalidated item with the last item
    i = collided.len;
    while(i) {
        --i;
        assert(tsr->nnz != 0);
        spt_SwapValues(tsr, collided.data[i], tsr->nnz-1, buffer);
        --tsr->nnz;
    }

    // Make sure all Vectors and Matrices have correct sizes
    for(i = 0; i < tsr->nmodes; ++i) {
        if(i != tsr->mode) {
            tsr->inds[i].len = tsr->nnz;
        }
    }
    tsr->values.nrows = tsr->nnz;

    sptFreeSizeVector(&collided);
    free(buffer);

    return sptSemiSparseTensorSortIndex(tsr);
}

static void spt_SwapValues(sptSemiSparseTensor *tsr, size_t ind1, size_t ind2, sptScalar buffer[]) {
    size_t i;
    for(i = 0; i < tsr->nmodes; ++i) {
        if(i != tsr->mode) {
            size_t eleind1 = tsr->inds[i].data[ind1];
            size_t eleind2 = tsr->inds[i].data[ind2];
            tsr->inds[i].data[ind1] = eleind2;
            tsr->inds[i].data[ind2] = eleind1;
        }
    }
    if(ind1 != ind2) {
        memcpy(buffer, &tsr->values.values[ind1*tsr->stride], tsr->stride * sizeof (sptScalar));
        memmove(&tsr->values.values[ind1*tsr->stride], &tsr->values.values[ind2*tsr->stride], tsr->stride * sizeof (sptScalar));
        memcpy(&tsr->values.values[ind2*tsr->stride], buffer, tsr->stride * sizeof (sptScalar));
    }
}
