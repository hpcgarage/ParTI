#include <SpTOL.h>
#include "sptensor.h"
#include <assert.h>

/**
 * compare two indices from two identical or distinct sparse tensors lexicographically
 * @param tsr1 the first sparse tensor
 * @param ind1 the order of the element in the first sparse tensor whose index is to be compared
 * @param tsr2 the second sparse tensor
 * @param ind2 the order of the element in the second sparse tensor whose index is to be compared
 * @return -1 for less, 0 for equal, 1 for greater
 */
int spt_SparseTensorCompareIndices(const sptSparseTensor *tsr1, size_t ind1, const sptSparseTensor *tsr2, size_t ind2) {
    size_t i;
    assert(tsr1->nmodes == tsr2->nmodes);
    for(i = 0; i < tsr1->nmodes; ++i) {
        size_t eleind1 = tsr1->inds[i].data[ind1];
        size_t eleind2 = tsr2->inds[i].data[ind2];
        if(eleind1 < eleind2) {
            return -1;
        } else if(eleind1 > eleind2) {
            return 1;
        }
    }
    return 0;
}

